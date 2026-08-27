#!/usr/bin/env python3
"""Sync a subtree from a Globus GCS v5 collection over HTTPS.

Globus Connect Personal cannot run on Gattaca2: its relay registration needs
outbound SSH to relay.globusonline.org:2223, which the cluster firewall blocks.
This fetches over the collection's HTTPS interface instead, which needs only
outbound 443. See NOTES.md for the diagnosis.

First run needs a browser once to authorize; the refresh token is cached in
--token-file afterwards, so later runs are unattended.

Usage:
    # one-time, prints a URL and waits for a code
    python globus_https_sync.py --login

    # list what would be fetched, transfer nothing
    python globus_https_sync.py --dry-run

    # fetch a single file (cheap end-to-end probe)
    python globus_https_sync.py --only BCI_50ha_big_shape_all.gpkg \
        --src /UAVSHARE/paula_mavic_products/Aligned/aux_files/

    # the real sync
    python globus_https_sync.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime
import json
import os
import pathlib
import stat
import sys
import threading
import time

import globus_sdk
import requests

# globus_sdk lazily imports submodules; `globus_sdk.scopes` is not populated by
# `import globus_sdk` alone, so bind it explicitly.
from globus_sdk import scopes as _scopes  # noqa: F401  (registers the submodule)

# The Globus CLI's own registered native app. Using it means no separate app
# registration, and it is the same client the `globus` command authenticates as.
CLIENT_ID = "95fdeba8-fac2-42bd-a357-e068d82ff78e"

COLLECTION = "075e93e2-6aab-4731-ab82-4a896271fa43"  # SI_STRI_ForestLandscapes_Shares
DEFAULT_SRC = "/UAVSHARE/paula_mavic_products/Aligned/Product_global/RGB/"
DEFAULT_DEST = "/scratch/tree-monitoring/stri/globus/RGB"
DEFAULT_TOKENS = pathlib.Path.home() / ".globus_https_sync_tokens.json"

CHUNK = 8 << 20  # 8 MiB


def scopes() -> list:
    """Transfer API for listing, plus the collection's HTTPS scope for data.

    Requested as two top-level scopes rather than one nested dependency, so Auth
    returns a separate token per resource server -- the HTTPS token keyed by the
    collection UUID. Guest collections need no additional data_access scope.

    Note TransferScopes.all is already a Scope in globus-sdk 4.x; wrapping it in
    Scope() again raises TypeError. It is deep-copied because with_dependency and
    friends mutate in place, and the module-level object is shared.
    """
    return [
        copy.deepcopy(globus_sdk.scopes.TransferScopes.all),
        globus_sdk.Scope(f"https://auth.globus.org/scopes/{COLLECTION}/https"),
    ]


def save_tokens(path: pathlib.Path, response) -> dict:
    data = {
        rs: {
            "access_token": d["access_token"],
            "refresh_token": d.get("refresh_token"),
            "expires_at_seconds": d.get("expires_at_seconds"),
        }
        for rs, d in response.by_resource_server.items()
    }
    path.write_text(json.dumps(data, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — these are credentials
    return data


def do_login(token_file: pathlib.Path) -> dict:
    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    client.oauth2_start_flow(requested_scopes=scopes(), refresh_tokens=True)
    print("\nOpen this URL in a browser, log in with JPL SSO, and Allow:\n")
    print("   ", client.oauth2_get_authorize_url(), "\n")
    code = input("Paste the resulting authorization code here: ").strip()
    tokens = save_tokens(token_file, client.oauth2_exchange_code_for_tokens(code))
    print(f"\nTokens cached in {token_file} (mode 0600).")
    print("Resource servers authorized:", ", ".join(tokens))
    return tokens


def load_tokens(token_file: pathlib.Path) -> dict:
    if not token_file.exists():
        sys.exit(
            f"No cached tokens at {token_file}.\n"
            "Run once with --login to authorize (needs a browser)."
        )
    return json.loads(token_file.read_text())


def authorizers(tokens: dict, token_file: pathlib.Path):
    """Build refreshing authorizers for the Transfer API and the HTTPS endpoint."""
    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

    def on_refresh(response, _tf=token_file, _t=tokens):
        for rs, d in response.by_resource_server.items():
            _t.setdefault(rs, {}).update(
                access_token=d["access_token"],
                refresh_token=d.get("refresh_token", _t.get(rs, {}).get("refresh_token")),
                expires_at_seconds=d.get("expires_at_seconds"),
            )
        _tf.write_text(json.dumps(_t, indent=2))
        _tf.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def make(rs: str):
        if rs not in tokens:
            sys.exit(
                f"No token for resource server {rs!r}; re-run with --login.\n"
                f"Have: {', '.join(tokens)}"
            )
        entry = tokens[rs]
        if entry.get("refresh_token"):
            return globus_sdk.RefreshTokenAuthorizer(
                entry["refresh_token"], client,
                access_token=entry.get("access_token"),
                expires_at=entry.get("expires_at_seconds"),
                on_refresh=on_refresh,
            )
        return globus_sdk.AccessTokenAuthorizer(entry["access_token"])

    return make("transfer.api.globus.org"), make(COLLECTION)


def walk(tc: globus_sdk.TransferClient, path: str):
    """Yield (remote_path, size, last_modified) for every file under path."""
    for item in tc.operation_ls(COLLECTION, path=path):
        child = path.rstrip("/") + "/" + item["name"]
        if item["type"] == "dir":
            yield from walk(tc, child)
        elif item["type"] == "file":
            yield child, item["size"], item["last_modified"]


def parse_mtime(value: str) -> float:
    """Globus returns e.g. '2026-07-12 22:44:05+00:00'; treat naive as UTC."""
    dt = datetime.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024


class GlobusAuth(requests.auth.AuthBase):
    """Re-read the token from the authorizer on every request.

    A long sync (355 GiB of rasters) outlives the ~48h access token, and a
    RefreshTokenAuthorizer renews it transparently -- but only if we ask it each
    time. Snapshotting the header once would fail partway through.
    """

    def __init__(self, authorizer):
        self.authorizer = authorizer

    def __call__(self, request):
        request.headers["Authorization"] = self.authorizer.get_authorization_header()
        return request


def fetch(session, url, dest: pathlib.Path, size: int, mtime: float) -> None:
    """Download to a .part file, resuming if one is already present, then rename.

    The collection advertises Accept-Ranges: bytes, so a partial file from an
    interrupted run is resumed rather than restarted -- worth a lot for ~4 GiB
    rasters over a WAN link.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    if have > size:  # stale/corrupt partial
        tmp.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    with session.get(url, stream=True, headers=headers, timeout=(30, 300)) as r:
        if have and r.status_code == 200:
            # Server ignored the Range header; start over.
            mode, have = "wb", 0
        elif have and r.status_code != 206:
            r.raise_for_status()
        else:
            r.raise_for_status()
        with open(tmp, mode) as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                fh.write(chunk)

    got = tmp.stat().st_size
    if got != size:
        raise IOError(f"size mismatch for {dest.name}: got {got:,}, expected {size:,}")
    tmp.replace(dest)              # atomic: never leaves a half file in place
    os.utime(dest, (mtime, mtime))  # so the next run can skip it


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--login", action="store_true",
                    help="run the one-time browser authorization and cache tokens")
    ap.add_argument("--src", default=DEFAULT_SRC, help="source path in the collection")
    ap.add_argument("--dest", default=DEFAULT_DEST, type=pathlib.Path,
                    help="local destination directory")
    ap.add_argument("--token-file", default=DEFAULT_TOKENS, type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched, transfer nothing")
    ap.add_argument("--only", metavar="NAME", action="append",
                    help="fetch only these filenames (repeatable)")
    ap.add_argument("--limit", type=int, help="stop after N files (for testing)")
    ap.add_argument("-j", "--jobs", type=int, default=8, metavar="N",
                    help="concurrent downloads (default 8); a single stream is "
                         "WAN-latency-limited to ~2 MiB/s")
    args = ap.parse_args()

    if args.login:
        do_login(args.token_file)
        return 0

    tokens = load_tokens(args.token_file)
    transfer_auth, https_auth = authorizers(tokens, args.token_file)
    tc = globus_sdk.TransferClient(authorizer=transfer_auth)

    base = tc.get_endpoint(COLLECTION)["https_server"].rstrip("/")
    print(f"collection HTTPS base: {base}")
    print(f"source:      {args.src}")
    print(f"destination: {args.dest}\n")

    entries = list(walk(tc, args.src))
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if os.path.basename(e[0]) in wanted]
        missing = wanted - {os.path.basename(e[0]) for e in entries}
        if missing:
            sys.exit(f"not found under {args.src}: {', '.join(sorted(missing))}")
    entries.sort(key=lambda e: e[0])

    # Decide what actually needs fetching before moving any bytes.
    todo, skip = [], 0
    for remote, size, last_modified in entries:
        local = args.dest / os.path.relpath(remote, args.src)
        mtime = parse_mtime(last_modified)
        if local.exists() and local.stat().st_size == size and local.stat().st_mtime >= mtime:
            skip += 1
            continue
        todo.append((remote, size, mtime, local))

    if args.limit:
        todo = todo[: args.limit]

    total = sum(t[1] for t in todo)
    print(f"{len(entries)} file(s) at source; {skip} already current; "
          f"{len(todo)} to fetch ({human(total)})\n")

    if args.dry_run:
        for remote, size, _, _ in todo:
            print(f"  would fetch {os.path.basename(remote):60} {human(size)}")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    # A single HTTPS stream sustains only ~2 MiB/s to this collection (measured),
    # which is a per-stream TCP limit over a ~70ms WAN RTT rather than a bandwidth
    # ceiling -- so fetching several files at once multiplies throughput. Each
    # worker gets its own Session because requests.Session is not thread-safe.
    lock = threading.Lock()
    counters = {"done": 0, "failed": 0, "moved": 0}
    start = time.monotonic()

    def worker(job):
        remote, size, mtime, local = job
        name = os.path.basename(remote)
        session = requests.Session()
        session.auth = GlobusAuth(https_auth)
        try:
            fetch(session, base + remote, local, size, mtime)
        except Exception as exc:
            with lock:
                counters["failed"] += 1
                n = counters["done"] + counters["failed"]
                print(f"[{n}/{len(todo)}] {name} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
            return
        finally:
            session.close()
        with lock:
            counters["done"] += 1
            counters["moved"] += size
            n = counters["done"] + counters["failed"]
            elapsed = time.monotonic() - start
            rate = counters["moved"] / elapsed if elapsed else 0
            remaining = total - counters["moved"]
            eta = remaining / rate / 60 if rate else 0
            print(f"[{n}/{len(todo)}] {name} ok  ({human(size)})  "
                  f"aggregate {human(rate)}/s  eta {eta:,.0f} min", flush=True)

    print(f"fetching with {args.jobs} concurrent stream(s)\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(worker, todo))

    elapsed = time.monotonic() - start
    rate = counters["moved"] / elapsed if elapsed else 0
    print(f"\n{counters['done']} fetched ({human(counters['moved'])}), "
          f"{skip} already current, {counters['failed']} failed")
    print(f"elapsed {elapsed/60:,.1f} min, aggregate {human(rate)}/s")
    return 1 if counters["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
