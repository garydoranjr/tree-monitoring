#!/usr/bin/env python3
"""Sync the full-resolution BCI orthomosaics out of a shared Google Drive folder.

The source folder (`orthomosaicsFullRes`, shared by mcgregori@caryinstitute.org)
is not public, and no Drive tooling is available here: rclone/gdown/gsutil are
not installed, and there is no OAuth client or service account for this
project. So this authenticates the way the browser does, with session cookies
lifted from a HAR export, and pulls over plain HTTPS.

Files this large never download directly -- Drive answers `uc?export=download`
with a "Virus scan warning" interstitial. Submitting that form's fields
(including a short-lived `uuid`/`at` pair) yields the real bytes, with
`Accept-Ranges: bytes`, so an interrupted transfer resumes instead of
restarting. That matters a great deal at ~66 GB/file.

The manifest below is baked in rather than scraped at runtime: the sizes were
verified against the server, and hardcoding them means a restart needs neither
the folder listing nor a still-valid HAR to know what "complete" looks like.

Usage:
    # what would be fetched, and what is already here
    python drive_folder_sync.py --dry-run
    python drive_folder_sync.py --status

    # cheap end-to-end probe: first 2 GiB of one file
    python drive_folder_sync.py --only 20250715_bciwhole_rx1rii_rgb.cog.tif \
        --max-bytes $((2 << 30))

    # the real sync (~856 GB, ~2.3h at 8 streams)
    python drive_folder_sync.py -j 8

Re-running is safe and cheap: complete files are skipped, partial ones resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import pathlib
import re
import sys
import threading
import time

import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_DEST = pathlib.Path("/scratch/tree-monitoring/full_island")
DEFAULT_HAR = REPO_ROOT / "drive.google.com_Archive [26-09-01 13-48-44].har"

# Firefox 140, matching the browser the HAR came from. Drive serves the
# interstitial differently to clients it does not recognize.
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64; rv:140.0) "
              "Gecko/20100101 Firefox/140.0")

# The shared folder lives under this account's Drive, not the default one; the
# download endpoints 302 to a sign-in page without the right authuser index.
AUTHUSER = "1"

CHUNK = 8 << 20  # 8 MiB, as in globus_https_sync.py

# (file_id, filename, exact size in bytes). Sizes came from the Content-Range
# of a probe request against each id, so they are the server's own numbers and
# are safe to assert on.
MANIFEST = [
    ("1ZhPjUCE1nWFod4GXnFBslUFg_v60-new", "2024-06-11_orthoWhole_bci_resFull.tif", 53031067268),
    ("19W5WUZc4dWbOxDNq5jlOx5z7eVD5t1pl", "2024-08-13_orthoWhole_bci_resFull.tif", 71149496862),
    ("1ttgHZ9uGwkYPog0BDYIWDRYSf2zt99OB", "2024-09-18_orthoWhole_bci_resFull.tif", 63986482542),
    ("1nhpazIcXHaF3VaZdAW7vrJDf0kaA-3Bv", "2024-10-14_orthoWhole_bci_resFull.tif", 69615701906),
    ("1CFb5K8gGMt0fBQLy94I3CWRRMDJPlubY", "2024-11-12_orthoWhole_bci_resFull.tif", 69148672186),
    ("1g_NfFXfsNiMjDJwyClknIaw30Pzk9Ds1", "2024-12-16_orthoWhole_bci_resFull.tif", 73706248655),
    ("1NTnGoQ2HS-Z5JRAH2oQ1_uKnrfhqRd2M", "2025-01-24_orthoWhole_bci_resFull.tif", 67238658411),
    ("1sVJ48DlqYDWitLXb2XCEm7fVZTaOh9Vm", "2025-02-17_orthoWhole_bci_resFull.tif", 70776583863),
    ("1ubXDcOiTD4pmpnjm-pwkWPnHQolgNFyc", "2025-03-17_orthoWhole_bci_resFull.tif", 72960351945),
    ("1BG3OtkbII0XNTBUR8DdoRUHB9w_ff-hF", "2025-04-14_orthoWhole_bci_resFull.tif", 72088177384),
    ("1z9a1-nW_NTROgdoC2lW5KUEQQyWlVvee", "20250715_bciwhole_rx1rii_rgb.cog.tif", 57114202917),
    ("1S4jppPYYxHNYiWZ7ADcyqunQOMsvSMqj", "20250818_bciwhole_rx1rii_rgb.cog.tif", 57466887075),
    ("1CtQF4dXHy5esZQfVEmxTCR36R8R7N6Rx", "20250915_bciwhole_rx1rii_rgb.cog.tif", 57468174905),
]


class ExpiredCredentials(RuntimeError):
    """Drive answered with HTML (a sign-in page) where bytes were expected.

    Raised as its own type because it is not worth retrying -- every attempt
    will fail the same way until the HAR is re-exported -- and because writing
    that HTML into a .tif would produce a file of plausible size and no value.
    """


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024


def hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def load_cookies(har_path: pathlib.Path) -> dict:
    """Pull drive.google.com request cookies out of a HAR export.

    Only cookie names/values are read; nothing else in the HAR is used, and
    values are never logged -- these are full Google account credentials.
    """
    if not har_path.exists():
        sys.exit(
            f"No HAR at {har_path}.\n"
            "Export one from a browser tab showing the Drive folder "
            "(devtools > Network > save all as HAR)."
        )
    try:
        har = json.loads(har_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        sys.exit(f"{har_path} is not valid JSON: {exc}")

    jar: dict[str, str] = {}
    for entry in har.get("log", {}).get("entries", []):
        if "drive.google.com" not in entry.get("request", {}).get("url", ""):
            continue
        for cookie in entry["request"].get("cookies", []):
            jar[cookie["name"]] = cookie["value"]
    if not jar:
        sys.exit(f"No drive.google.com cookies found in {har_path}.")
    # These two carry the authenticated session; without them Drive 302s to
    # a sign-in page and we would silently write HTML into .tif files.
    for required in ("SID", "SAPISID"):
        if required not in jar:
            sys.exit(f"HAR is missing the {required} cookie; re-export it while signed in.")
    return jar


def make_session(jar: dict) -> requests.Session:
    """One Session per worker -- requests.Session is not thread-safe."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for name, value in jar.items():
        session.cookies.set(name, value, domain=".google.com")
    return session


def resolve_download(session: requests.Session, file_id: str) -> tuple[str, dict]:
    """Clear the virus-scan interstitial and return (url, form params).

    The `uuid`/`at` fields are short-lived, so this is re-run per attempt
    rather than cached -- a stale confirm token is the most likely way a
    long-running transfer breaks.
    """
    landing = session.get(
        "https://drive.google.com/uc",
        params={"id": file_id, "export": "download", "authuser": AUTHUSER},
        timeout=(30, 120),
    )
    landing.raise_for_status()
    body = landing.text

    form = re.search(r'<form[^>]+action="([^"]+)"', body)
    if not form:
        if "accounts.google.com" in landing.url or "signin" in landing.url:
            raise ExpiredCredentials("redirected to sign-in")
        title = re.search(r"<title>(.*?)</title>", body, re.S)
        raise RuntimeError(
            f"no download form for {file_id}"
            + (f" (page: {title.group(1).strip()})" if title else "")
        )

    action = html.unescape(form.group(1))
    params = {
        name: html.unescape(value)
        for name, value in re.findall(
            r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', body
        )
    }
    params.setdefault("id", file_id)
    params.setdefault("export", "download")
    params.setdefault("confirm", "t")
    return action, params


def fetch(session, file_id, dest: pathlib.Path, size: int, on_bytes,
          max_bytes: int | None = None) -> None:
    """Download to a .part file, resuming an existing one, then rename.

    Writes only ever go to `<dest>.part`; the final name appears atomically and
    only after the size check, so a partial transfer can never be mistaken for
    a finished raster.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    if have > size:  # stale or corrupt partial -- start over
        tmp.unlink()
        have = 0

    target = min(size, have + max_bytes) if max_bytes else size
    if have >= target:
        return

    url, params = resolve_download(session, file_id)
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"

    with session.get(url, params=params, headers=headers, stream=True,
                     timeout=(30, 300)) as r:
        # An HTML body here means the session lapsed; writing it would corrupt
        # the output with a file of believable size.
        if "text/html" in r.headers.get("Content-Type", ""):
            raise ExpiredCredentials(f"HTML response for {dest.name}")
        if have and r.status_code == 200:
            mode, have = "wb", 0  # server ignored Range; restart cleanly
            on_bytes(-tmp.stat().st_size if tmp.exists() else 0)
        r.raise_for_status()

        with open(tmp, mode) as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                on_bytes(len(chunk))
                if max_bytes and fh.tell() >= target:
                    break

    got = tmp.stat().st_size
    if max_bytes and got < size:
        return  # bounded probe: leave the .part for a later full run to resume
    if got != size:
        raise IOError(f"size mismatch for {dest.name}: got {got:,}, expected {size:,}")
    tmp.replace(dest)


def read_progress(dest: pathlib.Path) -> tuple[int, int, int]:
    """Bytes on disk, plus complete/partial counts, from the filesystem alone.

    Deliberately independent of the state file so `--status` reports the truth
    even if a previous run was killed before it could write state.
    """
    done_bytes = complete = partial = 0
    for _, name, size in MANIFEST:
        final = dest / name
        part = dest / (name + ".part")
        if final.exists() and final.stat().st_size == size:
            done_bytes += size
            complete += 1
        elif part.exists():
            done_bytes += part.stat().st_size
            partial += 1
    return done_bytes, complete, partial


def print_status(dest: pathlib.Path) -> int:
    total = sum(s for _, _, s in MANIFEST)
    done, complete, partial = read_progress(dest)
    print(f"destination: {dest}\n")
    for _, name, size in MANIFEST:
        final, part = dest / name, dest / (name + ".part")
        if final.exists() and final.stat().st_size == size:
            state = f"complete  {human(size)}"
        elif final.exists():
            state = f"WRONG SIZE {human(final.stat().st_size)} != {human(size)}"
        elif part.exists():
            got = part.stat().st_size
            state = f"partial   {human(got)} / {human(size)} ({got / size:.0%})"
        else:
            state = f"missing   {human(size)}"
        print(f"  {name:42} {state}")
    print(f"\n{complete}/{len(MANIFEST)} complete, {partial} partial")
    print(f"{human(done)} of {human(total)} on disk ({done / total:.1%})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEFAULT_DEST, type=pathlib.Path,
                    help=f"local destination directory (default {DEFAULT_DEST})")
    ap.add_argument("--har", default=DEFAULT_HAR, type=pathlib.Path,
                    help="HAR export supplying Drive session cookies")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched, transfer nothing")
    ap.add_argument("--status", action="store_true",
                    help="summarize on-disk progress and exit (safe while a sync runs)")
    ap.add_argument("--only", metavar="NAME", action="append",
                    help="fetch only these filenames (repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch files that are already complete")
    ap.add_argument("--max-bytes", type=int, metavar="N",
                    help="stop each file after ~N new bytes (for probes)")
    ap.add_argument("-j", "--jobs", type=int, default=8, metavar="N",
                    help="concurrent downloads (default 8; ~99 MiB/s measured)")
    ap.add_argument("--retries", type=int, default=5, metavar="N",
                    help="attempts per file (default 5), resuming each time")
    ap.add_argument("--progress-interval", type=float, default=30.0, metavar="SEC",
                    help="seconds between progress lines (default 30; 0 disables)")
    ap.add_argument("--state-file", type=pathlib.Path,
                    help="progress JSON (default <dest>/.sync_state.json)")
    args = ap.parse_args()

    if args.status:
        return print_status(args.dest)

    entries = list(MANIFEST)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e[1] in wanted]
        missing = wanted - {e[1] for e in entries}
        if missing:
            sys.exit("not in manifest: " + ", ".join(sorted(missing)))

    # Decide everything before moving any bytes.
    todo, skipped, resuming = [], 0, 0
    for file_id, name, size in entries:
        final = args.dest / name
        if not args.force and final.exists() and final.stat().st_size == size:
            skipped += 1
            continue
        part = args.dest / (name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if have:
            resuming += 1
        todo.append((file_id, name, size, final, have))

    remaining = sum(size - have for _, _, size, _, have in todo)
    print(f"destination: {args.dest}")
    print(f"{len(entries)} file(s) in manifest; {skipped} already complete; "
          f"{len(todo)} to fetch ({human(remaining)} remaining"
          + (f", {resuming} resuming" if resuming else "") + ")\n")

    if args.dry_run:
        for _, name, size, _, have in todo:
            note = f"resume at {human(have)} of {human(size)}" if have else human(size)
            print(f"  would fetch {name:42} {note}")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    jar = load_cookies(args.har)
    state_file = args.state_file or args.dest / ".sync_state.json"

    lock = threading.Lock()
    moved = 0            # new bytes this run, for rate/ETA
    already = sum(have for *_, have in todo)  # resumed bytes, for percent-complete
    counters = {"done": 0, "failed": 0}
    status: dict[str, str] = {name: "pending" for _, name, _, _, _ in todo}
    active: set[str] = set()
    start = time.monotonic()
    finished = threading.Event()

    def on_bytes(n: int) -> None:
        nonlocal moved
        with lock:
            moved += n

    def save_state() -> None:
        payload = {
            "dest": str(args.dest),
            "total_bytes": sum(s for _, _, s in MANIFEST),
            "run_bytes": moved,
            "done": counters["done"],
            "failed": counters["failed"],
            "elapsed_seconds": round(time.monotonic() - start, 1),
            "files": status,
        }
        tmp = state_file.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(state_file)
        except OSError:
            pass  # progress reporting must never break the transfer

    def monitor() -> None:
        """Print aggregate throughput and an ETA on a fixed interval."""
        last_bytes, last_time = 0, time.monotonic()
        while not finished.wait(args.progress_interval):
            with lock:
                snapshot, done, streams = moved, counters["done"], sorted(active)
            now = time.monotonic()
            mean = snapshot / (now - start) if now > start else 0
            inst = (snapshot - last_bytes) / (now - last_time) if now > last_time else 0
            last_bytes, last_time = snapshot, now
            frac = (already + snapshot) / (already + remaining) if remaining else 1.0
            eta = (remaining - snapshot) / mean if mean > 0 else 0
            print(f"  ... {human(already + snapshot)} of {human(already + remaining)} "
                  f"({frac:.1%})  {human(inst)}/s now, {human(mean)}/s mean  "
                  f"{done}/{len(todo)} files, {len(streams)} active  "
                  f"eta {hms(eta)}", flush=True)
            with lock:
                save_state()

    def worker(job) -> None:
        file_id, name, size, final, _ = job
        session = make_session(jar)
        with lock:
            active.add(name)
            status[name] = "active"
        try:
            for attempt in range(1, args.retries + 1):
                try:
                    fetch(session, file_id, final, size, on_bytes, args.max_bytes)
                    with lock:
                        counters["done"] += 1
                        status[name] = "complete"
                        n = counters["done"] + counters["failed"]
                        elapsed = time.monotonic() - start
                        rate = moved / elapsed if elapsed else 0
                        eta = (remaining - moved) / rate if rate > 0 else 0
                        print(f"[{n}/{len(todo)}] {name} ok  ({human(size)})  "
                              f"aggregate {human(rate)}/s  eta {hms(eta)}", flush=True)
                        save_state()
                    return
                except ExpiredCredentials as exc:
                    with lock:
                        counters["failed"] += 1
                        status[name] = "auth-failed"
                        print(f"[!] {name}: {exc}\n"
                              f"    Drive credentials look expired -- re-export the HAR "
                              f"and re-run; partial files will resume.", flush=True)
                        save_state()
                    return
                except (requests.RequestException, IOError, OSError) as exc:
                    if attempt == args.retries:
                        with lock:
                            counters["failed"] += 1
                            status[name] = f"failed: {type(exc).__name__}"
                            print(f"[!] {name} FAILED after {attempt} attempts: "
                                  f"{type(exc).__name__}: {exc}", flush=True)
                            save_state()
                        return
                    backoff = min(5 * 2 ** (attempt - 1), 300)
                    with lock:
                        status[name] = f"retry {attempt}"
                        print(f"    {name}: {type(exc).__name__}: {exc} "
                              f"-- retry {attempt}/{args.retries - 1} in {backoff}s",
                              flush=True)
                    time.sleep(backoff)
        finally:
            session.close()
            with lock:
                active.discard(name)

    print(f"fetching with {args.jobs} concurrent stream(s); "
          f"progress every {args.progress_interval:g}s\n")
    if args.progress_interval > 0:
        threading.Thread(target=monitor, daemon=True).start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(worker, todo))
    except KeyboardInterrupt:
        finished.set()
        print("\ninterrupted -- partial files kept as .part; re-run to resume")
        save_state()
        return 130
    finished.set()

    elapsed = time.monotonic() - start
    rate = moved / elapsed if elapsed else 0
    with lock:
        save_state()
    print(f"\n{counters['done']} fetched ({human(moved)} this run), "
          f"{skipped} already complete, {counters['failed']} failed")
    print(f"elapsed {hms(elapsed)}, aggregate {human(rate)}/s")
    if counters["failed"]:
        print("re-run to resume the failed transfers from where they stopped")
    return 1 if counters["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
