# NOTES — STRI RGB sync onto Gattaca2

Findings from the first execution attempt of
[globus-gattaca2-sync-handoff.md](globus-gattaca2-sync-handoff.md).
That document had never been run against the cluster; this file records what
actually happened.

**Session 2026-08-27, on `cn003.cm.cluster`** (compute node, inside Slurm job
`19357244` — *not* the head node).

## Headline result

**Route A is not viable on Gattaca2.** Globus Connect Personal cannot complete
setup because its mandatory relay step needs **outbound SSH**, which the
cluster's egress firewall blocks. This is a different failure than the handoff
doc anticipated — the doc worried about GridFTP data-channel ports 50000–51000,
but the block hits earlier, at relay registration.

**Route C (HTTPS) works and is now in production use.** Implemented as
[scripts/globus_https_sync.py](scripts/globus_https_sync.py). It needs only
outbound 443. Verified end to end on 2026-08-27: probe file fetched byte-exact,
sync is idempotent, and the full 355 GiB pull is running.

### Throughput — measured, and the reason for parallelism

A single HTTPS stream sustains only **~2.3 MiB/s** to this collection, which
would put the full set at ~45 hours. This is the cost §5c warns about
("single-stream HTTP will usually be slower than Globus's parallel data
channels"). It is a **per-stream TCP limit over the ~70 ms California↔Panama
RTT, not a bandwidth ceiling**, so concurrent fetches scale it:

| Concurrent streams | Aggregate rate | Full-set ETA |
|---|---|---|
| 1 | 2.26 MiB/s | ~45 h |
| 4 | 11.2 MiB/s | ~9 h |
| 8 | **13.7 MiB/s** | **~7.3 h** |

Near-linear from 1→4, flattening by 8 — so 8 is about the practical knee and
higher `-j` is unlikely to pay. This measurement also settles half of §2's
"which route is fastest" question without needing the JPL staging leg: the WAN
path, not the receiving host, is the limiting factor.

---

## What worked

| Step | Result |
|---|---|
| `globus-cli` install | **3.43.0** (globus-sdk 4.9.0) into the `flower` conda env |
| `globus login` | Succeeded (JPL SSO, `--no-local-server`) |
| Identity | **`gdoran@jpl.nasa.gov`** — exactly one identity, no linked others |
| STRI read access | **Granted and working** on that identity |
| `data_access` consent | **Never needed.** `globus login` alone sufficed for `globus ls` |
| Source path | **Correct exactly as the doc states** — no guest-collection root surprise |
| Destination | `/scratch/tree-monitoring/stri/globus/RGB/` created, verified writable |

### The source tree (measured)

`/UAVSHARE/paula_mavic_products/Aligned/Product_global/RGB/`

- **96 files**, all `.tif`, **flat** (no subdirectories)
- **355 GiB** total (381,372,667,670 bytes)
- Sizes **3.1 – 4.9 GiB**, median 3.7 GiB, mean 3.8 GiB
- Dates **2024-03-06 → 2026-01-20**
- Naming: `BCI_50ha_<YYYY_MM_DD>_M3M_aligned_global_RGB.tif`

Consequences:

- **Well under §0 rule 4's multi-TB threshold** — no HPC volume consult needed.
  `/gpfs` has 753 TB free.
- **`--sync-level mtime` is clearly right.** 96 files at ~3.8 GiB means per-file
  overhead is irrelevant; `checksum` would re-read 355 GiB on both sides every
  run for no benefit.
- Sibling dirs confirmed present and correctly excluded: `Product_global_COG/`,
  `Product_cropped/`, `aux_files/`, and under `Product_global/` also
  `COREG_RESULT/`, `DSM/`, `MS/`.
- `aux_files/BCI_50ha_big_shape_all.gpkg` is only **104 KiB** — the only small
  file anywhere nearby, and therefore the right probe file. **No small file
  exists in `RGB/` itself**, contrary to what the handoff's §5 "transfer one
  small file first" step assumes.

---

## Why Route A failed — precise diagnosis

Per §0 rule 5, the failure stage matters. It failed at **relay setup**, *after*
successful authentication and endpoint creation, and *before* any data transfer.

1. `globusconnectpersonal -setup --no-gui` requires `--name`; the doc omits this.
   Correct invocation: `-setup --no-gui -n gattaca2` (or answer the prompt).
2. Authentication **succeeded**, and the endpoint **was created**:
   - name `gattaca2`, id **`fd70fed1-a261-11f1-abf3-0ee7ef9370d9`**
   - it still exists, and reports **`GCP Connected: False`**
3. Setup then failed at `relaytool`:
   ```
   RelayToolFailureError: ('relaytool setup failed', CompletedProcess(
       args='.../gt_amd64/bin/relaytool', returncode=1, stdout=b'', stderr=b''))
   ```
4. `relaytool` connects to `relay.globusonline.org:2223` **over SSH** using GCP's
   bundled client. Testing that path directly:
   ```
   debug1: Connecting to relay.globusonline.org [54.237.254.198] port 2223.
   debug1: Connection established.
   Connection timed out during banner exchange
   ```
   TCP connects, then the SSH banner exchange times out — the signature of a
   firewall that permits the SYN and drops application payload. **Confirmed by
   the user: this machine cannot make outgoing SSH connections.**
5. No usable local config resulted — `~/.globusonline/lta/client-id.txt` is
   absent, so `globus endpoint local-id` reports
   "No Globus Connect Personal installation found."

Because the relay is mandatory for GCP and is SSH-based, **no amount of
`config-paths` or firewall work on ports 50000–51000 will help.** GCP is simply
unusable here.

Incidental: `relay.globusonline.org` returns both A and AAAA records, and the
node has a private IPv6 ULA with **no IPv6 default route**. Not the cause (the
bundled ssh chose IPv4 and reached it), but a latent hazard for any tool that
prefers IPv6.

### Leftover state to be aware of

- Endpoint `fd70fed1-a261-11f1-abf3-0ee7ef9370d9` (`gattaca2`) is **registered
  but non-functional**. Delete it with
  `globus endpoint delete fd70fed1-a261-11f1-abf3-0ee7ef9370d9` if Route A is
  abandoned for good, so it doesn't mislead later.
- `~/.globusonline/lta/` holds only `register.log` and relay keys. Harmless.
- `config-paths` was **never created** — the §4 step was never reached.

---

## Why Route C should work

The STRI collection has a **live, functioning HTTPS interface**, all on port 443:

```
https_server                = https://g-2a8c53.6c70c9.a567.data.globus.org
tlsftp_server               = tlsftp://g-2a8c53.6c70c9.a567.data.globus.org:443
gcs_version                 = 5.4.98
high_assurance              = False
authentication_timeout_mins = None      (no forced periodic reauth)
```

Note `tlsftp_server` is on **443, not 50000–51000** — GCS 5.4 uses TLS-FTP over
443, so the doc's concern about the GridFTP port range is partly obsolete.

An unauthenticated `GET` of the probe file returned a well-formed
`307 Temporary Redirect` to `auth.globus.org` with full CORS and
`Accept-Ranges: bytes` headers. That proves the complete HTTPS request/response
cycle works from this node and only needs a bearer token. `Accept-Ranges` also
means **HTTP Range requests are supported**, so resumable fetches of the ~4 GiB
rasters are achievable.

Verified reachable over 443 from `cn003`: `auth.globus.org` (401),
`transfer.api.globus.org` (400), STRI data node (307), `pypi.org` (200).

### Verified end to end (2026-08-27)

**No native-app registration was needed** — contrary to §5c. The script
authenticates as the Globus CLI's own registered native client
(`95fdeba8-fac2-42bd-a357-e068d82ff78e`), requesting the Transfer scope and the
collection's `https` scope as two top-level scopes. Auth returns a token per
resource server, the HTTPS one keyed by the collection UUID. Refresh token is
cached at `~/.globus_https_sync_tokens.json` (mode 0600), so runs after the first
are unattended and there is **no 14-day reauth** (the collection sets
`authentication_timeout_mins = None`).

Probe: `aux_files/BCI_50ha_big_shape_all.gpkg`

- fetched in 11 s, **106,496 bytes — byte-exact** against the source listing
- `file` confirms a valid OGC GeoPackage; opens in sqlite3 with 15 tables and a
  `BCI_50ha_big_shape_all` features layer
- source mtime preserved (Jul 24), no `.part` residue
- **second run transferred zero bytes** — the idempotency check §5 insists on

### SDK gotchas found the hard way (globus-sdk 4.9.0)

- `globus_sdk.scopes.TransferScopes.all` **is already a `Scope`**, not a string.
  Wrapping it in `Scope(...)` raises `TypeError: argument of type 'Scope' is not
  iterable`. Its `__str__` renders as a plain scope string, which makes this
  easy to miss.
- `globus_sdk.scopes` is **lazily imported**; `import globus_sdk` alone leaves
  `globus_sdk.scopes` unresolved (`AttributeError`). Bind it explicitly.
- Scope objects are **mutated in place** by `with_dependency`, so the shared
  module-level object must be deep-copied before use.

### Fixes applied over the §5c skeleton

The doc's skeleton is a sketch, not working code. Beyond the SDK issues above:

1. **`NameError` in its sync check.** It tests
   `local.stat().st_mtime >= src_mtime` in a branch where `src_mtime` is not yet
   bound — it is only assigned later by a walrus inside the `os.utime` call. It
   would raise on the first already-present file.
2. **Timezone bug.** Globus returns `last_modified` as
   `2026-07-12 22:44:05+00:00`; comparing a naive parse against local-time
   `st_mtime` causes spurious re-fetches. Naive values are now treated as UTC.
3. **Resumable transfers.** The collection advertises `Accept-Ranges: bytes`, so
   an interrupted fetch resumes from its `.part` file via a `Range` request
   instead of restarting — worth a lot at ~3.8 GiB/file. Handles a server that
   ignores `Range` by restarting cleanly.
4. **Size verification before rename.** Output is checked against the size from
   the Transfer listing, so a truncated transfer fails loudly rather than
   leaving a plausible-looking corrupt raster. One failure no longer aborts the
   run.
5. **Token refresh mid-run.** Snapshotting the bearer header once (as the
   skeleton does) breaks a multi-hour sync when the ~48 h access token expires.
   A `requests.auth.AuthBase` re-reads it per request so refresh takes effect.
6. **Parallel fetches** with a per-thread `Session` (`requests.Session` is not
   thread-safe), giving the 6× speedup tabulated above.

---

## Corrections to the handoff document

1. **§3 Step 4a — `globus endpoint show $SRC_ID` fails** on CLI 3.43.0 for a GCS
   v5 guest collection: *"Expected ... to be an endpoint ID. Instead, found it
   was of type 'Globus Connect Server v5 Guest Collection'."* It suggests
   `globus gcs collection show`, but that then demands a `manage_collections`
   consent we don't need. Use `globus api transfer GET /endpoint/$SRC_ID`
   instead — it works and returns more.
2. **§3 Step 4b — the consent step never triggered.** The doc calls this "most
   likely place to get stuck"; in practice `globus login` covered it.
3. **§3 Step 1 — `-setup` needs `-n <name>`**, and `--no-gui` is required on a
   headless node (the bundled client is Tcl/Tk).
4. **§5 — "transfer one small file first" is not directly possible.** The
   smallest file in `RGB/` is 3.1 GiB. Use
   `aux_files/BCI_50ha_big_shape_all.gpkg` (104 KiB) instead.
5. **§8 flag spellings all confirmed correct** on CLI 3.43.0: `--encrypt-data`,
   `--preserve-mtime` (alias of `--preserve-timestamp`), `--sync-level`,
   `--delete-destination-extra`, `--jmespath`/`--jq`, `-F/--format`, and
   `globus timer` all exist as documented.
6. **§2's port worry is misdirected.** GCS 5.4 serves data over 443
   (`tlsftp_server`), and the actual blocker is outbound **SSH** for the GCP
   relay — a mechanism the doc never mentions.
7. **`globus login --no-local-server` is required** on a compute node; the
   default flow waits on a local callback that can't be reached.

## Environment notes

- The CLI lives in the **`flower` conda env**, not `pip install --user`:
  `/home/gdoran/miniconda3/envs/flower/bin/globus`. Any script must activate
  `flower` or use that absolute path.
- Installing it **upgraded two shared deps in `flower`**: `requests`
  2.32.5→2.34.2, `click` 8.2.1→8.4.2. `pip check` reports no broken
  requirements, but note this if something else in the env misbehaves.
- The `globus/connectpersonal/3.2.5` module exports the three JPL collection
  UUIDs exactly as the doc's §1 table claims.
- `mmlsquota -j tree-monitoring gpfs` fails ("no such fileset"), so a project
  quota could not be confirmed — only that the filesystem has 753 TB free.

## Current run

Full sync launched 2026-08-27 ~15:50 PDT:

```bash
cd ~/Documents/tree-flowering
nohup ~/miniconda3/envs/flower/bin/python scripts/globus_https_sync.py -j 8 \
  > ~/logs/stri-rgb-sync.log 2>&1 &
```

pid 2848132, 8 streams, 13.7 MiB/s, ETA ~23:10 PDT. Progress:
`tail -f ~/logs/stri-rgb-sync.log`.

**Caveat: it runs inside Slurm job `19357244` on `cn003`.** If that job ends the
sync dies with it. Not fatal — `.part` files mean a re-run resumes rather than
restarts — but for an unattended production sync this belongs in its own batch
job, or under `screen`/`tmux` on a node that will persist.

On completion, verify: 96 `.tif` files present, sizes match the source listing,
no `.part` files remain.

## Still open

- **§6.1 HPC ticket** — not yet filed. Now has a sharper question to ask:
  *GCP cannot work on Gattaca2 because outbound SSH to the Globus relay
  (`relay.globusonline.org:2223`) is blocked.* Ask whether that can be permitted,
  or confirm Route C/D as the sanctioned approach. Route D (NFS mounted on both
  DTN 1 and Gattaca2) is now considerably more attractive.
- **§6.3 user confirmations** — the Smithsonian PII/SPII notice, and terms of
  use / attribution for the share (`valdese@si.edu`).
- **STRI update cadence** — needed before choosing any sync interval. The file
  dates suggest roughly weekly flights, with the newest from 2026-01-20.
- Whether the head node (`gattaca2-hn1`) has different egress. Worth one test,
  though a cluster-wide SSH egress block is unlikely to differ there.
- **Scheduling** (§5a/§5b) — deliberately deferred. Note the cloud-timer option
  in §5a is moot: it requires a GCP destination, which cannot exist here. Any
  schedule must be cluster-driven (cron or a Slurm batch job) wrapping
  `globus_https_sync.py`, which is safe to re-run since it skips current files.
- **`/scratch` purge policy** — still unconfirmed, and now more relevant: 355 GiB
  that silently erodes would be re-fetched at ~7 h a time.
