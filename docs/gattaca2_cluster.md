# gattaca2: capabilities and limits

Notes on the JPL gattaca2 Slurm cluster as this project uses it, so they do not
have to be rediscovered. Checked 2026-10-17 with `sinfo`, `scontrol show
partition`, `sacctmgr show qos/assoc` and `df`. Re-check if a job behaves
unexpectedly.

## Partitions and nodes

| Partition | Nodes | Per node | Max time | Default time | Partition QOS |
|---|---|---|---|---|---|
| `compute` (default) | `cn[001-044]` | 64 CPU, 480 GB, no GPU | 10 days | — | `normal` |
| `debug` | `cn[001-044]` | same | 1 h | — | `debug` |
| `gpu` | `gn[01-03]` | 64 CPU, 480 GB, 2× A100 80 GB PCIe | 3 days | 3 h | `gpu` |
| `gpu-mig` | `gn04` | 64 CPU, 480 GB, 2× `3g.40gb` + 4× `2g.20gb` MIG slices | 3 days | — | `gpu-mig` |

The default memory is 1 GB per node (`DefMemPerNode=1024`) on both `compute`
and `gpu`, so always pass `--mem`. The default time on `gpu` is 3 h, so always
pass `--time`.

## Per-user limits (QOS `MaxTRESPerUser`)

- **`gpu`: 2 GPUs, 64 CPUs and 480 GB in total per user.** Jobs beyond that
  wait with reason `QOSMaxGRESPerUser`. An interactive GPU session (`srun
  --gres=gpu:1 ... bash`) counts against the limit, so while one is open only
  one batch GPU job runs at a time. A 6-task GPU array runs about one task at
  a time.
- `gpu-mig`: 16 CPUs, one `1g.20gb` slice and two `2g.20gb` slices.
- `normal` (compute): 512 CPUs and 4 TB per user, at most 512 jobs.
- `debug`: 24 CPUs and 192 GB per user, at most 5 jobs.
- `resv-gpu` (4 GPUs) and `resv` exist but need a reservation.

Every account (`rsp`, `tree-monitoring`, `ecopro-hpc`, ...) uses the `normal`
QOS, so switching `-A` does not raise a limit. The default account is `rsp`.
Pass `-A tree-monitoring` to charge this project.

**Working within the GPU limit.** When an interactive GPU session is open,
it is faster to run some work directly in that session's GPU
(`CUDA_VISIBLE_DEVICES` is already set) than to queue it all. Detach long runs
with `setsid nohup ... &`, because the session's own job end time (3 days at
most) kills them otherwise. If array tasks are moved out of Slurm this way,
`scancel` those tasks and update any `afterok` dependency on the array
(`scontrol update JobId=<id> Dependency=afterok:<array>_0:<array>_1...`),
otherwise the dependent job never starts.

**Queue waits.** The `gpu` partition is often full with other users' multi-day
jobs. On 2026-10-24 a 1-GPU job submitted at 11:19, with both of this user's
slots free, waited with reason `Resources` and a Slurm start estimate of
2026-10-26 05:48 (`squeue -j <id> --start`). Plan GPU work around waits of hours to days, not
minutes.

## Storage

| Path | What | Notes |
|---|---|---|
| `/scratch/tree-monitoring` | project data: chip sets, `planet_clipped/{4band,ocm}`, `stri/` drone orthos and crown maps, `results/` | GPFS (1.2 PB). Datasets moved here from `/scratch/ecopro-hpc/flower` on 2026-10-17 |
| `/scratch/tree-monitoring` (calibrated, 2026-10-23) | `20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated` (142 MODIS-calibrated chips, 2.7 GB) and `planet_clipped_calfill/4band` (2,138 calibrated 50ha clips, real files, 1.1 GB); apply output `202610_temporal_apply_cal/` | Copied from Earth03 with `rsync -aL` (calfill scenes are symlinks there) |
| `/scratch/tree-monitoring/hf_hub` (2026-10-25) | Hugging Face hub cache with the gated DINOv3 ViT-L/16 backbones (`facebook/dinov3-vitl16-pretrain-sat493m` and `-lvd1689m`, 1.1 GB each): `refs/` and `snapshots/` only, as plain files | Downloaded on the Mac and copied from Earth03 `flower/hf_hub`. Jobs read it with `HF_HUB_CACHE=/scratch/tree-monitoring/hf_hub HF_HUB_OFFLINE=1` (set by `config/temporal_cv_20261012_dinov3*.sh`), so no HF login is needed on the cluster |
| `/scratch/tree-monitoring/results/planet_calibration` (2026-10-27) | Planet-MODIS calibration tables: scene offsets, fits and matched pixels, evaluation, fixed-stretch metrics | Synced from Earth03 `flower/results/planet_calibration` with `rsync -a`. Scripts find it through `util.calibration_results()`, which uses the first of `/Volumes/Earth03/flower` and `/scratch/tree-monitoring` that exists (`TREE_DATA_ROOT` overrides) |
| `/home/gdoran` | repo and model runs (`~/Documents/tree-flowering/2026*_out*`, `202610_temporal/`, `202610_temporal_cal/`) | NFS, 76 TB shared |
| `/local` | node-local disk, ~680 GB free | Not shared between nodes; stage chip sets to `/local/gdoran_flower` |

**GPFS is slow for training reads.** Reading one 512 × 512 chip window off
GPFS takes 540–675 ms, against 14–17 ms from `/local`. Training straight from
`/scratch` leaves the GPU idle in `cxiWaitEventWait`. The `planet_temporal_*`
jobs stage chips with `scripts/slurm/stage_chips.sh`, which uses a per-node
lock and checks the copy by checksum. Python imports from the repo on `/home`
are also slow: importing the training script cold takes about a minute.

`du` on GPFS reports more than the files hold. For example, `planet_clipped`
shows 1.4 GB + 0.4 GB here against 1.2 GB + 0.26 GB on the Mac.

## Software

- Conda lives at `~/miniconda3`. Activate with
  `source ~/miniconda3/etc/profile.d/conda.sh; conda activate flower`.
- The gattaca2 `flower` env differs from `environment.yml`. It has torch
  2.8.0+cu128 and torchvision 0.23 (pip), not 2.12, and as of 2026-10-17
  numpy 2.4.0. `arosics`, `scikit-image` and `imageio` were added on
  2026-10-17. `omnicloudmask` is **not** installed (`pip install
  'omnicloudmask>=1,<2'`), so `cloud_mask_planet.py` cannot run here until it
  is.
- With about 100 pip packages in the env, a `conda install` takes about
  25 min to solve, and it may upgrade pip-installed packages such as numpy.
  Check `torch` and `numpy` afterwards.

## Data transfer

Globus to gattaca2 has had outages; see `globus-gattaca2-sync-handoff.md` in the repo root (not committed).
`rsync` from the Mac (`gattaca2:/scratch/tree-monitoring/...`) works.
