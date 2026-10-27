# Handoff: gattaca2 runs for "Why calibration did not help"

Written 2026-10-26. These are the cluster runs still pending from
[planet_temporal_generalization.md](planet_temporal_generalization.md), section
"Why calibration did not help" (subsections "Augmentation ablation and repeat
run" and "DINOv3 backbone"), plus the calibrated half of "Test-time offset
sensitivity". Read [gattaca2_cluster.md](gattaca2_cluster.md) first for the
2-GPU limit and queue waits. Run every command from the repo root on gattaca2
unless it is marked Mac.

## 0. Clear the failed first attempt (2026-10-26)

The first submission (jobs 20001851–20001860) failed for two reasons. The
three ablation runs inherited the DINOv3 trainer from the shell, which
cbfd084 fixes. The DINOv3 runs found no weights under
`/scratch/tree-monitoring/hf_hub`. Cancel whatever is left of it:

```bash
scancel 20001851 20001852 20001853 20001854 20001855 20001856 20001857 20001858 20001859 20001860
```

## 1. Preconditions

```bash
git pull && git log --oneline -1        # cbfd084 or later

# Mac: the gated DINOv3 weights (plain files, 2.2 GB). Keep the trailing
# slashes, or the files land in hf_hub/hf_hub/.
rsync -aL /Volumes/Earth03/flower/hf_hub/ gattaca2:/scratch/tree-monitoring/hf_hub/

# gattaca2: six files per model, including model.safetensors
ls /scratch/tree-monitoring/hf_hub/models--facebook--dinov3-vitl16-pretrain-*/snapshots/*/
```

## 2. Temporal CV and eval for all five runs

Use a fresh shell, because each config exports its settings into the shell
that sources it.

```bash
for C in nojit cal_nojit seed1 dinov3sat dinov3web; do
  source config/temporal_cv_20261012_$C.sh
  echo "$C: $TEMPORAL_TRAIN_SCRIPT -> $TEMPORAL_RUNS"
  CV=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_cv.sh)
  sbatch -A tree-monitoring --dependency=afterok:$CV scripts/slurm/planet_temporal_eval.sh
done
```

| Config | Trainer | Output (`$HOME/Documents/tree-flowering/`) | Tests |
|---|---|---|---|
| `nojit` | Mask R-CNN, TOA, no ColorJitter | `202610_temporal_nojit` | H1 |
| `cal_nojit` | Mask R-CNN, calibrated, no ColorJitter | `202610_temporal_cal_nojit` | H1 |
| `seed1` | Mask R-CNN, TOA, `--seed 1` | `202610_temporal_seed1` | H8: run-to-run spread |
| `dinov3sat` | DINOv3 ViT-L/16, SAT-493M, frozen | `202610_temporal_dinov3sat` | H3 |
| `dinov3web` | DINOv3 ViT-L/16, LVD-1689M, frozen | `202610_temporal_dinov3web` | H3 control |

What to expect:

- `squeue` shows array tasks waiting with `QOSMaxGRESPerUser`, and evals with
  `Dependency`. Both are normal.
- An eval showing `DependencyNeverSatisfied` means a task of its CV array
  failed. Look at `logs/temporal_cv_<array>_<task>.err`.
- Each Mask R-CNN run takes about 2–3 h of GPU time. GPU time per DINOv3 fold
  has not been measured.
- With 2 GPUs, the five runs take one to two days, longer if the partition is
  full.

Each eval writes `temporal_cv.json` and `temporal_cv_windows.csv` into its
output directory.

## 3. Test-time offset sensitivity of the calibrated Mask R-CNN

This needs the epoch-12 checkpoints of the existing calibrated CV
(`202610_temporal_cal/fold*`), which exist only on gattaca2. It also needs the
scene offsets, which live outside the repo with the other calibration tables,
so sync that directory first (Mac):

```bash
rsync -a /Volumes/Earth03/flower/results/planet_calibration/ \
  gattaca2:/scratch/tree-monitoring/results/planet_calibration/
```

`radiometric_sensitivity.py` reads
`/scratch/tree-monitoring/results/planet_calibration/planet_scene_offsets.csv`
by default on gattaca2 (`util.calibration_results()`), so the command below
needs no `--offsets-csv`. Pull 2026-10-27's commit first, which made that the
default. Then submit on gattaca2. It takes a GPU slot, so it queues behind the runs in
step 2. Use CUDA: on MPS it hits a float64 error in the mask head.

```bash
sbatch -A tree-monitoring -p gpu --gres=gpu:1 --cpus-per-task=2 --mem=32G --time=02:00:00 \
  -J radsens -o logs/radsens_%j.out -e logs/radsens_%j.err --wrap "
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flower &&
python scripts/radiometric_sensitivity.py \
  --chip-dir /scratch/tree-monitoring/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated \
  --folds config/temporal_folds_20261012.csv \
  --run-template '$HOME/Documents/tree-flowering/202610_temporal_cal/fold{fold}' \
  --units reflectance --device cuda --k -1,0,1 \
  --out-csv $HOME/Documents/tree-flowering/202610_temporal_cal/sensitivity_cal.csv"
```

Check: the k = 0 row should reproduce the calibrated run's epoch-12 pooled
numbers in `202610_temporal_cal/temporal_cv.json` (map_50 0.346, IoU 0.361).
The TOA result to compare against is in "Test-time offset sensitivity": at
most 0.007 pooled IoU lost at ±1 SD.

## 4. Bring the results back (Mac)

```bash
for R in nojit cal_nojit seed1 dinov3sat dinov3web; do
  rsync -a gattaca2:Documents/tree-flowering/202610_temporal_$R/temporal_cv\* \
    /Volumes/Earth03/flower/202610_temporal_$R/
done
rsync -a gattaca2:Documents/tree-flowering/202610_temporal_cal/sensitivity_cal.csv \
  /Volumes/Earth03/flower/202610_temporal_cal/
```

Then compare each run against the TOA baseline, for example:

```bash
F=/Volumes/Earth03/flower
python scripts/compare_temporal_cv.py \
  --run toa=$F/202610_temporal/temporal_cv.json \
  --run dinov3sat=$F/202610_temporal_dinov3sat/temporal_cv.json \
  --windows toa=$F/202610_temporal/temporal_cv_windows.csv \
  --windows dinov3sat=$F/202610_temporal_dinov3sat/temporal_cv_windows.csv \
  --chip-meta $F/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated/chips.csv
```

How to read them:

- **Jitter (H1).** It was standing in for calibration only if `nojit` drops
  clearly more than `cal_nojit`, relative to the toa-vs-`seed1` spread.
- **DINOv3 (H3).** Read the whole epoch grid, because its best epoch need not
  be 12. If the frozen SAT-493M backbone is competitive, the next run is
  phase C2: add `--finetune-blocks 4` to `TEMPORAL_TRAIN_ARGS`.

Record the results in the two subsections named above. Then delete this file,
or mark it done.
