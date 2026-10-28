# Handoff: gattaca2 runs for offset augmentation and the calibrated seed repeat

Written 2026-10-27. These runs follow from the results in
[planet_temporal_generalization.md](planet_temporal_generalization.md), section
"Why calibration did not help". Read [gattaca2_cluster.md](gattaca2_cluster.md)
first for the 2-GPU limit and queue waits. Run every command from the repo
root on gattaca2 unless it is marked Mac.

## Why these runs

The calibrated Mask R-CNN loses 0.12 pooled IoU when a 1 SD offset error is
added at test time. The TOA model loses at most 0.007. Under the fixed stretch
a 1 SD offset is about a third of the input range, and the training
ColorJitter is multiplicative, so it never shows the model shifts that large.
Borrowed offsets carry about 1 SD of error, which fits the recall the
calibrated model loses on borrowed-offset chips and scenes. Separately, a
second TOA seed moved pooled `map_50` and IoU by up to 0.022, but the
calibrated arm has only one run.

| Run | Config | Question |
|---|---|---|
| `cal_offjit1` | `config/temporal_cv_20261012_cal_offjit1.sh`: calibrated, `--offset-jitter 1.0` | Does training with offset errors of the size calibration leaves remove the sensitivity, without costing skill? |
| `cal_offjit05` | `config/temporal_cv_20261012_cal_offjit05.sh`: calibrated, `--offset-jitter 0.5` | The same, gentler. At σ = 1, one sample in eight is shifted by more than 1.5 SD, which pushes it close to saturation. |
| `cal_seed1` | `config/temporal_cv_20261012_cal_seed1.sh`: calibrated, `--seed 1` | H8 for the calibrated arm: is calibration's IoU and precision gain larger than the spread of both arms? |
| sensitivity | `scripts/slurm/radiometric_sensitivity.sh` | Offset response from −2 to +2 SD for TOA, calibrated, and both offset-jitter runs |
| `nojit_seed1`, `cal_nojit_seed1` (optional) | `config/temporal_cv_20261012_{nojit,cal_nojit}_seed1.sh` | H1 with two seeds per arm |

`--offset-jitter S` (`train_planet_image_maskrcnn.py`) adds k times the
across-scene SD of the fitted Planet–MODIS offsets to every band of each
training sample. The SDs are 0.0158 / 0.0149 / 0.0158 / 0.0355 in
reflectance for blue / green / red / NIR. k ~ N(0, S) is drawn once per
sample and shared by the bands, as in the sensitivity test. The offset is
added before the fixed stretch, and the flag requires `--fixed-stretch`. It is
stored in `params` as `offset_jitter` and `offset_sd`. The SDs are read
through `util.offset_sd()` from
`/scratch/tree-monitoring/results/planet_calibration/planet_scene_offsets.csv`.

## 1. Preconditions

```bash
git pull && git log --oneline -3        # the commit adding --offset-jitter, or later

# The scene offsets are already synced (they were used for sensitivity_cal.csv).
# This should print about [0.0158 0.0149 0.0158 0.0355]:
(cd scripts && conda run -n flower python -c "from util import offset_sd; print(offset_sd())")

# The existing fold checkpoints the sensitivity jobs need:
ls $HOME/Documents/tree-flowering/202610_temporal/fold0/epoch_012.pth \
   $HOME/Documents/tree-flowering/202610_temporal_cal/fold0/epoch_012.pth
```

## 2. Training runs, in priority order

Use a fresh shell, because each config exports its settings into the shell
that sources it. The offset-jitter runs chain a sensitivity job after their
eval. That job uses the run's own best epoch.

```bash
for C in cal_offjit1 cal_seed1 cal_offjit05; do
  source config/temporal_cv_20261012_$C.sh
  echo "$C: $TEMPORAL_TRAIN_ARGS -> $TEMPORAL_RUNS"
  CV=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_cv.sh)
  EV=$(sbatch --parsable -A tree-monitoring --dependency=afterok:$CV scripts/slurm/planet_temporal_eval.sh)
  case $C in cal_offjit*)
    sbatch -A tree-monitoring --dependency=afterok:$EV scripts/slurm/radiometric_sensitivity.sh ;;
  esac
done
```

Each Mask R-CNN run is six folds of about 35–40 min plus a 25-min eval, so
about 4 GPU-hours, or 2 h of wall time on both GPUs. With 2 GPUs the three
runs take most of a day, longer if the partition is full. Each eval writes
`temporal_cv.json` and `temporal_cv_windows.csv` into `$TEMPORAL_RUNS`. Each
sensitivity job writes `sensitivity_e<epoch>.csv` there.

As before, `squeue` shows array tasks waiting with `QOSMaxGRESPerUser`, and
dependent jobs with `Dependency`. `DependencyNeverSatisfied` means an upstream
task failed. Look at `logs/temporal_cv_<array>_<task>.err`.

## 3. Sensitivity of the existing TOA and calibrated runs on the same grid

These jobs use the existing epoch-12 checkpoints, so they need no dependency.
They put both reference models on the same seven-point grid as the
offset-jitter runs.

```bash
( source config/temporal_cv_20261012_cal.sh
  SENS_EPOCH=12 sbatch -A tree-monitoring --export=ALL scripts/slurm/radiometric_sensitivity.sh )
( source config/temporal_cv_20261012.sh
  SENS_EPOCH=12 SENS_CAL_CHIPS=/scratch/tree-monitoring/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated \
    sbatch -A tree-monitoring --export=ALL scripts/slurm/radiometric_sensitivity.sh )
```

The calibrated job at k = −1, 0, 1 has to reproduce `sensitivity_cal.csv`
(IoU 0.239 / 0.361 / 0.313). The TOA job has to reproduce the TOA table in
"Test-time offset sensitivity" (IoU 0.351 / 0.352 / 0.345).

## 4. Optional, if GPUs are idle: H1 with a second seed

```bash
for C in nojit_seed1 cal_nojit_seed1; do
  source config/temporal_cv_20261012_$C.sh
  CV=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_cv.sh)
  sbatch -A tree-monitoring --dependency=afterok:$CV scripts/slurm/planet_temporal_eval.sh
done
```

## 5. Bring the results back (Mac)

```bash
for R in cal_offjit1 cal_offjit05 cal_seed1 nojit_seed1 cal_nojit_seed1; do
  mkdir -p /Volumes/Earth03/flower/202610_temporal_$R
  rsync -a "gattaca2:Documents/tree-flowering/202610_temporal_$R/temporal_cv*" \
    "gattaca2:Documents/tree-flowering/202610_temporal_$R/sensitivity_*" \
    /Volumes/Earth03/flower/202610_temporal_$R/
done   # the two optional runs fail here if they were not run
rsync -a gattaca2:Documents/tree-flowering/202610_temporal/sensitivity_e12.csv /Volumes/Earth03/flower/202610_temporal/
rsync -a gattaca2:Documents/tree-flowering/202610_temporal_cal/sensitivity_e12.csv /Volumes/Earth03/flower/202610_temporal_cal/
```

## 6. Analysis (Mac)

```bash
F=/Volumes/Earth03/flower
M=$F/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated/chips.csv
# Offset jitter against the calibrated parent, with offset-source strata
python scripts/compare_temporal_cv.py \
  --run cal=$F/202610_temporal_cal/temporal_cv.json \
  --run cal_offjit1=$F/202610_temporal_cal_offjit1/temporal_cv.json \
  --windows cal=$F/202610_temporal_cal/temporal_cv_windows.csv \
  --windows cal_offjit1=$F/202610_temporal_cal_offjit1/temporal_cv_windows.csv \
  --chip-meta $M --epoch 12
# Offset response of all four models (each offset-jitter CSV is named after
# its run's best epoch)
O1=$(ls $F/202610_temporal_cal_offjit1/sensitivity_e*.csv)
O05=$(ls $F/202610_temporal_cal_offjit05/sensitivity_e*.csv)
python scripts/plot_radiometric_sensitivity.py \
  --run TOA=$F/202610_temporal/sensitivity_e12.csv \
  --run calibrated=$F/202610_temporal_cal/sensitivity_e12.csv \
  --run "offset jitter 1=$O1" --run "offset jitter 0.5=$O05" \
  --plot figs/planet_temporal/radiometric_sensitivity_offjit.png
```

Do the same for `cal_offjit05`, for `toa` vs `cal_seed1`, and for `cal` vs
`cal_seed1`. How to read them:

- **Offset jitter works** if three things hold:
  - Its pooled IoU loss at ±1 SD drops from 0.05–0.12 to near the TOA
    model's (≤ about 0.02).
  - Its pooled `map_50` and IoU stay within the seed spread of the
    calibrated run (about 0.02), using the whole epoch grid, not only
    epoch 12.
  - Recall on the 12 borrowed-offset chips recovers toward TOA's 0.37,
    from the calibrated run's 0.30.
- If it works, the real test is the apply run against the drone labels. It
  has 54 borrowed-offset scenes, against 12 chips in the CV. Train the final
  model with the offset-jitter config (`scripts/slurm/planet_temporal_final.sh`,
  with the apply settings restored from `temporal_cv_20261012_cal.sh` and a
  new `APPLY_OUT`). Then repeat "Calibrated apply run against the drone
  labels".
- **Calibrated seed (H8).** Calibration's IoU and precision gain counts as
  real if it is larger than the spread between the two calibrated runs and
  between the two TOA runs, at most epochs.
- **H1 with two seeds.** This holds if `nojit_seed1` again drops more than
  `cal_nojit_seed1`, against their parents.

Record the results in "Test-time offset sensitivity" and "Augmentation
ablation and repeat run". Then delete this file, or mark it done.
