# Handoff: apply the offset-jitter model against the drone labels

Written 2026-10-28. This run follows from "Offset augmentation (2026-10-28)" in
[planet_temporal_generalization.md](planet_temporal_generalization.md#offset-augmentation-2026-10-28).
Read [gattaca2_cluster.md](gattaca2_cluster.md) first for the 2-GPU limit and
queue waits. Run every command from the repo root on gattaca2, unless it is
marked Mac.

## Why this run

Training the calibrated Mask R-CNN with random offsets (`--offset-jitter 1`)
fixed most of the calibrated model's sensitivity to offset errors in the
temporal CV:
- It has the best pooled `map_50` of any run (0.361 at epoch 17).
- It has the best recall at matched detection counts.
- A 1 SD test-time offset costs 0.035 / 0.011 in `map_50`, against
  0.152 / 0.068 for the calibrated model.
- On the 12 borrowed-offset CV chips, recall rises to 0.41, against 0.30 and
  0.35 for the two calibrated seeds.

Two things are still open:
- **Too few borrowed-offset chips.** The CV has only 12, so the CV cannot
  settle whether the loss on borrowed-offset scenes is fixed.
- **Extra false detections.** A negative offset still nearly doubles the
  model's detections. The CV scores keep every detection down to a score of
  0.05, so they count all of them.

The apply comparison answers both. It has 54 borrowed-offset scenes in the
30-day envelope, against 12 chips. It also counts a crown as an event only
under detections scoring 0.5 or more. That is where the calibrated model lost
to TOA: on borrowed-offset scenes, precision fell from 0.553 to 0.459 and
recall from 0.299 to 0.209.

| Step | What | Output |
|---|---|---|
| Final model | `config/temporal_cv_20261012_cal_offjit1.sh`, all 142 chips, 17 epochs (the run's CV-selected epoch) | `$HOME/Documents/tree-flowering/202610_temporal_cal_offjit1/final_e17/` |
| Apply | the 798 coregistered non-drone-date scenes, calibrated (`planet_clipped_calfill/4band`), with the TOA coregistration | `/scratch/tree-monitoring/202610_temporal_apply_cal_offjit1/` |
| Compare (Mac) | crown agreement with the drone labels, against the TOA and calibrated apply runs | `202610_temporal_apply_cal_offjit1/crown_agreement_*` on Earth03 |

The config now keeps the parent's apply settings but sets its own `APPLY_OUT`.
The settings are `APPLY_PLANETDIR`, `APPLY_OCMDIR`, `APPLY_COREG_JSON` and
`APPLY_CROWNS`. The TOA and calibrated apply outputs are untouched.

## 1. Preconditions

```bash
git pull && git log --oneline -3   # the commit adding this file, or later

# The CV must have selected epoch 17 (this prints 17):
python -c "import json; print(json.load(open('$HOME/Documents/tree-flowering/202610_temporal_cal_offjit1/temporal_cv.json'))['best_epoch'])"

# Inputs already on scratch from the calibrated apply run:
ls -d /scratch/tree-monitoring/planet_clipped_calfill/4band \
      /scratch/tree-monitoring/planet_clipped/ocm
ls /scratch/tree-monitoring/202610_temporal_apply/coreg_nearest_drone.json

# The output must not exist yet:
ls -d /scratch/tree-monitoring/202610_temporal_apply_cal_offjit1   # "No such file"

# The offset SDs the training reads (about [0.0158 0.0149 0.0158 0.0355]):
(cd scripts && conda run -n flower python -c "from util import offset_sd; print(offset_sd())")
```

## 2. Train the final model and apply it

Use a fresh shell, because the config exports its settings into the shell that
sources it.

```bash
source config/temporal_cv_20261012_cal_offjit1.sh
export TEMPORAL_FINAL_EPOCHS=$(python -c "import json; print(json.load(open('$TEMPORAL_RUNS/temporal_cv.json'))['best_epoch'])")
echo "$TEMPORAL_FINAL_EPOCHS: $TEMPORAL_TRAIN_ARGS -> $APPLY_OUT"
FINAL=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_final.sh)
sbatch -A tree-monitoring --dependency=afterok:$FINAL scripts/slurm/planet_apply.sh
```

**Timing.** The calibrated final model took 12 min for 12 epochs, so this one
should take about 20 min. The apply array has 4 shards of about 8 min each;
with 2 GPUs they run two at a time.

**If a job fails.** `squeue` shows `DependencyNeverSatisfied` if the final job
failed. Look at `logs/temporal_final_<id>.err`. An apply shard that times out
can be resubmitted, because finished scenes are skipped.

When all four shards have finished, join the per-scene files. This step
needs only the CPU:

```bash
source config/temporal_cv_20261012_cal_offjit1.sh
python scripts/apply_planet_maskrcnn.py concat $APPLY_OUT
ls $APPLY_OUT/crown_timeseries.csv $APPLY_OUT/scene_summary.csv
```

**Check.** `scene_summary.csv` should have 792 rows: the 798 scenes, minus the
6 Dove Classic scenes that have no SR. Check the model column:

```bash
python -c "import pandas as pd; s=pd.read_csv('$APPLY_OUT/scene_summary.csv'); print(len(s), s.model.unique(), s.model_epoch.unique())"
```

## 3. Bring the results back (Mac)

```bash
rsync -a gattaca2:/scratch/tree-monitoring/202610_temporal_apply_cal_offjit1/ \
  /Volumes/Earth03/flower/202610_temporal_apply_cal_offjit1/
rsync -a gattaca2:Documents/tree-flowering/202610_temporal_cal_offjit1/final_e17 \
  /Volumes/Earth03/flower/202610_temporal_cal_offjit1/
```

## 4. Analysis (Mac)

```bash
F=/Volumes/Earth03/flower
A=$F/202610_temporal_apply_cal_offjit1
python scripts/plot_crown_species_comparison.py $A/crown_species --apply-dir $A
python scripts/make_crown_timeseries_html.py $A/crown_timeseries_viewer.html --apply-dir $A
# Against TOA (the current reference) and against the calibrated run
python scripts/compare_crown_agreement.py $A/crown_agreement_toa_vs_offjit1 \
  --toa-dir $F/202610_temporal_apply --cal-dir $A \
  --cal-label 'MODIS-calibrated, offset jitter 1' --cal-color '#8a5bd6'
python scripts/compare_crown_agreement.py $A/crown_agreement_cal_vs_offjit1 \
  --toa-dir $F/202610_temporal_apply_cal --cal-dir $A \
  --toa-label 'MODIS-calibrated, fixed stretch' --toa-color '#1baf7a' \
  --cal-label 'MODIS-calibrated, offset jitter 1' --cal-color '#8a5bd6'
```

`compare_crown_agreement.py` always names the first run `toa` and the second
`cal` in its column names and its "cal - toa" intervals, whatever the runs
are. Also compare the detection counts in `scene_summary.csv`
(`n_detections` and `n_detections_above_thresh`) with the calibrated run's.
Offset jitter should add detections that score below 0.5, but few above it.

How to read the results. The reference numbers are in "Calibrated apply run
against the drone labels". All are for in-envelope scenes, all species:

| Stratum | TOA precision / recall | Calibrated precision / recall |
|---|---|---|
| All, 202 scenes | 0.561 / 0.392 | 0.542 / 0.368 |
| Own MODIS fit, 148 | 0.562 / 0.409 | 0.552 / 0.398 |
| Borrowed offset, 54 | 0.553 / 0.299 | 0.459 / 0.209 |

- **Offset jitter fixes the borrowed-offset loss** if its borrowed-offset
  precision and recall return to TOA's, with the scene-bootstrap interval
  against the calibrated run excluding zero, and its own-fit row stays at
  least as good as the calibrated run's.
- **It becomes the reference apply output** if, in addition, its in-envelope
  all-species precision and recall match or beat TOA's (intervals against TOA
  covering zero or positive). The outside-envelope stratum, 48 scenes and
  mostly 2020, should not get worse. That is where calibration hurt precision
  (0.489 to 0.436).
- **If borrowed-offset scenes recover but not to TOA's level,** the next
  option is the one in "Calibrated apply run against the drone labels": use a
  per-image stretch only for borrowed-offset scenes.
- **The epoch is not matched.** The TOA and calibrated final models use epoch
  12, and this one uses epoch 17, each the epoch its CV selected. If the
  result is close, a `final_e12` of the offset-jitter config would separate
  the two effects. In the CV, it scores `map_50` 0.355 against 0.361.

Record the results as a new subsection after "Calibrated apply run against
the drone labels" in
[planet_temporal_generalization.md](planet_temporal_generalization.md). Add
the comparison figure's caption under "Figure captions", and update the
offset-augmentation verdict and "Where they stand". Then delete this file.
