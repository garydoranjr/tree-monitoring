# Handoff: retrain the Planet Mask R-CNN on MODIS-calibrated chips (gattaca2)

> **Temporary document.** Delete it once both experiments have run and the
> results are recorded in the permanent docs. Step 8 says how.

## Why

The crown model takes each Planet chip through a per-image stretch: every band
is divided by its own 99.9th percentile. The chips are TOA, so path radiance,
haze and clouds make the same forest look different on every date. The
forest's input value ranges 0.28–0.91 across chips.

We now have Planet surface reflectance calibrated to MODIS, together with one
fixed stretch per band. The method, figures and caveats are in
[planet_radiometric_calibration.md](planet_radiometric_calibration.md#classifier-input-stretch).
With them, the same reflectance gives the same input on every date. Across
chips, the spread of the median input roughly halves, and crowns span 2–4×
more of the input range.

This handoff checks whether that helps the model, using both existing
benchmarks:

| Experiment | Earlier run (TOA, per-image stretch) | New run (calibrated, fixed stretch) |
|---|---|---|
| **Spatial (east/west)**: train on the left windows of the 142 chips, score the right windows of the 56 phantom 2020–23 chips | `full_mkl`: `~/Documents/tree-flowering/202610_full2_mkl_curated_maskrcnn_out_min064`, scored in `logs/headtohead_sweep_202610_mkl.json` | `full_mkl_cal`: `~/Documents/tree-flowering/202610_full2_mkl_cal_curated_maskrcnn_out_min064` |
| **Temporal CV**: 6 date-blocked folds, both windows | `~/Documents/tree-flowering/202610_temporal/temporal_cv.json` (best epoch 12, pooled mAP@50 0.354) | `~/Documents/tree-flowering/202610_temporal_cal/` |

**What stays the same.** The new chip set is
`20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated`. It keeps all of
the following from the earlier runs:
- the chip names and grids;
- the crown labels (`.mask.png`) and coregistration (`coreg_log.json`);
- the OCM sidecars;
- the folds and the curation;
- the training settings.

**What changes.** Only two things differ:
- the `.tif` pixel values are calibrated reflectance × 10⁴ instead of TOA DN;
- training adds `--fixed-stretch config/planet_fixed_stretch.yml`. The stretch
  is stored in the checkpoint as `params['stretch']`, and the apply, deploy and
  evaluation scripts read it from there. Old checkpoints have no `stretch` and
  keep the old behaviour.

**Offsets and haze.**
- **Offset sources.** 130 of the 142 chips have their own MODIS fit. The other
  12 use an offset borrowed from nearby dates.
- **Hazy chips.** Four chips are hazy and come out mostly saturated under the
  fixed stretch: 2021-12-27 `1067`, 2022-01-26 `2448` (`145645_68`), 2023-04-16
  `2498`, and 2024-12-17 `24b3`. They are **kept**, so the comparison stays
  one-to-one. Dropping them is a possible follow-up (see "After this").

## 0. Before starting (on the Mac, by the user)

The repo commits must be pushed. The data must be copied, because the Mac
cannot reach gattaca2 from Claude's session. `-L` matters for the second copy:
calibrated scenes in `planet_clipped_calfill` are symlinks into
`planet_clipped_cal`.

```bash
F=/Volumes/Earth03/flower
rsync -a  $F/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated \
  gattaca2:/scratch/tree-monitoring/                                  # 2.6 GB, 712 files
rsync -aL $F/planet_clipped_calfill gattaca2:/scratch/tree-monitoring/  # 1.0 GB, 2,138 tifs
```

`planet_clipped_calfill` is needed only for the optional apply step (step 6).

## 1. Update the repo and check the inputs (gattaca2)

```bash
cd <tree-monitoring repo on gattaca2>
git pull
source ~/miniconda3/etc/profile.d/conda.sh; conda activate flower
mkdir -p logs
F=/scratch/tree-monitoring
ls $F/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated/*_4band.tif | wc -l   # 142
ls $F/20261012_full_label_extended2_mkl_x4_coreg_4band_cal_curated/*.mask.png | wc -l    # 142
find $F/planet_clipped_calfill/4band -name '*.tif' -type f | wc -l                       # 2138, real files
python scripts/train_planet_image_maskrcnn.py --help | grep -A2 fixed-stretch
python -c "import sys; sys.path.insert(0, 'scripts'); \
from train_planet_image_maskrcnn import load_stretch; print(load_stretch('config/planet_fixed_stretch.yml'))"
```

The stretch must print `lo [0.0039, 0.0177, 0.0074, 0.1825]` and `hi [0.0446,
0.0632, 0.0484, 0.4448]`.

The spatial benchmark scores the baselines on the 56 test chips *inside* the
142-chip TOA set, not in the `20260706` set the earlier sweeps used. On the
Mac the two copies are byte-identical (md5 of tif, mask and OCM sidecar for all
56). Step 3 checks this again on gattaca2 by reproducing the earlier numbers.

## 2. Submit everything (gattaca2)

There is a limit of 2 GPUs per user (see [gattaca2_cluster.md](gattaca2_cluster.md)).
So the spatial training job and the 6 temporal folds share two GPUs: expect
about 6–7 h wall time for all of it. Close any interactive GPU session first,
because it takes one of the two slots.

```bash
# spatial: train 200 epochs (checkpoints at the 12 benchmark epochs), then score
source config/spatial_20261012_cal.sh
TRAIN=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_spatial_train.sh)
sbatch -A tree-monitoring --dependency=afterok:$TRAIN scripts/slurm/planet_spatial_compare.sh

# temporal CV: 6 folds x 40 epochs, then the pooled evaluation
source config/temporal_cv_20261012_cal.sh
CV=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_cv.sh)
sbatch -A tree-monitoring --dependency=afterok:$CV scripts/slurm/planet_temporal_eval.sh
squeue -u $USER
```

Each `source` sets different variables (`SPATIAL_*`, `TEMPORAL_*`/`APPLY_*`),
so sourcing both in one shell is safe. The jobs stage the chip sets to
`/local/gdoran_flower` themselves (`scripts/slurm/stage_chips.sh`, which
verifies by checksum).

Earlier timings on one A100:
- `full_mkl`: 2 h 27 m (44 s/epoch).
- One temporal fold: 30–35 min.
- The 60-checkpoint sweep: 20 min. This one has 48 checkpoints.

**Check as soon as training starts.** Look at `logs/spatial_train_<id>.out` and
`logs/temporal_cv_<id>_0.out`. The first lines should show the staged
`..._cal_curated` path. Once `epoch_001.pth` exists, confirm that its params
match `full_mkl` apart from the stretch:

```bash
python - <<'EOF'
import sys, torch; sys.path.insert(0, 'scripts')
from compare_maskrcnn_runs import load_checkpoint
O = '/home/gdoran/Documents/tree-flowering'
_, a, _ = load_checkpoint(f'{O}/202610_full2_mkl_curated_maskrcnn_out_min064/epoch_001.pth', torch.device('cpu'))
_, b, _ = load_checkpoint(f'{O}/202610_full2_mkl_cal_curated_maskrcnn_out_min064/epoch_001.pth', torch.device('cpu'))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(k, '|', a.get(k), '|', b.get(k))
EOF
```

Expect differences only in `stretch` (None, then the dict) and in run
bookkeeping such as output paths or saved epochs. Anything else, for example
`lr`, `batch_size`, `num_epochs`, `min_instance_size`, `use_ocm_masks` or
`channel_kinds`, means the runs are not comparable: stop and investigate.

## 3. Spatial results (gattaca2)

`planet_spatial_compare.sh` writes `logs/headtohead_sweep_202610_mkl_cal.json`,
with the labels `base_eNNN`, `phantom_eNNN`, `full_mkl_eNNN` and
`full_mkl_cal_eNNN`.

**First, check the harness.** The baseline rows must reproduce the earlier
sweep exactly:

```bash
python - <<'EOF'
import json
old = json.load(open('logs/headtohead_sweep_202610_mkl.json'))['results']
new = json.load(open('logs/headtohead_sweep_202610_mkl_cal.json'))['results']
keys = [k for k in new if not k.startswith('full_mkl_cal') and k in old]
diff = max(abs(old[k][m] - new[k][m]) for k in keys for m in new[k]
           if isinstance(new[k][m], (int, float)) and m in old[k])
print(len(keys), 'baseline rows, max abs diff', diff)   # expect 36 rows, 0.0
EOF
```

If the difference is not 0, the 56 test chips in the 142-chip TOA set differ
from the `20260706` copies on gattaca2. Do not trust the comparison until that
is understood.

**Then read off the peaks** over the 12 epochs, as in the tables of
[planet_training_chips_50ha.md](planet_training_chips_50ha.md#colour-transfer-mavic-masks-20261012-set):

```bash
python - <<'EOF'
import json, collections
r = json.load(open('logs/headtohead_sweep_202610_mkl_cal.json'))['results']
runs = collections.defaultdict(dict)
for lab, row in r.items():
    run, ep = lab.rsplit('_e', 1); runs[run][int(ep)] = row
for run, eps in runs.items():
    m = max(eps, key=lambda e: eps[e]['test_map/map_50']); i = max(eps, key=lambda e: eps[e]['test_iou'])
    print(f"{run:13s} map_50 {eps[m]['test_map/map_50']:.4f} @e{m:03d}  iou {eps[i]['test_iou']:.4f} @e{i:03d}  "
          f"P/R {eps[m]['test_event/precision']:.3f}/{eps[m]['test_event/recall']:.3f}")
EOF
```

Earlier peaks for reference: `base` 0.1681 (e012) and IoU 0.3076 (e017);
`full_mkl` 0.1488 (e005) and IoU 0.2929 (e017). On 56 test chips a difference
of about 0.01 in `map_50` is within noise.

**Plot.** The plotters take explicit paths, so they run here as well as on the
Mac:

```bash
J=logs/headtohead_sweep_202610_mkl_cal.json
L=(--label base="phantom 2020-23 only (56)" --label phantom="+ phantom 2023-24 (77)"
   --label full_mkl="all three, TOA (142)" --label full_mkl_cal="all three, calibrated (142)")
R=(--run base --run phantom --run full_mkl --run full_mkl_cal)
python scripts/plot_headtohead_epochs.py $J logs/maskrcnn_headtohead_cal.png "${L[@]}" "${R[@]}" \
  --standalone iou --standalone pr
python scripts/plot_update_figures.py maskrcnn-summary --outdir logs --sweep $J \
  --name maskrcnn_summary_cal "${L[@]}" "${R[@]}"
```

## 4. Temporal CV results (gattaca2)

`planet_temporal_eval.sh` writes `$TEMPORAL_RUNS/temporal_cv.json` and
`temporal_cv_windows.csv`, and prints the pooled table and the selected epoch.
Compare with the TOA run:

```bash
source config/temporal_cv_20261012_cal.sh
python scripts/compare_temporal_cv.py \
  --run toa=$HOME/Documents/tree-flowering/202610_temporal/temporal_cv.json \
  --run cal=$TEMPORAL_RUNS/temporal_cv.json \
  --plot $TEMPORAL_RUNS/temporal_cv_toa_vs_cal.png
python scripts/plot_temporal_cv.py $TEMPORAL_RUNS/temporal_cv.json $TEMPORAL_RUNS/temporal_cv.png
python scripts/plot_temporal_cv_detail.py $TEMPORAL_RUNS/temporal_cv.json \
  $TEMPORAL_RUNS/temporal_cv_windows.csv $TEMPORAL_RUNS/temporal_cv
```

`compare_temporal_cv.py` refuses runs whose folds, epochs or scoring settings
differ. The TOA reference at its selected epoch 12 is:

| | pooled | 0–3 d | 4–7 d | 8–14 d | 15–30 d | >30 d |
|---|---|---|---|---|---|---|
| mAP@50 | 0.354 | 0.512 | 0.435 | 0.372 | 0.258 | 0.203 |
| IoU | 0.352 | 0.416 | 0.420 | 0.385 | 0.272 | 0.217 |

The main question is the far bins (15–30 and >30 days). Calibration should
help most where the date, and so the atmosphere, is furthest from training.
Look at fold 1 too: it holds most of the sparse 2021 flights.

## 5. Copy results back to Earth03 (on the Mac, by the user)

```bash
F=/Volumes/Earth03/flower
rsync -a gattaca2:'~/Documents/tree-flowering/202610_temporal_cal/temporal_cv*' $F/202610_temporal_cal/
mkdir -p $F/figs/202610_cal_updates
rsync -a gattaca2:<repo>/logs/headtohead_sweep_202610_mkl_cal.json \
         gattaca2:<repo>/logs/maskrcnn_headtohead_cal*.png \
         gattaca2:<repo>/logs/maskrcnn_summary_cal* $F/figs/202610_cal_updates/
```

## 6. Optional: final model and apply (gattaca2; ask the user first)

Only run this if the temporal CV improves; the user decides. It reuses the TOA
coregistration (`202610_temporal_apply/coreg_nearest_drone.json`). Do **not**
run `planet_coreg_nearest.sh` with the calibrated config.

```bash
source config/temporal_cv_20261012_cal.sh
export TEMPORAL_FINAL_EPOCHS=$(python -c "import json; print(json.load(open('$TEMPORAL_RUNS/temporal_cv.json'))['best_epoch'])")
FINAL=$(sbatch --parsable -A tree-monitoring scripts/slurm/planet_temporal_final.sh)
sbatch -A tree-monitoring --dependency=afterok:$FINAL scripts/slurm/planet_apply.sh
# when the apply array has finished:
python scripts/apply_planet_maskrcnn.py concat $APPLY_OUT
```

Apply reads `planet_clipped_calfill/4band` and writes to
`202610_temporal_apply_cal/`. It logs `skipping ... not found` for 6 of the
798 scenes: Dove Classic `1105` scenes that have no SR product. That is
expected.

## 7. Record the results (permanent docs)

Write each result into the doc that owns the experiment, in the existing
style: a short narrative with exact numbers, then a peak table. Add figure
captions per the CLAUDE.md caption rules.

- **Spatial:** a new subsection in
  [planet_training_chips_50ha.md](planet_training_chips_50ha.md), after
  "Colour-transfer mavic masks (`20261012` set)". Title it "MODIS-calibrated
  chips (`full_mkl_cal`)". Include:
  - the commands (step 2) and the run time;
  - the result of the param-diff check;
  - the harness check (max abs diff);
  - the peak table with `base`, `full_mkl` and `full_mkl_cal`.
- **Temporal:** a dated paragraph and table in the "Results" section of
  [planet_temporal_generalization.md](planet_temporal_generalization.md). Give
  pooled and per-bin mAP@50 and IoU at each run's selected epoch, and the
  selected epochs.
- **Calibration doc:** a short "Training results" paragraph at the end of
  "Classifier input stretch" in
  [planet_radiometric_calibration.md](planet_radiometric_calibration.md),
  linking to the two sections above.
- **Cluster notes:** add anything new about gattaca2 to
  [gattaca2_cluster.md](gattaca2_cluster.md). Add the new datasets to its
  storage table.

Commit with weekly-report-style subjects, for example "Retrain the Planet
Mask R-CNN on MODIS-calibrated chips; temporal-CV pooled mAP@50 0.354 → X".
Stage only the doc files, then push.

## 8. Remove this handoff document

```bash
git rm docs/handoff_calibrated_training.md
git commit -m "Remove calibrated-training handoff; results are in the chip, temporal and calibration docs"
git push
```

## After this (not part of the handoff)

These are possible follow-ups:
- drop the 4 hazy chips and retrain;
- add a dataset-derived mean/std for the stretched channels, instead of the
  ImageNet constants;
- apply the calibrated final model to the full record.

## Troubleshooting

- **A job waits with `QOSMaxGRESPerUser`.** The 2-GPU limit is in use (an
  interactive GPU session counts). The job will start once a GPU frees.
- **`Can't get attribute 'OCMMaskRCNN' on <module '__main__'>`** when loading a
  checkpoint by hand. Load it through `compare_maskrcnn_runs.load_checkpoint`,
  which registers the classes.
- **`Checksum mismatch staging ...`.** The `/local` copy is corrupt. Remove
  `/local/gdoran_flower/<set>` on that node and resubmit.
- **`eval_temporal_cv.py`: "settings differ from earlier runs".** Some fold was
  trained with different arguments. Check `TEMPORAL_TRAIN_ARGS` in that fold's
  log.
- **Input sanity check.** A calibrated chip through the fixed stretch should
  give a mean input of about 0.3–0.5 per channel. A TOA chip through the fixed
  stretch saturates at about 0.99. That means the wrong chip set was used with
  the stretch.
