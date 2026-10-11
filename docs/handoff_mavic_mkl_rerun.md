# Handoff: re-run mavic crown classification with the colour transfer (gattaca2)

> **Temporary document.** Delete it once the new run is finished and copied
> back. Step 7 says how.

## Why

On the two days both drones flew, the mavic flowering classifications came out
about 5× higher than phantom's. Most of the excess comes from colour: mavic
reaches the model greener and more saturated than phantom.
[`config/crown_classification_mavic_mkl.yml`](../config/crown_classification_mavic_mkl.yml)
replaces the per-band uint16 → uint8 map with a 3×3 colour transfer. On the test
crowns it cuts the mavic-only flowering calls from 42 to about 6–9, and it keeps
14 of the 16 crowns that flower on both cameras. For the analysis, see
[`docs/drone_radiometry.md`](drone_radiometry.md).

The old per-band results are archived rather than deleted:

| | Old per-band results | New colour-transfer results |
|---|---|---|
| Earth03 (local) | `/Volumes/Earth03/flower/results/older/classifications_mavic_perband/` (moved on 2026-10-10, done) | `/Volumes/Earth03/flower/results/classifications/` (step 6) |
| gattaca2 | `/scratch/tree-monitoring/results/older/globus_perband/` (step 2) | `/scratch/tree-monitoring/results/globus_mkl/` (steps 3–5) |

## Steps on gattaca2

### 1. Update the repo and check the environment

```bash
cd <tree-monitoring repo on gattaca2>     # where drone_{flower,decid}_geo_out/ live
git pull
conda activate flower
python -c "import transformers; print(transformers.__version__)"
```

`transformers` must be **below 5**. The checkpoints are pickled whole models,
and 5.x removed `SegformerEncoder`. If the check prints 5.x, run
`pip install "transformers<5"`.

### 2. Archive the old per-band results

Make sure no earlier mavic pipeline job is still running (`squeue -u $USER`).
Then move the results:

```bash
mkdir -p /scratch/tree-monitoring/results/older
test ! -e /scratch/tree-monitoring/results/older/globus_perband && \
  mv /scratch/tree-monitoring/results/globus /scratch/tree-monitoring/results/older/globus_perband
```

`mv` within `/scratch` is a rename, so nothing is copied or deleted.
`config/pipeline_mavic.sh` still points at `results/globus`. Don't run it
again unless you mean to reproduce the old per-band results.

### 3. Dry run

```bash
./run_crown_pipeline.sh config/pipeline_mavic_mkl.sh --dry-run
```

Check the dry-run output:
- `Scaling: config/crown_classification_mavic_mkl.yml`
- `Output dir: /scratch/tree-monitoring/results/globus_mkl`
- `Images: 96`, giving 192 classify / 192 merge / 96 concat tasks

### 4. Submit and monitor

```bash
./run_crown_pipeline.sh config/pipeline_mavic_mkl.sh
squeue -u $USER
```

Logs go to `logs/{classify,merge,concat}_<job>_<task>.{out,err}`. The first
lines of each classify log should show the `_mkl.yml` scaling config. With the
per-band config, each classify task took ~17 min. The colour transfer adds
only a 3×3 multiply per window, so expect the same.

Every stage skips outputs that already exist. A failed or timed-out array can
therefore be resubmitted with the same command without redoing finished work.

### 5. Check the outputs

```bash
ls /scratch/tree-monitoring/results/globus_mkl/*/*_classifications.tif | wc -l   # expect 96
```

### 6. Copy back to Earth03

Run this from the Mac, with Earth03 mounted:

```bash
STAGE=/Volumes/Earth03/flower/results/globus_mkl_staging   # same volume, so the mv is a rename
rsync -av --include='*/' --include='*_classifications.tif' --exclude='*' \
  gattaca2:/scratch/tree-monitoring/results/globus_mkl/ "$STAGE"/
find "$STAGE" -name '*_classifications.tif' \
  -exec mv -n {} /Volumes/Earth03/flower/results/classifications/ \;
ls /Volumes/Earth03/flower/results/classifications/*_M3M_*_classifications.tif | wc -l   # expect 96
find "$STAGE" -type f | wc -l    # expect 0, then: find "$STAGE" -type d -empty -delete
```

The 96 files total about 6.5 GB.

The new files have the same names as the old ones. The old ones are already in
`results/older/classifications_mavic_perband/`, so nothing is overwritten.
`plot_update_figures.py` will pick up the new files automatically.

Then refresh the comparison figures:
```bash
python scripts/plot_update_figures.py same-date
python scripts/plot_update_figures.py same-date-examples
python scripts/plot_update_figures.py classification-timeseries --decimate 16
```

### 7. Remove this handoff document

Once step 6 is done and the new results look right, delete this file:

```bash
git rm docs/handoff_mavic_mkl_rerun.md
git commit -m "Remove mavic colour-transfer re-run handoff; new classifications are in results/classifications"
git push
```

Also record the full-series outcome in
[`docs/drone_radiometry.md`](drone_radiometry.md) under "Production config".
Two numbers are useful: the mavic flowering fraction before and after, and the
same-day Spearman ρ against phantom.
