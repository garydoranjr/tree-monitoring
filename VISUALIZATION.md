# Interactive Mask R-CNN Visualization

Instructions for setting up the environment and running the interactive tree crown detection viewer.

## Prerequisites

- [Git](https://git-scm.com/)
- Conda — see below if not already installed

## Install Miniconda (if needed)

Download and run the Miniconda installer for your platform from https://docs.conda.io/en/latest/miniconda.html, then follow the on-screen instructions. After installation, open a new terminal and verify it worked:

```bash
conda --version
```

## Clone the repository

```bash
git clone https://github.com/garydoranjr/tree-monitoring.git
cd tree-monitoring
```

## Pulling updates

If the repository has been updated since you last cloned it, pull the latest changes:

```bash
git pull origin main
```

## Set up the conda environment

Create and activate the `flower` environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate flower
```

This only needs to be done once. On subsequent sessions, just run `conda activate flower`.

## Run the visualization

```bash
conda activate flower
python scripts/deploy_planet_image_maskrcnn_interactive.py \
    <path/to/data_directory> \
    --model <path/to/model.pth>
```

Replace `<path/to/data_directory>` with the path to the directory containing the Planet image tiles, and `<path/to/model.pth>` with the path to the trained Mask R-CNN weights file (`.pth`).

Once the server starts, open the URL printed in the terminal (typically http://127.0.0.1:8050) in a web browser.

### Reviewing coregistration instead of predictions

`--model` is optional. Leave it off and there is no prediction layer — the app becomes a comparator for the drone ortho, the Planet chip and the crown mask applied from the drone:

```bash
conda activate flower
python scripts/deploy_planet_image_maskrcnn_interactive.py \
    <path/to/data_directory> --split whole
```

`--split whole` shows the entire chip rather than the 512 px training window, which is where misalignment is easiest to see. A chip with no `.mask.png` simply loses the ground-truth layer.

Tick **Show drone overlay**, then compare it against the chip either way:

- **Blend** — the opacity slider fades the drone over the chip.
- **Swipe** — the drone covers the frame up to a moving edge, at full strength.

| Key | Action |
| --- | --- |
| `b` or space | Blink the drone layer on and off |
| `←` `→` | Previous / next chip |

The Δx/Δy readout in the corner is read from `coreg_log.json` and is informational only; a log that was rebuilt after a failed build shows `n/a (reconstructed)` rather than a shift it does not have.

## Vetting chips for the training set

A separate, much lighter tool rates chips `Poor` / `Fair` / `Good` on image quality — cloud, haze, partial coverage — to decide which belong in the training set at all. It reads only the RGB previews, so it starts immediately:

```bash
conda activate flower
python scripts/vet_planet_chips.py <path/to/data_directory>
```

Open http://127.0.0.1:8051. Every chip appears as a thumbnail bordered by its rating; click a thumbnail for the full-resolution view, with an optional cloud-mask overlay and a note field.

| Key | Action |
| --- | --- |
| `1` `2` `3` | Rate Poor / Fair / Good |
| `←` `→` | Previous / next chip |
| `esc` | Back to the contact sheet |

Clicking a chip's current rating again clears it. **Show: unrated** narrows the sheet to what is left to do.

Ratings are saved to `<data_directory>/vetting.json` (or `-o <path>`) after every click, so you can stop and reopen the tool at any point. `copy_good_planet_vetting.py --vetting <vetting.json>` then assembles the curated set from it.
