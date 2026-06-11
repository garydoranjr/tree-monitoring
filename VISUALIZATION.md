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
    <path/to/model.pth> \
    <path/to/data_directory>
```

Replace `<path/to/model.pth>` with the path to the trained Mask R-CNN weights file (`.pth`), and `<path/to/data_directory>` with the path to the directory containing the Planet image tiles.

Once the server starts, open the URL printed in the terminal (typically http://127.0.0.1:8050) in a web browser.
