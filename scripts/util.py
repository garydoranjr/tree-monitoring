"""
Tree Flowering Utilities
"""
import yaml

def load_config(configfile):
    with open(configfile, 'r') as f:
        return yaml.safe_load(f)


def select_device(pref="auto"):
    """Return a torch.device, preferring CUDA, then Apple MPS, then CPU.

    To make MPS usable, PYTORCH_ENABLE_MPS_FALLBACK=1 must be set *before*
    torch is imported (PyTorch reads it once at import). That guard lives at
    the top of each entry-point script, not here -- by the time this runs
    torch is already imported and the flag would be ignored.
    """
    import torch
    pref = (pref or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    else:
        name = pref
    return torch.device(name)


def is_readable_raster(path):
    """True if `path` exists and opens as a raster with at least one band.

    A non-empty file is not enough: an interrupted write leaves a truncated
    GeoTIFF that passes a size check but fails to open. Imported lazily so
    callers that only need load_config don't pay for rasterio.
    """
    import os
    import rasterio
    # Short-circuit: opening a missing path works but makes GDAL log an error
    # per call, which floods the log when most outputs are yet to be written.
    if not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as src:
            return src.count > 0
    except Exception:
        return False


# Data roots, in order of preference: the Earth03 drive on the Mac, then
# gattaca2's /scratch/tree-monitoring, which mirrors it (synced by rsync).
DATA_ROOTS = ('/Volumes/Earth03/flower', '/scratch/tree-monitoring')


def data_root():
    """Path of the project data root on this machine.

    $TREE_DATA_ROOT wins if set; otherwise the first of DATA_ROOTS that
    exists, falling back to the Earth03 path so that --help still works on a
    machine with neither.
    """
    import os
    from pathlib import Path
    if os.environ.get('TREE_DATA_ROOT'):
        return Path(os.environ['TREE_DATA_ROOT'])
    for root in DATA_ROOTS:
        if os.path.isdir(root):
            return Path(root)
    return Path(DATA_ROOTS[0])


def calibration_results():
    """Directory of the Planet-MODIS calibration tables (scene offsets, fits,
    evaluation, fixed-stretch metrics): results/planet_calibration under the
    data root."""
    return data_root() / 'results' / 'planet_calibration'


def offset_sd(offsets_csv=None):
    """Across-scene SD of the fitted Planet-MODIS offsets per band (blue,
    green, red, NIR; reflectance), over the own-fit scenes of
    planet_scene_offsets.csv: the size of the per-scene error that
    calibration removes."""
    import pandas as pd
    if offsets_csv is None:
        offsets_csv = calibration_results() / 'planet_scene_offsets.csv'
    o = pd.read_csv(offsets_csv)
    return (o[o['source'] == 'fit'].groupby('band')['offset'].std()
            .reindex(('blue', 'green', 'red', 'nir')).to_numpy())
