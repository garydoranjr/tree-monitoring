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
