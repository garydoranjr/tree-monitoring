"""
Tree Flowering Utilities
"""
import os
import yaml

def load_config(configfile):
    with open(configfile, 'r') as f:
        return yaml.safe_load(f)


def select_device(pref="auto"):
    """Return a torch.device, preferring CUDA, then Apple MPS, then CPU.

    When MPS is selected, set PYTORCH_ENABLE_MPS_FALLBACK=1 so ops not yet
    implemented for the MPS backend fall back to CPU instead of erroring.
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
    if name == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device(name)
