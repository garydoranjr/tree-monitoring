"""Brute-force NCC check of a Planet-to-drone coregistration shift.

AROSICS (`compute_coreg_shift` in apply_drone_labels_coreg.py) finds the
shift from one matching window and can lock onto a false match when its
search range is wide. This is an independent estimate of the same shift:
1. The Planet red band is resampled to a 1 m grid on the ortho.
2. Pixels that OmniCloudMask does not call clear, and nodata, are set to the
   clear mean.
3. The part of the scene that stays inside the ortho under any shift up to
   `search_m` is correlated (normalized cross-correlation) against the
   ortho's red band, block-averaged to 1 m, over every shift in that range.

The shift is returned in AROSICS's `corrected_shifts_map` convention (the
shift to apply to the Planet scene, metres east and north), so the two can be
compared directly. On 149 scenes AROSICS registers, NCC agrees within 3 m on
134. On the 95 where NCC is confident (`ncc_confident`) it agrees on 93. See
docs/handoff_rerun_coregistration.md.
"""
from functools import lru_cache

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from skimage.feature import match_template

NCC_RES = 1.0
NCC_SEARCH_M = 90.0
SECOND_EXCLUDE_M = 15.0
MIN_TEMPLATE_CLEAR = 0.05
# Confidence: the peak, and its ratio to the best NCC more than
# SECOND_EXCLUDE_M away. Chosen on the 149 controls above.
MIN_PEAK = 0.2
MIN_RATIO = 2.5
OCM_CLEAR = 0
OCM_NODATA = 255


@lru_cache(maxsize=2)
def drone_red_1m(dronefile):
    """The ortho's red band block-averaged to NCC_RES, its valid mask, transform
    and CRS. Cached because every scene of a label date shares one ortho."""
    with rasterio.open(dronefile) as src:
        b = src.bounds
        w = int((b.right - b.left) // NCC_RES)
        h = int((b.top - b.bottom) // NCC_RES)
        red = src.read(1, out_shape=(h, w), resampling=Resampling.average,
                       masked=True)
        transform = from_origin(b.left, b.top, NCC_RES, NCC_RES)
        crs = src.crs
    valid = ~np.ma.getmaskarray(red) & (red.filled(0) > 0)
    return red.filled(0).astype(np.float32), valid, transform, crs


def ocm_clear(ocm_path, src):
    """OCM clear mask on the grid of the open Planet scene `src`."""
    with rasterio.open(ocm_path) as ocm:
        cls = np.full(src.shape, OCM_NODATA, dtype=np.uint8)
        reproject(ocm.read(1), cls, src_transform=ocm.transform,
                  src_crs=ocm.crs, dst_transform=src.transform,
                  dst_crs=src.crs, resampling=Resampling.nearest,
                  src_nodata=OCM_NODATA, dst_nodata=OCM_NODATA)
    return cls == OCM_CLEAR


def ncc_shift(planetfile, dronefile, ocm_path=None, planet_band=3,
              search_m=NCC_SEARCH_M):
    """NCC shift of a Planet scene against the ortho.

    Returns a dict:
    - ncc_x_m, ncc_y_m: the shift (NaN when too little of the scene is clear);
    - ncc_peak: the peak NCC;
    - ncc_second: the best NCC more than SECOND_EXCLUDE_M from the peak;
    - ncc_clear: the clear share of the template.
    """
    red_d, valid_d, transform, crs = drone_red_1m(dronefile)
    with rasterio.open(planetfile) as src:
        red_p = src.read(planet_band).astype(np.float32)
        clear = src.read_masks(planet_band) > 0
        if ocm_path is not None:
            clear &= ocm_clear(ocm_path, src)
        on_d = np.zeros(red_d.shape, np.float32)
        clr_d = np.zeros(red_d.shape, np.uint8)
        reproject(red_p, on_d, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)
        reproject(clear.astype(np.uint8), clr_d, src_transform=src.transform,
                  src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    m = int(search_m / NCC_RES)
    tpl = on_d[m:-m, m:-m].copy()
    tclr = clr_d[m:-m, m:-m].astype(bool)
    out = {'ncc_x_m': np.nan, 'ncc_y_m': np.nan, 'ncc_peak': np.nan,
           'ncc_second': np.nan,
           'ncc_clear': float(tclr.mean()) if tclr.size else 0.0}
    # An ortho no wider than twice the search (e.g. the plot-only AVA ortho)
    # leaves no template, and the check cannot run.
    if out['ncc_clear'] < MIN_TEMPLATE_CLEAR:
        return out
    tpl[~tclr] = tpl[tclr].mean()
    img = red_d.copy()
    img[~valid_d] = img[valid_d].mean()
    r = match_template(img, tpl)
    i, j = np.unravel_index(np.argmax(r), r.shape)
    yy, xx = np.ogrid[:r.shape[0], :r.shape[1]]
    far = (yy - i) ** 2 + (xx - j) ** 2 > (SECOND_EXCLUDE_M / NCC_RES) ** 2
    # The best match is (j - m, i - m) pixels right and down of the scene's
    # claimed position. The correction to apply is its negative, in
    # AROSICS's corrected_shifts_map convention (checked on scenes AROSICS
    # registers: NCC (-11, -33) m against AROSICS (-10.4, -31.8) m).
    out.update(ncc_x_m=-(j - m) * NCC_RES, ncc_y_m=(i - m) * NCC_RES,
               ncc_peak=float(r[i, j]),
               ncc_second=float(r[far].max()) if far.any() else np.nan)
    return out


def ncc_confident(ncc, search_m=NCC_SEARCH_M):
    """Whether an ncc_shift result is a clear, unique peak inside the search
    range (a peak on its edge may lie beyond it)."""
    peak, second = ncc['ncc_peak'], ncc['ncc_second']
    if not np.isfinite(peak) or peak < MIN_PEAK:
        return False
    if np.isfinite(second) and second > 0 and peak / second < MIN_RATIO:
        return False
    edge = search_m - NCC_RES
    return max(abs(ncc['ncc_x_m']), abs(ncc['ncc_y_m'])) < edge
