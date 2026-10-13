#!/usr/bin/env python
"""Compare ML crown classifications to Vicente's ground-truth labels.

Joins per-(crown, date) ground-truth labels from
`vicente_20260618_labels.csv` to the ML output netCDF
(`crown_classifications.nc4`) and reports flowering / deciduous metrics,
overall, per species and per crown-size bin. The uuid->tag bridge, the
crown area and the gallery outlines come from the dated crown map
(--crownmap), whose per-date polygons are the ones the nc4 averages over.
Crowns that share a tag with another crown are dropped, since the nc4 keeps
only one of them. `--crownmap ''` falls back to the June 2026 bridge through
`flowering_dataset.gpkg`, which covers only its 251 crowns.

Phantom example (held-out: crown-dates in a same-date training chip dropped):

    python scripts/compare_classifications_to_gt.py --exclude-train-chips \
        --output-dir /Volumes/Earth03/flower/figs/20261013_updates/vicente_phantom_eval/heldout

With --labels-gpkg, the labels instead come from a labelling-tool export
that carries the crown-map `tag` on every crown-date polygon (e.g.
vicente_202610/latest_paulino_label.gpkg), so no uuid bridge is needed and
any nc4 built by extract_crown_stats_from_masks.py can be scored, including
crown_classifications_mavic.nc4. Example:

    python scripts/compare_classifications_to_gt.py \
        --labels-gpkg /Volumes/Earth03/flower/vicente_202610/latest_paulino_label.gpkg \
        --nc4 /Volumes/Earth03/flower/results/crown_classifications_mavic.nc4 \
        --drone-dir /Volumes/Earth03/flower/stri/globus/RGB \
        --drone-glob '*_M3M_aligned_global_RGB.tif' \
        --drone-scaling config/crown_classification_mavic_mkl.yml \
        --output-dir /Volumes/Earth03/flower/figs/20261013_updates/paulino_eval/mavic
"""
import base64
import html
import sys
from io import BytesIO
from pathlib import Path

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy.stats import pearsonr, spearmanr
from shapely.geometry import box
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from crown_timelapse_mosaic import (
    draw_polygon_on_image,
    extract_centered_window,
    get_date,
)


DRONE_DIR_DEFAULT = (
    "/Volumes/Earth03/flower/stri/24782016/"
    "BCI_50ha_timeseries_local_alignment"
)
GT_CSV_DEFAULT = (
    "/Volumes/Earth03/flower/vicente_20260618/vicente_20260618_labels.csv"
)
GPKG_DEFAULT = (
    "/Volumes/Earth03/flower/vicente_202509/flowering_dataset.gpkg"
)
NC4_DEFAULT = "/Volumes/Earth03/flower/results/crown_classifications.nc4"
GEO_FOLDS_DEFAULT = (
    "/Volumes/Earth03/flower/vicente_202509/geo_folds.csv"
)
TRAIN_CHIPS_DEFAULT = "/Volumes/Earth03/flower/vicente_202509/images"
CROWNMAP_DEFAULT = (
    "/Volumes/Earth03/flower/crowns_20240803/BCI_50ha_crownmap_timeseries.shp"
)
# Crown-area bin edges (m^2), about the quartiles of the good-segmentation
# phantom labels.
SIZE_BINS_DEFAULT = "60,110,200"

# Labelling-tool gpkg exports mix English and Spanish answers.
FLOWERING_MAP = {"no": "no", "yes": "yes", "sí": "yes", "si": "yes",
                 "maybe": "maybe", "quizás": "maybe", "quizas": "maybe"}
# Rows the annotator never opened keep the tool defaults (flowering "No",
# leafing 0) under these quality values, so they are not labels.
UNCHECKED_QUALITY = {"Not Checked", "Sin Revisar"}
GOOD_QUALITY = {"Excelente", "Excellent"}


def load_gt(path):
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["date"], format="%Y_%m_%d")
    df["uuid"] = df["polygon_id"].str.rsplit("_", n=1).str[0]
    return df


def _agg_flowering(s):
    """Majority vote over {yes, no} for one crown-date, ignoring 'maybe'.

    Returns "yes"/"no" on a clear majority, "maybe" when no yes/no vote
    exists, or the sentinel "__tie__" when yes and no votes are equal and
    nonzero (an unresolvable conflict to be dropped).
    """
    yes = int((s == "yes").sum())
    no = int((s == "no").sum())
    if yes == 0 and no == 0:
        return "maybe"
    if yes == no:
        return "__tie__"
    return "yes" if yes > no else "no"


def _agg_segmentation(s):
    """Majority-vote segmentation for one crown-date.

    On a tie, prefer a non-'good' label so the headline segmentation=='good'
    stratum stays conservative.
    """
    vc = s.value_counts()
    top = list(vc[vc == vc.max()].index)
    if len(top) == 1:
        return top[0]
    non_good = sorted(v for v in top if v != "good")
    return non_good[0] if non_good else "good"


def aggregate_gt_labels(gt):
    """Collapse multiple GT labels for the same (uuid, date) into one row.

    Multiple annotators sometimes label the same crown-date: observation_id
    is unique per label, but polygon_id/globalId/latin are constant within a
    crown-date. Without aggregation the inner join with predictions emits one
    row per label (each paired with the *same* prediction), over-weighting
    heavily-observed crown-dates and scoring conflicting labels against one
    prediction. We aggregate to one row per (uuid, dt):

      - isFlowering: majority vote over {yes, no} (see _agg_flowering); a
        yes/no tie is unresolvable and that crown-date is dropped.
      - leafing, floweringIntensity: mean (NaN-skipping).
      - segmentation: majority vote (see _agg_segmentation).
      - polygon_id, globalId, latin, date: first (constant within a crown-date).
      - n_labels: how many GT observations were aggregated.

    Returns (gt_agg, flower_conflicts), where flower_conflicts is the raw
    label rows for the dropped yes/no-tie crown-dates (for the conflict report).
    """
    grouped = gt.groupby(["uuid", "dt"], sort=False)
    agg = grouped.agg(
        isFlowering=("isFlowering", _agg_flowering),
        leafing=("leafing", "mean"),
        floweringIntensity=("floweringIntensity", "mean"),
        segmentation=("segmentation", _agg_segmentation),
        polygon_id=("polygon_id", "first"),
        globalId=("globalId", "first"),
        latin=("latin", "first"),
        date=("date", "first"),
        n_labels=("isFlowering", "size"),
    ).reset_index()

    mask = agg["isFlowering"] == "__tie__"
    conflict_keys = agg.loc[mask, ["uuid", "dt"]]
    flower_conflicts = gt.merge(conflict_keys, on=["uuid", "dt"])
    agg = agg[~mask].copy()
    return agg, flower_conflicts


def load_train_uuids(path):
    """Return the set of crown uuids that appear in the train split."""
    df = pd.read_csv(path)
    train = df[df["split"] == "train"]
    uuids = train["polygon_id"].str.rsplit("_", n=1).str[0]
    return set(uuids.unique())


def load_uuid_tag_map(gpkg_path):
    gdf = gpd.read_file(gpkg_path, layer="flowering_dataset")
    gdf = gdf.dropna(subset=["tag"])[["polygon_id", "tag"]].copy()
    gdf["uuid"] = gdf["polygon_id"].str.rsplit("_", n=1).str[0]
    gdf["tag"] = gdf["tag"].astype(int)
    return gdf[["uuid", "tag"]].drop_duplicates()


def load_crownmap(path):
    """Dated crown map: one polygon per (crown, date) with the scored outline.

    Returns (crowns, n_shared): crowns has uuid (GlobalID), date
    (YYYY_MM_DD), tag (int), crown_area (m^2) and geometry, without the
    crowns whose tag is shared with another crown (the nc4 is keyed by tag
    and keeps only one of them); n_shared counts those dropped crowns.
    """
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 32617:
        gdf = gdf.to_crs(epsg=32617)
    gdf = gdf.rename(columns={"GlobalID": "uuid"})
    gdf["tag"] = gdf["tag"].astype(int)
    pairs = gdf[["uuid", "tag"]].drop_duplicates()
    shared = pairs["tag"].duplicated(keep=False)
    gdf = gdf[~gdf["uuid"].isin(pairs.loc[shared, "uuid"])].copy()
    gdf["crown_area"] = gdf.geometry.area
    return gdf[["uuid", "date", "tag", "crown_area", "geometry"]], int(shared.sum())


def geoms_by_crown(gdf, key="uuid"):
    """{key: {"by_date": {YYYY_MM_DD: geom}, "any": geom}} for the galleries."""
    return {k: {"by_date": dict(zip(sub["date"], sub.geometry)),
                "any": sub.geometry.iloc[0]}
            for k, sub in gdf.groupby(key)}


def parse_size_bins(spec):
    """Bin edges (m^2) and labels with the matching effective diameter."""
    edges = [float(v) for v in spec.split(",")]

    def diam(a):
        return 2 * np.sqrt(a / np.pi)

    labels = [f"<{edges[0]:g} m² (d<{diam(edges[0]):.1f} m)"]
    labels += [f"{lo:g}–{hi:g} m² (d {diam(lo):.1f}–{diam(hi):.1f} m)"
               for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f"≥{edges[-1]:g} m² (d≥{diam(edges[-1]):.1f} m)")
    return [0.0] + edges + [np.inf], labels


def load_labels_gpkg(path):
    """Crown-date labels from a labelling-tool gpkg that carries `tag`.

    Drops rows flagged `ignore` and rows whose `quality` shows they were
    never checked (UNCHECKED_QUALITY), whose flowering/leafing values are
    tool defaults. Maps flowering to isFlowering {yes, no, maybe} and
    quality to segmentation ("good" for GOOD_QUALITY, else "other") so the
    rest of the pipeline is unchanged. The crown `tag` (as a string) is the
    uuid used for aggregation.

    Returns (gt, labels, geoms, counts): gt in the load_gt column layout,
    labels the kept rows as a GeoDataFrame (same index as gt), geoms the
    per-crown polygons for the galleries, and counts the row bookkeeping.
    """
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 32617:
        gdf = gdf.to_crs(epsg=32617)
    ignore = gdf["ignore"].astype(str).str.lower() == "true"
    unchecked = gdf["quality"].isin(UNCHECKED_QUALITY) & ~ignore
    counts = {"rows": len(gdf), "ignored": int(ignore.sum()),
              "unchecked": int(unchecked.sum())}
    gdf = gdf[~ignore & ~unchecked].copy()
    fl = gdf["flowering"].astype(str).str.strip().str.lower()
    unknown = sorted(set(fl) - set(FLOWERING_MAP))
    if unknown:
        raise ValueError(f"unrecognised flowering values in {path}: {unknown}")
    gdf["dt"] = pd.to_datetime(gdf["time"]).dt.normalize()
    gdf["date"] = gdf["dt"].dt.strftime("%Y_%m_%d")
    gdf["tag"] = gdf["tag"].astype(int)
    gt = pd.DataFrame({
        "uuid": gdf["tag"].astype(str),
        "dt": gdf["dt"],
        "date": gdf["date"],
        "isFlowering": fl.map(FLOWERING_MAP),
        "leafing": gdf["leafing"].astype(float),
        "floweringIntensity": np.nan,
        "segmentation": np.where(gdf["quality"].isin(GOOD_QUALITY), "good", "other"),
        "polygon_id": gdf["tag"].astype(str) + "_" + gdf["date"],
        "globalId": gdf["GlobalID"],
        "latin": gdf["latin"],
    }, index=gdf.index)
    counts["kept"] = len(gt)
    geoms = {uuid: {"by_date": dict(zip(sub["date"], sub.geometry)),
                    "any": sub.geometry.iloc[0]}
             for uuid, sub in gdf.groupby(gt["uuid"])}
    return gt, gdf, geoms, counts


def train_chip_overlap(labels, chips_dir):
    """True where a label polygon falls in a training chip of the same date.

    The drone models were trained on the chips in chips_dir (named
    <uuid>_<YYYY-MM-DD>.tif), and every labelled crown inside a chip is
    rasterized into its mask, so any crown overlapping a chip on that date
    was seen in training. Both train and test chips count, because the test
    fold was logged during training.
    """
    rows = []
    for path in sorted(Path(chips_dir).glob("*.tif")):
        with rasterio.open(path) as src:
            rows.append({"chip_date": path.stem.rsplit("_", 1)[1].replace("-", "_"),
                         "geometry": box(*src.bounds), "crs": src.crs})
    chips = gpd.GeoDataFrame(rows, geometry="geometry", crs=rows[0]["crs"])
    j = gpd.sjoin(labels[["date", "geometry"]], chips.drop(columns="crs"),
                  predicate="intersects")
    seen = j.index[j["date"] == j["chip_date"]].unique()
    return labels.index.isin(seen)


def load_predictions(nc4_path):
    ds = xr.open_dataset(nc4_path)
    species_pred = (
        ds["species"].to_series().rename("species_pred").reset_index()
    )
    probs = (
        ds[["flowering_probability", "deciduous_probability"]]
        .to_dataframe()
        .reset_index()
    )
    long = probs.merge(species_pred, on="tag", how="left")
    long = long.rename(columns={"date": "dt"})
    long["dt"] = pd.to_datetime(long["dt"])
    return long


def prf_from_confusion(cm):
    """precision/recall/accuracy from a [[TN,FP],[FN,TP]] matrix."""
    tn, fp, fn, tp = (
        int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    )
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else float("nan")
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


def binary_metrics(y_true, y_score, threshold):
    """Return AUROC/AUPRC/P/R/Acc/confusion at threshold; None if degenerate."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        **prf_from_confusion(cm),
        "confusion": cm,
    }


def bootstrap_ci(y_true, y_score, groups, threshold, n_boot=1000, seed=0):
    """95% percentile intervals for AUROC, AUPRC, precision and recall.

    Resamples whole crowns (groups) with replacement, since crown-dates of
    one crown are not independent. Resamples with a single class are
    skipped.
    """
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        yb, sb = y_true[idx], y_score[idx]
        if yb.min() == yb.max():
            continue
        pred = sb >= threshold
        tp = int((pred & (yb == 1)).sum())
        draws.append((roc_auc_score(yb, sb), average_precision_score(yb, sb),
                      tp / pred.sum() if pred.sum() else np.nan, tp / (yb == 1).sum()))
    lo, hi = np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)
    names = ("auroc", "auprc", "precision", "recall")
    return ({f"{k}_lo": v for k, v in zip(names, lo)}
            | {f"{k}_hi": v for k, v in zip(names, hi)})


def metrics_rows(target, metrics, strata, ycol, score_col, threshold, n_boot):
    """One row per stratum: point metrics plus crown-bootstrap intervals."""
    rows = []
    for name, m in metrics.items():
        if m is None:
            continue
        sub = strata[name]
        ci = bootstrap_ci(sub[ycol].to_numpy(), sub[score_col].to_numpy(),
                          sub["tag"].to_numpy(), threshold, n_boot)
        rows.append({"target": target, "stratum": name, "n": m["n"],
                     "n_pos": m["n_pos"], "n_crowns": int(sub["tag"].nunique()),
                     **{k: m[k] for k in ("auroc", "auprc", "precision", "recall",
                                          "accuracy")}, **ci})
    return rows


def fmt_metrics(label, m):
    if m is None:
        return f"  {label}: insufficient class diversity"
    cm = m["confusion"]
    mis = int(cm[0, 1] + cm[1, 0])
    mis_rate = mis / m["n"] if m["n"] else float("nan")
    return (
        f"  {label}: n={m['n']} pos={m['n_pos']} neg={m['n_neg']} "
        f"AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} Acc={m['accuracy']:.3f} "
        f"Mismatch={mis} ({mis_rate:.1%})\n"
        f"    confusion (rows=true, cols=pred; labels=[neg,pos]):\n"
        f"      [[{cm[0,0]:>5d} {cm[0,1]:>5d}]\n"
        f"       [{cm[1,0]:>5d} {cm[1,1]:>5d}]]"
    )


def per_group_table(df, group_cols, score_col, ycol, threshold, n_boot=0,
                    min_pos=10):
    """Metrics per group (e.g. species, or species x size bin).

    Groups with fewer than min_pos positives or negatives are skipped. With
    n_boot > 0, adds crown-bootstrap 95% intervals on AUROC; when the table
    has crown_area, adds the median crown area of the group's crown-dates.
    """
    rows = []
    for key, sub in df.groupby(group_cols, observed=True, sort=False):
        y = sub[ycol].to_numpy()
        s = sub[score_col].to_numpy()
        npos = int(y.sum())
        nneg = int(len(y) - npos)
        if npos < min_pos or nneg < min_pos:
            continue
        prf = prf_from_confusion(
            confusion_matrix(y, (s >= threshold).astype(int), labels=[0, 1]))
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row |= {"n": len(y), "n_pos": npos, "n_crowns": int(sub["tag"].nunique())}
        if "crown_area" in sub:
            row["median_area_m2"] = float(sub["crown_area"].median())
        row["auroc"] = roc_auc_score(y, s)
        if n_boot:
            ci = bootstrap_ci(y, s, sub["tag"].to_numpy(), threshold, n_boot)
            row |= {"auroc_lo": ci["auroc_lo"], "auroc_hi": ci["auroc_hi"]}
        row |= {k: prf[k] for k in ("precision", "recall", "accuracy")}
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["n"] if len(group_cols) == 1 else group_cols,
                              ascending=len(group_cols) > 1)
    return out


def size_metrics(df, ycol, score_col, threshold):
    """binary_metrics() per size bin, keyed by bin label, plus the strata."""
    strata = {str(b): sub for b, sub in df.groupby("size_bin", observed=True)}
    metrics = {name: binary_metrics(sub[ycol].to_numpy(), sub[score_col].to_numpy(),
                                    threshold)
               for name, sub in strata.items()}
    return metrics, strata


def plot_metric_by_size(size_df, bin_labels, out_path, title):
    """AUROC and AUPRC with crown-bootstrap intervals per crown-size bin."""
    targets = list(dict.fromkeys(size_df["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 4.6),
                             squeeze=False)
    for ax, target in zip(axes[0], targets):
        d = size_df[size_df["target"] == target].set_index("stratum").reindex(bin_labels)
        x = np.arange(len(bin_labels))
        for k, color, dx in (("auroc", "black", -0.06), ("auprc", "tab:red", 0.06)):
            ax.errorbar(x + dx, d[k], yerr=[d[k] - d[f"{k}_lo"], d[f"{k}_hi"] - d[k]],
                        fmt="o-", color=color, capsize=3, label=k.upper())
        ax.plot(x, d["n_pos"] / d["n"], ls="--", color="0.6",
                label="prevalence (AUPRC of chance)")
        for xi, (n, p) in enumerate(zip(d["n"], d["n_pos"])):
            if not np.isnan(n):
                ax.annotate(f"{int(p)}/{int(n)}", (xi, 0.02), ha="center", fontsize=8,
                            color="0.3")
        ax.set_xticks(x, [b.replace(" (", "\n(") for b in bin_labels], fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("crown area of the dated outline")
        ax.set_title(target)
        ax.grid(alpha=0.3)
        ax.legend(loc="center right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_roc_pr(curves, title, out_path, kind):
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, (y_true, y_score) in curves.items():
        if y_true is None or len(np.unique(y_true)) < 2:
            continue
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y_true, y_score)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{label} (AUC={a:.3f})")
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            a = average_precision_score(y_true, y_score)
            ax.plot(rec, prec, label=f"{label} (AP={a:.3f})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_decid_continuous(df, out_path):
    x = df["deciduous_probability"].to_numpy()
    y = (1.0 - df["leafing"].astype(float) / 100.0).to_numpy()
    rho, _ = spearmanr(x, y)
    r, _ = pearsonr(x, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=3, alpha=0.2)
    ax.set_xlabel("predicted deciduous_probability")
    ax.set_ylabel("1 - leafing/100  (GT)")
    ax.set_title(f"Deciduous prob vs leafing (Spearman ρ={rho:.3f}, Pearson r={r:.3f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    return fig


def load_crown_geometries(gpkg_path):
    """Map each crown uuid to its polygon(s) for chip extraction.

    Returns {uuid: {"by_date": {YYYY_MM_DD: geom}, "any": geom}}. The
    gpkg stores one polygon per (uuid, date) keyed by polygon_id
    (UUID_DATE, date as YYYY-MM-DD). Since a crown's location is stable
    across dates, the "any" representative geometry is a safe fallback
    when an exact date is missing.
    """
    gdf = gpd.read_file(gpkg_path, layer="flowering_dataset")
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.crs is not None and gdf.crs.to_epsg() != 32617:
        gdf = gdf.to_crs(epsg=32617)
    gdf["uuid"] = gdf["polygon_id"].str.rsplit("_", n=1).str[0]
    gdf["gdate"] = (
        gdf["polygon_id"].str.rsplit("_", n=1).str[1].str.replace("-", "_")
    )

    geoms = {}
    for uuid, sub in gdf.groupby("uuid"):
        by_date = dict(zip(sub["gdate"], sub.geometry))
        geoms[uuid] = {"by_date": by_date, "any": sub.geometry.iloc[0]}
    return geoms


def get_geometry(geoms, uuid, date):
    """Return (geom, source) for a uuid/date, or (None, None)."""
    entry = geoms.get(uuid)
    if entry is None:
        return None, None
    if date in entry["by_date"]:
        return entry["by_date"][date], "exact-date"
    return entry["any"], "fallback-uuid"


def index_drone_images(drone_dir, pattern="*_local.tif"):
    """Map YYYY_MM_DD -> drone GeoTIFF path."""
    index = {}
    for path in sorted(Path(drone_dir).glob(pattern)):
        try:
            index[get_date(str(path))] = path
        except ValueError:
            continue
    return index


def _chip_data_uri(tif_path, polygon, chip_size, scaling=None):
    """Crop the crown, outline it, resize, return a base64 PNG data URI.

    scaling (crown_classification.load_scaling) converts uint16 imagery,
    e.g. mavic, to the uint8 the model saw.
    """
    with rasterio.open(tif_path) as src:
        window, img = extract_centered_window(src, polygon, scaling=scaling)
        img = draw_polygon_on_image(
            img, src, window, polygon, outline=(255, 0, 0), width=3
        )
    img = img.resize((chip_size, chip_size))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fig_to_uri(fig, dpi=110):
    """Render a matplotlib figure to a base64 PNG data URI (does not close)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _esc(value):
    return html.escape("" if value is None else str(value))


def _confusion_html(m):
    """Render a binary_metrics confusion matrix as a compact 2x2 HTML table."""
    cm = m["confusion"]
    return (
        '<table class="cm"><tr><th></th><th>pred neg</th><th>pred pos</th></tr>'
        f'<tr><th>true neg</th><td>{cm[0, 0]:d}</td><td>{cm[0, 1]:d}</td></tr>'
        f'<tr><th>true pos</th><td>{cm[1, 0]:d}</td><td>{cm[1, 1]:d}</td></tr>'
        "</table>"
    )


def _metrics_table_html(metrics_by_stratum):
    """Render per-stratum binary_metrics dicts as one HTML table.

    metrics_by_stratum maps a stratum name to a binary_metrics() result
    (or None when class diversity was insufficient).
    """
    head = (
        "<table><thead><tr>"
        "<th>stratum</th><th>n</th><th>n_pos</th><th>n_neg</th>"
        "<th>AUROC</th><th>AUPRC</th><th>P</th><th>R</th><th>Acc</th>"
        "<th>mismatch</th><th>confusion</th>"
        "</tr></thead><tbody>"
    )
    rows = []
    for name, m in metrics_by_stratum.items():
        if m is None:
            rows.append(
                f'<tr><td>{_esc(name)}</td>'
                '<td colspan="10"><i>insufficient class diversity</i></td></tr>'
            )
            continue
        cm = m["confusion"]
        mis = int(cm[0, 1] + cm[1, 0])
        mis_rate = mis / m["n"] if m["n"] else float("nan")
        rows.append(
            f"<tr><td>{_esc(name)}</td>"
            f"<td>{m['n']:,}</td><td>{m['n_pos']:,}</td><td>{m['n_neg']:,}</td>"
            f"<td>{m['auroc']:.3f}</td><td>{m['auprc']:.3f}</td>"
            f"<td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td>"
            f"<td>{m['accuracy']:.3f}</td>"
            f"<td>{mis:,} ({mis_rate:.1%})</td>"
            f"<td>{_confusion_html(m)}</td></tr>"
        )
    return head + "".join(rows) + "</tbody></table>"


def _df_table_html(df, empty_msg="(none qualify)"):
    """Render a DataFrame as an HTML table, or a placeholder if empty."""
    if df is None or len(df) == 0:
        return f"<p class='muted'>{_esc(empty_msg)}</p>"
    return df.to_html(index=False, border=0, classes="df", float_format=
                      lambda v: f"{v:.3f}")


def _size_section_html(stats, prefix):
    """By-size metrics and species x size table, if the run has crown areas."""
    if f"{prefix}_size" not in stats:
        return ""
    return (
        "<h3>By crown area of the dated outline (segmentation==good)</h3>"
        + _metrics_table_html(stats[f"{prefix}_size"])
        + "<h3>Per species and crown-size bin (segmentation==good, n_pos&ge;10)</h3>"
        + _df_table_html(stats[f"{prefix}_species_size"])
    )


def render_summary_report(out_path, stats, plot_uris, link_targets,
                          flower_threshold, decid_threshold,
                          leafing_threshold, no_visualize):
    """Write the single high-level summary HTML report.

    stats is a dict of the captured statistics (join funnel, grouped
    means, per-stratum metrics, sweep/per-species tables, continuous
    correlation). plot_uris maps a plot key to a PNG data URI.
    link_targets maps a label to a relative href for the linked HTML/CSV
    artifacts.
    """
    f = stats["funnel"]
    funnel = (
        f"GT rows: {f['gt_rows']:,} (uuids {f['gt_uuids']:,}; aggregated from "
        f"{f['gt_rows_raw']:,} labels, {f['dropped_conflicts']} yes/no ties "
        f"dropped) &middot; "
        + (f"uuid&rarr;tag mappings: {f['bridge']:,} "
           + (f"from the dated crown map ({f['shared_tag']} crowns sharing a "
              "tag dropped) " if f.get("shared_tag") is not None else "")
           + (f"({f['in_train_chips']:,} crown-dates in same-date training "
              "chips excluded) " if f.get("in_train_chips") is not None else "")
           + "&middot; "
           if f["bridge"] is not None else
           f"tags from label gpkg ({_esc(f['label_counts'])}) &middot; ")
        + f"prediction rows: {f['pred_rows']:,} (tags {f['pred_tags']:,}) "
        f"&middot; after uuid&rarr;tag merge: {f['after_uuid']:,} &middot; "
        f"after (tag,date) join: {f['after_join']:,}"
    )

    links_html = "".join(
        f'<li><a href="{_esc(href)}">{_esc(label)}</a></li>'
        for label, href in link_targets.items()
    )
    novis_note = (
        '<p class="muted">Run with <code>--no-visualize</code>: the '
        "discrepancy galleries were not regenerated this run and may be "
        "absent or stale.</p>"
        if no_visualize else ""
    )

    cont = stats["decid_continuous"]
    cont_line = (
        f"n={cont['n']:,} &middot; Pearson r={cont['r']:.3f} &middot; "
        f"Spearman &rho;={cont['rho']:.3f}"
    )

    size_fig = ""
    if "by_size" in plot_uris:
        size_fig = (
            "<h2>By crown size</h2>"
            f'<div class="plots"><img src="{plot_uris["by_size"]}" '
            'alt="Metrics by crown size" style="max-width:100%"/></div>'
            f"<p>{_esc(stats['size_caption'])}</p>"
        )

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>GT vs ML classifier &mdash; summary report</title>
<style>
  body {{ font-family: sans-serif; margin: 24px; max-width: 1100px;
          color:#222; background:#fafafa; }}
  h1 {{ font-size: 24px; }}
  h2 {{ font-size: 19px; border-bottom:2px solid #ddd; padding-bottom:4px;
        margin-top:32px; }}
  h3 {{ font-size: 15px; margin-bottom:6px; }}
  .funnel {{ color:#333; background:#fff; border:1px solid #ddd;
             border-radius:6px; padding:10px 12px; }}
  table {{ border-collapse:collapse; margin:6px 0 14px; font-size:13px; }}
  th, td {{ border:1px solid #ddd; padding:4px 9px; text-align:right; }}
  thead th, table.cm th {{ background:#f0f0f0; }}
  td:first-child, th:first-child {{ text-align:left; }}
  table.cm {{ display:inline-table; margin:0; font-size:11px; }}
  table.cm td {{ text-align:right; }}
  .plots {{ display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start; }}
  .plots img {{ border:1px solid #ddd; border-radius:4px; background:#fff;
                max-width:420px; height:auto; }}
  .muted {{ color:#888; font-style:italic; }}
  code {{ background:#eee; padding:1px 4px; border-radius:3px; }}
  ul.links li {{ margin:3px 0; }}
</style></head><body>
<h1>ML crown classifier vs ground truth &mdash; summary</h1>
<div class="funnel">{funnel}</div>
<p class="muted">Thresholds: flowering&ge;{flower_threshold},
deciduous&ge;{decid_threshold}, GT-deciduous when
leafing&lt;{leafing_threshold}.</p>

<h2>Flowering</h2>
<h3>Mean predicted flowering_probability by GT <code>isFlowering</code></h3>
{stats['flower_groupby_html']}
<h3>Binary metrics (threshold={flower_threshold})</h3>
{_metrics_table_html(stats['flower_metrics'])}
<h3>Per-species AUROC (segmentation==good, n_pos&ge;10)</h3>
{_df_table_html(stats['flower_species'])}
{_size_section_html(stats, 'flower')}
<div class="plots">
  <img src="{plot_uris['flower_roc']}" alt="Flowering ROC"/>
  <img src="{plot_uris['flower_pr']}" alt="Flowering PR"/>
</div>

<h2>Deciduous</h2>
<h3>Binary metrics (leafing&lt;{leafing_threshold};
threshold={decid_threshold})</h3>
{_metrics_table_html(stats['decid_metrics'])}
<h3>AUROC swept over leafing thresholds (segmentation==good)</h3>
{_df_table_html(stats['decid_sweep'])}
<h3>Continuous deciduous_probability vs 1&minus;leafing/100
(segmentation==good)</h3>
<p>{cont_line}</p>
<h3>Per-species AUROC (segmentation==good, leafing&lt;{leafing_threshold},
n_pos&ge;10)</h3>
{_df_table_html(stats['decid_species'])}
{_size_section_html(stats, 'decid')}
<div class="plots">
  <img src="{plot_uris['decid_roc']}" alt="Deciduous ROC"/>
  <img src="{plot_uris['decid_pr']}" alt="Deciduous PR"/>
  <img src="{plot_uris['decid_continuous']}" alt="Deciduous continuous"/>
</div>

{size_fig}
<h2>Detailed reports</h2>
{novis_note}
<ul class="links">{links_html}</ul>
</body></html>"""
    Path(out_path).write_text(doc)
    print(f"Wrote summary report: {out_path}")


def render_discrepancy_report(merged, geoms, drone_index, out_path,
                              target, n, chip_size, scaling=None):
    """Write an HTML report of the worst GT-vs-prediction discrepancies.

    target is "flowering" or "deciduous". Ranks every joined row by how
    far the predicted probability sits from ground truth, then renders
    the worst n as cards with the crown chip and all identifiers needed
    to find the instance in gt_vs_ml_joined.csv.
    """
    df = merged.copy()
    if target == "flowering":
        df = df[df["isFlowering"].isin(["yes", "no"])].copy()
        df["gt_value"] = (df["isFlowering"] == "yes").astype(float)
        df["pred"] = df["flowering_probability"]
        df["gt_display"] = df["isFlowering"]
        gt_label = "isFlowering"
        pos_label, neg_label = "yes", "no"
    else:
        df = df.dropna(subset=["leafing"]).copy()
        df["gt_value"] = 1.0 - df["leafing"].astype(float) / 100.0
        df["pred"] = df["deciduous_probability"]
        df["gt_display"] = df.apply(
            lambda r: f"leafing={int(r['leafing'])} "
                      f"(decid={r['gt_value']:.2f})",
            axis=1,
        )
        gt_label = "leafing -> deciduous"
        pos_label, neg_label = "deciduous", "leafy"

    df = df.dropna(subset=["pred"]).copy()
    n_total = len(df)
    df["discrepancy"] = (df["pred"] - df["gt_value"]).abs()

    df["date_str"] = df["dt"].dt.strftime("%Y_%m_%d")

    # Companion CSV: polygon IDs (and identifiers) for the crown-dates whose
    # predicted class disagrees with GT (binary classes differ at threshold
    # 0.5 — i.e. a label mismatch), ranked by discrepancy. Includes crowns
    # lacking a drone image or geometry that the gallery below omits.
    opt_cols = [c for c in ("observation_id", "globalId") if c in df.columns]
    csv_cols = [
        c for c in (
            ["polygon_id", "uuid", "tag"] + opt_cols
            + ["date_str", "discrepancy", "pred", "gt_value", "gt_display",
               "segmentation", "latin", "species_pred"]
        ) if c in df.columns
    ]
    mismatch = (df["gt_value"] >= 0.5) != (df["pred"] >= 0.5)
    csv_df = df[mismatch]
    csv_path = Path(out_path).with_suffix(".csv")
    (csv_df.sort_values("discrepancy", ascending=False)[csv_cols]
       .rename(columns={"date_str": "date"})
       .to_csv(csv_path, index=False))
    print(f"Wrote {csv_path}  ({len(csv_df):,} mismatches of {len(df):,} rows)")

    df = df[df["date_str"].isin(drone_index)].copy()
    n_with_image = len(df)
    # A crown is hand-segmented on only ~5 dates but is GT-labelled and
    # imaged on nearly all of them; its location is stable across dates,
    # so any per-uuid polygon outlines it correctly. Require only that
    # the crown has *some* geometry, preferring the exact-date polygon
    # and falling back to a representative one (labelled per card).
    df = df[df["uuid"].isin(geoms)].copy()
    n_eligible = len(df)

    df = df.sort_values("discrepancy", ascending=False).head(n)

    def _cls_chip(value):
        """Colored chip for the binary class of a 0-1 value at threshold 0.5."""
        is_pos = value >= 0.5
        name = pos_label if is_pos else neg_label
        kind = "pos" if is_pos else "neg"
        return f'<span class="cls cls-{kind}">{name}</span>'

    cards = []
    n_failed = 0
    for _, row in df.iterrows():
        polygon, geom_src = get_geometry(geoms, row["uuid"], row["date_str"])
        tif_path = drone_index[row["date_str"]]
        try:
            uri = _chip_data_uri(tif_path, polygon, chip_size, scaling)
        except Exception as exc:  # noqa: BLE001 - one bad chip must not abort
            n_failed += 1
            print(f"  chip failed for {row['polygon_id']}: {exc}",
                  file=sys.stderr)
            continue

        seg = row.get("segmentation")
        seg_class = (
            "good" if seg == "good" else "poor" if seg == "poor" else "other"
        )
        id_rows = [
            ("polygon_id", row.get("polygon_id")),
            ("uuid", row.get("uuid")),
            ("tag", row.get("tag")),
            ("date", row["date_str"]),
        ]
        id_rows += [(c, row.get(c)) for c in opt_cols]
        id_rows.append(("drone tif", tif_path.name))
        id_rows.append(("geometry", geom_src))
        id_html = "".join(
            f'<div><span class="k">{_esc(k)}:</span>'
            f'<span class="v">{_esc(v)}</span></div>'
            for k, v in id_rows
        )
        cards.append(f"""
        <div class="card">
          <img src="{uri}" width="{chip_size}" height="{chip_size}"/>
          <div class="meta">
            <div class="disc">discrepancy = {row['discrepancy']:.3f}</div>
            <div class="vals">
              GT ({_esc(gt_label)}): <b>{_esc(row['gt_display'])}</b>
                {_cls_chip(row['gt_value'])}<br/>
              predicted prob: <b>{row['pred']:.3f}</b>
                {_cls_chip(row['pred'])}<br/>
              species (pred): {_esc(row.get('species_pred'))}<br/>
              species (GT): {_esc(row.get('latin'))}<br/>
              segmentation:
                <span class="seg seg-{seg_class}">{_esc(seg)}</span>
            </div>
            <div class="ids">{id_html}</div>
          </div>
        </div>""")

    summary = (
        f"target=<b>{_esc(target)}</b> &middot; "
        f"joined rows with valid GT/pred: {n_total:,} &middot; "
        f"with drone image: {n_with_image:,} &middot; "
        f"with geometry (eligible): {n_eligible:,} &middot; "
        f"showing worst {len(cards):,}"
        + (f" ({n_failed} chips failed)" if n_failed else "")
    )
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>GT vs ML discrepancies: {_esc(target)}</title>
<style>
  body {{ font-family: sans-serif; margin: 16px; background:#fafafa; }}
  h1 {{ font-size: 20px; }}
  .summary {{ color:#333; margin-bottom:16px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .card {{ background:#fff; border:1px solid #ddd; border-radius:6px;
           padding:8px; width:{chip_size + 18}px; box-shadow:0 1px 2px #0001; }}
  .card img {{ display:block; border-radius:4px; }}
  .meta {{ font-size:12px; margin-top:6px; }}
  .disc {{ font-weight:bold; font-size:14px; color:#b00; margin-bottom:4px; }}
  .vals {{ margin-bottom:6px; }}
  .ids {{ font-family:monospace; font-size:11px; background:#f4f4f4;
          border:1px solid #eee; border-radius:4px; padding:6px;
          user-select:all; }}
  .ids .k {{ color:#666; }}
  .ids .v {{ color:#000; margin-left:4px; }}
  .seg {{ padding:1px 5px; border-radius:3px; font-weight:bold; }}
  .seg-good {{ background:#d6f5d6; color:#0a0; }}
  .seg-poor {{ background:#fdd; color:#a00; }}
  .seg-other {{ background:#eee; color:#555; }}
  .cls {{ padding:1px 5px; border-radius:3px; font-weight:bold;
          font-size:11px; }}
  .cls-pos {{ background:#d3f9d8; color:#2b8a3e; }}
  .cls-neg {{ background:#ffe3e3; color:#c92a2a; }}
</style></head><body>
<h1>Largest GT-vs-classification discrepancies &mdash; {_esc(target)}</h1>
<div class="summary">{summary}</div>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    Path(out_path).write_text(doc)
    print(
        f"Wrote {out_path}  (eligible={n_eligible:,}, shown={len(cards):,}, "
        f"failed={n_failed})"
    )


def render_conflict_report(flower_conflicts, geoms, drone_index, out_path,
                           chip_size, scaling=None):
    """Write an HTML page of crown-dates dropped for a yes/no flowering tie.

    flower_conflicts holds the raw GT label rows (multiple per crown-date) for
    combos where the yes and no votes were equal and nonzero, so no majority
    label could be assigned. Each card shows the crown chip and every
    conflicting annotation (isFlowering, segmentation, observation_id).
    """
    by_crown = list(flower_conflicts.groupby(["uuid", "dt"], sort=False))
    cards = []
    n_failed = 0
    n_no_image = 0
    for (uuid, dt), sub in by_crown:
        date_str = dt.strftime("%Y_%m_%d")
        if date_str not in drone_index or uuid not in geoms:
            n_no_image += 1
            chip = '<div class="noimg">no drone image / geometry</div>'
        else:
            polygon, geom_src = get_geometry(geoms, uuid, date_str)
            try:
                uri = _chip_data_uri(drone_index[date_str], polygon, chip_size,
                                     scaling)
                chip = (f'<img src="{uri}" width="{chip_size}" '
                        f'height="{chip_size}"/>')
            except Exception as exc:  # noqa: BLE001 - one bad chip mustn't abort
                n_failed += 1
                print(f"  chip failed for {uuid} {date_str}: {exc}",
                      file=sys.stderr)
                chip = '<div class="noimg">chip failed</div>'

        opt_cols = [c for c in ("observation_id", "globalId") if c in sub.columns]
        rows_html = "".join(
            "<tr>"
            f'<td><span class="seg seg-{("good" if r.get("segmentation")=="good" else "poor" if r.get("segmentation")=="bad" else "other")}">'
            f'{_esc(r.get("isFlowering"))}</span></td>'
            f'<td>{_esc(r.get("segmentation"))}</td>'
            + "".join(f"<td>{_esc(r.get(c))}</td>" for c in opt_cols)
            + "</tr>"
            for _, r in sub.iterrows()
        )
        head = ("<tr><th>isFlowering</th><th>segmentation</th>"
                + "".join(f"<th>{_esc(c)}</th>" for c in opt_cols) + "</tr>")
        first = sub.iloc[0]
        ids = (f'<div><span class="k">polygon_id:</span>'
               f'<span class="v">{_esc(first.get("polygon_id"))}</span></div>'
               f'<div><span class="k">latin:</span>'
               f'<span class="v">{_esc(first.get("latin"))}</span></div>'
               f'<div><span class="k">date:</span>'
               f'<span class="v">{_esc(date_str)}</span></div>')
        cards.append(f"""
        <div class="card">
          {chip}
          <div class="meta">
            <div class="disc">{len(sub)} conflicting labels</div>
            <table class="votes"><thead>{head}</thead><tbody>{rows_html}</tbody></table>
            <div class="ids">{ids}</div>
          </div>
        </div>""")

    summary = (
        f"Crown-dates dropped from flowering metrics because their yes/no "
        f"annotator votes were tied: <b>{len(by_crown):,}</b>"
        + (f" &middot; {n_no_image} without drone image/geometry" if n_no_image else "")
        + (f" &middot; {n_failed} chips failed" if n_failed else "")
    )
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>GT flowering label conflicts</title>
<style>
  body {{ font-family: sans-serif; margin: 16px; background:#fafafa; }}
  h1 {{ font-size: 20px; }}
  .summary {{ color:#333; margin-bottom:16px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .card {{ background:#fff; border:1px solid #ddd; border-radius:6px;
           padding:8px; width:{chip_size + 18}px; box-shadow:0 1px 2px #0001; }}
  .card img {{ display:block; border-radius:4px; }}
  .noimg {{ width:{chip_size}px; height:{chip_size}px; display:flex;
            align-items:center; justify-content:center; background:#eee;
            color:#888; border-radius:4px; font-size:12px; text-align:center; }}
  .meta {{ font-size:12px; margin-top:6px; }}
  .disc {{ font-weight:bold; font-size:14px; color:#b00; margin-bottom:4px; }}
  table.votes {{ border-collapse:collapse; font-size:11px; margin-bottom:6px; }}
  table.votes th, table.votes td {{ border:1px solid #ddd; padding:2px 5px;
                                    text-align:left; }}
  .seg {{ padding:1px 5px; border-radius:3px; font-weight:bold; }}
  .seg-good {{ background:#d6f5d6; color:#0a0; }}
  .seg-poor {{ background:#fdd; color:#a00; }}
  .seg-other {{ background:#eee; color:#555; }}
  .ids {{ font-family:monospace; font-size:11px; background:#f4f4f4;
          border:1px solid #eee; border-radius:4px; padding:6px;
          user-select:all; }}
  .ids .k {{ color:#666; }}
  .ids .v {{ color:#000; margin-left:4px; }}
</style></head><body>
<h1>Dropped flowering label conflicts (yes/no ties)</h1>
<div class="summary">{summary}</div>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    Path(out_path).write_text(doc)
    print(f"Wrote {out_path}  (conflicts={len(by_crown):,})")


@click.command()
@click.option("--gt-csv", default=GT_CSV_DEFAULT, show_default=True)
@click.option("--gpkg", default=GPKG_DEFAULT, show_default=True)
@click.option("--nc4", default=NC4_DEFAULT, show_default=True)
@click.option("--geo-folds", default=GEO_FOLDS_DEFAULT, show_default=True,
              type=click.Path(dir_okay=False),
              help="CSV with polygon_id,split train/test folds.")
@click.option("--exclude-train", is_flag=True,
              help="Drop GT labels for any crown (uuid) in the train split "
                   "of --geo-folds, so metrics use held-out crowns only.")
@click.option("--labels-gpkg", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Score a labelling-tool gpkg with a `tag` per crown-date "
                   "(see load_labels_gpkg) instead of --gt-csv + --gpkg.")
@click.option("--exclude-train-chips", is_flag=True,
              help="Drop crown-dates whose polygon (from --labels-gpkg or "
                   "--crownmap) lies in a training chip of the same date "
                   "(--train-chips).")
@click.option("--crownmap", default=CROWNMAP_DEFAULT, show_default=True,
              help="Dated crown map giving the uuid->tag bridge, crown areas "
                   "and gallery outlines for --gt-csv; '' uses --gpkg instead.")
@click.option("--size-bins", default=SIZE_BINS_DEFAULT, show_default=True,
              help="Comma-separated crown-area bin edges (m^2).")
@click.option("--train-chips", default=TRAIN_CHIPS_DEFAULT, show_default=True,
              type=click.Path(file_okay=False),
              help="Directory of the drone models' training chips.")
@click.option("--n-boot", default=1000, show_default=True, type=int,
              help="Crown bootstrap resamples for the metric intervals.")
@click.option(
    "--output-dir",
    default="reports",
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option("--flower-threshold", default=0.5, show_default=True, type=float)
@click.option("--decid-threshold", default=0.5, show_default=True, type=float)
@click.option(
    "--leafing-threshold",
    default=50,
    show_default=True,
    type=int,
    help="leafing < threshold => GT-deciduous for binary metrics.",
)
@click.option(
    "--drone-dir",
    default=DRONE_DIR_DEFAULT,
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory of BCI_50ha_*_local.tif drone images for chips.",
)
@click.option("--drone-glob", default="*_local.tif", show_default=True,
              help="Filename pattern of the drone images in --drone-dir.")
@click.option("--drone-scaling", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="crown_classification scaling config for uint16 drone "
                   "images (e.g. config/crown_classification_mavic_mkl.yml).")
@click.option("--n-examples", default=40, show_default=True, type=int,
              help="Number of worst discrepancy examples per target.")
@click.option("--chip-size", default=300, show_default=True, type=int,
              help="Pixel size of each crown chip in the HTML report.")
@click.option("--no-visualize", is_flag=True,
              help="Skip the discrepancy HTML reports (metrics only).")
def main(gt_csv, gpkg, nc4, geo_folds, exclude_train, labels_gpkg,
         exclude_train_chips, crownmap, size_bins, train_chips, n_boot,
         output_dir, flower_threshold, decid_threshold, leafing_threshold,
         drone_dir, drone_glob, drone_scaling, n_examples, chip_size,
         no_visualize):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_uris = {}

    label_counts = None
    crowns = None
    n_shared = None
    in_train_chips = None
    if labels_gpkg:
        print(f"Loading labels gpkg: {labels_gpkg}")
        gt, label_gdf, label_geoms, label_counts = load_labels_gpkg(labels_gpkg)
        print(f"  rows: {label_counts['rows']:,}; ignored {label_counts['ignored']:,}; "
              f"unchecked (tool defaults) {label_counts['unchecked']:,}; "
              f"kept {label_counts['kept']:,}")
        if exclude_train_chips:
            seen = train_chip_overlap(label_gdf, train_chips)
            gt = gt[~seen].copy()
            label_counts["in_train_chips"] = int(seen.sum())
            print(f"  excluded {int(seen.sum()):,} crown-dates inside same-date "
                  f"training chips; GT rows -> {len(gt):,}")
    else:
        print(f"Loading GT  : {gt_csv}")
        gt = load_gt(gt_csv)
        print(f"  rows: {len(gt):,}  unique uuids: {gt['uuid'].nunique():,}")
        if crownmap:
            print(f"Loading dated crown map: {crownmap}")
            crowns, n_shared = load_crownmap(crownmap)
            print(f"  crown-dates: {len(crowns):,}; crowns: {crowns['uuid'].nunique():,} "
                  f"({n_shared} crowns sharing a tag dropped)")
            if exclude_train_chips:
                label_geo = crowns.merge(gt[["uuid", "date"]].drop_duplicates(),
                                         on=["uuid", "date"])
                seen = label_geo.loc[train_chip_overlap(label_geo, train_chips),
                                     ["uuid", "date"]]
                key = pd.MultiIndex.from_frame(gt[["uuid", "date"]])
                drop = key.isin(pd.MultiIndex.from_frame(seen))
                gt = gt[~drop].copy()
                in_train_chips = len(seen)
                print(f"  excluded {len(seen):,} crown-dates inside same-date "
                      f"training chips ({int(drop.sum()):,} labels); "
                      f"GT rows -> {len(gt):,}")
        elif exclude_train_chips:
            raise click.UsageError("--exclude-train-chips needs --labels-gpkg "
                                   "or --crownmap")

    if exclude_train and not labels_gpkg:
        train_uuids = load_train_uuids(geo_folds)
        before = len(gt)
        gt = gt[~gt["uuid"].isin(train_uuids)].copy()
        print(f"  excluded train crowns: {len(train_uuids):,} uuids; "
              f"GT rows {before:,} -> {len(gt):,}")

    gt_rows_raw = len(gt)
    gt, flower_conflicts = aggregate_gt_labels(gt)
    n_conf = flower_conflicts.groupby(["uuid", "dt"]).ngroups
    print(f"  aggregated to one row per (crown,date): {gt_rows_raw:,} -> "
          f"{len(gt):,} rows; dropped {n_conf} flowering yes/no ties")

    areas = None
    if labels_gpkg:
        bridge = None
        merged_uuid = gt.assign(tag=gt["uuid"].astype(int))
        areas = pd.DataFrame({"uuid": label_gdf["tag"].astype(str),
                              "date": label_gdf["date"],
                              "crown_area": label_gdf.geometry.area})
    elif crowns is not None:
        bridge = crowns[["uuid", "tag"]].drop_duplicates()
        print(f"  uuid->tag mappings from the dated crown map: {len(bridge):,}")
        merged_uuid = gt.merge(bridge, on="uuid", how="inner")
        print(f"After UUID->tag merge: {len(merged_uuid):,} rows")
        areas = crowns[["uuid", "date", "crown_area"]]
    else:
        print(f"Loading bridge: {gpkg}")
        bridge = load_uuid_tag_map(gpkg)
        print(f"  uuid->tag mappings: {len(bridge):,}")
        merged_uuid = gt.merge(bridge, on="uuid", how="inner")
        print(f"After UUID->tag merge: {len(merged_uuid):,} rows")

    bin_edges, bin_labels = parse_size_bins(size_bins)
    if areas is not None:
        areas = areas.drop_duplicates(["uuid", "date"])
        merged_uuid = merged_uuid.merge(areas, on=["uuid", "date"], how="left")
        merged_uuid["size_bin"] = pd.cut(merged_uuid["crown_area"], bin_edges,
                                         right=False, labels=bin_labels)
        print(f"  crown area found for {merged_uuid['crown_area'].notna().mean():.1%} "
              "of crown-dates")

    print(f"Loading nc4 : {nc4}")
    preds = load_predictions(nc4)
    print(f"  prediction rows: {len(preds):,}  tags: {preds['tag'].nunique():,}")

    merged = merged_uuid.merge(preds, on=["tag", "dt"], how="inner")
    print(f"After (tag, date) join: {len(merged):,} rows")
    no_pred = merged[["flowering_probability", "deciduous_probability"]].isna().any(axis=1)
    if no_pred.any():
        # Crowns outside the classifier's windows have no prediction.
        merged = merged[~no_pred].copy()
        print(f"  dropped {int(no_pred.sum()):,} rows with no prediction -> {len(merged):,}")
    if len(merged) == 0:
        print("No overlapping rows; aborting.", file=sys.stderr)
        sys.exit(1)

    csv_out = out_dir / "gt_vs_ml_joined.csv"
    merged_out = merged.drop(columns=["dt"]).copy()
    merged_out.to_csv(csv_out, index=False)
    print(f"Wrote joined CSV: {csv_out}  ({len(merged_out):,} rows)")

    # ----- Flowering -----
    print("\n=== Flowering ===")
    print("Mean predicted flowering_probability by GT isFlowering bucket:")
    grp = merged.groupby("isFlowering")["flowering_probability"].agg(
        ["count", "mean", "median"]
    )
    print(grp.to_string())

    binary_flower = merged[merged["isFlowering"].isin(["yes", "no"])].copy()
    binary_flower["y"] = (binary_flower["isFlowering"] == "yes").astype(int)

    strata = {
        "all": binary_flower,
        "segmentation==good": binary_flower[
            binary_flower["segmentation"] == "good"
        ],
    }

    print(f"\nFlowering metrics (threshold={flower_threshold}):")
    flower_curves_roc = {}
    flower_curves_pr = {}
    flower_metrics = {}
    for name, sub in strata.items():
        m = binary_metrics(
            sub["y"].to_numpy(),
            sub["flowering_probability"].to_numpy(),
            flower_threshold,
        )
        flower_metrics[name] = m
        print(fmt_metrics(name, m))
        flower_curves_roc[name] = (
            sub["y"].to_numpy(), sub["flowering_probability"].to_numpy()
        )
        flower_curves_pr[name] = flower_curves_roc[name]

    print("\nPer-species flowering AUROC (segmentation==good, n_pos>=10):")
    sp_tbl = per_group_table(
        strata["segmentation==good"], ["latin"], "flowering_probability", "y",
        flower_threshold, n_boot,
    )
    print(sp_tbl.to_string(index=False) if len(sp_tbl) else "  (none qualify)")

    fig = plot_roc_pr(
        flower_curves_roc,
        "Flowering ROC",
        out_dir / "gt_vs_ml_flowering_roc.pdf",
        "roc",
    )
    plot_uris["flower_roc"] = _fig_to_uri(fig)
    plt.close(fig)
    fig = plot_roc_pr(
        flower_curves_pr,
        "Flowering PR",
        out_dir / "gt_vs_ml_flowering_pr.pdf",
        "pr",
    )
    plot_uris["flower_pr"] = _fig_to_uri(fig)
    plt.close(fig)

    # ----- Deciduous -----
    print("\n=== Deciduous ===")
    valid_leaf = merged.dropna(subset=["leafing"]).copy()
    valid_leaf["y_leaf"] = (
        valid_leaf["leafing"].astype(float) < leafing_threshold
    ).astype(int)

    strata_d = {
        "all": valid_leaf,
        "segmentation==good": valid_leaf[valid_leaf["segmentation"] == "good"],
    }

    print(
        f"\nDeciduous binary metrics (leafing<{leafing_threshold}; "
        f"threshold={decid_threshold}):"
    )
    decid_curves_roc = {}
    decid_curves_pr = {}
    decid_metrics = {}
    for name, sub in strata_d.items():
        m = binary_metrics(
            sub["y_leaf"].to_numpy(),
            sub["deciduous_probability"].to_numpy(),
            decid_threshold,
        )
        decid_metrics[name] = m
        print(fmt_metrics(name, m))
        decid_curves_roc[name] = (
            sub["y_leaf"].to_numpy(),
            sub["deciduous_probability"].to_numpy(),
        )
        decid_curves_pr[name] = decid_curves_roc[name]

    print("\nDeciduous AUROC swept over leafing thresholds (segmentation==good):")
    good = strata_d["segmentation==good"]
    sweep_rows = []
    for thr in (1, 25, 50, 75):
        y = (good["leafing"].astype(float) < thr).astype(int)
        if len(np.unique(y)) < 2:
            sweep_rows.append({"leafing<": thr, "n_pos": int(y.sum()),
                               "auroc": float("nan"), "precision": float("nan"),
                               "recall": float("nan"), "accuracy": float("nan")})
            continue
        y_pred = (good["deciduous_probability"] >= decid_threshold).astype(int)
        prf = prf_from_confusion(confusion_matrix(y, y_pred, labels=[0, 1]))
        sweep_rows.append({
            "leafing<": thr,
            "n_pos": int(y.sum()),
            "auroc": roc_auc_score(y, good["deciduous_probability"]),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "accuracy": prf["accuracy"],
        })
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False))

    cont = good.dropna(subset=["deciduous_probability"])
    cont_y = 1.0 - cont["leafing"].astype(float) / 100.0
    rho, _ = spearmanr(cont["deciduous_probability"], cont_y)
    r, _ = pearsonr(cont["deciduous_probability"], cont_y)
    print(
        f"\nContinuous (segmentation==good): n={len(cont):,}  "
        f"Pearson r={r:.3f}  Spearman ρ={rho:.3f}"
    )

    print("\nPer-species deciduous AUROC (segmentation==good, leafing<"
          f"{leafing_threshold}, n_pos>=10):")
    sp_tbl_d = per_group_table(
        strata_d["segmentation==good"], ["latin"], "deciduous_probability",
        "y_leaf", decid_threshold, n_boot,
    )
    print(sp_tbl_d.to_string(index=False) if len(sp_tbl_d) else "  (none qualify)")

    fig = plot_roc_pr(
        decid_curves_roc,
        f"Deciduous ROC (leafing<{leafing_threshold})",
        out_dir / "gt_vs_ml_deciduous_roc.pdf",
        "roc",
    )
    plot_uris["decid_roc"] = _fig_to_uri(fig)
    plt.close(fig)
    fig = plot_roc_pr(
        decid_curves_pr,
        f"Deciduous PR (leafing<{leafing_threshold})",
        out_dir / "gt_vs_ml_deciduous_pr.pdf",
        "pr",
    )
    plot_uris["decid_pr"] = _fig_to_uri(fig)
    plt.close(fig)
    fig = plot_decid_continuous(cont, out_dir / "gt_vs_ml_deciduous_continuous.pdf")
    plot_uris["decid_continuous"] = _fig_to_uri(fig)
    plt.close(fig)

    metrics_df = pd.DataFrame(
        metrics_rows("flowering", flower_metrics, strata, "y",
                     "flowering_probability", flower_threshold, n_boot)
        + metrics_rows(f"deciduous (leafing<{leafing_threshold})", decid_metrics,
                       strata_d, "y_leaf", "deciduous_probability",
                       decid_threshold, n_boot))
    print(f"\nMetrics with 95% crown-bootstrap intervals ({n_boot} resamples):")
    print(metrics_df.round(3).to_string(index=False))

    species_df = pd.concat([sp_tbl.assign(target="flowering"),
                            sp_tbl_d.assign(target=f"deciduous (leafing<{leafing_threshold})")])
    species_df.to_csv(out_dir / "gt_vs_ml_by_species.csv", index=False)

    # ----- By crown size (segmentation==good) -----
    size_stats = {}
    if "size_bin" in merged:
        print("\n=== By crown area of the dated outline (segmentation==good) ===")
        decid_target = f"deciduous (leafing<{leafing_threshold})"
        size_rows = []
        for prefix, target, sub, ycol, col, thr in (
                ("flower", "flowering", strata["segmentation==good"], "y",
                 "flowering_probability", flower_threshold),
                ("decid", decid_target, strata_d["segmentation==good"], "y_leaf",
                 "deciduous_probability", decid_threshold)):
            m, st = size_metrics(sub, ycol, col, thr)
            size_stats[f"{prefix}_size"] = {b: m[b] for b in bin_labels if b in m}
            for b in bin_labels:
                if b in m:
                    print(fmt_metrics(b, m[b]))
            size_rows += metrics_rows(target, m, st, ycol, col, thr, n_boot)
            sxs = per_group_table(sub, ["latin", "size_bin"], col, ycol, thr, n_boot)
            size_stats[f"{prefix}_species_size"] = sxs
            print(f"\n{target} per species x size bin (n_pos>=10):")
            print(sxs.round(3).to_string(index=False) if len(sxs) else "  (none qualify)")
        size_df = pd.DataFrame(size_rows)
        edges = dict(zip(bin_labels, zip(bin_edges[:-1], bin_edges[1:])))
        size_df["area_lo_m2"] = size_df["stratum"].map(lambda b: edges[b][0])
        size_df["area_hi_m2"] = size_df["stratum"].map(lambda b: edges[b][1])
        size_df.to_csv(out_dir / "gt_vs_ml_by_size.csv", index=False)
        metrics_df = pd.concat([metrics_df, size_df.drop(
            columns=["area_lo_m2", "area_hi_m2"]).assign(
            stratum=lambda d: "segmentation==good, area " + d["stratum"])])
        pd.concat([g.assign(target=t) for t, g in (
            ("flowering", size_stats["flower_species_size"]),
            (decid_target, size_stats["decid_species_size"]))]).to_csv(
            out_dir / "gt_vs_ml_by_species_size.csv", index=False)

        dates = merged["dt"]
        size_stats["size_caption"] = (
            "Agreement between the drone classifier and the ground-truth labels "
            "as a function of crown size: AUROC (black) and AUPRC (red) of the "
            "crown-mean predicted probability, with the label prevalence "
            "(dashed grey, the AUPRC of a random classifier), for flowering "
            "(label yes vs no; 'maybe' excluded) and deciduousness (leafing "
            f"< {leafing_threshold}%). Crown size is the area of the crown's "
            "outline on that date in the dated crown map, the same polygon "
            "the probability is averaged over, binned in m² (effective "
            "diameter d = 2√(area/π)). Only crown-dates with good segmentation "
            "are included; numbers along the bottom are positives/total "
            "crown-dates per bin, and error bars are 95% intervals from 1,000 "
            "bootstrap resamples of whole crowns. Labels cover "
            f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}."
        ).replace("1,000", f"{n_boot:,}")
        fig = plot_metric_by_size(
            size_df, bin_labels, out_dir / "gt_vs_ml_by_size.pdf",
            "Classifier vs labels by crown area (segmentation==good; "
            "95% crown-bootstrap intervals)")
        plot_uris["by_size"] = _fig_to_uri(fig)
        fig.savefig(out_dir / "gt_vs_ml_by_size.png", dpi=130)
        plt.close(fig)

    metrics_df.to_csv(out_dir / "gt_vs_ml_metrics.csv", index=False)

    print(f"\nPlots written to: {out_dir}/")

    # ----- Discrepancy visualization -----
    if not no_visualize:
        print("\n=== Discrepancy reports ===")
        if labels_gpkg:
            geoms = label_geoms
        elif crowns is not None:
            geoms = geoms_by_crown(crowns[crowns["uuid"].isin(merged["uuid"])])
        else:
            print(f"Loading crown geometries: {gpkg}")
            geoms = load_crown_geometries(gpkg)
        print(f"  crowns with geometry: {len(geoms):,}")
        print(f"Indexing drone images: {drone_dir} ({drone_glob})")
        drone_index = index_drone_images(drone_dir, drone_glob)
        print(f"  drone dates: {len(drone_index):,}")
        scaling = None
        if drone_scaling:
            from crown_classification import load_scaling
            scaling = load_scaling(drone_scaling)

        for target in ("flowering", "deciduous"):
            render_discrepancy_report(
                merged,
                geoms,
                drone_index,
                out_dir / f"gt_vs_ml_{target}_discrepancies.html",
                target,
                n_examples,
                chip_size,
                scaling,
            )

        render_conflict_report(
            flower_conflicts,
            geoms,
            drone_index,
            out_dir / "gt_vs_ml_flowering_conflicts.html",
            chip_size,
            scaling,
        )

    # ----- Single summary report -----
    print("\n=== Summary report ===")
    stats = {
        "funnel": {
            "gt_rows": len(gt),
            "gt_rows_raw": gt_rows_raw,
            "dropped_conflicts": int(n_conf),
            "gt_uuids": int(gt["uuid"].nunique()),
            "bridge": len(bridge) if bridge is not None else None,
            "shared_tag": n_shared,
            "in_train_chips": in_train_chips,
            "label_counts": label_counts,
            "pred_rows": len(preds),
            "pred_tags": int(preds["tag"].nunique()),
            "after_uuid": len(merged_uuid),
            "after_join": len(merged),
        },
        "flower_groupby_html": grp.reset_index().to_html(
            index=False, border=0, classes="df",
            float_format=lambda v: f"{v:.3f}",
        ),
        "flower_metrics": flower_metrics,
        "flower_species": sp_tbl,
        "decid_metrics": decid_metrics,
        "decid_sweep": sweep_df,
        "decid_continuous": {"n": len(cont), "r": float(r), "rho": float(rho)},
        "decid_species": sp_tbl_d,
        **size_stats,
    }
    link_targets = {
        "Joined GT/prediction table (CSV)": "gt_vs_ml_joined.csv",
        "Metrics with crown-bootstrap intervals (CSV)": "gt_vs_ml_metrics.csv",
        "Per-species metrics (CSV)": "gt_vs_ml_by_species.csv",
        "Flowering discrepancy gallery (HTML)":
            "gt_vs_ml_flowering_discrepancies.html",
        "Deciduous discrepancy gallery (HTML)":
            "gt_vs_ml_deciduous_discrepancies.html",
        "Dropped flowering label conflicts (HTML)":
            "gt_vs_ml_flowering_conflicts.html",
    }
    if size_stats:
        link_targets |= {
            "Metrics by crown size (CSV)": "gt_vs_ml_by_size.csv",
            "Metrics by species and crown size (CSV)": "gt_vs_ml_by_species_size.csv",
        }
    render_summary_report(
        out_dir / "gt_vs_ml_report.html",
        stats,
        plot_uris,
        link_targets,
        flower_threshold,
        decid_threshold,
        leafing_threshold,
        no_visualize,
    )


if __name__ == "__main__":
    main()
