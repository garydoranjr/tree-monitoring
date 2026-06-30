#!/usr/bin/env python
"""Compare ML crown classifications to Vicente's ground-truth labels.

Joins per-(crown, date) ground-truth labels from
`vicente_20260618_labels.csv` to the ML output netCDF
(`crown_classifications.nc4`) via the UUID->tag mapping in
`flowering_dataset.gpkg`, and reports flowering / deciduous metrics.
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


def per_species_table(df, score_col, target, threshold, min_pos=10):
    rows = []
    for sp, sub in df.groupby("latin"):
        y = target.loc[sub.index]
        s = sub[score_col]
        npos = int(y.sum())
        nneg = int(len(y) - npos)
        if npos < min_pos or nneg < min_pos:
            continue
        y_pred = (s >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        prf = prf_from_confusion(cm)
        rows.append({
            "latin": sp,
            "n": len(y),
            "n_pos": npos,
            "auroc": roc_auc_score(y, s),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "accuracy": prf["accuracy"],
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


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


def index_drone_images(drone_dir):
    """Map YYYY_MM_DD -> drone GeoTIFF path."""
    index = {}
    for path in sorted(Path(drone_dir).glob("*_local.tif")):
        try:
            index[get_date(str(path))] = path
        except ValueError:
            continue
    return index


def _chip_data_uri(tif_path, polygon, chip_size):
    """Crop the crown, outline it, resize, return a base64 PNG data URI."""
    with rasterio.open(tif_path) as src:
        window, img = extract_centered_window(src, polygon)
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
        f"uuid&rarr;tag mappings: {f['bridge']:,} &middot; "
        f"prediction rows: {f['pred_rows']:,} (tags {f['pred_tags']:,}) "
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
<div class="plots">
  <img src="{plot_uris['decid_roc']}" alt="Deciduous ROC"/>
  <img src="{plot_uris['decid_pr']}" alt="Deciduous PR"/>
  <img src="{plot_uris['decid_continuous']}" alt="Deciduous continuous"/>
</div>

<h2>Detailed reports</h2>
{novis_note}
<ul class="links">{links_html}</ul>
</body></html>"""
    Path(out_path).write_text(doc)
    print(f"Wrote summary report: {out_path}")


def render_discrepancy_report(merged, geoms, drone_index, out_path,
                              target, n, chip_size):
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

    # Companion CSV: polygon IDs (and identifiers) for *every* row ranked by
    # discrepancy — the full set with a computable discrepancy, including
    # crowns lacking a drone image or geometry that the gallery below omits.
    opt_cols = [c for c in ("observation_id", "globalId") if c in df.columns]
    csv_cols = [
        c for c in (
            ["polygon_id", "uuid", "tag"] + opt_cols
            + ["date_str", "discrepancy", "pred", "gt_value", "gt_display",
               "segmentation", "latin", "species_pred"]
        ) if c in df.columns
    ]
    csv_path = Path(out_path).with_suffix(".csv")
    (df.sort_values("discrepancy", ascending=False)[csv_cols]
       .rename(columns={"date_str": "date"})
       .to_csv(csv_path, index=False))
    print(f"Wrote {csv_path}  ({len(df):,} rows)")

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
            uri = _chip_data_uri(tif_path, polygon, chip_size)
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
                           chip_size):
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
                uri = _chip_data_uri(drone_index[date_str], polygon, chip_size)
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
@click.option("--n-examples", default=40, show_default=True, type=int,
              help="Number of worst discrepancy examples per target.")
@click.option("--chip-size", default=300, show_default=True, type=int,
              help="Pixel size of each crown chip in the HTML report.")
@click.option("--no-visualize", is_flag=True,
              help="Skip the discrepancy HTML reports (metrics only).")
def main(gt_csv, gpkg, nc4, geo_folds, exclude_train, output_dir,
         flower_threshold, decid_threshold, leafing_threshold, drone_dir,
         n_examples, chip_size, no_visualize):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_uris = {}

    print(f"Loading GT  : {gt_csv}")
    gt = load_gt(gt_csv)
    print(f"  rows: {len(gt):,}  unique uuids: {gt['uuid'].nunique():,}")

    if exclude_train:
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

    print(f"Loading bridge: {gpkg}")
    bridge = load_uuid_tag_map(gpkg)
    print(f"  uuid->tag mappings: {len(bridge):,}")

    print(f"Loading nc4 : {nc4}")
    preds = load_predictions(nc4)
    print(f"  prediction rows: {len(preds):,}  tags: {preds['tag'].nunique():,}")

    merged_uuid = gt.merge(bridge, on="uuid", how="inner")
    print(f"After UUID->tag merge: {len(merged_uuid):,} rows")

    merged = merged_uuid.merge(preds, on=["tag", "dt"], how="inner")
    print(f"After (tag, date) join: {len(merged):,} rows")
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
    sp_tbl = per_species_table(
        strata["segmentation==good"],
        "flowering_probability",
        strata["segmentation==good"]["y"],
        flower_threshold,
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
    sp_tbl_d = per_species_table(
        strata_d["segmentation==good"],
        "deciduous_probability",
        strata_d["segmentation==good"]["y_leaf"],
        decid_threshold,
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

    print(f"\nPlots written to: {out_dir}/")

    # ----- Discrepancy visualization -----
    if not no_visualize:
        print("\n=== Discrepancy reports ===")
        print(f"Loading crown geometries: {gpkg}")
        geoms = load_crown_geometries(gpkg)
        print(f"  crowns with geometry: {len(geoms):,}")
        print(f"Indexing drone images: {drone_dir}")
        drone_index = index_drone_images(drone_dir)
        print(f"  drone dates: {len(drone_index):,}")

        for target in ("flowering", "deciduous"):
            render_discrepancy_report(
                merged,
                geoms,
                drone_index,
                out_dir / f"gt_vs_ml_{target}_discrepancies.html",
                target,
                n_examples,
                chip_size,
            )

        render_conflict_report(
            flower_conflicts,
            geoms,
            drone_index,
            out_dir / "gt_vs_ml_flowering_conflicts.html",
            chip_size,
        )

    # ----- Single summary report -----
    print("\n=== Summary report ===")
    stats = {
        "funnel": {
            "gt_rows": len(gt),
            "gt_rows_raw": gt_rows_raw,
            "dropped_conflicts": int(n_conf),
            "gt_uuids": int(gt["uuid"].nunique()),
            "bridge": len(bridge),
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
    }
    link_targets = {
        "Joined GT/prediction table (CSV)": "gt_vs_ml_joined.csv",
        "Flowering discrepancy gallery (HTML)":
            "gt_vs_ml_flowering_discrepancies.html",
        "Deciduous discrepancy gallery (HTML)":
            "gt_vs_ml_deciduous_discrepancies.html",
        "Dropped flowering label conflicts (HTML)":
            "gt_vs_ml_flowering_conflicts.html",
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
