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


def load_gt(path):
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["date"], format="%Y_%m_%d")
    df["uuid"] = df["polygon_id"].str.rsplit("_", n=1).str[0]
    return df


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


def binary_metrics(y_true, y_score, threshold):
    """Return AUROC/AUPRC/confusion at threshold; (None,)*4 if degenerate."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    y_pred = (y_score >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def fmt_metrics(label, m):
    if m is None:
        return f"  {label}: insufficient class diversity"
    cm = m["confusion"]
    return (
        f"  {label}: n={m['n']} pos={m['n_pos']} neg={m['n_neg']} "
        f"AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}\n"
        f"    confusion (rows=true, cols=pred; labels=[neg,pos]):\n"
        f"      [[{cm[0,0]:>5d} {cm[0,1]:>5d}]\n"
        f"       [{cm[1,0]:>5d} {cm[1,1]:>5d}]]"
    )


def per_species_table(df, score_col, target, min_pos=10):
    rows = []
    for sp, sub in df.groupby("latin"):
        y = target.loc[sub.index]
        s = sub[score_col]
        npos = int(y.sum())
        nneg = int(len(y) - npos)
        if npos < min_pos or nneg < min_pos:
            continue
        rows.append({
            "latin": sp,
            "n": len(y),
            "n_pos": npos,
            "auroc": roc_auc_score(y, s),
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
    plt.close(fig)


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
    plt.close(fig)


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


def _esc(value):
    return html.escape("" if value is None else str(value))


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

    df = df.dropna(subset=["pred"]).copy()
    n_total = len(df)
    df["discrepancy"] = (df["pred"] - df["gt_value"]).abs()

    df["date_str"] = df["dt"].dt.strftime("%Y_%m_%d")
    df = df[df["date_str"].isin(drone_index)].copy()
    n_with_image = len(df)
    df = df[df["uuid"].isin(geoms)].copy()
    n_eligible = len(df)

    df = df.sort_values("discrepancy", ascending=False).head(n)

    opt_cols = [c for c in ("observation_id", "globalId") if c in df.columns]

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
              GT ({_esc(gt_label)}): <b>{_esc(row['gt_display'])}</b><br/>
              predicted prob: <b>{row['pred']:.3f}</b><br/>
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


@click.command()
@click.option("--gt-csv", default=GT_CSV_DEFAULT, show_default=True)
@click.option("--gpkg", default=GPKG_DEFAULT, show_default=True)
@click.option("--nc4", default=NC4_DEFAULT, show_default=True)
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
def main(gt_csv, gpkg, nc4, output_dir, flower_threshold, decid_threshold,
         leafing_threshold, drone_dir, n_examples, chip_size, no_visualize):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GT  : {gt_csv}")
    gt = load_gt(gt_csv)
    print(f"  rows: {len(gt):,}  unique uuids: {gt['uuid'].nunique():,}")

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
    for name, sub in strata.items():
        m = binary_metrics(
            sub["y"].to_numpy(),
            sub["flowering_probability"].to_numpy(),
            flower_threshold,
        )
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
    )
    print(sp_tbl.to_string(index=False) if len(sp_tbl) else "  (none qualify)")

    plot_roc_pr(
        flower_curves_roc,
        "Flowering ROC",
        out_dir / "gt_vs_ml_flowering_roc.pdf",
        "roc",
    )
    plot_roc_pr(
        flower_curves_pr,
        "Flowering PR",
        out_dir / "gt_vs_ml_flowering_pr.pdf",
        "pr",
    )

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
    for name, sub in strata_d.items():
        m = binary_metrics(
            sub["y_leaf"].to_numpy(),
            sub["deciduous_probability"].to_numpy(),
            decid_threshold,
        )
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
                               "auroc": float("nan")})
            continue
        sweep_rows.append({
            "leafing<": thr,
            "n_pos": int(y.sum()),
            "auroc": roc_auc_score(y, good["deciduous_probability"]),
        })
    print(pd.DataFrame(sweep_rows).to_string(index=False))

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
    )
    print(sp_tbl_d.to_string(index=False) if len(sp_tbl_d) else "  (none qualify)")

    plot_roc_pr(
        decid_curves_roc,
        f"Deciduous ROC (leafing<{leafing_threshold})",
        out_dir / "gt_vs_ml_deciduous_roc.pdf",
        "roc",
    )
    plot_roc_pr(
        decid_curves_pr,
        f"Deciduous PR (leafing<{leafing_threshold})",
        out_dir / "gt_vs_ml_deciduous_pr.pdf",
        "pr",
    )
    plot_decid_continuous(cont, out_dir / "gt_vs_ml_deciduous_continuous.pdf")

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


if __name__ == "__main__":
    main()
