#!/usr/bin/env python
"""Compare ML crown classifications to Vicente's ground-truth labels.

Joins per-(crown, date) ground-truth labels from
`vicente_20260618_labels.csv` to the ML output netCDF
(`crown_classifications.nc4`) via the UUID->tag mapping in
`flowering_dataset.gpkg`, and reports flowering / deciduous metrics.
"""
import sys
from pathlib import Path

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
def main(gt_csv, gpkg, nc4, output_dir, flower_threshold, decid_threshold,
         leafing_threshold):
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


if __name__ == "__main__":
    main()
