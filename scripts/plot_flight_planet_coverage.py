#!/usr/bin/env python
"""Plot whether whole-island drone flights are covered by clear Planet imagery.

For each BCI whole-island drone flight, decide whether a Planet image with a
clear enough view of the canopy (fraction_clear >= --threshold) was acquired
within +/- --window-days of the flight's multi-day acquisition span. Both the
clear threshold and the temporal window are parametrizable.

Inputs:
  --clear-csv    Per-Planet-scene clear fractions from bci_clear_fraction.py
                 (columns: image_name, datetime_utc, fraction_clear, ...).
  --flights-csv  Drone flight metadata (BCImetadataFlights_*.csv). Whole-island
                 flights are scale in {full, Full}; actual acquisition dates are
                 the comma-separated dateFlights_ymd (mixed ISO / US formats).

Only flights whose acquisition span overlaps the Planet record are kept.

Output: a multi-page PDF -- page 1 is a full-timeline overview (all Planet
images as points at their clear %, each flight window shaded and colored by
whether it is covered); following pages are per-flight zoomed panels where the
+/- window (only a few days wide) and nearby Planet points are legible.

Typical usage:
    python scripts/plot_flight_planet_coverage.py
    python scripts/plot_flight_planet_coverage.py --threshold 0.5 --window-days 3
"""
from __future__ import annotations

import math
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_CLEAR_CSV = Path(
    "/Volumes/Earth03/flower/whole_island/bci_clear_fraction.csv"
)
DEFAULT_FLIGHTS_CSV = Path(
    "/Volumes/Earth03/flower/whole_island/BCImetadataFlights_2026-2-17.csv"
)
DEFAULT_OUTPUT = Path(
    "/Volumes/Earth03/flower/whole_island/flight_planet_coverage.pdf"
)

COVERED_COLOR = "#2ca02c"       # green: flight covered by a clear Planet image
UNCOVERED_COLOR = "#d62728"     # red: no clear Planet image in window
PT_BELOW_COLOR = "#bdbdbd"      # Planet image below threshold
PT_ABOVE_COLOR = "#1f77b4"      # Planet image at/above threshold
THRESH_COLOR = "#333333"


# ---------------------------------------------------------------------------
# Flight-date parsing


def _parse_one_date(token: str) -> pd.Timestamp | None:
    """Parse a single flight date token; try ISO (Y-M-D) then US (m/d/Y)."""
    token = token.strip()
    if not token or "?" in token:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return pd.to_datetime(token, format=fmt)
        except (ValueError, TypeError):
            continue
    return None


def parse_flight_dates(cell: str) -> list[pd.Timestamp]:
    """Parse the comma-separated dateFlights_ymd cell into sorted unique dates."""
    if not isinstance(cell, str):
        return []
    dates = {d for tok in cell.split(",") if (d := _parse_one_date(tok)) is not None}
    return sorted(dates)


def build_flights(flights_csv: Path, planet_min: pd.Timestamp,
                  planet_max: pd.Timestamp, window: pd.Timedelta) -> pd.DataFrame:
    """Return whole-island flights overlapping the Planet record, with windows."""
    raw = pd.read_csv(flights_csv, dtype=str)
    is_full = raw["scale"].fillna("").str.strip().str.lower() == "full"
    recs = []
    for _, row in raw[is_full].iterrows():
        dates = parse_flight_dates(row.get("dateFlights_ymd", ""))
        if not dates:
            continue
        first, last = dates[0].normalize(), dates[-1].normalize()
        # Keep only flights whose span overlaps the Planet record.
        if last < planet_min.normalize() or first > planet_max.normalize():
            continue
        # Calendar-day window: "within N days" includes the whole trailing/leading
        # calendar day. win_start/win_end are day boundaries; membership is tested
        # against Planet acquisition *dates* (see annotate_coverage).
        recs.append({
            "first": first,
            "last": last,
            "win_start": first - window,
            "win_end": last + window,
            "mission": (row.get("missionName") or "").strip(),
        })
    flights = pd.DataFrame(recs).sort_values("first").reset_index(drop=True)
    return flights


# ---------------------------------------------------------------------------
# Coverage computation


def annotate_coverage(flights: pd.DataFrame, planet: pd.DataFrame,
                      thresh_pct: float) -> pd.DataFrame:
    """Add `covered` and `max_pct_in_win` columns per flight.

    Membership is by calendar day: a Planet image counts if its acquisition
    date falls in [win_start, win_end] inclusive (both are day boundaries), so
    the whole trailing/leading calendar day of the +/- window is included.
    """
    day = planet["datetime_utc"].dt.normalize().values
    pct = planet["pct_clear"].values
    covered, max_in_win = [], []
    for _, f in flights.iterrows():
        in_win = (day >= np.datetime64(f["win_start"])) & \
                 (day <= np.datetime64(f["win_end"]))
        window_pct = pct[in_win]
        mx = float(window_pct.max()) if window_pct.size else float("nan")
        max_in_win.append(mx)
        covered.append(bool(window_pct.size) and mx >= thresh_pct)
    out = flights.copy()
    out["covered"] = covered
    out["max_pct_in_win"] = max_in_win
    return out


# ---------------------------------------------------------------------------
# Plotting


def plot_overview(ax, planet, flights, thresh_pct, window_days):
    below = planet["pct_clear"] < thresh_pct
    ax.scatter(planet.loc[below, "datetime_utc"], planet.loc[below, "pct_clear"],
               s=6, c=PT_BELOW_COLOR, alpha=0.4, linewidths=0, zorder=1)
    ax.scatter(planet.loc[~below, "datetime_utc"], planet.loc[~below, "pct_clear"],
               s=8, c=PT_ABOVE_COLOR, alpha=0.6, linewidths=0, zorder=2)
    ax.axhline(thresh_pct, color=THRESH_COLOR, lw=1.2, ls="--", zorder=3)

    day = pd.Timedelta(days=1)
    for _, f in flights.iterrows():
        color = COVERED_COLOR if f["covered"] else UNCOVERED_COLOR
        # Shade through the end of the last in-window calendar day.
        span_end = f["win_end"] + day
        ax.axvspan(f["win_start"], span_end, color=color, alpha=0.35, zorder=0)
        mid = f["win_start"] + (span_end - f["win_start"]) / 2
        ax.axvline(mid, color=color, lw=0.9, alpha=0.9, zorder=4)

    n_cov = int(flights["covered"].sum())
    n = len(flights)
    ax.set_ylim(0, 100)
    ax.set_xlim(planet["datetime_utc"].min(), planet["datetime_utc"].max())
    ax.set_ylabel("Planet clear view of BCI canopy (%)")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Whole-island drone flights covered by clear Planet imagery: "
        f"{n_cov} of {n} flights have a Planet image ≥{thresh_pct:.0f}% "
        f"clear within ±{window_days:g} day(s) of the flight"
    )
    ax.grid(True, ls="--", alpha=0.4)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PT_ABOVE_COLOR,
               markersize=6, label=f"Planet image ≥{thresh_pct:.0f}% clear"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PT_BELOW_COLOR,
               markersize=6, label=f"Planet image <{thresh_pct:.0f}% clear"),
        Line2D([0], [0], color=THRESH_COLOR, ls="--",
               label=f"{thresh_pct:.0f}% threshold"),
        Patch(facecolor=COVERED_COLOR, alpha=0.35, label="Flight window (covered)"),
        Patch(facecolor=UNCOVERED_COLOR, alpha=0.35,
              label="Flight window (not covered)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)


def plot_panels(pdf, planet, flights, thresh_pct, window, window_days,
                ncols=4, per_page=12):
    pad = max(pd.Timedelta(days=15), 3 * window)
    dt = planet["datetime_utc"]
    n = len(flights)
    for start in range(0, n, per_page):
        chunk = flights.iloc[start:start + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for i, (_, f) in enumerate(chunk.iterrows()):
            ax = axes.flat[i]
            ax.set_visible(True)
            lo, hi = f["first"] - pad, f["last"] + pad
            m = (dt >= lo) & (dt <= hi)
            sub = planet[m]
            below = sub["pct_clear"] < thresh_pct
            ax.scatter(sub.loc[below, "datetime_utc"], sub.loc[below, "pct_clear"],
                       s=14, c=PT_BELOW_COLOR, alpha=0.7, linewidths=0)
            ax.scatter(sub.loc[~below, "datetime_utc"], sub.loc[~below, "pct_clear"],
                       s=18, c=PT_ABOVE_COLOR, alpha=0.9, linewidths=0)
            color = COVERED_COLOR if f["covered"] else UNCOVERED_COLOR
            ax.axvspan(f["win_start"], f["win_end"] + pd.Timedelta(days=1),
                       color=color, alpha=0.3)
            ax.axhline(thresh_pct, color=THRESH_COLOR, lw=1.0, ls="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 100)
            mx = f["max_pct_in_win"]
            mx_txt = "no Planet img" if math.isnan(mx) else f"max {mx:.0f}%"
            status = "COVERED" if f["covered"] else "not covered"
            ax.set_title(
                f"{f['first'].date()}–{f['last'].date()}\n"
                f"{status} ({mx_txt} in window)",
                fontsize=9, color=color,
            )
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, ls="--", alpha=0.4)
        fig.suptitle(
            f"Per-flight coverage (±{window_days:g} d window, "
            f"≥{thresh_pct:.0f}% clear)", fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI


@click.command(
    help="Plot whole-island flight coverage by clear Planet imagery.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--clear-csv", type=click.Path(path_type=Path, dir_okay=False),
              default=DEFAULT_CLEAR_CSV, show_default=True)
@click.option("--flights-csv", type=click.Path(path_type=Path, dir_okay=False),
              default=DEFAULT_FLIGHTS_CSV, show_default=True)
@click.option("--output", type=click.Path(path_type=Path, dir_okay=False),
              default=DEFAULT_OUTPUT, show_default=True)
@click.option("--threshold", type=float, default=0.25, show_default=True,
              help="Min fraction_clear (0-1) counted as a clear Planet image.")
@click.option("--window-days", type=float, default=1.0, show_default=True,
              help="Days added to each side of the flight's acquisition span.")
def main(clear_csv: Path, flights_csv: Path, output: Path,
         threshold: float, window_days: float) -> None:
    planet = pd.read_csv(clear_csv, parse_dates=["datetime_utc"])
    # Drop tz so Planet timestamps compare with tz-naive flight dates (day-level).
    if planet["datetime_utc"].dt.tz is not None:
        planet["datetime_utc"] = planet["datetime_utc"].dt.tz_localize(None)
    planet["pct_clear"] = planet["fraction_clear"] * 100.0
    planet = planet.sort_values("datetime_utc").reset_index(drop=True)
    p_min, p_max = planet["datetime_utc"].min(), planet["datetime_utc"].max()
    thresh_pct = threshold * 100.0
    window = pd.Timedelta(days=window_days)

    flights = build_flights(flights_csv, p_min, p_max, window)
    if flights.empty:
        raise click.ClickException("no whole-island flights overlap the Planet record")
    flights = annotate_coverage(flights, planet, thresh_pct)

    # Console summary + per-flight table (verification aid).
    click.echo(
        f"{len(flights)} in-range whole-island flights; "
        f"{int(flights['covered'].sum())} covered at ≥{thresh_pct:.0f}% "
        f"within ±{window_days:g} d"
    )
    for _, f in flights.iterrows():
        mx = f["max_pct_in_win"]
        mx_txt = "  n/a" if math.isnan(mx) else f"{mx:5.1f}%"
        click.echo(
            f"  {f['first'].date()}–{f['last'].date()} "
            f"[{f['win_start'].date()}..{f['win_end'].date()}]  "
            f"max={mx_txt}  {'COVERED' if f['covered'] else 'not covered'}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(15, 6))
        plot_overview(ax, planet, flights, thresh_pct, window_days)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        plot_panels(pdf, planet, flights, thresh_pct, window, window_days)

    click.echo(f"wrote {output}")


if __name__ == "__main__":
    main()
