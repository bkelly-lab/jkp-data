"""Compare baseline (main) vs Bessembinder-corrected HML factor returns.

Both inputs are the ``hml.parquet`` output of ``jkp portfolio`` from two runs:

- ``hml_main.parquet``          -- baseline, no Bessembinder correction
- ``hml_bessembinder.parquet``  -- with the Bessembinder price-error correction

Each file holds the long-short (high-minus-low) portfolio return for *every*
characteristic, per country and month:

    keys    : eom, excntry, characteristic
    returns : ret_ew, ret_vw, ret_vw_cap   (equal / value / capped-value weighted)

The two runs are joined on the keys, and per (country x characteristic) we
compute the correlation and the annualized mean / volatility / Sharpe of each
side. Figures are rendered live by the companion ``.qmd``.

Exposes:
- load_pair()       -> merged frame with ret_*_main / ret_*_bess
- by_cntry_char()   -> per (country x characteristic): n, corr_*, mean_*, std_*
- corr_hist(scope)  -> correlation histograms (all / us / exus)
- moment_scatter()  -> 3x3 annualized mean/vol/Sharpe scatter grid vs y=x
- summary_table()   -> median correlations by sample and weighting
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

RET_COLS = ["ret_ew", "ret_vw", "ret_vw_cap"]
KEYS = ["eom", "excntry", "characteristic"]
WEIGHT_LABELS = {
    "ret_ew": "Equal-weighted",
    "ret_vw": "Value-weighted",
    "ret_vw_cap": "Capped value-weighted",
}

# Data fixtures live in the sibling data/ folder next to this module.
# Point these at your own baseline / corrected runs to regenerate.
DATA_DIR = Path(__file__).resolve().parent / "data"
MAIN_PATH = DATA_DIR / "hml_main.parquet"
BESS_PATH = DATA_DIR / "hml_bessembinder.parquet"
# Precomputed per-country x characteristic stats (correlations + moments).
# This is the only file the report needs; the two hml_*.parquet raw runs above
# are optional and kept untracked. Regenerate with rebuild_stats().
STATS_PATH = DATA_DIR / "bess_factor_stats.parquet"

# Filters (match the validation notebook): drop thin panels and two countries
# whose returns are dominated by hyperinflation-era noise.
MIN_N = 12
EXCLUDE_CNTRY = ["VEN", "ZWE"]

BLUE = "#2a78d6"  # World ex-US
RED = "#e03131"  # US
GREY = "#52514e"

# Row order for the moment grid: capped VW on top, EW at the bottom.
WEIGHT_ORDER = ["ret_vw_cap", "ret_vw", "ret_ew"]

# Monthly -> annualized scaling.
ANN_MEAN = 12.0
ANN_STD = np.sqrt(12.0)


def load_pair(main_path: Path = MAIN_PATH, bess_path: Path = BESS_PATH) -> pl.DataFrame:
    """Inner-join the two runs on (eom, excntry, characteristic)."""
    main = (
        pl.scan_parquet(main_path)
        .select(KEYS + RET_COLS)
        .rename({c: f"{c}_main" for c in RET_COLS})
    )
    bess = (
        pl.scan_parquet(bess_path)
        .select(KEYS + RET_COLS)
        .rename({c: f"{c}_bess" for c in RET_COLS})
    )
    return main.join(bess, on=KEYS, how="inner").collect()


def by_cntry_char(
    merged: pl.DataFrame | None = None,
    min_n: int = MIN_N,
    exclude: list[str] = EXCLUDE_CNTRY,
) -> pl.DataFrame:
    """Per (country x characteristic): overlap n, correlation, and the
    per-side mean / std of each weighting (monthly, unannualized)."""
    if merged is None:
        merged = load_pair()
    agg = [pl.len().alias("n")]
    agg += [pl.corr(f"{c}_main", f"{c}_bess").alias(f"corr_{c}") for c in RET_COLS]
    for c in RET_COLS:
        agg += [
            pl.mean(f"{c}_main").alias(f"mean_{c}_main"),
            pl.mean(f"{c}_bess").alias(f"mean_{c}_bess"),
            pl.std(f"{c}_main").alias(f"std_{c}_main"),
            pl.std(f"{c}_bess").alias(f"std_{c}_bess"),
        ]
    return (
        merged.lazy()
        .group_by("excntry", "characteristic")
        .agg(agg)
        .filter(pl.col("n") > min_n)
        .filter(pl.col("excntry").is_in(exclude).not_())
        .sort("corr_ret_vw_cap")
        .collect()
    )


def load_stats(path: Path = STATS_PATH) -> pl.DataFrame:
    """Read the precomputed per-country x characteristic stats table."""
    return pl.read_parquet(path)


def rebuild_stats(path: Path = STATS_PATH) -> pl.DataFrame:
    """Recompute the stats table from the raw hml runs and overwrite it."""
    stats = by_cntry_char(load_pair())
    stats.write_parquet(path)
    return stats


def _scope(df: pl.DataFrame, scope: str) -> pl.DataFrame:
    if scope == "us":
        return df.filter(pl.col("excntry") == "USA")
    if scope == "exus":
        return df.filter(pl.col("excntry") != "USA")
    return df


def corr_hist(df: pl.DataFrame | None = None, scope: str = "all", bins: int | None = None):
    """Histogram of main-vs-bess correlations, one panel per weighting.

    scope: 'all' | 'us' | 'exus'. US is zoomed to [0.95, 1] with fine bins.
    """
    if df is None:
        df = by_cntry_char()
    d = _scope(df, scope)
    # zoom to where the data lives; bin within the visible window
    if scope == "us":
        xlim, bins = (0.98, 1.0), bins or 40
    else:
        xlim, bins = (0.0, 1.0), bins or 50

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, c in zip(axes, RET_COLS):
        vals = d[f"corr_{c}"].drop_nulls().drop_nans()
        ax.hist(vals, bins=bins, range=xlim, color=BLUE, edgecolor="white", linewidth=0.4)
        med = vals.median()
        ax.axvline(med, color=GREY, linewidth=1, linestyle="--")
        ax.text(
            med - 0.06 * (xlim[1] - xlim[0]),
            0.985,
            f"Median: {med:.3f}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8.5,
            color=GREY,
        )
        ax.set_title(WEIGHT_LABELS[c], fontsize=11, pad=8)
        ax.set_xlabel("Correlation")
        ax.set_xlim(*xlim)
        if scope == "us":
            ticks = np.arange(xlim[0], xlim[1] + 1e-9, 0.005)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.3f}" for t in ticks])
        ax.margins(y=0.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("n")
    # Pin margins so axes positions are identical across tabs regardless of
    # tick-label width (US y-max "140" vs WxUS "6000" would shift tight_layout).
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.17, wspace=0.18)
    return fig


def corr_stats(df: pl.DataFrame | None = None, scope: str = "us") -> pl.DataFrame:
    """For one sample ('us' or 'exus'): pair count plus the five-number summary
    (Min, Q1, Q2/median, Q3, Max) of the main-vs-bess correlation, per weighting."""
    if df is None:
        df = by_cntry_char()
    d = _scope(df, scope)
    rows = []
    for c in RET_COLS:
        v = d[f"corr_{c}"].drop_nulls().drop_nans()
        rows.append(
            {
                "Weighting": WEIGHT_LABELS[c],
                "Pairs": v.len(),
                "Min": v.min(),
                "Q1": v.quantile(0.25),
                "Q2 / Median": v.median(),
                "Q3": v.quantile(0.75),
                "Max": v.max(),
            }
        )
    out = pl.DataFrame(rows)
    stat_cols = ["Min", "Q1", "Q2 / Median", "Q3", "Max"]
    return out.with_columns([pl.col(c).round(3) for c in stat_cols])


def _moment_xy(df: pl.DataFrame, metric: str, c: str, src: str) -> np.ndarray:
    if metric == "mean":
        return df[f"mean_{c}_{src}"].to_numpy() * ANN_MEAN * 100  # percent
    if metric == "std":
        return df[f"std_{c}_{src}"].to_numpy() * ANN_STD * 100  # percent
    # annualized Sharpe = (mean / std) * sqrt(12); dimensionless ratio
    return (df[f"mean_{c}_{src}"].to_numpy() / df[f"std_{c}_{src}"].to_numpy()) * ANN_STD


def moment_scatter(df: pl.DataFrame | None = None):
    """3x3 grid: rows = weighting (capped VW / VW / EW), cols = annualized
    moment (mean / vol / Sharpe). Each point is one country x characteristic,
    baseline (x) vs corrected (y), dashed line is y = x. US points are red,
    World ex-US points are blue."""
    if df is None:
        df = by_cntry_char()
    us = _scope(df, "us")
    exus = _scope(df, "exus")
    metrics = [
        ("mean", "Annualized Mean (%)"),
        ("std", "Annualized Volatility (%)"),
        ("sharpe", "Annualized Sharpe"),
    ]
    limits = {
        "mean": (-75, 75),
        "std": (0, 140),
        "sharpe": (-4, 4),
    }
    ticks = {
        "mean": np.arange(-75, 76, 25),
        "std": np.arange(0, 141, 20),
        "sharpe": np.arange(-4, 5, 2),
    }
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=RED,
            markeredgecolor="none",
            markersize=7,
            label="US",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE,
            markeredgecolor="none",
            markersize=7,
            label="World ex-US",
        ),
    ]
    fig = plt.figure(figsize=(12, 10.6))
    subfigs = fig.subfigures(3, 1, hspace=0.20)
    for i, c in enumerate(WEIGHT_ORDER):
        sf = subfigs[i]
        sf.suptitle(
            WEIGHT_LABELS[c],
            x=0.5,
            y=1.11,
            ha="center",
            fontsize=15,
            fontweight="normal",
            color="#1a1818",
        )
        axes = sf.subplots(1, 3, gridspec_kw={"wspace": 0.32})
        sf.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=2,
            frameon=False,
            fontsize=9,
            handletextpad=0.4,
            columnspacing=1.6,
        )
        for j, (m, label) in enumerate(metrics):
            ax = axes[j]
            xb, yb = _moment_xy(exus, m, c, "main"), _moment_xy(exus, m, c, "bess")
            xr, yr = _moment_xy(us, m, c, "main"), _moment_xy(us, m, c, "bess")
            ax.scatter(xb, yb, s=8, color=BLUE, alpha=0.4, edgecolor="none", label="World ex-US")
            ax.scatter(xr, yr, s=8, color=RED, alpha=0.6, edgecolor="none", label="US")
            lo, hi = limits[m]
            ax.plot([lo, hi], [lo, hi], color=GREY, linewidth=1, linestyle="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xticks(ticks[m])
            ax.set_yticks(ticks[m])
            ax.set_aspect("equal", "box")
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Current")
            ax.set_ylabel("Post-Correction")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
    return fig


def summary_table(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Median main-vs-bess correlation by sample (All / USA / Ex-USA) and weighting."""
    if df is None:
        df = by_cntry_char()
    rows = []
    for scope, label in [("all", "All countries"), ("us", "USA"), ("exus", "Ex-USA")]:
        d = _scope(df, scope)
        row = {"Sample": label, "Pairs": d.height}
        for c in RET_COLS:
            med = d[f"corr_{c}"].drop_nulls().drop_nans().median()
            row[f"Median corr — {WEIGHT_LABELS[c]}"] = round(float(med), 4)
        rows.append(row)
    return pl.DataFrame(rows)


def date_window_note(merged: pl.DataFrame | None = None) -> str:
    if merged is None:
        merged = load_pair()
    start = str(merged["eom"].min())
    end = str(merged["eom"].max())
    n_char = merged["characteristic"].n_unique()
    n_cntry = merged["excntry"].n_unique()
    return (
        f"Overlap window: {start} to {end}; "
        f"{n_char} characteristics across {n_cntry} countries "
        f"({len(merged):,} country x characteristic x month observations)."
    )
