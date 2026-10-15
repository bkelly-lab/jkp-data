"""Statistical diagnostics for the economic themes in economic_themes.py.

Description:
    Checks how well the sub-themes hold together in the data, using the average monthly
    cross-sectional rank correlation between characteristics (U.S. stocks, START to END).
Steps:
    1) Percentile-rank each characteristic within month, correlate the ranks across the
       stocks that have both characteristics, and average over months (each month weighted
       equally). Cached to OUT_DIR/corr.parquet; recomputed when RERUN_CORR is set or the
       cache lacks a characteristic.
    2) Sub-theme composites: equal-weighted average of member ranks, each multiplied by its
       sub_theme_sign. Composite correlations are implied by the average correlation matrix.
       This is an approximation: it ignores differences in missing data and ties across
       months, and the averaged matrix need not be a valid correlation matrix. A warning is
       printed if a fit exceeds 1.
    3) Per characteristic: signed correlation with the composite of the other members of its
       own sub-theme (fit; negative means the data disagree with its sign), and the
       best-fitting other sub-theme. Clear misfits are listed.
    4) Per sub-theme: mean within-sub-theme |corr|; nearest other sub-theme by |composite corr|.
    5) Per theme: pairs of sub-themes that correlate negatively once theme signs are applied.
Output:
    OUT_DIR/theme_members.parquet, OUT_DIR/theme_summary.parquet,
    OUT_DIR/theme_composite_corr.parquet, plus a console summary.

Run with: uv run python scripts/themes/theme_diagnostics.py
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from economic_themes import NEGATIVE_IN_THEME, SUB_THEMES, THEMES, themes_frame  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "data" / "processed" / "characteristics" / "USA.parquet"
OUT_DIR = REPO / "data" / "interim" / "themes"
RERUN_CORR = False  # recompute the correlations (after a new data run)
START = date(1972, 1, 1)
END = date(2025, 12, 31)
MIN_OBS = 30  # min stocks with both characteristics for a month's pair correlation to count
WEAK_FIT = 0.3  # flag characteristics whose fit with their own sub-theme is below this
MISFIT_GAP = 0.15  # clear misfit: fits another sub-theme at least this much better ...
MISFIT_MIN = 0.4  # ... and at least this well
SIGN_CONFLICT = 0.1  # flag same-theme sub-theme pairs correlating below minus this
CLOSE_PAIR = 0.7  # flag sub-theme pairs whose composites correlate at least this much


def monthly_rank_corr(x: np.ndarray) -> np.ndarray:
    """
    Description:
        Pairwise-complete Pearson correlation of the columns of x (NaN = missing), computed
        with masked matrix products instead of looping over pairs.
    Steps:
        1) Zero-fill x and build the availability mask m.
        2) For each pair (i, j), use sums over rows where both are present.
    Output:
        p x p correlation matrix (NaN where fewer than MIN_OBS rows or zero variance).
    """
    m = (~np.isnan(x)).astype(np.float64)
    x0 = np.nan_to_num(x, nan=0.0)
    n = m.T @ m
    sx = x0.T @ m  # sx[i, j] = sum of x_i over rows where i and j are present
    sxx = (x0 * x0).T @ m
    sxy = x0.T @ x0
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sx.T / n
        var_i = sxx - sx**2 / n
        corr = cov / np.sqrt(var_i * var_i.T)
    corr[(n < MIN_OBS) | ~np.isfinite(corr)] = np.nan
    return corr


def compute_corr(characteristics_path: Path, features: list[str]) -> pl.DataFrame:
    """
    Description:
        Average monthly cross-sectional rank correlation of the characteristics.
    Steps:
        1) Scan the characteristics file, keep START <= eom <= END, percentile-rank each
           characteristic within eom (nulls stay null).
        2) Correlate ranks within each month; average over months, ignoring missing pairs.
    Output:
        DataFrame with a feature column and one column per characteristic.
    """
    ranks = (
        pl.scan_parquet(characteristics_path)
        .filter(pl.col("eom").is_between(START, END))
        .select(["eom", *features])
        .with_columns(
            (pl.col(f).rank("average").over("eom") / pl.col(f).count().over("eom")).cast(pl.Float64)
            for f in features
        )
        .sort("eom")
        .collect()
    )
    p = len(features)
    corr_sum = np.zeros((p, p))
    n_months = np.zeros((p, p))
    months = ranks.partition_by("eom", maintain_order=True)
    for i, month in enumerate(months):
        corr = monthly_rank_corr(month.select(features).to_numpy().astype(np.float64))
        valid = ~np.isnan(corr)
        corr_sum[valid] += corr[valid]
        n_months += valid
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(months)} months")
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = corr_sum / n_months
    return pl.DataFrame(avg, schema=features).insert_column(0, pl.Series("feature", features))


def corr_with_composite(corr: np.ndarray, x: int, idx: np.ndarray, w: np.ndarray) -> float:
    """Correlation of characteristic x with the composite of idx weighted by signs w."""
    var = w @ corr[np.ix_(idx, idx)] @ w
    return float((w @ corr[idx, x]) / np.sqrt(var))


def load_corr(features: list[str]) -> np.ndarray:
    """Correlation matrix of features (diagonal 1, symmetric), from the cache when possible."""
    cache = OUT_DIR / "corr.parquet"
    cached = pl.read_parquet(cache) if cache.exists() and not RERUN_CORR else None
    if cached is None or not set(features) <= set(cached["feature"]):
        print(f"Computing rank correlations from {INPUT} ({START} to {END})")
        cached = compute_corr(INPUT, features)
        cached.write_parquet(cache)
    order = {f: i for i, f in enumerate(cached["feature"])}
    idx = [order[f] for f in features]
    corr = np.nan_to_num(cached.drop("feature").to_numpy()[np.ix_(idx, idx)])
    np.fill_diagonal(corr, 1.0)
    return (corr + corr.T) / 2


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(SUB_THEMES)
    features = [f for t in names for f in SUB_THEMES[t]]
    corr = load_corr(features)
    pos = {f: i for i, f in enumerate(features)}
    groups = {t: np.array([pos[f] for f in SUB_THEMES[t]]) for t in names}
    sign = dict(themes_frame().select("characteristic", "sub_theme_sign").iter_rows())
    signs = {t: np.array([sign[f] for f in SUB_THEMES[t]], dtype=float) for t in names}

    # Characteristics: fit with own sub-theme (leave-one-out) and best other sub-theme -----
    rows = []
    for t in names:
        for f in SUB_THEMES[t]:
            x = pos[f]
            keep = groups[t] != x
            own = (
                sign[f] * corr_with_composite(corr, x, groups[t][keep], signs[t][keep])
                if keep.any()
                else None
            )
            other = {
                u: abs(corr_with_composite(corr, x, groups[u], signs[u])) for u in names if u != t
            }
            best = max(other, key=lambda u: other[u])
            rows.append(
                {
                    "feature": f,
                    "sub_theme": t,
                    "own": own,
                    "best_other": best,
                    "best_other_r": other[best],
                }
            )
    members = pl.DataFrame(rows)
    if (members["own"].abs() > 1).any() or (members["best_other_r"] > 1).any():
        print("Warning: some fits exceed 1; the average correlation matrix is too far from valid.")

    # Sub-themes: coherence and nearest neighbour -------------------------------------------
    w = np.zeros((len(features), len(names)))
    for g, t in enumerate(names):
        w[groups[t], g] = signs[t]
    s = w.T @ corr @ w
    comp = s / np.sqrt(np.outer(np.diag(s), np.diag(s)))
    comp_abs = np.abs(comp)
    np.fill_diagonal(comp_abs, 0.0)
    summary = []
    for g, t in enumerate(names):
        idx = groups[t]
        n = len(idx)
        sub = np.abs(corr[np.ix_(idx, idx)])
        within = float((sub.sum() - n) / (n * (n - 1))) if n > 1 else None
        nb = int(np.argmax(comp_abs[g]))
        summary.append(
            {
                "sub_theme": t,
                "n": n,
                "mean_abs_r": within,
                "nearest_sub_theme": names[nb],
                "nearest_r": float(comp[g, nb]),
            }
        )
    summary = pl.DataFrame(summary)
    composite = pl.DataFrame(comp, schema=names).insert_column(0, pl.Series("sub_theme", names))

    members.write_parquet(OUT_DIR / "theme_members.parquet")
    summary.write_parquet(OUT_DIR / "theme_summary.parquet")
    composite.write_parquet(OUT_DIR / "theme_composite_corr.parquet")

    with pl.Config(tbl_rows=-1, tbl_width_chars=160, float_precision=2):
        print(f"Median mean |r| within sub-themes: {summary['mean_abs_r'].median():.2f}\n")
        print(summary.sort("mean_abs_r", nulls_last=True))
        print(
            f"\nClear misfits (fit another sub-theme at least {MISFIT_GAP} better, "
            f"at {MISFIT_MIN} or more); move them or record why they stay:"
        )
        misfit = (pl.col("best_other_r") >= pl.col("own") + MISFIT_GAP) & (
            pl.col("best_other_r") >= MISFIT_MIN
        )
        print(members.filter(misfit).sort("sub_theme"))
        print(f"\nWeakly attached characteristics (fit < {WEAK_FIT}):")
        print(members.filter(pl.col("own") < WEAK_FIT).sort("own"))
        print(f"\nSub-theme pairs whose composites correlate {CLOSE_PAIR} or more:")
        pairs = [
            (names[i], names[j], float(comp[i, j]))
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if comp_abs[i, j] >= CLOSE_PAIR
        ]
        print(pl.DataFrame(pairs, schema=["sub_theme", "other", "r"], orient="row").sort("r"))
        print("\nCharacteristics whose sign the data disagree with (negative fit):")
        print(members.filter(pl.col("own") < 0).sort("own"))
        print(
            f"\nSub-themes in the same theme correlating below -{SIGN_CONFLICT} after theme signs:"
        )
        theme_sign = {t: -1 if t in NEGATIVE_IN_THEME else 1 for t in names}
        conflicts = [
            (theme, a, b, comp[names.index(a), names.index(b)] * theme_sign[a] * theme_sign[b])
            for theme, subs in THEMES.items()
            for i, a in enumerate(subs)
            for b in subs[i + 1 :]
        ]
        conflicts = [c for c in conflicts if c[3] < -SIGN_CONFLICT]
        print(pl.DataFrame(conflicts, schema=["theme", "sub_theme", "other", "r"], orient="row"))
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
