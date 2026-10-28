"""Statistical diagnostics for the economic themes in themes.csv.

Description:
    Checks how well the subthemes hold together in the data, using the average monthly
    cross-sectional rank correlation between characteristics (U.S. stocks, START to END).
Steps:
    1) Percentile-rank each characteristic within month, correlate the ranks across the
       stocks that have both characteristics, and average over months (each month weighted
       equally). Cached to OUT_DIR/corr.parquet; recomputed when the cache lacks a
       characteristic or is older than INPUT.
    2) Subtheme composites: equal-weighted average of member ranks, each multiplied by its
       subtheme_sign. Composite correlations are implied by the average correlation matrix.
       This is an approximation: it ignores differences in missing data and ties across
       months, and the averaged matrix need not be a valid correlation matrix. A warning is
       printed if a fit exceeds 1.
    3) Per characteristic: signed correlation with the composite of the other members of its
       own subtheme (fit; negative means the data disagree with its sign), and the
       best-fitting other subtheme. Clear misfits are listed.
    4) Per subtheme: mean within-subtheme |corr|; nearest other subtheme by |composite corr|.
    5) Per theme: pairs of subthemes that correlate negatively once theme signs are applied.
Output:
    OUT_DIR/theme_members.parquet, OUT_DIR/theme_summary.parquet,
    OUT_DIR/theme_composite_corr.parquet, plus a console summary.

Run with: uv run python scripts/themes/theme_diagnostics.py
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
THEMES_CSV = REPO / "src" / "jkp" / "data" / "resources" / "themes" / "themes.csv"
INPUT = REPO / "data" / "processed" / "characteristics" / "USA.parquet"
OUT_DIR = REPO / "data" / "interim" / "themes"
START = date(1972, 1, 1)
END = date(2025, 12, 31)
MIN_OBS = 30  # min stocks with both characteristics for a month's pair correlation to count
WEAK_FIT = 0.3  # flag characteristics whose fit with their own subtheme is below this
MISFIT_GAP = 0.15  # clear misfit: fits another subtheme at least this much better ...
MISFIT_MIN = 0.4  # ... and at least this well
SIGN_CONFLICT = 0.1  # flag same-theme subtheme pairs correlating below minus this
CLOSE_PAIR = 0.7  # flag subtheme pairs whose composites correlate at least this much


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
    fresh = cache.exists() and (
        not INPUT.exists() or cache.stat().st_mtime >= INPUT.stat().st_mtime
    )
    cached = pl.read_parquet(cache) if fresh else None
    if cached is None or not set(features) <= set(cached["feature"]):
        print(f"Computing rank correlations from {INPUT} ({START} to {END})")
        cached = compute_corr(INPUT, features)
        cached.write_parquet(cache)
    order = {f: i for i, f in enumerate(cached["feature"])}
    idx = [order[f] for f in features]
    corr = np.nan_to_num(cached.drop("feature").to_numpy()[np.ix_(idx, idx)])
    np.fill_diagonal(corr, 1.0)
    return (corr + corr.T) / 2


def subtheme_groups(themes: pl.DataFrame) -> pl.DataFrame:
    """One row per subtheme, in file order: theme, theme_sign, member rows (pos) and signs."""
    return (
        themes.with_row_index("pos")
        .group_by("subtheme", maintain_order=True)
        .agg(pl.col("theme", "theme_sign").first(), "pos", "subtheme_sign")
    )


def member_fits(corr: np.ndarray, themes: pl.DataFrame) -> pl.DataFrame:
    """
    Description:
        How well each characteristic fits its own subtheme and every other subtheme.
    Steps:
        1) Own fit: correlation with the signed composite of the other members of its own
           subtheme (leave-one-out), times its own subtheme_sign. None for a one-member
           subtheme.
        2) Best other: the other subtheme whose composite has the largest |correlation|.
    Output:
        One row per characteristic, in file order: feature, subtheme, own, best_other,
        best_other_r. corr rows and columns follow the rows of themes.
    """
    groups = subtheme_groups(themes)
    idx = {t: np.array(pos) for t, pos in groups.select("subtheme", "pos").iter_rows()}
    w = {
        t: np.array(s, dtype=float)
        for t, s in groups.select("subtheme", "subtheme_sign").iter_rows()
    }
    rows = []
    for x, (f, t, sign) in enumerate(
        themes.select("characteristic", "subtheme", "subtheme_sign").iter_rows()
    ):
        keep = idx[t] != x
        own = sign * corr_with_composite(corr, x, idx[t][keep], w[t][keep]) if keep.any() else None
        other = {u: abs(corr_with_composite(corr, x, idx[u], w[u])) for u in idx if u != t}
        best = max(other, key=lambda u: other[u])
        rows.append(
            {
                "feature": f,
                "subtheme": t,
                "own": own,
                "best_other": best,
                "best_other_r": other[best],
            }
        )
    return pl.DataFrame(rows)


def misfits(members: pl.DataFrame) -> pl.DataFrame:
    """Characteristics that fit another subtheme at least MISFIT_GAP better, at MISFIT_MIN or more."""
    return members.filter(
        (pl.col("best_other_r") >= pl.col("own") + MISFIT_GAP)
        & (pl.col("best_other_r") >= MISFIT_MIN)
    )


def composite_corr(corr: np.ndarray, themes: pl.DataFrame) -> pl.DataFrame:
    """
    Description:
        Correlations between subtheme composites, implied by the characteristic correlations.
    Steps:
        1) Weight matrix: each subtheme's members get their subtheme_sign, others zero.
        2) Composite covariance W' C W, scaled to correlations.
    Output:
        A subtheme column, then one column per subtheme, both in file order.
    """
    groups = subtheme_groups(themes)
    names = groups["subtheme"].to_list()
    w = np.zeros((themes.height, len(names)))
    for g, (pos, signs) in enumerate(groups.select("pos", "subtheme_sign").iter_rows()):
        w[pos, g] = signs
    s = w.T @ corr @ w
    comp = s / np.sqrt(np.outer(np.diag(s), np.diag(s)))
    return pl.DataFrame(comp, schema=names).insert_column(0, pl.Series("subtheme", names))


def subtheme_summary(
    corr: np.ndarray, composite: pl.DataFrame, themes: pl.DataFrame
) -> pl.DataFrame:
    """
    Description:
        How tightly each subtheme holds together, and which other subtheme is closest.
    Steps:
        1) Mean |correlation| between distinct members (None for a one-member subtheme).
        2) Nearest other subtheme by |composite correlation|.
    Output:
        One row per subtheme: subtheme, n, mean_abs_r, nearest_subtheme, nearest_r.
    """
    groups = subtheme_groups(themes)
    names = groups["subtheme"].to_list()
    comp = composite.drop("subtheme").to_numpy()
    comp_abs = np.abs(comp)
    np.fill_diagonal(comp_abs, 0.0)
    rows = []
    for g, (t, pos) in enumerate(groups.select("subtheme", "pos").iter_rows()):
        n = len(pos)
        sub = np.abs(corr[np.ix_(pos, pos)])
        within = float((sub.sum() - n) / (n * (n - 1))) if n > 1 else None
        nb = int(np.argmax(comp_abs[g]))
        rows.append(
            {
                "subtheme": t,
                "n": n,
                "mean_abs_r": within,
                "nearest_subtheme": names[nb],
                "nearest_r": float(comp[g, nb]),
            }
        )
    return pl.DataFrame(rows)


def print_report(
    members: pl.DataFrame, summary: pl.DataFrame, composite: pl.DataFrame, themes: pl.DataFrame
) -> None:
    """
    Description:
        Console summary of the diagnostics.
    Steps:
        1) Warn if a fit exceeds 1.
        2) Print coherence, clear misfits, weak fits, close subtheme pairs, signs the data
           disagree with, and same-theme subthemes that correlate negatively after theme signs.
    Output:
        None; prints to the console.
    """
    if (members["own"].abs() > 1).any() or (members["best_other_r"] > 1).any():
        print("Warning: some fits exceed 1; the average correlation matrix is too far from valid.")
    names = composite["subtheme"].to_list()
    comp = composite.drop("subtheme").to_numpy()
    groups = subtheme_groups(themes)
    theme_sign = dict(groups.select("subtheme", "theme_sign").iter_rows())
    theme_subs = groups.group_by("theme", maintain_order=True).agg("subtheme").iter_rows()

    with pl.Config(tbl_rows=-1, tbl_width_chars=160, float_precision=2):
        print(f"Median mean |r| within subthemes: {summary['mean_abs_r'].median():.2f}\n")
        print(summary.sort("mean_abs_r", nulls_last=True))
        print(
            f"\nClear misfits (fit another subtheme at least {MISFIT_GAP} better, "
            f"at {MISFIT_MIN} or more); move them or record why they stay:"
        )
        print(misfits(members).sort("subtheme"))
        print(f"\nWeakly attached characteristics (fit < {WEAK_FIT}):")
        print(members.filter(pl.col("own") < WEAK_FIT).sort("own"))
        print(f"\nSubtheme pairs whose composites correlate {CLOSE_PAIR} or more:")
        pairs = [
            (names[i], names[j], float(comp[i, j]))
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if abs(comp[i, j]) >= CLOSE_PAIR
        ]
        print(pl.DataFrame(pairs, schema=["subtheme", "other", "r"], orient="row").sort("r"))
        print("\nCharacteristics whose sign the data disagree with (negative fit):")
        print(members.filter(pl.col("own") < 0).sort("own"))
        print(
            f"\nSubthemes in the same theme correlating below -{SIGN_CONFLICT} after theme signs:"
        )
        conflicts = [
            (theme, a, b, comp[names.index(a), names.index(b)] * theme_sign[a] * theme_sign[b])
            for theme, subs in theme_subs
            for i, a in enumerate(subs)
            for b in subs[i + 1 :]
        ]
        conflicts = [c for c in conflicts if c[3] < -SIGN_CONFLICT]
        print(pl.DataFrame(conflicts, schema=["theme", "subtheme", "other", "r"], orient="row"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    themes = pl.read_csv(THEMES_CSV)
    corr = load_corr(themes["characteristic"].to_list())
    members = member_fits(corr, themes)
    composite = composite_corr(corr, themes)
    summary = subtheme_summary(corr, composite, themes)

    members.write_parquet(OUT_DIR / "theme_members.parquet")
    summary.write_parquet(OUT_DIR / "theme_summary.parquet")
    composite.write_parquet(OUT_DIR / "theme_composite_corr.parquet")
    print_report(members, summary, composite, themes)
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
