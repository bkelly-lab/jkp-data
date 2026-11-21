"""
Compustat return corrections.

Repairs decimal-shift price errors in the daily Compustat security files, then
screens stock-months (drops unreliable months, repairs jumps in shares and market
value) before returns are computed, following the data corrections of
Bessembinder et al. (2023). Daily rows follow their month's decision.

Layering: the public `correct_decimal_errors` / `screen_observations` /
`apply_monthly_screen` functions are domain-level orchestrators (sort -> build
typed inputs -> run -> reattach). A dedicated adapter converts the Polars panel to NumPy once, the
numba kernels (at the bottom of this file) do the per-security work, and the
result is reattached lazily. NumPy is only the kernel boundary, not the whole
design.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit, prange

from .config import (
    CORRECTION_SPILL_COMPRESSION,
    DECIMAL_CORRECTION_METHODS,
    DECIMAL_DETECTION_WINDOWS,
    EARLY_JUMP_MONTHS,
    EARLY_JUMP_SHARE,
    FILTER_GAP_MONTHS,
    JUMP_PAIR_MAX_MONTHS,
    LOW_PRICE_COUNTRIES,
    QUNIT_PRICE_CHANGE_BOUND,
)


@dataclass(frozen=True)
class DecimalCorrectionState:
    """Per-variable detection state at the decimal-correction kernel boundary.
    The kernels mutate factor/window_size/endpoints in place."""

    values: np.ndarray  # zero-nulled working copy (feeds detection and output)
    factor: np.ndarray  # correction multiplier, 1.0 = clean
    window_size: np.ndarray  # detection window, -1 = null
    left_endpoint: np.ndarray  # clean value to the left of a flagged spike
    right_endpoint: np.ndarray  # clean value to the right


def _group_starts(df: pl.DataFrame, group_cols: list[str]) -> np.ndarray:
    """
    Description:
        Contiguous group-boundary offsets for a frame sorted by group_cols + date.
    Output:
        int64 array of length n_groups + 1 (offsets into the frame).
    """
    n = df.height
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    gid = df.select(pl.struct(group_cols).rle_id().alias("_g"))["_g"].to_numpy()
    changes = np.flatnonzero(np.diff(gid)) + 1
    return np.concatenate(([0], changes, [n])).astype(np.int64)


# Decimal-error corrections
def _correct_variable_arrays(
    values: np.ndarray,
    starts: np.ndarray,
    window_sizes: list[int],
    correction_method: str,
    price_floor: bool = False,
    variation_threshold: float = 1.3,
) -> np.ndarray:
    """
    Description:
        Correct one variable: single- then multi-period detection, cascading
        validation, then apply. Zeros become NaN in the working copy that feeds
        both detection and output, so a zero input ends up null.
    Output:
        Corrected float64 array (NaN = null).
    """
    n = len(values)
    x = values.copy()
    x[x == 0.0] = np.nan
    state = DecimalCorrectionState(
        values=x,
        factor=np.ones(n, dtype=np.float64),
        window_size=np.full(n, -1, dtype=np.int32),
        left_endpoint=np.full(n, np.nan, dtype=np.float64),
        right_endpoint=np.full(n, np.nan, dtype=np.float64),
    )

    detect_single_period_all(
        state.values, starts, state.factor, state.left_endpoint, state.right_endpoint
    )
    state.window_size[state.factor != 1.0] = 1
    nlags = np.array(sorted(w for w in window_sizes if w > 1), dtype=np.int64)
    if len(nlags) > 0:
        detect_multi_period_all(
            state.values,
            starts,
            state.factor,
            state.window_size,
            state.left_endpoint,
            state.right_endpoint,
            nlags,
            variation_threshold,
        )
    validate_cascading_all(starts, state.factor, state.window_size)

    if price_floor:
        is_multiply = state.factor > 1.0  # correction direction is multiply
        is_sub_dollar = state.values < 1.0  # original price below $1
        # keep only divide-direction corrections on >=$1 prices
        state.factor[(state.factor != 1.0) & (is_multiply | is_sub_dollar)] = 1.0

    if correction_method not in ("interpolation", "floor_interp"):
        return state.values * state.factor

    # interpolation: replace a flagged spike with the geometric mean of its
    # clean endpoints (falling back to whichever single endpoint exists)
    flagged = state.factor != 1.0
    ep_l, ep_r = state.left_endpoint, state.right_endpoint
    has_l, has_r = ~np.isnan(ep_l), ~np.isnan(ep_r)
    corrected = state.values.copy()
    both = flagged & has_l & has_r
    left_only = flagged & has_l & ~has_r
    right_only = flagged & ~has_l & has_r
    corrected[both] = np.sqrt(ep_l[both] * ep_r[both])
    corrected[left_only] = ep_l[left_only]
    corrected[right_only] = ep_r[right_only]
    return corrected


def _load_correction_inputs(
    df: pl.LazyFrame, group_cols: list[str], sort_col: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Collect the keys and correction value columns (as f64), sorted by security and
    date; _row is each row's position in df. Return (data, starts)."""
    schema_names = df.collect_schema().names()
    value_cols = [
        v for v in ("trfd", "qunit", "adrrc", "ajexdi", "prccd", "cshoc") if v in schema_names
    ]
    data = (
        df.with_row_index("_row")
        .select(
            ["_row"] + group_cols + [sort_col] + [pl.col(c).cast(pl.Float64) for c in value_cols]
        )
        .sort(group_cols + [sort_col])
        .collect()
    )
    return data, _group_starts(data, group_cols)


def _run_decimal_correction(
    data: pl.DataFrame,
    starts: np.ndarray,
    window_sizes: list[int],
    has_adrrc: bool,
    correction_method: str,
    variation_threshold: float,
) -> dict[str, np.ndarray]:
    """Correct each variable independently and return {name: corrected array}."""
    cols = set(data.columns)  # only the value columns actually present were collected
    corrected: dict[str, np.ndarray] = {}
    # trfd, qunit and (NA data only) adrrc are corrected independently
    for variable in ("trfd", "qunit", "adrrc"):
        if variable in cols and (variable != "adrrc" or has_adrrc):
            corrected[variable] = _correct_variable_arrays(
                data[variable].to_numpy(),
                starts,
                window_sizes,
                correction_method,
                variation_threshold=variation_threshold,
            )

    # reconstruct prccd/cshoc from the corrected split-adjusted price and shares
    if {"ajexdi", "prccd", "cshoc"} <= cols:
        ajexdi = data["ajexdi"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            # floor variants gate the price only (divide-direction, >=$1)
            adjprc = _correct_variable_arrays(
                data["prccd"].to_numpy() / ajexdi,
                starts,
                window_sizes,
                correction_method,
                price_floor=correction_method in ("floor", "floor_interp"),
                variation_threshold=variation_threshold,
            )
            adjcsho = _correct_variable_arrays(
                data["cshoc"].to_numpy() * ajexdi,
                starts,
                window_sizes,
                correction_method,
                variation_threshold=variation_threshold,
            )
            corrected["prccd"] = adjprc * ajexdi
            corrected["cshoc"] = adjcsho / ajexdi
    return corrected


def _reattach_corrected(
    df: pl.LazyFrame, row: pl.Series, corrected: dict[str, np.ndarray], spill_dir: Path
) -> pl.LazyFrame:
    """Splice corrected columns over the originals (which move to the end), in df's row order."""
    corr_path = spill_dir / "__corr_corrected_cols.parquet"
    (
        pl.DataFrame(corrected)
        .with_columns([pl.col(name).fill_nan(None) for name in corrected], _row=row)
        .sort("_row")
        .drop("_row")
        .write_parquet(corr_path, compression=CORRECTION_SPILL_COMPRESSION)
    )
    return pl.concat([df.drop(list(corrected)), pl.scan_parquet(corr_path)], how="horizontal")


def correct_decimal_errors(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    window_sizes: list[int] | None = None,
    has_adrrc: bool = False,
    correction_method: str = "multiplier",
    spill_dir: Path | None = None,
    variation_threshold: float = 1.3,
) -> pl.LazyFrame:
    """
    Description:
        Repair decimal-shift errors in the price/shares/return-factor columns,
        implementing the decimal-error corrections of Section 6 of Bessembinder
        et al. (2023) Data Appendix. Correct TRFD, QUNIT and (NA only) ADRRC
        independently, then adjPRC = PRCCD/AJEXDI and adjCSHO = CSHOC*AJEXDI,
        and reconstruct PRCCD/CSHOC. Only the keys and value columns are
        sorted and collected; spill files go to spill_dir (required).
        df is read twice, once to collect the kernel columns and once inside
        the returned plan to splice the results back by row position, so it
        must yield the same rows in the same order both times (e.g. a scan of
        a file that is not overwritten before the result is collected or sunk).
    Args:
        correction_method: 'multiplier' (fixed 10x/100x/1000x multipliers),
            'interpolation' (endpoint geometric mean), 'floor' (multipliers,
            price divide-direction on >=$1 only), 'floor_interp' (both).
        has_adrrc: correct the ADRRC column (NA data only).
        variation_threshold: multi-period interior-variation bound.
    Output:
        Corrected LazyFrame in df's row order (corrected columns move to the
        end; NaN -> null).
    """
    if correction_method not in DECIMAL_CORRECTION_METHODS:
        raise ValueError(
            f"Unknown correction_method: {correction_method!r}; expected one of {DECIMAL_CORRECTION_METHODS}"
        )
    if spill_dir is None:
        raise ValueError("correct_decimal_errors requires spill_dir")
    group_cols = group_cols or ["gvkey", "iid"]
    window_sizes = window_sizes if window_sizes is not None else DECIMAL_DETECTION_WINDOWS

    data, starts = _load_correction_inputs(df, group_cols, sort_col)
    corrected = _run_decimal_correction(
        data, starts, window_sizes, has_adrrc, correction_method, variation_threshold
    )
    return _reattach_corrected(df, data["_row"], corrected, spill_dir)


# Observation filters (Section 8), run on stock-months
def _month_index(eom: pl.Expr) -> pl.Expr:
    """Months since year 0, so consecutive months differ by 1."""
    return (eom.dt.year() * 12 + eom.dt.month()).cast(pl.Int32)


def _load_monthly_panel(
    monthly: pl.LazyFrame, group_cols: list[str], country_col: str
) -> pl.DataFrame:
    """Collect the monthly kernel columns sorted by security and month: f64 values, month
    index, currency code (0 = missing) and country flags."""
    return (
        monthly.select(
            *group_cols,
            "eom",
            *[pl.col(c).cast(pl.Float64) for c in ["prc", "me", "ri", "cshoc", "ajexdi", "qunit"]],
            mi=_month_index(pl.col("eom")),
            curcd=pl.col("curcdd").rank("dense").fill_null(0).cast(pl.Int64),
            low=pl.col(country_col).is_in(LOW_PRICE_COUNTRIES).fill_null(False),
            chn=(pl.col(country_col) == "CHN").fill_null(False),
        )
        .sort([*group_cols, "eom"])
        .collect()
    )


def _security_flags(
    daily: pl.LazyFrame, panel: pl.DataFrame, group_cols: list[str]
) -> pl.DataFrame:
    """
    Description:
        Security-level removal decisions that use the whole history (Section 8a-b, AJEXDI).
    Steps:
        1) Low volume: mean positive daily dollar volume per security; the bottom 2% of
           securities are removed. Securities without positive volume are kept.
        2) Zero AJEXDI: any daily or monthly AJEXDI equal to zero.
    Output:
        DataFrame [group_cols, remove_low_volume, zero_ajexdi], one row per panel
        security, in panel order.
    """
    avg_vol = (
        daily.filter(pl.col("dolvol") > 0)
        .group_by(group_cols)
        .agg(_avg_vol=pl.col("dolvol").cast(pl.Float64).mean())
        .collect()
    )
    cutoff = avg_vol["_avg_vol"].quantile(0.02)
    zero_ajexdi = pl.concat(
        [
            daily.filter(pl.col("ajexdi") == 0).select(group_cols).unique().collect(),
            panel.filter(pl.col("ajexdi") == 0).select(group_cols).unique(),
        ]
    ).unique()
    return (
        panel.select(group_cols)
        .unique(maintain_order=True)
        .join(avg_vol, on=group_cols, how="left", nulls_equal=True, maintain_order="left")
        .join(
            zero_ajexdi.with_columns(_zero=pl.lit(True)),
            on=group_cols,
            how="left",
            nulls_equal=True,
            maintain_order="left",
        )
        .select(
            *group_cols,
            remove_low_volume=pl.col("_avg_vol").le(cutoff).fill_null(False),
            zero_ajexdi=pl.col("_zero").fill_null(False),
        )
    )


def _event_dates(daily: pl.LazyFrame, events: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """
    Description:
        Place each repaired monthly jump on a trading day, so daily rows before the jump
        keep the previous month's multiplier.
    Steps:
        1) Daily rows of securities with a jump, with the step from the previous trading
           day in split-adjusted shares (shares jumps) or in market value net of the
           return (market-value jumps).
        2) In each jump month, the day with the largest absolute log step.
    Output:
        DataFrame [group_cols, eom, event_date].
    """
    prev = lambda c: pl.col(c).shift(1).over(group_cols)  # noqa: E731
    return (
        daily.join(events.lazy().select(group_cols).unique(), on=group_cols, how="semi")
        .select(
            *group_cols,
            "eom",
            pl.col("datadate").cast(pl.Date),
            adj=(pl.col("cshoc") * pl.col("ajexdi")).cast(pl.Float64),
            me=pl.col("me").cast(pl.Float64),
            ri=pl.col("ri").cast(pl.Float64),
        )
        .sort([*group_cols, "datadate"])
        .with_columns(
            shares_step=(pl.col("adj") / prev("adj")).log().abs().fill_nan(None),
            value_step=((pl.col("me") / prev("me")) / (pl.col("ri") / prev("ri")))
            .log()
            .abs()
            .fill_nan(None),
        )
        .join(events.lazy(), on=[*group_cols, "eom"], how="inner", nulls_equal=True)
        .with_columns(
            step=pl.when(pl.col("event") == 2)
            .then(pl.col("value_step"))
            .otherwise(pl.col("shares_step"))
        )
        .group_by([*group_cols, "eom"])
        .agg(
            event_date=pl.col("datadate")
            .sort_by(["step", "datadate"], descending=[True, False], nulls_last=True)
            .first()
        )
        .collect()
    )


def screen_observations(
    daily: pl.LazyFrame,
    monthly: pl.LazyFrame,
    group_cols: list[str] | None = None,
    country_col: str = "excntry",
    pair_max_months: int = JUMP_PAIR_MAX_MONTHS,
) -> pl.DataFrame:
    """
    Description:
        Observation filters and jump repairs of Section 8 of Bessembinder et al. (2023)
        Data Appendix, run on stock-months as in the paper. Returns a keep decision and a
        shares multiplier for every security-month of daily and monthly, so daily and
        monthly rows follow one decision (apply_monthly_screen).
    Steps:
        1) Load the monthly panel (one row per security-month) and the security-level
           low-volume and zero-AJEXDI decisions.
        2) run_monthly_screen, one numba pass per security: low volume, zero AJEXDI,
           QUNIT changes, low price or market value (months after the breach month),
           gaps, shares and market-value jumps (early jumps delete the history through
           the jump; later jumps are repaired), return/market-value mismatches, initial
           ratio errors.
        3) Place each repaired jump on a trading day (_event_dates).
        4) Spread the decisions over every security-month of daily and monthly: months
           without a panel row take the security-level cuts and the previous kept
           month's multiplier.
    Output:
        DataFrame [group_cols, eom, keep, factor, factor_prev, event_date]. factor applies
        to shares and market value from event_date on, factor_prev before it.
    """
    group_cols = group_cols or ["gvkey", "iid"]
    panel = _load_monthly_panel(monthly, group_cols, country_col)
    flags = _security_flags(daily, panel, group_cols)
    starts = _group_starts(panel, group_cols)
    n, n_sec = panel.height, len(starts) - 1
    reason = np.zeros(n, dtype=np.int8)
    factor = np.ones(n, dtype=np.float64)
    event = np.zeros(n, dtype=np.int8)
    killed = np.zeros(n_sec, dtype=np.bool_)
    start_cut = np.full(n_sec, -1, dtype=np.int64)
    end_cut = np.full(n_sec, -1, dtype=np.int64)
    mi = panel["mi"].to_numpy()
    run_monthly_screen(
        starts,
        reason,
        factor,
        event,
        killed,
        start_cut,
        end_cut,
        flags["remove_low_volume"].to_numpy(),
        flags["zero_ajexdi"].to_numpy(),
        *[panel[c].to_numpy() for c in ["prc", "me", "ri", "cshoc", "ajexdi", "qunit"]],
        panel["curcd"].to_numpy(),
        mi,
        panel["low"].to_numpy(),
        panel["chn"].to_numpy(),
        FILTER_GAP_MONTHS + 1,
        float(EARLY_JUMP_MONTHS),
        EARLY_JUMP_SHARE,
        pair_max_months,
        QUNIT_PRICE_CHANGE_BOUND,
    )
    cut_mi = lambda idx: pl.Series(np.where(idx >= 0, mi[np.maximum(idx, 0)], -1)).replace(  # noqa: E731
        -1, None
    )
    cuts = flags.select(group_cols).with_columns(
        killed=pl.Series(killed), start_cut=cut_mi(start_cut), end_cut=cut_mi(end_cut)
    )
    alive = (
        panel.select(*group_cols, "eom")
        .with_columns(reason=pl.Series(reason), factor=pl.Series(factor), event=pl.Series(event))
        .filter(pl.col("reason") == 0)
        .with_columns(factor_prev=pl.col("factor").shift(1).over(group_cols).fill_null(1.0))
    )
    events = alive.filter(pl.col("event") > 0).select(*group_cols, "eom", "event")
    removed = panel.select(*group_cols, "eom").filter(pl.Series(reason) != 0)
    months = pl.concat(
        [daily.select(*group_cols, "eom").unique().collect(), panel.select(*group_cols, "eom")]
    ).unique()
    return (
        months.join(
            alive.drop("reason", "event"), on=[*group_cols, "eom"], how="left", nulls_equal=True
        )
        .join(
            _event_dates(daily, events, group_cols),
            on=[*group_cols, "eom"],
            how="left",
            nulls_equal=True,
        )
        .join(
            removed.with_columns(_removed=pl.lit(True)),
            on=[*group_cols, "eom"],
            how="left",
            nulls_equal=True,
        )
        .join(cuts, on=group_cols, how="left", nulls_equal=True)
        .sort([*group_cols, "eom"])
        .with_columns(
            mi=_month_index(pl.col("eom")),
            _filled=pl.col("factor").forward_fill().over(group_cols).fill_null(1.0),
        )
        .select(
            *group_cols,
            "eom",
            keep=~pl.col("killed").fill_null(False)
            & ~pl.col("_removed").fill_null(False)
            & (pl.col("start_cut").is_null() | (pl.col("mi") > pl.col("start_cut")))
            & (pl.col("end_cut").is_null() | (pl.col("mi") <= pl.col("end_cut"))),
            factor=pl.col("_filled"),
            factor_prev=pl.coalesce("factor_prev", "_filled"),
            event_date="event_date",
        )
    )


def apply_monthly_screen(
    df: pl.LazyFrame, decisions: pl.LazyFrame, group_cols: list[str] | None = None
) -> pl.LazyFrame:
    """
    Description:
        Apply the security-month decisions of screen_observations to daily or monthly rows.
    Steps:
        1) Join the decision of each row's security-month; keep the kept months.
        2) Multiply shares (cshoc) and market value (me) by factor, or by factor_prev
           before the month's event_date.
    Output:
        LazyFrame with df's columns.
    """
    group_cols = group_cols or ["gvkey", "iid"]
    f = (
        pl.when(pl.col("_event_date").is_not_null() & (pl.col("datadate") < pl.col("_event_date")))
        .then(pl.col("_factor_prev"))
        .otherwise(pl.col("_factor"))
    )
    return (
        df.join(
            decisions.select(
                *group_cols,
                "eom",
                _keep="keep",
                _factor="factor",
                _factor_prev="factor_prev",
                _event_date="event_date",
            ),
            on=[*group_cols, "eom"],
            how="left",
            nulls_equal=True,
        )
        .filter(pl.col("_keep"))
        .with_columns(cshoc=pl.col("cshoc") * f, me=pl.col("me") * f)
        .drop("_keep", "_factor", "_factor_prev", "_event_date")
    )


# Numba kernels: per-security loops over date-sorted array slices, parallelized
# with prange over group boundaries; kernels mutate preallocated arrays in place.
#
# Detection-kernel outputs: factor = correction multiplier (1.0 = clean),
# wsize = detection window (-1 = null), ep_l/ep_r = clean endpoints (used by
# interpolation). The filter kernels mark removed rows in `reason` (0 = kept,
# 1 = removed).
#
# Equivalence notes (load-bearing — do not "simplify"):
#   - Out-of-range endpoint factor counts as clean (1.0) but its VALUE is NaN,
#     so the reversal ratio fails and no detection fires.
#   - Interior variation max/min skip NaN cells; an all-NaN interior kills it.
#   - Propagated rows carry the detection point's endpoints and window_size=nlag,
#     claimable only while their factor is still 1.0.
#   - Multi-period classifies magnitudes with the same nested 500/50/5 chain as
#     single-period (deliberate divergence from the original dead-code loop;
#     commit 507df7c).


@njit(cache=True, error_model="numpy")
def _magnitude_factor(rl, rr):
    """
    Correction factor for a spike whose ratios to both endpoints are rl, rr.
    Nested first-match order (power-of-10 magnitude); 1.0 means no error.
    """
    if rl > 500.0 and rr > 500.0:
        return 0.001
    if rl > 50.0 and rr > 50.0:
        return 0.01
    if rl > 5.0 and rr > 5.0:
        return 0.1
    if rl < 0.002 and rr < 0.002:
        return 1000.0
    if rl < 0.02 and rr < 0.02:
        return 100.0
    if rl < 0.2 and rr < 0.2:
        return 10.0
    return 1.0


@njit(cache=True, error_model="numpy")
def _detect_single_one(x, factor, ep_l, ep_r):
    """Single-period detection for one security slice (arrays are views)."""
    n = len(x)
    for t in range(1, n - 1):
        xt = x[t]
        xp = x[t - 1]
        xn = x[t + 1]
        if np.isnan(xt) or np.isnan(xp) or np.isnan(xn):
            continue
        f = _magnitude_factor(xt / xp, xt / xn)
        if f == 1.0:
            continue
        factor[t] = f
        ep_l[t] = xp
        ep_r[t] = xn


@njit(parallel=True, cache=True, error_model="numpy")
def detect_single_period_all(x, starts, factor, ep_l, ep_r):
    """Run single-period detection over all securities in parallel."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        _detect_single_one(x[s:e], factor[s:e], ep_l[s:e], ep_r[s:e])


@njit(cache=True, error_model="numpy")
def _multi_pass_one(
    x, factor, wsize, ep_l, ep_r, nlag, ep_lag, ep_lead, ilo, ihi, vthr, det, det_epl, det_epr
):
    """
    One window pass (FULL/SUB_A/SUB_B) for one security. Phase 1 detects into
    the det* scratch arrays against the pass-start factor state; phase 2
    propagates first-write-wins in ascending offset order onto still-1.0 rows.
    """
    n = len(x)
    for k in range(n):
        det[k] = np.nan

    # Phase 1: detection
    for t in range(n):
        if factor[t] != 1.0:
            continue
        il = t - ep_lag
        ir = t + ep_lead
        # Out-of-range endpoint: clean guard sees 1.0, value is NaN
        fl = factor[il] if 0 <= il < n else 1.0
        fr = factor[ir] if 0 <= ir < n else 1.0
        if fl != 1.0 or fr != 1.0:
            continue
        xt = x[t]
        if np.isnan(xt):
            continue
        xl = x[il] if 0 <= il < n else np.nan
        xr = x[ir] if 0 <= ir < n else np.nan
        rl = xt / xl
        rr = xt / xr
        high = rl > 5.0 and rr > 5.0
        low = rl < 0.2 and rr < 0.2
        if not (high or low):
            continue
        # Interior variation: skip NaN/out-of-range cells; all-NaN -> no detect
        imax = -np.inf
        imin = np.inf
        count = 0
        for off in range(ilo, ihi + 1):
            j = t + off
            if j < 0 or j >= n:
                continue
            v = x[j]
            if np.isnan(v):
                continue
            count += 1
            if v > imax:
                imax = v
            if v < imin:
                imin = v
        if count == 0:
            continue
        if not (imax / imin < vthr):
            continue
        # high/low gate above guarantees a non-1.0 magnitude here
        det[t] = _magnitude_factor(rl, rr)
        det_epl[t] = xl
        det_epr[t] = xr

    # Phase 2: propagation, first-write-wins in ascending offset order
    for off in range(ilo, ihi + 1):
        for r in range(n):
            t = r - off  # shift(off) moves value from row r-off to row r
            if t < 0 or t >= n:
                continue
            if np.isnan(det[t]) or factor[r] != 1.0:
                continue
            factor[r] = det[t]
            wsize[r] = nlag
            ep_l[r] = det_epl[t]
            ep_r[r] = det_epr[t]


@njit(cache=True, error_model="numpy")
def _multi_one(x, factor, wsize, ep_l, ep_r, nlags, vthr):
    """All multi-period passes for one security: nlags ascending, FULL -> SUB_A -> SUB_B."""
    n = len(x)
    det = np.empty(n, dtype=np.float64)
    det_epl = np.empty(n, dtype=np.float64)
    det_epr = np.empty(n, dtype=np.float64)
    for i in range(len(nlags)):
        nlag = nlags[i]
        if nlag <= 1:
            continue
        # Three window passes, each (ep_lag, ep_lead, ilo, ihi):
        #   FULL : endpoints t-nlag  / t+nlag   , interior -(nlag-1)..(nlag-1)
        #   SUB_A: endpoints t-nlag  / t+nlag-1 , interior -(nlag-1)..(nlag-2)
        #   SUB_B: endpoints t-nlag+1/ t+nlag   , interior -(nlag-2)..(nlag-1)
        for ep_lag, ep_lead, ilo, ihi in (
            (nlag, nlag, -(nlag - 1), nlag - 1),
            (nlag, nlag - 1, -(nlag - 1), nlag - 2),
            (nlag - 1, nlag, -(nlag - 2), nlag - 1),
        ):
            _multi_pass_one(
                x,
                factor,
                wsize,
                ep_l,
                ep_r,
                nlag,
                ep_lag,
                ep_lead,
                ilo,
                ihi,
                vthr,
                det,
                det_epl,
                det_epr,
            )


@njit(parallel=True, cache=True, error_model="numpy")
def detect_multi_period_all(x, starts, factor, wsize, ep_l, ep_r, nlags, vthr):
    """Run multi-period detection over all securities in parallel."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        _multi_one(x[s:e], factor[s:e], wsize[s:e], ep_l[s:e], ep_r[s:e], nlags, vthr)


@njit(cache=True, error_model="numpy")
def _validate_one(factor, wsize, reject):
    """
    Cascading validation for one security: reject corrections whose BOTH
    endpoint positions (p +/- window_size, null -> 1) are themselves flagged.
    Rejections applied simultaneously per round, max 10 rounds.
    """
    n = len(factor)
    for _ in range(10):
        any_reject = False
        for p in range(n):
            reject[p] = False
        for p in range(n):
            if factor[p] == 1.0:
                continue
            w = wsize[p] if wsize[p] != -1 else 1  # -1 = null window size
            lp = p - w
            rp = p + w
            lf = 0 <= lp < n and factor[lp] != 1.0
            rf = 0 <= rp < n and factor[rp] != 1.0
            if lf and rf:
                reject[p] = True
                any_reject = True
        if not any_reject:
            break
        for p in range(n):
            if reject[p]:
                factor[p] = 1.0


@njit(parallel=True, cache=True, error_model="numpy")
def validate_cascading_all(starts, factor, wsize):
    """Run cascading validation over all securities (factor reset in place)."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        scratch = np.empty(e - s, dtype=np.bool_)
        _validate_one(factor[s:e], wsize[s:e], scratch)


# Observation-filter kernels (thresholds from Bessembinder et al. 2023 Data Appendix)


@njit(cache=True, error_model="numpy")
def _kill_all(reason):
    """Remove every still-alive row of the security."""
    for i in range(reason.size):
        if reason[i] == 0:
            reason[i] = 1


@njit(cache=True, error_model="numpy")
def _alive_rows(reason):
    """Positions of the rows still alive, in date order."""
    n = 0
    for i in range(reason.size):
        if reason[i] == 0:
            n += 1
    idx = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(reason.size):
        if reason[i] == 0:
            idx[k] = i
            k += 1
    return idx


@njit(cache=True, error_model="numpy")
def _jump_direction(x_prev, x_cur, c_prev, c_cur, return_based, chn):
    """
    +1 / -1 for an up / down jump in x between two months, 0 otherwise. The shares
    test (return_based False) needs the market value c to move with x (looser CHN
    bounds); the market-value test (True) needs the return from c (a return index) not
    to explain the move. NaN operands fail every test.
    """
    jump = x_cur / x_prev
    if return_based:
        ret = c_cur / c_prev - 1.0
        if jump > 10.0 and ret < 2.0:
            return 1
        if jump < 0.1 and ret > -0.5:
            return -1
        return 0
    confirm = c_cur / c_prev
    if chn:
        up = jump >= 50.0 and confirm >= 25.0
    else:
        up = jump >= 5.0 and confirm >= 2.5
    if up:
        return 1
    if jump <= 0.2 and confirm <= 0.4:
        return -1
    return 0


@njit(cache=True, error_model="numpy")
def _early_jump_stage(reason, x, confirm, return_based, chn, early_months, early_share):
    """
    A jump within the lesser of the first early_months months or the first early_share
    of the security's months deletes the alive rows through the last such jump. Returns
    the position of that jump, -1 if none.
    """
    idx = _alive_rows(reason)
    window = min(early_months, early_share * idx.size)
    last = -1
    for k in range(1, idx.size):
        if k >= window:
            break
        i, p = idx[k], idx[k - 1]
        if _jump_direction(x[p], x[i], confirm[p], confirm[i], return_based, chn[i]) != 0:
            last = k
    for k in range(last + 1):
        reason[idx[k]] = 1
    return idx[last] if last >= 0 else -1


@njit(cache=True, error_model="numpy")
def _repair_jumps(reason, x, confirm, return_based, chn, mi, pair_max_months, factor, event, code):
    """
    Repair the jumps left after the early stage, rescaling x in place and multiplying
    factor by the same amount; marks repaired jump months in event with code.
    - Paired (the next jump is the opposite one, within pair_max_months): the months
      from the jump to before the reversal get min(x before the jump, x at the reversal),
      each end net of its month's return (market-value test only).
    - Unpaired up-jump: x from the jump on is scaled down to the level before it.
    - Unpaired down-jump: x before the jump is scaled down to the level after it.
    """
    idx = _alive_rows(reason)
    m = idx.size
    k = 1
    while k < m:
        i, p = idx[k], idx[k - 1]
        d = 0
        if x[p] > 0.0 and x[i] > 0.0:
            d = _jump_direction(x[p], x[i], confirm[p], confirm[i], return_based, chn[i])
        if d == 0:
            k += 1
            continue
        growth = confirm[i] / confirm[p] if return_based else 1.0
        pair = -1
        for q in range(k + 1, m):
            j, jp = idx[q], idx[q - 1]
            if mi[j] - mi[i] > pair_max_months:
                break
            if not (x[jp] > 0.0 and x[j] > 0.0):
                continue
            dj = _jump_direction(x[jp], x[j], confirm[jp], confirm[j], return_based, chn[j])
            if dj == -d:
                pair = q
                break
            if dj == d:
                break
        if pair >= 0:
            j, jp = idx[pair], idx[pair - 1]
            growth_s = confirm[j] / confirm[jp] if return_based else 1.0
            level = min(x[p] * growth, x[j] / growth_s)
            for r in range(k, pair):
                u = idx[r]
                factor[u] *= level / x[u]
                x[u] = level
            event[i] = code
            event[j] = code
            k = pair + 1
            continue
        if d == 1:
            scale = x[p] * growth / x[i]
            lo, hi = k, m
        else:
            scale = x[i] / (x[p] * growth)
            lo, hi = 0, k
        for r in range(lo, hi):
            u = idx[r]
            x[u] *= scale
            factor[u] *= scale
        event[i] = code
        k += 1


# Each filter reads the rows still alive at its start (sequential survivor) and
# marks removals in `reason`. Those that can void the whole security return True
# to stop the chain.


@njit(cache=True, error_model="numpy")
def _filter_qunit_changes(reason, qunit, curcd, prc, ajexdi, max_change):
    """
    A change in the quote unit QUNIT without a change in currency, together with a
    split-adjusted price change beyond +-max_change, removes the whole security.
    """
    idx = _alive_rows(reason)
    for k in range(1, idx.size):
        i, p = idx[k], idx[k - 1]
        unit_changed = qunit[i] == qunit[i] and qunit[p] == qunit[p] and qunit[i] != qunit[p]
        same_currency = curcd[i] > 0 and curcd[i] == curcd[p]
        if unit_changed and same_currency:
            change = (prc[i] / ajexdi[i]) / (prc[p] / ajexdi[p]) - 1.0
            if change > max_change or change < -max_change:
                _kill_all(reason)
                return True
    return False


@njit(cache=True, error_model="numpy")
def _filter_low_price_or_me(reason, prc, me, low_thr):
    """
    Price below threshold ($0.01, or $0.001 in low_thr countries) or market value below
    $1M at a month-end. A breach in the first month voids the security; a later breach
    removes the months after it. Returns the breach position, -2 if the security is
    voided, -1 if there is no breach.
    """
    idx = _alive_rows(reason)
    for k in range(idx.size):
        i = idx[k]
        thr = 0.001 if low_thr[i] else 0.01
        if me[i] < 1.0 or prc[i] < thr:
            if k == 0:
                _kill_all(reason)
                return -2
            for r in range(k + 1, idx.size):
                reason[idx[r]] = 1
            return i
    return -1


@njit(cache=True, error_model="numpy")
def _filter_data_gaps(reason, mi, max_step):
    """Drop the month after a step of more than max_step months."""
    prev = -1
    for i in range(reason.size):
        if reason[i] != 0:
            continue
        if prev >= 0 and mi[i] - mi[prev] > max_step:
            reason[i] = 1
        # prev advances even past a removed row: gaps are measured against the
        # pre-stage rows, so a removed row is still the next row's reference
        prev = i


@njit(cache=True, error_model="numpy")
def _filter_return_me_mismatch(reason, ri, me):
    """|return| > 80% with |market-value change| < 50%."""
    prev = -1
    for i in range(reason.size):
        if reason[i] != 0:
            continue
        if prev >= 0:
            ret = ri[i] / ri[prev] - 1.0
            me_chg = me[i] / me[prev] - 1.0
            if abs(ret) > 0.8 and abs(me_chg) < 0.5:
                reason[i] = 1
        # a flagged row still advances prev, staying the reference for the next row
        prev = i


@njit(cache=True, error_model="numpy")
def _filter_initial_ratio_errors(reason, adj_prc, me):
    """A split-adjusted price or market-value ratio outside [0.1, 10] among the first
    3 observations."""
    prev = -1
    obs = -1
    for i in range(reason.size):
        if reason[i] != 0:
            continue
        obs += 1
        if obs > 2:  # only the first 3 observations (0-indexed)
            break
        if prev >= 0:
            pr = adj_prc[i] / adj_prc[prev]
            mr = me[i] / me[prev]
            if pr > 10.0 or pr < 0.1 or mr > 10.0 or mr < 0.1:
                reason[i] = 1
        prev = i  # flagged rows still advance prev (see _filter_return_me_mismatch)


@njit(cache=True, error_model="numpy")
def _screen_one_security(
    reason,
    factor,
    event,
    remove_low_volume,
    zero_ajexdi,
    prc,
    me,
    ri,
    cshoc,
    ajexdi,
    qunit,
    curcd,
    mi,
    low_thr,
    chn,
    max_step,
    early_months,
    early_share,
    pair_max_months,
    qunit_max_change,
):
    """
    Run the Section 8 chain in order on one security's month-sorted slice. Returns
    (killed, start_cut, end_cut): start_cut is the position of the last early jump
    (rows through it deleted), end_cut the breach month (rows after it deleted), -1 if
    none.
    """
    if remove_low_volume or zero_ajexdi:
        _kill_all(reason)
        return True, -1, -1
    if _filter_qunit_changes(reason, qunit, curcd, prc, ajexdi, qunit_max_change):
        return True, -1, -1
    breach = _filter_low_price_or_me(reason, prc, me, low_thr)
    if breach == -2:
        return True, -1, -1
    _filter_data_gaps(reason, mi, max_step)
    # the breach month only carries its return: held out of the jump stages, so its
    # drop never rescales the history before it
    held_out = breach >= 0 and reason[breach] == 0
    if held_out:
        reason[breach] = 2
    # shares jumps: adjCSHO against market value, then market value is rebuilt
    adj = cshoc * ajexdi
    cut_shares = _early_jump_stage(reason, adj, me, False, chn, early_months, early_share)
    _repair_jumps(reason, adj, me, False, chn, mi, pair_max_months, factor, event, 1)
    # market-value jumps on the rebuilt market value, against the return
    value = me * factor
    cut_value = _early_jump_stage(reason, value, ri, True, chn, early_months, early_share)
    _repair_jumps(reason, value, ri, True, chn, mi, pair_max_months, factor, event, 2)
    if held_out:
        reason[breach] = 0
        # the breach month follows the multiplier of the month before it
        k = breach - 1
        while k >= 0 and reason[k] != 0:
            k -= 1
        if k >= 0:
            factor[breach] = factor[k]
            value[breach] = me[breach] * factor[k]
    _filter_return_me_mismatch(reason, ri, value)
    _filter_initial_ratio_errors(reason, prc / ajexdi, value)
    return False, max(cut_shares, cut_value), breach


@njit(parallel=True, cache=True, error_model="numpy")
def run_monthly_screen(
    starts,
    reason,
    factor,
    event,
    killed,
    start_cut,
    end_cut,
    remove_low_volume,
    zero_ajexdi,
    prc,
    me,
    ri,
    cshoc,
    ajexdi,
    qunit,
    curcd,
    mi,
    low_thr,
    chn,
    max_step,
    early_months,
    early_share,
    pair_max_months,
    qunit_max_change,
):
    """Run the Section 8 chain over all securities in parallel; start_cut and end_cut
    hold absolute row positions."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        k, c0, c1 = _screen_one_security(
            reason[s:e],
            factor[s:e],
            event[s:e],
            remove_low_volume[g],
            zero_ajexdi[g],
            prc[s:e],
            me[s:e],
            ri[s:e],
            cshoc[s:e],
            ajexdi[s:e],
            qunit[s:e],
            curcd[s:e],
            mi[s:e],
            low_thr[s:e],
            chn[s:e],
            max_step,
            early_months,
            early_share,
            pair_max_months,
            qunit_max_change,
        )
        killed[g] = k
        start_cut[g] = s + c0 if c0 >= 0 else -1
        end_cut[g] = s + c1 if c1 >= 0 else -1
