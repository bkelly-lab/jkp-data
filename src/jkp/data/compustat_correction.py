"""
The Compustat correction, based on the methodology of Bessembinder, Chen, Choi and Wei
(2023), 'Long-Term Shareholder Returns: Evidence from 64,000 Global Stocks', Financial
Analysts Journal, Internet Data Appendix.

Repairs decimal-shift price errors in the daily Compustat security files, then screens
stock-months before returns are computed: drops unreliable months and repairs jumps in
shares outstanding and market value. Daily rows follow their month's decision.

`correct_decimal_errors` and `screen_stock_months` sort the data into one Polars panel,
convert it to NumPy arrays once and run numba kernels (at the bottom of this file) that
loop over one security at a time; the results are joined back onto the data.
`apply_screen` applies the screen's decisions in Polars.
"""

import math
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
        Repair decimal-shift errors in the price/shares/return-factor columns, the
        first step of the Compustat correction. Correct TRFD, QUNIT and (NA only) ADRRC
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


# Stock-month screen. In the kernels, each row of a security's month-sorted slice has a
# status, and each repaired jump month a kind.
_KEPT, _DROPPED, _HELD_OUT = 0, 1, 2
_SHARES_JUMP, _VALUE_JUMP = 1, 2


def _month_number(eom: pl.Expr) -> pl.Expr:
    """
    Description:
        A month counter for measuring steps between months.
    Steps:
        1) year * 12 + month.
    Output:
        Int32 expression; consecutive months differ by 1.
    """
    return (eom.dt.year() * 12 + eom.dt.month()).cast(pl.Int32)


def _load_screen_panel(
    monthly: pl.LazyFrame, group_cols: list[str], country_col: str
) -> pl.DataFrame:
    """
    Description:
        The monthly panel the screen kernels read, one row per security-month.
    Steps:
        1) Values as f64, the month number, the currency as a code (0 = missing), and
           whether the country has the low minimum price or is China.
        2) Sort by security and month.
    Output:
        DataFrame [group_cols, eom, prc, me, ri, cshoc, ajexdi, qunit, month, currency,
        low_price_country, is_china].
    """
    return (
        monthly.select(
            *group_cols,
            "eom",
            *[pl.col(c).cast(pl.Float64) for c in ["prc", "me", "ri", "cshoc", "ajexdi", "qunit"]],
            month=_month_number(pl.col("eom")),
            currency=pl.col("curcdd").rank("dense").fill_null(0).cast(pl.Int64),
            low_price_country=pl.col(country_col).is_in(LOW_PRICE_COUNTRIES).fill_null(False),
            is_china=(pl.col(country_col) == "CHN").fill_null(False),
        )
        .sort([*group_cols, "eom"])
        .collect()
    )


def _security_removals(
    daily: pl.LazyFrame, panel: pl.DataFrame, group_cols: list[str]
) -> pl.DataFrame:
    """
    Description:
        The removals that judge a security on its whole history: low trading volume and
        a zero adjustment factor.
    Steps:
        1) Low volume: mean positive daily dollar volume per security; the bottom 2% of
           securities are removed. Securities without positive volume are kept.
        2) Zero AJEXDI: any daily or monthly AJEXDI equal to zero.
    Output:
        DataFrame [group_cols, low_volume, zero_ajexdi], one row per panel security, in
        panel order.
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
            low_volume=pl.col("_avg_vol").le(cutoff).fill_null(False),
            zero_ajexdi=pl.col("_zero").fill_null(False),
        )
    )


def _jump_days(daily: pl.LazyFrame, jumps: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """
    Description:
        Place each repaired monthly jump on a trading day, so daily rows before the jump
        keep the previous month's multiplier.
    Steps:
        1) Daily rows of securities with a jump, with the step from the previous trading
           day in split-adjusted shares (shares jumps) or in market value net of the
           return (market-value jumps).
        2) In each jump month, the day with the largest absolute log step; days where
           either value is zero or missing have no step.
    Output:
        DataFrame [group_cols, eom, event_date].
    """

    def growth(c: str) -> pl.Expr:
        return pl.col(c) / pl.col(c).shift(1).over(group_cols)

    def log_step(ratio: pl.Expr) -> pl.Expr:
        # a zero or missing value on either day gives no step
        step = ratio.log().abs()
        return pl.when(step.is_finite()).then(step)

    return (
        daily.join(jumps.lazy().select(group_cols).unique(), on=group_cols, how="semi")
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
            shares_step=log_step(growth("adj")),
            value_step=log_step(growth("me") / growth("ri")),
        )
        .join(jumps.lazy(), on=[*group_cols, "eom"], how="inner", nulls_equal=True)
        .with_columns(
            step=pl.when(pl.col("jump") == _VALUE_JUMP)
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


def screen_stock_months(
    daily: pl.LazyFrame,
    monthly: pl.LazyFrame,
    group_cols: list[str] | None = None,
    country_col: str = "excntry",
    pair_max_months: int = JUMP_PAIR_MAX_MONTHS,
) -> pl.DataFrame:
    """
    Description:
        The stock-month screen of the Compustat correction: a keep decision and a shares
        multiplier for every security-month of daily and monthly, so daily and monthly
        rows follow one decision (apply_screen).
    Steps:
        1) Load the monthly panel (one row per security-month) and the security-level
           low-volume and zero-AJEXDI removals.
        2) screen_all_securities, one numba pass per security: low volume, zero AJEXDI,
           QUNIT changes, low price or market value (months after the breach month),
           gaps, early jumps (the history through the last one is dropped), later jumps
           (repaired), return/market-value mismatches, initial ratio errors.
        3) Spread the decisions over every security-month of daily and monthly
           (_spread_to_all_months).
    Output:
        DataFrame [group_cols, eom, keep, factor, factor_prev, event_date]. factor applies
        to shares and market value from event_date on, factor_prev before it.
    """
    group_cols = group_cols or ["gvkey", "iid"]
    panel = _load_screen_panel(monthly, group_cols, country_col)
    removals = _security_removals(daily, panel, group_cols)
    starts = _group_starts(panel, group_cols)
    n, n_sec = panel.height, len(starts) - 1
    status = np.zeros(n, dtype=np.int8)
    factor = np.ones(n, dtype=np.float64)
    jump = np.zeros(n, dtype=np.int8)
    dropped = np.zeros(n_sec, dtype=np.bool_)
    early_cut = np.full(n_sec, -1, dtype=np.int64)
    breach = np.full(n_sec, -1, dtype=np.int64)
    screen_all_securities(
        starts=starts,
        status=status,
        factor=factor,
        jump=jump,
        dropped=dropped,
        early_cut=early_cut,
        breach=breach,
        low_volume=removals["low_volume"].to_numpy(),
        zero_ajexdi=removals["zero_ajexdi"].to_numpy(),
        prc=panel["prc"].to_numpy(),
        me=panel["me"].to_numpy(),
        ri=panel["ri"].to_numpy(),
        cshoc=panel["cshoc"].to_numpy(),
        ajexdi=panel["ajexdi"].to_numpy(),
        qunit=panel["qunit"].to_numpy(),
        currency=panel["currency"].to_numpy(),
        month=panel["month"].to_numpy(),
        low_price_country=panel["low_price_country"].to_numpy(),
        is_china=panel["is_china"].to_numpy(),
        max_step=FILTER_GAP_MONTHS + 1,
        early_months=float(EARLY_JUMP_MONTHS),
        early_share=EARLY_JUMP_SHARE,
        pair_max_months=pair_max_months,
        qunit_max_change=QUNIT_PRICE_CHANGE_BOUND,
    )
    screened = panel.select(*group_cols, "eom").with_columns(
        status=pl.Series(status), factor=pl.Series(factor), jump=pl.Series(jump)
    )
    security_cuts = removals.select(group_cols).with_columns(
        dropped=pl.Series(dropped),
        early_cut=pl.Series(early_cut).replace(-1, None),
        breach=pl.Series(breach).replace(-1, None),
    )
    return _spread_to_all_months(daily, screened, security_cuts, group_cols)


def _spread_to_all_months(
    daily: pl.LazyFrame,
    screened: pl.DataFrame,
    security_cuts: pl.DataFrame,
    group_cols: list[str],
) -> pl.DataFrame:
    """
    Description:
        Turn the screen's panel rows and security-level cuts into one decision for every
        security-month of the daily data and the panel. Months without a panel row
        (daily rows with no valid month-end) follow the security-level cuts and the
        multiplier of the nearest kept month, the previous one where there is one.
    Steps:
        1) Kept panel months: their multiplier and the previous kept month's multiplier.
        2) Place each repaired jump on a trading day (_jump_days).
        3) For every security-month: keep unless the security is dropped, the panel
           month is dropped, the month is at or before the early cut, or after the
           breach month; the multiplier is the last kept month's, or the first kept
           month's before it (1 if none).
    Output:
        DataFrame [group_cols, eom, keep, factor, factor_prev, event_date].
    """
    keys = [*group_cols, "eom"]
    kept = screened.filter(pl.col("status") == _KEPT).with_columns(
        factor_prev=pl.col("factor").shift(1).over(group_cols).fill_null(1.0)
    )
    jumps = kept.filter(pl.col("jump") > 0).select(*keys, "jump")
    dropped_months = screened.filter(pl.col("status") != _KEPT).select(
        *keys, _dropped_month=pl.lit(True)
    )
    months = pl.concat([daily.select(*keys).unique().collect(), screened.select(keys)]).unique()
    return (
        months.join(kept.drop("status", "jump"), on=keys, how="left", nulls_equal=True)
        .join(_jump_days(daily, jumps, group_cols), on=keys, how="left", nulls_equal=True)
        .join(dropped_months, on=keys, how="left", nulls_equal=True)
        .join(security_cuts, on=group_cols, how="left", nulls_equal=True)
        .sort(keys)
        .with_columns(
            month=_month_number(pl.col("eom")),
            _nearest_factor=pl.col("factor")
            .forward_fill()
            .backward_fill()
            .over(group_cols)
            .fill_null(1.0),
        )
        .select(
            *keys,
            keep=~pl.col("dropped").fill_null(False)
            & ~pl.col("_dropped_month").fill_null(False)
            & (pl.col("early_cut").is_null() | (pl.col("month") > pl.col("early_cut")))
            & (pl.col("breach").is_null() | (pl.col("month") <= pl.col("breach"))),
            factor=pl.col("_nearest_factor"),
            factor_prev=pl.coalesce("factor_prev", "_nearest_factor"),
            event_date="event_date",
        )
    )


def apply_screen(
    df: pl.LazyFrame, decisions: pl.LazyFrame, group_cols: list[str] | None = None
) -> pl.LazyFrame:
    """
    Description:
        Apply the security-month decisions of screen_stock_months to daily or monthly
        rows.
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


# Decimal-correction kernels: per-security loops over date-sorted array slices, parallelized
# with prange over group boundaries; kernels mutate preallocated arrays in place.
#
# Detection-kernel outputs: factor = correction multiplier (1.0 = clean),
# wsize = detection window (-1 = null), ep_l/ep_r = clean endpoints (used by
# interpolation).
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


# Stock-month screen kernels. Each works on one security's month-sorted slice, in
# the order of _screen_one_security, and reads only the rows still kept when it starts.


@njit(cache=True, error_model="numpy")
def _drop_security(status: np.ndarray) -> None:
    """
    Description:
        Drop the whole security.
    Steps:
        1) Mark every row still kept as dropped.
    Output:
        status updated in place.
    """
    for i in range(status.size):
        if status[i] == _KEPT:
            status[i] = _DROPPED


@njit(cache=True, error_model="numpy")
def _kept_rows(status: np.ndarray) -> np.ndarray:
    """
    Description:
        The rows the next screen step sees.
    Steps:
        1) Positions of the rows still kept.
    Output:
        Array of row positions, in month order.
    """
    return np.flatnonzero(status == _KEPT)


@njit(cache=True, error_model="numpy")
def _jump_direction(
    x_prev: float,
    x_cur: float,
    confirm_prev: float,
    confirm_cur: float,
    return_based: bool,
    is_china: bool,
) -> int:
    """
    Description:
        Whether x jumps between two months, confirmed by a second series.
    Steps:
        1) Shares (return_based False): x up 5x with market value (confirm) up 2.5x
           (50x and 25x in China), or x down to 0.2x with market value down to 0.4x.
        2) Market value (True): x up 10x with a return from the return index (confirm)
           below 200%, or x down to 0.1x with a return above -50%.
    Output:
        +1 / -1 for an up / down jump, 0 otherwise; NaN operands fail every test.
    """
    jump = x_cur / x_prev
    if return_based:
        ret = confirm_cur / confirm_prev - 1.0
        if jump > 10.0 and ret < 2.0:
            return 1
        if jump < 0.1 and ret > -0.5:
            return -1
        return 0
    confirm = confirm_cur / confirm_prev
    if is_china:
        up = jump >= 50.0 and confirm >= 25.0
    else:
        up = jump >= 5.0 and confirm >= 2.5
    if up:
        return 1
    if jump <= 0.2 and confirm <= 0.4:
        return -1
    return 0


@njit(cache=True, error_model="numpy")
def _drop_early_history(
    status: np.ndarray,
    adj: np.ndarray,
    me: np.ndarray,
    ri: np.ndarray,
    is_china: np.ndarray,
    early_months: float,
    early_share: float,
) -> int:
    """
    Description:
        Drop the history through the last jump early in the security's life.
    Steps:
        1) The window is the lesser of early_months and early_share of the kept months.
        2) Within it, find the last month with a shares jump (split-adjusted shares adj
           against market value me) or a market-value jump (me against the return
           index ri).
        3) Drop the kept rows through that month.
    Output:
        status updated in place. Returns the row of the last early jump, -1 if none.
    """
    rows = _kept_rows(status)
    window = min(early_months, early_share * rows.size)
    last = -1
    for k in range(1, min(rows.size, math.ceil(window))):
        i, p = rows[k], rows[k - 1]
        if (
            _jump_direction(adj[p], adj[i], me[p], me[i], False, is_china[i]) != 0
            or _jump_direction(me[p], me[i], ri[p], ri[i], True, is_china[i]) != 0
        ):
            last = k
    for k in range(last + 1):
        status[rows[k]] = _DROPPED
    return rows[last] if last >= 0 else -1


@njit(cache=True, error_model="numpy")
def _rescale_across_jump(
    rows: np.ndarray,
    x: np.ndarray,
    confirm: np.ndarray,
    return_based: bool,
    factor: np.ndarray,
    k: int,
    direction: int,
) -> None:
    """
    Description:
        Remove an unpaired jump at kept position k.
    Steps:
        1) Up-jump: scale x from k on down to the level before the jump.
        2) Down-jump: scale x before k down to the level at k.
        3) Market value (return_based) keeps the month's return across the jump.
    Output:
        x and factor multiplied in place by the same amount.
    """
    i, p = rows[k], rows[k - 1]
    growth = confirm[i] / confirm[p] if return_based else 1.0
    if direction == 1:
        scale = x[p] * growth / x[i]
        lo, hi = k, rows.size
    else:
        scale = x[i] / (x[p] * growth)
        lo, hi = 0, k
    for r in range(lo, hi):
        u = rows[r]
        x[u] *= scale
        factor[u] *= scale


@njit(cache=True, error_model="numpy")
def _find_reversal(
    rows: np.ndarray,
    x: np.ndarray,
    confirm: np.ndarray,
    return_based: bool,
    is_china: np.ndarray,
    month: np.ndarray,
    pair_max_months: int,
    k: int,
    direction: int,
) -> int:
    """
    Description:
        The opposite jump that pairs with the jump at kept position k.
    Steps:
        1) Walk the kept months after k, up to pair_max_months after the jump month.
           A step where either month has no positive x is skipped, so a reversal
           across such a month is not seen.
        2) Stop at the first jump: in the opposite direction it is the reversal, in the
           same direction it ends the search.
    Output:
        Kept position of the reversal, -1 if none.
    """
    jump_month = month[rows[k]]
    for q in range(k + 1, rows.size):
        cur, prev = rows[q], rows[q - 1]
        if month[cur] - jump_month > pair_max_months:
            break
        if not (x[prev] > 0.0 and x[cur] > 0.0):
            continue
        found = _jump_direction(
            x[prev], x[cur], confirm[prev], confirm[cur], return_based, is_china[cur]
        )
        if found == -direction:
            return q
        if found == direction:
            break
    return -1


@njit(cache=True, error_model="numpy")
def _repair_pair(
    rows: np.ndarray,
    x: np.ndarray,
    confirm: np.ndarray,
    return_based: bool,
    is_china: np.ndarray,
    factor: np.ndarray,
    k: int,
    reversal: int,
) -> None:
    """
    Description:
        Repair a jump at kept position k together with its reversal at kept position
        reversal.
    Steps:
        1) Bring the months from k to before the reversal to the lower of the two ends.
           Shares get that level flat; market value is scaled by one multiplier, so the
           window keeps its returns, each end net of its month's return.
        2) Repair a jump left at the other end as an unpaired jump. Shares test the
           step from before the jump to the reversal, as market value is not yet
           repaired; market value tests the one-month step at that end with its own
           return.
    Output:
        x and factor rescaled in place by the same multiplier.
    """
    # i: the jump month, p: the month before it; j: the reversal month, jp: the month before it
    i, p = rows[k], rows[k - 1]
    j, jp = rows[reversal], rows[reversal - 1]
    if return_based:
        to_before = x[p] * (confirm[i] / confirm[p]) / x[i]
        to_after = x[j] / (confirm[j] / confirm[jp]) / x[jp]
        scale = min(to_before, to_after)
        for r in range(k, reversal):
            u = rows[r]
            x[u] *= scale
            factor[u] *= scale
        # the window now meets the end whose multiplier was the smaller; any step is
        # left at the other end, tested with that month's return
        side = reversal if to_before <= to_after else k
        a, b = rows[side - 1], rows[side]
        leftover = _jump_direction(x[a], x[b], confirm[a], confirm[b], True, is_china[b])
    else:
        level = min(x[p], x[j])
        for r in range(k, reversal):
            u = rows[r]
            factor[u] *= level / x[u]
            x[u] = level
        # the window sits at the lower end, any step is left at the higher one; market
        # value is not yet repaired, so the step is tested end to end
        side = reversal if x[p] <= x[j] else k
        leftover = _jump_direction(x[p], x[j], confirm[p], confirm[j], False, is_china[j])
    if leftover != 0:
        _rescale_across_jump(rows, x, confirm, return_based, factor, side, leftover)


@njit(cache=True, error_model="numpy")
def _repair_later_jumps(
    status: np.ndarray,
    x: np.ndarray,
    confirm: np.ndarray,
    return_based: bool,
    is_china: np.ndarray,
    month: np.ndarray,
    pair_max_months: int,
    factor: np.ndarray,
    jump: np.ndarray,
    kind: int,
) -> None:
    """
    Description:
        Repair the jumps left after the early cut in x: split-adjusted shares tested
        against market value (return_based False), or market value tested against the
        return index (True).
    Steps:
        1) Walk the kept months; skip a step where either month has no positive x.
        2) At a jump, look for its reversal (_find_reversal).
        3) Repair it.
           - Paired: the jump and its reversal together (_repair_pair).
           - Unpaired up-jump: x from the jump on is scaled down to the level before it.
           - Unpaired down-jump: x before the jump is scaled down to the level after it.
    Output:
        x and factor rescaled in place by the same multiplier; each repaired jump month
        (both ends of a pair) marked kind in jump.
    """
    rows = _kept_rows(status)
    k = 1
    while k < rows.size:
        # i: the jump month, p: the month before it
        i, p = rows[k], rows[k - 1]
        direction = 0
        if x[p] > 0.0 and x[i] > 0.0:
            direction = _jump_direction(
                x[p], x[i], confirm[p], confirm[i], return_based, is_china[i]
            )
        if direction == 0:
            k += 1
            continue
        jump[i] = kind
        reversal = _find_reversal(
            rows, x, confirm, return_based, is_china, month, pair_max_months, k, direction
        )
        if reversal < 0:
            _rescale_across_jump(rows, x, confirm, return_based, factor, k, direction)
            k += 1
        else:
            _repair_pair(rows, x, confirm, return_based, is_china, factor, k, reversal)
            jump[rows[reversal]] = kind
            k = reversal + 1


@njit(cache=True, error_model="numpy")
def _drop_on_qunit_change(
    status: np.ndarray,
    qunit: np.ndarray,
    currency: np.ndarray,
    prc: np.ndarray,
    ajexdi: np.ndarray,
    max_change: float,
) -> bool:
    """
    Description:
        Drop a security whose quote unit (QUNIT) changes with a large price move.
    Steps:
        1) Between consecutive kept months, a QUNIT change without a currency change.
        2) A split-adjusted price change beyond +-max_change at that step drops the
           security.
    Output:
        status updated in place. Returns whether the security was dropped.
    """
    rows = _kept_rows(status)
    for k in range(1, rows.size):
        i, p = rows[k], rows[k - 1]
        unit_changed = not np.isnan(qunit[i]) and not np.isnan(qunit[p]) and qunit[i] != qunit[p]
        same_currency = currency[i] > 0 and currency[i] == currency[p]
        if unit_changed and same_currency:
            change = (prc[i] / ajexdi[i]) / (prc[p] / ajexdi[p]) - 1.0
            if change > max_change or change < -max_change:
                _drop_security(status)
                return True
    return False


@njit(cache=True, error_model="numpy")
def _drop_after_low_price_or_me(
    status: np.ndarray, prc: np.ndarray, me: np.ndarray, low_price_country: np.ndarray
) -> int:
    """
    Description:
        Cut the history at the first month-end with a price below the minimum ($0.01,
        or $0.001 in low_price_country) or a market value below $1M.
    Steps:
        1) A breach in the first kept month drops the security.
        2) A later breach drops the months after it; the breach month is kept.
    Output:
        status updated in place. Returns the breach row, -2 if the security was dropped,
        -1 if there is no breach.
    """
    rows = _kept_rows(status)
    for k in range(rows.size):
        i = rows[k]
        min_price = 0.001 if low_price_country[i] else 0.01
        if me[i] < 1.0 or prc[i] < min_price:
            if k == 0:
                _drop_security(status)
                return -2
            for r in range(k + 1, rows.size):
                status[rows[r]] = _DROPPED
            return i
    return -1


@njit(cache=True, error_model="numpy")
def _drop_month_after_gap(status: np.ndarray, month: np.ndarray, max_step: int) -> None:
    """
    Description:
        Drop the month after a gap.
    Steps:
        1) Between consecutive kept months, a step of more than max_step months drops
           the later month.
    Output:
        status updated in place.
    """
    prev = -1
    for i in range(status.size):
        if status[i] != _KEPT:
            continue
        if prev >= 0 and month[i] - month[prev] > max_step:
            status[i] = _DROPPED
        # prev advances past a dropped month too: each gap is measured between the
        # months kept when the step starts
        prev = i


@njit(cache=True, error_model="numpy")
def _drop_return_me_mismatches(status: np.ndarray, ri: np.ndarray, me: np.ndarray) -> None:
    """
    Description:
        Drop months whose return is not reflected in market value.
    Steps:
        1) Between consecutive kept months, an absolute return above 80% with an
           absolute market-value change below 50% drops the later month.
    Output:
        status updated in place.
    """
    prev = -1
    for i in range(status.size):
        if status[i] != _KEPT:
            continue
        if prev >= 0:
            ret = ri[i] / ri[prev] - 1.0
            me_chg = me[i] / me[prev] - 1.0
            if abs(ret) > 0.8 and abs(me_chg) < 0.5:
                status[i] = _DROPPED
        # a dropped month still advances prev, staying the reference for the next month
        prev = i


@njit(cache=True, error_model="numpy")
def _drop_initial_ratio_errors(status: np.ndarray, adj_prc: np.ndarray, me: np.ndarray) -> None:
    """
    Description:
        Drop implausible moves among the first kept months.
    Steps:
        1) Among the first 3 kept months, a split-adjusted price or market-value ratio
           to the month before outside [0.1, 10] drops the month.
    Output:
        status updated in place.
    """
    prev = -1
    obs = -1
    for i in range(status.size):
        if status[i] != _KEPT:
            continue
        obs += 1
        if obs > 2:
            break
        if prev >= 0:
            pr = adj_prc[i] / adj_prc[prev]
            mr = me[i] / me[prev]
            if pr > 10.0 or pr < 0.1 or mr > 10.0 or mr < 0.1:
                status[i] = _DROPPED
        prev = i  # a dropped month still advances prev, as in _drop_return_me_mismatches


@njit(cache=True, error_model="numpy")
def _screen_one_security(
    status: np.ndarray,
    factor: np.ndarray,
    jump: np.ndarray,
    low_volume: bool,
    zero_ajexdi: bool,
    prc: np.ndarray,
    me: np.ndarray,
    ri: np.ndarray,
    cshoc: np.ndarray,
    ajexdi: np.ndarray,
    qunit: np.ndarray,
    currency: np.ndarray,
    month: np.ndarray,
    low_price_country: np.ndarray,
    is_china: np.ndarray,
    max_step: int,
    early_months: float,
    early_share: float,
    pair_max_months: int,
    qunit_max_change: float,
) -> tuple[bool, int, int]:
    """
    Description:
        Screen one security's month-sorted slice.
    Steps:
        1) Drop the security for low volume, a zero AJEXDI or a QUNIT change, or when
           its first month breaches the minimum price or market value.
        2) Drop the months after a later breach and the month after each gap.
        3) Hold the breach month out of the jump steps.
        4) Drop the history through the last early jump, then repair later shares
           jumps and, on market value rebuilt with the shares multiplier, later
           market-value jumps.
        5) Restore the breach month with the multiplier of the month before it.
        6) Drop return/market-value mismatches and initial ratio errors.
    Output:
        status, factor and jump filled in place. Returns (dropped, early_cut, breach):
        early_cut is the row of the last early jump (rows through it dropped), breach the
        breach row (rows after it dropped), -1 if none.
    """
    if low_volume or zero_ajexdi:
        _drop_security(status)
        return True, -1, -1
    if _drop_on_qunit_change(status, qunit, currency, prc, ajexdi, qunit_max_change):
        return True, -1, -1
    breach = _drop_after_low_price_or_me(status, prc, me, low_price_country)
    if breach == -2:
        return True, -1, -1
    _drop_month_after_gap(status, month, max_step)
    # the breach month only carries its return: held out of the jump steps, so its drop
    # never rescales the history before it
    held_out = breach >= 0 and status[breach] == _KEPT
    if held_out:
        status[breach] = _HELD_OUT
    # one early window for both kinds of jump, then the later jumps are repaired: shares
    # (split-adjusted shares against market value) first, then market value is rebuilt
    # and repaired against the return
    adj = cshoc * ajexdi
    early_cut = _drop_early_history(status, adj, me, ri, is_china, early_months, early_share)
    _repair_later_jumps(
        status, adj, me, False, is_china, month, pair_max_months, factor, jump, _SHARES_JUMP
    )
    value = me * factor
    _repair_later_jumps(
        status, value, ri, True, is_china, month, pair_max_months, factor, jump, _VALUE_JUMP
    )
    if held_out:
        status[breach] = _KEPT
        # the breach month follows the multiplier of the month before it
        k = breach - 1
        while k >= 0 and status[k] != _KEPT:
            k -= 1
        if k >= 0:
            factor[breach] = factor[k]
            value[breach] = me[breach] * factor[k]
    _drop_return_me_mismatches(status, ri, value)
    _drop_initial_ratio_errors(status, prc / ajexdi, value)
    return False, early_cut, breach


@njit(parallel=True, cache=True, error_model="numpy")
def screen_all_securities(
    starts: np.ndarray,
    status: np.ndarray,
    factor: np.ndarray,
    jump: np.ndarray,
    dropped: np.ndarray,
    early_cut: np.ndarray,
    breach: np.ndarray,
    low_volume: np.ndarray,
    zero_ajexdi: np.ndarray,
    prc: np.ndarray,
    me: np.ndarray,
    ri: np.ndarray,
    cshoc: np.ndarray,
    ajexdi: np.ndarray,
    qunit: np.ndarray,
    currency: np.ndarray,
    month: np.ndarray,
    low_price_country: np.ndarray,
    is_china: np.ndarray,
    max_step: int,
    early_months: float,
    early_share: float,
    pair_max_months: int,
    qunit_max_change: float,
) -> None:
    """
    Description:
        Screen every security in parallel (_screen_one_security on each slice).
    Steps:
        1) For each security, the rows starts[g]:starts[g + 1].
        2) Store its result, with the two cut rows converted to month numbers.
    Output:
        status, factor and jump filled in place per row; dropped, early_cut and breach
        per security (cuts as month numbers, -1 if none).
    """
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        is_dropped, cut_row, breach_row = _screen_one_security(
            status[s:e],
            factor[s:e],
            jump[s:e],
            low_volume[g],
            zero_ajexdi[g],
            prc[s:e],
            me[s:e],
            ri[s:e],
            cshoc[s:e],
            ajexdi[s:e],
            qunit[s:e],
            currency[s:e],
            month[s:e],
            low_price_country[s:e],
            is_china[s:e],
            max_step,
            early_months,
            early_share,
            pair_max_months,
            qunit_max_change,
        )
        dropped[g] = is_dropped
        early_cut[g] = month[s + cut_row] if cut_row >= 0 else -1
        breach[g] = month[s + breach_row] if breach_row >= 0 else -1
