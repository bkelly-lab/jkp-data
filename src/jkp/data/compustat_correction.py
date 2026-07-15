"""
Compustat return corrections.

Repairs decimal-shift price errors and drops unreliable daily observations in
the Compustat security files before returns are computed, following the data
corrections of Bessembinder et al. (2023).

Layering: the public `correct_decimal_errors` / `drop_unreliable_observations`
functions are domain-level orchestrators (sort -> build typed inputs -> run ->
reattach). A dedicated adapter converts the Polars panel to NumPy once, the
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
    FILTER_GAP_TRADING_DAYS,
    LOW_PRICE_COUNTRIES,
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


@dataclass(frozen=True)
class FilterInputs:
    """Typed NumPy view of one filter panel, ready for `run_all_filters`."""

    starts: np.ndarray  # group-boundary offsets
    remove_low_volume: np.ndarray  # per-security bottom-2%-volume decision
    ajexdi: np.ndarray
    prc: np.ndarray
    me: np.ndarray
    ri: np.ndarray
    cshoc: np.ndarray
    dates: np.ndarray  # physical int32 days
    low: np.ndarray  # $0.001 price-threshold country flag
    chn: np.ndarray  # China flag (looser jump bounds)
    gap_days: int  # data-gap calendar-day bound


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


def _sort_to_spill(df: pl.LazyFrame, sort_keys: list[str], path: Path) -> None:
    """Streaming-sort the input by sort_keys to a spill parquet."""
    df.sort(sort_keys).sink_parquet(path, compression=CORRECTION_SPILL_COMPRESSION)


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
    sorted_path: Path, group_cols: list[str], sort_col: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Collect the correction value columns as f64; return (data, starts)."""
    lf = pl.scan_parquet(sorted_path)
    schema_names = lf.collect_schema().names()
    value_cols = [
        v for v in ("trfd", "qunit", "adrrc", "ajexdi", "prccd", "cshoc") if v in schema_names
    ]
    data = lf.select(
        group_cols + [sort_col] + [pl.col(c).cast(pl.Float64) for c in value_cols]
    ).collect()
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
    sorted_path: Path, corrected: dict[str, np.ndarray], spill_dir: Path
) -> pl.LazyFrame:
    """Splice corrected columns over the originals (which move to the end)."""
    corr_path = spill_dir / "__corr_corrected_cols.parquet"
    pl.DataFrame(corrected).with_columns(
        [pl.col(name).fill_nan(None) for name in corrected]
    ).write_parquet(corr_path, compression=CORRECTION_SPILL_COMPRESSION)
    return pl.concat(
        [pl.scan_parquet(sorted_path).drop(list(corrected)), pl.scan_parquet(corr_path)],
        how="horizontal",
    )


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
        and reconstruct PRCCD/CSHOC. Memory-bounded array path using spill
        files in spill_dir (required).
    Args:
        correction_method: 'multiplier' (fixed 10x/100x/1000x multipliers),
            'interpolation' (endpoint geometric mean), 'floor' (multipliers,
            price divide-direction on >=$1 only), 'floor_interp' (both).
        has_adrrc: correct the ADRRC column (NA data only).
        variation_threshold: multi-period interior-variation bound.
    Output:
        Corrected LazyFrame (corrected columns move to the end; NaN -> null).
    """
    if correction_method not in DECIMAL_CORRECTION_METHODS:
        raise ValueError(
            f"Unknown correction_method: {correction_method!r}; expected one of {DECIMAL_CORRECTION_METHODS}"
        )
    if spill_dir is None:
        raise ValueError("correct_decimal_errors requires spill_dir")
    group_cols = group_cols or ["gvkey", "iid"]
    window_sizes = window_sizes if window_sizes is not None else DECIMAL_DETECTION_WINDOWS

    sorted_path = spill_dir / "__corr_sorted.parquet"
    _sort_to_spill(df, group_cols + [sort_col], sorted_path)
    data, starts = _load_correction_inputs(sorted_path, group_cols, sort_col)
    corrected = _run_decimal_correction(
        data, starts, window_sizes, has_adrrc, correction_method, variation_threshold
    )
    return _reattach_corrected(sorted_path, corrected, spill_dir)


# Observation filters
def _load_filter_inputs(
    sorted_path: Path, group_cols: list[str], sort_col: str, country_col: str
) -> pl.DataFrame:
    """Collect kernel inputs in one pass: f64 values, int32 date, country flags."""
    value_cols = ["ajexdi", "prc", "me", "ri", "cshoc", "dolvol"]
    return (
        pl.scan_parquet(sorted_path)
        .select(
            group_cols
            + [sort_col]
            + [pl.col(c).cast(pl.Float64) for c in value_cols]
            + [
                # named to match the FilterInputs fields (see _build_filter_inputs)
                pl.col(sort_col).cast(pl.Date).to_physical().cast(pl.Int32).alias("dates"),
                pl.col(country_col).is_in(LOW_PRICE_COUNTRIES).fill_null(False).alias("low"),
                (pl.col(country_col) == "CHN").fill_null(False).alias("chn"),
            ]
        )
        .collect()
    )


def _low_volume_decision(data: pl.DataFrame, group_cols: list[str]) -> np.ndarray:
    """
    Global cross-security low-volume decision: per-security mean of positive
    volume, dropping the bottom 2% (scalar quantile, nulls ignored). Securities
    with no positive volume (null mean) are KEPT. One row per group, in
    group_starts (sorted) order.
    """
    avg_vol = (
        data.lazy()
        .filter(pl.col("dolvol") > 0)
        .group_by(group_cols)
        .agg(pl.mean("dolvol").alias("_avg_vol"))
        .collect()
    )
    cutoff = avg_vol["_avg_vol"].quantile(0.02)
    return (
        data.select(group_cols)
        .unique()
        .join(avg_vol, on=group_cols, how="left")
        .sort(group_cols)
        .get_column("_avg_vol")
        .le(cutoff)
        .fill_null(False)
        .to_numpy()
    )


def _build_filter_inputs(data: pl.DataFrame, group_cols: list[str]) -> FilterInputs:
    """Adapter: the collected filter panel -> typed NumPy kernel inputs. The
    per-row array columns are named to match the FilterInputs fields."""
    array_fields = ("ajexdi", "prc", "me", "ri", "cshoc", "dates", "low", "chn")
    return FilterInputs(
        starts=_group_starts(data, group_cols),
        remove_low_volume=_low_volume_decision(data, group_cols),
        gap_days=int(FILTER_GAP_TRADING_DAYS * 365 / 252),  # trading -> calendar days
        **{f: data[f].to_numpy() for f in array_fields},
    )


def _verify_presorted_order(
    data: pl.DataFrame, group_cols: list[str], inp: FilterInputs, sorted_path: Path
) -> None:
    """Trusting an externally sorted spill: a stale or misordered file would
    silently corrupt the panel, so check groups are contiguous and dates rise."""
    n_groups = data.select(pl.struct(group_cols).n_unique()).item()
    if len(inp.starts) - 1 != n_groups:
        raise ValueError(f"presorted file {sorted_path}: groups are not contiguous")
    d = inp.dates
    interior = np.ones(len(d), dtype=np.bool_)
    interior[inp.starts[:-1]] = False  # group starts are exempt from the diff check
    idx = np.flatnonzero(interior)
    if not np.all(d[idx] >= d[idx - 1]):
        raise ValueError(f"presorted file {sorted_path}: dates not sorted within a group")


def _reattach_reason(sorted_path: Path, reason: np.ndarray, spill_dir: Path) -> pl.LazyFrame:
    """Attach the kernel's keep/remove decision and filter to the kept rows."""
    reason_path = spill_dir / "__corr_filter_reason.parquet"
    pl.DataFrame({"_reason": reason}).write_parquet(
        reason_path, compression=CORRECTION_SPILL_COMPRESSION
    )
    return (
        pl.concat([pl.scan_parquet(sorted_path), pl.scan_parquet(reason_path)], how="horizontal")
        .filter(pl.col("_reason") == 0)
        .drop("_reason")
    )


def drop_unreliable_observations(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    country_col: str = "excntry",
    spill_dir: Path | None = None,
    presorted_path: Path | None = None,
) -> pl.LazyFrame:
    """
    Description:
        Drop unreliable daily observations, implementing the observation filters
        of Section 8 of Bessembinder et al. (2023) Data Appendix. Runs the full
        filter chain as one numba pass per security, preserving
        sequential-survivor semantics. Memory-bounded array path.
    Steps:
        1) Streaming-sort input to a spill file (or verify presorted_path).
        2) Build typed NumPy inputs (kernel arrays + low-volume decision).
        3) Run run_all_filters -> per-row keep/remove; filter lazily.
    Note:
        With presorted_path, the panel is read from that file and df is NOT
        re-read — the caller must have written df's data there, sorted by
        group_cols + sort_col.
    Output:
        Filtered LazyFrame.
    """
    group_cols = group_cols or ["gvkey", "iid"]
    if spill_dir is None:
        raise ValueError("drop_unreliable_observations requires spill_dir")
    if presorted_path is not None and not presorted_path.exists():
        raise FileNotFoundError(f"presorted_path does not exist: {presorted_path}")

    sorted_path = presorted_path or (spill_dir / "__corr_filter_sorted.parquet")
    if presorted_path is None:
        _sort_to_spill(df, group_cols + [sort_col], sorted_path)

    data = _load_filter_inputs(sorted_path, group_cols, sort_col, country_col)
    inp = _build_filter_inputs(data, group_cols)
    if presorted_path is not None:
        _verify_presorted_order(data, group_cols, inp, sorted_path)

    reason = np.zeros(data.height, dtype=np.int8)
    run_all_filters(
        inp.starts,
        reason,
        inp.remove_low_volume,
        inp.ajexdi,
        inp.prc,
        inp.me,
        inp.ri,
        inp.cshoc,
        inp.dates,
        inp.low,
        inp.chn,
        inp.gap_days,
    )
    return _reattach_reason(sorted_path, reason, spill_dir)


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
def _early_jump_stage(reason, jump_series, confirm_series, return_based, chn):
    """
    Shared early-jump stage for the adjCSHO and ME filters. A jump in
    jump_series (confirmed by confirm_series) within the early period deletes
    alive rows 0..max-jump-obs. The return-based variant reads confirm_series as
    a return; the adjCSHO variant uses looser CHN bounds. NaN operands fail all
    tests (like polars).
    """
    n = reason.size
    total = 0
    for i in range(n):
        if reason[i] == 0:
            total += 1
    if total == 0:
        return
    delete_through = -1
    obs = -1
    prev = -1
    for i in range(n):
        if reason[i] != 0:
            continue
        obs += 1
        if prev >= 0:
            jump = jump_series[i] / jump_series[prev]
            if return_based:
                # ME jump not backed by a commensurate return
                ret = confirm_series[i] / confirm_series[prev] - 1.0
                up = jump > 10.0 and ret < 2.0
                down = jump < 0.1 and ret > -0.5
            else:
                # adjCSHO jump vs ME (CHN gets looser bounds)
                confirm = confirm_series[i] / confirm_series[prev]
                if chn[i]:
                    up = jump >= 50.0 and confirm >= 25.0
                else:
                    up = jump >= 5.0 and confirm >= 2.5
                down = jump <= 0.2 and confirm <= 0.4
            # early period: first ~24 months (obs < 504) or 20% of the history
            in_early_period = obs < 504.0 or obs < 0.2 * total
            if (up or down) and in_early_period:
                delete_through = obs
        prev = i
    if delete_through < 0:
        return
    obs = -1
    for i in range(n):
        if reason[i] != 0:
            continue
        obs += 1
        if obs <= delete_through:
            reason[i] = 1


# Each filter reads the rows still alive at its start (sequential survivor) and
# marks removals in `reason`. Those that can void the whole security (the
# low-volume, zero-AJEXDI, and initial low-price/ME filters) return True to stop
# the chain.


@njit(cache=True, error_model="numpy")
def _filter_low_volume(reason, remove_low_volume):
    """Drop the whole security if it is in the bottom 2% by average positive volume."""
    if remove_low_volume:
        _kill_all(reason)
        return True
    return False


@njit(cache=True, error_model="numpy")
def _filter_zero_ajexdi(reason, ajexdi):
    """Any AJEXDI == 0 removes the whole security."""
    for i in range(reason.size):
        if ajexdi[i] == 0.0:
            _kill_all(reason)
            return True
    return False


@njit(cache=True, error_model="numpy")
def _filter_low_price_or_me(reason, prc, me, dates, low_thr):
    """
    Price below threshold ($0.01, or $0.001 in low_thr countries) or ME < $1M.
    A breach on the first alive date voids the security; a later breach removes
    all history from the breach date onward.
    """
    n = reason.size
    first = -1
    for i in range(n):
        if reason[i] == 0:
            first = i
            break
    if first < 0:
        return True
    breach_idx = -1
    for i in range(first, n):
        if reason[i] != 0:
            continue
        thr = 0.001 if low_thr[i] else 0.01
        if me[i] < 1.0 or prc[i] < thr:
            breach_idx = i
            break
    if breach_idx < 0:
        return False
    initial = False
    for i in range(first, n):
        if reason[i] != 0 or dates[i] != dates[first]:
            break
        thr = 0.001 if low_thr[i] else 0.01
        if me[i] < 1.0 or prc[i] < thr:
            initial = True
            break
    if initial:
        _kill_all(reason)
        return True
    for i in range(n):
        if reason[i] == 0 and dates[i] >= dates[breach_idx]:
            reason[i] = 1
    return False


@njit(cache=True, error_model="numpy")
def _filter_data_gaps(reason, dates, gap_days):
    """Drop the observation after a calendar gap wider than gap_days."""
    prev = -1
    for i in range(reason.size):
        if reason[i] != 0:
            continue
        if prev >= 0 and dates[i] - dates[prev] > gap_days:
            reason[i] = 1
        # prev advances even past a removed row: polars measures gaps against
        # the pre-stage frame, so a removed row is still the next row's reference
        prev = i


@njit(cache=True, error_model="numpy")
def _filter_adjcsho_jumps(reason, cshoc, ajexdi, me, chn):
    """Early adjCSHO (= cshoc*ajexdi) jump not matched by ME (looser CHN bounds)."""
    _early_jump_stage(reason, cshoc * ajexdi, me, False, chn)


@njit(cache=True, error_model="numpy")
def _filter_me_jumps(reason, me, ri, chn):
    """Early ME jump not backed by a commensurate return (ri)."""
    _early_jump_stage(reason, me, ri, True, chn)


@njit(cache=True, error_model="numpy")
def _filter_return_me_mismatch(reason, ri, me):
    """|return| > 80% with |ME change| < 50%."""
    prev = -1
    for i in range(reason.size):
        if reason[i] != 0:
            continue
        if prev >= 0:
            ret = ri[i] / ri[prev] - 1.0
            me_chg = me[i] / me[prev] - 1.0
            if abs(ret) > 0.8 and abs(me_chg) < 0.5:
                reason[i] = 1
        # a flagged row still advances prev, staying the shift source for the
        # next row (matching the polars stage-input shift)
        prev = i


@njit(cache=True, error_model="numpy")
def _filter_initial_ratio_errors(reason, prc, me):
    """A price or ME ratio outside [0.1, 10] among the first 3 observations."""
    prev = -1
    obs = -1
    for i in range(reason.size):
        if reason[i] != 0:
            continue
        obs += 1
        if obs > 2.0:  # only the first 3 observations (0-indexed)
            break
        if prev >= 0:
            pr = prc[i] / prc[prev]
            mr = me[i] / me[prev]
            if pr > 10.0 or pr < 0.1 or mr > 10.0 or mr < 0.1:
                reason[i] = 1
        prev = i  # flagged rows still advance prev (see _filter_return_me_mismatch)


@njit(cache=True, error_model="numpy")
def _run_filters_one_security(
    reason, remove_low_volume, ajexdi, prc, me, ri, cshoc, dates, low_thr, chn, gap_days
):
    """Run the filter chain in order on one security's date-sorted slice."""
    if _filter_low_volume(reason, remove_low_volume):
        return
    if _filter_zero_ajexdi(reason, ajexdi):
        return
    if _filter_low_price_or_me(reason, prc, me, dates, low_thr):
        return
    _filter_data_gaps(reason, dates, gap_days)
    _filter_adjcsho_jumps(reason, cshoc, ajexdi, me, chn)
    _filter_me_jumps(reason, me, ri, chn)
    _filter_return_me_mismatch(reason, ri, me)
    _filter_initial_ratio_errors(reason, prc, me)


@njit(parallel=True, cache=True, error_model="numpy")
def run_all_filters(
    starts, reason, remove_low_volume, ajexdi, prc, me, ri, cshoc, dates, low_thr, chn, gap_days
):
    """Run the filter chain over all securities in parallel."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        _run_filters_one_security(
            reason[s:e],
            remove_low_volume[g],
            ajexdi[s:e],
            prc[s:e],
            me[s:e],
            ri[s:e],
            cshoc[s:e],
            dates[s:e],
            low_thr[s:e],
            chn[s:e],
            gap_days,
        )
