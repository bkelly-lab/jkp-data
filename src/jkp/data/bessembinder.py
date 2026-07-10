"""
Bessembinder et al. (2023) Data Corrections for Compustat

Implements the Section 6 decimal-error corrections and Section 8 filters from
Bessembinder et al. (2023) "Do Global Stocks Outperform US Treasury Bills?"
Data Appendix.

Layering: the public `apply_bessembinder_section6/8` functions are domain-level
orchestrators (sort -> build typed inputs -> run -> reattach). A dedicated
adapter converts the Polars panel to NumPy once, the numba kernels
(bessembinder_kernels) do the per-security work, and the result is reattached
lazily. NumPy is only the kernel boundary, not the whole design.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import bessembinder_kernels as bk
from .config import (
    BESS_DEFAULT_WINDOWS,
    BESS_LOW_PRICE_COUNTRIES,
    BESS_SECTION6_METHODS,
    BESS_SECTION8_GAP_TRADING_DAYS,
    BESS_SPILL_COMPRESSION,
)


@dataclass(frozen=True)
class Section6State:
    """Per-variable detection state at the Section 6 kernel boundary. The
    kernels mutate factor/window_size/endpoints in place."""

    values: np.ndarray  # zero-nulled working copy (feeds detection and output)
    factor: np.ndarray  # correction multiplier, 1.0 = clean
    window_size: np.ndarray  # detection window, -1 = null
    left_endpoint: np.ndarray  # clean value to the left of a flagged spike
    right_endpoint: np.ndarray  # clean value to the right


@dataclass(frozen=True)
class Section8Inputs:
    """Typed NumPy view of one Section 8 panel, ready for `section8_all`."""

    starts: np.ndarray  # group-boundary offsets
    remove_8a: np.ndarray  # per-security 8a (bottom-2% volume) decision
    ajexdi: np.ndarray
    prc: np.ndarray
    me: np.ndarray
    ri: np.ndarray
    cshoc: np.ndarray
    dates: np.ndarray  # physical int32 days
    low: np.ndarray  # $0.001 price-threshold country flag (8c)
    chn: np.ndarray  # China flag (looser 8e bounds)
    gap_days: int  # filter 8d calendar-day gap bound


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
    df.sort(sort_keys).sink_parquet(path, compression=BESS_SPILL_COMPRESSION)


# Section 6: decimal-error corrections
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
    state = Section6State(
        values=x,
        factor=np.ones(n, dtype=np.float64),
        window_size=np.full(n, -1, dtype=np.int32),
        left_endpoint=np.full(n, np.nan, dtype=np.float64),
        right_endpoint=np.full(n, np.nan, dtype=np.float64),
    )

    bk.detect_single_period_all(
        state.values, starts, state.factor, state.left_endpoint, state.right_endpoint
    )
    state.window_size[state.factor != 1.0] = 1
    nlags = np.array(sorted(w for w in window_sizes if w > 1), dtype=np.int64)
    if len(nlags) > 0:
        bk.detect_multi_period_all(
            state.values,
            starts,
            state.factor,
            state.window_size,
            state.left_endpoint,
            state.right_endpoint,
            nlags,
            variation_threshold,
        )
    bk.validate_cascading_all(starts, state.factor, state.window_size)

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


def _load_section6(
    sorted_path: Path, group_cols: list[str], sort_col: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Collect the Section 6 value columns as f64; return (data, starts)."""
    lf = pl.scan_parquet(sorted_path)
    schema_names = lf.collect_schema().names()
    value_cols = [
        v for v in ("trfd", "qunit", "adrrc", "ajexdi", "prccd", "cshoc") if v in schema_names
    ]
    data = lf.select(
        group_cols + [sort_col] + [pl.col(c).cast(pl.Float64) for c in value_cols]
    ).collect()
    return data, _group_starts(data, group_cols)


def _run_section6(
    data: pl.DataFrame,
    starts: np.ndarray,
    window_sizes: list[int],
    has_adrrc: bool,
    correction_method: str,
    variation_threshold: float,
) -> dict[str, np.ndarray]:
    """Correct each variable (per Section 6c) and return {name: corrected array}."""
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
    corr_path = spill_dir / "__bess_corrected_cols.parquet"
    pl.DataFrame(corrected).with_columns(
        [pl.col(name).fill_nan(None) for name in corrected]
    ).write_parquet(corr_path, compression=BESS_SPILL_COMPRESSION)
    return pl.concat(
        [pl.scan_parquet(sorted_path).drop(list(corrected)), pl.scan_parquet(corr_path)],
        how="horizontal",
    )


def apply_bessembinder_section6(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    window_sizes: list[int] | None = None,
    has_adrrc: bool = False,
    correction_method: str = "bessembinder",
    spill_dir: Path | None = None,
    variation_threshold: float = 1.3,
) -> pl.LazyFrame:
    """
    Description:
        Apply Section 6 corrections (per 6c): correct TRFD, QUNIT and (NA only)
        ADRRC independently, then adjPRC = PRCCD/AJEXDI and adjCSHO =
        CSHOC*AJEXDI, and reconstruct PRCCD/CSHOC. Memory-bounded array path
        using spill files in spill_dir (required).
    Args:
        correction_method: 'bessembinder' (fixed 10x/100x/1000x multipliers),
            'interpolation' (endpoint geometric mean), 'floor' (multipliers,
            price divide-direction on >=$1 only), 'floor_interp' (both).
        has_adrrc: correct the ADRRC column (NA data only).
        variation_threshold: multi-period interior-variation bound.
    Output:
        Corrected LazyFrame (corrected columns move to the end; NaN -> null).
    """
    if correction_method not in BESS_SECTION6_METHODS:
        raise ValueError(
            f"Unknown correction_method: {correction_method!r}; expected one of {BESS_SECTION6_METHODS}"
        )
    if spill_dir is None:
        raise ValueError("apply_bessembinder_section6 requires spill_dir")
    group_cols = group_cols or ["gvkey", "iid"]
    window_sizes = window_sizes if window_sizes is not None else BESS_DEFAULT_WINDOWS

    sorted_path = spill_dir / "__bess_sorted.parquet"
    _sort_to_spill(df, group_cols + [sort_col], sorted_path)
    data, starts = _load_section6(sorted_path, group_cols, sort_col)
    corrected = _run_section6(
        data, starts, window_sizes, has_adrrc, correction_method, variation_threshold
    )
    return _reattach_corrected(sorted_path, corrected, spill_dir)


# Section 8: additional filters
def _load_section8(
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
                # named to match the Section8Inputs fields (see _section8_inputs)
                pl.col(sort_col).cast(pl.Date).to_physical().cast(pl.Int32).alias("dates"),
                pl.col(country_col).is_in(BESS_LOW_PRICE_COUNTRIES).fill_null(False).alias("low"),
                (pl.col(country_col) == "CHN").fill_null(False).alias("chn"),
            ]
        )
        .collect()
    )


def _filter_8a_decision(data: pl.DataFrame, group_cols: list[str]) -> np.ndarray:
    """
    Filter 8a's global cross-security decision: per-security mean of positive
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


def _section8_inputs(data: pl.DataFrame, group_cols: list[str]) -> Section8Inputs:
    """Adapter: the collected Section 8 panel -> typed NumPy kernel inputs. The
    per-row array columns are named to match the Section8Inputs fields."""
    array_fields = ("ajexdi", "prc", "me", "ri", "cshoc", "dates", "low", "chn")
    return Section8Inputs(
        starts=_group_starts(data, group_cols),
        remove_8a=_filter_8a_decision(data, group_cols),
        gap_days=int(BESS_SECTION8_GAP_TRADING_DAYS * 365 / 252),  # trading -> calendar days
        **{f: data[f].to_numpy() for f in array_fields},
    )


def _verify_presorted_order(
    data: pl.DataFrame, group_cols: list[str], inp: Section8Inputs, sorted_path: Path
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
    reason_path = spill_dir / "__bess_s8_reason.parquet"
    pl.DataFrame({"_reason": reason}).write_parquet(reason_path, compression=BESS_SPILL_COMPRESSION)
    return (
        pl.concat([pl.scan_parquet(sorted_path), pl.scan_parquet(reason_path)], how="horizontal")
        .filter(pl.col("_reason") == 0)
        .drop("_reason")
    )


def apply_bessembinder_section8(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    country_col: str = "excntry",
    spill_dir: Path | None = None,
    presorted_path: Path | None = None,
) -> pl.LazyFrame:
    """
    Description:
        Apply the Section 8 filter chain (8a-8h) as one numba pass per security,
        preserving sequential-survivor semantics. Memory-bounded array path.
    Steps:
        1) Streaming-sort input to a spill file (or verify presorted_path).
        2) Build typed NumPy inputs (kernel arrays + 8a decision).
        3) Run section8_all -> per-row keep/remove; filter lazily.
    Note:
        With presorted_path, the panel is read from that file and df is NOT
        re-read — the caller must have written df's data there, sorted by
        group_cols + sort_col.
    Output:
        Filtered LazyFrame.
    """
    group_cols = group_cols or ["gvkey", "iid"]
    if spill_dir is None:
        raise ValueError("apply_bessembinder_section8 requires spill_dir")
    if presorted_path is not None and not presorted_path.exists():
        raise FileNotFoundError(f"presorted_path does not exist: {presorted_path}")

    sorted_path = presorted_path or (spill_dir / "__bess_s8_sorted.parquet")
    if presorted_path is None:
        _sort_to_spill(df, group_cols + [sort_col], sorted_path)

    data = _load_section8(sorted_path, group_cols, sort_col, country_col)
    inp = _section8_inputs(data, group_cols)
    if presorted_path is not None:
        _verify_presorted_order(data, group_cols, inp, sorted_path)

    reason = np.zeros(data.height, dtype=np.int8)
    bk.section8_all(
        inp.starts,
        reason,
        inp.remove_8a,
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
