"""
Bessembinder et al. (2023) Data Corrections for Compustat

Implements the Section 6 decimal-error corrections and Section 8 filters from
Bessembinder et al. (2023) "Do Global Stocks Outperform US Treasury Bills?"
Data Appendix. Both sections run as memory-bounded array/numba passes: the
input is streaming-sorted to a spill file, corrections/filters run in numpy,
and the result is reattached via a lazy horizontal concat.
"""

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


# Section 6: decimal-error corrections
def _correct_variable_arrays(
    x_raw: np.ndarray,
    starts: np.ndarray,
    window_sizes: list[int],
    correction_method: str,
    price_floor: bool = False,
    variation_threshold: float = 1.3,
) -> np.ndarray:
    """
    Description:
        Full Section 6 correction for one variable on a numpy array:
        single- then multi-period detection, cascading validation, correction.
    Steps:
        1) Zeros -> NaN in a working copy that feeds detection AND output, so a
           zero input becomes null (legacy zero-to-null behavior).
        2) Detect single- then multi-period errors; validate cascading.
        3) price_floor: keep only divide-direction corrections on >=$1 prices.
        4) Apply: 'bessembinder'/'floor' multiply by the factor;
           'interpolation'/'floor_interp' use the endpoint geometric mean
           (single-endpoint fallback).
    Output:
        Corrected float64 array (NaN = null).
    """
    n = len(x_raw)
    x_det = x_raw.copy()
    x_det[x_det == 0.0] = np.nan
    factor = np.ones(n, dtype=np.float64)
    wsize = np.full(n, -1, dtype=np.int32)  # -1 = null window size
    ep_l = np.full(n, np.nan, dtype=np.float64)
    ep_r = np.full(n, np.nan, dtype=np.float64)

    bk.detect_single_period_all(x_det, starts, factor, ep_l, ep_r)
    wsize[factor != 1.0] = 1
    nlags = np.array(sorted(w for w in window_sizes if w > 1), dtype=np.int64)
    if len(nlags) > 0:
        bk.detect_multi_period_all(
            x_det, starts, factor, wsize, ep_l, ep_r, nlags, variation_threshold
        )
    bk.validate_cascading_all(starts, factor, wsize)

    if price_floor:
        is_multiply = factor > 1.0  # correction direction is multiply
        is_sub_dollar = x_det < 1.0  # original price below $1
        # keep only divide-direction corrections on >=$1 prices
        gated = (factor != 1.0) & (is_multiply | is_sub_dollar)
        factor[gated] = 1.0

    if correction_method not in ("interpolation", "floor_interp"):
        return x_det * factor

    flagged = factor != 1.0
    corrected = x_det.copy()
    both = flagged & ~np.isnan(ep_l) & ~np.isnan(ep_r)
    corrected[both] = np.sqrt(ep_l[both] * ep_r[both])
    left_only = flagged & ~np.isnan(ep_l) & np.isnan(ep_r)
    corrected[left_only] = ep_l[left_only]
    right_only = flagged & np.isnan(ep_l) & ~np.isnan(ep_r)
    corrected[right_only] = ep_r[right_only]
    return corrected


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
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    if window_sizes is None:
        window_sizes = BESS_DEFAULT_WINDOWS

    sorted_path = spill_dir / "__bess_sorted.parquet"
    df.sort(group_cols + [sort_col]).sink_parquet(sorted_path, compression=BESS_SPILL_COMPRESSION)
    lf = pl.scan_parquet(sorted_path)
    schema_names = lf.collect_schema().names()

    value_cols = [
        v for v in ("trfd", "qunit", "adrrc", "ajexdi", "prccd", "cshoc") if v in schema_names
    ]
    data = lf.select(
        group_cols + [sort_col] + [pl.col(c).cast(pl.Float64) for c in value_cols]
    ).collect()
    starts = _group_starts(data, group_cols)
    corrected_cols: dict[str, np.ndarray] = {}

    # trfd, qunit and (NA data only) adrrc are corrected independently.
    # (_correct_variable_arrays copies its input internally, so no .copy() here.)
    for variable in ("trfd", "qunit", "adrrc"):
        if variable in schema_names and (variable != "adrrc" or has_adrrc):
            corrected_cols[variable] = _correct_variable_arrays(
                data[variable].to_numpy(),
                starts,
                window_sizes,
                correction_method,
                variation_threshold=variation_threshold,
            )

    # Reconstruct prccd/cshoc from corrected split-adjusted values; only when
    # all three inputs are present (always so on the Compustat security files).
    if {"ajexdi", "prccd", "cshoc"} <= set(schema_names):
        ajexdi = data["ajexdi"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            # The floor variants gate the price variable only (divide-direction, >=$1).
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
            corrected_cols["prccd"] = adjprc * ajexdi
            corrected_cols["cshoc"] = adjcsho / ajexdi

    corr_path = spill_dir / "__bess_corrected_cols.parquet"
    pl.DataFrame(corrected_cols).with_columns(
        [pl.col(name).fill_nan(None) for name in corrected_cols]
    ).write_parquet(corr_path, compression=BESS_SPILL_COMPRESSION)

    # Reattach the corrected columns in place of the originals (which are
    # dropped from the sorted scan and reappear at the end of the column order).
    return pl.concat(
        [pl.scan_parquet(sorted_path).drop(list(corrected_cols)), pl.scan_parquet(corr_path)],
        how="horizontal",
    )


# Section 8: additional filters
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
        preserving sequential-survivor semantics. Memory-bounded array path
        with spill files.
    Steps:
        1) Streaming-sort input to a spill file (or verify presorted_path).
        2) Collect keys, kernel inputs and country flags in one pass.
        3) Filter 8a global decision: positive-volume mean + 2% quantile.
        4) section8_all kernel -> per-row keep/remove; filter lazily.
    Note:
        With presorted_path, the panel is read from that file and df is NOT
        re-read — the caller must have written df's data there, sorted by
        group_cols + sort_col.
    Output:
        Filtered LazyFrame.
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    if spill_dir is None:
        raise ValueError("apply_bessembinder_section8 requires spill_dir")
    if presorted_path is not None and not presorted_path.exists():
        raise FileNotFoundError(f"presorted_path does not exist: {presorted_path}")

    sorted_path = presorted_path or (spill_dir / "__bess_s8_sorted.parquet")
    if presorted_path is None:
        df.sort(group_cols + [sort_col]).sink_parquet(
            sorted_path, compression=BESS_SPILL_COMPRESSION
        )
    lf = pl.scan_parquet(sorted_path)

    # Collect every kernel input in one pass: float value columns, the date as
    # physical int32 days, and the two country flags.
    value_cols = ["ajexdi", "prc", "me", "ri", "cshoc", "dolvol"]
    data = lf.select(
        group_cols
        + [sort_col]
        + [pl.col(c).cast(pl.Float64) for c in value_cols]
        + [
            pl.col(sort_col).cast(pl.Date).to_physical().cast(pl.Int32).alias("_date"),
            pl.col(country_col).is_in(BESS_LOW_PRICE_COUNTRIES).fill_null(False).alias("_low"),
            (pl.col(country_col) == "CHN").fill_null(False).alias("_chn"),
        ]
    ).collect()
    starts = _group_starts(data, group_cols)
    arr = {c: data[c].to_numpy() for c in value_cols + ["_date", "_low", "_chn"]}

    if presorted_path is not None:
        # Trusting an externally sorted file: a stale or differently-ordered
        # spill would silently corrupt the whole panel, so verify the order.
        n_groups = data.select(pl.struct(group_cols).n_unique()).item()
        if len(starts) - 1 != n_groups:
            raise ValueError(f"presorted file {sorted_path}: groups are not contiguous")
        d = arr["_date"]
        interior = np.ones(len(d), dtype=np.bool_)
        interior[starts[:-1]] = False  # group starts are exempt from the diff check
        idx = np.flatnonzero(interior)
        if not np.all(d[idx] >= d[idx - 1]):
            raise ValueError(f"presorted file {sorted_path}: dates not sorted within a group")

    # Filter 8a's global decision (the only cross-security statistic): the
    # per-security mean of positive volume, computed once, with the bottom-2%
    # cutoff taken as a scalar quantile (nulls ignored). Groups with no positive
    # volume (null mean) are KEPT; rows are realigned to the group_starts order.
    avg_vol = (
        data.lazy()
        .filter(pl.col("dolvol") > 0)
        .group_by(group_cols)
        .agg(pl.mean("dolvol").alias("_avg_vol"))
        .collect()
    )
    cutoff = avg_vol["_avg_vol"].quantile(0.02)
    remove_8a = (
        data.select(group_cols)
        .unique()
        .join(avg_vol, on=group_cols, how="left")
        .sort(group_cols)
        .select((pl.col("_avg_vol") <= cutoff).fill_null(False))
        .to_series()
        .to_numpy()
        .copy()
    )

    reason = np.zeros(data.height, dtype=np.int8)
    # Filter 8d gap bound: trading days -> calendar days (~11 months).
    gap_calendar_days = int(BESS_SECTION8_GAP_TRADING_DAYS * 365 / 252)
    bk.section8_all(
        starts,
        reason,
        remove_8a,
        arr["ajexdi"],
        arr["prc"],
        arr["me"],
        arr["ri"],
        arr["cshoc"],
        arr["_date"],
        arr["_low"],
        arr["_chn"],
        gap_calendar_days,
    )

    reason_path = spill_dir / "__bess_s8_reason.parquet"
    pl.DataFrame({"_reason": reason}).write_parquet(reason_path, compression=BESS_SPILL_COMPRESSION)
    return (
        pl.concat([pl.scan_parquet(sorted_path), pl.scan_parquet(reason_path)], how="horizontal")
        .filter(pl.col("_reason") == 0)
        .drop("_reason")
    )
