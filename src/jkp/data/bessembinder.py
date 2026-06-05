"""
Bessembinder et al. (2023) Data Corrections for Compustat

This module implements the decimal error corrections (Section 6) and additional
filters (Section 8) from Bessembinder et al. (2023) "Do Global Stocks Outperform US Treasury Bills?"
Data Appendix.

See BESSEMBINDER_IMPLEMENTATION_PLAN.md for detailed algorithm documentation
including deviations from the original paper.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from polars import col

from . import bessembinder_kernels as bk

logger = logging.getLogger(__name__)

# =============================================================================
# Bessembinder et al. (2023) Data Corrections - Section 6: Decimal Errors
# =============================================================================

# =============================================================================
# Kernel marshalling helpers
# =============================================================================

_ERROR_TYPE_BY_CODE = ["high_10x", "high_100x", "high_1000x", "low_10x", "low_100x", "low_1000x"]
_CODE_BY_ERROR_TYPE = {name: i for i, name in enumerate(_ERROR_TYPE_BY_CODE)}
_WINDOW_TYPE_BY_CODE = ["single", "full", "sub_a", "sub_b"]
_CODE_BY_WINDOW_TYPE = {name: i for i, name in enumerate(_WINDOW_TYPE_BY_CODE)}


def _group_starts(df: pl.DataFrame, group_cols: list[str]) -> np.ndarray:
    """
    Description:
        Compute contiguous group boundary offsets for a frame sorted by
        group_cols (+ date).
    Steps:
        1) rle_id over the group-key struct -> contiguous int codes.
        2) Boundaries where the code changes, plus 0 and n.
    Output:
        int64 array of length n_groups + 1 (offsets into the frame).
    """
    n = df.height
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    gid = df.select(pl.struct(group_cols).rle_id().alias("_g"))["_g"].to_numpy()
    changes = np.flatnonzero(np.diff(gid)) + 1
    return np.concatenate(([0], changes, [n])).astype(np.int64)


def _values_for_detection(df: pl.DataFrame, col_name: str) -> np.ndarray:
    """
    Description:
        Float64 detection array for a value column: nulls and zeros become
        NaN (replicating the original zero->null replacement before ratio
        computation; NaN inputs stay NaN).
    Steps:
        1) Cast to Float64, nulls -> NaN via to_numpy.
        2) Zeros -> NaN.
    Output:
        Writable float64 numpy array aligned with the frame rows.
    """
    x = df[col_name].cast(pl.Float64).to_numpy().copy()
    x[x == 0.0] = np.nan
    return x


def _codes_to_strings(codes: np.ndarray, names: list[str]) -> pl.Series:
    """Map int8 codes (-1 = null) to their string labels as a Utf8 Series."""
    return pl.Series(codes).replace_strict(
        dict(enumerate(names)) | {-1: None},
        return_dtype=pl.Utf8,
    )


def _detection_output_arrays(n: int) -> dict[str, np.ndarray]:
    """Preallocate kernel output arrays (null sentinels: NaN / -1)."""
    return {
        "factor": np.ones(n, dtype=np.float64),
        "etype": np.full(n, -1, dtype=np.int8),
        "wsize": np.full(n, -1, dtype=np.int32),
        "wtype": np.full(n, -1, dtype=np.int8),
        "ep_l": np.full(n, np.nan, dtype=np.float64),
        "ep_r": np.full(n, np.nan, dtype=np.float64),
        "rat_l": np.full(n, np.nan, dtype=np.float64),
        "rat_r": np.full(n, np.nan, dtype=np.float64),
        "var": np.full(n, np.nan, dtype=np.float64),
    }


def _read_detection_arrays(df: pl.DataFrame, col_name: str) -> dict[str, np.ndarray]:
    """
    Description:
        Read pre-existing detection-state columns into kernel arrays (used by
        the multi-period shim, whose callers may pre-initialize factor /
        error_type and may or may not carry window/debug columns).
    Steps:
        1) factor: existing column (nulls -> 1.0) or ones.
        2) error_type / window_type: strings -> int8 codes (-1 = null).
        3) window_size: Int32 with -1 sentinel; debug columns -> NaN-filled.
    Output:
        Dict of writable numpy arrays matching _detection_output_arrays.
    """
    n = df.height
    out = _detection_output_arrays(n)
    cols = df.columns
    factor_col = f"{col_name}_correction_factor"
    error_type_col = f"{col_name}_error_type"
    window_col = f"{col_name}_window_size"
    window_type_col = f"{col_name}_window_type"
    debug = {
        "ep_l": f"{col_name}_endpoint_left",
        "ep_r": f"{col_name}_endpoint_right",
        "rat_l": f"{col_name}_ratio_to_left",
        "rat_r": f"{col_name}_ratio_to_right",
        "var": f"{col_name}_variation_ratio",
    }
    if factor_col in cols:
        out["factor"] = df[factor_col].cast(pl.Float64).fill_null(1.0).to_numpy().copy()
    if error_type_col in cols:
        out["etype"] = (
            df[error_type_col]
            .replace_strict(_CODE_BY_ERROR_TYPE, default=-1, return_dtype=pl.Int8)
            .fill_null(-1)
            .to_numpy()
            .copy()
        )
    if window_col in cols:
        out["wsize"] = df[window_col].cast(pl.Int32).fill_null(-1).to_numpy().copy()
    if window_type_col in cols:
        out["wtype"] = (
            df[window_type_col]
            .replace_strict(_CODE_BY_WINDOW_TYPE, default=-1, return_dtype=pl.Int8)
            .fill_null(-1)
            .to_numpy()
            .copy()
        )
    for key, name in debug.items():
        if name in cols:
            out[key] = df[name].cast(pl.Float64).to_numpy().copy()
    return out


def _write_detection_columns(
    df: pl.DataFrame, col_name: str, arrays: dict[str, np.ndarray], include_window: bool
) -> pl.DataFrame:
    """
    Description:
        Write kernel output arrays back onto the frame as the original
        detection columns (factor, error_type, window_type, endpoints,
        ratios; plus window_size / variation_ratio for multi-period).
    Steps:
        1) Map code arrays to Utf8; NaN debug values stay as nulls via masks.
        2) with_columns in one call (aligned: frame is date-sorted).
    Output:
        Frame with detection columns attached/overwritten.
    """
    debug_names = [
        (f"{col_name}_endpoint_left", arrays["ep_l"]),
        (f"{col_name}_endpoint_right", arrays["ep_r"]),
        (f"{col_name}_ratio_to_left", arrays["rat_l"]),
        (f"{col_name}_ratio_to_right", arrays["rat_r"]),
    ]
    if include_window:
        debug_names.append((f"{col_name}_variation_ratio", arrays["var"]))
    new_cols = [
        pl.Series(f"{col_name}_correction_factor", arrays["factor"]),
        _codes_to_strings(arrays["etype"], _ERROR_TYPE_BY_CODE).alias(f"{col_name}_error_type"),
        _codes_to_strings(arrays["wtype"], _WINDOW_TYPE_BY_CODE).alias(f"{col_name}_window_type"),
    ]
    if include_window:
        new_cols.append(pl.Series(f"{col_name}_window_size", arrays["wsize"], dtype=pl.Int32))
    new_cols += [pl.Series(name, arr) for name, arr in debug_names]
    df = df.with_columns(new_cols)
    # Null sentinels: -1 window size -> null, NaN debug values -> null
    fixups = [col(name).fill_nan(None).alias(name) for name, _ in debug_names]
    if include_window:
        ws = f"{col_name}_window_size"
        fixups.append(
            pl.when(col(ws) == bk.WS_NULL).then(pl.lit(None)).otherwise(col(ws)).alias(ws)
        )
    return df.with_columns(fixups)


def _zero_to_null(df: pl.LazyFrame | pl.DataFrame, col_name: str) -> pl.LazyFrame | pl.DataFrame:
    """Replace zeros with null in the value column (keeps the input dtype)."""
    return df.with_columns(
        pl.when(col(col_name) == 0).then(pl.lit(None)).otherwise(col(col_name)).alias(col_name)
    )


def _detect_decimal_error_single_period(
    df: pl.LazyFrame,
    col_name: str,
    group_cols: list[str],
    sort_col: str,
) -> pl.LazyFrame:
    """
    Description:
        Detect single-period decimal shift errors (Bessembinder Section 6a)
        via the numba kernel; byte-identical to the original Polars version.
    Steps:
        1) Sort by group + date, zeros -> null, collect.
        2) Run detect_single_period_all on per-security slices.
        3) Write factor / error_type / window_type / endpoint / ratio columns.
    Output:
        LazyFrame with detection columns added.
    """
    eager = _zero_to_null(df.sort(group_cols + [sort_col]), col_name).collect()
    arrays = _detection_output_arrays(eager.height)
    starts = _group_starts(eager, group_cols)
    bk.detect_single_period_all(
        _values_for_detection(eager, col_name),
        starts,
        arrays["factor"],
        arrays["etype"],
        arrays["wtype"],
        arrays["ep_l"],
        arrays["ep_r"],
        arrays["rat_l"],
        arrays["rat_r"],
    )
    return _write_detection_columns(eager, col_name, arrays, include_window=False).lazy()


def _detect_decimal_error_multi_period(
    df: pl.LazyFrame,
    col_name: str,
    group_cols: list[str],
    sort_col: str,
    window_sizes: list[int],
    variation_threshold: float = 1.3,
) -> pl.LazyFrame:
    """
    Description:
        Detect multi-period decimal shift errors (Bessembinder Section 6b)
        via the numba kernel: per nlag (ascending) the FULL, SUB_A and SUB_B
        window passes with first-write-wins interior propagation. Magnitudes
        are classed with the nested 500/50/5 chain like single-period — a
        deliberate fix over the original, whose magnitude [1, 2, 3] loop was
        dead code beyond 10x (see commit 507df7c), under-correcting
        multi-day 100x/1000x errors. All other semantics are byte-identical
        to the original Polars version.
    Steps:
        1) Sort by group + date, zeros -> null, collect.
        2) Read pre-existing detection state (callers may pre-initialize).
        3) Run detect_multi_period_all on per-security slices.
        4) Write detection + window/debug columns back.
    Output:
        LazyFrame with detection columns added/updated.
    """
    eager = _zero_to_null(df.sort(group_cols + [sort_col]), col_name).collect()
    arrays = _read_detection_arrays(eager, col_name)
    starts = _group_starts(eager, group_cols)
    nlags = np.array(sorted(w for w in window_sizes if w > 1), dtype=np.int64)
    if len(nlags) > 0:
        bk.detect_multi_period_all(
            _values_for_detection(eager, col_name),
            starts,
            arrays["factor"],
            arrays["etype"],
            arrays["wsize"],
            arrays["wtype"],
            arrays["ep_l"],
            arrays["ep_r"],
            arrays["rat_l"],
            arrays["rat_r"],
            arrays["var"],
            nlags,
            variation_threshold,
        )
    return _write_detection_columns(eager, col_name, arrays, include_window=True).lazy()


def correct_decimal_errors(
    df: pl.LazyFrame,
    col_name: str,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    window_sizes: list[int] | None = None,
    log_corrections: bool = True,
    validate_cascading: bool = True,
    correction_method: str = "bessembinder",
) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
    """
    Apply Bessembinder Section 6 decimal error corrections to a column.

    This function detects and corrects temporary decimal shift errors by looking
    for spike-and-reversal patterns. It handles both single-period errors and
    multi-period errors.

    Correction Methods:
        correction_method='bessembinder' (default):
            Uses fixed multipliers (0.1, 0.01, 0.001 for high errors; 10, 100, 1000
            for low errors) as described in Bessembinder et al. (2023).

        correction_method='interpolation':
            Uses the geometric mean of surrounding clean values (endpoints) to
            determine the correct price level. This handles non-10x errors (e.g.,
            5.5x, 7x) that fixed multipliers would over- or under-correct.

            CRSP validation showed 'bessembinder' method achieves ~3% accuracy for
            high_* corrections vs 92-100% for low_*. The 'interpolation' method
            addresses this by using actual market context instead of assuming
            exact powers of 10.

    Cascading Error Prevention:
        When validate_cascading=True (default), the algorithm validates corrections
        to prevent cascading errors where a correct value is incorrectly flagged
        because its neighbors are errors. Two validation checks are applied:

        1. Endpoint Validation: Reject corrections where either endpoint matches
           another error value (within 1% tolerance).

        2. Mixed Direction Validation: Reject corrections in clusters where both
           HIGH and LOW corrections occur within 5 days of each other.

        Set validate_cascading=False to disable these checks (for comparison
        with previous algorithm behavior).

    Window Size Selection:
        Default uses priority windows [1, 2, 3, 5, 10, 21] covering common error
        durations from 1 day to ~1 month. This provides ~10x performance improvement
        over exhaustive 63-window search while catching most real errors.

        Bessembinder et al. (2023) used nlag ∈ {1, 2, 3} on monthly data, reasoning
        that "anomalies that remain uncorrected beyond 3 periods are increasingly
        likely to reflect true underlying price changes." Our extended windows for
        daily data follow this principle while allowing for slightly longer errors.

        For exhaustive coverage, pass window_sizes=list(range(1, 64)) explicitly.

    Args:
        df: LazyFrame with security data
        col_name: Column to correct (e.g., 'prccd', 'cshoc', 'trfd', 'qunit')
        group_cols: Columns defining the security (default: ['gvkey', 'iid'])
        sort_col: Column to sort by (default: 'datadate')
        window_sizes: Window sizes in trading days (default: [1, 2, 3, 5, 10, 21])
        log_corrections: Whether to return a log of corrections made
        validate_cascading: Whether to validate corrections to prevent cascading
            errors (default: True). Set to False for legacy behavior.
        correction_method: Method for computing corrected values. Options:
            - 'bessembinder': Fixed 10x/100x/1000x multipliers (default)
            - 'interpolation': Geometric mean of surrounding clean values

    Returns:
        Tuple of (corrected_df, corrections_log)
        - corrected_df: LazyFrame with corrected values
        - corrections_log: LazyFrame with details of corrections (or None)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    if window_sizes is None:
        # Priority windows covering common error durations:
        # - 1: Single-day errors (most common)
        # - 2: 2-3 day errors
        # - 3: 4-5 day errors
        # - 5: 8-9 day errors
        # - 10: 18-19 day errors (~2 weeks)
        # - 21: 40-41 day errors (~1 month)
        # This reduces iterations from 558 to 54 (~10x speedup) while covering
        # the most common error durations. Bessembinder et al. (2023) only used
        # 1-3 periods on monthly data; we extend slightly for daily frequency.
        # For exhaustive coverage, use window_sizes=list(range(1, 64)) explicitly.
        window_sizes = [1, 2, 3, 5, 10, 21]

    factor_col = f"{col_name}_correction_factor"
    error_type_col = f"{col_name}_error_type"
    window_col = f"{col_name}_window_size"
    window_type_col = f"{col_name}_window_type"
    original_col = f"{col_name}_original"
    # Debug columns for tracking detection details
    endpoint_left_col = f"{col_name}_endpoint_left"
    endpoint_right_col = f"{col_name}_endpoint_right"
    ratio_left_col = f"{col_name}_ratio_to_left"
    ratio_right_col = f"{col_name}_ratio_to_right"
    variation_col = f"{col_name}_variation_ratio"

    # Store original value for logging
    df = df.with_columns(col(col_name).alias(original_col))

    # First pass: detect single-period errors (window=1)
    # This adds factor_col, error_type_col, and window_type_col
    print(f"  [correct] {col_name}: single-period detection", flush=True)
    df = _detect_decimal_error_single_period(df, col_name, group_cols, sort_col)

    # Add window size for single-period detections
    df = df.with_columns(
        pl.when(col(factor_col) != 1.0)
        .then(pl.lit(1))
        .otherwise(pl.lit(None))
        .cast(pl.Int32)
        .alias(window_col)
    )

    # Second pass: detect multi-period errors
    # Only check windows > 1 since single-period is already done
    multi_windows = [w for w in window_sizes if w > 1]
    if multi_windows:
        print(
            f"  [correct] {col_name}: multi-period detection, windows={multi_windows}", flush=True
        )
        df = _detect_decimal_error_multi_period(df, col_name, group_cols, sort_col, multi_windows)

    # Third pass: Validate corrections to prevent cascading errors. A flagged
    # row whose BOTH endpoint positions (pos +/- window_size) are themselves
    # flagged is a likely false positive sandwiched between two errors; the
    # kernel iterates rejection until stable (max 10 rounds) and resets only
    # factor and error_type on rejected rows (matching the original).
    if validate_cascading:
        print(f"  [correct] {col_name}: cascading validation", flush=True)
        eager = df.collect()
        arrays = _read_detection_arrays(eager, col_name)
        starts = _group_starts(eager, group_cols)
        rejected = np.zeros(eager.height, dtype=np.bool_)
        bk.validate_cascading_all(
            starts, arrays["factor"], arrays["etype"], arrays["wsize"], rejected
        )
        n_rejected = int(rejected.sum())
        if n_rejected > 0:
            logger.info(f"{col_name}: rejected {n_rejected} cascading false positives")
        df = eager.with_columns(
            pl.Series(factor_col, arrays["factor"]),
            _codes_to_strings(arrays["etype"], _ERROR_TYPE_BY_CODE).alias(error_type_col),
        ).lazy()

    # Apply corrections (validated if validate_cascading=True)
    if correction_method == "bessembinder":
        # Fixed multipliers: multiply original value by 0.1/0.01/0.001 or 10/100/1000
        df = df.with_columns((col(col_name) * col(factor_col)).alias(col_name))
    elif correction_method == "interpolation":
        # Interpolation: use geometric mean of surrounding clean values (endpoints)
        # This handles non-10x errors (e.g., 5.5x, 7x) that fixed multipliers miss.
        # For positions flagged as errors (factor != 1.0), replace with geometric mean
        # of left and right endpoints. If either endpoint is null, fall back to the
        # non-null endpoint; if both null, keep original value (shouldn't happen).
        df = df.with_columns(
            pl.when(col(factor_col) != 1.0)
            .then(
                pl.when(
                    col(endpoint_left_col).is_not_null() & col(endpoint_right_col).is_not_null()
                )
                .then((col(endpoint_left_col) * col(endpoint_right_col)).sqrt())
                .when(col(endpoint_left_col).is_not_null())
                .then(col(endpoint_left_col))
                .when(col(endpoint_right_col).is_not_null())
                .then(col(endpoint_right_col))
                .otherwise(col(col_name))  # Both null - keep original (shouldn't happen)
            )
            .otherwise(col(col_name))
            .alias(col_name)
        )
    else:
        raise ValueError(
            f"Unknown correction_method: {correction_method}. "
            "Expected 'bessembinder' or 'interpolation'."
        )

    # Build final corrections log if requested
    corrections_log = None
    if log_corrections:
        corrections_log = df.filter(col(factor_col) != 1.0).select(
            group_cols
            + [
                sort_col,
                pl.lit(col_name).alias("variable"),
                col(original_col).alias("original_value"),
                col(col_name).alias("corrected_value"),
                col(factor_col).alias("correction_factor"),
                pl.lit(correction_method).alias("correction_method"),
                col(error_type_col).alias("error_type"),
                col(window_col).alias("window_size"),
                col(window_type_col).alias("window_type"),
                # Debug columns
                col(endpoint_left_col).alias("endpoint_left"),
                col(endpoint_right_col).alias("endpoint_right"),
                col(ratio_left_col).alias("ratio_to_left"),
                col(ratio_right_col).alias("ratio_to_right"),
                col(variation_col).alias("variation_ratio"),
            ]
        )

        # Log summary of corrections
        try:
            log_collected = corrections_log.collect()
            n_corrections = len(log_collected)
            if n_corrections > 0:
                # Group by error_type and window_type for summary
                summary = (
                    log_collected.group_by(["error_type", "window_type"])
                    .agg(pl.len().alias("count"))
                    .sort("count", descending=True)
                )
                summary_str = ", ".join(
                    f"{row['error_type']}/{row['window_type']}:{row['count']}"
                    for row in summary.iter_rows(named=True)
                )
                logger.info(
                    f"{col_name}: {n_corrections} corrections applied "
                    f"(method={correction_method}, {summary_str})"
                )
            else:
                logger.debug(f"{col_name}: no decimal errors detected")
            # Convert back to lazy for return
            corrections_log = log_collected.lazy()
        except Exception as e:
            logger.warning(f"{col_name}: could not summarize corrections: {e}")

    # Clean up temporary columns
    df = df.drop(
        [
            factor_col,
            error_type_col,
            window_col,
            window_type_col,
            original_col,
            endpoint_left_col,
            endpoint_right_col,
            ratio_left_col,
            ratio_right_col,
            variation_col,
        ]
    )

    return df, corrections_log


def _correct_variable_arrays(
    x_raw: np.ndarray,
    starts: np.ndarray,
    window_sizes: list[int],
    correction_method: str,
    price_floor: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Description:
        Run the full Section 6 correction pipeline for one variable on raw
        numpy arrays: single-period detection, multi-period detection,
        cascading validation, and correction application.
    Steps:
        1) zeros -> NaN copy for detection (raw kept for logging).
        2) detect_single_period_all; window_size = 1 where flagged.
        3) detect_multi_period_all over windows > 1 (ascending).
        4) validate_cascading_all (always on for the array path).
        5) If price_floor: drop flags where the original value is < 1.0 or
           the correction direction is multiply (factor > 1) — the handoff
           research found divide corrections on >=$1 prices are the only
           reliably accurate class.
        6) Apply correction: 'bessembinder'/'floor' multiplies by the
           factor; 'interpolation' uses geometric mean of endpoints with
           fallbacks.
    Output:
        (corrected float64 array with NaN for nulls, kernel output arrays,
        flagged row indices).
    """
    x_det = x_raw.copy()
    x_det[x_det == 0.0] = np.nan
    arrays = _detection_output_arrays(len(x_raw))
    bk.detect_single_period_all(
        x_det,
        starts,
        arrays["factor"],
        arrays["etype"],
        arrays["wtype"],
        arrays["ep_l"],
        arrays["ep_r"],
        arrays["rat_l"],
        arrays["rat_r"],
    )
    arrays["wsize"][arrays["factor"] != 1.0] = 1
    nlags = np.array(sorted(w for w in window_sizes if w > 1), dtype=np.int64)
    if len(nlags) > 0:
        bk.detect_multi_period_all(
            x_det,
            starts,
            arrays["factor"],
            arrays["etype"],
            arrays["wsize"],
            arrays["wtype"],
            arrays["ep_l"],
            arrays["ep_r"],
            arrays["rat_l"],
            arrays["rat_r"],
            arrays["var"],
            nlags,
            1.3,
        )
    rejected = np.zeros(len(x_raw), dtype=np.bool_)
    bk.validate_cascading_all(starts, arrays["factor"], arrays["etype"], arrays["wsize"], rejected)

    if price_floor:
        gated = (arrays["factor"] != 1.0) & ((arrays["factor"] > 1.0) | (x_det < 1.0))
        arrays["factor"][gated] = 1.0

    flagged = arrays["factor"] != 1.0
    if correction_method in ("bessembinder", "floor"):
        corrected = x_det * arrays["factor"]
    elif correction_method == "interpolation":
        corrected = x_det.copy()
        ep_l, ep_r = arrays["ep_l"], arrays["ep_r"]
        both = flagged & ~np.isnan(ep_l) & ~np.isnan(ep_r)
        corrected[both] = np.sqrt(ep_l[both] * ep_r[both])
        left_only = flagged & ~np.isnan(ep_l) & np.isnan(ep_r)
        corrected[left_only] = ep_l[left_only]
        right_only = flagged & np.isnan(ep_l) & ~np.isnan(ep_r)
        corrected[right_only] = ep_r[right_only]
    else:
        raise ValueError(
            f"Unknown correction_method: {correction_method}. "
            "Expected 'bessembinder' or 'interpolation'."
        )
    return corrected, arrays, np.flatnonzero(flagged)


def _log_from_arrays(
    keys: pl.DataFrame,
    group_cols: list[str],
    sort_col: str,
    variable: str,
    x_raw: np.ndarray,
    corrected: np.ndarray,
    arrays: dict[str, np.ndarray],
    flag_idx: np.ndarray,
    correction_method: str,
) -> pl.DataFrame:
    """Build the pinned-order corrections log for one variable from kernel arrays."""
    log = keys[flag_idx].with_columns(
        pl.lit(variable).alias("variable"),
        pl.Series("original_value", x_raw[flag_idx]),
        pl.Series("corrected_value", corrected[flag_idx]),
        pl.Series("correction_factor", arrays["factor"][flag_idx]),
        pl.lit(correction_method).alias("correction_method"),
        _codes_to_strings(arrays["etype"][flag_idx], _ERROR_TYPE_BY_CODE).alias("error_type"),
        pl.Series("window_size", arrays["wsize"][flag_idx], dtype=pl.Int32),
        _codes_to_strings(arrays["wtype"][flag_idx], _WINDOW_TYPE_BY_CODE).alias("window_type"),
        pl.Series("endpoint_left", arrays["ep_l"][flag_idx]),
        pl.Series("endpoint_right", arrays["ep_r"][flag_idx]),
        pl.Series("ratio_to_left", arrays["rat_l"][flag_idx]),
        pl.Series("ratio_to_right", arrays["rat_r"][flag_idx]),
        pl.Series("variation_ratio", arrays["var"][flag_idx]),
    )
    debug_cols = [
        "endpoint_left",
        "endpoint_right",
        "ratio_to_left",
        "ratio_to_right",
        "variation_ratio",
    ]
    return log.with_columns([col(c).fill_nan(None).alias(c) for c in debug_cols]).select(
        group_cols
        + [
            sort_col,
            "variable",
            "original_value",
            "corrected_value",
            "correction_factor",
            "correction_method",
            "error_type",
            "window_size",
            "window_type",
        ]
        + debug_cols
    )


def _apply_section6_slim(
    df: pl.LazyFrame,
    group_cols: list[str],
    sort_col: str,
    window_sizes: list[int],
    has_adrrc: bool,
    correction_method: str,
    spill_dir: Path,
) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
    """
    Description:
        Memory-bounded Section 6 path for full-size cluster data. Avoids ever
        materializing the wide input frame: the input is sorted once with a
        streaming sink to a spill file, each variable's values are collected
        as a single float64 column, all corrections run in numpy/numba
        arrays, and the corrected columns are reattached lazily via a
        horizontal concat of two parquet scans.
    Steps:
        1) Streaming-sort input to spill_dir/__bess_sorted.parquet.
        2) Collect group keys once; build group-start offsets.
        3) Correct trfd, qunit, adrrc (if present) from per-column arrays.
        4) adjprc = prccd/ajexdi and adjcsho = cshoc*ajexdi in numpy; correct
           both; reconstruct prccd and cshoc.
        5) Write corrected columns parquet; return lazy horizontal concat of
           (sorted scan minus replaced columns) and the corrected columns,
           plus the combined corrections log.
    Output:
        (corrected LazyFrame, corrections log LazyFrame or None). Note: NaN
        values in corrected columns become null on writeback, and replaced
        columns move to the end of the column order.
    """
    sorted_path = spill_dir / "__bess_sorted.parquet"
    print("[section6] streaming-sorting input to spill file...", flush=True)
    df.sort(group_cols + [sort_col]).sink_parquet(sorted_path)
    lf = pl.scan_parquet(sorted_path)
    schema_names = lf.collect_schema().names()

    print("[section6] building group index...", flush=True)
    keys = lf.select(group_cols + [sort_col]).collect()
    starts = _group_starts(keys, group_cols)

    def load(name: str) -> np.ndarray:
        return lf.select(pl.col(name).cast(pl.Float64)).collect()[name].to_numpy().copy()

    def correct(variable: str, x_raw: np.ndarray) -> np.ndarray:
        print(f"[section6] correcting {variable}...", flush=True)
        # 'floor' gates the price variable only (divide-direction, >=$1);
        # non-price variables keep standard bessembinder corrections.
        corrected, arrays, flag_idx = _correct_variable_arrays(
            x_raw,
            starts,
            window_sizes,
            correction_method,
            price_floor=(correction_method == "floor" and variable == "adjprc"),
        )
        logger.info(f"{variable}: {len(flag_idx)} corrections applied")
        logs.append(
            _log_from_arrays(
                keys,
                group_cols,
                sort_col,
                variable,
                x_raw,
                corrected,
                arrays,
                flag_idx,
                correction_method,
            )
        )
        return corrected

    logs: list[pl.DataFrame] = []
    corrected_cols: dict[str, np.ndarray] = {}

    # Steps 1-2: correct trfd, qunit and (NA data only) adrrc independently
    for variable in ["trfd", "qunit"]:
        if variable in schema_names:
            corrected_cols[variable] = correct(variable, load(variable))
    if has_adrrc:
        if "adrrc" in schema_names:
            corrected_cols["adrrc"] = correct("adrrc", load("adrrc"))
        else:
            logger.warning("ADRRC column expected (has_adrrc=True) but not found in data schema")

    # Steps 3-5: correct split-adjusted price/shares, reconstruct prccd/cshoc
    ajexdi = load("ajexdi")
    with np.errstate(divide="ignore", invalid="ignore"):
        adjprc_corr = correct("adjprc", load("prccd") / ajexdi)
        adjcsho_corr = correct("adjcsho", load("cshoc") * ajexdi)
        corrected_cols["prccd"] = adjprc_corr * ajexdi
        corrected_cols["cshoc"] = adjcsho_corr / ajexdi

    print("[section6] writing corrected columns...", flush=True)
    corr_path = spill_dir / "__bess_corrected_cols.parquet"
    pl.DataFrame(corrected_cols).with_columns(
        [col(name).fill_nan(None) for name in corrected_cols]
    ).write_parquet(corr_path)

    result = pl.concat(
        [pl.scan_parquet(sorted_path).drop(list(corrected_cols)), pl.scan_parquet(corr_path)],
        how="horizontal",
    )
    all_corrections = pl.concat(logs, how="vertical_relaxed").lazy() if logs else None
    return result, all_corrections


def apply_bessembinder_section6(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    window_sizes: list[int] | None = None,
    has_adrrc: bool = False,
    correction_method: str = "bessembinder",
    spill_dir: Path | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Apply Bessembinder Section 6 decimal corrections in the correct order.

    Order of application (per Section 6c):
    1. Correct TRFD (total return factor) and QUNIT independently
    2. Correct ADRRC if present (ADR ratio, NA data only)
    3. Compute adjusted price and shares: adjPRC = PRCCD/AJEXDI, adjCSHO = CSHOC*AJEXDI
    4. Correct adjPRC and adjCSHO
    5. Reconstruct PRCCD and CSHOC from corrected adjusted values

    Args:
        df: LazyFrame with raw Compustat security data
        group_cols: Security identifier columns
        sort_col: Date column for sorting
        window_sizes: Window sizes for multi-period detection
        has_adrrc: Whether the data has ADRRC column (NA data only)
        correction_method: Method for computing corrected values. Options:
            - 'bessembinder': Fixed 10x/100x/1000x multipliers (default)
            - 'interpolation': Geometric mean of surrounding clean values
            - 'floor': As 'bessembinder', but price (adjprc) corrections are
              applied only in the divide direction and only where the
              original adjusted price is >= $1 (handoff research
              recommendation). Slim path (spill_dir) only.
        spill_dir: When set, use the memory-bounded array path
            (_apply_section6_slim) with spill files in this directory —
            required for full-size cluster data, where collecting the wide
            input frame does not fit in memory. When None (default), the
            frame-based path is used (fine for small/test data).

    Returns:
        Tuple of (corrected_df, all_corrections_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    if window_sizes is None:
        # Priority windows covering common error durations (matches correct_decimal_errors)
        window_sizes = [1, 2, 3, 5, 10, 21]

    if spill_dir is not None:
        return _apply_section6_slim(
            df, group_cols, sort_col, window_sizes, has_adrrc, correction_method, spill_dir
        )

    all_logs = []

    # Step 1: Correct TRFD independently
    if "trfd" in df.collect_schema().names():
        print("[section6] step 1/5: correcting trfd", flush=True)
        df, log = correct_decimal_errors(
            df, "trfd", group_cols, sort_col, window_sizes, correction_method=correction_method
        )
        if log is not None:
            all_logs.append(log)

    # Step 1: Correct QUNIT independently
    if "qunit" in df.collect_schema().names():
        print("[section6] step 1/5: correcting qunit", flush=True)
        df, log = correct_decimal_errors(
            df, "qunit", group_cols, sort_col, window_sizes, correction_method=correction_method
        )
        if log is not None:
            all_logs.append(log)

    # Step 2: Correct ADRRC if present (NA data only)
    adrrc_in_schema = "adrrc" in df.collect_schema().names()
    if has_adrrc:
        if adrrc_in_schema:
            logger.info("ADRRC column found in data - applying decimal corrections")
            print("[section6] step 2/5: correcting adrrc", flush=True)
            df, log = correct_decimal_errors(
                df, "adrrc", group_cols, sort_col, window_sizes, correction_method=correction_method
            )
            if log is not None:
                n_corrections = log.select(pl.len()).collect().item()
                logger.info(f"ADRRC: {n_corrections} decimal corrections applied")
                all_logs.append(log)
            else:
                logger.info("ADRRC: no decimal errors detected")
        else:
            logger.warning("ADRRC column expected (has_adrrc=True) but not found in data schema")
    else:
        if adrrc_in_schema:
            logger.debug("ADRRC column present in data but has_adrrc=False - skipping correction")

    # Step 3: Compute adjusted price and shares
    # adjPRC = PRCCD / AJEXDI (split-adjusted price)
    # adjCSHO = CSHOC * AJEXDI (split-adjusted shares)
    df = df.with_columns(
        [
            (col("prccd") / col("ajexdi")).alias("_adjprc"),
            (col("cshoc") * col("ajexdi")).alias("_adjcsho"),
        ]
    )

    # Step 4: Correct adjPRC and adjCSHO
    print("[section6] step 4/5: correcting adjprc", flush=True)
    df, log = correct_decimal_errors(
        df, "_adjprc", group_cols, sort_col, window_sizes, correction_method=correction_method
    )
    if log is not None:
        # Rename variable in log from _adjprc to adjprc
        log = log.with_columns(
            pl.when(col("variable") == "_adjprc")
            .then(pl.lit("adjprc"))
            .otherwise(col("variable"))
            .alias("variable")
        )
        all_logs.append(log)

    print("[section6] step 4/5: correcting adjcsho", flush=True)
    df, log = correct_decimal_errors(
        df, "_adjcsho", group_cols, sort_col, window_sizes, correction_method=correction_method
    )
    if log is not None:
        log = log.with_columns(
            pl.when(col("variable") == "_adjcsho")
            .then(pl.lit("adjcsho"))
            .otherwise(col("variable"))
            .alias("variable")
        )
        all_logs.append(log)

    print("[section6] step 5/5: reconstructing prccd/cshoc", flush=True)
    # Step 5: Reconstruct PRCCD and CSHOC from corrected adjusted values
    df = df.with_columns(
        [
            (col("_adjprc") * col("ajexdi")).alias("prccd"),
            (col("_adjcsho") / col("ajexdi")).alias("cshoc"),
        ]
    )

    # Clean up temporary columns
    df = df.drop(["_adjprc", "_adjcsho"])

    # Combine all correction logs
    if all_logs:
        all_corrections = pl.concat(all_logs, how="vertical_relaxed")
    else:
        all_corrections = None

    return df, all_corrections


# =============================================================================
# Bessembinder Data Quality Metrics and Validation
# =============================================================================


def log_data_quality_metrics(
    df: pl.LazyFrame,
    stage: str,
    price_col: str = "prccd",
    group_cols: list[str] | None = None,
) -> dict:
    """
    Log data quality metrics for debugging Bessembinder corrections.

    This function computes and logs summary statistics about the data at various
    stages of processing. Useful for comparing before/after corrections.

    Args:
        df: LazyFrame with price data
        stage: Label for this stage (e.g., "before_corrections", "after_corrections")
        price_col: Column with price data
        group_cols: Security identifier columns

    Returns:
        Dictionary with computed metrics for programmatic use
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    try:
        # Collect a sample for metrics (avoid full materialization for large data)
        data = df.collect()
        n_rows = len(data)
        n_securities = data.select(group_cols).unique().height

        metrics = {
            "stage": stage,
            "n_rows": n_rows,
            "n_securities": n_securities,
        }

        # Price metrics - cast to float to handle Decimal types
        if price_col in data.columns:
            # Cast to float to handle Decimal[38,8] types from Compustat
            prices = data[price_col].cast(pl.Float64).drop_nulls()
            if len(prices) > 0:
                metrics["price_min"] = float(prices.min())
                metrics["price_max"] = float(prices.max())
                metrics["price_median"] = float(prices.median())
                metrics["price_null_pct"] = round(100 * (n_rows - len(prices)) / n_rows, 2)

        # Compute price ratios for extreme detection
        if price_col in data.columns and len(data) > 1:
            # Cast to float to handle Decimal types, then compute ratios
            data_float = data.with_columns(col(price_col).cast(pl.Float64).alias("_price_float"))
            # Group by security and compute day-over-day ratios
            with_ratios = (
                data_float.sort(group_cols + ["datadate"])
                .with_columns(
                    (col("_price_float") / col("_price_float").shift(1).over(group_cols)).alias(
                        "_ratio"
                    )
                )
                .filter(col("_ratio").is_not_null() & col("_ratio").is_finite())
            )
            if len(with_ratios) > 0:
                ratios = with_ratios["_ratio"]
                # Count extreme ratios (potential errors)
                metrics["ratio_gt_5x"] = int((ratios > 5).sum())
                metrics["ratio_gt_10x"] = int((ratios > 10).sum())
                metrics["ratio_gt_100x"] = int((ratios > 100).sum())
                metrics["ratio_lt_0.2x"] = int((ratios < 0.2).sum())
                metrics["ratio_lt_0.1x"] = int((ratios < 0.1).sum())
                metrics["ratio_lt_0.01x"] = int((ratios < 0.01).sum())
                metrics["ratio_max"] = float(ratios.max())
                metrics["ratio_min"] = float(ratios.min())

        # Log summary
        logger.info(f"Data quality [{stage}]: {n_rows:,} rows, {n_securities:,} securities")
        if "ratio_gt_5x" in metrics:
            logger.info(
                f"  Extreme ratios: >5x={metrics['ratio_gt_5x']}, >10x={metrics['ratio_gt_10x']}, "
                f">100x={metrics['ratio_gt_100x']}, <0.2x={metrics['ratio_lt_0.2x']}, "
                f"<0.1x={metrics['ratio_lt_0.1x']}, <0.01x={metrics['ratio_lt_0.01x']}"
            )
            logger.info(
                f"  Ratio range: min={metrics['ratio_min']:.4f}, max={metrics['ratio_max']:.4f}"
            )

        return metrics

    except Exception as e:
        logger.warning(f"Could not compute data quality metrics for {stage}: {e}")
        return {"stage": stage, "error": str(e)}


def log_correction_summary_by_country(
    corrections_log: pl.LazyFrame | pl.DataFrame,
    country_col: str = "excntry",
) -> None:
    """
    Log correction counts broken down by country.

    Args:
        corrections_log: Log of corrections from Section 6
        country_col: Column with country identifier
    """
    try:
        if isinstance(corrections_log, pl.LazyFrame):
            log_df = corrections_log.collect()
        else:
            log_df = corrections_log

        if country_col not in log_df.columns:
            logger.info("Country column not in corrections log, skipping country breakdown")
            return

        by_country = (
            log_df.group_by(country_col).agg(pl.len().alias("count")).sort("count", descending=True)
        )

        if len(by_country) > 0:
            top_countries = by_country.head(10)
            summary_str = ", ".join(
                f"{row[country_col]}:{row['count']}" for row in top_countries.iter_rows(named=True)
            )
            logger.info(f"Corrections by country (top 10): {summary_str}")
        else:
            logger.info("No corrections to break down by country")

    except Exception as e:
        logger.warning(f"Could not compute country breakdown: {e}")


def log_correction_summary_by_year(
    corrections_log: pl.LazyFrame | pl.DataFrame,
    date_col: str = "datadate",
) -> None:
    """
    Log correction counts broken down by year.

    Args:
        corrections_log: Log of corrections from Section 6
        date_col: Column with date
    """
    try:
        if isinstance(corrections_log, pl.LazyFrame):
            log_df = corrections_log.collect()
        else:
            log_df = corrections_log

        if date_col not in log_df.columns:
            logger.info("Date column not in corrections log, skipping year breakdown")
            return

        by_year = (
            log_df.with_columns(col(date_col).dt.year().alias("_year"))
            .group_by("_year")
            .agg(pl.len().alias("count"))
            .sort("_year")
        )

        if len(by_year) > 0:
            summary_str = ", ".join(
                f"{row['_year']}:{row['count']}" for row in by_year.iter_rows(named=True)
            )
            logger.info(f"Corrections by year: {summary_str}")
        else:
            logger.info("No corrections to break down by year")

    except Exception as e:
        logger.warning(f"Could not compute year breakdown: {e}")


def detect_potential_boundary_errors(
    df: pl.LazyFrame,
    price_col: str = "prccd",
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    ratio_threshold: float = 5.0,
) -> pl.DataFrame:
    """
    Detect potential errors at series boundaries (first/last observations).

    These errors cannot be detected by the spike-and-reversal algorithm because
    they lack a comparison point on one side.

    Args:
        df: LazyFrame with price data
        price_col: Column with price data
        group_cols: Security identifier columns
        sort_col: Date column

    Returns:
        DataFrame with potential boundary errors for manual review
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    try:
        # Stay lazy on a slim projection — only the (small) filtered result
        # is collected, never the full daily frame.
        data = df.select(group_cols + [sort_col, price_col]).sort(group_cols + [sort_col])

        # Add row numbers within each security
        data = data.with_columns(
            pl.col(price_col).count().over(group_cols).alias("_n_obs"),
            pl.col(price_col).cum_count().over(group_cols).alias("_row_num"),
        )

        # Flag first and last observations
        data = data.with_columns(
            ((col("_row_num") == 1) | (col("_row_num") == col("_n_obs"))).alias("_is_boundary"),
            # Ratio to next (for first obs)
            (col(price_col) / col(price_col).shift(-1).over(group_cols)).alias("_ratio_to_next"),
            # Ratio to prior (for last obs)
            (col(price_col) / col(price_col).shift(1).over(group_cols)).alias("_ratio_to_prior"),
        )

        # Find boundary observations with extreme ratios
        boundary_errors = (
            data.filter(
                col("_is_boundary")
                & (
                    (col("_ratio_to_next") > ratio_threshold)
                    | (col("_ratio_to_next") < 1 / ratio_threshold)
                    | (col("_ratio_to_prior") > ratio_threshold)
                    | (col("_ratio_to_prior") < 1 / ratio_threshold)
                )
            )
            .select(
                group_cols
                + [
                    sort_col,
                    price_col,
                    "_row_num",
                    "_n_obs",
                    "_ratio_to_next",
                    "_ratio_to_prior",
                ]
            )
            .collect()
        )

        n_boundary_errors = len(boundary_errors)
        if n_boundary_errors > 0:
            logger.warning(
                f"Found {n_boundary_errors} potential boundary errors "
                f"(first/last obs with ratio > {ratio_threshold}x)"
            )
            # Log a few examples
            examples = boundary_errors.head(5)
            for row in examples.iter_rows(named=True):
                logger.debug(
                    f"  Boundary error: {row[group_cols[0]]}/{row[group_cols[1]]} "
                    f"on {row[sort_col]}: {price_col}={row[price_col]:.2f}, "
                    f"ratio_to_next={row['_ratio_to_next']}, "
                    f"ratio_to_prior={row['_ratio_to_prior']}"
                )
        else:
            logger.info("No potential boundary errors detected")

        return boundary_errors

    except Exception as e:
        logger.warning(f"Could not detect boundary errors: {e}")
        return pl.DataFrame()


def compute_correction_rate(
    corrections_log: pl.LazyFrame | None,
    total_observations: int,
    price_threshold: float = 0.01,
) -> dict:
    """
    Compute the correction rate and compare to Bessembinder benchmark.

    The research note found Bessembinder corrects ~0.02% of observations,
    vs 0.06% for 0.1% winsorization and 0.89% for 1% winsorization.

    Since Section 8c filters will remove observations with price < $0.01,
    corrections on sub-$0.01 prices are excluded from the benchmark comparison
    (they would be deleted anyway).

    Args:
        corrections_log: LazyFrame with corrections (or None if no corrections)
        total_observations: Total number of observations processed
        price_threshold: Minimum price for Section 8c filter (default $0.01).
            Corrections on prices below this are excluded from benchmark comparison.

    Returns:
        Dictionary with correction statistics including both raw and filtered rates
    """
    if corrections_log is None or total_observations == 0:
        return {
            "n_corrections": 0,
            "n_corrections_above_threshold": 0,
            "n_corrections_below_threshold": 0,
            "n_observations": total_observations,
            "correction_rate_pct": 0.0,
            "correction_rate_filtered_pct": 0.0,
            "benchmark_pct": 0.02,
            "vs_benchmark": "N/A",
            "price_threshold": price_threshold,
        }

    # Collect the log once for efficiency
    log_collected = (
        corrections_log.collect() if isinstance(corrections_log, pl.LazyFrame) else corrections_log
    )

    n_corrections = len(log_collected)
    correction_rate = 100 * n_corrections / total_observations

    # Filter out sub-threshold corrections (these would be deleted by Section 8c)
    # Use original_value since that's the price being corrected
    if "original_value" in log_collected.columns:
        above_threshold = log_collected.filter(col("original_value") >= price_threshold)
        n_above = len(above_threshold)
        n_below = n_corrections - n_above
    else:
        # Fallback if column not present
        n_above = n_corrections
        n_below = 0

    correction_rate_filtered = 100 * n_above / total_observations

    # Compare FILTERED rate to benchmark (sub-$0.01 would be deleted anyway)
    benchmark = 0.02
    if correction_rate_filtered < benchmark * 0.5:
        vs_benchmark = "LOW (< 50% of benchmark)"
    elif correction_rate_filtered > benchmark * 2:
        vs_benchmark = "HIGH (> 200% of benchmark)"
    else:
        vs_benchmark = "NORMAL (within 50-200% of benchmark)"

    result = {
        "n_corrections": n_corrections,
        "n_corrections_above_threshold": n_above,
        "n_corrections_below_threshold": n_below,
        "n_observations": total_observations,
        "correction_rate_pct": round(correction_rate, 4),
        "correction_rate_filtered_pct": round(correction_rate_filtered, 4),
        "benchmark_pct": benchmark,
        "vs_benchmark": vs_benchmark,
        "price_threshold": price_threshold,
    }

    # Log both rates
    if n_below > 0:
        logger.info(
            f"Correction rate: {n_corrections:,}/{total_observations:,} = "
            f"{correction_rate:.4f}% (raw), {correction_rate_filtered:.4f}% "
            f"(excluding {n_below} sub-${price_threshold} corrections)"
        )
        logger.info(
            f"Filtered rate vs benchmark: {correction_rate_filtered:.4f}% vs "
            f"{benchmark}% (status: {vs_benchmark})"
        )
    else:
        logger.info(
            f"Correction rate: {n_corrections:,}/{total_observations:,} = "
            f"{correction_rate:.4f}% (benchmark: {benchmark}%, status: {vs_benchmark})"
        )

    return result


def detect_non_reversing_errors(
    df: pl.LazyFrame,
    price_col: str = "prccd",
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    spike_threshold: float = 5.0,
    lookforward_days: int = 21,
) -> pl.DataFrame:
    """
    Detect large price spikes that don't reverse - potential uncorrected errors.

    From the research note: The Bessembinder algorithm fails when errors don't
    reverse within the detection window. Examples include gvkey 165768 where
    price jumped from 0.0145 to 2.54 and persisted.

    Args:
        df: LazyFrame with price data
        price_col: Column with price data
        group_cols: Security identifier columns
        sort_col: Date column
        spike_threshold: Minimum ratio to consider a spike (default 5x)
        lookforward_days: How many days to check for reversal

    Returns:
        DataFrame with suspected non-reversing errors
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    try:
        data = df.sort(group_cols + [sort_col]).collect()

        # Cast to float to handle Decimal types
        if price_col in data.columns:
            data = data.with_columns(col(price_col).cast(pl.Float64).alias("_price"))
        else:
            logger.warning(f"Price column {price_col} not found")
            return pl.DataFrame()

        # Compute ratio to prior day
        data = data.with_columns(
            (col("_price") / col("_price").shift(1).over(group_cols)).alias("_spike_ratio")
        )

        # Find large spikes (up or down)
        spikes = data.filter(
            (col("_spike_ratio") > spike_threshold) | (col("_spike_ratio") < 1 / spike_threshold)
        )

        if len(spikes) == 0:
            logger.info("No large spikes detected")
            return pl.DataFrame()

        # For each spike, check if it reverses within lookforward_days
        non_reversing = []

        for spike_row in spikes.iter_rows(named=True):
            gvkey = spike_row[group_cols[0]]
            iid = spike_row[group_cols[1]]
            spike_date = spike_row[sort_col]
            spike_ratio = spike_row["_spike_ratio"]
            spike_price = spike_row["_price"]

            # Get subsequent prices for this security
            subsequent = data.filter(
                (col(group_cols[0]) == gvkey)
                & (col(group_cols[1]) == iid)
                & (col(sort_col) > spike_date)
            ).head(lookforward_days)

            if len(subsequent) == 0:
                continue

            # Check if price returns to pre-spike level
            pre_spike_price = spike_price / spike_ratio if spike_ratio else None
            if pre_spike_price is None or pre_spike_price == 0:
                continue

            # Calculate max deviation from pre-spike price in subsequent days
            subsequent_prices = subsequent["_price"].to_list()
            max_ratio = max(p / pre_spike_price for p in subsequent_prices if p and pre_spike_price)
            min_ratio = min(p / pre_spike_price for p in subsequent_prices if p and pre_spike_price)

            # If price stays elevated (or depressed), it's non-reversing
            if spike_ratio > spike_threshold:
                # Upward spike - check if it reverses down
                if min_ratio > 0.5:  # Still more than 50% above pre-spike
                    non_reversing.append(
                        {
                            "gvkey": gvkey,
                            "iid": iid,
                            "datadate": spike_date,
                            "spike_ratio": spike_ratio,
                            "pre_spike_price": pre_spike_price,
                            "spike_price": spike_price,
                            "min_subsequent_ratio": min_ratio,
                            "days_checked": len(subsequent),
                            "direction": "up",
                        }
                    )
            else:
                # Downward spike - check if it reverses up
                if max_ratio < 2.0:  # Still less than 2x pre-spike
                    non_reversing.append(
                        {
                            "gvkey": gvkey,
                            "iid": iid,
                            "datadate": spike_date,
                            "spike_ratio": spike_ratio,
                            "pre_spike_price": pre_spike_price,
                            "spike_price": spike_price,
                            "max_subsequent_ratio": max_ratio,
                            "days_checked": len(subsequent),
                            "direction": "down",
                        }
                    )

        if non_reversing:
            result = pl.DataFrame(non_reversing)
            logger.warning(
                f"Found {len(result)} non-reversing spikes (potential uncorrected errors). "
                f"Examples: {result.head(3).to_dicts()}"
            )
            return result
        else:
            logger.info("All detected spikes reversed within lookforward window")
            return pl.DataFrame()

    except Exception as e:
        logger.warning(f"Could not detect non-reversing errors: {e}")
        return pl.DataFrame()


def compare_to_crsp(
    compustat_df: pl.LazyFrame,
    crsp_df: pl.LazyFrame,
    link_table: pl.LazyFrame | None = None,
    compustat_ret_col: str = "ret",
    crsp_ret_col: str = "ret",
    group_cols: list[str] | None = None,
    date_col: str = "datadate",
) -> dict:
    """
    Compare Compustat returns to CRSP returns for validation.

    From the research note, they tracked:
    - % of observations where |diff| <= 0.1%, 1%, 10%
    - By size group (Mega, Large, Small, Micro, Nano)
    - Mean/median absolute differences

    Args:
        compustat_df: LazyFrame with Compustat returns
        crsp_df: LazyFrame with CRSP returns
        link_table: Optional link table for matching (gvkey/iid to permno)
        compustat_ret_col: Column name for Compustat returns
        crsp_ret_col: Column name for CRSP returns
        group_cols: Security identifier columns
        date_col: Date column for matching

    Returns:
        Dictionary with comparison statistics
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    try:
        # Collect data
        comp_data = compustat_df.collect()
        crsp_data = crsp_df.collect()

        # For now, assume data is already matched on gvkey/iid/date
        # In production, would use link_table to match permno to gvkey
        merged = comp_data.join(
            crsp_data.select(group_cols + [date_col, crsp_ret_col]),
            on=group_cols + [date_col],
            how="inner",
            suffix="_crsp",
        )

        if len(merged) == 0:
            logger.warning("No matching observations between Compustat and CRSP")
            return {"n_matched": 0, "error": "No matches found"}

        # Compute absolute difference
        crsp_col_actual = (
            f"{crsp_ret_col}_crsp" if crsp_ret_col in comp_data.columns else crsp_ret_col
        )
        merged = merged.with_columns(
            (col(compustat_ret_col).cast(pl.Float64) - col(crsp_col_actual).cast(pl.Float64))
            .abs()
            .alias("_abs_diff")
        )

        # Filter to valid differences
        valid = merged.filter(col("_abs_diff").is_not_null() & col("_abs_diff").is_finite())
        n_valid = len(valid)

        if n_valid == 0:
            logger.warning("No valid return differences to compute")
            return {"n_matched": len(merged), "n_valid": 0}

        # Compute threshold percentages
        pct_le_0_1 = 100 * len(valid.filter(col("_abs_diff") <= 0.001)) / n_valid
        pct_le_1 = 100 * len(valid.filter(col("_abs_diff") <= 0.01)) / n_valid
        pct_le_10 = 100 * len(valid.filter(col("_abs_diff") <= 0.10)) / n_valid

        # Compute mean/median absolute difference
        mean_abs_diff = valid["_abs_diff"].mean()
        median_abs_diff = valid["_abs_diff"].median()

        result = {
            "n_matched": len(merged),
            "n_valid": n_valid,
            "pct_diff_le_0.1%": round(pct_le_0_1, 2),
            "pct_diff_le_1%": round(pct_le_1, 2),
            "pct_diff_le_10%": round(pct_le_10, 2),
            "mean_abs_diff": round(mean_abs_diff, 6) if mean_abs_diff else None,
            "median_abs_diff": round(median_abs_diff, 6) if median_abs_diff else None,
        }

        logger.info(
            f"CRSP comparison: {n_valid:,} valid pairs, "
            f"≤0.1%: {pct_le_0_1:.1f}%, ≤1%: {pct_le_1:.1f}%, ≤10%: {pct_le_10:.1f}%, "
            f"mean|diff|: {mean_abs_diff:.6f}"
        )

        return result

    except Exception as e:
        logger.warning(f"Could not compare to CRSP: {e}")
        return {"error": str(e)}


def find_partial_corrections(
    corrections_log: pl.LazyFrame,
    compustat_df: pl.LazyFrame,
    crsp_df: pl.LazyFrame,
    ret_col: str = "ret",
    group_cols: list[str] | None = None,
    date_col: str = "datadate",
    mismatch_threshold: float = 0.10,
) -> pl.DataFrame:
    """
    Find corrections that still don't match CRSP - partial corrections.

    From the research note: gvkey 017422 on Dec 22, 2022 was corrected from
    2299 to 1.29, but CRSP shows -0.29. The correction helped but didn't
    fully fix the problem.

    Args:
        corrections_log: LazyFrame with corrections made
        compustat_df: LazyFrame with corrected Compustat data
        crsp_df: LazyFrame with CRSP data
        ret_col: Return column name
        group_cols: Security identifier columns
        date_col: Date column
        mismatch_threshold: Minimum |diff| to consider a mismatch

    Returns:
        DataFrame with partial corrections (corrected but still mismatched)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    try:
        # Get the dates/securities that were corrected
        corrections = corrections_log.collect()
        if len(corrections) == 0:
            return pl.DataFrame()

        # Get corrected Compustat returns
        comp_data = compustat_df.collect()
        crsp_data = crsp_df.collect()

        # Join corrections with both datasets
        corrected_with_crsp = (
            corrections.select(
                group_cols + [date_col, "original_value", "corrected_value", "correction_factor"]
            )
            .join(
                comp_data.select(group_cols + [date_col, ret_col]),
                on=group_cols + [date_col],
                how="left",
            )
            .join(
                crsp_data.select(group_cols + [date_col, col(ret_col).alias("crsp_ret")]),
                on=group_cols + [date_col],
                how="left",
            )
        )

        # Filter to valid comparisons
        valid = corrected_with_crsp.filter(
            col(ret_col).is_not_null() & col("crsp_ret").is_not_null()
        )

        if len(valid) == 0:
            logger.info("No corrected observations have matching CRSP returns")
            return pl.DataFrame()

        # Find where corrected return still doesn't match CRSP
        partial = valid.with_columns(
            (col(ret_col).cast(pl.Float64) - col("crsp_ret").cast(pl.Float64)).abs().alias("_diff")
        ).filter(col("_diff") > mismatch_threshold)

        if len(partial) > 0:
            logger.warning(
                f"Found {len(partial)} partial corrections (corrected but |diff from CRSP| > {mismatch_threshold}). "
                f"Examples: {partial.head(5).to_dicts()}"
            )
        else:
            logger.info("All corrections align with CRSP within threshold")

        return partial

    except Exception as e:
        logger.warning(f"Could not find partial corrections: {e}")
        return pl.DataFrame()


# =============================================================================
# Bessembinder et al. (2023) Data Corrections - Section 8: Additional Filters
# =============================================================================

# Countries with $0.001 price threshold instead of $0.01
LOW_PRICE_THRESHOLD_COUNTRIES = ["BRA", "IDN", "NGA", "TUR"]


def filter_8a_trading_volume(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    volume_col: str = "dolvol",
    percentile: float = 0.02,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8a: Eliminate bottom 2% of stocks by average daily positive volume.

    Args:
        df: LazyFrame with USD-converted data
        group_cols: Security identifier columns
        volume_col: Column with dollar volume
        percentile: Bottom percentile to eliminate (default 0.02 = 2%)

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    # Compute average positive volume per security
    avg_vol = (
        df.filter(col(volume_col) > 0)
        .group_by(group_cols)
        .agg(pl.mean(volume_col).alias("_avg_vol"))
    )

    # Find the percentile cutoff; cross join broadcasts the single-row cutoff
    cutoff = avg_vol.select(pl.col("_avg_vol").quantile(percentile).alias("_cutoff"))
    df = df.join(avg_vol, on=group_cols, how="left")
    df = df.join(cutoff, how="cross")

    # Log removed records
    removed = df.filter(col("_avg_vol") <= col("_cutoff")).select(
        group_cols + ["datadate", pl.lit("8a_trading_volume").alias("filter_reason")]
    )

    # Keep records above cutoff
    df = df.filter((col("_avg_vol") > col("_cutoff")) | col("_avg_vol").is_null())
    df = df.drop(["_avg_vol", "_cutoff"])

    return df, removed


def filter_8b_ajex_qunit(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8b: Drop stocks where AJEXDI=0, or QUNIT changes without currency change
    and |price change| > 50%.

    Args:
        df: LazyFrame with data
        group_cols: Security identifier columns
        sort_col: Date column

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    df = df.sort(group_cols + [sort_col])

    # Identify securities where AJEXDI is ever zero
    has_zero_ajex = (
        df.filter(col("ajexdi") == 0)
        .select(group_cols)
        .unique()
        .with_columns(pl.lit(True).alias("_has_zero_ajex"))
    )

    # Detect QUNIT changes without currency change and large price change.
    # Post-merge data (gen_comp_dsf) has neither qunit nor prccd — qunit is
    # consumed into prc_local = prccd / qunit by the SQL stage — so the
    # QUNIT-change check is only possible when those columns are present.
    schema_names = df.collect_schema().names()
    has_qunit = "qunit" in schema_names
    qunit_helper_cols: list[str] = []
    if has_qunit:
        price_col = "prccd" if "prccd" in schema_names else "prc_local"
        df = df.with_columns(
            [
                col("qunit").shift(1).over(group_cols).alias("_qunit_lag"),
                col("curcdd").shift(1).over(group_cols).alias("_curcdd_lag"),
                col(price_col).shift(1).over(group_cols).alias("_prccd_lag"),
            ]
        )

        df = df.with_columns(
            [
                (
                    (col("qunit") != col("_qunit_lag"))
                    & (col("curcdd") == col("_curcdd_lag"))
                    & (((col(price_col) / col("_prccd_lag")) - 1).abs() > 0.5)
                ).alias("_bad_qunit_change")
            ]
        )
        qunit_helper_cols = ["_qunit_lag", "_curcdd_lag", "_prccd_lag", "_bad_qunit_change"]

        # Identify securities with bad QUNIT changes
        has_bad_qunit = (
            df.filter(col("_bad_qunit_change"))
            .select(group_cols)
            .unique()
            .with_columns(pl.lit(True).alias("_has_bad_qunit"))
        )
    else:
        logger.warning("filter 8b: qunit column not available; skipping QUNIT-change check")
        # Explicit empty frame (a head(0) of df here trips the lazy
        # optimizer's projection pushdown into the later join)
        full_schema = df.collect_schema()
        has_bad_qunit = pl.DataFrame(
            schema={c: full_schema[c] for c in group_cols} | {"_has_bad_qunit": pl.Boolean}
        ).lazy()

    # Join flags
    df = df.join(has_zero_ajex, on=group_cols, how="left")
    df = df.join(has_bad_qunit, on=group_cols, how="left")

    # Log removed records
    removed = df.filter(
        col("_has_zero_ajex").fill_null(False) | col("_has_bad_qunit").fill_null(False)
    ).select(
        group_cols
        + [
            sort_col,
            pl.when(col("_has_zero_ajex").fill_null(False))
            .then(pl.lit("8b_ajex_zero"))
            .otherwise(pl.lit("8b_qunit_change"))
            .alias("filter_reason"),
        ]
    )

    # Filter out bad securities
    df = df.filter(
        col("_has_zero_ajex").fill_null(False).not_()
        & col("_has_bad_qunit").fill_null(False).not_()
    )

    # Clean up
    df = df.drop(qunit_helper_cols + ["_has_zero_ajex", "_has_bad_qunit"])

    return df, removed


def filter_8c_low_price_me(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    country_col: str = "excntry",
    me_threshold: float = 1.0,  # $1M USD
    price_threshold: float = 0.01,  # $0.01 USD
    price_threshold_low: float = 0.001,  # $0.001 for BRA, IDN, NGA, TUR
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8c: Delete remaining history if ME < $1M or price < $0.01 USD.
    Exception: $0.001 threshold for BRA, IDN, NGA, TUR.

    Also delete entire stock if initial observation has low price/ME.

    Args:
        df: LazyFrame with USD-converted data
        group_cols: Security identifier columns
        sort_col: Date column
        country_col: Country identifier column
        me_threshold: Minimum market cap in millions USD
        price_threshold: Minimum price in USD (standard)
        price_threshold_low: Minimum price for exception countries

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    df = df.sort(group_cols + [sort_col])

    # Determine price threshold based on country
    df = df.with_columns(
        pl.when(col(country_col).is_in(LOW_PRICE_THRESHOLD_COUNTRIES))
        .then(pl.lit(price_threshold_low))
        .otherwise(pl.lit(price_threshold))
        .alias("_price_threshold")
    )

    # Flag observations that breach thresholds
    df = df.with_columns(
        ((col("me") < me_threshold) | (col("prc") < col("_price_threshold"))).alias(
            "_breaches_threshold"
        )
    )

    # For each security, find the first breach date
    first_breach = (
        df.filter(col("_breaches_threshold"))
        .group_by(group_cols)
        .agg(pl.min(sort_col).alias("_first_breach_date"))
    )

    df = df.join(first_breach, on=group_cols, how="left")

    # Also check if initial observation breaches (delete entire stock)
    df = df.with_columns(
        (col(sort_col) == col(sort_col).min().over(group_cols)).alias("_is_first_obs")
    )

    initial_breach = (
        df.filter(col("_is_first_obs") & col("_breaches_threshold"))
        .select(group_cols)
        .unique()
        .with_columns(pl.lit(True).alias("_initial_breach"))
    )

    df = df.join(initial_breach, on=group_cols, how="left")

    # Log removed records
    removed = df.filter(
        col("_initial_breach").fill_null(False)
        | (col("_first_breach_date").is_not_null() & (col(sort_col) >= col("_first_breach_date")))
    ).select(
        group_cols
        + [
            sort_col,
            pl.when(col("_initial_breach").fill_null(False))
            .then(pl.lit("8c_initial_low_price_me"))
            .otherwise(pl.lit("8c_low_price_me"))
            .alias("filter_reason"),
        ]
    )

    # Keep only records before breach (and not initial breach stocks)
    df = df.filter(
        col("_initial_breach").fill_null(False).not_()
        & (col("_first_breach_date").is_null() | (col(sort_col) < col("_first_breach_date")))
    )

    # Clean up
    df = df.drop(
        [
            "_price_threshold",
            "_breaches_threshold",
            "_first_breach_date",
            "_is_first_obs",
            "_initial_breach",
        ]
    )

    return df, removed


def filter_8d_data_gaps(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    max_gap_days: int = 231,  # ~11 months in trading days
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8d: Drop first observation after gaps > 231 trading days (~11 months).

    Args:
        df: LazyFrame with data
        group_cols: Security identifier columns
        sort_col: Date column
        max_gap_days: Maximum gap in trading days before dropping

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    df = df.sort(group_cols + [sort_col])

    # Compute row number within each security to measure gaps
    df = df.with_columns(
        pl.arange(0, pl.len()).over(group_cols).alias("_row_num"),
        col(sort_col).shift(1).over(group_cols).alias("_prev_date"),
    )

    # Calculate gap in calendar days
    df = df.with_columns(
        (col(sort_col).cast(pl.Date) - col("_prev_date").cast(pl.Date))
        .dt.total_days()
        .alias("_gap_days")
    )

    # Flag observations after large gaps
    # Using calendar days, 11 months ≈ 335 days, but we use trading days ≈ 231
    # Convert to approximate calendar days: 231 * 365/252 ≈ 335
    calendar_gap_threshold = int(max_gap_days * 365 / 252)

    df = df.with_columns((col("_gap_days") > calendar_gap_threshold).alias("_after_large_gap"))

    # Log removed records
    removed = df.filter(col("_after_large_gap").fill_null(False)).select(
        group_cols + [sort_col, pl.lit("8d_data_gap").alias("filter_reason")]
    )

    # Filter out records after large gaps
    df = df.filter(col("_after_large_gap").fill_null(False).not_())

    # Clean up
    df = df.drop(["_row_num", "_prev_date", "_gap_days", "_after_large_gap"])

    return df, removed


def filter_8e_adjcsho_jumps(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    country_col: str = "excntry",
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8e: Detect large adjCSHO changes without commensurate ME changes.

    Thresholds:
    - Standard: ≥5x adjCSHO with ≥2.5x ME, or ≤0.2x adjCSHO with ≤0.4x ME
    - China: ≥50x adjCSHO with ≥25x ME (reverse mergers common)

    For early observations (first 24 months or 20%), delete up to and including jump.
    For later observations, adjust adjCSHO to smooth the jump.

    Args:
        df: LazyFrame with data
        group_cols: Security identifier columns
        sort_col: Date column
        country_col: Country identifier

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    df = df.sort(group_cols + [sort_col])

    # Compute adjCSHO = CSHOC * AJEXDI
    df = df.with_columns((col("cshoc") * col("ajexdi")).alias("_adjcsho"))

    # Compute ratios
    df = df.with_columns(
        [
            (col("_adjcsho") / col("_adjcsho").shift(1).over(group_cols)).alias("_adjcsho_ratio"),
            (col("me") / col("me").shift(1).over(group_cols)).alias("_me_ratio"),
            pl.arange(0, pl.len()).over(group_cols).alias("_obs_num"),
            pl.len().over(group_cols).alias("_total_obs"),
        ]
    )

    # Determine thresholds based on country (China gets higher thresholds)
    is_china = col(country_col) == "CHN"

    # Detect up-jumps
    up_jump = (
        pl.when(is_china)
        .then((col("_adjcsho_ratio") >= 50) & (col("_me_ratio") >= 25))
        .otherwise((col("_adjcsho_ratio") >= 5) & (col("_me_ratio") >= 2.5))
    )

    # Detect down-jumps
    down_jump = (col("_adjcsho_ratio") <= 0.2) & (col("_me_ratio") <= 0.4)

    df = df.with_columns((up_jump | down_jump).alias("_is_jump"))

    # Determine if jump is in early period (first 24 months or 20% of obs)
    # For daily data, 24 months ≈ 504 trading days
    early_period = (col("_obs_num") < 504) | (col("_obs_num") < (col("_total_obs") * 0.2))

    df = df.with_columns((col("_is_jump") & early_period).alias("_early_jump"))

    # For early jumps, find securities and their first jump date
    early_jump_info = (
        df.filter(col("_early_jump"))
        .group_by(group_cols)
        .agg(pl.max("_obs_num").alias("_delete_through_obs"))
    )

    df = df.join(early_jump_info, on=group_cols, how="left")

    # Log removed records (early period jumps - delete up to and including)
    removed = df.filter(
        col("_delete_through_obs").is_not_null() & (col("_obs_num") <= col("_delete_through_obs"))
    ).select(group_cols + [sort_col, pl.lit("8e_adjcsho_jump_early").alias("filter_reason")])

    # Filter out early jump observations
    df = df.filter(
        col("_delete_through_obs").is_null() | (col("_obs_num") > col("_delete_through_obs"))
    )

    # For later jumps, we would adjust adjCSHO (simplified: just flag for now)
    # Full implementation would smooth paired jumps or scale unpaired jumps

    # Clean up
    df = df.drop(
        [
            "_adjcsho",
            "_adjcsho_ratio",
            "_me_ratio",
            "_obs_num",
            "_total_obs",
            "_is_jump",
            "_early_jump",
            "_delete_through_obs",
        ]
    )

    return df, removed


def filter_8f_me_jumps(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8f: Detect ME jumps not supported by returns.

    Thresholds:
    - Up-jump: ME ratio > 10 but RET < 2 (200%)
    - Down-jump: ME ratio < 0.1 but RET > -0.5 (-50%)

    Args:
        df: LazyFrame with data
        group_cols: Security identifier columns
        sort_col: Date column

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    df = df.sort(group_cols + [sort_col])

    # Compute ME ratio and get return
    df = df.with_columns(
        [
            (col("me") / col("me").shift(1).over(group_cols)).alias("_me_ratio"),
            # Compute return from ri (return index) if available
            (col("ri") / col("ri").shift(1).over(group_cols) - 1).alias("_ret"),
            pl.arange(0, pl.len()).over(group_cols).alias("_obs_num"),
            pl.len().over(group_cols).alias("_total_obs"),
        ]
    )

    # Detect jumps
    up_jump = (col("_me_ratio") > 10) & (col("_ret") < 2)
    down_jump = (col("_me_ratio") < 0.1) & (col("_ret") > -0.5)

    df = df.with_columns((up_jump | down_jump).alias("_is_jump"))

    # Early period check (first 24 months or 20%)
    early_period = (col("_obs_num") < 504) | (col("_obs_num") < (col("_total_obs") * 0.2))

    df = df.with_columns((col("_is_jump") & early_period).alias("_early_jump"))

    # For early jumps, delete up to and including
    early_jump_info = (
        df.filter(col("_early_jump"))
        .group_by(group_cols)
        .agg(pl.max("_obs_num").alias("_delete_through_obs"))
    )

    df = df.join(early_jump_info, on=group_cols, how="left")

    # Log removed records
    removed = df.filter(
        col("_delete_through_obs").is_not_null() & (col("_obs_num") <= col("_delete_through_obs"))
    ).select(group_cols + [sort_col, pl.lit("8f_me_jump_early").alias("filter_reason")])

    # Filter out early jump observations
    df = df.filter(
        col("_delete_through_obs").is_null() | (col("_obs_num") > col("_delete_through_obs"))
    )

    # Clean up
    df = df.drop(
        [
            "_me_ratio",
            "_ret",
            "_obs_num",
            "_total_obs",
            "_is_jump",
            "_early_jump",
            "_delete_through_obs",
        ]
    )

    return df, removed


def filter_8g_return_filter(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8g: Delete observations where |RET| > 80% but |ME change| < 50%.

    Args:
        df: LazyFrame with data
        group_cols: Security identifier columns
        sort_col: Date column

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    df = df.sort(group_cols + [sort_col])

    # Compute return and ME change
    df = df.with_columns(
        [
            (col("ri") / col("ri").shift(1).over(group_cols) - 1).alias("_ret"),
            (col("me") / col("me").shift(1).over(group_cols) - 1).alias("_me_change"),
        ]
    )

    # Flag problematic observations
    df = df.with_columns(
        ((col("_ret").abs() > 0.8) & (col("_me_change").abs() < 0.5)).alias("_bad_return")
    )

    # Log removed records
    removed = df.filter(col("_bad_return").fill_null(False)).select(
        group_cols + [sort_col, pl.lit("8g_return_filter").alias("filter_reason")]
    )

    # Filter out bad observations
    df = df.filter(col("_bad_return").fill_null(False).not_())

    # Clean up
    df = df.drop(["_ret", "_me_change", "_bad_return"])

    return df, removed


def filter_8h_initial_errors(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Filter 8h: Delete first 3 observations if adjPRC or ME ratio > 10x.

    Args:
        df: LazyFrame with data
        group_cols: Security identifier columns
        sort_col: Date column

    Returns:
        Tuple of (filtered_df, removed_records_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    df = df.sort(group_cols + [sort_col])

    # Compute observation number and ratios
    df = df.with_columns(
        [
            pl.arange(0, pl.len()).over(group_cols).alias("_obs_num"),
            (col("prc") / col("prc").shift(1).over(group_cols)).alias("_prc_ratio"),
            (col("me") / col("me").shift(1).over(group_cols)).alias("_me_ratio"),
        ]
    )

    # Flag initial observations with large jumps (obs 1, 2, or 3 with ratio > 10)
    # Note: obs_num is 0-indexed, so first 3 obs are 0, 1, 2
    df = df.with_columns(
        (
            (col("_obs_num") <= 2)
            & (
                (col("_prc_ratio") > 10)
                | (col("_prc_ratio") < 0.1)
                | (col("_me_ratio") > 10)
                | (col("_me_ratio") < 0.1)
            )
        ).alias("_initial_error")
    )

    # Log removed records
    removed = df.filter(col("_initial_error").fill_null(False)).select(
        group_cols + [sort_col, pl.lit("8h_initial_error").alias("filter_reason")]
    )

    # Filter out initial errors
    df = df.filter(col("_initial_error").fill_null(False).not_())

    # Clean up
    df = df.drop(["_obs_num", "_prc_ratio", "_me_ratio", "_initial_error"])

    return df, removed


_REASON_BY_CODE = [
    "8a_trading_volume",
    "8b_ajex_zero",
    "8b_qunit_change",
    "8c_initial_low_price_me",
    "8c_low_price_me",
    "8d_data_gap",
    "8e_adjcsho_jump_early",
    "8f_me_jump_early",
    "8g_return_filter",
    "8h_initial_error",
]


def _apply_section8_slim(
    df: pl.LazyFrame,
    group_cols: list[str],
    sort_col: str,
    country_col: str,
    spill_dir: Path,
) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
    """
    Description:
        Memory-bounded Section 8 path for full-size cluster data. The polars
        filter chain's shift/cum_count-over-group plans fall back to
        in-memory execution on the whole frame; this path instead sorts the
        input once with a streaming sink, runs the entire filter chain as a
        single numba pass per security (sequential-survivor semantics
        preserved inside the kernel), and reattaches the keep decision via a
        lazy horizontal concat.
    Steps:
        1) Streaming-sort input to spill_dir/__bess_s8_sorted.parquet.
        2) Collect group keys once; build group-start offsets.
        3) Filter 8a's global cross-security decision: per-security mean of
           positive volume + 2% quantile via the same polars expressions as
           the reference, mapped to a per-security bool.
        4) section8_all kernel -> per-row removal-reason codes.
        5) Removal log from gathered keys + reason strings; kept frame =
           sorted scan + reason column, filtered lazily.
    Output:
        (filtered LazyFrame, removal log LazyFrame or None).
    """
    sorted_path = spill_dir / "__bess_s8_sorted.parquet"
    print("[section8] streaming-sorting input to spill file...", flush=True)
    df.sort(group_cols + [sort_col]).sink_parquet(sorted_path)
    lf = pl.scan_parquet(sorted_path)

    print("[section8] building group index...", flush=True)
    keys = lf.select(group_cols + [sort_col]).collect()
    starts = _group_starts(keys, group_cols)

    # Filter 8a's global decision (the only cross-security statistic): same
    # expressions as filter_8a_trading_volume, evaluated once on the full
    # panel; securities with no positive volume (null mean) are KEPT.
    print("[section8] computing 8a volume decisions...", flush=True)
    avg_vol = (
        lf.filter(col("dolvol") > 0).group_by(group_cols).agg(pl.mean("dolvol").alias("_avg_vol"))
    )
    decisions = (
        keys.lazy()
        .group_by(group_cols)
        .len()
        .join(avg_vol, on=group_cols, how="left")
        .join(avg_vol.select(pl.col("_avg_vol").quantile(0.02).alias("_cutoff")), how="cross")
        # joins do not preserve row order: re-sort into group_starts order
        .sort(group_cols)
        .select((pl.col("_avg_vol") <= pl.col("_cutoff")).fill_null(False).alias("_remove"))
        .collect()
    )
    remove_8a = decisions["_remove"].to_numpy().copy()

    def load(name: str) -> np.ndarray:
        return lf.select(pl.col(name).cast(pl.Float64)).collect()[name].to_numpy().copy()

    print("[section8] running filter kernel...", flush=True)
    reason = np.full(keys.height, bk.R_NULL, dtype=np.int8)
    bk.section8_all(
        starts,
        reason,
        remove_8a,
        load("ajexdi"),
        load("prc"),
        load("me"),
        load("ri"),
        load("cshoc"),
        keys[sort_col].cast(pl.Date).to_physical().cast(pl.Int32).to_numpy().copy(),
        lf.select(col(country_col).is_in(LOW_PRICE_THRESHOLD_COUNTRIES).fill_null(False))
        .collect()
        .to_series()
        .to_numpy()
        .copy(),
        lf.select((col(country_col) == "CHN").fill_null(False))
        .collect()
        .to_series()
        .to_numpy()
        .copy(),
    )

    removed_idx = np.flatnonzero(reason != bk.R_NULL)
    removed = (
        keys[removed_idx]
        .with_columns(
            _codes_to_strings(reason[removed_idx], _REASON_BY_CODE).alias("filter_reason")
        )
        .lazy()
        if len(removed_idx)
        else None
    )

    reason_path = spill_dir / "__bess_s8_reason.parquet"
    pl.DataFrame({"_reason": reason}).write_parquet(reason_path)
    kept = (
        pl.concat([pl.scan_parquet(sorted_path), pl.scan_parquet(reason_path)], how="horizontal")
        .filter(col("_reason") == bk.R_NULL)
        .drop("_reason")
    )
    return kept, removed


def apply_bessembinder_section8(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    country_col: str = "excntry",
    spill_dir: Path | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Apply all Bessembinder Section 8 filters in sequence.

    Filter order:
    8a. Trading volume (eliminate bottom 2%)
    8b. AJEX/QUNIT (drop if AJEXDI=0 or bad QUNIT change)
    8c. Low price/ME (delete history after breach)
    8d. Data gaps (drop first obs after gaps > 11 months)
    8e. adjCSHO jumps (detect and handle)
    8f. ME jumps (detect and handle)
    8g. Return filter (delete mismatched return/ME)
    8h. Initial errors (delete first 3 obs if large jumps)
    8i. Delisting returns (already implemented in codebase)

    Args:
        df: LazyFrame with USD-converted data
        group_cols: Security identifier columns
        sort_col: Date column
        country_col: Country identifier column
        spill_dir: When set, run the whole chain as a single numba pass per
            security (_apply_section8_slim) with spill files in this
            directory — required for full-size cluster data. When None
            (default), the polars filter chain runs (fine for small/test
            data; it is the reference implementation).

    Returns:
        Tuple of (filtered_df, all_removed_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]

    if spill_dir is not None:
        return _apply_section8_slim(df, group_cols, sort_col, country_col, spill_dir)

    all_removed = []

    # 8a: Trading volume filter
    df, removed = filter_8a_trading_volume(df, group_cols)
    if removed is not None:
        all_removed.append(removed)

    # 8b: AJEX/QUNIT filter
    df, removed = filter_8b_ajex_qunit(df, group_cols, sort_col)
    if removed is not None:
        all_removed.append(removed)

    # 8c: Low price/ME filter
    df, removed = filter_8c_low_price_me(df, group_cols, sort_col, country_col)
    if removed is not None:
        all_removed.append(removed)

    # 8d: Data gaps filter
    df, removed = filter_8d_data_gaps(df, group_cols, sort_col)
    if removed is not None:
        all_removed.append(removed)

    # 8e: adjCSHO jumps filter
    df, removed = filter_8e_adjcsho_jumps(df, group_cols, sort_col, country_col)
    if removed is not None:
        all_removed.append(removed)

    # 8f: ME jumps filter
    df, removed = filter_8f_me_jumps(df, group_cols, sort_col)
    if removed is not None:
        all_removed.append(removed)

    # 8g: Return filter
    df, removed = filter_8g_return_filter(df, group_cols, sort_col)
    if removed is not None:
        all_removed.append(removed)

    # 8h: Initial errors filter
    df, removed = filter_8h_initial_errors(df, group_cols, sort_col)
    if removed is not None:
        all_removed.append(removed)

    # 8i: Delisting returns - already implemented in gen_delist_df()

    # Combine all removal logs
    if all_removed:
        all_removed_log = pl.concat(all_removed, how="vertical_relaxed")

        # Log summary of removals by filter
        try:
            log_collected = (
                all_removed_log.collect()
                if isinstance(all_removed_log, pl.LazyFrame)
                else all_removed_log
            )
            if "filter_reason" in log_collected.columns:
                summary = (
                    log_collected.group_by("filter_reason")
                    .agg(pl.len().alias("count"))
                    .sort("count", descending=True)
                )
                total_removed = len(log_collected)
                summary_str = ", ".join(
                    f"{row['filter_reason']}:{row['count']}"
                    for row in summary.iter_rows(named=True)
                )
                logger.info(
                    f"Section 8 filters removed {total_removed} observations ({summary_str})"
                )
            else:
                logger.info(f"Section 8 filters removed {len(log_collected)} observations")
            all_removed_log = log_collected.lazy()
        except Exception as e:
            logger.warning(f"Could not summarize Section 8 filter removals: {e}")
    else:
        all_removed_log = None
        logger.info("Section 8 filters: no observations removed")

    return df, all_removed_log
