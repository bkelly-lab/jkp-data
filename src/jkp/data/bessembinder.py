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

import polars as pl
from polars import col

logger = logging.getLogger(__name__)

# =============================================================================
# Bessembinder et al. (2023) Data Corrections - Section 6: Decimal Errors
# =============================================================================


def _validate_corrections_no_cascading(
    df: pl.LazyFrame,
    corrections_log: pl.LazyFrame,
    col_name: str,
    group_cols: list[str],
    sort_col: str,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Validate corrections to prevent cascading errors.

    The cascading error bug occurs when a correct value is incorrectly flagged
    because BOTH of its endpoint POSITIONS are themselves errors. This function
    identifies and rejects such false positives.

    Algorithm:
    1. Build a set of all positions (gvkey, iid, date) flagged for correction
    2. For each correction, determine its endpoint positions (date ± window_size)
    3. Check if those endpoint positions are ALSO flagged for correction
    4. If BOTH endpoints are flagged → reject (likely false positive)
    5. If at least one endpoint is clean → keep (likely true error)
    6. Iterate until no more rejections (handles chains of false positives)

    Key insight: A true error has at least one clean endpoint (where the "true"
    price level is). A false positive is sandwiched between two errors, so both
    its endpoints are error positions.

    Args:
        df: LazyFrame with the data (after detection, before correction applied)
        corrections_log: LazyFrame with detected corrections
        col_name: Column being corrected
        group_cols: Security identifier columns
        sort_col: Date column

    Returns:
        Tuple of:
        - df: LazyFrame with invalid correction factors reset to 1.0
        - valid_log: LazyFrame with valid corrections only
        - rejected_log: LazyFrame with rejected corrections and rejection reason
    """
    factor_col = f"{col_name}_correction_factor"
    error_type_col = f"{col_name}_error_type"

    # Map each flagged correction to its row position within the security.
    # df is sorted by group_cols + sort_col upstream (detection functions sort),
    # so within-group row order equals date order. Only the corrections log is
    # ever materialized — the full data frame stays lazy throughout.
    pos_index = df.select(group_cols + [sort_col]).with_columns(
        pl.int_range(pl.len()).over(group_cols).alias("_pos")
    )
    log_df = corrections_log.join(pos_index, on=group_cols + [sort_col], how="left").collect()

    if len(log_df) == 0:
        # No corrections to validate
        return df, corrections_log, pl.LazyFrame()

    print(f"    [validate] {col_name}: {len(log_df)} corrections to validate", flush=True)

    orig_schema = corrections_log.collect_schema()

    # Endpoint positions: pos ± window_size (single-period window_size is 1)
    log_df = log_df.with_columns(
        (col("_pos") - col("window_size").fill_null(1)).alias("_lpos"),
        (col("_pos") + col("window_size").fill_null(1)).alias("_rpos"),
    )

    # Iteratively reject false positives until stable
    current_log = log_df
    rejected_logs = []
    max_iterations = 10  # Safety limit

    for iteration in range(max_iterations):
        if len(current_log) == 0:
            break

        # A correction is rejected when BOTH endpoint positions are themselves
        # flagged (likely false positive sandwiched between two errors).
        flagged = current_log.select(group_cols + ["_pos"])
        rejected = current_log.join(
            flagged.rename({"_pos": "_lpos"}), on=group_cols + ["_lpos"], how="semi"
        ).join(flagged.rename({"_pos": "_rpos"}), on=group_cols + ["_rpos"], how="semi")

        if len(rejected) == 0:
            # No more rejections, we're done
            break

        rejected_logs.append(
            rejected.with_columns(pl.lit("both_endpoints_flagged").alias("rejection_reason"))
        )
        current_log = current_log.join(rejected, on=group_cols + [sort_col, "variable"], how="anti")

        print(
            f"    [validate] {col_name}: iteration {iteration + 1}, "
            f"rejected {len(rejected)} corrections",
            flush=True,
        )

    # Build final results
    helper_cols = ["_pos", "_lpos", "_rpos"]
    valid_corrections = current_log.drop(helper_cols)
    rejected_corrections = (
        pl.concat(rejected_logs, how="vertical").drop(helper_cols)
        if rejected_logs
        else pl.DataFrame(schema=dict(orig_schema) | {"rejection_reason": pl.Utf8})
    )

    # Log rejection statistics
    n_rejected = len(rejected_corrections)
    n_total = len(log_df)
    if n_rejected > 0:
        logger.info(f"{col_name}: rejected {n_rejected}/{n_total} cascading false positives")
        print(
            f"    [validate] {col_name}: rejected {n_rejected}/{n_total} cascading false positives",
            flush=True,
        )

    # Reset invalid corrections on the main dataframe — lazy join + when/otherwise
    # instead of materializing the full frame.
    if n_rejected > 0:
        rejected_keys = (
            rejected_corrections.lazy()
            .select(group_cols + [sort_col])
            .with_columns(pl.lit(True).alias("_rejected"))
        )
        df = (
            df.join(rejected_keys, on=group_cols + [sort_col], how="left")
            .with_columns(
                pl.when(col("_rejected"))
                .then(pl.lit(1.0))  # Reset to no correction
                .otherwise(col(factor_col))
                .alias(factor_col),
                pl.when(col("_rejected"))
                .then(pl.lit(None).cast(pl.Utf8))
                .otherwise(col(error_type_col))
                .alias(error_type_col),
            )
            .drop("_rejected")
        )

    return df, valid_corrections.lazy(), rejected_corrections.lazy()


def _detect_decimal_error_single_period(
    df: pl.LazyFrame,
    col_name: str,
    group_cols: list[str],
    sort_col: str,
) -> pl.LazyFrame:
    """
    Detect single-period decimal shift errors (Bessembinder Section 6a).

    A decimal error is detected when the value spikes relative to both its
    prior and next values. For example, if price goes 8.56 -> 69.50 -> 7.32,
    the middle value likely has a decimal error.

    Detection rules (for values that are too HIGH):
    - If ratio to prior > 5 AND ratio to next > 5: error magnitude is 10x
    - If both ratios > 50: error magnitude is 100x
    - If both ratios > 500: error magnitude is 1000x

    Analogous rules apply for values that are too LOW (ratios < 0.2, 0.02, 0.002).

    Args:
        df: LazyFrame with the data
        col_name: Column to check for errors
        group_cols: Columns defining the security (e.g., ['gvkey', 'iid'])
        sort_col: Column to sort by within groups (e.g., 'datadate')

    Returns:
        LazyFrame with additional columns:
        - {col_name}_correction_factor: multiplier to apply (1.0 = no correction)
        - {col_name}_error_type: 'high_10x', 'high_100x', 'high_1000x',
                                 'low_10x', 'low_100x', 'low_1000x', or None
        - {col_name}_window_type: 'single' for single-period detection
        - {col_name}_endpoint_left: value of prior observation (debug)
        - {col_name}_endpoint_right: value of next observation (debug)
        - {col_name}_ratio_to_left: ratio to prior value (debug)
        - {col_name}_ratio_to_right: ratio to next value (debug)
    """
    factor_col = f"{col_name}_correction_factor"
    error_type_col = f"{col_name}_error_type"
    window_type_col = f"{col_name}_window_type"
    endpoint_left_col = f"{col_name}_endpoint_left"
    endpoint_right_col = f"{col_name}_endpoint_right"
    ratio_left_col = f"{col_name}_ratio_to_left"
    ratio_right_col = f"{col_name}_ratio_to_right"

    df = df.sort(group_cols + [sort_col])

    # Replace zero values with null to avoid division by zero errors
    df = df.with_columns(
        pl.when(col(col_name) == 0).then(pl.lit(None)).otherwise(col(col_name)).alias(col_name)
    )

    # Compute endpoint values and ratios within each security
    df = df.with_columns(
        [
            col(col_name).shift(1).over(group_cols).alias(endpoint_left_col),
            col(col_name).shift(-1).over(group_cols).alias(endpoint_right_col),
            (col(col_name) / col(col_name).shift(1).over(group_cols)).alias("_ratio_to_prior"),
            (col(col_name) / col(col_name).shift(-1).over(group_cols)).alias("_ratio_to_next"),
        ]
    )

    # Detect errors and determine correction factor
    # High errors: both ratios > threshold -> divide by 10^N
    # Low errors: both ratios < 1/threshold -> multiply by 10^N
    df = df.with_columns(
        [
            pl.when((col("_ratio_to_prior") > 500) & (col("_ratio_to_next") > 500))
            .then(pl.lit(0.001))  # divide by 1000
            .when((col("_ratio_to_prior") > 50) & (col("_ratio_to_next") > 50))
            .then(pl.lit(0.01))  # divide by 100
            .when((col("_ratio_to_prior") > 5) & (col("_ratio_to_next") > 5))
            .then(pl.lit(0.1))  # divide by 10
            .when((col("_ratio_to_prior") < 0.002) & (col("_ratio_to_next") < 0.002))
            .then(pl.lit(1000.0))  # multiply by 1000
            .when((col("_ratio_to_prior") < 0.02) & (col("_ratio_to_next") < 0.02))
            .then(pl.lit(100.0))  # multiply by 100
            .when((col("_ratio_to_prior") < 0.2) & (col("_ratio_to_next") < 0.2))
            .then(pl.lit(10.0))  # multiply by 10
            .otherwise(pl.lit(1.0))
            .alias(factor_col),
            # Detailed error type with magnitude
            pl.when((col("_ratio_to_prior") > 500) & (col("_ratio_to_next") > 500))
            .then(pl.lit("high_1000x"))
            .when((col("_ratio_to_prior") > 50) & (col("_ratio_to_next") > 50))
            .then(pl.lit("high_100x"))
            .when((col("_ratio_to_prior") > 5) & (col("_ratio_to_next") > 5))
            .then(pl.lit("high_10x"))
            .when((col("_ratio_to_prior") < 0.002) & (col("_ratio_to_next") < 0.002))
            .then(pl.lit("low_1000x"))
            .when((col("_ratio_to_prior") < 0.02) & (col("_ratio_to_next") < 0.02))
            .then(pl.lit("low_100x"))
            .when((col("_ratio_to_prior") < 0.2) & (col("_ratio_to_next") < 0.2))
            .then(pl.lit("low_10x"))
            .otherwise(pl.lit(None))
            .alias(error_type_col),
            # Window type for single-period detection
            pl.when(
                (col("_ratio_to_prior") > 5) & (col("_ratio_to_next") > 5)
                | (col("_ratio_to_prior") < 0.2) & (col("_ratio_to_next") < 0.2)
            )
            .then(pl.lit("single"))
            .otherwise(pl.lit(None))
            .alias(window_type_col),
            # Keep ratios for debugging (only when correction applied)
            pl.when(
                (col("_ratio_to_prior") > 5) & (col("_ratio_to_next") > 5)
                | (col("_ratio_to_prior") < 0.2) & (col("_ratio_to_next") < 0.2)
            )
            .then(col("_ratio_to_prior"))
            .otherwise(pl.lit(None))
            .alias(ratio_left_col),
            pl.when(
                (col("_ratio_to_prior") > 5) & (col("_ratio_to_next") > 5)
                | (col("_ratio_to_prior") < 0.2) & (col("_ratio_to_next") < 0.2)
            )
            .then(col("_ratio_to_next"))
            .otherwise(pl.lit(None))
            .alias(ratio_right_col),
            # Keep endpoint values only when correction applied
            pl.when(
                (col("_ratio_to_prior") > 5) & (col("_ratio_to_next") > 5)
                | (col("_ratio_to_prior") < 0.2) & (col("_ratio_to_next") < 0.2)
            )
            .then(col(endpoint_left_col))
            .otherwise(pl.lit(None))
            .alias(endpoint_left_col),
            pl.when(
                (col("_ratio_to_prior") > 5) & (col("_ratio_to_next") > 5)
                | (col("_ratio_to_prior") < 0.2) & (col("_ratio_to_next") < 0.2)
            )
            .then(col(endpoint_right_col))
            .otherwise(pl.lit(None))
            .alias(endpoint_right_col),
        ]
    )

    # Clean up temporary columns
    df = df.drop(["_ratio_to_prior", "_ratio_to_next"])

    return df


def _propagate_interior(
    df: pl.LazyFrame,
    col_name: str,
    marker_prefix: str,
    offsets: list[int],
    group_cols: list[str],
    nlag: int,
) -> pl.LazyFrame:
    """
    Propagate detection markers to all interior positions in a single pass.

    Replaces the previous per-offset sequential update loop. That loop was
    first-write-wins in offset order: each iteration only wrote where the
    target column was still unset (factor == 1.0 / others null), and marker
    columns are immutable during propagation, so the first non-null shifted
    marker in offset order won. pl.coalesce over the shifted markers in the
    same offset order reproduces that exactly, in one with_columns instead of
    len(offsets) sequential passes.

    Guards per column (matching the original loop):
    - factor: == 1.0 (1.0 is non-null, so a bare coalesce would never write)
    - window: source is the literal nlag, gated on the detection marker
    - all others: null-gated, so coalesce([existing, first_hit]) is exact
    """
    factor_col = f"{col_name}_correction_factor"
    window_col = f"{col_name}_window_size"

    def first_hit(marker: str) -> pl.Expr:
        return pl.coalesce(
            [col(f"{marker_prefix}{marker}").shift(o).over(group_cols) for o in offsets]
        )

    # (target column suffix, marker suffix) pairs sharing the null-gated form
    null_gated = [
        ("error_type", "type"),
        ("window_type", "wtype"),
        ("endpoint_left", "ep_left"),
        ("endpoint_right", "ep_right"),
        ("ratio_to_left", "ratio_left"),
        ("ratio_to_right", "ratio_right"),
        ("variation_ratio", "variation"),
    ]
    return df.with_columns(
        [
            pl.when((col(factor_col) == 1.0) & first_hit("det").is_not_null())
            .then(first_hit("det"))
            .otherwise(col(factor_col))
            .alias(factor_col),
            pl.when(col(window_col).is_null() & first_hit("det").is_not_null())
            .then(pl.lit(nlag))
            .otherwise(col(window_col))
            .alias(window_col),
        ]
        + [
            pl.coalesce([col(f"{col_name}_{target}"), first_hit(marker)]).alias(
                f"{col_name}_{target}"
            )
            for target, marker in null_gated
        ]
    )


def _detect_decimal_error_multi_period(
    df: pl.LazyFrame,
    col_name: str,
    group_cols: list[str],
    sort_col: str,
    window_sizes: list[int],
    variation_threshold: float = 1.3,
    spill_dir: Path | None = None,
) -> pl.LazyFrame:
    """
    Detect multi-period decimal shift errors (Bessembinder Section 6b).

    For each nlag, checks THREE windows independently at each position t:

    1. FULL WINDOW (size 2*nlag + 1):
       - Endpoints: t-nlag and t+nlag
       - Reversal check: X(t)/X(t-nlag) > 5 AND X(t)/X(t+nlag) > 5
       - If detected, check variation on interior (t-nlag+1 to t+nlag-1)
       - If variation < 1.3, flag interior for correction

    2. SUB-WINDOW A (size 2*nlag):
       - Endpoints: t-nlag and t+nlag-1
       - Reversal check: X(t)/X(t-nlag) > 5 AND X(t)/X(t+nlag-1) > 5
       - If detected, check variation on interior (t-nlag+1 to t+nlag-2)
       - If variation < 1.3, flag interior for correction

    3. SUB-WINDOW B (size 2*nlag):
       - Endpoints: t-nlag+1 and t+nlag
       - Reversal check: X(t)/X(t-nlag+1) > 5 AND X(t)/X(t+nlag) > 5
       - If detected, check variation on interior (t-nlag+2 to t+nlag-1)
       - If variation < 1.3, flag interior for correction

    Key insight: Each window type has its own reversal condition using that
    window's specific endpoints. The variation check excludes endpoints
    because they are the clean comparison values.

    Args:
        df: LazyFrame with the data
        col_name: Column to check for errors
        group_cols: Columns defining the security (e.g., ['gvkey', 'iid'])
        sort_col: Column to sort by within groups (e.g., 'datadate')
        window_sizes: List of window sizes to check (in trading days)
        variation_threshold: Max ratio of max/min within interior (default 1.3 = 30%)

    Returns:
        LazyFrame with correction factors for multi-period errors
    """
    factor_col = f"{col_name}_correction_factor"
    error_type_col = f"{col_name}_error_type"
    window_col = f"{col_name}_window_size"
    window_type_col = f"{col_name}_window_type"
    # Debug columns for tracking detection details
    endpoint_left_col = f"{col_name}_endpoint_left"
    endpoint_right_col = f"{col_name}_endpoint_right"
    ratio_left_col = f"{col_name}_ratio_to_left"
    ratio_right_col = f"{col_name}_ratio_to_right"
    variation_col = f"{col_name}_variation_ratio"

    df = df.sort(group_cols + [sort_col])

    # Replace zero values with null to avoid division by zero errors
    # (zero prices/shares are invalid and should not trigger false positives)
    df = df.with_columns(
        pl.when(col(col_name) == 0).then(pl.lit(None)).otherwise(col(col_name)).alias(col_name)
    )

    # Initialize correction columns if not present
    if factor_col not in df.collect_schema().names():
        df = df.with_columns(
            [
                pl.lit(1.0).alias(factor_col),
                pl.lit(None).cast(pl.Utf8).alias(error_type_col),
            ]
        )

    # Add window size tracking
    if window_col not in df.collect_schema().names():
        df = df.with_columns(pl.lit(None).cast(pl.Int32).alias(window_col))

    # Add window type tracking
    if window_type_col not in df.collect_schema().names():
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(window_type_col))

    # Add debug columns for endpoint and variation tracking
    for debug_col in [
        endpoint_left_col,
        endpoint_right_col,
        ratio_left_col,
        ratio_right_col,
        variation_col,
    ]:
        if debug_col not in df.collect_schema().names():
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(debug_col))

    # Process smallest nlag first to ensure single-day errors are corrected
    # before multi-day detection runs, preventing cascading corrections
    # Materialize periodically to prevent LazyFrame plan from becoming too large
    materialize_interval = 5  # Materialize every N nlag iterations
    iteration_count = 0

    for nlag in sorted(window_sizes):
        if nlag <= 1:
            continue  # Single period handled by _detect_decimal_error_single_period

        print(f"    [detect] {col_name}: multi-period window nlag={nlag}", flush=True)

        # For each threshold magnitude (10x, 100x, 1000x)
        # Single magnitude pass (10x). The previous loop over magnitudes
        # [1, 2, 3] was dead code for magnitudes 2 and 3: every detection flag
        # requires col(factor_col) == 1.0, and any position whose endpoint
        # ratios exceed 50 (100x threshold) also exceeds 5 (10x threshold), so
        # the 10x pass always fired first and its offset-0 propagation marked
        # the detection point itself, blocking the 100x/1000x passes. NOTE:
        # this means multi-period errors are always corrected by 10x regardless
        # of true magnitude -- pre-existing behavior, deliberately preserved.
        # (Single-period detection retains its 1000x -> 100x -> 10x chain.)
        threshold_high = 5.0
        threshold_low = 0.2  # 1 / threshold_high
        correction_factor_high = 0.1  # divide by 10
        correction_factor_low = 10.0  # multiply by 10
        error_type_high = "high_10x"
        error_type_low = "low_10x"

        # ===================================================================
        # FULL WINDOW: endpoints at t-nlag and t+nlag
        # Interior: positions t-nlag+1 to t+nlag-1 (size 2*nlag-1)
        #
        # Detection: At position t, check reversal using endpoints.
        # If reversal + variation OK, mark ALL interior positions.
        # ===================================================================

        # Build list of shifted values for interior variation calculation
        # Interior spans from t-nlag+1 to t+nlag-1 relative to detection point t
        interior_shifts = list(range(-(nlag - 1), nlag))  # e.g., for nlag=2: [-1, 0, 1]

        # Compute max/min across interior positions explicitly
        interior_cols = [
            col(col_name).shift(-s).over(group_cols).alias(f"_int_{s}") for s in interior_shifts
        ]
        df = df.with_columns(interior_cols)

        int_col_names = [f"_int_{s}" for s in interior_shifts]
        df = df.with_columns(
            [
                pl.max_horizontal(*[col(c) for c in int_col_names]).alias("_full_int_max"),
                pl.min_horizontal(*[col(c) for c in int_col_names]).alias("_full_int_min"),
                # Endpoints
                col(col_name).shift(nlag).over(group_cols).alias("_full_ep_lag"),
                col(col_name).shift(-nlag).over(group_cols).alias("_full_ep_lead"),
            ]
        )

        # Drop interior shift columns
        df = df.drop(int_col_names)

        # FIX: Check that endpoints are clean (not already flagged for correction)
        # This prevents cascading false positives where an error value becomes
        # a comparison point for detecting "errors" in subsequent normal data.
        # fill_null(1.0) ensures boundary positions are treated as clean.
        endpoint_left_clean_full = (
            col(factor_col).shift(nlag).over(group_cols).fill_null(1.0) == 1.0
        )
        endpoint_right_clean_full = (
            col(factor_col).shift(-nlag).over(group_cols).fill_null(1.0) == 1.0
        )

        # Reversal check for FULL window (only if endpoints are clean)
        full_high_reversal = (
            endpoint_left_clean_full
            & endpoint_right_clean_full
            & (col(col_name) / col("_full_ep_lag") > threshold_high)
            & (col(col_name) / col("_full_ep_lead") > threshold_high)
        )
        full_low_reversal = (
            endpoint_left_clean_full
            & endpoint_right_clean_full
            & (col(col_name) / col("_full_ep_lag") < threshold_low)
            & (col(col_name) / col("_full_ep_lead") < threshold_low)
        )

        # Variation check on interior
        full_variation_ok = (col("_full_int_max") / col("_full_int_min")) < variation_threshold

        # Determine which positions to flag
        full_high_flag = full_high_reversal & full_variation_ok & (col(factor_col) == 1.0)
        full_low_flag = full_low_reversal & full_variation_ok & (col(factor_col) == 1.0)

        # Create detection marker with debug info
        df = df.with_columns(
            [
                pl.when(full_high_flag)
                .then(pl.lit(correction_factor_high))
                .when(full_low_flag)
                .then(pl.lit(correction_factor_low))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("_full_det"),
                pl.when(full_high_flag)
                .then(pl.lit(error_type_high))
                .when(full_low_flag)
                .then(pl.lit(error_type_low))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("_full_type"),
                pl.when(full_high_flag | full_low_flag)
                .then(pl.lit("full"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("_full_wtype"),
                # Debug: endpoint values
                pl.when(full_high_flag | full_low_flag)
                .then(col("_full_ep_lag"))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("_full_ep_left"),
                pl.when(full_high_flag | full_low_flag)
                .then(col("_full_ep_lead"))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("_full_ep_right"),
                # Debug: ratios to endpoints
                pl.when(full_high_flag | full_low_flag)
                .then(col(col_name) / col("_full_ep_lag"))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("_full_ratio_left"),
                pl.when(full_high_flag | full_low_flag)
                .then(col(col_name) / col("_full_ep_lead"))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("_full_ratio_right"),
                # Debug: variation ratio
                pl.when(full_high_flag | full_low_flag)
                .then(col("_full_int_max") / col("_full_int_min"))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("_full_variation"),
            ]
        )

        # Propagate to all interior positions (single coalesce pass)
        # Interior is at offsets -(nlag-1) to +(nlag-1) from detection point
        df = _propagate_interior(df, col_name, "_full_", interior_shifts, group_cols, nlag)

        df = df.drop(
            [
                "_full_int_max",
                "_full_int_min",
                "_full_ep_lag",
                "_full_ep_lead",
                "_full_det",
                "_full_type",
                "_full_wtype",
                "_full_ep_left",
                "_full_ep_right",
                "_full_ratio_left",
                "_full_ratio_right",
                "_full_variation",
            ]
        )

        # ===================================================================
        # SUB-WINDOW A: endpoints at t-nlag and t+nlag-1
        # Interior: positions t-nlag+1 to t+nlag-2 (size 2*nlag-2)
        # ===================================================================

        sub_interior_size = 2 * nlag - 2
        if sub_interior_size >= 1:
            # Interior spans from t-nlag+1 to t+nlag-2
            sub_a_shifts = list(range(-(nlag - 1), nlag - 1))  # e.g., for nlag=2: [-1, 0]

            sub_a_cols = [
                col(col_name).shift(-s).over(group_cols).alias(f"_sa_{s}") for s in sub_a_shifts
            ]
            df = df.with_columns(sub_a_cols)

            sa_col_names = [f"_sa_{s}" for s in sub_a_shifts]
            df = df.with_columns(
                [
                    pl.max_horizontal(*[col(c) for c in sa_col_names]).alias("_sa_int_max"),
                    pl.min_horizontal(*[col(c) for c in sa_col_names]).alias("_sa_int_min"),
                    col(col_name).shift(nlag).over(group_cols).alias("_sa_ep_lag"),
                    col(col_name).shift(-(nlag - 1)).over(group_cols).alias("_sa_ep_lead"),
                ]
            )
            df = df.drop(sa_col_names)

            # FIX: Check that endpoints are clean for sub-window A
            # Endpoints at t-nlag and t+nlag-1
            endpoint_left_clean_sa = (
                col(factor_col).shift(nlag).over(group_cols).fill_null(1.0) == 1.0
            )
            endpoint_right_clean_sa = (
                col(factor_col).shift(-(nlag - 1)).over(group_cols).fill_null(1.0) == 1.0
            )

            sub_a_high_reversal = (
                endpoint_left_clean_sa
                & endpoint_right_clean_sa
                & (col(col_name) / col("_sa_ep_lag") > threshold_high)
                & (col(col_name) / col("_sa_ep_lead") > threshold_high)
            )
            sub_a_low_reversal = (
                endpoint_left_clean_sa
                & endpoint_right_clean_sa
                & (col(col_name) / col("_sa_ep_lag") < threshold_low)
                & (col(col_name) / col("_sa_ep_lead") < threshold_low)
            )

            sub_a_variation_ok = (col("_sa_int_max") / col("_sa_int_min")) < variation_threshold

            sub_a_high_flag = sub_a_high_reversal & sub_a_variation_ok & (col(factor_col) == 1.0)
            sub_a_low_flag = sub_a_low_reversal & sub_a_variation_ok & (col(factor_col) == 1.0)

            df = df.with_columns(
                [
                    pl.when(sub_a_high_flag)
                    .then(pl.lit(correction_factor_high))
                    .when(sub_a_low_flag)
                    .then(pl.lit(correction_factor_low))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sa_det"),
                    pl.when(sub_a_high_flag)
                    .then(pl.lit(error_type_high))
                    .when(sub_a_low_flag)
                    .then(pl.lit(error_type_low))
                    .otherwise(pl.lit(None).cast(pl.Utf8))
                    .alias("_sa_type"),
                    pl.when(sub_a_high_flag | sub_a_low_flag)
                    .then(pl.lit("sub_a"))
                    .otherwise(pl.lit(None).cast(pl.Utf8))
                    .alias("_sa_wtype"),
                    # Debug: endpoint values
                    pl.when(sub_a_high_flag | sub_a_low_flag)
                    .then(col("_sa_ep_lag"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sa_ep_left"),
                    pl.when(sub_a_high_flag | sub_a_low_flag)
                    .then(col("_sa_ep_lead"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sa_ep_right"),
                    # Debug: ratios to endpoints
                    pl.when(sub_a_high_flag | sub_a_low_flag)
                    .then(col(col_name) / col("_sa_ep_lag"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sa_ratio_left"),
                    pl.when(sub_a_high_flag | sub_a_low_flag)
                    .then(col(col_name) / col("_sa_ep_lead"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sa_ratio_right"),
                    # Debug: variation ratio
                    pl.when(sub_a_high_flag | sub_a_low_flag)
                    .then(col("_sa_int_max") / col("_sa_int_min"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sa_variation"),
                ]
            )

            # Propagate to all interior positions (single coalesce pass)
            df = _propagate_interior(df, col_name, "_sa_", sub_a_shifts, group_cols, nlag)

            df = df.drop(
                [
                    "_sa_int_max",
                    "_sa_int_min",
                    "_sa_ep_lag",
                    "_sa_ep_lead",
                    "_sa_det",
                    "_sa_type",
                    "_sa_wtype",
                    "_sa_ep_left",
                    "_sa_ep_right",
                    "_sa_ratio_left",
                    "_sa_ratio_right",
                    "_sa_variation",
                ]
            )

        # ===================================================================
        # SUB-WINDOW B: endpoints at t-nlag+1 and t+nlag
        # Interior: positions t-nlag+2 to t+nlag-1 (size 2*nlag-2)
        # ===================================================================

        if sub_interior_size >= 1:
            # Interior spans from t-nlag+2 to t+nlag-1
            sub_b_shifts = list(range(-(nlag - 2), nlag))  # e.g., for nlag=2: [0, 1]

            sub_b_cols = [
                col(col_name).shift(-s).over(group_cols).alias(f"_sb_{s}") for s in sub_b_shifts
            ]
            df = df.with_columns(sub_b_cols)

            sb_col_names = [f"_sb_{s}" for s in sub_b_shifts]
            df = df.with_columns(
                [
                    pl.max_horizontal(*[col(c) for c in sb_col_names]).alias("_sb_int_max"),
                    pl.min_horizontal(*[col(c) for c in sb_col_names]).alias("_sb_int_min"),
                    col(col_name).shift(nlag - 1).over(group_cols).alias("_sb_ep_lag"),
                    col(col_name).shift(-nlag).over(group_cols).alias("_sb_ep_lead"),
                ]
            )
            df = df.drop(sb_col_names)

            # FIX: Check that endpoints are clean for sub-window B
            # Endpoints at t-nlag+1 and t+nlag
            endpoint_left_clean_sb = (
                col(factor_col).shift(nlag - 1).over(group_cols).fill_null(1.0) == 1.0
            )
            endpoint_right_clean_sb = (
                col(factor_col).shift(-nlag).over(group_cols).fill_null(1.0) == 1.0
            )

            sub_b_high_reversal = (
                endpoint_left_clean_sb
                & endpoint_right_clean_sb
                & (col(col_name) / col("_sb_ep_lag") > threshold_high)
                & (col(col_name) / col("_sb_ep_lead") > threshold_high)
            )
            sub_b_low_reversal = (
                endpoint_left_clean_sb
                & endpoint_right_clean_sb
                & (col(col_name) / col("_sb_ep_lag") < threshold_low)
                & (col(col_name) / col("_sb_ep_lead") < threshold_low)
            )

            sub_b_variation_ok = (col("_sb_int_max") / col("_sb_int_min")) < variation_threshold

            sub_b_high_flag = sub_b_high_reversal & sub_b_variation_ok & (col(factor_col) == 1.0)
            sub_b_low_flag = sub_b_low_reversal & sub_b_variation_ok & (col(factor_col) == 1.0)

            df = df.with_columns(
                [
                    pl.when(sub_b_high_flag)
                    .then(pl.lit(correction_factor_high))
                    .when(sub_b_low_flag)
                    .then(pl.lit(correction_factor_low))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sb_det"),
                    pl.when(sub_b_high_flag)
                    .then(pl.lit(error_type_high))
                    .when(sub_b_low_flag)
                    .then(pl.lit(error_type_low))
                    .otherwise(pl.lit(None).cast(pl.Utf8))
                    .alias("_sb_type"),
                    pl.when(sub_b_high_flag | sub_b_low_flag)
                    .then(pl.lit("sub_b"))
                    .otherwise(pl.lit(None).cast(pl.Utf8))
                    .alias("_sb_wtype"),
                    # Debug: endpoint values
                    pl.when(sub_b_high_flag | sub_b_low_flag)
                    .then(col("_sb_ep_lag"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sb_ep_left"),
                    pl.when(sub_b_high_flag | sub_b_low_flag)
                    .then(col("_sb_ep_lead"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sb_ep_right"),
                    # Debug: ratios to endpoints
                    pl.when(sub_b_high_flag | sub_b_low_flag)
                    .then(col(col_name) / col("_sb_ep_lag"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sb_ratio_left"),
                    pl.when(sub_b_high_flag | sub_b_low_flag)
                    .then(col(col_name) / col("_sb_ep_lead"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sb_ratio_right"),
                    # Debug: variation ratio
                    pl.when(sub_b_high_flag | sub_b_low_flag)
                    .then(col("_sb_int_max") / col("_sb_int_min"))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias("_sb_variation"),
                ]
            )

            # Propagate to all interior positions (single coalesce pass)
            df = _propagate_interior(df, col_name, "_sb_", sub_b_shifts, group_cols, nlag)

            df = df.drop(
                [
                    "_sb_int_max",
                    "_sb_int_min",
                    "_sb_ep_lag",
                    "_sb_ep_lead",
                    "_sb_det",
                    "_sb_type",
                    "_sb_wtype",
                    "_sb_ep_left",
                    "_sb_ep_right",
                    "_sb_ratio_left",
                    "_sb_ratio_right",
                    "_sb_variation",
                ]
            )

        # Materialize periodically to prevent LazyFrame plan from becoming too large
        # This prevents segmentation faults from overly complex query plans
        iteration_count += 1
        if iteration_count % materialize_interval == 0:
            if spill_dir is not None:
                # Spill to disk (streaming) instead of collecting in RAM: the full
                # frame with debug columns does not fit in memory on cluster runs.
                spill_path = spill_dir / f"__bess_spill_{col_name}_{iteration_count}.parquet"
                print(
                    f"    [detect] {col_name}: spilling plan to {spill_path.name} "
                    f"after {iteration_count} windows",
                    flush=True,
                )
                df.sink_parquet(spill_path)
                df = pl.scan_parquet(spill_path)
            else:
                print(
                    f"    [detect] {col_name}: materializing plan after {iteration_count} windows",
                    flush=True,
                )
                df = df.collect().lazy()

    return df


def correct_decimal_errors(
    df: pl.LazyFrame,
    col_name: str,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    window_sizes: list[int] | None = None,
    log_corrections: bool = True,
    validate_cascading: bool = True,
    correction_method: str = "bessembinder",
    spill_dir: Path | None = None,
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
        df = _detect_decimal_error_multi_period(
            df, col_name, group_cols, sort_col, multi_windows, spill_dir=spill_dir
        )

    # Third pass: Validate corrections to prevent cascading errors
    if validate_cascading:
        print(f"  [correct] {col_name}: validating corrections (cascading check)", flush=True)
        # Build preliminary corrections log for validation
        preliminary_log = df.filter(col(factor_col) != 1.0).select(
            group_cols
            + [
                sort_col,
                pl.lit(col_name).alias("variable"),
                col(original_col).alias("original_value"),
                col(factor_col).alias("correction_factor"),
                col(error_type_col).alias("error_type"),
                col(window_col).alias("window_size"),
                col(window_type_col).alias("window_type"),
                col(endpoint_left_col).alias("endpoint_left"),
                col(endpoint_right_col).alias("endpoint_right"),
                col(ratio_left_col).alias("ratio_to_left"),
                col(ratio_right_col).alias("ratio_to_right"),
                col(variation_col).alias("variation_ratio"),
            ]
        )

        # Validate and filter corrections
        df, valid_log, rejected_log = _validate_corrections_no_cascading(
            df, preliminary_log, col_name, group_cols, sort_col
        )

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

    Returns:
        Tuple of (corrected_df, all_corrections_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
    if window_sizes is None:
        # Priority windows covering common error durations (matches correct_decimal_errors)
        window_sizes = [1, 2, 3, 5, 10, 21]

    all_logs = []

    # Step 1: Correct TRFD independently
    if "trfd" in df.collect_schema().names():
        print("[section6] step 1/5: correcting trfd", flush=True)
        df, log = correct_decimal_errors(
            df,
            "trfd",
            group_cols,
            sort_col,
            window_sizes,
            correction_method=correction_method,
            spill_dir=spill_dir,
        )
        if log is not None:
            all_logs.append(log)

    # Step 1: Correct QUNIT independently
    if "qunit" in df.collect_schema().names():
        print("[section6] step 1/5: correcting qunit", flush=True)
        df, log = correct_decimal_errors(
            df,
            "qunit",
            group_cols,
            sort_col,
            window_sizes,
            correction_method=correction_method,
            spill_dir=spill_dir,
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
                df,
                "adrrc",
                group_cols,
                sort_col,
                window_sizes,
                correction_method=correction_method,
                spill_dir=spill_dir,
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
        df,
        "_adjprc",
        group_cols,
        sort_col,
        window_sizes,
        correction_method=correction_method,
        spill_dir=spill_dir,
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
        df,
        "_adjcsho",
        group_cols,
        sort_col,
        window_sizes,
        correction_method=correction_method,
        spill_dir=spill_dir,
    )
    if log is not None:
        log = log.with_columns(
            pl.when(col("variable") == "_adjcsho")
            .then(pl.lit("adjcsho"))
            .otherwise(col("variable"))
            .alias("variable")
        )
        all_logs.append(log)

    # Step 5: Reconstruct PRCCD and CSHOC from corrected adjusted values
    print("[section6] step 5/5: reconstructing prccd/cshoc", flush=True)
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
        # Stay lazy on a slim projection — only the (small) filtered result is
        # collected, never the full daily frame.
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

    # Find the percentile cutoff
    cutoff = avg_vol.select(pl.col("_avg_vol").quantile(percentile).alias("_cutoff"))

    # Join back and filter
    df = df.join(avg_vol, on=group_cols, how="left")
    df = df.with_columns(cutoff.select("_cutoff"))

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

    # Detect QUNIT changes without currency change and large price change
    df = df.with_columns(
        [
            col("qunit").shift(1).over(group_cols).alias("_qunit_lag"),
            col("curcdd").shift(1).over(group_cols).alias("_curcdd_lag"),
            col("prccd").shift(1).over(group_cols).alias("_prccd_lag"),
        ]
    )

    df = df.with_columns(
        [
            (
                (col("qunit") != col("_qunit_lag"))
                & (col("curcdd") == col("_curcdd_lag"))
                & (((col("prccd") / col("_prccd_lag")) - 1).abs() > 0.5)
            ).alias("_bad_qunit_change")
        ]
    )

    # Identify securities with bad QUNIT changes
    has_bad_qunit = (
        df.filter(col("_bad_qunit_change"))
        .select(group_cols)
        .unique()
        .with_columns(pl.lit(True).alias("_has_bad_qunit"))
    )

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
    df = df.drop(
        [
            "_qunit_lag",
            "_curcdd_lag",
            "_prccd_lag",
            "_bad_qunit_change",
            "_has_zero_ajex",
            "_has_bad_qunit",
        ]
    )

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


def apply_bessembinder_section8(
    df: pl.LazyFrame,
    group_cols: list[str] | None = None,
    sort_col: str = "datadate",
    country_col: str = "excntry",
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

    Returns:
        Tuple of (filtered_df, all_removed_log)
    """
    if group_cols is None:
        group_cols = ["gvkey", "iid"]
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
