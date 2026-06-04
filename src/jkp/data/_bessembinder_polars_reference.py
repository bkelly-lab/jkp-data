"""
Original Polars implementation of Bessembinder Section 6 detection/validation.

Description:
    Verbatim copy of the handoff implementation (commit 605f131) of the three
    Section 6 core functions. Kept solely as the differential-test oracle for
    the numba kernel rewrite in bessembinder_kernels.py — production code in
    bessembinder.py does NOT import this module.
Steps:
    1) _polars_validate_corrections_no_cascading: iterative cascade rejection.
    2) _polars_detect_decimal_error_single_period: spike-and-reversal, window 1.
    3) _polars_detect_decimal_error_multi_period: FULL/SUB_A/SUB_B windows per
       nlag with magnitude loop [1, 2, 3].
Output:
    Functions imported by tests/unit/test_bessembinder_differential.py only.
"""

import logging

import polars as pl
from polars import col

logger = logging.getLogger(__name__)


def _polars_validate_corrections_no_cascading(
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

    # Materialize to work with actual values
    log_df = corrections_log.collect()

    if len(log_df) == 0:
        # No corrections to validate
        return df, corrections_log, pl.LazyFrame()

    # We need the full data to look up dates for endpoint positions
    data_df = df.collect()

    # Build a date index for each security to map positions to dates
    # Group data by security and create position -> date mapping
    date_index_by_security = {}
    for key, group in data_df.group_by(group_cols):
        group_sorted = group.sort(sort_col)
        dates = group_sorted[sort_col].to_list()
        # Create mapping from date to position index
        date_to_pos = {d: i for i, d in enumerate(dates)}
        pos_to_date = dict(enumerate(dates))
        date_index_by_security[key] = {
            "dates": dates,
            "date_to_pos": date_to_pos,
            "pos_to_date": pos_to_date,
        }

    # Iteratively reject false positives until stable
    current_log = log_df.clone()
    all_rejected = []
    max_iterations = 10  # Safety limit

    for iteration in range(max_iterations):
        # Build set of currently flagged positions
        flagged_positions = set()
        for row in current_log.iter_rows(named=True):
            key = tuple(row[gc] for gc in group_cols) + (row[sort_col],)
            flagged_positions.add(key)

        if len(flagged_positions) == 0:
            break

        # Check each correction: are BOTH endpoint positions also flagged?
        to_reject = []
        to_keep = []

        for row in current_log.iter_rows(named=True):
            security_key = tuple(row[gc] for gc in group_cols)
            current_date = row[sort_col]
            window_size = row.get("window_size", 1) or 1

            # Get date index for this security
            if security_key not in date_index_by_security:
                to_keep.append(row)
                continue

            index_info = date_index_by_security[security_key]
            date_to_pos = index_info["date_to_pos"]
            pos_to_date = index_info["pos_to_date"]

            if current_date not in date_to_pos:
                to_keep.append(row)
                continue

            current_pos = date_to_pos[current_date]

            # Determine endpoint positions (for single-period: pos-1 and pos+1)
            # For multi-period: pos-window_size and pos+window_size
            left_pos = current_pos - window_size
            right_pos = current_pos + window_size

            # Get endpoint dates
            left_date = pos_to_date.get(left_pos)
            right_date = pos_to_date.get(right_pos)

            # Check if endpoint positions are flagged
            left_flagged = False
            right_flagged = False

            if left_date is not None:
                left_key = security_key + (left_date,)
                left_flagged = left_key in flagged_positions

            if right_date is not None:
                right_key = security_key + (right_date,)
                right_flagged = right_key in flagged_positions

            # Reject if BOTH endpoints are flagged (likely false positive)
            if left_flagged and right_flagged:
                to_reject.append(dict(row) | {"rejection_reason": "both_endpoints_flagged"})
            else:
                to_keep.append(row)

        if len(to_reject) == 0:
            # No more rejections, we're done
            break

        all_rejected.extend(to_reject)
        current_log = pl.DataFrame(to_keep) if to_keep else pl.DataFrame(schema=log_df.schema)

        logger.debug(
            f"{col_name}: iteration {iteration + 1}, rejected {len(to_reject)} corrections"
        )

    # Build final results
    valid_corrections = current_log
    rejected_corrections = (
        pl.DataFrame(all_rejected)
        if all_rejected
        else pl.DataFrame(schema=log_df.schema | {"rejection_reason": pl.Utf8})
    )

    # Log rejection statistics
    n_rejected = len(rejected_corrections)
    n_total = len(log_df)
    if n_rejected > 0:
        logger.info(f"{col_name}: rejected {n_rejected}/{n_total} cascading false positives")

    # Now update the main dataframe to reset invalid corrections
    if n_rejected > 0:
        # Build a lookup for rejection
        rejection_lookup = set()
        for row in rejected_corrections.iter_rows(named=True):
            key = tuple(row[gc] for gc in group_cols) + (row[sort_col],)
            rejection_lookup.add(key)

        # Reset correction factors and error_types for rejected corrections
        new_factors = []
        new_types = []
        for row in data_df.iter_rows(named=True):
            key = tuple(row[gc] for gc in group_cols) + (row[sort_col],)
            if key in rejection_lookup:
                new_factors.append(1.0)  # Reset to no correction
                new_types.append(None)
            else:
                new_factors.append(row[factor_col])
                new_types.append(row[error_type_col])

        data_df = data_df.with_columns(
            [
                pl.Series(factor_col, new_factors),
                pl.Series(error_type_col, new_types).cast(pl.Utf8),
            ]
        )

        df = data_df.lazy()
    else:
        df = data_df.lazy()

    # Clean up validation columns before returning
    valid_cols = [c for c in valid_corrections.columns if c != "rejection_reason"]
    if len(valid_cols) > 0 and len(valid_corrections) > 0:
        valid_corrections = valid_corrections.select(valid_cols)

    return df, valid_corrections.lazy(), rejected_corrections.lazy()


def _polars_detect_decimal_error_single_period(
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


def _polars_detect_decimal_error_multi_period(
    df: pl.LazyFrame,
    col_name: str,
    group_cols: list[str],
    sort_col: str,
    window_sizes: list[int],
    variation_threshold: float = 1.3,
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

        # For each threshold magnitude (10x, 100x, 1000x)
        for magnitude in [1, 2, 3]:
            threshold_high = 5 * (10 ** (magnitude - 1))  # 5, 50, 500
            threshold_low = 1.0 / threshold_high  # 0.2, 0.02, 0.002
            correction_factor_high = 10.0 ** (-magnitude)  # 0.1, 0.01, 0.001
            correction_factor_low = 10.0**magnitude  # 10, 100, 1000
            # Error type labels with magnitude
            magnitude_suffix = f"{10**magnitude}x"  # "10x", "100x", "1000x"
            error_type_high = f"high_{magnitude_suffix}"
            error_type_low = f"low_{magnitude_suffix}"

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

            # Propagate to all interior positions
            # Interior is at offsets -(nlag-1) to +(nlag-1) from detection point
            for offset in interior_shifts:
                df = df.with_columns(
                    [
                        pl.when(
                            (col(factor_col) == 1.0)
                            & col("_full_det").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_det").shift(offset).over(group_cols))
                        .otherwise(col(factor_col))
                        .alias(factor_col),
                        pl.when(
                            col(error_type_col).is_null()
                            & col("_full_type").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_type").shift(offset).over(group_cols))
                        .otherwise(col(error_type_col))
                        .alias(error_type_col),
                        pl.when(
                            col(window_col).is_null()
                            & col("_full_det").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(pl.lit(nlag))
                        .otherwise(col(window_col))
                        .alias(window_col),
                        pl.when(
                            col(window_type_col).is_null()
                            & col("_full_wtype").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_wtype").shift(offset).over(group_cols))
                        .otherwise(col(window_type_col))
                        .alias(window_type_col),
                        # Propagate debug columns
                        pl.when(
                            col(endpoint_left_col).is_null()
                            & col("_full_ep_left").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_ep_left").shift(offset).over(group_cols))
                        .otherwise(col(endpoint_left_col))
                        .alias(endpoint_left_col),
                        pl.when(
                            col(endpoint_right_col).is_null()
                            & col("_full_ep_right").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_ep_right").shift(offset).over(group_cols))
                        .otherwise(col(endpoint_right_col))
                        .alias(endpoint_right_col),
                        pl.when(
                            col(ratio_left_col).is_null()
                            & col("_full_ratio_left").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_ratio_left").shift(offset).over(group_cols))
                        .otherwise(col(ratio_left_col))
                        .alias(ratio_left_col),
                        pl.when(
                            col(ratio_right_col).is_null()
                            & col("_full_ratio_right").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_ratio_right").shift(offset).over(group_cols))
                        .otherwise(col(ratio_right_col))
                        .alias(ratio_right_col),
                        pl.when(
                            col(variation_col).is_null()
                            & col("_full_variation").shift(offset).over(group_cols).is_not_null()
                        )
                        .then(col("_full_variation").shift(offset).over(group_cols))
                        .otherwise(col(variation_col))
                        .alias(variation_col),
                    ]
                )

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

                sub_a_high_flag = (
                    sub_a_high_reversal & sub_a_variation_ok & (col(factor_col) == 1.0)
                )
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

                for offset in sub_a_shifts:
                    df = df.with_columns(
                        [
                            pl.when(
                                (col(factor_col) == 1.0)
                                & col("_sa_det").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sa_det").shift(offset).over(group_cols))
                            .otherwise(col(factor_col))
                            .alias(factor_col),
                            pl.when(
                                col(error_type_col).is_null()
                                & col("_sa_type").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sa_type").shift(offset).over(group_cols))
                            .otherwise(col(error_type_col))
                            .alias(error_type_col),
                            pl.when(
                                col(window_col).is_null()
                                & col("_sa_det").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(pl.lit(nlag))
                            .otherwise(col(window_col))
                            .alias(window_col),
                            pl.when(
                                col(window_type_col).is_null()
                                & col("_sa_wtype").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sa_wtype").shift(offset).over(group_cols))
                            .otherwise(col(window_type_col))
                            .alias(window_type_col),
                            # Propagate debug columns
                            pl.when(
                                col(endpoint_left_col).is_null()
                                & col("_sa_ep_left").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sa_ep_left").shift(offset).over(group_cols))
                            .otherwise(col(endpoint_left_col))
                            .alias(endpoint_left_col),
                            pl.when(
                                col(endpoint_right_col).is_null()
                                & col("_sa_ep_right").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sa_ep_right").shift(offset).over(group_cols))
                            .otherwise(col(endpoint_right_col))
                            .alias(endpoint_right_col),
                            pl.when(
                                col(ratio_left_col).is_null()
                                & col("_sa_ratio_left").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sa_ratio_left").shift(offset).over(group_cols))
                            .otherwise(col(ratio_left_col))
                            .alias(ratio_left_col),
                            pl.when(
                                col(ratio_right_col).is_null()
                                & col("_sa_ratio_right")
                                .shift(offset)
                                .over(group_cols)
                                .is_not_null()
                            )
                            .then(col("_sa_ratio_right").shift(offset).over(group_cols))
                            .otherwise(col(ratio_right_col))
                            .alias(ratio_right_col),
                            pl.when(
                                col(variation_col).is_null()
                                & col("_sa_variation").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sa_variation").shift(offset).over(group_cols))
                            .otherwise(col(variation_col))
                            .alias(variation_col),
                        ]
                    )

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

                sub_b_high_flag = (
                    sub_b_high_reversal & sub_b_variation_ok & (col(factor_col) == 1.0)
                )
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

                for offset in sub_b_shifts:
                    df = df.with_columns(
                        [
                            pl.when(
                                (col(factor_col) == 1.0)
                                & col("_sb_det").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sb_det").shift(offset).over(group_cols))
                            .otherwise(col(factor_col))
                            .alias(factor_col),
                            pl.when(
                                col(error_type_col).is_null()
                                & col("_sb_type").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sb_type").shift(offset).over(group_cols))
                            .otherwise(col(error_type_col))
                            .alias(error_type_col),
                            pl.when(
                                col(window_col).is_null()
                                & col("_sb_det").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(pl.lit(nlag))
                            .otherwise(col(window_col))
                            .alias(window_col),
                            pl.when(
                                col(window_type_col).is_null()
                                & col("_sb_wtype").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sb_wtype").shift(offset).over(group_cols))
                            .otherwise(col(window_type_col))
                            .alias(window_type_col),
                            # Propagate debug columns
                            pl.when(
                                col(endpoint_left_col).is_null()
                                & col("_sb_ep_left").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sb_ep_left").shift(offset).over(group_cols))
                            .otherwise(col(endpoint_left_col))
                            .alias(endpoint_left_col),
                            pl.when(
                                col(endpoint_right_col).is_null()
                                & col("_sb_ep_right").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sb_ep_right").shift(offset).over(group_cols))
                            .otherwise(col(endpoint_right_col))
                            .alias(endpoint_right_col),
                            pl.when(
                                col(ratio_left_col).is_null()
                                & col("_sb_ratio_left").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sb_ratio_left").shift(offset).over(group_cols))
                            .otherwise(col(ratio_left_col))
                            .alias(ratio_left_col),
                            pl.when(
                                col(ratio_right_col).is_null()
                                & col("_sb_ratio_right")
                                .shift(offset)
                                .over(group_cols)
                                .is_not_null()
                            )
                            .then(col("_sb_ratio_right").shift(offset).over(group_cols))
                            .otherwise(col(ratio_right_col))
                            .alias(ratio_right_col),
                            pl.when(
                                col(variation_col).is_null()
                                & col("_sb_variation").shift(offset).over(group_cols).is_not_null()
                            )
                            .then(col("_sb_variation").shift(offset).over(group_cols))
                            .otherwise(col(variation_col))
                            .alias(variation_col),
                        ]
                    )

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
            df = df.collect().lazy()

    return df


def _polars_correct_decimal_errors(
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
    df = _polars_detect_decimal_error_single_period(df, col_name, group_cols, sort_col)

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
        df = _polars_detect_decimal_error_multi_period(
            df, col_name, group_cols, sort_col, multi_windows
        )

    # Third pass: Validate corrections to prevent cascading errors
    if validate_cascading:
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
        df, valid_log, rejected_log = _polars_validate_corrections_no_cascading(
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
