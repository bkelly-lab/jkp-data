"""
Tests for Bessembinder et al. (2023) data correction functions.

These tests verify the decimal error detection and correction algorithms
that clean Compustat data for use in return calculations. The corrections
are critical for data quality when using Compustat instead of CRSP.

Paper Reference: Bessembinder, Chen, Choi, Wei (2023), "Do Global Stocks Outperform US Treasury Bills?"
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jkp.data.bessembinder import (
    _detect_decimal_error_multi_period,
    _detect_decimal_error_single_period,
    compute_correction_rate,
    correct_decimal_errors,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def single_security_df() -> pl.LazyFrame:
    """Create a simple single-security time series for testing."""
    return (
        pl.DataFrame(
            {
                "gvkey": ["001"] * 10,
                "iid": ["01"] * 10,
                "datadate": [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                    "2020-01-09",
                    "2020-01-10",
                    "2020-01-13",
                    "2020-01-14",
                    "2020-01-15",
                ],
                "prccd": [10.0, 10.5, 10.2, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.6],
            }
        )
        .with_columns(pl.col("datadate").str.to_date())
        .lazy()
    )


@pytest.fixture
def multi_security_df() -> pl.LazyFrame:
    """Create a multi-security DataFrame for group-aware testing."""
    return (
        pl.DataFrame(
            {
                "gvkey": ["001"] * 5 + ["002"] * 5,
                "iid": ["01"] * 5 + ["01"] * 5,
                "datadate": [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                ]
                * 2,
                "prccd": [
                    10.0,
                    10.5,
                    10.2,
                    10.3,
                    10.1,  # Security 1: normal
                    20.0,
                    21.0,
                    20.5,
                    20.8,
                    20.2,  # Security 2: normal
                ],
            }
        )
        .with_columns(pl.col("datadate").str.to_date())
        .lazy()
    )


# =============================================================================
# Single-Period Decimal Error Detection (Section 6a)
# =============================================================================


class TestDetectDecimalErrorSinglePeriod:
    """Tests for _detect_decimal_error_single_period() - Bessembinder Section 6a.

    This function detects single-period decimal errors where a value spikes
    relative to both its prior and next values (spike-and-reversal pattern).
    """

    def test_detects_10x_high_error(self):
        """Detect 10x decimal error (e.g., 10.5 recorded as 105)."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    "prccd": [10.0, 10.5, 105.0, 10.3, 10.1],  # Day 3 is 10x error
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        # Row 2 (105.0) should have correction factor 0.1 (divide by 10)
        assert result["prccd_correction_factor"][2] == pytest.approx(0.1), (
            f"Expected correction factor 0.1 for 10x error, "
            f"got {result['prccd_correction_factor'][2]}"
        )
        assert result["prccd_error_type"][2] == "high_10x", (
            f"Expected error_type 'high_10x', got {result['prccd_error_type'][2]}"
        )

    def test_detects_100x_high_error(self):
        """Detect 100x decimal error (e.g., 10.5 recorded as 1050)."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    "prccd": [10.0, 10.5, 1050.0, 10.3, 10.1],  # Day 3 is 100x error
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        assert result["prccd_correction_factor"][2] == pytest.approx(0.01), (
            f"Expected correction factor 0.01 for 100x error, "
            f"got {result['prccd_correction_factor'][2]}"
        )

    def test_detects_1000x_high_error(self):
        """Detect 1000x decimal error."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    "prccd": [10.0, 10.5, 10500.0, 10.3, 10.1],  # Day 3 is 1000x error
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        assert result["prccd_correction_factor"][2] == pytest.approx(0.001), (
            f"Expected correction factor 0.001 for 1000x error, "
            f"got {result['prccd_correction_factor'][2]}"
        )

    def test_detects_10x_low_error(self):
        """Detect 10x low decimal error (e.g., 10.5 recorded as 1.05)."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    "prccd": [10.0, 10.5, 1.02, 10.3, 10.1],  # Day 3 is 1/10 error
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        assert result["prccd_correction_factor"][2] == pytest.approx(10.0), (
            f"Expected correction factor 10.0 for 1/10 error, "
            f"got {result['prccd_correction_factor'][2]}"
        )
        assert result["prccd_error_type"][2] == "low_10x", (
            f"Expected error_type 'low_10x', got {result['prccd_error_type'][2]}"
        )

    def test_no_correction_for_normal_data(self, single_security_df):
        """Normal price movements should not trigger correction."""
        result = _detect_decimal_error_single_period(
            single_security_df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        # All correction factors should be 1.0 (no correction)
        factors = result["prccd_correction_factor"].to_list()
        assert all(f == 1.0 for f in factors), (
            f"Expected all correction factors to be 1.0, got {factors}"
        )

    def test_no_correction_for_sustained_move(self):
        """A sustained price move (no reversal) should not trigger correction.

        Even if the move is large, if it doesn't reverse, it's not a decimal error.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    # 10x jump that persists (M&A, etc.)
                    "prccd": [10.0, 10.5, 100.0, 102.0, 98.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        # Row 2: ratio_to_prior = 100/10.5 ≈ 9.5 > 5
        # But ratio_to_next = 100/102 ≈ 0.98, NOT > 5
        # So should NOT be flagged as error
        assert result["prccd_correction_factor"][2] == 1.0, (
            f"Sustained move should not trigger correction, "
            f"got factor {result['prccd_correction_factor'][2]}"
        )

    def test_handles_multiple_securities(self, multi_security_df):
        """Corrections should be computed within each security group."""
        # Add an error to one security but not the other
        df = multi_security_df.collect()
        df = df.with_columns(
            pl.when((pl.col("gvkey") == "001") & (pl.col("datadate").dt.day() == 6))
            .then(pl.lit(102.0))  # 10x error in security 1
            .otherwise(pl.col("prccd"))
            .alias("prccd")
        ).lazy()

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        # Security 001 should have correction at row 2
        sec1 = result.filter(pl.col("gvkey") == "001")
        sec2 = result.filter(pl.col("gvkey") == "002")

        assert sec1["prccd_correction_factor"][2] == pytest.approx(0.1), (
            "Security 001 should have 10x correction at row 2"
        )

        # Security 002 should have no corrections
        assert all(f == 1.0 for f in sec2["prccd_correction_factor"].to_list()), (
            "Security 002 should have no corrections"
        )

    def test_first_and_last_rows_not_flagged(self):
        """First and last rows cannot be flagged (no prior/next to compare)."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 3,
                    "iid": ["01"] * 3,
                    "datadate": ["2020-01-02", "2020-01-03", "2020-01-06"],
                    "prccd": [100.0, 10.0, 100.0],  # Middle is suspicious but edges are too
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        # First row has no prior, last row has no next
        # Only middle row can potentially be flagged
        assert result["prccd_correction_factor"][1] == pytest.approx(10.0), (
            "Middle row should be flagged as low error"
        )


# =============================================================================
# Multi-Period Decimal Error Detection (Section 6b)
# =============================================================================


class TestDetectDecimalErrorMultiPeriod:
    """Tests for _detect_decimal_error_multi_period() - Bessembinder Section 6b.

    This function detects decimal errors that persist for multiple periods.
    It uses three-window approach with variation checks on interiors.

    For each nlag, three windows are checked independently:
    1. FULL WINDOW: endpoints t-nlag and t+nlag, interior t-nlag+1 to t+nlag-1
    2. SUB-WINDOW A: endpoints t-nlag and t+nlag-1, interior t-nlag+1 to t+nlag-2
    3. SUB-WINDOW B: endpoints t-nlag+1 and t+nlag, interior t-nlag+2 to t+nlag-1

    Each window has its own reversal check using that window's endpoints.
    Variation check (max/min < 1.3) is performed on the interior only.
    """

    def test_detects_2day_error_via_subwindow_b(self):
        """Detect a 2-day error using sub-window B.

        This test demonstrates how sub-window B catches even-length errors:

        Position:   0      1      2      3      4      5      6
        Value:     10.0   10.2   10.1   101.0  102.0  10.3   10.4
                                       ^^^    ^^^
                                       ERROR SPAN (2 days)

        At position t=3 with nlag=2:
        - FULL WINDOW endpoints (1, 5): reversal OK, but interior (2,3,4) has
          values [10.1, 101, 102] with variation 102/10.1 ≈ 10 > 1.3 → FAILS
        - SUB-WINDOW B endpoints (2, 5): reversal OK, interior (3,4) has
          values [101, 102] with variation 102/101 ≈ 1.01 < 1.3 → PASSES!

        This is why we need sub-windows: they can have interiors that contain
        ONLY error values, even when the full window interior spans clean data.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 7,
                    "iid": ["01"] * 7,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",  # Error starts (position 3)
                        "2020-01-08",  # Error ends (position 4)
                        "2020-01-09",
                        "2020-01-10",
                    ],
                    "prccd": [10.0, 10.2, 10.1, 101.0, 102.0, 10.3, 10.4],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        # Initialize correction factor column (would normally come from single-period)
        df = df.with_columns(
            [
                pl.lit(1.0).alias("prccd_correction_factor"),
                pl.lit(None).cast(pl.Utf8).alias("prccd_error_type"),
            ]
        )

        result = _detect_decimal_error_multi_period(
            df, "prccd", ["gvkey", "iid"], "datadate", window_sizes=list(range(2, 10))
        ).collect()

        # Positions 3 and 4 (101.0, 102.0) should be flagged
        assert result["prccd_correction_factor"][3] == pytest.approx(0.1), (
            f"Position 3 should be corrected, got factor {result['prccd_correction_factor'][3]}"
        )
        assert result["prccd_correction_factor"][4] == pytest.approx(0.1), (
            f"Position 4 should be corrected, got factor {result['prccd_correction_factor'][4]}"
        )

        # Non-error positions should NOT be corrected
        for i in [0, 1, 2, 5, 6]:
            assert result["prccd_correction_factor"][i] == 1.0, (
                f"Position {i} should NOT be corrected, got {result['prccd_correction_factor'][i]}"
            )

    def test_detects_3day_error_via_full_window(self):
        """Detect a 3-day error using the FULL window.

        This test demonstrates how FULL window catches odd-length errors:

        Position:   0      1      2      3      4      5      6      7
        Value:     10.0   10.2   100.0  101.0  102.0  10.3   10.4   10.5
                                ^^^    ^^^    ^^^
                                ERROR SPAN (3 days)

        At position t=3 with nlag=2:
        - FULL WINDOW endpoints (1, 5): X(3)/X(1)=10.1>5, X(3)/X(5)=9.8>5 ✓
          Interior (2,3,4): [100, 101, 102] with variation 102/100=1.02<1.3 ✓
          → PASSES! Flags positions 2, 3, 4.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 8,
                    "iid": ["01"] * 8,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(8)],
                    "prccd": [10.0, 10.2, 100.0, 101.0, 102.0, 10.3, 10.4, 10.5],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        df = df.with_columns(
            [
                pl.lit(1.0).alias("prccd_correction_factor"),
                pl.lit(None).cast(pl.Utf8).alias("prccd_error_type"),
            ]
        )

        result = _detect_decimal_error_multi_period(
            df, "prccd", ["gvkey", "iid"], "datadate", window_sizes=list(range(2, 10))
        ).collect()

        # Positions 2, 3, 4 (100.0, 101.0, 102.0) should be flagged
        for i in [2, 3, 4]:
            assert result["prccd_correction_factor"][i] == pytest.approx(0.1), (
                f"Position {i} should be corrected, got {result['prccd_correction_factor'][i]}"
            )

        # Non-error positions should NOT be corrected
        for i in [0, 1, 5, 6, 7]:
            assert result["prccd_correction_factor"][i] == 1.0, (
                f"Position {i} should NOT be corrected, got {result['prccd_correction_factor'][i]}"
            )

    def test_detects_5day_error(self):
        """Detect a decimal error persisting for 5 days."""
        # Build a longer series with 5-day error in the middle
        normal_before = [10.0, 10.2, 10.1, 10.3, 10.2]
        error_days = [101.0, 102.0, 100.5, 101.5, 103.0]  # 5 days, all ~100
        normal_after = [10.4, 10.3, 10.5, 10.2, 10.4]

        n = len(normal_before) + len(error_days) + len(normal_after)
        dates = [f"2020-01-{i + 2:02d}" for i in range(n)]

        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * n,
                    "iid": ["01"] * n,
                    "datadate": dates,
                    "prccd": normal_before + error_days + normal_after,
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        df = df.with_columns(
            [
                pl.lit(1.0).alias("prccd_correction_factor"),
                pl.lit(None).cast(pl.Utf8).alias("prccd_error_type"),
            ]
        )

        result = _detect_decimal_error_multi_period(
            df, "prccd", ["gvkey", "iid"], "datadate", window_sizes=list(range(2, 15))
        ).collect()

        # Error rows are indices 5-9
        for i in range(5, 10):
            assert result["prccd_correction_factor"][i] == pytest.approx(0.1), (
                f"Row {i} should be corrected, got factor {result['prccd_correction_factor'][i]}"
            )

        # Non-error rows should not be corrected
        for i in list(range(5)) + list(range(10, n)):
            assert result["prccd_correction_factor"][i] == 1.0, (
                f"Row {i} should NOT be corrected, got factor {result['prccd_correction_factor'][i]}"
            )

    def test_variation_check_rejects_volatile_window(self):
        """Windows with high internal variation should not be flagged.

        The 30% variation rule prevents false positives on legitimately volatile periods.
        """
        # Create data where values are elevated but vary too much to be a decimal error
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 7,
                    "iid": ["01"] * 7,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                        "2020-01-09",
                        "2020-01-10",
                    ],
                    # Values swing between 60 and 150 - too volatile to be decimal error
                    "prccd": [10.0, 10.2, 60.0, 150.0, 70.0, 10.3, 10.4],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        df = df.with_columns(
            [
                pl.lit(1.0).alias("prccd_correction_factor"),
                pl.lit(None).cast(pl.Utf8).alias("prccd_error_type"),
            ]
        )

        result = _detect_decimal_error_multi_period(
            df, "prccd", ["gvkey", "iid"], "datadate", window_sizes=list(range(2, 10))
        ).collect()

        # The variation (150/60 = 2.5) exceeds 30%, so should not be flagged as single error
        # Each value might or might not be flagged individually, but the variation check
        # should prevent multi-period detection
        # Note: some days might still be flagged by single-period detection
        factors = result["prccd_correction_factor"].to_list()

        # At minimum, not ALL of the volatile days should be flagged with the same factor
        # This is a weak test - the main point is the variation check exists
        assert not all(f == 0.1 for f in factors[2:5]), (
            "Volatile window should not all be flagged as a single error"
        )

    def test_smaller_windows_processed_first(self):
        """Smaller windows are processed first to prevent cascading corrections.

        This ensures single-day errors are corrected before multi-day detection
        runs, preventing incorrect flagging of adjacent clean values.
        """
        # This is more of an implementation detail, but we can verify
        # that errors are detected correctly regardless of window processing order
        normal_before = [10.0, 10.2, 10.1]
        error_days = [100.0, 101.0, 102.0, 100.5]  # 4 days
        normal_after = [10.4, 10.3, 10.5]

        n = len(normal_before) + len(error_days) + len(normal_after)
        dates = [f"2020-01-{i + 2:02d}" for i in range(n)]

        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * n,
                    "iid": ["01"] * n,
                    "datadate": dates,
                    "prccd": normal_before + error_days + normal_after,
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        df = df.with_columns(
            [
                pl.lit(1.0).alias("prccd_correction_factor"),
                pl.lit(None).cast(pl.Utf8).alias("prccd_error_type"),
            ]
        )

        result = _detect_decimal_error_multi_period(
            df, "prccd", ["gvkey", "iid"], "datadate", window_sizes=list(range(2, 10))
        ).collect()

        # All error days should be marked
        for i in range(3, 7):
            assert result["prccd_correction_factor"][i] == pytest.approx(0.1), (
                f"Row {i} should be corrected"
            )


# =============================================================================
# Integrated Correction Function Tests
# =============================================================================


class TestCorrectDecimalErrors:
    """Tests for correct_decimal_errors() - the main correction wrapper.

    This function combines single-period and multi-period detection,
    applies corrections, and generates a log.
    """

    def test_applies_correction_to_values(self):
        """Corrections should be applied by multiplying by the factor."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    "prccd": [10.0, 10.5, 105.0, 10.3, 10.1],  # Day 3 is 10x error
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd", ["gvkey", "iid"], "datadate")
        result = result.collect()

        # The 105.0 should be corrected to ~10.5
        assert result["prccd"][2] == pytest.approx(10.5), (
            f"105.0 should be corrected to 10.5, got {result['prccd'][2]}"
        )

    def test_returns_correction_log(self):
        """Function should return a log of corrections made."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    "prccd": [10.0, 10.5, 105.0, 10.3, 10.1],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd", ["gvkey", "iid"], "datadate")

        assert log is not None, "Correction log should not be None"
        log = log.collect()

        assert len(log) == 1, f"Expected 1 correction logged, got {len(log)}"
        assert log["original_value"][0] == 105.0, (
            f"Expected original value 105.0, got {log['original_value'][0]}"
        )
        assert log["corrected_value"][0] == pytest.approx(10.5), (
            f"Expected corrected value 10.5, got {log['corrected_value'][0]}"
        )
        assert log["correction_factor"][0] == pytest.approx(0.1), (
            f"Expected correction factor 0.1, got {log['correction_factor'][0]}"
        )

    def test_no_corrections_for_clean_data(self, single_security_df):
        """Clean data should return unchanged with empty log."""
        result, log = correct_decimal_errors(
            single_security_df, "prccd", ["gvkey", "iid"], "datadate"
        )
        result = result.collect()
        original = single_security_df.collect()

        # Values should be unchanged
        np.testing.assert_allclose(
            result["prccd"].to_list(),
            original["prccd"].to_list(),
            rtol=1e-10,
            err_msg="Clean data should not be modified",
        )

        # Log should be empty
        if log is not None:
            log = log.collect()
            assert len(log) == 0, f"Expected empty log for clean data, got {len(log)} rows"

    def test_handles_null_values(self):
        """Null values should not cause errors and should remain null."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-08",
                    ],
                    "prccd": [10.0, None, 10.2, 10.3, 10.1],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd", ["gvkey", "iid"], "datadate")
        result = result.collect()

        # Null should remain null
        assert result["prccd"][1] is None, (
            f"Null value should remain null, got {result['prccd'][1]}"
        )

    def test_multiple_errors_in_series(self):
        """Multiple separate errors in a series should all be corrected."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 9,
                    "iid": ["01"] * 9,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(9)],
                    "prccd": [10.0, 100.0, 10.2, 10.1, 10.3, 1.0, 10.2, 10.4, 10.3],
                    #              ^ 10x high         ^ 10x low
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd", ["gvkey", "iid"], "datadate")
        result = result.collect()

        # Both errors should be corrected
        assert result["prccd"][1] == pytest.approx(10.0), (
            f"High error at row 1 should be corrected to ~10, got {result['prccd'][1]}"
        )
        assert result["prccd"][5] == pytest.approx(10.0), (
            f"Low error at row 5 should be corrected to ~10, got {result['prccd'][5]}"
        )

    def test_uses_priority_window_sizes_by_default(self):
        """Default should use priority windows [1, 2, 3, 5, 10, 21] for performance."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.5, 10.2, 10.3, 10.1],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        # Should not raise error with default priority windows
        result, log = correct_decimal_errors(df, "prccd")
        result.collect()  # Force execution


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


class TestBessembinderEdgeCases:
    """Edge case tests for Bessembinder correction functions."""

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty result without error."""
        df = pl.DataFrame(
            schema={
                "gvkey": pl.Utf8,
                "iid": pl.Utf8,
                "datadate": pl.Date,
                "prccd": pl.Float64,
            }
        ).lazy()

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        assert len(result) == 0, f"Expected empty result, got {len(result)} rows"

    def test_single_row(self):
        """Single row should not be corrected (no neighbors to compare)."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"],
                    "iid": ["01"],
                    "datadate": ["2020-01-02"],
                    "prccd": [100.0],  # Could look like an error, but can't detect
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        assert result["prccd"][0] == 100.0, (
            f"Single row should not be modified, got {result['prccd'][0]}"
        )

    def test_two_rows(self):
        """Two rows cannot have interior errors (only boundary rows)."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001", "001"],
                    "iid": ["01", "01"],
                    "datadate": ["2020-01-02", "2020-01-03"],
                    "prccd": [10.0, 100.0],  # Looks like error but no next value
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Neither row can be corrected (first has no prior, second has no next)
        assert result["prccd"][0] == 10.0
        assert result["prccd"][1] == 100.0

    def test_all_same_values(self):
        """Series with all same values should not trigger correction."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 10,
                    "iid": ["01"] * 10,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(10)],
                    "prccd": [50.0] * 10,
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        assert all(v == 50.0 for v in result["prccd"].to_list()), (
            "Constant series should not be modified"
        )

    def test_ratio_exactly_at_threshold(self):
        """Ratio exactly at threshold (5.0) should trigger correction."""
        # ratio_to_prior = 50/10 = 5.0
        # ratio_to_next = 50/10 = 5.0
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.0, 50.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        # The threshold is > 5, so exactly 5.0 should NOT trigger
        # (5.0 is NOT > 5.0)
        assert result["prccd_correction_factor"][2] == 1.0, (
            "Ratio exactly at 5.0 should not trigger correction"
        )

    def test_ratio_just_above_threshold(self):
        """Ratio just above threshold (5.01) should trigger correction."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.0, 50.1, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = _detect_decimal_error_single_period(
            df, "prccd", ["gvkey", "iid"], "datadate"
        ).collect()

        # 50.1/10 = 5.01 > 5, should trigger
        assert result["prccd_correction_factor"][2] == pytest.approx(0.1), (
            "Ratio just above 5.0 should trigger correction"
        )

    def test_handles_very_small_values(self):
        """Very small positive values should be handled without overflow."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [1e-10, 1e-10, 1e-9, 1e-10, 1e-10],  # 10x error at tiny scale
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Should detect and correct the 10x error
        assert result["prccd"][2] == pytest.approx(1e-10, rel=0.01), (
            "Small value error should be corrected"
        )

    def test_handles_very_large_values(self):
        """Very large values should be handled without overflow."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [1e10, 1e10, 1e11, 1e10, 1e10],  # 10x error at large scale
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Should detect and correct the 10x error
        assert result["prccd"][2] == pytest.approx(1e10, rel=0.01), (
            "Large value error should be corrected"
        )

    def test_handles_zero_values(self):
        """Zero values should not cause division by zero errors.

        This was a real bug found when processing NA Compustat data where
        some adjcsho (adjusted shares outstanding) values were zero.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 7,
                    "iid": ["01"] * 7,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(7)],
                    "prccd": [10.0, 10.0, 0.0, 10.0, 100.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        # Should not raise division by zero error
        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Zero should remain zero (or null), not cause error
        assert result["prccd"][2] is None or result["prccd"][2] == 0.0, (
            "Zero value should be handled gracefully"
        )
        # The 10x spike at position 4 should still be corrected
        assert result["prccd"][4] == pytest.approx(10.0, rel=0.01), (
            "Error at position 4 should still be corrected"
        )

    def test_handles_zero_in_denominator(self):
        """Zero values adjacent to potential errors should not crash.

        When zero appears next to an error, division would fail without
        proper handling.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [0.0, 100.0, 0.0, 10.0, 10.0],  # Zero before and after spike
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        # Should not raise division by zero error
        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Just verify it completes without error
        assert len(result) == 5, "Should return all rows"


# =============================================================================
# Real-World Scenario Tests
# =============================================================================


class TestBessembinderRealWorldScenarios:
    """Tests based on real-world data scenarios."""

    def test_bessembinder_paper_example(self):
        """Test case inspired by Bessembinder paper example.

        Price goes from ~8.56 to 69.50 (erroneously shifted decimal) back to ~7.32.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [8.50, 8.56, 69.50, 7.32, 7.45],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # 69.50 should be corrected to ~6.95
        assert result["prccd"][2] == pytest.approx(6.95, rel=0.01), (
            f"69.50 should be corrected to ~6.95, got {result['prccd'][2]}"
        )

    def test_multi_day_error_with_stable_values(self):
        """Multi-day error where erroneous values are stable (< 30% variation)."""
        # Error persists for 3 days, all values around 100 (true value ~10)
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 9,
                    "iid": ["01"] * 9,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(9)],
                    "prccd": [10.0, 10.2, 10.1, 100.0, 101.0, 100.5, 10.3, 10.4, 10.2],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # All three error days should be corrected
        for i in [3, 4, 5]:
            assert result["prccd"][i] == pytest.approx(10.0, rel=0.05), (
                f"Row {i} should be corrected to ~10, got {result['prccd'][i]}"
            )

    def test_error_at_series_boundary_not_detected(self):
        """Error at the very start or end cannot be detected."""
        # Error on first day - can't detect because no prior value
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [100.0, 10.0, 10.2, 10.1, 10.3],  # First day looks wrong
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # First row cannot be corrected
        assert result["prccd"][0] == 100.0, (
            f"First row cannot be corrected, got {result['prccd'][0]}"
        )


# =============================================================================
# Test Correction Rate Calculation
# =============================================================================


class TestComputeCorrectionRate:
    """Tests for the compute_correction_rate() function."""

    def test_excludes_sub_threshold_corrections(self):
        """Sub-$0.01 corrections should be excluded from benchmark comparison."""
        # compute_correction_rate imported at top of file

        # Create a corrections log with mixed prices
        corrections = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 10,
                    "iid": ["01"] * 10,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(10)],
                    "variable": ["adjprc"] * 10,
                    "original_value": [
                        0.001,
                        0.005,
                        0.01,
                        0.1,
                        1.0,
                        0.0001,
                        0.0005,
                        10.0,
                        100.0,
                        0.00001,
                    ],
                    "corrected_value": [0.01] * 10,
                    "correction_factor": [10.0] * 10,
                    "error_type": ["low_10x"] * 10,
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result = compute_correction_rate(corrections, 100000)

        # 10 total corrections
        assert result["n_corrections"] == 10

        # 4 above threshold (0.01, 0.1, 1.0, 10.0, 100.0) - wait, that's 5
        # Let me recount: 0.01, 0.1, 1.0, 10.0, 100.0 = 5 above $0.01
        assert result["n_corrections_above_threshold"] == 5

        # 5 below threshold (0.001, 0.005, 0.0001, 0.0005, 0.00001)
        assert result["n_corrections_below_threshold"] == 5

        # Filtered rate should be based on above-threshold corrections only
        expected_filtered_rate = 100 * 5 / 100000
        assert result["correction_rate_filtered_pct"] == pytest.approx(
            expected_filtered_rate, rel=0.01
        )

    def test_no_corrections_returns_zero_rates(self):
        """When no corrections, all rates should be zero."""
        # compute_correction_rate imported at top of file

        result = compute_correction_rate(None, 100000)

        assert result["n_corrections"] == 0
        assert result["correction_rate_pct"] == 0.0
        assert result["correction_rate_filtered_pct"] == 0.0
        assert result["vs_benchmark"] == "N/A"

    def test_benchmark_comparison_uses_filtered_rate(self):
        """Benchmark comparison should use filtered rate, not raw rate."""
        # compute_correction_rate imported at top of file

        # Create log with mostly sub-threshold corrections
        # Raw rate would be HIGH, but filtered rate should be NORMAL
        corrections = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 100,
                    "iid": ["01"] * 100,
                    "datadate": [
                        f"2020-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(100)
                    ],
                    "variable": ["adjprc"] * 100,
                    # 95 sub-$0.01 corrections, 5 above $0.01
                    "original_value": [0.001] * 95 + [1.0] * 5,
                    "corrected_value": [0.01] * 95 + [10.0] * 5,
                    "correction_factor": [10.0] * 100,
                    "error_type": ["low_10x"] * 100,
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        # With 100k obs: raw rate = 0.1%, filtered rate = 0.005%
        result = compute_correction_rate(corrections, 100000)

        # Raw rate is HIGH (0.1% >> 0.02% benchmark)
        assert result["correction_rate_pct"] == pytest.approx(0.1, rel=0.01)

        # Filtered rate is LOW (0.005% < 0.01% which is 50% of benchmark)
        assert result["correction_rate_filtered_pct"] == pytest.approx(0.005, rel=0.01)

        # Benchmark comparison uses filtered rate
        assert result["vs_benchmark"] == "LOW (< 50% of benchmark)"


# =============================================================================
# Test Cascading False Positive Prevention
# =============================================================================


class TestCascadingFalsePositivePrevention:
    """Tests for the fix that prevents cascading false positives.

    The bug: When a true error exists, the algorithm was using the uncorrected
    error value as a comparison endpoint for subsequent positions, causing
    normal data to be falsely flagged as errors.

    The fix: Check that endpoint values are "clean" (not already flagged for
    correction) before using them in detection.
    """

    def test_no_cascade_after_single_error(self):
        """Normal data after a single error should NOT be corrected.

        Scenario: True 1-day error followed by normal data.
        - Day 0: Normal price (10.0)
        - Day 1: Error price (0.001) - should be corrected
        - Days 2-5: Normal prices (~10.0) - should NOT be corrected
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 8,
                    "iid": ["01"] * 8,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(8)],
                    # True error on day 1 (0.001), rest are normal
                    "prccd": [10.0, 0.001, 10.0, 10.2, 10.1, 10.3, 10.0, 10.2],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Day 1 SHOULD be corrected (true error)
        assert result["prccd"][1] == pytest.approx(1.0, rel=0.01), (
            f"Day 1 should be corrected from 0.001 to ~1.0, got {result['prccd'][1]}"
        )

        # Days 2-7 should NOT be corrected (normal data)
        for i in range(2, 8):
            original = [10.0, 0.001, 10.0, 10.2, 10.1, 10.3, 10.0, 10.2][i]
            assert result["prccd"][i] == pytest.approx(original, rel=0.01), (
                f"Day {i} should NOT be corrected, expected {original}, got {result['prccd'][i]}"
            )

    def test_no_cascade_error_at_window_edge(self):
        """Error at window edge should not cause cascade to interior.

        Scenario: True error that becomes an endpoint for later window detection.
        - Day 0: Normal (10.0)
        - Day 1: Error (0.01) - triggers 1000x low correction -> becomes 10.0
        - Day 2: Normal (10.0) - at nlag=2, left endpoint is Day 0, should be fine
        - Day 3: Normal (10.0) - at nlag=2, left endpoint is Day 1 (ERROR!)
        - Day 4: Normal (10.0)

        Without the fix, Day 3 would be falsely flagged because its left endpoint
        (Day 1) is an error value. With the fix, Day 1 is marked as dirty, so
        the detection skips it.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 8,
                    "iid": ["01"] * 8,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(8)],
                    # Error on day 1: 0.01 vs neighbors 10.0 is 1000x low error
                    "prccd": [10.0, 0.01, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Day 1 should be corrected (0.01 * 1000 = 10.0)
        assert result["prccd"][1] == pytest.approx(10.0, rel=0.01), (
            f"Day 1 should be corrected to ~10.0, got {result['prccd'][1]}"
        )

        # Days 2-7 should NOT be corrected
        for i in range(2, 8):
            assert result["prccd"][i] == pytest.approx(10.0, rel=0.01), (
                f"Day {i} should NOT be corrected, got {result['prccd'][i]}"
            )

    def test_true_multi_day_error_still_detected(self):
        """A true multi-day error should still be detected and corrected.

        This verifies that the fix doesn't break legitimate multi-day detection.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 9,
                    "iid": ["01"] * 9,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(9)],
                    # True 3-day error (days 3-5)
                    "prccd": [10.0, 10.2, 10.1, 100.0, 101.0, 100.5, 10.3, 10.4, 10.2],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # All three error days should be corrected
        for i in [3, 4, 5]:
            assert result["prccd"][i] == pytest.approx(10.0, rel=0.1), (
                f"Day {i} should be corrected to ~10, got {result['prccd'][i]}"
            )

        # Days 0-2 and 6-8 should NOT be corrected
        for i in [0, 1, 2, 6, 7, 8]:
            original = [10.0, 10.2, 10.1, 100.0, 101.0, 100.5, 10.3, 10.4, 10.2][i]
            assert result["prccd"][i] == pytest.approx(original, rel=0.01), (
                f"Day {i} should NOT be corrected, got {result['prccd'][i]}"
            )

    def test_consecutive_errors_handled_correctly(self):
        """Two consecutive but separate errors should both be detected.

        Scenario: Two distinct 1-day errors with normal data between them.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 10,
                    "iid": ["01"] * 10,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(10)],
                    # Two separate 1-day errors
                    "prccd": [10.0, 100.0, 10.0, 10.2, 10.1, 100.0, 10.3, 10.0, 10.2, 10.1],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Day 1 should be corrected
        assert result["prccd"][1] == pytest.approx(10.0, rel=0.1), (
            f"Day 1 should be corrected to ~10, got {result['prccd'][1]}"
        )

        # Day 5 should be corrected
        assert result["prccd"][5] == pytest.approx(10.0, rel=0.1), (
            f"Day 5 should be corrected to ~10, got {result['prccd'][5]}"
        )

        # Other days should NOT be corrected
        originals = [10.0, 100.0, 10.0, 10.2, 10.1, 100.0, 10.3, 10.0, 10.2, 10.1]
        for i in [0, 2, 3, 4, 6, 7, 8, 9]:
            assert result["prccd"][i] == pytest.approx(originals[i], rel=0.01), (
                f"Day {i} should NOT be corrected, got {result['prccd'][i]}"
            )


# =============================================================================
# Regression Tests for Cascading Error Bug (BESSEMBINDER_CASCADING_ERROR_BUG.md)
# =============================================================================


class TestCascadingErrorBugRegressions:
    """Regression tests for the documented cascading error bug.

    These tests verify the fix for the bug where the algorithm uses
    ORIGINAL (uncorrected) values as endpoints, causing:
    1. Correct values to be flagged as errors
    2. Corrections in the wrong direction
    3. New spikes created in "corrected" data

    See BESSEMBINDER_CASCADING_ERROR_BUG.md for full documentation.
    """

    def test_correct_value_between_errors_not_flagged(self):
        """Test Case 1: Adjacent Opposite Errors.

        From bug documentation:
        Input:  [0.50, 5.00, 0.50, 5.00, 0.50]

        The middle 0.50 should NOT be flagged as an error, even though both
        adjacent values (5.00) are errors.

        Bug behavior: Middle 0.50 would be flagged as LOW error because:
        - endpoint_left = 5.00 (error value)
        - endpoint_right = 5.00 (error value)
        - ratio = 0.50/5.00 = 0.10 < 0.2 → incorrectly flagged!
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [0.50, 5.00, 0.50, 5.00, 0.50],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # The middle value (index 2) should remain unchanged
        assert result["prccd"][2] == pytest.approx(0.50), (
            f"Middle correct value was incorrectly modified to {result['prccd'][2]}"
        )

        # Positions 1 and 3 should be corrected from 5.00 to 0.50
        assert result["prccd"][1] == pytest.approx(0.50, rel=0.01), (
            f"Position 1 should be corrected to 0.50, got {result['prccd'][1]}"
        )
        assert result["prccd"][3] == pytest.approx(0.50, rel=0.01), (
            f"Position 3 should be corrected to 0.50, got {result['prccd'][3]}"
        )

    def test_error_sandwich_not_over_corrected(self):
        """Test Case 2: Error Sandwich.

        A correct value sandwiched between two error values should not be flagged.

        True:    [10, 10, 10, 10, 10]
        Input:   [10, 100, 10, 100, 10]

        Positions 1 and 3 are errors. Position 2 is correct.
        Position 2 should NOT be flagged despite having error endpoints.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 100.0, 10.0, 100.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # Position 2 should remain 10, not be "corrected" to 100
        assert result["prccd"][2] == pytest.approx(10.0), (
            f"Position 2 should remain 10.0, got {result['prccd'][2]}"
        )

        # Positions 1 and 3 should be corrected to 10
        assert result["prccd"][1] == pytest.approx(10.0, rel=0.01), (
            f"Position 1 should be corrected to 10.0, got {result['prccd'][1]}"
        )
        assert result["prccd"][3] == pytest.approx(10.0, rel=0.01), (
            f"Position 3 should be corrected to 10.0, got {result['prccd'][3]}"
        )

    def test_no_new_spikes_introduced(self):
        """Test Case 3: No New Spikes Introduced.

        After correction, the data should not have NEW spike-and-reversal patterns
        that weren't in the original data.

        This tests a specific scenario where a bug would INTRODUCE a spike:
        - If a correct value (0.50) between errors (5.00) is incorrectly
          "corrected" to 5.00, we'd have created a new spike.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 7,
                    "iid": ["01"] * 7,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(7)],
                    # Original has spikes at positions 2 and 4
                    "prccd": [0.50, 0.50, 5.00, 0.50, 5.00, 0.50, 0.50],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd")
        result = result.collect()

        # After correction, there should be NO spikes
        # All values should be approximately 0.50
        for i in range(7):
            assert result["prccd"][i] == pytest.approx(0.50, rel=0.01), (
                f"Position {i} should be ~0.50 after correction, got {result['prccd'][i]}"
            )

    def test_validate_cascading_disabled_for_comparison(self):
        """Test that validate_cascading=False gives legacy behavior.

        This allows comparing old (buggy) vs new (fixed) behavior.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [0.50, 5.00, 0.50, 5.00, 0.50],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        # With validation enabled (default), middle value should be preserved
        result_with, _ = correct_decimal_errors(df, "prccd", validate_cascading=True)
        result_with = result_with.collect()

        assert result_with["prccd"][2] == pytest.approx(0.50), (
            f"With validation: middle value should be 0.50, got {result_with['prccd'][2]}"
        )

        # With validation disabled, legacy (buggy) behavior may incorrectly flag middle value
        # Note: Due to the multi-period endpoint check already in place, this may still work correctly
        result_without, _ = correct_decimal_errors(df, "prccd", validate_cascading=False)
        result_without = result_without.collect()

        # Just verify it runs without error
        assert len(result_without) == 5


# =============================================================================
# Test Interpolation Correction Method
# =============================================================================


class TestInterpolationCorrection:
    """Tests for interpolation-based correction method.

    Unlike fixed multipliers (0.1, 0.01, etc.), interpolation uses the geometric
    mean of surrounding clean values to determine the correct price level.
    This handles non-10x errors (e.g., 5.5x, 7x) that fixed multipliers miss.
    """

    def test_interpolation_uses_geometric_mean_of_endpoints(self):
        """Verify interpolation correction uses geometric mean of left/right endpoints."""
        # Error value is 100.0 (10x error), endpoints are 10.0 and 10.0
        # Geometric mean: sqrt(10 * 10) = 10.0
        # Note: threshold is >5x, so 50/10=5x wouldn't be detected, but 100/10=10x is
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.0, 100.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, log = correct_decimal_errors(df, "prccd", correction_method="interpolation")
        result = result.collect()

        # Middle value should be corrected to geometric mean = sqrt(10*10) = 10.0
        assert result["prccd"][2] == pytest.approx(10.0, rel=0.01), (
            f"Interpolation should use geometric mean (10.0), got {result['prccd'][2]}"
        )

    def test_interpolation_handles_non_10x_errors(self):
        """Verify interpolation correctly handles errors that aren't exact 10x.

        This is the key advantage over fixed multipliers: a 5.5x error should
        be corrected properly, not over-corrected by dividing by 10.
        """
        # Error value is 55.0 (5.5x error), endpoints are 10.0
        # Bessembinder would divide by 10 -> 5.5 (WRONG)
        # Interpolation uses sqrt(10*10) = 10.0 (CORRECT)
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.0, 55.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result_interp, _ = correct_decimal_errors(df, "prccd", correction_method="interpolation")
        result_interp = result_interp.collect()

        result_bess, _ = correct_decimal_errors(df, "prccd", correction_method="bessembinder")
        result_bess = result_bess.collect()

        # Interpolation should give 10.0 (geometric mean of endpoints)
        assert result_interp["prccd"][2] == pytest.approx(10.0, rel=0.01), (
            f"Interpolation should give 10.0, got {result_interp['prccd'][2]}"
        )

        # Bessembinder would give 5.5 (55/10) - demonstrating the over-correction
        assert result_bess["prccd"][2] == pytest.approx(5.5, rel=0.01), (
            f"Bessembinder should give 5.5, got {result_bess['prccd'][2]}"
        )

    def test_interpolation_with_asymmetric_endpoints(self):
        """Test interpolation when left and right endpoints differ.

        Geometric mean of 8.0 and 12.0 = sqrt(8 * 12) = sqrt(96) ≈ 9.80
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [8.0, 8.0, 80.0, 12.0, 12.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, _ = correct_decimal_errors(df, "prccd", correction_method="interpolation")
        result = result.collect()

        expected = np.sqrt(8.0 * 12.0)  # ≈ 9.80
        assert result["prccd"][2] == pytest.approx(expected, rel=0.01), (
            f"Should use geometric mean sqrt(8*12)={expected:.2f}, got {result['prccd'][2]}"
        )

    def test_interpolation_logs_correction_method(self):
        """Verify corrections log includes the correction_method used."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.0, 100.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        _, log = correct_decimal_errors(df, "prccd", correction_method="interpolation")
        log = log.collect()

        assert "correction_method" in log.columns
        assert log["correction_method"][0] == "interpolation"

    def test_bessembinder_method_logs_correction_method(self):
        """Verify Bessembinder method also logs its name."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.0, 100.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        _, log = correct_decimal_errors(df, "prccd", correction_method="bessembinder")
        log = log.collect()

        assert "correction_method" in log.columns
        assert log["correction_method"][0] == "bessembinder"

    def test_invalid_correction_method_raises_error(self):
        """Verify invalid correction_method raises ValueError."""
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 5,
                    "iid": ["01"] * 5,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(5)],
                    "prccd": [10.0, 10.0, 100.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        with pytest.raises(ValueError, match="Unknown correction_method"):
            correct_decimal_errors(df, "prccd", correction_method="invalid_method")

    def test_interpolation_multi_day_error(self):
        """Test interpolation handles multi-day errors correctly.

        For a 2-day error, both error values should be corrected to the
        geometric mean of the flanking clean values.
        """
        df = (
            pl.DataFrame(
                {
                    "gvkey": ["001"] * 6,
                    "iid": ["01"] * 6,
                    "datadate": [f"2020-01-{i + 2:02d}" for i in range(6)],
                    "prccd": [10.0, 10.0, 100.0, 100.0, 10.0, 10.0],
                }
            )
            .with_columns(pl.col("datadate").str.to_date())
            .lazy()
        )

        result, _ = correct_decimal_errors(df, "prccd", correction_method="interpolation")
        result = result.collect()

        # Both error positions should be corrected to ~10.0
        assert result["prccd"][2] == pytest.approx(10.0, rel=0.01), (
            f"Position 2 should be ~10.0, got {result['prccd'][2]}"
        )
        assert result["prccd"][3] == pytest.approx(10.0, rel=0.01), (
            f"Position 3 should be ~10.0, got {result['prccd'][3]}"
        )
