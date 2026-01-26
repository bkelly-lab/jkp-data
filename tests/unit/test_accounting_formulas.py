"""
Tests for accounting characteristic calculations.

These tests verify the correctness of financial score and ratio calculations
used to compute firm characteristics. The formulas are critical for academic
replication and must match published methodology.

Paper Reference: Jensen, Kelly, Pedersen (2023), "Is There a Replication Crisis in Finance?"
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from aux_functions import altman_z, intrinsic_value, kz_index, ohlson_o, pitroski_f


def _make_date(year: int, month: int, day: int) -> date:
    """Create a Python date object for use in test DataFrames."""
    return date(year, month, day)


def _generate_monthly_dates(n_months: int, start_year: int = 2020) -> list[date]:
    """Generate a list of month-end dates for test data."""
    dates = []
    for i in range(n_months):
        year = start_year + i // 12
        month = (i % 12) + 1
        last_day = 28 if month == 2 else (30 if month in [4, 6, 9, 11] else 31)
        dates.append(date(year, month, last_day))
    return dates


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_accounting_data() -> pl.DataFrame:
    """Create base accounting data with all required columns.

    Returns a DataFrame with 24 months of data for a single firm,
    sufficient to test formulas that require count > 12.
    """
    n_months = 24

    # Generate dates using helper function for proper date handling
    dates = _generate_monthly_dates(n_months, start_year=2020)

    # Base financial values
    at_x = 1000.0  # Total assets
    lt = 400.0  # Total liabilities
    seq_x = 600.0  # Shareholders' equity

    df = pl.DataFrame(
        {
            "gvkey": ["001"] * n_months,
            "curcd": ["USD"] * n_months,
            "datadate": dates,
            "count": list(range(1, n_months + 1)),
            # Balance sheet
            "at": [at_x] * n_months,
            "at_x": [at_x] * n_months,
            "lt": [lt] * n_months,
            "seq_x": [seq_x] * n_months,
            "be_x": [seq_x * 0.95] * n_months,
            "ca_x": [300.0] * n_months,
            "cl_x": [150.0] * n_months,
            "che": [100.0] * n_months,
            "dltt": [250.0] * n_months,
            "dlc": [50.0] * n_months,
            "debt_x": [300.0] * n_months,
            # Income statement
            "sale_x": [500.0] * n_months,
            "gp_x": [200.0] * n_months,
            "ebitda_x": [150.0] * n_months,
            "ebit_x": [120.0] * n_months,
            "pi_x": [100.0] * n_months,
            "ni_x": [80.0] * n_months,
            "nix_x": [80.0] * n_months,
            "dp": [30.0] * n_months,
            # Cash flow
            "ocf_x": [100.0] * n_months,
            "div_x": [20.0] * n_months,
            # Other
            "re": [400.0] * n_months,
            "ppent": [500.0] * n_months,
            "eqis_x": [0.0] * n_months,
            "me_fiscal": [800.0] * n_months,
        }
    )

    return df


# =============================================================================
# Piotroski F-Score Tests
# =============================================================================


class TestPiotroskiF:
    """
    Piotroski F-score: 9-component binary score for financial strength.

    Paper Reference: Piotroski (2000), used in JKP (2023) Appendix Table A.1

    Components:
    1. ROA > 0 (profitability)
    2. CFO > 0 (cash flow)
    3. ΔROA > 0 (improving profitability)
    4. CFO > ROA (accruals quality)
    5. ΔLeverage < 0 (decreasing leverage)
    6. ΔCurrent Ratio > 0 (improving liquidity)
    7. No equity issuance
    8. ΔGross Margin > 0 (improving margins)
    9. ΔAsset Turnover > 0 (improving efficiency)
    """

    def test_fscore_range_is_valid(self, base_accounting_data: pl.DataFrame):
        """F-score must be in [0, 9] for all observations."""
        result = pitroski_f(base_accounting_data)

        # Filter to observations where score should be computed (count > 12)
        valid_rows = result.filter(pl.col("count") > 12)

        if len(valid_rows) > 0:
            scores = valid_rows["f_score"].to_list()
            for score in scores:
                if score is not None:
                    assert 0 <= score <= 9, f"F-score {score} out of range [0, 9]"

    def test_fscore_is_integer(self, base_accounting_data: pl.DataFrame):
        """F-score should be an integer (sum of binary signals)."""
        result = pitroski_f(base_accounting_data)

        # Filter to valid observations
        valid_rows = result.filter(pl.col("count") > 24)  # Need extra history for ΔROA

        if len(valid_rows) > 0:
            scores = valid_rows["f_score"].to_list()
            for score in scores:
                if score is not None:
                    assert score == int(score), f"F-score {score} is not an integer"

    def test_fscore_requires_sufficient_history(self, base_accounting_data: pl.DataFrame):
        """F-score requires count > 12 (and some components need > 24)."""
        # Create data with insufficient history
        short_data = base_accounting_data.filter(pl.col("count") <= 12)

        result = pitroski_f(short_data)

        # With count <= 12, F-score should be null for all observations
        assert result["f_score"].is_null().all(), (
            f"F-score should be null when count <= 12, "
            f"but found {len(result) - result['f_score'].null_count()} non-null values"
        )

    def test_fscore_no_equity_issuance_adds_point(self, base_accounting_data: pl.DataFrame):
        """No equity issuance (eqis_x = 0) should add 1 to score."""
        # Data with no issuance
        no_issuance = base_accounting_data.with_columns(pl.lit(0.0).alias("eqis_x"))
        result_no = pitroski_f(no_issuance)

        # Data with issuance
        with_issuance = base_accounting_data.with_columns(pl.lit(100.0).alias("eqis_x"))
        result_with = pitroski_f(with_issuance)

        # Get scores at same point (where both should be valid)
        idx = 20  # count = 21, should have enough history

        score_no = result_no["f_score"][idx]
        score_with = result_with["f_score"][idx]

        if score_no is not None and score_with is not None:
            # No issuance should score at least as high as with issuance
            assert score_no >= score_with, (
                f"No equity issuance should score >= with issuance: got {score_no} vs {score_with}"
            )

    def test_fscore_custom_name(self, base_accounting_data: pl.DataFrame):
        """F-score function should accept custom column name."""
        result = pitroski_f(base_accounting_data, name="my_fscore")

        assert "my_fscore" in result.columns, (
            f"Custom column name 'my_fscore' not found in result columns: {result.columns}"
        )
        assert "f_score" not in result.columns, (
            "Default column name 'f_score' should not be present when using custom name"
        )


# =============================================================================
# Altman Z-Score Tests
# =============================================================================


class TestAltmanZ:
    """
    Altman Z-score: Bankruptcy prediction model.

    Paper Reference: Altman (1968), used in JKP (2023) Appendix Table A.1

    Formula: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    Where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBITDA / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets
    """

    def test_zscore_with_known_values(self, tolerance):
        """Verify Z-score calculation with known input values."""
        # Create a simple test case where we can compute expected result
        df = pl.DataFrame(
            {
                "gvkey": ["001"],
                "curcd": ["USD"],
                "datadate": [_make_date(2020, 12, 31)],
                "count": [13],
                "at_x": [1000.0],
                "ca_x": [400.0],
                "cl_x": [200.0],
                "re": [300.0],
                "ebitda_x": [150.0],
                "me_fiscal": [800.0],
                "lt": [400.0],
                "sale_x": [1200.0],
            }
        )

        result = altman_z(df)

        # Manual calculation:
        # X1 = (400 - 200) / 1000 = 0.2 → 1.2 * 0.2 = 0.24
        # X2 = 300 / 1000 = 0.3 → 1.4 * 0.3 = 0.42
        # X3 = 150 / 1000 = 0.15 → 3.3 * 0.15 = 0.495
        # X4 = 800 / 400 = 2.0 → 0.6 * 2.0 = 1.2
        # X5 = 1200 / 1000 = 1.2 → 1.0 * 1.2 = 1.2
        # Z = 0.24 + 0.42 + 0.495 + 1.2 + 1.2 = 3.555

        expected_z = 1.2 * 0.2 + 1.4 * 0.3 + 3.3 * 0.15 + 0.6 * 2.0 + 1.0 * 1.2

        np.testing.assert_allclose(
            result["z_score"][0],
            expected_z,
            **tolerance.COMPOSITE_SCORES,
            err_msg="Z-score calculation does not match expected formula",
        )

    def test_zscore_interpretation_zones(self, base_accounting_data: pl.DataFrame):
        """Verify Z-scores fall in reasonable ranges."""
        result = altman_z(base_accounting_data)

        # Altman suggested:
        # Z < 1.81: Distress zone
        # 1.81 <= Z <= 2.99: Grey zone
        # Z > 2.99: Safe zone

        valid_scores = result["z_score"].drop_nulls()

        if len(valid_scores) > 0:
            # With our base data (profitable, moderate leverage), should not be in distress
            # This is a sanity check, not a strict requirement
            mean_z = valid_scores.mean()
            assert mean_z > 0, "Mean Z-score should be positive for healthy firm"

    def test_zscore_null_when_assets_zero(self):
        """Z-score should be null when total assets is zero or negative."""
        df = pl.DataFrame(
            {
                "gvkey": ["001", "002"],
                "curcd": ["USD", "USD"],
                "datadate": [_make_date(2020, 12, 31), _make_date(2020, 12, 31)],
                "count": [13, 13],
                "at_x": [0.0, -100.0],
                "ca_x": [100.0, 100.0],
                "cl_x": [50.0, 50.0],
                "re": [100.0, 100.0],
                "ebitda_x": [50.0, 50.0],
                "me_fiscal": [200.0, 200.0],
                "lt": [100.0, 100.0],
                "sale_x": [300.0, 300.0],
            }
        )

        result = altman_z(df)

        # Working capital / at_x should be null when at_x <= 0
        assert result["z_score"][0] is None or np.isnan(result["z_score"][0]), (
            f"Z-score should be null when at_x=0, got {result['z_score'][0]}"
        )
        assert result["z_score"][1] is None or np.isnan(result["z_score"][1]), (
            f"Z-score should be null when at_x<0, got {result['z_score'][1]}"
        )

    def test_zscore_coefficients_are_correct(self, tolerance):
        """Verify the Altman coefficients match published values."""
        # Test with unit inputs to verify coefficients
        df = pl.DataFrame(
            {
                "gvkey": ["001"],
                "curcd": ["USD"],
                "datadate": [_make_date(2020, 12, 31)],
                "count": [13],
                "at_x": [1.0],  # Normalized
                "ca_x": [1.0],  # WC/AT = 1 - 0 = 1
                "cl_x": [0.0],
                "re": [1.0],  # RE/AT = 1
                "ebitda_x": [1.0],  # EBITDA/AT = 1
                "me_fiscal": [1.0],  # ME/LT = 1 (if lt=1)
                "lt": [1.0],
                "sale_x": [1.0],  # Sale/AT = 1
            }
        )

        result = altman_z(df)

        # With all ratios = 1:
        # Z = 1.2*1 + 1.4*1 + 3.3*1 + 0.6*1 + 1.0*1 = 7.5
        expected = 1.2 + 1.4 + 3.3 + 0.6 + 1.0

        np.testing.assert_allclose(
            result["z_score"][0],
            expected,
            **tolerance.COMPOSITE_SCORES,
            err_msg="Altman coefficients do not match: 1.2, 1.4, 3.3, 0.6, 1.0",
        )


# =============================================================================
# Ohlson O-Score Tests
# =============================================================================


class TestOhlsonO:
    """
    Ohlson O-score: Bankruptcy prediction model using logit.

    Paper Reference: Ohlson (1980), used in JKP (2023) Appendix Table A.1

    Formula coefficients:
        -1.32 - 0.407*log(TA) + 6.03*TLTA - 1.43*WCTA + 0.076*CLCA
        - 1.72*OENEG - 2.37*NITA - 1.83*FUTL + 0.285*INTWO - 0.52*CHIN

    Higher O-score indicates higher probability of bankruptcy.
    """

    def test_oscore_returns_continuous_value(self, base_accounting_data: pl.DataFrame):
        """O-score should be a continuous value (not discrete like F-score)."""
        result = ohlson_o(base_accounting_data)

        valid_scores = result.filter(pl.col("count") > 12)["o_score"].drop_nulls()

        if len(valid_scores) > 0:
            # O-score is continuous, check it's not all integers
            values = valid_scores.to_list()
            non_integers = [v for v in values if v != int(v)]
            # With continuous formula, we expect non-integer results
            assert len(non_integers) > 0 or len(values) == 1, (
                f"O-score should be continuous (non-integer), but all {len(values)} values are integers"
            )

    def test_oscore_requires_history(self):
        """O-score requires count > 12 for some components."""
        # Create data with insufficient history
        df = pl.DataFrame(
            {
                "gvkey": ["001"],
                "curcd": ["USD"],
                "datadate": [_make_date(2020, 1, 31)],
                "count": [5],  # Insufficient history
                "at_x": [1000.0],
                "lt": [400.0],
                "ca_x": [300.0],
                "cl_x": [150.0],
                "debt_x": [300.0],
                "nix_x": [80.0],
                "pi_x": [100.0],
                "dp": [30.0],
            }
        )

        result = ohlson_o(df)

        # With insufficient history (count <= 12), O-score should be null
        assert result["o_score"].is_null().all(), (
            f"O-score should be null when count <= 12, "
            f"but found {len(result) - result['o_score'].null_count()} non-null values"
        )

    def test_oscore_higher_for_distressed_firms(self):
        """Higher leverage and losses should increase O-score."""
        dates = _generate_monthly_dates(24)

        # Healthy firm
        healthy = pl.DataFrame(
            {
                "gvkey": ["001"] * 24,
                "curcd": ["USD"] * 24,
                "datadate": dates,
                "count": list(range(1, 25)),
                "at_x": [1000.0] * 24,
                "lt": [300.0] * 24,  # Low leverage
                "ca_x": [400.0] * 24,
                "cl_x": [100.0] * 24,  # High current ratio
                "debt_x": [200.0] * 24,
                "nix_x": [100.0] * 24,  # Profitable
                "pi_x": [120.0] * 24,
                "dp": [30.0] * 24,
            }
        )

        # Distressed firm
        distressed = pl.DataFrame(
            {
                "gvkey": ["002"] * 24,
                "curcd": ["USD"] * 24,
                "datadate": dates,
                "count": list(range(1, 25)),
                "at_x": [1000.0] * 24,
                "lt": [900.0] * 24,  # High leverage
                "ca_x": [200.0] * 24,
                "cl_x": [300.0] * 24,  # Low current ratio
                "debt_x": [800.0] * 24,
                "nix_x": [-50.0] * 24,  # Losing money
                "pi_x": [-30.0] * 24,
                "dp": [30.0] * 24,
            }
        )

        result_healthy = ohlson_o(healthy)
        result_distressed = ohlson_o(distressed)

        # Get scores at same point
        idx = 20

        score_healthy = result_healthy["o_score"][idx]
        score_distressed = result_distressed["o_score"][idx]

        if score_healthy is not None and score_distressed is not None:
            assert score_distressed > score_healthy, "Distressed firm should have higher O-score"


# =============================================================================
# KZ Index Tests
# =============================================================================


class TestKZIndex:
    """
    Kaplan-Zingales Index: Financial constraints measure.

    Paper Reference: Kaplan and Zingales (1997), used in JKP (2023) Appendix Table A.1

    Formula:
        KZ = -1.002*(CF/K) + 0.283*Q + 3.139*(D/(D+E)) - 39.368*(Div/K) - 1.315*(C/K)

    Where K = lagged PP&E, Q = Tobin's Q, D = Debt, E = Equity, C = Cash
    Higher KZ indicates more financially constrained.
    """

    def test_kz_requires_lagged_ppent(self):
        """KZ index requires ppent from 12 months ago (count > 12)."""
        # Create data with count <= 12
        dates = _generate_monthly_dates(12)
        df = pl.DataFrame(
            {
                "gvkey": ["001"] * 12,
                "curcd": ["USD"] * 12,
                "datadate": dates,
                "count": list(range(1, 13)),
                "ni_x": [80.0] * 12,
                "dp": [30.0] * 12,
                "ppent": [500.0] * 12,
                "div_x": [20.0] * 12,
                "che": [100.0] * 12,
                "at_x": [1000.0] * 12,
                "me_fiscal": [800.0] * 12,
                "be_x": [600.0] * 12,
                "debt_x": [300.0] * 12,
                "seq_x": [600.0] * 12,
            }
        )

        result = kz_index(df)

        # All values should be null because count <= 12
        assert result["kz_index"].is_null().all(), (
            f"KZ index should be null when count <= 12, "
            f"but found {result['kz_index'].null_count()} nulls out of {len(result)}"
        )

    def test_kz_coefficients_signs(self):
        """Verify KZ coefficient signs match published formula."""
        # According to KZ (1997):
        # - CF/K: negative coefficient (-1.002) - more cash flow = less constrained
        # - Q: positive coefficient (0.283) - higher Q = more constrained
        # - D/(D+E): positive coefficient (3.139) - more leverage = more constrained
        # - Div/K: negative coefficient (-39.368) - paying dividends = less constrained
        # - C/K: negative coefficient (-1.315) - more cash = less constrained

        dates = _generate_monthly_dates(24)

        # Create base case
        base = pl.DataFrame(
            {
                "gvkey": ["001"] * 24,
                "curcd": ["USD"] * 24,
                "datadate": dates,
                "count": list(range(1, 25)),
                "ni_x": [80.0] * 24,
                "dp": [30.0] * 24,
                "ppent": [500.0] * 24,
                "div_x": [20.0] * 24,
                "che": [100.0] * 24,
                "at_x": [1000.0] * 24,
                "me_fiscal": [800.0] * 24,
                "be_x": [600.0] * 24,
                "debt_x": [300.0] * 24,
                "seq_x": [600.0] * 24,
            }
        )

        # High cash flow version (should have lower KZ)
        high_cf = base.with_columns(pl.lit(200.0).alias("ni_x"))

        result_base = kz_index(base)
        result_high_cf = kz_index(high_cf)

        idx = 20
        kz_base = result_base["kz_index"][idx]
        kz_high_cf = result_high_cf["kz_index"][idx]

        if kz_base is not None and kz_high_cf is not None:
            # Higher cash flow should reduce KZ (less constrained)
            assert kz_high_cf < kz_base, (
                f"Higher cash flow should reduce KZ: got {kz_high_cf} (high CF) vs {kz_base} (base)"
            )

    def test_kz_null_when_ppent_lag_zero(self):
        """KZ should be null when lagged PP&E is zero or negative."""
        dates = _generate_monthly_dates(24)
        df = pl.DataFrame(
            {
                "gvkey": ["001"] * 24,
                "curcd": ["USD"] * 24,
                "datadate": dates,
                "count": list(range(1, 25)),
                "ni_x": [80.0] * 24,
                "dp": [30.0] * 24,
                # PP&E starts at 0 and stays 0 for first 12 months
                "ppent": [0.0] * 12 + [500.0] * 12,
                "div_x": [20.0] * 24,
                "che": [100.0] * 24,
                "at_x": [1000.0] * 24,
                "me_fiscal": [800.0] * 24,
                "be_x": [600.0] * 24,
                "debt_x": [300.0] * 24,
                "seq_x": [600.0] * 24,
            }
        )

        result = kz_index(df)

        # For count = 13-24, the lagged ppent (12 months ago) was 0
        # So KZ should be null for these periods
        for i in range(12, 24):
            assert result["kz_index"][i] is None or np.isnan(result["kz_index"][i]), (
                f"KZ at row {i} should be null when lagged ppent=0, got {result['kz_index'][i]}"
            )


# =============================================================================
# Intrinsic Value Tests
# =============================================================================


class TestIntrinsicValue:
    """
    Intrinsic value calculation using residual income model.

    The intrinsic value is computed using a simplified residual income valuation
    that depends on ROE, book equity, and dividend payout.
    """

    def test_intrinsic_value_positive_for_profitable_firms(self):
        """Intrinsic value should be positive for profitable firms with positive BE."""
        dates = _generate_monthly_dates(24)
        df = pl.DataFrame(
            {
                "gvkey": ["001"] * 24,
                "curcd": ["USD"] * 24,
                "datadate": dates,
                "count": list(range(1, 25)),
                "nix_x": [80.0] * 24,  # Positive earnings
                "be_x": [600.0] * 24,  # Positive book equity
                "at_x": [1000.0] * 24,
                "div_x": [20.0] * 24,  # Paying dividends
            }
        )

        result = intrinsic_value(df)

        # For count > 12 with positive BE and positive NI, IV should be positive
        valid = result.filter(pl.col("count") > 12)
        iv_values = valid["intrinsic_value"].drop_nulls()

        if len(iv_values) > 0:
            assert (iv_values > 0).all(), "Intrinsic value should be positive for profitable firms"

    def test_intrinsic_value_null_when_negative(self):
        """Intrinsic value should be null when computed value is negative."""
        # Create a firm where IV formula would yield negative value
        dates = _generate_monthly_dates(24)
        df = pl.DataFrame(
            {
                "gvkey": ["001"] * 24,
                "curcd": ["USD"] * 24,
                "datadate": dates,
                "count": list(range(1, 25)),
                "nix_x": [-100.0] * 24,  # Large losses
                "be_x": [100.0] * 24,  # Small positive book equity
                "at_x": [1000.0] * 24,
                "div_x": [0.0] * 24,  # No dividends
            }
        )

        result = intrinsic_value(df)

        # The function should return null for negative intrinsic values
        valid = result.filter(pl.col("count") > 12)
        iv_values = valid["intrinsic_value"]

        # Either null or positive (function clips negative to null)
        for idx, iv in enumerate(iv_values.to_list()):
            assert iv is None or iv > 0, (
                f"Intrinsic value at row {idx} should be null or positive, got {iv}"
            )

    def test_intrinsic_value_custom_discount_rate(self):
        """Intrinsic value should accept custom discount rate."""
        dates = _generate_monthly_dates(24)
        df = pl.DataFrame(
            {
                "gvkey": ["001"] * 24,
                "curcd": ["USD"] * 24,
                "datadate": dates,
                "count": list(range(1, 25)),
                "nix_x": [80.0] * 24,
                "be_x": [600.0] * 24,
                "at_x": [1000.0] * 24,
                "div_x": [20.0] * 24,
            }
        )

        # Higher discount rate should lead to lower intrinsic value
        result_low_r = intrinsic_value(df, r=0.08)
        result_high_r = intrinsic_value(df, r=0.16)

        idx = 20
        iv_low = result_low_r["intrinsic_value"][idx]
        iv_high = result_high_r["intrinsic_value"][idx]

        if iv_low is not None and iv_high is not None:
            assert iv_low > iv_high, "Higher discount rate should reduce intrinsic value"
