"""
Unit tests for rolling daily metric helpers in aux_functions.py.

This module starts with rvol() and can be extended with tests for
other functions in the same section (rmax, zero_trades, mktcorr, ...).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Keep import resolution consistent with existing tests and editor static analysis.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from aux_functions import (
    ami,
    capm,
    capm_ext,
    dimsonbeta,
    dolvol,
    downbeta,
    ff3,
    hxz4,
    mktcorr,
    mktrf_vol,
    prc_to_high,
    rmax,
    rvol,
    skew,
    turnover,
    zero_trades,
)


def _is_none_or_nan(value: float | None) -> bool:
    """Return True when value is None or NaN-like."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def _empty_df(schema: dict[str, Any]) -> pl.DataFrame:
    """Build an empty DataFrame with explicit dtypes."""
    return pl.DataFrame({k: pl.Series(name=k, values=[], dtype=v) for k, v in schema.items()})


class TestRvol:
    """Tests for rvol() rolling volatility helper."""

    def test_rvol_grouped_std_computation(self, tolerance):
        """rvol should compute per-(id_int, group_number) sample std of ret_exc."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20, 20],
                "ret_exc": [1.0, 2.0, 3.0, 2.0, 4.0, 4.0, 3.0, 3.0, 3.0, -1.0, 0.0, 1.0],
            }
        )

        result = rvol(df, "_21d", __min=15).sort(["id_int", "group_number"])

        assert result.columns == ["id_int", "group_number", "rvol_21d"], (
            f"Unexpected columns: {result.columns}"
        )
        assert len(result) == 4, f"Expected 4 grouped rows, got {len(result)}"

        # Polars std() is sample std (ddof=1), so:
        # (1,10): std([1,2,3]) = 1.0
        # (1,20): std([2,4,4]) = sqrt(4/3)
        # (2,10): std([3,3,3]) = 0.0
        # (2,20): std([-1,0,1]) = 1.0
        np.testing.assert_allclose(
            result["rvol_21d"].to_list(),
            [1.0, np.sqrt(4.0 / 3.0), 0.0, 1.0],
            **tolerance.STANDARD,
            err_msg=f"Unexpected rvol values: {result['rvol_21d'].to_list()}",
        )

    def test_rvol_uses_suffix_for_output_name(self):
        """rvol output column name should include the provided suffix."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1],
                "group_number": [1, 1],
                "ret_exc": [0.1, 0.2],
            }
        )

        result = rvol(df, "_252d", __min=120)

        assert "rvol_252d" in result.columns, (
            f"Expected output column 'rvol_252d', got {result.columns}"
        )

    def test_rvol_single_observation_group_returns_null(self):
        """Sample std is undefined for 1 observation, so rvol should be null."""
        df = pl.DataFrame(
            {
                "id_int": [1],
                "group_number": [99],
                "ret_exc": [0.05],
            }
        )

        result = rvol(df, "_21d", __min=15)

        assert result["rvol_21d"][0] is None, (
            f"Expected null std for single observation, got {result['rvol_21d'][0]}"
        )

    def test_rvol_does_not_apply_min_filter_parameter(self):
        """rvol currently ignores __min and returns grouped output regardless of threshold."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 2, 2],
                "group_number": [10, 10, 20, 20],
                "ret_exc": [0.01, 0.02, 0.03, 0.04],
            }
        )

        result = rvol(df, "_21d", __min=9999)
        assert len(result) == 2, f"Expected both groups despite huge __min, got {len(result)}"

    def test_rvol_empty_input_returns_empty(self):
        df = _empty_df({"id_int": pl.Int64, "group_number": pl.Int64, "ret_exc": pl.Float64})
        result = rvol(df, "_21d", __min=15)
        assert len(result) == 0


class TestRmax:
    """Tests for rmax() rolling extreme-return helpers."""

    def test_rmax_grouped_top5_mean_and_max(self, tolerance):
        """rmax should compute per-group top-5 mean and max from ret."""
        df = pl.DataFrame(
            {
                "id_int": [1] * 8 + [1] * 6 + [2] * 6,
                "group_number": [10] * 8 + [20] * 6 + [10] * 6,
                "ret": [
                    -0.05,
                    0.01,
                    0.02,
                    0.10,
                    0.03,
                    0.20,
                    0.04,
                    0.05,  # (1,10)
                    0.00,
                    -0.01,
                    0.03,
                    0.02,
                    0.08,
                    0.07,  # (1,20)
                    -0.20,
                    -0.10,
                    0.00,
                    0.10,
                    0.15,
                    0.20,  # (2,10)
                ],
            }
        )

        result = rmax(df, "_21d", __min=15).sort(["id_int", "group_number"])

        assert result.columns == ["id_int", "group_number", "rmax5_21d", "rmax1_21d"], (
            f"Unexpected columns: {result.columns}"
        )
        assert len(result) == 3, f"Expected 3 grouped rows, got {len(result)}"

        # (1,10): sorted desc -> [0.20, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, -0.05]
        # rmax5 = mean(top5) = (0.20+0.10+0.05+0.04+0.03)/5 = 0.084 ; rmax1=0.20
        # (1,20): sorted desc -> [0.08,0.07,0.03,0.02,0.00,-0.01]
        # rmax5 = (0.08+0.07+0.03+0.02+0.00)/5 = 0.04 ; rmax1=0.08
        # (2,10): sorted desc -> [0.20,0.15,0.10,0.00,-0.10,-0.20]
        # rmax5 = (0.20+0.15+0.10+0.00-0.10)/5 = 0.07 ; rmax1=0.20
        np.testing.assert_allclose(
            result["rmax5_21d"].to_list(),
            [0.084, 0.04, 0.07],
            **tolerance.STANDARD,
            err_msg=f"Unexpected rmax5 values: {result['rmax5_21d'].to_list()}",
        )
        np.testing.assert_allclose(
            result["rmax1_21d"].to_list(),
            [0.20, 0.08, 0.20],
            **tolerance.STANDARD,
            err_msg=f"Unexpected rmax1 values: {result['rmax1_21d'].to_list()}",
        )

    def test_rmax_with_fewer_than_five_values_uses_available_values(self, tolerance):
        """top_k(5) should average available rows when group size < 5."""
        df = pl.DataFrame(
            {
                "id_int": [3, 3, 3, 3],
                "group_number": [30, 30, 30, 30],
                "ret": [0.01, 0.03, -0.02, 0.00],
            }
        )

        result = rmax(df, "_252d", __min=120)

        # top 5 of 4 values => all 4 values, mean = (0.01+0.03-0.02+0.00)/4 = 0.005
        np.testing.assert_allclose(
            result["rmax5_252d"][0],
            0.005,
            **tolerance.STANDARD,
            err_msg=f"Expected rmax5_252d=0.005, got {result['rmax5_252d'][0]}",
        )
        np.testing.assert_allclose(
            result["rmax1_252d"][0],
            0.03,
            **tolerance.STANDARD,
            err_msg=f"Expected rmax1_252d=0.03, got {result['rmax1_252d'][0]}",
        )

    def test_rmax_handles_null_returns(self):
        """Null returns should not crash grouped top-k/max aggregation."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1],
                "group_number": [10, 10, 10, 10],
                "ret": [0.01, None, 0.03, -0.02],
            }
        )
        result = rmax(df, "_21d", __min=15)
        assert "rmax5_21d" in result.columns and "rmax1_21d" in result.columns

    def test_rmax_ignores_min_parameter(self):
        """rmax currently ignores __min and should return same output for any threshold."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20],
                "ret": [0.01, 0.02, -0.01, 0.03, 0.04, 0.05],
            }
        )
        low_min = rmax(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = rmax(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_rmax_empty_input_returns_empty(self):
        df = _empty_df({"id_int": pl.Int64, "group_number": pl.Int64, "ret": pl.Float64})
        result = rmax(df, "_21d", __min=15)
        assert len(result) == 0


class TestSkew:
    """Tests for skew() rolling skewness helper."""

    def test_skew_symmetric_returns_zero(self, tolerance):
        """A symmetric return distribution should have skew close to 0."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10],
                "ret_exc": [-2.0, -1.0, 0.0, 1.0, 2.0],
            }
        )

        result = skew(df, "_21d", __min=15)
        np.testing.assert_allclose(
            result["rskew_21d"][0],
            0.0,
            **tolerance.STANDARD,
            err_msg=f"Expected skew ~0 for symmetric data, got {result['rskew_21d'][0]}",
        )

    def test_skew_constant_series_is_undefined(self):
        """Skew on a constant series should be undefined (null/NaN)."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1],
                "group_number": [10, 10, 10, 10],
                "ret_exc": [0.5, 0.5, 0.5, 0.5],
            }
        )
        result = skew(df, "_21d", __min=15)
        assert _is_none_or_nan(result["rskew_21d"][0]), (
            f"Expected undefined skew for constant series, got {result['rskew_21d'][0]}"
        )

    def test_skew_ignores_min_parameter(self):
        """skew currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20],
                "ret_exc": [0.1, 0.2, -0.1, 0.3, 0.4, 0.1],
            }
        )
        low_min = skew(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = skew(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_skew_empty_input_returns_empty(self):
        df = _empty_df({"id_int": pl.Int64, "group_number": pl.Int64, "ret_exc": pl.Float64})
        result = skew(df, "_21d", __min=15)
        assert len(result) == 0


class TestPrcToHigh:
    """Tests for prc_to_high() price-to-high helper."""

    def test_prc_to_high_sorts_by_date_and_applies_min_filter(self, tolerance):
        """Should use last price by date, divide by max price, and keep n >= __min groups."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2],
                "group_number": [10, 10, 10, 20, 20],
                "date": [
                    date(2024, 1, 2),
                    date(2024, 1, 1),
                    date(2024, 1, 3),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                ],
                "prc_adj": [12.0, 10.0, 11.0, 20.0, 25.0],
            }
        )

        result = prc_to_high(df, "_21d", __min=3).sort(["id_int", "group_number"])

        # group (1,10): last by date = 11.0, max = 12.0 -> 11/12
        # group (2,20): only 2 obs, should be filtered out by __min=3
        assert len(result) == 1, f"Expected 1 group after min filter, got {len(result)}"
        np.testing.assert_allclose(
            result["prc_highprc_21d"][0],
            11.0 / 12.0,
            **tolerance.STANDARD,
            err_msg=f"Expected prc_highprc_21d=11/12, got {result['prc_highprc_21d'][0]}",
        )

    def test_prc_to_high_zero_prices_yield_undefined_ratio(self):
        """When both last and max price are zero, ratio is undefined (0/0)."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1],
                "group_number": [10, 10, 10],
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "prc_adj": [0.0, 0.0, 0.0],
            }
        )
        result = prc_to_high(df, "_21d", __min=1)
        assert _is_none_or_nan(result["prc_highprc_21d"][0]), (
            f"Expected undefined ratio for zero prices, got {result['prc_highprc_21d'][0]}"
        )

    def test_prc_to_high_empty_input_returns_empty(self):
        df = _empty_df(
            {"id_int": pl.Int64, "group_number": pl.Int64, "date": pl.Date, "prc_adj": pl.Float64}
        )
        result = prc_to_high(df, "_21d", __min=3)
        assert len(result) == 0


class TestCapm:
    """Tests for capm() beta and idiosyncratic volatility helper."""

    def test_capm_perfect_linear_relation(self, tolerance):
        """If ret_exc = 2*mktrf, beta should be 2 and ivol should be 0."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1],
                "group_number": [10, 10, 10, 10],
                "mktrf": [1.0, 2.0, 3.0, 4.0],
                "ret_exc": [2.0, 4.0, 6.0, 8.0],
            }
        )

        result = capm(df, "_21d", __min=15)

        np.testing.assert_allclose(
            result["beta_21d"][0],
            2.0,
            **tolerance.STANDARD,
            err_msg=f"Expected beta_21d=2.0, got {result['beta_21d'][0]}",
        )
        np.testing.assert_allclose(
            result["ivol_capm_21d"][0],
            0.0,
            **tolerance.STANDARD,
            err_msg=f"Expected ivol_capm_21d=0.0, got {result['ivol_capm_21d'][0]}",
        )

    def test_capm_constant_market_factor_is_undefined(self):
        """CAPM beta/ivol are undefined when var(mktrf)=0."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1],
                "group_number": [10, 10, 10, 10],
                "mktrf": [2.0, 2.0, 2.0, 2.0],
                "ret_exc": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = capm(df, "_21d", __min=15)
        assert _is_none_or_nan(result["beta_21d"][0])
        assert _is_none_or_nan(result["ivol_capm_21d"][0])

    def test_capm_ignores_min_parameter(self):
        """capm currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 2, 2, 2, 2],
                "group_number": [10, 10, 10, 10, 20, 20, 20, 20],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                "ret_exc": [2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 5.0],
            }
        )
        low_min = capm(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = capm(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_capm_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "mktrf": pl.Float64,
                "ret_exc": pl.Float64,
            }
        )
        result = capm(df, "_21d", __min=15)
        assert len(result) == 0


class TestAmi:
    """Tests for ami() Amihud illiquidity helper."""

    def test_ami_zero_dollar_volume_ignored_and_min_filter_applied(self, tolerance):
        """dolvol_d == 0 should become null in ratio; groups keep only if n >= __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2],
                "group_number": [10, 10, 10, 20, 20],
                "ret": [0.10, -0.20, 0.05, 0.03, 0.04],
                "dolvol_d": [1000.0, 0.0, 2000.0, 1000.0, 1200.0],
            }
        )

        result = ami(df, "_21d", __min=3).sort(["id_int", "group_number"])

        # group (1,10):
        # abs(ret)/dolvol*1e6 -> [100, null, 25] -> mean = 62.5
        # n = count(dolvol_d) = 3, passes
        # group (2,20): n=2, filtered out
        assert len(result) == 1, f"Expected 1 group after min filter, got {len(result)}"
        np.testing.assert_allclose(
            result["ami_21d"][0],
            62.5,
            **tolerance.STANDARD,
            err_msg=f"Expected ami_21d=62.5, got {result['ami_21d'][0]}",
        )

    def test_ami_all_zero_dollar_volume_gives_null(self):
        """If all dolvol_d are zero, the Amihud ratio should be undefined."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1],
                "group_number": [10, 10, 10],
                "ret": [0.01, -0.02, 0.03],
                "dolvol_d": [0.0, 0.0, 0.0],
            }
        )
        result = ami(df, "_21d", __min=1)
        assert _is_none_or_nan(result["ami_21d"][0]), (
            f"Expected undefined ami with zero dollar volume, got {result['ami_21d'][0]}"
        )

    def test_ami_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "ret": pl.Float64,
                "dolvol_d": pl.Float64,
            }
        )
        result = ami(df, "_21d", __min=3)
        assert len(result) == 0


class TestDownbeta:
    """Tests for downbeta() downside beta helper."""

    def test_downbeta_uses_only_negative_market_days_and_threshold(self, tolerance):
        """Should filter to mktrf < 0 and require n >= __min / 2 per group."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 2, 2, 2, 2],
                "group_number": [10, 10, 10, 10, 20, 20, 20, 20],
                "mktrf": [-1.0, -2.0, 1.0, -3.0, 1.0, -1.0, 2.0, 3.0],
                "ret_exc": [-2.0, -4.0, 2.0, -6.0, 1.0, -2.0, 2.0, 3.0],
            }
        )

        result = downbeta(df, "_21d", __min=4).sort(["id_int", "group_number"])

        # __min/2 = 2. group (1,10) has 3 negative mktrf rows -> keep
        # On those rows ret_exc = 2*mktrf -> downside beta = 2
        # group (2,20) has 1 negative row -> drop
        assert len(result) == 1, f"Expected 1 group after downside threshold, got {len(result)}"
        np.testing.assert_allclose(
            result["betadown_21d"][0],
            2.0,
            **tolerance.STANDARD,
            err_msg=f"Expected betadown_21d=2.0, got {result['betadown_21d'][0]}",
        )

    def test_downbeta_no_negative_market_days_returns_empty(self):
        """If no mktrf < 0 rows exist, downbeta should return no groups."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1],
                "group_number": [10, 10, 10],
                "mktrf": [0.0, 0.5, 1.0],
                "ret_exc": [0.0, 1.0, 2.0],
            }
        )
        result = downbeta(df, "_21d", __min=2)
        assert len(result) == 0, f"Expected empty result when no downside days, got {len(result)}"

    def test_downbeta_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "mktrf": pl.Float64,
                "ret_exc": pl.Float64,
            }
        )
        result = downbeta(df, "_21d", __min=4)
        assert len(result) == 0


class TestMktrfVol:
    """Tests for mktrf_vol() market-factor volatility helper."""

    def test_mktrf_vol_grouped_std(self, tolerance):
        """mktrf_vol should compute sample std of mktrf per group."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20],
                "mktrf": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            }
        )

        result = mktrf_vol(df, "_21d", __min=15).sort(["id_int", "group_number"])

        assert result.columns == ["id_int", "group_number", "__mktvol_21d"], (
            f"Unexpected columns: {result.columns}"
        )
        np.testing.assert_allclose(
            result["__mktvol_21d"].to_list(),
            [1.0, 0.0],
            **tolerance.STANDARD,
            err_msg=f"Unexpected __mktvol_21d values: {result['__mktvol_21d'].to_list()}",
        )

    def test_mktrf_vol_single_observation_is_null(self):
        """Sample std for one observation should be null."""
        df = pl.DataFrame({"id_int": [1], "group_number": [10], "mktrf": [0.1]})
        result = mktrf_vol(df, "_21d", __min=15)
        assert result["__mktvol_21d"][0] is None

    def test_mktrf_vol_ignores_min_parameter(self):
        """mktrf_vol currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 4.0, 4.0],
            }
        )
        low_min = mktrf_vol(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = mktrf_vol(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_mktrf_vol_empty_input_returns_empty(self):
        df = _empty_df({"id_int": pl.Int64, "group_number": pl.Int64, "mktrf": pl.Float64})
        result = mktrf_vol(df, "_21d", __min=15)
        assert len(result) == 0


class TestCapmExt:
    """Tests for capm_ext() extended CAPM diagnostics helper."""

    def test_capm_ext_ivol_zero_when_ret_equals_market(self, tolerance):
        """If ret_exc == mktrf exactly, residuals are zero so ivol should be zero."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 5.0],
                "ret_exc": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = capm_ext(df, "_21d", __min=15)

        np.testing.assert_allclose(
            result["beta_21d"][0],
            1.0,
            **tolerance.STANDARD,
            err_msg=f"Expected beta_21d=1.0, got {result['beta_21d'][0]}",
        )
        np.testing.assert_allclose(
            result["ivol_capm_21d"][0],
            0.0,
            **tolerance.STANDARD,
            err_msg=f"Expected ivol_capm_21d=0.0, got {result['ivol_capm_21d'][0]}",
        )

    def test_capm_ext_constant_market_has_undefined_outputs(self):
        """With constant mktrf (var=0), regression diagnostics are undefined."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10],
                "mktrf": [2.0, 2.0, 2.0, 2.0, 2.0],
                "ret_exc": [1.0, 1.5, 2.0, 2.5, 3.0],
            }
        )

        result = capm_ext(df, "_21d", __min=15)

        # beta and downstream moments can be null or NaN when var(mktrf)=0.
        beta = result["beta_21d"][0]
        ivol = result["ivol_capm_21d"][0]
        iskew = result["iskew_capm_21d"][0]
        coskew = result["coskew_21d"][0]

        assert beta is None or np.isnan(beta), f"Expected beta undefined, got {beta}"
        assert ivol is None or np.isnan(ivol), f"Expected ivol undefined, got {ivol}"
        assert iskew is None or np.isnan(iskew), f"Expected iskew undefined, got {iskew}"
        assert coskew is None or np.isnan(coskew), f"Expected coskew undefined, got {coskew}"

    def test_capm_ext_outputs_expected_columns_and_reasonable_values(self, tolerance):
        """capm_ext should emit beta/ivol/iskew/coskew with finite non-null values."""
        mktrf = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        ret_exc = np.array([2.2, 3.9, 6.4, 7.8, 10.5], dtype=float)

        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10],
                "mktrf": mktrf,
                "ret_exc": ret_exc,
            }
        )

        result = capm_ext(df, "_21d", __min=15)

        expected_cols = [
            "id_int",
            "group_number",
            "beta_21d",
            "ivol_capm_21d",
            "iskew_capm_21d",
            "coskew_21d",
        ]
        assert result.columns == expected_cols, f"Unexpected columns: {result.columns}"

        expected_beta = np.cov(ret_exc, mktrf, ddof=1)[0, 1] / np.var(mktrf, ddof=1)
        np.testing.assert_allclose(
            result["beta_21d"][0],
            expected_beta,
            **tolerance.STANDARD,
            err_msg=f"Unexpected beta_21d: {result['beta_21d'][0]}",
        )

        assert result["ivol_capm_21d"][0] is not None, "ivol_capm_21d should not be null"
        assert result["ivol_capm_21d"][0] > 0, (
            f"ivol_capm_21d should be positive, got {result['ivol_capm_21d'][0]}"
        )
        assert np.isfinite(result["iskew_capm_21d"][0]), (
            f"iskew_capm_21d should be finite, got {result['iskew_capm_21d'][0]}"
        )
        assert np.isfinite(result["coskew_21d"][0]), (
            f"coskew_21d should be finite, got {result['coskew_21d'][0]}"
        )

    def test_capm_ext_single_observation_is_undefined(self):
        """Extended CAPM moments are undefined for one-row groups."""
        df = pl.DataFrame({"id_int": [1], "group_number": [10], "mktrf": [0.1], "ret_exc": [0.2]})
        result = capm_ext(df, "_21d", __min=15)
        assert _is_none_or_nan(result["beta_21d"][0])
        assert _is_none_or_nan(result["ivol_capm_21d"][0])

    def test_capm_ext_ignores_min_parameter(self):
        """capm_ext currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 5.0],
                "ret_exc": [2.2, 3.9, 6.4, 7.8, 10.5],
            }
        )
        low_min = capm_ext(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = capm_ext(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_capm_ext_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "mktrf": pl.Float64,
                "ret_exc": pl.Float64,
            }
        )
        result = capm_ext(df, "_21d", __min=15)
        assert len(result) == 0


class TestFf3:
    """Tests for ff3() FF3 residual diagnostics helper."""

    def test_ff3_zero_residual_for_exact_linear_model(self, tolerance):
        """Exact FF3 linear relation should produce near-zero residual volatility."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10, 10],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "smb_ff": [0.5, 1.0, 1.5, 0.0, -0.5, 2.0],
                "hml": [2.0, 1.0, 0.0, 3.0, 4.0, -1.0],
            }
        ).with_columns(
            (1.0 + 2.0 * pl.col("mktrf") + 3.0 * pl.col("smb_ff") + 4.0 * pl.col("hml")).alias(
                "ret_exc"
            )
        )

        result = ff3(df, "_21d", __min=15)

        np.testing.assert_allclose(
            result["ivol_ff3_21d"][0],
            0.0,
            **tolerance.STANDARD,
            err_msg=f"Expected ivol_ff3_21d near zero, got {result['ivol_ff3_21d'][0]}",
        )

    def test_ff3_drops_rows_with_missing_factors(self):
        """Rows with null SMB/HML should be filtered before grouping."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20],
                "ret_exc": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "mktrf": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "smb_ff": [1.0, 1.0, 1.0, None, None, None],
                "hml": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        result = ff3(df, "_21d", __min=15).sort(["id_int", "group_number"])
        assert len(result) == 1, f"Expected only one surviving group, got {len(result)}"
        assert result["id_int"][0] == 1 and result["group_number"][0] == 10, (
            f"Unexpected group(s) after factor filter: {result[['id_int', 'group_number']]}"
        )

    def test_ff3_all_rows_missing_factors_returns_empty(self):
        """If every row misses SMB/HML, ff3 should return an empty frame."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1],
                "group_number": [10, 10],
                "ret_exc": [1.0, 2.0],
                "mktrf": [1.0, 2.0],
                "smb_ff": [None, None],
                "hml": [1.0, 1.0],
            }
        )
        result = ff3(df, "_21d", __min=15)
        assert len(result) == 0

    def test_ff3_ignores_min_parameter(self):
        """ff3 currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10, 10],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "smb_ff": [0.5, 1.0, 1.5, 0.0, -0.5, 2.0],
                "hml": [2.0, 1.0, 0.0, 3.0, 4.0, -1.0],
            }
        ).with_columns(
            (1.0 + 2.0 * pl.col("mktrf") + 3.0 * pl.col("smb_ff") + 4.0 * pl.col("hml")).alias(
                "ret_exc"
            )
        )
        low_min = ff3(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = ff3(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_ff3_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "ret_exc": pl.Float64,
                "mktrf": pl.Float64,
                "smb_ff": pl.Float64,
                "hml": pl.Float64,
            }
        )
        result = ff3(df, "_21d", __min=15)
        assert len(result) == 0


class TestHxz4:
    """Tests for hxz4() HXZ4 residual diagnostics helper."""

    def test_hxz4_zero_residual_for_exact_linear_model(self, tolerance):
        """Exact HXZ4 linear relation should produce near-zero residual volatility."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10, 10, 10],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "smb_hxz": [0.5, 1.0, 1.5, 0.0, -0.5, 2.0, 1.0],
                "roe": [2.0, 1.0, 0.0, 3.0, 4.0, -1.0, 2.0],
                "inv": [0.0, 1.0, 0.5, 2.0, 1.5, -0.5, 1.0],
            }
        ).with_columns(
            (
                1.0
                + 2.0 * pl.col("mktrf")
                + 3.0 * pl.col("smb_hxz")
                + 4.0 * pl.col("roe")
                + 5.0 * pl.col("inv")
            ).alias("ret_exc")
        )

        result = hxz4(df, "_21d", __min=15)

        np.testing.assert_allclose(
            result["ivol_hxz4_21d"][0],
            0.0,
            **tolerance.STANDARD,
            err_msg=f"Expected ivol_hxz4_21d near zero, got {result['ivol_hxz4_21d'][0]}",
        )

    def test_hxz4_ignores_min_parameter(self):
        """hxz4 currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10, 10, 10],
                "mktrf": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "smb_hxz": [0.5, 1.0, 1.5, 0.0, -0.5, 2.0, 1.0],
                "roe": [2.0, 1.0, 0.0, 3.0, 4.0, -1.0, 2.0],
                "inv": [0.0, 1.0, 0.5, 2.0, 1.5, -0.5, 1.0],
            }
        ).with_columns(
            (
                1.0
                + 2.0 * pl.col("mktrf")
                + 3.0 * pl.col("smb_hxz")
                + 4.0 * pl.col("roe")
                + 5.0 * pl.col("inv")
            ).alias("ret_exc")
        )
        low_min = hxz4(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = hxz4(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_hxz4_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "ret_exc": pl.Float64,
                "mktrf": pl.Float64,
                "smb_hxz": pl.Float64,
                "roe": pl.Float64,
                "inv": pl.Float64,
            }
        )
        result = hxz4(df, "_21d", __min=15)
        assert len(result) == 0


class TestZeroTrades:
    """Tests for zero_trades() turnover-adjusted zero-trade metric."""

    def test_zero_trades_composite_rank_plus_zero_days(self, tolerance):
        """Composite should equal zero-trade-days + turnover-rank adjustment."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 10, 10, 10],
                "tvol": [0.0, 0.0, 10.0, 20.0, 20.0, 20.0],
                "shares": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        result = zero_trades(df, "_21d", __min=15).sort(["id_int", "group_number"])

        # id=1: zero_trades = (2/3)*21 = 14.0, lower turnover so rank=2 of 2 => 2/2=1.0 => +0.01
        # id=2: zero_trades = 0.0, higher turnover so rank=1 of 2 => 1/2=0.5 => +0.005
        np.testing.assert_allclose(
            result["zero_trades_21d"].to_list(),
            [14.01, 0.005],
            **tolerance.STANDARD,
            err_msg=f"Unexpected zero_trades_21d values: {result['zero_trades_21d'].to_list()}",
        )

    def test_zero_trades_ignores_min_parameter(self):
        """zero_trades currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 10, 10, 10],
                "tvol": [0.0, 0.0, 10.0, 20.0, 20.0, 20.0],
                "shares": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        low_min = zero_trades(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = zero_trades(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_zero_trades_empty_input_returns_empty(self):
        df = _empty_df(
            {"id_int": pl.Int64, "group_number": pl.Int64, "tvol": pl.Float64, "shares": pl.Float64}
        )
        result = zero_trades(df, "_21d", __min=15)
        assert len(result) == 0


class TestDolvol:
    """Tests for dolvol() dollar-volume level and variability helper."""

    def test_dolvol_mean_and_variability(self, tolerance):
        """dolvol should return group mean and std/mean variability."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20],
                "dolvol_d": [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
            }
        )

        result = dolvol(df, "_21d", __min=15).sort(["id_int", "group_number"])

        # group (1,10): mean=20, std=10 (sample), var ratio=0.5
        # group (2,20): mean=0 => dolvol_var should be null by guard
        np.testing.assert_allclose(
            result["dolvol_21d"].to_list(),
            [20.0, 0.0],
            **tolerance.STANDARD,
            err_msg=f"Unexpected dolvol_21d values: {result['dolvol_21d'].to_list()}",
        )
        np.testing.assert_allclose(
            result["dolvol_var_21d"][0],
            0.5,
            **tolerance.STANDARD,
            err_msg=f"Expected dolvol_var_21d=0.5, got {result['dolvol_var_21d'][0]}",
        )
        assert result["dolvol_var_21d"][1] is None, (
            f"Expected null dolvol_var_21d when mean=0, got {result['dolvol_var_21d'][1]}"
        )

    def test_dolvol_single_observation_variability_is_null(self):
        """Sample std-based variability is undefined for one observation."""
        df = pl.DataFrame({"id_int": [1], "group_number": [10], "dolvol_d": [100.0]})
        result = dolvol(df, "_21d", __min=15)
        assert result["dolvol_21d"][0] == 100.0
        assert result["dolvol_var_21d"][0] is None

    def test_dolvol_ignores_min_parameter(self):
        """dolvol currently ignores __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2, 2],
                "group_number": [10, 10, 10, 20, 20, 20],
                "dolvol_d": [10.0, 20.0, 30.0, 5.0, 5.0, 5.0],
            }
        )
        low_min = dolvol(df, "_21d", __min=1).sort(["id_int", "group_number"])
        high_min = dolvol(df, "_21d", __min=10_000).sort(["id_int", "group_number"])
        assert low_min.equals(high_min), "Expected identical output regardless of __min"

    def test_dolvol_empty_input_returns_empty(self):
        df = _empty_df({"id_int": pl.Int64, "group_number": pl.Int64, "dolvol_d": pl.Float64})
        result = dolvol(df, "_21d", __min=15)
        assert len(result) == 0


class TestTurnover:
    """Tests for turnover() turnover level and variability helper."""

    def test_turnover_computation_and_min_filter(self, tolerance):
        """turnover should compute mean/std ratio and apply n >= __min filter."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2],
                "group_number": [10, 10, 10, 20, 20],
                "tvol": [10.0, 20.0, 30.0, 5.0, 15.0],
                "shares": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        result = turnover(df, "_21d", __min=3).sort(["id_int", "group_number"])

        # group (1,10) turnover_d: [10,20,30]/1e6 => mean=20e-6, std=10e-6 -> var=0.5
        # group (2,20) has n=2 -> filtered out
        assert len(result) == 1, f"Expected 1 group after __min filter, got {len(result)}"
        np.testing.assert_allclose(
            result["turnover_21d"][0],
            20.0 / 1e6,
            **tolerance.STANDARD,
            err_msg=f"Unexpected turnover_21d: {result['turnover_21d'][0]}",
        )
        np.testing.assert_allclose(
            result["turnover_var_21d"][0],
            0.5,
            **tolerance.STANDARD,
            err_msg=f"Unexpected turnover_var_21d: {result['turnover_var_21d'][0]}",
        )

    def test_turnover_all_zero_shares_gives_null_metrics(self):
        """All-zero shares make turnover undefined while n can still satisfy __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1],
                "group_number": [10, 10, 10],
                "tvol": [5.0, 10.0, 15.0],
                "shares": [0.0, 0.0, 0.0],
            }
        )
        result = turnover(df, "_21d", __min=3)
        assert len(result) == 1
        assert _is_none_or_nan(result["turnover_21d"][0])
        assert _is_none_or_nan(result["turnover_var_21d"][0])

    def test_turnover_empty_input_returns_empty(self):
        df = _empty_df(
            {"id_int": pl.Int64, "group_number": pl.Int64, "tvol": pl.Float64, "shares": pl.Float64}
        )
        result = turnover(df, "_21d", __min=3)
        assert len(result) == 0


class TestMktcorr:
    """Tests for mktcorr() rolling correlation helper."""

    def test_mktcorr_computation_and_min_filter(self, tolerance):
        """Should compute correlation and keep only groups with n >= __min."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 2, 2],
                "group_number": [10, 10, 10, 20, 20],
                "ret_exc_3l": [1.0, 2.0, 3.0, 1.0, 2.0],
                "mkt_exc_3l": [2.0, 4.0, 6.0, 2.0, 4.0],
            }
        )

        result = mktcorr(df, "_21d", __min=3).sort(["id_int", "group_number"])

        assert len(result) == 1, f"Expected 1 group after __min filter, got {len(result)}"
        np.testing.assert_allclose(
            result["corr_21d"][0],
            1.0,
            **tolerance.STANDARD,
            err_msg=f"Expected corr_21d=1.0, got {result['corr_21d'][0]}",
        )

    def test_mktcorr_constant_series_is_undefined(self):
        """Correlation is undefined if one side is constant."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1],
                "group_number": [10, 10, 10],
                "ret_exc_3l": [1.0, 1.0, 1.0],
                "mkt_exc_3l": [2.0, 4.0, 6.0],
            }
        )
        result = mktcorr(df, "_21d", __min=3)
        assert _is_none_or_nan(result["corr_21d"][0])

    def test_mktcorr_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "ret_exc_3l": pl.Float64,
                "mkt_exc_3l": pl.Float64,
            }
        )
        result = mktcorr(df, "_21d", __min=3)
        assert len(result) == 0


class TestDimsonbeta:
    """Tests for dimsonbeta() lead-lag adjusted beta helper."""

    def test_dimsonbeta_sums_market_coefficients(self, tolerance):
        """Across multiple groups, beta_dimson should equal b1+b2+b3 per group."""
        mktrf = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        mktrf_ld1 = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
        mktrf_lg1 = np.array([2.0, 1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)

        df = pl.concat(
            [
                # Group A: sum = 2 + 3 + 4 = 9
                pl.DataFrame(
                    {
                        "id_int": [1] * len(mktrf),
                        "group_number": [10] * len(mktrf),
                        "mktrf": mktrf,
                        "mktrf_ld1": mktrf_ld1,
                        "mktrf_lg1": mktrf_lg1,
                        "ret_exc": 1.0 + 2.0 * mktrf + 3.0 * mktrf_ld1 + 4.0 * mktrf_lg1,
                    }
                ),
                # Group B: sum = 1 + 0 + 2 = 3
                pl.DataFrame(
                    {
                        "id_int": [2] * len(mktrf),
                        "group_number": [20] * len(mktrf),
                        "mktrf": mktrf,
                        "mktrf_ld1": mktrf_ld1,
                        "mktrf_lg1": mktrf_lg1,
                        "ret_exc": -2.0 + 1.0 * mktrf + 0.0 * mktrf_ld1 + 2.0 * mktrf_lg1,
                    }
                ),
                # Group C: sum = 0.5 - 1 + 1.5 = 1
                pl.DataFrame(
                    {
                        "id_int": [3] * len(mktrf),
                        "group_number": [30] * len(mktrf),
                        "mktrf": mktrf,
                        "mktrf_ld1": mktrf_ld1,
                        "mktrf_lg1": mktrf_lg1,
                        "ret_exc": 0.7 + 0.5 * mktrf - 1.0 * mktrf_ld1 + 1.5 * mktrf_lg1,
                    }
                ),
            ]
        )

        result = dimsonbeta(df, "_21d", __min=15).sort(["id_int", "group_number"])

        np.testing.assert_allclose(
            result["beta_dimson_21d"].to_list(),
            [9.0, 3.0, 1.0],
            **tolerance.STANDARD,
            err_msg=f"Unexpected beta_dimson_21d values: {result['beta_dimson_21d'].to_list()}",
        )

    def test_dimsonbeta_ignores_min_parameter(self):
        """dimsonbeta currently does not use __min and should still return grouped output."""
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10],
                "mktrf": [0.0, 1.0, 2.0, 3.0, 4.0],
                "mktrf_ld1": [1.0, 0.0, 1.0, 0.0, 1.0],
                "mktrf_lg1": [0.0, 1.0, 0.0, 1.0, 0.0],
                "ret_exc": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = dimsonbeta(df, "_21d", __min=10_000)
        assert len(result) == 1

    def test_dimsonbeta_empty_input_returns_empty(self):
        df = _empty_df(
            {
                "id_int": pl.Int64,
                "group_number": pl.Int64,
                "mktrf": pl.Float64,
                "mktrf_ld1": pl.Float64,
                "mktrf_lg1": pl.Float64,
                "ret_exc": pl.Float64,
            }
        )
        result = dimsonbeta(df, "_21d", __min=15)
        assert len(result) == 0
