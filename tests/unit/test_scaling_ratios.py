"""
Tests for scaling and ratio helper functions in aux_functions.py.

Covers FX-adjusted scaling (scale_me, scale_mev), year-mean helpers
(mean_year), parameterized ratio builders (temp_liq_rat, temp_rat_other,
temp_rat_other_spc), and consecutive-earnings / abnormal-capex DataFrame
transforms. Pins null-handling, count-gate, and zero-denominator behavior
so downstream accounting characteristics cannot silently regress.

Paper Reference: Jensen, Kelly, Pedersen (2023), "Is There a Replication Crisis in Finance?"
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from jkp.data.aux_functions import (
    calculate_consecutive_earnings_increases,
    compute_capex_abn,
    mean_year,
    scale_me,
    scale_mev,
    temp_liq_rat,
    temp_rat_other,
    temp_rat_other_spc,
    update_ni_inc_and_decrease,
)

# =============================================================================
# Local Helpers / Fixtures
# =============================================================================


def _monthly_dates(n: int, start_year: int = 2010) -> list[date]:
    """Generate n monthly dates on the 28th starting Jan of start_year."""
    out: list[date] = []
    for i in range(n):
        year = start_year + i // 12
        month = (i % 12) + 1
        out.append(date(year, month, 28))
    return out


def _single_firm_panel(n: int, **cols: list) -> pl.DataFrame:
    """Build a sorted monthly panel for one gvkey/curcd with arbitrary columns."""
    data: dict[str, list] = {
        "gvkey": [1] * n,
        "curcd": ["USD"] * n,
        "datadate": _monthly_dates(n),
        "count": list(range(1, n + 1)),
    }
    for k, v in cols.items():
        data[k] = v
    return pl.DataFrame(data)


# =============================================================================
# scale_me
# =============================================================================


class TestScaleMe:
    """Tests for scale_me() — FX-adjusted scaling by me_company."""

    def test_basic_positive(self, tolerance):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "me_company": [50.0]})
        result = df.select(scale_me("ni_x"))["ni_me"][0]
        np.testing.assert_allclose(result, 2.0, **tolerance.TIGHT)

    def test_alias_strips_x_suffix(self):
        df = pl.DataFrame({"at_x": [1.0], "fx": [1.0], "me_company": [1.0]})
        assert df.select(scale_me("at_x")).columns == ["at_me"]

    def test_alias_no_x_suffix(self):
        df = pl.DataFrame({"ebitda": [1.0], "fx": [1.0], "me_company": [1.0]})
        assert df.select(scale_me("ebitda")).columns == ["ebitda_me"]

    def test_alias_substring_replace_is_literal(self):
        """`.replace('_x', '')` strips every '_x' substring, not just trailing."""
        df = pl.DataFrame({"xrd_x": [1.0], "fx": [1.0], "me_company": [1.0]})
        assert df.select(scale_me("xrd_x")).columns == ["xrd_me"]

    def test_zero_denominator_returns_null(self):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "me_company": [0.0]})
        assert df.select(scale_me("ni_x"))["ni_me"][0] is None

    def test_negative_me_company_flows_through(self, tolerance):
        """No sign guard — negative me_company produces negative ratio."""
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "me_company": [-50.0]})
        np.testing.assert_allclose(df.select(scale_me("ni_x"))["ni_me"][0], -2.0, **tolerance.TIGHT)

    def test_null_var_propagates(self):
        df = pl.DataFrame({"ni_x": [None], "fx": [1.0], "me_company": [50.0]})
        assert df.select(scale_me("ni_x"))["ni_me"][0] is None

    def test_null_fx_propagates(self):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [None], "me_company": [50.0]})
        assert df.select(scale_me("ni_x"))["ni_me"][0] is None

    def test_null_me_company_returns_null(self):
        """`None != 0` evaluates to null → when-clause null → output null."""
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "me_company": [None]})
        assert df.select(scale_me("ni_x"))["ni_me"][0] is None

    def test_fx_less_than_one(self, tolerance):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [0.5], "me_company": [10.0]})
        np.testing.assert_allclose(df.select(scale_me("ni_x"))["ni_me"][0], 5.0, **tolerance.TIGHT)

    def test_fx_greater_than_one(self, tolerance):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.25], "me_company": [50.0]})
        np.testing.assert_allclose(df.select(scale_me("ni_x"))["ni_me"][0], 2.5, **tolerance.TIGHT)

    def test_fx_zero_returns_zero(self, tolerance):
        """fx=0 is not guarded — numerator becomes 0, result is 0 (not null)."""
        df = pl.DataFrame({"ni_x": [100.0], "fx": [0.0], "me_company": [50.0]})
        np.testing.assert_allclose(df.select(scale_me("ni_x"))["ni_me"][0], 0.0, **tolerance.TIGHT)

    def test_negative_var_flows_through(self, tolerance):
        df = pl.DataFrame({"ni_x": [-100.0], "fx": [1.0], "me_company": [50.0]})
        np.testing.assert_allclose(df.select(scale_me("ni_x"))["ni_me"][0], -2.0, **tolerance.TIGHT)

    def test_multi_row(self):
        df = pl.DataFrame(
            {
                "ni_x": [100.0, 200.0, -50.0, 0.0, 100.0],
                "fx": [1.0, 0.5, 2.0, 1.0, 1.0],
                "me_company": [50.0, 100.0, 25.0, 10.0, 0.0],
            }
        )
        result = df.select(scale_me("ni_x"))["ni_me"].to_list()
        assert result == [2.0, 1.0, -4.0, 0.0, None]

    def test_extreme_values(self, tolerance):
        df = pl.DataFrame({"ni_x": [1e-9], "fx": [1.0], "me_company": [1e12]})
        np.testing.assert_allclose(
            df.select(scale_me("ni_x"))["ni_me"][0], 1e-21, **tolerance.STANDARD
        )


# =============================================================================
# scale_mev
# =============================================================================


class TestScaleMev:
    """Tests for scale_mev() — FX-adjusted scaling by mev."""

    def test_basic_positive(self, tolerance):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "mev": [50.0]})
        np.testing.assert_allclose(
            df.select(scale_mev("ni_x"))["ni_mev"][0], 2.0, **tolerance.TIGHT
        )

    def test_alias_strips_x_suffix(self):
        df = pl.DataFrame({"at_x": [1.0], "fx": [1.0], "mev": [1.0]})
        assert df.select(scale_mev("at_x")).columns == ["at_mev"]

    def test_alias_no_x_suffix(self):
        df = pl.DataFrame({"ebitda": [1.0], "fx": [1.0], "mev": [1.0]})
        assert df.select(scale_mev("ebitda")).columns == ["ebitda_mev"]

    def test_zero_denominator_returns_null(self):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "mev": [0.0]})
        assert df.select(scale_mev("ni_x"))["ni_mev"][0] is None

    def test_negative_mev_flows_through(self, tolerance):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "mev": [-50.0]})
        np.testing.assert_allclose(
            df.select(scale_mev("ni_x"))["ni_mev"][0], -2.0, **tolerance.TIGHT
        )

    def test_null_var_propagates(self):
        df = pl.DataFrame({"ni_x": [None], "fx": [1.0], "mev": [50.0]})
        assert df.select(scale_mev("ni_x"))["ni_mev"][0] is None

    def test_null_fx_propagates(self):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [None], "mev": [50.0]})
        assert df.select(scale_mev("ni_x"))["ni_mev"][0] is None

    def test_null_mev_returns_null(self):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [1.0], "mev": [None]})
        assert df.select(scale_mev("ni_x"))["ni_mev"][0] is None

    def test_fx_scaling(self, tolerance):
        df = pl.DataFrame({"ni_x": [100.0], "fx": [0.5], "mev": [10.0]})
        np.testing.assert_allclose(
            df.select(scale_mev("ni_x"))["ni_mev"][0], 5.0, **tolerance.TIGHT
        )

    def test_negative_var_flows_through(self, tolerance):
        df = pl.DataFrame({"ni_x": [-100.0], "fx": [1.0], "mev": [50.0]})
        np.testing.assert_allclose(
            df.select(scale_mev("ni_x"))["ni_mev"][0], -2.0, **tolerance.TIGHT
        )

    def test_multi_row(self):
        df = pl.DataFrame(
            {
                "ni_x": [100.0, 200.0, -50.0, 0.0, 100.0],
                "fx": [1.0, 0.5, 2.0, 1.0, 1.0],
                "mev": [50.0, 100.0, 25.0, 10.0, 0.0],
            }
        )
        result = df.select(scale_mev("ni_x"))["ni_mev"].to_list()
        assert result == [2.0, 1.0, -4.0, 0.0, None]


# =============================================================================
# mean_year
# =============================================================================


class TestMeanYear:
    """Tests for mean_year() — year-mean with 12-month lag fallbacks."""

    def test_both_present_returns_average(self, tolerance):
        """Row 12 of [1..13] → mean(13, 1) = 7."""
        df = _single_firm_panel(13, v=[float(i) for i in range(1, 14)])
        result = df.select(mean_year("v").alias("out"))["out"][12]
        np.testing.assert_allclose(result, 7.0, **tolerance.TIGHT)

    def test_partial_history_returns_current(self, tolerance):
        """Row 0..11 lack a 12-month lag → fallback to current value."""
        df = _single_firm_panel(13, v=[float(i) for i in range(1, 14)])
        result = df.select(mean_year("v").alias("out"))["out"][5]
        np.testing.assert_allclose(result, 6.0, **tolerance.TIGHT)

    def test_only_current_returns_current(self, tolerance):
        df = _single_firm_panel(13, v=[None] * 12 + [5.0])
        result = df.select(mean_year("v").alias("out"))["out"][12]
        np.testing.assert_allclose(result, 5.0, **tolerance.TIGHT)

    def test_only_lag_returns_lag(self, tolerance):
        df = _single_firm_panel(13, v=[7.0] + [None] * 12)
        result = df.select(mean_year("v").alias("out"))["out"][12]
        np.testing.assert_allclose(result, 7.0, **tolerance.TIGHT)

    def test_both_null_returns_null(self):
        df = _single_firm_panel(13, v=[None] * 13)
        assert df.select(mean_year("v").alias("out"))["out"][12] is None

    def test_negative_values_average_to_zero(self, tolerance):
        v = [None] * 13
        v[0] = 10.0
        v[12] = -10.0
        df = _single_firm_panel(13, v=v)
        result = df.select(mean_year("v").alias("out"))["out"][12]
        np.testing.assert_allclose(result, 0.0, **tolerance.TIGHT)

    def test_does_not_cross_gvkey_boundary(self, tolerance):
        """Two firms — shift(12) must stay within each gvkey partition."""
        firm_a = _single_firm_panel(13, v=[float(i) for i in range(1, 14)]).with_columns(
            pl.lit(1, dtype=pl.Int64).alias("gvkey")
        )
        firm_b = _single_firm_panel(13, v=[100.0] * 13).with_columns(
            pl.lit(2, dtype=pl.Int64).alias("gvkey")
        )
        df = pl.concat([firm_a, firm_b]).sort(["gvkey", "datadate"])
        out = df.select([pl.col("gvkey"), mean_year("v").alias("m")])
        # Firm B row 12 (overall row 25): both current and lag are 100.0 → mean=100.0,
        # NOT (100 + firm_A_value)/2 if shift leaked.
        np.testing.assert_allclose(
            out.filter(pl.col("gvkey") == 2)["m"][12], 100.0, **tolerance.TIGHT
        )

    def test_does_not_cross_curcd_boundary(self, tolerance):
        """Same gvkey, two curcds → lag must respect currency partition."""
        n = 13
        df = pl.concat(
            [
                pl.DataFrame(
                    {
                        "gvkey": [1] * n,
                        "curcd": ["USD"] * n,
                        "datadate": _monthly_dates(n),
                        "v": [float(i) for i in range(1, n + 1)],
                    }
                ),
                pl.DataFrame(
                    {
                        "gvkey": [1] * n,
                        "curcd": ["EUR"] * n,
                        "datadate": _monthly_dates(n),
                        "v": [999.0] * n,
                    }
                ),
            ]
        ).sort(["gvkey", "curcd", "datadate"])
        out = df.select([pl.col("curcd"), mean_year("v").alias("m")])
        np.testing.assert_allclose(
            out.filter(pl.col("curcd") == "USD")["m"][12], 7.0, **tolerance.TIGHT
        )
        np.testing.assert_allclose(
            out.filter(pl.col("curcd") == "EUR")["m"][12], 999.0, **tolerance.TIGHT
        )

    def test_full_panel_values(self, tolerance):
        df = _single_firm_panel(15, v=[float(i) for i in range(1, 16)])
        result = df.select(mean_year("v").alias("out"))["out"].to_list()
        # rows 0..11 fall back to current; rows 12..14 average current with lag-12.
        expected = [float(i) for i in range(1, 13)] + [7.0, 8.0, 9.0]
        np.testing.assert_allclose(result, expected, **tolerance.TIGHT)


# =============================================================================
# temp_liq_rat
# =============================================================================


class TestTempLiqRat:
    """Tests for temp_liq_rat() — liquidity ratio gated on count > 12."""

    def test_normal_ratio(self, tolerance):
        df = _single_firm_panel(
            13,
            col_avg=[100.0] * 13,
            den=[50.0] * 13,
        )
        # count column from panel goes 1..13; gate is count > 12 → only row 12 passes.
        result = df.select(temp_liq_rat("col_avg", "den", "out"))["out"][12]
        np.testing.assert_allclose(result, 365 * 100 / 50, **tolerance.FINANCIAL_RATIOS)

    def test_count_at_threshold_returns_null(self):
        df = _single_firm_panel(13, col_avg=[100.0] * 13, den=[50.0] * 13).with_columns(
            pl.lit(12).alias("count")
        )
        out = df.select(temp_liq_rat("col_avg", "den", "out"))["out"]
        assert all(v is None for v in out.to_list())

    def test_count_just_above_threshold(self, tolerance):
        df = _single_firm_panel(13, col_avg=[100.0] * 13, den=[50.0] * 13).with_columns(
            pl.lit(13).alias("count")
        )
        result = df.select(temp_liq_rat("col_avg", "den", "out"))["out"][12]
        np.testing.assert_allclose(result, 365 * 100 / 50, **tolerance.FINANCIAL_RATIOS)

    def test_zero_denominator_returns_null(self):
        df = _single_firm_panel(13, col_avg=[100.0] * 13, den=[0.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        out = df.select(temp_liq_rat("col_avg", "den", "out"))["out"][12]
        assert out is None

    def test_negative_denominator_flows_through(self, tolerance):
        df = _single_firm_panel(13, col_avg=[100.0] * 13, den=[-50.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        result = df.select(temp_liq_rat("col_avg", "den", "out"))["out"][12]
        np.testing.assert_allclose(result, 365 * 100 / -50, **tolerance.FINANCIAL_RATIOS)

    def test_null_history_returns_null(self):
        df = _single_firm_panel(13, col_avg=[None] * 13, den=[50.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        assert df.select(temp_liq_rat("col_avg", "den", "out"))["out"][12] is None

    def test_alias_parameter_respected(self):
        df = _single_firm_panel(13, col_avg=[100.0] * 13, den=[50.0] * 13)
        assert df.select(temp_liq_rat("col_avg", "den", "days_inv")).columns == ["days_inv"]

    def test_null_count_column_returns_null(self):
        df = _single_firm_panel(13, col_avg=[100.0] * 13, den=[50.0] * 13).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("count")
        )
        assert df.select(temp_liq_rat("col_avg", "den", "out"))["out"][12] is None


# =============================================================================
# temp_rat_other
# =============================================================================


class TestTempRatOther:
    """Tests for temp_rat_other() — generic ratio with year-mean denominator."""

    def test_normal_ratio(self, tolerance):
        df = _single_firm_panel(13, num=[100.0] * 13, den=[20.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        result = df.select(temp_rat_other("num", "den", "out"))["out"][12]
        np.testing.assert_allclose(result, 100 / 20, **tolerance.FINANCIAL_RATIOS)

    def test_count_at_threshold_returns_null(self):
        df = _single_firm_panel(13, num=[100.0] * 13, den=[20.0] * 13).with_columns(
            pl.lit(12).alias("count")
        )
        assert df.select(temp_rat_other("num", "den", "out"))["out"][12] is None

    def test_count_above_threshold_returns_value(self, tolerance):
        df = _single_firm_panel(13, num=[100.0] * 13, den=[20.0] * 13).with_columns(
            pl.lit(13).alias("count")
        )
        result = df.select(temp_rat_other("num", "den", "out"))["out"][12]
        np.testing.assert_allclose(result, 5.0, **tolerance.FINANCIAL_RATIOS)

    def test_zero_mean_denominator_returns_null(self):
        df = _single_firm_panel(13, num=[100.0] * 13, den=[0.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        assert df.select(temp_rat_other("num", "den", "out"))["out"][12] is None

    def test_null_numerator_returns_null(self):
        df = _single_firm_panel(13, num=[None] * 13, den=[20.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        assert df.select(temp_rat_other("num", "den", "out"))["out"][12] is None

    def test_negative_numerator_flows_through(self, tolerance):
        df = _single_firm_panel(13, num=[-100.0] * 13, den=[20.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        result = df.select(temp_rat_other("num", "den", "out"))["out"][12]
        np.testing.assert_allclose(result, -5.0, **tolerance.FINANCIAL_RATIOS)

    def test_alias_parameter_respected(self):
        df = _single_firm_panel(13, num=[1.0] * 13, den=[1.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        assert df.select(temp_rat_other("num", "den", "asset_turnover")).columns == [
            "asset_turnover"
        ]


# =============================================================================
# temp_rat_other_spc (ap_turnover)
# =============================================================================


class TestTempRatOtherSpc:
    """Tests for temp_rat_other_spc() — hard-coded AP turnover."""

    def test_normal_ratio(self, tolerance):
        # cogs+invt-invt[0] at row 12: (50+30-30)=50; mean_year(ap)[12]=(20+10)/2=15
        df = _single_firm_panel(
            13,
            cogs=[50.0] * 13,
            invt=[30.0] + [30.0] * 12,
            ap=[10.0] + [None] * 11 + [20.0],
        ).with_columns(pl.lit(24).alias("count"))
        result = df.select(temp_rat_other_spc())["ap_turnover"][12]
        expected = 50.0 / 15.0
        np.testing.assert_allclose(result, expected, **tolerance.FINANCIAL_RATIOS)

    def test_count_at_threshold_returns_null(self):
        df = _single_firm_panel(
            13, cogs=[50.0] * 13, invt=[30.0] * 13, ap=[20.0] * 13
        ).with_columns(pl.lit(12).alias("count"))
        assert df.select(temp_rat_other_spc())["ap_turnover"][12] is None

    def test_zero_mean_ap_returns_null(self):
        df = _single_firm_panel(13, cogs=[50.0] * 13, invt=[30.0] * 13, ap=[0.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        assert df.select(temp_rat_other_spc())["ap_turnover"][12] is None

    def test_alias_is_ap_turnover(self):
        df = _single_firm_panel(
            13, cogs=[50.0] * 13, invt=[30.0] * 13, ap=[20.0] * 13
        ).with_columns(pl.lit(24).alias("count"))
        assert df.select(temp_rat_other_spc()).columns == ["ap_turnover"]

    def test_null_cogs_propagates(self):
        cogs = [50.0] * 13
        cogs[12] = None
        df = _single_firm_panel(13, cogs=cogs, invt=[30.0] * 13, ap=[20.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        assert df.select(temp_rat_other_spc())["ap_turnover"][12] is None

    def test_null_invt_lag_propagates(self):
        invt = [30.0] * 13
        invt[0] = None
        df = _single_firm_panel(13, cogs=[50.0] * 13, invt=invt, ap=[20.0] * 13).with_columns(
            pl.lit(24).alias("count")
        )
        assert df.select(temp_rat_other_spc())["ap_turnover"][12] is None

    def test_does_not_cross_gvkey(self, tolerance):
        """Two firms — invt.shift(12) must respect gvkey partition."""
        firm_a = _single_firm_panel(
            13, cogs=[50.0] * 13, invt=[30.0] * 13, ap=[20.0] * 13
        ).with_columns(pl.lit(24).alias("count"))
        firm_b = _single_firm_panel(
            13, cogs=[50.0] * 13, invt=[30.0] * 13, ap=[20.0] * 13
        ).with_columns([pl.lit(2, dtype=pl.Int64).alias("gvkey"), pl.lit(24).alias("count")])
        df = pl.concat([firm_a, firm_b]).sort(["gvkey", "curcd", "datadate"])
        out = df.select([pl.col("gvkey"), temp_rat_other_spc()])
        # numerator for firm B row 12 = 50+30-30 = 50; denom = 20 → 2.5
        np.testing.assert_allclose(
            out.filter(pl.col("gvkey") == 2)["ap_turnover"][12], 2.5, **tolerance.FINANCIAL_RATIOS
        )


# =============================================================================
# update_ni_inc_and_decrease
# =============================================================================


def _streak_frame(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """Build a small DataFrame for update_ni_inc_and_decrease tests.

    rows: list of (ni_inc, no_decrease, ni_inc8q) tuples in monthly order.
    """
    n = len(rows)
    return pl.DataFrame(
        {
            "gvkey": [1] * n,
            "curcd": ["USD"] * n,
            "datadate": _monthly_dates(n),
            "ni_inc": [r[0] for r in rows],
            "no_decrease": [r[1] for r in rows],
            "ni_inc8q": [r[2] for r in rows],
        }
    )


class TestUpdateNiIncAndDecrease:
    """Tests for update_ni_inc_and_decrease() — streak-counter helper."""

    def test_increments_when_both_conditions_met_lag0(self):
        df = _streak_frame([(1, 1, 0), (1, 1, 0), (1, 1, 0)])
        out = update_ni_inc_and_decrease(df, 0)
        assert out["ni_inc8q"].to_list() == [1, 1, 1]
        assert out["no_decrease"].to_list() == [1, 1, 1]

    def test_resets_no_decrease_when_ni_inc_zero(self):
        df = _streak_frame([(0, 1, 0), (0, 1, 0), (0, 1, 0)])
        out = update_ni_inc_and_decrease(df, 0)
        assert out["ni_inc8q"].to_list() == [0, 0, 0]
        assert out["no_decrease"].to_list() == [0, 0, 0]

    def test_preserves_when_already_broken(self):
        df = _streak_frame([(1, 0, 3), (1, 0, 3), (1, 0, 3)])
        out = update_ni_inc_and_decrease(df, 0)
        # no_decrease was 0 — condition fails → no_decrease forced to 0 (no change).
        assert out["ni_inc8q"].to_list() == [3, 3, 3]
        assert out["no_decrease"].to_list() == [0, 0, 0]

    def test_lag_nonzero_uses_shift(self):
        # ni_inc=[1, 1, 1]; with lag=2, shift(2) = [None, None, 1].
        # Only row 2 has shifted ni_inc == 1, no_decrease == 1 → increment row 2.
        df = _streak_frame([(1, 1, 0), (1, 1, 0), (1, 1, 0)])
        out = update_ni_inc_and_decrease(df, 2)
        assert out["ni_inc8q"].to_list() == [0, 0, 1]
        assert out["no_decrease"].to_list() == [0, 0, 1]

    def test_null_shifted_value_resets_no_decrease(self):
        # lag=1: shift = [None, 1, 1]. Row 0: condition null → no_decrease=0, no inc.
        df = _streak_frame([(1, 1, 0), (1, 1, 0), (1, 1, 0)])
        out = update_ni_inc_and_decrease(df, 1)
        assert out["no_decrease"][0] == 0
        assert out["ni_inc8q"][0] == 0

    def test_returns_sorted_output(self):
        n = 4
        df = pl.DataFrame(
            {
                "gvkey": [1, 1, 1, 1],
                "curcd": ["USD"] * 4,
                "datadate": _monthly_dates(n)[::-1],
                "ni_inc": [1, 1, 1, 1],
                "no_decrease": [1, 1, 1, 1],
                "ni_inc8q": [0, 0, 0, 0],
            }
        )
        out = update_ni_inc_and_decrease(df, 0)
        assert out["datadate"].to_list() == sorted(df["datadate"].to_list())

    def test_idempotent_when_no_decrease_zero(self):
        df = _streak_frame([(1, 0, 5)])
        out = update_ni_inc_and_decrease(df, 0)
        assert out["no_decrease"][0] == 0
        assert out["ni_inc8q"][0] == 5

    def test_multi_firm_shift_is_global(self):
        """`.shift(lag)` is applied without `.over` — documents observed behavior.

        First row of firm B sees last row of firm A through the shift, since
        the shift is global to the sorted DataFrame. Future refactor to add
        `.over(['gvkey','curcd'])` would change this and is intentional.
        """
        n = 3
        df = pl.DataFrame(
            {
                "gvkey": [1, 1, 1, 2, 2, 2],
                "curcd": ["USD"] * 6,
                "datadate": _monthly_dates(n) + _monthly_dates(n),
                "ni_inc": [1, 1, 1, 1, 1, 1],
                "no_decrease": [1, 1, 1, 1, 1, 1],
                "ni_inc8q": [0, 0, 0, 0, 0, 0],
            }
        ).sort(["gvkey", "curcd", "datadate"])
        out = update_ni_inc_and_decrease(df, 1)
        # Firm B row 0 (overall row 3): shift(1) = firm A row 2's ni_inc = 1.
        # Both conditions met → increment to 1.
        firm_b = out.filter(pl.col("gvkey") == 2)
        assert firm_b["ni_inc8q"][0] == 1


# =============================================================================
# calculate_consecutive_earnings_increases
# =============================================================================


def _ni_panel(n: int, ni_x: list[float | None]) -> pl.DataFrame:
    return _single_firm_panel(n, ni_x=ni_x)


class TestCalculateConsecutiveEarningsIncreases:
    """Tests for calculate_consecutive_earnings_increases()."""

    def test_eight_consecutive_increases(self):
        n = 36
        df = _ni_panel(n, [float(i) for i in range(n)])
        out = calculate_consecutive_earnings_increases(df)
        assert out["ni_inc8q"][35] == 8

    def test_decrease_at_end_zeros_streak(self):
        n = 36
        ni_x = [float(i) for i in range(n)]
        ni_x[35] = -100.0
        df = _ni_panel(n, ni_x)
        out = calculate_consecutive_earnings_increases(df)
        # Row 35 ni_inc=0 → lag=0 fails → no_decrease=0, counter stays 0.
        assert out["ni_inc8q"][35] == 0

    def test_decrease_mid_window_partial_streak(self):
        n = 36
        ni_x = [float(i) for i in range(n)]
        ni_x[26] = -100.0  # ni_inc[26] = 0; at row 35 lag=9 reads row 26 → break
        df = _ni_panel(n, ni_x)
        out = calculate_consecutive_earnings_increases(df)
        assert 0 <= out["ni_inc8q"][35] < 8

    def test_flat_earnings_zero_streak(self):
        n = 36
        df = _ni_panel(n, [5.0] * n)
        out = calculate_consecutive_earnings_increases(df)
        # ni_x not greater than lag → ni_inc=0 → counter stays 0
        assert out["ni_inc8q"][35] == 0

    def test_count_below_33_returns_null(self):
        n = 36
        df = _ni_panel(n, [float(i) for i in range(n)]).with_columns(
            (pl.col("count") - 10).alias("count")
        )
        # No row has count >= 33 here.
        out = calculate_consecutive_earnings_increases(df)
        assert all(v is None for v in out["ni_inc8q"].to_list())

    def test_short_history_returns_null(self):
        n = 10
        df = _ni_panel(n, [float(i) for i in range(n)])
        out = calculate_consecutive_earnings_increases(df)
        assert all(v is None for v in out["ni_inc8q"].to_list())

    def test_null_ni_x_in_lag_window_returns_null(self):
        n = 36
        ni_x: list[float | None] = [float(i) for i in range(n)]
        ni_x[14] = None  # shift(21) at row 35 reads row 14 → null ni_inc
        df = _ni_panel(n, ni_x)
        out = calculate_consecutive_earnings_increases(df)
        # n_ni_inc < 8 → c1 fails → null
        assert out["ni_inc8q"][35] is None

    def test_intermediate_columns_dropped(self):
        df = _ni_panel(36, [float(i) for i in range(36)])
        out = calculate_consecutive_earnings_increases(df)
        for col_name in ("ni_inc", "no_decrease", "n_ni_inc"):
            assert col_name not in out.columns

    def test_input_columns_preserved(self):
        df = _ni_panel(36, [float(i) for i in range(36)])
        out = calculate_consecutive_earnings_increases(df)
        for col_name in ("gvkey", "curcd", "datadate", "ni_x", "count"):
            assert col_name in out.columns

    def test_output_sorted(self):
        df = _ni_panel(36, [float(i) for i in range(36)]).reverse()
        out = calculate_consecutive_earnings_increases(df)
        assert out["datadate"].to_list() == sorted(out["datadate"].to_list())

    def test_multi_firm_independent(self):
        n = 36
        firm_a = _ni_panel(n, [float(i) for i in range(n)])
        ni_b = [float(i) for i in range(n)]
        ni_b[35] = -100.0
        firm_b = _ni_panel(n, ni_b).with_columns(pl.lit(2, dtype=pl.Int64).alias("gvkey"))
        df = pl.concat([firm_a, firm_b])
        out = calculate_consecutive_earnings_increases(df)
        a = out.filter(pl.col("gvkey") == 1)
        b = out.filter(pl.col("gvkey") == 2)
        assert a["ni_inc8q"][35] == 8
        assert b["ni_inc8q"][35] == 0

    def test_count_gate_boundary(self):
        """Count must be >=33 AND all 8 lags non-null; row 33 (count=34) is first
        position where both can hold with monotone data starting at index 0."""
        n = 36
        df = _ni_panel(n, [float(i) for i in range(n)])
        out = calculate_consecutive_earnings_increases(df)
        # Row 32 (count=33) → shift(21) reads row 11 which has no 12-month lag
        # (ni_inc[11] is null) → n_ni_inc<8 → null.
        assert out["ni_inc8q"][32] is None
        # Row 33 (count=34) → all 8 lags non-null and increasing → 8.
        assert out["ni_inc8q"][33] == 8


# =============================================================================
# compute_capex_abn
# =============================================================================


class TestComputeCapexAbn:
    """Tests for compute_capex_abn() — abnormal capex vs 3-year trailing avg."""

    def test_constant_ratio_returns_zero(self, tolerance):
        n = 37
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=[100.0] * n)
        out = compute_capex_abn(df)
        np.testing.assert_allclose(out["capex_abn"][36], 0.0, **tolerance.FINANCIAL_RATIOS)

    def test_doubled_current_ratio(self, tolerance):
        n = 37
        capx = [10.0] * n
        capx[36] = 20.0  # double the ratio at current row
        df = _single_firm_panel(n, capx=capx, sale_x=[100.0] * n)
        out = compute_capex_abn(df)
        np.testing.assert_allclose(out["capex_abn"][36], 1.0, **tolerance.FINANCIAL_RATIOS)

    def test_halved_current_ratio(self, tolerance):
        n = 37
        capx = [10.0] * n
        capx[36] = 5.0
        df = _single_firm_panel(n, capx=capx, sale_x=[100.0] * n)
        out = compute_capex_abn(df)
        np.testing.assert_allclose(out["capex_abn"][36], -0.5, **tolerance.FINANCIAL_RATIOS)

    def test_count_at_threshold_returns_null(self):
        n = 37
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=[100.0] * n)
        # Row 35 has count=36 → fails count>36 gate.
        out = compute_capex_abn(df)
        assert out["capex_abn"][35] is None

    def test_count_above_threshold_returns_value(self):
        n = 37
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=[100.0] * n)
        out = compute_capex_abn(df)
        assert out["capex_abn"][36] is not None

    def test_zero_sale_at_current_returns_null(self):
        n = 37
        sale = [100.0] * n
        sale[36] = 0.0
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=sale)
        out = compute_capex_abn(df)
        assert out["capex_abn"][36] is None

    def test_negative_sale_returns_null(self):
        """safe_div mode 3 requires sale_x > 0 → negative sale → __capex_sale null."""
        n = 37
        sale = [100.0] * n
        sale[36] = -50.0
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=sale)
        out = compute_capex_abn(df)
        assert out["capex_abn"][36] is None

    def test_zero_sale_in_lag_window_returns_null(self):
        n = 37
        sale = [100.0] * n
        sale[24] = 0.0  # row 36 - 12 = row 24 → __capex_sale[24] null
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=sale)
        out = compute_capex_abn(df)
        assert out["capex_abn"][36] is None

    def test_null_capx_propagates(self):
        n = 37
        capx: list[float | None] = [10.0] * n
        capx[36] = None
        df = _single_firm_panel(n, capx=capx, sale_x=[100.0] * n)
        out = compute_capex_abn(df)
        assert out["capex_abn"][36] is None

    def test_helper_column_dropped(self):
        n = 37
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=[100.0] * n)
        out = compute_capex_abn(df)
        assert "__capex_sale" not in out.columns

    def test_output_sorted(self):
        n = 37
        df = _single_firm_panel(n, capx=[10.0] * n, sale_x=[100.0] * n).reverse()
        out = compute_capex_abn(df)
        assert out["datadate"].to_list() == sorted(out["datadate"].to_list())

    def test_multi_firm_independent(self, tolerance):
        n = 37
        firm_a = _single_firm_panel(n, capx=[10.0] * n, sale_x=[100.0] * n)
        capx_b = [10.0] * n
        capx_b[36] = 20.0
        firm_b = _single_firm_panel(n, capx=capx_b, sale_x=[100.0] * n).with_columns(
            pl.lit(2, dtype=pl.Int64).alias("gvkey")
        )
        df = pl.concat([firm_a, firm_b])
        out = compute_capex_abn(df)
        a = out.filter(pl.col("gvkey") == 1)
        b = out.filter(pl.col("gvkey") == 2)
        np.testing.assert_allclose(a["capex_abn"][36], 0.0, **tolerance.FINANCIAL_RATIOS)
        np.testing.assert_allclose(b["capex_abn"][36], 1.0, **tolerance.FINANCIAL_RATIOS)
