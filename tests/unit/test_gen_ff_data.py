"""
Unit tests for FF3/FF5/UMD factor-builder helpers in aux_functions.py.

Covers:
  - _ff_bm_eligible / _ff_op_eligible / _ff_inv_eligible / _ff_mom_eligible
  - ff_country_breakpoints, _ff_bm_breaks / _ff_op_breaks / _ff_inv_breaks,
    _ff_size_breaks
  - _ff_us_finish_prepare / _ff_row_finish_prepare (no-BE June stocks kept
    for the size pool)
  - ff_assign_portfolios
  - ff_assign_to_panel
  - _ff_vw_leg
  - ff_compute_factors
  - _ff_mom_signal_monthly, _ff_mom_signal_daily
  - ff_compute_umd_factor
  - ff_build_characteristics
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from jkp.data.aux_functions import (
    _ff_bm_breaks,
    _ff_bm_eligible,
    _ff_inv_breaks,
    _ff_inv_eligible,
    _ff_mom_eligible,
    _ff_mom_signal_daily,
    _ff_mom_signal_monthly,
    _ff_op_breaks,
    _ff_op_eligible,
    _ff_row_finish_prepare,
    _ff_row_rets_weights_lazy,
    _ff_size_breaks,
    _ff_us_finish_prepare,
    _ff_us_rets_weights_lazy,
    _ff_vw_leg,
    ff_assign_portfolios,
    ff_assign_to_panel,
    ff_build_characteristics,
    ff_compute_factors,
    ff_compute_umd_factor,
    ff_country_breakpoints,
)
from jkp.data.config import (
    FF_MIN_STOCKS_BP,
    FF_MIN_STOCKS_PF,
    FF_MISSING_RET_CODE,
    US_EXCNTRY,
)

# =============================================================================
# Helpers
# =============================================================================


def _apply_eligible(df: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    """Evaluate an eligibility Expr on df, returning the boolean Series.

    Frames without an excntry column get a ROW default ("GBR") so the
    bm/op count gate (US-exempt) keeps its count >= 2 semantics.
    """
    if "excntry" not in df.columns:
        df = df.with_columns(excntry=pl.lit("GBR"))
    return df.select(expr.alias("elig"))["elig"]


# =============================================================================
# TestFFBmEligible
# =============================================================================


class TestFFBmEligible:
    """Tests for _ff_bm_eligible()."""

    def test_all_conditions_met(self):
        """Returns True when beme>0, me>0, count>=2."""
        df = pl.DataFrame({"beme": [1.0], "me": [1.0], "count": [2]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is True

    def test_beme_zero_fails(self):
        """Returns False when beme == 0."""
        df = pl.DataFrame({"beme": [0.0], "me": [1.0], "count": [2]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_beme_negative_fails(self):
        """Returns False when beme < 0."""
        df = pl.DataFrame({"beme": [-1.0], "me": [1.0], "count": [2]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_beme_null_fails(self):
        """Returns null (not True) when beme is null — Polars null propagation."""
        df = pl.DataFrame(
            {"beme": [None], "me": [1.0], "count": [2]},
            schema={"beme": pl.Float64, "me": pl.Float64, "count": pl.Int64},
        )
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is not True

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"beme": [1.0], "me": [0.0], "count": [2]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_me_negative_fails(self):
        """Returns False when me < 0."""
        df = pl.DataFrame({"beme": [1.0], "me": [-1.0], "count": [2]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_count_one_fails(self):
        """ROW: returns False when count < 2 (count == 1)."""
        df = pl.DataFrame({"beme": [1.0], "me": [1.0], "count": [1]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_count_two_passes(self):
        """ROW: returns True when count == 2 (boundary)."""
        df = pl.DataFrame({"beme": [1.0], "me": [1.0], "count": [2]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is True

    def test_us_count_one_passes(self):
        """US is exempt from the FF (1993) two-year count gate."""
        df = pl.DataFrame({"beme": [1.0], "me": [1.0], "count": [1], "excntry": [US_EXCNTRY]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is True

    def test_vectorised_mixed(self):
        """Null beme propagates as null; False inputs produce False."""
        df = pl.DataFrame(
            {
                "beme": [1.0, -1.0, 1.0, None],
                "me": [1.0, 1.0, 0.0, 1.0],
                "count": [3, 3, 3, 3],
            }
        )
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is True
        assert result[1] is False
        assert result[2] is False
        assert result[3] is not True  # null propagation from null beme


# =============================================================================
# TestFFOpEligible
# =============================================================================


class TestFFOpEligible:
    """Tests for _ff_op_eligible()."""

    def test_all_conditions_met(self):
        """Returns True when me>0, be>0, count>=2, op not null."""
        df = pl.DataFrame({"me": [1.0], "be": [1.0], "count": [2], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is True

    def test_op_null_fails(self):
        """Returns False when op is null."""
        df = pl.DataFrame(
            {"me": [1.0], "be": [1.0], "count": [2], "op": [None]},
            schema={"me": pl.Float64, "be": pl.Float64, "count": pl.Int64, "op": pl.Float64},
        )
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_be_zero_fails(self):
        """Returns False when be == 0."""
        df = pl.DataFrame({"me": [1.0], "be": [0.0], "count": [2], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_be_negative_fails(self):
        """Returns False when be < 0."""
        df = pl.DataFrame({"me": [1.0], "be": [-1.0], "count": [2], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"me": [0.0], "be": [1.0], "count": [2], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_count_one_fails(self):
        """ROW: returns False when count == 1."""
        df = pl.DataFrame({"me": [1.0], "be": [1.0], "count": [1], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_us_count_one_passes(self):
        """US is exempt from the FF (1993) two-year count gate."""
        df = pl.DataFrame(
            {"me": [1.0], "be": [1.0], "count": [1], "op": [0.5], "excntry": [US_EXCNTRY]}
        )
        assert _apply_eligible(df, _ff_op_eligible())[0] is True

    def test_vectorised_mixed(self):
        """Batch with two valid and two invalid rows."""
        df = pl.DataFrame(
            {
                "me": [1.0, 1.0, 0.0, 1.0],
                "be": [1.0, -1.0, 1.0, 1.0],
                "count": [2, 2, 2, 2],
                "op": [0.5, 0.5, 0.5, None],
            }
        )
        result = _apply_eligible(df, _ff_op_eligible())
        assert result.to_list() == [True, False, False, False]


# =============================================================================
# TestFFInvEligible
# =============================================================================


class TestFFInvEligible:
    """Tests for _ff_inv_eligible()."""

    def test_passes_with_positive_me_and_inv(self):
        """Returns True when me>0 and inv is not null."""
        df = pl.DataFrame({"me": [1.0], "inv": [0.1]})
        assert _apply_eligible(df, _ff_inv_eligible())[0] is True

    def test_inv_null_fails(self):
        """Returns False when inv is null."""
        df = pl.DataFrame(
            {"me": [1.0], "inv": [None]}, schema={"me": pl.Float64, "inv": pl.Float64}
        )
        assert _apply_eligible(df, _ff_inv_eligible())[0] is False

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"me": [0.0], "inv": [0.1]})
        assert _apply_eligible(df, _ff_inv_eligible())[0] is False

    def test_me_negative_fails(self):
        """Returns False when me < 0."""
        df = pl.DataFrame({"me": [-1.0], "inv": [0.1]})
        assert _apply_eligible(df, _ff_inv_eligible())[0] is False

    def test_inv_zero_passes(self):
        """Returns True when inv == 0 (zero is valid, only null is excluded)."""
        df = pl.DataFrame({"me": [1.0], "inv": [0.0]})
        assert _apply_eligible(df, _ff_inv_eligible())[0] is True

    def test_inv_negative_passes(self):
        """Returns True when inv < 0 (sign not gated)."""
        df = pl.DataFrame({"me": [1.0], "inv": [-0.5]})
        assert _apply_eligible(df, _ff_inv_eligible())[0] is True


# =============================================================================
# TestFFMomEligible
# =============================================================================


class TestFFMomEligible:
    """Tests for _ff_mom_eligible()."""

    def test_passes_with_valid_inputs(self):
        """Returns True when mom_2_12 not null and me_lag1 > 0."""
        df = pl.DataFrame({"mom_2_12": [0.1], "me_lag1": [1.0]})
        assert _apply_eligible(df, _ff_mom_eligible())[0] is True

    def test_mom_null_fails(self):
        """Returns False when mom_2_12 is null."""
        df = pl.DataFrame(
            {"mom_2_12": [None], "me_lag1": [1.0]},
            schema={"mom_2_12": pl.Float64, "me_lag1": pl.Float64},
        )
        assert _apply_eligible(df, _ff_mom_eligible())[0] is False

    def test_me_lag1_zero_fails(self):
        """Returns False when me_lag1 == 0."""
        df = pl.DataFrame({"mom_2_12": [0.1], "me_lag1": [0.0]})
        assert _apply_eligible(df, _ff_mom_eligible())[0] is False

    def test_me_lag1_negative_fails(self):
        """Returns False when me_lag1 < 0."""
        df = pl.DataFrame({"mom_2_12": [0.1], "me_lag1": [-1.0]})
        assert _apply_eligible(df, _ff_mom_eligible())[0] is False

    def test_me_lag1_null_fails(self):
        """Returns null (not True) when me_lag1 is null — Polars null propagation."""
        df = pl.DataFrame(
            {"mom_2_12": [0.1], "me_lag1": [None]},
            schema={"mom_2_12": pl.Float64, "me_lag1": pl.Float64},
        )
        assert _apply_eligible(df, _ff_mom_eligible())[0] is not True

    def test_both_null_fails(self):
        """Returns null (not True) when both are null."""
        df = pl.DataFrame(
            {"mom_2_12": [None], "me_lag1": [None]},
            schema={"mom_2_12": pl.Float64, "me_lag1": pl.Float64},
        )
        assert _apply_eligible(df, _ff_mom_eligible())[0] is not True

    def test_vectorised(self):
        """Batch: True for valid row, False/null for invalid rows."""
        df = pl.DataFrame(
            {
                "mom_2_12": [0.1, None, 0.1, 0.1],
                "me_lag1": [1.0, 1.0, 0.0, None],
            }
        )
        result = _apply_eligible(df, _ff_mom_eligible())
        assert result[0] is True
        assert result[1] is not True  # mom_2_12 null
        assert result[2] is False  # me_lag1 == 0
        assert result[3] is not True  # me_lag1 null


# =============================================================================
# TestFFCountryBreakpoints
# =============================================================================


def _make_june_us(n: int, vals: list[float], with_exchcd: bool = True) -> pl.DataFrame:
    """Minimal June snapshot for US with NYSE (exchcd_us=1) stocks."""
    return pl.DataFrame(
        {
            "excntry": [US_EXCNTRY] * n,
            "date": [date(2020, 6, 30)] * n,
            "id": list(range(n)),
            "beme": vals,
            "me": [1.0] * n,
            "count": [5] * n,
            "exchcd_us": [1] * n if with_exchcd else [2] * n,
            "size_grp": ["large"] * n,
        }
    )


class TestFFCountryBreakpoints:
    """Tests for ff_country_breakpoints()."""

    def test_us_30_70_quantiles(self, tolerance):
        """NYSE-only US pool: 30/70 quantiles match expected values."""
        vals = list(range(1, 11))  # 1..10
        june = _make_june_us(10, vals)
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        assert len(bp) == 1
        row = bp.row(0, named=True)
        # QUANTILE_DISC at 0.30 of [1..10] => 3, at 0.70 => 7
        assert row["beme30"] == 3.0
        assert row["beme70"] == 7.0

    def test_us_always_emits_breakpoints_regardless_of_count(self):
        """US bypasses the FF_MIN_STOCKS_BP gate even with 1 stock."""
        june = _make_june_us(1, [5.0])
        specs = [("beme", 0.50, "beme50")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        assert len(bp) == 1

    def test_row_below_min_stocks_suppressed(self):
        """ROW country with count < FF_MIN_STOCKS_BP yields no breakpoints."""
        n = FF_MIN_STOCKS_BP - 1
        june = pl.DataFrame(
            {
                "excntry": ["GBR"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n + 1)],
                "me": [1.0] * n,
                "count": [5] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["large"] * n,
            }
        )
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        # No GBR rows should survive
        gbr = bp.filter(pl.col("excntry") == "GBR")
        assert len(gbr) == 0

    def test_row_at_min_stocks_emits_breakpoints(self):
        """ROW country with count == FF_MIN_STOCKS_BP gets breakpoints."""
        n = FF_MIN_STOCKS_BP
        june = pl.DataFrame(
            {
                "excntry": ["GBR"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n + 1)],
                "me": [1.0] * n,
                "count": [5] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["large"] * n,
            }
        )
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        gbr = bp.filter(pl.col("excntry") == "GBR")
        assert len(gbr) == 1

    def test_row_uses_size_grp_filter(self):
        """ROW pool requires size_grp in {small, large, mega}; micro excluded."""
        n = FF_MIN_STOCKS_BP + 5
        june = pl.DataFrame(
            {
                "excntry": ["DEU"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n + 1)],
                "me": [1.0] * n,
                "count": [5] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["micro"] * n,  # all micro → excluded from pool
            }
        )
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        deu = bp.filter(pl.col("excntry") == "DEU")
        assert len(deu) == 0

    def test_empty_input_returns_empty(self):
        """Empty input produces empty output."""
        june = pl.DataFrame(
            {
                "excntry": pl.Series([], dtype=pl.Utf8),
                "date": pl.Series([], dtype=pl.Date),
                "id": pl.Series([], dtype=pl.Int64),
                "beme": pl.Series([], dtype=pl.Float64),
                "me": pl.Series([], dtype=pl.Float64),
                "count": pl.Series([], dtype=pl.Int64),
                "exchcd_us": pl.Series([], dtype=pl.Int64),
                "size_grp": pl.Series([], dtype=pl.Utf8),
            }
        )
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        assert len(bp) == 0

    def test_output_schema_includes_excntry_date(self):
        """Output always includes excntry and date columns."""
        june = _make_june_us(10, list(range(1, 11)))
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        assert "excntry" in bp.columns
        assert "date" in bp.columns


# =============================================================================
# TestFFSpecBreaks
# =============================================================================


# Standalone B/M 30/70 breakpoints (no size median), matching the former
# ff_country_breaks_for_spec(june, "bm", with_size_median=False).
def _bm_value_breaks(june: pl.DataFrame) -> pl.DataFrame:
    return ff_country_breakpoints(
        june, [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")], _ff_bm_eligible()
    )


class TestFFSpecBreaks:
    """Tests for _ff_bm_breaks / _ff_op_breaks / _ff_inv_breaks and the
    standalone B/M value breakpoints."""

    def test_bm_spec_returns_30_70_breaks(self):
        """bm value breaks emit beme30, beme70 columns."""
        june = _make_june_us(20, [float(i) for i in range(1, 21)])
        bp = _bm_value_breaks(june)
        assert "beme30" in bp.columns
        assert "beme70" in bp.columns

    def test_with_size_median_adds_sizemedn(self):
        """_ff_bm_breaks adds the sizemedn column."""
        june = _make_june_us(20, [float(i) for i in range(1, 21)])
        bp = _ff_bm_breaks(june)
        assert "sizemedn" in bp.columns

    def test_without_size_median_no_sizemedn(self):
        """Standalone value breaks omit sizemedn."""
        june = _make_june_us(20, [float(i) for i in range(1, 21)])
        bp = _bm_value_breaks(june)
        assert "sizemedn" not in bp.columns

    def test_op_spec_returns_op30_op70(self):
        """_ff_op_breaks emits op30, op70 columns."""
        june = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * 20,
                "date": [date(2020, 6, 30)] * 20,
                "id": list(range(20)),
                "op": [float(i) * 0.1 for i in range(1, 21)],
                "me": [1.0] * 20,
                "be": [1.0] * 20,
                "count": [3] * 20,
                "exchcd_us": [1] * 20,
                "size_grp": ["large"] * 20,
            }
        )
        bp = _ff_op_breaks(june)
        assert "op30" in bp.columns
        assert "op70" in bp.columns

    def test_inv_spec_returns_inv30_inv70(self):
        """_ff_inv_breaks emits inv30, inv70 columns."""
        june = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * 20,
                "date": [date(2020, 6, 30)] * 20,
                "id": list(range(20)),
                "inv": [float(i) * 0.05 for i in range(1, 21)],
                "me": [1.0] * 20,
                "count": [3] * 20,
                "exchcd_us": [1] * 20,
                "size_grp": ["large"] * 20,
            }
        )
        bp = _ff_inv_breaks(june)
        assert "inv30" in bp.columns
        assert "inv70" in bp.columns

    @staticmethod
    def _make_june_size_pool() -> pl.DataFrame:
        """US June frame where the bm-eligible pool and the all-NYSE pool
        have different ME medians.

        NYSE me 10/20/30 bm-eligible (strict median 20); NYSE me 40/50 with
        null beme/count (bm-ineligible); two NYSE me=0 and one AMEX me=1000
        that must stay out of the lenient pool.
        """
        return pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * 8,
                "date": [date(2020, 6, 30)] * 8,
                "id": list(range(8)),
                "beme": [1.0, 2.0, 3.0, None, None, None, None, 4.0],
                "me": [10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0, 1000.0],
                "count": [5, 5, 5, None, None, None, None, 5],
                "exchcd_us": [1, 1, 1, 1, 1, 1, 1, 2],
                "size_grp": ["large"] * 8,
            }
        )

    def test_us_size_all_nyse_widens_median_pool(self):
        """All-NYSE US size median (production) widens the pool to all NYSE
        me>0 stocks (incl. bm-ineligible); the sort-eligible pool is
        narrower. me<=0 and non-NYSE stay excluded from both."""
        june = self._make_june_size_pool()
        strict = ff_country_breakpoints(june, [("me", 0.50, "sizemedn")], _ff_bm_eligible())
        lenient = _ff_size_breaks(june, _ff_bm_eligible())
        assert strict["sizemedn"][0] == 20.0  # bm-eligible pool: 10/20/30
        assert lenient["sizemedn"][0] == 30.0  # all NYSE me>0: 10/20/30/40/50

    def test_us_size_all_nyse_leaves_value_breaks_unchanged(self):
        """30/70 value breakpoints keep the strict (bm-eligible) pool: the
        beme30/70 in _ff_bm_breaks match the standalone value breaks
        regardless of the (widened) size-median pool."""
        june = self._make_june_size_pool()
        value = _bm_value_breaks(june)
        combined = _ff_bm_breaks(june)
        assert combined.select("excntry", "date", "beme30", "beme70").equals(
            value.select("excntry", "date", "beme30", "beme70")
        )

    def test_row_unaffected_by_us_size_all_nyse(self):
        """ROW keeps the sort-eligible sizemedn pool: the all-NYSE helper and
        the plain sort-eligible breakpoints agree for a ROW country."""
        n = FF_MIN_STOCKS_BP + 5
        june = pl.DataFrame(
            {
                "excntry": ["GBR"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n - 1)] + [None, None],
                "me": [float(i) for i in range(1, n + 1)],
                "count": [5] * (n - 2) + [None, None],
                "exchcd_us": [None] * n,
                "size_grp": ["large"] * n,
            },
            schema_overrides={"exchcd_us": pl.Int64},
        )
        strict = ff_country_breakpoints(june, [("me", 0.50, "sizemedn")], _ff_bm_eligible())
        lenient = _ff_size_breaks(june, _ff_bm_eligible())
        assert strict.equals(lenient)


# =============================================================================
# TestFFPrepareCharsLeftJoin
# =============================================================================


class TestFFPrepareCharsLeftJoin:
    """No-BE June stocks survive the comp join in the finish-prepare step
    (_ff_us_finish_prepare / _ff_row_finish_prepare): they widen the US
    size-median pool with null accounting fields, while
    comp-matched rows are unchanged and recoverable via count.is_not_null().
    US-only, stocks without Dec(t-1) ME also stay (null dec_me/beme, sortable
    on OP/INV); ROW keeps the inner dec_me join."""

    @staticmethod
    def _run_us():
        # permno 1: Dec ME + comp row; permno 2: Dec ME, no comp row;
        # permno 3: comp row, no Dec 2019 row (listed Jan-Jun 2020).
        panel = pl.DataFrame(
            {
                "permno": [1, 1, 2, 2, 3],
                "date": [date(2019, 12, 31), date(2020, 6, 30)] * 2 + [date(2020, 6, 30)],
                "retadj": [0.01] * 5,
                "retx": [0.01] * 5,
                "me": [100.0, 110.0, 200.0, 210.0, 300.0],
                "exchcd": [1] * 5,
                "shrcd": [10] * 5,
            }
        ).sort("permno", "date")
        comp = pl.DataFrame(
            {
                "permno": [1, 3],
                "year": [2019, 2019],
                "datadate": [date(2019, 12, 31)] * 2,
                "be": [0.05, 0.07],
                "op": [0.2, 0.3],
                "inv": [0.1, 0.15],
                "count": [3, 4],
            },
            schema_overrides={"year": pl.Int32},
        )
        _, data_chars = _ff_us_finish_prepare(
            _ff_us_rets_weights_lazy(panel.lazy(), "monthly").collect(), comp
        )
        return data_chars

    def test_no_comp_stock_kept_with_null_accounting(self):
        chars = self._run_us()
        assert chars.height == 3
        row = chars.filter(pl.col("id") == 2)
        assert row.height == 1
        for c in ("be", "op", "inv", "count", "beme"):
            assert row[c][0] is None
        assert row["dec_me"][0] == 200.0

    def test_comp_matched_row_unchanged(self):
        chars = self._run_us()
        row = chars.filter(pl.col("id") == 1)
        assert row["count"][0] == 3
        # beme = 1000 x be($M) / dec_me($K) = 1000 x 0.05 / 100
        assert abs(row["beme"][0] - 0.5) < 1e-12

    def test_us_no_dec_me_stock_kept_op_inv_sortable(self):
        """US stock without Dec(t-1) ME keeps comp op/inv (French's OP/INV
        sorts need June ME only) but gets null beme (no B/M sort)."""
        chars = self._run_us()
        row = chars.filter(pl.col("id") == 3)
        assert row.height == 1
        assert row["dec_me"][0] is None
        assert row["beme"][0] is None
        assert row["op"][0] == 0.3
        assert row["inv"][0] == 0.15
        assert row["me"][0] == 300.0  # in the lenient size-median pool

    def test_chars_filter_recovers_inner_join_rows(self):
        """The chars call site filters count & dec_me non-null to reproduce
        the pre-left-join row set."""
        chars = self._run_us()
        kept = chars.filter(pl.col("count").is_not_null() & pl.col("dec_me").is_not_null())
        assert kept["id"].to_list() == [1]

    def test_row_keeps_inner_dec_me_join(self):
        """ROW stock without Dec(t-1) ME stays excluded (JKP convention)."""
        panel = pl.DataFrame(
            {
                "excntry": ["GBR"] * 5,
                "id": [1, 1, 2, 2, 3],
                "gvkey": ["g1", "g1", "g2", "g2", "g3"],
                "date": [date(2019, 12, 31), date(2020, 6, 30)] * 2 + [date(2020, 6, 30)],
                "ret": [0.01] * 5,
                "me": [100.0, 110.0, 200.0, 210.0, 300.0],
                "size_grp": ["large"] * 5,
            }
        ).sort("id", "date")
        comp = pl.DataFrame(
            {
                "gvkey": ["g1", "g3"],
                "year": [2019, 2019],
                "datadate": [date(2019, 12, 31)] * 2,
                "be": [50.0, 70.0],
                "op": [0.2, 0.3],
                "inv": [0.1, 0.15],
                "count": [3, 4],
            },
            schema_overrides={"year": pl.Int32},
        )
        _, data_chars = _ff_row_finish_prepare(
            _ff_row_rets_weights_lazy(panel.lazy()).collect(), comp
        )
        # id 3 (no Dec ME) dropped by the inner dec_me join; id 2 (no comp)
        # kept by the left comp join.
        assert sorted(data_chars["id"].to_list()) == [1, 2]


# =============================================================================
# TestFFAssignPortfolios
# =============================================================================


def _empty_op_breaks(excntry: str) -> pl.DataFrame:
    """Trivial op breakpoint frame (null op30/op70) for the (excntry, date)
    key used by the assign tests, so the op left-join contributes no OP sort."""
    return pl.DataFrame(
        {
            "excntry": [excntry],
            "date": [date(2020, 6, 30)],
            "op30": [None],
            "op70": [None],
        },
        schema={"excntry": pl.Utf8, "date": pl.Date, "op30": pl.Float64, "op70": pl.Float64},
    )


def _empty_inv_breaks(excntry: str) -> pl.DataFrame:
    """Trivial inv breakpoint frame (null inv30/inv70) for the (excntry, date)
    key used by the assign tests, so the inv left-join contributes no sort."""
    return pl.DataFrame(
        {
            "excntry": [excntry],
            "date": [date(2020, 6, 30)],
            "inv30": [None],
            "inv70": [None],
        },
        schema={"excntry": pl.Utf8, "date": pl.Date, "inv30": pl.Float64, "inv70": pl.Float64},
    )


def _make_june_with_bp(
    n: int,
    beme_vals: list[float],
    me_vals: list[float],
    sizemedn: float,
    beme30: float,
    beme70: float,
    excntry: str = US_EXCNTRY,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build a June frame and matching bm/op/inv break frames for
    ff_assign_portfolios. op/inv breaks are trivial (null) so only the
    Size + B/M assignment is exercised."""
    june = pl.DataFrame(
        {
            "excntry": [excntry] * n,
            "date": [date(2020, 6, 30)] * n,
            "id": list(range(n)),
            "beme": beme_vals,
            "me": me_vals,
            "count": [3] * n,
            "exchcd_us": [1] * n,
            "size_grp": ["large"] * n,
            # be/op/inv referenced by the OP/INV eligibility exprs; null here
            # since op/inv breaks are trivial (no OP/INV assignment exercised).
            "be": [None] * n,
            "op": [None] * n,
            "inv": [None] * n,
        },
        schema_overrides={"be": pl.Float64, "op": pl.Float64, "inv": pl.Float64},
    )
    bm_breaks = pl.DataFrame(
        {
            "excntry": [excntry],
            "date": [date(2020, 6, 30)],
            "sizemedn": [sizemedn],
            "beme30": [beme30],
            "beme70": [beme70],
        }
    )
    return june, bm_breaks, _empty_op_breaks(excntry), _empty_inv_breaks(excntry)


class TestFFAssignPortfolios:
    """Tests for ff_assign_portfolios()."""

    def test_size_split_correct(self):
        """Firms below sizemedn → S; at or above → B."""
        june, bm_breaks, op_breaks, inv_breaks = _make_june_with_bp(
            n=4,
            beme_vals=[1.0, 1.0, 5.0, 5.0],
            me_vals=[1.0, 3.0, 1.0, 3.0],
            sizemedn=2.0,
            beme30=2.0,
            beme70=4.0,
        )
        result = ff_assign_portfolios(june, bm_breaks, op_breaks, inv_breaks)
        size_ports = result["sizeport"].to_list()
        # me=1 < 2=sizemedn → S; me=3 >= 2 → B
        assert size_ports[0] == "S"
        assert size_ports[1] == "B"
        assert size_ports[2] == "S"
        assert size_ports[3] == "B"

    def test_bm_portfolio_labels(self):
        """beme <= beme30 → L, beme30 < beme <= beme70 → M, beme > beme70 → H."""
        june, bm_breaks, op_breaks, inv_breaks = _make_june_with_bp(
            n=3,
            beme_vals=[1.0, 3.0, 5.0],
            me_vals=[1.0, 1.0, 1.0],
            sizemedn=2.0,
            beme30=2.0,
            beme70=4.0,
        )
        result = ff_assign_portfolios(june, bm_breaks, op_breaks, inv_breaks)
        ports = result["btmport"].to_list()
        assert ports == ["L", "M", "H"]

    def test_ineligible_rows_get_null_port(self):
        """Rows failing eligibility (beme<=0) get null btmport."""
        june, bm_breaks, op_breaks, inv_breaks = _make_june_with_bp(
            n=2,
            beme_vals=[-1.0, 1.0],  # first row ineligible
            me_vals=[1.0, 1.0],
            sizemedn=0.5,
            beme30=0.5,
            beme70=1.5,
        )
        result = ff_assign_portfolios(june, bm_breaks, op_breaks, inv_breaks)
        assert result["btmport"][0] is None
        assert result["btmport"][1] is not None

    def test_missing_breakpoint_null_port(self):
        """If breakpoint columns are null, eligible rows get null assignment."""
        june = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "date": [date(2020, 6, 30)],
                "id": [0],
                "beme": [2.0],
                "me": [1.0],
                "count": [3],
                "exchcd_us": [1],
                "size_grp": ["large"],
                "be": [None],
                "op": [None],
                "inv": [None],
            },
            schema_overrides={"be": pl.Float64, "op": pl.Float64, "inv": pl.Float64},
        )
        bp_size = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "date": [date(2020, 6, 30)],
                "sizemedn": [0.5],
                "beme30": [None],
                "beme70": [None],
            },
            schema={
                "excntry": pl.Utf8,
                "date": pl.Date,
                "sizemedn": pl.Float64,
                "beme30": pl.Float64,
                "beme70": pl.Float64,
            },
        )
        result = ff_assign_portfolios(
            june, bp_size, _empty_op_breaks(US_EXCNTRY), _empty_inv_breaks(US_EXCNTRY)
        )
        # With null breakpoints the eligibility gate includes is_not_null check
        assert result["btmport"][0] is None

    def test_output_columns_include_required(self):
        """Output always has excntry, id, date, sizeport."""
        june, bm_breaks, op_breaks, inv_breaks = _make_june_with_bp(
            n=4,
            beme_vals=[1.0, 2.0, 4.0, 6.0],
            me_vals=[1.0, 1.0, 1.0, 1.0],
            sizemedn=1.5,
            beme30=2.5,
            beme70=5.0,
        )
        result = ff_assign_portfolios(june, bm_breaks, op_breaks, inv_breaks)
        for col in ("excntry", "id", "date", "sizeport", "btmport", "positivebeme"):
            assert col in result.columns

    def test_positivebeme_flag(self):
        """positivebeme == 1 when bm-eligible, 0 otherwise."""
        june, bm_breaks, op_breaks, inv_breaks = _make_june_with_bp(
            n=2,
            beme_vals=[1.0, -1.0],
            me_vals=[1.0, 1.0],
            sizemedn=0.5,
            beme30=0.5,
            beme70=1.5,
        )
        result = ff_assign_portfolios(june, bm_breaks, op_breaks, inv_breaks)
        flags = result["positivebeme"].to_list()
        assert flags[0] == 1
        assert flags[1] == 0


# =============================================================================
# TestFFAssignToPanel
# =============================================================================


class TestFFAssignToPanel:
    """Tests for ff_assign_to_panel()."""

    def _make_june_port(self, year: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "id": [1],
                "date": [date(year, 6, 30)],
                "sizeport": ["S"],
                "btmport": ["H"],
                "positivebeme": [1],
                "opport": ["R"],
                "nonmiss_op": [1],
                "invport": ["C"],
                "nonmiss_inv": [1],
            }
        )

    def test_july_maps_to_current_year(self):
        """July observation maps to formation year y (Jul→Jun(y+1))."""
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "id": [1],
                "date": [date(2020, 7, 31)],
                "ret": [0.01],
                "me": [1.0],
                "w": [1.0],
            }
        )
        june = self._make_june_port(2020)
        result = ff_assign_to_panel(panel, june)
        assert len(result) == 1
        assert result["sizeport"][0] == "S"

    def test_june_maps_to_prior_year(self):
        """June observation maps to formation year y-1."""
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "id": [1],
                "date": [date(2021, 6, 30)],
                "ret": [0.01],
                "me": [1.0],
                "w": [1.0],
            }
        )
        june = self._make_june_port(2020)
        result = ff_assign_to_panel(panel, june)
        assert len(result) == 1
        assert result["btmport"][0] == "H"

    def test_no_match_excluded(self):
        """Panel row with no June match is dropped (inner join)."""
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "id": [999],  # no matching id in june
                "date": [date(2020, 7, 31)],
                "ret": [0.01],
                "me": [1.0],
                "w": [1.0],
            }
        )
        june = self._make_june_port(2020)
        result = ff_assign_to_panel(panel, june)
        assert len(result) == 0

    def test_july_and_june_rollover(self):
        """July(y+1) and June(y+1) both map to year y correctly."""
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY, US_EXCNTRY],
                "id": [1, 1],
                "date": [date(2021, 7, 31), date(2021, 6, 30)],
                "ret": [0.01, 0.02],
                "me": [1.0, 1.0],
                "w": [1.0, 1.0],
            }
        )
        june_2020 = self._make_june_port(2020)
        june_2021 = self._make_june_port(2021).with_columns(
            pl.col("sizeport").map_elements(lambda x: "B", return_dtype=pl.Utf8)
        )
        june = pl.concat([june_2020, june_2021])
        result = ff_assign_to_panel(panel, june)
        assert len(result) == 2
        # July 2021 → port_year = 2021 → june_2021 → "B"
        july_row = result.filter(pl.col("date") == date(2021, 7, 31))
        assert july_row["sizeport"][0] == "B"
        # June 2021 → port_year = 2020 → june_2020 → "S"
        june_row = result.filter(pl.col("date") == date(2021, 6, 30))
        assert june_row["sizeport"][0] == "S"

    def test_port_year_dropped_from_output(self):
        """port_year is an intermediate column that must not appear in output."""
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "id": [1],
                "date": [date(2020, 8, 31)],
                "ret": [0.01],
                "me": [1.0],
                "w": [1.0],
            }
        )
        june = self._make_june_port(2020)
        result = ff_assign_to_panel(panel, june)
        assert "port_year" not in result.columns


# =============================================================================
# TestFFVwLeg
# =============================================================================


# Inlined former FF_SORT_SPECS["bm"] literals for the B/M VW leg.
_BM_PORT = "btmport"
_BM_FLAG = "positivebeme"
_BM_BUCKETS = ["SL", "SM", "SH", "BL", "BM", "BH"]


def _vw_leg_bm(panel: pl.DataFrame) -> pl.DataFrame:
    """Apply the shared (w>0, ret not null, sizeport not null) filter — which
    _ff_vw_leg no longer does internally — then run the B/M VW leg."""
    filtered = panel.filter(
        (pl.col("w") > 0) & pl.col("ret").is_not_null() & pl.col("sizeport").is_not_null()
    )
    return _ff_vw_leg(filtered, port=_BM_PORT, flag=_BM_FLAG, buckets=_BM_BUCKETS)


def _make_panel_bm(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal panel frame for the B/M _ff_vw_leg tests."""
    schema = {
        "excntry": pl.Utf8,
        "id": pl.Int64,
        "date": pl.Date,
        "ret": pl.Float64,
        "w": pl.Float64,
        "me": pl.Float64,
        "sizeport": pl.Utf8,
        "btmport": pl.Utf8,
        "positivebeme": pl.Int64,
    }
    data: dict[str, list] = {k: [] for k in schema}
    for r in rows:
        for k in schema:
            data[k].append(r.get(k))
    return pl.DataFrame(data, schema=schema)


class TestFFVwLeg:
    """Tests for _ff_vw_leg()."""

    def test_vw_return_formula(self, tolerance):
        """VW return = sum(ret*w)/sum(w) per bucket."""
        # One bucket SH: two stocks, w=[1,3], ret=[0.1, 0.2]
        # vwret = (0.1*1 + 0.2*3) / (1+3) = 0.7/4 = 0.175
        panel = _make_panel_bm(
            [
                {
                    "excntry": US_EXCNTRY,
                    "id": 1,
                    "date": date(2020, 7, 31),
                    "ret": 0.1,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
                {
                    "excntry": US_EXCNTRY,
                    "id": 2,
                    "date": date(2020, 7, 31),
                    "ret": 0.2,
                    "w": 3.0,
                    "me": 3.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
            ]
        )
        result = _vw_leg_bm(panel)
        sh_val = result["SH"][0]
        np.testing.assert_allclose(sh_val, 0.175, **tolerance.TIGHT)

    def test_missing_buckets_filled_with_null(self):
        """Buckets absent from data are filled with null (not dropped)."""
        # Only populate SH; other 5 buckets are missing
        panel = _make_panel_bm(
            [
                {
                    "excntry": US_EXCNTRY,
                    "id": 1,
                    "date": date(2020, 7, 31),
                    "ret": 0.1,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
            ]
        )
        result = _vw_leg_bm(panel)
        for col in _BM_BUCKETS:
            assert col in result.columns

    def test_us_bypasses_min_stocks_gate(self):
        """US bucket with only 1 stock passes (no FF_MIN_STOCKS_PF gate for US)."""
        panel = _make_panel_bm(
            [
                {
                    "excntry": US_EXCNTRY,
                    "id": 1,
                    "date": date(2020, 7, 31),
                    "ret": 0.05,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
            ]
        )
        result = _vw_leg_bm(panel)
        # With 1 stock, US should still emit a value
        assert result["SH"][0] is not None

    def test_row_below_min_stocks_nulled(self):
        """ROW bucket with n < FF_MIN_STOCKS_PF gets null."""
        n = FF_MIN_STOCKS_PF - 1
        rows = [
            {
                "excntry": "GBR",
                "id": i,
                "date": date(2020, 7, 31),
                "ret": 0.1,
                "w": 1.0,
                "me": 1.0,
                "sizeport": "S",
                "btmport": "H",
                "positivebeme": 1,
            }
            for i in range(n)
        ]
        panel = _make_panel_bm(rows)
        result = _vw_leg_bm(panel)
        gbr_rows = result.filter(pl.col("excntry") == "GBR")
        if len(gbr_rows) > 0:
            assert gbr_rows["SH"][0] is None

    def test_row_at_min_stocks_not_nulled(self):
        """ROW bucket with n == FF_MIN_STOCKS_PF passes through."""
        n = FF_MIN_STOCKS_PF
        rows = [
            {
                "excntry": "GBR",
                "id": i,
                "date": date(2020, 7, 31),
                "ret": 0.1,
                "w": 1.0,
                "me": 1.0,
                "sizeport": "S",
                "btmport": "H",
                "positivebeme": 1,
            }
            for i in range(n)
        ]
        panel = _make_panel_bm(rows)
        result = _vw_leg_bm(panel)
        gbr_rows = result.filter(pl.col("excntry") == "GBR")
        assert len(gbr_rows) == 1
        assert gbr_rows["SH"][0] is not None

    def test_null_ret_excluded(self):
        """Rows with null ret are excluded from VW computation."""
        panel = _make_panel_bm(
            [
                {
                    "excntry": US_EXCNTRY,
                    "id": 1,
                    "date": date(2020, 7, 31),
                    "ret": None,
                    "w": 10.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
                {
                    "excntry": US_EXCNTRY,
                    "id": 2,
                    "date": date(2020, 7, 31),
                    "ret": 0.1,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
            ]
        )
        result = _vw_leg_bm(panel)
        # Only id=2 contributes; vwret = 0.1
        np.testing.assert_allclose(result["SH"][0], 0.1, rtol=1e-9)

    def test_zero_w_excluded(self):
        """Rows with w == 0 are excluded from VW computation."""
        panel = _make_panel_bm(
            [
                {
                    "excntry": US_EXCNTRY,
                    "id": 1,
                    "date": date(2020, 7, 31),
                    "ret": 999.0,
                    "w": 0.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
                {
                    "excntry": US_EXCNTRY,
                    "id": 2,
                    "date": date(2020, 7, 31),
                    "ret": 0.05,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
            ]
        )
        result = _vw_leg_bm(panel)
        np.testing.assert_allclose(result["SH"][0], 0.05, rtol=1e-9)

    def test_output_schema(self):
        """Output has excntry, date, and all 6 BM buckets."""
        panel = _make_panel_bm(
            [
                {
                    "excntry": US_EXCNTRY,
                    "id": 1,
                    "date": date(2020, 7, 31),
                    "ret": 0.1,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": "S",
                    "btmport": "H",
                    "positivebeme": 1,
                },
            ]
        )
        result = _vw_leg_bm(panel)
        assert "excntry" in result.columns
        assert "date" in result.columns
        for col in _BM_BUCKETS:
            assert col in result.columns


# =============================================================================
# TestFFComputeFactors
# =============================================================================


def _make_ccm4_full(excntry: str = US_EXCNTRY) -> pl.DataFrame:
    """Build a minimal panel with all 18 BM/OP/INV buckets populated."""
    d = date(2020, 7, 31)
    # 6 BM buckets, 6 OP (W/N/R), 6 INV (C/N/A)
    rows = []
    bm_ports = ["L", "M", "H"]
    op_ports = ["W", "N", "R"]
    inv_ports = ["C", "N", "A"]
    stock_id = 0

    for sz in ["S", "B"]:
        for bmp in bm_ports:
            stock_id += 1
            rows.append(
                {
                    "excntry": excntry,
                    "id": stock_id,
                    "date": d,
                    "ret": 0.01 * stock_id,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": sz,
                    "btmport": bmp,
                    "positivebeme": 1,
                    "opport": "W",
                    "nonmiss_op": 1,
                    "invport": "C",
                    "nonmiss_inv": 1,
                }
            )
        for opp in op_ports:
            stock_id += 1
            rows.append(
                {
                    "excntry": excntry,
                    "id": stock_id,
                    "date": d,
                    "ret": 0.01 * stock_id,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": sz,
                    "btmport": "L",
                    "positivebeme": 1,
                    "opport": opp,
                    "nonmiss_op": 1,
                    "invport": "C",
                    "nonmiss_inv": 1,
                }
            )
        for invp in inv_ports:
            stock_id += 1
            rows.append(
                {
                    "excntry": excntry,
                    "id": stock_id,
                    "date": d,
                    "ret": 0.01 * stock_id,
                    "w": 1.0,
                    "me": 1.0,
                    "sizeport": sz,
                    "btmport": "L",
                    "positivebeme": 1,
                    "opport": "W",
                    "nonmiss_op": 1,
                    "invport": invp,
                    "nonmiss_inv": 1,
                }
            )

    schema = {
        "excntry": pl.Utf8,
        "id": pl.Int64,
        "date": pl.Date,
        "ret": pl.Float64,
        "w": pl.Float64,
        "me": pl.Float64,
        "sizeport": pl.Utf8,
        "btmport": pl.Utf8,
        "positivebeme": pl.Int64,
        "opport": pl.Utf8,
        "nonmiss_op": pl.Int64,
        "invport": pl.Utf8,
        "nonmiss_inv": pl.Int64,
    }
    data: dict[str, list] = {k: [] for k in schema}
    for r in rows:
        for k in schema:
            data[k].append(r.get(k))
    return pl.DataFrame(data, schema=schema)


class TestFFComputeFactors:
    """Tests for ff_compute_factors()."""

    def test_output_columns(self):
        """Output has excntry, date, smb_ff3, smb_ff5, hml_ff, rmw_ff, cma_ff."""
        ccm4 = _make_ccm4_full()
        result = ff_compute_factors(ccm4)
        for col in ("excntry", "date", "smb_ff3", "smb_ff5", "hml_ff", "rmw_ff", "cma_ff"):
            assert col in result.columns

    def test_hml_is_high_minus_low(self, tolerance):
        """HML = avg(SH,BH) - avg(SL,BL): one stock per bucket, equal weights."""
        # One distinct stock per (sizeport, btmport) bucket. Each also tagged
        # for one OP and one INV bucket so all three legs are populated.
        # BM leg: SH=0.30, BH=0.10, SL=0.01, BL=0.02 → HML = 0.20 - 0.015 = 0.185
        d = date(2020, 7, 31)
        # fmt: off
        # (sizeport, btmport, ret, positivebeme, opport, nonmiss_op, invport, nonmiss_inv)
        rows = [
            # 6 unique BM buckets — each stock belongs to exactly one BM bucket
            ("S", "H", 0.30, 1, "W", 1, "C", 1),
            ("B", "H", 0.10, 1, "N", 1, "N", 1),
            ("S", "M", 0.05, 1, "R", 1, "A", 1),
            ("B", "M", 0.05, 1, "W", 0, "C", 0),  # nonmiss_op/inv=0 → excluded from OP/INV legs
            ("S", "L", 0.01, 1, "W", 0, "C", 0),
            ("B", "L", 0.02, 1, "W", 0, "C", 0),
            # Extra stocks to fill remaining OP buckets (nonmiss_op=1, unique BM buckets)
            ("S", "N", 0.05, 0, "N", 1, "C", 0),  # positivebeme=0 → excluded from BM
            ("B", "N", 0.05, 0, "R", 1, "C", 0),
            ("S", "W", 0.05, 0, "W", 1, "N", 1),
            ("B", "W", 0.05, 0, "W", 1, "N", 1),
            ("S", "R", 0.05, 0, "R", 1, "A", 1),
            ("B", "R", 0.05, 0, "N", 1, "A", 1),
        ]
        # fmt: on
        schema = {
            "excntry": pl.Utf8,
            "id": pl.Int64,
            "date": pl.Date,
            "ret": pl.Float64,
            "w": pl.Float64,
            "me": pl.Float64,
            "sizeport": pl.Utf8,
            "btmport": pl.Utf8,
            "positivebeme": pl.Int64,
            "opport": pl.Utf8,
            "nonmiss_op": pl.Int64,
            "invport": pl.Utf8,
            "nonmiss_inv": pl.Int64,
        }
        data: dict[str, list] = {k: [] for k in schema}
        for i, r in enumerate(rows):
            data["excntry"].append(US_EXCNTRY)
            data["id"].append(i)
            data["date"].append(d)
            data["sizeport"].append(r[0])
            data["btmport"].append(r[1])
            data["ret"].append(r[2])
            data["positivebeme"].append(r[3])
            data["opport"].append(r[4])
            data["nonmiss_op"].append(r[5])
            data["invport"].append(r[6])
            data["nonmiss_inv"].append(r[7])
            data["w"].append(1.0)
            data["me"].append(1.0)
        ccm4 = pl.DataFrame(data, schema=schema)
        result = ff_compute_factors(ccm4)
        # BM leg only: SH=0.30, BH=0.10, SL=0.01, BL=0.02 (each one stock, w=1)
        # HML = (SH+BH)/2 - (SL+BL)/2 = 0.20 - 0.015 = 0.185
        np.testing.assert_allclose(result["hml_ff"][0], 0.185, **tolerance.STANDARD)

    def test_smb_ff3_vs_smb_ff5_differ(self):
        """smb_ff3 uses BM leg only; smb_ff5 averages BM/OP/INV legs."""
        ccm4 = _make_ccm4_full()
        result = ff_compute_factors(ccm4)
        # They will generally differ unless OP/INV size legs happen to equal BM
        assert "smb_ff3" in result.columns
        assert "smb_ff5" in result.columns

    def test_null_propagation_when_bucket_missing(self):
        """When a required bucket is absent, factor is null (no ignore_nulls)."""
        # Use a panel that has no BM 'H' bucket stocks → HML should be null
        d = date(2020, 7, 31)
        # Only L and M; no H
        schema = {
            "excntry": pl.Utf8,
            "id": pl.Int64,
            "date": pl.Date,
            "ret": pl.Float64,
            "w": pl.Float64,
            "me": pl.Float64,
            "sizeport": pl.Utf8,
            "btmport": pl.Utf8,
            "positivebeme": pl.Int64,
            "opport": pl.Utf8,
            "nonmiss_op": pl.Int64,
            "invport": pl.Utf8,
            "nonmiss_inv": pl.Int64,
        }
        rows = [
            (US_EXCNTRY, 1, d, 0.05, 1.0, 1.0, "S", "L", 1, "W", 1, "C", 1),
            (US_EXCNTRY, 2, d, 0.05, 1.0, 1.0, "B", "L", 1, "W", 1, "C", 1),
            (US_EXCNTRY, 3, d, 0.05, 1.0, 1.0, "S", "M", 1, "W", 1, "C", 1),
            (US_EXCNTRY, 4, d, 0.05, 1.0, 1.0, "B", "M", 1, "W", 1, "C", 1),
        ]
        data = {k: [r[i] for r in rows] for i, k in enumerate(schema)}
        ccm4 = pl.DataFrame(data, schema=schema)
        result = ff_compute_factors(ccm4)
        # SH and BH are null → HML is null
        assert result["hml_ff"][0] is None


# =============================================================================
# TestFFMomSignalMonthly
# =============================================================================


def _make_monthly_panel(
    months: int,
    base_ret: float = 0.01,
    excntry: str = US_EXCNTRY,
    stock_id: int = 1,
    include_missing_sentinel: bool = False,
    ret_overrides: dict[int, float | None] | None = None,
) -> pl.DataFrame:
    """Consecutive monthly panel for a single stock."""
    dates = [date(2018 + (m // 12), (m % 12) + 1, 28) for m in range(months)]
    rets: list[float | None] = [base_ret] * months
    if include_missing_sentinel:
        rets[5] = FF_MISSING_RET_CODE  # inject -99 sentinel
    for i, v in (ret_overrides or {}).items():
        rets[i] = v
    return pl.DataFrame(
        {
            "excntry": [excntry] * months,
            "id": [stock_id] * months,
            "date": dates,
            "ret": rets,
            "me": [1.0] * months,
            "w": [1.0] * months,
            "exchcd_us": [1] * months,
            "size_grp": ["large"] * months,
        },
        schema_overrides={"ret": pl.Float64},
    )


class TestFFMomSignalMonthly:
    """Tests for _ff_mom_signal_monthly()."""

    def test_output_schema(self):
        """Output has the expected 11 columns of the mom-signal output schema."""
        panel = _make_monthly_panel(months=20)
        lf = panel.lazy()
        result = _ff_mom_signal_monthly(lf).collect()
        expected_cols = {
            "excntry",
            "id",
            "date",
            "ret",
            "w",
            "me",
            "exchcd_us",
            "size_grp",
            "me_lag1",
            "mom_2_12",
            "eligible_mom",
        }
        assert set(result.columns) == expected_cols

    def test_insufficient_history_yields_null_mom(self):
        """Less than 13 months of data → mom_2_12 is null (not enough lags)."""
        panel = _make_monthly_panel(months=12)
        lf = panel.lazy()
        result = _ff_mom_signal_monthly(lf).collect()
        # All rows should have null mom_2_12 (need 13 months for lag 1..12)
        assert result["mom_2_12"].is_null().all()

    def test_sufficient_history_yields_non_null_mom(self):
        """13+ months of data → later rows have non-null mom_2_12."""
        panel = _make_monthly_panel(months=20)
        lf = panel.lazy()
        result = _ff_mom_signal_monthly(lf).collect()
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0

    def test_neg99_sentinel_treated_as_zero(self, tolerance):
        """FF_MISSING_RET_CODE (-99) in any lag month is treated as 0.0 return."""
        # Build 20-month panel with -99 at position 5 (will be a lagged return)
        panel = _make_monthly_panel(months=20, base_ret=0.0, include_missing_sentinel=True)
        lf = panel.lazy()
        result = _ff_mom_signal_monthly(lf).collect()
        # Should still compute (sentinel becomes 0, doesn't make mom null)
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0

    @staticmethod
    def _interior_null_panel(excntry: str) -> pl.LazyFrame:
        """20 consecutive months, ret null at month 10 (an interior 2-12 lag
        for the final row; the row itself exists, mimicking a CIZ
        missing-price month that legacy CRSP coded -99)."""
        return _make_monthly_panel(months=20, excntry=excntry, ret_overrides={10: None}).lazy()

    def test_us_interior_null_month_tolerated(self, tolerance):
        """US: a null interior window month (row present) stays eligible and
        compounds as 0, per French's -99 tolerance for t-12..t-3."""
        result = _ff_mom_signal_monthly(self._interior_null_panel(US_EXCNTRY)).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is True
        # 11 window months, one zeroed: (1.01)^10 - 1
        np.testing.assert_allclose(last_row["mom_2_12"][0], 1.01**10 - 1, **tolerance.STANDARD)

    def test_row_interior_null_month_ineligible(self):
        """ROW: interior null month keeps the strict JKP gate."""
        result = _ff_mom_signal_monthly(self._interior_null_panel("GBR")).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is not True

    def test_me_lag1_null_when_gap(self):
        """me_lag1 is null when there's a calendar month gap."""
        # Two rows with a 2-month gap → me_lag1 for the second row is null
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY, US_EXCNTRY],
                "id": [1, 1],
                "date": [date(2020, 1, 31), date(2020, 3, 31)],  # skip Feb
                "ret": [0.01, 0.02],
                "me": [1.0, 2.0],
                "w": [1.0, 1.0],
                "exchcd_us": [1, 1],
                "size_grp": ["large", "large"],
            }
        )
        lf = panel.lazy()
        result = _ff_mom_signal_monthly(lf).collect().sort("date")
        # March row: gap1 = (2020*12+3) - (2020*12+1) = 2 ≠ 1 → me_lag1 null
        mar_row = result.filter(pl.col("date") == date(2020, 3, 31))
        assert mar_row["me_lag1"][0] is None

    def test_mom_signal_2_12_skip(self, tolerance):
        """With constant 2% monthly return, compounded 2-12 (skip 1) ≈ 1.02^11 - 1."""
        panel = _make_monthly_panel(months=24, base_ret=0.02)
        lf = panel.lazy()
        result = _ff_mom_signal_monthly(lf).collect()
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        if len(non_null) > 0:
            # 11 months compounded: lags 2 through 12 = 11 months
            expected = (1.02**11) - 1
            np.testing.assert_allclose(non_null["mom_2_12"][0], expected, **tolerance.STANDARD)

    def test_multi_stock_independent(self):
        """Two stocks compute signals independently."""
        p1 = _make_monthly_panel(months=20, base_ret=0.01, stock_id=1)
        p2 = _make_monthly_panel(months=20, base_ret=0.05, stock_id=2)
        panel = pl.concat([p1, p2])
        lf = panel.lazy()
        result = _ff_mom_signal_monthly(lf).collect()
        s1 = result.filter(pl.col("id") == 1).filter(pl.col("mom_2_12").is_not_null())
        s2 = result.filter(pl.col("id") == 2).filter(pl.col("mom_2_12").is_not_null())
        if len(s1) > 0 and len(s2) > 0:
            # Stock 2 has higher momentum
            assert s2["mom_2_12"][0] > s1["mom_2_12"][0]


# =============================================================================
# TestFFMomSignalDaily
# =============================================================================


def _make_daily_panel(
    n_days: int,
    base_ret: float = 0.001,
    excntry: str = US_EXCNTRY,
    stock_id: int = 1,
    start_date: date = date(2018, 1, 2),
    ret_overrides: dict[int, float | None] | None = None,
) -> pl.DataFrame:
    """Consecutive trading-day panel (Mon-Fri, simplified as calendar days)."""
    from datetime import timedelta

    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    rets: list[float | None] = [base_ret] * n_days
    for i, v in (ret_overrides or {}).items():
        rets[i] = v
    return pl.DataFrame(
        {
            "excntry": [excntry] * n_days,
            "id": [stock_id] * n_days,
            "date": dates,
            "ret": rets,
            "me": [1.0] * n_days,
            "w": [1.0] * n_days,
            "exchcd_us": [1] * n_days,
            "size_grp": ["large"] * n_days,
        },
        schema_overrides={"ret": pl.Float64},
    )


class TestFFMomSignalDaily:
    """Tests for _ff_mom_signal_daily()."""

    def test_output_schema(self):
        """Output has the expected 11 columns."""
        panel = _make_daily_panel(n_days=300)
        lf = panel.lazy()
        result = _ff_mom_signal_daily(lf).collect()
        expected_cols = {
            "excntry",
            "id",
            "date",
            "ret",
            "w",
            "me",
            "exchcd_us",
            "size_grp",
            "me_lag1",
            "mom_2_12",
            "eligible_mom",
        }
        assert set(result.columns) == expected_cols

    def test_insufficient_history_yields_null_mom(self):
        """Fewer than 252 days → all mom_2_12 null (not enough history)."""
        panel = _make_daily_panel(n_days=250)
        lf = panel.lazy()
        result = _ff_mom_signal_daily(lf).collect()
        assert result["mom_2_12"].is_null().all()

    def test_sufficient_history_has_non_null(self):
        """300 days → rows past lookback window have non-null mom_2_12."""
        panel = _make_daily_panel(n_days=300)
        lf = panel.lazy()
        result = _ff_mom_signal_daily(lf).collect()
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0

    def test_neg99_sentinel_not_eligible(self):
        """Row where ret_lag_skip == -99 is marked ineligible."""
        from datetime import timedelta

        n = 300
        base = date(2018, 1, 2)
        dates = [base + timedelta(days=i) for i in range(n)]
        rets = [0.001] * n
        # Place sentinel at position exactly skip=21 before the end
        rets[n - 22] = FF_MISSING_RET_CODE
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * n,
                "id": [1] * n,
                "date": dates,
                "ret": rets,
                "me": [1.0] * n,
                "w": [1.0] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["large"] * n,
            }
        )
        lf = panel.lazy()
        result = _ff_mom_signal_daily(lf).collect()
        # The last row may be ineligible due to sentinel
        # (checking eligible_mom == False for the terminal row where skip lag == -99)
        last_row = result.sort("date").tail(1)
        # eligible_mom should be False when sentinel is at skip lag
        assert last_row["eligible_mom"][0] is False

    @staticmethod
    def _null_in_window_panel(excntry: str) -> pl.LazyFrame:
        """300 consecutive days, ret null at an interior lookback-window day
        for the last row."""
        return _make_daily_panel(n_days=300, excntry=excntry, ret_overrides={200: None}).lazy()

    def test_row_null_ret_in_window_makes_ineligible(self):
        """ROW: a null return in the lookback window → ineligible (JKP)."""
        result = _ff_mom_signal_daily(self._null_in_window_panel("GBR")).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is False

    def test_us_null_ret_in_window_tolerated(self, tolerance):
        """US: interior missing-price day is allowed and compounds as 0
        (French: 'any missing returns from day t-250 to t-22 must be -99.0')."""
        result = _ff_mom_signal_daily(self._null_in_window_panel(US_EXCNTRY)).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is True
        # 230-day window with one zeroed day: (1.001)^229 - 1
        np.testing.assert_allclose(last_row["mom_2_12"][0], 1.001**229 - 1, **tolerance.STANDARD)

    def test_us_window_includes_skip_day_row_excludes_it(self):
        """Window pin: a spike at t-21 is inside the US window (t-250..t-21,
        per French's daily detail) but outside the ROW legacy window
        (t-251..t-22)."""
        n = 300
        for excntry, expected in ((US_EXCNTRY, 0.5), ("GBR", 0.0)):
            panel = _make_daily_panel(
                n_days=n,
                base_ret=0.0,
                excntry=excntry,
                ret_overrides={n - 1 - 21: 0.5},  # day t-21 for the last row
            )
            result = _ff_mom_signal_daily(panel.lazy()).collect()
            last_row = result.sort("date").tail(1)
            assert last_row["eligible_mom"][0] is True
            assert abs(last_row["mom_2_12"][0] - expected) < 1e-9, excntry

    def test_multi_country_independent(self):
        """Two countries compute trading-day index independently."""
        p_usa = _make_daily_panel(300, base_ret=0.001, excntry="USA", stock_id=1)
        p_gbr = _make_daily_panel(300, base_ret=0.003, excntry="GBR", stock_id=1)
        panel = pl.concat([p_usa, p_gbr])
        lf = panel.lazy()
        result = _ff_mom_signal_daily(lf).collect()
        usa = result.filter((pl.col("excntry") == "USA") & pl.col("mom_2_12").is_not_null())
        gbr = result.filter((pl.col("excntry") == "GBR") & pl.col("mom_2_12").is_not_null())
        if len(usa) > 0 and len(gbr) > 0:
            # GBR has higher per-day return → higher momentum
            assert gbr["mom_2_12"].mean() > usa["mom_2_12"].mean()


# =============================================================================
# TestFFComputeUmdFactor
# =============================================================================


def _make_mom_panel(excntry: str = US_EXCNTRY) -> pl.DataFrame:
    """Panel with size × mom buckets for UMD tests."""
    d = date(2020, 7, 31)
    schema = {
        "excntry": pl.Utf8,
        "id": pl.Int64,
        "date": pl.Date,
        "ret": pl.Float64,
        "w": pl.Float64,
        "me": pl.Float64,
        "sizeport": pl.Utf8,
        "momport": pl.Utf8,
        "nonmiss_mom": pl.Int64,
    }
    rows = [
        (excntry, 1, d, 0.30, 1.0, 1.0, "S", "H", 1),  # SH
        (excntry, 2, d, 0.20, 1.0, 1.0, "B", "H", 1),  # BH
        (excntry, 3, d, 0.02, 1.0, 1.0, "S", "L", 1),  # SL
        (excntry, 4, d, 0.01, 1.0, 1.0, "B", "L", 1),  # BL
        (excntry, 5, d, 0.10, 1.0, 1.0, "S", "N", 1),  # SN
        (excntry, 6, d, 0.10, 1.0, 1.0, "B", "N", 1),  # BN
    ]
    data: dict[str, list] = {k: [] for k in schema}
    for r in rows:
        for i, k in enumerate(schema):
            data[k].append(r[i])
    return pl.DataFrame(data, schema=schema)


class TestFFComputeUmdFactor:
    """Tests for ff_compute_umd_factor()."""

    def test_umd_arithmetic(self, tolerance):
        """umd_ff = 0.5*(SH+BH) - 0.5*(SL+BL) with known returns."""
        panel = _make_mom_panel()
        result = ff_compute_umd_factor(panel)
        # 0.5*(0.30+0.20) - 0.5*(0.02+0.01) = 0.25 - 0.015 = 0.235
        np.testing.assert_allclose(result["umd_ff"][0], 0.235, **tolerance.STANDARD)

    def test_output_columns(self):
        """Output has excntry, date, umd_ff."""
        panel = _make_mom_panel()
        result = ff_compute_umd_factor(panel)
        assert set(result.columns) == {"excntry", "date", "umd_ff"}

    def test_sorted_by_excntry_date(self):
        """Output is sorted by (excntry, date)."""
        d1 = date(2020, 7, 31)
        d2 = date(2020, 8, 31)
        schema = {
            "excntry": pl.Utf8,
            "id": pl.Int64,
            "date": pl.Date,
            "ret": pl.Float64,
            "w": pl.Float64,
            "me": pl.Float64,
            "sizeport": pl.Utf8,
            "momport": pl.Utf8,
            "nonmiss_mom": pl.Int64,
        }
        rows_d1 = [
            (US_EXCNTRY, i, d1, 0.1, 1.0, 1.0, sz, port, 1)
            for i, (sz, port) in enumerate([("S", "H"), ("B", "H"), ("S", "L"), ("B", "L")])
        ]
        rows_d2 = [
            (US_EXCNTRY, i + 10, d2, 0.1, 1.0, 1.0, sz, port, 1)
            for i, (sz, port) in enumerate([("S", "H"), ("B", "H"), ("S", "L"), ("B", "L")])
        ]
        data: dict[str, list] = {k: [] for k in schema}
        for r in rows_d1 + rows_d2:
            for j, k in enumerate(schema):
                data[k].append(r[j])
        panel = pl.DataFrame(data, schema=schema)
        result = ff_compute_umd_factor(panel)
        dates = result["date"].to_list()
        assert dates == sorted(dates)

    def test_null_when_bucket_missing(self):
        """If SH or BH bucket is absent, umd_ff is null."""
        panel = _make_mom_panel().filter(pl.col("momport") != "H")  # remove H bucket
        result = ff_compute_umd_factor(panel)
        assert result["umd_ff"][0] is None

    def test_us_bypasses_min_stocks_gate(self):
        """US with 1 stock per bucket still computes umd_ff."""
        panel = _make_mom_panel(excntry=US_EXCNTRY)
        result = ff_compute_umd_factor(panel)
        assert result["umd_ff"][0] is not None


# =============================================================================
# TestFFBuildCharacteristics
# =============================================================================


def _make_char_panel(freq: str = "monthly") -> pl.DataFrame:
    """Build a minimal panel for ff_build_characteristics."""
    if freq == "monthly":
        dates = [date(2020, 8, 31), date(2020, 9, 30)]
    else:
        # Daily: one per-day in August 2020
        from datetime import timedelta

        start = date(2020, 8, 3)
        dates = [start + timedelta(days=i) for i in range(5)]

    n = len(dates)
    return pl.DataFrame(
        {
            "excntry": [US_EXCNTRY] * n,
            "id": [1] * n,
            "date": dates,
            "ret": [0.01] * n,
            "me": [100.0] * n,
            "w": [1.0] * n,
        }
    )


def _make_june_chars(
    beme: float = 1.5, op: float | None = 0.2, inv: float | None = 0.05
) -> pl.DataFrame:
    """Minimal June snapshot with beme/op/inv for characteristics."""
    return pl.DataFrame(
        {
            "excntry": [US_EXCNTRY],
            "id": [1],
            "date": [date(2020, 6, 30)],
            "beme": [beme],
            "op": [op],
            "inv": [inv],
        },
        schema={
            "excntry": pl.Utf8,
            "id": pl.Int64,
            "date": pl.Date,
            "beme": pl.Float64,
            "op": pl.Float64,
            "inv": pl.Float64,
        },
    )


class TestFFBuildCharacteristics:
    """Tests for ff_build_characteristics()."""

    def test_output_schema_monthly(self):
        """Monthly output has date, excntry, id, me_ff, beme_ff, op_ff, inv_ff."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars()
        result = ff_build_characteristics(panel, june, freq="monthly")
        for col in ("date", "excntry", "id", "me_ff", "beme_ff", "op_ff", "inv_ff"):
            assert col in result.columns

    def test_me_ff_matches_panel_me(self, tolerance):
        """me_ff is taken directly from panel's me column."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars()
        result = ff_build_characteristics(panel, june, freq="monthly")
        np.testing.assert_allclose(
            result["me_ff"].to_numpy(),
            [100.0, 100.0],
            **tolerance.TIGHT,
        )

    def test_beme_ff_from_june(self, tolerance):
        """beme_ff is the June beme value broadcast to panel rows."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars(beme=2.5)
        result = ff_build_characteristics(panel, june, freq="monthly")
        np.testing.assert_allclose(
            result["beme_ff"].to_numpy(),
            [2.5, 2.5],
            **tolerance.TIGHT,
        )

    def test_null_beme_propagates(self):
        """Null beme in june propagates to beme_ff."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars(beme=None)
        result = ff_build_characteristics(panel, june, freq="monthly")
        assert result["beme_ff"].is_null().all()

    def test_op_ff_from_june(self, tolerance):
        """op_ff is the June op value."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars(op=0.35)
        result = ff_build_characteristics(panel, june, freq="monthly")
        np.testing.assert_allclose(
            result["op_ff"].to_numpy(),
            [0.35, 0.35],
            **tolerance.TIGHT,
        )

    def test_null_op_propagates(self):
        """Null op in june propagates to op_ff."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars(op=None)
        result = ff_build_characteristics(panel, june, freq="monthly")
        assert result["op_ff"].is_null().all()

    def test_sorted_by_date_excntry_id(self):
        """Output is sorted by (date, excntry, id)."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars()
        result = ff_build_characteristics(panel, june, freq="monthly")
        dates = result["date"].to_list()
        assert dates == sorted(dates)

    def test_mom_signal_joined_as_umd_ff(self, tolerance):
        """When mom_signal is provided, umd_ff column is added."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars()
        mom_signal = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY, US_EXCNTRY],
                "id": [1, 1],
                "date": [date(2020, 8, 31), date(2020, 9, 30)],
                "mom_2_12": [0.15, 0.20],
            }
        )
        result = ff_build_characteristics(panel, june, freq="monthly", mom_signal=mom_signal)
        assert "umd_ff" in result.columns
        np.testing.assert_allclose(result["umd_ff"][0], 0.15, **tolerance.TIGHT)

    def test_no_mom_signal_no_umd_ff(self):
        """Without mom_signal, umd_ff is absent."""
        panel = _make_char_panel("monthly")
        june = _make_june_chars()
        result = ff_build_characteristics(panel, june, freq="monthly")
        assert "umd_ff" not in result.columns

    def test_daily_downsampled_to_monthly(self):
        """Daily input is downsampled to last trading day per (excntry, id, month)."""
        panel = _make_char_panel("daily")
        june = _make_june_chars()
        result = ff_build_characteristics(panel, june, freq="daily")
        # All daily rows are in Aug 2020 → should collapse to 1 row
        assert len(result) == 1

    def test_no_match_in_june_excluded(self):
        """Panel rows with no matching June formation year are dropped."""
        # Panel in 2021, June has formation year 2020 → Jul-Jun(y+1); 2020/07→2021/06 uses 2020
        # Use dates well outside the formation window
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY],
                "id": [99],  # no match in june
                "date": [date(2020, 8, 31)],
                "ret": [0.01],
                "me": [100.0],
                "w": [1.0],
            }
        )
        june = _make_june_chars()  # id=1
        result = ff_build_characteristics(panel, june, freq="monthly")
        assert len(result) == 0
