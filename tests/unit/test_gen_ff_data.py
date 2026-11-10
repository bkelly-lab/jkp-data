"""
Unit tests for FF3/FF5/UMD factor-builder helpers in aux_functions.py.

Covers:
  - _ff_bm_eligible / _ff_op_eligible / _ff_inv_eligible / _ff_mom_eligible
  - ff_country_breakpoints, _ff_bm_breaks / _ff_op_breaks / _ff_inv_breaks,
    _ff_size_breaks
  - ff_annual_characteristics, ff_load_crsp_panel (monthly ME)
  - ff_assign_portfolios
  - ff_assign_to_panel
  - _ff_vw_leg
  - ff_compute_factors
  - _ff_mom_signal_monthly, _ff_mom_signal_daily
  - ff_compute_umd_factor
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import (
    _ff_bm_breaks,
    _ff_bm_eligible,
    _ff_inv_breaks,
    _ff_inv_eligible,
    _ff_mom_eligible,
    _ff_mom_signal_daily,
    _ff_mom_signal_monthly,
    _ff_mom_source_row,
    _ff_mom_source_us,
    _ff_op_breaks,
    _ff_op_eligible,
    _ff_size_breaks,
    _ff_vw_leg,
    ff_annual_characteristics,
    ff_assign_portfolios,
    ff_assign_to_panel,
    ff_compute_factors,
    ff_compute_umd_factor,
    ff_country_breakpoints,
    ff_load_crsp_panel,
    ff_load_world_panel,
)
from jkp.data.config import (
    FACTOR_MIN_STOCKS_BP,
    FACTOR_MIN_STOCKS_PF,
    FF_MISSING_RET_CODE,
    US_EXCNTRY,
)

# =============================================================================
# Helpers
# =============================================================================


def _apply_eligible(df: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    """Evaluate an eligibility Expr on df, returning the boolean Series.

    Frames without dec_me / dec_universe get a positive Dec(t-1) ME in the
    December universe so the gates test their other conditions.
    """
    if "dec_me" not in df.columns:
        df = df.with_columns(dec_me=pl.lit(1.0))
    if "dec_universe" not in df.columns:
        df = df.with_columns(dec_universe=pl.lit(True))
    return df.select(expr.alias("elig"))["elig"]


# =============================================================================
# TestFFBmEligible
# =============================================================================


class TestFFBmEligible:
    """Tests for _ff_bm_eligible()."""

    def test_all_conditions_met(self):
        """Returns True when beme>0, me>0, in the December universe."""
        df = pl.DataFrame({"beme": [1.0], "me": [1.0]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is True

    def test_beme_zero_fails(self):
        """Returns False when beme == 0."""
        df = pl.DataFrame({"beme": [0.0], "me": [1.0]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_beme_negative_fails(self):
        """Returns False when beme < 0."""
        df = pl.DataFrame({"beme": [-1.0], "me": [1.0]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_beme_null_fails(self):
        """Returns null (not True) when beme is null — Polars null propagation."""
        df = pl.DataFrame(
            {"beme": [None], "me": [1.0]},
            schema={"beme": pl.Float64, "me": pl.Float64},
        )
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is not True

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"beme": [1.0], "me": [0.0]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_me_negative_fails(self):
        """Returns False when me < 0."""
        df = pl.DataFrame({"beme": [1.0], "me": [-1.0]})
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is False

    def test_vectorised_mixed(self):
        """Null beme propagates as null; False inputs produce False."""
        df = pl.DataFrame(
            {
                "beme": [1.0, -1.0, 1.0, None],
                "me": [1.0, 1.0, 0.0, 1.0],
            }
        )
        result = _apply_eligible(df, _ff_bm_eligible())
        assert result[0] is True
        assert result[1] is False
        assert result[2] is False
        assert result[3] is not True  # null propagation from null beme

    def test_outside_december_universe_fails(self):
        """Ex US: a stock outside the universe in December t-1 (e.g. a secondary
        line then) stays out of the B/M sort even with a valid beme."""
        df = pl.DataFrame({"beme": [1.0], "me": [1.0], "dec_universe": [False]})
        assert _apply_eligible(df, _ff_bm_eligible())[0] is False


# =============================================================================
# TestFFOpEligible
# =============================================================================


class TestFFOpEligible:
    """Tests for _ff_op_eligible()."""

    def test_all_conditions_met(self):
        """Returns True when me>0, be>0, op not null."""
        df = pl.DataFrame({"me": [1.0], "be": [1.0], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is True

    def test_op_null_fails(self):
        """Returns False when op is null."""
        df = pl.DataFrame(
            {"me": [1.0], "be": [1.0], "op": [None]},
            schema={"me": pl.Float64, "be": pl.Float64, "op": pl.Float64},
        )
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_be_zero_fails(self):
        """Returns False when be == 0."""
        df = pl.DataFrame({"me": [1.0], "be": [0.0], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_be_negative_fails(self):
        """Returns False when be < 0."""
        df = pl.DataFrame({"me": [1.0], "be": [-1.0], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"me": [0.0], "be": [1.0], "op": [0.5]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_vectorised_mixed(self):
        """Batch with two valid and two invalid rows."""
        df = pl.DataFrame(
            {
                "me": [1.0, 1.0, 0.0, 1.0],
                "be": [1.0, -1.0, 1.0, 1.0],
                "op": [0.5, 0.5, 0.5, None],
            }
        )
        result = _apply_eligible(df, _ff_op_eligible())
        assert result.to_list() == [True, False, False, False]

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    @pytest.mark.parametrize("dec_me", [None, 0.0])
    def test_no_dec_me_fails(self, excntry, dec_me):
        """French: RMW needs Dec(t-1) ME as well as June ME, in every country."""
        df = pl.DataFrame(
            {"me": [1.0], "dec_me": [dec_me], "be": [1.0], "op": [0.5]},
            schema_overrides={"dec_me": pl.Float64},
        ).with_columns(excntry=pl.lit(excntry))
        assert _apply_eligible(df, _ff_op_eligible())[0] is False

    def test_outside_december_universe_fails(self):
        df = pl.DataFrame({"me": [1.0], "be": [1.0], "op": [0.5], "dec_universe": [False]})
        assert _apply_eligible(df, _ff_op_eligible())[0] is False


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

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    @pytest.mark.parametrize("dec_me", [None, 0.0])
    def test_no_dec_me_fails(self, excntry, dec_me):
        """French: CMA needs Dec(t-1) ME as well as June ME, in every country."""
        df = pl.DataFrame(
            {"me": [1.0], "dec_me": [dec_me], "inv": [0.1]},
            schema_overrides={"dec_me": pl.Float64},
        ).with_columns(excntry=pl.lit(excntry))
        assert _apply_eligible(df, _ff_inv_eligible())[0] is False

    def test_outside_december_universe_fails(self):
        df = pl.DataFrame({"me": [1.0], "inv": [0.1], "dec_universe": [False]})
        assert _apply_eligible(df, _ff_inv_eligible())[0] is False


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
            "exchcd_us": [1] * n if with_exchcd else [2] * n,
            "size_grp": ["large"] * n,
            "dec_universe": [True] * n,
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
        """US bypasses the FACTOR_MIN_STOCKS_BP gate even with 1 stock."""
        june = _make_june_us(1, [5.0])
        specs = [("beme", 0.50, "beme50")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        assert len(bp) == 1

    def test_row_below_min_stocks_suppressed(self):
        """ROW country with count < FACTOR_MIN_STOCKS_BP yields no breakpoints."""
        n = FACTOR_MIN_STOCKS_BP - 1
        june = pl.DataFrame(
            {
                "excntry": ["GBR"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n + 1)],
                "me": [1.0] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["large"] * n,
                "dec_universe": [True] * n,
            }
        )
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        # No GBR rows should survive
        gbr = bp.filter(pl.col("excntry") == "GBR")
        assert len(gbr) == 0

    def test_row_at_min_stocks_emits_breakpoints(self):
        """ROW country with count == FACTOR_MIN_STOCKS_BP gets breakpoints."""
        n = FACTOR_MIN_STOCKS_BP
        june = pl.DataFrame(
            {
                "excntry": ["GBR"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n + 1)],
                "me": [1.0] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["large"] * n,
                "dec_universe": [True] * n,
            }
        )
        specs = [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")]
        bp = ff_country_breakpoints(june, specs, _ff_bm_eligible())
        gbr = bp.filter(pl.col("excntry") == "GBR")
        assert len(gbr) == 1

    def test_row_uses_size_grp_filter(self):
        """ROW pool requires size_grp in {small, large, mega}; micro excluded."""
        n = FACTOR_MIN_STOCKS_BP + 5
        june = pl.DataFrame(
            {
                "excntry": ["DEU"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n + 1)],
                "me": [1.0] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["micro"] * n,  # all micro → excluded from pool
                "dec_universe": [True] * n,
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
                "exchcd_us": pl.Series([], dtype=pl.Int64),
                "size_grp": pl.Series([], dtype=pl.Utf8),
                "dec_universe": pl.Series([], dtype=pl.Boolean),
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
                "dec_me": [1.0] * 20,
                "be": [1.0] * 20,
                "exchcd_us": [1] * 20,
                "size_grp": ["large"] * 20,
                "dec_universe": [True] * 20,
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
                "dec_me": [1.0] * 20,
                "exchcd_us": [1] * 20,
                "size_grp": ["large"] * 20,
                "dec_universe": [True] * 20,
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
        null beme (bm-ineligible); two NYSE me=0 and one AMEX me=1000
        that must stay out of the lenient pool.
        """
        return pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * 8,
                "date": [date(2020, 6, 30)] * 8,
                "id": list(range(8)),
                "beme": [1.0, 2.0, 3.0, None, None, None, None, 4.0],
                "me": [10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0, 1000.0],
                "exchcd_us": [1, 1, 1, 1, 1, 1, 1, 2],
                "size_grp": ["large"] * 8,
                "dec_universe": [True] * 8,
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
        n = FACTOR_MIN_STOCKS_BP + 5
        june = pl.DataFrame(
            {
                "excntry": ["GBR"] * n,
                "date": [date(2020, 6, 30)] * n,
                "id": list(range(n)),
                "beme": [float(i) for i in range(1, n - 1)] + [None, None],
                "me": [float(i) for i in range(1, n + 1)],
                "exchcd_us": [None] * n,
                "size_grp": ["large"] * n,
                "dec_universe": [True] * n,
            },
            schema_overrides={"exchcd_us": pl.Int64},
        )
        strict = ff_country_breakpoints(june, [("me", 0.50, "sizemedn")], _ff_bm_eligible())
        lenient = _ff_size_breaks(june, _ff_bm_eligible())
        assert strict.equals(lenient)


# =============================================================================
# TestFFAnnualCharacteristics
# =============================================================================


def _write_annual_world_msf(test_paths) -> None:
    """world_msf rows for ff_annual_characteristics (all fiscal 2019 -> port_year 2020).

    GBR (Compustat): 10 Dec ME 50; 11 Dec ME 0; 12 Dec ME 30 as a secondary
    line (primary_sec 0); 13 halted, last row in Nov. US: 20 Compustat-only
    (me 55, me_company 60); 21 Compustat-only with a Nov row only; 30 CRSP (me 40,
    me_company 60, world_msf gvkey g20); 31 CRSP with no accounting.
    """
    rows = [
        # id, excntry, gvkey, eom, me, me_company, source_crsp, primary_sec
        (10, "GBR", "g10", date(2019, 12, 31), 50.0, 50.0, 0, 1),
        (11, "GBR", "g11", date(2019, 12, 31), 0.0, 0.0, 0, 1),
        (12, "GBR", "g12", date(2019, 12, 31), 30.0, 30.0, 0, 0),
        (13, "GBR", "g13", date(2019, 11, 30), 200.0, 200.0, 0, 1),
        (20, US_EXCNTRY, "g20", date(2019, 12, 31), 55.0, 60.0, 0, 1),
        (21, US_EXCNTRY, "g21", date(2019, 11, 30), 80.0, 80.0, 0, 1),
        (30, US_EXCNTRY, "g20", date(2019, 12, 31), 40.0, 60.0, 1, 1),
        (31, US_EXCNTRY, None, date(2019, 12, 31), 70.0, 70.0, 1, 1),
    ]
    cols = ["id", "excntry", "gvkey", "eom", "me", "me_company", "source_crsp", "primary_sec"]
    pl.DataFrame(rows, schema=cols, orient="row").with_columns(
        common=pl.lit(1), obs_main=pl.lit(1), exch_main=pl.lit(1)
    ).write_parquet(test_paths.interim_dir / "world_msf.parquet")


def _acc(gvkeys: list[str], be: float, op: float, inv: float | None) -> pl.DataFrame:
    n = len(gvkeys)
    return pl.DataFrame(
        {
            "gvkey": gvkeys,
            "datadate": [date(2019, 12, 31)] * n,
            "year": [2019] * n,
            "be": [be] * n,
            "op": [op] * n,
            "inv": [inv] * n,
        },
        schema_overrides={"inv": pl.Float64},
    )


class TestFFAnnualCharacteristics:
    """ff_annual_characteristics: one table for the published characteristics and
    the June sorts; it reads no sample screens."""

    @staticmethod
    def _run(test_paths) -> dict[int, dict]:
        _write_annual_world_msf(test_paths)
        acc_us = _acc(["g20", "g21"], be=30.0, op=0.2, inv=None)
        comp_world = _acc(["g10", "g11", "g12", "g13"], be=25.0, op=0.1, inv=0.05)
        ccm2a = _acc(["g30"], be=40.0, op=0.3, inv=0.02).with_columns(permno=pl.lit(30))
        # FF's CRSP sample in December 2019: permno 30 only (31 is outside it).
        crsp_m = pl.LazyFrame({"permno": [30], "date": [date(2019, 12, 31)]})
        out = ff_annual_characteristics(test_paths, acc_us, ccm2a, comp_world, crsp_m)
        assert out.select("id", "port_year").is_duplicated().sum() == 0
        assert out["port_year"].unique().to_list() == [2020]
        return {r["id"]: r for r in out.iter_rows(named=True)}

    def test_row_dec_me_and_beme(self, test_paths):
        r = self._run(test_paths)[10]
        assert r["dec_me"] == 50.0
        assert r["beme"] == pytest.approx(25.0 / 50.0)

    def test_zero_dec_me_nulls_beme_only(self, test_paths):
        r = self._run(test_paths)[11]
        assert r["dec_me"] is None and r["beme"] is None
        assert r["op"] == pytest.approx(0.1)
        assert r["inv"] == pytest.approx(0.05)

    def test_row_dec_me_needs_december(self, test_paths):
        """No December row (halted): Dec ME and beme are null; op is unaffected."""
        r = self._run(test_paths)[13]
        assert r["dec_me"] is None and r["beme"] is None
        assert r["op"] == pytest.approx(0.1)

    def test_dec_universe_is_a_screen_not_an_input(self, test_paths):
        """A secondary line in December keeps its published beme but is flagged
        out of the December universe."""
        out = self._run(test_paths)
        assert out[12]["beme"] == pytest.approx(25.0 / 30.0)
        assert [out[i]["dec_universe"] for i in (10, 12)] == [True, False]

    def test_us_dec_universe_is_ff_crsp_sample(self, test_paths):
        """US: in the December universe only if the permno is in FF's CRSP sample
        that December; Compustat-only securities never are. Dec ME is unaffected."""
        out = self._run(test_paths)
        assert [out[i]["dec_universe"] for i in (20, 30, 31)] == [False, True, False]
        assert out[31]["dec_me"] == 70.0

    def test_us_compustat_only_uses_me_company_and_acc_us(self, test_paths):
        r = self._run(test_paths)[20]
        assert r["dec_me"] == 60.0
        assert r["beme"] == pytest.approx(30.0 / 60.0)
        assert r["op"] == pytest.approx(0.2)

    def test_us_dec_me_needs_december(self, test_paths):
        r = self._run(test_paths)[21]
        assert r["dec_me"] is None and r["beme"] is None
        assert r["be"] == 30.0

    def test_us_crsp_uses_ccm2a_not_world_msf_gvkey(self, test_paths):
        r = self._run(test_paths)[30]
        assert (r["be"], r["dec_me"]) == (40.0, 60.0)
        assert r["beme"] == pytest.approx(40.0 / 60.0)

    def test_dec_me_without_accounting_kept(self, test_paths):
        r = self._run(test_paths)[31]
        assert r["dec_me"] == 70.0
        assert r["be"] is None and r["beme"] is None


# =============================================================================
# TestFFLoadCrspPanel
# =============================================================================


class TestFFLoadCrspPanel:
    """ff_load_crsp_panel: one permno per permco; monthly ME is JKP's me_company."""

    def test_monthly_me_is_me_company(self, test_paths):
        n = 3
        pl.DataFrame(
            {
                "permno": [1, 2, 3],
                "permco": [100, 100, 200],
                "mthcaldt": [date(2020, 6, 30)] * n,
                "mthret": [0.01] * n,
                "mthretx": [0.01] * n,
                "mthprc": [10.0, 5.0, 8.0],
                "shrout": [1000.0] * n,
                "sharetype": ["NS"] * n,
                "securitytype": ["EQTY"] * n,
                "securitysubtype": ["COM"] * n,
                "usincflg": ["Y"] * n,
                "issuertype": ["CORP"] * n,
                "primaryexch": ["N"] * n,
            }
        ).write_parquet(test_paths.raw_tables_dir / "crsp_msf_v2.parquet")
        pl.DataFrame(
            {
                "id": [1, 2],
                "eom": [date(2020, 6, 30)] * 2,
                "source_crsp": [1, 1],
                "me_company": [25.0, 25.0],
            }
        ).write_parquet(test_paths.interim_dir / "world_msf.parquet")
        out = ff_load_crsp_panel(test_paths, "monthly").collect().sort("permno")
        # permno 1 has the larger CIZ ME (10 vs 5); its me is me_company (25), not
        # the CIZ permco sum (15). permno 3 has no world_msf row.
        assert out["permno"].to_list() == [1, 3]
        assert out["me"].to_list() == [25.0, None]


# =============================================================================
# TestFFLoadWorldPanel
# =============================================================================


class TestFFLoadWorldPanel:
    """Universe filter of the ROW panel loader."""

    def test_gvkey_null_stock_stays_in_monthly_panel(self, test_paths):
        """A stock without a Compustat link (null gvkey) passes the universe
        filter (US parity: it stays in size pools, VW legs and momentum)."""
        pl.DataFrame(
            {
                "excntry": ["GBR", "GBR"],
                "id": [1, 2],
                "gvkey": ["g1", None],
                "eom": [date(2020, 6, 30)] * 2,
                "ret": [0.01, 0.02],
                "ret_lag_dif": [1, 1],
                "me": [100.0, 200.0],
                "size_grp": ["large", "large"],
                "common": [1, 1],
                "obs_main": [1, 1],
                "primary_sec": [1, 1],
                "exch_main": [1, 1],
            }
        ).write_parquet(test_paths.interim_dir / "world_msf.parquet")
        panel = ff_load_world_panel(test_paths, "monthly").collect()
        assert sorted(panel["id"].to_list()) == [1, 2]
        assert panel.filter(pl.col("id") == 2)["gvkey"][0] is None


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
            "exchcd_us": [1] * n,
            "size_grp": ["large"] * n,
            "dec_universe": [True] * n,
            # be/op/inv/dec_me referenced by the OP/INV eligibility exprs; null
            # here since op/inv breaks are trivial (no OP/INV assignment exercised).
            "dec_me": [None] * n,
            "be": [None] * n,
            "op": [None] * n,
            "inv": [None] * n,
        },
        schema_overrides={
            "dec_me": pl.Float64,
            "be": pl.Float64,
            "op": pl.Float64,
            "inv": pl.Float64,
        },
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
                "exchcd_us": [1],
                "size_grp": ["large"],
                "dec_universe": [True],
                "dec_me": [1.0],
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
        """US bucket with only 1 stock passes (no FACTOR_MIN_STOCKS_PF gate for US)."""
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
        """ROW bucket with n < FACTOR_MIN_STOCKS_PF gets null."""
        n = FACTOR_MIN_STOCKS_PF - 1
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
        assert len(gbr_rows) > 0
        assert gbr_rows["SH"][0] is None

    def test_row_at_min_stocks_not_nulled(self):
        """ROW bucket with n == FACTOR_MIN_STOCKS_PF passes through."""
        n = FACTOR_MIN_STOCKS_PF
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

    def test_smb_ff3_vs_smb_ff5_differ(self, tolerance):
        """smb_ff3 uses BM leg only; smb_ff5 averages BM/OP/INV legs. Build a
        panel where BM-eligible stocks (positivebeme=1, nonmiss_op=nonmiss_inv=0)
        are disjoint from OP/INV-eligible stocks (positivebeme=0), with
        different size spreads per leg, so the two SMBs are not just present
        but numerically distinct."""
        d = date(2020, 7, 31)
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
            # BM-only stocks: one per (size, btm) bucket; excluded from OP/INV legs.
            ("S", "L", 0.10, 1, "W", 0, "C", 0),
            ("S", "M", 0.05, 1, "W", 0, "C", 0),
            ("S", "H", 0.02, 1, "W", 0, "C", 0),
            ("B", "L", 0.01, 1, "W", 0, "C", 0),
            ("B", "M", 0.01, 1, "W", 0, "C", 0),
            ("B", "H", 0.01, 1, "W", 0, "C", 0),
            # OP-only stocks: one per (size, op) bucket; excluded from BM/INV legs.
            ("S", "L", 0.01, 0, "W", 1, "C", 0),
            ("S", "L", 0.01, 0, "N", 1, "C", 0),
            ("S", "L", 0.01, 0, "R", 1, "C", 0),
            ("B", "L", 0.01, 0, "W", 1, "C", 0),
            ("B", "L", 0.01, 0, "N", 1, "C", 0),
            ("B", "L", 0.01, 0, "R", 1, "C", 0),
            # INV-only stocks: one per (size, inv) bucket; excluded from BM/OP legs.
            ("S", "L", 0.01, 0, "W", 0, "C", 1),
            ("S", "L", 0.01, 0, "W", 0, "N", 1),
            ("S", "L", 0.01, 0, "W", 0, "A", 1),
            ("B", "L", 0.20, 0, "W", 0, "C", 1),
            ("B", "L", 0.20, 0, "W", 0, "N", 1),
            ("B", "L", 0.20, 0, "W", 0, "A", 1),
        ]
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
        assert "smb_ff3" in result.columns
        assert "smb_ff5" in result.columns
        smb_bm = (0.10 + 0.05 + 0.02) / 3 - (0.01 + 0.01 + 0.01) / 3
        smb_op = (0.01 + 0.01 + 0.01) / 3 - (0.01 + 0.01 + 0.01) / 3
        smb_inv = (0.01 + 0.01 + 0.01) / 3 - (0.20 + 0.20 + 0.20) / 3
        np.testing.assert_allclose(result["smb_ff3"][0], smb_bm, **tolerance.STANDARD)
        np.testing.assert_allclose(
            result["smb_ff5"][0], (smb_bm + smb_op + smb_inv) / 3, **tolerance.STANDARD
        )
        assert abs(result["smb_ff3"][0] - result["smb_ff5"][0]) > 1e-6

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
            "prc_us": [1.0] * months,
            "w": [1.0] * months,
            "exchcd_us": [1] * months,
            "size_grp": ["large"] * months,
        },
        schema_overrides={"ret": pl.Float64},
    )


def _jkp_mom(panel: pl.LazyFrame) -> pl.LazyFrame:
    """JKP ret_12_1 keyed by holding month, as market_chars_monthly builds it:
    return index over zero-filled returns, ri[t-2] / ri[t-13] - 1."""
    by = ["excntry", "id"]
    r = pl.col("ret")
    r0 = pl.when((r - FF_MISSING_RET_CODE).abs() < 1e-9).then(0.0).otherwise(r).fill_null(0.0)
    return (
        panel.sort([*by, "date"])
        .with_columns(ri=(1.0 + r0).cum_prod().over(by))
        .select(
            pl.col("id").cast(pl.Int64),
            mth_idx=(pl.col("date").dt.year() * 12 + pl.col("date").dt.month()).cast(pl.Int64),
            ret_12_1=(pl.col("ri").shift(2) / pl.col("ri").shift(13) - 1.0).over(by),
        )
    )


def _mom_src(panel: pl.LazyFrame, freq: str) -> pl.LazyFrame:
    """UMD eligibility source for a test panel: US rows via _ff_mom_source_us;
    non-US rows as a world_msf/world_dsf stand-in (prc = prc_us, ret_lag_dif =
    periods since the previous row), via _ff_mom_source_row."""
    by = ["excntry", "id"]
    if freq == "monthly":
        period = pl.col("date").dt.year() * 12 + pl.col("date").dt.month()
    else:
        period = pl.col("date").cast(pl.Int64)
    world = (
        panel.filter(pl.col("excntry") != US_EXCNTRY)
        .sort([*by, "date"])
        .select(
            "excntry",
            "id",
            "date",
            "me",
            "ret",
            prc=pl.col("prc_us"),
            ret_lag_dif=period.diff().over(by).fill_null(1),
        )
    )
    return pl.concat(
        [_ff_mom_source_us(panel), _ff_mom_source_row(world, freq)], how="vertical_relaxed"
    )


def _run_monthly(panel: pl.LazyFrame) -> pl.LazyFrame:
    return _ff_mom_signal_monthly(panel, _mom_src(panel, "monthly"), _jkp_mom(panel))


def _run_daily(panel: pl.LazyFrame) -> pl.LazyFrame:
    return _ff_mom_signal_daily(panel, _mom_src(panel, "daily"))


class TestFFMomSignalMonthly:
    """Tests for _ff_mom_signal_monthly()."""

    def test_output_schema(self):
        """Output has the expected 11 columns of the mom-signal output schema."""
        panel = _make_monthly_panel(months=20)
        lf = panel.lazy()
        result = _run_monthly(lf).collect()
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
        result = _run_monthly(lf).collect()
        # All rows should have null mom_2_12 (need 13 months for lag 1..12)
        assert result["mom_2_12"].is_null().all()

    def test_sufficient_history_yields_non_null_mom(self):
        """13+ months of data → later rows have non-null mom_2_12."""
        panel = _make_monthly_panel(months=20)
        lf = panel.lazy()
        result = _run_monthly(lf).collect()
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0

    def test_neg99_sentinel_treated_as_zero(self, tolerance):
        """FF_MISSING_RET_CODE (-99) in any lag month is treated as 0.0 return."""
        # Build 20-month panel with -99 at position 5 (will be a lagged return)
        panel = _make_monthly_panel(months=20, base_ret=0.0, include_missing_sentinel=True)
        lf = panel.lazy()
        result = _run_monthly(lf).collect()
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
        result = _run_monthly(self._interior_null_panel(US_EXCNTRY)).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is True
        # 11 window months, one zeroed: (1.01)^10 - 1
        np.testing.assert_allclose(last_row["mom_2_12"][0], 1.01**10 - 1, **tolerance.STANDARD)

    def test_row_interior_null_month_tolerated(self, tolerance):
        """ROW: same gate as the US; the null month counts as 0 in ret_12_1."""
        result = _run_monthly(self._interior_null_panel("GBR")).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is True
        np.testing.assert_allclose(last_row["mom_2_12"][0], 1.01**10 - 1, **tolerance.STANDARD)

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
                "prc_us": [1.0, 1.0],
                "w": [1.0, 1.0],
                "exchcd_us": [1, 1],
                "size_grp": ["large", "large"],
            }
        )
        lf = panel.lazy()
        result = _run_monthly(lf).collect().sort("date")
        # March row: gap1 = (2020*12+3) - (2020*12+1) = 2 ≠ 1 → me_lag1 null
        mar_row = result.filter(pl.col("date") == date(2020, 3, 31))
        assert mar_row["me_lag1"][0] is None

    def test_mom_signal_2_12_skip(self, tolerance):
        """With constant 2% monthly return, compounded 2-12 (skip 1) ≈ 1.02^11 - 1."""
        panel = _make_monthly_panel(months=24, base_ret=0.02)
        lf = panel.lazy()
        result = _run_monthly(lf).collect()
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0
        # 11 months compounded: lags 2 through 12 = 11 months
        expected = (1.02**11) - 1
        np.testing.assert_allclose(non_null["mom_2_12"][0], expected, **tolerance.STANDARD)

    def test_multi_stock_independent(self):
        """Two stocks compute signals independently."""
        p1 = _make_monthly_panel(months=20, base_ret=0.01, stock_id=1)
        p2 = _make_monthly_panel(months=20, base_ret=0.05, stock_id=2)
        panel = pl.concat([p1, p2])
        lf = panel.lazy()
        result = _run_monthly(lf).collect()
        s1 = result.filter(pl.col("id") == 1).filter(pl.col("mom_2_12").is_not_null())
        s2 = result.filter(pl.col("id") == 2).filter(pl.col("mom_2_12").is_not_null())
        assert len(s1) > 0 and len(s2) > 0
        # Stock 2 has higher momentum
        assert s2["mom_2_12"][0] > s1["mom_2_12"][0]

    # Last row of a 20-month panel is index 19: t-13 = 6, t-2 = 17, window t-12..t-2 = 7..17.

    @staticmethod
    def _last(result: pl.DataFrame) -> dict:
        return result.sort("date").row(-1, named=True)

    def test_us_missing_price_at_start_ineligible(self):
        """US: row at t-13 without a price fails French's start check, even though
        the permco-summed me is present."""
        panel = _make_monthly_panel(months=20).with_columns(
            prc_us=pl.when(pl.int_range(pl.len()) == 6).then(None).otherwise(pl.col("prc_us"))
        )
        assert self._last(_run_monthly(panel.lazy()).collect())["eligible_mom"] is False

    def test_us_missing_interior_row_ineligible(self):
        """US: French requires a row (possibly -99) for every month t-12..t-2."""
        panel = _make_monthly_panel(months=20).filter(pl.int_range(pl.len()) != 10)
        assert self._last(_run_monthly(panel.lazy()).collect())["eligible_mom"] is False

    @pytest.mark.parametrize(("halt", "eligible"), [(range(9, 14), True), (range(9, 15), False)])
    def test_row_halt_tolerance(self, halt, eligible):
        """Non-US: a halt is missing rows; 5 missing of 11 window months (6 left)
        stays eligible, 6 missing does not."""
        panel = _make_monthly_panel(months=20, excntry="GBR").filter(
            ~pl.int_range(pl.len()).is_in(list(halt))
        )
        last = self._last(_run_monthly(panel.lazy()).collect())
        assert last["eligible_mom"] is eligible
        assert (last["mom_2_12"] is not None) is eligible

    @pytest.mark.parametrize("missing", [6, 17])
    def test_row_halt_at_endpoint_ineligible(self, missing):
        """Non-US: endpoints stay strict (price at t-13, good return at t-2)."""
        panel = _make_monthly_panel(months=20, excntry="GBR").filter(
            pl.int_range(pl.len()) != missing
        )
        assert self._last(_run_monthly(panel.lazy()).collect())["eligible_mom"] is False

    def test_row_price_without_me_at_start_eligible(self):
        """Non-US: the start check reads world_msf directly, so a priced row whose
        me is null (absent from the FF panel) still counts."""
        full = _make_monthly_panel(months=20, excntry="GBR").with_columns(
            me=pl.when(pl.int_range(pl.len()) == 6).then(None).otherwise(pl.col("me"))
        )
        panel = full.filter(pl.col("me").is_not_null()).lazy()
        src = _mom_src(full.lazy(), "monthly")
        result = _ff_mom_signal_monthly(panel, src, _jkp_mom(full.lazy())).collect()
        assert self._last(result)["eligible_mom"] is True


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
            "prc_us": [1.0] * n_days,
            "w": [1.0] * n_days,
            "exchcd_us": [1] * n_days,
            "size_grp": ["large"] * n_days,
        },
        schema_overrides={"ret": pl.Float64},
    )


class TestFFMomSignalDaily:
    """Tests for _run_daily()."""

    def test_output_schema(self):
        """Output has the expected 11 columns."""
        panel = _make_daily_panel(n_days=300)
        lf = panel.lazy()
        result = _run_daily(lf).collect()
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
        result = _run_daily(lf).collect()
        assert result["mom_2_12"].is_null().all()

    def test_sufficient_history_has_non_null(self):
        """300 days → rows past lookback window have non-null mom_2_12."""
        panel = _make_daily_panel(n_days=300)
        lf = panel.lazy()
        result = _run_daily(lf).collect()
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
                "prc_us": [1.0] * n,
                "w": [1.0] * n,
                "exchcd_us": [1] * n,
                "size_grp": ["large"] * n,
            }
        )
        lf = panel.lazy()
        result = _run_daily(lf).collect()
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

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    def test_null_ret_in_window_tolerated(self, excntry, tolerance):
        """Interior missing-price day is allowed and compounds as 0, in every country
        (French: 'any missing returns from day t-250 to t-22 must be -99.0')."""
        result = _run_daily(self._null_in_window_panel(excntry)).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is True
        # 230-day window with one zeroed day: (1.001)^229 - 1
        np.testing.assert_allclose(last_row["mom_2_12"][0], 1.001**229 - 1, **tolerance.STANDARD)

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    def test_window_includes_skip_day(self, excntry):
        """Window pin: a spike at t-21 is inside the window (t-250..t-21, per
        French's daily detail) in every country."""
        n = 300
        panel = _make_daily_panel(
            n_days=n,
            base_ret=0.0,
            excntry=excntry,
            ret_overrides={n - 1 - 21: 0.5},  # day t-21 for the last row
        )
        result = _run_daily(panel.lazy()).collect()
        last_row = result.sort("date").tail(1)
        assert last_row["eligible_mom"][0] is True
        assert abs(last_row["mom_2_12"][0] - 0.5) < 1e-9

    def test_multi_country_independent(self):
        """Two countries compute trading-day index independently."""
        p_usa = _make_daily_panel(300, base_ret=0.001, excntry="USA", stock_id=1)
        p_gbr = _make_daily_panel(300, base_ret=0.003, excntry="GBR", stock_id=1)
        panel = pl.concat([p_usa, p_gbr])
        lf = panel.lazy()
        result = _run_daily(lf).collect()
        usa = result.filter((pl.col("excntry") == "USA") & pl.col("mom_2_12").is_not_null())
        gbr = result.filter((pl.col("excntry") == "GBR") & pl.col("mom_2_12").is_not_null())
        assert len(usa) > 0 and len(gbr) > 0
        # GBR has higher per-day return → higher momentum
        assert gbr["mom_2_12"].mean() > usa["mom_2_12"].mean()

    # Last row of a 300-day panel is index 299: t-251 = 48, t-21 = 278, window = 49..278.
    # A second, complete stock keeps every date in the country's trading calendar.

    @staticmethod
    def _with_full_peer(panel: pl.DataFrame) -> pl.LazyFrame:
        peer = _make_daily_panel(300, excntry=panel["excntry"][0], stock_id=2)
        return pl.concat([panel, peer]).lazy()

    @staticmethod
    def _last_id1(result: pl.DataFrame) -> dict:
        return result.filter(pl.col("id") == 1).sort("date").row(-1, named=True)

    def test_us_missing_price_at_start_ineligible(self):
        """US: no price at t-251 fails French's start check."""
        panel = _make_daily_panel(300).with_columns(
            prc_us=pl.when(pl.int_range(pl.len()) == 48).then(None).otherwise(pl.col("prc_us"))
        )
        assert (
            self._last_id1(_run_daily(self._with_full_peer(panel)).collect())["eligible_mom"]
            is False
        )

    def test_us_skip_window_rows_not_required(self):
        """US: rows t-20..t-2 lie outside French's window and may be missing."""
        panel = _make_daily_panel(300).filter(~pl.int_range(pl.len()).is_between(279, 297))
        assert (
            self._last_id1(_run_daily(self._with_full_peer(panel)).collect())["eligible_mom"]
            is True
        )

    def test_row_halt_spanning_return_included(self):
        """Non-US: a 51-day halt is tolerated and the first return after it
        (p_end / p_start - 1) enters the compounded window return in full."""
        panel = _make_daily_panel(
            300, base_ret=0.0, excntry="GBR", ret_overrides={151: 0.5}
        ).filter(~pl.int_range(pl.len()).is_between(100, 150))
        last = self._last_id1(_run_daily(self._with_full_peer(panel)).collect())
        assert last["eligible_mom"] is True
        assert abs(last["mom_2_12"] - 0.5) < 1e-9

    @pytest.mark.parametrize(("n_missing", "eligible"), [(115, True), (116, False)])
    def test_row_halt_min_days(self, n_missing, eligible):
        """Non-US: at least 115 of the 230 window days must be present."""
        panel = _make_daily_panel(300, excntry="GBR").filter(
            ~pl.int_range(pl.len()).is_between(100, 100 + n_missing - 1)
        )
        assert (
            self._last_id1(_run_daily(self._with_full_peer(panel)).collect())["eligible_mom"]
            is eligible
        )


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
