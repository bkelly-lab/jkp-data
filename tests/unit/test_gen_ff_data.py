"""
Unit tests for the FF3/FF5/UMD factor builder helpers in aux_functions.py.

Covers:
  - ff_sort_definitions (eligibility of each sort)
  - compute_ff_breakpoints (breakpoint pool, minimum stocks, size median, sort joins)
  - compute_ff_annual_characteristics (accounting records, December ME from the panel)
  - load_ff_universe (rule columns, US vs elsewhere, monthly and daily)
  - build_ff_firm_panel (class aggregation, ME, null-ME rows kept)
  - add_ff_buy_and_hold_weights (buy-and-hold weights, gap-spanning returns, null-ME rows kept)
  - assign_ff_portfolios
  - compute_ff_portfolio_returns
  - compute_ff_factor_returns (July-June broadcast, FF3/FF5/UMD spreads)
  - compute_ff_momentum_signal (monthly and daily)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import jkp.data.aux_functions as af
from jkp.data.aux_functions import (
    add_ff_buy_and_hold_weights,
    assign_ff_portfolios,
    build_ff_firm_panel,
    compute_ff_annual_characteristics,
    compute_ff_breakpoints,
    compute_ff_factor_returns,
    compute_ff_momentum_signal,
    compute_ff_portfolio_returns,
    ff_sort_definitions,
    load_ff_universe,
)
from jkp.data.config import (
    FACTOR_MIN_STOCKS_BP,
    FACTOR_MIN_STOCKS_PF,
    US_EXCNTRY,
)

# =============================================================================
# Helpers
# =============================================================================


def _apply_eligible(df: pl.DataFrame, sort: str) -> pl.Series:
    """Evaluate the sort's eligibility (ff_sort_definitions) on df, returning the boolean Series.

    Frames without dec_me get a positive December ME so the gates test their
    other conditions.
    """
    if "dec_me" not in df.columns:
        df = df.with_columns(dec_me=pl.lit(1.0))
    return df.select(ff_sort_definitions()[sort][2].alias("elig"))["elig"]


# =============================================================================
# TestFFBmEligible
# =============================================================================


class TestFFBmEligible:
    """Tests for the bm eligibility in ff_sort_definitions()."""

    def test_all_conditions_met(self):
        """Returns True when beme>0, me>0, in the December universe."""
        df = pl.DataFrame({"beme": [1.0], "me": [1.0]})
        result = _apply_eligible(df, "bm")
        assert result[0] is True

    def test_beme_zero_fails(self):
        """Returns False when beme == 0."""
        df = pl.DataFrame({"beme": [0.0], "me": [1.0]})
        result = _apply_eligible(df, "bm")
        assert result[0] is False

    def test_beme_negative_fails(self):
        """Returns False when beme < 0."""
        df = pl.DataFrame({"beme": [-1.0], "me": [1.0]})
        result = _apply_eligible(df, "bm")
        assert result[0] is False

    def test_beme_null_fails(self):
        """Returns null (not True) when beme is null — Polars null propagation."""
        df = pl.DataFrame(
            {"beme": [None], "me": [1.0]},
            schema={"beme": pl.Float64, "me": pl.Float64},
        )
        result = _apply_eligible(df, "bm")
        assert result[0] is not True

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"beme": [1.0], "me": [0.0]})
        result = _apply_eligible(df, "bm")
        assert result[0] is False

    def test_me_negative_fails(self):
        """Returns False when me < 0."""
        df = pl.DataFrame({"beme": [1.0], "me": [-1.0]})
        result = _apply_eligible(df, "bm")
        assert result[0] is False

    def test_vectorised_mixed(self):
        """Null beme propagates as null; False inputs produce False."""
        df = pl.DataFrame(
            {
                "beme": [1.0, -1.0, 1.0, None],
                "me": [1.0, 1.0, 0.0, 1.0],
            }
        )
        result = _apply_eligible(df, "bm")
        assert result[0] is True
        assert result[1] is False
        assert result[2] is False
        assert result[3] is not True  # null propagation from null beme

    def test_no_december_me_fails(self):
        """beme is null for a unit with no December ME, so it is not B/M eligible
        even with a valid book equity."""
        df = pl.DataFrame(
            {"beme": [None], "me": [1.0]}, schema={"beme": pl.Float64, "me": pl.Float64}
        )
        assert _apply_eligible(df, "bm")[0] is not True


# =============================================================================
# TestFFOpEligible
# =============================================================================


class TestFFOpEligible:
    """Tests for the op eligibility in ff_sort_definitions()."""

    def test_all_conditions_met(self):
        """Returns True when me>0, be>0, op not null."""
        df = pl.DataFrame({"me": [1.0], "be": [1.0], "op": [0.5]})
        assert _apply_eligible(df, "op")[0] is True

    def test_op_null_fails(self):
        """Returns False when op is null."""
        df = pl.DataFrame(
            {"me": [1.0], "be": [1.0], "op": [None]},
            schema={"me": pl.Float64, "be": pl.Float64, "op": pl.Float64},
        )
        assert _apply_eligible(df, "op")[0] is False

    def test_be_zero_fails(self):
        """Returns False when be == 0."""
        df = pl.DataFrame({"me": [1.0], "be": [0.0], "op": [0.5]})
        assert _apply_eligible(df, "op")[0] is False

    def test_be_negative_fails(self):
        """Returns False when be < 0."""
        df = pl.DataFrame({"me": [1.0], "be": [-1.0], "op": [0.5]})
        assert _apply_eligible(df, "op")[0] is False

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"me": [0.0], "be": [1.0], "op": [0.5]})
        assert _apply_eligible(df, "op")[0] is False

    def test_vectorised_mixed(self):
        """Batch with two valid and two invalid rows."""
        df = pl.DataFrame(
            {
                "me": [1.0, 1.0, 0.0, 1.0],
                "be": [1.0, -1.0, 1.0, 1.0],
                "op": [0.5, 0.5, 0.5, None],
            }
        )
        result = _apply_eligible(df, "op")
        assert result.to_list() == [True, False, False, False]

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    @pytest.mark.parametrize("dec_me", [None, 0.0])
    def test_no_dec_me_fails(self, excntry, dec_me):
        """French: RMW needs Dec(t-1) ME as well as June ME, in every country."""
        df = pl.DataFrame(
            {"me": [1.0], "dec_me": [dec_me], "be": [1.0], "op": [0.5]},
            schema_overrides={"dec_me": pl.Float64},
        ).with_columns(excntry=pl.lit(excntry))
        assert _apply_eligible(df, "op")[0] is False


# =============================================================================
# TestFFInvEligible
# =============================================================================


class TestFFInvEligible:
    """Tests for the inv eligibility in ff_sort_definitions()."""

    def test_passes_with_positive_me_and_inv(self):
        """Returns True when me>0 and inv is not null."""
        df = pl.DataFrame({"me": [1.0], "inv": [0.1]})
        assert _apply_eligible(df, "inv")[0] is True

    def test_inv_null_fails(self):
        """Returns False when inv is null."""
        df = pl.DataFrame(
            {"me": [1.0], "inv": [None]}, schema={"me": pl.Float64, "inv": pl.Float64}
        )
        assert _apply_eligible(df, "inv")[0] is False

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"me": [0.0], "inv": [0.1]})
        assert _apply_eligible(df, "inv")[0] is False

    def test_me_negative_fails(self):
        """Returns False when me < 0."""
        df = pl.DataFrame({"me": [-1.0], "inv": [0.1]})
        assert _apply_eligible(df, "inv")[0] is False

    def test_inv_zero_passes(self):
        """Returns True when inv == 0 (zero is valid, only null is excluded)."""
        df = pl.DataFrame({"me": [1.0], "inv": [0.0]})
        assert _apply_eligible(df, "inv")[0] is True

    def test_inv_negative_passes(self):
        """Returns True when inv < 0 (sign not gated)."""
        df = pl.DataFrame({"me": [1.0], "inv": [-0.5]})
        assert _apply_eligible(df, "inv")[0] is True

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    @pytest.mark.parametrize("dec_me", [None, 0.0])
    def test_no_dec_me_fails(self, excntry, dec_me):
        """French: CMA needs Dec(t-1) ME as well as June ME, in every country."""
        df = pl.DataFrame(
            {"me": [1.0], "dec_me": [dec_me], "inv": [0.1]},
            schema_overrides={"dec_me": pl.Float64},
        ).with_columns(excntry=pl.lit(excntry))
        assert _apply_eligible(df, "inv")[0] is False


# =============================================================================
# TestFFMomEligible
# =============================================================================


class TestFFMomEligible:
    """Tests for the mom eligibility in ff_sort_definitions()."""

    def test_passes_with_valid_inputs(self):
        """Returns True when mom_2_12 not null and me > 0."""
        df = pl.DataFrame({"mom_2_12": [0.1], "me": [1.0]})
        assert _apply_eligible(df, "mom")[0] is True

    def test_mom_null_fails(self):
        """Returns False when mom_2_12 is null."""
        df = pl.DataFrame(
            {"mom_2_12": [None], "me": [1.0]},
            schema={"mom_2_12": pl.Float64, "me": pl.Float64},
        )
        assert _apply_eligible(df, "mom")[0] is False

    def test_me_zero_fails(self):
        """Returns False when me == 0."""
        df = pl.DataFrame({"mom_2_12": [0.1], "me": [0.0]})
        assert _apply_eligible(df, "mom")[0] is False

    def test_me_negative_fails(self):
        """Returns False when me < 0."""
        df = pl.DataFrame({"mom_2_12": [0.1], "me": [-1.0]})
        assert _apply_eligible(df, "mom")[0] is False

    def test_me_null_fails(self):
        """Returns null (not True) when me is null — Polars null propagation."""
        df = pl.DataFrame(
            {"mom_2_12": [0.1], "me": [None]},
            schema={"mom_2_12": pl.Float64, "me": pl.Float64},
        )
        assert _apply_eligible(df, "mom")[0] is not True

    def test_both_null_fails(self):
        """Returns null (not True) when both are null."""
        df = pl.DataFrame(
            {"mom_2_12": [None], "me": [None]},
            schema={"mom_2_12": pl.Float64, "me": pl.Float64},
        )
        assert _apply_eligible(df, "mom")[0] is not True

    def test_vectorised(self):
        """Batch: True for valid row, False/null for invalid rows."""
        df = pl.DataFrame(
            {
                "mom_2_12": [0.1, None, 0.1, 0.1],
                "me": [1.0, 1.0, 0.0, None],
            }
        )
        result = _apply_eligible(df, "mom")
        assert result[0] is True
        assert result[1] is not True  # mom_2_12 null
        assert result[2] is False  # me == 0
        assert result[3] is not True  # me null


# =============================================================================
# TestFFBreakpoints
# =============================================================================


def _june(
    n: int,
    vals: list[float | None],
    excntry: str = US_EXCNTRY,
    bp_pool: bool = True,
    me: list[float] | None = None,
) -> pl.DataFrame:
    """Minimal June snapshot of n stocks for the B/M sort."""
    return pl.DataFrame(
        {
            "excntry": [excntry] * n,
            "date": [date(2020, 6, 30)] * n,
            "id": list(range(n)),
            "beme": vals,
            "me": me or [1.0] * n,
            "bp_pool": [bp_pool] * n,
        },
        schema_overrides={"beme": pl.Float64},
    )


class TestFFBreakpoints:
    """Tests for compute_ff_breakpoints()."""

    def test_us_30_70_quantiles(self):
        """30/70 quantiles of the pool's B/M match the expected values."""
        bp = compute_ff_breakpoints(_june(10, [float(i) for i in range(1, 11)]), ["bm"], 1)
        assert len(bp) == 1
        row = bp.row(0, named=True)
        # QUANTILE_DISC at 0.30 of [1..10] => 3, at 0.70 => 7
        assert row["bm_lo"] == 3.0
        assert row["bm_hi"] == 7.0

    def test_us_below_min_stocks_suppressed(self):
        """The US is gated like every country: 1 pool stock yields no breakpoints."""
        bp = compute_ff_breakpoints(_june(1, [5.0]), ["bm"], FACTOR_MIN_STOCKS_BP)
        assert len(bp) == 0

    def test_row_below_min_stocks_suppressed(self):
        """A country with fewer than FACTOR_MIN_STOCKS_BP pool stocks has no row."""
        n = FACTOR_MIN_STOCKS_BP - 1
        june = _june(n, [float(i) for i in range(1, n + 1)], excntry="GBR")
        assert len(compute_ff_breakpoints(june, ["bm"], FACTOR_MIN_STOCKS_BP)) == 0

    def test_row_at_min_stocks_emits_breakpoints(self):
        """A country with exactly FACTOR_MIN_STOCKS_BP pool stocks gets breakpoints."""
        n = FACTOR_MIN_STOCKS_BP
        june = _june(n, [float(i) for i in range(1, n + 1)], excntry="GBR")
        assert len(compute_ff_breakpoints(june, ["bm"], FACTOR_MIN_STOCKS_BP)) == 1

    def test_stocks_outside_pool_excluded(self):
        """Stocks with bp_pool False (e.g. micro caps) never form breakpoints."""
        n = FACTOR_MIN_STOCKS_BP + 5
        june = _june(n, [float(i) for i in range(1, n + 1)], excntry="DEU", bp_pool=False)
        assert len(compute_ff_breakpoints(june, ["bm"], FACTOR_MIN_STOCKS_BP)) == 0

    def test_empty_input_returns_empty(self):
        """Empty input produces empty output."""
        june = pl.DataFrame(
            schema={
                "excntry": pl.Utf8,
                "date": pl.Date,
                "id": pl.Int64,
                "beme": pl.Float64,
                "me": pl.Float64,
                "bp_pool": pl.Boolean,
            }
        )
        assert len(compute_ff_breakpoints(june, ["bm"], FACTOR_MIN_STOCKS_BP)) == 0

    def test_output_schema(self):
        """Output is [excntry, date, sizemedn, <sort>_lo, <sort>_hi]."""
        bp = compute_ff_breakpoints(_june(10, [float(i) for i in range(1, 11)]), ["bm"], 1)
        assert bp.columns == ["excntry", "date", "sizemedn", "bm_lo", "bm_hi"]

    def test_op_sort_returns_op_lo_hi(self):
        """The OP sort emits op_lo, op_hi, from stocks with BE, OP and both MEs."""
        june = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * 20,
                "date": [date(2020, 6, 30)] * 20,
                "id": list(range(20)),
                "op": [float(i) * 0.1 for i in range(1, 21)],
                "me": [1.0] * 20,
                "dec_me": [1.0] * 20,
                "be": [1.0] * 20,
                "bp_pool": [True] * 20,
            }
        )
        bp = compute_ff_breakpoints(june, ["op"], 1)
        assert bp.columns == ["excntry", "date", "sizemedn", "op_lo", "op_hi"]
        assert bp["op_lo"][0] < bp["op_hi"][0]

    def test_inv_sort_returns_inv_lo_hi(self):
        """The INV sort emits inv_lo, inv_hi."""
        june = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * 20,
                "date": [date(2020, 6, 30)] * 20,
                "id": list(range(20)),
                "inv": [float(i) * 0.05 for i in range(1, 21)],
                "me": [1.0] * 20,
                "dec_me": [1.0] * 20,
                "bp_pool": [True] * 20,
            }
        )
        bp = compute_ff_breakpoints(june, ["inv"], 1)
        assert bp.columns == ["excntry", "date", "sizemedn", "inv_lo", "inv_hi"]

    def test_sorts_are_independent(self):
        """A country-date with the size median forms each sort that has its own
        breakpoints: no B/M breakpoints leaves bm null but still forms INV."""
        june = _june(10, [None] * 10).with_columns(
            inv=pl.Series([float(i) for i in range(1, 11)]), dec_me=pl.lit(1.0)
        )
        bp = compute_ff_breakpoints(june, ["bm", "inv"], 1)
        assert bp.height == 1
        row = bp.row(0, named=True)
        assert row["bm_lo"] is None and row["bm_hi"] is None
        assert row["inv_lo"] == 3.0 and row["inv_hi"] == 7.0

    @staticmethod
    def _june_size_pool() -> pl.DataFrame:
        """US June frame where the B/M-eligible pool and the whole pool have
        different ME medians.

        Pool me 10/20/30 is B/M eligible (strict median 20); pool me 40/50 has null
        beme (ineligible); two pool stocks have me 0 and one stock outside the pool
        has me 1 and beme 4, and none of those may count.
        """
        return pl.DataFrame(
            {
                "excntry": [US_EXCNTRY] * 8,
                "date": [date(2020, 6, 30)] * 8,
                "id": list(range(8)),
                "beme": [1.0, 2.0, 3.0, None, None, None, None, 4.0],
                "me": [10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0, 1.0],
                "bp_pool": [True] * 7 + [False],
            }
        )

    def test_size_median_pools_all_me_positive(self):
        """The size median runs over every pool stock with me > 0, B/M eligible or
        not; me <= 0 and stocks outside the pool stay out."""
        june = self._june_size_pool()
        strict = af.country_breakpoints(
            june,
            [("me", 0.50, "sizemedn")],
            group_keys=["excntry", "date"],
            pool=pl.col("bp_pool") & ff_sort_definitions()["bm"][2],
            min_stocks=1,
        )
        wide = compute_ff_breakpoints(june, ["bm"], 1)
        assert strict["sizemedn"][0] == 20.0  # B/M-eligible pool: 10/20/30
        assert wide["sizemedn"][0] == 30.0  # pool with me > 0: 10/20/30/40/50

    def test_value_breaks_use_eligible_pool(self):
        """30/70 breakpoints keep the sort-eligible pool, whatever the (wider)
        size-median pool: the pool B/M values are 1, 2, 3."""
        bp = compute_ff_breakpoints(self._june_size_pool(), ["bm"], 1)
        assert (bp["bm_lo"][0], bp["bm_hi"][0]) == (1.0, 3.0)

    def test_row_size_median_excludes_non_pool_stocks(self):
        """Elsewhere the same rule: every pool stock with me > 0, B/M eligible or
        not; stocks outside the pool (micro caps) stay out."""
        n = FACTOR_MIN_STOCKS_BP + 5
        june = _june(
            n,
            [float(i) for i in range(1, n - 1)] + [None, None],
            excntry="GBR",
            me=[float(i) for i in range(1, n + 1)],
        )
        micro = june.head(3).with_columns(
            id=pl.int_range(n, n + 3), me=pl.lit(1e6), bp_pool=pl.lit(False)
        )
        bp = compute_ff_breakpoints(pl.concat([june, micro]), ["bm"], FACTOR_MIN_STOCKS_BP)
        # median (QUANTILE_DISC) of me 1..15 is 8; the B/M-eligible 1..13 give 7
        assert bp["sizemedn"][0] == 8.0


# =============================================================================
# TestFFAnnualSignals
# =============================================================================


def _records(rows: list[tuple]) -> pl.DataFrame:
    """link_accounting_to_securities output rows: (gvkey, id, be, op, inv, datadate); fiscal 2019."""
    return pl.DataFrame(
        [(g, i, 2019, d, be, op, inv) for g, i, be, op, inv, d in rows],
        schema={
            "gvkey": pl.Utf8,
            "id": pl.Int64,
            "year": pl.Int32,
            "datadate": pl.Date,
            "be": pl.Float64,
            "op": pl.Float64,
            "inv": pl.Float64,
        },
        orient="row",
    )


def _dec_panel(rows: list[tuple[str, int, float | None]]) -> pl.DataFrame:
    """Panel rows (excntry, id, me) at December 2019; a null me is a row without ME."""
    return pl.DataFrame(
        [(e, i, date(2019, 12, 31), me) for e, i, me in rows],
        schema={"excntry": pl.Utf8, "id": pl.Int64, "date": pl.Date, "me": pl.Float64},
        orient="row",
    )


def _annual_signals(
    test_paths, monkeypatch, records: pl.DataFrame, panel: pl.DataFrame, permcos: dict[int, int]
) -> dict[int, dict]:
    """compute_ff_annual_characteristics with the accounting records fed in directly. world_msf lists
    every record's security: a CRSP security with its permco (permcos), else Compustat."""
    _write_world_msf(
        test_paths,
        [
            {
                "id": i,
                "eom": date(2020, 6, 30),
                "source_crsp": int(i in permcos),
                "permco": permcos.get(i),
            }
            for i in records["id"].unique().to_list()
        ],
    )
    monkeypatch.setattr(af, "load_jkp_annual_accounting", lambda paths: None)
    monkeypatch.setattr(af, "select_ff_fiscal_records", lambda acc: None)
    monkeypatch.setattr(
        af, "link_accounting_to_securities", lambda paths, comp, *, dff_path: records
    )
    out = compute_ff_annual_characteristics(test_paths, panel)
    assert out.select("id", "port_year").is_duplicated().sum() == 0
    assert out["port_year"].unique().to_list() == [2020]
    return {r["id"]: r for r in out.iter_rows(named=True)}


class TestFFAnnualSignals:
    """compute_ff_annual_characteristics: one row per unit with the accounting of fiscal y-1 and
    the panel's December y-1 ME; a unit not in the panel that December has no ME."""

    @staticmethod
    def _run(test_paths, monkeypatch) -> dict[int, dict]:
        fy = date(2019, 12, 31)
        records = _records(
            [
                ("g10", 10, 25.0, 0.1, 0.05, fy),
                ("g11", 11, 25.0, 0.1, 0.05, fy),
                ("g12", 12, 25.0, 0.1, 0.05, fy),
                ("g13", 13, 25.0, 0.1, 0.05, fy),
                ("g30", 30, 40.0, 0.3, 0.02, fy),
                ("g32", 32, 30.0, 0.2, None, fy),
            ]
        )
        # 11 has ME 0, 12 is outside the universe that December, 13 has no December
        # row (halted); 31 has December ME but no accounting
        panel = _dec_panel(
            [
                ("GBR", 10, 50.0),
                ("GBR", 11, 0.0),
                (US_EXCNTRY, 30, 60.0),
                (US_EXCNTRY, 31, 70.0),
            ]
        )
        return _annual_signals(test_paths, monkeypatch, records, panel, {30: 30, 32: 32})

    def test_row_dec_me_and_beme(self, test_paths, monkeypatch):
        r = self._run(test_paths, monkeypatch)[10]
        assert r["dec_me"] == 50.0
        assert r["beme"] == pytest.approx(25.0 / 50.0)

    def test_zero_dec_me_nulls_beme_only(self, test_paths, monkeypatch):
        r = self._run(test_paths, monkeypatch)[11]
        assert r["dec_me"] is None and r["beme"] is None
        assert r["op"] == pytest.approx(0.1)
        assert r["inv"] == pytest.approx(0.05)

    def test_dec_me_needs_december(self, test_paths, monkeypatch):
        """No December row (halted): Dec ME and beme are null; op is unaffected."""
        r = self._run(test_paths, monkeypatch)[13]
        assert r["dec_me"] is None and r["beme"] is None
        assert r["op"] == pytest.approx(0.1)

    def test_unit_outside_universe_in_december_has_no_dec_me(self, test_paths, monkeypatch):
        """A unit absent from the panel in December (e.g. a secondary line) keeps
        its accounting but has no December ME, so no beme."""
        r = self._run(test_paths, monkeypatch)[12]
        assert r["dec_me"] is None and r["beme"] is None
        assert r["be"] == 25.0

    def test_us_crsp_uses_linked_accounting_and_panel_me(self, test_paths, monkeypatch):
        r = self._run(test_paths, monkeypatch)[30]
        assert (r["be"], r["dec_me"]) == (40.0, 60.0)
        assert r["beme"] == pytest.approx(40.0 / 60.0)

    def test_us_dec_me_needs_december(self, test_paths, monkeypatch):
        r = self._run(test_paths, monkeypatch)[32]
        assert r["dec_me"] is None and r["beme"] is None
        assert r["be"] == 30.0

    def test_dec_me_without_accounting_kept(self, test_paths, monkeypatch):
        r = self._run(test_paths, monkeypatch)[31]
        assert r["dec_me"] == 70.0
        assert r["be"] is None and r["beme"] is None


class TestFFAnnualSignalsFirms:
    """compute_ff_annual_characteristics keys US CRSP securities by their FF firm: a company's
    classes share one row, whichever class carries the accounting."""

    @staticmethod
    def _run(test_paths, monkeypatch) -> dict[int, dict]:
        fy, dff = date(2019, 12, 31), date(2020, 6, 30)
        records = _records(
            [
                ("g40", 41, 50.0, 0.2, 0.1, fy),
                # DFF record (no gvkey) on the other class
                (None, 40, 45.0, None, None, dff),
                ("g50", 50, 20.0, 0.3, 0.05, fy),
                ("g51", 51, 10.0, 0.4, 0.06, fy),
                ("g60", 60, 5.0, 0.5, 0.0, date(2019, 6, 30)),
                ("g61", 61, 6.0, 0.6, 0.0, fy),
                ("g71", 71, None, 0.1, 0.0, fy),
                (None, 70, 8.0, None, None, dff),
            ]
        )
        # December rows of the firms; 51 has none
        panel = _dec_panel(
            [
                (US_EXCNTRY, 40, 100.0),
                (US_EXCNTRY, 50, 70.0),
                (US_EXCNTRY, 60, 50.0),
                (US_EXCNTRY, 70, 16.0),
            ]
        )
        permcos = {40: 40, 41: 40, 50: 50, 51: 51, 60: 60, 61: 60, 70: 70, 71: 70}
        return _annual_signals(test_paths, monkeypatch, records, panel, permcos)

    def test_one_row_per_firm(self, test_paths, monkeypatch):
        assert sorted(self._run(test_paths, monkeypatch)) == [40, 50, 51, 60, 70]

    def test_other_class_accounting_reaches_the_firm_compustat_first(self, test_paths, monkeypatch):
        """Class 41 carries the Compustat record, class 40 a DFF BE: the firm keeps
        the Compustat record, with the company's December ME."""
        r = self._run(test_paths, monkeypatch)[40]
        assert (r["be"], r["op"], r["inv"], r["dec_me"]) == (50.0, 0.2, 0.1, 100.0)
        assert r["beme"] == pytest.approx(0.5)

    def test_dff_on_other_class_fills_missing_compustat_be(self, test_paths, monkeypatch):
        """Class 71's Compustat record has no BE; class 70's DFF BE fills it."""
        r = self._run(test_paths, monkeypatch)[70]
        assert (r["be"], r["op"]) == (8.0, 0.1)

    def test_firm_dec_me_comes_from_the_panel(self, test_paths, monkeypatch):
        r = self._run(test_paths, monkeypatch)[50]
        assert r["dec_me"] == 70.0
        assert r["beme"] == pytest.approx(20.0 / 70.0)

    def test_firm_without_december_row_keeps_accounting_without_dec_me(
        self, test_paths, monkeypatch
    ):
        """A firm with no December row in the panel keeps its accounting but has no
        December ME, so it is out of every sort."""
        r = self._run(test_paths, monkeypatch)[51]
        assert r["be"] == 10.0
        assert r["dec_me"] is None and r["beme"] is None

    def test_latest_fiscal_year_end_wins_within_firm(self, test_paths, monkeypatch):
        r = self._run(test_paths, monkeypatch)[60]
        assert (r["be"], r["op"]) == (6.0, 0.6)


# =============================================================================
# World files for the universe and panel tests
# =============================================================================


_MSF_SCHEMA = {
    "id": pl.Int64,
    "eom": pl.Date,
    "excntry": pl.Utf8,
    "source_crsp": pl.Int64,
    "common": pl.Int64,
    "obs_main": pl.Int64,
    "primary_sec": pl.Int64,
    "exch_main": pl.Int64,
    "primaryexch": pl.Utf8,
    "gvkey": pl.Utf8,
    "prc": pl.Float64,
    "me": pl.Float64,
    "me_company": pl.Float64,
    "ret": pl.Float64,
    "ret_lag_dif": pl.Int64,
    "size_grp": pl.Utf8,
    "permco": pl.Int64,
}

_MSF_DEFAULTS = {
    "excntry": US_EXCNTRY,
    "source_crsp": 1,
    "common": 1,
    "obs_main": 1,
    "primary_sec": 1,
    "exch_main": 1,
    "primaryexch": None,
    "gvkey": None,
    "prc": 8.0,
    "me": 8.0,
    "me_company": 8.0,
    "ret": 0.01,
    "ret_lag_dif": 1,
    "size_grp": None,
}


def _write_world_msf(test_paths, rows: list[dict]) -> None:
    """world_msf from rows that give id, eom and what differs from _MSF_DEFAULTS; the
    permco defaults to the id, a company of one class."""
    pl.DataFrame(
        [{**_MSF_DEFAULTS, "permco": r["id"], **r} for r in rows], schema=_MSF_SCHEMA
    ).write_parquet(test_paths.interim_dir / "world_msf.parquet")


def _write_raw_msf(test_paths, rows: list[tuple[int, date, str]]) -> None:
    """Raw CIZ monthly rows (permno, mthcaldt, usincflg)."""
    pl.DataFrame(rows, schema=["permno", "mthcaldt", "usincflg"], orient="row").write_parquet(
        test_paths.raw_tables_dir / "crsp_msf_v2.parquet"
    )


def _write_world_dsf(test_paths, rows: list[dict]) -> None:
    """world_dsf from rows that give id, date, eom and what differs from _MSF_DEFAULTS."""
    cols = ["excntry", "source_crsp", "common", "obs_main", "primary_sec", "exch_main"]
    cols += ["prc", "me", "ret", "ret_lag_dif"]
    schema = {"id": pl.Int64, "date": pl.Date, "eom": pl.Date} | {c: _MSF_SCHEMA[c] for c in cols}
    pl.DataFrame(
        [{**{c: _MSF_DEFAULTS[c] for c in cols}, **r} for r in rows], schema=schema
    ).write_parquet(test_paths.interim_dir / "world_dsf.parquet")


# =============================================================================
# TestFFUniverseRows
# =============================================================================


class TestFFUniverseRows:
    """load_ff_universe: the rule columns, where the US and other countries differ."""

    @staticmethod
    def _monthly(test_paths) -> dict[int, dict]:
        may, jun = date(2020, 5, 31), date(2020, 6, 30)
        _write_raw_msf(test_paths, [(p, jun, "Y") for p in (1, 2, 3)] + [(1, may, "Y")])
        _write_world_msf(
            test_paths,
            [
                {"id": 1, "eom": may, "primaryexch": "N", "me": 30.0, "ret_lag_dif": 1},
                # the US takes the world file's ret_lag_dif, as other countries do
                {
                    "id": 1,
                    "eom": jun,
                    "primaryexch": "N",
                    "me": 33.0,
                    "me_company": 44.0,
                    "ret_lag_dif": 7,
                },
                {"id": 2, "eom": jun, "primaryexch": "Q", "me": 11.0, "me_company": 44.0},
                {"id": 3, "eom": jun, "primaryexch": "N", "me": 20.0, "me_company": 100.0},
                {
                    "id": 101,
                    "eom": jun,
                    "excntry": "GBR",
                    "source_crsp": 0,
                    "size_grp": "large",
                    "ret_lag_dif": 3,
                },
                {"id": 102, "eom": jun, "excntry": "GBR", "source_crsp": 0, "size_grp": "micro"},
                {
                    "id": 103,
                    "eom": jun,
                    "excntry": "GBR",
                    "source_crsp": 0,
                    "size_grp": "small",
                    "me": None,
                },
                {"id": 104, "eom": jun, "excntry": "DEU", "source_crsp": 0, "size_grp": "mega"},
            ],
        )
        out = load_ff_universe(test_paths, "monthly").collect()
        return {r["id"]: r for r in out.filter(pl.col("date") == jun).iter_rows(named=True)}

    def test_us_rule_columns(self, test_paths):
        r = self._monthly(test_paths)[1]
        assert r["ret_lag_dif"] == 7  # the world file's value
        assert r["me_company"] == 44.0
        assert r["w_cls"] == 30.0  # the class's ME at t-1
        assert r["bp_pool"] is True  # NYSE

    def test_us_bp_pool_is_nyse_only(self, test_paths):
        out = self._monthly(test_paths)
        assert [out[i]["bp_pool"] for i in (1, 2, 3)] == [True, False, True]

    def test_other_country_rule_columns(self, test_paths):
        r = self._monthly(test_paths)[101]
        assert r["ret_lag_dif"] == 3  # the world file's value
        assert r["me_company"] is None
        assert r["w_cls"] is None
        assert r["bp_pool"] is True

    def test_other_country_bp_pool_is_size_grp(self, test_paths):
        """small, large and mega form the pool; micro does not."""
        out = self._monthly(test_paths)
        assert [out[i]["bp_pool"] for i in (101, 102, 103, 104)] == [True, False, True, True]

    def test_other_country_row_without_me_stays(self, test_paths):
        r = self._monthly(test_paths)[103]
        assert r["me"] is None

    def test_daily_pool_columns(self, test_paths):
        """Daily: the US pool is NYSE on the raw dsf primaryexch; elsewhere size_grp
        comes from the previous month's world_msf row, not the month's own."""
        day, eom, prev = date(2020, 6, 15), date(2020, 6, 30), date(2020, 5, 31)
        pl.DataFrame(
            {
                "permno": [1, 2],
                "dlycaldt": [day] * 2,
                "usincflg": ["Y"] * 2,
                "primaryexch": ["N", "Q"],
                "permco": [1, 2],
            }
        ).write_parquet(test_paths.raw_tables_dir / "crsp_dsf_v2.parquet")
        _write_world_dsf(
            test_paths,
            [
                {"id": 1, "date": day, "eom": eom, "ret_lag_dif": 9},
                {"id": 2, "date": day, "eom": eom},
                {
                    "id": 101,
                    "date": day,
                    "eom": eom,
                    "excntry": "GBR",
                    "source_crsp": 0,
                    "ret_lag_dif": 4,
                },
                {"id": 102, "date": day, "eom": eom, "excntry": "GBR", "source_crsp": 0},
            ],
        )
        _write_world_msf(
            test_paths,
            [
                {"id": 101, "eom": prev, "excntry": "GBR", "source_crsp": 0, "size_grp": "mega"},
                {"id": 101, "eom": eom, "excntry": "GBR", "source_crsp": 0, "size_grp": "micro"},
                {"id": 102, "eom": prev, "excntry": "GBR", "source_crsp": 0, "size_grp": "micro"},
                {"id": 102, "eom": eom, "excntry": "GBR", "source_crsp": 0, "size_grp": "mega"},
            ],
        )
        out = load_ff_universe(test_paths, "daily").collect()
        r = {x["id"]: x for x in out.iter_rows(named=True)}
        assert [r[i]["bp_pool"] for i in (1, 2, 101, 102)] == [True, False, True, False]
        assert [r[i]["ret_lag_dif"] for i in (1, 2, 101)] == [9, 1, 4]
        assert r[1]["me_company"] is None


# =============================================================================
# TestFFLoadPanel
# =============================================================================


class TestFFLoadPanel:
    """build_ff_firm_panel: one row per FF firm and date. Class rows come from the
    world files, with usincflg (and daily primaryexch) from raw CIZ. Class returns are
    weighted by class ME at t-1; monthly ME is JKP's me_company; returns are JKP's CRSP returns, not raw CIZ."""

    @staticmethod
    def _run_monthly(test_paths) -> pl.DataFrame:
        may, jun = date(2020, 5, 31), date(2020, 6, 30)
        # permno, mthcaldt, usincflg; permno 3 is foreign-incorporated
        raw = [
            (1, date(2020, 5, 29), "Y"),
            (2, date(2020, 5, 29), "Y"),
            (1, date(2020, 6, 30), "Y"),
            (2, date(2020, 6, 30), "Y"),
            (3, date(2020, 6, 30), "N"),
            (4, date(2020, 6, 30), "Y"),
            (5, date(2020, 6, 30), "Y"),
            (6, date(2020, 6, 30), "Y"),
            (7, date(2020, 6, 30), "Y"),
            (8, date(2020, 6, 30), "Y"),
        ]
        _write_raw_msf(test_paths, raw)
        # id, eom, source_crsp, common, primaryexch, prc, me, me_company, ret.
        # 7 is not a common share, 8 trades off the main exchanges, 9 is Compustat.
        # Companies: 1 and 2; 4, 5 and 6; the rest have one class.
        permco = {2: 1, 5: 4, 6: 4}
        world = [
            (1, may, 1, 1, "Q", 10.0, 30.0, 40.0, 0.0),
            (2, may, 1, 1, "N", 5.0, 10.0, 40.0, 0.0),
            (1, jun, 1, 1, "Q", -12.0, 33.0, 44.0, 0.02),
            (2, jun, 1, 1, "N", 6.0, 11.0, 44.0, 0.10),
            (3, jun, 1, 1, "Q", 8.0, 8.0, 8.0, 0.01),
            # -0.29: a delisting month whose world return includes the delisting return
            (4, jun, 1, 1, "N", 8.0, 20.0, 100.0, -0.29),
            (5, jun, 1, 1, "N", 8.0, 5.0, 100.0, 0.0),
            (6, jun, 1, 1, "N", 8.0, 75.0, 100.0, 0.0),
            (7, jun, 1, 0, "N", 8.0, 8.0, 8.0, 0.01),
            (8, jun, 1, 1, "X", 8.0, 8.0, 8.0, 0.01),
            (9, jun, 0, 1, None, 8.0, 8.0, 8.0, 0.01),
        ]
        _write_world_msf(
            test_paths,
            [
                {
                    "id": i,
                    "eom": d,
                    "source_crsp": src,
                    "common": common,
                    "primaryexch": px,
                    "prc": prc,
                    "me": me,
                    "me_company": mec,
                    "ret": ret,
                    "permco": permco.get(i, i),
                }
                for i, d, src, common, px, prc, me, mec, ret in world
            ],
        )
        out = build_ff_firm_panel(test_paths, "monthly")
        return out.filter(pl.col("date") == jun).sort("id")

    def test_output_columns(self, test_paths):
        out = self._run_monthly(test_paths)
        assert out.columns == [
            "excntry",
            "id",
            "date",
            "ret",
            "me",
            "prc",
            "ret_lag_dif",
            "bp_pool",
        ]

    def test_universe(self, test_paths):
        """3 (raw usincflg N), 7 (world common 0), 8 (exchange X) and 9 (Compustat)
        are out."""
        out = self._run_monthly(test_paths)
        assert out["id"].to_list() == [1, 4]

    def test_two_class_firm(self, test_paths, tolerance):
        """Return weighted by May class ME (30, 10); ME is me_company; NYSE if any
        class is; prc is the largest |price|."""
        r = self._run_monthly(test_paths).row(0, named=True)
        np.testing.assert_allclose(r["ret"], (0.02 * 30 + 0.10 * 10) / 40, **tolerance.STANDARD)
        assert (r["excntry"], r["me"], r["bp_pool"], r["prc"]) == (US_EXCNTRY, 44.0, True, 12.0)
        assert r["ret_lag_dif"] == 1

    def test_three_class_firm_me_is_me_company(self, test_paths):
        """Classes 4, 5 and 6 are one permco: ME is me_company (100). No class has
        ME at t-1, so the firm's return is null."""
        r = self._run_monthly(test_paths).row(1, named=True)
        assert r["me"] == 100.0
        assert r["ret"] is None

    def test_daily_from_world_dsf(self, test_paths):
        day = date(2020, 6, 30)
        pl.DataFrame(
            {
                "permno": [1, 2],
                "dlycaldt": [day] * 2,
                "usincflg": ["Y"] * 2,
                "primaryexch": ["N", "X"],
                "permco": [1, 2],
            }
        ).write_parquet(test_paths.raw_tables_dir / "crsp_dsf_v2.parquet")
        _write_world_dsf(
            test_paths,
            [
                {"id": 1, "date": day, "eom": day, "prc": 10.0, "me": 10.0, "ret": -0.29},
                {"id": 2, "date": day, "eom": day, "prc": 8.0, "me": 8.0, "ret": 0.5},
            ],
        )
        _write_world_msf(test_paths, [{"id": 1, "eom": day}])
        out = build_ff_firm_panel(test_paths, "daily")
        # Daily primaryexch comes from raw CIZ: permno 2 trades on exchange X.
        assert out.select("id", "ret", "me").rows() == [(1, -0.29, 10.0)]

    @staticmethod
    def _write_ex_us(test_paths, rows: list[dict]) -> None:
        jun = date(2020, 6, 30)
        _write_raw_msf(test_paths, [(1, jun, "Y")])
        _write_world_msf(
            test_paths,
            [
                {"eom": jun, "excntry": "GBR", "source_crsp": 0, "size_grp": "large"} | r
                for r in rows
            ],
        )

    def test_single_row_unit_passes_through(self, test_paths):
        """A unit with one row that date keeps that row's values."""
        self._write_ex_us(
            test_paths, [{"id": 100, "prc": 9.0, "me": 120.0, "ret": 0.04, "ret_lag_dif": 2}]
        )
        out = build_ff_firm_panel(test_paths, "monthly")
        assert out.rows(named=True) == [
            {
                "excntry": "GBR",
                "id": 100,
                "date": date(2020, 6, 30),
                "ret": 0.04,
                "me": 120.0,
                "prc": 9.0,
                "ret_lag_dif": 2,
                "bp_pool": True,
            }
        ]

    def test_gvkey_null_stock_stays_in_panel(self, test_paths):
        """A stock without a Compustat link (null gvkey) passes the universe filter
        (it stays in size pools, VW legs and momentum)."""
        self._write_ex_us(
            test_paths,
            [{"id": 1, "gvkey": "g1", "me": 100.0}, {"id": 2, "gvkey": None, "me": 200.0}],
        )
        out = build_ff_firm_panel(test_paths, "monthly")
        assert sorted(out["id"].to_list()) == [1, 2]

    def test_ex_us_row_without_me_stays(self, test_paths):
        """A priced row whose ME is missing stays in the panel and the factor rows."""
        self._write_ex_us(test_paths, [{"id": 1, "me": 100.0}, {"id": 2, "me": None, "prc": 7.0}])
        panel = build_ff_firm_panel(test_paths, "monthly").sort("id")
        assert panel["id"].to_list() == [1, 2]
        assert panel["me"].to_list() == [100.0, None]
        assert panel["prc"].to_list() == [8.0, 7.0]
        assert add_ff_buy_and_hold_weights(panel, "monthly")["id"].to_list() == [1, 2]


# =============================================================================
# TestFFFactorRows
# =============================================================================


def _factor_panel(rows: list[tuple]) -> pl.DataFrame:
    """Panel rows (excntry, id, date, ret, me) with no gaps, as add_ff_buy_and_hold_weights reads."""
    return pl.DataFrame(
        rows,
        schema={
            "excntry": pl.Utf8,
            "id": pl.Int64,
            "date": pl.Date,
            "ret": pl.Float64,
            "me": pl.Float64,
        },
        orient="row",
    ).with_columns(ret_lag_dif=pl.lit(0), bp_pool=pl.lit(False))


def _weights(rows: list[tuple], freq: str = "monthly") -> dict[tuple[int, date], float | None]:
    """add_ff_buy_and_hold_weights on (excntry, id, date, ret, me) rows -> {(id, date): w}."""
    out = add_ff_buy_and_hold_weights(_factor_panel(rows), freq)
    return {(i, d): w for i, d, w in out.select("id", "date", "w").iter_rows()}


class TestFFFactorRows:
    """add_ff_buy_and_hold_weights: buy-and-hold from June in every country. The weight is
    the end-of-June ME grown by the total return since then; a null return
    compounds as 0; a listing gap nulls the weight unless the whole market skipped
    those dates."""

    def test_drifts_from_june_me_by_ret(self):
        """July weight = June ME; later months grow it by ret, not by the ME path
        (the share count doubles in August)."""
        w = _weights(
            [
                ("GBR", 1, date(2020, 6, 30), 0.0, 100.0),
                ("GBR", 1, date(2020, 7, 31), 0.10, 110.0),
                ("GBR", 1, date(2020, 8, 31), None, 400.0),
                ("GBR", 1, date(2020, 9, 30), -0.5, 200.0),
                ("GBR", 1, date(2020, 10, 31), 0.0, 200.0),
            ]
        )
        assert w[(1, date(2020, 7, 31))] == 100.0
        assert w[(1, date(2020, 8, 31))] == pytest.approx(110.0)
        # August ret is null: compounds as 0
        assert w[(1, date(2020, 9, 30))] == pytest.approx(110.0)
        assert w[(1, date(2020, 10, 31))] == pytest.approx(55.0)

    def test_new_year_restarts_from_june(self):
        """The next July starts again from the June ME (row before the year)."""
        rows = [("USA", 1, date(2020, 6, 30), 0.0, 100.0)]
        rows += [("USA", 1, date(2020, m, 28), 0.1, 100.0) for m in range(7, 13)]
        rows += [("USA", 1, date(2021, m, 28), 0.1, 100.0) for m in range(1, 7)]
        rows += [("USA", 1, date(2021, 7, 28), 0.1, 100.0)]
        w = _weights(rows)
        assert w[(1, date(2021, 6, 28))] == pytest.approx(100.0 * 1.1**11)
        assert w[(1, date(2021, 7, 28))] == 100.0

    def test_listing_gap_nulls_weight(self):
        """A stock back after a missing month has no weight that month; the rest
        of the year keeps the June base."""
        w = _weights(
            [
                ("GBR", 1, date(2020, 6, 30), 0.0, 100.0),
                ("GBR", 1, date(2020, 7, 31), 0.10, 110.0),
                ("GBR", 1, date(2020, 9, 30), 0.0, 120.0),
                ("GBR", 1, date(2020, 10, 31), 0.0, 120.0),
                # another stock keeps September a trading month of the market
                ("GBR", 2, date(2020, 8, 31), 0.0, 1.0),
                ("GBR", 2, date(2020, 9, 30), 0.0, 1.0),
            ]
        )
        assert w[(1, date(2020, 9, 30))] is None
        assert w[(1, date(2020, 10, 31))] == pytest.approx(110.0)

    def test_market_closure_is_not_a_gap(self):
        """Daily: a stock whose previous row is the market's previous trading date
        keeps its weight even 7 days later (an exchange closure)."""
        w = _weights(
            [
                ("USA", 1, date(2001, 6, 29), 0.0, 100.0),
                ("USA", 1, date(2001, 9, 10), 0.0, 100.0),
                ("USA", 1, date(2001, 9, 17), -0.05, 95.0),
                ("USA", 2, date(2001, 9, 10), 0.0, 50.0),
                ("USA", 2, date(2001, 9, 17), -0.05, 47.5),
            ],
            freq="daily",
        )
        assert w[(1, date(2001, 9, 17))] == 100.0
        assert w[(2, date(2001, 9, 17))] is None  # no June row: first row's base is null

    def test_nonpositive_lagged_me_nulls_weight(self):
        w = _weights(
            [
                ("GBR", 1, date(2020, 6, 30), 0.0, 100.0),
                ("GBR", 1, date(2020, 7, 31), 0.0, 0.0),
                ("GBR", 1, date(2020, 8, 31), 0.0, 100.0),
            ]
        )
        assert w[(1, date(2020, 7, 31))] == 100.0
        assert w[(1, date(2020, 8, 31))] is None

    def test_stock_listed_mid_year_has_no_weight_until_next_july(self):
        """No row before the year's first row -> no base ME for the rest of the year."""
        w = _weights(
            [
                ("GBR", 1, date(2020, 9, 30), 0.0, 100.0),
                ("GBR", 1, date(2020, 10, 31), 0.0, 100.0),
                ("GBR", 1, date(2021, 6, 30), 0.0, 100.0),
                ("GBR", 1, date(2021, 7, 31), 0.0, 100.0),
            ]
        )
        assert w[(1, date(2020, 10, 31))] is None
        assert w[(1, date(2021, 7, 31))] == 100.0

    @pytest.mark.parametrize(("lag_dif", "kept"), [(1, True), (2, False)])
    def test_gap_spanning_return_nulled_monthly(self, lag_dif, kept):
        """A return over more than one month is nulled, the row stays."""
        panel = _factor_panel(
            [
                ("GBR", 1, date(2020, 6, 30), 0.0, 100.0),
                ("GBR", 1, date(2020, 8, 31), 0.1, 110.0),
            ]
        ).with_columns(ret_lag_dif=pl.Series([0, lag_dif]))
        out = add_ff_buy_and_hold_weights(panel, "monthly")
        assert out["id"].to_list() == [1, 1]
        assert (out["ret"][1] is not None) is kept

    @pytest.mark.parametrize(("lag_dif", "kept"), [(5, True), (6, False)])
    def test_gap_spanning_return_nulled_daily(self, lag_dif, kept):
        """Daily: more than 5 days nulls the return."""
        panel = _factor_panel(
            [
                ("GBR", 1, date(2020, 6, 30), 0.0, 100.0),
                ("GBR", 1, date(2020, 7, 7), 0.1, 110.0),
            ]
        ).with_columns(ret_lag_dif=pl.Series([0, lag_dif]))
        out = add_ff_buy_and_hold_weights(panel, "daily")
        assert out["id"].to_list() == [1, 1]
        assert (out["ret"][1] is not None) is kept

    def test_rows_without_me_kept(self):
        """A row without ME at t stays in every country."""
        panel = _factor_panel(
            [
                ("GBR", 1, date(2020, 7, 31), 0.0, None),
                ("GBR", 2, date(2020, 7, 31), 0.0, 5.0),
                ("USA", 3, date(2020, 7, 31), 0.0, None),
            ]
        )
        assert add_ff_buy_and_hold_weights(panel, "monthly")["id"].sort().to_list() == [1, 2, 3]


# =============================================================================
# TestFFMembership
# =============================================================================


def _june_with_bp(
    beme_vals: list[float],
    me_vals: list[float],
    sizemedn: float,
    bm_lo: float,
    bm_hi: float,
    excntry: str = US_EXCNTRY,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A June frame and its breakpoint frame for the B/M sort."""
    n = len(beme_vals)
    june = pl.DataFrame(
        {
            "excntry": [excntry] * n,
            "date": [date(2020, 6, 30)] * n,
            "id": list(range(n)),
            "beme": beme_vals,
            "me": me_vals,
        }
    )
    breaks = pl.DataFrame(
        {
            "excntry": [excntry],
            "date": [date(2020, 6, 30)],
            "sizemedn": [sizemedn],
            "bm_lo": [bm_lo],
            "bm_hi": [bm_hi],
        }
    )
    return june, breaks


class TestFFMembership:
    """Tests for assign_ff_portfolios()."""

    def test_size_split_correct(self):
        """Firms below sizemedn -> S; at or above -> B."""
        june, breaks = _june_with_bp(
            [1.0, 1.0, 5.0, 5.0], [1.0, 3.0, 1.0, 3.0], sizemedn=2.0, bm_lo=2.0, bm_hi=4.0
        )
        result = assign_ff_portfolios(june, breaks, ["bm"])
        # me=1 < 2=sizemedn -> S; me=3 >= 2 -> B
        assert [p[0] for p in result["bm_port"]] == ["S", "B", "S", "B"]

    def test_bm_portfolio_labels(self):
        """beme <= lo -> L, lo < beme <= hi -> M, beme > hi -> H."""
        june, breaks = _june_with_bp(
            [1.0, 3.0, 5.0], [1.0, 1.0, 1.0], sizemedn=2.0, bm_lo=2.0, bm_hi=4.0
        )
        result = assign_ff_portfolios(june, breaks, ["bm"])
        assert result["bm_port"].to_list() == ["SL", "SM", "SH"]

    def test_ineligible_rows_get_null_port(self):
        """Rows failing eligibility (beme <= 0) get a null port."""
        june, breaks = _june_with_bp([-1.0, 1.0], [1.0, 1.0], sizemedn=0.5, bm_lo=0.5, bm_hi=1.5)
        result = assign_ff_portfolios(june, breaks, ["bm"])
        assert result["bm_port"][0] is None
        assert result["bm_port"][1] is not None

    def test_missing_breakpoint_null_port(self):
        """If the sort's breakpoints are null, eligible rows get a null port."""
        june, breaks = _june_with_bp([2.0], [1.0], sizemedn=0.5, bm_lo=None, bm_hi=None)
        breaks = breaks.cast({"bm_lo": pl.Float64, "bm_hi": pl.Float64})
        assert assign_ff_portfolios(june, breaks, ["bm"])["bm_port"][0] is None

    def test_country_date_without_breakpoints_dropped(self):
        """Units in a country-date with no breakpoint row are not in the output."""
        june, breaks = _june_with_bp([1.0, 2.0], [1.0, 1.0], 0.5, 1.0, 2.0)
        june = pl.concat([june, june.with_columns(excntry=pl.lit("GBR"), id=pl.col("id") + 10)])
        assert set(assign_ff_portfolios(june, breaks, ["bm"])["excntry"]) == {US_EXCNTRY}

    def test_output_columns(self):
        """Output keeps frame's columns plus <sort>_port; no breakpoint or flag columns."""
        june, breaks = _june_with_bp([1.0, 2.0], [1.0, 1.0], 0.5, 1.0, 2.0)
        result = assign_ff_portfolios(june, breaks, ["bm"])
        assert result.columns == [*june.columns, "bm_port"]

    def test_two_sorts(self):
        """Each sort gets its own column: OP labels W/N/R after the size letter."""
        june, breaks = _june_with_bp([1.0, 1.0], [1.0, 3.0], 2.0, 0.5, 1.5)
        june = june.with_columns(dec_me=pl.lit(1.0), be=pl.lit(1.0), op=pl.Series([0.1, 0.9]))
        breaks = breaks.with_columns(op_lo=pl.lit(0.3), op_hi=pl.lit(0.6))
        result = assign_ff_portfolios(june, breaks, ["bm", "op"])
        assert result["bm_port"].to_list() == ["SM", "BM"]
        assert result["op_port"].to_list() == ["SW", "BR"]


# =============================================================================
# TestFFLeg
# =============================================================================


_BM_BUCKETS = ["bm_SL", "bm_SM", "bm_SH", "bm_BL", "bm_BM", "bm_BH"]


def _leg_frame(rows: list[dict]) -> pl.DataFrame:
    """Frame of (excntry, id, date, ret, w, bm_port) rows for compute_ff_portfolio_returns."""
    schema = {
        "excntry": pl.Utf8,
        "id": pl.Int64,
        "date": pl.Date,
        "ret": pl.Float64,
        "w": pl.Float64,
        "bm_port": pl.Utf8,
    }
    return pl.DataFrame(
        [{"excntry": US_EXCNTRY, "date": date(2020, 7, 31), "bm_port": "SH"} | r for r in rows],
        schema=schema,
    )


class TestFFLeg:
    """Tests for compute_ff_portfolio_returns() on the B/M sort."""

    def test_vw_return_formula(self, tolerance):
        """VW return = sum(ret*w)/sum(w) per bucket: (0.1*1 + 0.2*3) / 4 = 0.175."""
        frame = _leg_frame([{"id": 1, "ret": 0.1, "w": 1.0}, {"id": 2, "ret": 0.2, "w": 3.0}])
        result = compute_ff_portfolio_returns(frame, "bm", 1)
        np.testing.assert_allclose(result["bm_SH"][0], 0.175, **tolerance.TIGHT)

    def test_missing_buckets_filled_with_null(self):
        """Buckets absent from data are filled with null (not dropped)."""
        result = compute_ff_portfolio_returns(
            _leg_frame([{"id": 1, "ret": 0.1, "w": 1.0}]), "bm", 1
        )
        for col in _BM_BUCKETS:
            assert col in result.columns
        assert result["bm_SL"][0] is None

    def test_us_below_min_stocks_gated(self):
        """The US is gated like every country: a 1-stock bucket is null."""
        result = compute_ff_portfolio_returns(
            _leg_frame([{"id": 1, "ret": 0.05, "w": 1.0}]), "bm", FACTOR_MIN_STOCKS_PF
        )
        assert result["bm_SH"][0] is None

    def test_row_below_min_stocks_nulled(self):
        """A bucket with n < FACTOR_MIN_STOCKS_PF gets null."""
        rows = [
            {"excntry": "GBR", "id": i, "ret": 0.1, "w": 1.0}
            for i in range(FACTOR_MIN_STOCKS_PF - 1)
        ]
        result = compute_ff_portfolio_returns(_leg_frame(rows), "bm", FACTOR_MIN_STOCKS_PF)
        gbr = result.filter(pl.col("excntry") == "GBR")
        assert len(gbr) > 0
        assert gbr["bm_SH"][0] is None

    def test_row_at_min_stocks_not_nulled(self):
        """A bucket with n == FACTOR_MIN_STOCKS_PF passes through."""
        rows = [
            {"excntry": "GBR", "id": i, "ret": 0.1, "w": 1.0} for i in range(FACTOR_MIN_STOCKS_PF)
        ]
        result = compute_ff_portfolio_returns(_leg_frame(rows), "bm", FACTOR_MIN_STOCKS_PF)
        gbr = result.filter(pl.col("excntry") == "GBR")
        assert len(gbr) == 1
        assert gbr["bm_SH"][0] is not None

    def test_null_ret_excluded(self):
        """Rows with null ret are excluded from VW computation."""
        frame = _leg_frame([{"id": 1, "ret": None, "w": 10.0}, {"id": 2, "ret": 0.1, "w": 1.0}])
        np.testing.assert_allclose(
            compute_ff_portfolio_returns(frame, "bm", 1)["bm_SH"][0], 0.1, rtol=1e-9
        )

    def test_zero_w_excluded(self):
        """Rows with w == 0 are excluded from VW computation."""
        frame = _leg_frame([{"id": 1, "ret": 999.0, "w": 0.0}, {"id": 2, "ret": 0.05, "w": 1.0}])
        np.testing.assert_allclose(
            compute_ff_portfolio_returns(frame, "bm", 1)["bm_SH"][0], 0.05, rtol=1e-9
        )

    def test_null_port_excluded(self):
        """Rows with no membership in the sort are excluded."""
        frame = _leg_frame(
            [{"id": 1, "ret": 999.0, "w": 1.0, "bm_port": None}, {"id": 2, "ret": 0.05, "w": 1.0}]
        )
        np.testing.assert_allclose(
            compute_ff_portfolio_returns(frame, "bm", 1)["bm_SH"][0], 0.05, rtol=1e-9
        )

    def test_output_schema(self):
        """Output has excntry, date, and all 6 prefixed B/M buckets."""
        result = compute_ff_portfolio_returns(
            _leg_frame([{"id": 1, "ret": 0.1, "w": 1.0}]), "bm", 1
        )
        assert result.columns == ["excntry", "date", *_BM_BUCKETS]


# =============================================================================
# TestFFReturns
# =============================================================================


def _full_annual(bm: bool = True) -> list[tuple]:
    """One stock per size x label bucket of the B/M (if bm), OP and INV sorts, each
    stock in one sort only: (ret, bm_port, op_port, inv_port)."""
    out: list[tuple] = []
    for sz in "SB":
        if bm:
            out += [(0.01, f"{sz}{lab}", None, None) for lab in "LMH"]
        out += [(0.01, None, f"{sz}{lab}", None) for lab in "WNR"]
        out += [(0.01, None, None, f"{sz}{lab}") for lab in "CNA"]
    return out


_FULL_MOM = [(0.01, "SL"), (0.01, "SH"), (0.01, "BL"), (0.01, "BH")]


_ANNUAL = ["bm", "op", "inv"]


def _returns(
    annual: list[tuple],
    mom: list[tuple] = _FULL_MOM,
    excntry: str = US_EXCNTRY,
    min_stocks_pf: int = 1,
    d: date = date(2020, 7, 31),
) -> pl.DataFrame:
    """compute_ff_factor_returns on stocks held July-June from the June 2020 memberships.

    annual: (ret, bm_port, op_port, inv_port) per stock; mom: (ret, mom_port) per
    stock. Every stock has weight 1.
    """
    n = len(annual)
    rows = pl.DataFrame(
        {
            "excntry": [excntry] * n,
            "id": list(range(n)),
            "date": [d] * n,
            "ret": [a[0] for a in annual],
            "w": [1.0] * n,
        }
    )
    members = pl.DataFrame(
        {
            "excntry": [excntry] * n,
            "id": list(range(n)),
            "date": [date(2020, 6, 30)] * n,
            "bm_port": [a[1] for a in annual],
            "op_port": [a[2] for a in annual],
            "inv_port": [a[3] for a in annual],
        },
        schema_overrides={"bm_port": pl.Utf8, "op_port": pl.Utf8, "inv_port": pl.Utf8},
    )
    mom_frame = pl.DataFrame(
        {
            "excntry": [excntry] * len(mom),
            "id": list(range(len(mom))),
            "date": [d] * len(mom),
            "ret": [m[0] for m in mom],
            "w": [1.0] * len(mom),
            "mom_port": [m[1] for m in mom],
        }
    )
    return compute_ff_factor_returns(rows, members, mom_frame, _ANNUAL, min_stocks_pf)


class TestFFReturns:
    """Tests for compute_ff_factor_returns()."""

    def test_output_columns(self):
        """Output is [excntry, date, smb_ff3, smb_ff5, hml_ff, rmw_ff, cma_ff, umd_ff]."""
        result = _returns(_full_annual())
        assert result.columns == [
            "excntry",
            "date",
            "smb_ff3",
            "smb_ff5",
            "hml_ff",
            "rmw_ff",
            "cma_ff",
            "umd_ff",
        ]

    def test_hml_is_high_minus_low(self, tolerance):
        """HML = avg(SH,BH) - avg(SL,BL): one stock per bucket, equal weights."""
        # BM leg: SH=0.30, BH=0.10, SL=0.01, BL=0.02 -> HML = 0.20 - 0.015 = 0.185
        bm = [
            (0.30, "SH", None, None),
            (0.10, "BH", None, None),
            (0.05, "SM", None, None),
            (0.05, "BM", None, None),
            (0.01, "SL", None, None),
            (0.02, "BL", None, None),
        ]
        result = _returns([*bm, *_full_annual(bm=False)])
        np.testing.assert_allclose(result["hml_ff"][0], 0.185, **tolerance.STANDARD)

    def test_smb_ff3_vs_smb_ff5_differ(self, tolerance):
        """smb_ff3 uses the B/M sort only; smb_ff5 averages the B/M, OP and INV size
        spreads. The three sorts hold disjoint stocks with different size spreads,
        so the two SMBs are not just present but numerically distinct."""
        # fmt: off
        stocks = [
            # B/M-only stocks: one per (size, btm) bucket
            (0.10, "SL", None, None), (0.05, "SM", None, None), (0.02, "SH", None, None),
            (0.01, "BL", None, None), (0.01, "BM", None, None), (0.01, "BH", None, None),
            # OP-only stocks: one per (size, op) bucket
            (0.01, None, "SW", None), (0.01, None, "SN", None), (0.01, None, "SR", None),
            (0.01, None, "BW", None), (0.01, None, "BN", None), (0.01, None, "BR", None),
            # INV-only stocks: one per (size, inv) bucket
            (0.01, None, None, "SC"), (0.01, None, None, "SN"), (0.01, None, None, "SA"),
            (0.20, None, None, "BC"), (0.20, None, None, "BN"), (0.20, None, None, "BA"),
        ]
        # fmt: on
        result = _returns(stocks)
        smb_bm = (0.10 + 0.05 + 0.02) / 3 - (0.01 + 0.01 + 0.01) / 3
        smb_op = (0.01 + 0.01 + 0.01) / 3 - (0.01 + 0.01 + 0.01) / 3
        smb_inv = (0.01 + 0.01 + 0.01) / 3 - (0.20 + 0.20 + 0.20) / 3
        np.testing.assert_allclose(result["smb_ff3"][0], smb_bm, **tolerance.STANDARD)
        np.testing.assert_allclose(
            result["smb_ff5"][0], (smb_bm + smb_op + smb_inv) / 3, **tolerance.STANDARD
        )
        assert abs(result["smb_ff3"][0] - result["smb_ff5"][0]) > 1e-6

    def test_null_propagation_when_bucket_missing(self):
        """When a required bucket is absent, the factor is null (no ignore_nulls)."""
        # no B/M 'H' bucket stocks -> HML is null
        bm = [(0.05, f"{sz}{lab}", None, None) for sz in "SB" for lab in "LM"]
        result = _returns([*bm, *_full_annual(bm=False)])
        assert result["hml_ff"][0] is None

    def test_june_memberships_hold_july_to_june(self):
        """Memberships formed in June 2020 apply to July 2020 through June 2021 and
        those formed in June 2021 from July 2021; a stock with no membership in the
        year is not in the legs."""
        ids = [1, 2, 3, 4, 999]
        rets = [0.1, 0.2, 0.3, 0.4, 100.0]
        rows = pl.concat(
            [
                pl.DataFrame(
                    {
                        "excntry": [US_EXCNTRY] * 5,
                        "id": ids,
                        "date": [d] * 5,
                        "ret": rets,
                        "w": [1.0] * 5,
                    }
                )
                for d in (date(2020, 7, 31), date(2021, 6, 30), date(2021, 7, 31))
            ]
        )
        same = {"op_port": ["SW", "SR", "BW", "BR"], "inv_port": ["SC", "SA", "BC", "BA"]}
        members = pl.concat(
            [
                pl.DataFrame(
                    {
                        "excntry": [US_EXCNTRY] * 4,
                        "id": [1, 2, 3, 4],
                        "date": [june] * 4,
                        "bm_port": bm,
                    }
                    | same
                )
                for june, bm in (
                    (date(2020, 6, 30), ["SL", "SH", "BL", "BH"]),
                    (date(2021, 6, 30), ["SH", "SL", "BH", "BL"]),
                )
            ]
        )
        mom = rows.filter(pl.col("id") < 5).with_columns(
            mom_port=pl.col("id").replace_strict(
                {1: "SL", 2: "SH", 3: "BL", 4: "BH"}, return_dtype=pl.Utf8
            )
        )
        out = compute_ff_factor_returns(rows, members, mom, _ANNUAL, 1)
        assert out["date"].to_list() == sorted(out["date"].to_list())
        hml = dict(out.select("date", "hml_ff").iter_rows())
        # 2020 memberships: (0.2+0.4)/2 - (0.1+0.3)/2 = 0.1; 2021 memberships: -0.1
        assert hml[date(2020, 7, 31)] == pytest.approx(0.1)
        assert hml[date(2021, 6, 30)] == pytest.approx(0.1)
        assert hml[date(2021, 7, 31)] == pytest.approx(-0.1)

    def test_umd_arithmetic(self, tolerance):
        """umd_ff = 0.5*(SH+BH) - 0.5*(SL+BL) with known returns."""
        mom = [(0.30, "SH"), (0.20, "BH"), (0.02, "SL"), (0.01, "BL"), (0.10, "SN"), (0.10, "BN")]
        result = _returns(_full_annual(), mom)
        # 0.5*(0.30+0.20) - 0.5*(0.02+0.01) = 0.25 - 0.015 = 0.235
        np.testing.assert_allclose(result["umd_ff"][0], 0.235, **tolerance.STANDARD)

    def test_umd_null_when_bucket_missing(self):
        """If the SH or BH bucket is absent, umd_ff is null."""
        mom = [(0.02, "SL"), (0.01, "BL"), (0.10, "SN"), (0.10, "BN")]
        assert _returns(_full_annual(), mom)["umd_ff"][0] is None

    def test_us_below_min_stocks_gated(self):
        """The US is gated like every country: 1 stock per bucket gives null umd_ff."""
        result = _returns(_full_annual(), min_stocks_pf=FACTOR_MIN_STOCKS_PF)
        assert result["umd_ff"][0] is None


# =============================================================================
# TestFFMomSignalMonthly
# =============================================================================


def _with_lag(panel: pl.DataFrame, freq: str) -> pl.DataFrame:
    """Add the panel's ret_lag_dif as world_msf and world_dsf carry it: 1 for the US
    (JKP's value on every CRSP row), and for other countries the periods since the
    stock's previous row."""
    by = ["excntry", "id"]
    if freq == "monthly":
        period = pl.col("date").dt.year() * 12 + pl.col("date").dt.month()
    else:
        period = pl.col("date").cast(pl.Int64)
    return panel.sort([*by, "date"]).with_columns(
        ret_lag_dif=pl.when(pl.col("excntry") == US_EXCNTRY)
        .then(1)
        .otherwise(period.diff().over(by).fill_null(1))
    )


def _run_mom(panel: pl.DataFrame, freq: str, rows: pl.DataFrame | None = None) -> pl.DataFrame:
    """compute_ff_momentum_signal on a test panel; rows default to the whole panel."""
    full = _with_lag(panel, freq)
    rows = full if rows is None else rows
    return compute_ff_momentum_signal(
        full, rows.select("excntry", "id", "date", "ret", "bp_pool"), freq
    )


def _run_monthly(panel: pl.DataFrame, rows: pl.DataFrame | None = None) -> pl.DataFrame:
    return _run_mom(panel, "monthly", rows)


def _run_daily(panel: pl.DataFrame) -> pl.DataFrame:
    return _run_mom(panel, "daily")


_MOM_COLS = {"excntry", "id", "date", "ret", "bp_pool", "me", "w", "mom_2_12"}


def _make_monthly_panel(
    months: int,
    base_ret: float = 0.01,
    excntry: str = US_EXCNTRY,
    stock_id: int = 1,
    ret_overrides: dict[int, float | None] | None = None,
) -> pl.DataFrame:
    """Consecutive monthly panel for a single stock."""
    dates = [date(2018 + (m // 12), (m % 12) + 1, 28) for m in range(months)]
    rets: list[float | None] = [base_ret] * months
    for i, v in (ret_overrides or {}).items():
        rets[i] = v
    return pl.DataFrame(
        {
            "excntry": [excntry] * months,
            "id": [stock_id] * months,
            "date": dates,
            "ret": rets,
            "me": [1.0] * months,
            "prc": [1.0] * months,
            "bp_pool": [True] * months,
        },
        schema_overrides={"ret": pl.Float64},
    )


class TestFFMomSignalMonthly:
    """Tests for compute_ff_momentum_signal() on monthly data."""

    def test_output_schema(self):
        """Output has the mom-signal columns."""
        result = _run_monthly(_make_monthly_panel(months=20))
        assert set(result.columns) == _MOM_COLS

    def test_bp_pool_is_lagged(self):
        """UMD forms at t-1, so the breakpoint pool is the row's at t-1."""
        panel = _make_monthly_panel(months=3).with_columns(bp_pool=pl.Series([True, False, True]))
        result = _run_monthly(panel).sort("date")
        assert result["bp_pool"].to_list() == [False, True, False]

    def test_minus_100_return_nulls_every_window_holding_it(self):
        """A -100% return at month 5 nulls the windows t-12..t-2 that hold it
        (t = 13..17); later windows compound normally."""
        panel = _make_monthly_panel(months=20, ret_overrides={5: -1.0})
        mom = _run_monthly(panel).sort("date")["mom_2_12"].to_list()
        assert all(m is None for m in mom[:18])
        assert mom[18] == pytest.approx(1.01**11 - 1, abs=1e-12)
        assert mom[19] == pytest.approx(1.01**11 - 1, abs=1e-12)

    def test_me_and_w_are_lagged_me(self):
        """UMD sizes and weights by ME at t-1."""
        panel = _make_monthly_panel(months=3).with_columns(me=pl.Series([1.0, 2.0, 4.0]))
        result = _run_monthly(panel).sort("date")
        assert result["me"].to_list() == [None, 1.0, 2.0]
        assert result["w"].to_list() == result["me"].to_list()

    def test_insufficient_history_yields_null_mom(self):
        """Less than 13 months of data -> mom_2_12 is null (not enough lags)."""
        result = _run_monthly(_make_monthly_panel(months=12))
        # All rows should have null mom_2_12 (need 13 months for lag 1..12)
        assert result["mom_2_12"].is_null().all()

    def test_sufficient_history_yields_non_null_mom(self):
        """13+ months of data -> later rows have non-null mom_2_12."""
        result = _run_monthly(_make_monthly_panel(months=20))
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0

    @staticmethod
    def _interior_null_panel(excntry: str) -> pl.DataFrame:
        """20 consecutive months, ret null at month 10 (an interior 2-12 lag
        for the final row; the row itself exists, mimicking a CIZ
        missing-price month that legacy CRSP coded -99)."""
        return _make_monthly_panel(months=20, excntry=excntry, ret_overrides={10: None})

    def test_us_interior_null_month_tolerated(self, tolerance):
        """US: a null interior window month (row present) stays eligible and
        compounds as 0, per French's -99 tolerance for t-12..t-3."""
        result = _run_monthly(self._interior_null_panel(US_EXCNTRY))
        last_row = result.sort("date").tail(1)
        # 11 window months, one zeroed: (1.01)^10 - 1
        np.testing.assert_allclose(last_row["mom_2_12"][0], 1.01**10 - 1, **tolerance.STANDARD)

    def test_row_interior_null_month_tolerated(self, tolerance):
        """ROW: same gate as the US; the null month compounds as 0."""
        result = _run_monthly(self._interior_null_panel("GBR"))
        last_row = result.sort("date").tail(1)
        np.testing.assert_allclose(last_row["mom_2_12"][0], 1.01**10 - 1, **tolerance.STANDARD)

    def test_me_null_when_gap(self):
        """me (ME at t-1) is null when there's a calendar month gap."""
        # Two rows with a 2-month gap -> me for the second row is null
        panel = pl.DataFrame(
            {
                "excntry": [US_EXCNTRY, US_EXCNTRY],
                "id": [1, 1],
                "date": [date(2020, 1, 31), date(2020, 3, 31)],  # skip Feb
                "ret": [0.01, 0.02],
                "me": [1.0, 2.0],
                "prc": [1.0, 1.0],
                "bp_pool": [True, True],
            }
        )
        result = _run_monthly(panel).sort("date")
        mar_row = result.filter(pl.col("date") == date(2020, 3, 31))
        assert mar_row["me"][0] is None

    def test_mom_signal_2_12_skip(self, tolerance):
        """With constant 2% monthly return, compounded 2-12 (skip 1) ~ 1.02^11 - 1."""
        result = _run_monthly(_make_monthly_panel(months=24, base_ret=0.02))
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0
        # 11 months compounded: lags 2 through 12 = 11 months
        expected = (1.02**11) - 1
        np.testing.assert_allclose(non_null["mom_2_12"][0], expected, **tolerance.STANDARD)

    def test_multi_stock_independent(self):
        """Two stocks compute signals independently."""
        p1 = _make_monthly_panel(months=20, base_ret=0.01, stock_id=1)
        p2 = _make_monthly_panel(months=20, base_ret=0.05, stock_id=2)
        result = _run_monthly(pl.concat([p1, p2]))
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
            prc=pl.when(pl.int_range(pl.len()) == 6).then(None).otherwise(pl.col("prc"))
        )
        assert self._last(_run_monthly(panel))["mom_2_12"] is None

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    @pytest.mark.parametrize(("halt", "eligible"), [(range(9, 14), True), (range(9, 15), False)])
    def test_halt_tolerance(self, excntry, halt, eligible):
        """Every country, the US included: a halt is missing rows; 5 missing of 11
        window months (6 left) stays eligible, 6 missing does not."""
        panel = _make_monthly_panel(months=20, excntry=excntry).filter(
            ~pl.int_range(pl.len()).is_in(list(halt))
        )
        last = self._last(_run_monthly(panel))
        assert (last["mom_2_12"] is not None) is eligible

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    @pytest.mark.parametrize("missing", [6, 17])
    def test_halt_at_endpoint_ineligible(self, excntry, missing):
        """Endpoints stay strict (price at t-13, good return at t-2)."""
        panel = _make_monthly_panel(months=20, excntry=excntry).filter(
            pl.int_range(pl.len()) != missing
        )
        assert self._last(_run_monthly(panel))["mom_2_12"] is None

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    def test_compounds_source_returns(self, excntry, tolerance):
        """Every country compounds its source returns over t-12..t-2 (in the US the
        FF firm's own panel returns, a company's combined return)."""
        result = _run_monthly(_make_monthly_panel(months=20, excntry=excntry))
        np.testing.assert_allclose(
            self._last(result)["mom_2_12"], 1.01**11 - 1, **tolerance.STANDARD
        )

    def test_row_halt_spanning_return_included(self):
        """Non-US: the first return after a halt (p_end / p_start - 1) enters the
        compounded window return in full."""
        panel = _make_monthly_panel(
            months=20, base_ret=0.0, excntry="GBR", ret_overrides={13: 0.5}
        ).filter(~pl.int_range(pl.len()).is_between(10, 12))
        last = self._last(_run_monthly(panel))
        assert abs(last["mom_2_12"] - 0.5) < 1e-9

    def test_row_price_without_me_at_start_eligible(self):
        """Non-US: eligibility is judged on the panel, so a priced row whose me is
        null (not a factor row) still counts."""
        full = _make_monthly_panel(months=20, excntry="GBR").with_columns(
            me=pl.when(pl.int_range(pl.len()) == 6).then(None).otherwise(pl.col("me"))
        )
        rows = full.filter(pl.col("me").is_not_null())
        result = _run_monthly(full, rows)
        assert self._last(result)["mom_2_12"] is not None


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
            "prc": [1.0] * n_days,
            "bp_pool": [True] * n_days,
        },
        schema_overrides={"ret": pl.Float64},
    )


class TestFFMomSignalDaily:
    """Tests for compute_ff_momentum_signal() on daily data."""

    def test_output_schema(self):
        """Output has the mom-signal columns."""
        result = _run_daily(_make_daily_panel(n_days=300))
        assert set(result.columns) == _MOM_COLS

    def test_insufficient_history_yields_null_mom(self):
        """Fewer than 252 days -> all mom_2_12 null (not enough history)."""
        result = _run_daily(_make_daily_panel(n_days=250))
        assert result["mom_2_12"].is_null().all()

    def test_sufficient_history_has_non_null(self):
        """300 days -> rows past lookback window have non-null mom_2_12."""
        result = _run_daily(_make_daily_panel(n_days=300))
        non_null = result.filter(pl.col("mom_2_12").is_not_null())
        assert len(non_null) > 0

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    def test_null_ret_at_window_end_ineligible(self, excntry):
        """A missing return at t-21, the window end, fails the good-return check."""
        n = 300
        panel = _make_daily_panel(n_days=n, excntry=excntry, ret_overrides={n - 22: None})
        last_row = _run_daily(panel).sort("date").tail(1)
        assert last_row["mom_2_12"][0] is None

    @staticmethod
    def _null_in_window_panel(excntry: str) -> pl.DataFrame:
        """300 consecutive days, ret null at an interior lookback-window day
        for the last row."""
        return _make_daily_panel(n_days=300, excntry=excntry, ret_overrides={200: None})

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    def test_null_ret_in_window_tolerated(self, excntry, tolerance):
        """Interior missing-price day is allowed and compounds as 0, in every country
        (French: 'any missing returns from day t-250 to t-22 must be -99.0')."""
        result = _run_daily(self._null_in_window_panel(excntry))
        last_row = result.sort("date").tail(1)
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
        last_row = _run_daily(panel).sort("date").tail(1)
        assert abs(last_row["mom_2_12"][0] - 0.5) < 1e-9

    def test_multi_country_independent(self):
        """Two countries compute trading-day index independently."""
        p_usa = _make_daily_panel(300, base_ret=0.001, excntry="USA", stock_id=1)
        p_gbr = _make_daily_panel(300, base_ret=0.003, excntry="GBR", stock_id=1)
        result = _run_daily(pl.concat([p_usa, p_gbr]))
        usa = result.filter((pl.col("excntry") == "USA") & pl.col("mom_2_12").is_not_null())
        gbr = result.filter((pl.col("excntry") == "GBR") & pl.col("mom_2_12").is_not_null())
        assert len(usa) > 0 and len(gbr) > 0
        # GBR has higher per-day return -> higher momentum
        assert gbr["mom_2_12"].mean() > usa["mom_2_12"].mean()

    # Last row of a 300-day panel is index 299: t-251 = 48, t-21 = 278, window = 49..278.
    # A second, complete stock keeps every date in the country's trading calendar.

    @staticmethod
    def _with_full_peer(panel: pl.DataFrame) -> pl.DataFrame:
        peer = _make_daily_panel(300, excntry=panel["excntry"][0], stock_id=2)
        return pl.concat([panel, peer])

    @staticmethod
    def _last_id1(result: pl.DataFrame) -> dict:
        return result.filter(pl.col("id") == 1).sort("date").row(-1, named=True)

    def test_us_missing_price_at_start_ineligible(self):
        """US: no price at t-251 fails French's start check."""
        panel = _make_daily_panel(300).with_columns(
            prc=pl.when(pl.int_range(pl.len()) == 48).then(None).otherwise(pl.col("prc"))
        )
        assert self._last_id1(_run_daily(self._with_full_peer(panel)))["mom_2_12"] is None

    def test_us_skip_window_rows_not_required(self):
        """US: rows t-20..t-2 lie outside French's window and may be missing."""
        panel = _make_daily_panel(300).filter(~pl.int_range(pl.len()).is_between(279, 297))
        assert self._last_id1(_run_daily(self._with_full_peer(panel)))["mom_2_12"] is not None

    def test_row_halt_spanning_return_included(self):
        """Non-US: a 51-day halt is tolerated and the first return after it
        (p_end / p_start - 1) enters the compounded window return in full."""
        panel = _make_daily_panel(
            300, base_ret=0.0, excntry="GBR", ret_overrides={151: 0.5}
        ).filter(~pl.int_range(pl.len()).is_between(100, 150))
        last = self._last_id1(_run_daily(self._with_full_peer(panel)))
        assert abs(last["mom_2_12"] - 0.5) < 1e-9

    @pytest.mark.parametrize("excntry", [US_EXCNTRY, "GBR"])
    @pytest.mark.parametrize(("n_missing", "eligible"), [(115, True), (116, False)])
    def test_halt_min_days(self, excntry, n_missing, eligible):
        """Every country, the US included: at least 115 of the 230 window days
        must be present."""
        panel = _make_daily_panel(300, excntry=excntry).filter(
            ~pl.int_range(pl.len()).is_between(100, 100 + n_missing - 1)
        )
        last = self._last_id1(_run_daily(self._with_full_peer(panel)))
        assert (last["mom_2_12"] is not None) is eligible
