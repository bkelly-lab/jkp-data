"""Tests for the accounting-stage column blocks in ``aux_functions``.

Covers ``add_accounting_misc_cols_1``, ``add_accounting_misc_cols_2``,
``add_liq_and_efficiency_ratios``, ``add_profit_scaled_by_lagged_vars`` and
``financial_soundness_and_misc_ratios_exps``.

Frames are monthly panels for two firms ("A" and "B"). Every input column is
the linear series ``base + slope * t`` (firm A slope 1, firm B slope 2), so the
lagged expectations are computed by hand from the characteristic definitions
(``accounting_chars.sas``), never from the code under test. ``count`` is the
1-based observation number per firm, as ``create_acc_chars`` builds it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import (
    add_accounting_misc_cols_1,
    add_accounting_misc_cols_2,
    add_liq_and_efficiency_ratios,
    add_profit_scaled_by_lagged_vars,
    financial_soundness_and_misc_ratios_exps,
)

pytestmark = pytest.mark.unit

DEFAULT_BASE = 100.0
SLOPE = {"A": 1.0, "B": 2.0}
Gen = Callable[[str, str, int], float | None]


def _lin(base: dict[str, float] | None = None) -> Gen:
    """Generator for ``base[col] (default 100) + SLOPE[firm] * t``."""
    base = base or {}

    def gen(name: str, firm: str, t: int) -> float | None:
        return base.get(name, DEFAULT_BASE) + SLOPE[firm] * t

    return gen


def _panel(cols: Sequence[str], n: int, gen: Gen | None = None) -> pl.DataFrame:
    """Two firms x ``n`` months; ``_t`` is the 0-based month, ``count`` is ``_t + 1``."""
    gen = gen or _lin()
    firms = [f for f in ("A", "B") for _ in range(n)]
    ts = [t for _ in ("A", "B") for t in range(n)]
    data: dict[str, list] = {
        "gvkey": firms,
        "curcd": ["USD"] * len(firms),
        "datadate": [date(2000 + t // 12, t % 12 + 1, 28) for t in ts],
        "_t": ts,
        "count": [t + 1 for t in ts],
    }
    for c in cols:
        data[c] = [gen(c, f, t) for f, t in zip(firms, ts, strict=True)]
    return pl.DataFrame(data, schema_overrides=dict.fromkeys(cols, pl.Float64))


def _set(df: pl.DataFrame, firm: str, t: int, **vals: float | None) -> pl.DataFrame:
    """Overwrite columns at one (firm, month) cell; ``None`` makes the cell null."""
    hit = (pl.col("gvkey") == firm) & (pl.col("_t") == t)
    return df.with_columns(
        pl.when(hit).then(pl.lit(v, dtype=pl.Float64)).otherwise(pl.col(c)).alias(c)
        for c, v in vals.items()
    )


def _row(df: pl.DataFrame, firm: str, t: int) -> dict:
    return df.filter((pl.col("gvkey") == firm) & (pl.col("_t") == t)).row(0, named=True)


def _approx(x: float) -> object:
    return pytest.approx(x, rel=1e-9, abs=1e-12)


def _check(row: dict, expected: dict[str, float]) -> None:
    for name, value in expected.items():
        assert row[name] == _approx(value), name


def _all_null(row: dict, names: Sequence[str]) -> None:
    for name in names:
        assert row[name] is None, name


# --------------------------------------------------------------------------------------
# add_accounting_misc_cols_1
# --------------------------------------------------------------------------------------

GROWTH_VARS = [
    "at_x", "ca_x", "nca_x", "lt", "cl_x", "ncl_x", "be_x", "pstk_x", "debt_x",
    "sale_x", "cogs", "xsga", "opex_x", "capx", "invt",
]  # fmt: skip
CH_ASSET_VARS = [
    "che", "invt", "rect", "ppegt", "ivao", "ivst", "intan", "dlc", "ap", "txp", "dltt",
    "txditc", "coa_x", "col_x", "cowc_x", "ncoa_x", "ncol_x", "nncoa_x", "oa_x", "ol_x",
    "fna_x", "fnl_x", "nfna_x", "gp_x", "ebitda_x", "ebit_x", "ope_x", "ni_x", "nix_x", "dp",
    "fincf_x", "ocf_x", "fcf_x", "nwc_x", "eqnetis_x", "dltnetis_x", "dstnetis_x",
    "dbnetis_x", "netis_x", "eqnpo_x", "txt", "eqbb_x", "eqis_x", "div_x", "eqpo_x", "capx",
    "be_x",
]  # fmt: skip
COLS_1 = sorted(
    {
        *GROWTH_VARS, *CH_ASSET_VARS,
        "xrd", "spi", "xido_x", "pi_x", "ni", "fi_x", "cop_x", "bev_x", "ppent", "xint",
        "oacc_x", "tacc_x", "noa_x", "ebit_x", "ebitda_x", "fcf_x", "ocf_x", "gp_x", "ope_x",
    }
)  # fmt: skip

# Raw inputs for one hand-computed month (firm A, t=20).
RAW_1 = {
    "at_x": 100.0, "sale_x": 200.0, "be_x": 50.0, "bev_x": 80.0, "ppent": 40.0, "xint": 5.0,
    "capx": 5.0, "xrd": 3.0, "spi": -2.0, "xido_x": 4.0, "gp_x": 60.0, "ebitda_x": 40.0,
    "ebit_x": 30.0, "pi_x": 25.0, "ni_x": 20.0, "ni": 18.0, "ocf_x": 36.0, "fcf_x": 24.0,
    "fi_x": 10.0, "cop_x": 12.0, "ope_x": 16.0, "nix_x": -10.0, "debt_x": 70.0,
    "fincf_x": -8.0, "netis_x": 4.0, "eqnetis_x": 2.0, "eqis_x": 6.0, "dbnetis_x": 2.0,
    "dltnetis_x": 1.0, "dstnetis_x": 1.0, "eqnpo_x": -5.0, "eqbb_x": 7.0, "div_x": 3.0,
    "pstk_x": 10.0, "che": 20.0, "dltt": 30.0, "dlc": 10.0, "oacc_x": -4.0, "tacc_x": -6.0,
    "noa_x": 54.0,
}  # fmt: skip
# at_x is 100 + 8 = 108 at t=8, the 12-month lag of t=20.
EXPECTED_1 = {
    "capx_at": 0.05, "rd_at": 0.03, "spi_at": -0.02, "xido_at": 0.04, "nri_at": (-2 + 4) / 100,
    "gp_sale": 0.3, "ebitda_sale": 0.2, "ebit_sale": 0.15, "pi_sale": 0.125, "ni_sale": 0.1,
    "nix_sale": 0.09, "ocf_sale": 0.18, "fcf_sale": 0.12,
    "gp_at": 0.6, "ebitda_at": 0.4, "ebit_at": 0.3, "fi_at": 0.1, "cop_at": 0.12, "ni_at": 0.2,
    "ope_be": 0.32, "ni_be": 0.4, "nix_be": -0.2, "ocf_be": 0.72, "fcf_be": 0.48,
    "gp_bev": 0.75, "ebitda_bev": 0.5, "ebit_bev": 0.375, "fi_bev": 0.125, "cop_bev": 0.15,
    "gp_ppen": 1.5, "ebitda_ppen": 1.0, "fcf_ppen": 0.6,
    "fincf_at": -0.08, "netis_at": 0.04, "eqnetis_at": 0.02, "eqis_at": 0.06,
    "dbnetis_at": 0.02, "dltnetis_at": 0.01, "dstnetis_at": 0.01, "eqnpo_at": -0.05,
    "eqbb_at": 0.07, "div_at": 0.03,
    "debt_at": 0.7, "debt_be": 1.4, "ebit_int": 6.0,
    "be_bev": 0.625, "debt_bev": 0.875, "pstk_bev": 0.125, "cash_bev": 0.25,
    "debtlt_bev": 0.375, "debtst_bev": 0.125,
    "oaccruals_at": -0.04, "taccruals_at": -0.06,
    "oaccruals_ni": -4 / 10, "taccruals_ni": -6 / 10,  # |nix_x| = 10
    "noa_at": 54 / 108,
}  # fmt: skip
# Everything above except noa_at divides by the contemporaneous at_x, sale_x, be_x, bev_x,
# ppent, xint or nix_x.
RATIOS_1 = [c for c in EXPECTED_1 if c != "noa_at"]


def _run_1(df: pl.DataFrame) -> pl.DataFrame:
    return add_accounting_misc_cols_1(df.lazy()).collect()


@pytest.fixture(scope="module")
def panel_1() -> pl.DataFrame:
    return _run_1(_panel(COLS_1, 40))


@pytest.fixture(scope="module")
def ratios_1() -> pl.DataFrame:
    df = _panel(COLS_1, 24)
    df = _set(df, "A", 20, **RAW_1)
    # t=21: every contemporaneous denominator is zero.
    df = _set(df, "A", 21, at_x=0.0, sale_x=0.0, be_x=0.0, bev_x=0.0, ppent=0.0, xint=0.0)
    df = _set(df, "A", 21, nix_x=0.0)
    # t=22: sale_x and spi are null.
    df = _set(df, "A", 22, sale_x=None, spi=None)
    return _run_1(df)


class TestMiscCols1Growth:
    def test_one_and_three_year_growth(self, panel_1: pl.DataFrame) -> None:
        for firm in ("A", "B"):
            row = _row(panel_1, firm, 39)
            for var in GROWTH_VARS:
                base = var.replace("_x", "")
                v = _lin()
                assert row[f"{base}_gr1"] == _approx(v(var, firm, 39) / v(var, firm, 27) - 1), (
                    firm,
                    var,
                )
                assert row[f"{base}_gr3"] == _approx(v(var, firm, 39) / v(var, firm, 3) - 1), (
                    firm,
                    var,
                )

    def test_growth_hand_value(self, panel_1: pl.DataFrame) -> None:
        # firm B, t=12: at_x = 100 + 2*12 = 124 vs 100 at t=0.
        assert _row(panel_1, "B", 12)["at_gr1"] == _approx(124 / 100 - 1)

    def test_short_history_is_null(self, panel_1: pl.DataFrame) -> None:
        for firm in ("A", "B"):
            # count = 12 is not > 12; count = 36 is not > 36.
            assert _row(panel_1, firm, 11)["at_gr1"] is None
            assert _row(panel_1, firm, 12)["at_gr1"] is not None
            assert _row(panel_1, firm, 35)["at_gr3"] is None
            assert _row(panel_1, firm, 36)["at_gr3"] is not None

    def test_nonpositive_or_null_lag_is_null(self) -> None:
        df = _panel(COLS_1, 16)
        df = _set(df, "A", 0, pstk_x=0.0, at_x=None)
        df = _set(df, "A", 1, pstk_x=-1.0)
        out = _run_1(df)
        assert _row(out, "A", 12)["pstk_gr1"] is None  # lag = 0
        assert _row(out, "A", 12)["at_gr1"] is None  # lag null
        assert _row(out, "A", 13)["pstk_gr1"] is None  # lag < 0
        assert _row(out, "A", 14)["pstk_gr1"] == _approx(114 / 102 - 1)
        assert _row(out, "B", 12)["pstk_gr1"] == _approx(124 / 100 - 1)

    def test_null_current_value_is_null(self) -> None:
        out = _run_1(_set(_panel(COLS_1, 16), "A", 14, sale_x=None))
        assert _row(out, "A", 14)["sale_gr1"] is None

    def test_output_is_sorted_by_firm_and_date(self) -> None:
        out = _run_1(_panel(COLS_1, 14).reverse())
        assert out["gvkey"].to_list() == ["A"] * 14 + ["B"] * 14
        assert out["_t"].to_list() == list(range(14)) * 2
        assert _row(out, "B", 13)["at_gr1"] == _approx(126 / 102 - 1)


class TestMiscCols1ChangeScaledByAssets:
    def test_one_and_three_year_changes(self, panel_1: pl.DataFrame) -> None:
        for firm in ("A", "B"):
            row = _row(panel_1, firm, 39)
            at = DEFAULT_BASE + SLOPE[firm] * 39
            for var in CH_ASSET_VARS:
                base = var.replace("_x", "")
                # Change over h months of a linear series is slope * h.
                assert row[f"{base}_gr1a"] == _approx(SLOPE[firm] * 12 / at), (firm, var)
                assert row[f"{base}_gr3a"] == _approx(SLOPE[firm] * 36 / at), (firm, var)

    def test_hand_value(self, panel_1: pl.DataFrame) -> None:
        # che(A, t=39) = 139, che(A, t=27) = 127, at_x = 139.
        assert _row(panel_1, "A", 39)["che_gr1a"] == _approx((139 - 127) / 139)

    def test_short_history_is_null(self, panel_1: pl.DataFrame) -> None:
        for firm in ("A", "B"):
            assert _row(panel_1, firm, 11)["che_gr1a"] is None
            assert _row(panel_1, firm, 12)["che_gr1a"] is not None
            assert _row(panel_1, firm, 35)["che_gr3a"] is None
            assert _row(panel_1, firm, 36)["che_gr3a"] is not None

    def test_nonpositive_assets_is_null(self) -> None:
        df = _set(_panel(COLS_1, 16), "A", 13, at_x=0.0)
        df = _set(df, "A", 14, at_x=-5.0)
        out = _run_1(df)
        for t in (13, 14):
            assert _row(out, "A", t)["che_gr1a"] is None
        assert _row(out, "A", 15)["che_gr1a"] == _approx(12 / 115)

    def test_null_lag_is_null(self) -> None:
        out = _run_1(_set(_panel(COLS_1, 16), "A", 1, che=None))
        assert _row(out, "A", 13)["che_gr1a"] is None
        assert _row(out, "B", 13)["che_gr1a"] == _approx(24 / 126)


class TestMiscCols1Ratios:
    def test_hand_computed_ratios(self, ratios_1: pl.DataFrame) -> None:
        _check(_row(ratios_1, "A", 20), EXPECTED_1)

    def test_other_firm_uses_its_own_values(self, ratios_1: pl.DataFrame) -> None:
        row = _row(ratios_1, "B", 20)
        # B at t=20: gp_x = 140, sale_x = 140, at_x = 140; at_x at t=8 is 116.
        assert row["gp_sale"] == _approx(1.0)
        assert row["capx_at"] == _approx(1.0)
        assert row["noa_at"] == _approx(140 / 116)

    def test_zero_denominators_are_null(self, ratios_1: pl.DataFrame) -> None:
        _all_null(_row(ratios_1, "A", 21), RATIOS_1)

    def test_null_inputs_propagate(self, ratios_1: pl.DataFrame) -> None:
        row = _row(ratios_1, "A", 22)
        _all_null(row, ["nri_at", "spi_at", "gp_sale", "ni_sale", "fcf_sale", "nix_sale"])
        assert row["capx_at"] == _approx(122 / 122)

    def test_negative_denominators(self) -> None:
        df = _set(_panel(COLS_1, 24), "A", 20, **RAW_1)
        df = _set(df, "A", 20, be_x=-50.0, at_x=-100.0)
        row = _row(_run_1(df), "A", 20)
        # Mode-1 divisions keep negative denominators; accruals_ni uses |nix_x|.
        assert row["ni_be"] == _approx(20 / -50)
        assert row["capx_at"] == _approx(5 / -100)
        assert row["oaccruals_ni"] == _approx(-4 / 10)

    def test_noa_at_uses_lagged_assets(self) -> None:
        df = _panel(COLS_1, 24)
        df = _set(df, "A", 20, noa_x=54.0)
        df = _set(df, "A", 21, noa_x=54.0)
        df = _set(df, "A", 22, noa_x=54.0)
        df = _set(df, "A", 23, noa_x=54.0)
        df = _set(df, "A", 9, at_x=0.0)
        df = _set(df, "A", 10, at_x=-5.0)
        df = _set(df, "A", 11, at_x=None)
        out = _run_1(df)
        assert _row(out, "A", 20)["noa_at"] == _approx(54 / 108)
        assert _row(out, "A", 21)["noa_at"] is None  # lag = 0
        assert _row(out, "A", 22)["noa_at"] is None  # lag < 0
        assert _row(out, "A", 23)["noa_at"] is None  # lag null
        assert _row(out, "A", 11)["noa_at"] is None  # count = 12
        assert _row(out, "B", 12)["noa_at"] == _approx(124 / 100)  # first month with count > 12


# --------------------------------------------------------------------------------------
# financial_soundness_and_misc_ratios_exps
# --------------------------------------------------------------------------------------

COLS_FIN = [
    "act", "at_x", "be_x", "bev_x", "che", "cl_x", "debt_x", "div_x", "dlc", "dltt",
    "ebitda_x", "emp", "fcf_x", "invt", "lt", "ni_x", "nix_x", "nwc_x", "ocf_x", "opex_x",
    "pi_x", "ppent", "rect", "sale_x", "txt", "xad", "xint", "xlr", "xrd",
]  # fmt: skip
RAW_FIN = {
    "debt_x": 70.0, "xint": 5.0, "ocf_x": 36.0, "ebitda_x": 40.0, "dlc": 10.0, "dltt": 30.0,
    "cl_x": 40.0, "che": 20.0, "lt": 80.0, "invt": 15.0, "act": 60.0, "rect": 12.0,
    "opex_x": 50.0, "at_x": 100.0, "nwc_x": 25.0, "ppent": 40.0, "be_x": 50.0, "fcf_x": 24.0,
    "xad": 4.0, "xlr": 6.0, "sale_x": 200.0, "bev_x": 80.0, "xrd": 3.0, "txt": 7.0,
    "pi_x": 28.0, "emp": 10.0, "ni_x": 20.0, "nix_x": 18.0, "div_x": 9.0,
}  # fmt: skip
EXPECTED_FIN = {
    "int_debt": 5 / 70, "ocf_debt": 36 / 70, "ebitda_debt": 40 / 70, "debtst_debt": 10 / 70,
    "debtlt_debt": 30 / 70, "int_debtlt": 5 / 30, "profit_cl": 1.0, "ocf_cl": 0.9,
    "cash_lt": 0.25, "cl_lt": 0.5, "inv_act": 0.25, "rec_act": 0.2, "opex_at": 0.5,
    "nwc_at": 0.25, "lt_ppen": 2.0, "debtlt_be": 0.6, "fcf_ocf": 24 / 36,
    "adv_sale": 0.02, "staff_sale": 0.03, "sale_bev": 2.5, "rd_sale": 0.015, "sale_be": 4.0,
    "sale_nwc": 8.0, "tax_pi": 0.25, "cash_at": 0.2, "ni_emp": 2.0, "sale_emp": 20.0,
    "div_ni": 0.5,
}  # fmt: skip
# Divisions that are null for a zero denominator (mode 1) ...
FIN_MODE1 = [
    "int_debt", "ocf_debt", "ebitda_debt", "debtst_debt", "debtlt_debt", "int_debtlt",
    "profit_cl", "ocf_cl", "cash_lt", "cl_lt", "inv_act", "rec_act", "opex_at", "nwc_at",
    "lt_ppen", "debtlt_be", "adv_sale", "staff_sale", "sale_bev", "rd_sale", "sale_be",
]  # fmt: skip
# ... and those that also need a strictly positive denominator (mode 3).
FIN_MODE3 = ["fcf_ocf", "sale_nwc", "tax_pi", "cash_at", "ni_emp", "sale_emp"]


@pytest.fixture(scope="module")
def fin() -> pl.DataFrame:
    df = _panel(COLS_FIN, 30)
    df = _set(df, "A", 20, **RAW_FIN)
    zeros = dict.fromkeys(("debt_x", "dltt", "cl_x", "lt", "act", "at_x", "ppent", "be_x"), 0.0)
    zeros |= dict.fromkeys(("sale_x", "bev_x", "ocf_x", "nwc_x", "pi_x", "emp"), 0.0)
    df = _set(df, "A", 21, **RAW_FIN)
    df = _set(df, "A", 21, **zeros)
    negs = {"ocf_x": -36.0, "nwc_x": -25.0, "pi_x": -28.0, "at_x": -100.0, "emp": -10.0}
    df = _set(df, "A", 22, **RAW_FIN)
    df = _set(df, "A", 22, **negs, debt_x=-70.0)
    return df.with_columns(financial_soundness_and_misc_ratios_exps())


class TestFinancialSoundnessAndMisc:
    def test_hand_computed_ratios(self, fin: pl.DataFrame) -> None:
        _check(_row(fin, "A", 20), EXPECTED_FIN)

    def test_every_expected_column_is_produced(self, fin: pl.DataFrame) -> None:
        assert len(financial_soundness_and_misc_ratios_exps()) == len(EXPECTED_FIN)
        assert set(EXPECTED_FIN) <= set(fin.columns)

    def test_other_firm_uses_its_own_values(self, fin: pl.DataFrame) -> None:
        # B at t=0: dltt = 100, debt_x = 100 -> 1.0; at t=20: cash_at = che/at_x = 140/140.
        assert _row(fin, "B", 0)["debtlt_debt"] == _approx(1.0)
        assert _row(fin, "B", 20)["cash_at"] == _approx(1.0)

    def test_zero_denominators_are_null(self, fin: pl.DataFrame) -> None:
        _all_null(_row(fin, "A", 21), FIN_MODE1 + FIN_MODE3)

    def test_mode3_needs_positive_denominator(self, fin: pl.DataFrame) -> None:
        row = _row(fin, "A", 22)
        _all_null(row, FIN_MODE3)
        # Mode-1 ratios keep a negative denominator.
        assert row["int_debt"] == _approx(5 / -70)

    @pytest.mark.parametrize(
        ("nix_x", "ni_x", "div_x", "expected"),
        [
            (18.0, 20.0, 9.0, 0.5),  # nix_x > 0
            (None, 20.0, 9.0, None),  # nix_x null: coalesce > 0 but div/nix is null
            (-5.0, 20.0, 9.0, None),  # coalesce picks the negative nix_x
            (0.0, 20.0, 9.0, None),  # coalesce picks zero
            (None, None, 9.0, None),  # both null
            (18.0, 20.0, None, None),  # null dividend
            (-5.0, None, 9.0, None),  # nix_x < 0
        ],
    )
    def test_div_ni(
        self, nix_x: float | None, ni_x: float | None, div_x: float | None, expected: float | None
    ) -> None:
        df = _set(_panel(COLS_FIN, 3), "A", 1, nix_x=nix_x, ni_x=ni_x, div_x=div_x)
        out = df.with_columns(financial_soundness_and_misc_ratios_exps())
        got = _row(out, "A", 1)["div_ni"]
        if expected is None:
            assert got is None
        else:
            assert got == _approx(expected)


# --------------------------------------------------------------------------------------
# add_liq_and_efficiency_ratios
# --------------------------------------------------------------------------------------

COLS_LIQ = ["ap", "at_x", "ca_x", "caliq_x", "che", "cl_x", "cogs", "invt", "rect", "sale_x"]
BASE_LIQ = {
    "invt": 40.0, "rect": 60.0, "ap": 30.0, "cogs": 200.0, "sale_x": 400.0, "at_x": 1000.0,
    "che": 10.0, "cl_x": 100.0, "caliq_x": 50.0, "ca_x": 80.0,
}  # fmt: skip


def _liq_panel(n: int = 16) -> pl.DataFrame:
    return _panel(COLS_LIQ, n, _lin(BASE_LIQ))


def _run_liq(df: pl.DataFrame) -> pl.DataFrame:
    return add_liq_and_efficiency_ratios(df.lazy()).collect()


class TestLiqAndEfficiency:
    def test_hand_computed_firm_a(self) -> None:
        row = _row(_run_liq(_liq_panel()), "A", 13)
        # t=13 vs t=1: invt 53/41, rect 73/61, ap 43/31, cogs 213, sale_x 413, at_x 1013/1001.
        inv_days = 365 * ((53 + 41) / 2) / 213
        rec_days = 365 * ((73 + 61) / 2) / 413
        ap_days = 365 * ((43 + 31) / 2) / 213
        _check(
            row,
            {
                "inv_days": inv_days,
                "rec_days": rec_days,
                "ap_days": ap_days,
                "cash_conversion": inv_days + rec_days - ap_days,
                "cash_cl": 23 / 113,  # che / cl_x
                "caliq_cl": 63 / 113,
                "ca_cl": 93 / 113,
                "inv_turnover": 213 / ((53 + 41) / 2),
                "at_turnover": 413 / ((1013 + 1001) / 2),
                "rec_turnover": 413 / ((73 + 61) / 2),
                "ap_turnover": (213 + 53 - 41) / ((43 + 31) / 2),
            },
        )

    def test_hand_computed_firm_b(self) -> None:
        row = _row(_run_liq(_liq_panel()), "B", 13)
        # B has slope 2: invt 66/42, cogs 226, ap 56/32.
        assert row["inv_days"] == _approx(365 * ((66 + 42) / 2) / 226)
        assert row["ap_turnover"] == _approx((226 + 66 - 42) / ((56 + 32) / 2))
        assert row["cash_cl"] == _approx(36 / 126)

    def test_short_history_nulls_days_and_turnovers(self) -> None:
        out = _run_liq(_liq_panel())
        names = ["inv_days", "rec_days", "ap_days", "cash_conversion", "inv_turnover"]
        names += ["at_turnover", "rec_turnover", "ap_turnover"]
        for firm in ("A", "B"):
            _all_null(_row(out, firm, 11), names)  # count = 12
            assert _row(out, firm, 12)["inv_days"] is not None  # count = 13
            assert _row(out, firm, 12)["ap_turnover"] is not None
            # Balance-sheet ratios do not depend on history.
            assert _row(out, firm, 0)["cash_cl"] is not None

    def test_mean_year_falls_back_to_available_value(self) -> None:
        df = _set(_liq_panel(), "A", 1, invt=None)  # lag null at t=13
        df = _set(df, "A", 14, rect=None)  # current null at t=14, lag (t=2) = 62
        row13 = _row(_run_liq(df), "A", 13)
        assert row13["inv_days"] == _approx(365 * 53 / 213)
        row14 = _row(_run_liq(df), "A", 14)
        assert row14["rec_days"] == _approx(365 * 62 / 414)

    def test_zero_denominators_are_null(self) -> None:
        df = _set(_liq_panel(), "A", 13, cogs=0.0, sale_x=0.0, cl_x=0.0)
        df = _set(df, "A", 14, invt=0.0, at_x=0.0)
        df = _set(df, "A", 2, invt=0.0, at_x=0.0)  # mean_year = 0 at t=14
        out = _run_liq(df)
        row = _row(out, "A", 13)
        _all_null(row, ["inv_days", "ap_days", "rec_days", "cash_conversion"])
        _all_null(row, ["cash_cl", "caliq_cl", "ca_cl"])
        assert row["inv_turnover"] == _approx(0 / 47)  # cogs = 0 is a valid numerator
        row = _row(out, "A", 14)
        _all_null(row, ["inv_turnover", "at_turnover"])
        assert row["rec_turnover"] is not None

    def test_nonpositive_current_liabilities_null_liquidity_ratios(self) -> None:
        df = _set(_liq_panel(), "A", 5, cl_x=-10.0)
        row = _row(_run_liq(df), "A", 5)
        _all_null(row, ["cash_cl", "caliq_cl", "ca_cl"])

    def test_negative_cash_conversion_is_null_zero_is_kept(self) -> None:
        # Huge payables: ap_days exceeds inv_days + rec_days.
        df = _set(_liq_panel(), "A", 13, ap=4000.0)
        row = _row(_run_liq(df), "A", 13)
        assert row["ap_days"] > row["inv_days"] + row["rec_days"]
        assert row["cash_conversion"] is None
        # All three means zero -> all three days zero -> cash conversion exactly 0.
        df = _liq_panel()
        for t in (1, 13):
            df = _set(df, "A", t, invt=0.0, rect=0.0, ap=0.0)
        row = _row(_run_liq(df), "A", 13)
        assert row["cash_conversion"] == 0.0

    def test_null_days_make_cash_conversion_null(self) -> None:
        df = _set(_liq_panel(), "A", 13, invt=None)
        df = _set(df, "A", 1, invt=None)
        row = _row(_run_liq(df), "A", 13)
        assert row["inv_days"] is None
        assert row["cash_conversion"] is None
        assert row["rec_days"] is not None

    def test_unsorted_input_is_sorted(self) -> None:
        out = _run_liq(_liq_panel().reverse())
        assert out["_t"].to_list() == list(range(16)) * 2
        assert _row(out, "A", 13)["inv_turnover"] == _approx(213 / 47)


# --------------------------------------------------------------------------------------
# add_profit_scaled_by_lagged_vars
# --------------------------------------------------------------------------------------

COLS_LAG = ["at_x", "be_x", "cop_x", "gp_x", "op_x", "ope_x"]
BASE_LAG = {"at_x": 100.0, "be_x": 50.0, "cop_x": 20.0, "gp_x": 60.0, "op_x": 30.0, "ope_x": 10.0}
LAG_NAMES = ["op_atl1", "gp_atl1", "ope_bel1", "cop_atl1"]


def _run_lag(df: pl.DataFrame) -> pl.DataFrame:
    return add_profit_scaled_by_lagged_vars(df.lazy()).collect()


def _lag_panel() -> pl.DataFrame:
    return _panel(COLS_LAG, 16, _lin(BASE_LAG))


class TestProfitScaledByLaggedVars:
    def test_hand_computed(self) -> None:
        out = _run_lag(_lag_panel())
        # A at t=13: numerators 43, 73, 23, 33; lagged (t=1) at_x = 101, be_x = 51.
        _check(
            _row(out, "A", 13),
            {
                "op_atl1": 43 / 101,
                "gp_atl1": 73 / 101,
                "ope_bel1": 23 / 51,
                "cop_atl1": 33 / 101,
            },
        )
        # B slope 2: op 56, gp 86, ope 36, cop 46; lagged at_x 102, be_x 52.
        _check(
            _row(out, "B", 13),
            {
                "op_atl1": 56 / 102,
                "gp_atl1": 86 / 102,
                "ope_bel1": 36 / 52,
                "cop_atl1": 46 / 102,
            },
        )

    def test_short_history_is_null(self) -> None:
        out = _run_lag(_lag_panel())
        for firm in ("A", "B"):
            _all_null(_row(out, firm, 11), LAG_NAMES)  # count = 12
            _all_null(_row(out, firm, 0), LAG_NAMES)
            assert _row(out, firm, 12)["op_atl1"] is not None  # count = 13

    def test_nonpositive_lagged_assets_null_asset_scaled_only(self) -> None:
        df = _set(_lag_panel(), "A", 1, at_x=0.0)
        df = _set(df, "A", 2, at_x=-5.0)
        out = _run_lag(df)
        for t in (13, 14):
            row = _row(out, "A", t)
            _all_null(row, ["op_atl1", "gp_atl1", "cop_atl1"])
        assert _row(out, "A", 13)["ope_bel1"] == _approx(23 / 51)
        assert _row(out, "A", 15)["op_atl1"] == _approx(45 / 103)

    def test_nonpositive_lagged_book_equity_nulls_ope_only(self) -> None:
        df = _set(_lag_panel(), "A", 1, be_x=0.0)
        df = _set(df, "A", 2, be_x=-1.0)
        out = _run_lag(df)
        for t in (13, 14):
            assert _row(out, "A", t)["ope_bel1"] is None
        assert _row(out, "A", 13)["op_atl1"] == _approx(43 / 101)

    def test_null_lag_and_null_numerator(self) -> None:
        df = _set(_lag_panel(), "A", 1, at_x=None)
        df = _set(df, "A", 14, op_x=None)
        out = _run_lag(df)
        _all_null(_row(out, "A", 13), ["op_atl1", "gp_atl1", "cop_atl1"])
        assert _row(out, "A", 14)["op_atl1"] is None
        assert _row(out, "A", 14)["gp_atl1"] == _approx(74 / 102)

    def test_other_firm_unaffected_by_lag_edits(self) -> None:
        out = _run_lag(_set(_lag_panel(), "A", 1, at_x=0.0))
        assert _row(out, "B", 13)["op_atl1"] == _approx(56 / 102)

    def test_unsorted_input_is_sorted(self) -> None:
        out = _run_lag(_lag_panel().reverse())
        assert out["_t"].to_list() == list(range(16)) * 2
        assert _row(out, "A", 13)["op_atl1"] == _approx(43 / 101)


# --------------------------------------------------------------------------------------
# add_accounting_misc_cols_2
# --------------------------------------------------------------------------------------

COLS_2_SPEC = {  # column -> (base, slope) for firm A
    "at_x": (100.0, 0.0), "sale_x": (200.0, 0.0), "be_x": (50.0, 0.0),
    "gp_x": (10.0, 0.1), "ni_x": (5.0, 0.1), "ocf_x": (8.0, 0.2), "nix_x": (5.0, 0.1),
    "ca_x": (50.0, 0.0), "cl_x": (30.0, 0.0), "dltt": (20.0, 0.0), "eqis_x": (0.0, 0.0),
    "debt_x": (40.0, 0.0), "lt": (80.0, 0.0), "pi_x": (8.0, 0.0), "dp": (2.0, 0.0),
    "re": (20.0, 0.0), "ebitda_x": (10.0, 0.0), "me_fiscal": (60.0, 0.0),
    "div_x": (2.0, 0.0), "ppent": (40.0, 0.0), "che": (10.0, 0.0), "seq_x": (50.0, 0.0),
    "__ocfq_saleq": (0.0, 1.0), "__niq_saleq": (0.0, 1.0), "__roeq": (0.0, 1.0),
    "__roe": (0.0, 1.0),
}  # fmt: skip
VOL_COLS = ["__ocfq_saleq", "__niq_saleq", "__roeq", "__roe"]


def _gen_2(name: str, firm: str, t: int) -> float:
    base, slope = COLS_2_SPEC[name]
    if firm == "B":
        # B: doubled volatility drivers and gross profit; negative book equity.
        if name in VOL_COLS or name == "gp_x":
            base, slope = 2 * base, 2 * slope
        if name == "be_x":
            base = -50.0
    return base + slope * t


def _panel_2(n: int = 61) -> pl.DataFrame:
    return _panel(list(COLS_2_SPEC), n, _gen_2)


def _run_2(df: pl.DataFrame) -> pl.DataFrame:
    return add_accounting_misc_cols_2(df.lazy()).collect()


def _std_of_step_sequence(step: float, n: int) -> float:
    """Sample std of n equally spaced values (spacing ``step``): step * sqrt(n(n+1)/12)."""
    return step * math.sqrt(n * (n + 1) / 12)


@pytest.fixture(scope="module")
def out_2() -> pl.DataFrame:
    return _run_2(_panel_2())


class TestMiscCols2:
    def test_helper_columns_are_dropped(self, out_2: pl.DataFrame) -> None:
        for name in ["count", *VOL_COLS]:
            assert name not in out_2.columns

    def test_all_new_columns_present(self, out_2: pl.DataFrame) -> None:
        expected = [
            "ocfq_saleq_std", "niq_saleq_std", "roeq_be_std", "roe_be_std",
            "earnings_variability", "ed_cd", "ed_cd_w", "ed_constant", "ed_err", "f_score",
            "o_score", "z_score", "intrinsic_value", "kz_index",
            "gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5",
        ]  # fmt: skip
        assert set(expected) <= set(out_2.columns)

    def test_volatilities_hand_computed(self, out_2: pl.DataFrame) -> None:
        # Drivers equal t, so window values are spaced 3 (quarterly) or 12 (annual) apart.
        for firm, scale in (("A", 1.0), ("B", 2.0)):
            row = _row(out_2, firm, 60)
            _check(
                row,
                {
                    "ocfq_saleq_std": scale * _std_of_step_sequence(3, 16),
                    "niq_saleq_std": scale * _std_of_step_sequence(3, 16),
                    "roeq_be_std": scale * _std_of_step_sequence(3, 20),
                    "roe_be_std": scale * _std_of_step_sequence(12, 5),
                },
            )

    def test_volatility_short_history_is_null(self, out_2: pl.DataFrame) -> None:
        for firm in ("A", "B"):
            # count must exceed (qtrs - 1) * 3 = 45 / 57 and (yrs - 1) * 12 = 48.
            _all_null(_row(out_2, firm, 15), ["ocfq_saleq_std", "niq_saleq_std", "roeq_be_std"])
            assert _row(out_2, firm, 44)["ocfq_saleq_std"] is None
            assert _row(out_2, firm, 45)["ocfq_saleq_std"] is not None
            assert _row(out_2, firm, 56)["roeq_be_std"] is None
            assert _row(out_2, firm, 57)["roeq_be_std"] is not None
            assert _row(out_2, firm, 47)["roe_be_std"] is None
            assert _row(out_2, firm, 48)["roe_be_std"] == _approx(
                (2 if firm == "B" else 1) * _std_of_step_sequence(12, 5)
            )

    def test_volatility_needs_minimum_observations(self) -> None:
        df = _set(_panel_2(), "A", 36, __roe=None)
        df = _set(df, "A", 24, __roe=None)  # annual window at t=60 keeps 3 of 5
        df = df.with_columns(
            pl.when((pl.col("gvkey") == "A") & (pl.col("_t") % 3 != 0))
            .then(None)
            .otherwise(pl.col("__ocfq_saleq"))
            .alias("__ocfq_saleq")
        )
        out = _run_2(df)
        assert _row(out, "A", 60)["roe_be_std"] is None  # n = 3 < 5
        assert _row(out, "B", 60)["roe_be_std"] == _approx(2 * _std_of_step_sequence(12, 5))
        # Window shifts 0,3,...,45 all land on multiples of 3 at t=60, so n stays 16.
        assert _row(out, "A", 60)["ocfq_saleq_std"] is not None
        # At t=61 would not exist; use t=59: shifts 59,56,... skip multiples -> n = 0 < 8.
        assert _row(out, "A", 59)["ocfq_saleq_std"] is None

    def test_five_year_changes(self, out_2: pl.DataFrame) -> None:
        # gp_x, ni_x, ocf_x rise by 6, 6, 12 over 60 months; at_x = 100, be_x = 50, sale_x = 200.
        row = _row(out_2, "A", 60)
        _check(
            row,
            {
                "gpoa_ch5": 6 / 100,
                "roe_ch5": 6 / 50,
                "roa_ch5": 6 / 100,
                "cfoa_ch5": 12 / 100,
                "gmar_ch5": 6 / 200,
            },
        )
        # B: gp_x doubled; negative book equity nulls roe_ch5.
        row = _row(out_2, "B", 60)
        _check(row, {"gpoa_ch5": 12 / 100, "gmar_ch5": 12 / 200, "roa_ch5": 6 / 100})
        assert row["roe_ch5"] is None

    def test_five_year_changes_short_history_is_null(self, out_2: pl.DataFrame) -> None:
        for firm in ("A", "B"):
            row = _row(out_2, firm, 59)  # count = 60, not > 60
            _all_null(row, ["gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5"])

    def test_nonpositive_denominator_nulls_change(self) -> None:
        df = _set(_panel_2(), "A", 60, at_x=0.0)
        row = _row(_run_2(df), "A", 60)
        _all_null(row, ["gpoa_ch5", "roa_ch5", "cfoa_ch5"])
        assert row["gmar_ch5"] == _approx(6 / 200)

    def test_earnings_variability(self, out_2: pl.DataFrame) -> None:
        # ROA steps 0.012 and CROA steps 0.024 per year over five annual points.
        row = _row(out_2, "A", 60)
        assert row["earnings_variability"] == _approx(0.5)
        assert _row(out_2, "A", 59)["earnings_variability"] is None  # count = 60, not > 60

    def test_equity_duration_columns(self, out_2: pl.DataFrame) -> None:
        row = _row(out_2, "A", 60)
        assert row["ed_constant"] == _approx(10 + 1.12 / 0.12)
        assert row["ed_err"] == 0.0
        # AR(1) paths by hand: ROE0 = 11/50, g0 = 200/200 - 1 = 0, BE0 = 50.
        r, roe, g, be = 0.12, 11 / 50, 0.0, 50.0
        ed_cd = ed_cd_w = 0.0
        for t in range(1, 11):
            g = 0.06 * (1 - 0.24) + 0.24 * g
            roe = 0.12 * (1 - 0.57) + 0.57 * roe
            cd = be * (roe - g)
            be *= 1 + g
            ed_cd += cd / (1 + r) ** t
            ed_cd_w += t * cd / (1 + r) ** t
        assert row["ed_cd"] == _approx(ed_cd)
        assert row["ed_cd_w"] == _approx(ed_cd_w)

    def test_composite_scores(self, out_2: pl.DataFrame) -> None:
        row = _row(out_2, "A", 60)
        # Altman Z: WC/TA .2, RE/TA .2, EBITDA/TA .1, ME/LT .75, Sales/TA 2.0.
        assert row["z_score"] == _approx(1.2 * 0.2 + 1.4 * 0.2 + 3.3 * 0.1 + 0.6 * 0.75 + 2.0)
        # Piotroski: roa, cfo, droa, accruals, no issuance, gross margin pass; the rest are flat.
        assert row["f_score"] == 6
        # Ohlson: lev .4, roe .11, cacl .6, wc .2, ffo .125, no negative equity or earnings.
        nich = (11 - 9.8) / (11 + 9.8)
        o = (
            -1.32 - 0.407 * math.log(100) + 6.03 * 0.4 - 1.43 * 0.2 + 0.076 * 0.6
            - 2.37 * 0.11 - 1.83 * 0.125 - 0.52 * nich
        )  # fmt: skip
        assert row["o_score"] == _approx(o)
        # Kaplan-Zingales: lagged ppent 40; cf (11+2)/40, dv 2/40, cs 10/40, q 1.1, db 40/90.
        kz = -1.002 * (13 / 40) + 0.283 * 1.1 + 3.139 * (40 / 90) - 39.368 * (2 / 40) - 1.315 * 0.25
        assert row["kz_index"] == _approx(kz)
        # Intrinsic value: payout 2/11, roe 11/50, r = 12%.
        roe, po, be = 11 / 50, 2 / 11, 50.0
        be1 = (1 + (1 - po) * roe) * be
        iv = be + ((roe - 0.12) / 1.12) * be + ((roe - 0.12) / (1.12 * 0.12)) * be1
        assert row["intrinsic_value"] == _approx(iv)
