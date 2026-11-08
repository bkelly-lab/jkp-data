"""Tests for add_helper_vars (monthly grid and the *_x helper variables).

Expected values are worked out by hand from the SAS definitions in
accounting_chars.sas (%add_helper_vars). Single-row tests use one firm-month
(count = 1); the accrual and issuance tests use a 14-month series so that the
12-month differences are available at count 13 and 14.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from jkp.data.aux_functions import add_helper_vars
from jkp.data.paths import DataPaths

INPUT_COLS = [
    "at", "sale", "revt", "dv", "che", "dltt", "dlc", "pstkrv", "pstkl", "pstk",
    "xopr", "cogs", "xsga", "sstk", "dvt", "prstkc", "purtshr", "xido", "xi", "do",
    "act", "rect", "invt", "aco", "lct", "ap", "txp", "lco", "ivst", "ivao",
    "ppegt", "ppent", "intan", "ao", "lo", "dp", "txditc", "txdb", "itcb", "gp",
    "seq", "ceq", "lt", "ebitda", "oibdp", "icapt", "mib", "ebit", "oiadp", "xrd",
    "xint", "pi", "spi", "nopi", "ib", "ni", "txt", "mii", "oancf", "dltis",
    "dltr", "ltdch", "dlcch", "wcapt", "capx", "fincf", "fiao", "txbcof",
]  # fmt: skip

JAN = date(2020, 1, 31)


def _month_end(i: int) -> date:
    """Month-end of the i-th month after January 2020 (i = 0 gives 2020-01-31)."""
    m = i + 1
    first_next = date(2020 + m // 12, m % 12 + 1, 1)
    return first_next - timedelta(days=1)


def _frame(rows: list[dict[str, Any]]) -> pl.LazyFrame:
    """Build the input frame; unspecified input columns are null."""
    data: dict[str, list[Any]] = {
        "gvkey": [r.get("gvkey", "A") for r in rows],
        "curcd": [r.get("curcd", "USD") for r in rows],
        "datadate": [r.get("datadate", JAN) for r in rows],
    }
    for c in INPUT_COLS:
        data[c] = [r.get(c) for r in rows]
    schema = {"gvkey": pl.String, "curcd": pl.String, "datadate": pl.Date}
    schema.update(dict.fromkeys(INPUT_COLS, pl.Float64))
    return pl.DataFrame(data, schema=schema).lazy()


def _run(test_paths: DataPaths, rows: list[dict[str, Any]]) -> pl.DataFrame:
    return add_helper_vars(test_paths, _frame(rows)).collect().sort("gvkey", "curcd", "datadate")


def _one(test_paths: DataPaths, **vals: float | None) -> dict[str, Any]:
    """Run one firm-month (count = 1) and return its output row."""
    return _run(test_paths, [vals]).row(0, named=True)


def _check(row: dict[str, Any], expected: dict[str, float | None]) -> None:
    for name, want in expected.items():
        if want is None:
            assert row[name] is None, name
        else:
            assert row[name] == pytest.approx(want), name


def _series(gvkey: str, n: int, fn: Any) -> list[dict[str, Any]]:
    """n consecutive month-ends (t = 1..n) for one firm; fn(t) gives the inputs."""
    return [{"gvkey": gvkey, "datadate": _month_end(t - 1), **fn(t)} for t in range(1, n + 1)]


# --------------------------------------------------------------------------
# Monthly grid
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestGrid:
    def test_gap_months_are_filled_with_null_rows(self, test_paths: DataPaths) -> None:
        # Jan 31 and Apr 30 observed; Feb 29 (leap year) and Mar 31 are grid-only.
        out = _run(
            test_paths,
            [
                {"datadate": date(2020, 1, 31), "sale": 1.0},
                {"datadate": date(2020, 4, 30), "sale": 4.0},
            ],
        )
        assert out["datadate"].to_list() == [
            date(2020, 1, 31),
            date(2020, 2, 29),
            date(2020, 3, 31),
            date(2020, 4, 30),
        ]
        assert out["data_available"].to_list() == [1, 0, 0, 1]
        assert out["sale_x"].to_list() == [1.0, None, None, 4.0]
        assert out["gvkey"].to_list() == ["A"] * 4
        assert out["curcd"].to_list() == ["USD"] * 4

    def test_single_observation_gives_single_row(self, test_paths: DataPaths) -> None:
        out = _run(test_paths, [{"sale": 5.0}])
        assert out.height == 1
        assert out["data_available"].to_list() == [1]

    def test_grid_is_per_firm_and_per_currency(self, test_paths: DataPaths) -> None:
        out = _run(
            test_paths,
            [
                {"gvkey": "A", "datadate": date(2020, 1, 31)},
                {"gvkey": "A", "datadate": date(2020, 3, 31)},
                {"gvkey": "B", "datadate": date(2020, 3, 31)},
                {"gvkey": "B", "datadate": date(2020, 4, 30)},
                {"gvkey": "B", "curcd": "CAD", "datadate": date(2020, 6, 30)},
            ],
        )
        got = {
            (g, c): d
            for g, c, d in out.group_by("gvkey", "curcd", maintain_order=True)
            .agg(pl.col("datadate"))
            .iter_rows()
        }
        assert got == {
            ("A", "USD"): [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)],
            ("B", "CAD"): [date(2020, 6, 30)],
            ("B", "USD"): [date(2020, 3, 31), date(2020, 4, 30)],
        }

    def test_non_month_end_observation_is_not_matched(self, test_paths: DataPaths) -> None:
        # The grid is on month-ends (SAS intnx 'e'); a mid-month datadate joins to nothing.
        out = _run(
            test_paths,
            [
                {"datadate": date(2020, 1, 15), "sale": 9.0},
                {"datadate": date(2020, 3, 31), "sale": 3.0},
            ],
        )
        assert out["datadate"].to_list() == [
            date(2020, 1, 31),
            date(2020, 2, 29),
            date(2020, 3, 31),
        ]
        assert out["data_available"].to_list() == [0, 0, 1]
        assert out["sale_x"].to_list() == [None, None, 3.0]

    def test_duplicate_rows_collapse_to_one(self, test_paths: DataPaths) -> None:
        out = _run(
            test_paths,
            [{"sale": 1.0}, {"sale": 1.0}, {"datadate": date(2020, 2, 29), "sale": 2.0}],
        )
        assert out["datadate"].to_list() == [date(2020, 1, 31), date(2020, 2, 29)]
        assert out["data_available"].to_list() == [1, 1]

    def test_output_columns(self, test_paths: DataPaths) -> None:
        out = _run(test_paths, [{"at": 1.0}])
        assert "count" not in out.columns
        assert "at" in out.columns
        assert "at_var" not in out.columns
        assert "data_available" in out.columns

    def test_negative_values_set_to_null_before_helpers(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, at=-1.0, sale=-2.0, revt=-3.0, dv=-4.0, che=-5.0)
        for name in ["at", "sale", "revt", "dv", "che"]:
            assert row[name] is None, name
        assert row["sale_x"] is None

    def test_zero_is_kept(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, sale=0.0, revt=7.0)
        assert row["sale"] == 0.0
        assert row["sale_x"] == 0.0


# --------------------------------------------------------------------------
# Income statement helpers
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestIncomeStatement:
    def test_sale_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, sale=10.0, revt=20.0)["sale_x"] == 10.0
        assert _one(test_paths, revt=20.0)["sale_x"] == 20.0
        assert _one(test_paths, sale=-5.0, revt=7.0)["sale_x"] == 7.0  # negative sale cleared
        assert _one(test_paths)["sale_x"] is None

    def test_gp_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, gp=4.0, sale=10.0, cogs=3.0)["gp_x"] == 4.0
        assert _one(test_paths, sale=10.0, cogs=3.0)["gp_x"] == 7.0  # 10 - 3
        assert _one(test_paths, revt=12.0, cogs=3.0)["gp_x"] == 9.0  # sale_x = revt
        assert _one(test_paths, sale=10.0)["gp_x"] is None  # cogs missing

    def test_opex_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, xopr=8.0, cogs=3.0, xsga=2.0)["opex_x"] == 8.0
        assert _one(test_paths, cogs=3.0, xsga=2.0)["opex_x"] == 5.0
        assert _one(test_paths, cogs=3.0)["opex_x"] is None
        assert _one(test_paths)["opex_x"] is None

    def test_ebitda_x_chain(self, test_paths: DataPaths) -> None:
        base = {"sale": 100.0, "xopr": 60.0, "gp": 50.0, "xsga": 15.0}
        assert _one(test_paths, ebitda=1.0, oibdp=2.0, **base)["ebitda_x"] == 1.0
        assert _one(test_paths, oibdp=2.0, **base)["ebitda_x"] == 2.0
        assert _one(test_paths, **base)["ebitda_x"] == 40.0  # sale_x - opex_x = 100 - 60
        # opex_x missing: gp_x - xsga = 50 - 15
        assert _one(test_paths, gp=50.0, xsga=15.0)["ebitda_x"] == 35.0
        assert _one(test_paths)["ebitda_x"] is None

    def test_ebit_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, ebit=1.0, oiadp=2.0, oibdp=9.0, dp=3.0)["ebit_x"] == 1.0
        assert _one(test_paths, oiadp=2.0, oibdp=9.0, dp=3.0)["ebit_x"] == 2.0
        assert _one(test_paths, oibdp=9.0, dp=3.0)["ebit_x"] == 6.0  # ebitda_x - dp
        assert _one(test_paths, oibdp=9.0)["ebit_x"] is None
        assert _one(test_paths)["ebit_x"] is None

    def test_op_x_and_ope_x(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, oibdp=10.0, xrd=2.0, xint=3.0)
        _check(row, {"op_x": 12.0, "ope_x": 7.0})  # 10 + 2; 10 - 3
        row = _one(test_paths, oibdp=10.0)
        _check(row, {"op_x": 10.0, "ope_x": None})  # xrd missing counts as 0; xint missing
        _check(_one(test_paths), {"op_x": None, "ope_x": None})

    def test_pi_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, pi=5.0, ebit=9.0, xint=1.0)["pi_x"] == 5.0
        # ebit_x - xint + spi + nopi = 9 - 1 + 2 + 3
        assert _one(test_paths, ebit=9.0, xint=1.0, spi=2.0, nopi=3.0)["pi_x"] == 13.0
        # spi and nopi missing count as 0
        assert _one(test_paths, ebit=9.0, xint=1.0)["pi_x"] == 8.0
        assert _one(test_paths, ebit=9.0)["pi_x"] is None  # xint missing
        assert _one(test_paths)["pi_x"] is None

    def test_xido_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, xido=1.0, xi=5.0, do=2.0)["xido_x"] == 1.0
        assert _one(test_paths, xi=5.0, do=2.0)["xido_x"] == 7.0
        assert _one(test_paths, xi=5.0)["xido_x"] == 5.0  # do missing counts as 0
        assert _one(test_paths, do=2.0)["xido_x"] is None
        assert _one(test_paths)["xido_x"] is None

    def test_ni_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, ib=4.0, ni=9.0, xi=1.0)["ni_x"] == 4.0
        # ni - xido_x = 9 - (1 + 2)
        assert _one(test_paths, ni=9.0, xi=1.0, do=2.0)["ni_x"] == 6.0
        # no ib, ni, xido_x: pi_x - txt - mii = 10 - 3 - 1
        assert _one(test_paths, pi=10.0, txt=3.0, mii=1.0)["ni_x"] == 6.0
        assert _one(test_paths, pi=10.0, txt=3.0)["ni_x"] == 7.0  # mii missing counts as 0
        assert _one(test_paths, pi=10.0)["ni_x"] is None  # txt missing
        assert _one(test_paths)["ni_x"] is None

    def test_nix_x_chain_and_fi_x(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, ni=9.0, ib=4.0)["nix_x"] == 9.0
        # ni_x + xido_x = ib + xido = 4 + 3 (xido given, ni missing)
        assert _one(test_paths, ib=4.0, xido=3.0)["nix_x"] == 7.0
        # xido_x missing: ni_x + 0 = 4
        assert _one(test_paths, ib=4.0)["nix_x"] == 4.0
        assert _one(test_paths)["nix_x"] is None
        _check(_one(test_paths, ni=9.0, xint=2.0), {"fi_x": 11.0})
        _check(_one(test_paths, ni=9.0), {"fi_x": None})


# --------------------------------------------------------------------------
# Balance sheet helpers
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestBalanceSheet:
    def test_debt_x_uses_sas_sum(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, dltt=3.0, dlc=4.0)["debt_x"] == 7.0
        assert _one(test_paths, dltt=3.0)["debt_x"] == 3.0
        assert _one(test_paths, dlc=4.0)["debt_x"] == 4.0
        assert _one(test_paths)["debt_x"] is None

    def test_pstk_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, pstkrv=1.0, pstkl=2.0, pstk=3.0)["pstk_x"] == 1.0
        assert _one(test_paths, pstkl=2.0, pstk=3.0)["pstk_x"] == 2.0
        assert _one(test_paths, pstk=3.0)["pstk_x"] == 3.0
        assert _one(test_paths)["pstk_x"] is None

    def test_ca_x_and_cl_x_chains(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, act=50.0, rect=1.0, invt=2.0, che=3.0, aco=4.0)["ca_x"] == 50.0
        # rect + invt + che + aco = 1 + 2 + 3 + 4
        assert _one(test_paths, rect=1.0, invt=2.0, che=3.0, aco=4.0)["ca_x"] == 10.0
        assert _one(test_paths, rect=1.0, invt=2.0, che=3.0)["ca_x"] is None
        assert _one(test_paths, lct=30.0, ap=1.0, dlc=2.0, txp=3.0, lco=4.0)["cl_x"] == 30.0
        # ap + dlc + txp + lco = 1 + 2 + 3 + 4
        assert _one(test_paths, ap=1.0, dlc=2.0, txp=3.0, lco=4.0)["cl_x"] == 10.0
        assert _one(test_paths, ap=1.0, dlc=2.0, txp=3.0)["cl_x"] is None

    def test_current_and_noncurrent_pieces(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, act=50.0, che=10.0, lct=30.0, dlc=5.0, lt=80.0, dltt=20.0)
        _check(
            row,
            {
                "coa_x": 40.0,  # 50 - 10
                "col_x": 25.0,  # 30 - 5
                "cowc_x": 15.0,  # 40 - 25
                "ncl_x": 50.0,  # 80 - 30
                "ncol_x": 30.0,  # 80 - 30 - 20
                "ol_x": 55.0,  # col_x + ncol_x = 25 + 30
                "nwc_x": 20.0,  # 50 - 30
            },
        )

    def test_dlc_missing_counts_as_zero_in_col_x(self, test_paths: DataPaths) -> None:
        _check(_one(test_paths, lct=30.0), {"col_x": 30.0})
        _check(_one(test_paths), {"col_x": None, "coa_x": None, "nwc_x": None})

    def test_ncol_x_needs_dltt(self, test_paths: DataPaths) -> None:
        _check(_one(test_paths, lt=80.0, lct=30.0), {"ncol_x": None, "ncl_x": 50.0})

    def test_fna_x_fnl_x_nfna_x(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, ivst=2.0, ivao=3.0, dltt=10.0, dlc=4.0, pstk=1.0)
        _check(row, {"fna_x": 5.0, "fnl_x": 15.0, "nfna_x": -10.0})  # 14 + 1; 5 - 15
        # fna_x is 0 when both investments are missing; fnl_x needs debt_x
        _check(_one(test_paths, ivst=2.0), {"fna_x": 2.0, "fnl_x": None, "nfna_x": None})
        _check(_one(test_paths), {"fna_x": 0.0})

    def test_caliq_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, act=50.0, invt=8.0, che=3.0, rect=2.0)["caliq_x"] == 42.0
        assert _one(test_paths, che=3.0, rect=2.0)["caliq_x"] == 5.0  # ca_x missing
        assert _one(test_paths, che=3.0)["caliq_x"] is None

    def test_netdebt_x(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, dltt=10.0, dlc=4.0, che=6.0)["netdebt_x"] == 8.0
        assert _one(test_paths, dltt=10.0)["netdebt_x"] == 10.0  # che missing counts as 0
        assert _one(test_paths, che=6.0)["netdebt_x"] is None

    def test_txditc_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, txditc=5.0, txdb=1.0, itcb=2.0)["txditc_x"] == 5.0
        assert _one(test_paths, txdb=1.0, itcb=2.0)["txditc_x"] == 3.0
        assert _one(test_paths, itcb=2.0)["txditc_x"] == 2.0
        assert _one(test_paths)["txditc_x"] is None

    def test_seq_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, seq=10.0, ceq=20.0, at=50.0, lt=30.0)["seq_x"] == 10.0
        assert _one(test_paths, ceq=20.0, pstk=4.0)["seq_x"] == 24.0  # ceq + pstk_x
        assert _one(test_paths, ceq=20.0)["seq_x"] == 20.0  # no preferred
        assert _one(test_paths, at=50.0, lt=30.0)["seq_x"] == 20.0  # at - lt
        assert _one(test_paths, at=50.0)["seq_x"] is None
        assert _one(test_paths)["seq_x"] is None

    def test_at_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, at=100.0, seq=10.0, dltt=5.0)["at_x"] == 100.0
        # seq_x + dltt + lct + lo + txditc = 10 + 5 + 2 + 3 + 4
        row = _one(test_paths, seq=10.0, dltt=5.0, lct=2.0, lo=3.0, txditc=4.0)
        assert row["at_x"] == 24.0
        assert _one(test_paths, seq=10.0, dltt=5.0)["at_x"] == 15.0  # optional terms are 0
        assert _one(test_paths, seq=10.0)["at_x"] is None  # dltt missing
        # negative at is cleared, so the fallback applies
        assert _one(test_paths, at=-1.0, seq=10.0, dltt=5.0)["at_x"] == 15.0

    def test_derived_asset_measures(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, at=100.0, act=40.0, che=10.0, ivao=6.0, intan=8.0)
        _check(
            row,
            {
                "nca_x": 60.0,  # 100 - 40
                "ncoa_x": 54.0,  # 100 - 40 - 6
                "oa_x": 84.0,  # coa_x + ncoa_x = 30 + 54
                "aliq_x": 58.5,  # 10 + 0.75 * 30 + 0.5 * (60 - 8)
            },
        )

    def test_ncoa_x_without_ivao_and_intan(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, at=100.0, act=40.0, che=10.0)
        _check(row, {"ncoa_x": 60.0, "aliq_x": 10.0 + 22.5 + 30.0})

    def test_nncoa_noa(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, at=100.0, act=40.0, che=10.0, ivao=6.0, lct=30.0, lt=80.0, dltt=20.0)
        # ncoa_x = 54, ncol_x = 80 - 30 - 20 = 30 -> nncoa_x = 24
        # oa_x = 30 + 54 = 84, ol_x = col_x + ncol_x = 30 + 30 = 60 -> noa_x = 24
        _check(row, {"nncoa_x": 24.0, "oa_x": 84.0, "ol_x": 60.0, "noa_x": 24.0})

    def test_ppeinv_x_and_lnoa_x(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, ppegt=10.0, invt=2.0, ppent=6.0, intan=3.0, ao=4.0, lo=1.0, dp=2.0)
        _check(row, {"ppeinv_x": 12.0, "lnoa_x": 14.0})  # 6 + 3 + 4 - 1 + 2
        _check(_one(test_paths, ppegt=10.0), {"ppeinv_x": None, "lnoa_x": None})

    def test_be_x(self, test_paths: DataPaths) -> None:
        # seq_x + txditc_x - pstk_x = 30 + 5 - 4
        assert _one(test_paths, seq=30.0, txditc=5.0, pstk=4.0)["be_x"] == 31.0
        assert _one(test_paths, seq=30.0)["be_x"] == 30.0  # optional terms are 0
        assert _one(test_paths, ceq=30.0, pstk=4.0, txdb=5.0)["be_x"] == 30.0 + 4.0 + 5.0 - 4.0
        assert _one(test_paths)["be_x"] is None

    def test_be_x_nonpositive_is_null(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, seq=0.0)["be_x"] is None
        assert _one(test_paths, seq=-3.0)["be_x"] is None
        assert _one(test_paths, seq=4.0, pstk=4.0)["be_x"] is None  # 4 - 4 = 0

    def test_bev_x_chain(self, test_paths: DataPaths) -> None:
        # icapt + dlc - che = 50 + 4 - 6
        assert _one(test_paths, icapt=50.0, dlc=4.0, che=6.0, seq=99.0)["bev_x"] == 48.0
        assert _one(test_paths, icapt=50.0)["bev_x"] == 50.0  # dlc, che count as 0
        # netdebt_x + seq_x + mib = (10 + 4 - 6) + 30 + 2
        row = _one(test_paths, dltt=10.0, dlc=4.0, che=6.0, seq=30.0, mib=2.0)
        assert row["bev_x"] == 40.0
        assert _one(test_paths, dltt=10.0, seq=30.0)["bev_x"] == 40.0  # che, mib count as 0
        assert _one(test_paths, seq=30.0)["bev_x"] is None  # netdebt_x missing
        assert _one(test_paths)["bev_x"] is None

    def test_bev_x_nonpositive_is_null(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, icapt=5.0, che=5.0)["bev_x"] is None  # 0
        assert _one(test_paths, icapt=1.0, che=5.0)["bev_x"] is None  # negative


# --------------------------------------------------------------------------
# Cash flow / financing helpers (single month)
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestFinancingSingleMonth:
    def test_eqbb_x_uses_sas_sum(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, prstkc=2.0, purtshr=3.0)["eqbb_x"] == 5.0
        assert _one(test_paths, purtshr=3.0)["eqbb_x"] == 3.0
        assert _one(test_paths, prstkc=2.0)["eqbb_x"] == 2.0
        assert _one(test_paths)["eqbb_x"] is None

    def test_eqis_and_eqnetis(self, test_paths: DataPaths) -> None:
        _check(_one(test_paths, sstk=10.0, prstkc=4.0), {"eqis_x": 10.0, "eqnetis_x": 6.0})
        _check(_one(test_paths, sstk=10.0), {"eqis_x": 10.0, "eqnetis_x": 10.0})
        _check(_one(test_paths, prstkc=4.0), {"eqis_x": None, "eqnetis_x": -4.0})
        _check(_one(test_paths), {"eqis_x": None, "eqnetis_x": None})

    def test_div_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, dvt=4.0, dv=3.0)["div_x"] == 4.0
        assert _one(test_paths, dv=3.0)["div_x"] == 3.0
        assert _one(test_paths, dv=-3.0)["div_x"] is None
        assert _one(test_paths)["div_x"] is None

    def test_eqpo_and_eqnpo(self, test_paths: DataPaths) -> None:
        # div_x = 4, eqbb_x = 5, eqnetis_x = 10 - 5 = 5
        row = _one(test_paths, dvt=4.0, prstkc=2.0, purtshr=3.0, sstk=10.0)
        _check(row, {"eqpo_x": 9.0, "eqnpo_x": -1.0})
        # eqbb_x missing makes eqpo_x null (plain +), eqnetis_x = 10
        _check(_one(test_paths, dvt=4.0, sstk=10.0), {"eqpo_x": None, "eqnpo_x": -6.0})

    def test_dltnetis_x_chain_at_count_one(self, test_paths: DataPaths) -> None:
        _check(_one(test_paths, dltis=5.0, dltr=2.0, ltdch=9.0), {"dltnetis_x": 3.0})
        _check(_one(test_paths, dltis=5.0), {"dltnetis_x": 5.0})
        _check(_one(test_paths, dltr=2.0), {"dltnetis_x": -2.0})
        _check(_one(test_paths, ltdch=4.0), {"dltnetis_x": 4.0})
        # nothing available in the first 12 months: no dif12 fallback
        _check(_one(test_paths, dltt=10.0), {"dltnetis_x": None})

    def test_dstnetis_x_at_count_one(self, test_paths: DataPaths) -> None:
        _check(_one(test_paths, dlcch=7.0), {"dstnetis_x": 7.0})
        _check(_one(test_paths, dlc=7.0), {"dstnetis_x": None})

    def test_dbnetis_netis_fincf(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, sstk=10.0, dlcch=7.0, dv=2.0, fiao=1.0)
        # dbnetis_x = sum(7, null) = 7; netis_x = 10 + 7 = 17; fincf_x = 17 - 2 + 1 + 0
        _check(row, {"dbnetis_x": 7.0, "netis_x": 17.0, "fincf_x": 16.0})

    def test_dbnetis_x_sums_both_parts(self, test_paths: DataPaths) -> None:
        row = _one(test_paths, dlcch=7.0, ltdch=4.0)
        _check(row, {"dbnetis_x": 11.0, "netis_x": None})  # eqnetis_x missing

    def test_fincf_x_chain(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, fincf=3.0, sstk=10.0, dv=2.0)["fincf_x"] == 3.0
        # netis_x - dv + fiao + txbcof = 10 - 2 + 1 + 4
        row = _one(test_paths, sstk=10.0, dlcch=0.0, dv=2.0, fiao=1.0, txbcof=4.0)
        assert row["fincf_x"] == 13.0
        assert _one(test_paths, sstk=10.0, dlcch=0.0)["fincf_x"] is None  # dv missing
        assert _one(test_paths)["fincf_x"] is None

    def test_ocf_x_chain_at_count_one(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, oancf=7.0, ib=10.0, dp=3.0)["ocf_x"] == 7.0
        # oacc_x is null in the first 12 months, so ni_x + dp - wcapt = 10 + 3 - 2
        assert _one(test_paths, ib=10.0, dp=3.0, wcapt=2.0)["ocf_x"] == 11.0
        assert _one(test_paths, ib=10.0, dp=3.0)["ocf_x"] == 13.0
        assert _one(test_paths, ib=10.0)["ocf_x"] is None  # dp missing
        assert _one(test_paths)["ocf_x"] is None

    def test_fcf_x(self, test_paths: DataPaths) -> None:
        assert _one(test_paths, oancf=7.0, capx=2.0)["fcf_x"] == 5.0
        assert _one(test_paths, oancf=7.0)["fcf_x"] is None

    def test_oacc_from_cash_flow_needs_history(self, test_paths: DataPaths) -> None:
        # ib - oancf would be 3, but count <= 12 forces null for oacc_x, tacc_x, cop_x
        row = _one(test_paths, ib=10.0, oancf=7.0, oibdp=20.0)
        _check(row, {"oacc_x": None, "tacc_x": None, "cop_x": None})


# --------------------------------------------------------------------------
# 12-month differences (14-month series)
# --------------------------------------------------------------------------


def _accrual_inputs(t: int) -> dict[str, float]:
    """Balance sheet with moving pieces. Derived by hand:

    ca_x = act = 100 + t;  coa_x = act - che = 90 + t
    col_x = lct - dlc = 50 - (5 + t) = 45 - t;  cowc_x = 45 + 2t
    ncoa_x = at - act = 100 + t;  ncol_x = lt - lct - dltt = 80 - 50 - (20 + 2t) = 10 - 2t
    nncoa_x = 90 + 3t
    fna_x = 0;  fnl_x = debt_x = dltt + dlc = 25 + 3t;  nfna_x = -(25 + 3t)
    """
    return {
        "act": 100.0 + t,
        "che": 10.0,
        "lct": 50.0,
        "dlc": 5.0 + t,
        "at": 200.0 + 2 * t,
        "lt": 80.0,
        "dltt": 20.0 + 2 * t,
        "oibdp": 100.0,
    }


@pytest.mark.unit
class TestTwelveMonthDifferences:
    def test_accruals_from_balance_sheet_changes(self, test_paths: DataPaths) -> None:
        out = _run(test_paths, _series("A", 14, _accrual_inputs))
        # at t = 13 vs t = 1: dif12(cowc_x) = 2 * 12 = 24, dif12(nncoa_x) = 3 * 12 = 36,
        # dif12(nfna_x) = -3 * 12 = -36
        for t in (13, 14):
            row = out.row(t - 1, named=True)
            _check(
                row,
                {
                    "oacc_x": 60.0,  # 24 + 36
                    "tacc_x": 24.0,  # 60 - 36
                    "cop_x": 40.0,  # ebitda 100 + xrd 0 - 60
                },
            )

    def test_accruals_null_through_count_twelve(self, test_paths: DataPaths) -> None:
        out = _run(test_paths, _series("A", 14, _accrual_inputs))
        first12 = out.head(12)
        assert first12["oacc_x"].null_count() == 12
        assert first12["tacc_x"].null_count() == 12
        assert first12["cop_x"].null_count() == 12

    def test_oacc_prefers_income_minus_operating_cash_flow(self, test_paths: DataPaths) -> None:
        def inputs(t: int) -> dict[str, float]:
            return {**_accrual_inputs(t), "ib": 10.0, "oancf": 4.0}

        out = _run(test_paths, _series("A", 14, inputs))
        row = out.row(12, named=True)  # t = 13
        _check(row, {"oacc_x": 6.0, "tacc_x": 6.0 - 36.0, "ocf_x": 4.0, "cop_x": 94.0})
        assert out["oacc_x"][11] is None  # t = 12, still guarded despite ni_x and oancf

    def test_ocf_x_falls_back_to_ni_minus_oacc(self, test_paths: DataPaths) -> None:
        def inputs(t: int) -> dict[str, float]:
            return {**_accrual_inputs(t), "ib": 10.0}

        out = _run(test_paths, _series("A", 14, inputs))
        # t = 13: oacc_x = ni_x - oancf is null, so oacc_x = 60 from the diffs; ocf_x = 10 - 60
        _check(out.row(12, named=True), {"oacc_x": 60.0, "ocf_x": -50.0})

    def test_ocf_x_without_ni_is_null_after_count_twelve(self, test_paths: DataPaths) -> None:
        out = _run(test_paths, _series("A", 14, _accrual_inputs))
        # ni_x null, so neither the second nor third ocf_x branch exists
        assert out["ocf_x"].null_count() == 14

    def test_dltnetis_and_dstnetis_use_dif12_after_count_twelve(
        self, test_paths: DataPaths
    ) -> None:
        out = _run(test_paths, _series("A", 14, _accrual_inputs))
        assert out["dltnetis_x"].head(12).null_count() == 12
        assert out["dstnetis_x"].head(12).null_count() == 12
        row = out.row(12, named=True)  # t = 13
        # dltt rises 2 per month -> 24; dlc rises 1 per month -> 12; dbnetis_x = 12 + 24
        _check(row, {"dltnetis_x": 24.0, "dstnetis_x": 12.0, "dbnetis_x": 36.0})

    def test_flow_items_override_dif12_after_count_twelve(self, test_paths: DataPaths) -> None:
        def inputs(t: int) -> dict[str, float]:
            return {**_accrual_inputs(t), "dltis": 8.0, "dltr": 3.0, "dlcch": 2.0}

        row = _run(test_paths, _series("A", 14, inputs)).row(12, named=True)
        _check(row, {"dltnetis_x": 5.0, "dstnetis_x": 2.0, "dbnetis_x": 7.0})

    def test_netis_x_after_count_twelve(self, test_paths: DataPaths) -> None:
        def inputs(t: int) -> dict[str, float]:
            return {**_accrual_inputs(t), "sstk": 10.0, "prstkc": 4.0}

        row = _run(test_paths, _series("A", 14, inputs)).row(12, named=True)
        _check(row, {"eqnetis_x": 6.0, "dbnetis_x": 36.0, "netis_x": 42.0})

    def test_missing_month_breaks_the_difference(self, test_paths: DataPaths) -> None:
        # t = 1 is absent, so the grid starts at t = 2 and t = 14 is only count 13;
        # its t - 12 = 2 partner has no data (grid-only row of nulls is not the case here:
        # the partner row exists with data), so we remove t = 2 instead.
        rows = [r for r in _series("A", 14, _accrual_inputs) if r["datadate"] != _month_end(1)]
        out = _run(test_paths, rows)
        assert out.height == 14
        row13 = out.row(12, named=True)  # t = 13: partner t = 1 exists
        assert row13["oacc_x"] == pytest.approx(60.0)
        row14 = out.row(13, named=True)  # t = 14: partner t = 2 is a null grid row
        _check(row14, {"oacc_x": None, "tacc_x": None})

    def test_firms_do_not_mix(self, test_paths: DataPaths) -> None:
        def b_inputs(t: int) -> dict[str, float]:
            return {**_accrual_inputs(t), "dltt": 1000.0 * t, "act": 500.0}

        rows = _series("A", 14, _accrual_inputs) + _series("B", 14, b_inputs)
        out = _run(test_paths, rows)
        a = out.filter(pl.col("gvkey") == "A")
        b = out.filter(pl.col("gvkey") == "B")
        _check(a.row(12, named=True), {"oacc_x": 60.0, "dltnetis_x": 24.0})
        # B: dltt rises 1000 per month -> dltnetis_x = 12000; act constant -> cowc_x moves
        # only through col_x (-1 per month): dif12 = +12; ncol_x = 30 - 1000t -> dif12(nncoa)
        # = (ncoa constant: at - act changes by 2 per month = +24) - (-12000) = 12024
        b13 = b.row(12, named=True)
        _check(b13, {"dltnetis_x": 12000.0, "oacc_x": 12.0 + 24.0 + 12000.0})
        assert b["oacc_x"].head(12).null_count() == 12
        assert a["oacc_x"].head(12).null_count() == 12

    def test_currencies_do_not_mix(self, test_paths: DataPaths) -> None:
        rows = _series("A", 13, _accrual_inputs) + [
            {**r, "curcd": "CAD"} for r in _series("A", 3, _accrual_inputs)
        ]
        out = _run(test_paths, rows)
        cad = out.filter(pl.col("curcd") == "CAD")
        assert cad["oacc_x"].null_count() == 3
        assert out.filter(pl.col("curcd") == "USD").row(12, named=True)["oacc_x"] == pytest.approx(
            60.0
        )
