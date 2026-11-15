"""
Unit tests for the fiscal-year-change dedup in the FF Compustat loaders.

A firm that moves its fiscal-year-end reports two `datadate` in one calendar
year. The loaders collapse to one row per (gvkey, year) at the source (latest
datadate), which removes the downstream `(id, eom)` fan-out. US BE, OP and INV
are JKP's values at the kept record.

Covers:
  - _ff_load_funda_global     (ROW per-source collapse)
  - ff_link_accounting      (every security -> gvkey through world_msf; US CRSP early links)
  - _jkp_annual_acc           (FF OP definition, at_gr1 from achars_world)
"""

from __future__ import annotations

from datetime import date

import polars as pl

from jkp.data.aux_functions import (
    _ff_load_funda_global,
    _jkp_annual_acc,
    ff_link_accounting,
    ff_load_world_compustat,
)

# Every float column either loader casts, defaulted to None so callers can
# override only what a test needs.
_FLOAT_COLS = [
    "pstk",
    "seq",
    "ceq",
    "lt",
    "txditc",
    "txdb",
    "revt",
    "sale",
    "cogs",
    "xsga",
    "xopr",
    "xint",
    "ebitda",
    "oibdp",
    "at",
    "pstkrv",
    "pstkl",
    "itcb",
    "mib",
    "mibt",
    "mibn",
]


def _funda_row(gvkey, datadate, at_, *, indl, **over):
    """One synthetic comp_funda row. `indl` selects NA-style (INDL/STD/D) vs
    Global-style (INDL/HIST_STD/I) filter-passing values."""
    row = dict.fromkeys(_FLOAT_COLS)
    row.update(
        gvkey=gvkey,
        datadate=datadate,
        curcd="USD",
        indfmt="INDL",
        datafmt="STD" if indl else "HIST_STD",
        popsrc="D" if indl else "I",
        consol="C",
        # be = seq + txditc - ps (ps=pstk here); keep positive and finite.
        seq=at_ * 0.5,
        txditc=0.0,
        pstk=0.0,
        at=at_,
    )
    row.update(over)
    return row


def _write_funda(path, rows):
    pl.DataFrame(rows).with_columns(
        pl.col("datadate").cast(pl.Date),
        *[pl.col(c).cast(pl.Float64) for c in _FLOAT_COLS],
    ).write_parquet(path)


# Fiscal-year-change firm: two datadate in 2019 (06-30 stub + 12-31), plus a
# clean prior year. at grows 100 -> 120 across the *annual* boundary.
def _fiscal_change_rows(gvkey, *, indl):
    return [
        _funda_row(gvkey, date(2018, 12, 31), 100.0, indl=indl),
        _funda_row(gvkey, date(2019, 6, 30), 110.0, indl=indl),  # stub, must drop
        _funda_row(gvkey, date(2019, 12, 31), 120.0, indl=indl),
    ]


# =============================================================================
# ROW path
# =============================================================================


def test_ff_load_funda_global_collapses_fiscal_change(test_paths):
    _write_funda(
        test_paths.raw_tables_dir / "comp_g_funda.parquet", _fiscal_change_rows("001", indl=False)
    )

    out = _ff_load_funda_global(test_paths).collect()

    # One row per (gvkey, year); the 2019 stub (06-30) is dropped for 12-31.
    assert out.height == 2
    rows_2019 = out.filter(pl.col("year") == 2019)
    assert rows_2019.height == 1
    assert rows_2019["datadate"].item() == date(2019, 12, 31)


def test_ff_load_funda_global_indfmt_rule(test_paths):
    rows = [
        _funda_row("001", date(2018, 12, 31), 100.0, indl=False),
        _funda_row("001", date(2018, 12, 31), 999.0, indl=False, indfmt="FS"),
        _funda_row("002", date(2018, 12, 31), 100.0, indl=False, indfmt="FS"),
        _funda_row("002", date(2018, 12, 31), 100.0, indl=False, indfmt="FS"),
    ]
    _write_funda(test_paths.raw_tables_dir / "comp_g_funda.parquet", rows)

    out = _ff_load_funda_global(test_paths).collect().sort("gvkey")

    # INDL wins over FS; a date reported only in FS is kept (#329's rule).
    assert out.select("gvkey", "at").rows() == [("001", 100.0), ("002", 100.0)]


# =============================================================================
# ff_link_accounting links a security to its fiscal records
# through the gvkey on the security's June row in world_msf.
# =============================================================================


def _comp(rows):
    """rows: (gvkey, fiscal year, be, op, inv) with a December 31 fiscal year-end."""
    return pl.DataFrame(
        [(g, date(y, 12, 31), y, be, op, inv) for g, y, be, op, inv in rows],
        schema={
            "gvkey": pl.String,
            "datadate": pl.Date,
            "year": pl.Int32,
            "be": pl.Float64,
            "op": pl.Float64,
            "inv": pl.Float64,
        },
        orient="row",
    )


def _write_us_msf(test_paths, rows):
    """rows: (permno, eom, gvkey) for US CRSP securities in world_msf."""
    pl.DataFrame(
        rows, schema={"id": pl.Int64, "eom": pl.Date, "gvkey": pl.String}, orient="row"
    ).with_columns(excntry=pl.lit("USA"), source_crsp=pl.lit(1)).write_parquet(
        test_paths.interim_dir / "world_msf.parquet"
    )


def test_ff_link_accounting_through_june_gvkey(test_paths):
    # The June 2020 row links fiscal 2019, the June 2021 row fiscal 2020; fiscal
    # 2018 has no June 2019 row and stays unlinked.
    _write_us_msf(
        test_paths,
        [(10001, date(2020, 6, 30), "001"), (10001, date(2021, 6, 30), "001")],
    )
    comp = _comp([("001", y, 50.0 + y - 2018, 0.1, 0.2) for y in (2018, 2019, 2020)])
    out = ff_link_accounting(test_paths, comp)
    assert out.select("id", "year", "be").rows() == [(10001, 2019, 51.0), (10001, 2020, 52.0)]


def test_ff_link_accounting_ex_us_security_links_through_june_gvkey(test_paths):
    # A source_crsp 0 security links through its June-row gvkey like a CRSP one, but
    # gets no early link: its June 1963 row has no gvkey, so fiscal 1962 stays unlinked.
    pl.DataFrame(
        [
            (20001, date(2020, 6, 30), "001"),
            (20002, date(1963, 6, 30), None),
            (20002, date(1965, 1, 31), "002"),
        ],
        schema={"id": pl.Int64, "eom": pl.Date, "gvkey": pl.String},
        orient="row",
    ).with_columns(excntry=pl.lit("GBR"), source_crsp=pl.lit(0)).write_parquet(
        test_paths.interim_dir / "world_msf.parquet"
    )
    comp = _comp([("001", 2019, 50.0, 0.1, 0.2), ("002", 1962, 70.0, None, None)])
    out = ff_link_accounting(test_paths, comp)
    assert out.select("id", "year", "gvkey", "be").rows() == [(20001, 2019, "001", 50.0)]


def test_ff_link_accounting_june_row_without_gvkey(test_paths):
    # Outside the early-link years, a June row without a gvkey links nothing.
    _write_us_msf(
        test_paths,
        [(10001, date(2020, 6, 30), None), (10001, date(2020, 7, 31), "001")],
    )
    assert ff_link_accounting(test_paths, _comp([("001", 2019, 50.0, None, None)])).height == 0


def test_ff_link_accounting_record_without_jkp_values_dropped(test_paths):
    # JKP has no be_x, ope_x or at_gr1 for the record (here: seq missing), so it
    # leaves the fiscal records and the US sorts.
    _write_funda(
        test_paths.raw_tables_dir / "comp_funda.parquet",
        [_funda_row("001", date(2019, 12, 31), 100.0, indl=True, seq=None)],
    )
    _write_funda(
        test_paths.raw_tables_dir / "comp_g_funda.parquet",
        [_funda_row("900", date(2019, 12, 31), 100.0, indl=False)],
    )
    _write_acc_std_ann(
        test_paths.interim_dir / "acc_std_ann.parquet", [("001", None, None, None, None, None)]
    )
    _write_achars_world(test_paths.interim_dir / "achars_world.parquet", [])
    _write_us_msf(test_paths, [(10001, date(2020, 6, 30), "001")])
    comp = ff_load_world_compustat(test_paths, _jkp_annual_acc(test_paths))
    assert ff_link_accounting(test_paths, comp).height == 0


class TestEarlyLinks:
    """June 1963-66 formations (FF_CCM_BACKDATE_*): Compustat backfilled fy1962-64
    statements for firms whose gvkey link starts years later."""

    @staticmethod
    def _run(test_paths, fiscal_year, gvkey, msf_rows):
        _write_us_msf(test_paths, msf_rows)
        return ff_link_accounting(test_paths, _comp([(gvkey, fiscal_year, 50.0, None, None)]))

    def test_backfilled_year_linked(self, test_paths):
        # fy1962 feeds June 1963; the permno's first gvkey starts January 1965.
        out = self._run(
            test_paths,
            1962,
            "001",
            [(10001, date(1963, 6, 30), None), (10001, date(1965, 1, 31), "001")],
        )
        assert out.select("id", "year", "gvkey").rows() == [(10001, 1962, "001")]

    def test_no_early_link_after_gate(self, test_paths):
        # fy1970 feeds June 1971, after FF_CCM_BACKDATE_THROUGH.
        out = self._run(
            test_paths,
            1970,
            "001",
            [(10001, date(1971, 6, 30), None), (10001, date(1973, 1, 31), "001")],
        )
        assert out.height == 0

    def test_no_early_link_before_gate(self, test_paths):
        # fy1960 feeds June 1961, before FF_CCM_BACKDATE_FROM.
        out = self._run(
            test_paths,
            1960,
            "001",
            [(10001, date(1961, 6, 30), None), (10001, date(1963, 1, 31), "001")],
        )
        assert out.height == 0

    def test_distance_cap(self, test_paths):
        # A first gvkey in 1990 is more than FF_CCM_BACKDATE_MAX_YEARS after June 1963.
        out = self._run(
            test_paths,
            1962,
            "001",
            [(10001, date(1963, 6, 30), None), (10001, date(1990, 1, 31), "001")],
        )
        assert out.height == 0

    def test_first_gvkey_before_formation_not_used(self, test_paths):
        # The permno's first gvkey (001, 1955-60) predates the formation: an ended
        # link, not a late one, so gvkey 002's record from 1965 on is not linked.
        out = self._run(
            test_paths,
            1962,
            "002",
            [
                (10001, date(1955, 1, 31), "001"),
                (10001, date(1963, 6, 30), None),
                (10001, date(1965, 1, 31), "002"),
            ],
        )
        assert out.height == 0

    def test_june_gvkey_wins(self, test_paths):
        # The June row has a gvkey: it links, and no early link is looked for.
        _write_us_msf(
            test_paths,
            [(10001, date(1963, 6, 30), "001"), (10001, date(1965, 1, 31), "002")],
        )
        comp = _comp([("001", 1962, 50.0, None, None), ("002", 1962, 70.0, None, None)])
        out = ff_link_accounting(test_paths, comp)
        assert out.select("id", "year", "gvkey").rows() == [(10001, 1962, "001")]


# =============================================================================
# JKP's annual accounting for the FF loaders (_jkp_annual_acc):
# op = ope_x / (be_x + coalesce(mibt, mib + mibn)), at_gr1 from achars_world.
# =============================================================================


def _write_acc_std_ann(path, rows):
    """rows: (gvkey, be_x, ope_x, mib, mibt, mibn) on a 2019-12-31 report row."""
    pl.DataFrame(
        {
            "gvkey": [r[0] for r in rows],
            "curcd": ["USD"] * len(rows),
            "datadate": [date(2019, 12, 31)] * len(rows),
            "source": ["GLOBAL"] * len(rows),
            "be_x": [r[1] for r in rows],
            "ope_x": [r[2] for r in rows],
            "mib": [r[3] for r in rows],
            "mibt": [r[4] for r in rows],
            "mibn": [r[5] for r in rows],
            "ni_x": [1.0] * len(rows),
        },
        schema_overrides=dict.fromkeys(["be_x", "ope_x", "mib", "mibt", "mibn"], pl.Float64),
    ).write_parquet(path)


def _write_achars_world(path, rows):
    """rows: (gvkey, datadate, public_date, at_gr1)."""
    pl.DataFrame(
        rows,
        schema={
            "gvkey": pl.String,
            "datadate": pl.Date,
            "public_date": pl.Date,
            "at_gr1": pl.Float64,
        },
        orient="row",
    ).write_parquet(path)


def test_ff_load_world_compustat_op_definition(test_paths):
    gvkeys = ["101", "102", "103", "104", "105"]
    _write_funda(
        test_paths.raw_tables_dir / "comp_g_funda.parquet",
        [_funda_row(g, date(2019, 12, 31), 100.0, indl=False) for g in gvkeys],
    )
    _write_funda(
        test_paths.raw_tables_dir / "comp_funda.parquet",
        [_funda_row("999", date(2019, 12, 31), 100.0, indl=True)],
    )
    _write_acc_std_ann(
        test_paths.interim_dir / "acc_std_ann.parquet",
        [
            ("101", 50.0, None, None, None, None),  # ope_x missing -> null
            ("102", 50.0, 10.0, 2.0, 10.0, 8.0),  # mibt wins
            ("103", 50.0, 10.0, 2.0, None, 8.0),  # mib + mibn
            ("104", 50.0, 10.0, None, None, None),  # no minority interest
            ("105", -5.0, 10.0, None, None, None),  # non-positive BE -> null
        ],
    )
    _write_achars_world(test_paths.interim_dir / "achars_world.parquet", [])
    op = dict(
        ff_load_world_compustat(test_paths, _jkp_annual_acc(test_paths))
        .select("gvkey", "op")
        .iter_rows()
    )
    assert op.get("101") is None
    assert abs(op["102"] - 10.0 / 60.0) < 1e-12
    assert abs(op["103"] - 10.0 / 60.0) < 1e-12
    assert abs(op["104"] - 10.0 / 50.0) < 1e-12
    assert op.get("105") is None


def test_jkp_annual_acc_at_gr1_at_first_public_date(test_paths):
    _write_acc_std_ann(
        test_paths.interim_dir / "acc_std_ann.parquet",
        [("101", 50.0, 10.0, None, None, None), ("102", 50.0, 10.0, None, None, None)],
    )
    fy = date(2019, 12, 31)
    _write_achars_world(
        test_paths.interim_dir / "achars_world.parquet",
        [
            ("101", fy, date(2020, 5, 31), 0.9),  # later month of the same record
            ("101", fy, date(2020, 4, 30), 0.1),
        ],
    )
    out = _jkp_annual_acc(test_paths).sort("gvkey")
    # The first public_date's value; a record missing from achars_world gets null.
    assert out.select("gvkey", "at_gr1").rows() == [("101", 0.1), ("102", None)]
