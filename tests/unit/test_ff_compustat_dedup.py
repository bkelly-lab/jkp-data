"""
Unit tests for the fiscal-year-change dedup in the FF Compustat loaders.

A firm that moves its fiscal-year-end reports two `datadate` in one calendar
year. The loaders collapse to one row per (gvkey, year) at the source (latest
datadate), which removes the downstream `(id, eom)` fan-out. US BE, OP and INV
are JKP's values at the kept record.

Covers:
  - _ff_load_funda_global     (ROW per-source collapse)
  - ff_load_compustat_us      (US collapse + JKP be/op/inv at the kept record)
  - _jkp_annual_acc           (FF OP definition, at_gr1 from achars_world)
"""

from __future__ import annotations

from datetime import date

import polars as pl

from jkp.data.aux_functions import (
    _ff_load_funda_global,
    _ff_us_accounting,
    _jkp_annual_acc,
    ff_load_compustat_us,
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


def _acc_us(test_paths, at_gr1=None, op=None):
    """_ff_us_accounting(ff5=True) on the written comp_funda, with JKP's annual accounting
    mocked per written record: be_x = seq (the rows carry no deferred taxes or preferred
    stock), null when not positive; op and at_gr1 from {(gvkey, datadate): value} maps,
    none by default."""
    at_gr1, op = at_gr1 or {}, op or {}
    funda = pl.read_parquet(
        test_paths.raw_tables_dir / "comp_funda.parquet", columns=["gvkey", "datadate", "seq"]
    )
    keys = list(zip(funda["gvkey"], funda["datadate"], strict=True))
    jkp_acc = pl.DataFrame(
        {
            "gvkey": funda["gvkey"],
            "datadate": funda["datadate"],
            "be_x": [v if v is not None and v > 0 else None for v in funda["seq"]],
            "op": [op.get(k) for k in keys],
            "at_gr1": [at_gr1.get(k) for k in keys],
        },
        schema={
            "gvkey": pl.String,
            "datadate": pl.Date,
            "be_x": pl.Float64,
            "op": pl.Float64,
            "at_gr1": pl.Float64,
        },
    )
    return _ff_us_accounting(test_paths, True, jkp_acc).collect()


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
# US path
# =============================================================================


def _write_lnkhist(path, gvkey, permno):
    pl.DataFrame(
        {
            "gvkey": [gvkey],
            "lpermno": [permno],
            "linkprim": ["P"],
            "linktype": ["LC"],
            "linkdt": [date(1990, 1, 1)],
            "linkenddt": [None],
        }
    ).with_columns(
        pl.col("linkdt").cast(pl.Date),
        pl.col("linkenddt").cast(pl.Date),
    ).write_parquet(path)


def test_ff_load_compustat_us_collapses_and_takes_jkp_values(test_paths):
    _write_funda(
        test_paths.raw_tables_dir / "comp_funda.parquet", _fiscal_change_rows("001", indl=True)
    )
    _write_lnkhist(test_paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet", "001", 10001)
    stub, kept = ("001", date(2019, 6, 30)), ("001", date(2019, 12, 31))
    acc = _acc_us(test_paths, at_gr1={stub: 0.10, kept: 0.20}, op={stub: 0.3, kept: 0.4})

    out = ff_load_compustat_us(test_paths, acc, ff5=True)

    # Unique per (permno, year) — no fiscal-change fan-out.
    keyed = out.select("permno", "year")
    assert keyed.height == keyed.unique().height

    # BE, OP and INV are JKP's at the kept (latest) 2019 record; null where JKP has none.
    assert out.sort("year").select("year", "be", "op", "inv").rows() == [
        (2018, 50.0, None, None),
        (2019, 60.0, 0.4, 0.20),
    ]


def test_ff_load_compustat_us_drops_record_without_jkp_be(test_paths):
    # JKP's be_x is null (here: seq missing), so the record leaves the US sorts.
    _write_funda(
        test_paths.raw_tables_dir / "comp_funda.parquet",
        [_funda_row("001", date(2019, 12, 31), 100.0, indl=True, seq=None)],
    )
    _write_lnkhist(test_paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet", "001", 10001)
    assert ff_load_compustat_us(test_paths, _acc_us(test_paths), ff5=True).height == 0


# =============================================================================
# CCM backfill rescue (FF_CCM_BACKDATE_*): June 1963-66 formations may use
# fiscal rows that predate the permno's first link window (Compustat added
# ~200 small firms/yr around 1964-66 and backfilled fy1962-64; CCM stamps
# linkdt at coverage start, so the standard window filter rejects exactly
# the rows French's portfolios include).
# =============================================================================


def _write_lnkhist_rows(path, rows):
    """rows: list of (gvkey, permno, linkdt[, linkenddt]) — P/LC links, open-ended
    unless a linkenddt is given."""
    pl.DataFrame(
        {
            "gvkey": [r[0] for r in rows],
            "lpermno": [r[1] for r in rows],
            "linkprim": ["P"] * len(rows),
            "linktype": ["LC"] * len(rows),
            "linkdt": [r[2] for r in rows],
            "linkenddt": [r[3] if len(r) > 3 else None for r in rows],
        }
    ).with_columns(
        pl.col("linkdt").cast(pl.Date),
        pl.col("linkenddt").cast(pl.Date),
    ).write_parquet(path)


class TestCcmBackdateRescue:
    @staticmethod
    def _run(test_paths, funda_rows, lnk_rows):
        _write_funda(test_paths.raw_tables_dir / "comp_funda.parquet", funda_rows)
        _write_lnkhist_rows(test_paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet", lnk_rows)
        return ff_load_compustat_us(test_paths, _acc_us(test_paths), ff5=True)

    def test_backfilled_year_rescued(self, test_paths):
        # fy1962 -> jun_end 1963-06-30, inside [1963, 1966]; link starts
        # 1965-01-01 (permno's first, within 5y of the formation).
        out = self._run(
            test_paths,
            [_funda_row("001", date(1962, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1965, 1, 1))],
        )
        assert out.filter((pl.col("permno") == 10001) & (pl.col("year") == 1962)).height == 1

    def test_no_rescue_after_gate(self, test_paths):
        # fy1970 -> jun_end 1971-06-30 > THROUGH: standard window only.
        out = self._run(
            test_paths,
            [_funda_row("001", date(1970, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1973, 1, 1))],
        )
        assert out.height == 0

    def test_no_rescue_before_gate(self, test_paths):
        # fy1960 -> jun_end 1961-06-30 < FROM.
        out = self._run(
            test_paths,
            [_funda_row("001", date(1960, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1963, 1, 1))],
        )
        assert out.height == 0

    def test_distance_cap(self, test_paths):
        # Link 1990 claiming fy1962: > MAX_YEARS after the formation.
        out = self._run(
            test_paths,
            [_funda_row("001", date(1962, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1990, 1, 1))],
        )
        assert out.height == 0

    def test_first_link_guard(self, test_paths):
        # permno's first link belongs to another gvkey (1955): the 1965 link
        # is not first -> no rescue (recycled-permno protection).
        out = self._run(
            test_paths,
            [_funda_row("002", date(1962, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1955, 1, 1)), ("002", 10001, date(1965, 1, 1))],
        )
        assert out.height == 0

    def test_ended_first_link_not_rescued(self, test_paths):
        # The permno's first link runs 1960-01-01 to 1963-03-31, so it ended
        # before the June 1963 formation: it is not a late-starting link.
        out = self._run(
            test_paths,
            [_funda_row("001", date(1962, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1960, 1, 1), date(1963, 3, 31))],
        )
        assert out.height == 0

    def test_in_window_beats_backdated(self, test_paths):
        # Two gvkeys with fy1962 rows for one permno: gvkey 001 in-window
        # (first link, 1955), gvkey 002 backdated. Dedup keeps the in-window
        # match. (002 also fails the first-link guard; the sort-priority is
        # belt-and-braces.)
        out = self._run(
            test_paths,
            [
                _funda_row("001", date(1962, 12, 31), 100.0, indl=True),
                _funda_row("002", date(1962, 12, 31), 200.0, indl=True),
            ],
            [("001", 10001, date(1955, 1, 1)), ("002", 10001, date(1965, 1, 1))],
        )
        rows = out.filter((pl.col("permno") == 10001) & (pl.col("year") == 1962))
        assert rows.height == 1
        assert rows["gvkey"][0] == "001"


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
