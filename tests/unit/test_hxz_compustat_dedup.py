"""
Unit tests for the fiscal-year-change dedup in the HXZ Compustat loaders
(Part B of the HXZ fix — mirrors test_ff_compustat_dedup.py).

A firm that moves its fiscal-year-end reports two `datadate` in one calendar
year. Before the fix, `inv` (= `ia_hxz`) was computed over the raw datadate
sequence, so the `at.shift(1)` lag (gated on `fy_gap == 1`) spanned the
sub-annual stub and nulled `inv` on the surviving row. The loaders now collapse
to one row per `(gvkey, year)` (latest datadate) BEFORE the lag.

Covers:
  - hxz_load_funda          (US annual inv)
  - hxz_load_compustat_row  (ROW INDL/FS rule, NA record preferred, annual roe_a)
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import hxz_load_compustat_row, hxz_load_funda

# Columns hxz_load_funda references, defaulted to None.
_FLOAT_COLS = [
    "pstk",
    "pstkrv",
    "pstkl",
    "itcb",
    "seq",
    "ceq",
    "lt",
    "txditc",
    "txdb",
    "at",
    "ib",
    "csho",
    "ajex",
]


def _row(gvkey, datadate, at_, **over):
    row = dict.fromkeys(_FLOAT_COLS)
    row.update(gvkey=gvkey, datadate=datadate, seq=at_ * 0.5, at=at_, ib=1.0)
    row.update(over)
    return row


# Fiscal-year-change firm: 2019 has a 06-30 stub + a 12-31 close; at grows
# 100 -> 120 across the *annual* (2018-12 -> 2019-12) boundary.
def _fiscal_change_rows(gvkey):
    return [
        _row(gvkey, date(2018, 12, 31), 100.0),
        _row(gvkey, date(2019, 6, 30), 110.0),  # stub, must drop
        _row(gvkey, date(2019, 12, 31), 120.0),
    ]


def test_hxz_load_funda_collapses_and_recovers_inv(test_paths):
    pl.DataFrame(_fiscal_change_rows("001")).with_columns(
        pl.col("datadate").cast(pl.Date),
        *[pl.col(c).cast(pl.Float64) for c in _FLOAT_COLS],
        indfmt=pl.lit("INDL"),
        datafmt=pl.lit("STD"),
        popsrc=pl.lit("D"),
        consol=pl.lit("C"),
    ).write_parquet(test_paths.raw_tables_dir / "comp_funda.parquet")

    out = hxz_load_funda(test_paths)

    keyed = out.with_columns(year=pl.col("datadate").dt.year()).select("gvkey", "year")
    assert keyed.height == keyed.unique().height  # one row per (gvkey, year)
    r2019 = out.filter(pl.col("datadate") == date(2019, 12, 31))
    assert r2019["inv"].item() is not None
    assert abs(r2019["inv"].item() - 0.20) < 1e-9


def _write_compustat(path, rows, *, na):
    pl.DataFrame(rows).with_columns(
        pl.col("datadate").cast(pl.Date),
        *[pl.col(c).cast(pl.Float64) for c in _FLOAT_COLS],
        datafmt=pl.lit("STD" if na else "HIST_STD"),
        popsrc=pl.lit("D" if na else "I"),
        consol=pl.lit("C"),
    ).write_parquet(path)


def test_hxz_load_compustat_row_indfmt_rule(test_paths):
    d = date(2019, 12, 31)
    _write_compustat(
        test_paths.raw_tables_dir / "comp_funda.parquet",
        [_row("999", d, 100.0, indfmt="INDL")],
        na=True,
    )
    _write_compustat(
        test_paths.raw_tables_dir / "comp_g_funda.parquet",
        [
            _row("001", d, 100.0, indfmt="INDL"),
            _row("001", d, 999.0, indfmt="FS"),
            _row("002", d, 100.0, indfmt="FS"),
            _row("002", d, 100.0, indfmt="FS"),
        ],
        na=False,
    )
    pl.DataFrame(
        {
            "gvkey": ["001", "002"],
            "curcd": ["USD", "USD"],
            "datadate": [d, d],
            "source": ["GLOBAL", "GLOBAL"],
            "at_x": [100.0, 100.0],
            "ni_x": [1.0, 1.0],
        }
    ).with_columns(
        pl.lit(10.0).alias(c) for c in ("be_x", "ebitda_x", "xint", "mib", "mibt", "mibn")
    ).write_parquet(test_paths.interim_dir / "acc_std_ann.parquet")

    out = hxz_load_compustat_row(test_paths)

    # One record per date; a date reported only in FS (002) is kept (#329's rule).
    assert out.select("gvkey", "datadate").rows() == [("001", d), ("002", d), ("999", d)]


def _write_keys(path, rows, *, na):
    """Fiscal-record keys only; the loaders keep (gvkey, datadate, year)."""
    pl.DataFrame(
        {"gvkey": [r[0] for r in rows], "datadate": [r[1] for r in rows]},
        schema={"gvkey": pl.Utf8, "datadate": pl.Date},
    ).with_columns(
        indfmt=pl.lit("INDL"),
        datafmt=pl.lit("STD" if na else "HIST_STD"),
        popsrc=pl.lit("D" if na else "I"),
        consol=pl.lit("C"),
    ).write_parquet(path)


def _write_acc(path, rows):
    """acc_std_ann rows as (gvkey, datadate, be_x, ni_x)."""
    pl.DataFrame(
        {
            "gvkey": [r[0] for r in rows],
            "datadate": [r[1] for r in rows],
            "be_x": [r[2] for r in rows],
            "ni_x": [r[3] for r in rows],
        },
        schema={"gvkey": pl.Utf8, "datadate": pl.Date, "be_x": pl.Float64, "ni_x": pl.Float64},
    ).with_columns(
        curcd=pl.lit("USD"),
        source=pl.lit("GLOBAL"),
        at_x=pl.lit(100.0),
        **{c: pl.lit(0.0) for c in ("ebitda_x", "xint", "mib", "mibt", "mibn")},
    ).write_parquet(path)


def test_hxz_load_compustat_row_na_record_preferred_over_global(test_paths):
    # gvkey 001 is in both NA and Global for 2019, with different datadate.
    _write_keys(
        test_paths.raw_tables_dir / "comp_funda.parquet", [("001", date(2019, 12, 31))], na=True
    )
    _write_keys(
        test_paths.raw_tables_dir / "comp_g_funda.parquet", [("001", date(2019, 11, 30))], na=False
    )
    _write_acc(
        test_paths.interim_dir / "acc_std_ann.parquet",
        [("001", date(2019, 12, 31), 10.0, 1.0), ("001", date(2019, 11, 30), 99.0, 1.0)],
    )

    out = hxz_load_compustat_row(test_paths)

    # One record for 2019: the NA one, valued from its own acc_std_ann row.
    assert out.select("gvkey", "datadate", "year", "be_a").rows() == [
        ("001", date(2019, 12, 31), 2019, 10.0)
    ]


def test_hxz_load_compustat_row_roe_a_uses_previous_be(test_paths):
    dates = [date(y, 12, 31) for y in range(2018, 2025)]
    _write_keys(
        test_paths.raw_tables_dir / "comp_funda.parquet", [("002", d) for d in dates], na=True
    )
    _write_keys(test_paths.raw_tables_dir / "comp_g_funda.parquet", [("999", dates[0])], na=False)
    be = [50.0, 100.0, None, 80.0, 40.0, -10.0, 60.0]
    ni = [5.0, 20.0, 30.0, 8.0, 4.0, 1.0, 6.0]
    _write_acc(
        test_paths.interim_dir / "acc_std_ann.parquet",
        [("002", d, b, n) for d, b, n in zip(dates, be, ni, strict=True)],
    )

    out = hxz_load_compustat_row(test_paths).filter(pl.col("gvkey") == "002").sort("datadate")

    got = out["roe_a"].to_list()
    # roe_a = ni_x / previous record's be_x: null for the first record, a missing previous
    # be_x (2021) and a previous be_x <= 0 (2024).
    assert [g is None for g in got] == [True, False, False, True, False, False, True]
    assert got[1] == pytest.approx(20.0 / 50.0)
    assert got[2] == pytest.approx(30.0 / 100.0)
    assert got[4] == pytest.approx(4.0 / 80.0)
    assert got[5] == pytest.approx(1.0 / 40.0)


def test_hxz_load_compustat_row_roe_a_needs_previous_fiscal_year(test_paths):
    # 003: no 2020 record (gap). 004: fiscal year-end moves Dec -> Mar with no FY 2020,
    # then Mar 2021 -> Mar 2022, one fiscal year later.
    keys = [
        ("003", date(2018, 12, 31)),
        ("003", date(2019, 12, 31)),
        ("003", date(2021, 12, 31)),
        ("004", date(2019, 12, 31)),
        ("004", date(2021, 3, 31)),
        ("004", date(2022, 3, 31)),
    ]
    _write_keys(test_paths.raw_tables_dir / "comp_funda.parquet", keys, na=True)
    _write_keys(test_paths.raw_tables_dir / "comp_g_funda.parquet", [("999", keys[0][1])], na=False)
    _write_acc(
        test_paths.interim_dir / "acc_std_ann.parquet", [(g, d, 100.0, 10.0) for g, d in keys]
    )

    out = (
        hxz_load_compustat_row(test_paths)
        .filter(pl.col("gvkey") != "999")
        .sort("gvkey", "datadate")
    )

    # roe_a needs a record for fiscal year y-1 (fiscal years named by the calendar year
    # they end in); a record two fiscal years back doesn't count.
    assert out["roe_a"].to_list() == [
        None,
        pytest.approx(0.1),
        None,
        None,
        None,
        pytest.approx(0.1),
    ]
