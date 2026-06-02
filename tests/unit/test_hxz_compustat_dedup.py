"""
Unit tests for the fiscal-year-change dedup in the HXZ Compustat loaders
(Part B of the HXZ fix — mirrors test_ff_compustat_dedup.py).

A firm that moves its fiscal-year-end reports two `datadate` in one calendar
year. Before the fix, `inv` (= `ia_hxz`) was computed over the raw datadate
sequence, so the `at.shift(1)` lag (gated on `fy_gap == 1`) spanned the
sub-annual stub and nulled `inv` on the surviving row. The loaders now collapse
to one row per `(gvkey, year)` (latest datadate) BEFORE the lag.

Covers:
  - _hxz_compustat_be_inv   (ROW annual inv)
  - hxz_load_funda          (US annual inv)
"""

from __future__ import annotations

from datetime import date

import polars as pl

from jkp.data.aux_functions import _hxz_compustat_be_inv, hxz_load_funda

# Columns _hxz_compustat_be_inv / hxz_load_funda reference, defaulted to None.
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


def _lazy(rows):
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("datadate").cast(pl.Date),
            *[pl.col(c).cast(pl.Float64) for c in _FLOAT_COLS],
        )
        .lazy()
    )


def test_hxz_compustat_be_inv_collapses_and_recovers_inv():
    out = _hxz_compustat_be_inv(_lazy(_fiscal_change_rows("001")), is_global=True).collect()

    # One row per (gvkey, year); the 2019 stub (06-30) dropped for 12-31.
    assert out.height == 2
    r2019 = out.filter(pl.col("year") == 2019)
    assert r2019.height == 1
    assert r2019["datadate"].item() == date(2019, 12, 31)
    # at_lag is the 2018 row (100) -> fy_gap == 1 -> inv = (120-100)/100.
    assert r2019["inv"].item() is not None
    assert abs(r2019["inv"].item() - 0.20) < 1e-9


def test_hxz_load_funda_collapses_and_recovers_inv(tmp_path):
    pl.DataFrame(_fiscal_change_rows("001")).with_columns(
        pl.col("datadate").cast(pl.Date),
        *[pl.col(c).cast(pl.Float64) for c in _FLOAT_COLS],
        indfmt=pl.lit("INDL"),
        datafmt=pl.lit("STD"),
        popsrc=pl.lit("D"),
        consol=pl.lit("C"),
    ).write_parquet(tmp_path / "comp_funda.parquet")

    out = hxz_load_funda(tmp_path)

    keyed = out.with_columns(year=pl.col("datadate").dt.year()).select("gvkey", "year")
    assert keyed.height == keyed.unique().height  # one row per (gvkey, year)
    r2019 = out.filter(pl.col("datadate") == date(2019, 12, 31))
    assert r2019["inv"].item() is not None
    assert abs(r2019["inv"].item() - 0.20) < 1e-9
