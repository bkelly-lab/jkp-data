"""
Unit tests for the fiscal-year-change dedup in the FF Compustat loaders.

A firm that moves its fiscal-year-end reports two `datadate` in one calendar
year. Before the fix, `be/op/inv` were computed over the raw datadate sequence
and the loaders then kept the latest datadate per (gvkey/permno, year) — so the
`inv` lag (`at.shift(1)`, gated on `fy_gap == 1`) spanned the sub-annual stub,
nulling `inv` on the surviving row. The loaders now collapse to one row per
(gvkey, year) at the source (latest datadate) BEFORE computing `inv`, which both
removes the downstream `(id, eom)` fan-out and recovers the annual `inv`.

Covers:
  - _ff_load_world_funda      (ROW per-source collapse)
  - _ff_compute_be_op_inv     (ROW inv recovery)
  - ff_load_compustat_us      (US collapse + inv recovery)
"""

from __future__ import annotations

from datetime import date

import polars as pl

from jkp.data.aux_functions import (
    _ff_compute_be_op_inv,
    _ff_load_world_funda,
    ff_load_compustat_us,
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


def test_ff_load_world_funda_collapses_fiscal_change(tmp_path):
    _write_funda(tmp_path / "g_funda.parquet", _fiscal_change_rows("001", indl=False))

    out = _ff_load_world_funda(tmp_path, "g_funda.parquet", is_global=True).collect()

    # One row per (gvkey, year); the 2019 stub (06-30) is dropped for 12-31.
    assert out.height == 2
    rows_2019 = out.filter(pl.col("year") == 2019)
    assert rows_2019.height == 1
    assert rows_2019["datadate"].item() == date(2019, 12, 31)


def test_ff_compute_be_op_inv_recovers_annual_inv(tmp_path):
    _write_funda(tmp_path / "g_funda.parquet", _fiscal_change_rows("001", indl=False))

    lf = _ff_load_world_funda(tmp_path, "g_funda.parquet", is_global=True)
    out = _ff_compute_be_op_inv(lf, is_global=True).collect()

    inv_2019 = out.filter(pl.col("year") == 2019)["inv"].item()
    # at_lag is the 2018 row (100) -> fy_gap == 1 -> inv = (120-100)/100.
    assert inv_2019 is not None
    assert abs(inv_2019 - 0.20) < 1e-9


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


def test_ff_load_compustat_us_collapses_and_recovers_inv(tmp_path):
    _write_funda(tmp_path / "comp_funda.parquet", _fiscal_change_rows("001", indl=True))
    _write_lnkhist(tmp_path / "crsp_ccmxpf_lnkhist.parquet", "001", 10001)

    out = ff_load_compustat_us(tmp_path, ff5=True)

    # Unique per (permno, year) — no fiscal-change fan-out.
    keyed = out.select("permno", "year")
    assert keyed.height == keyed.unique().height

    inv_2019 = out.filter(pl.col("year") == 2019)["inv"].item()
    assert inv_2019 is not None
    assert abs(inv_2019 - 0.20) < 1e-9
