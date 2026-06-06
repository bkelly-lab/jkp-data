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
  - _ff_load_funda_global     (ROW per-source collapse)
  - _ff_global_be_op_inv      (ROW inv recovery)
  - ff_load_compustat_us      (US collapse + inv recovery + BE ladder)
"""

from __future__ import annotations

from datetime import date

import polars as pl

from jkp.data.aux_functions import (
    _ff_global_be_op_inv,
    _ff_load_funda_global,
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
    "mib",
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


def test_ff_load_funda_global_collapses_fiscal_change(tmp_path):
    _write_funda(tmp_path / "comp_g_funda.parquet", _fiscal_change_rows("001", indl=False))

    out = _ff_load_funda_global(tmp_path).collect()

    # One row per (gvkey, year); the 2019 stub (06-30) is dropped for 12-31.
    assert out.height == 2
    rows_2019 = out.filter(pl.col("year") == 2019)
    assert rows_2019.height == 1
    assert rows_2019["datadate"].item() == date(2019, 12, 31)


def test_ff_global_be_op_inv_recovers_annual_inv(tmp_path):
    _write_funda(tmp_path / "comp_g_funda.parquet", _fiscal_change_rows("001", indl=False))

    lf = _ff_load_funda_global(tmp_path)
    out = _ff_global_be_op_inv(lf).collect()

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


# =============================================================================
# US BE ladder: SHE = coalesce(seq, ceq + pstk, at - lt)
# (French data library / DFF 2000: "the value reported by Moody's or
# Compustat, if it is available. If not, ... the book value of common equity
# plus the par value of preferred stock, or the book value of assets minus
# total liabilities (in that order)")
# =============================================================================


class TestUsBeLadder:
    @staticmethod
    def _be(tmp_path, **over):
        """BE from a single 2019 funda row with the given overrides.

        _funda_row defaults: seq = at/2, pstk = 0, txditc = 0 (and the
        FASB-109 gate zeroes txditc for years >= 1993 anyway), so
        ps = coalesce(pstkrv, pstkl, pstk, 0) = pstk.
        """
        row = _funda_row("001", date(2019, 12, 31), 100.0, indl=True, **over)
        _write_funda(tmp_path / "comp_funda.parquet", [row])
        _write_lnkhist(tmp_path / "crsp_ccmxpf_lnkhist.parquet", "001", 10001)
        out = ff_load_compustat_us(tmp_path, ff5=True)
        assert out.height == 1
        return out

    def test_ceq_pstk_rung(self, tmp_path):
        # she = ceq + pstk = 45; be = she - ps = she - pstk = ceq = 40.
        out = self._be(tmp_path, seq=None, ceq=40.0, pstk=5.0, lt=30.0)
        assert abs(out["be"][0] - 40.0) < 1e-12

    def test_at_lt_rung(self, tmp_path):
        out = self._be(tmp_path, seq=None, ceq=None, pstk=None, lt=30.0)
        assert abs(out["be"][0] - 70.0) < 1e-12  # at(100) - lt(30)

    def test_null_pstk_falls_to_at_lt(self, tmp_path):
        # SAS sum semantics: ceq + null pstk -> null rung -> at - lt, not ceq.
        out = self._be(tmp_path, seq=None, ceq=40.0, pstk=None, lt=30.0)
        assert abs(out["be"][0] - 70.0) < 1e-12

    def test_all_rungs_null_be_null(self, tmp_path):
        out = self._be(tmp_path, seq=None, ceq=None, pstk=None, at=None, lt=None)
        assert out["be"][0] is None

    def test_negative_ladder_be_nulled(self, tmp_path):
        out = self._be(tmp_path, seq=None, ceq=None, pstk=None, lt=170.0)
        assert out["be"][0] is None  # at(100) - lt(170) < 0

    def test_seq_dominates(self, tmp_path):
        out = self._be(tmp_path, seq=50.0, ceq=40.0, lt=30.0)
        assert abs(out["be"][0] - 50.0) < 1e-12

    def test_op_computed_on_ladder_be(self, tmp_path):
        # be = at - lt = 70; op = (revt - cogs) / be = 15 / 70.
        out = self._be(tmp_path, seq=None, ceq=None, pstk=None, lt=30.0, revt=20.0, cogs=5.0)
        assert abs(out["op"][0] - 15.0 / 70.0) < 1e-12


# =============================================================================
# CCM backfill rescue (FF_CCM_BACKDATE_*): June 1963-66 formations may use
# fiscal rows that predate the permno's first link window (Compustat added
# ~200 small firms/yr around 1964-66 and backfilled fy1962-64; CCM stamps
# linkdt at coverage start, so the standard window filter rejects exactly
# the rows French's portfolios include).
# =============================================================================


def _write_lnkhist_rows(path, rows):
    """rows: list of (gvkey, permno, linkdt) — P/LC links, open-ended."""
    pl.DataFrame(
        {
            "gvkey": [r[0] for r in rows],
            "lpermno": [r[1] for r in rows],
            "linkprim": ["P"] * len(rows),
            "linktype": ["LC"] * len(rows),
            "linkdt": [r[2] for r in rows],
            "linkenddt": [None] * len(rows),
        }
    ).with_columns(
        pl.col("linkdt").cast(pl.Date),
        pl.col("linkenddt").cast(pl.Date),
    ).write_parquet(path)


class TestCcmBackdateRescue:
    @staticmethod
    def _run(tmp_path, funda_rows, lnk_rows):
        _write_funda(tmp_path / "comp_funda.parquet", funda_rows)
        _write_lnkhist_rows(tmp_path / "crsp_ccmxpf_lnkhist.parquet", lnk_rows)
        return ff_load_compustat_us(tmp_path, ff5=True)

    def test_backfilled_year_rescued(self, tmp_path):
        # fy1962 -> jun_end 1963-06-30, inside [1963, 1966]; link starts
        # 1965-01-01 (permno's first, within 5y of the formation).
        out = self._run(
            tmp_path,
            [_funda_row("001", date(1962, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1965, 1, 1))],
        )
        assert out.filter((pl.col("permno") == 10001) & (pl.col("year") == 1962)).height == 1

    def test_no_rescue_after_gate(self, tmp_path):
        # fy1970 -> jun_end 1971-06-30 > THROUGH: standard window only.
        out = self._run(
            tmp_path,
            [_funda_row("001", date(1970, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1973, 1, 1))],
        )
        assert out.height == 0

    def test_no_rescue_before_gate(self, tmp_path):
        # fy1960 -> jun_end 1961-06-30 < FROM.
        out = self._run(
            tmp_path,
            [_funda_row("001", date(1960, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1963, 1, 1))],
        )
        assert out.height == 0

    def test_distance_cap(self, tmp_path):
        # Link 1990 claiming fy1962: > MAX_YEARS after the formation.
        out = self._run(
            tmp_path,
            [_funda_row("001", date(1962, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1990, 1, 1))],
        )
        assert out.height == 0

    def test_first_link_guard(self, tmp_path):
        # permno's first link belongs to another gvkey (1955): the 1965 link
        # is not first -> no rescue (recycled-permno protection).
        out = self._run(
            tmp_path,
            [_funda_row("002", date(1962, 12, 31), 100.0, indl=True)],
            [("001", 10001, date(1955, 1, 1)), ("002", 10001, date(1965, 1, 1))],
        )
        assert out.height == 0

    def test_in_window_beats_backdated(self, tmp_path):
        # Two gvkeys with fy1962 rows for one permno: gvkey 001 in-window
        # (first link, 1955), gvkey 002 backdated. Dedup keeps the in-window
        # match. (002 also fails the first-link guard; the sort-priority is
        # belt-and-braces.)
        out = self._run(
            tmp_path,
            [
                _funda_row("001", date(1962, 12, 31), 100.0, indl=True),
                _funda_row("002", date(1962, 12, 31), 200.0, indl=True),
            ],
            [("001", 10001, date(1955, 1, 1)), ("002", 10001, date(1965, 1, 1))],
        )
        rows = out.filter((pl.col("permno") == 10001) & (pl.col("year") == 1962))
        assert rows.height == 1
        assert rows["gvkey"][0] == "001"
