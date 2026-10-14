"""
Unit tests for DHS ROW helpers in aux_functions.py.

Covers _dhs_row_chars: NS = log(1 + chcsho_12m), IR = -eqnpo_60m (both sampled
at June rows), the YEAR relabel split (IR emitted at YEAR=t, NS at YEAR=t-1),
and the lagBE join (be_x of the fiscal year ending in calendar YEAR-1, latest
statement per gvkey-fiscal-year). Also covers _dhs_ibes_announcements (ticker
bridging via both security masters, SEDOL > CUSIP > ibtic link precedence,
measure-agnostic filter, periodicity gate flag) and the periodicity-aware
staleness gate in _dhs_align_abr.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import (
    _dhs_align_abr,
    _dhs_ibes_announcements,
    _dhs_ibes_gvkey_link,
    _dhs_non_financial,
    _dhs_row_chars,
)
from jkp.data.config import US_EXCNTRY


@pytest.fixture
def dhs_interim_dir(tmp_path):
    """Interim dir with a toy world_data_prelim + acc_std_ann for one ROW id."""
    months = pl.date_range(date(2000, 1, 31), date(2002, 12, 31), "1mo", eager=True)
    prelim = pl.DataFrame(
        {
            "id": [10] * len(months),
            "gvkey": ["001000"] * len(months),
            "excntry": ["JPN"] * len(months),
            "eom": months,
            "chcsho_12m": [0.1] * len(months),
            "eqnpo_60m": [-0.25] * len(months),
        }
    )
    # A US id that must be excluded, plus a null-char ROW row.
    prelim = pl.concat(
        [
            prelim,
            pl.DataFrame(
                {
                    "id": [20, 30],
                    "gvkey": ["002000", "003000"],
                    "excntry": [US_EXCNTRY, "JPN"],
                    "eom": [date(2001, 6, 30), date(2001, 6, 30)],
                    "chcsho_12m": [0.1, None],
                    "eqnpo_60m": [-0.25, None],
                }
            ),
        ]
    )
    prelim.write_parquet(tmp_path / "world_data_prelim.parquet")
    acc = pl.DataFrame(
        {
            "gvkey": ["001000", "001000", "001000"],
            "datadate": [date(2000, 3, 31), date(2000, 12, 31), date(2001, 12, 31)],
            "be_x": [5.0, 7.0, 9.0],
        }
    )
    acc.write_parquet(tmp_path / "acc_std_ann.parquet")
    return tmp_path


class TestDhsNonFinancial:
    """NAICS 52-53 when NAICS is present, else SIC 6000-6999; missing both is kept."""

    def test_naics_first_then_sic_fallback(self):
        df = pl.DataFrame(
            {
                "naics": ["522110", "531120", "334111", "334111", None, None, None],
                "siccd": [3711, 3711, 6020, None, 6020, 3711, None],
            }
        )
        kept = df.select(_dhs_non_financial().alias("keep"))["keep"].to_list()
        assert kept == [False, False, True, True, False, True, True]


class TestDhsRowChars:
    def test_ir_is_minus_eqnpo_60m_at_june(self, dhs_interim_dir):
        out = _dhs_row_chars(dhs_interim_dir, 2000, 2002)
        ir = out.filter(
            (pl.col("id") == 10) & (pl.col("YEAR") == 2001) & pl.col("IR").is_not_null()
        )
        assert ir.height == 1
        assert ir["datadate"][0] == date(2001, 6, 30)
        assert math.isclose(ir["IR"][0], 0.25)

    def test_ns_is_log1p_chcsho_relabelled_to_prior_year(self, dhs_interim_dir):
        out = _dhs_row_chars(dhs_interim_dir, 2000, 2002)
        # NS from June(2001) is emitted at YEAR=2000 (fin_factor filters YEAR+1==cyear)
        ns = out.filter(
            (pl.col("id") == 10) & (pl.col("YEAR") == 2000) & pl.col("NS").is_not_null()
        )
        assert ns.height == 1
        assert ns["datadate"][0] == date(2001, 6, 30)
        assert math.isclose(ns["NS"][0], math.log(1.1))

    def test_lag_be_joins_prior_calendar_year_latest_statement(self, dhs_interim_dir):
        out = _dhs_row_chars(dhs_interim_dir, 2000, 2002)
        # YEAR=2001 rows join be_x of fiscal years ending in 2000; the Dec-2000
        # statement (7.0) wins over Mar-2000 (5.0) as the latest datadate.
        ir = out.filter(
            (pl.col("id") == 10) & (pl.col("YEAR") == 2001) & pl.col("IR").is_not_null()
        )
        assert ir["lagBE"][0] == 7.0
        ir_2002 = out.filter(
            (pl.col("id") == 10) & (pl.col("YEAR") == 2002) & pl.col("IR").is_not_null()
        )
        assert ir_2002["lagBE"][0] == 9.0

    def test_us_rows_excluded_and_null_chars_stay_null(self, dhs_interim_dir):
        out = _dhs_row_chars(dhs_interim_dir, 2000, 2002)
        assert out.filter(pl.col("id") == 20).height == 0
        null_row = out.filter((pl.col("id") == 30) & (pl.col("YEAR") == 2001))
        assert null_row["IR"].null_count() == null_row.height


def _security(
    gvkey: list[str],
    ibtic: list[str | None] | None = None,
    sedol: list[str | None] | None = None,
    cusip: list[str | None] | None = None,
) -> pl.DataFrame:
    """Toy Compustat security master; omitted identifier columns are all null."""
    n = len(gvkey)
    return pl.DataFrame(
        {
            "gvkey": gvkey,
            "ibtic": ibtic or [None] * n,
            "sedol": sedol or [None] * n,
            "cusip": cusip or [None] * n,
        },
        schema={"gvkey": pl.Utf8, "ibtic": pl.Utf8, "sedol": pl.Utf8, "cusip": pl.Utf8},
    )


@pytest.fixture
def ibes_dirs(tmp_path):
    """raw + interim dirs with toy comp_g_security / comp_security /
    ibes_actu_epsint / prelim."""
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    # no SEDOL/CUSIP matches here: these tickers bridge through the ibtic fallback
    _security(
        ibtic=["T1", "T2", "T3", "T5"], gvkey=["001000", "002000", "003000", "005000"]
    ).write_parquet(raw / "comp_g_security.parquet")
    # T4 is bridged ONLY via the North America security master
    _security(ibtic=["T4"], gvkey=["004000"]).write_parquet(raw / "comp_security.parquet")
    pl.DataFrame(
        {
            "ticker": ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T5"],
            "usfirm": [0, 0, 0, 0, 1, 0, 0, 0],
            # T5: non-EPS measure (FFO) must still supply an announcement
            "measure": ["EPS"] * 6 + ["EPS", "FFO"],
            # T1: SAN row must be admitted; T2: QTR+SAN same pends dedupe to the
            # earliest anndats; T3: usfirm==1 excluded, ANN admitted.
            "pdicity": ["QTR", "SAN", "QTR", "SAN", "QTR", "ANN", "QTR", "SAN"],
            "anndats": [
                date(2010, 5, 3),
                date(2010, 8, 16),
                date(2010, 7, 30),
                date(2010, 7, 20),
                date(2010, 5, 1),
                date(2010, 5, 1),
                date(2010, 4, 28),
                date(2010, 9, 15),
            ],
            "pends": [
                date(2010, 3, 31),
                date(2010, 6, 30),
                date(2010, 6, 30),
                date(2010, 6, 30),
                date(2010, 3, 31),
                date(2010, 3, 31),
                date(2010, 3, 31),
                date(2010, 6, 30),
            ],
            "cusip": pl.Series([None] * 8, dtype=pl.Utf8),
        }
    ).write_parquet(raw / "ibes_actu_epsint.parquet")
    pl.DataFrame(
        {
            "id": [310001000, 320002000, 330003000, 340004000, 350005000],
            "gvkey": ["001000", "002000", "003000", "004000", "005000"],
            "excntry": ["GBR"] * 5,
            "eom": [date(2010, 6, 30)] * 5,
            "common": [1] * 5,
            "primary_sec": [1] * 5,
            "obs_main": [1] * 5,
            "exch_main": [1] * 5,
            "me": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    ).write_parquet(interim / "world_data_prelim.parquet")
    return raw, interim


class TestDhsIbesAnnouncements:
    def test_san_ann_admitted_qtr_san_deduped_usfirm_excluded(self, ibes_dirs):
        raw, interim = ibes_dirs
        out = _dhs_ibes_announcements(raw, interim, 2000, 2020)
        # T1: both its QTR and SAN announcements survive (distinct period-ends)
        t1 = out.filter(pl.col("id") == 310001000).sort("datadate")
        assert t1.height == 2
        assert t1["rdq"].to_list() == [date(2010, 5, 3), date(2010, 8, 16)]
        # T2: QTR and SAN share pends -> one row with the EARLIEST anndats (the SAN one)
        t2 = out.filter(pl.col("id") == 320002000)
        assert t2.height == 1
        assert t2["rdq"][0] == date(2010, 7, 20)
        # T3: the usfirm==1 QTR row is excluded; its usfirm==0 ANN row is admitted
        t3 = out.filter(pl.col("id") == 330003000)
        assert t3["rdq"].to_list() == [date(2010, 5, 1)]

    def test_comp_security_only_ticker_bridged(self, ibes_dirs):
        raw, interim = ibes_dirs
        out = _dhs_ibes_announcements(raw, interim, 2000, 2020)
        # T4's ibtic link lives only in comp_security (NA master)
        t4 = out.filter(pl.col("id") == 340004000)
        assert t4.height == 1
        assert t4["rdq"][0] == date(2010, 4, 28)

    def test_non_eps_measure_supplies_announcement(self, ibes_dirs):
        raw, interim = ibes_dirs
        out = _dhs_ibes_announcements(raw, interim, 2000, 2020)
        # T5 reports only FFO; anndats is measure-agnostic
        t5 = out.filter(pl.col("id") == 350005000)
        assert t5.height == 1
        assert t5["rdq"][0] == date(2010, 9, 15)

    def test_gate_reflects_periodicity_of_kept_row(self, ibes_dirs):
        raw, interim = ibes_dirs
        out = _dhs_ibes_announcements(raw, interim, 2000, 2020)
        t1 = out.filter(pl.col("id") == 310001000).sort("datadate")
        assert t1["gate"].to_list() == [6, 9]  # QTR then SAN period-end
        # T2's kept row is the earlier-anndats SAN one; T3 ANN; T4 QTR; T5 SAN
        assert out.filter(pl.col("id") == 320002000)["gate"].to_list() == [9]
        assert out.filter(pl.col("id") == 330003000)["gate"].to_list() == [15]
        assert out.filter(pl.col("id") == 340004000)["gate"].to_list() == [6]
        assert out.filter(pl.col("id") == 350005000)["gate"].to_list() == [9]

    def test_same_day_tie_keeps_shortest_gate(self, ibes_dirs):
        raw, interim = ibes_dirs
        # T4 also files the same period-end, same day, as ANN: the QTR row must win
        ib = pl.read_parquet(raw / "ibes_actu_epsint.parquet")
        extra = ib.filter(pl.col("ticker") == "T4").with_columns(pdicity=pl.lit("ANN"))
        pl.concat([ib, extra]).write_parquet(raw / "ibes_actu_epsint.parquet")
        out = _dhs_ibes_announcements(raw, interim, 2000, 2020)
        assert out.filter(pl.col("id") == 340004000)["gate"].to_list() == [6]


class TestDhsIbesGvkeyLink:
    """Identifier precedence SEDOL > CUSIP > ibtic; majority vote within one identifier."""

    @pytest.fixture
    def link(self, tmp_path):
        _security(
            gvkey=["001000", "002000", "003000", "004000", "005000"],
            ibtic=["X1", "X2", None, None, None],
            sedol=["9999999", "1234567", "7654321", "1111111", "2222222"],
            cusip=[None, None, None, "87654321X", "GB555555X"],
        ).write_parquet(tmp_path / "comp_g_security.parquet")
        _security(
            gvkey=["006000", "009000", "008000"],
            sedol=[None, "3333333", "4444444"],
            cusip=["12345678Z", None, None],
        ).write_parquet(tmp_path / "comp_security.parquet")
        ibes = pl.LazyFrame(
            {
                "ticker": ["X1", "X2", "X3", "X3", "X4", "X4", "X4", "X5", "X5", "X6"],
                "cusip": [
                    "GB123456",  # X1: SEDOL of 002000 beats ibtic -> 001000
                    "12345678",  # X2: CUSIP of 006000 beats ibtic -> 002000
                    "GB765432",  # X3: SEDOL of 003000 ...
                    "87654321",  # ... beats CUSIP of 004000
                    "GB111111",  # X4: two rows -> 004000
                    "GB111111",
                    "GB222222",  # one row -> 005000
                    "GB333333",  # X5: tie 009000 vs 008000 -> lowest gvkey
                    "GB444444",
                    "GB555555",  # X6: SEDOL form never matched as a CUSIP
                ],
            }
        )
        return dict(_dhs_ibes_gvkey_link(ibes, tmp_path).collect().iter_rows())

    def test_sedol_beats_ibtic(self, link):
        assert link["X1"] == "002000"

    def test_cusip_beats_ibtic(self, link):
        assert link["X2"] == "006000"

    def test_sedol_beats_cusip(self, link):
        assert link["X3"] == "003000"

    def test_majority_vote_then_lowest_gvkey(self, link):
        assert link["X4"] == "004000"
        assert link["X5"] == "008000"

    def test_sedol_form_not_tried_as_cusip(self, link):
        assert "X6" not in link


def _align_abr_frames(stock_id: int, gate: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Minimal (panel_rows, abr) pair for _dhs_align_abr: one non-financial id with
    monthly panel rows Jun-2010..Dec-2011 and a single Jun-2010 period-end announced
    2010-07-05, so the period-end-to-CRSP-month gap runs 0..18 across the panel."""
    eoms = pl.date_range(date(2010, 6, 1), date(2011, 12, 1), "1mo", eager=True).dt.month_end()
    panel_rows = pl.DataFrame(
        {
            "excntry": ["GBR"] * len(eoms),
            "eom": eoms,
            "id": [stock_id] * len(eoms),
            "primaryexch": ["L"] * len(eoms),
            "lagCRSPSIZE": [100.0] * len(eoms),
            "siccd": [3711] * len(eoms),
            "naics": [None] * len(eoms),
            "size_grp": ["large"] * len(eoms),
            "year": eoms.dt.year(),
            "month": eoms.dt.month(),
            "CRSPDATE": eoms,
            "ret": [0.01] * len(eoms),
        }
    )
    abr = pl.DataFrame(
        {
            "id": [stock_id],
            "DATADATE": [date(2010, 6, 30)],
            "RDQ": [date(2010, 7, 5)],
            "Abr": [0.05],
            "eom": [date(2010, 7, 31)],
            "gate": [gate],
        }
    )
    return panel_rows, abr


class TestDhsAlignAbr:
    def test_san_row_survives_gap_7_to_9_qtr_does_not(self):
        # QTR (gate 6): forward-filled rows drop once the gap exceeds 6 months
        panel_q, abr_q = _align_abr_frames(1, gate=6)
        out_q = _dhs_align_abr(panel_q, abr_q)
        assert out_q["eom"].max() == date(2010, 12, 31)  # gap 6 kept, gap 7 dropped
        # SAN (gate 9): the gate widens to 9 months
        panel_s, abr_s = _align_abr_frames(2, gate=9)
        out_s = _dhs_align_abr(panel_s, abr_s)
        assert out_s["eom"].max() == date(2011, 3, 31)  # gap 9 kept, gap 10 dropped
        gaps_7_to_9 = out_s.filter(pl.col("eom") > date(2010, 12, 31))
        assert gaps_7_to_9.height == 3
        assert gaps_7_to_9["lagAbr"].null_count() == 0

    def test_ann_row_survives_gap_to_15(self):
        # ANN (gate 15): annual-only reporters carry Abr to the next annual announcement
        panel_a, abr_a = _align_abr_frames(3, gate=15)
        out_a = _dhs_align_abr(panel_a, abr_a)
        assert out_a["eom"].max() == date(2011, 9, 30)  # gap 15 kept, gap 16 dropped

    def test_missing_gate_column_defaults_to_six_month_gate(self):
        # the US path passes abr without a gate column; behavior must match gate=6
        panel_rows, abr = _align_abr_frames(1, gate=6)
        out_without = _dhs_align_abr(panel_rows, abr.drop("gate"))
        out_with = _dhs_align_abr(panel_rows, abr)
        assert out_without.equals(out_with)
        assert out_without["eom"].max() == date(2010, 12, 31)
