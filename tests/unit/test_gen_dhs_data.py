"""
Unit tests for DHS ROW helpers in aux_functions.py.

Covers _dhs_row_chars: NS = log(1 + chcsho_12m), IR = -eqnpo_60m (both sampled
at June rows), the YEAR relabel split (IR emitted at YEAR=t, NS at YEAR=t-1),
and the lagBE join (be_x of the fiscal year ending in calendar YEAR-1, latest
statement per gvkey-fiscal-year). Also covers _dhs_ibes_announcements (ticker
bridging via both security masters, measure-agnostic filter, san periodicity
flag) and the periodicity-aware staleness gate in _dhs_align_abr.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import _dhs_align_abr, _dhs_ibes_announcements, _dhs_row_chars
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


@pytest.fixture
def ibes_dirs(tmp_path):
    """raw + interim dirs with toy comp_g_security / comp_security /
    ibes_actu_epsint / prelim."""
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    pl.DataFrame(
        {"ibtic": ["T1", "T2", "T3", "T5"], "gvkey": ["001000", "002000", "003000", "005000"]}
    ).write_parquet(raw / "comp_g_security.parquet")
    # T4 is bridged ONLY via the North America security master
    pl.DataFrame({"ibtic": ["T4"], "gvkey": ["004000"]}).write_parquet(
        raw / "comp_security.parquet"
    )
    pl.DataFrame(
        {
            "ticker": ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T5"],
            "usfirm": [0, 0, 0, 0, 1, 0, 0, 0],
            # T5: non-EPS measure (FFO) must still supply an announcement
            "measure": ["EPS"] * 6 + ["EPS", "FFO"],
            # T1: SAN row must be admitted; T2: QTR+SAN same pends dedupe to the
            # earliest anndats; T3: usfirm==1 and ANN both excluded.
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
    def test_san_admitted_qtr_san_deduped_ann_usfirm_excluded(self, ibes_dirs):
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
        # T3: usfirm==1 (QTR) and ANN rows are both excluded
        assert out.filter(pl.col("id") == 330003000).height == 0

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

    def test_san_column_reflects_periodicity_of_kept_row(self, ibes_dirs):
        raw, interim = ibes_dirs
        out = _dhs_ibes_announcements(raw, interim, 2000, 2020)
        assert out.schema["san"] == pl.Boolean
        t1 = out.filter(pl.col("id") == 310001000).sort("datadate")
        assert t1["san"].to_list() == [False, True]  # QTR then SAN period-end
        # T2's kept row is the earlier-anndats SAN one; T4 QTR; T5 SAN
        assert out.filter(pl.col("id") == 320002000)["san"].to_list() == [True]
        assert out.filter(pl.col("id") == 340004000)["san"].to_list() == [False]
        assert out.filter(pl.col("id") == 350005000)["san"].to_list() == [True]


def _align_abr_frames(stock_id: int, san: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Minimal (panel_rows, abr) pair for _dhs_align_abr: one non-financial id with
    monthly panel rows Jun-2010..May-2011 and a single Jun-2010 period-end announced
    2010-07-05, so the period-end-to-CRSP-month gap runs 1..11 across the panel."""
    eoms = pl.date_range(date(2010, 6, 1), date(2011, 5, 1), "1mo", eager=True).dt.month_end()
    panel_rows = pl.DataFrame(
        {
            "excntry": ["GBR"] * len(eoms),
            "eom": eoms,
            "id": [stock_id] * len(eoms),
            "primaryexch": ["L"] * len(eoms),
            "lagCRSPSIZE": [100.0] * len(eoms),
            "siccd": [3711] * len(eoms),
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
            "san": [san],
        }
    )
    return panel_rows, abr


class TestDhsAlignAbr:
    def test_san_row_survives_gap_7_to_9_qtr_does_not(self):
        # QTR (san=False): forward-filled rows drop once the gap exceeds 6 months
        panel_q, abr_q = _align_abr_frames(1, san=False)
        out_q = _dhs_align_abr(panel_q, abr_q)
        assert out_q["eom"].max() == date(2010, 12, 31)  # gap 6 kept, gap 7 dropped
        # SAN (san=True): the gate widens to 9 months
        panel_s, abr_s = _align_abr_frames(2, san=True)
        out_s = _dhs_align_abr(panel_s, abr_s)
        assert out_s["eom"].max() == date(2011, 3, 31)  # gap 9 kept, gap 10 dropped
        gaps_7_to_9 = out_s.filter(pl.col("eom") > date(2010, 12, 31))
        assert gaps_7_to_9.height == 3
        assert gaps_7_to_9["lagAbr"].null_count() == 0

    def test_missing_san_column_defaults_to_six_month_gate(self):
        # the US path passes abr without a san column; behavior must match san=False
        panel_rows, abr = _align_abr_frames(1, san=False)
        out_without = _dhs_align_abr(panel_rows, abr.drop("san"))
        out_with = _dhs_align_abr(panel_rows, abr)
        assert out_without.equals(out_with)
        assert out_without["eom"].max() == date(2010, 12, 31)
