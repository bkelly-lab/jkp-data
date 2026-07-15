"""
Unit tests for DHS ROW characteristic helper in aux_functions.py.

Covers _dhs_row_chars: NS = log(1 + chcsho_12m), IR = -eqnpo_60m (both sampled
at June rows), the YEAR relabel split (IR emitted at YEAR=t, NS at YEAR=t-1),
and the lagBE join (be_x of the fiscal year ending in calendar YEAR-1, latest
statement per gvkey-fiscal-year).
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import _dhs_ibes_announcements, _dhs_row_chars
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
    """raw + interim dirs with toy comp_g_security / ibes_actu_epsint / prelim."""
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    pl.DataFrame(
        {"ibtic": ["T1", "T2", "T3"], "gvkey": ["001000", "002000", "003000"]}
    ).write_parquet(raw / "comp_g_security.parquet")
    pl.DataFrame(
        {
            "ticker": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "usfirm": [0, 0, 0, 0, 1, 0],
            "measure": ["EPS"] * 6,
            # T1: SAN row must be admitted; T2: QTR+SAN same pends dedupe to the
            # earliest anndats; T3: usfirm==1 and ANN both excluded.
            "pdicity": ["QTR", "SAN", "QTR", "SAN", "QTR", "ANN"],
            "anndats": [
                date(2010, 5, 3),
                date(2010, 8, 16),
                date(2010, 7, 30),
                date(2010, 7, 20),
                date(2010, 5, 1),
                date(2010, 5, 1),
            ],
            "pends": [
                date(2010, 3, 31),
                date(2010, 6, 30),
                date(2010, 6, 30),
                date(2010, 6, 30),
                date(2010, 3, 31),
                date(2010, 3, 31),
            ],
        }
    ).write_parquet(raw / "ibes_actu_epsint.parquet")
    pl.DataFrame(
        {
            "id": [310001000, 320002000, 330003000],
            "gvkey": ["001000", "002000", "003000"],
            "excntry": ["GBR", "GBR", "GBR"],
            "eom": [date(2010, 6, 30)] * 3,
            "common": [1, 1, 1],
            "primary_sec": [1, 1, 1],
            "obs_main": [1, 1, 1],
            "exch_main": [1, 1, 1],
            "me": [100.0, 200.0, 300.0],
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
