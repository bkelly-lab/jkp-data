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

from jkp.data.aux_functions import _dhs_row_chars
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
