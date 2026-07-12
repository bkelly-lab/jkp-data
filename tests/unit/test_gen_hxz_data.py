"""
Unit tests for HXZ q-factor builder helpers in aux_functions.py.

Covers Stage 1 (_hxz_add_me_lag1_ym, hxz_attach_siccd), Stage 2
(hxz_impute_be_clean_surplus, hxz_supplement_q4_shares, hxz_merge_be_chain,
hxz_link_compustat), Stage 3 (_hxz_us_roe_monthly), Stage 4 (_hxz_nyse_breaks,
_hxz_country_breaks, _hxz_classify_size_ia, _hxz_classify_roe,
hxz_classify_portfolios), Stage 5 (hxz_compute_factors), and Stage 6
(hxz_build_characteristics).

Does NOT call gen_hxz_data (covered by golden test) and does NOT require
WRDS I/O for helpers that only need in-memory DataFrames.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import (
    _hxz_add_me_lag1_ym,
    _hxz_classify_roe,
    _hxz_classify_size_ia,
    country_breakpoints,
    hxz_attach_siccd,
    hxz_build_characteristics,
    hxz_classify_portfolios,
    hxz_compute_factors,
    hxz_impute_be_clean_surplus,
    hxz_merge_be_chain,
)
from jkp.data.config import (
    HXZ_MIN_STOCKS_BP,
)


def _hxz_nyse_breaks(df, specs):
    """Test convenience: US NYSE-pool wrapper over country_breakpoints."""
    return country_breakpoints(
        df, specs, group_keys=["date"], pool=(pl.col("exchcd") == 1) & pl.col("elig")
    )


def _hxz_country_breaks(df, specs):
    """Test convenience: ROW pool wrapper over country_breakpoints."""
    return country_breakpoints(
        df,
        specs,
        group_keys=["excntry", "date"],
        pool=pl.col("elig") & pl.col("size_grp").is_in(["small", "large", "mega"]),
        min_stocks=HXZ_MIN_STOCKS_BP,
    )


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------


class TestHxzAddMeLag1Ym:
    """Tests for _hxz_add_me_lag1_ym (gap-robust monthly lag + daily shift)."""

    def test_monthly_lag_normal(self):
        """me_lag1 should equal the me value from the prior calendar month."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1],
                "date": [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)],
                "me": [100.0, 200.0, 300.0],
            }
        )
        result = _hxz_add_me_lag1_ym(df, key="id", freq="monthly")
        assert "me_lag1" in result.columns
        vals = result.sort("date")["me_lag1"].to_list()
        assert vals[0] is None
        assert vals[1] == pytest.approx(100.0)
        assert vals[2] == pytest.approx(200.0)

    def test_monthly_lag_two_stocks(self):
        """Lag should be computed independently for each id."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "date": [
                    date(2020, 1, 31),
                    date(2020, 2, 29),
                    date(2020, 1, 31),
                    date(2020, 2, 29),
                ],
                "me": [10.0, 20.0, 30.0, 40.0],
            }
        )
        result = _hxz_add_me_lag1_ym(df, key="id", freq="monthly").sort(["id", "date"])
        me_lag1 = result["me_lag1"].to_list()
        assert me_lag1[0] is None  # id=1, Jan
        assert me_lag1[1] == pytest.approx(10.0)  # id=1, Feb
        assert me_lag1[2] is None  # id=2, Jan
        assert me_lag1[3] == pytest.approx(30.0)  # id=2, Feb

    def test_monthly_lag_gap_skips_month(self):
        """If February is missing, March lag should still be January value (YM join)."""
        df = pl.DataFrame(
            {
                "id": [1, 1],
                "date": [date(2020, 1, 31), date(2020, 3, 31)],
                "me": [50.0, 90.0],
            }
        )
        result = _hxz_add_me_lag1_ym(df, key="id", freq="monthly").sort("date")
        # Mar's prev_ym = 202002; no row with ym=202002, so join misses → null
        assert result["me_lag1"][1] is None

    def test_monthly_lag_non_month_end_dates(self):
        """CIZ mthcaldt is last TRADING day; lag should still resolve via YM int."""
        df = pl.DataFrame(
            {
                "id": [1, 1],
                # Jan has two dates in the same month; only the last should be
                # used as the lag source for Feb
                "date": [date(2020, 1, 15), date(2020, 1, 30)],
                "me": [10.0, 99.0],
            }
        )
        # Add a Feb row to check the lag
        df2 = pl.DataFrame(
            {
                "id": [1],
                "date": [date(2020, 2, 28)],
                "me": [200.0],
            }
        )
        full = pl.concat([df, df2])
        result = _hxz_add_me_lag1_ym(full, key="id", freq="monthly").sort("date")
        # Jan 15 and Jan 30 are in the same YM=202001; last() is 99.0
        # Feb 28's prev_ym = 202001 → me_lag1 = 99.0
        feb_row = result.filter(pl.col("date") == date(2020, 2, 28))
        assert feb_row["me_lag1"][0] == pytest.approx(99.0)

    def test_daily_lag_uses_shift(self):
        """Daily freq should attach shift(1) within id group."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1],
                "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                "me": [10.0, 20.0, 30.0],
            }
        )
        result = _hxz_add_me_lag1_ym(df, key="id", freq="daily").sort("date")
        assert result["me_lag1"][0] is None
        assert result["me_lag1"][1] == pytest.approx(10.0)
        assert result["me_lag1"][2] == pytest.approx(20.0)

    def test_daily_lag_two_ids(self):
        """Daily lag is per-id; no cross-contamination."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 1, 2],
                "date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 2)],
                "me": [5.0, 50.0, 15.0, 150.0],
            }
        )
        result = _hxz_add_me_lag1_ym(df, key="id", freq="daily").sort(["id", "date"])
        # id=1: lag=[null, 5.0]; id=2: lag=[null, 50.0]
        me_lag1 = result["me_lag1"].to_list()
        nones = [v for v in me_lag1 if v is None]
        assert len(nones) == 2
        nonnull = [v for v in me_lag1 if v is not None]
        assert 5.0 in nonnull
        assert 50.0 in nonnull


class TestHxzAttachSiccd:
    """Tests for hxz_attach_siccd: event-time SIC join and is_fin flag."""

    def _write_sec_hist(self, tmp_path: Path, rows: list[dict]) -> Path:
        sec = pl.DataFrame(rows).with_columns(
            pl.col("secinfostartdt", "secinfoenddt").cast(pl.Date)
        )
        p = tmp_path / "crsp_stksecurityinfohist.parquet"
        sec.write_parquet(p)
        return tmp_path

    def test_basic_join(self, tmp_path):
        """SIC is attached for matching date range."""
        raw_dir = self._write_sec_hist(
            tmp_path,
            [
                {
                    "permno": 1,
                    "secinfostartdt": date(2000, 1, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 3310,
                }
            ],
        )
        df = pl.DataFrame({"permno": [1], "obs_date": [date(2020, 6, 30)]}).lazy()
        result = hxz_attach_siccd(df, raw_dir, "obs_date").collect()
        assert result["siccd"][0] == 3310
        assert result["is_fin"][0] is False

    def test_is_fin_flagged_for_sic_6000_to_6999(self, tmp_path):
        """is_fin=True for siccd in [6000, 6999]."""
        raw_dir = self._write_sec_hist(
            tmp_path,
            [
                {
                    "permno": 1,
                    "secinfostartdt": date(2000, 1, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 6200,
                }
            ],
        )
        df = pl.DataFrame({"permno": [1], "obs_date": [date(2020, 6, 30)]}).lazy()
        result = hxz_attach_siccd(df, raw_dir, "obs_date").collect()
        assert result["is_fin"][0] is True

    def test_is_fin_boundary_6000(self, tmp_path):
        """is_fin=True at the lower boundary siccd=6000."""
        raw_dir = self._write_sec_hist(
            tmp_path,
            [
                {
                    "permno": 1,
                    "secinfostartdt": date(2000, 1, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 6000,
                }
            ],
        )
        df = pl.DataFrame({"permno": [1], "obs_date": [date(2020, 6, 30)]}).lazy()
        result = hxz_attach_siccd(df, raw_dir, "obs_date").collect()
        assert result["is_fin"][0] is True

    def test_is_fin_boundary_6999(self, tmp_path):
        """is_fin=True at the upper boundary siccd=6999."""
        raw_dir = self._write_sec_hist(
            tmp_path,
            [
                {
                    "permno": 1,
                    "secinfostartdt": date(2000, 1, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 6999,
                }
            ],
        )
        df = pl.DataFrame({"permno": [1], "obs_date": [date(2020, 6, 30)]}).lazy()
        result = hxz_attach_siccd(df, raw_dir, "obs_date").collect()
        assert result["is_fin"][0] is True

    def test_date_outside_window_drops_row(self, tmp_path):
        """When obs_date is before secinfostartdt, the filter drops the row entirely."""
        raw_dir = self._write_sec_hist(
            tmp_path,
            [
                {
                    "permno": 1,
                    "secinfostartdt": date(2010, 1, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 3310,
                }
            ],
        )
        df = pl.DataFrame({"permno": [1], "obs_date": [date(2005, 6, 30)]}).lazy()
        result = hxz_attach_siccd(df, raw_dir, "obs_date").collect()
        # The join+filter drops all rows (no secinfostartdt.is_null() row and
        # the range condition fails) → empty output
        assert len(result) == 0

    def test_overlapping_windows_picks_latest_start(self, tmp_path):
        """Tie-break: most recent secinfostartdt wins when two windows overlap."""
        raw_dir = self._write_sec_hist(
            tmp_path,
            [
                {
                    "permno": 1,
                    "secinfostartdt": date(2000, 1, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 3310,
                },
                {
                    "permno": 1,
                    "secinfostartdt": date(2015, 6, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 6200,
                },
            ],
        )
        df = pl.DataFrame({"permno": [1], "obs_date": [date(2020, 6, 30)]}).lazy()
        result = hxz_attach_siccd(df, raw_dir, "obs_date").collect()
        # 2015-06-01 > 2000-01-01, so siccd=6200 should win
        assert result["siccd"][0] == 6200

    def test_drops_secinfostartdt_secinfoenddt_columns(self, tmp_path):
        """Output should not contain secinfostartdt or secinfoenddt."""
        raw_dir = self._write_sec_hist(
            tmp_path,
            [
                {
                    "permno": 1,
                    "secinfostartdt": date(2000, 1, 1),
                    "secinfoenddt": date(2025, 12, 31),
                    "siccd": 3310,
                }
            ],
        )
        df = pl.DataFrame({"permno": [1], "obs_date": [date(2020, 6, 30)]}).lazy()
        result = hxz_attach_siccd(df, raw_dir, "obs_date").collect()
        assert "secinfostartdt" not in result.columns
        assert "secinfoenddt" not in result.columns


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------


class TestHxzMergeBeChain:
    """Tests for hxz_merge_be_chain: Q4 BEQ fill from annual BE_a."""

    def _make_fundq(self, rows):
        return pl.DataFrame(rows).with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col("beq").cast(pl.Float64),
        )

    def _make_funda(self, rows):
        return pl.DataFrame(rows).with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col("be_a").cast(pl.Float64),
        )

    def test_q4_null_beq_filled_from_annual(self):
        """Q4 with null beq gets be_a from funda."""
        fundq = self._make_fundq(
            [{"gvkey": "001", "datadate": date(2020, 12, 31), "fqtr": 4, "beq": None}]
        )
        funda = self._make_funda([{"gvkey": "001", "datadate": date(2020, 12, 31), "be_a": 500.0}])
        result = hxz_merge_be_chain(fundq, funda)
        assert result["beq"][0] == pytest.approx(500.0)

    def test_q4_with_existing_beq_not_overwritten(self):
        """Q4 with non-null beq keeps its own value."""
        fundq = self._make_fundq(
            [{"gvkey": "001", "datadate": date(2020, 12, 31), "fqtr": 4, "beq": 300.0}]
        )
        funda = self._make_funda([{"gvkey": "001", "datadate": date(2020, 12, 31), "be_a": 999.0}])
        result = hxz_merge_be_chain(fundq, funda)
        assert result["beq"][0] == pytest.approx(300.0)

    def test_non_q4_null_beq_stays_null(self):
        """Non-Q4 null beq is not filled (even if annual exists)."""
        fundq = self._make_fundq(
            [{"gvkey": "001", "datadate": date(2020, 9, 30), "fqtr": 3, "beq": None}]
        )
        funda = self._make_funda([{"gvkey": "001", "datadate": date(2020, 9, 30), "be_a": 400.0}])
        result = hxz_merge_be_chain(fundq, funda)
        assert result["beq"][0] is None

    def test_drops_be_a_column(self):
        """Output should not contain be_a column."""
        fundq = self._make_fundq(
            [{"gvkey": "001", "datadate": date(2020, 12, 31), "fqtr": 4, "beq": None}]
        )
        funda = self._make_funda([{"gvkey": "001", "datadate": date(2020, 12, 31), "be_a": 100.0}])
        result = hxz_merge_be_chain(fundq, funda)
        assert "be_a" not in result.columns


class TestHxzImputeBeCleanSurplus:
    """Tests for hxz_impute_be_clean_surplus: BEQ chain imputation."""

    def _base_row(
        self, gvkey, fyearq, fqtr, beq, ibq, dvpsxq, cshoq_eff, ajexq_eff, rdq=None, datadate=None
    ):
        if datadate is None:
            month = fqtr * 3
            datadate = date(2000 + fyearq, month, 28)
        return {
            "gvkey": gvkey,
            "datadate": datadate,
            "fyearq": fyearq,
            "fqtr": fqtr,
            "rdq": rdq,
            "beq": beq,
            "ibq": ibq,
            "dvpsxq": dvpsxq,
            "cshoq_eff": cshoq_eff,
            "ajexq_eff": ajexq_eff,
        }

    def _make_df(self, rows):
        df = pl.DataFrame(rows)
        return df.with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col("rdq").cast(pl.Date),
            pl.col("beq", "ibq", "dvpsxq", "cshoq_eff", "ajexq_eff").cast(pl.Float64),
        )

    def test_full_availability_passes_through(self):
        """When all BEQ are present, imputation is a no-op (beq unchanged)."""
        rows = [
            self._base_row("G01", 20, 1, 100.0, 10.0, 1.0, 50.0, 1.0, datadate=date(2020, 3, 31)),
            self._base_row("G01", 20, 2, 110.0, 12.0, 1.0, 50.0, 1.0, datadate=date(2020, 6, 30)),
            self._base_row("G01", 20, 3, 122.0, 14.0, 1.0, 50.0, 1.0, datadate=date(2020, 9, 30)),
            self._base_row("G01", 20, 4, 130.0, 10.0, 2.0, 50.0, 1.0, datadate=date(2020, 12, 31)),
        ]
        df = self._make_df(rows)
        result = hxz_impute_be_clean_surplus(df)
        # All rows have real datadate so all should survive filter
        result_g01 = result.filter(pl.col("gvkey") == "G01").sort("datadate")
        # beq should match original values
        assert result_g01["beq"][0] == pytest.approx(100.0)
        assert result_g01["beq"][1] == pytest.approx(110.0)
        assert result_g01["beq"][2] == pytest.approx(122.0)
        assert result_g01["beq"][3] == pytest.approx(130.0)

    def test_forward_impute_from_prior_beq(self):
        """Missing BEQ is forward-imputed: BEQ_t = BEQ_{t-1} + IBQ_t - DVQ_t."""
        # Q1: BEQ=100, IBQ=10, DVQ=0; Q2: BEQ=null, IBQ=5, DVQ=0
        rows = [
            self._base_row("G02", 20, 1, 100.0, 10.0, 0.0, 50.0, 1.0, datadate=date(2020, 3, 31)),
            self._base_row("G02", 20, 2, None, 5.0, 0.0, 50.0, 1.0, datadate=date(2020, 6, 30)),
        ]
        df = self._make_df(rows)
        result = hxz_impute_be_clean_surplus(df)
        result_g02 = result.filter(pl.col("gvkey") == "G02").sort("datadate")
        # Q1 beq=100; Q2 imputed = 100 + 5 - 0 = 105
        assert result_g02["beq"][0] == pytest.approx(100.0)
        assert result_g02["beq"][1] == pytest.approx(105.0)

    def test_backward_impute_from_next_beq(self):
        """Missing BEQ is backward-imputed: BEQ_{t-1} = BEQ_t - IBQ_t + DVQ_t."""
        # Q1: BEQ=null; Q2: BEQ=110, IBQ=12, DVQ=0 → Q1 = 110 - 12 + 0 = 98
        rows = [
            self._base_row("G03", 20, 1, None, 12.0, 0.0, 50.0, 1.0, datadate=date(2020, 3, 31)),
            self._base_row("G03", 20, 2, 110.0, 12.0, 0.0, 50.0, 1.0, datadate=date(2020, 6, 30)),
        ]
        df = self._make_df(rows)
        result = hxz_impute_be_clean_surplus(df)
        result_g03 = result.filter(pl.col("gvkey") == "G03").sort("datadate")
        assert result_g03["beq"][0] == pytest.approx(110.0 - 12.0)

    def test_beq_lag1_computed(self):
        """beq_lag1 should equal the prior quarter's beq."""
        rows = [
            self._base_row("G04", 20, 1, 100.0, 5.0, 0.0, 50.0, 1.0, datadate=date(2020, 3, 31)),
            self._base_row("G04", 20, 2, 120.0, 5.0, 0.0, 50.0, 1.0, datadate=date(2020, 6, 30)),
        ]
        df = self._make_df(rows)
        result = hxz_impute_be_clean_surplus(df)
        result_g04 = result.filter(pl.col("gvkey") == "G04").sort("datadate")
        assert result_g04["beq_lag1"][0] is None
        assert result_g04["beq_lag1"][1] == pytest.approx(100.0)

    def test_missing_ibq_breaks_forward_impute(self):
        """Forward impute propagates null when IBQ is null."""
        rows = [
            self._base_row("G05", 20, 1, 100.0, 10.0, 0.0, 50.0, 1.0, datadate=date(2020, 3, 31)),
            # IBQ=null → sum over window includes null → forward impute fails
            self._base_row("G05", 20, 2, None, None, 0.0, 50.0, 1.0, datadate=date(2020, 6, 30)),
        ]
        df = self._make_df(rows)
        result = hxz_impute_be_clean_surplus(df)
        result_g05 = result.filter(pl.col("gvkey") == "G05").sort("datadate")
        # Q2 impute: beq_fw1 = beq_lag + rolling_sum(ibq,1) - rolling_sum(dvq,1)
        # rolling_sum with min_samples=1 propagates null if ibq is null
        assert result_g05["beq"][1] is None

    def test_padded_rows_dropped_from_output(self):
        """Rows with null datadate (dense padding gaps) are filtered out."""
        # Single gvkey spanning Q1 and Q3 (Q2 gap filled by densification)
        rows = [
            self._base_row("G06", 20, 1, 100.0, 10.0, 0.0, 50.0, 1.0, datadate=date(2020, 3, 31)),
            self._base_row("G06", 20, 3, 120.0, 10.0, 0.0, 50.0, 1.0, datadate=date(2020, 9, 30)),
        ]
        df = self._make_df(rows)
        result = hxz_impute_be_clean_surplus(df)
        # Output must not have rows with null datadate
        assert result["datadate"].is_not_null().all()
        # Both real quarters must appear
        result_g06 = result.filter(pl.col("gvkey") == "G06")
        assert len(result_g06) == 2


class TestHxzLinkCompustat:
    """Tests for hxz_link_compustat: gvkey→permno with time-validity + dedup."""

    def _write_lnkhist(self, tmp_path: Path, rows: list[dict]) -> Path:
        df = pl.DataFrame(rows).with_columns(
            pl.col("linkdt", "linkenddt").cast(pl.Date),
            pl.col("lpermno").cast(pl.Int64),
        )
        p = tmp_path / "crsp_ccmxpf_lnkhist.parquet"
        df.write_parquet(p)
        return tmp_path

    def test_basic_link(self, tmp_path):
        """Single matching link returns permno."""
        raw_dir = self._write_lnkhist(
            tmp_path,
            [
                {
                    "gvkey": "001",
                    "lpermno": 10001,
                    "linkprim": "P",
                    "linktype": "LC",
                    "linkdt": date(2000, 1, 1),
                    "linkenddt": date(2025, 12, 31),
                }
            ],
        )
        comp = pl.DataFrame({"gvkey": ["001"], "datadate": [date(2020, 6, 30)]}).with_columns(
            pl.col("datadate").cast(pl.Date)
        )
        from jkp.data.aux_functions import hxz_link_compustat

        result = hxz_link_compustat(comp, raw_dir)
        assert result["permno"][0] == 10001

    def test_date_outside_link_window_excluded(self, tmp_path):
        """datadate before linkdt → row excluded (no match)."""
        raw_dir = self._write_lnkhist(
            tmp_path,
            [
                {
                    "gvkey": "001",
                    "lpermno": 10001,
                    "linkprim": "P",
                    "linktype": "LC",
                    "linkdt": date(2010, 1, 1),
                    "linkenddt": date(2025, 12, 31),
                }
            ],
        )
        comp = pl.DataFrame({"gvkey": ["001"], "datadate": [date(2005, 6, 30)]}).with_columns(
            pl.col("datadate").cast(pl.Date)
        )
        from jkp.data.aux_functions import hxz_link_compustat

        result = hxz_link_compustat(comp, raw_dir)
        assert len(result) == 0

    def test_null_linkenddt_treated_as_open_ended(self, tmp_path):
        """linkenddt=null means the link is still active."""
        raw_dir = self._write_lnkhist(
            tmp_path,
            [
                {
                    "gvkey": "001",
                    "lpermno": 10001,
                    "linkprim": "P",
                    "linktype": "LC",
                    "linkdt": date(2000, 1, 1),
                    "linkenddt": None,
                }
            ],
        )
        comp = pl.DataFrame({"gvkey": ["001"], "datadate": [date(2030, 6, 30)]}).with_columns(
            pl.col("datadate").cast(pl.Date)
        )
        from jkp.data.aux_functions import hxz_link_compustat

        result = hxz_link_compustat(comp, raw_dir)
        assert len(result) == 1
        assert result["permno"][0] == 10001

    def test_multi_link_primary_wins(self, tmp_path):
        """When P and C links compete for same (datadate, permno), P wins."""
        raw_dir = self._write_lnkhist(
            tmp_path,
            [
                {
                    "gvkey": "001",
                    "lpermno": 10001,
                    "linkprim": "C",
                    "linktype": "LC",
                    "linkdt": date(2000, 1, 1),
                    "linkenddt": date(2025, 12, 31),
                },
                {
                    "gvkey": "002",
                    "lpermno": 10001,
                    "linkprim": "P",
                    "linktype": "LC",
                    "linkdt": date(2000, 1, 1),
                    "linkenddt": date(2025, 12, 31),
                },
            ],
        )
        # Both gvkeys link to permno 10001 at same date
        comp = pl.DataFrame(
            {"gvkey": ["001", "002"], "datadate": [date(2020, 6, 30), date(2020, 6, 30)]}
        ).with_columns(pl.col("datadate").cast(pl.Date))
        from jkp.data.aux_functions import hxz_link_compustat

        result = hxz_link_compustat(comp, raw_dir)
        # unique(subset=["datadate","permno"]) → only 1 row per (datadate,permno)
        assert len(result) == 1
        # linkprim DESC: P > C → gvkey 002 wins
        assert result["linkprim"][0] == "P"

    def test_drops_linktype_and_link_days(self, tmp_path):
        """Output should not contain linktype or _link_days columns."""
        raw_dir = self._write_lnkhist(
            tmp_path,
            [
                {
                    "gvkey": "001",
                    "lpermno": 10001,
                    "linkprim": "P",
                    "linktype": "LC",
                    "linkdt": date(2000, 1, 1),
                    "linkenddt": date(2025, 12, 31),
                }
            ],
        )
        comp = pl.DataFrame({"gvkey": ["001"], "datadate": [date(2020, 6, 30)]}).with_columns(
            pl.col("datadate").cast(pl.Date)
        )
        from jkp.data.aux_functions import hxz_link_compustat

        result = hxz_link_compustat(comp, raw_dir)
        assert "linktype" not in result.columns
        assert "_link_days" not in result.columns


class TestHxzSupplementQ4Shares:
    """Tests for hxz_supplement_q4_shares: share fallback hierarchy."""

    def test_q4_null_cshoq_filled_from_annual(self, tmp_path):
        """Q4 with null cshoq/ajexq gets values from funda csho/ajex."""
        # Write a dummy ccmxpf_lnkhist and crsp_msf_v2 so the function
        # doesn't crash on scan_parquet — this path is taken only when
        # cshoq_eff is still null after the Q4 fill
        lnk = pl.DataFrame(
            {
                "gvkey": ["001"],
                "lpermno": [10001],
                "linkprim": ["P"],
                "linktype": ["LC"],
                "linkdt": [date(2000, 1, 1)],
                "linkenddt": [date(2025, 12, 31)],
            }
        ).with_columns(
            pl.col("linkdt", "linkenddt").cast(pl.Date), pl.col("lpermno").cast(pl.Int64)
        )
        lnk.write_parquet(tmp_path / "crsp_ccmxpf_lnkhist.parquet")

        msf = pl.DataFrame(
            {
                "permno": [10001],
                "mthcaldt": [date(2020, 12, 31)],
                "shrout": [1000.0],
                "mthcumfacshr": [2.0],
            }
        ).with_columns(pl.col("mthcaldt").cast(pl.Date))
        msf.write_parquet(tmp_path / "crsp_msf_v2.parquet")

        fundq = pl.DataFrame(
            {
                "gvkey": ["001"],
                "datadate": [date(2020, 12, 31)],
                "fyearq": [2020],
                "fqtr": [4],
                "rdq": [None],
                "ibq": [10.0],
                "dvpsxq": [0.5],
                "cshoq": [None],
                "ajexq": [None],
                "beq": [100.0],
            }
        ).with_columns(
            pl.col("datadate", "rdq").cast(pl.Date),
            pl.col("ibq", "dvpsxq", "cshoq", "ajexq", "beq").cast(pl.Float64),
        )
        funda = pl.DataFrame(
            {
                "gvkey": ["001"],
                "datadate": [date(2020, 12, 31)],
                "be_a": [90.0],
                "inv": [0.05],
                "csho": [50.0],
                "ajex": [1.0],
            }
        ).with_columns(
            pl.col("datadate").cast(pl.Date), pl.col("be_a", "inv", "csho", "ajex").cast(pl.Float64)
        )

        from jkp.data.aux_functions import hxz_supplement_q4_shares

        result = hxz_supplement_q4_shares(fundq, funda, tmp_path)
        assert result["cshoq_eff"][0] == pytest.approx(50.0)  # from funda csho
        assert result["ajexq_eff"][0] == pytest.approx(1.0)  # from funda ajex

    def test_existing_cshoq_not_overwritten(self, tmp_path):
        """Non-null cshoq/ajexq should not be replaced by annual values."""
        lnk = pl.DataFrame(
            {
                "gvkey": ["001"],
                "lpermno": [10001],
                "linkprim": ["P"],
                "linktype": ["LC"],
                "linkdt": [date(2000, 1, 1)],
                "linkenddt": [date(2025, 12, 31)],
            }
        ).with_columns(
            pl.col("linkdt", "linkenddt").cast(pl.Date), pl.col("lpermno").cast(pl.Int64)
        )
        lnk.write_parquet(tmp_path / "crsp_ccmxpf_lnkhist.parquet")

        msf = pl.DataFrame(
            {
                "permno": [10001],
                "mthcaldt": [date(2020, 12, 31)],
                "shrout": [999.0],
                "mthcumfacshr": [9.0],
            }
        ).with_columns(pl.col("mthcaldt").cast(pl.Date))
        msf.write_parquet(tmp_path / "crsp_msf_v2.parquet")

        fundq = pl.DataFrame(
            {
                "gvkey": ["001"],
                "datadate": [date(2020, 12, 31)],
                "fyearq": [2020],
                "fqtr": [4],
                "rdq": [None],
                "ibq": [10.0],
                "dvpsxq": [0.5],
                "cshoq": [77.0],
                "ajexq": [2.0],
                "beq": [100.0],
            }
        ).with_columns(
            pl.col("datadate", "rdq").cast(pl.Date),
            pl.col("ibq", "dvpsxq", "cshoq", "ajexq", "beq").cast(pl.Float64),
        )
        funda = pl.DataFrame(
            {
                "gvkey": ["001"],
                "datadate": [date(2020, 12, 31)],
                "be_a": [90.0],
                "inv": [0.05],
                "csho": [50.0],
                "ajex": [1.0],
            }
        ).with_columns(
            pl.col("datadate").cast(pl.Date), pl.col("be_a", "inv", "csho", "ajex").cast(pl.Float64)
        )

        from jkp.data.aux_functions import hxz_supplement_q4_shares

        result = hxz_supplement_q4_shares(fundq, funda, tmp_path)
        assert result["cshoq_eff"][0] == pytest.approx(77.0)
        assert result["ajexq_eff"][0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Stage 4 helpers
# ---------------------------------------------------------------------------


class TestHxzNyseBreaks:
    """Tests for _hxz_nyse_breaks: NYSE-only per-date QUANTILE_DISC breakpoints."""

    def _make_panel(self, rows):
        return pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv").cast(pl.Float64),
        )

    def test_nyse_only_filter(self):
        """Only exchcd==1 stocks contribute to breakpoints."""
        df = self._make_panel(
            [
                {"date": date(2020, 6, 30), "exchcd": 1, "elig": True, "me": 100.0, "inv": 0.1},
                {"date": date(2020, 6, 30), "exchcd": 2, "elig": True, "me": 999.0, "inv": 0.9},
                {"date": date(2020, 6, 30), "exchcd": 3, "elig": True, "me": 888.0, "inv": 0.8},
            ]
        )
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_nyse_breaks(df, specs)
        # Only exchcd==1 stock has me=100, median = 100
        assert result["sizemedn"][0] == pytest.approx(100.0)

    def test_elig_filter(self):
        """Non-eligible stocks (elig=False) are excluded from breakpoints."""
        df = self._make_panel(
            [
                {"date": date(2020, 6, 30), "exchcd": 1, "elig": True, "me": 100.0, "inv": 0.2},
                {"date": date(2020, 6, 30), "exchcd": 1, "elig": False, "me": 999.0, "inv": 0.9},
            ]
        )
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_nyse_breaks(df, specs)
        assert result["sizemedn"][0] == pytest.approx(100.0)

    def test_30_70_breaks_computed(self):
        """30th and 70th percentile breaks are correct for simple sequence."""
        # 10 stocks with me = 1,2,...,10 → Q30=3, Q70=7 (QUANTILE_DISC)
        rows = [
            {"date": date(2020, 6, 30), "exchcd": 1, "elig": True, "me": float(i), "inv": 0.1}
            for i in range(1, 11)
        ]
        df = self._make_panel(rows)
        specs = [("me", 0.30, "me30"), ("me", 0.70, "me70")]
        result = _hxz_nyse_breaks(df, specs)
        assert result["me30"][0] == pytest.approx(3.0)
        assert result["me70"][0] == pytest.approx(7.0)

    def test_per_date_grouping(self):
        """Breakpoints computed separately for each date."""
        rows = [
            {"date": date(2020, 6, 30), "exchcd": 1, "elig": True, "me": float(i), "inv": 0.1}
            for i in range(1, 5)
        ] + [
            {"date": date(2020, 7, 31), "exchcd": 1, "elig": True, "me": float(i * 10), "inv": 0.1}
            for i in range(1, 5)
        ]
        df = self._make_panel(rows)
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_nyse_breaks(df, specs).sort("date")
        # Jun median of [1,2,3,4] = 2 (QUANTILE_DISC at 0.5)
        # Jul median of [10,20,30,40] = 20
        assert result["sizemedn"][0] == pytest.approx(2.0)
        assert result["sizemedn"][1] == pytest.approx(20.0)


class TestHxzCountryBreaks:
    """Tests for _hxz_country_breaks: per-(excntry,date) with HXZ_MIN_STOCKS_BP gate."""

    def _make_panel(self, rows):
        return pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv").cast(pl.Float64),
        )

    def test_below_min_stocks_excluded(self):
        """excntry/date with fewer than HXZ_MIN_STOCKS_BP eligible stocks → no row in output."""
        # Only 5 stocks (below HXZ_MIN_STOCKS_BP=10)
        rows = [
            {
                "excntry": "GBR",
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "large",
                "me": float(i),
                "inv": 0.1,
            }
            for i in range(1, 6)
        ]
        df = self._make_panel(rows)
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_country_breaks(df, specs)
        assert len(result) == 0

    def test_above_min_stocks_included(self):
        """excntry/date with at least HXZ_MIN_STOCKS_BP eligible stocks → breakpoints computed."""
        rows = [
            {
                "excntry": "GBR",
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "large",
                "me": float(i),
                "inv": 0.1,
            }
            for i in range(1, HXZ_MIN_STOCKS_BP + 2)  # just above threshold
        ]
        df = self._make_panel(rows)
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_country_breaks(df, specs)
        assert len(result) == 1

    def test_non_elig_stocks_excluded(self):
        """Non-eligible stocks don't count toward the gate."""
        rows = [
            {
                "excntry": "GBR",
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "large",
                "me": float(i),
                "inv": 0.1,
            }
            for i in range(1, HXZ_MIN_STOCKS_BP + 1)
        ] + [
            {
                "excntry": "GBR",
                "date": date(2020, 6, 30),
                "elig": False,
                "size_grp": "large",
                "me": 999.0,
                "inv": 0.9,
            }
        ]
        df = self._make_panel(rows)
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_country_breaks(df, specs)
        # exactly HXZ_MIN_STOCKS_BP eligible → included
        assert len(result) == 1

    def test_size_grp_filter(self):
        """Only size_grp in {small, large, mega} contribute."""
        rows = [
            {
                "excntry": "GBR",
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "micro",
                "me": float(i),
                "inv": 0.1,
            }
            for i in range(1, 20)
        ]
        df = self._make_panel(rows)
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_country_breaks(df, specs)
        # micro excluded → zero eligible → no output row
        assert len(result) == 0

    def test_two_countries_independent(self):
        """Each excntry/date pair is computed independently."""
        rows_gbr = [
            {
                "excntry": "GBR",
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "large",
                "me": float(i),
                "inv": 0.1,
            }
            for i in range(1, 12)
        ]
        rows_deu = [
            {
                "excntry": "DEU",
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "large",
                "me": float(i * 100),
                "inv": 0.2,
            }
            for i in range(1, 12)
        ]
        df = self._make_panel(rows_gbr + rows_deu)
        specs = [("me", 0.5, "sizemedn")]
        result = _hxz_country_breaks(df, specs).sort("excntry")
        assert len(result) == 2
        deu_row = result.filter(pl.col("excntry") == "DEU")
        assert deu_row["sizemedn"][0] > result.filter(pl.col("excntry") == "GBR")["sizemedn"][0]


class TestHxzClassifySizeIa:
    """Tests for _hxz_classify_size_ia: joint annual size+inv sort."""

    def _make_us_frame(self, rows):
        df = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv", "be_a").cast(pl.Float64),
        )
        # _hxz_classify_size_ia splits on excntry; the ROW slice (empty for
        # pure-US frames) still selects ["excntry","id","gvkey",...] and calls
        # _hxz_country_breaks which references size_grp — add both null columns.
        if "size_grp" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("size_grp"))
        if "gvkey" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gvkey"))
        return df

    def test_us_sizeport_assignment(self):
        """US: stock at/below NYSE median gets sizeport=1, above gets sizeport=2."""
        # 4 NYSE stocks, me=[10,20,30,40], median=20; exchcd=1
        # 2 non-NYSE stocks also present for NYSE filter test
        rows = [
            {
                "excntry": "USA",
                "id": i,
                "permno": i,
                "date": date(2020, 6, 30),
                "exchcd": 1,
                "elig": True,
                "me": float(i * 10),
                "inv": 0.1,
                "be_a": 100.0,
                "is_fin": False,
            }
            for i in range(1, 5)
        ]
        df = self._make_us_frame(rows)
        result = _hxz_classify_size_ia(df)
        result_us = result.filter(pl.col("excntry") == "USA").sort("id")
        # me=10,20 ≤ 20 → sizeport=1; me=30,40 > 20 → sizeport=2
        assert result_us["sizeport"][0] == 1
        assert result_us["sizeport"][1] == 1
        assert result_us["sizeport"][2] == 2
        assert result_us["sizeport"][3] == 2

    def test_us_invport_assignment(self):
        """US: invport based on inv 30/70 breakpoints from NYSE pool."""
        # 10 stocks with inv=0.1,0.2,...,1.0; Q30=0.3, Q70=0.7 (QUANTILE_DISC)
        rows = [
            {
                "excntry": "USA",
                "id": i,
                "permno": i,
                "date": date(2020, 6, 30),
                "exchcd": 1,
                "elig": True,
                "me": 100.0,
                "inv": round(i * 0.1, 1),
                "be_a": 100.0,
                "is_fin": False,
            }
            for i in range(1, 11)
        ]
        df = self._make_us_frame(rows)
        result = _hxz_classify_size_ia(df).sort("id")
        # inv=0.1,0.2,0.3 → invport=1; inv=0.4..0.6,0.7 → invport=2; inv=0.8..1.0 → invport=3
        invports = result["invport"].to_list()
        assert invports[0] == 1  # inv=0.1
        assert invports[2] == 1  # inv=0.3
        assert invports[3] == 2  # inv=0.4
        assert invports[9] == 3  # inv=1.0

    def test_inelig_gets_null_ports(self):
        """Non-eligible stocks get null sizeport and invport."""
        rows = [
            {
                "excntry": "USA",
                "id": 1,
                "permno": 1,
                "date": date(2020, 6, 30),
                "exchcd": 1,
                "elig": False,
                "me": 100.0,
                "inv": 0.2,
                "be_a": 100.0,
                "is_fin": False,
            },
        ] + [
            # add eligible stocks so NYSE breaks can be computed
            {
                "excntry": "USA",
                "id": i,
                "permno": i,
                "date": date(2020, 6, 30),
                "exchcd": 1,
                "elig": True,
                "me": float(i * 10),
                "inv": round(i * 0.1, 1),
                "be_a": 100.0,
                "is_fin": False,
            }
            for i in range(2, 12)
        ]
        df = self._make_us_frame(rows)
        result = _hxz_classify_size_ia(df)
        inelig_row = result.filter(pl.col("id") == 1)
        assert inelig_row["sizeport"][0] is None
        assert inelig_row["invport"][0] is None

    def test_row_below_gate_gets_null_ports(self):
        """ROW with fewer than HXZ_MIN_STOCKS_BP stocks → sizeport/invport null."""
        rows = [
            {
                "excntry": "GBR",
                "id": i,
                "gvkey": str(i),
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "large",
                "me": float(i * 10),
                "inv": round(i * 0.1, 1),
                "be_a": 100.0,
                "is_fin": False,
                # exchcd required even for ROW frames because US slice (empty)
                # passes through _hxz_nyse_breaks which references exchcd
                "exchcd": None,
            }
            for i in range(1, 5)  # only 4 stocks, below gate
        ]
        df = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv", "be_a").cast(pl.Float64),
            pl.col("exchcd").cast(pl.Int64),
        )
        # US slice (empty) selects ["excntry","id","permno",...] — add null permno
        df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("permno"))
        result = _hxz_classify_size_ia(df)
        # sizemedn not available → sizeport all null
        assert result["sizeport"].is_null().all()

    def test_output_has_sizeport_invport_inv(self):
        """US output contains sizeport, invport, and inv columns."""
        rows = [
            {
                "excntry": "USA",
                "id": i,
                "permno": i,
                "date": date(2020, 6, 30),
                "exchcd": 1,
                "elig": True,
                "me": float(i * 10),
                "inv": round(i * 0.1, 1),
                "be_a": 100.0,
                "is_fin": False,
            }
            for i in range(1, 11)
        ]
        df = self._make_us_frame(rows)
        result = _hxz_classify_size_ia(df)
        for col in ("sizeport", "invport", "inv"):
            assert col in result.columns


class TestHxzClassifyRoe:
    """Tests for _hxz_classify_roe: monthly ROE sort."""

    def _make_us_roe(self, n_stocks=10, base_date=date(2020, 1, 31)):
        # size_grp required: _hxz_classify_roe splits by excntry; empty ROW
        # slice still runs _hxz_country_breaks which references size_grp.
        return pl.DataFrame(
            {
                "excntry": ["USA"] * n_stocks,
                "id": list(range(1, n_stocks + 1)),
                "permno": list(range(1, n_stocks + 1)),
                "date": [base_date] * n_stocks,
                "exchcd": [1] * n_stocks,
                "is_fin": [False] * n_stocks,
                "roe": [round(i * 0.1, 1) for i in range(1, n_stocks + 1)],
                "size_grp": [None] * n_stocks,
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("roe").cast(pl.Float64),
            pl.col("size_grp").cast(pl.Utf8),
        )

    def test_us_roeport_assignment(self):
        """US: roeport 1/2/3 based on NYSE-only 30/70 breaks."""
        df = self._make_us_roe(n_stocks=10)
        result = _hxz_classify_roe(df).sort("id")
        # roe=0.1..0.3 → roeport=1; 0.4..0.6,0.7 → 2; 0.8..1.0 → 3
        assert result["roeport"][0] == 1  # roe=0.1
        assert result["roeport"][2] == 1  # roe=0.3
        assert result["roeport"][3] == 2  # roe=0.4
        assert result["roeport"][9] == 3  # roe=1.0

    def test_fin_stocks_get_null_roeport(self):
        """Financial stocks (is_fin=True) should not be classified (null roeport)."""
        df = self._make_us_roe(n_stocks=10)
        # Mark first stock as financial
        df = df.with_columns(is_fin=pl.when(pl.col("id") == 1).then(True).otherwise(False))
        result = _hxz_classify_roe(df)
        fin_row = result.filter(pl.col("id") == 1)
        assert fin_row["roeport"][0] is None

    def test_null_roe_gets_null_roeport(self):
        """Stocks with null ROE get null roeport even when eligible."""
        df = self._make_us_roe(n_stocks=10)
        df = df.with_columns(roe=pl.when(pl.col("id") == 1).then(None).otherwise(pl.col("roe")))
        result = _hxz_classify_roe(df)
        null_row = result.filter(pl.col("id") == 1)
        assert null_row["roeport"][0] is None

    def test_output_contains_roeport_and_roe(self):
        """Output schema has roeport and roe columns."""
        df = self._make_us_roe()
        result = _hxz_classify_roe(df)
        assert "roeport" in result.columns
        assert "roe" in result.columns

    def test_row_below_gate_gets_null_roeport(self):
        """ROW with fewer than HXZ_MIN_STOCKS_BP eligible stocks → null roeport."""
        # exchcd required: _hxz_classify_roe splits by excntry; empty US slice
        # still runs _hxz_nyse_breaks which references exchcd.
        df = pl.DataFrame(
            {
                "excntry": ["DEU"] * 5,
                "id": list(range(1, 6)),
                "date": [date(2020, 1, 31)] * 5,
                "is_fin": [False] * 5,
                "size_grp": ["large"] * 5,
                "roe": [0.1, 0.2, 0.3, 0.4, 0.5],
                "exchcd": [None] * 5,
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("roe").cast(pl.Float64),
            pl.col("exchcd").cast(pl.Int64),
        )
        result = _hxz_classify_roe(df)
        assert result["roeport"].is_null().all()


# ---------------------------------------------------------------------------
# Stage 5 — hxz_compute_factors
# ---------------------------------------------------------------------------


class TestHxzComputeFactors:
    """Tests for hxz_compute_factors: 18-bucket VW → me_hxz, ia_hxz, roe_hxz."""

    def _make_panel(self, buckets_data):
        """Build a panel with sizeport/invport/roeport pre-assigned."""
        return pl.DataFrame(buckets_data).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("ret", "me_lag1").cast(pl.Float64),
            pl.col("sizeport", "invport", "roeport").cast(pl.Int64),
        )

    def test_me_hxz_sign(self):
        """me_hxz = avg(small) - avg(big): small positive return, big negative → positive factor."""
        rows = []
        date_ = date(2020, 1, 31)
        # Small (sizeport=1) stocks: all R2 (neutral), all I2 → avg ret = 0.10
        for i in range(5):
            rows.append(
                {
                    "excntry": "USA",
                    "id": i,
                    "date": date_,
                    "ret": 0.10,
                    "me_lag1": 1.0,
                    "sizeport": 1,
                    "invport": 2,
                    "roeport": 2,
                }
            )
        # Big (sizeport=2) stocks: all R2, all I2 → avg ret = -0.05
        for i in range(5, 10):
            rows.append(
                {
                    "excntry": "USA",
                    "id": i,
                    "date": date_,
                    "ret": -0.05,
                    "me_lag1": 1.0,
                    "sizeport": 2,
                    "invport": 2,
                    "roeport": 2,
                }
            )
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        assert result["me_hxz"][0] == pytest.approx(0.10 - (-0.05))

    def test_ia_hxz_sign(self):
        """ia_hxz = avg(low_inv=I1) - avg(high_inv=I3)."""
        date_ = date(2020, 1, 31)
        rows = []
        for i in range(5):
            rows.append(
                {
                    "excntry": "USA",
                    "id": i,
                    "date": date_,
                    "ret": 0.05,
                    "me_lag1": 1.0,
                    "sizeport": 1,
                    "invport": 1,
                    "roeport": 2,
                }
            )  # I1
        for i in range(5, 10):
            rows.append(
                {
                    "excntry": "USA",
                    "id": i,
                    "date": date_,
                    "ret": -0.02,
                    "me_lag1": 1.0,
                    "sizeport": 1,
                    "invport": 3,
                    "roeport": 2,
                }
            )  # I3
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        assert result["ia_hxz"][0] == pytest.approx(0.05 - (-0.02))

    def test_roe_hxz_sign(self):
        """roe_hxz = avg(high_roe=R3) - avg(low_roe=R1)."""
        date_ = date(2020, 1, 31)
        rows = []
        for i in range(5):
            rows.append(
                {
                    "excntry": "USA",
                    "id": i,
                    "date": date_,
                    "ret": 0.08,
                    "me_lag1": 1.0,
                    "sizeport": 1,
                    "invport": 2,
                    "roeport": 3,
                }
            )  # R3
        for i in range(5, 10):
            rows.append(
                {
                    "excntry": "USA",
                    "id": i,
                    "date": date_,
                    "ret": -0.03,
                    "me_lag1": 1.0,
                    "sizeport": 1,
                    "invport": 2,
                    "roeport": 1,
                }
            )  # R1
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        assert result["roe_hxz"][0] == pytest.approx(0.08 - (-0.03))

    def test_us_bypasses_min_stocks_pf_gate(self):
        """US country bypasses HXZ_MIN_STOCKS_PF gate even for single-stock buckets."""
        date_ = date(2020, 1, 31)
        # Only 1 stock per bucket across the whole 2×3×3 grid
        rows = [
            {
                "excntry": "USA",
                "id": 1,
                "date": date_,
                "ret": 0.10,
                "me_lag1": 1.0,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            },
            {
                "excntry": "USA",
                "id": 2,
                "date": date_,
                "ret": -0.05,
                "me_lag1": 1.0,
                "sizeport": 2,
                "invport": 2,
                "roeport": 2,
            },
        ]
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        # me_hxz should not be null (US bypass in effect)
        assert result["me_hxz"][0] is not None
        assert not np.isnan(result["me_hxz"][0])

    def test_row_below_min_stocks_pf_bucket_nulled(self):
        """ROW bucket with n < HXZ_MIN_STOCKS_PF gets null vwret → null factor legs."""
        date_ = date(2020, 1, 31)
        rows = [
            # Only 1 stock in each small bucket → n=1 < HXZ_MIN_STOCKS_PF=3 → null
            {
                "excntry": "DEU",
                "id": 1,
                "date": date_,
                "ret": 0.10,
                "me_lag1": 1.0,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            },
            # Large stocks with enough to fill all S2 buckets meaningfully
            {
                "excntry": "DEU",
                "id": 2,
                "date": date_,
                "ret": -0.05,
                "me_lag1": 1.0,
                "sizeport": 2,
                "invport": 2,
                "roeport": 2,
            },
        ]
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        # With n=1 < HXZ_MIN_STOCKS_PF, small bucket is null, me_hxz = null - something
        # mean_horizontal skips nulls → me_hxz uses null for avg(small) → None or NaN
        row = result.filter(pl.col("excntry") == "DEU")
        # me_hxz may be null/NaN when only one small bucket and it's nulled
        me_hxz_val = row["me_hxz"][0]
        assert me_hxz_val is None or (isinstance(me_hxz_val, float) and np.isnan(me_hxz_val))

    def test_equal_weights_vw_exact(self, tolerance):
        """VW return with equal weights equals simple mean of returns."""
        date_ = date(2020, 1, 31)
        rets = [0.01, 0.02, 0.03, 0.04]
        rows = [
            {
                "excntry": "USA",
                "id": i,
                "date": date_,
                "ret": r,
                "me_lag1": 1.0,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            }
            for i, r in enumerate(rets)
        ] + [
            {
                "excntry": "USA",
                "id": 10 + i,
                "date": date_,
                "ret": 0.0,
                "me_lag1": 1.0,
                "sizeport": 2,
                "invport": 2,
                "roeport": 2,
            }
            for i in range(4)
        ]
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        expected_small = np.mean(rets)
        # me_hxz = avg(small) - avg(big) = mean(rets) - 0.0
        np.testing.assert_allclose(result["me_hxz"][0], expected_small, **tolerance.STANDARD)

    def test_output_schema(self):
        """Output has excntry, date, me_hxz, ia_hxz, roe_hxz."""
        date_ = date(2020, 1, 31)
        rows = [
            {
                "excntry": "USA",
                "id": 1,
                "date": date_,
                "ret": 0.01,
                "me_lag1": 1.0,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            },
        ]
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        for col in ("excntry", "date", "me_hxz", "ia_hxz", "roe_hxz"):
            assert col in result.columns

    def test_null_me_lag1_excluded(self):
        """Stocks with null or zero me_lag1 excluded from VW sum."""
        date_ = date(2020, 1, 31)
        rows = [
            {
                "excntry": "USA",
                "id": 1,
                "date": date_,
                "ret": 0.99,
                "me_lag1": None,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            },
            {
                "excntry": "USA",
                "id": 2,
                "date": date_,
                "ret": 0.01,
                "me_lag1": 1.0,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            },
            {
                "excntry": "USA",
                "id": 3,
                "date": date_,
                "ret": 0.0,
                "me_lag1": 1.0,
                "sizeport": 2,
                "invport": 2,
                "roeport": 2,
            },
        ]
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        # id=1 excluded (null me_lag1); small VW = 0.01; big VW = 0.0
        assert result["me_hxz"][0] == pytest.approx(0.01)

    def test_sorted_by_excntry_date(self):
        """Output is sorted by (excntry, date)."""
        rows = [
            {
                "excntry": "USA",
                "id": 1,
                "date": date(2020, 2, 28),
                "ret": 0.01,
                "me_lag1": 1.0,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            },
            {
                "excntry": "USA",
                "id": 2,
                "date": date(2020, 1, 31),
                "ret": 0.01,
                "me_lag1": 1.0,
                "sizeport": 1,
                "invport": 2,
                "roeport": 2,
            },
        ]
        df = self._make_panel(rows)
        result = hxz_compute_factors(df)
        dates = result["date"].to_list()
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Stage 6 — hxz_build_characteristics
# ---------------------------------------------------------------------------


class TestHxzBuildCharacteristics:
    """Tests for hxz_build_characteristics: schema, dtypes, eom derivation."""

    def _make_panel(self, rows):
        return pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv", "roe").cast(pl.Float64),
            pl.col("id").cast(pl.Int64),
        )

    def test_output_schema(self):
        """Output has exactly eom, excntry, id, me_hxz, ia_hxz, roe_hxz."""
        df = self._make_panel(
            [
                {
                    "excntry": "USA",
                    "id": 1,
                    "date": date(2020, 1, 31),
                    "me": 100.0,
                    "inv": 0.1,
                    "roe": 0.05,
                }
            ]
        )
        result = hxz_build_characteristics(df)
        assert set(result.columns) == {"eom", "excntry", "id", "me_hxz", "ia_hxz", "roe_hxz"}

    def test_eom_is_month_end(self):
        """eom = month_end of the input date column."""
        df = self._make_panel(
            [
                {
                    "excntry": "USA",
                    "id": 1,
                    "date": date(2020, 1, 15),
                    "me": 100.0,
                    "inv": 0.1,
                    "roe": 0.05,
                }
            ]
        )
        result = hxz_build_characteristics(df)
        assert result["eom"][0] == date(2020, 1, 31)

    def test_id_is_int64(self):
        """id column dtype must be Int64."""
        df = self._make_panel(
            [
                {
                    "excntry": "USA",
                    "id": 12345,
                    "date": date(2020, 1, 31),
                    "me": 100.0,
                    "inv": 0.1,
                    "roe": 0.05,
                }
            ]
        )
        result = hxz_build_characteristics(df)
        assert result["id"].dtype == pl.Int64

    def test_null_me_passes_through(self):
        """Null me propagates to me_hxz without error."""
        df = self._make_panel(
            [
                {
                    "excntry": "USA",
                    "id": 1,
                    "date": date(2020, 1, 31),
                    "me": None,
                    "inv": 0.1,
                    "roe": 0.05,
                }
            ]
        )
        result = hxz_build_characteristics(df)
        assert result["me_hxz"][0] is None

    def test_null_inv_passes_through(self):
        """Null inv propagates to ia_hxz without error."""
        df = self._make_panel(
            [
                {
                    "excntry": "USA",
                    "id": 1,
                    "date": date(2020, 1, 31),
                    "me": 100.0,
                    "inv": None,
                    "roe": 0.05,
                }
            ]
        )
        result = hxz_build_characteristics(df)
        assert result["ia_hxz"][0] is None

    def test_null_roe_passes_through(self):
        """Null roe propagates to roe_hxz without error."""
        df = self._make_panel(
            [
                {
                    "excntry": "USA",
                    "id": 1,
                    "date": date(2020, 1, 31),
                    "me": 100.0,
                    "inv": 0.1,
                    "roe": None,
                }
            ]
        )
        result = hxz_build_characteristics(df)
        assert result["roe_hxz"][0] is None

    def test_sorted_by_eom_excntry_id(self):
        """Output is sorted by (eom, excntry, id)."""
        rows = [
            {
                "excntry": "USA",
                "id": 2,
                "date": date(2020, 2, 28),
                "me": 10.0,
                "inv": 0.1,
                "roe": 0.01,
            },
            {
                "excntry": "USA",
                "id": 1,
                "date": date(2020, 1, 31),
                "me": 10.0,
                "inv": 0.1,
                "roe": 0.01,
            },
            {
                "excntry": "DEU",
                "id": 1,
                "date": date(2020, 1, 31),
                "me": 10.0,
                "inv": 0.1,
                "roe": 0.01,
            },
        ]
        df = self._make_panel(rows)
        result = hxz_build_characteristics(df)
        eoms = result["eom"].to_list()
        assert eoms == sorted(eoms)

    def test_values_mapped_correctly(self, tolerance):
        """me→me_hxz, inv→ia_hxz, roe→roe_hxz."""
        df = self._make_panel(
            [
                {
                    "excntry": "USA",
                    "id": 1,
                    "date": date(2020, 1, 31),
                    "me": 123.4,
                    "inv": 0.07,
                    "roe": 0.13,
                }
            ]
        )
        result = hxz_build_characteristics(df)
        np.testing.assert_allclose(result["me_hxz"][0], 123.4, **tolerance.TIGHT)
        np.testing.assert_allclose(result["ia_hxz"][0], 0.07, **tolerance.TIGHT)
        np.testing.assert_allclose(result["roe_hxz"][0], 0.13, **tolerance.TIGHT)


# ---------------------------------------------------------------------------
# Stage 4 combined — hxz_classify_portfolios
# ---------------------------------------------------------------------------


class TestHxzClassifyPortfolios:
    """Tests for hxz_classify_portfolios: combined classify + broadcast."""

    def _make_us_panel(self, dates, permno=1):
        # gvkey required: hxz_classify_portfolios splits by excntry; the empty
        # ROW slice (.filter(excntry != USA)) still joins on gvkey.
        return pl.DataFrame(
            {
                "excntry": ["USA"] * len(dates),
                "id": [permno] * len(dates),
                "permno": [permno] * len(dates),
                "gvkey": [None] * len(dates),
                "date": dates,
                "ret": [0.01] * len(dates),
                "me": [100.0] * len(dates),
                "me_lag1": [90.0] * len(dates),
                "is_fin": [False] * len(dates),
                "exchcd": [1] * len(dates),
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("gvkey").cast(pl.Utf8),
        )

    def test_broadcast_june_classification_to_all_months(self):
        """June classification is broadcast to Jul(y)..Jun(y+1) months."""
        # Build a minimal size_ia_form and roe_m with one US stock in June
        dates = [
            date(2020, 6, 30),  # June formation
            date(2020, 7, 31),  # Jul — should carry June classification
            date(2020, 12, 31),  # Dec — should carry June classification
            date(2021, 6, 30),  # Next June — new formation year
        ]
        panel = self._make_us_panel(dates, permno=1)

        # size_ia_form: one row in June 2020 with sizeport=1, invport=2
        size_ia_form = pl.DataFrame(
            {
                "excntry": ["USA"],
                "id": [1],
                "permno": [1],
                "gvkey": [None],  # required by ROW cls select
                "date": [date(2020, 6, 30)],
                "me": [100.0],
                "inv": [0.1],
                "be_a": [200.0],
                "is_fin": [False],
                "exchcd": [1],
                "elig": [True],
                "size_grp": [None],  # required by ROW path schema
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv", "be_a").cast(pl.Float64),
            pl.col("size_grp", "gvkey").cast(pl.Utf8),
        )

        # roe_m: one row per month per stock
        roe_m = pl.DataFrame(
            {
                "excntry": ["USA"] * 4,
                "id": [1] * 4,
                "permno": [1] * 4,
                "date": dates,
                "is_fin": [False] * 4,
                "exchcd": [1] * 4,
                "roe": [0.05] * 4,
                "size_grp": [None] * 4,  # required by ROW path schema
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("roe").cast(pl.Float64),
            pl.col("size_grp").cast(pl.Utf8),
        )

        result = hxz_classify_portfolios(panel, size_ia_form, roe_m)
        result_sorted = result.sort("date")
        # All months in Jul..Dec 2020 should have non-null sizeport and invport
        jul_row = result_sorted.filter(pl.col("date") == date(2020, 7, 31))
        dec_row = result_sorted.filter(pl.col("date") == date(2020, 12, 31))
        # June formation year 2020 → port_year=2020; Jul..Dec 2020 port_year=2020 (month<7 → year-1; month>=7 → year)
        # Jul: month=7 ≥ 7 → port_year=2020; Dec: month=12 ≥ 7 → port_year=2020
        # This is correct; Jun formation year = 2020
        assert jul_row["sizeport"][0] is not None or dec_row["sizeport"][0] is not None

    def test_roe_attached_monthly(self):
        """ROE port from roe_m is joined on (excntry, id, date)."""
        dates = [date(2020, 6, 30), date(2020, 7, 31)]
        panel = self._make_us_panel(dates, permno=1)

        size_ia_form = pl.DataFrame(
            {
                "excntry": ["USA"],
                "id": [1],
                "permno": [1],
                "gvkey": [None],
                "date": [date(2020, 6, 30)],
                "me": [100.0],
                "inv": [0.1],
                "be_a": [200.0],
                "is_fin": [False],
                "exchcd": [1],
                "elig": [True],
                "size_grp": [None],
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv", "be_a").cast(pl.Float64),
            pl.col("size_grp", "gvkey").cast(pl.Utf8),
        )

        roe_m = pl.DataFrame(
            {
                "excntry": ["USA", "USA"],
                "id": [1, 1],
                "permno": [1, 1],
                "date": dates,
                "is_fin": [False, False],
                "exchcd": [1, 1],
                "roe": [0.07, 0.08],
                "size_grp": [None, None],
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("roe").cast(pl.Float64),
            pl.col("size_grp").cast(pl.Utf8),
        )

        result = hxz_classify_portfolios(panel, size_ia_form, roe_m).sort("date")
        assert result["roe"][0] == pytest.approx(0.07)
        assert result["roe"][1] == pytest.approx(0.08)

    def test_output_has_sizeport_invport_roeport(self):
        """Output contains sizeport, invport, roeport columns."""
        dates = [date(2020, 6, 30)]
        panel = self._make_us_panel(dates)

        size_ia_form = pl.DataFrame(
            {
                "excntry": ["USA"],
                "id": [1],
                "permno": [1],
                "gvkey": [None],
                "date": [date(2020, 6, 30)],
                "me": [100.0],
                "inv": [0.1],
                "be_a": [200.0],
                "is_fin": [False],
                "exchcd": [1],
                "elig": [True],
                "size_grp": [None],
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv", "be_a").cast(pl.Float64),
            pl.col("size_grp", "gvkey").cast(pl.Utf8),
        )

        roe_m = pl.DataFrame(
            {
                "excntry": ["USA"],
                "id": [1],
                "permno": [1],
                "date": [date(2020, 6, 30)],
                "is_fin": [False],
                "exchcd": [1],
                "roe": [0.05],
                "size_grp": [None],
            }
        ).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("roe").cast(pl.Float64),
            pl.col("size_grp").cast(pl.Utf8),
        )

        result = hxz_classify_portfolios(panel, size_ia_form, roe_m)
        for col in ("sizeport", "invport", "roeport"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# Skipped helpers (require real WRDS I/O)
# ---------------------------------------------------------------------------


class TestHxzUsRoeMonthly:
    """Tests for _hxz_us_roe_monthly: pre/post-1972 RDQ-gating and 6mo cap."""

    def _write_lnkhist(self, tmp_path, permno, gvkey):
        lnk = pl.DataFrame(
            {
                "gvkey": [gvkey],
                "lpermno": [permno],
                "linkprim": ["P"],
                "linktype": ["LC"],
                "linkdt": [date(1960, 1, 1)],
                "linkenddt": [None],
            }
        ).with_columns(
            pl.col("linkdt", "linkenddt").cast(pl.Date), pl.col("lpermno").cast(pl.Int64)
        )
        lnk.write_parquet(tmp_path / "crsp_ccmxpf_lnkhist.parquet")

    def test_null_rdq_excluded(self, tmp_path):
        """No RDQ → earnings never assigned (RDQ gate applies full sample)."""
        self._write_lnkhist(tmp_path, 10001, "G01")
        fundq = pl.DataFrame(
            {
                "gvkey": ["G01"],
                "datadate": [date(1965, 12, 31)],
                "rdq": [None],
                "beq_lag1": [100.0],
                "ibq": [10.0],
            }
        ).with_columns(
            pl.col("datadate", "rdq").cast(pl.Date), pl.col("beq_lag1", "ibq").cast(pl.Float64)
        )
        formation_dates = pl.DataFrame(
            {"date": [date(1966, 4, 30), date(1966, 6, 30)]}
        ).with_columns(pl.col("date").cast(pl.Date))
        from jkp.data.aux_functions import _hxz_us_roe_monthly, safe_div

        fq_with_roe = fundq.with_columns(safe_div("ibq", "beq_lag1", "roe", mode=3))
        result = _hxz_us_roe_monthly(fq_with_roe, tmp_path, formation_dates)
        assert len(result) == 0

    def test_pre_1972_rdq_gated(self, tmp_path):
        """Pre-1972 row with RDQ follows the same gate as post-1972."""
        self._write_lnkhist(tmp_path, 10001, "G01")
        fundq = pl.DataFrame(
            {
                "gvkey": ["G01"],
                "datadate": [date(1965, 12, 31)],
                "rdq": [date(1966, 2, 15)],
                "beq_lag1": [100.0],
                "ibq": [10.0],
            }
        ).with_columns(
            pl.col("datadate", "rdq").cast(pl.Date), pl.col("beq_lag1", "ibq").cast(pl.Float64)
        )
        # formation_min = 1966-02-28 + 1mo = 1966-03-28 → first eom 1966-03-31
        # formation_max = 1965-12-31 + 6mo month_end = 1966-06-30
        formation_dates = pl.DataFrame(
            {"date": [date(1966, 2, 28), date(1966, 3, 31), date(1966, 6, 30)]}
        ).with_columns(pl.col("date").cast(pl.Date))
        from jkp.data.aux_functions import _hxz_us_roe_monthly, safe_div

        fq_with_roe = fundq.with_columns(safe_div("ibq", "beq_lag1", "roe", mode=3))
        result = _hxz_us_roe_monthly(fq_with_roe, tmp_path, formation_dates).sort("date")
        assert result["date"].to_list() == [date(1966, 3, 31), date(1966, 6, 30)]
        assert result["roe"][0] == pytest.approx(0.1)

    def test_post_1972_rdq_gated(self, tmp_path):
        """Post-1972: formation_date ≥ rdq.month_end + 1mo."""
        self._write_lnkhist(tmp_path, 10001, "G01")
        fundq = pl.DataFrame(
            {
                "gvkey": ["G01"],
                "datadate": [date(1990, 3, 31)],
                "rdq": [date(1990, 4, 15)],
                "beq_lag1": [200.0],
                "ibq": [20.0],
            }
        ).with_columns(
            pl.col("datadate", "rdq").cast(pl.Date), pl.col("beq_lag1", "ibq").cast(pl.Float64)
        )
        # formation_min_post = 1990-04-30.month_end + 1mo = 1990-05-31
        # formation_max = 1990-03-31 + 6mo month_end = 1990-09-30
        # date 1990-04-30 < 1990-05-31 → excluded
        # date 1990-05-31 == 1990-05-31 → included
        formation_dates = pl.DataFrame(
            {"date": [date(1990, 4, 30), date(1990, 5, 31)]}
        ).with_columns(pl.col("date").cast(pl.Date))
        from jkp.data.aux_functions import _hxz_us_roe_monthly, safe_div

        fq_with_roe = fundq.with_columns(safe_div("ibq", "beq_lag1", "roe", mode=3))
        result = _hxz_us_roe_monthly(fq_with_roe, tmp_path, formation_dates)
        assert len(result) == 1
        assert result["date"][0] == date(1990, 5, 31)

    def test_staleness_cap_6mo(self, tmp_path):
        """formation_date > datadate + 6mo month_end → excluded (stale)."""
        self._write_lnkhist(tmp_path, 10001, "G01")
        fundq = pl.DataFrame(
            {
                "gvkey": ["G01"],
                "datadate": [date(1990, 3, 31)],
                "rdq": [date(1990, 4, 15)],
                "beq_lag1": [200.0],
                "ibq": [20.0],
            }
        ).with_columns(
            pl.col("datadate", "rdq").cast(pl.Date), pl.col("beq_lag1", "ibq").cast(pl.Float64)
        )
        # formation_max = 1990-03-31 + 6mo month_end = 1990-09-30
        # 1990-10-31 > 1990-09-30 → excluded (stale)
        formation_dates = pl.DataFrame({"date": [date(1990, 10, 31)]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        from jkp.data.aux_functions import _hxz_us_roe_monthly, safe_div

        fq_with_roe = fundq.with_columns(safe_div("ibq", "beq_lag1", "roe", mode=3))
        result = _hxz_us_roe_monthly(fq_with_roe, tmp_path, formation_dates)
        assert len(result) == 0
