"""
Unit tests for HXZ q-factor builder helpers in aux_functions.py.

Covers Stage 1 (_hxz_add_me_lag1_ym, hxz_attach_siccd), Stage 2
(hxz_impute_be_clean_surplus, hxz_supplement_q4_shares, hxz_merge_be_chain,
hxz_link_compustat), the characteristic tables (hxz_annual_characteristics,
hxz_roe_by_month), Stage 4 (_hxz_nyse_breaks, _hxz_country_breaks,
_hxz_classify_size_ia, _hxz_classify_roe, hxz_classify_portfolios), Stage 5
(hxz_compute_factors), and the two-calendar quarter dedup (one_row_per_date).

Does NOT call gen_hxz_data (covered by golden test) and does NOT require
WRDS I/O for helpers that only need in-memory DataFrames.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data import aux_functions as af
from jkp.data.aux_functions import (
    _hxz_add_me_lag1_ym,
    _hxz_classify_roe,
    _hxz_classify_size_ia,
    country_breakpoints,
    hxz_annual_characteristics,
    hxz_attach_siccd,
    hxz_classify_portfolios,
    hxz_compute_factors,
    hxz_impute_be_clean_surplus,
    hxz_load_crsp,
    hxz_load_fundq,
    hxz_load_panel_row,
    hxz_load_returns,
    hxz_merge_be_chain,
    hxz_roe_by_month,
    one_row_per_date,
)
from jkp.data.config import (
    FACTOR_MIN_STOCKS_BP,
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
        min_stocks=FACTOR_MIN_STOCKS_BP,
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

    def _write_sec_hist(self, test_paths, rows: list[dict]) -> None:
        sec = pl.DataFrame(rows).with_columns(
            pl.col("secinfostartdt", "secinfoenddt").cast(pl.Date)
        )
        p = test_paths.raw_tables_dir / "crsp_stksecurityinfohist.parquet"
        sec.write_parquet(p)

    def test_basic_join(self, test_paths):
        """SIC is attached for matching date range."""
        self._write_sec_hist(
            test_paths,
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
        result = hxz_attach_siccd(test_paths, df, "obs_date").collect()
        assert result["siccd"][0] == 3310
        assert result["is_fin"][0] is False

    def test_is_fin_flagged_for_sic_6000_to_6999(self, test_paths):
        """is_fin=True for siccd in [6000, 6999]."""
        self._write_sec_hist(
            test_paths,
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
        result = hxz_attach_siccd(test_paths, df, "obs_date").collect()
        assert result["is_fin"][0] is True

    def test_is_fin_boundary_6000(self, test_paths):
        """is_fin=True at the lower boundary siccd=6000."""
        self._write_sec_hist(
            test_paths,
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
        result = hxz_attach_siccd(test_paths, df, "obs_date").collect()
        assert result["is_fin"][0] is True

    def test_is_fin_boundary_6999(self, test_paths):
        """is_fin=True at the upper boundary siccd=6999."""
        self._write_sec_hist(
            test_paths,
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
        result = hxz_attach_siccd(test_paths, df, "obs_date").collect()
        assert result["is_fin"][0] is True

    def test_date_outside_window_keeps_row_with_null_siccd(self, test_paths):
        """When obs_date is before secinfostartdt (no covering window), the
        panel row is kept with null siccd/is_fin rather than being dropped."""
        self._write_sec_hist(
            test_paths,
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
        result = hxz_attach_siccd(test_paths, df, "obs_date").collect()
        assert len(result) == 1
        assert result["siccd"][0] is None
        assert result["is_fin"][0] is None

    def test_overlapping_windows_picks_latest_start(self, test_paths):
        """Tie-break: most recent secinfostartdt wins when two windows overlap."""
        self._write_sec_hist(
            test_paths,
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
        result = hxz_attach_siccd(test_paths, df, "obs_date").collect()
        # 2015-06-01 > 2000-01-01, so siccd=6200 should win
        assert result["siccd"][0] == 6200

    def test_drops_secinfostartdt_secinfoenddt_columns(self, test_paths):
        """Output should not contain secinfostartdt or secinfoenddt."""
        self._write_sec_hist(
            test_paths,
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
        result = hxz_attach_siccd(test_paths, df, "obs_date").collect()
        assert "secinfostartdt" not in result.columns
        assert "secinfoenddt" not in result.columns


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------


class TestHxzLoadCrsp:
    """Tests for hxz_load_crsp: the HXZ month-end price rule."""

    def test_me_missing_when_price_not_from_last_trading_day(self, test_paths):
        eom = date(2020, 6, 30)
        n = 2
        pl.DataFrame(
            {
                "permno": [1, 2],
                "permco": [1, 2],
                "mthcaldt": [eom, eom],
                "mthprcdt": [eom, date(2020, 6, 26)],
                "mthret": [0.01, 0.02],
                "mthretx": [0.01, 0.02],
                "mthprc": [10.0, 20.0],
                "shrout": [1000.0, 1000.0],
                "primaryexch": ["N"] * n,
                "sharetype": ["NS"] * n,
                "securitytype": ["EQTY"] * n,
                "securitysubtype": ["COM"] * n,
                "usincflg": ["Y"] * n,
                "issuertype": ["CORP"] * n,
            }
        ).write_parquet(test_paths.raw_tables_dir / "crsp_msf_v2.parquet")
        pl.DataFrame(
            {
                "permno": [1, 2],
                "secinfostartdt": [date(2000, 1, 1)] * n,
                "secinfoenddt": [None] * n,
                "siccd": [3000, 3000],
            },
            schema_overrides={"secinfoenddt": pl.Date},
        ).write_parquet(test_paths.raw_tables_dir / "crsp_stksecurityinfohist.parquet")

        out = hxz_load_crsp(test_paths, "monthly").sort("permno")

        # Price from June 26 (not the last trading day): ME missing, return kept.
        assert out["me"].to_list() == [10.0, None]
        assert out["ret"].to_list() == [0.01, 0.02]


class TestHxzLoadCrspPrices:
    """hxz_load_crsp: D4 month-end price rule on ME, and the industry columns by frequency."""

    _UNIVERSE = {
        "sharetype": "NS",
        "securitytype": "EQTY",
        "securitysubtype": "COM",
        "usincflg": "Y",
        "issuertype": "CORP",
        "primaryexch": "N",
    }

    def test_monthly_me_is_abs_prc_times_shrout_only_on_price_date(self, test_paths):
        """ME = |prc| * shrout / 1000 when mthprcdt == mthcaldt; null when mthprcdt is null
        or differs. (The mismatch case alone is in TestHxzLoadCrsp.)"""
        eom = date(2020, 6, 30)
        n = 4
        pl.DataFrame(
            {
                "permno": [1, 2, 3, 4],
                "permco": [1, 2, 3, 4],
                "mthcaldt": [eom] * n,
                "mthprcdt": [eom, None, eom, date(2020, 6, 29)],
                "mthret": [0.01] * n,
                "mthretx": [0.01] * n,
                "mthprc": [10.0, 20.0, -8.0, 30.0],
                "shrout": [1000.0, 1000.0, 500.0, 1000.0],
                **{k: [v] * n for k, v in self._UNIVERSE.items()},
            },
            schema_overrides={"mthprcdt": pl.Date},
        ).write_parquet(test_paths.raw_tables_dir / "crsp_msf_v2.parquet")
        pl.DataFrame(
            {
                "permno": [1, 2, 3, 4],
                "secinfostartdt": [date(2000, 1, 1)] * n,
                "secinfoenddt": [None] * n,
                "siccd": [3000] * n,
            },
            schema_overrides={"secinfoenddt": pl.Date},
        ).write_parquet(test_paths.raw_tables_dir / "crsp_stksecurityinfohist.parquet")

        out = hxz_load_crsp(test_paths, "monthly").sort("permno")

        assert out["me"].to_list() == [10.0, None, 4.0, None]  # |-8| * 500 / 1000
        assert {"siccd", "is_fin"} <= set(out.columns)

    def test_daily_has_me_without_industry_columns(self, test_paths):
        """Daily ME has no price-date rule, and no stksecurityinfohist is read (none written)."""
        n = 2
        pl.DataFrame(
            {
                "permno": [1, 1],
                "permco": [1, 1],
                "dlycaldt": [date(2020, 6, 29), date(2020, 6, 30)],
                "dlyret": [0.01, 0.02],
                "dlyretx": [0.01, 0.02],
                "dlyprc": [10.0, -12.0],
                "shrout": [1000.0, 1000.0],
                **{k: [v] * n for k, v in self._UNIVERSE.items()},
            }
        ).write_parquet(test_paths.raw_tables_dir / "crsp_dsf_v2.parquet")

        out = hxz_load_crsp(test_paths, "daily").sort("date")

        assert out.columns == [
            "permno", "permco", "date", "ret", "retx", "prc", "shrout", "me", "exchcd",
        ]  # fmt: skip
        assert out["me"].to_list() == [10.0, 12.0]


class TestHxzLoadFundq:
    """hxz_load_fundq: NA screen, one row per (gvkey, fyearq, fqtr), sorted output."""

    _FLOATS = [
        "ibq", "dvpsxq", "cshoq", "ajexq",
        "seqq", "ceqq", "pstkq", "atq", "ltq", "txditcq", "pstkrq",
    ]  # fmt: skip

    @classmethod
    def _write(cls, test_paths, rows: list[dict]) -> None:
        base = {
            "gvkey": "g1",
            "fyr": 12,
            "fyearq": 2020,
            "fqtr": 1,
            "datadate": date(2020, 3, 31),
            "rdq": date(2020, 5, 1),
            "indfmt": "INDL",
            "datafmt": "STD",
            "popsrc": "D",
            "consol": "C",
            **dict.fromkeys(cls._FLOATS),
            "seqq": 10.0,
            "ibq": 1.0,
        }
        schema = {
            "gvkey": pl.Utf8,
            "fyr": pl.Int64,
            "fyearq": pl.Int64,
            "fqtr": pl.Int64,
            "datadate": pl.Date,
            "rdq": pl.Date,
            "indfmt": pl.Utf8,
            "datafmt": pl.Utf8,
            "popsrc": pl.Utf8,
            "consol": pl.Utf8,
            **dict.fromkeys(cls._FLOATS, pl.Float64),
        }
        pl.DataFrame([base | r for r in rows], schema=schema).write_parquet(
            test_paths.raw_tables_dir / "comp_fundq.parquet"
        )

    def test_dedup_screens_and_sort(self, test_paths):
        self._write(
            test_paths,
            [
                # 0-1: same (gvkey, fyearq, fqtr) under two datadate: the latest datadate wins
                # even though its rdq is earlier.
                {"gvkey": "g1", "ibq": 1.0},
                {"gvkey": "g1", "datadate": date(2020, 4, 30), "rdq": date(2020, 4, 1), "ibq": 2.0},
                # 2-3: same datadate: the latest rdq wins.
                {"gvkey": "g2", "datadate": date(2020, 6, 30), "rdq": date(2020, 8, 1), "ibq": 3.0},
                {"gvkey": "g2", "datadate": date(2020, 6, 30), "rdq": date(2020, 9, 1), "ibq": 4.0},
                # 4: no fiscal quarter. 5-8: fail one NA screen each.
                {"gvkey": "g3", "fqtr": None},
                {"gvkey": "g4", "indfmt": "FS"},
                {"gvkey": "g5", "datafmt": "SUMM_STD"},
                {"gvkey": "g6", "popsrc": "I"},
                {"gvkey": "g7", "consol": "D"},
                # 9: sorts before g1 although written last.
                {"gvkey": "g0", "fyearq": 2019, "fqtr": 4, "ibq": 5.0},
            ],
        )

        out = hxz_load_fundq(test_paths)

        assert out.columns == [
            "gvkey", "datadate", "fyr", "fyearq", "fqtr", "rdq",
            "ibq", "dvpsxq", "cshoq", "ajexq", "beq", "n",
        ]  # fmt: skip
        assert out.select("gvkey", "fyearq", "fqtr", "ibq", "n").rows() == [
            ("g0", 2019, 4, 5.0, 9),
            ("g1", 2020, 1, 2.0, 1),
            ("g2", 2020, 1, 4.0, 3),
        ]
        assert out["fyr"].to_list() == [12, 12, 12]
        assert out["beq"].to_list() == [10.0, 10.0, 10.0]  # seqq


class TestHxzLoadPanelRow:
    """hxz_load_panel_row: gvkey screen, (id, eom) industry dedup and is_fin (monthly only)."""

    _SCHEMA = {
        "excntry": pl.Utf8,
        "id": pl.Int64,
        "gvkey": pl.Utf8,
        "eom": pl.Date,
        "ret": pl.Float64,
        "me": pl.Float64,
        "size_grp": pl.Utf8,
        "common": pl.Int64,
        "obs_main": pl.Int64,
        "primary_sec": pl.Int64,
        "exch_main": pl.Int64,
        "ret_lag_dif": pl.Int64,
        "naics": pl.Utf8,
        "sic": pl.Int64,
    }

    @classmethod
    def _write_msf(cls, test_paths, rows: list[dict]) -> None:
        """world_msf rows that pass the ROW universe unless overridden."""
        base = {
            "excntry": "GBR",
            "id": 1,
            "gvkey": "g1",
            "eom": date(2020, 6, 30),
            "ret": 0.01,
            "me": 10.0,
            "size_grp": "small",
            "common": 1,
            "obs_main": 1,
            "primary_sec": 1,
            "exch_main": 1,
            "ret_lag_dif": 1,
            "naics": None,
            "sic": None,
        }
        pl.DataFrame([base | r for r in rows], schema=cls._SCHEMA).write_parquet(
            test_paths.interim_dir / "world_msf.parquet"
        )

    def test_monthly_is_fin_rule_and_null_gvkey_dropped(self, test_paths):
        self._write_msf(
            test_paths,
            [
                {"id": 1, "gvkey": "g1", "naics": "522110"},
                {"id": 2, "gvkey": "g2", "naics": "531210"},
                {"id": 3, "gvkey": "g3", "sic": 6500},  # naics null: SIC fallback
                {"id": 4, "gvkey": "g4"},  # no industry at all
                {"id": 5, "gvkey": "g5", "naics": "331110", "sic": 6500},  # NAICS wins
                {"id": 6, "gvkey": None, "naics": "522110"},
            ],
        )

        out = hxz_load_panel_row(test_paths, "monthly").sort("id")

        assert out.columns == [
            "excntry", "id", "gvkey", "date", "ret", "me", "size_grp", "naics", "sic", "is_fin",
        ]  # fmt: skip
        assert out["id"].to_list() == [1, 2, 3, 4, 5]
        assert out["is_fin"].to_list() == [True, True, True, False, False]

    def test_monthly_duplicate_industry_rows_do_not_multiply_panel_rows(self, test_paths):
        # Each id has a kept row plus a twin outside the universe (obs_main 0) that carries
        # the winning industry: non-null naics, then larger naics, then larger sic.
        self._write_msf(
            test_paths,
            [
                {"id": 11, "sic": 2000},
                {"id": 11, "obs_main": 0, "naics": "52", "sic": 100},
                {"id": 12, "naics": "33", "sic": 1000},
                {"id": 12, "obs_main": 0, "naics": "52", "sic": 5},
                {"id": 13, "sic": 2000},
                {"id": 13, "obs_main": 0, "sic": 6500},
            ],
        )

        out = hxz_load_panel_row(test_paths, "monthly").sort("id")

        assert out["id"].to_list() == [11, 12, 13]
        assert out["naics"].to_list() == ["52", "52", None]
        assert out["sic"].to_list() == [100, 5, 6500]
        assert out["is_fin"].to_list() == [True, True, True]

    def test_daily_has_no_industry_columns_and_drops_null_gvkey(self, test_paths):
        # Daily gvkey and size_grp come from the month's world_msf row.
        self._write_msf(
            test_paths,
            [
                {"id": 1, "gvkey": "g1", "naics": "52", "sic": 6000},
                {"id": 2, "gvkey": None, "naics": "52", "sic": 6000},
            ],
        )
        dates = [date(2020, 6, 29), date(2020, 6, 30)]
        pl.DataFrame(
            {
                "excntry": ["GBR"] * 4,
                "id": [1, 1, 2, 2],
                "eom": [date(2020, 6, 30)] * 4,
                "date": dates * 2,
                "ret": [0.01] * 4,
                "me": [10.0] * 4,
                "ret_lag_dif": [1] * 4,
                "common": [1] * 4,
                "obs_main": [1] * 4,
                "primary_sec": [1] * 4,
                "exch_main": [1] * 4,
            }
        ).write_parquet(test_paths.interim_dir / "world_dsf.parquet")

        out = hxz_load_panel_row(test_paths, "daily").sort("date")

        assert out.columns == ["excntry", "id", "gvkey", "date", "ret", "me", "size_grp"]
        assert out["id"].to_list() == [1, 1]
        assert out["date"].to_list() == dates


class TestHxzLoadReturns:
    """hxz_load_returns: US + ROW concatenation, me_lag1 and date_start. The two loaders are
    replaced by small frames."""

    @staticmethod
    def _patch(monkeypatch, us: pl.DataFrame, row: pl.DataFrame) -> None:
        def fake_crsp(paths, freq, date_start=None):
            return us if date_start is None else us.filter(pl.col("date") >= date_start)

        monkeypatch.setattr(af, "hxz_load_crsp", fake_crsp)
        monkeypatch.setattr(af, "hxz_load_panel_row", lambda paths, freq: row)

    @staticmethod
    def _monthly_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
        us = pl.DataFrame(
            {
                "permno": [1, 1, 1, 2],
                "date": [
                    date(2020, 1, 31),
                    date(2020, 2, 29),
                    date(2020, 4, 30),
                    date(2020, 1, 31),
                ],
                "ret": [0.01, 0.02, 0.03, 0.04],
                "me": [10.0, 20.0, 40.0, 7.0],
            }
        )
        row = pl.DataFrame(
            {
                "excntry": ["GBR", "GBR"],
                "id": [10, 10],
                "gvkey": ["g10", "g10"],
                "date": [date(2020, 1, 31), date(2020, 2, 29)],
                "ret": [0.05, 0.06],
                "me": [5.0, 6.0],
            }
        )
        return us, row

    def test_monthly_us_keys_and_me_lag1_by_calendar_month(self, test_paths, monkeypatch):
        self._patch(monkeypatch, *self._monthly_frames())

        out = hxz_load_returns(test_paths, "monthly").sort("excntry", "id", "date")

        us = out.filter(pl.col("excntry") == "USA")
        assert (us["id"] == us["permno"]).all()
        assert us["id"].to_list() == [1, 1, 1, 2]
        # Feb sees January's ME; April has no March row; first rows have no lag.
        assert us["me_lag1"].to_list() == [None, 10.0, None, None]
        gbr = out.filter(pl.col("excntry") == "GBR")
        assert gbr["id"].to_list() == [10, 10]
        assert gbr["me_lag1"].to_list() == [None, 5.0]

    def test_date_start_filters_us_and_row(self, test_paths, monkeypatch):
        self._patch(monkeypatch, *self._monthly_frames())

        out = hxz_load_returns(test_paths, "monthly", date_start=date(2020, 2, 1))

        assert out.sort("excntry", "id", "date").select("excntry", "id", "date").rows() == [
            ("GBR", 10, date(2020, 2, 29)),
            ("USA", 1, date(2020, 2, 29)),
            ("USA", 1, date(2020, 4, 30)),
        ]

    def test_daily_me_lag1_is_previous_row(self, test_paths, monkeypatch):
        days = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
        us = pl.DataFrame(
            {
                "permno": [1, 1, 1, 2],
                "date": [*days, days[0]],
                "ret": [0.01] * 4,
                "me": [10.0, 11.0, 12.0, 7.0],
            }
        )
        row = pl.DataFrame(
            {
                "excntry": ["GBR"] * 2,
                "id": [10, 10],
                "gvkey": ["g10"] * 2,
                "date": days[:2],
                "ret": [0.01] * 2,
                "me": [5.0, 6.0],
            }
        )
        self._patch(monkeypatch, us, row)

        out = hxz_load_returns(test_paths, "daily").sort("excntry", "id", "date")

        assert out.filter(pl.col("id") == 1)["me_lag1"].to_list() == [None, 10.0, 11.0]
        assert out.filter(pl.col("id") == 2)["me_lag1"].to_list() == [None]  # not 1's last ME
        assert out.filter(pl.col("id") == 10)["me_lag1"].to_list() == [None, 5.0]


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

    def _write_lnkhist(self, test_paths, rows: list[dict]) -> None:
        df = pl.DataFrame(rows).with_columns(
            pl.col("linkdt", "linkenddt").cast(pl.Date),
            pl.col("lpermno").cast(pl.Int64),
        )
        p = test_paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet"
        df.write_parquet(p)

    def test_basic_link(self, test_paths):
        """Single matching link returns permno."""
        self._write_lnkhist(
            test_paths,
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

        result = hxz_link_compustat(test_paths, comp)
        assert result["permno"][0] == 10001

    def test_date_outside_link_window_excluded(self, test_paths):
        """datadate before linkdt → row excluded (no match)."""
        self._write_lnkhist(
            test_paths,
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

        result = hxz_link_compustat(test_paths, comp)
        assert len(result) == 0

    def test_null_linkenddt_treated_as_open_ended(self, test_paths):
        """linkenddt=null means the link is still active."""
        self._write_lnkhist(
            test_paths,
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

        result = hxz_link_compustat(test_paths, comp)
        assert len(result) == 1
        assert result["permno"][0] == 10001

    def test_multi_link_primary_wins(self, test_paths):
        """When P and C links compete for same (datadate, permno), P wins."""
        self._write_lnkhist(
            test_paths,
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

        result = hxz_link_compustat(test_paths, comp)
        # unique(subset=["datadate","permno"]) → only 1 row per (datadate,permno)
        assert len(result) == 1
        # linkprim DESC: P > C → gvkey 002 wins
        assert result["linkprim"][0] == "P"

    def test_drops_linktype_and_link_days(self, test_paths):
        """Output should not contain linktype or _link_days columns."""
        self._write_lnkhist(
            test_paths,
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

        result = hxz_link_compustat(test_paths, comp)
        assert "linktype" not in result.columns
        assert "_link_days" not in result.columns


class TestHxzSupplementQ4Shares:
    """Tests for hxz_supplement_q4_shares: share fallback hierarchy."""

    def test_q4_null_cshoq_filled_from_annual(self, test_paths):
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
        lnk.write_parquet(test_paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet")

        msf = pl.DataFrame(
            {
                "permno": [10001],
                "mthcaldt": [date(2020, 12, 31)],
                "shrout": [1000.0],
                "mthcumfacshr": [2.0],
            }
        ).with_columns(pl.col("mthcaldt").cast(pl.Date))
        msf.write_parquet(test_paths.raw_tables_dir / "crsp_msf_v2.parquet")

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

        result = hxz_supplement_q4_shares(test_paths, fundq, funda)
        assert result["cshoq_eff"][0] == pytest.approx(50.0)  # from funda csho
        assert result["ajexq_eff"][0] == pytest.approx(1.0)  # from funda ajex

    def test_existing_cshoq_not_overwritten(self, test_paths):
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
        lnk.write_parquet(test_paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet")

        msf = pl.DataFrame(
            {
                "permno": [10001],
                "mthcaldt": [date(2020, 12, 31)],
                "shrout": [999.0],
                "mthcumfacshr": [9.0],
            }
        ).with_columns(pl.col("mthcaldt").cast(pl.Date))
        msf.write_parquet(test_paths.raw_tables_dir / "crsp_msf_v2.parquet")

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

        result = hxz_supplement_q4_shares(test_paths, fundq, funda)
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
    """Tests for _hxz_country_breaks: per-(excntry,date) with FACTOR_MIN_STOCKS_BP gate."""

    def _make_panel(self, rows):
        return pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("me", "inv").cast(pl.Float64),
        )

    def test_below_min_stocks_excluded(self):
        """excntry/date with fewer than FACTOR_MIN_STOCKS_BP eligible stocks → no row in output."""
        # Only 5 stocks (below FACTOR_MIN_STOCKS_BP=10)
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
        """excntry/date with at least FACTOR_MIN_STOCKS_BP eligible stocks → breakpoints computed."""
        rows = [
            {
                "excntry": "GBR",
                "date": date(2020, 6, 30),
                "elig": True,
                "size_grp": "large",
                "me": float(i),
                "inv": 0.1,
            }
            for i in range(1, FACTOR_MIN_STOCKS_BP + 2)  # just above threshold
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
            for i in range(1, FACTOR_MIN_STOCKS_BP + 1)
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
        # exactly FACTOR_MIN_STOCKS_BP eligible → included
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
        """ROW with fewer than FACTOR_MIN_STOCKS_BP stocks → sizeport/invport null."""
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
        """ROW with fewer than FACTOR_MIN_STOCKS_BP eligible stocks → null roeport."""
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
        """US country bypasses FACTOR_MIN_STOCKS_PF gate even for single-stock buckets."""
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
        """ROW bucket with n < FACTOR_MIN_STOCKS_PF gets null vwret → null factor legs."""
        date_ = date(2020, 1, 31)
        rows = [
            # Only 1 stock in each small bucket → n=1 < FACTOR_MIN_STOCKS_PF=3 → null
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
        # With n=1 < FACTOR_MIN_STOCKS_PF, small bucket is null, me_hxz = null - something
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
# Stage 4 combined — hxz_classify_portfolios
# ---------------------------------------------------------------------------


class TestHxzClassifyPortfolios:
    """hxz_classify_portfolios: June size and I/A sorts from the annual table,
    monthly Roe sorts from the Roe table, both attached to the monthly panel."""

    DATES = [date(2020, 6, 30), date(2020, 7, 31), date(2020, 12, 31), date(2021, 6, 30)]

    def _panel(self, is_fin: bool = False) -> pl.DataFrame:
        n = len(self.DATES)
        return pl.DataFrame(
            {
                "excntry": ["USA"] * n,
                "id": [1] * n,
                "date": self.DATES,
                "ret": [0.01] * n,
                "me": [100.0] * n,
                "me_lag1": [90.0] * n,
                "is_fin": [is_fin] * n,
                "exchcd": [1] * n,
                "size_grp": [None] * n,
            },
            schema_overrides={"size_grp": pl.Utf8},
        )

    def _annual(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"id": [1, 1], "port_year": [2020, 2021], "inv": [0.1, 0.2], "be_a": [200.0, 210.0]},
            schema_overrides={"port_year": pl.Int32},
        )

    def _roe(self) -> pl.DataFrame:
        yms = [2020 * 12 + 6, 2020 * 12 + 7, 2020 * 12 + 12, 2021 * 12 + 6]
        return pl.DataFrame({"id": [1] * 4, "ym": yms, "roe": [0.05, 0.06, 0.07, 0.08]})

    def _classify(self, is_fin: bool = False) -> pl.DataFrame:
        return hxz_classify_portfolios(self._panel(is_fin), self._annual(), self._roe()).sort(
            "date"
        )

    def test_june_sort_held_july_to_june(self):
        """The June 2020 sort is held from July 2020 to June 2021."""
        out = self._classify()
        assert out["sizeport"].to_list() == [None, 1, 1, 1]
        assert out["inv"].to_list() == [None, 0.1, 0.1, 0.1]

    def test_roe_attached_by_month(self):
        out = self._classify()
        assert out["roe"].to_list() == [0.05, 0.06, 0.07, 0.08]
        assert out["roeport"].to_list() == [1, 1, 1, 1]

    def test_screen_changes_membership_not_values(self):
        """A financial firm enters no sort; its I/A and Roe are still the table values."""
        out = self._classify(is_fin=True)
        for col in ("sizeport", "invport", "roeport"):
            assert out[col].null_count() == out.height
        assert out["inv"].to_list() == [None, 0.1, 0.1, 0.1]
        assert out["roe"].to_list() == [0.05, 0.06, 0.07, 0.08]


# ---------------------------------------------------------------------------
# Characteristic tables — hxz_annual_characteristics, hxz_roe_by_month
# ---------------------------------------------------------------------------


class TestHxzCharacteristicTables:
    """The I/A and Roe tables cover every security: a GBR Compustat security
    (id 10), a US Compustat security without a CCM link (id 200001), a US CRSP
    permno whose world_msf gvkey differs from its CCM link (id 30), and a CRSP
    permno with 1960s quarters (id 40)."""

    def _inputs(self, test_paths) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        eoms = pl.date_range(date(2019, 6, 1), date(2020, 6, 1), "1mo", eager=True).dt.month_end()
        n = len(eoms)
        pl.DataFrame(
            {
                "excntry": ["GBR"] * n + ["USA"] * 2 * n,
                "id": [10] * n + [200001] * n + [30] * n,
                "gvkey": ["g10"] * n + ["g20"] * 2 * n,
                "source_crsp": [0] * 2 * n + [1] * n,
                "eom": pl.concat([eoms, eoms, eoms]),
            }
        ).write_parquet(test_paths.interim_dir / "world_msf.parquet")
        pl.DataFrame(
            {
                "gvkey": ["g30", "g40"],
                "lpermno": [30, 40],
                "linkprim": ["P", "P"],
                "linktype": ["LC", "LC"],
                "linkdt": [date(2000, 1, 1), date(1960, 1, 1)],
                "linkenddt": [None, None],
            },
            schema_overrides={"linkenddt": pl.Date},
        ).write_parquet(test_paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet")
        funda_us = pl.DataFrame(
            {
                "gvkey": ["g20", "g30"],
                "datadate": [date(2018, 12, 31), date(2018, 12, 31)],
                "inv": [0.05, 0.07],
                "be_a": [1.0, 1.0],
            }
        )
        fundq_us = pl.DataFrame(
            {
                "gvkey": ["g20", "g20", "g30", "g40", "g40"],
                "datadate": [
                    date(2019, 9, 30),
                    date(2019, 12, 31),  # rdq <= datadate: dropped
                    date(2019, 9, 30),
                    date(1965, 12, 31),
                    date(1966, 3, 31),  # no rdq: never usable
                ],
                "rdq": [
                    date(2019, 10, 25),
                    date(2019, 12, 31),
                    date(2019, 10, 25),
                    date(1966, 2, 15),
                    None,
                ],
                "ibq": [1.0, 5.0, 2.0, 10.0, 50.0],
                "beq_lag1": [10.0, 10.0, 10.0, 100.0, 100.0],
            }
        )
        funda_row = pl.DataFrame(
            {
                "gvkey": ["g10", "g10"],
                "year": [2018, 2019],
                "datadate": [date(2018, 12, 31), date(2019, 12, 31)],
                "inv": [0.1, 0.4],
                "be_a": [1.0, 1.0],
                "roe_a": [0.2, 0.3],
            }
        )
        return funda_us, fundq_us, funda_row

    def test_annual_table(self, test_paths):
        """Fiscal year y-1 applies to port_year y; CRSP permnos use their CCM link."""
        funda_us, _, funda_row = self._inputs(test_paths)
        out = hxz_annual_characteristics(test_paths, funda_us, funda_row)
        assert out.columns == ["id", "port_year", "inv", "be_a"]
        assert out.rows() == [
            (10, 2019, 0.1, 1.0),
            (10, 2020, 0.4, 1.0),
            (30, 2019, 0.07, 1.0),
            (200001, 2019, 0.05, 1.0),
        ]

    def test_roe_table(self, test_paths):
        funda_us, fundq_us, funda_row = self._inputs(test_paths)
        roe = hxz_roe_by_month(test_paths, fundq_us, funda_row)
        assert roe.columns == ["id", "ym", "roe"]

        def at(i: int, y: int, m: int) -> list[float]:
            return roe.filter((pl.col("id") == i) & (pl.col("ym") == y * 12 + m))["roe"].to_list()

        # GBR: annual roe_a from 6 to 18 months after the fiscal year-end; latest year wins
        assert [at(10, 2019, 5), at(10, 2019, 6), at(10, 2020, 6)] == [[], [0.2], [0.3]]
        # US Compustat: month after rdq to six months after the quarter end
        assert [at(200001, 2019, m) for m in (10, 11)] == [[], [0.1]]
        assert [at(200001, 2020, m) for m in (3, 4)] == [[0.1], []]
        # US CRSP: the CCM-linked gvkey (g30), not the world_msf gvkey
        assert at(30, 2019, 11) == [0.2]
        # No sample start: a 1966 Roe exists; a quarter without rdq is never used
        assert [at(40, 1966, m) for m in (2, 3, 6, 7)] == [[], [0.1], [0.1], []]


# ---------------------------------------------------------------------------
# Two fiscal calendars on one quarter end — one_row_per_date after the BEQ chain
# ---------------------------------------------------------------------------


class TestHxzTwoCalendarQuarters:
    """A firm that changes its fiscal year-end reports some quarter ends under
    both calendars. The BEQ chain runs on both; one_row_per_date then keeps the
    established calendar's row, and the new calendar's next quarter still lags
    to the shared quarter end."""

    def test_established_calendar_kept_and_lag_intact(self):
        # FYE December, then June: 1999-09-30 and 1999-12-31 appear as FY1999
        # Q3/Q4 (fyr 12) and as FY2000 Q1/Q2 (fyr 6).
        fq = pl.DataFrame(
            [
                (date(1999, 3, 31), 12, 1999, 1, 100.0, 0),
                (date(1999, 6, 30), 12, 1999, 2, 101.0, 1),
                (date(1999, 9, 30), 12, 1999, 3, 102.0, 2),
                (date(1999, 12, 31), 12, 1999, 4, 103.0, 3),
                (date(1999, 9, 30), 6, 2000, 1, 102.5, 4),
                (date(1999, 12, 31), 6, 2000, 2, 103.5, 5),
                (date(2000, 3, 31), 6, 2000, 3, 104.0, 6),
            ],
            schema=["datadate", "fyr", "fyearq", "fqtr", "beq", "n"],
            orient="row",
        ).with_columns(
            gvkey=pl.lit("g1"),
            ibq=pl.lit(1.0),
            dvpsxq=pl.lit(0.0),
            cshoq_eff=pl.lit(1.0),
            ajexq_eff=pl.lit(1.0),
        )
        out = (
            hxz_impute_be_clean_surplus(fq)
            .lazy()
            .with_columns(source=pl.lit("NA"))
            .pipe(one_row_per_date)
            .collect()
            .sort("datadate")
        )
        assert out["datadate"].to_list() == [
            date(1999, 3, 31),
            date(1999, 6, 30),
            date(1999, 9, 30),
            date(1999, 12, 31),
            date(2000, 3, 31),
        ]
        assert out["fyr"].to_list() == [12, 12, 12, 12, 6]
        assert out["beq_lag1"].to_list() == [None, 100.0, 101.0, 102.0, 103.5]
