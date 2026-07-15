"""Unit tests for the daily-coverage patches (zero-return gate, mktrf join, Gao–Ritter)."""

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import (
    adj_trd_vol_NASDAQ,
    base_data_filter_exp,
    prepare_daily,
    zero_obs_gate_ok,
)
from jkp.data.paths import DataPaths


class TestZeroObsGate:
    def test_crsp_passes_even_with_many_zeros(self):
        df = pl.DataFrame({"source_crsp": [1], "zero_obs": [25]})
        assert df.select(zero_obs_gate_ok().alias("ok"))["ok"][0] is True

    def test_compustat_fails_when_zero_obs_ge_10(self):
        df = pl.DataFrame({"source_crsp": [0], "zero_obs": [10]})
        assert df.select(zero_obs_gate_ok().alias("ok"))["ok"][0] is False

    def test_compustat_passes_when_zero_obs_lt_10(self):
        df = pl.DataFrame({"source_crsp": [0], "zero_obs": [9]})
        assert df.select(zero_obs_gate_ok().alias("ok"))["ok"][0] is True


class TestBaseDataFilter:
    def test_rvol_keeps_crsp_high_zero_obs(self):
        df = pl.DataFrame(
            {
                "ret_exc": [0.01],
                "mktrf": [0.001],
                "source_crsp": [1],
                "zero_obs": [20],
            }
        )
        out = df.filter(base_data_filter_exp("rvol"))
        assert out.height == 1

    def test_rvol_drops_compustat_high_zero_obs(self):
        df = pl.DataFrame(
            {
                "ret_exc": [0.01],
                "mktrf": [0.001],
                "source_crsp": [0],
                "zero_obs": [20],
            }
        )
        out = df.filter(base_data_filter_exp("rvol"))
        assert out.height == 0

    def test_return_stat_keeps_row_regardless_of_mktrf(self):
        # mktrf-null rows are dropped upstream in prepare_daily, so base_data_filter_exp
        # no longer re-checks mktrf for any stat.
        df = pl.DataFrame(
            {
                "ret_exc": [0.01, 0.02],
                "mktrf": [0.001, None],
                "source_crsp": [1, 1],
                "zero_obs": [0, 0],
            }
        )
        assert df.filter(base_data_filter_exp("capm")).height == 2
        assert df.filter(base_data_filter_exp("rvol")).height == 2


class TestGaoRitterBoundary:
    def test_2003_12_31_is_adjusted(self):
        df = pl.DataFrame(
            {
                "date": [date(2003, 12, 31)],
                "vol": [1600.0],
                "is_nasdaq": [True],
            }
        ).with_columns(
            adj_trd_vol_NASDAQ("date", "vol", pl.col("is_nasdaq")),
        )
        assert df["vol"][0] == pytest.approx(1000.0)

    def test_2004_01_01_unchanged(self):
        df = pl.DataFrame(
            {
                "date": [date(2004, 1, 1)],
                "vol": [1600.0],
                "is_nasdaq": [True],
            }
        ).with_columns(
            adj_trd_vol_NASDAQ("date", "vol", pl.col("is_nasdaq")),
        )
        assert df["vol"][0] == pytest.approx(1600.0)


class TestPrepareDailyCoverage:
    def _write_inputs(self, test_paths: DataPaths, rows: list[dict], fcts_rows: list[dict]):
        dsf_path = test_paths.interim_dir / "dsf.parquet"
        fcts_path = test_paths.interim_dir / "fcts.parquet"
        pl.DataFrame(rows).write_parquet(dsf_path)
        pl.DataFrame(fcts_rows).write_parquet(fcts_path)
        return dsf_path, fcts_path

    def test_drops_stock_day_when_mktrf_null(self, test_paths: DataPaths):
        rows = [
            {
                "excntry": "USA",
                "id": "A",
                "date": date(2020, 1, 2),
                "eom": date(2020, 1, 31),
                "prc": 10.0,
                "adjfct": 1.0,
                "ret": 0.01,
                "ret_exc": 0.01,
                "dolvol": 1e6,
                "shares": 100.0,
                "tvol": 50.0,
                "ret_lag_dif": 1,
                "ret_local": 0.01,
                "source_crsp": 1,
            }
        ]
        # Factor file deliberately missing that date, so the stock-day has no mktrf.
        fcts_rows = [
            {
                "excntry": "USA",
                "date": date(2020, 1, 3),
                "mktrf": 0.001,
                "hml": 0.0,
                "smb_ff": 0.0,
                "inv": 0.0,
                "roe": 0.0,
                "smb_hxz": 0.0,
            }
        ]
        dsf_path, fcts_path = self._write_inputs(test_paths, rows, fcts_rows)
        prepare_daily(test_paths, dsf_path, fcts_path)
        dsf1 = pl.read_parquet(test_paths.interim_dir / "dsf1.parquet")
        assert dsf1.height == 0

    def test_crsp_zero_month_kept_in_corr_data(self, test_paths: DataPaths):
        dates = [date(2020, 1, d) for d in range(2, 12)]  # 10 days, all zeros
        rows = [
            {
                "excntry": "USA",
                "id": "CRSP",
                "date": d,
                "eom": date(2020, 1, 31),
                "prc": 10.0,
                "adjfct": 1.0,
                "ret": 0.0,
                "ret_exc": 0.0,
                "dolvol": 0.0,
                "shares": 100.0,
                "tvol": 0.0,
                "ret_lag_dif": 1,
                "ret_local": 0.0,
                "source_crsp": 1,
            }
            for d in dates
        ]
        fcts_rows = [
            {
                "excntry": "USA",
                "date": d,
                "mktrf": 0.001,
                "hml": 0.0,
                "smb_ff": 0.0,
                "inv": 0.0,
                "roe": 0.0,
                "smb_hxz": 0.0,
            }
            for d in dates
        ]
        dsf_path, fcts_path = self._write_inputs(test_paths, rows, fcts_rows)
        prepare_daily(test_paths, dsf_path, fcts_path)
        corr = pl.read_parquet(test_paths.interim_dir / "corr_data.parquet")
        # 10 days → 8 non-null 3-day sums for a contiguous series.
        assert corr.height == 8
