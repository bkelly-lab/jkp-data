"""Unit tests for the daily-coverage patches (zero-return gate, mktrf join, Gao–Ritter)."""

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import (
    adj_trd_vol_NASDAQ,
    base_data_filter_exp,
    zero_obs_gate_ok,
)


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
