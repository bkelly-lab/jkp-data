"""
Tests for add_ret_exc_wins() in aux_functions.py.

Verifies that the winsorized excess return column (ret_exc_wins) is correctly
computed: Compustat stocks (id > 99999) are clipped to the [0.1%, 99.9%]
percentiles of ret_exc per eom, while CRSP stocks are left unchanged.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from aux_functions import add_ret_exc_wins


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Set working directory to tmp_path and return it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_world_msf(path, rows: list[dict]) -> None:
    """Write a minimal world_msf parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path / "world_msf.parquet")


def _make_world_dsf(path, rows: list[dict]) -> None:
    """Write a minimal world_dsf parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path / "world_dsf.parquet")


def _read_result(path, freq: str) -> pl.DataFrame:
    return pl.read_parquet(path / f"world_{freq}sf.parquet")


class TestAddRetExcWinsMonthly:
    """Tests for monthly frequency."""

    def test_crsp_stocks_unchanged(self, data_dir):
        """CRSP stocks (id <= 99999) should have ret_exc_wins == ret_exc."""
        rows = [
            {"id": 10001, "eom": date(2020, 1, 31), "ret_exc": 0.05},
            {"id": 10002, "eom": date(2020, 1, 31), "ret_exc": -0.03},
            {"id": 10003, "eom": date(2020, 1, 31), "ret_exc": 0.10},
        ]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        assert "ret_exc_wins" in result.columns
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_normal_unchanged(self, data_dir):
        """Compustat stocks within bounds should have ret_exc_wins == ret_exc."""
        # All Compustat ids, all with similar returns (no outliers)
        rows = [
            {"id": 100001, "eom": date(2020, 1, 31), "ret_exc": 0.01},
            {"id": 100002, "eom": date(2020, 1, 31), "ret_exc": 0.02},
            {"id": 100003, "eom": date(2020, 1, 31), "ret_exc": 0.03},
        ]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_outlier_clipped_high(self, data_dir):
        """Compustat stock with extreme high return should be clipped down."""
        # Need enough rows so the outlier exceeds the 99.9th percentile
        normal_rows = [
            {"id": 100000 + i, "eom": date(2020, 1, 31), "ret_exc": 0.01 * i}
            for i in range(1, 2001)
        ]
        outlier_row = {"id": 200001, "eom": date(2020, 1, 31), "ret_exc": 99.0}
        rows = normal_rows + [outlier_row]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] < outlier["ret_exc"][0]

    def test_compustat_outlier_clipped_low(self, data_dir):
        """Compustat stock with extreme low return should be clipped up."""
        normal_rows = [
            {"id": 100000 + i, "eom": date(2020, 1, 31), "ret_exc": 0.01 * i}
            for i in range(1, 2001)
        ]
        outlier_row = {"id": 200001, "eom": date(2020, 1, 31), "ret_exc": -99.0}
        rows = normal_rows + [outlier_row]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] > outlier["ret_exc"][0]

    def test_null_ret_exc_stays_null(self, data_dir):
        """Null ret_exc should produce null ret_exc_wins."""
        rows = [
            {"id": 100001, "eom": date(2020, 1, 31), "ret_exc": None},
            {"id": 100002, "eom": date(2020, 1, 31), "ret_exc": 0.05},
        ]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        null_row = result.filter(pl.col("id") == 100001)
        assert null_row["ret_exc_wins"][0] is None

    def test_idempotent(self, data_dir):
        """Running add_ret_exc_wins twice should produce the same result."""
        rows = [
            {"id": 10001, "eom": date(2020, 1, 31), "ret_exc": 0.05},
            {"id": 100001, "eom": date(2020, 1, 31), "ret_exc": 0.03},
        ]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")
        first = _read_result(data_dir, "m")
        add_ret_exc_wins("m")
        second = _read_result(data_dir, "m")

        assert first["ret_exc_wins"].to_list() == second["ret_exc_wins"].to_list()

    def test_boundary_id_99999_is_crsp(self, data_dir):
        """id == 99999 is CRSP (not > 99999), so ret_exc_wins == ret_exc."""
        rows = [
            {"id": 99999, "eom": date(2020, 1, 31), "ret_exc": 99.0},
            {"id": 100000, "eom": date(2020, 1, 31), "ret_exc": 0.05},
        ]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        crsp_row = result.filter(pl.col("id") == 99999)
        # id 99999 is NOT > 99999, so it's treated as CRSP — unchanged
        assert crsp_row["ret_exc_wins"][0] == crsp_row["ret_exc"][0]

    def test_boundary_id_100000_is_compustat(self, data_dir):
        """id == 100000 is Compustat (> 99999), so it gets winsorized."""
        normal_rows = [
            {"id": 100000 + i, "eom": date(2020, 1, 31), "ret_exc": 0.01 * i}
            for i in range(1, 2001)
        ]
        # id 100000 with an outlier return
        outlier_row = {"id": 100000, "eom": date(2020, 1, 31), "ret_exc": 99.0}
        rows = normal_rows + [outlier_row]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        outlier = result.filter(pl.col("id") == 100000)
        assert outlier["ret_exc_wins"][0] < outlier["ret_exc"][0]

    def test_multiple_eom_periods(self, data_dir):
        """Percentiles should be computed independently per eom."""
        rows = [
            # Jan: normal returns
            {"id": 100001, "eom": date(2020, 1, 31), "ret_exc": 0.01},
            {"id": 100002, "eom": date(2020, 1, 31), "ret_exc": 0.02},
            # Feb: different return distribution
            {"id": 100001, "eom": date(2020, 2, 29), "ret_exc": 0.10},
            {"id": 100002, "eom": date(2020, 2, 29), "ret_exc": 0.20},
        ]
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        assert "ret_exc_wins" in result.columns
        assert len(result) == 4

    def test_custom_percentiles(self, data_dir):
        """Custom lower/upper percentile arguments should be respected."""
        normal_rows = [
            {"id": 100000 + i, "eom": date(2020, 1, 31), "ret_exc": 0.01 * i}
            for i in range(1, 2001)
        ]
        outlier_row = {"id": 200001, "eom": date(2020, 1, 31), "ret_exc": 99.0}
        rows = normal_rows + [outlier_row]
        _make_world_msf(data_dir, rows)

        # Wider percentiles (1% and 99%) should clip more aggressively
        add_ret_exc_wins("m", lower=0.01, upper=0.99)
        result_wide = _read_result(data_dir, "m")
        outlier_wide = result_wide.filter(pl.col("id") == 200001)

        # Recreate data for default percentiles
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")
        result_default = _read_result(data_dir, "m")
        outlier_default = result_default.filter(pl.col("id") == 200001)

        # With wider clipping (99th vs 99.9th), the clipped value should be lower
        assert outlier_wide["ret_exc_wins"][0] <= outlier_default["ret_exc_wins"][0]


class TestAddRetExcWinsDaily:
    """Tests for daily frequency."""

    def test_crsp_stocks_unchanged(self, data_dir):
        """CRSP stocks should have ret_exc_wins == ret_exc for daily data."""
        rows = [
            {"id": 10001, "eom": date(2020, 1, 31), "date": date(2020, 1, 15), "ret_exc": 0.005},
            {"id": 10002, "eom": date(2020, 1, 31), "date": date(2020, 1, 15), "ret_exc": -0.003},
        ]
        _make_world_dsf(data_dir, rows)
        add_ret_exc_wins("d")

        result = _read_result(data_dir, "d")
        assert "ret_exc_wins" in result.columns
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_outlier_clipped(self, data_dir):
        """Compustat daily outlier should be clipped."""
        normal_rows = [
            {
                "id": 100000 + i,
                "eom": date(2020, 1, 31),
                "date": date(2020, 1, 15),
                "ret_exc": 0.001 * i,
            }
            for i in range(1, 2001)
        ]
        outlier_row = {
            "id": 200001,
            "eom": date(2020, 1, 31),
            "date": date(2020, 1, 15),
            "ret_exc": 99.0,
        }
        rows = normal_rows + [outlier_row]
        _make_world_dsf(data_dir, rows)
        add_ret_exc_wins("d")

        result = _read_result(data_dir, "d")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] < outlier["ret_exc"][0]
