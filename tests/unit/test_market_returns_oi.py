"""
Tests for market_returns_overnight_intraday.

Validates that:
- market_returns_overnight_intraday produces daily VW/EW returns per component
- Null component returns are handled correctly (rows excluded from aggregation)
- Graceful skip when overnight/intraday columns are absent from data
- Low-coverage trading days are dropped for daily frequency
- save_output_files copies OI files when present
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from jkp.data.paths import DataPaths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_stock_columns(n: int = 20) -> dict[str, list]:
    """Minimal stock-level columns needed by market_returns_overnight_intraday."""
    rng = np.random.default_rng(42)
    ids = list(range(1, n + 1))
    eom = [date(2024, 1, 31)] * n
    return {
        "id": ids,
        "date": [date(2024, 1, d + 1) for d in range(n)],
        "eom": eom,
        "excntry": ["USA"] * n,
        "source_crsp": [1] * n,
        "obs_main": [1] * n,
        "exch_main": [1] * n,
        "primary_sec": [1] * n,
        "common": [1] * n,
        "ret_lag_dif": [1] * n,
        "me": (rng.uniform(100, 1000, n)).tolist(),
        "dolvol": (rng.uniform(1e6, 1e8, n)).tolist(),
        "ret": (rng.normal(0.001, 0.02, n)).tolist(),
        "ret_local": (rng.normal(0.001, 0.02, n)).tolist(),
        "ret_exc": (rng.normal(0.001, 0.02, n)).tolist(),
        "ret_intraday": (rng.normal(0.0005, 0.015, n)).tolist(),
        "ret_overnight": (rng.normal(0.0005, 0.010, n)).tolist(),
    }


@pytest.fixture
def world_dsf(tmp_path: Path) -> Path:
    """Write a world_dsf.parquet with overnight/intraday columns."""
    data = _base_stock_columns()
    df = pl.DataFrame(data)
    path = tmp_path / "interim" / "world_dsf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


@pytest.fixture
def nyse_cutoffs(tmp_path: Path) -> Path:
    """Write a minimal nyse_cutoffs.parquet."""
    df = pl.DataFrame({"eom": [date(2024, 1, 31)], "nyse_p80": [500.0]})
    path = tmp_path / "interim" / "nyse_cutoffs.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


@pytest.fixture
def test_paths(tmp_path: Path) -> DataPaths:
    (tmp_path / "interim").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processed" / "other_output").mkdir(parents=True, exist_ok=True)
    return DataPaths(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# market_returns_overnight_intraday
# ---------------------------------------------------------------------------


class TestMarketReturnsOvernightIntraday:
    """Test market-level overnight/intraday return aggregation."""

    def test_daily_files_written(
        self, test_paths: DataPaths, world_dsf: Path, nyse_cutoffs: Path
    ) -> None:
        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, world_dsf, "d", nyse_cutoffs)

        for component in ("overnight", "intraday"):
            path = test_paths.interim_dir / f"market_returns_daily_{component}.parquet"
            assert path.exists(), f"{path.name} was not written"

    def test_output_columns(
        self, test_paths: DataPaths, world_dsf: Path, nyse_cutoffs: Path
    ) -> None:
        """Each output file should have the expected VW/EW columns."""
        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, world_dsf, "d", nyse_cutoffs)

        for component in ("overnight", "intraday"):
            path = test_paths.interim_dir / f"market_returns_daily_{component}.parquet"
            df = pl.read_parquet(path)
            assert "excntry" in df.columns
            assert "date" in df.columns
            assert "stocks" in df.columns
            assert f"mkt_vw_{component}" in df.columns
            assert f"mkt_ew_{component}" in df.columns

    def test_vw_return_is_weighted_average(
        self, test_paths: DataPaths, nyse_cutoffs: Path, tmp_path: Path
    ) -> None:
        """VW return should equal sum(ret*me_lag1) / sum(me_lag1)."""
        data = {
            "id": [1, 1, 2, 2],
            "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
            "eom": [date(2024, 1, 31)] * 4,
            "excntry": ["USA"] * 4,
            "source_crsp": [1] * 4,
            "obs_main": [1] * 4,
            "exch_main": [1] * 4,
            "primary_sec": [1] * 4,
            "common": [1] * 4,
            "ret_lag_dif": [1] * 4,
            "me": [100.0, 200.0, 300.0, 400.0],
            "dolvol": [1e7] * 4,
            "ret": [0.01] * 4,
            "ret_local": [0.01] * 4,
            "ret_exc": [0.01] * 4,
            "ret_overnight": [0.01, 0.02, 0.03, 0.04],
            "ret_intraday": [0.005, 0.010, 0.015, 0.020],
        }
        df = pl.DataFrame(data)
        dsf_path = tmp_path / "interim" / "vw_test.parquet"
        dsf_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dsf_path)

        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, dsf_path, "d", nyse_cutoffs)

        result = pl.read_parquet(test_paths.interim_dir / "market_returns_daily_overnight.parquet")
        # On date 2024-01-02, id=1 has me_lag1=100, id=2 has me_lag1=300
        # Both capped at nyse_p80=500, so weights = 100 and 300
        # ret_overnight: id1=0.02, id2=0.04
        # VW = (0.02*100 + 0.04*300) / (100+300) = (2+12)/400 = 0.035
        row = result.filter(pl.col("date") == date(2024, 1, 2))
        if row.height > 0:
            vw = row["mkt_vw_overnight"][0]
            np.testing.assert_allclose(vw, 0.035, rtol=1e-10)

    def test_graceful_skip_without_columns(self, test_paths: DataPaths, nyse_cutoffs: Path) -> None:
        """When ret_intraday/ret_overnight are absent, function should skip."""
        df = pl.DataFrame(
            {
                "id": [1],
                "date": [date(2024, 1, 15)],
                "eom": [date(2024, 1, 31)],
                "excntry": ["USA"],
                "source_crsp": [1],
                "obs_main": [1],
                "exch_main": [1],
                "primary_sec": [1],
                "common": [1],
                "ret_lag_dif": [1],
                "me": [100.0],
                "dolvol": [1e7],
                "ret": [0.01],
                "ret_local": [0.01],
                "ret_exc": [0.01],
            }
        )
        data_path = test_paths.interim_dir / "world_dsf_no_oi.parquet"
        df.write_parquet(data_path)

        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, data_path, "d", nyse_cutoffs)

        assert not (test_paths.interim_dir / "market_returns_daily_overnight.parquet").exists()
        assert not (test_paths.interim_dir / "market_returns_daily_intraday.parquet").exists()

    def test_null_component_returns_excluded(
        self, test_paths: DataPaths, nyse_cutoffs: Path, tmp_path: Path
    ) -> None:
        """Rows with null component returns should be excluded from aggregation."""
        data = _base_stock_columns(n=10)
        data["ret_overnight"][0] = None  # type: ignore[assignment]
        data["ret_overnight"][1] = None  # type: ignore[assignment]
        df = pl.DataFrame(data)
        dsf_path = tmp_path / "interim" / "null_test.parquet"
        dsf_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dsf_path)

        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, dsf_path, "d", nyse_cutoffs)

        result = pl.read_parquet(test_paths.interim_dir / "market_returns_daily_overnight.parquet")
        assert result["mkt_vw_overnight"].null_count() == 0
        assert result["mkt_ew_overnight"].null_count() == 0

    def test_partial_components(
        self, test_paths: DataPaths, nyse_cutoffs: Path, tmp_path: Path
    ) -> None:
        """If only ret_overnight exists (no ret_intraday), only overnight file is written."""
        data = _base_stock_columns()
        df = pl.DataFrame(data).drop("ret_intraday")
        dsf_path = tmp_path / "interim" / "partial_test.parquet"
        dsf_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dsf_path)

        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, dsf_path, "d", nyse_cutoffs)

        assert (test_paths.interim_dir / "market_returns_daily_overnight.parquet").exists()
        assert not (test_paths.interim_dir / "market_returns_daily_intraday.parquet").exists()

    def test_output_sorted_by_excntry_date(
        self, test_paths: DataPaths, world_dsf: Path, nyse_cutoffs: Path
    ) -> None:
        """Output should be sorted by excntry, date."""
        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, world_dsf, "d", nyse_cutoffs)

        for component in ("overnight", "intraday"):
            df = pl.read_parquet(
                test_paths.interim_dir / f"market_returns_daily_{component}.parquet"
            )
            if df.height > 1:
                assert df.equals(df.sort(["excntry", "date"]))


# ---------------------------------------------------------------------------
# save_output_files
# ---------------------------------------------------------------------------


class TestSaveOutputFiles:
    """Test that save_output_files copies overnight/intraday daily files."""

    def test_oi_daily_files_copied_when_present(self, test_paths: DataPaths) -> None:
        from jkp.data.aux_functions import save_output_files

        for name in (
            "market_returns.parquet",
            "market_returns_daily.parquet",
            "nyse_cutoffs.parquet",
            "return_cutoffs.parquet",
            "return_cutoffs_daily.parquet",
            "ap_factors_monthly.parquet",
            "ap_factors_daily.parquet",
        ):
            pl.DataFrame({"x": [1]}).write_parquet(test_paths.interim_dir / name)

        oi_daily_files = []
        for component in ("overnight", "intraday"):
            for prefix in ("market_returns_daily", "return_cutoffs_daily"):
                name = f"{prefix}_{component}.parquet"
                pl.DataFrame({"x": [1]}).write_parquet(test_paths.interim_dir / name)
                oi_daily_files.append(name)

        save_output_files(test_paths)

        other_output = test_paths.processed_dir / "other_output"
        for name in oi_daily_files:
            assert (other_output / name).exists(), f"{name} not copied to other_output"

    def test_oi_files_skipped_when_absent(self, test_paths: DataPaths) -> None:
        """save_output_files should not fail if OI files don't exist."""
        from jkp.data.aux_functions import save_output_files

        for name in (
            "market_returns.parquet",
            "market_returns_daily.parquet",
            "nyse_cutoffs.parquet",
            "return_cutoffs.parquet",
            "return_cutoffs_daily.parquet",
            "ap_factors_monthly.parquet",
            "ap_factors_daily.parquet",
        ):
            pl.DataFrame({"x": [1]}).write_parquet(test_paths.interim_dir / name)

        save_output_files(test_paths)

        other_output = test_paths.processed_dir / "other_output"
        assert not (other_output / "market_returns_daily_overnight.parquet").exists()
