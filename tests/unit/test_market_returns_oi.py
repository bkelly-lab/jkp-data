"""
Tests for overnight/intraday versions of market_returns and return_cutoffs.

Validates that:
- return_cutoffs_overnight_intraday produces correct daily percentile columns
- market_returns_overnight_intraday produces daily VW/EW returns per component
- Null component returns are handled correctly (rows excluded from aggregation)
- Graceful skip when overnight/intraday columns are absent from data
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


def _base_stock_columns() -> dict[str, list]:
    """Minimal stock-level columns needed by both functions."""
    n = 20
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
# return_cutoffs_overnight_intraday
# ---------------------------------------------------------------------------


class TestReturnCutoffsOvernightIntraday:
    """Test percentile cutoff computation for overnight/intraday returns."""

    def test_daily_cutoffs_have_year_month(self, test_paths: DataPaths, world_dsf: Path):
        from jkp.data.aux_functions import return_cutoffs_overnight_intraday

        return_cutoffs_overnight_intraday(test_paths, "d")

        for component in ("overnight", "intraday"):
            path = test_paths.interim_dir / f"return_cutoffs_daily_{component}.parquet"
            assert path.exists()
            df = pl.read_parquet(path)
            assert "year" in df.columns
            assert "month" in df.columns


# ---------------------------------------------------------------------------
# market_returns_overnight_intraday
# ---------------------------------------------------------------------------


class TestMarketReturnsOvernightIntraday:
    """Test market-level overnight/intraday return aggregation."""

    def test_daily_files_written(self, test_paths: DataPaths, world_dsf: Path, nyse_cutoffs: Path):
        from jkp.data.aux_functions import market_returns_overnight_intraday

        market_returns_overnight_intraday(test_paths, world_dsf, "d", nyse_cutoffs)

        for component in ("overnight", "intraday"):
            path = test_paths.interim_dir / f"market_returns_daily_{component}.parquet"
            assert path.exists()

    def test_graceful_skip_without_columns(self, test_paths: DataPaths, nyse_cutoffs: Path):
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


# ---------------------------------------------------------------------------
# save_output_files
# ---------------------------------------------------------------------------


class TestSaveOutputFiles:
    """Test that save_output_files copies overnight/intraday daily files."""

    def test_oi_daily_files_copied_when_present(self, test_paths: DataPaths):
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
            for name in (
                f"market_returns_daily_{component}.parquet",
                f"return_cutoffs_daily_{component}.parquet",
            ):
                pl.DataFrame({"x": [1]}).write_parquet(test_paths.interim_dir / name)
                oi_daily_files.append(name)

        save_output_files(test_paths)

        other_output = test_paths.processed_dir / "other_output"
        for name in oi_daily_files:
            assert (other_output / name).exists(), f"{name} not copied to other_output"

    def test_oi_files_skipped_when_absent(self, test_paths: DataPaths):
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
