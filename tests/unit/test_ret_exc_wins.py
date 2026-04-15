"""
Tests for add_ret_exc_wins() in aux_functions.py.

Verifies that the winsorized excess return column (ret_exc_wins) is correctly
computed: Compustat stocks (source_crsp == 0) are clipped to the precomputed
[lower, upper] cutoffs from return_cutoffs{,_daily}.parquet, while CRSP stocks
are left unchanged.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from jkp.data.aux_functions import add_ret_exc_wins


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Set working directory to tmp_path and return it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_world_msf(path: Path, rows: list[dict]) -> None:
    """Write a minimal world_msf parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path / "world_msf.parquet")


def _make_world_dsf(path: Path, rows: list[dict]) -> None:
    """Write a minimal world_dsf parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path / "world_dsf.parquet")


def _make_cutoffs_monthly(path: Path, rows: list[dict]) -> None:
    """Write a minimal return_cutoffs.parquet with the given rows.

    Each row should have an `eom` and any of the cutoff columns
    (ret_exc_0_1, ret_exc_1, ret_exc_99, ret_exc_99_9).
    """
    pl.DataFrame(rows).write_parquet(path / "return_cutoffs.parquet")


def _make_cutoffs_daily(path: Path, rows: list[dict]) -> None:
    """Write a minimal return_cutoffs_daily.parquet with the given rows.

    Each row should have `year`, `month`, and any of the cutoff columns
    (ret_exc_0_1, ret_exc_1, ret_exc_99, ret_exc_99_9).
    """
    pl.DataFrame(rows).write_parquet(path / "return_cutoffs_daily.parquet")


def _read_result(path: Path, freq: str) -> pl.DataFrame:
    return pl.read_parquet(path / f"world_{freq}sf.parquet")


class TestAddRetExcWinsMonthly:
    """Tests for monthly frequency."""

    def test_crsp_stocks_unchanged(self, data_dir):
        """CRSP stocks (source_crsp == 1) should have ret_exc_wins == ret_exc."""
        rows = [
            {"id": 10001, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 0.05},
            {"id": 10002, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": -0.03},
            {"id": 10003, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        assert "ret_exc_wins" in result.columns
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_normal_unchanged(self, data_dir):
        """Compustat stocks within bounds should have ret_exc_wins == ret_exc."""
        rows = [
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.01},
            {"id": 100002, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.02},
            {"id": 100003, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.03},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_outlier_clipped_high(self, data_dir):
        """Compustat stock above the upper cutoff should be clipped to the cutoff."""
        rows = [
            {"id": 200001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.05}],
        )
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] == pytest.approx(0.05)

    def test_compustat_outlier_clipped_low(self, data_dir):
        """Compustat stock below the lower cutoff should be clipped to the cutoff."""
        rows = [
            {"id": 200001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": -99.0},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.05, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] == pytest.approx(-0.05)

    def test_null_ret_exc_stays_null(self, data_dir):
        """Null ret_exc should produce null ret_exc_wins."""
        rows = [
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": None},
            {"id": 100002, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.05},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        null_row = result.filter(pl.col("id") == 100001)
        assert null_row["ret_exc_wins"][0] is None

    def test_idempotent(self, data_dir):
        """Running add_ret_exc_wins twice should produce the same result."""
        rows = [
            {"id": 10001, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 0.05},
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.03},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins("m")
        first = _read_result(data_dir, "m")
        add_ret_exc_wins("m")
        second = _read_result(data_dir, "m")

        assert first["ret_exc_wins"].to_list() == second["ret_exc_wins"].to_list()
        assert first.columns == second.columns

    def test_source_crsp_boundary(self, data_dir):
        """Two rows with the same out-of-bounds ret_exc: only the Compustat row is clipped."""
        rows = [
            {"id": 10001, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 99.0},
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.05}],
        )
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m")
        crsp_row = result.filter(pl.col("source_crsp") == 1)
        comp_row = result.filter(pl.col("source_crsp") == 0)
        assert crsp_row["ret_exc_wins"][0] == pytest.approx(99.0)
        assert comp_row["ret_exc_wins"][0] == pytest.approx(0.05)

    def test_multiple_eom_periods(self, data_dir):
        """Each row uses cutoffs from its own period."""
        rows = [
            # Jan: outlier clipped to Jan's cutoff
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
            # Feb: outlier clipped to Feb's (different) cutoff
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 2, 29), "ret_exc": 99.0},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [
                {"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.05},
                {"eom": date(2020, 2, 29), "ret_exc_0_1": -0.20, "ret_exc_99_9": 0.15},
            ],
        )
        add_ret_exc_wins("m")

        result = _read_result(data_dir, "m").sort("eom")
        assert result["ret_exc_wins"].to_list() == pytest.approx([0.05, 0.15])

    def test_custom_percentiles(self, data_dir):
        """Custom lower/upper arguments should select the corresponding cutoff columns."""
        rows = [
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(data_dir, rows)
        _make_cutoffs_monthly(
            data_dir,
            [
                {
                    "eom": date(2020, 1, 31),
                    "ret_exc_0_1": -0.10,
                    "ret_exc_1": -0.05,
                    "ret_exc_99": 0.04,
                    "ret_exc_99_9": 0.08,
                }
            ],
        )

        # 1% / 99% cutoffs (ret_exc_99 = 0.04)
        add_ret_exc_wins("m", lower=0.01, upper=0.99)
        wide = _read_result(data_dir, "m")
        assert wide["ret_exc_wins"][0] == pytest.approx(0.04)

        # Re-create source data and run with the default 0.1% / 99.9% (ret_exc_99_9 = 0.08)
        _make_world_msf(data_dir, rows)
        add_ret_exc_wins("m")
        default = _read_result(data_dir, "m")
        assert default["ret_exc_wins"][0] == pytest.approx(0.08)


class TestAddRetExcWinsDaily:
    """Tests for daily frequency."""

    def test_crsp_stocks_unchanged(self, data_dir):
        """CRSP stocks should have ret_exc_wins == ret_exc for daily data."""
        rows = [
            {
                "id": 10001,
                "source_crsp": 1,
                "eom": date(2020, 1, 31),
                "date": date(2020, 1, 15),
                "ret_exc": 0.005,
            },
            {
                "id": 10002,
                "source_crsp": 1,
                "eom": date(2020, 1, 31),
                "date": date(2020, 1, 15),
                "ret_exc": -0.003,
            },
        ]
        _make_world_dsf(data_dir, rows)
        _make_cutoffs_daily(
            data_dir,
            [{"year": 2020, "month": 1, "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins("d")

        result = _read_result(data_dir, "d")
        assert "ret_exc_wins" in result.columns
        assert "year" not in result.columns
        assert "month" not in result.columns
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_outlier_clipped(self, data_dir):
        """Compustat daily outlier should be clipped to the daily cutoff."""
        rows = [
            {
                "id": 200001,
                "source_crsp": 0,
                "eom": date(2020, 1, 31),
                "date": date(2020, 1, 15),
                "ret_exc": 99.0,
            },
        ]
        _make_world_dsf(data_dir, rows)
        _make_cutoffs_daily(
            data_dir,
            [{"year": 2020, "month": 1, "ret_exc_0_1": -0.05, "ret_exc_99_9": 0.05}],
        )
        add_ret_exc_wins("d")

        result = _read_result(data_dir, "d")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] == pytest.approx(0.05)


class TestAddRetExcWinsValidation:
    """Tests for input validation of lower/upper percentile parameters."""

    def test_lower_negative_raises(self, data_dir):
        """A negative lower bound should raise ValueError."""
        with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
            add_ret_exc_wins("m", lower=-0.1, upper=0.999)

    def test_upper_above_one_raises(self, data_dir):
        """An upper bound > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
            add_ret_exc_wins("m", lower=0.001, upper=1.1)

    def test_lower_gte_upper_raises(self, data_dir):
        """lower >= upper should raise ValueError."""
        with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
            add_ret_exc_wins("m", lower=0.5, upper=0.4)

    def test_unsupported_percentile_raises(self, data_dir):
        """A percentile not in the precomputed cutoffs should raise ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            add_ret_exc_wins("m", lower=0.005, upper=0.999)
