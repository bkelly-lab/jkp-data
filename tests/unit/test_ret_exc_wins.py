"""
Tests for add_ret_exc_wins() in aux_functions.py.

Verifies that the winsorized excess return column (ret_exc_wins) is correctly
computed: Compustat stocks (source_crsp == 0) are clipped to the precomputed
[lower, upper] cutoffs from return_cutoffs{,_daily}.parquet, while CRSP stocks
are left unchanged.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import add_ret_exc_wins


def _make_world_msf(path, rows: list[dict]) -> None:
    """Write a minimal world_msf parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path.interim_dir / "world_msf.parquet")


def _make_world_dsf(path, rows: list[dict]) -> None:
    """Write a minimal world_dsf parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path.interim_dir / "world_dsf.parquet")


def _make_cutoffs_monthly(path, rows: list[dict]) -> None:
    """Write a minimal return_cutoffs.parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path.interim_dir / "return_cutoffs.parquet")


def _make_cutoffs_daily(path, rows: list[dict]) -> None:
    """Write a minimal return_cutoffs_daily.parquet with the given rows."""
    pl.DataFrame(rows).write_parquet(path.interim_dir / "return_cutoffs_daily.parquet")


def _read_result(path, freq: str) -> pl.DataFrame:
    return pl.read_parquet(path.interim_dir / f"world_{freq}sf.parquet")


class TestAddRetExcWinsMonthly:
    """Tests for monthly frequency."""

    def test_crsp_stocks_unchanged(self, test_paths):
        """CRSP stocks (source_crsp == 1) should have ret_exc_wins == ret_exc."""
        rows = [
            {"id": 10001, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 0.05},
            {"id": 10002, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": -0.03},
            {"id": 10003, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins(test_paths, "m")

        result = _read_result(test_paths, "m")
        assert "ret_exc_wins" in result.columns
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_normal_unchanged(self, test_paths):
        """Compustat stocks within bounds should have ret_exc_wins == ret_exc."""
        rows = [
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.01},
            {"id": 100002, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.02},
            {"id": 100003, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.03},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins(test_paths, "m")

        result = _read_result(test_paths, "m")
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_outlier_clipped_high(self, test_paths):
        """Compustat stock above the upper cutoff should be clipped to the cutoff."""
        rows = [
            {"id": 200001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.05}],
        )
        add_ret_exc_wins(test_paths, "m")

        result = _read_result(test_paths, "m")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] == pytest.approx(0.05)

    def test_compustat_outlier_clipped_low(self, test_paths):
        """Compustat stock below the lower cutoff should be clipped to the cutoff."""
        rows = [
            {"id": 200001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": -99.0},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.05, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins(test_paths, "m")

        result = _read_result(test_paths, "m")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] == pytest.approx(-0.05)

    def test_null_ret_exc_stays_null(self, test_paths):
        """Null ret_exc should produce null ret_exc_wins."""
        rows = [
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": None},
            {"id": 100002, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.05},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins(test_paths, "m")

        result = _read_result(test_paths, "m")
        null_row = result.filter(pl.col("id") == 100001)
        assert null_row["ret_exc_wins"][0] is None

    def test_idempotent(self, test_paths):
        """Running add_ret_exc_wins twice should produce the same result."""
        rows = [
            {"id": 10001, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 0.05},
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 0.03},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins(test_paths, "m")
        first = _read_result(test_paths, "m")
        add_ret_exc_wins(test_paths, "m")
        second = _read_result(test_paths, "m")

        assert first["ret_exc_wins"].to_list() == second["ret_exc_wins"].to_list()
        assert first.columns == second.columns

    def test_source_crsp_boundary(self, test_paths):
        """Two rows with the same out-of-bounds ret_exc: only the Compustat row is clipped."""
        rows = [
            {"id": 10001, "source_crsp": 1, "eom": date(2020, 1, 31), "ret_exc": 99.0},
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [{"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.05}],
        )
        add_ret_exc_wins(test_paths, "m")

        result = _read_result(test_paths, "m")
        crsp_row = result.filter(pl.col("source_crsp") == 1)
        comp_row = result.filter(pl.col("source_crsp") == 0)
        assert crsp_row["ret_exc_wins"][0] == pytest.approx(99.0)
        assert comp_row["ret_exc_wins"][0] == pytest.approx(0.05)

    def test_multiple_eom_periods(self, test_paths):
        """Each row uses cutoffs from its own period."""
        rows = [
            # Jan: outlier clipped to Jan's cutoff
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
            # Feb: outlier clipped to Feb's (different) cutoff
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 2, 29), "ret_exc": 99.0},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
            [
                {"eom": date(2020, 1, 31), "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.05},
                {"eom": date(2020, 2, 29), "ret_exc_0_1": -0.20, "ret_exc_99_9": 0.15},
            ],
        )
        add_ret_exc_wins(test_paths, "m")

        result = _read_result(test_paths, "m").sort("eom")
        assert result["ret_exc_wins"].to_list() == pytest.approx([0.05, 0.15])

    def test_custom_percentiles(self, test_paths):
        """Custom lower/upper arguments should select the corresponding cutoff columns."""
        rows = [
            {"id": 100001, "source_crsp": 0, "eom": date(2020, 1, 31), "ret_exc": 99.0},
        ]
        _make_world_msf(test_paths, rows)
        _make_cutoffs_monthly(
            test_paths,
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
        add_ret_exc_wins(test_paths, "m", lower=0.01, upper=0.99)
        wide = _read_result(test_paths, "m")
        assert wide["ret_exc_wins"][0] == pytest.approx(0.04)

        # Re-create source data and run with the default 0.1% / 99.9% (ret_exc_99_9 = 0.08)
        _make_world_msf(test_paths, rows)
        add_ret_exc_wins(test_paths, "m")
        default = _read_result(test_paths, "m")
        assert default["ret_exc_wins"][0] == pytest.approx(0.08)


class TestAddRetExcWinsDaily:
    """Tests for daily frequency."""

    def test_crsp_stocks_unchanged(self, test_paths):
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
        _make_world_dsf(test_paths, rows)
        _make_cutoffs_daily(
            test_paths,
            [{"year": 2020, "month": 1, "ret_exc_0_1": -0.10, "ret_exc_99_9": 0.10}],
        )
        add_ret_exc_wins(test_paths, "d")

        result = _read_result(test_paths, "d")
        assert "ret_exc_wins" in result.columns
        assert "year" not in result.columns
        assert "month" not in result.columns
        assert result["ret_exc_wins"].to_list() == result["ret_exc"].to_list()

    def test_compustat_outlier_clipped(self, test_paths):
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
        _make_world_dsf(test_paths, rows)
        _make_cutoffs_daily(
            test_paths,
            [{"year": 2020, "month": 1, "ret_exc_0_1": -0.05, "ret_exc_99_9": 0.05}],
        )
        add_ret_exc_wins(test_paths, "d")

        result = _read_result(test_paths, "d")
        outlier = result.filter(pl.col("id") == 200001)
        assert outlier["ret_exc_wins"][0] == pytest.approx(0.05)


class TestAddRetExcWinsValidation:
    """Tests for input validation of lower/upper percentile parameters."""

    def test_lower_negative_raises(self, test_paths):
        """A negative lower bound should raise ValueError."""
        with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
            add_ret_exc_wins(test_paths, "m", lower=-0.1, upper=0.999)

    def test_upper_above_one_raises(self, test_paths):
        """An upper bound > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
            add_ret_exc_wins(test_paths, "m", lower=0.001, upper=1.1)

    def test_lower_gte_upper_raises(self, test_paths):
        """lower >= upper should raise ValueError."""
        with pytest.raises(ValueError, match="0 <= lower < upper <= 1"):
            add_ret_exc_wins(test_paths, "m", lower=0.5, upper=0.4)

    def test_unsupported_percentile_raises(self, test_paths):
        """A percentile not in the precomputed cutoffs should raise ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            add_ret_exc_wins(test_paths, "m", lower=0.005, upper=0.999)
