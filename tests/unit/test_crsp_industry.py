"""Tests for crsp_industry() (Issue #155).

Covers the daily SIC/NAICS expansion of CRSP name-history rows:

- Date-range expansion via ``pl.date_ranges(secinfostartdt, secinfoenddt)``
- Null/sentinel handling for ``sic == 0`` (must become null)
- Preservation of null ``naics`` values
- Dedup on overlapping name-history spans via ``.unique(["permno","date"])``
- Sort invariant on ``(permno, date)``
- A regression golden fixture locking the output bit-for-bit.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import crsp_industry
from jkp.data.paths import DataPaths

GOLDEN_DIR = Path(__file__).parent.parent / "golden" / "fixtures" / "crsp_industry"


def _write_permno0(paths: DataPaths, df: pl.DataFrame) -> None:
    """Persist a permno0 fixture in the location ``crsp_industry`` reads from."""
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    raw_data_dfs.mkdir(parents=True, exist_ok=True)
    df.write_parquet(raw_data_dfs / "permno0.parquet")


class TestCrspIndustry:
    """Tests for ``crsp_industry``."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths
        self.output_path = self.paths.interim_dir / "crsp_ind.parquet"

    def test_date_range_expansion(self) -> None:
        """A single span produces one row per calendar date in [start, end]."""
        df = pl.DataFrame(
            {
                "permno": [10001],
                "permco": [1],
                "secinfostartdt": [date(2020, 1, 1)],
                "secinfoenddt": [date(2020, 1, 5)],
                "sic": [7372],
                "naics": [511210],
            },
            schema={
                "permno": pl.Int64,
                "permco": pl.Int64,
                "secinfostartdt": pl.Date,
                "secinfoenddt": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_permno0(self.paths, df)

        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path).sort("date")
        assert result.height == 5, "5-day span should explode to 5 daily rows"
        expected_dates = [
            date(2020, 1, 1),
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 4),
            date(2020, 1, 5),
        ]
        assert result["date"].to_list() == expected_dates, (
            f"Expected contiguous dates {expected_dates}, got {result['date'].to_list()}"
        )
        assert result["sic"].to_list() == [7372] * 5, (
            f"Expected sic=7372 for all rows, got {result['sic'].to_list()}"
        )
        assert result["naics"].to_list() == [511210] * 5, (
            f"Expected naics=511210 for all rows, got {result['naics'].to_list()}"
        )

    def test_multiple_non_overlapping_spans_per_permno(self) -> None:
        """Two non-overlapping spans expand independently; rows are contiguous within each."""
        df = pl.DataFrame(
            {
                "permno": [10001, 10001],
                "permco": [1, 1],
                "secinfostartdt": [date(2020, 1, 1), date(2020, 1, 6)],
                "secinfoenddt": [date(2020, 1, 3), date(2020, 1, 8)],
                "sic": [7372, 7370],
                "naics": [511210, 511200],
            },
            schema={
                "permno": pl.Int64,
                "permco": pl.Int64,
                "secinfostartdt": pl.Date,
                "secinfoenddt": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_permno0(self.paths, df)

        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path).sort("date")
        # 3 days + 3 days = 6 rows; no fill across the Jan 4-5 gap.
        assert result.height == 6, f"Expected 6 rows (3+3 from two spans), got {result.height}"
        assert date(2020, 1, 4) not in result["date"].to_list(), (
            "Jan 4 should not appear — it falls in the gap between spans"
        )
        assert date(2020, 1, 5) not in result["date"].to_list(), (
            "Jan 5 should not appear — it falls in the gap between spans"
        )
        # Codes track their span.
        first_span = result.filter(pl.col("date") <= date(2020, 1, 3))
        second_span = result.filter(pl.col("date") >= date(2020, 1, 6))
        assert first_span["sic"].unique().to_list() == [7372], (
            f"First span should carry sic=7372, got {first_span['sic'].unique().to_list()}"
        )
        assert second_span["sic"].unique().to_list() == [7370], (
            f"Second span should carry sic=7370, got {second_span['sic'].unique().to_list()}"
        )

    def test_sic_zero_becomes_null(self) -> None:
        """``sic == 0`` is rewritten to a typed null."""
        df = pl.DataFrame(
            {
                "permno": [10002],
                "permco": [2],
                "secinfostartdt": [date(2020, 2, 1)],
                "secinfoenddt": [date(2020, 2, 2)],
                "sic": [0],
                "naics": [None],
            },
            schema={
                "permno": pl.Int64,
                "permco": pl.Int64,
                "secinfostartdt": pl.Date,
                "secinfoenddt": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_permno0(self.paths, df)

        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result["sic"].dtype == pl.Int64, (
            f"Expected sic dtype Int64, got {result['sic'].dtype}"
        )
        assert result["sic"].null_count() == result.height, (
            "All sic values should be null after sic==0 rewrite"
        )

    def test_null_naics_preserved(self) -> None:
        """Null ``naics`` values pass through unchanged."""
        df = pl.DataFrame(
            {
                "permno": [10002],
                "permco": [2],
                "secinfostartdt": [date(2020, 2, 1)],
                "secinfoenddt": [date(2020, 2, 3)],
                "sic": [7372],
                "naics": [None],
            },
            schema={
                "permno": pl.Int64,
                "permco": pl.Int64,
                "secinfostartdt": pl.Date,
                "secinfoenddt": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_permno0(self.paths, df)

        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result["naics"].null_count() == result.height, (
            f"Expected all naics null, got {result.height - result['naics'].null_count()} non-null"
        )

    def test_overlapping_spans_deduplicated(self) -> None:
        """Two spans sharing dates collapse to one row per ``(permno, date)``."""
        df = pl.DataFrame(
            {
                "permno": [10003, 10003],
                "permco": [3, 3],
                "secinfostartdt": [date(2020, 3, 1), date(2020, 3, 3)],
                "secinfoenddt": [date(2020, 3, 4), date(2020, 3, 6)],
                "sic": [6020, 6020],
                "naics": [522110, 522110],
            },
            schema={
                "permno": pl.Int64,
                "permco": pl.Int64,
                "secinfostartdt": pl.Date,
                "secinfoenddt": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_permno0(self.paths, df)

        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        # Union of [Mar 1-4] and [Mar 3-6] is Mar 1-6 = 6 distinct dates.
        assert result.height == 6, (
            f"Expected 6 distinct dates from overlapping spans, got {result.height}"
        )
        # No duplicate (permno, date) keys.
        assert result.unique(["permno", "date"]).height == result.height, (
            f"Found duplicate (permno, date) rows: {result.height} total vs "
            f"{result.unique(['permno', 'date']).height} unique"
        )

    def test_sorted_by_permno_date(self) -> None:
        """Output rows are sorted by ``(permno, date)`` ascending."""
        df = pl.DataFrame(
            {
                "permno": [20001, 10001],
                "permco": [2, 1],
                "secinfostartdt": [date(2020, 1, 1), date(2020, 1, 2)],
                "secinfoenddt": [date(2020, 1, 3), date(2020, 1, 4)],
                "sic": [7370, 7372],
                "naics": [511200, 511210],
            },
            schema={
                "permno": pl.Int64,
                "permco": pl.Int64,
                "secinfostartdt": pl.Date,
                "secinfoenddt": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_permno0(self.paths, df)

        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        permnos = result["permno"].to_list()
        dates = result["date"].to_list()
        assert permnos == sorted(permnos), f"permnos not sorted ascending: {permnos}"
        for i in range(1, len(result)):
            same_permno = permnos[i] == permnos[i - 1]
            assert (not same_permno) or dates[i] >= dates[i - 1], (
                f"Dates not sorted within permno {permnos[i]}: "
                f"{dates[i - 1]} > {dates[i]} at rows {i - 1},{i}"
            )

    def test_preserves_permco(self) -> None:
        """``permco`` is carried through to the daily output."""
        df = pl.DataFrame(
            {
                "permno": [10001],
                "permco": [42],
                "secinfostartdt": [date(2020, 1, 1)],
                "secinfoenddt": [date(2020, 1, 2)],
                "sic": [7372],
                "naics": [511210],
            },
            schema={
                "permno": pl.Int64,
                "permco": pl.Int64,
                "secinfostartdt": pl.Date,
                "secinfoenddt": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_permno0(self.paths, df)

        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result["permco"].unique().to_list() == [42], (
            f"Expected permco=42 preserved, got {result['permco'].unique().to_list()}"
        )

    @pytest.mark.regression
    def test_crsp_industry_golden_fixture(self) -> None:
        """Bit-identical match against the locked golden fixture."""
        from tests.golden.generate_crsp_industry_golden import build_permno0_input

        _write_permno0(self.paths, build_permno0_input(seed=42))
        crsp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        golden = pl.read_parquet(GOLDEN_DIR / "crsp_ind.parquet")
        assert_frame_equal(result, golden, check_exact=True)
