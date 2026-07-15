"""
Tests for deterministic deduplication in the Compustat stock file pipeline.

Verifies that the dedup locations covered in this file resolve duplicates
using economically meaningful tie-breaking rules (issue #69):

1. add_primary_sec(): prefer primary_sec=1 over primary_sec=0
2. gen_returns_df(): keep highest prcstd per {gvkey, iid, datadate}
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest


class TestAddPrimarySecDedup:
    """Test that add_primary_sec() prefers primary_sec=1 when range-join
    fan-out produces conflicting classifications."""

    @pytest.fixture()
    def setup_paths(self, test_paths):
        """Set up minimal parquet files for add_primary_sec() under DataPaths layout."""
        raw_data_dfs = test_paths.interim_dir / "raw_data_dfs"
        # raw_data_dfs is created by the temp_data_dir fixture; ensure exists.
        raw_data_dfs.mkdir(parents=True, exist_ok=True)

        # Input security file: two securities for one company
        input_df = pl.DataFrame(
            {
                "gvkey": ["001000", "001000", "001000", "001000"],
                "iid": ["01", "01", "02", "02"],
                "datadate": [
                    date(2024, 1, 31),
                    date(2024, 2, 29),
                    date(2024, 1, 31),
                    date(2024, 2, 29),
                ],
                "prc": [10.0, 11.0, 20.0, 21.0],
            }
        )
        input_path = test_paths.interim_dir / "input.parquet"
        input_df.write_parquet(input_path)

        # prihistrow: TWO overlapping ranges for gvkey 001000
        # Range 1 says prihistrow="01", range 2 says prihistrow="02"
        # Both cover datadate 2024-01-31 and 2024-02-29
        pl.DataFrame(
            {
                "gvkey": ["001000", "001000"],
                "prihistrow": ["01", "02"],
                "effdate": [date(2023, 1, 1), date(2024, 1, 1)],
                "thrudate": [date(2024, 6, 30), date(2024, 12, 31)],
            }
        ).write_parquet(raw_data_dfs / "__prihistrow.parquet")

        # prihistusa / prihistcan: empty (no matches)
        for name in ["__prihistusa", "__prihistcan"]:
            pl.DataFrame(
                {
                    "gvkey": pl.Series([], dtype=pl.Utf8),
                    f"{name.lstrip('_')}": pl.Series([], dtype=pl.Utf8),
                    "effdate": pl.Series([], dtype=pl.Date),
                    "thrudate": pl.Series([], dtype=pl.Date),
                }
            ).write_parquet(raw_data_dfs / f"{name}.parquet")

        # header: fallback values (won't matter since prihistrow matches)
        pl.DataFrame(
            {
                "gvkey": ["001000"],
                "prirow": ["01"],
                "priusa": pl.Series([None], dtype=pl.Utf8),
                "prican": pl.Series([None], dtype=pl.Utf8),
            }
        ).write_parquet(raw_data_dfs / "__header.parquet")

        return test_paths, input_path, test_paths.interim_dir / "output.parquet"

    def test_primary_sec_prefers_one(self, setup_paths) -> None:
        """When overlapping ranges produce both primary_sec=0 and primary_sec=1,
        the output should deterministically keep primary_sec=1."""
        from jkp.data.aux_functions import add_primary_sec

        paths, input_path, output_path = setup_paths
        add_primary_sec(paths, input_path, "datadate", output_path)

        result = pl.read_parquet(output_path)

        # iid="01" should be primary (matches prihistrow="01" from range 1)
        iid01 = result.filter(pl.col("iid") == "01")
        assert iid01["primary_sec"].to_list() == [1, 1], (
            "iid='01' should be primary_sec=1 for both dates"
        )

        # Each (gvkey, iid, datadate) should appear exactly once
        key_counts = result.group_by(["gvkey", "iid", "datadate"]).len()
        assert key_counts["len"].max() == 1, "No duplicate rows should remain"

    def test_non_primary_stays_zero(self, setup_paths) -> None:
        """Securities that don't match any prihist record should get primary_sec=0."""
        from jkp.data.aux_functions import add_primary_sec

        paths, input_path, output_path = setup_paths
        add_primary_sec(paths, input_path, "datadate", output_path)

        result = pl.read_parquet(output_path)

        # iid="02" matches prihistrow="02" from range 2, so it IS primary
        iid02 = result.filter(pl.col("iid") == "02")
        assert iid02["primary_sec"].to_list() == [1, 1], (
            "iid='02' matches prihistrow='02' from range 2, so primary_sec=1 should win"
        )


class TestGenReturnsDfDedup:
    """Test that gen_returns_df() keeps the row with the highest prcstd
    when duplicates exist for the same {gvkey, iid, datadate}."""

    def test_keeps_highest_prcstd(self, test_paths) -> None:
        """When two rows share {gvkey, iid, datadate} but differ in prcstd,
        the row with higher prcstd should survive."""
        # Build a minimal __comp_msf.parquet with duplicate rows
        df = pl.DataFrame(
            {
                "gvkey": ["001", "001", "001", "001"],
                "iid": ["01", "01", "01", "01"],
                "datadate": [
                    date(2024, 1, 31),
                    date(2024, 1, 31),  # duplicate
                    date(2024, 2, 29),
                    date(2024, 3, 29),
                ],
                "eom": [
                    date(2024, 1, 31),
                    date(2024, 1, 31),
                    date(2024, 2, 29),
                    date(2024, 3, 29),
                ],
                "prcstd": [3, 10, 4, 4],  # row 0 has prcstd=3, row 1 has prcstd=10
                "ri": [100.0, 105.0, 110.0, 115.0],  # different ri for the duplicate
                "ri_local": [100.0, 105.0, 110.0, 115.0],
                "curcdd": ["USD", "USD", "USD", "USD"],
            }
        )
        df.write_parquet(test_paths.interim_dir / "__comp_msf.parquet")

        from jkp.data.aux_functions import gen_returns_df

        result = gen_returns_df(test_paths, "m")

        # Should have 3 rows (duplicate resolved)
        assert len(result) == 3, f"Expected 3 rows after dedup, got {len(result)}"

        # The first row (Jan 31) should use ri=105.0 (from prcstd=10)
        jan_row = result.filter(pl.col("datadate") == date(2024, 1, 31))
        assert len(jan_row) == 1, "Should have exactly one Jan 31 row"

        # The return for Feb should be (110 - 105) / 105
        feb_row = result.filter(pl.col("datadate") == date(2024, 2, 29))
        expected_ret = (110.0 - 105.0) / 105.0
        actual_ret = feb_row["ret"][0]
        assert actual_ret == pytest.approx(expected_ret, rel=1e-10), (
            f"Feb return should be based on ri=105 (prcstd=10 survivor), "
            f"got {actual_ret}, expected {expected_ret}"
        )

    def test_no_duplicates_unchanged(self, test_paths) -> None:
        """When no duplicates exist, the output should be the same as before."""
        df = pl.DataFrame(
            {
                "gvkey": ["001", "001", "001"],
                "iid": ["01", "01", "01"],
                "datadate": [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 29)],
                "eom": [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 29)],
                "prcstd": [4, 4, 4],
                "ri": [100.0, 110.0, 121.0],
                "ri_local": [100.0, 110.0, 121.0],
                "curcdd": ["USD", "USD", "USD"],
            }
        )
        df.write_parquet(test_paths.interim_dir / "__comp_msf.parquet")

        from jkp.data.aux_functions import gen_returns_df

        result = gen_returns_df(test_paths, "m")
        assert len(result) == 3

        feb_ret = result.filter(pl.col("datadate") == date(2024, 2, 29))["ret"][0]
        assert feb_ret == pytest.approx(0.1, rel=1e-10)
