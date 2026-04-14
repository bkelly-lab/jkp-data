"""
Tests for aug_msf_v2() idempotency and output correctness.

Validates the fix for #56: aug_msf_v2 writes to raw_data_dfs/crsp_msf_v2_aug.parquet
(separate from raw input) so re-runs don't duplicate columns.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from aux_functions import aug_msf_v2


def _write_aug_msf_fixtures(raw_tables: Path) -> None:
    """Write minimal msf_v2 and dsf_v2 parquet fixtures for aug_msf_v2()."""
    # Monthly file: two months for one stock, one with TR pricing, one with BA
    pl.DataFrame(
        {
            "permno": [10001, 10001],
            "yyyymm": [202001, 202002],
            "mthcaldt": [date(2020, 1, 31), date(2020, 2, 29)],
            "mthprc": [100.0, 50.0],
            "mthprcflg": ["TR", "BA"],
            "mthret": [0.05, -0.02],
        }
    ).write_parquet(raw_tables / "crsp_msf_v2.parquet")

    # Daily file: several days in Jan 2020 for the same stock
    pl.DataFrame(
        {
            "permno": [10001, 10001, 10001],
            "dlycaldt": [date(2020, 1, 2), date(2020, 1, 15), date(2020, 1, 31)],
            "dlyprc": [95.0, 105.0, 100.0],
            "dlyprcflg": ["TR", "TR", "TR"],
        }
    ).write_parquet(raw_tables / "crsp_dsf_v2.parquet")


class TestAugMsfV2:
    """Tests for aug_msf_v2()."""

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        temp_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Set up fixtures and chdir for all tests in this class."""
        raw_tables = temp_data_dir / "raw" / "raw_tables"
        self.code_dir = temp_data_dir / "code"
        self.code_dir.mkdir()
        (self.code_dir / "raw_data_dfs").mkdir()
        _write_aug_msf_fixtures(raw_tables)
        monkeypatch.chdir(self.code_dir)

    def test_output_has_askhi_bidlo_columns(self) -> None:
        """aug_msf_v2() output should contain mthaskhi and mthbidlo columns."""
        aug_msf_v2()

        result = pl.read_parquet(self.code_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")
        assert "mthaskhi" in result.columns
        assert "mthbidlo" in result.columns

    def test_askhi_bidlo_values_for_tr_rows(self) -> None:
        """For TR-flagged months, mthaskhi/mthbidlo should reflect daily high/low."""
        aug_msf_v2()

        result = pl.read_parquet(self.code_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")
        tr_row = result.filter(pl.col("mthprcflg") == "TR")
        assert tr_row["mthaskhi"][0] == 105.0  # max of 95, 105, 100
        assert tr_row["mthbidlo"][0] == 95.0  # min of 95, 105, 100

    def test_askhi_bidlo_null_for_non_tr_rows(self) -> None:
        """For non-TR months, mthaskhi/mthbidlo should be null."""
        aug_msf_v2()

        result = pl.read_parquet(self.code_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")
        ba_row = result.filter(pl.col("mthprcflg") == "BA")
        assert ba_row["mthaskhi"][0] is None
        assert ba_row["mthbidlo"][0] is None

    def test_idempotent_on_rerun(self) -> None:
        """Running aug_msf_v2() twice should produce identical output (fix for #56)."""
        aug_msf_v2()
        first_run = pl.read_parquet(self.code_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")

        aug_msf_v2()
        second_run = pl.read_parquet(self.code_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")

        assert first_run.columns == second_run.columns, (
            f"Column mismatch between runs: {first_run.columns} vs {second_run.columns}"
        )
        assert first_run.shape == second_run.shape
        assert first_run.equals(second_run)

    def test_preserves_original_columns(self) -> None:
        """aug_msf_v2() should preserve all original msf_v2 columns in the output."""
        original = pl.read_parquet("../raw/raw_tables/crsp_msf_v2.parquet")

        aug_msf_v2()

        result = pl.read_parquet(self.code_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")
        for col_name in original.columns:
            assert col_name in result.columns, f"Original column {col_name!r} missing from output"
