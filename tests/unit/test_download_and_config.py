"""
Tests for WRDS download functions and config module.

These tests cover:
- download_wrds_table: WHERE clause generation for date-filtered downloads
- download_raw_data_tables: date_columns mapping passed correctly to download_wrds_table
- save_main_data: filtering logic (no end_date parameter)
- config: END_DATE constant

Paper Reference: Jensen, Kelly, Pedersen (2023), "Is There a Replication Crisis in Finance?"
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from config import END_DATE

# =============================================================================
# Tests: config
# =============================================================================


class TestConfig:
    """Tests for the config module constants."""

    def test_end_date_is_date(self):
        """END_DATE should be a datetime.date instance."""
        assert isinstance(END_DATE, date), f"END_DATE should be datetime.date, got {type(END_DATE)}"

    def test_end_date_is_month_end(self):
        """END_DATE should fall on the last day of its month."""
        next_day = date(END_DATE.year + (END_DATE.month // 12), END_DATE.month % 12 + 1, 1)
        last_day = (next_day - __import__("datetime").timedelta(days=1)).day
        assert END_DATE.day == last_day, (
            f"END_DATE ({END_DATE}) is not the last day of its month (expected day {last_day})"
        )


# =============================================================================
# Tests: download_wrds_table
# =============================================================================


class TestDownloadWrdsTable:
    """Tests for download_wrds_table().

    This function downloads a WRDS table via DuckDB postgres_scan, optionally
    filtering rows by a date column. Tests mock DuckDB and get_columns to
    verify the SQL query is constructed correctly.
    """

    @pytest.fixture(autouse=True)
    def _patch_helpers(self):
        """Patch get_columns and build_projection for all tests."""
        with (
            patch("aux_functions.get_columns", return_value=["col_a", "col_b"]),
            patch("aux_functions.build_projection", return_value="*"),
        ):
            yield

    def _run(
        self,
        date_column: str | None = None,
        end_date: date | None = None,
    ) -> str:
        """Call download_wrds_table with a mock conn and return the executed SQL."""
        from aux_functions import download_wrds_table

        mock_conn = MagicMock()
        download_wrds_table(
            conninfo="host=test",
            duckdb_conn=mock_conn,
            table_name="comp.funda",
            filename="out.parquet",
            date_column=date_column,
            end_date=end_date,
        )
        return mock_conn.execute.call_args[0][0]

    def test_no_date_filter_when_params_absent(self):
        """SQL should have no WHERE clause when date_column and end_date are None."""
        sql = self._run()
        assert "WHERE" not in sql, f"Unexpected WHERE clause in SQL: {sql}"

    def test_no_date_filter_when_only_date_column(self):
        """SQL should have no WHERE clause when only date_column is provided."""
        sql = self._run(date_column="datadate")
        assert "WHERE" not in sql, f"Unexpected WHERE clause in SQL: {sql}"

    def test_no_date_filter_when_only_end_date(self):
        """SQL should have no WHERE clause when only end_date is provided."""
        sql = self._run(end_date=date(2025, 12, 31))
        assert "WHERE" not in sql, f"Unexpected WHERE clause in SQL: {sql}"

    def test_where_clause_when_both_params_provided(self):
        """SQL should contain a WHERE clause filtering on the date column."""
        sql = self._run(date_column="datadate", end_date=date(2025, 12, 31))
        assert "WHERE datadate <= '2025-12-31'" in sql, (
            f"Expected WHERE clause with date filter, got: {sql}"
        )

    def test_where_clause_uses_correct_column_name(self):
        """WHERE clause should use the provided date_column name."""
        sql = self._run(date_column="mthcaldt", end_date=date(2024, 6, 30))
        assert "WHERE mthcaldt <= '2024-06-30'" in sql, (
            f"Expected mthcaldt in WHERE clause, got: {sql}"
        )

    def test_sql_targets_correct_table(self):
        """SQL should reference the correct library and table via postgres_scan."""
        sql = self._run()
        assert "'comp'" in sql, f"Expected lib 'comp' in SQL, got: {sql}"
        assert "'funda'" in sql, f"Expected table 'funda' in SQL, got: {sql}"

    def test_sql_outputs_to_correct_filename(self):
        """SQL COPY should target the provided filename."""
        sql = self._run()
        assert "'out.parquet'" in sql, f"Expected filename in SQL, got: {sql}"


# =============================================================================
# Tests: download_raw_data_tables
# =============================================================================


class TestDownloadRawDataTables:
    """Tests for download_raw_data_tables().

    This function orchestrates downloading multiple WRDS tables. Tests mock
    the WRDS connection and download_wrds_table to verify that date filtering
    parameters are passed correctly for each table.
    """

    @pytest.fixture()
    def captured_calls(self):
        """Run download_raw_data_tables and capture all download_wrds_table calls."""
        with (
            patch("aux_functions.gen_wrds_connection_info", return_value="host=test"),
            patch("aux_functions.duckdb") as mock_duckdb,
            patch("aux_functions.download_wrds_table") as mock_download,
        ):
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn

            from aux_functions import download_raw_data_tables

            download_raw_data_tables("user", "pass", end_date=date(2025, 12, 31))
            yield mock_download.call_args_list

    def test_date_filtered_tables_get_date_column(self, captured_calls):
        """Tables with known date columns should receive the date_column kwarg."""
        expected_date_cols = {
            "crsp.msf_v2": "mthcaldt",
            "crsp.dsf_v2": "dlycaldt",
            "comp.secd": "datadate",
            "comp.g_secd": "datadate",
            "comp.secm": "datadate",
            "comp.funda": "datadate",
            "comp.fundq": "datadate",
            "comp.g_funda": "datadate",
            "comp.g_fundq": "datadate",
        }
        for c in captured_calls:
            table_name = c.args[2] if len(c.args) > 2 else c.kwargs.get("table_name")
            date_col = c.kwargs.get("date_column")
            if table_name in expected_date_cols:
                assert date_col == expected_date_cols[table_name], (
                    f"Table {table_name}: expected date_column={expected_date_cols[table_name]}, "
                    f"got {date_col}"
                )

    def test_reference_tables_get_no_date_column(self, captured_calls):
        """Reference/metadata tables should have date_column=None."""
        reference_tables = {
            "comp.exrt_dly",
            "ff.factors_monthly",
            "comp.g_security",
            "comp.security",
            "comp.r_ex_codes",
        }
        for c in captured_calls:
            table_name = c.args[2] if len(c.args) > 2 else c.kwargs.get("table_name")
            if table_name in reference_tables:
                date_col = c.kwargs.get("date_column")
                assert date_col is None, (
                    f"Reference table {table_name} should not have date_column, got {date_col}"
                )

    def test_end_date_passed_to_all_calls(self, captured_calls):
        """Every download_wrds_table call should receive the end_date."""
        for c in captured_calls:
            table_name = c.args[2] if len(c.args) > 2 else c.kwargs.get("table_name")
            end = c.kwargs.get("end_date")
            assert end == date(2025, 12, 31), (
                f"Table {table_name}: expected end_date=2025-12-31, got {end}"
            )

    def test_all_expected_tables_downloaded(self, captured_calls):
        """All tables in the canonical list should be downloaded."""
        downloaded = {
            c.args[2] if len(c.args) > 2 else c.kwargs.get("table_name") for c in captured_calls
        }
        expected_subset = {"comp.funda", "crsp.msf_v2", "crsp.dsf_v2", "comp.secd"}
        assert expected_subset <= downloaded, f"Missing tables: {expected_subset - downloaded}"


# =============================================================================
# Tests: save_main_data
# =============================================================================


class TestSaveMainData:
    """Tests for save_main_data().

    This function filters world_data to main securities. The key change
    is the removal of the end_date parameter — data is now pre-filtered
    at download time. Tests verify the filter conditions.
    """

    def test_no_end_date_parameter(self):
        """save_main_data should accept no arguments (end_date removed)."""
        import inspect

        from aux_functions import save_main_data

        # measure_time wraps the function; inspect the inner function via closure
        inner_func = save_main_data.__closure__[0].cell_contents
        sig = inspect.signature(inner_func)
        assert len(sig.parameters) == 0, (
            f"save_main_data should take no parameters, got: {list(sig.parameters)}"
        )

    def _run_save_main_data(self, tmp_path: Path) -> None:
        """Chdir to tmp_path, run save_main_data, and restore cwd."""
        import os

        from aux_functions import save_main_data

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            with (
                patch("aux_functions.os.chdir"),
                patch("aux_functions.os.system"),
                patch("aux_functions.duckdb") as mock_duckdb,
            ):
                mock_duckdb.connect.return_value = MagicMock()
                save_main_data()
        finally:
            os.chdir(original_cwd)

    def test_filters_to_main_securities(self, tmp_path):
        """Output should only contain rows where primary_sec, common, obs_main, exch_main are all 1."""
        world_data = pl.DataFrame(
            {
                "id": ["A", "B", "C", "D"],
                "eom": [date(2020, 1, 31)] * 4,
                "me": [100.0, 200.0, 300.0, 400.0],
                "primary_sec": [1, 0, 1, 1],
                "common": [1, 1, 0, 1],
                "obs_main": [1, 1, 1, 0],
                "exch_main": [1, 1, 1, 1],
                "excntry": ["USA"] * 4,
            }
        )
        world_data.write_parquet(tmp_path / "world_data.parquet")
        self._run_save_main_data(tmp_path)

        output = pl.read_parquet(tmp_path / "world_data_filtered.parquet")
        assert len(output) == 1, f"Expected 1 row after filtering, got {len(output)}"
        assert output["id"][0] == "A", f"Expected row A, got {output['id'][0]}"

    def test_no_eom_date_filter(self, tmp_path):
        """All dates should pass through — there should be no eom <= end_date filter."""
        world_data = pl.DataFrame(
            {
                "id": ["A", "A"],
                "eom": [date(2020, 1, 31), date(2099, 12, 31)],
                "me": [100.0, 200.0],
                "primary_sec": [1, 1],
                "common": [1, 1],
                "obs_main": [1, 1],
                "exch_main": [1, 1],
                "excntry": ["USA"] * 2,
            }
        )
        world_data.write_parquet(tmp_path / "world_data.parquet")
        self._run_save_main_data(tmp_path)

        output = pl.read_parquet(tmp_path / "world_data_filtered.parquet")
        assert len(output) == 2, f"Expected both rows (no date filter), got {len(output)}"
