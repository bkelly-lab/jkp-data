"""
Tests for WRDS download functions, config module, and filter functions.

These tests cover:
- download_wrds_table: WHERE clause generation for date-filtered downloads
- download_raw_data_tables: date_columns mapping passed correctly to download_wrds_table
- save_main_data: me_lag1 computation and output (no filtering, no end_date parameter)
- filter_dsf, filter_msf, filter_world: MAIN_FILTERS screening
- config: END_DATE and MAIN_FILTERS constants

Paper Reference: Jensen, Kelly, Pedersen (2023), "Is There a Replication Crisis in Finance?"
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from jkp.data.config import END_DATE, MAIN_FILTERS

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

    def test_main_filters_is_dict(self):
        """MAIN_FILTERS should be a dict."""
        assert isinstance(MAIN_FILTERS, dict), (
            f"MAIN_FILTERS should be a dict, got {type(MAIN_FILTERS)}"
        )

    def test_main_filters_has_expected_keys(self):
        """MAIN_FILTERS should contain the four standard screening columns."""
        expected = {"primary_sec", "common", "obs_main", "exch_main"}
        assert set(MAIN_FILTERS.keys()) == expected, (
            f"MAIN_FILTERS keys should be {expected}, got {set(MAIN_FILTERS.keys())}"
        )

    def test_main_filters_values_are_one(self):
        """All MAIN_FILTERS values should be 1 (the passing value)."""
        for k, v in MAIN_FILTERS.items():
            assert v == 1, f"MAIN_FILTERS['{k}'] should be 1, got {v}"


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
            patch("jkp.data.aux_functions.get_columns", return_value=["col_a", "col_b"]),
            patch("jkp.data.aux_functions.build_projection", return_value="*"),
        ):
            yield

    def _run(
        self,
        date_column: str | None = None,
        end_date: date | None = None,
    ) -> str:
        """Call download_wrds_table with a mock conn and return the executed SQL."""
        from jkp.data.aux_functions import download_wrds_table

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
            patch("jkp.data.aux_functions.gen_wrds_connection_info", return_value="host=test"),
            patch("jkp.data.aux_functions.duckdb") as mock_duckdb,
            patch("jkp.data.aux_functions.download_wrds_table") as mock_download,
        ):
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn

            from jkp.data.aux_functions import download_raw_data_tables

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

    This function computes lagged market equity and exports country-level files.
    Filtering is now done upstream by filter_world(). Tests verify me_lag1
    computation and that all rows pass through.
    """

    def test_accepts_paths_parameter(self):
        """save_main_data should accept a paths parameter."""
        import inspect

        from jkp.data.aux_functions import save_main_data

        # measure_time wraps the function; inspect the inner function via closure
        inner_func = save_main_data.__closure__[0].cell_contents
        sig = inspect.signature(inner_func)
        assert "paths" in sig.parameters, (
            f"save_main_data should accept 'paths' parameter, got: {list(sig.parameters)}"
        )

    def _run_save_main_data(self, tmp_path: Path) -> None:
        """Chdir to tmp_path, run save_main_data, and restore cwd."""
        import os

        from jkp.data.aux_functions import save_main_data
        from jkp.data.paths import DataPaths

        paths = DataPaths(base_dir=tmp_path)

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            with (
                patch("jkp.data.aux_functions.os.chdir"),
                patch("jkp.data.aux_functions.os.system"),
                patch("jkp.data.aux_functions.duckdb") as mock_duckdb,
            ):
                mock_duckdb.connect.return_value = MagicMock()
                save_main_data(paths)
        finally:
            os.chdir(original_cwd)

    def test_all_rows_pass_through(self, tmp_path):
        """save_main_data should not filter — all rows should appear in output."""
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
        world_data.write_parquet(tmp_path / "world_data_output.parquet")
        self._run_save_main_data(tmp_path)

        output = pl.read_parquet(tmp_path / "world_data_output.parquet")
        assert len(output) == 4, f"Expected all 4 rows (no filtering), got {len(output)}"

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
        world_data.write_parquet(tmp_path / "world_data_output.parquet")
        self._run_save_main_data(tmp_path)

        output = pl.read_parquet(tmp_path / "world_data_output.parquet")
        assert len(output) == 2, f"Expected both rows (no date filter), got {len(output)}"

    def test_me_lag1_computed(self, tmp_path):
        """save_main_data should add me_lag1 column with lagged market equity."""
        world_data = pl.DataFrame(
            {
                "id": ["A", "A", "A"],
                "eom": [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)],
                "me": [100.0, 200.0, 300.0],
                "primary_sec": [1, 1, 1],
                "common": [1, 1, 1],
                "obs_main": [1, 1, 1],
                "exch_main": [1, 1, 1],
                "excntry": ["USA"] * 3,
            }
        )
        world_data.write_parquet(tmp_path / "world_data_output.parquet")
        self._run_save_main_data(tmp_path)

        output = pl.read_parquet(tmp_path / "world_data_output.parquet").sort("eom")
        assert "me_lag1" in output.columns, "me_lag1 column should be present"
        assert output["me_lag1"][0] is None or output["me_lag1"][0] != output["me_lag1"][0]
        assert output["me_lag1"][1] == pytest.approx(100.0)
        assert output["me_lag1"][2] == pytest.approx(200.0)


# =============================================================================
# Tests: filter functions
# =============================================================================


class TestFilterFunctions:
    """Tests for filter_dsf(), filter_msf(), and filter_world().

    These functions apply MAIN_FILTERS screening to interim parquet files,
    keeping only rows where all four filter columns equal 1.
    """

    @staticmethod
    def _make_test_data() -> pl.DataFrame:
        """Create a toy DataFrame with mixed filter values."""
        return pl.DataFrame(
            {
                "id": ["A", "B", "C", "D", "E"],
                "eom": [date(2020, 1, 31)] * 5,
                "me": [100.0, 200.0, 300.0, 400.0, 500.0],
                "ret": [0.01, 0.02, 0.03, 0.04, 0.05],
                "primary_sec": [1, 0, 1, 1, 1],
                "common": [1, 1, 0, 1, 1],
                "obs_main": [1, 1, 1, 0, 1],
                "exch_main": [1, 1, 1, 1, 0],
                "excntry": ["USA"] * 5,
            }
        )

    def _run_filter(self, tmp_path: Path, func_name: str, source: str, output: str) -> pl.DataFrame:
        """Write test data, run a filter function, and return the result."""
        import os

        import jkp.data.aux_functions as aux_functions

        data = self._make_test_data()
        data.write_parquet(tmp_path / source)

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            getattr(aux_functions, func_name)()
        finally:
            os.chdir(original_cwd)

        return pl.read_parquet(tmp_path / output)

    def test_filter_dsf_keeps_only_passing_rows(self, tmp_path):
        """filter_dsf should keep only rows where all four filter columns are 1."""
        result = self._run_filter(
            tmp_path, "filter_dsf", "world_dsf.parquet", "world_dsf_output.parquet"
        )
        assert len(result) == 1, f"Expected 1 passing row, got {len(result)}"
        assert result["id"][0] == "A"

    def test_filter_msf_keeps_only_passing_rows(self, tmp_path):
        """filter_msf should keep only rows where all four filter columns are 1."""
        result = self._run_filter(
            tmp_path, "filter_msf", "world_msf.parquet", "world_msf_output.parquet"
        )
        assert len(result) == 1, f"Expected 1 passing row, got {len(result)}"
        assert result["id"][0] == "A"

    def test_filter_world_keeps_only_passing_rows(self, tmp_path):
        """filter_world should keep only rows where all four filter columns are 1."""
        result = self._run_filter(
            tmp_path, "filter_world", "world_data.parquet", "world_data_output.parquet"
        )
        assert len(result) == 1, f"Expected 1 passing row, got {len(result)}"
        assert result["id"][0] == "A"

    def test_filter_preserves_all_columns(self, tmp_path):
        """Filtered output should retain all original columns."""
        original_cols = set(self._make_test_data().columns)
        result = self._run_filter(
            tmp_path, "filter_world", "world_data.parquet", "world_data_output.parquet"
        )
        assert set(result.columns) == original_cols, (
            f"Column mismatch: expected {original_cols}, got {set(result.columns)}"
        )

    def test_filter_does_not_modify_source(self, tmp_path):
        """Source file should be unchanged after filtering."""
        import os

        import jkp.data.aux_functions as aux_functions

        data = self._make_test_data()
        data.write_parquet(tmp_path / "world_data.parquet")

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            aux_functions.filter_world()
        finally:
            os.chdir(original_cwd)

        source = pl.read_parquet(tmp_path / "world_data.parquet")
        assert len(source) == 5, f"Source should be unchanged (5 rows), got {len(source)}"

    def test_filter_is_idempotent(self, tmp_path):
        """Running filter twice should produce the same result."""
        import os

        import jkp.data.aux_functions as aux_functions

        data = self._make_test_data()
        data.write_parquet(tmp_path / "world_data.parquet")

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            aux_functions.filter_world()
            first_run = pl.read_parquet(tmp_path / "world_data_output.parquet")
            aux_functions.filter_world()
            second_run = pl.read_parquet(tmp_path / "world_data_output.parquet")
        finally:
            os.chdir(original_cwd)

        assert first_run.equals(second_run), "Second run should produce identical output"

    def test_filter_all_pass(self, tmp_path):
        """When all rows pass the filter, all should be retained."""
        import os

        import jkp.data.aux_functions as aux_functions

        data = pl.DataFrame(
            {
                "id": ["A", "B"],
                "eom": [date(2020, 1, 31)] * 2,
                "me": [100.0, 200.0],
                "ret": [0.01, 0.02],
                "primary_sec": [1, 1],
                "common": [1, 1],
                "obs_main": [1, 1],
                "exch_main": [1, 1],
                "excntry": ["USA"] * 2,
            }
        )
        data.write_parquet(tmp_path / "world_data.parquet")

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            aux_functions.filter_world()
        finally:
            os.chdir(original_cwd)

        result = pl.read_parquet(tmp_path / "world_data_output.parquet")
        assert len(result) == 2, f"Expected both rows to pass, got {len(result)}"

    def test_filter_none_pass(self, tmp_path):
        """When no rows pass the filter, output should be empty."""
        import os

        import jkp.data.aux_functions as aux_functions

        data = pl.DataFrame(
            {
                "id": ["A"],
                "eom": [date(2020, 1, 31)],
                "me": [100.0],
                "ret": [0.01],
                "primary_sec": [0],
                "common": [0],
                "obs_main": [0],
                "exch_main": [0],
                "excntry": ["USA"],
            }
        )
        data.write_parquet(tmp_path / "world_data.parquet")

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            aux_functions.filter_world()
        finally:
            os.chdir(original_cwd)

        result = pl.read_parquet(tmp_path / "world_data_output.parquet")
        assert len(result) == 0, f"Expected 0 rows, got {len(result)}"
