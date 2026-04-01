"""
Tests for WRDS download functionality.

This module tests the download_raw_data_tables function and its helper functions,
particularly the persistent connection feature that uses ATTACH instead of postgres_scan().
"""

from unittest.mock import MagicMock, patch

import pytest


class TestBuildProjection:
    """Tests for build_projection() function."""

    def test_no_special_columns(self):
        """When no special columns present, return simple wildcard."""
        from aux_functions import build_projection

        cols = ["date", "value", "name"]
        result = build_projection(cols)
        assert result == "*"

    def test_permno_column_cast(self):
        """permno column should be cast to BIGINT."""
        from aux_functions import build_projection

        cols = ["permno", "date", "ret"]
        result = build_projection(cols)
        assert "TRY_CAST(permno AS BIGINT) AS permno" in result
        assert result.startswith("* REPLACE (")

    def test_multiple_special_columns(self):
        """Multiple special columns should all be cast."""
        from aux_functions import build_projection

        cols = ["permno", "permco", "sic", "sich", "date"]
        result = build_projection(cols)
        assert "TRY_CAST(permno AS BIGINT) AS permno" in result
        assert "TRY_CAST(permco AS BIGINT) AS permco" in result
        assert "TRY_CAST(sic AS BIGINT) AS sic" in result
        assert "TRY_CAST(sich AS BIGINT) AS sich" in result


class TestGenWrdsConnectionInfo:
    """Tests for gen_wrds_connection_info() function."""

    def test_connection_string_format(self):
        """Connection string should have correct format."""
        from aux_functions import gen_wrds_connection_info

        result = gen_wrds_connection_info("testuser", "testpass")

        assert "host=wrds-pgdata.wharton.upenn.edu" in result
        assert "port=9737" in result
        assert "dbname=wrds" in result
        assert "user=testuser" in result
        assert "password=testpass" in result
        assert "sslmode=require" in result


class TestDownloadRawDataTablesBranching:
    """Tests for download_raw_data_tables() branching logic.

    These tests verify that the correct download method is used based on
    the persistent_connection parameter.
    """

    @pytest.fixture
    def mock_duckdb(self):
        """Create a mock DuckDB connection."""
        with patch("aux_functions.duckdb") as mock:
            mock_conn = MagicMock()
            mock.connect.return_value = mock_conn
            mock_result = MagicMock()
            mock_result.description = [("col1",), ("col2",)]
            mock_conn.execute.return_value = mock_result
            yield mock, mock_conn

    def test_persistent_connection_false_uses_postgres_scan(self, mock_duckdb):
        """When persistent_connection=False, should use postgres_scan()."""
        from aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables("user", "pass", persistent_connection=False)

        executed_sql = [
            str(c[0][0])
            for c in mock_conn.execute.call_args_list
            if c[0] and isinstance(c[0][0], str)
        ]
        sql_joined = " ".join(executed_sql)

        assert "postgres_scan" in sql_joined
        assert "ATTACH" not in sql_joined

    def test_persistent_connection_true_uses_attach(self, mock_duckdb):
        """When persistent_connection=True, should use ATTACH."""
        from aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables("user", "pass", persistent_connection=True)

        executed_sql = [
            str(c[0][0])
            for c in mock_conn.execute.call_args_list
            if c[0] and isinstance(c[0][0], str)
        ]
        sql_joined = " ".join(executed_sql)

        assert "ATTACH" in sql_joined
        assert "DETACH" in sql_joined
        assert "wrds." in sql_joined

    def test_persistent_connection_true_single_attach(self, mock_duckdb):
        """Persistent connection should only ATTACH once for all tables."""
        from aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables("user", "pass", persistent_connection=True)

        executed_sql = [
            str(c[0][0])
            for c in mock_conn.execute.call_args_list
            if c[0] and isinstance(c[0][0], str)
        ]

        attach_count = sum(1 for sql in executed_sql if "ATTACH" in sql and "DETACH" not in sql)
        detach_count = sum(1 for sql in executed_sql if "DETACH" in sql)

        assert attach_count == 1, f"Expected 1 ATTACH, got {attach_count}"
        assert detach_count == 1, f"Expected 1 DETACH, got {detach_count}"

    def test_connection_closed_after_download(self, mock_duckdb):
        """Connection should be closed after download completes."""
        from aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables("user", "pass", persistent_connection=False)
        mock_conn.close.assert_called_once()

        mock_conn.reset_mock()

        download_raw_data_tables("user", "pass", persistent_connection=True)
        mock_conn.close.assert_called_once()


class TestGetColumnsAttached:
    """Tests for get_columns_attached() function."""

    def test_returns_column_names(self):
        """Should extract column names from query description."""
        from aux_functions import get_columns_attached

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("permno",), ("date",), ("ret",)]
        mock_conn.execute.return_value = mock_result

        result = get_columns_attached(mock_conn, "wrds", "crsp", "msf")

        assert result == ["permno", "date", "ret"]

    def test_queries_attached_database(self):
        """Should query the attached database with correct syntax."""
        from aux_functions import get_columns_attached

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("col1",)]
        mock_conn.execute.return_value = mock_result

        get_columns_attached(mock_conn, "mydb", "mylib", "mytable")

        call_args = mock_conn.execute.call_args[0][0]
        assert "mydb.mylib.mytable" in call_args
        assert "LIMIT 0" in call_args


class TestDownloadWrdsTableAttached:
    """Tests for download_wrds_table_attached() function."""

    def test_copies_to_parquet(self):
        """Should execute COPY TO parquet command."""
        from aux_functions import download_wrds_table_attached

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("col1",), ("col2",)]
        mock_conn.execute.return_value = mock_result

        download_wrds_table_attached(mock_conn, "wrds", "crsp.msf", "/tmp/test.parquet")

        copy_calls = [
            c for c in mock_conn.execute.call_args_list if c[0] and "COPY" in str(c[0][0])
        ]

        assert len(copy_calls) == 1, "Should have exactly one COPY command"
        copy_sql = copy_calls[0][0][0]
        assert "wrds.crsp.msf" in copy_sql
        assert "/tmp/test.parquet" in copy_sql
        assert "FORMAT PARQUET" in copy_sql
