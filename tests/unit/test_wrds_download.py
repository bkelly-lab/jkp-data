"""
Tests for WRDS download functionality.

This module tests the download_raw_data_tables function and its helper functions,
particularly the persistent connection feature that uses ATTACH instead of postgres_scan().
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestBuildProjection:
    """Tests for build_projection() function."""

    def test_no_special_columns(self):
        """When no special columns present, return simple wildcard."""
        from jkp.data.aux_functions import build_projection

        cols = ["date", "value", "name"]
        result = build_projection(cols)
        assert result == "*"

    def test_permno_column_cast(self):
        """permno column should be cast to BIGINT."""
        from jkp.data.aux_functions import build_projection

        cols = ["permno", "date", "ret"]
        result = build_projection(cols)
        assert "TRY_CAST(permno AS BIGINT) AS permno" in result
        assert result.startswith("* REPLACE (")

    def test_multiple_special_columns(self):
        """Multiple special columns should all be cast."""
        from jkp.data.aux_functions import build_projection

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
        from jkp.data.aux_functions import gen_wrds_connection_info

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
        with patch("jkp.data.aux_functions.duckdb") as mock:
            mock_conn = MagicMock()
            mock.connect.return_value = mock_conn
            mock_result = MagicMock()
            mock_result.description = [("col1",), ("col2",)]
            mock_conn.execute.return_value = mock_result
            yield mock, mock_conn

    def test_persistent_connection_false_uses_postgres_scan(self, mock_duckdb, test_paths):
        """When persistent_connection=False, should use postgres_scan()."""
        from jkp.data.aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables(test_paths, "user", "pass", persistent_connection=False)

        executed_sql = [
            str(c[0][0])
            for c in mock_conn.execute.call_args_list
            if c[0] and isinstance(c[0][0], str)
        ]
        sql_joined = " ".join(executed_sql)

        assert "postgres_scan" in sql_joined
        assert "ATTACH" not in sql_joined

    def test_persistent_connection_true_uses_attach(self, mock_duckdb, test_paths):
        """When persistent_connection=True, should use ATTACH."""
        from jkp.data.aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables(test_paths, "user", "pass", persistent_connection=True)

        executed_sql = [
            str(c[0][0])
            for c in mock_conn.execute.call_args_list
            if c[0] and isinstance(c[0][0], str)
        ]
        sql_joined = " ".join(executed_sql)

        assert "ATTACH" in sql_joined
        assert "DETACH" in sql_joined
        assert "wrds." in sql_joined

    def test_persistent_connection_true_single_attach(self, mock_duckdb, test_paths):
        """Persistent connection should only ATTACH once for all tables."""
        from jkp.data.aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables(test_paths, "user", "pass", persistent_connection=True)

        executed_sql = [
            str(c[0][0])
            for c in mock_conn.execute.call_args_list
            if c[0] and isinstance(c[0][0], str)
        ]

        attach_count = sum(1 for sql in executed_sql if "ATTACH" in sql and "DETACH" not in sql)
        detach_count = sum(1 for sql in executed_sql if "DETACH" in sql)

        assert attach_count == 1, f"Expected 1 ATTACH, got {attach_count}"
        assert detach_count == 1, f"Expected 1 DETACH, got {detach_count}"

    def test_connection_closed_after_download(self, mock_duckdb, test_paths):
        """Connection should be closed after download completes."""
        from jkp.data.aux_functions import download_raw_data_tables

        mock, mock_conn = mock_duckdb

        download_raw_data_tables(test_paths, "user", "pass", persistent_connection=False)
        mock_conn.close.assert_called_once()

        mock_conn.reset_mock()

        download_raw_data_tables(test_paths, "user", "pass", persistent_connection=True)
        mock_conn.close.assert_called_once()


class TestGetColumnsAttached:
    """Tests for get_columns_attached() function."""

    def test_returns_column_names(self):
        """Should extract column names from query description."""
        from jkp.data.aux_functions import get_columns_attached

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("permno",), ("date",), ("ret",)]
        mock_conn.execute.return_value = mock_result

        result = get_columns_attached(mock_conn, "wrds", "crsp", "msf")

        assert result == ["permno", "date", "ret"]

    def test_queries_attached_database(self):
        """Should query the attached database with correct syntax."""
        from jkp.data.aux_functions import get_columns_attached

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
        from jkp.data.aux_functions import download_wrds_table_attached

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


class TestEffectiveDownloadWorkers:
    """Tests for _effective_download_workers() clamping logic."""

    def test_one_or_less_means_sequential(self):
        from jkp.data.aux_functions import _effective_download_workers

        assert _effective_download_workers(1, 25) == 1
        assert _effective_download_workers(0, 25) == 1
        assert _effective_download_workers(-3, 25) == 1

    def test_clamped_to_connection_limit(self):
        from jkp.data.aux_functions import WRDS_MAX_CONNECTIONS, _effective_download_workers

        # Leaves one connection of headroom under the WRDS per-account limit.
        assert _effective_download_workers(100, 25) == WRDS_MAX_CONNECTIONS - 1

    def test_never_more_workers_than_tables(self):
        from jkp.data.aux_functions import _effective_download_workers

        assert _effective_download_workers(6, 3) == 3

    def test_passthrough_within_bounds(self):
        from jkp.data.aux_functions import _effective_download_workers

        assert _effective_download_workers(4, 25) == 4


class TestParallelDownload:
    """Tests for the parallel (max_workers > 1) download path."""

    @pytest.fixture
    def mock_duckdb_multi(self):
        """Patch duckdb so each connect() returns a distinct mock connection.

        Each mock's __enter__ returns itself, mirroring a real DuckDB connection used as a
        context manager (``with duckdb.connect() as con`` binds con to the connection), so the
        worker's ATTACH/DETACH calls are recorded on the same mock and __exit__ closes it.
        """
        with patch("jkp.data.aux_functions.duckdb") as mock:
            conns = [MagicMock(name=f"conn{i}") for i in range(32)]
            for c in conns:
                c.__enter__.return_value = c
            mock.connect.side_effect = conns
            yield mock, conns

    def _record_tables(self, mock_dl):
        """Make download_wrds_table_attached record the tables it's asked to download."""
        recorded: list[str] = []
        lock = threading.Lock()

        def record(con, alias, table, filename, **kwargs):
            with lock:
                recorded.append(table)

        mock_dl.side_effect = record
        return recorded

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_parallel_covers_same_tables_as_sequential(
        self, mock_dl, mock_duckdb_multi, test_paths
    ):
        """Parallel download must cover exactly the same tables, each once, as the sequential path."""
        from jkp.data.aux_functions import download_raw_data_tables

        seq = self._record_tables(mock_dl)
        download_raw_data_tables(
            test_paths, "user", "pass", persistent_connection=True, max_workers=1
        )
        sequential_tables = set(seq)

        mock_dl.reset_mock()
        par = self._record_tables(mock_dl)
        download_raw_data_tables(test_paths, "user", "pass", max_workers=4)

        assert len(par) == len(set(par)), "a table was downloaded more than once"
        assert set(par) == sequential_tables
        assert len(sequential_tables) > 0

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_one_connection_per_worker_with_attach_detach(
        self, mock_dl, mock_duckdb_multi, test_paths
    ):
        from jkp.data.aux_functions import download_raw_data_tables

        mock, conns = mock_duckdb_multi
        self._record_tables(mock_dl)

        download_raw_data_tables(test_paths, "user", "pass", max_workers=3)

        assert mock.connect.call_count == 3, "should open exactly one connection per worker"
        for con in conns[:3]:
            sqls = " ".join(
                str(c[0][0])
                for c in con.execute.call_args_list
                if c[0] and isinstance(c[0][0], str)
            )
            assert "ATTACH" in sqls
            assert "DETACH" in sqls
            # Connection is closed via the context manager (`with duckdb.connect()`), i.e. __exit__.
            con.__exit__.assert_called()

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_worker_count_clamped_to_connection_limit(self, mock_dl, mock_duckdb_multi, test_paths):
        from jkp.data.aux_functions import WRDS_MAX_CONNECTIONS, download_raw_data_tables

        mock, _ = mock_duckdb_multi
        self._record_tables(mock_dl)

        download_raw_data_tables(test_paths, "user", "pass", max_workers=50)

        assert mock.connect.call_count == WRDS_MAX_CONNECTIONS - 1

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_table_failure_is_aggregated(self, mock_dl, mock_duckdb_multi, test_paths):
        from jkp.data.aux_functions import download_raw_data_tables

        def maybe_fail(con, alias, table, filename, **kwargs):
            if table == "crsp.dsf_v2":
                raise ValueError("simulated download failure")

        mock_dl.side_effect = maybe_fail

        with pytest.raises(RuntimeError, match="download.*failed"):
            download_raw_data_tables(test_paths, "user", "pass", max_workers=4)

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_password_not_leaked_in_aggregated_error(self, mock_dl, mock_duckdb_multi, test_paths):
        from jkp.data.aux_functions import download_raw_data_tables

        def fail_with_secret(con, alias, table, filename, **kwargs):
            raise ValueError("connection failed for password=hunter2secret while reading")

        mock_dl.side_effect = fail_with_secret

        with pytest.raises(RuntimeError) as exc_info:
            download_raw_data_tables(test_paths, "user", "hunter2secret", max_workers=2)

        assert "hunter2secret" not in str(exc_info.value)

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_persistent_connection_forces_sequential(
        self, mock_dl, mock_duckdb_multi, test_paths, capsys
    ):
        """--persistent-connection must win over max_workers: a single connection, no parallelism.

        Parallel workers would each open a connection (one MFA prompt apiece), defeating the
        single-MFA-prompt purpose of --persistent-connection, so it stays sequential.
        """
        from jkp.data.aux_functions import download_raw_data_tables

        mock, _ = mock_duckdb_multi
        self._record_tables(mock_dl)

        download_raw_data_tables(
            test_paths, "user", "pass", persistent_connection=True, max_workers=6
        )

        # One connection (sequential ATTACH path), not six (parallel pool).
        assert mock.connect.call_count == 1
        # The override notice is a warning -> stderr.
        assert "single WRDS connection" in capsys.readouterr().err
