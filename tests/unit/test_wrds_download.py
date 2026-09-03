"""
Tests for WRDS download functionality.

This module tests the download_raw_data_tables function and its helper functions,
particularly the persistent connection feature that uses ATTACH instead of postgres_scan().
"""

import datetime as dt
import threading
from collections import Counter
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


@pytest.fixture
def mock_duckdb_multi():
    """Patch duckdb so each connect() returns a distinct mock connection.

    Each mock's __enter__ returns itself, mirroring a real DuckDB connection used as a
    context manager (``with duckdb.connect() as con`` binds con to the connection), so the
    worker's ATTACH/DETACH calls are recorded on the same mock and __exit__ closes it.

    Module-scoped rather than nested in one test class: every test that reaches the
    download path needs the dual patch below, and hand-rolled single-module patches
    silently fall through to a real INSTALL.
    """
    # Patch duckdb in both modules that open connections on the download path:
    # the per-worker ATTACH connections live in aux_functions, while the one
    # up-front INSTALL connection is opened by wrds_connection._install_postgres_extension.
    # Pointing both names at the same mock keeps a single ordered connect() sequence
    # (install consumes conns[0], workers consume conns[1:]).
    with (
        patch("jkp.data.aux_functions.duckdb") as mock,
        patch("jkp.data.wrds_connection.duckdb", mock),
    ):
        conns = [MagicMock(name=f"conn{i}") for i in range(32)]
        for c in conns:
            c.__enter__.return_value = c
        mock.connect.side_effect = conns
        yield mock, conns


class TestParallelDownload:
    """Tests for the parallel (max_workers > 1) download path."""

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

        # One up-front connection to INSTALL the extension, then one per worker.
        assert mock.connect.call_count == 1 + 3
        install_sqls = " ".join(
            str(c[0][0])
            for c in conns[0].execute.call_args_list
            if c[0] and isinstance(c[0][0], str)
        )
        assert "INSTALL" in install_sqls and "ATTACH" not in install_sqls  # install-only, no WRDS
        for con in conns[1:4]:  # the three worker connections
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

        # Clamped worker connections, plus the one up-front extension-install connection.
        assert mock.connect.call_count == (WRDS_MAX_CONNECTIONS - 1) + 1

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

        err = str(exc_info.value)
        assert "hunter2secret" not in err  # credential redacted
        assert "***" in err  # ... replaced in place
        assert "connection failed" in err and "while reading" in err  # ... diagnostic preserved

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_special_char_password_not_leaked_in_aggregated_error(
        self, mock_dl, mock_duckdb_multi, test_paths
    ):
        """A password with a quote/backslash appears in DuckDB errors in its
        libpq-escaped form, so redaction must scrub that form too — a raw
        ``str(e).replace(password, ...)`` would miss it and leak the secret."""
        from jkp.data.aux_functions import download_raw_data_tables

        def fail_with_secret(con, alias, table, filename, **kwargs):
            # the libpq-escaped form is what a real DuckDB error would contain
            raise ValueError(r"connection failed for password='ab\'cd' while reading")

        mock_dl.side_effect = fail_with_secret

        with pytest.raises(RuntimeError) as exc_info:
            download_raw_data_tables(test_paths, "user", "ab'cd", max_workers=2)

        err = str(exc_info.value)
        assert "ab'cd" not in err  # raw form absent
        assert r"ab\'cd" not in err  # libpq-escaped form absent (the form that appeared)
        assert "***" in err
        assert "connection failed" in err and "while reading" in err

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    @patch("jkp.data.aux_functions._attach_wrds")
    def test_one_attach_failure_is_non_fatal_when_queue_drains(
        self, mock_attach, mock_dl, mock_duckdb_multi, test_paths, capsys
    ):
        """One worker failing to attach must not fail the run if survivors download every table."""
        from jkp.data.aux_functions import download_raw_data_tables

        seen = {"n": 0}
        seen_lock = threading.Lock()

        def attach(con, conninfo, password):
            with seen_lock:
                seen["n"] += 1
                first = seen["n"] == 1
            if first:  # exactly one worker fails to attach
                raise RuntimeError("Failed to attach WRDS connection. Check credentials and MFA.")

        mock_attach.side_effect = attach
        self._record_tables(mock_dl)

        # Must NOT raise: the other workers drain the shared queue.
        download_raw_data_tables(test_paths, "user", "pass", max_workers=3)

        err = capsys.readouterr().err
        assert "failed to start" in err  # warned (immediately + in the summary)

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    @patch("jkp.data.aux_functions._attach_wrds")
    def test_all_attach_failures_raise(self, mock_attach, mock_dl, mock_duckdb_multi, test_paths):
        """If every worker fails to attach, the queue never drains -> raise (nothing downloaded)."""
        from jkp.data.aux_functions import download_raw_data_tables

        mock_attach.side_effect = RuntimeError("Failed to attach WRDS connection.")
        with pytest.raises(RuntimeError, match="failed to start"):
            download_raw_data_tables(test_paths, "user", "pass", max_workers=3)
        mock_dl.assert_not_called()  # no table was ever downloaded

    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    @patch("jkp.data.aux_functions._attach_wrds")
    @patch("jkp.data.aux_functions.duckdb")
    def test_worker_respects_stop_event(self, mock_duckdb, mock_attach, mock_dl):
        """A set stop_event makes a worker exit its drain loop without pulling any task (Ctrl-C path)."""
        import queue as _queue

        from jkp.data.aux_functions import _attach_download_worker, _DownloadTask

        con = MagicMock()
        con.__enter__.return_value = con
        mock_duckdb.connect.return_value = con
        task_queue: _queue.Queue = _queue.Queue()
        task_queue.put(_DownloadTask("crsp.dsf_v2", "/tmp/x.parquet", None, None, None))
        stop_event = threading.Event()
        stop_event.set()  # already asked to stop

        _attach_download_worker(task_queue, "conninfo", "pw", [], [], threading.Lock(), stop_event)

        mock_dl.assert_not_called()  # nothing pulled/downloaded
        assert task_queue.qsize() == 1  # task left un-pulled

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


class TestDateRangeSplitting:
    """Tests for splitting the giant daily tables into row-balanced date-range chunks."""

    def test_date_where_clause(self):
        from jkp.data.aux_functions import _date_where

        assert _date_where(None, None, None) == ""
        assert _date_where("d", None, dt.date(2025, 12, 31)) == "WHERE d <= '2025-12-31'"
        assert _date_where("d", dt.date(2020, 1, 1), None) == "WHERE d >= '2020-01-01'"
        assert (
            _date_where("d", dt.date(2020, 1, 1), dt.date(2025, 12, 31))
            == "WHERE d >= '2020-01-01' AND d <= '2025-12-31'"
        )

    def test_download_attached_emits_date_range(self):
        from jkp.data.aux_functions import download_wrds_table_attached

        con = MagicMock()
        result = MagicMock()
        result.description = [("dlycaldt",), ("x",)]
        con.execute.return_value = result

        download_wrds_table_attached(
            con,
            "wrds",
            "crsp.dsf_v2",
            "/tmp/c.parquet",
            date_column="dlycaldt",
            start_date=dt.date(2020, 1, 1),
            end_date=dt.date(2025, 12, 31),
        )

        copy_sql = next(
            str(c[0][0]) for c in con.execute.call_args_list if c[0] and "COPY" in str(c[0][0])
        )
        assert "dlycaldt >= '2020-01-01'" in copy_sql
        assert "dlycaldt <= '2025-12-31'" in copy_sql

    def test_balanced_chunks_cover_and_balance(self):
        from jkp.data.aux_functions import _balanced_year_chunks

        # Increasing rows per year (recent years denser), like the real daily tables.
        hist = [(y, y - 1924) for y in range(1925, 2026)]
        end = dt.date(2025, 12, 31)
        ranges = _balanced_year_chunks(hist, 6, end)

        assert len(ranges) == 6
        assert ranges[0][0] is None  # unbounded below
        assert ranges[-1][1] == end  # last bounded by overall end_date
        # Contiguous and non-overlapping (year boundaries line up).
        for (_s1, e1), (s2, _e2) in zip(ranges[:-1], ranges[1:], strict=True):
            assert s2.year == e1.year + 1

        def rows(start, end_):
            return sum(n for y, n in hist if (start is None or y >= start.year) and y <= end_.year)

        sizes = [rows(s, e) for s, e in ranges]
        assert sum(sizes) == sum(n for _, n in hist)  # complete coverage, no double count
        target = sum(sizes) / 6
        assert max(sizes) <= 1.5 * target  # reasonably balanced

    def test_balanced_chunks_degenerate_cases(self):
        from jkp.data.aux_functions import _balanced_year_chunks

        end = dt.date(2025, 12, 31)
        assert _balanced_year_chunks([], 6, end) == [(None, end)]
        assert _balanced_year_chunks([(2025, 100)], 6, end) == [(None, end)]  # single year
        assert _balanced_year_chunks([(y, 1) for y in range(2000, 2010)], 1, end) == [(None, end)]

    def test_build_tasks_splits_only_split_tables(self):
        from jkp.data.aux_functions import _build_download_tasks

        end = dt.date(2025, 12, 31)
        hist = [(y, 10) for y in range(2010, 2022)]  # 12 equal years
        tables = ["ff.factors_monthly", "crsp.dsf_v2"]
        filenames = {
            "ff.factors_monthly": "/o/ff_factors_monthly.parquet",
            "crsp.dsf_v2": "/o/crsp_dsf_v2.parquet",
        }
        tasks, concat_map = _build_download_tasks(
            tables,
            filenames,
            {"crsp.dsf_v2": "dlycaldt"},
            end,
            frozenset({"crsp.dsf_v2"}),
            4,
            {"crsp.dsf_v2": hist},
        )

        ff = [t for t in tasks if t.table == "ff.factors_monthly"]
        dsf = [t for t in tasks if t.table == "crsp.dsf_v2"]
        assert len(ff) == 1 and ff[0].out == filenames["ff.factors_monthly"]
        assert len(dsf) == 4  # split into 4 chunks
        assert all(".part" in t.out for t in dsf)
        assert dsf[0].start_date is None and dsf[-1].end_date == end
        assert concat_map[filenames["crsp.dsf_v2"]] == [t.out for t in dsf]

    def test_build_tasks_no_split_without_histogram(self):
        from jkp.data.aux_functions import _build_download_tasks

        # No histogram (e.g. end_date was None upstream) -> single whole-table task, no concat.
        tasks, concat_map = _build_download_tasks(
            ["crsp.dsf_v2"],
            {"crsp.dsf_v2": "/o/crsp_dsf_v2.parquet"},
            {"crsp.dsf_v2": "dlycaldt"},
            None,
            frozenset({"crsp.dsf_v2"}),
            4,
            {},
        )
        assert len(tasks) == 1
        assert tasks[0].out == "/o/crsp_dsf_v2.parquet"
        assert not concat_map

    def test_concat_chunks_combines_and_removes(self, tmp_path):
        import polars as pl

        from jkp.data.aux_functions import _concat_chunks

        c0 = tmp_path / "t.part00.parquet"
        c1 = tmp_path / "t.part01.parquet"
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_parquet(c0)
        pl.DataFrame({"a": [3], "b": ["z"]}).write_parquet(c1)
        final = tmp_path / "t.parquet"

        _concat_chunks(str(final), [str(c0), str(c1)])

        out = pl.read_parquet(final)
        assert out.height == 3
        assert out["a"].to_list() == [1, 2, 3]  # chunk order preserved
        assert not c0.exists() and not c1.exists()  # chunks cleaned up

    def test_remove_chunk_parts(self, tmp_path):
        from jkp.data.aux_functions import _remove_chunk_parts

        final = tmp_path / "crsp_dsf_v2.parquet"
        final.write_bytes(b"keep")
        (tmp_path / "crsp_dsf_v2.part00.parquet").write_bytes(b"x")
        (tmp_path / "crsp_dsf_v2.part09.parquet").write_bytes(b"x")  # high index from a prior run
        (tmp_path / "crsp_dsf_v2.partial.parquet").write_bytes(b"nope")  # must NOT be swept
        (tmp_path / "comp_secd.part00.parquet").write_bytes(b"other")  # different table

        _remove_chunk_parts(str(final))

        assert not (tmp_path / "crsp_dsf_v2.part00.parquet").exists()
        assert not (tmp_path / "crsp_dsf_v2.part09.parquet").exists()
        assert final.exists()  # the final (non-part) file is left untouched
        assert (tmp_path / "crsp_dsf_v2.partial.parquet").exists()  # `part[0-9][0-9]` excludes this
        assert (tmp_path / "comp_secd.part00.parquet").exists()  # other tables untouched

    @patch("jkp.data.aux_functions._concat_chunks")
    @patch("jkp.data.aux_functions._compute_histograms")
    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_parallel_download_aggregates_concat_failures(
        self, mock_dl, mock_hist, mock_concat, mock_duckdb_multi, test_paths
    ):
        """A concat failure surfaces as one aggregated error, not a raw first-failure exception."""
        from jkp.data.aux_functions import SPLIT_TABLES, download_raw_data_tables

        mock_hist.return_value = dict.fromkeys(SPLIT_TABLES, [(y, 10) for y in range(2010, 2022)])
        mock_concat.side_effect = RuntimeError("disk full")

        with pytest.raises(RuntimeError, match="concatenation"):
            download_raw_data_tables(
                test_paths, "user", "pass", end_date=dt.date(2025, 12, 31), max_workers=4
            )

    @patch("jkp.data.aux_functions._concat_chunks")
    @patch("jkp.data.aux_functions._compute_histograms")
    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_parallel_download_splits_giant_tables(
        self, mock_dl, mock_hist, mock_concat, mock_duckdb_multi, test_paths
    ):
        """End-to-end (mocked): with an end_date, the giant tables expand into max_workers chunks."""
        from jkp.data.aux_functions import SPLIT_TABLES, download_raw_data_tables

        hist = [(y, 10) for y in range(2010, 2022)]
        mock_hist.return_value = dict.fromkeys(SPLIT_TABLES, hist)

        recorded: list[str] = []
        lock = threading.Lock()

        def record(con, alias, table, out, **kwargs):
            with lock:
                recorded.append(table)

        mock_dl.side_effect = record

        download_raw_data_tables(
            test_paths, "user", "pass", end_date=dt.date(2025, 12, 31), max_workers=4
        )

        counts = Counter(recorded)
        for split_table in SPLIT_TABLES:
            assert counts[split_table] == 4, (
                f"{split_table}: expected 4 chunks, got {counts[split_table]}"
            )
        assert counts["ff.factors_monthly"] == 1  # non-split table downloaded once

    def test_balanced_chunks_skewed_distribution(self):
        # One year dwarfs the rest (its count exceeds the per-chunk target): chunks must still
        # cover everything exactly once with contiguous, non-empty year ranges.
        from jkp.data.aux_functions import _balanced_year_chunks

        end = dt.date(2025, 12, 31)
        hist = [(2000, 1), (2001, 1), (2002, 1000), (2003, 1), (2004, 1)]
        ranges = _balanced_year_chunks(hist, 4, end)

        assert ranges[0][0] is None
        assert ranges[-1][1] == end
        for (_s1, e1), (s2, _e2) in zip(ranges[:-1], ranges[1:], strict=True):
            assert s2.year == e1.year + 1  # contiguous, non-overlapping
        for s, e in ranges:
            if s is not None:
                assert s <= e  # no empty/inverted spans

        def rows(start, end_):
            return sum(n for y, n in hist if (start is None or y >= start.year) and y <= end_.year)

        assert sum(rows(s, e) for s, e in ranges) == sum(n for _, n in hist)  # complete coverage

    @patch("jkp.data.aux_functions._concat_chunks")
    @patch("jkp.data.aux_functions._compute_histograms")
    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_parallel_download_concatenates_each_split_table(
        self, mock_dl, mock_hist, mock_concat, mock_duckdb_multi, test_paths
    ):
        """The concat_map -> ThreadPoolExecutor wiring invokes _concat_chunks once per split table."""
        from jkp.data.aux_functions import SPLIT_TABLES, download_raw_data_tables

        mock_hist.return_value = dict.fromkeys(SPLIT_TABLES, [(y, 10) for y in range(2010, 2022)])
        download_raw_data_tables(
            test_paths, "user", "pass", end_date=dt.date(2025, 12, 31), max_workers=4
        )

        assert mock_concat.call_count == len(SPLIT_TABLES)
        for call in mock_concat.call_args_list:
            _final_file, chunk_files = call.args
            assert len(chunk_files) == 4
            assert chunk_files == sorted(chunk_files)  # chunks concatenated in part00..part03 order
            assert all(".part" in cf for cf in chunk_files)

    def test_compute_histograms_attach_failure_is_redacted(self):
        """A histogram-phase ATTACH failure must not leak the password (regression for M1)."""
        from jkp.data.aux_functions import _compute_histograms

        password = "topsecret"  # noqa: S105

        def execute(sql, *args):
            if "ATTACH" in sql:
                raise RuntimeError(f"Connection failed: host=x password={password} dbname=wrds")
            return MagicMock()

        with patch("jkp.data.aux_functions.duckdb") as mock_duckdb:
            con = MagicMock()
            con.__enter__.return_value = con
            con.execute.side_effect = execute
            mock_duckdb.connect.return_value = con
            with pytest.raises(Exception) as exc_info:  # noqa: PT011
                _compute_histograms(
                    f"host=x password={password}",
                    ["crsp.dsf_v2"],
                    {"crsp.dsf_v2": "dlycaldt"},
                    dt.date(2025, 12, 31),
                    4,
                    password,
                )
        assert password not in str(exc_info.value)


class TestAttachWrds:
    """The shared WRDS ATTACH helper (password redaction)."""

    def test_redacts_password_on_attach_error(self):
        from jkp.data.aux_functions import _attach_wrds

        con = MagicMock()
        con.execute.side_effect = RuntimeError("ATTACH failed: host=x password=hunter2 dbname=wrds")
        with pytest.raises(RuntimeError) as exc_info:
            _attach_wrds(con, "host=x password=hunter2", "hunter2")
        assert "hunter2" not in str(exc_info.value)
        assert "credentials" in str(exc_info.value).lower()

    def test_redacts_special_char_password_on_attach_error(self):
        """The escaped form of a special-character password is what appears in the
        ATTACH error, so detection must match it — a raw ``password in str(e)``
        would miss it and re-raise the secret."""
        from jkp.data.aux_functions import _attach_wrds

        con = MagicMock()
        con.execute.side_effect = RuntimeError(r"ATTACH failed: password='ab\'cd' dbname=wrds")
        with pytest.raises(RuntimeError) as exc_info:
            _attach_wrds(con, r"host=x password='ab\'cd'", "ab'cd")
        msg = str(exc_info.value)
        assert "ab'cd" not in msg and r"ab\'cd" not in msg  # neither form leaks
        assert "credentials" in msg.lower()

    def test_passes_through_non_password_error(self):
        from jkp.data.aux_functions import _attach_wrds

        con = MagicMock()
        con.execute.side_effect = RuntimeError("FATAL: too many connections for role")
        # A non-credential error (no password in it) propagates unchanged.
        with pytest.raises(RuntimeError, match="too many connections"):
            _attach_wrds(con, "host=x password=hunter2", "hunter2")


class TestMapInterruptible:
    """The interrupt-responsive concurrent map used by the histogram and concat phases."""

    def test_preserves_order_and_returns_results(self):
        from jkp.data.aux_functions import _map_interruptible

        assert _map_interruptible(lambda x: x * 2, [1, 2, 3, 4], max_workers=3) == [2, 4, 6, 8]

    def test_propagates_task_exception(self):
        from jkp.data.aux_functions import _map_interruptible

        def boom(x):
            if x == 2:
                raise ValueError("boom")
            return x

        with pytest.raises(ValueError, match="boom"):
            _map_interruptible(boom, [1, 2, 3], max_workers=3)

    def test_propagates_keyboard_interrupt(self):
        """A Ctrl-C during the wait loop unwinds promptly rather than blocking in shutdown."""
        from jkp.data.aux_functions import _map_interruptible

        with (
            patch("jkp.data.aux_functions.wait", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            _map_interruptible(lambda x: x, [1, 2, 3], max_workers=2)
