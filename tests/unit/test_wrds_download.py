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


class TestGenWrdsConnectionInfo:
    """Tests for gen_wrds_connection_info() function."""

    def test_connection_string_format(self):
        """Connection string should have correct format."""
        from jkp.data.aux_functions import gen_wrds_connection_info

        result = gen_wrds_connection_info("testuser", "testpass")

        assert "host=wrds-pgdata.wharton.upenn.edu" in result
        assert "port=9737" in result
        assert "dbname=wrds" in result
        assert "user='testuser'" in result
        assert "password='testpass'" in result
        assert "sslmode=require" in result

    def test_password_with_special_characters_is_quoted_and_escaped(self):
        """A password containing spaces/quotes/backslashes must be single-quoted
        with libpq escaping so it can't break the conninfo."""
        from jkp.data.aux_functions import gen_wrds_connection_info

        result = gen_wrds_connection_info("testuser", "p ss'w\\rd")

        # spaces stay inside the quotes; ' and \ are backslash-escaped
        assert "password='p ss\\'w\\\\rd'" in result
        assert "sslmode=require" in result

    def test_username_with_special_characters_is_quoted_and_escaped(self):
        """A username with a space or quote must be single-quoted and escaped too,
        or it breaks libpq's conninfo parsing just as an unquoted password would."""
        from jkp.data.aux_functions import gen_wrds_connection_info

        result = gen_wrds_connection_info("od d'user", None)

        assert "user='od d\\'user'" in result

    def test_password_omitted_when_none(self):
        """With password=None the password= field is omitted so libpq reads
        ~/.pgpass / $PGPASSFILE."""
        from jkp.data.aux_functions import gen_wrds_connection_info

        result = gen_wrds_connection_info("testuser", None)

        assert "user='testuser'" in result
        assert "password=" not in result
        assert "sslmode=require" in result

    def test_sql_literal_escapes_conninfo_for_embedding(self):
        """_sql_literal must escape the conninfo so it embeds in single-quoted
        ATTACH/postgres_scan SQL without a quoted password terminating the literal.
        Proven at the parse layer — no postgres extension or network — so it runs
        unconditionally in CI, unlike the ATTACH integration check below."""
        duckdb = pytest.importorskip("duckdb")
        from jkp.data.aux_functions import _sql_literal, gen_wrds_connection_info

        conninfo = gen_wrds_connection_info("testuser", "p ss'w\\d")
        con = duckdb.connect()

        # Escaped: the whole conninfo parses as one string literal and round-trips.
        assert con.execute(f"SELECT '{_sql_literal(conninfo)}'").fetchone()[0] == conninfo
        # Unescaped: the raw quote terminates the literal early -> ParserException.
        with pytest.raises(Exception) as excinfo:
            con.execute(f"SELECT '{conninfo}'")
        assert "Parser" in type(excinfo.value).__name__

    def test_conninfo_embeds_in_sql_without_parse_error(self):
        """Integration check (needs the DuckDB postgres extension): a real ATTACH
        with the escaped conninfo must fail at the *connection* stage, not with a
        ParserException — proving the SQL literal held together AND libpq accepted
        the quoted conninfo. Also pins the password's echo format for the masking
        in _attach_wrds."""
        duckdb = pytest.importorskip("duckdb")
        from jkp.data.aux_functions import _sql_literal, gen_wrds_connection_info

        con = duckdb.connect()
        try:
            con.execute("INSTALL postgres; LOAD postgres")
        except Exception:  # pragma: no cover - environment without the extension
            pytest.skip("duckdb postgres extension unavailable")

        # A password with a space, a single quote, and a backslash — the exact
        # class the libpq quoting targets. Point at a dead local endpoint so the
        # ATTACH fails to *connect* rather than hanging on a real socket.
        conninfo = (
            gen_wrds_connection_info("testuser", "p ss'w\\d")
            .replace("host=wrds-pgdata.wharton.upenn.edu", "host=127.0.0.1")
            .replace("port=9737", "port=9")
            .replace("sslmode=require", "sslmode=disable")
            # cap any hang if something ever happens to listen on :9
            + " connect_timeout=2"
        )
        # Guard the neutering: if the WRDS host/port constants ever change, the
        # .replace() calls above would silently no-op and this test would dial the
        # real WRDS endpoint. Assert the substitutions actually took effect.
        assert "host=127.0.0.1" in conninfo and "port=9 " in conninfo
        assert "wrds-pgdata.wharton.upenn.edu" not in conninfo

        from jkp.data.aux_functions import _password_in_error

        with pytest.raises(Exception) as excinfo:
            con.execute(f"ATTACH '{_sql_literal(conninfo)}' AS wrds (TYPE postgres, READ_ONLY)")
        err = str(excinfo.value)
        # A ParserException would mean the SQL literal was broken by the password's
        # quote (the SQL-escaping half of the fix).
        assert "Parser" not in type(excinfo.value).__name__, err
        # Reaching the connection stage proves libpq accepted the quoted conninfo
        # (the libpq-escaping half): a regression from \' to ''-style escaping would
        # fail here as a conninfo-syntax error even though the parse test still passes.
        assert "connect" in err.lower(), err
        # And DuckDB really echoes the password in the libpq-escaped form the masking
        # matches — pinned against a real error, not a synthesized one, so a DuckDB
        # echo-format change can't silently regress _attach_wrds's redaction.
        assert _password_in_error(err, "p ss'w\\d")

    def test_password_masking_matches_libpq_escaped_form(self):
        """DuckDB echoes the connection string with the password in libpq-escaped
        form, so the masking must match that form — a raw ``password in text``
        check misses every special-character password."""
        from jkp.data.aux_functions import (
            _password_in_error,
            _pg_escape_value,
            _redact_password,
        )

        pw = "ab'cd\\e"
        err = f"IO Error: Unable to connect at password='{_pg_escape_value(pw)}' sslmode=disable"

        assert pw not in err  # the raw password does not appear verbatim
        assert _password_in_error(err, pw)
        redacted = _redact_password(err, pw)
        assert "***" in redacted
        assert _pg_escape_value(pw) not in redacted

    def test_password_masking_matches_sql_doubled_form(self):
        """A parser error echoes the *raw statement text*, where the password is
        SQL-escaped over the libpq-escaped form. Masking must match that composed
        form too — neither the raw nor the plain libpq-escaped form appears."""
        from jkp.data.aux_functions import (
            _password_in_error,
            _pg_escape_value,
            _redact_password,
            _sql_literal,
        )

        pw = "ab'cd\\e"
        composed = _sql_literal(_pg_escape_value(pw))
        err = f"Parser Error: syntax error near ...{composed}... in statement"

        assert pw not in err and _pg_escape_value(pw) not in err  # only the composed form
        assert _password_in_error(err, pw)
        assert composed not in _redact_password(err, pw)

    def test_empty_password_is_a_noop(self):
        """An empty password has no forms to find/replace. Without an early-out, "" is a substring
        of everything: _password_in_error would report a match and _redact_password would interleave
        *** between every character. Guard against that footgun (it's truthiness-gated in the guards,
        but the helpers must be safe on their own)."""
        from jkp.data.aux_functions import (
            _password_in_error,
            _password_leak_in_error,
            _redact_password,
        )

        text = "IO Error: connection reset by peer"
        assert _password_in_error(text, "") is False
        assert _password_leak_in_error(text, "") is False
        assert _redact_password(text, "") == text

    def test_password_leak_detects_truncated_fragment(self):
        """A truncated statement echo (DuckDB's caret-centered ``LINE 1: ...`` window) can slice a
        password, leaving only a tail fragment with no complete form. _password_in_error misses that
        (whole forms only); _password_leak_in_error must catch a >= 8-char fragment so the guard
        doesn't fail open."""
        from jkp.data.aux_functions import _password_in_error, _password_leak_in_error

        password = "SEKRIT-abcdefghijklmnop"  # noqa: S105
        # Front-elided echo: the head (incl. `password='`) is cut; a 12-char tail survives.
        truncated = (
            "Parser Error: syntax error\n\nLINE 1: ...efghijklmnop', 'comp', 'funda')) TO 'x'"
        )
        assert password not in truncated  # no complete form present...
        assert _password_in_error(truncated, password) is False  # ...so the whole-form check misses
        assert (
            _password_leak_in_error(truncated, password) is True
        )  # ...but the fragment check hits

    def test_password_leak_ignores_short_coincidental_runs(self):
        """Fragment matching needs a run of >= 8 password chars, so a short coincidental overlap
        (here 'reset' happens to be inside the password) does not trigger a false leak."""
        from jkp.data.aux_functions import _password_leak_in_error

        password = "reset-Xq"  # noqa: S105  # 'reset' overlaps common error text, but only 5 chars
        assert _password_leak_in_error("IO Error: connection reset by peer", password) is False

    def test_truncation_below_fragment_length_is_a_documented_gap(self):
        """Documented limitation, not a bug: detection needs a run of _MIN_LEAK_FRAGMENT consecutive
        form chars, so ANY echo retaining <= 7 of them fails open -- regardless of password length. A
        short password is the guaranteed case (any truncation); a long password hits it only when the
        cut leaves a <= 7-char run. Intentional -- any window short enough to catch a <= 7-char run
        would false-positive on ordinary error text (see _password_leak_in_error). Pinned so the
        boundary is a conscious choice: if you make this pass, you have changed the detection
        contract."""
        from jkp.data.aux_functions import _password_leak_in_error

        short = "hunter2"  # noqa: S105  # 7 chars: every truncation leaves <= 6
        assert _password_leak_in_error("... password=hunter2 ...", short) is True  # full: caught
        assert _password_leak_in_error("LINE 1: ...unter2', 'comp')", short) is False  # tail: not

        # A LONG password truncated to a 7-char surviving run also fails open -- not a length issue.
        long = "SEKRIT-abcdefghijklmnop"  # noqa: S105
        assert _password_leak_in_error("LINE 1: ...klmnop', 'comp')", long) is False  # 6-char tail


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

    def test_sequential_postgres_scan_masks_password_end_to_end(self, mock_duckdb, test_paths):
        """Default-flag download: a wrong password surfaced by the first postgres_scan probe must
        not leak through the orchestrator (issue #250). Exercises the real get_columns/
        download_wrds_table and pins the ``password=`` plumbing at the sequential call site.

        The broad ``pytest.raises(Exception)`` only proves *something* raised; the ``credentials``
        assertion below is what makes this strict (that the leak was masked). Do not drop it."""
        from jkp.data.aux_functions import download_raw_data_tables

        _mock, mock_conn = mock_duckdb
        password = "topsecret"  # noqa: S105

        def execute(sql, *args):
            if "LIMIT 0" in sql:  # get_columns probe -> first postgres_scan, wrong password
                raise RuntimeError(f"IO Error: Connection failed: password={password} dbname=wrds")
            return MagicMock(description=[("col1",), ("col2",)])

        mock_conn.execute.side_effect = execute  # side_effect overrides the fixture's return_value
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            download_raw_data_tables(test_paths, "user", password, end_date=dt.date(2025, 12, 31))
        assert password not in str(exc_info.value)
        assert "credentials" in str(exc_info.value).lower()

    def test_persistent_connection_redacts_password_end_to_end(self, mock_duckdb, test_paths):
        """Persistent-connection path: ATTACH succeeds (valid password), but a later per-table
        query that echoes the conninfo must not leak that working password (issue #250 sibling).

        As above, the ``connection reset`` assertion (redaction kept the real diagnostic) is what
        makes the broad ``pytest.raises`` strict -- do not drop it."""
        from jkp.data.aux_functions import download_raw_data_tables

        _mock, mock_conn = mock_duckdb
        password = "topsecret"  # noqa: S105

        def execute(sql, *args):
            # ATTACH / LOAD / the LIMIT 0 probe succeed; the COPY drops the socket mid-download.
            if "COPY" in sql:
                raise RuntimeError(f"IO Error: password={password} dbname=wrds connection reset")
            return MagicMock(description=[("col1",), ("col2",)])

        mock_conn.execute.side_effect = execute  # side_effect overrides the fixture's return_value
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            download_raw_data_tables(
                test_paths,
                "user",
                password,
                end_date=dt.date(2025, 12, 31),
                persistent_connection=True,
            )
        msg = str(exc_info.value)
        assert password not in msg
        assert "connection reset" in msg  # redaction keeps the real diagnostic
        # Per-table guard tags the failing table so a mid-run failure keeps its identity. The first
        # table in the canonical list is comp.exrt_dly, so its COPY is the one that fails here.
        assert msg.startswith("comp.exrt_dly: ")

    def test_persistent_detach_failure_does_not_bury_redacted_error(self, mock_duckdb, test_paths):
        """The mid-download socket-drop scenario: the COPY fails (redacted, password-free) AND the
        DETACH in the finally then fails on the dead connection. The DETACH must be suppressed, or it
        would displace the redacted per-table diagnostic (which then survives only as __context__)."""
        from jkp.data.aux_functions import download_raw_data_tables

        _mock, mock_conn = mock_duckdb
        password = "topsecret"  # noqa: S105

        def execute(sql, *args):
            if "COPY" in sql:
                raise RuntimeError(f"IO Error: password={password} dbname=wrds connection reset")
            if "DETACH" in sql:
                raise RuntimeError("Cannot DETACH: connection already closed")
            return MagicMock(description=[("col1",), ("col2",)])

        mock_conn.execute.side_effect = execute
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            download_raw_data_tables(
                test_paths,
                "user",
                password,
                end_date=dt.date(2025, 12, 31),
                persistent_connection=True,
            )
        msg = str(exc_info.value)
        assert "connection reset" in msg  # the redacted COPY diagnostic survived...
        assert "DETACH" not in msg  # ...and the DETACH failure did not displace it
        assert password not in msg


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
        self, mock_dl, mock_hist, mock_concat, test_paths
    ):
        """A concat failure surfaces as one aggregated error, not a raw first-failure exception."""
        from jkp.data.aux_functions import SPLIT_TABLES, download_raw_data_tables

        mock_hist.return_value = dict.fromkeys(SPLIT_TABLES, [(y, 10) for y in range(2010, 2022)])
        mock_concat.side_effect = RuntimeError("disk full")

        with patch("jkp.data.aux_functions.duckdb") as mock_duckdb:
            conns = [MagicMock(name=f"c{i}") for i in range(40)]
            for c in conns:
                c.__enter__.return_value = c
            mock_duckdb.connect.side_effect = conns
            with pytest.raises(RuntimeError, match="concatenation"):
                download_raw_data_tables(
                    test_paths, "user", "pass", end_date=dt.date(2025, 12, 31), max_workers=4
                )

    @patch("jkp.data.aux_functions._concat_chunks")
    @patch("jkp.data.aux_functions._compute_histograms")
    @patch("jkp.data.aux_functions.download_wrds_table_attached")
    def test_parallel_download_splits_giant_tables(
        self, mock_dl, mock_hist, mock_concat, test_paths
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

        with patch("jkp.data.aux_functions.duckdb") as mock_duckdb:
            conns = [MagicMock(name=f"c{i}") for i in range(40)]
            for c in conns:
                c.__enter__.return_value = c
            mock_duckdb.connect.side_effect = conns
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
        self, mock_dl, mock_hist, mock_concat, test_paths
    ):
        """The concat_map -> ThreadPoolExecutor wiring invokes _concat_chunks once per split table."""
        from jkp.data.aux_functions import SPLIT_TABLES, download_raw_data_tables

        mock_hist.return_value = dict.fromkeys(SPLIT_TABLES, [(y, 10) for y in range(2010, 2022)])
        with patch("jkp.data.aux_functions.duckdb") as mock_duckdb:
            conns = [MagicMock(name=f"c{i}") for i in range(40)]
            for c in conns:
                c.__enter__.return_value = c
            mock_duckdb.connect.side_effect = conns
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
    """The shared WRDS ATTACH helper (password masking -- _attach_wrds uses _mask_password_errors,
    so a leak surfaces the generic credential message, not a redacted error)."""

    def test_masks_password_on_attach_error(self):
        from jkp.data.aux_functions import _attach_wrds

        con = MagicMock()
        con.execute.side_effect = RuntimeError("ATTACH failed: host=x password=hunter2 dbname=wrds")
        with pytest.raises(RuntimeError) as exc_info:
            _attach_wrds(con, "host=x password=hunter2", "hunter2")
        msg = str(exc_info.value)
        assert "hunter2" not in msg
        assert "credentials" in msg.lower()
        # Pin the ATTACH-specific message so a dropped `message=` (falling back to the generic
        # get_columns text) is caught, not silently accepted.
        assert "attach" in msg.lower()

    def test_masks_special_char_password_on_attach_error(self):
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


class TestPostgresScanPasswordMasking:
    """The default sequential postgres_scan() download path must not leak the password.

    Unlike the ATTACH path (see TestAttachWrds), get_columns()/download_wrds_table() embed the
    full conninfo in ``postgres_scan('...')`` SQL, so a wrong/expired password makes DuckDB echo
    the conninfo — and thus the password — in the IOException it raises.
    """

    def test_get_columns_masks_password_on_error(self):
        """get_columns()'s probe must scrub the password from a postgres_scan failure (#250).

        The probe establishes the connection, so a failure most likely means bad credentials —
        hence the generic credential-check message (mask), not a redacted raw error."""
        from jkp.data.aux_functions import get_columns

        password = "topsecret"  # noqa: S105
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError(
            f"IO Error: Connection failed: host=x password={password} dbname=wrds"
        )
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            get_columns(conn, f"host=x password={password}", "comp", "funda", password=password)
        assert password not in str(exc_info.value)
        assert "credentials" in str(exc_info.value).lower()

    def test_get_columns_requires_password_keyword(self):
        """password is keyword-only and required, so a caller can't silently omit it and re-open
        the leak (the fail-open regression shape). Omitting it is a TypeError, not a quiet no-op."""
        from jkp.data.aux_functions import get_columns

        with pytest.raises(TypeError):
            get_columns(MagicMock(), "host=x", "comp", "funda")  # type: ignore[call-arg]

    def test_download_wrds_table_requires_password_keyword(self):
        """Same fail-open guard on download_wrds_table: omitting password is a TypeError, so a
        reintroduced ``password=None`` default (which would disable masking) can't slip through."""
        from jkp.data.aux_functions import download_wrds_table

        with pytest.raises(TypeError):
            download_wrds_table("host=x", MagicMock(), "comp.funda", "out.parquet")  # type: ignore[call-arg]

    def test_download_wrds_table_redacts_password_on_copy_error(self):
        """download_wrds_table()'s COPY runs after the probe already opened the connection, so a
        failure there is a genuine diagnostic: redact the secret but keep the rest (#250)."""
        from jkp.data.aux_functions import download_wrds_table

        password = "topsecret"  # noqa: S105

        def execute(sql, *args):
            # get_columns' LIMIT 0 probe succeeds; the COPY fails echoing the conninfo.
            if "COPY" in sql:
                raise RuntimeError(f"IO Error: host=x password={password} dbname=wrds table full")
            return MagicMock(description=[("col_a",), ("col_b",)])

        conn = MagicMock()
        conn.execute.side_effect = execute
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            download_wrds_table(
                f"host=x password={password}",
                conn,
                "comp.funda",
                "out.parquet",
                password=password,
            )
        msg = str(exc_info.value)
        assert password not in msg
        assert "table full" in msg  # redaction preserves the diagnostic

    def test_masked_error_does_not_leak_via_exception_chain(self):
        """The password must be gone from the whole chain, not just str(e): a context-manager guard
        scrubs the message but leaves the original on __context__, which Sentry-style chain-walking
        recovers. The guard wraps the call and raises outside the except, so the chain is clean."""
        from jkp.data.aux_functions import get_columns

        password = "topsecret"  # noqa: S105
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError(f"IO Error: password={password} dbname=wrds")
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            get_columns(conn, f"host=x password={password}", "comp", "funda", password=password)

        node: BaseException | None = exc_info.value
        while node is not None:
            assert password not in str(node)
            node = node.__context__ or node.__cause__

    def test_guard_frame_does_not_retain_raw_error_in_locals(self):
        """The raise-site frame (_guard_password_errors) is what ``pytest --showlocals`` /
        Sentry-style local capture render, and it is the only frame that holds the raw
        password-bearing error text. It must drop that from its locals. A unique marker present
        *only* in the raw error (not in the password or conninfo, so caller-frame locals can't
        account for it) must not survive in that frame's locals."""
        import traceback

        from jkp.data.aux_functions import get_columns

        password = "topsecret"  # noqa: S105
        marker = "RAWERR_MARKER_9x8y7z"  # appears only inside the raw DuckDB error string
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError(
            f"IO Error: password={password} dbname=wrds {marker}"
        )
        with pytest.raises(Exception) as exc_info:  # noqa: PT011
            get_columns(conn, f"host=x password={password}", "comp", "funda", password=password)

        assert marker not in str(exc_info.value)
        guard_frames = [
            frame
            for frame, _lineno in traceback.walk_tb(exc_info.value.__traceback__)
            if frame.f_code.co_name == "_guard_password_errors"
        ]
        assert guard_frames, "guard frame not found in traceback"
        for frame in guard_frames:
            assert marker not in repr(frame.f_locals)

    @staticmethod
    def _raw_reachable(exc: BaseException, sentinel: str) -> bool:
        """True if ``sentinel`` is reachable from ``exc`` via its chain (``str``) or ANY traceback
        frame's locals. ``sentinel`` must be a string that exists ONLY inside the raw error text --
        never as a standalone test local -- so a hit means the raw error survived on some frame.
        (An earlier version filtered by frame name and so skipped the transform/callback frames
        where these leaks actually live, letting mutants pass; search every frame instead.)"""
        node: BaseException | None = exc
        while node is not None:
            if sentinel in str(node):
                return True
            tb = node.__traceback__
            while tb is not None:
                if sentinel in repr(tb.tb_frame.f_locals):
                    return True
                tb = tb.tb_next
            node = node.__context__ or node.__cause__
        return False

    def test_transform_failure_does_not_leak_raw_text(self):
        """Exotic exit: if the transform itself raises (realistically MemoryError), the raw text is
        on the transform's frame. That sub-traceback must be dropped so it can't be walked out.

        The sentinel is an inline literal that appears only inside the raised error's text (never as
        a standalone local), so _raw_reachable's all-frames search can't false-hit a test local -- a
        hit means the transform frame's ``_text`` survived (which reverting exc.__traceback__=None
        would cause)."""
        from jkp.data.aux_functions import _guard_password_errors

        password = "topsecret"  # noqa: S105

        def boom():
            raise RuntimeError(f"IO Error: password={password} dbname=wrds RAWSENTINEL_XFORM_MEM")

        def bad_transform(_text):
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError) as exc_info:
            _guard_password_errors(password, bad_transform, boom)
        # Compute on its own line, NOT inline in the assert: pytest's assertion rewriter binds the
        # literal argument to a temp local in this frame, and this frame is in the traceback being
        # walked, so an inline sentinel would false-hit.
        reachable = self._raw_reachable(exc_info.value, "RAWSENTINEL_XFORM_MEM")
        assert not reachable

    def test_interrupt_during_detection_does_not_leak(self):
        """Exotic exit: a Ctrl-C landing in the detection window must not carry the raw exception
        out on the KeyboardInterrupt's chain, nor on the detection/guard frames."""
        from jkp.data.aux_functions import _guard_password_errors

        password = "topsecret"  # noqa: S105

        def boom():
            raise RuntimeError(f"IO Error: password={password} dbname=wrds RAWSENTINEL_DETECT")

        def interrupt_detection(_text, _password):
            raise KeyboardInterrupt

        with (
            patch("jkp.data.aux_functions._password_leak_in_error", interrupt_detection),
            pytest.raises(KeyboardInterrupt) as exc_info,
        ):
            _guard_password_errors(password, lambda t: t, boom)
        assert exc_info.value.__context__ is None  # chain severed
        reachable = self._raw_reachable(exc_info.value, "RAWSENTINEL_DETECT")  # own line: see above
        assert not reachable

    def test_interrupt_during_transform_does_not_leak(self):
        """Sibling of the above: a Ctrl-C landing *inside* transform (not the detection window) is a
        BaseException, so the transform escape hatch must catch BaseException -- not just Exception
        -- or the raw text rides out on transform's frame."""
        from jkp.data.aux_functions import _guard_password_errors

        password = "topsecret"  # noqa: S105

        def boom():
            raise RuntimeError(f"IO Error: password={password} dbname=wrds RAWSENTINEL_XFORM_KI")

        def interrupt_transform(_text):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt) as exc_info:
            _guard_password_errors(password, interrupt_transform, boom)
        reachable = self._raw_reachable(
            exc_info.value, "RAWSENTINEL_XFORM_KI"
        )  # own line: see above
        assert not reachable

    def test_redact_errors_prefixes_context(self):
        """_redact_password_errors(..., context=table) tags the scrubbed message with the failing
        item's identity, so a per-table loop failure isn't a bare, table-less error."""
        from jkp.data.aux_functions import _redact_password_errors

        password = "topsecret"  # noqa: S105

        def boom():
            raise RuntimeError(f"IO Error: password={password} dbname=wrds reset")

        with pytest.raises(RuntimeError) as exc_info:
            _redact_password_errors(password, boom, context="crsp.dsf_v2")
        msg = str(exc_info.value)
        assert msg.startswith("crsp.dsf_v2: ")
        assert password not in msg
        assert "reset" in msg  # diagnostic preserved

    def test_redact_fails_closed_on_truncated_fragment(self):
        """When only a truncated fragment matched (no complete form), _redact_password can't scrub
        it, so the redact path must fail closed to the generic message -- not surface the raw error
        with the password tail intact (issue #250, the truncated-echo fail-open)."""
        from jkp.data.aux_functions import _password_in_error, _redact_password_errors

        password = "SEKRIT-abcdefghijklmnop"  # noqa: S105
        tail = "efghijklmnop', 'comp', 'funda')) TO 'x'"  # >=8-char fragment, no complete form

        def boom():
            raise RuntimeError(f"Parser Error: syntax error\n\nLINE 1: ...{tail}")

        assert not _password_in_error(f"...{tail}", password)  # precondition: fragment only
        with pytest.raises(RuntimeError) as exc_info:
            _redact_password_errors(password, boom, context="crsp.dsf_v2")
        msg = str(exc_info.value)
        assert "efghijklmnop" not in msg  # the raw fragment must not survive
        assert "credentials" in msg.lower()  # failed closed to the generic message
        assert msg.startswith("crsp.dsf_v2: ")  # context still preserved

    def test_redact_fails_closed_on_mixed_full_and_fragment_echo(self):
        """The dangerous case: one error carries BOTH a complete conninfo echo AND an independent
        truncated fragment (DuckDB can emit a header echo plus a sliced LINE 1: excerpt). Redacting
        whole forms leaves the fragment; the re-check on the redacted body must catch it and fail
        closed, or the fragment ships inside 'redacted' output."""
        from jkp.data.aux_functions import _redact_password_errors

        password = "SEKRIT-abcdefghijklmnop"  # noqa: S105

        def boom():
            # complete form (scrubbable) AND a sliced tail fragment (not scrubbable) in one message
            raise RuntimeError(
                f"IO Error: ... password='{password}' ... connection dropped\n\n"
                "LINE 1: ...efghijklmnop', 'crsp', 'msf_v2') TO 'out.parquet'"
            )

        with pytest.raises(RuntimeError) as exc_info:
            _redact_password_errors(password, boom, context="crsp.msf_v2")
        msg = str(exc_info.value)
        assert "efghijklmnop" not in msg  # the sliced fragment must not survive the redaction
        assert password not in msg
        assert "credentials" in msg.lower()  # failed closed rather than ship a partial redaction

    def test_get_columns_passes_through_non_password_error(self):
        """A failure that doesn't carry the password propagates unchanged."""
        from jkp.data.aux_functions import get_columns

        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("Catalog Error: table does not exist")
        with pytest.raises(RuntimeError, match="does not exist"):
            get_columns(conn, "host=x password=hunter2", "comp", "nope", password="hunter2")

    def test_get_columns_none_password_propagates_raw_error(self):
        """The ~/.pgpass path passes password=None (no secret in the conninfo), so masking is a
        no-op and the original error propagates unchanged."""
        from jkp.data.aux_functions import get_columns

        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("IO Error: connection reset by peer")
        with pytest.raises(RuntimeError, match="connection reset by peer"):
            get_columns(conn, "host=x sslmode=require", "comp", "funda", password=None)

    def test_compute_histograms_query_failure_is_redacted(self):
        """A histogram query (post-ATTACH) failure must not leak the password, but must keep the
        real diagnostic (redaction, not the credential-check message) since the password is valid."""
        from jkp.data.aux_functions import _compute_histograms

        password = "topsecret"  # noqa: S105

        def execute(sql, *args):
            # ATTACH succeeds; the per-year COUNT(*) query fails echoing the conninfo.
            if "EXTRACT(YEAR" in sql:
                raise RuntimeError(f"IO Error: host=x password={password} dbname=wrds reset")
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
        msg = str(exc_info.value)
        assert password not in msg
        assert "reset" in msg  # redaction preserves the diagnostic


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
