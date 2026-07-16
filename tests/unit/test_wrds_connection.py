"""
Tests for WRDS connection primitives (jkp.data.wrds_connection).

Covers conninfo building, libpq/SQL escaping, and the password-masking helpers
used to keep credentials out of error text, plus verify_wrds_connection's
error-handling contract.
"""

import pytest


class TestGenWrdsConnectionInfo:
    """Tests for gen_wrds_connection_info() function."""

    def test_connection_string_format(self):
        """Connection string should have correct format."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

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
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("testuser", "p ss'w\\rd")

        # spaces stay inside the quotes; ' and \ are backslash-escaped
        assert "password='p ss\\'w\\\\rd'" in result
        assert "sslmode=require" in result

    def test_username_with_special_characters_is_quoted_and_escaped(self):
        """A username with a space or quote must be single-quoted and escaped too,
        or it breaks libpq's conninfo parsing just as an unquoted password would."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("od d'user", None)

        assert "user='od d\\'user'" in result

    def test_password_omitted_when_none(self):
        """With password=None the password= field is omitted so libpq reads
        ~/.pgpass / $PGPASSFILE."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

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
        from jkp.data.wrds_connection import _sql_literal, gen_wrds_connection_info

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
        from jkp.data.wrds_connection import _sql_literal, gen_wrds_connection_info

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

        from jkp.data.wrds_connection import _password_in_error

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
        from jkp.data.wrds_connection import (
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
        from jkp.data.wrds_connection import (
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

    def test_connect_timeout_included_when_set(self):
        """connect_timeout, when given, is appended to the conninfo."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("u", "p", connect_timeout=10)

        assert "connect_timeout=10" in result

    def test_connect_timeout_omitted_by_default(self):
        """Without an explicit connect_timeout, libpq's default (no timeout) applies."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("u", "p")

        assert "connect_timeout" not in result

    def test_password_none_omits_password_field(self):
        """password=None (no positional password arg) omits password= entirely."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("u", None)

        assert "password=" not in result


class TestVerifyWrdsConnection:
    """Tests for verify_wrds_connection() — the CLI-facing connectivity check."""

    def test_raises_password_free_runtime_error_on_attach_failure(self, monkeypatch):
        """A failed ATTACH (e.g. bad credentials) must surface as a RuntimeError
        whose message never contains the password, even though the underlying
        DuckDB error echoes it in libpq-escaped form."""
        from jkp.data import wrds_connection as mod

        password = "hunter2"  # noqa: S105

        class FakeConnection:
            def execute(self, sql, *args):
                if "ATTACH" in sql:
                    escaped = mod._pg_escape_value(password)
                    raise RuntimeError(
                        f"IO Error: Unable to connect at password='{escaped}' sslmode=require"
                    )
                return None

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        monkeypatch.setattr(mod.duckdb, "connect", lambda *a, **kw: FakeConnection())

        with pytest.raises(RuntimeError) as exc_info:
            mod.verify_wrds_connection("testuser", password, connect_timeout=1)

        msg = str(exc_info.value)
        assert password not in msg
        assert "credentials" in msg.lower()

    def test_pgpass_auth_failure_wrapped_as_runtime_error(self, monkeypatch):
        """With password=None (the ~/.pgpass auth path) _attach_wrds re-raises the
        raw DuckDB exception; verify_wrds_connection must still wrap it in a
        RuntimeError so `jkp connect` exits cleanly instead of dumping a traceback."""
        from jkp.data import wrds_connection as mod

        class FakeConnection:
            def execute(self, sql, *args):
                if "ATTACH" in sql:
                    # A non-RuntimeError, mirroring the raw duckdb IOException that
                    # escapes _attach_wrds when there is no password to detect/redact.
                    raise OSError("IO Error: connection to WRDS refused")
                return None

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        monkeypatch.setattr(mod.duckdb, "connect", lambda *a, **kw: FakeConnection())

        with pytest.raises(RuntimeError) as exc_info:
            mod.verify_wrds_connection("testuser", None, connect_timeout=1)

        assert "credentials" in str(exc_info.value).lower()

    def test_install_extension_failure_wrapped_as_runtime_error(self, monkeypatch):
        """A failed INSTALL postgres (e.g. no network to the extension repo on a
        headless HPC node) must be wrapped in a RuntimeError, not escape as a raw
        traceback — it runs before the ATTACH but is still inside the guarded path."""
        from jkp.data import wrds_connection as mod

        def boom():
            raise OSError("HTTP Error: failed to download extension 'postgres'")

        monkeypatch.setattr(mod, "_install_postgres_extension", boom)

        with pytest.raises(RuntimeError) as exc_info:
            mod.verify_wrds_connection("testuser", "pw", connect_timeout=1)  # noqa: S106

        assert "wrds" in str(exc_info.value).lower()
