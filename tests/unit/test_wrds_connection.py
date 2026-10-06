"""
Tests for WRDS connection primitives (jkp.data.wrds_connection).

Covers conninfo building and libpq/SQL escaping, passing the password to libpq
via PGPASSWORD, and verify_wrds_connection's error-handling contract.
"""

import os

import pytest


class TestGenWrdsConnectionInfo:
    """Tests for gen_wrds_connection_info() function."""

    def test_connection_string_format(self):
        """Connection string should have correct format and never carry a password
        (libpq reads it from PGPASSWORD or ~/.pgpass)."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("testuser")

        assert "host=wrds-pgdata.wharton.upenn.edu" in result
        assert "port=9737" in result
        assert "dbname=wrds" in result
        assert "user='testuser'" in result
        assert "sslmode=require" in result
        assert "password" not in result

    def test_username_with_special_characters_is_quoted_and_escaped(self):
        """A username with a space or quote must be single-quoted and escaped, or
        it breaks libpq's conninfo parsing."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("od d'user")

        assert "user='od d\\'user'" in result

    def test_sql_literal_escapes_conninfo_for_embedding(self):
        """_sql_literal must escape the conninfo so it embeds in single-quoted
        ATTACH/postgres_scan SQL without a quoted username terminating the literal.
        Proven at the parse layer — no postgres extension or network — so it runs
        unconditionally in CI, unlike the ATTACH integration check below."""
        duckdb = pytest.importorskip("duckdb")
        from jkp.data.wrds_connection import _sql_literal, gen_wrds_connection_info

        conninfo = gen_wrds_connection_info("od d'user")
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
        the quoted conninfo."""
        duckdb = pytest.importorskip("duckdb")
        from jkp.data.wrds_connection import _sql_literal, gen_wrds_connection_info

        con = duckdb.connect()
        try:
            con.execute("INSTALL postgres; LOAD postgres")
        except Exception:  # pragma: no cover - environment without the extension
            pytest.skip("duckdb postgres extension unavailable")

        # A username with a space and a single quote — the class the libpq quoting
        # targets. Point at a dead local endpoint so the ATTACH fails to *connect*
        # rather than hanging on a real socket.
        conninfo = (
            gen_wrds_connection_info("od d'user")
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

        with pytest.raises(Exception) as excinfo:
            con.execute(f"ATTACH '{_sql_literal(conninfo)}' AS wrds (TYPE postgres, READ_ONLY)")
        err = str(excinfo.value)
        # A ParserException would mean the SQL literal was broken by the quote.
        assert "Parser" not in type(excinfo.value).__name__, err
        # Reaching the connection stage proves libpq accepted the quoted conninfo.
        assert "connect" in err.lower(), err

    def test_connect_timeout_included_when_set(self):
        """connect_timeout, when given, is appended to the conninfo."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("u", connect_timeout=10)

        assert "connect_timeout=10" in result

    def test_connect_timeout_omitted_by_default(self):
        """Without an explicit connect_timeout, libpq's default (no timeout) applies."""
        from jkp.data.wrds_connection import gen_wrds_connection_info

        result = gen_wrds_connection_info("u")

        assert "connect_timeout" not in result


class TestWrdsPasswordEnv:
    """wrds_password_env() hands the password to libpq via PGPASSWORD."""

    def test_sets_and_unsets(self, monkeypatch):
        from jkp.data.wrds_connection import wrds_password_env

        monkeypatch.delenv("PGPASSWORD", raising=False)
        with wrds_password_env("hunter2"):
            assert os.environ["PGPASSWORD"] == "hunter2"
        assert "PGPASSWORD" not in os.environ

    def test_restores_previous_value_on_error(self, monkeypatch):
        from jkp.data.wrds_connection import wrds_password_env

        monkeypatch.setenv("PGPASSWORD", "users-own")
        with pytest.raises(ValueError), wrds_password_env("hunter2"):
            raise ValueError("boom")
        assert os.environ["PGPASSWORD"] == "users-own"

    def test_none_leaves_environment_untouched(self, monkeypatch):
        """password=None is the ~/.pgpass path: libpq reads the file, so nothing is set."""
        from jkp.data.wrds_connection import wrds_password_env

        monkeypatch.delenv("PGPASSWORD", raising=False)
        with wrds_password_env(None):
            assert "PGPASSWORD" not in os.environ


class TestVerifyWrdsConnection:
    """Tests for verify_wrds_connection() — the CLI-facing connectivity check."""

    @staticmethod
    def _fake_connect(monkeypatch, execute):
        """Route duckdb.connect to a context-manager connection whose execute is ``execute``."""
        from jkp.data import wrds_connection as mod

        class FakeConnection:
            def execute(self, sql, *args):
                return execute(sql)

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        monkeypatch.setattr(mod.duckdb, "connect", lambda *a, **kw: FakeConnection())
        return mod

    def test_success_runs_attach_and_probe_with_default_timeout(self, monkeypatch):
        """On the success path verify must actually ATTACH *and* run the
        information_schema probe, with the default 25s connect_timeout in the
        conninfo and the password supplied via PGPASSWORD, never in the SQL."""
        monkeypatch.delenv("PGPASSWORD", raising=False)
        recorded: list[tuple[str, str | None]] = []
        mod = self._fake_connect(
            monkeypatch, lambda sql: recorded.append((sql, os.environ.get("PGPASSWORD")))
        )

        # No explicit connect_timeout → exercises the 25s default.
        result = mod.verify_wrds_connection("testuser", "hunter2")  # noqa: S106

        assert result is None
        attach_sql, attach_env = next(r for r in recorded if "ATTACH" in r[0])
        assert "connect_timeout=25" in attach_sql  # default flows into the conninfo
        assert attach_env == "hunter2"  # libpq sees the password via PGPASSWORD
        assert all("hunter2" not in sql for sql, _ in recorded)
        assert any("information_schema" in sql for sql, _ in recorded)  # the probe ran
        assert "PGPASSWORD" not in os.environ  # restored afterwards

    def test_attach_failure_wrapped_as_runtime_error(self, monkeypatch):
        """A failed ATTACH (e.g. bad credentials) surfaces as a RuntimeError that
        keeps DuckDB's message, so `jkp connect` exits cleanly with something useful."""

        def execute(sql):
            if "ATTACH" in sql:
                raise OSError("IO Error: password authentication failed")

        mod = self._fake_connect(monkeypatch, execute)

        with pytest.raises(RuntimeError) as exc_info:
            mod.verify_wrds_connection("testuser", "hunter2", connect_timeout=1)  # noqa: S106

        msg = str(exc_info.value)
        assert "password authentication failed" in msg
        assert "credentials" in msg.lower()
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_pgpass_path_does_not_set_pgpassword(self, monkeypatch):
        """With password=None (the ~/.pgpass path) PGPASSWORD is left alone so libpq
        reads the file."""
        monkeypatch.delenv("PGPASSWORD", raising=False)
        seen: list[str | None] = []
        mod = self._fake_connect(monkeypatch, lambda sql: seen.append(os.environ.get("PGPASSWORD")))

        mod.verify_wrds_connection("testuser", None, connect_timeout=1)

        assert seen and all(v is None for v in seen)

    def test_probe_query_failure_wrapped_as_runtime_error(self, monkeypatch):
        """ATTACH succeeds but the information_schema probe fails — a valid login with
        revoked or limited WRDS product permissions, or a connection dropped between the
        attach and the first query. It must surface as the same RuntimeError."""

        def execute(sql):
            if "information_schema" in sql:
                raise OSError("Permission denied for schema information_schema")

        mod = self._fake_connect(monkeypatch, execute)

        with pytest.raises(RuntimeError, match="Permission denied"):
            mod.verify_wrds_connection("testuser", "hunter2", connect_timeout=1)  # noqa: S106

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
