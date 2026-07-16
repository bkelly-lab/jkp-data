"""WRDS connection primitives.

Builds the libpq conninfo for the WRDS Postgres endpoint, attaches it read-only
on a DuckDB connection, and verifies connectivity. Kept separate from the heavy
``aux_functions`` pipeline module so the CLI's ``jkp connect`` command can check
a connection without importing the whole pipeline.

Password handling: the password only ever appears inside the conninfo and the
DuckDB ATTACH statement. DuckDB's postgres extension echoes the full connection
string (password included) in ATTACH error text, so the masking helpers here
detect and redact every escaped form the password can take in an error message.
"""

import duckdb

from .wrds_credentials import WRDS_DB, WRDS_HOST, WRDS_PORT


def _pg_escape_value(value: str) -> str:
    """libpq-escape a conninfo value (user or password) for a single-quoted field.

    libpq accepts single-quoted values with backslash-escaped ``\\`` and ``'``,
    so quoting lets a value hold spaces or special characters without breaking the
    conninfo. For the password this is also the form that appears in any error text
    echoing the connection string, so the credential-masking checks reuse it rather
    than matching the raw password.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _sql_literal(value: str) -> str:
    """Escape a string for embedding inside a single-quoted DuckDB SQL literal.

    The conninfo is interpolated into ``ATTACH '...'`` / ``postgres_scan('...')``
    SQL, so any single quote it contains (e.g. around a libpq-quoted password)
    must be doubled or it terminates the SQL string literal.
    """
    return value.replace("'", "''")


def gen_wrds_connection_info(
    user, password: str | None = None, *, connect_timeout: int | None = None
) -> str:
    """Build a libpq conninfo for WRDS.

    When ``password`` is ``None`` the ``password=`` field is omitted, so libpq
    authenticates from ``$PGPASSFILE`` / ``~/.pgpass`` instead. When
    ``connect_timeout`` is set, libpq gives up after that many seconds rather than
    hanging on an unreachable host.
    """
    parts = [
        f"host={WRDS_HOST}",
        f"port={WRDS_PORT}",
        f"dbname={WRDS_DB}",
        # Quote the username too: a space or quote in it would otherwise break
        # libpq's conninfo parsing exactly as an unquoted password would.
        f"user='{_pg_escape_value(user)}'",
    ]
    if password is not None:
        # Single-quote and escape so a password containing spaces or special
        # characters can't break the conninfo (or split the value, which would
        # defeat the password-masking check in _attach_wrds). The conninfo is
        # itself embedded in a single-quoted SQL literal at each use site, so
        # callers must additionally pass it through _sql_literal.
        parts.append(f"password='{_pg_escape_value(password)}'")
    parts.append("sslmode=require")
    if connect_timeout is not None:
        parts.append(f"connect_timeout={connect_timeout}")
    return " ".join(parts)


def _password_forms(password: str) -> tuple[str, str, str]:
    """The forms the password can take on its way into an error message: raw, the
    libpq-escaped conninfo form (echoed by a connection IOException), and the
    SQL-escaped-then-libpq-escaped form (echoed from the raw statement text by a
    parser error). Ordered most-escaped first so redaction replaces the longest
    match before its shorter substrings."""
    escaped = _pg_escape_value(password)
    return (_sql_literal(escaped), escaped, password)


def _password_in_error(text: str, password: str) -> bool:
    """True if the password appears in ``text`` in any of the forms it can take in
    an error message (see :func:`_password_forms`)."""
    return any(form in text for form in _password_forms(password))


def _redact_password(text: str, password: str) -> str:
    """Replace the password with ``***`` in every form it can take in an error
    message (see :func:`_password_forms`)."""
    for form in _password_forms(password):
        text = text.replace(form, "***")
    return text


def _attach_wrds(con: duckdb.DuckDBPyConnection, conninfo: str, password: str | None) -> None:
    """ATTACH the WRDS Postgres database read-only on an existing DuckDB connection.

    DuckDB's postgres extension embeds the full connection string (including the password) in
    ATTACH error text, so on failure suppress the original exception and raise a generic,
    password-free error. Errors that don't contain the password propagate unchanged.
    """
    try:
        con.execute(f"ATTACH '{_sql_literal(conninfo)}' AS wrds (TYPE postgres, READ_ONLY)")
    except Exception as e:
        if password and _password_in_error(str(e), password):
            raise RuntimeError(
                "Failed to attach WRDS connection. Check credentials and MFA approval."
            ) from None
        raise


def _install_postgres_extension() -> None:
    """Install the DuckDB postgres extension once, up front (idempotent).

    Doing it in the main thread means the parallel workers only ``LOAD`` it (a per-connection,
    no-download operation), which avoids a concurrent-INSTALL race across the pool writing the same
    extension file, and surfaces a fetch failure as one clean error here rather than N worker
    tracebacks.
    """
    with duckdb.connect(":memory:") as con:
        con.execute("INSTALL postgres;")


def verify_wrds_connection(
    username: str, password: str | None, *, connect_timeout: int = 10
) -> None:
    """Open a real WRDS connection and confirm it is queryable.

    Attaches the WRDS Postgres database read-only and runs a trivial query against
    it. The ATTACH authenticates eagerly (opening the libpq connection triggers the
    WRDS Duo MFA push), so a successful return means credentials, connectivity, and
    MFA all succeeded. Raises :class:`RuntimeError` with a password-free message on
    any failure. ``connect_timeout`` bounds how long libpq waits on an unreachable
    host before failing.
    """
    conninfo = gen_wrds_connection_info(username, password, connect_timeout=connect_timeout)
    _install_postgres_extension()
    try:
        with duckdb.connect(":memory:") as con:
            con.execute("LOAD postgres;")
            _attach_wrds(con, conninfo, password)
            con.execute("SELECT 1 FROM wrds.information_schema.schemata LIMIT 1")
    except RuntimeError:
        # _attach_wrds already raises a friendly, password-free RuntimeError when
        # the failure text embeds the password; pass it through unchanged.
        raise
    except Exception as e:
        # Any other failure (e.g. a ~/.pgpass auth path where password is None, so
        # _attach_wrds re-raises the raw DuckDB exception, or a LOAD/query error).
        # Surface one actionable, password-free message so callers that only handle
        # RuntimeError (e.g. `jkp connect`) exit cleanly instead of dumping a traceback.
        raise RuntimeError("Failed to connect to WRDS. Check credentials and MFA approval.") from e
