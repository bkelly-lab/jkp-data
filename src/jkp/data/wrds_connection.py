"""WRDS connection primitives.

Builds the libpq conninfo for the WRDS Postgres endpoint, attaches it read-only
on a DuckDB connection, and verifies connectivity. Kept separate from the heavy
``aux_functions`` pipeline module so the CLI's ``jkp connect`` command can check
a connection without importing the whole pipeline.

Password handling: the password is never put in the conninfo. It reaches libpq
through the ``PGPASSWORD`` environment variable (see :func:`wrds_password_env`),
so it can't show up in the ATTACH / postgres_scan SQL or in DuckDB's error text,
which echoes the full connection string.
"""

import contextlib
import os
from collections.abc import Iterator

import duckdb

from .wrds_credentials import WRDS_DB, WRDS_HOST, WRDS_PORT


def _pg_escape_value(value: str) -> str:
    """libpq-escape a conninfo value for a single-quoted field.

    libpq accepts single-quoted values with backslash-escaped ``\\`` and ``'``,
    so quoting lets a value hold spaces or special characters without breaking the
    conninfo.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _sql_literal(value: str) -> str:
    """Escape a string for embedding inside a single-quoted DuckDB SQL literal.

    The conninfo is interpolated into ``ATTACH '...'`` / ``postgres_scan('...')``
    SQL, so any single quote it contains (e.g. around a libpq-quoted username)
    must be doubled or it terminates the SQL string literal.
    """
    return value.replace("'", "''")


def gen_wrds_connection_info(user: str, *, connect_timeout: int | None = None) -> str:
    """Build a libpq conninfo for WRDS.

    The conninfo never carries the password: libpq reads it from ``PGPASSWORD``
    (set by :func:`wrds_password_env`) or, failing that, from ``$PGPASSFILE`` /
    ``~/.pgpass``. When ``connect_timeout`` is set, libpq gives up after that many
    seconds rather than hanging on an unreachable host.
    """
    parts = [
        f"host={WRDS_HOST}",
        f"port={WRDS_PORT}",
        f"dbname={WRDS_DB}",
        # Quote the username: a space or quote in it would otherwise break
        # libpq's conninfo parsing. The conninfo is itself embedded in a
        # single-quoted SQL literal at each use site, so callers must
        # additionally pass it through _sql_literal.
        f"user='{_pg_escape_value(user)}'",
        "sslmode=require",
    ]
    if connect_timeout is not None:
        parts.append(f"connect_timeout={connect_timeout}")
    return " ".join(parts)


@contextlib.contextmanager
def wrds_password_env(password: str | None) -> Iterator[None]:
    """Expose the password to libpq via ``PGPASSWORD`` for the duration of the block.

    Keeps the password out of the conninfo, and so out of the SQL and DuckDB's
    error text. With ``None`` nothing is set and libpq falls back to ``~/.pgpass``.
    Set before any worker thread starts and restored after they finish, so the
    process-wide environment is never mutated concurrently.
    """
    if password is None:
        yield
        return
    previous = os.environ.get("PGPASSWORD")
    os.environ["PGPASSWORD"] = password
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PGPASSWORD", None)
        else:
            os.environ["PGPASSWORD"] = previous


def _attach_wrds(con: duckdb.DuckDBPyConnection, conninfo: str) -> None:
    """ATTACH the WRDS Postgres database read-only on an existing DuckDB connection."""
    con.execute(f"ATTACH '{_sql_literal(conninfo)}' AS wrds (TYPE postgres, READ_ONLY)")


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
    username: str, password: str | None, *, connect_timeout: int = 25
) -> None:
    """Open a real WRDS connection and confirm it is queryable.

    Attaches the WRDS Postgres database read-only and runs a trivial query against
    it. The ATTACH authenticates eagerly (opening the libpq connection triggers the
    WRDS Duo MFA push), so a successful return means credentials, connectivity, and
    MFA all succeeded. Raises :class:`RuntimeError` on any failure, so `jkp connect`
    can print one actionable message instead of a traceback. ``connect_timeout``
    bounds how long libpq waits before failing — it must leave the user time to
    approve the Duo MFA push, so it defaults to 25s.
    """
    conninfo = gen_wrds_connection_info(username, connect_timeout=connect_timeout)
    try:
        with wrds_password_env(password):
            # Inside the try: INSTALL can itself fail (e.g. no network to DuckDB's
            # extension repo, plausible on the headless HPC nodes this targets).
            _install_postgres_extension()
            with duckdb.connect(":memory:") as con:
                con.execute("LOAD postgres;")
                _attach_wrds(con, conninfo)
                con.execute("SELECT 1 FROM wrds.information_schema.schemata LIMIT 1")
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to WRDS ({e}). Check your network, credentials, and MFA approval."
        ) from e
