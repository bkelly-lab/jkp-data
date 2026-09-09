"""JKP Data CLI - Factor data generation pipeline."""

from enum import StrEnum
from pathlib import Path

import typer

from . import __version__


class OutputFormat(StrEnum):
    """Supported output file formats."""

    parquet = "parquet"
    csv = "csv"


app = typer.Typer(
    name="jkp",
    help="JKP Factor Data generation pipeline.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
) -> None:
    """JKP Factor Data generation pipeline."""


@app.command()
def build(
    output_dir: Path = typer.Argument(
        help="Directory for pipeline output (raw, interim, and processed data).",
    ),
    persistent_connection: bool = typer.Option(
        False,
        "--persistent-connection",
        "-p",
        help="Use a single persistent WRDS connection (reduces MFA prompts on NAT-rotated networks).",
    ),
    download_workers: int = typer.Option(
        1,
        "--download-workers",
        "-j",
        help=(
            "Number of concurrent WRDS download connections (default 1 = sequential). "
            "Higher values download tables in parallel and cut wall time, capped at the WRDS "
            "per-account connection limit. Ignored when --persistent-connection is set (which "
            "forces a single connection). On NAT-rotated networks each worker may trigger its "
            "own MFA prompt at startup; keep at 1 there."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing data in output directory without prompting.",
    ),
) -> None:
    """Run the full data generation pipeline."""
    from .main import run_pipeline

    if not force and output_dir.exists() and any(output_dir.iterdir()):
        typer.confirm(
            f"Output directory '{output_dir}' already contains data. Overwrite?",
            abort=True,
        )

    run_pipeline(
        persistent_connection=persistent_connection,
        max_workers=download_workers,
        output_dir=output_dir,
    )


@app.command()
def portfolio(
    output_dir: Path = typer.Argument(
        help="Directory containing pipeline output (must match output_dir from build).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.parquet,
        "--output-format",
        help="Output file format.",
    ),
) -> None:
    """Generate factor portfolios from characteristics data."""
    from .portfolio import run_portfolio

    run_portfolio(output_format=output_format.value, output_dir=output_dir)


# This help text is the canonical description of the credential precedence
# order; the README, the wrds_credentials module docstring, and the tests point
# here rather than restating it. Update here first.
@app.command()
def connect(
    reset: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="Reset stored WRDS credentials.",
    ),
) -> None:
    """Verify the WRDS connection, or configure/reset stored credentials.

    Opens a real WRDS connection (attaches the database read-only and runs a
    trivial query), so a successful run confirms credentials, connectivity,
    and MFA all actually work. A password entered at the prompt is verified
    before it is stored, so a typo is never persisted and simply re-prompts on
    the next run. This may trigger a WRDS Duo MFA push: WRDS trusts a
    username/IP pair for 30 days after a successful approval, so expect one on the
    first run from a new IP and again once that window expires.

    Credential precedence (highest first):

      1. WRDS_USERNAME and WRDS_PASSWORD environment variables. Useful for
         containers, CI, and shared service accounts.
      2. The system keyring (Keychain on macOS, Secret Service on Linux desktop,
         Credential Vault on Windows). Default for interactive desktop sessions.
      3. A libpq password file: $PGPASSFILE, else ~/.pgpass
         (%APPDATA%\\postgresql\\pgpass.conf on Windows). The standard
         Postgres/WRDS mechanism, ideal for headless HPC nodes — jkp omits the
         password from the connection string and libpq reads the file.

    Running `jkp connect` stores the password in the system keyring where one is
    available. On a headless login node (no keyring, but an interactive
    terminal) it writes ~/.pgpass (mode 600) instead; because $HOME is shared
    with the compute nodes, batch jobs then read it without any further setup.
    The selected source is printed to stderr on every run.
    """
    from .wrds_credentials import get_wrds_credentials, reset_credentials

    try:
        if reset:
            reset_credentials(full_reset=True)
            typer.echo("Credentials reset.")
            return

        # Imported before resolving credentials, not after: this is the process's
        # first `duckdb` import, and a broken native wheel (GLIBC mismatch on an HPC
        # node) raises ImportError, which is not in the except tuple below and so
        # escapes to Typer's pretty-exception handler. That handler prints frame
        # locals on typer < 0.23.0, and pyproject allows typer>=0.15.0 — so no
        # plaintext password may be bound in this frame when the import runs.
        from .wrds_connection import verify_wrds_connection

        # Injected rather than called on the result: on the freshly-prompted path the
        # check has to run between the prompt and the store, which is inside
        # get_wrds_credentials. It verifies every resolution path exactly once, so
        # verifying again here would open a second connection (and risk a second Duo
        # push). Passing the function also keeps the plaintext password out of this
        # frame on the prompt path.
        creds = get_wrds_credentials(verify=verify_wrds_connection)
        typer.echo(f"Connected as: {creds.username}")
    except (RuntimeError, ValueError, OSError) as exc:
        # Anticipated, actionable failures from credential resolution and connection
        # verification: RuntimeError (no/empty username, unreadable ~/.pgpass, a failed
        # WRDS attach), ValueError (e.g. a password containing a newline), and OSError
        # (e.g. an unwritable state dir when persisting the username / writing ~/.pgpass).
        # Their messages are password-free; surface the message and exit non-zero rather
        # than dumping a traceback.
        typer.echo(str(exc), err=True)
        # Still worth pointing at: a password typed at the prompt is now verified
        # before it is stored, but one that arrived from an env var, an entry
        # predating this check, or a hand-edited ~/.pgpass can still be wrong, and
        # resolution never re-prompts while a stored value exists. Kept at the CLI
        # layer because the wrds_connection messages are shared with pipeline worker
        # paths, where --reset is not the right advice.
        typer.echo(
            "If a stored password is wrong, run `jkp connect --reset` and re-enter credentials.",
            err=True,
        )
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
