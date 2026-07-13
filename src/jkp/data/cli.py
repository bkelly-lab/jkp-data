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
    """Test or configure the WRDS connection.

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

    if reset:
        reset_credentials(full_reset=True)
        typer.echo("Credentials reset.")
        return

    creds = get_wrds_credentials()
    typer.echo(f"Connected as: {creds.username}")


if __name__ == "__main__":
    app()
