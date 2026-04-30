"""JKP Data CLI - Factor data generation pipeline."""

from enum import StrEnum
from pathlib import Path

import typer


class OutputFormat(StrEnum):
    """Supported output file formats."""

    parquet = "parquet"
    csv = "csv"


app = typer.Typer(
    name="jkp",
    help="JKP Factor Data generation pipeline.",
    no_args_is_help=True,
)


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

    run_pipeline(persistent_connection=persistent_connection, output_dir=output_dir)


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
         containers and shared service accounts.
      2. The system keyring (Keychain on macOS, Secret Service on Linux desktop,
         Credential Vault on Windows). Default for interactive sessions.
      3. The file-backed keyring (keyrings.alt.file.PlaintextKeyring), which
         stores the password in a mode-600 file under
         ~/.local/share/python_keyring/. Selected only when
         JKP_ALLOW_PLAINTEXT_KEYRING=1 is set; appropriate for headless
         environments (HPC compute nodes, minimal Docker images) where no
         system keyring daemon is available.
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
