"""JKP Data CLI - Factor data generation pipeline."""

from pathlib import Path

import typer

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
) -> None:
    """Run the full data generation pipeline."""
    from .main import run_pipeline

    run_pipeline(persistent_connection=persistent_connection, output_dir=output_dir)


@app.command()
def portfolio(
    output_dir: Path = typer.Argument(
        help="Directory containing pipeline output (must match output_dir from build).",
    ),
    output_format: str = typer.Option(
        "parquet",
        "--output-format",
        help="Output file format (parquet or csv).",
    ),
) -> None:
    """Generate factor portfolios from characteristics data."""
    from .portfolio import run_portfolio

    run_portfolio(output_format=output_format, output_dir=output_dir)


@app.command()
def connect(
    reset: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="Reset stored WRDS credentials.",
    ),
) -> None:
    """Test or configure WRDS connection."""
    from .wrds_credentials import get_wrds_credentials, reset_credentials

    if reset:
        reset_credentials(full_reset=True)
        typer.echo("Credentials reset.")
        return

    creds = get_wrds_credentials()
    typer.echo(f"Connected as: {creds.username}")


if __name__ == "__main__":
    app()
