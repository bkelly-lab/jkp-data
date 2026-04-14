"""
Output writer module for writing DataFrames.

The pipeline always writes parquet files internally. If CSV output is configured,
call convert_outputs_to_csv() at the end to convert all parquet files to CSV.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import polars as pl


class OutputFormat(Enum):
    """Supported output file formats."""

    PARQUET = "parquet"
    CSV = "csv"


# Valid format strings for configure_output_format()
VALID_OUTPUT_FORMATS = ("csv", "parquet")

# Global configuration (immutable after first configuration)
_output_format: OutputFormat = OutputFormat.PARQUET
_configured: bool = False


def configure_output_format(format_str: str, *, _allow_reset: bool = False) -> None:
    """
    Description:
        Configure the final output format for pipeline outputs. This function
        can only be called once per process to prevent accidental reconfiguration.

    Steps:
        1) Validate format_str is "csv" or "parquet".
        2) Check if already configured (raise RuntimeError unless _allow_reset).
        3) Set global _output_format and mark as configured.

    Output:
        None. Sets module-level state. Prints format confirmation message.
    """
    global _output_format, _configured

    if format_str not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output format: '{format_str}'. Must be one of: {VALID_OUTPUT_FORMATS}"
        )

    if _configured and not _allow_reset:
        raise RuntimeError(
            "Output format already configured. "
            "configure_output_format() can only be called once per process."
        )

    if format_str == "csv":
        _output_format = OutputFormat.CSV
        print("Output format: CSV (will convert at end of pipeline)")
    else:
        _output_format = OutputFormat.PARQUET
        print("Output format: Parquet")
    _configured = True


def get_output_format() -> OutputFormat:
    """
    Description:
        Get the current output format configuration.

    Steps:
        1) Return the module-level _output_format value.

    Output:
        OutputFormat enum value (PARQUET or CSV).
    """
    return _output_format


def _collect_dataframe(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    streaming: bool = False,
    shrink_dtype: bool = False,
) -> pl.DataFrame:
    """
    Description:
        Collect a LazyFrame to a DataFrame, applying optional transformations.

    Steps:
        1) If input is LazyFrame and shrink_dtype requested, apply shrink_dtype.
        2) Collect using streaming engine if requested, otherwise standard collect.
        3) If input is DataFrame and shrink_dtype requested, apply shrink_dtype.
        4) Return the resulting DataFrame.

    Output:
        Collected pl.DataFrame with optional dtype shrinking applied.
    """
    if isinstance(df, pl.LazyFrame):
        if shrink_dtype:
            df = df.select(pl.all().shrink_dtype())
        if streaming:
            return df.collect(engine="streaming")
        return df.collect()
    elif shrink_dtype:
        return df.select(pl.all().shrink_dtype())
    return df


def write_dataframe(
    df: pl.DataFrame | pl.LazyFrame,
    path: str | Path,
    *,
    streaming: bool = False,
    shrink_dtype: bool = False,
) -> None:
    """
    Description:
        Write a DataFrame to Parquet format. Always writes parquet internally;
        use convert_outputs_to_csv() at pipeline end for CSV output. This function
        differs from collect_and_write() in aux_functions.py by supporting
        shrink_dtype and ensuring .parquet extension.

    Steps:
        1) Collect the DataFrame (handles LazyFrame input).
        2) Ensure path has .parquet extension.
        3) Create parent directories if needed.
        4) Write to parquet format.

    Output:
        Parquet file written to disk at the specified path.
    """
    path = Path(path)
    collected_df = _collect_dataframe(df, streaming=streaming, shrink_dtype=shrink_dtype)

    # Always write parquet (ensure extension)
    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    collected_df.write_parquet(path)


def _convert_parquet_to_csv(parquet_path: Path, csv_path: Path) -> None:
    """
    Description:
        Convert a single parquet file to CSV with quoted strings using streaming.

    Steps:
        1) Scan parquet file lazily.
        2) Sink to CSV with quote_style="non_numeric" to preserve leading zeros.

    Output:
        CSV file written to csv_path.
    """
    pl.scan_parquet(parquet_path).sink_csv(
        csv_path,
        quote_style="non_numeric",
        null_value="",
    )


def convert_outputs_to_csv(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path | None = None,
) -> None:
    """
    Description:
        Convert all parquet output files to CSV format and remove the parquet files.
        Uses parallel threads for efficient I/O. Call at end of pipeline if CSV
        output is configured.

    Steps:
        1) Check if CSV format is configured; skip if not.
        2) Find all .parquet files recursively in processed_dir.
        3) Convert each file to CSV using streaming (parallel threads).
        4) Remove original parquet files after successful conversion.

    Output:
        CSV files written to output_dir (or processed_dir if output_dir is None).
        Original parquet files are deleted.
    """
    if _output_format != OutputFormat.CSV:
        print("Output format is parquet, skipping CSV conversion")
        return

    start_time = time.time()

    processed_dir = Path(processed_dir)
    if output_dir is None:
        output_base = processed_dir
    else:
        output_base = Path(output_dir)

    # Collect all parquet files first
    parquet_files = list(processed_dir.rglob("*.parquet"))

    if not parquet_files:
        print("No parquet files to convert")
        return

    # Use half of available CPUs to avoid saturating disk I/O
    max_workers = max(1, (os.cpu_count() or 4) // 2)
    print(f"Converting {len(parquet_files)} parquet files to CSV ({max_workers} threads)...")

    def convert_single_file(parquet_file: Path) -> None:
        """Convert a single parquet file to CSV and remove the original."""
        rel_path = parquet_file.relative_to(processed_dir)
        csv_file = output_base / rel_path.with_suffix(".csv")
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        _convert_parquet_to_csv(parquet_file, csv_file)
        parquet_file.unlink()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_single_file, f): f for f in parquet_files}
        for future in as_completed(futures):
            # Re-raise any exceptions from worker threads
            future.result()

    elapsed = time.time() - start_time
    print(f"CSV conversion complete: {len(parquet_files)} files in {elapsed:.1f}s")
