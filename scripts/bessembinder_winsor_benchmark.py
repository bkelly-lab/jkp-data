"""Winsorization benchmark for the Bessembinder validation.

Description:
    Naive-cleaning baseline: instead of Bessembinder corrections, winsorize
    the UNCORRECTED Compustat daily returns cross-sectionally (per date) at
    the 0.1% / 99.9% quantiles, then compare to CRSP with the same harness
    as the corrected runs.
Steps:
    1) Build permno-mapped returns from __comp_dsf_uncorrected.parquet.
    2) Per datadate, clip ret to [q0.001, q0.999] (only tail obs affected).
    3) compare_compustat_crsp_before_after(before=uncorrected,
       after=winsorized) -> stats to interim/winsor_0.1/.
Output:
    Comparison parquets + printed improvement summary.

Usage (from a checkout of the bessembinder_correction branch):
    PYTHONPATH=src python scripts/bessembinder_winsor_benchmark.py [source]

source: which __comp_dsf_<source>.parquet to winsorize ('uncorrected'
default; pass a correction method, e.g. 'bessembinder', to measure
winsorization applied ON TOP of corrected data).
"""

import sys

import polars as pl
from bessembinder_us_validation import BASE, REAL, comp_dsf_with_permno, paths
from polars import col

from jkp.data.aux_functions import compare_compustat_crsp_before_after

LOWER, UPPER = 0.001, 0.999


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "uncorrected"
    uncorrected = comp_dsf_with_permno(paths.interim_dir / f"__comp_dsf_{source}.parquet")

    # Cross-sectional (per-day) winsorization of returns at 0.1% / 99.9%
    winsorized = uncorrected.with_columns(
        col("ret")
        .clip(
            col("ret").quantile(LOWER).over("datadate"),
            col("ret").quantile(UPPER).over("datadate"),
        )
        .alias("ret")
    )

    crsp = pl.scan_parquet(REAL / "interim" / "crsp_dsf.parquet").select(
        "permno", "date", "prc", "ret"
    )
    out_dir = BASE / "interim" / ("winsor_0.1" if source == "uncorrected" else f"winsor_{source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = compare_compustat_crsp_before_after(
        uncorrected, winsorized, crsp, output_dir=str(out_dir)
    )
    print(f"method=winsor_0.1_99.9_cross_sectional source={source}", flush=True)
    print(results["improvement"], flush=True)


if __name__ == "__main__":
    main()
