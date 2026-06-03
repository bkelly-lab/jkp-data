"""One-off cluster validation of Bessembinder corrections against CRSP (US stocks).

Runs gen_comp_dsf twice (without/with Bessembinder corrections) in a sandboxed
data directory, maps permno onto the Compustat side via the CCM link table, and
compares both versions to CRSP daily data using
compare_compustat_crsp_before_after.

All outputs are contained in the sandbox (BASE) — the clean pipeline run under
REAL is only read, never written.

Usage (from a checkout of the bessembinder_correction branch):
    PYTHONPATH=src python scripts/bessembinder_us_validation.py
"""

import shutil
from pathlib import Path

import polars as pl
from polars import col

from jkp.data.aux_functions import compare_compustat_crsp_before_after, gen_comp_dsf
from jkp.data.paths import DataPaths

# Sandbox: all outputs land here. Inputs are symlinked in from REAL (read-only).
BASE = Path.home() / "bessembinder_validation" / "data"
# Clean pipeline run — read-only.
REAL = Path.home() / "jkp-data" / "data"

paths = DataPaths(base_dir=BASE)


def comp_dsf_with_permno(parquet_path: Path) -> pl.LazyFrame:
    """Load a __comp_dsf parquet, compute daily returns from ri, and map permno.

    compare_compustat_crsp_daily expects columns: permno, datadate, prccd, ret.
    """
    df = (
        pl.scan_parquet(parquet_path)
        .sort(["gvkey", "iid", "datadate"])
        .with_columns(
            (col("ri") / col("ri").shift(1).over(["gvkey", "iid"]) - 1).alias("ret"),
            col("prc").alias("prccd"),
        )
    )
    link = (
        pl.scan_parquet(REAL / "raw" / "raw_tables" / "crsp_ccmxpf_lnkhist.parquet")
        .filter(col("linktype").is_in(["LU", "LC"]) & col("linkprim").is_in(["P", "C"]))
        .select(
            "gvkey",
            col("liid").alias("iid"),
            col("lpermno").alias("permno"),
            "linkdt",
            col("linkenddt").fill_null(pl.date(2100, 1, 1)).alias("linkenddt"),
        )
    )
    return (
        df.join(link, on=["gvkey", "iid"], how="inner")
        .filter((col("datadate") >= col("linkdt")) & (col("datadate") <= col("linkenddt")))
        .drop(["linkdt", "linkenddt"])
    )


def main() -> None:
    uncorrected_path = paths.interim_dir / "__comp_dsf_uncorrected.parquet"
    corrected_path = paths.interim_dir / "__comp_dsf.parquet"

    # 1) Baseline: no corrections
    if not uncorrected_path.exists():
        print("=== Running gen_comp_dsf WITHOUT Bessembinder corrections ===", flush=True)
        gen_comp_dsf(paths, apply_bessembinder=False)
        shutil.move(corrected_path, uncorrected_path)
    else:
        print(f"Reusing existing {uncorrected_path}", flush=True)

    # 2) Corrected: Section 6 + Section 8 (writes corrections log alongside)
    print("=== Running gen_comp_dsf WITH Bessembinder corrections ===", flush=True)
    gen_comp_dsf(paths, apply_bessembinder=True)

    # 3) Map permno onto both Compustat versions
    before = comp_dsf_with_permno(uncorrected_path)
    after = comp_dsf_with_permno(corrected_path)

    # 4) CRSP daily (ground truth) and before/after comparison
    crsp = pl.scan_parquet(REAL / "interim" / "crsp_dsf.parquet").select(
        "permno", "date", "prc", "ret"
    )
    results = compare_compustat_crsp_before_after(
        before, after, crsp, output_dir=str(BASE / "interim")
    )
    print(results["improvement"], flush=True)


if __name__ == "__main__":
    main()
