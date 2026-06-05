"""One-off cluster validation of Bessembinder corrections against CRSP (US stocks).

Runs gen_comp_dsf twice (without/with Bessembinder corrections) in a sandboxed
data directory, maps permno onto the Compustat side via the CCM link table, and
compares both versions to CRSP daily data using
compare_compustat_crsp_before_after.

All outputs are contained in the sandbox (BASE) — the clean pipeline run under
REAL is only read, never written.

Usage (from a checkout of the bessembinder_correction branch):
    PYTHONPATH=src python scripts/bessembinder_us_validation.py [method] [overrides.json]

method: Section 6 correction method, 'bessembinder' (default) or
'interpolation'. Corrected outputs and comparison files are suffixed with
the method so runs can coexist.

overrides.json: optional threshold-sweep spec, e.g.
    {"tag": "g05", "variation_threshold": 1.3, "s8": {"g_ret": 0.5}}
"tag" suffixes all outputs as <method>_<tag>; "s8" keys are Section8Params
fields (unset fields keep paper defaults).
"""

import json
import shutil
import sys
from pathlib import Path

import polars as pl
from polars import col

from jkp.data.aux_functions import compare_compustat_crsp_before_after, gen_comp_dsf
from jkp.data.bessembinder import Section8Params
from jkp.data.paths import DataPaths

# Sandbox: all outputs land here. Inputs are symlinked in from REAL (read-only).
BASE = Path.home() / "bessembinder_validation" / "data"
# Clean pipeline run — read-only.
REAL = Path.home() / "jkp-data" / "data"

paths = DataPaths(base_dir=BASE)


def comp_dsf_with_permno(parquet_path: Path, assume_sorted: bool = False) -> pl.LazyFrame:
    """Load a __comp_dsf parquet, compute daily returns from ri, and map permno.

    compare_compustat_crsp_daily expects columns: permno, datadate, prccd, ret.
    assume_sorted skips the (gvkey, iid, datadate) sort — correct for corrected
    outputs, which the pipeline writes in exactly that order.
    """
    lf = pl.scan_parquet(parquet_path)
    if not assume_sorted:
        lf = lf.sort(["gvkey", "iid", "datadate"])
    df = lf.with_columns(
        (col("ri") / col("ri").shift(1).over(["gvkey", "iid"]) - 1).alias("ret"),
        col("prc").alias("prccd"),
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
    method = sys.argv[1] if len(sys.argv) > 1 else "bessembinder"
    overrides = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
    s8_params = Section8Params(**overrides.get("s8", {}))
    variation_threshold = overrides.get("variation_threshold", 1.3)
    variant = f"{method}_{overrides['tag']}" if overrides.get("tag") else method

    uncorrected_path = paths.interim_dir / "__comp_dsf_uncorrected.parquet"
    pipeline_path = paths.interim_dir / "__comp_dsf.parquet"
    corrected_path = paths.interim_dir / f"__comp_dsf_{variant}.parquet"

    # 1) Baseline: no corrections
    if not uncorrected_path.exists():
        print("=== Running gen_comp_dsf WITHOUT Bessembinder corrections ===", flush=True)
        gen_comp_dsf(paths, apply_bessembinder=False)
        shutil.move(pipeline_path, uncorrected_path)
    else:
        print(f"Reusing existing {uncorrected_path}", flush=True)

    # 2) Corrected: Section 6 + Section 8 (writes corrections log alongside)
    if not corrected_path.exists():
        print(f"=== Running gen_comp_dsf WITH Bessembinder corrections ({variant}) ===", flush=True)
        gen_comp_dsf(
            paths,
            apply_bessembinder=True,
            correction_method=method,
            variation_threshold=variation_threshold,
            s8_params=s8_params,
        )
        shutil.move(pipeline_path, corrected_path)
        log_path = paths.interim_dir / "bessembinder_corrections_log.parquet"
        if log_path.exists():
            shutil.move(
                log_path, paths.interim_dir / f"bessembinder_corrections_log_{variant}.parquet"
            )
    else:
        print(f"Reusing existing {corrected_path}", flush=True)

    # 3) Map permno onto both Compustat versions (corrected outputs are
    # written already sorted; the uncorrected baseline is not)
    before = comp_dsf_with_permno(uncorrected_path)
    after = comp_dsf_with_permno(corrected_path, assume_sorted=True)

    # 4) CRSP daily (ground truth) and before/after comparison. The BEFORE
    # side is method-invariant: cached once next to the uncorrected file.
    crsp = pl.scan_parquet(REAL / "interim" / "crsp_dsf.parquet").select(
        "permno", "date", "prc", "ret"
    )
    out_dir = BASE / "interim" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    results = compare_compustat_crsp_before_after(
        before,
        after,
        crsp,
        output_dir=str(out_dir),
        before_cache_path=str(paths.interim_dir / "crsp_comparison_before_cache.parquet"),
    )
    print(f"method={variant}", flush=True)
    print(results["improvement"], flush=True)


if __name__ == "__main__":
    main()
