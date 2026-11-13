"""
Check that the synthetic Compustat inputs copy nothing from the real ones (issue #303, step 6).
Generates the inputs (synthetic_compustat.py) into a temporary folder and compares them with DATA_DIR.

Runs read-only on Yale (check_no_leak.slurm) and prints only counts, all of which must be 0:
    gvkeys        synthetic gvkeys that occur in the real Compustat tables or world_msf
    <table>       synthetic rows whose (datadate, at, sale), or (datadate, atq, saleq) in the
                  quarterly tables, equal a real row's with both items rounded to DECIMALS

Usage:
    python -m tests.golden.calibration.check_no_leak DATA_DIR
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import polars as pl

from jkp.data.paths import DataPaths
from tests.golden.synthetic_compustat import generate

# table -> its asset and sales items
ITEMS = {
    "comp_funda": ("at", "sale"),
    "comp_g_funda": ("at", "sale"),
    "comp_fundq": ("atq", "saleq"),
    "comp_g_fundq": ("atq", "saleq"),
}
DECIMALS = 3  # synthetic items are rounded to 3 decimals (synthetic_compustat.to_local_currency)


def _gvkeys(paths: DataPaths) -> pl.LazyFrame:
    tables = [pl.scan_parquet(paths.raw_tables_dir / f"{t}.parquet") for t in ITEMS]
    tables.append(pl.scan_parquet(paths.interim_dir / "world_msf.parquet"))
    return pl.concat([t.select("gvkey") for t in tables]).drop_nulls().unique()


def _rows(paths: DataPaths, table: str) -> pl.LazyFrame:
    """(datadate, assets, sales) of every row with both items, rounded to DECIMALS."""
    items = [pl.col(c).cast(pl.Float64).round(DECIMALS) for c in ITEMS[table]]
    return (
        pl.scan_parquet(paths.raw_tables_dir / f"{table}.parquet")
        .select("datadate", *items)
        .drop_nulls()
    )


def leaks(real: DataPaths, synthetic: DataPaths) -> dict[str, int]:
    """
    Description:
        Count the synthetic gvkeys and rows that also occur in the real data.
    Steps:
        1) Synthetic gvkeys semi-joined to the real gvkeys of the four tables and world_msf.
        2) Per table, synthetic (datadate, assets, sales) semi-joined to the real ones.
    Output:
        Check name -> number of matches.
    """
    counts = {"gvkeys": _gvkeys(synthetic).join(_gvkeys(real), on="gvkey", how="semi")}
    for table, (at, sale) in ITEMS.items():
        on = ["datadate", at, sale]
        counts[table] = _rows(synthetic, table).join(_rows(real, table), on=on, how="semi")
    return {
        name: lf.select(pl.len()).collect(engine="streaming").item() for name, lf in counts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("data_dir", type=Path, help="real data root (raw/raw_tables, interim)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
        counts = leaks(DataPaths(args.data_dir), DataPaths(Path(tmp)))
    print(json.dumps(counts, indent=2))
    sys.exit(1 if any(counts.values()) else 0)


if __name__ == "__main__":
    main()
