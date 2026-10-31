"""Generate golden fixtures for the final merge and save stage.

Run with:
    uv run python -m tests.golden.generate_final_stage_golden

Writes to tests/golden/fixtures/final_stage/:
    world_data_prelim, world_data_-1, __msf_world2, world_ret_monthly,
    daily_rets (all per-country daily files stacked, with a ``country`` column),
    accounting_quarterly, accounting_annual.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.golden.final_stage_inputs import write_full_inputs

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "final_stage"

# fixture name -> sort keys applied before writing and comparing
SORT_KEYS: dict[str, list[str]] = {
    "world_data_prelim": ["id", "eom"],
    "world_data_-1": ["id", "eom"],
    "__msf_world2": ["id", "eom"],
    "world_ret_monthly": ["id", "eom"],
    "daily_rets": ["country", "id", "date"],
    "accounting_quarterly": ["gvkey", "datadate"],
    "accounting_annual": ["gvkey", "datadate"],
}


def compute_outputs(paths: DataPaths) -> dict[str, pl.DataFrame]:
    """Run the merge chain and the save functions on the synthetic inputs.

    Description:
        Executes create_world_data_prelim, merge_world_data_prelim,
        merge_industry_to_world_msf, then the filter/save steps up to
        save_accounting_data, and reads back the golden outputs.
    Steps:
        1) Create the folder tree and write all interim inputs.
        2) Run the merges and the save functions.
        3) Read each output, stack the daily country files, sort deterministically.
    Output:
        Dict of fixture name to sorted DataFrame.
    """
    af.setup_folder_structure(paths)
    write_full_inputs(paths)
    i = paths.interim_dir
    af.create_world_data_prelim(
        paths,
        i / "world_msf.parquet",
        i / "market_chars_m.parquet",
        i / "acc_chars_world.parquet",
        i / "world_data_prelim.parquet",
    )
    af.merge_world_data_prelim(paths)
    af.merge_industry_to_world_msf(paths)
    af.filter_dsf(paths)
    af.filter_msf(paths)
    af.save_daily_ret(paths)
    af.save_monthly_ret(paths)
    af.save_accounting_data(paths)

    daily_dir = paths.processed_dir / "return_data" / "daily_rets_by_country"
    daily = pl.concat(
        [
            pl.read_parquet(f).with_columns(country=pl.lit(f.stem))
            for f in sorted(daily_dir.glob("*.parquet"))
        ],
        how="diagonal",
    )
    out = {
        "world_data_prelim": pl.read_parquet(i / "world_data_prelim.parquet"),
        "world_data_-1": pl.read_parquet(i / "world_data_-1.parquet"),
        "__msf_world2": pl.read_parquet(i / "__msf_world2.parquet"),
        "world_ret_monthly": pl.read_parquet(
            paths.processed_dir / "return_data" / "world_ret_monthly.parquet"
        ),
        "daily_rets": daily,
        "accounting_quarterly": pl.read_parquet(
            paths.processed_dir / "accounting_data" / "quarterly.parquet"
        ),
        "accounting_annual": pl.read_parquet(
            paths.processed_dir / "accounting_data" / "annual.parquet"
        ),
    }
    return {k: v.sort(SORT_KEYS[k], maintain_order=True) for k, v in out.items()}


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        outputs = compute_outputs(DataPaths(base_dir=Path(td)))
    for name, df in outputs.items():
        df.write_parquet(GOLDEN_DIR / f"{name}.parquet")
        print(f"{name}: {df.height} rows")


if __name__ == "__main__":
    main()
