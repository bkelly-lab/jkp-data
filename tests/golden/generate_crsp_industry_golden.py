"""Generate golden fixture for crsp_industry().

Run with:
    uv run python -m tests.golden.generate_crsp_industry_golden

Writes:
    tests/golden/fixtures/crsp_industry/crsp_ind.parquet
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import crsp_industry
from jkp.data.paths import DataPaths
from tests.golden.crsp_industry_inputs import build_permno0_input

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "crsp_industry"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        (paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
        build_permno0_input().write_parquet(paths.interim_dir / "raw_data_dfs" / "permno0.parquet")
        crsp_industry(paths)
        out_path = GOLDEN_DIR / "crsp_ind.parquet"
        pl.read_parquet(paths.interim_dir / "crsp_ind.parquet").write_parquet(out_path)
        n_rows = pl.read_parquet(out_path).height
        print(f"crsp_ind.parquet: {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
