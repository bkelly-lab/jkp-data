"""Generate golden fixture for market_beta().

Run with:
    uv run python -m tests.golden.generate_market_beta_golden

Writes:
    tests/golden/fixtures/market_beta/beta_60m.parquet
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import market_beta
from jkp.data.paths import DataPaths
from tests.golden.market_beta_inputs import (
    build_ap_factors_monthly_input,
    build_world_msf_input,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "market_beta"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        paths.interim_dir.mkdir(parents=True, exist_ok=True)
        data_path = paths.interim_dir / "world_msf.parquet"
        fcts_path = paths.interim_dir / "ap_factors_monthly.parquet"
        build_world_msf_input(seed=42).write_parquet(data_path)
        build_ap_factors_monthly_input(seed=42).write_parquet(fcts_path)

        out_path = paths.interim_dir / "beta_60m.parquet"
        market_beta(paths, out_path, data_path, fcts_path, 60, 36)

        fixture_path = GOLDEN_DIR / "beta_60m.parquet"
        df = pl.read_parquet(out_path)
        df.write_parquet(fixture_path)
        print(f"beta_60m.parquet: {df.height} rows -> {fixture_path}")


if __name__ == "__main__":
    main()
