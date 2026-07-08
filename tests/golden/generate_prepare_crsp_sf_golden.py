"""Generate golden fixtures for prepare_crsp_sf().

Run with:
    uv run python -m tests.golden.generate_prepare_crsp_sf_golden

Writes:
    tests/golden/fixtures/prepare_crsp_sf/crsp_msf.parquet
    tests/golden/fixtures/prepare_crsp_sf/crsp_dsf.parquet
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import prepare_crsp_sf
from jkp.data.paths import DataPaths
from tests.golden.prepare_crsp_sf_inputs import write_all_inputs

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "prepare_crsp_sf"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        for freq in ("m", "d"):
            write_all_inputs(paths, freq)
            prepare_crsp_sf(paths, freq)
            out_path = GOLDEN_DIR / f"crsp_{freq}sf.parquet"
            pl.read_parquet(paths.interim_dir / f"crsp_{freq}sf.parquet").write_parquet(out_path)
            print(f"crsp_{freq}sf.parquet: {pl.read_parquet(out_path).height} rows -> {out_path}")


if __name__ == "__main__":
    main()
