"""Generate golden fixture for comp_sic_naics().

Run with:
    uv run python -m tests.golden.generate_comp_sic_naics_golden

Writes:
    tests/golden/fixtures/comp_sic_naics/comp_other.parquet
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import comp_sic_naics
from jkp.data.paths import DataPaths
from tests.golden.comp_sic_naics_inputs import build_sic_naics_inputs

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "comp_sic_naics"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        (paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
        sic_naics_na, sic_naics_gl = build_sic_naics_inputs()
        sic_naics_na.write_parquet(paths.interim_dir / "raw_data_dfs" / "sic_naics_na.parquet")
        sic_naics_gl.write_parquet(paths.interim_dir / "raw_data_dfs" / "sic_naics_gl.parquet")
        comp_sic_naics(paths)
        out_path = GOLDEN_DIR / "comp_other.parquet"
        pl.read_parquet(paths.interim_dir / "comp_other.parquet").write_parquet(out_path)
        n_rows = pl.read_parquet(out_path).height
        print(f"comp_other.parquet: {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
