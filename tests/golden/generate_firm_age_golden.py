"""Generate golden fixture for firm_age().

Run with:
    uv run python -m tests.golden.generate_firm_age_golden

Writes:
    tests/golden/fixtures/firm_age/firm_age.parquet
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import firm_age
from jkp.data.paths import DataPaths
from tests.golden.firm_age_inputs import write_full_inputs

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "firm_age"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        write_full_inputs(paths)
        firm_age(paths, paths.interim_dir / "world_msf.parquet")
        out_path = GOLDEN_DIR / "firm_age.parquet"
        actual = pl.read_parquet(paths.interim_dir / "firm_age.parquet").sort(["id", "eom"])
        actual.write_parquet(out_path)
        print(f"firm_age.parquet: {actual.height} rows -> {out_path}")


if __name__ == "__main__":
    main()
