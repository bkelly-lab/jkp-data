"""Regenerate the FF (FF3/FF5/UMD) golden parquets.

Generates the synthetic WRDS slices into a temporary directory,
stages them into a temporary DataPaths layout, runs ``gen_ff_data``, and writes the results under
``tests/golden/fixtures/ff/``.

Usage:
    uv run python -m tests.golden.generate_ff_golden \\
        [--out tests/golden/fixtures/ff]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from jkp.data.aux_functions import gen_ff_data
from jkp.data.paths import DataPaths
from tests.golden._golden_helpers import (
    cwd,
    stage_ff_mom_input,
    stage_synthetic_slices,
)
from tests.golden.generate_synthetic_wrds import write_synthetic_wrds

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "ff"
OUTPUTS = (
    "ff_factors_monthly.parquet",
    "ff_factors_daily.parquet",
    "ff_characteristics.parquet",
)


def regenerate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = DataPaths(base_dir=Path(tmp))
        paths.raw_tables_dir.mkdir(parents=True, exist_ok=True)
        paths.interim_dir.mkdir(parents=True, exist_ok=True)
        slices_dir = Path(tmp) / "synthetic_wrds"
        write_synthetic_wrds(slices_dir)
        stage_synthetic_slices(paths, slices_dir)
        stage_ff_mom_input(paths)
        with cwd(paths.interim_dir):
            gen_ff_data(
                paths,
                monthly_factors_path="ff_factors_monthly.parquet",
                daily_factors_path="ff_factors_daily.parquet",
                chars_path="ff_characteristics.parquet",
            )
        for name in OUTPUTS:
            shutil.copy2(paths.interim_dir / name, out_dir / name)
            print(f"wrote {out_dir / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=GOLDEN_DIR, type=Path)
    args = ap.parse_args()
    regenerate(args.out)


if __name__ == "__main__":
    main()
