"""Regenerate the HXZ4 golden parquets.

Mirrors ``generate_ff_golden.py`` for ``gen_hxz_data``: generates the synthetic
WRDS slices into a temporary directory, runs the builder on them, and writes the
results under ``tests/golden/fixtures/hxz/``.

Usage:
    uv run python -m tests.golden.generate_hxz_golden \\
        [--out tests/golden/fixtures/hxz]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from jkp.data.aux_functions import gen_hxz_data
from jkp.data.paths import DataPaths
from tests.golden._golden_helpers import cwd, stage_synthetic_slices
from tests.golden.generate_synthetic_wrds import write_synthetic_wrds

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "hxz"
OUTPUTS = (
    "hxz_factors_monthly.parquet",
    "hxz_factors_daily.parquet",
    "hxz_characteristics.parquet",
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
        with cwd(paths.interim_dir):
            gen_hxz_data(
                paths,
                monthly_factors_path="hxz_factors_monthly.parquet",
                daily_factors_path="hxz_factors_daily.parquet",
                chars_path="hxz_characteristics.parquet",
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
