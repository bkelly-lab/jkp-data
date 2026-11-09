"""Regenerate the HXZ4 golden parquets.

Mirrors ``generate_ff_golden.py`` for ``gen_hxz_data``.

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
from tests.golden._golden_helpers import SYNTHETIC_SLICES_DIR, cwd, stage_synthetic_slices

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "hxz"
OUTPUTS = (
    "hxz_factors_monthly.parquet",
    "hxz_factors_daily.parquet",
    "hxz_characteristics.parquet",
)


def regenerate(out_dir: Path, slices_dir: Path = SYNTHETIC_SLICES_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = DataPaths(base_dir=Path(tmp))
        paths.raw_tables_dir.mkdir(parents=True, exist_ok=True)
        paths.interim_dir.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--slices", default=SYNTHETIC_SLICES_DIR, type=Path)
    args = ap.parse_args()
    regenerate(args.out, args.slices)


if __name__ == "__main__":
    main()
