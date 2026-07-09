"""Generate golden fixtures for ``residual_momentum``.

Run with:
    PYTHONPATH=src uv run --no-sync python -m tests.golden.generate_residual_momentum_golden
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from jkp.data.aux_functions import residual_momentum
from jkp.data.paths import DataPaths
from tests.golden.residual_momentum_inputs import (
    build_ap_factors_monthly_input,
    build_world_msf_input,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "residual_momentum"

# residual_momentum(paths, output_path, data_path, fcts_path, __n, __min, incl, skip)
_N = 36
_MIN = 24
_VARIANTS = ((12, 1), (6, 1))
_OUTPUT_STEM = "resmom_ff3"


def _run(paths: DataPaths) -> None:
    """Write inputs then run both residual_momentum variants into ``interim_dir``."""
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    world_path = paths.interim_dir / "world_msf.parquet"
    fcts_path = paths.interim_dir / "ap_factors_monthly.parquet"
    build_world_msf_input(seed=42, bulk=True).write_parquet(world_path)
    build_ap_factors_monthly_input(seed=42, bulk=True).write_parquet(fcts_path)
    for incl, skip in _VARIANTS:
        residual_momentum(paths, _OUTPUT_STEM, world_path, fcts_path, _N, _MIN, incl, skip)


def main() -> None:
    """Generate and copy both resmom golden parquets to the fixtures directory."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = DataPaths(base_dir=Path(tmpdir))
        _run(paths)
        for incl, skip in _VARIANTS:
            name = f"{_OUTPUT_STEM}_{incl}_{skip}.parquet"
            src = paths.interim_dir / name
            dst = GOLDEN_DIR / name
            shutil.copyfile(src, dst)
            print(f"{name} -> {dst}")


if __name__ == "__main__":
    main()
