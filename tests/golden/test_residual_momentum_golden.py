"""Golden-fixture regression test for ``residual_momentum``.

Run with --regen-golden to regenerate the fixtures:
    PYTHONPATH=src uv run --no-sync python -m pytest \
        tests/golden/test_residual_momentum_golden.py --regen-golden -v

Run without to verify against the committed fixtures:
    PYTHONPATH=src uv run --no-sync python -m pytest \
        tests/golden/test_residual_momentum_golden.py -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.testing
import pytest

from jkp.data.aux_functions import residual_momentum
from tests.golden.residual_momentum_inputs import (
    build_ap_factors_monthly_input,
    build_world_msf_input,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "residual_momentum"
_OUTPUT_STEM = "resmom_ff3"
_N = 36
_MIN = 24
_VARIANTS = ((12, 1), (6, 1))


@pytest.mark.regression
@pytest.mark.parametrize(("incl", "skip"), _VARIANTS)
def test_residual_momentum_golden(
    test_paths,
    tolerance,
    request: pytest.FixtureRequest,
    incl: int,
    skip: int,
) -> None:
    """residual_momentum output matches the committed golden fixture."""
    world_path = test_paths.interim_dir / "world_msf.parquet"
    fcts_path = test_paths.interim_dir / "ap_factors_monthly.parquet"
    build_world_msf_input(seed=42).write_parquet(world_path)
    build_ap_factors_monthly_input(seed=42).write_parquet(fcts_path)

    residual_momentum(test_paths, _OUTPUT_STEM, world_path, fcts_path, _N, _MIN, incl, skip)

    name = f"{_OUTPUT_STEM}_{incl}_{skip}.parquet"
    value_col = f"resff3_{incl}_{skip}"
    actual = pl.read_parquet(test_paths.interim_dir / name).sort(["id", "eom"])

    fixture_path = FIXTURE_DIR / name
    if request.config.getoption("--regen-golden"):
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        actual.write_parquet(fixture_path)
        pytest.skip(f"regenerated golden fixture: {name}")

    assert fixture_path.exists(), f"missing golden fixture: {fixture_path}"
    expected = pl.read_parquet(fixture_path).sort(["id", "eom"])

    # Key columns must match exactly (no tolerance).
    polars.testing.assert_frame_equal(
        actual.select(["id", "eom"]),
        expected.select(["id", "eom"]),
        check_exact=True,
    )
    # Float column: tight golden tolerance (same seed, same platform ops).
    polars.testing.assert_series_equal(
        actual[value_col],
        expected[value_col],
        **tolerance.GOLDEN,
        check_names=True,
    )
