"""Golden-fixture regression test for quality_minus_junk.

Run with --regen-golden to regenerate the fixture:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_quality_minus_junk_golden.py --regen-golden -v

Run without to verify against the committed fixture:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_quality_minus_junk_golden.py -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.testing
import pytest

from jkp.data.aux_functions import quality_minus_junk
from tests.golden.generate_qmj_golden import build_qmj_world_data_input

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qmj" / "qmj.parquet"


@pytest.mark.regression
def test_quality_minus_junk_golden(
    test_paths,
    tolerance,
    request: pytest.FixtureRequest,
) -> None:
    """quality_minus_junk output matches the committed golden fixture."""
    df_input = build_qmj_world_data_input(seed=42)
    input_path = test_paths.interim_dir / "world_data_-1.parquet"
    df_input.write_parquet(input_path)

    quality_minus_junk(test_paths, input_path, 10)

    actual = pl.read_parquet(test_paths.interim_dir / "qmj.parquet").sort(["excntry", "id", "eom"])

    if request.config.getoption("--regen-golden"):
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        actual.write_parquet(FIXTURE_PATH)
        pytest.skip("regenerated golden fixture")

    assert FIXTURE_PATH.exists(), f"missing golden fixture: {FIXTURE_PATH}"
    expected = pl.read_parquet(FIXTURE_PATH)

    # Key columns must match exactly (no tolerance)
    key_cols = ["excntry", "id", "eom"]
    polars.testing.assert_frame_equal(
        actual.select(key_cols),
        expected.select(key_cols),
        check_exact=True,
    )

    # Float columns: tight tolerance (same seed, same platform ops)
    float_cols = ["qmj_prof", "qmj_growth", "qmj_safety", "qmj"]
    for col in float_cols:
        polars.testing.assert_series_equal(
            actual[col],
            expected[col],
            **tolerance.GOLDEN,
            check_names=True,
        )
