"""Golden-fixture regression test for the final merge and save stage.

Run with --regen-golden to regenerate the fixtures:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_final_stage_golden.py --regen-golden -v

Run without to verify against the committed fixtures:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_final_stage_golden.py -v
"""

from __future__ import annotations

import polars as pl
import polars.testing
import pytest

from tests.golden.generate_final_stage_golden import GOLDEN_DIR, SORT_KEYS, compute_outputs


@pytest.fixture
def outputs(test_paths) -> dict[str, pl.DataFrame]:
    return compute_outputs(test_paths)


@pytest.mark.regression
@pytest.mark.parametrize("name", list(SORT_KEYS))
def test_final_stage_golden(name: str, outputs, request: pytest.FixtureRequest) -> None:
    """Each merge/save output matches its committed fixture exactly, dtypes included."""
    actual = outputs[name]
    path = GOLDEN_DIR / f"{name}.parquet"

    if request.config.getoption("--regen-golden"):
        path.parent.mkdir(parents=True, exist_ok=True)
        actual.write_parquet(path)
        pytest.skip("regenerated golden fixture")

    assert path.exists(), f"missing golden fixture: {path}"
    expected = pl.read_parquet(path).sort(SORT_KEYS[name], maintain_order=True)
    polars.testing.assert_frame_equal(actual, expected, check_exact=True)
