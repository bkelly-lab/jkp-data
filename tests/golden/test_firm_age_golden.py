"""Golden-fixture regression test for firm_age.

Run with --regen-golden to regenerate the fixture:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_firm_age_golden.py --regen-golden -v

Run without to verify against the committed fixture:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_firm_age_golden.py -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.testing
import pytest

from jkp.data.aux_functions import firm_age
from tests.golden.firm_age_inputs import write_full_inputs

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "firm_age" / "firm_age.parquet"


@pytest.mark.regression
def test_firm_age_golden(test_paths, request: pytest.FixtureRequest) -> None:
    """firm_age output matches the committed golden fixture (exact Int64 ages)."""
    write_full_inputs(test_paths)

    firm_age(test_paths, test_paths.interim_dir / "world_msf.parquet")

    actual = pl.read_parquet(test_paths.interim_dir / "firm_age.parquet").sort(["id", "eom"])

    if request.config.getoption("--regen-golden"):
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        actual.write_parquet(FIXTURE_PATH)
        pytest.skip("regenerated golden fixture")

    assert FIXTURE_PATH.exists(), f"missing golden fixture: {FIXTURE_PATH}"
    expected = pl.read_parquet(FIXTURE_PATH).sort(["id", "eom"])

    polars.testing.assert_frame_equal(actual, expected, check_exact=True)
