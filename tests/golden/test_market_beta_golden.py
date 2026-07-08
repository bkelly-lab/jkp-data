"""Golden-fixture regression test for market_beta.

Run with --regen-golden to regenerate the fixture:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_market_beta_golden.py --regen-golden -v

Run without to verify against the committed fixture:
    PYTHONPATH=src uv run --no-sync python -m pytest tests/golden/test_market_beta_golden.py -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.testing
import pytest

from jkp.data.aux_functions import market_beta
from tests.golden.market_beta_inputs import (
    build_ap_factors_monthly_input,
    build_world_msf_input,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "market_beta" / "beta_60m.parquet"


@pytest.mark.regression
def test_market_beta_golden(
    test_paths,
    tolerance,
    request: pytest.FixtureRequest,
) -> None:
    """market_beta output matches the committed golden fixture."""
    data_path = test_paths.interim_dir / "world_msf.parquet"
    fcts_path = test_paths.interim_dir / "ap_factors_monthly.parquet"
    build_world_msf_input(seed=42).write_parquet(data_path)
    build_ap_factors_monthly_input(seed=42).write_parquet(fcts_path)

    out_path = test_paths.interim_dir / "beta_60m.parquet"
    market_beta(test_paths, out_path, data_path, fcts_path, 60, 36)

    actual = pl.read_parquet(out_path).sort(["id", "eom"])

    if request.config.getoption("--regen-golden"):
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        actual.write_parquet(FIXTURE_PATH)
        pytest.skip("regenerated golden fixture")

    assert FIXTURE_PATH.exists(), f"missing golden fixture: {FIXTURE_PATH}"
    expected = pl.read_parquet(FIXTURE_PATH)

    # Key columns must match exactly (no tolerance).
    key_cols = ["id", "eom"]
    polars.testing.assert_frame_equal(
        actual.select(key_cols),
        expected.select(key_cols),
        check_exact=True,
    )

    # Float columns: tight tolerance (same seed, same platform ops).
    for col in ["beta_60m", "ivol_capm_60m"]:
        polars.testing.assert_series_equal(
            actual[col],
            expected[col],
            **tolerance.GOLDEN,
            check_names=True,
        )
