"""Golden-fixture regression test for ``gen_hxz_data``.

Mirrors ``test_gen_ff_data_golden`` for the HXZ4 builder. Compares
``hxz_factors_monthly.parquet``, ``hxz_factors_daily.parquet``, and
``hxz_characteristics.parquet`` against the committed golden parquets.

Regenerate with:
    pytest tests/golden/test_gen_hxz_data_golden.py --regen-golden -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jkp.data.aux_functions import gen_hxz_data
from jkp.data.paths import DataPaths
from tests.golden._golden_helpers import (
    cwd,
    regen_or_compare,
    stage_synthetic_slices,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "hxz"

OUTPUTS = {
    "hxz_factors_monthly.parquet": ["excntry", "eom"],
    "hxz_factors_daily.parquet": ["excntry", "date"],
    "hxz_characteristics.parquet": ["id", "eom"],
}


@pytest.mark.regression
def test_gen_hxz_data_golden(test_paths: DataPaths, request: pytest.FixtureRequest) -> None:
    stage_synthetic_slices(test_paths)
    with cwd(test_paths.interim_dir):
        gen_hxz_data(
            test_paths,
            monthly_factors_path="hxz_factors_monthly.parquet",
            daily_factors_path="hxz_factors_daily.parquet",
            chars_path="hxz_characteristics.parquet",
        )

    regen = request.config.getoption("--regen-golden")
    failures: list[str] = []
    for name, keys in OUTPUTS.items():
        failures.extend(
            regen_or_compare(
                actual=test_paths.interim_dir / name,
                golden=GOLDEN_DIR / name,
                key_cols=keys,
                regen=regen,
            )
        )
    if regen:
        pytest.skip("regenerated golden parquets")
    assert not failures, "\n".join(failures)
