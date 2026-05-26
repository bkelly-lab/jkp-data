"""Golden-fixture regression test for ``gen_ff_data``.

Runs the real builder against the committed synthetic slices and
compares ``ff_factors_monthly.parquet``, ``ff_factors_daily.parquet``,
and ``ff_characteristics.parquet`` against the committed golden parquets
within ``rtol=1e-6, atol=1e-10`` (key columns matched exactly).

The synthetic inputs are committed under
``tests/golden/fixtures/synthetic_wrds/`` and contain no real WRDS data
(see ``generate_synthetic_wrds.py``).

Regenerate goldens with:
    pytest tests/golden/test_gen_ff_data_golden.py --regen-golden -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jkp.data.aux_functions import gen_ff_data
from jkp.data.paths import DataPaths
from tests.golden._golden_helpers import (
    cwd,
    regen_or_compare,
    stage_synthetic_slices,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "ff"

OUTPUTS = {
    "ff_factors_monthly.parquet": ["excntry", "eom"],
    "ff_factors_daily.parquet": ["excntry", "date"],
    "ff_characteristics.parquet": ["id", "eom"],
}


def test_gen_ff_data_golden(test_paths: DataPaths, request: pytest.FixtureRequest) -> None:
    stage_synthetic_slices(test_paths)
    with cwd(test_paths.interim_dir):
        gen_ff_data(
            test_paths,
            monthly_factors_path="ff_factors_monthly.parquet",
            daily_factors_path="ff_factors_daily.parquet",
            chars_path="ff_characteristics.parquet",
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
