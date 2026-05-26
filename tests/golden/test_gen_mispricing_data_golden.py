"""Golden-fixture regression test for ``gen_mispricing_data``.

``gen_mispricing_data`` takes no kwargs — it writes ``mp_*.parquet``
into cwd. Test chdirs to ``test_paths.interim_dir`` first.

Regenerate with:
    pytest tests/golden/test_gen_mispricing_data_golden.py --regen-golden -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jkp.data.aux_functions import gen_mispricing_data
from jkp.data.paths import DataPaths
from tests.golden._golden_helpers import (
    cwd,
    regen_or_compare,
    stage_synthetic_slices,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "mispricing"

OUTPUTS = {
    "mp_factors_monthly.parquet": ["excntry", "eom"],
    "mp_factors_daily.parquet": ["excntry", "date"],
    "mp_characteristics.parquet": ["id", "eom"],
}


def test_gen_mispricing_data_golden(test_paths: DataPaths, request: pytest.FixtureRequest) -> None:
    stage_synthetic_slices(test_paths)
    with cwd(test_paths.interim_dir):
        gen_mispricing_data(test_paths)

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
