"""Golden-fixture regression test for ``gen_mispricing_data``.

``gen_mispricing_data`` writes ``mp_*.parquet`` into ``test_paths.interim_dir``.

The golden parquets are pinned to x86-64 (regenerate on the cluster, where
CI also runs): the world COMPOSITE_ISSUE/MOMENTUM/DISTRESS chains go
through libm (log/exp), whose last-ULP results differ across
architectures; on the synthetic slice that shifts a percentile break and
flips one (excntry, eom) bucket past the min-obs gate, so an exact
row-set comparison cannot hold on both arches at once.

Regenerate with (on x86-64):
    pytest tests/golden/test_gen_mispricing_data_golden.py --regen-golden -v
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from jkp.data.aux_functions import gen_mispricing_data
from jkp.data.paths import DataPaths
from tests.golden._golden_helpers import (
    regen_or_compare,
    stage_synthetic_slices,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "mispricing"

OUTPUTS = {
    "mp_factors_monthly.parquet": ["excntry", "eom"],
    "mp_factors_daily.parquet": ["excntry", "date"],
    "mp_characteristics.parquet": ["id", "eom"],
}


@pytest.mark.regression
@pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64"),
    reason="golden pinned to x86-64 (CI/cluster); libm ULP flips a rank-boundary bucket here",
)
def test_gen_mispricing_data_golden(test_paths: DataPaths, request: pytest.FixtureRequest) -> None:
    stage_synthetic_slices(test_paths)
    interim = test_paths.interim_dir
    gen_mispricing_data(
        test_paths,
        interim / "mp_factors_monthly.parquet",
        interim / "mp_factors_daily.parquet",
        interim / "mp_characteristics.parquet",
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
