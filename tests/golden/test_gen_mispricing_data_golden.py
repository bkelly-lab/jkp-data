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
    WRDS_SLICES_DIR,
    cwd,
    regen_or_compare,
    stage_wrds_slices,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "mispricing"

OUTPUTS = {
    "mp_factors_monthly.parquet": ["excntry", "eom"],
    "mp_factors_daily.parquet": ["excntry", "date"],
    "mp_characteristics.parquet": ["id", "eom"],
}

REQUIRED_SLICES = (
    "crsp_msf_v2.parquet",
    "crsp_dsf_v2.parquet",
    "comp_funda.parquet",
    "comp_fundq.parquet",
    "comp_g_fundq.parquet",
    "crsp_ccmxpf_lnkhist.parquet",
    "crsp_a_indexes_msp500.parquet",
    "crsp_a_indexes_acti.parquet",
    "world_data.parquet",
    "world_dsf.parquet",
    "market_returns.parquet",
    "market_returns_daily.parquet",
)


def _missing_slices() -> list[str]:
    return [s for s in REQUIRED_SLICES if not (WRDS_SLICES_DIR / s).exists()]


def test_gen_mispricing_data_golden(test_paths: DataPaths, request: pytest.FixtureRequest) -> None:
    missing = _missing_slices()
    if missing:
        pytest.skip(f"missing WRDS slices: {missing}. Run tests/golden/generate_wrds_slices.py.")

    stage_wrds_slices(test_paths)
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
