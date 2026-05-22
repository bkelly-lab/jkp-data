"""Golden-fixture regression test for ``gen_ff_data``.

Runs the real builder against the committed WRDS slices and compares
``ff_factors_monthly.parquet``, ``ff_factors_daily.parquet``, and
``ff_characteristics.parquet`` against the committed golden parquets.

Regenerate with:
    pytest tests/golden/test_gen_ff_data_golden.py --regen-golden -v

Skips when the WRDS slices or golden parquets are missing (the slices
are too large for plain dev environments; run the slice + golden
generator on the cluster first).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jkp.data.aux_functions import gen_ff_data
from jkp.data.paths import DataPaths
from tests.golden._golden_helpers import (
    WRDS_SLICES_DIR,
    cwd,
    regen_or_compare,
    stage_wrds_slices,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "ff"

OUTPUTS = {
    "ff_factors_monthly.parquet": ["excntry", "eom"],
    "ff_factors_daily.parquet": ["excntry", "date"],
    "ff_characteristics.parquet": ["id", "eom"],
}

REQUIRED_SLICES = (
    "crsp_msf_v2.parquet",
    "crsp_dsf_v2.parquet",
    "comp_funda.parquet",
    "comp_g_funda.parquet",
    "crsp_ccmxpf_lnkhist.parquet",
    "world_msf.parquet",
    "world_dsf.parquet",
)


def _missing_slices() -> list[str]:
    return [s for s in REQUIRED_SLICES if not (WRDS_SLICES_DIR / s).exists()]


def test_gen_ff_data_golden(test_paths: DataPaths, request: pytest.FixtureRequest) -> None:
    missing = _missing_slices()
    if missing:
        pytest.skip(f"missing WRDS slices: {missing}. Run tests/golden/generate_wrds_slices.py.")

    stage_wrds_slices(test_paths)
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
