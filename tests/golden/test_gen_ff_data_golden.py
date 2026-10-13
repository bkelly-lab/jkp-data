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

import polars as pl
import pytest

from jkp.data.aux_functions import gen_ff_data, mom_rev_cols
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


@pytest.mark.regression
def test_gen_ff_data_golden(test_paths: DataPaths, request: pytest.FixtureRequest) -> None:
    stage_synthetic_slices(test_paths)
    interim = test_paths.interim_dir
    # UMD reads ret_12_1 from market_chars_m. The synthetic world_msf lacks most
    # inputs of market_chars_monthly, so build just ret_12_1 with its helper.
    (
        pl.read_parquet(interim / "world_msf.parquet")
        .sort(["id", "eom"])
        .with_columns(
            ret_x=pl.col("ret"),
            ri_x=(1 + pl.col("ret").fill_null(0.0)).cum_prod().over("id"),
            count=pl.col("id").cum_count().over("id"),
        )
        .with_columns(mom_rev_cols(1, 12).over("id"))
        .select("id", "eom", "ret_12_1")
        .write_parquet(interim / "market_chars_m.parquet")
    )
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
