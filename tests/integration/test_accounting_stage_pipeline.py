"""Integration test for the accounting stage (issue #303, step 4).

Runs the stage's calls in main.py's order on the synthetic Compustat inputs
(tests/golden/fixtures/synthetic_compustat/) and checks the shape of what it writes. The values
are locked by tests/golden/test_acc_chars_golden.py.
"""

from __future__ import annotations

import polars as pl
import pytest

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.conftest import assert_unique_keys
from tests.golden.generate_acc_chars_golden import KEYS, run_stage

pytestmark = pytest.mark.integration

# file -> its key columns
STAGE_FILES = {
    "acc_std_ann": ["gvkey", "datadate"],
    "acc_std_qtr": ["gvkey", "datadate"],
    "ni_ar_res": ["gvkey", "curcd", "datadate"],
    "achars_world": KEYS,
    "qchars_world": KEYS,
    "acc_chars_world": KEYS,
}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pl.DataFrame]:
    paths = DataPaths(tmp_path_factory.mktemp("acc_stage"))
    run_stage(paths)
    return {name: pl.read_parquet(paths.interim_dir / f"{name}.parquet") for name in STAGE_FILES}


def _public_months(std: pl.DataFrame) -> int:
    """
    Description:
        Public dates the expansion should produce from a standardized table, counted directly.
    Steps:
        1) Keep reports with data_available = 1.
        2) A report is public from the month-end 4 months after datadate until the month before
           the firm's next report becomes public, for at most 18 months after datadate.
    Output:
        Total number of (report, public month) pairs.
    """
    month = pl.col("datadate").dt.offset_by
    spans = (
        std.filter(pl.col("data_available") == 1)
        .sort("gvkey", "datadate")
        .with_columns(start=month("4mo").dt.month_end())
        .with_columns(
            end=pl.min_horizontal(
                pl.col("start").shift(-1).over("gvkey").dt.offset_by("-1mo").dt.month_end(),
                month("18mo").dt.month_end(),
            )
        )
    )
    months = (
        (pl.col("end").dt.year() - pl.col("start").dt.year()) * 12
        + pl.col("end").dt.month()
        - pl.col("start").dt.month()
        + 1
    )
    return spans.select(months.clip(0).sum()).item()


@pytest.mark.parametrize(("name", "keys"), STAGE_FILES.items())
def test_stage_file_is_written_with_unique_keys(
    name: str, keys: list[str], outputs: dict[str, pl.DataFrame]
) -> None:
    assert outputs[name].height > 0
    assert_unique_keys(outputs[name], keys)


def test_output_has_keys_and_every_characteristic(outputs: dict[str, pl.DataFrame]) -> None:
    assert outputs["acc_chars_world"].columns == ["source", *KEYS, *af.acc_chars_list()]


@pytest.mark.parametrize(
    ("std", "chars"), [("acc_std_ann", "achars_world"), ("acc_std_qtr", "qchars_world")]
)
def test_public_date_expansion_row_counts(
    std: str, chars: str, outputs: dict[str, pl.DataFrame]
) -> None:
    assert outputs[chars].height == _public_months(outputs[std])


def test_combined_output_is_union_of_annual_and_quarterly(outputs: dict[str, pl.DataFrame]) -> None:
    union = pl.concat([outputs["achars_world"].select(KEYS), outputs["qchars_world"].select(KEYS)])
    assert outputs["acc_chars_world"].height == union.unique().height


def test_no_characteristic_is_all_null_or_nan(outputs: dict[str, pl.DataFrame]) -> None:
    chars = outputs["acc_chars_world"]
    counts = chars.select(
        pl.col(c).cast(pl.Float64).fill_nan(None).count() for c in af.acc_chars_list()
    ).row(0, named=True)
    assert [c for c, n in counts.items() if n == 0] == []
