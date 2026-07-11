"""Golden regression test for prepare_crsp_sf().

Reconstructs the synthetic inputs from ``prepare_crsp_sf_inputs``, runs the real
``prepare_crsp_sf`` for both freqs on a tmp DataPaths, and asserts every emitted
parquet equals its committed fixture. Pass ``--regen-golden`` to overwrite.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jkp.data.aux_functions import prepare_crsp_sf
from jkp.data.paths import DataPaths
from tests.conftest import ToleranceSpec, assert_series_equal
from tests.golden.prepare_crsp_sf_inputs import write_all_inputs, write_empty_inputs

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "prepare_crsp_sf"


def _assert_frames_equal(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    """NaN/null-aware full-frame comparison (columns, dtypes, values)."""
    actual = actual.sort(["permno", "date"])
    expected = expected.sort(["permno", "date"])
    assert actual.columns == expected.columns, (
        f"Column mismatch:\n actual={actual.columns}\n expected={expected.columns}"
    )
    assert actual.schema == expected.schema, (
        f"Dtype mismatch:\n actual={actual.schema}\n expected={expected.schema}"
    )
    assert actual.height == expected.height, (
        f"Row count differs: {actual.height} vs {expected.height}"
    )
    for name in expected.columns:
        a, e = actual[name], expected[name]
        if a.dtype.is_numeric():
            assert_series_equal(a.cast(pl.Float64), e.cast(pl.Float64), **ToleranceSpec.GOLDEN)
        else:
            assert a.to_list() == e.to_list(), f"Column {name} differs"


@pytest.mark.regression
@pytest.mark.parametrize("freq", ["m", "d"])
def test_prepare_crsp_sf_golden(freq: str, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    paths = DataPaths(base_dir=tmp_path)
    write_all_inputs(paths, freq)
    prepare_crsp_sf(paths, freq)
    actual = pl.read_parquet(paths.interim_dir / f"crsp_{freq}sf.parquet")

    fixture_path = GOLDEN_DIR / f"crsp_{freq}sf.parquet"
    if request.config.getoption("--regen-golden"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        actual.write_parquet(fixture_path)
        pytest.skip(f"Regenerated {fixture_path}")

    _assert_frames_equal(actual, pl.read_parquet(fixture_path))


@pytest.mark.regression
@pytest.mark.parametrize("freq", ["m", "d"])
def test_prepare_crsp_sf_empty_input(freq: str, tmp_path: Path) -> None:
    """0-row typed inputs must not crash and must yield a 0-row, golden-schema output."""
    paths = DataPaths(base_dir=tmp_path)
    write_empty_inputs(paths, freq)
    prepare_crsp_sf(paths, freq)
    actual = pl.read_parquet(paths.interim_dir / f"crsp_{freq}sf.parquet")

    golden = pl.read_parquet(GOLDEN_DIR / f"crsp_{freq}sf.parquet")
    assert actual.height == 0, f"Expected 0 rows from empty inputs, got {actual.height}"
    assert actual.columns == golden.columns, (
        f"Column mismatch on empty input:\n actual={actual.columns}\n expected={golden.columns}"
    )
