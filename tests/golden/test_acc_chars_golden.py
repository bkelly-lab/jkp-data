"""Golden-fixture regression test for the accounting stage (issue #303, step 4).

Run with --regen-golden to regenerate the fixtures:
    PYTHONPATH=src:. uv run --no-sync python -m pytest tests/golden/test_acc_chars_golden.py --regen-golden

Run without to verify against the committed fixtures:
    PYTHONPATH=src:. uv run --no-sync python -m pytest tests/golden/test_acc_chars_golden.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import polars.testing
import pytest

from jkp.data.paths import DataPaths
from tests.conftest import ToleranceSpec
from tests.golden.generate_acc_chars_golden import (
    GOLDEN_DIR,
    GOLDEN_NAMES,
    compute_outputs,
    write_goldens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pl.DataFrame]:
    return compute_outputs(DataPaths(tmp_path_factory.mktemp("acc_stage")))


def assert_matches_golden(actual: pl.DataFrame, name: str, tolerance: ToleranceSpec) -> None:
    """
    Description:
        Compare one golden frame with its fixture: keys, dtypes and nulls exactly, values to a
        tolerance, since the earnings-persistence regression runs in DuckDB.
    Steps:
        1) personas and random hold Float32 values: STANDARD tolerance, above Float32 rounding.
        2) The fingerprint is full precision: GOLDEN tolerance, plus 1e-10 absolute for column
           sums that cancel to near zero (the acceptance rule).
    Output:
        None; raises AssertionError on a mismatch.
    """
    path = GOLDEN_DIR / f"{name}.parquet"
    assert path.exists(), f"missing golden fixture: {path}"
    tol = {**tolerance.GOLDEN, "atol": 1e-10} if name == "fingerprint" else tolerance.STANDARD
    polars.testing.assert_frame_equal(actual, pl.read_parquet(path), check_exact=False, **tol)


@pytest.mark.regression
@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_acc_chars_golden(
    name: str,
    outputs: dict[str, pl.DataFrame],
    tolerance: ToleranceSpec,
    request: pytest.FixtureRequest,
) -> None:
    """Each golden frame of the stage's output matches its committed fixture."""
    if request.config.getoption("--regen-golden"):
        write_goldens({name: outputs[name]})
        pytest.skip("regenerated golden fixture")
    assert_matches_golden(outputs[name], name, tolerance)


@pytest.mark.regression
def test_single_thread_run_matches_golden(tmp_path: Path, tolerance: ToleranceSpec) -> None:
    """The stage on one Polars thread, in a fresh process, reproduces every golden frame."""
    env = {
        **os.environ,
        "POLARS_MAX_THREADS": "1",
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)]),
    }
    run = subprocess.run(
        [sys.executable, "-m", "tests.golden.generate_acc_chars_golden", str(tmp_path)],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    for name in GOLDEN_NAMES:
        assert_matches_golden(pl.read_parquet(tmp_path / f"{name}.parquet"), name, tolerance)
