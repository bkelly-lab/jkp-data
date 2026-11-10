"""Golden-fixture regression test for the accounting stage (issue #303, step 4).

Run with --regen-golden to regenerate the fingerprint:
    PYTHONPATH=src:. uv run --no-sync python -m pytest tests/golden/test_acc_chars_golden.py --regen-golden

Run without to verify against the committed fingerprint:
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
from tests.golden.generate_acc_chars_golden import GOLDEN_DIR, stage_fingerprint, write_golden

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.regression


def assert_matches_golden(actual: pl.DataFrame, tolerance: ToleranceSpec) -> None:
    """
    Description:
        Compare a fingerprint with the committed one: columns, dtypes and counts exactly; sums,
        minima and maxima to a tolerance, since the earnings-persistence regression runs in DuckDB.
    Steps:
        1) GOLDEN tolerance, plus 1e-10 absolute for column sums that cancel to near zero (the
           acceptance rule).
    Output:
        None; raises AssertionError on a mismatch.
    """
    path = GOLDEN_DIR / "fingerprint.parquet"
    assert path.exists(), f"missing golden fixture: {path}"
    tol = {**tolerance.GOLDEN, "atol": 1e-10}
    polars.testing.assert_frame_equal(actual, pl.read_parquet(path), check_exact=False, **tol)


def test_acc_chars_golden(
    acc_stage: DataPaths, tolerance: ToleranceSpec, request: pytest.FixtureRequest
) -> None:
    """The fingerprint of the stage's output matches the committed one."""
    actual = stage_fingerprint(acc_stage)
    if request.config.getoption("--regen-golden"):
        write_golden(actual)
        pytest.skip("regenerated golden fixture")
    assert_matches_golden(actual, tolerance)


def test_single_thread_run_matches_golden(tmp_path: Path, tolerance: ToleranceSpec) -> None:
    """The inputs and the stage on one Polars thread, in a fresh process, reproduce the
    fingerprint."""
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
    assert_matches_golden(pl.read_parquet(tmp_path / "fingerprint.parquet"), tolerance)
