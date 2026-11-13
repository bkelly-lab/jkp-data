"""Determinism of the portfolio build.

Each build runs ``run_portfolio`` and then ``run_themes`` for v0 in a fresh Python
process on the same synthetic inputs, which include the overnight/intraday columns. The
baseline uses four Polars and four Numba threads. ``repeat`` builds with the same
settings again: if it fails, the outputs vary from run to run and a failing variant
says nothing about its setting. Each other variant changes one thing that must not
change the outputs:

- ``shuffled_inputs``: the rows of every parquet input in another order. The themes
  and country classification files are written by the build process and keep theirs,
- ``polars_one_thread``: one Polars thread,
- ``numba_one_thread``: one Numba thread,
- ``hash_seed``: another Python hash seed, which changes the iteration order of sets
  of strings.

Every output file must match the baseline: the same files, schemas and row order, keys
and integer columns exactly, and floats within ``TIGHT``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests.conftest import ToleranceSpec
from tests.fixtures.portfolio_build import build_oi_inputs, output_files

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[2]
# A build takes seconds; the limit only stops a hung one.
BUILD_TIMEOUT = 300
BASELINE_ENV = {"POLARS_MAX_THREADS": "4", "NUMBA_NUM_THREADS": "4", "PYTHONHASHSEED": "0"}
VARIANTS = {
    "repeat": ({}, False),
    "shuffled_inputs": ({}, True),
    "polars_one_thread": ({"POLARS_MAX_THREADS": "1"}, False),
    "numba_one_thread": ({"NUMBA_NUM_THREADS": "1"}, False),
    "hash_seed": ({"PYTHONHASHSEED": "1"}, False),
}


def _run_build(inputs: Path, root: Path, env: dict[str, str], shuffle: bool) -> Path:
    """Copy the inputs to ``root``, optionally shuffle their rows, and build in a new process."""
    shutil.copytree(inputs, root / "processed")
    if shuffle:
        for i, path in enumerate(sorted((root / "processed").rglob("*.parquet"))):
            pl.read_parquet(path).sample(fraction=1.0, shuffle=True, seed=i).write_parquet(path)
    code = (
        "import sys; from pathlib import Path; "
        "from tests.fixtures.portfolio_build import run_oi_build; "
        "run_oi_build(Path(sys.argv[1]))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(root)],
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        # The captured stderr is bytes here, whatever ``text`` says.
        stderr = (e.stderr or b"").decode(errors="replace")
        raise AssertionError(f"build hung past {BUILD_TIMEOUT} s:\n{stderr[-4000:]}") from e
    assert result.returncode == 0, result.stderr[-4000:]
    return root / "processed" / "portfolios"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("inputs")
    build_oi_inputs(root)
    return root / "processed"


@pytest.fixture(scope="module")
def baseline(inputs: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The baseline build, which must write the files the variants are compared on."""
    out = _run_build(inputs, tmp_path_factory.mktemp("baseline"), BASELINE_ENV, shuffle=False)
    for rel in ("pfs.parquet", "pfs_overnight.parquet", "themes/v0/lms.parquet"):
        path = out / rel
        assert path.exists() and pl.read_parquet(path).height > 0, rel
    return out


@pytest.mark.parametrize("variant", VARIANTS)
def test_outputs_match_baseline(
    inputs: Path, baseline: Path, tmp_path: Path, variant: str, tolerance: ToleranceSpec
) -> None:
    env, shuffle = VARIANTS[variant]
    out = _run_build(inputs, tmp_path, {**BASELINE_ENV, **env}, shuffle)
    assert output_files(out) == output_files(baseline)
    for rel in sorted(output_files(baseline)):
        try:
            assert_frame_equal(
                pl.read_parquet(out / rel),
                pl.read_parquet(baseline / rel),
                check_exact=False,
                **tolerance.polars(tolerance.TIGHT),
            )
        except AssertionError as e:
            raise AssertionError(f"{variant}: {rel}: {e}") from e
