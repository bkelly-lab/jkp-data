"""Determinism of the portfolio build.

Each build runs ``run_portfolio`` and then ``run_themes`` for v0 in a fresh Python
process on the same synthetic inputs, which include the overnight/intraday columns. The
baseline uses four Polars and four Numba threads; each variant changes one thing that
must not change the outputs:

- ``shuffled_inputs``: the rows of every input file in another order,
- ``one_thread``: one Polars and one Numba thread,
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

from jkp.data.portfolio import run_portfolio, run_themes
from tests.integration.test_oi_single_pass_equivalence import SETTINGS, _build_inputs, _patch

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[2]
BASELINE_ENV = {"POLARS_MAX_THREADS": "4", "NUMBA_NUM_THREADS": "4", "PYTHONHASHSEED": "0"}
VARIANTS = {
    "shuffled_inputs": ({}, True),
    "one_thread": ({"POLARS_MAX_THREADS": "1", "NUMBA_NUM_THREADS": "1"}, False),
    "hash_seed": ({"PYTHONHASHSEED": "1"}, False),
}


def _build(root: Path) -> None:
    """Run the portfolio build on ``root``: the entry point of each build process."""
    resources = root / "resources"
    resources.mkdir()
    mp = pytest.MonkeyPatch()
    _patch(mp, resources, SETTINGS)
    run_portfolio(output_format="parquet", output_dir=root)
    run_themes(output_dir=root, version="v0")


def _run_build(inputs: Path, root: Path, env: dict[str, str], shuffle: bool) -> Path:
    """Copy the inputs to ``root``, optionally shuffle their rows, and build in a new process."""
    shutil.copytree(inputs, root / "processed")
    if shuffle:
        for i, path in enumerate(sorted((root / "processed").rglob("*.parquet"))):
            pl.read_parquet(path).sample(fraction=1.0, shuffle=True, seed=i).write_parquet(path)
    code = (
        "import sys; from pathlib import Path; "
        "from tests.integration.test_portfolio_determinism import _build; "
        "_build(Path(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return root / "processed" / "portfolios"


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.parquet"))


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("inputs")
    _build_inputs(root)
    return root / "processed"


@pytest.fixture(scope="module")
def baseline(inputs: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _run_build(inputs, tmp_path_factory.mktemp("baseline"), BASELINE_ENV, shuffle=False)


@pytest.mark.parametrize("variant", VARIANTS)
def test_outputs_match_baseline(
    inputs: Path, baseline: Path, tmp_path: Path, variant: str, tolerance
) -> None:
    env, shuffle = VARIANTS[variant]
    out = _run_build(inputs, tmp_path, {**BASELINE_ENV, **env}, shuffle)
    assert _files(out) == _files(baseline)
    for rel in _files(baseline):
        try:
            assert_frame_equal(
                pl.read_parquet(out / rel),
                pl.read_parquet(baseline / rel),
                check_exact=False,
                **tolerance.TIGHT,
            )
        except AssertionError as e:
            raise AssertionError(f"{variant}: {rel}: {e}") from e
