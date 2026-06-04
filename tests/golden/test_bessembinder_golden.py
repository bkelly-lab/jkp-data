"""Golden regression test for the Bessembinder Section 6 correction pipeline."""

import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent.parent / "unit"))

from test_bessembinder_differential import GROUP_COLS, LOG_SORT, SORT_COL, _make_synthetic

from jkp.data.bessembinder import correct_decimal_errors

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "bessembinder"

pytestmark = pytest.mark.skipif(
    not (GOLDEN_DIR / "corrected.parquet").exists(),
    reason="golden fixtures missing; run tests/golden/generate_bessembinder_golden.py",
)


def test_corrected_frame_matches_golden():
    corrected, _ = correct_decimal_errors(_make_synthetic(), "prc", GROUP_COLS, SORT_COL)
    assert_frame_equal(
        corrected.sort(GROUP_COLS + [SORT_COL]).collect(),
        pl.read_parquet(GOLDEN_DIR / "corrected.parquet"),
    )


def test_corrections_log_matches_golden():
    _, log = correct_decimal_errors(_make_synthetic(), "prc", GROUP_COLS, SORT_COL)
    assert_frame_equal(
        log.sort(LOG_SORT).collect(),
        pl.read_parquet(GOLDEN_DIR / "log.parquet"),
    )
