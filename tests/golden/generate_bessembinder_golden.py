"""
Description:
    Generate golden fixtures for the Bessembinder Section 6 correction
    pipeline. The input panel is the seeded synthetic generator from the
    differential test suite; outputs are produced by the production
    implementation (numba kernels) and pinned as regression fixtures.
Steps:
    1) Build the seeded synthetic panel (_make_synthetic, seed 42).
    2) Run correct_decimal_errors with default settings.
    3) Write input / corrected / log parquets to fixtures/bessembinder/.
Output:
    tests/golden/fixtures/bessembinder/{input,corrected,log}.parquet

Usage:
    PYTHONPATH=src:tests/unit python tests/golden/generate_bessembinder_golden.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "unit"))

from test_bessembinder_differential import GROUP_COLS, LOG_SORT, SORT_COL, _make_synthetic

from jkp.data.bessembinder import correct_decimal_errors

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "bessembinder"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    synthetic = _make_synthetic()
    corrected, log = correct_decimal_errors(synthetic, "prc", GROUP_COLS, SORT_COL)
    synthetic.collect().write_parquet(GOLDEN_DIR / "input.parquet")
    corrected.sort(GROUP_COLS + [SORT_COL]).collect().write_parquet(
        GOLDEN_DIR / "corrected.parquet"
    )
    log.sort(LOG_SORT).collect().write_parquet(GOLDEN_DIR / "log.parquet")
    print(f"Golden fixtures written to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
