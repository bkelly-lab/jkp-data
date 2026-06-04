"""
Description:
    Benchmark the numba kernel implementation of Bessembinder Section 6
    against the original Polars reference, sweeping panel sizes. Mirrors
    scripts/benchmark_dimsonbeta.py: median wall-clock + equality check.
Steps:
    1) Generate a seeded synthetic panel per group count.
    2) Time correct_decimal_errors (kernel) and, for the smaller sizes,
       _polars_correct_decimal_errors (reference).
    3) Assert outputs identical where both ran; report timings.
Output:
    Console table: rows, kernel median seconds, reference median seconds,
    speedup, outputs-identical flag.

Usage:
    PYTHONPATH=src:tests/unit python scripts/benchmark_bessembinder.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "unit"))

from polars.testing import assert_frame_equal
from test_bessembinder_differential import GROUP_COLS, LOG_SORT, SORT_COL, _make_synthetic

from jkp.data._bessembinder_polars_reference import _polars_correct_decimal_errors
from jkp.data.bessembinder import correct_decimal_errors

REFERENCE_MAX_GROUPS = 1_000  # reference is too slow beyond this
REPS = 3


def _time(fn) -> tuple[float, object]:
    times = []
    result = None
    for _ in range(REPS):
        t0 = time.time()
        result = fn()
        times.append(time.time() - t0)
    return statistics.median(times), result


def main() -> None:
    print(f"{'groups':>8} {'rows':>10} {'kernel_s':>9} {'ref_s':>9} {'speedup':>8} identical")
    for n_groups in [250, 1_000, 4_000, 16_000]:
        df = _make_synthetic(seed=42, n_groups=n_groups)
        n_rows = df.collect().height
        kernel_t, kernel_out = _time(
            lambda d=df: tuple(
                x.collect() for x in correct_decimal_errors(d, "prc", GROUP_COLS, SORT_COL)
            )
        )
        if n_groups <= REFERENCE_MAX_GROUPS:
            ref_t, ref_out = _time(
                lambda d=df: tuple(
                    x.collect()
                    for x in _polars_correct_decimal_errors(d, "prc", GROUP_COLS, SORT_COL)
                )
            )
            sort_cols = GROUP_COLS + [SORT_COL]
            assert_frame_equal(
                kernel_out[0].sort(sort_cols),
                ref_out[0].sort(sort_cols),
                check_column_order=False,
            )
            assert_frame_equal(kernel_out[1].sort(LOG_SORT), ref_out[1].sort(LOG_SORT))
            ref_s, speedup, identical = f"{ref_t:9.2f}", f"{ref_t / kernel_t:7.1f}x", "yes"
        else:
            ref_s, speedup, identical = "      --", "      --", "--"
        print(f"{n_groups:>8} {n_rows:>10} {kernel_t:9.3f} {ref_s} {speedup} {identical}")


if __name__ == "__main__":
    main()
