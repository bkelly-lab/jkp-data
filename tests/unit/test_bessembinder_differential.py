"""
Differential tests: numba kernel implementation vs original Polars reference.

The production Section 6 implementation (numba kernels behind the public
functions in jkp.data.bessembinder) must be byte-identical to the original
handoff implementation, preserved verbatim in
jkp.data._bessembinder_polars_reference. These tests run both on the same
synthetic inputs and assert identical corrected frames and correction logs.

The reference multi-period detector still contains the original magnitude
[1, 2, 3] loop, so passing these tests also re-proves that the kernel's
single-magnitude multi-period pass is equivalent (the dead-code claim of
commit 507df7c).
"""

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from polars.testing import assert_frame_equal

from jkp.data._bessembinder_polars_reference import _polars_correct_decimal_errors
from jkp.data.bessembinder import correct_decimal_errors

GROUP_COLS = ["gvkey", "iid"]
SORT_COL = "datadate"
LOG_SORT = GROUP_COLS + [SORT_COL, "variable"]


def _make_synthetic(seed: int = 42, n_groups: int = 250) -> pl.LazyFrame:
    """
    Description:
        Seeded synthetic security panel with injected decimal errors covering
        every detection path of the Section 6 algorithm.
    Steps:
        1) Per group: lognormal random-walk base series (length 30-220).
        2) Inject single-day high/low 10x/100x/1000x spikes; multi-day stable
           error spans of widths 2, 3, 5, 10, 21 (x10 and /10); cascade chains
           (adjacent opposite errors); group-edge errors (positions 0/1 and
           last) to exercise out-of-range endpoint/interior semantics; zeros
           and nulls.
    Output:
        LazyFrame with gvkey, iid, datadate, prc.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        n = int(rng.integers(30, 220))
        base = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        x = base.copy()
        kind = g % 10
        if kind == 0 and n > 10:  # single-day errors, all magnitudes
            for mag, pos in [(10, 3), (100, 6), (1000, 9)]:
                if pos < n - 1:
                    x[pos] *= mag if rng.random() < 0.5 else 1.0 / mag
        elif kind == 1:  # multi-day stable spans
            w = [2, 3, 5, 10, 21][g % 5]
            s = min(5, n - w - 2)
            if s >= 1:
                x[s : s + w] *= 10.0 if rng.random() < 0.5 else 0.1
        elif kind == 2 and n > 8:  # cascade chain: alternating opposite errors
            for p in range(2, min(7, n - 2)):
                x[p] *= 10.0 if p % 2 == 0 else 0.1
        elif kind == 3:  # error at the very edge of the group
            x[0] *= 10.0
            if n > 3:
                x[1] *= 10.0
            x[n - 1] *= 0.1
        elif kind == 4 and n > 12:  # span touching group start (interior off-range)
            x[0:4] *= 10.0
        elif kind == 5 and n > 10:  # zeros and nulls adjacent to an error
            x[4] = 0.0
            x[5] *= 100.0
            x[7] = np.nan  # becomes null via Polars round-trip
        elif kind == 6 and n > 25:  # sandwich pattern: clean value between errors
            x[10] *= 10.0
            x[12] *= 10.0
        # kinds 7-9: clean series (no injection)
        frames.append(
            pl.DataFrame(
                {
                    "gvkey": [f"{g:06d}"] * n,
                    "iid": ["01"] * n,
                    "datadate": np.arange(n, dtype=np.int64),
                    "prc": x,
                }
            )
        )
    out = pl.concat(frames)
    # Encode injected NaN as proper nulls (parquet/null semantics of real data)
    return out.with_columns(
        pl.when(pl.col("prc").is_nan()).then(None).otherwise(pl.col("prc")).alias("prc")
    ).lazy()


def _assert_equivalent(df: pl.LazyFrame, **kwargs) -> None:
    """Run kernel and reference implementations; assert identical outputs."""
    new_df, new_log = correct_decimal_errors(df, "prc", GROUP_COLS, SORT_COL, **kwargs)
    ref_df, ref_log = _polars_correct_decimal_errors(df, "prc", GROUP_COLS, SORT_COL, **kwargs)
    assert_frame_equal(
        new_df.sort(GROUP_COLS + [SORT_COL]).collect(),
        ref_df.sort(GROUP_COLS + [SORT_COL]).collect(),
        check_column_order=False,
    )
    assert (new_log is None) == (ref_log is None)
    if new_log is not None:
        assert_frame_equal(
            new_log.sort(LOG_SORT).collect(),
            ref_log.sort(LOG_SORT).collect(),
            check_column_order=True,
        )


class TestDifferentialSynthetic:
    """Kernel vs reference on the seeded synthetic panel."""

    @pytest.fixture(scope="class")
    def synthetic(self) -> pl.LazyFrame:
        return _make_synthetic()

    def test_default_bessembinder_method(self, synthetic):
        _assert_equivalent(synthetic)

    def test_interpolation_method(self, synthetic):
        _assert_equivalent(synthetic, correction_method="interpolation")

    def test_no_cascading_validation(self, synthetic):
        _assert_equivalent(synthetic, validate_cascading=False)

    def test_interpolation_no_cascading(self, synthetic):
        _assert_equivalent(synthetic, correction_method="interpolation", validate_cascading=False)

    def test_exhaustive_small_windows(self, synthetic):
        _assert_equivalent(synthetic, window_sizes=list(range(1, 8)))


@st.composite
def _series_strategy(draw):
    """Short random per-group series mixing clean values and 10^k spikes."""
    n = draw(st.integers(min_value=3, max_value=40))
    base = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    values = []
    for v in base:
        kind = draw(st.integers(min_value=0, max_value=9))
        if kind == 0:
            values.append(v * 10.0)
        elif kind == 1:
            values.append(v * 0.1)
        elif kind == 2:
            values.append(v * 100.0)
        elif kind == 3:
            values.append(0.0)
        elif kind == 4:
            values.append(None)
        else:
            values.append(v)
    return values


class TestDifferentialFuzz:
    """Hypothesis fuzz: random short series must agree between implementations."""

    @given(values=_series_strategy(), method=st.sampled_from(["bessembinder", "interpolation"]))
    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_equivalence(self, values, method):
        n = len(values)
        df = pl.LazyFrame(
            {
                "gvkey": ["000001"] * n,
                "iid": ["01"] * n,
                "datadate": list(range(n)),
                "prc": pl.Series(values, dtype=pl.Float64),
            }
        )
        _assert_equivalent(df, correction_method=method, window_sizes=[1, 2, 3, 5])
