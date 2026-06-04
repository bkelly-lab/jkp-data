"""
Differential tests: numba kernel implementation vs original Polars reference.

The production Section 6 implementation (numba kernels behind the public
functions in jkp.data.bessembinder) must be byte-identical to the original
handoff implementation, preserved verbatim in
jkp.data._bessembinder_polars_reference. These tests run both on the same
synthetic inputs and assert identical corrected frames and correction logs.

One DELIBERATE divergence exists: the kernel classes multi-period magnitudes
with the nested 500/50/5 chain (like single-period), fixing the original's
under-correction of multi-day 100x/1000x errors (its magnitude [1, 2, 3]
loop was dead code beyond 10x — commit 507df7c). The equivalence tests below
exercise only patterns outside that divergence; TestMultiPeriodMagnitudeFix
pins the divergence itself in both directions.
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
            # 10x (not 100x): a zero neighbor pushes this onto the multi-period
            # path, where magnitude classing deliberately diverges from the
            # reference (see TestMultiPeriodMagnitudeFix)
            x[4] = 0.0
            x[5] *= 10.0
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


def _span_frame(mult: float, width: int = 2, n: int = 12) -> pl.LazyFrame:
    """Flat series with a stable multi-day error span of the given multiplier."""
    x = np.full(n, 10.0)
    x[4 : 4 + width] *= mult
    return pl.LazyFrame(
        {
            "gvkey": ["000001"] * n,
            "iid": ["01"] * n,
            "datadate": list(range(n)),
            "prc": x,
        }
    )


class TestMultiPeriodMagnitudeFix:
    """
    Pin the deliberate divergence: multi-day 100x/1000x errors get true
    magnitude corrections from the kernel, while the original reference
    always applied 10x (dead-code magnitude loop).
    """

    @pytest.mark.parametrize(
        ("mult", "factor", "error_type"),
        [
            (100.0, 0.01, "high_100x"),
            (1000.0, 0.001, "high_1000x"),
            (0.01, 100.0, "low_100x"),
            (0.001, 1000.0, "low_1000x"),
        ],
    )
    def test_kernel_corrects_true_magnitude(self, mult, factor, error_type):
        _, log = correct_decimal_errors(_span_frame(mult), "prc", GROUP_COLS, SORT_COL)
        rows = log.collect()
        assert rows.height == 2
        assert set(rows["correction_factor"].to_list()) == {factor}
        assert set(rows["error_type"].to_list()) == {error_type}
        assert set(rows["window_type"].to_list()) <= {"full", "sub_a", "sub_b"}

    def test_reference_undercorrected_to_10x(self):
        _, ref_log = _polars_correct_decimal_errors(_span_frame(100.0), "prc", GROUP_COLS, SORT_COL)
        rows = ref_log.collect()
        assert rows.height == 2
        assert set(rows["correction_factor"].to_list()) == {0.1}
        assert set(rows["error_type"].to_list()) == {"high_10x"}

    def test_10x_spans_unchanged_vs_reference(self):
        for mult in (10.0, 0.1):
            _assert_equivalent(_span_frame(mult))


@st.composite
def _series_strategy(draw):
    """
    Short random per-group series mixing clean values, 10x spikes (one
    direction per example), zeros and nulls.

    The value space is constrained to stay outside the deliberate multi-period
    magnitude divergence (see module docstring): base in [10, 30] with a
    single spike direction bounds every pairwise ratio to (50, 1/50) — no
    pattern can reach the 100x/1000x classing thresholds, so reference and
    kernel must agree exactly.
    """
    n = draw(st.integers(min_value=3, max_value=40))
    base = draw(
        st.lists(
            st.floats(min_value=10.0, max_value=30.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    spike = 10.0 if draw(st.booleans()) else 0.1
    values = []
    for v in base:
        kind = draw(st.integers(min_value=0, max_value=9))
        if kind in (0, 1):
            values.append(v * spike)
        elif kind == 2:
            values.append(v * (5.5 if spike == 10.0 else 1 / 5.5))
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


def _section6_frame(seed: int = 7, n_groups: int = 40) -> pl.LazyFrame:
    """Small panel with the full Section 6 input schema and injected errors."""
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        n = int(rng.integers(20, 120))
        prc = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        trfd = np.cumprod(1 + rng.normal(0.0002, 0.001, n))
        qunit = np.ones(n)
        adrrc = np.full(n, 2.0)
        cshoc = np.full(n, 1e6) * (1 + rng.normal(0, 0.001, n))
        if g % 4 == 0 and n > 12:
            prc[5] *= 10.0  # single-day price error
            w = [2, 3, 5][g % 3]
            prc[8 : 8 + w] *= 0.1  # multi-day price error
        if g % 4 == 1 and n > 8:
            trfd[4] *= 100.0
            cshoc[6] *= 10.0
        if g % 4 == 2 and n > 10:
            qunit[3] *= 10.0
            adrrc[7] *= 0.1
            prc[9] = 0.0
        frames.append(
            pl.DataFrame(
                {
                    "gvkey": [f"{g:06d}"] * n,
                    "iid": ["01"] * n,
                    "datadate": np.arange(n, dtype=np.int64),
                    "prccd": prc,
                    "ajexdi": np.where(rng.random(n) < 0.02, 2.0, 1.0),
                    "cshoc": cshoc,
                    "trfd": trfd,
                    "qunit": qunit,
                    "adrrc": adrrc,
                }
            )
        )
    return pl.concat(frames).lazy()


class TestSlimPathEquivalence:
    """The spill_dir array path must match the frame-based default path."""

    @pytest.mark.parametrize("has_adrrc", [True, False])
    def test_slim_matches_default(self, tmp_path, has_adrrc):
        from jkp.data.bessembinder import apply_bessembinder_section6

        df = _section6_frame()
        sort_cols = GROUP_COLS + [SORT_COL]
        default_df, default_log = apply_bessembinder_section6(df, has_adrrc=has_adrrc)
        slim_df, slim_log = apply_bessembinder_section6(df, has_adrrc=has_adrrc, spill_dir=tmp_path)
        assert_frame_equal(
            slim_df.sort(sort_cols).collect(),
            default_df.sort(sort_cols).collect(),
            check_column_order=False,
        )
        assert (slim_log is None) == (default_log is None)
        assert_frame_equal(
            slim_log.sort(LOG_SORT).collect(),
            default_log.sort(LOG_SORT).collect(),
            check_column_order=True,
        )
