"""Unit tests for ``_sum_by_portfolio``, the one pass behind monthly and daily returns.

Output sums are compared with a Polars group-by and across thread counts: the
equivalence with the old per-characteristic code is in ``test_speedup_equivalence.py``.
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import _SUM_VALUES, _sum_by_portfolio

N_PFS = 3


@pytest.fixture
def rows() -> pl.DataFrame:
    """Stock-days of 40 stock-months over 25 periods of very different sizes."""
    rng = np.random.default_rng(3)
    n = 4000
    return pl.DataFrame(
        {
            "period": np.sort(np.minimum(rng.integers(0, 25, n), rng.integers(0, 25, n))),
            "sm": rng.integers(0, 40, n),
            "a": rng.normal(0, 0.02, n),
            "b": rng.lognormal(0, 3, n),
        },
        schema_overrides={"sm": pl.UInt32},
    ).sort("period", "sm")


@pytest.fixture
def pf_by_row() -> np.ndarray:
    """Portfolios of 40 stock-months in 2 characteristics, with 0 for unassigned."""
    rng = np.random.default_rng(4)
    return rng.integers(0, N_PFS + 1, (40, 2)).astype(np.uint8)


@pytest.mark.unit
class TestSumByPortfolio:
    def test_matches_a_group_by(self, rows: pl.DataFrame, pf_by_row: np.ndarray) -> None:
        periods, totals = _sum_by_portfolio(rows, "period", ["a", "b"], pf_by_row, N_PFS)
        assert periods.name == "period"
        assert totals.shape == (periods.len(), 2, N_PFS, 2)
        for char in range(2):
            expected = (
                rows.with_columns(pl.Series("pf", pf_by_row[rows["sm"].to_numpy(), char]))
                .filter(pl.col("pf") > 0)
                .group_by("period", "pf")
                .agg(pl.sum("a"), pl.sum("b"))
            )
            found = np.zeros_like(totals[:, char])
            for period, pf, a, b in expected.iter_rows():
                found[periods.search_sorted(period), pf - 1] = (a, b)
            np.testing.assert_allclose(totals[:, char], found, rtol=1e-12, atol=1e-12)

    def test_any_thread_count_gives_the_same_bits(
        self, rows: pl.DataFrame, pf_by_row: np.ndarray
    ) -> None:
        before = numba.get_num_threads()
        results = []
        try:
            for threads in {1, 2, numba.config.NUMBA_NUM_THREADS}:
                numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
                _, totals = _sum_by_portfolio(rows, "period", ["a", "b"], pf_by_row, N_PFS)
                results.append(totals.tobytes())
        finally:
            numba.set_num_threads(before)
        assert len(set(results)) == 1

    def test_nan_stays_in_its_portfolio(self) -> None:
        rows = pl.DataFrame(
            {"period": [0, 0, 0], "sm": [0, 1, 2], "a": [np.nan, 1.0, 2.0]},
            schema_overrides={"sm": pl.UInt32},
        )
        pf_by_row = np.array([[1, 3], [2, 3], [3, 1]], dtype=np.uint8)
        _, totals = _sum_by_portfolio(rows, "period", ["a"], pf_by_row, N_PFS)
        assert np.isnan(totals[0, 0, 0, 0])
        assert totals[0, 0, 1:, 0].tolist() == [1.0, 2.0]
        assert totals[0, 1, 0, 0] == 2.0
        assert totals[0, 1, 1, 0] == 0.0
        assert np.isnan(totals[0, 1, 2, 0])

    def test_unassigned_rows_add_nothing(self) -> None:
        rows = pl.DataFrame(
            {"period": [0, 0], "sm": [0, 1], "a": [np.nan, 5.0]},
            schema_overrides={"sm": pl.UInt32},
        )
        pf_by_row = np.array([[0], [1]], dtype=np.uint8)
        _, totals = _sum_by_portfolio(rows, "period", ["a"], pf_by_row, N_PFS)
        assert totals[0, 0, :, 0].tolist() == [5.0, 0.0, 0.0]

    def test_no_rows(self) -> None:
        rows = pl.DataFrame(
            schema={"period": pl.Date, "sm": pl.UInt32, "a": pl.Float64},
        )
        periods, totals = _sum_by_portfolio(
            rows, "period", ["a"], np.zeros((0, 2), dtype=np.uint8), N_PFS
        )
        assert periods.len() == 0
        assert totals.shape == (0, 2, N_PFS, 1)

    def test_all_value_columns_fit(self) -> None:
        cols = {f"v{i}": [float(i)] for i in range(_SUM_VALUES)}
        rows = pl.DataFrame({"period": [0], "sm": [0], **cols}, schema_overrides={"sm": pl.UInt32})
        _, totals = _sum_by_portfolio(
            rows, "period", list(cols), np.array([[2]], dtype=np.uint8), N_PFS
        )
        assert totals[0, 0, 1].tolist() == [float(i) for i in range(_SUM_VALUES)]

    def test_too_many_value_columns_raises(self) -> None:
        cols = [f"v{i}" for i in range(_SUM_VALUES + 1)]
        rows = pl.DataFrame(
            {"period": [0], "sm": [0], **{c: [1.0] for c in cols}},
            schema_overrides={"sm": pl.UInt32},
        )
        with pytest.raises(ValueError, match="value columns"):
            _sum_by_portfolio(rows, "period", cols, np.ones((1, 1), dtype=np.uint8), N_PFS)
