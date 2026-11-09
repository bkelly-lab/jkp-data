"""Unit tests for ``_rank_by_month``, the one sort behind portfolios, medians and CMP.

Small cases that pin each rule by hand. The equivalence with the old per-characteristic
code on awkward data is in ``test_speedup_equivalence.py``.
"""

from __future__ import annotations

from datetime import date

import numba
import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import _cmp_returns, _rank_by_month

EOM = date(2020, 1, 31)


def _frame(
    values: list[float | None],
    bp_stock: list[bool | None] | None = None,
    size_grp: list[str] | None = None,
    ret: list[float] | None = None,
    eom: date = EOM,
) -> pl.DataFrame:
    n = len(values)
    return pl.DataFrame(
        {
            "eom": [eom] * n,
            "c": values,
            "bp_stock": bp_stock if bp_stock is not None else [True] * n,
            "size_grp": size_grp if size_grp is not None else ["mega"] * n,
            "ret_exc_lead1m": ret if ret is not None else [0.0] * n,
        },
        schema_overrides={
            "eom": pl.Date,
            "c": pl.Float64,
            "bp_stock": pl.Boolean,
            "size_grp": pl.String,
            "ret_exc_lead1m": pl.Float64,
        },
    )


def _portfolios(data: pl.DataFrame, n_pfs: int, bp_min_n: int = 1) -> list[int]:
    return _rank_by_month(data, ["c"], n_pfs, bp_min_n)[0][:, 0].tolist()


@pytest.mark.unit
class TestPortfolios:
    def test_each_breakpoint_stock_moves_the_ecdf_up(self) -> None:
        data = _frame([float(v) for v in range(1, 11)])
        assert _portfolios(data, 10) == list(range(1, 11))

    def test_tied_values_share_a_portfolio(self) -> None:
        assert _portfolios(_frame([1.0, 1.0, 2.0, 2.0]), 2) == [1, 1, 2, 2]

    def test_other_stocks_take_the_ecdf_of_the_breakpoint_stock_below(self) -> None:
        data = _frame([1.0, 2.0, 3.0, 2.5], bp_stock=[True, True, True, False])
        assert _portfolios(data, 3) == [1, 2, 3, 2]

    def test_stocks_below_every_breakpoint_stock_go_to_portfolio_one(self) -> None:
        data = _frame([10.0, 20.0, 5.0], bp_stock=[True, True, False])
        assert _portfolios(data, 3) == [2, 3, 1]

    def test_a_month_without_breakpoint_stocks_all_share_portfolio_one(self) -> None:
        data = _frame([1.0, 2.0, 3.0], bp_stock=[False, False, False])
        assert _portfolios(data, 3, bp_min_n=0) == [1, 1, 1]

    def test_lowest_value_is_in_portfolio_one_when_many_stocks_tie_there(self) -> None:
        data = _frame([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
        assert _portfolios(data, 3) == [1, 1, 1, 1, 3, 3]

    def test_months_are_independent(self) -> None:
        data = pl.concat([_frame([1.0, 2.0]), _frame([20.0, 10.0], eom=date(2020, 2, 29))])
        assert _portfolios(data, 2) == [1, 2, 2, 1]

    def test_thin_month_has_no_portfolios(self) -> None:
        data = _frame([1.0, 2.0, 3.0], bp_stock=[True, True, False])
        pf_by_row, medians, _ = _rank_by_month(data, ["c"], 3, 3)
        assert pf_by_row[:, 0].tolist() == [0, 0, 0]
        assert np.isnan(medians).all()

    def test_null_breakpoint_flag_is_not_a_breakpoint_stock(self) -> None:
        data = _frame([1.0, 2.0, 3.0], bp_stock=[True, None, True])
        assert _portfolios(data, 2, bp_min_n=3) == [0, 0, 0]
        assert _portfolios(data, 2, bp_min_n=2) == [1, 1, 2]

    def test_null_values_have_no_portfolio(self) -> None:
        assert _portfolios(_frame([None, 1.0, 2.0]), 2) == [0, 1, 2]


@pytest.mark.unit
class TestMedians:
    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0],
            [-np.inf, 1.0],
            [1.0, np.inf],
            [-np.inf, np.inf],
            [-np.inf, 1.0, 2.0, np.inf],
        ],
    )
    def test_matches_polars(self, values: list[float]) -> None:
        data = _frame(values)
        expected = data.group_by("eom").agg(pl.median("c"))["c"].to_numpy()
        medians = _rank_by_month(data, ["c"], 1, 1)[1]
        np.testing.assert_array_equal(medians[0, 0, 0], expected[0])

    def test_one_median_per_portfolio(self) -> None:
        medians = _rank_by_month(_frame([4.0, 3.0, 2.0, 1.0]), ["c"], 2, 1)[1]
        assert medians[0, 0].tolist() == [1.5, 3.5]


@pytest.mark.unit
class TestCmp:
    @staticmethod
    def _cmp(data: pl.DataFrame, n_groups: int) -> pl.DataFrame:
        months = data["eom"].unique(maintain_order=True)
        cmp_totals = _rank_by_month(data, ["c"], 3, 1)[2]
        return _cmp_returns(months, cmp_totals, ["c"], n_groups, "usa")

    def test_rank_weights(self) -> None:
        row = self._cmp(_frame([1.0, 2.0, 3.0, 4.0], ret=[0.01, 0.02, 0.03, 0.04]), 1).row(
            0, named=True
        )
        # Centered ranks -3, -1, 1, 3, with half their absolute sum 4.
        assert row["size_grp"] == "all"
        assert row["n_stocks"] == 4
        assert row["ret_weighted"] == pytest.approx(0.1 / 4)
        assert row["signal_weighted"] == pytest.approx(10.0 / 4)
        assert row["eom"] == date(2020, 2, 29)
        assert row["excntry"] == "USA"

    def test_tied_stocks_share_their_average_rank(self) -> None:
        row = self._cmp(_frame([1.0, 1.0, 2.0, 2.0], ret=[0.01, 0.03, 0.05, 0.07]), 1).row(
            0, named=True
        )
        # Centered ranks -2, -2, 2, 2, with half their absolute sum 4.
        assert row["ret_weighted"] == pytest.approx((2 * (0.12 - 0.04)) / 4)
        assert row["signal_weighted"] == pytest.approx((2 * 4.0 - 2 * 2.0) / 4)

    def test_whole_number_characteristic_is_exact(self) -> None:
        row = self._cmp(_frame([400.0] * 1000 + [424.0] * 2001), 1).row(0, named=True)
        assert row["signal_weighted"] == 24.0

    def test_each_grouping_ranks_within_its_own_groups(self) -> None:
        data = _frame(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            size_grp=["mega", "mega", "large", "small", "micro", "nano"],
        )
        assert self._cmp(data, 1)["n_stocks"].to_list() == [6]
        two = self._cmp(data, 2).sort("size_grp")
        assert two["size_grp"].to_list() == ["big", "small"]
        assert two["n_stocks"].to_list() == [3, 3]
        five = self._cmp(data, 5)
        assert five["size_grp"].to_list() == ["mega"]
        assert five["n_stocks"].to_list() == [2]

    def test_constant_group_is_dropped(self) -> None:
        assert self._cmp(_frame([2.5, 2.5, 2.5]), 1).height == 0

    @pytest.mark.parametrize(
        "values",
        [
            [0.5, 0.5 + 1e-14],
            [0.1, 0.1 + 1e-14, 0.1 + 2e-14],
            [0.0, 1e-16],
            [-1e-16, 1e-16],
            [0.5, 0.5 + 1e-10],
        ],
    )
    def test_group_at_or_below_the_minimum_std_is_dropped(self, values: list[float]) -> None:
        assert self._cmp(_frame(values), 1).height == 0

    @pytest.mark.parametrize(
        "values",
        [
            [0.5, 0.5 + 1e-9],
            [0.0, 2e-10],
            [1e6, 1e6 + 1e-8],
            [1.0, np.inf],
            [np.inf, np.inf],
            [-np.inf, np.inf],
            [-1e308, 1e308],
        ],
    )
    def test_group_above_the_minimum_std_stays(self, values: list[float]) -> None:
        assert self._cmp(_frame(values), 1).height == 1

    @pytest.mark.parametrize(("target", "rows"), [(0.99e-10, 0), (1.01e-10, 1)])
    def test_minimum_is_on_the_sample_std_with_ties(self, target: float, rows: int) -> None:
        raw = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 7.0, 7.0, 7.0, 7.0])
        values = 0.25 + raw * target / raw.std(ddof=1)
        assert self._cmp(_frame(values.tolist()), 1).height == rows

    @pytest.mark.parametrize(("n_tied", "rows"), [(399, 0), (99, 1)])
    def test_minimum_is_on_the_std_not_the_range(self, n_tied: int, rows: int) -> None:
        # One stock 1.1e-9 from the rest: the std is 5.5e-11 with 400 stocks, 1.1e-10 with 100.
        assert self._cmp(_frame([0.0] * n_tied + [1.1e-9]), 1).height == rows

    def test_minimum_std_does_not_depend_on_row_order(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.normal(size=50)
        values = 0.25 + raw * 1e-10 / raw.std(ddof=1)
        rows = {self._cmp(_frame(rng.permutation(values).tolist()), 1).height for _ in range(20)}
        assert len(rows) == 1

    def test_minimum_std_applies_group_by_group(self) -> None:
        data = _frame(
            [0.5, 0.5 + 1e-14, 1.0, 2.0],
            size_grp=["mega", "mega", "large", "large"],
        )
        assert self._cmp(data, 5)["size_grp"].to_list() == ["large"]
        assert self._cmp(data, 1)["size_grp"].to_list() == ["all"]

    def test_single_stock_group_is_dropped(self) -> None:
        assert self._cmp(_frame([2.5]), 1).height == 0

    def test_group_of_infinite_values_stays_with_nan(self) -> None:
        row = self._cmp(_frame([np.inf, np.inf]), 1).row(0, named=True)
        assert row["n_stocks"] == 2
        assert np.isnan(row["ret_weighted"])

    def test_more_portfolios_than_the_array_holds_raises(self) -> None:
        with pytest.raises(ValueError, match="256 portfolios, at most 255 fit"):
            _rank_by_month(pl.DataFrame(), [], 256, 1)

    def test_unknown_size_group_raises(self) -> None:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            _rank_by_month(_frame([1.0, 2.0], size_grp=["mega", "giant"]), ["c"], 3, 1)


@pytest.mark.unit
class TestKernel:
    def test_any_thread_count_gives_the_same_bits(self) -> None:
        rng = np.random.default_rng(6)
        months = [date(2020, m, 28) for m in range(1, 13)]
        data = pl.concat(
            [
                pl.DataFrame(
                    {
                        "eom": [eom] * n,
                        "c": rng.integers(0, 20, n).astype(float),
                        "d": rng.normal(0, 1, n),
                        "bp_stock": rng.random(n) < 0.5,
                        "size_grp": rng.choice(["mega", "large", "small", "micro", "nano"], n),
                        "ret_exc_lead1m": rng.normal(0, 0.1, n),
                    }
                )
                for eom, n in zip(months, rng.integers(10, 400, 12), strict=True)
            ]
        )
        before = numba.get_num_threads()
        results = set()
        try:
            for threads in {1, 2, numba.config.NUMBA_NUM_THREADS}:
                numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
                arrays = _rank_by_month(data, ["c", "d"], 3, 5)
                results.add(b"".join(a.tobytes() for a in arrays))
        finally:
            numba.set_num_threads(before)
        assert len(results) == 1

    def test_no_rows(self) -> None:
        data = _frame([])
        pf_by_row, medians, cmp_totals = _rank_by_month(data, ["c"], 3, 1)
        assert pf_by_row.shape == (0, 1)
        assert medians.shape == (0, 1, 3)
        assert cmp_totals.shape == (0, 1, 8, 4)
