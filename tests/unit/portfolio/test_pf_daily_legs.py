"""Unit tests for ``_pf_daily_returns`` and ``_portfolio_rows``.

All daily return columns are summed in one pass over a shared stock-day table, which
gives one long frame per column. These tests pin the per-leg counts, weighted sums
and the dropping of legs with no non-null returns on a date, and compare the
``_portfolio_rows`` kernel with a frozen copy of the NumPy code it replaced.
"""

from __future__ import annotations

from datetime import date

import numba
import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import _month_totals, _pf_daily_returns, _portfolio_rows, _stock_days

RET_COLS = ["ret_exc", "ret_overnight"]


@pytest.fixture
def legs() -> dict[str, pl.DataFrame]:
    """One portfolio of two stocks (me 1 and 3) applied to three February days.

    Day 3: both stocks have ret_exc and one has ret_overnight.
    Day 4: only ret_overnight is present.
    Day 5: every return is null, so the row is dropped.
    """
    eom = date(2020, 1, 31)
    data = pl.DataFrame(
        {
            "sm": [0, 1],
            "id": [1, 2],
            "eom": [eom, eom],
            "ret_exc_lead1m": [0.0, 0.0],
            "me": [1.0, 3.0],
            "me_cap": [1.0, 1.0],
        },
        schema_overrides={"sm": pl.UInt32},
    )
    daily = pl.DataFrame(
        {
            "id": [1, 2, 1, 2, 1, 2],
            "date": [date(2020, 2, d) for d in (3, 3, 4, 4, 5, 5)],
            "eom_lag1": [eom] * 6,
            "ret_exc": [0.01, 0.02, None, None, None, None],
            "ret_overnight": [0.03, None, 0.01, 0.02, None, None],
        }
    )
    pf_by_row = np.array([[1], [1]], dtype=np.uint8)
    months, month_totals = _month_totals(data, pf_by_row, 3)
    stock_days = _stock_days(data, daily, RET_COLS)
    return _pf_daily_returns(stock_days, pf_by_row, months, month_totals, ["char_a"], RET_COLS, 3)


@pytest.mark.unit
class TestPfDailyReturns:
    def test_per_leg_counts(self, legs: dict[str, pl.DataFrame]) -> None:
        assert legs["ret_exc"]["n"].to_list() == [2]
        assert legs["ret_overnight"]["n"].to_list() == [1, 2]
        assert legs["ret_exc"].schema["n"] == pl.UInt32

    def test_per_leg_weighted_sums(self, legs: dict[str, pl.DataFrame]) -> None:
        ret_exc = legs["ret_exc"].row(0, named=True)
        assert ret_exc["ret_ew"] == pytest.approx(0.015)
        assert ret_exc["ret_vw"] == pytest.approx(0.25 * 0.01 + 0.75 * 0.02)
        assert ret_exc["ret_vw_cap"] == pytest.approx(0.015)
        overnight = legs["ret_overnight"].row(0, named=True)
        assert overnight["ret_ew"] == pytest.approx(0.5 * 0.03)
        assert overnight["ret_vw"] == pytest.approx(0.25 * 0.03)

    def test_drops_dates_without_leg_returns(self, legs: dict[str, pl.DataFrame]) -> None:
        leg = legs["ret_exc"]
        assert leg.columns == [
            "pf",
            "date",
            "characteristic",
            "n",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
        ]
        assert leg["date"].to_list() == [date(2020, 2, 3)]
        assert (leg["characteristic"] == "char_a").all()

    def test_keeps_leg_only_dates(self, legs: dict[str, pl.DataFrame]) -> None:
        leg = legs["ret_overnight"]
        assert leg["date"].to_list() == [date(2020, 2, 3), date(2020, 2, 4)]
        assert leg["ret_ew"].to_list() == pytest.approx([0.015, 0.015])


# Five characteristics out of name order, three portfolios, three formation months and
# four dates per month.
CHARS = ["zeta", "alpha", "mid_1", "beta", "alpha_2"]
N_PFS = 3
N_MONTHS = 3
DAYS_PER_MONTH = 4


def _old_portfolio_rows(
    periods: pl.Series, numerators: np.ndarray, denominators: np.ndarray, chars: list[str]
) -> pl.DataFrame:
    """Frozen copy of the NumPy ``_portfolio_rows`` the kernel replaced.

    Its rows come in (period, characteristic index, pf) order, and ``denominators`` holds
    one row per period rather than per month.
    """
    period_i, char_i, pf_i = np.nonzero(numerators[:, :, :, 0] > 0)
    numerator = numerators[period_i, char_i, pf_i]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = numerator[:, 1:] / denominators[period_i, char_i, pf_i]
    return pl.DataFrame(
        {
            "pf": pf_i + 1.0,
            periods.name: periods.gather(period_i),
            "characteristic": pl.Series(chars).gather(char_i),
            "n": numerator[:, 0].astype(np.uint32),
            "ret_ew": returns[:, 0],
            "ret_vw": returns[:, 1],
            "ret_vw_cap": returns[:, 2],
        }
    )


def _sums(n_periods: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sums by (period, characteristic, pf - 1, value), and monthly denominators.

    The sums sit in the middle of a wider array, so the kernel gets a strided view as
    it does from ``_sum_by_portfolio``. About a quarter of the counts are zero, and
    the sums and denominators include NaN, infinity, -0.0 and zeros (0/0 and x/0).
    """
    rng = np.random.default_rng(seed)
    wide = rng.normal(size=(n_periods, len(CHARS), N_PFS, 8))
    wide[..., 2] = rng.integers(0, 4, size=(n_periods, len(CHARS), N_PFS))
    for value, special in zip([3, 4, 5, 3], [np.nan, np.inf, -np.inf, -0.0], strict=True):
        wide.reshape(-1, 8)[rng.choice(wide[..., 0].size, 6, replace=False), value] = special
    denominators = rng.uniform(0.5, 2.0, size=(N_MONTHS, len(CHARS), N_PFS, 3))
    denominators.reshape(-1, 3)[rng.choice(denominators[..., 0].size, 8, replace=False)] = 0.0
    return wide[..., 2:6], denominators


def _bits(frame: pl.DataFrame) -> list:
    """Every column as bits for floats and values otherwise, so -0.0 and NaN must match."""
    return [
        frame[c].to_numpy().view(np.uint64).tobytes()
        if frame.schema[c] == pl.Float64
        else frame[c].to_list()
        for c in frame.columns
    ]


@pytest.mark.unit
class TestPortfolioRows:
    def test_daily_matches_the_old_code_bit_for_bit(self) -> None:
        dates = pl.Series("date", pl.date_range(date(2020, 1, 1), date(2020, 1, 12), eager=True))
        numerators, denominators = _sums(dates.len(), seed=1)
        formation = np.repeat(np.arange(N_MONTHS), DAYS_PER_MONTH)
        new = _portfolio_rows(dates, numerators, denominators, formation, CHARS)
        old = _old_portfolio_rows(dates, numerators, denominators[formation], CHARS)
        assert new.height == old.height > 0
        assert _bits(new) == _bits(old.sort("characteristic", "pf", "date"))

    def test_monthly_signal_matches_the_old_mask(self) -> None:
        months = pl.Series("eom", [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)])
        numerators, denominators = _sums(N_MONTHS, seed=2)
        medians = np.random.default_rng(3).normal(size=(N_MONTHS, len(CHARS), N_PFS))
        new = _portfolio_rows(
            months, numerators, denominators, np.arange(N_MONTHS), CHARS, signals=medians
        )
        old = _old_portfolio_rows(months, numerators, denominators, CHARS).with_columns(
            pl.Series("signal", medians[numerators[:, :, :, 0] > 0])
        )
        assert _bits(new) == _bits(old.sort("characteristic", "pf", "eom"))

    def test_rows_come_sorted_by_characteristic_name(self) -> None:
        dates = pl.Series("date", pl.date_range(date(2020, 1, 1), date(2020, 1, 12), eager=True))
        numerators, denominators = _sums(dates.len(), seed=4)
        formation = np.repeat(np.arange(N_MONTHS), DAYS_PER_MONTH)
        keys = _portfolio_rows(dates, numerators, denominators, formation, CHARS).select(
            "characteristic", "pf", "date"
        )
        assert keys.equals(keys.sort(keys.columns))
        assert keys["characteristic"].unique(maintain_order=True).to_list() == sorted(CHARS)

    def test_any_thread_count_gives_the_same_bits(self) -> None:
        dates = pl.Series("date", pl.date_range(date(2020, 1, 1), date(2020, 1, 12), eager=True))
        numerators, denominators = _sums(dates.len(), seed=5)
        formation = np.repeat(np.arange(N_MONTHS), DAYS_PER_MONTH)
        before = numba.get_num_threads()
        results = []
        try:
            for threads in {1, 2, numba.config.NUMBA_NUM_THREADS}:
                numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
                rows = _portfolio_rows(dates, numerators, denominators, formation, CHARS)
                results.append(_bits(rows))
        finally:
            numba.set_num_threads(before)
        assert all(result == results[0] for result in results)

    def test_a_formation_month_outside_the_denominators_raises(self) -> None:
        """The kernel does not check bounds, so ``_portfolio_rows`` checks its indexes."""
        months = pl.Series("eom", [date(2020, 1, 31)])
        numerators = np.ones((1, len(CHARS), N_PFS, 4))
        denominators = np.ones((1, len(CHARS), N_PFS, 3))
        with pytest.raises(ValueError, match="outside denominators"):
            _portfolio_rows(months, numerators, denominators, np.array([1]), CHARS)
        with pytest.raises(ValueError, match="do not line up"):
            _portfolio_rows(months, numerators, denominators[:, :1], np.array([0]), CHARS)

    def test_no_stocks_gives_an_empty_frame_with_the_dtypes(self) -> None:
        months = pl.Series("eom", [date(2020, 1, 31)])
        numerators = np.zeros((1, len(CHARS), N_PFS, 4))
        denominators = np.ones((1, len(CHARS), N_PFS, 3))
        rows = _portfolio_rows(months, numerators, denominators, np.arange(1), CHARS)
        assert rows.height == 0
        assert rows.schema == _old_portfolio_rows(months, numerators, denominators, CHARS).schema
