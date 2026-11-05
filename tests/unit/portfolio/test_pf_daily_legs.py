"""Unit tests for ``_pf_daily_returns``.

All daily return columns are summed in one pass over a shared stock-day table, which
gives one long frame per column. These tests pin the per-leg counts, weighted sums
and the dropping of legs with no non-null returns on a date.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import _month_totals, _pf_daily_returns, _stock_days

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
