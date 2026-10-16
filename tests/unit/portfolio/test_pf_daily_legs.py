"""Unit tests for ``_pf_daily_lazy`` and ``_split_daily_leg``.

The two helpers aggregate several daily return columns through one shared
weights-to-daily join and then split the wide result into one long frame per
column. These tests pin the per-leg counts, weighted sums and the dropping of
legs with no non-null returns on a date.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import _pf_daily_lazy, _split_daily_leg

RET_COLS = ["ret_exc", "ret_overnight"]


@pytest.fixture
def wide() -> pl.DataFrame:
    """One portfolio of two stocks (me 1 and 3) applied to three February days.

    Day 3: both stocks have ret_exc and one has ret_overnight.
    Day 4: only ret_overnight is present.
    Day 5: every return is null, so the row is dropped by the join filter.
    """
    eom = date(2020, 1, 31)
    sub = pl.LazyFrame(
        {
            "eom": [eom, eom],
            "pf": [1.0, 1.0],
            "id": [1, 2],
            "me": [1.0, 3.0],
            "me_cap": [1.0, 1.0],
        }
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
    return _pf_daily_lazy(sub, daily, "char_a", RET_COLS).collect().sort("date")


@pytest.mark.unit
class TestPfDailyLazy:
    def test_rows_kept_when_any_leg_present(self, wide: pl.DataFrame) -> None:
        assert wide["date"].to_list() == [date(2020, 2, 3), date(2020, 2, 4)]
        assert (wide["characteristic"] == "char_a").all()

    def test_per_leg_counts(self, wide: pl.DataFrame) -> None:
        assert wide["n_ret_exc"].to_list() == [2, 0]
        assert wide["n_ret_overnight"].to_list() == [1, 2]
        assert wide.schema["n_ret_exc"] == pl.UInt32

    def test_per_leg_weighted_sums(self, wide: pl.DataFrame) -> None:
        day3 = wide.row(0, named=True)
        assert day3["ret_ew_ret_exc"] == pytest.approx(0.015)
        assert day3["ret_vw_ret_exc"] == pytest.approx(0.25 * 0.01 + 0.75 * 0.02)
        assert day3["ret_vw_cap_ret_exc"] == pytest.approx(0.015)
        assert day3["ret_ew_ret_overnight"] == pytest.approx(0.5 * 0.03)
        assert day3["ret_vw_ret_overnight"] == pytest.approx(0.25 * 0.03)


@pytest.mark.unit
class TestSplitDailyLeg:
    def test_drops_dates_without_leg_returns(self, wide: pl.DataFrame) -> None:
        leg = _split_daily_leg(wide, "ret_exc")
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
        assert leg["n"].to_list() == [2]

    def test_keeps_leg_only_dates(self, wide: pl.DataFrame) -> None:
        leg = _split_daily_leg(wide, "ret_overnight")
        assert leg["date"].to_list() == [date(2020, 2, 3), date(2020, 2, 4)]
        assert leg["n"].to_list() == [1, 2]
        assert leg["ret_ew"].to_list() == pytest.approx([0.015, 0.015])
