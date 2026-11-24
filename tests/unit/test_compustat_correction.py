"""
Unit tests for the Compustat decimal-error corrections and the stock-month screen
(production slim/numba path).
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.compustat_correction import (
    apply_screen,
    correct_decimal_errors,
    screen_stock_months,
)

GROUP = ["gvkey", "iid"]
SORT = "datadate"


def _correction_frame(prccd: list[float], **overrides) -> pl.LazyFrame:
    """One security with a flat clean baseline for the non-price variables."""
    n = len(prccd)
    cols = {
        "gvkey": ["000001"] * n,
        "iid": ["01"] * n,
        "datadate": pl.Series(np.arange(n, dtype=np.int64)).cast(pl.Date),
        "ajexdi": [1.0] * n,
        "prccd": prccd,
        "cshoc": [8.0] * n,
        "trfd": [1.0] * n,
        "qunit": [1.0] * n,
    }
    cols.update(overrides)
    return pl.DataFrame(cols).lazy()


def _corrected_prccd(lf: pl.LazyFrame, tmp_path, method="multiplier", **kwargs) -> list[float]:
    out = correct_decimal_errors(lf, spill_dir=tmp_path, correction_method=method, **kwargs)
    return out.sort(GROUP + [SORT]).collect()["prccd"].to_list()


class TestDecimalCorrection:
    def test_single_period_high_10x_corrected(self, tmp_path):
        prc = _corrected_prccd(_correction_frame([20.0, 20.0, 200.0, 20.0, 20.0]), tmp_path)
        assert prc[2] == pytest.approx(20.0)

    def test_single_period_high_100x_corrected(self, tmp_path):
        prc = _corrected_prccd(_correction_frame([20.0, 20.0, 2000.0, 20.0, 20.0]), tmp_path)
        assert prc[2] == pytest.approx(20.0)

    def test_single_period_low_10x_corrected(self, tmp_path):
        prc = _corrected_prccd(_correction_frame([20.0, 20.0, 2.0, 20.0, 20.0]), tmp_path)
        assert prc[2] == pytest.approx(20.0)

    def test_multi_period_span_corrected(self, tmp_path):
        raw = [20.0, 20.0, 200.0, 200.0, 200.0, 20.0, 20.0]
        prc = _corrected_prccd(_correction_frame(raw), tmp_path)
        assert prc[2:5] == pytest.approx([20.0, 20.0, 20.0])

    def test_clean_series_unchanged(self, tmp_path):
        raw = [20.0, 20.5, 19.8, 20.1, 20.3]
        prc = _corrected_prccd(_correction_frame(raw), tmp_path)
        assert prc == pytest.approx(raw)

    def test_interpolation_uses_endpoint_geomean(self, tmp_path):
        # neighbors 16 and 25 -> geometric mean 20, not the fixed-multiplier 20.
        raw = [16.0, 160.0, 25.0]
        prc = _corrected_prccd(_correction_frame(raw), tmp_path, method="interpolation")
        assert prc[1] == pytest.approx(np.sqrt(16.0 * 25.0))

    def test_floor_skips_multiply_direction(self, tmp_path):
        # A low error (divide-direction correction is a multiply) is left alone
        # by the floor method, which only corrects divide-direction on >=$1.
        raw = [20.0, 20.0, 2.0, 20.0, 20.0]
        prc = _corrected_prccd(_correction_frame(raw), tmp_path, method="floor")
        assert prc[2] == pytest.approx(2.0)

    def test_trfd_corrected(self, tmp_path):
        lf = _correction_frame([20.0] * 5, trfd=[1.0, 1.0, 100.0, 1.0, 1.0])
        out = correct_decimal_errors(lf, spill_dir=tmp_path)
        trfd = out.sort(GROUP + [SORT]).collect()["trfd"].to_list()
        assert trfd[2] == pytest.approx(1.0)

    def test_requires_spill_dir(self):
        with pytest.raises(ValueError, match="requires spill_dir"):
            correct_decimal_errors(_correction_frame([20.0, 20.0, 20.0]))

    def test_rejects_unknown_method(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown correction_method"):
            correct_decimal_errors(
                _correction_frame([20.0]), spill_dir=tmp_path, correction_method="bogus"
            )

    def test_determinism_rerun(self, tmp_path):
        lf = _correction_frame([20.0, 20.0, 200.0, 200.0, 20.0, 2.0, 20.0])
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a = _corrected_prccd(lf, tmp_path / "a")
        b = _corrected_prccd(lf, tmp_path / "b")
        assert a == b


N_MONTHS = 60
TARGET = "000059"


def _months(n: int) -> list[date]:
    return (
        pl.date_range(date(2000, 1, 1), date(2000 + n // 12 + 1, 1, 1), "1mo", eager=True)[:n]
        .dt.month_end()
        .to_list()
    )


def _monthly_security(g: int, n: int = N_MONTHS, **cols) -> pl.DataFrame:
    """One security, one row per month: flat shares, price drifting up 0.1% a month,
    return index equal to the price, volume rising with g (so the low-volume cut takes
    the smallest g). Keyword arguments replace columns; me is price x shares unless given."""
    prc = 10.0 * np.exp(np.cumsum(np.full(n, 0.001)))
    base = {
        "gvkey": [f"{g:06d}"] * n,
        "iid": ["01"] * n,
        "eom": _months(n),
        "prc": prc,
        "cshoc": np.full(n, 5.0),
        "ajexdi": np.ones(n),
        "qunit": np.full(n, np.nan),
        "curcdd": ["USD"] * n,
        "excntry": ["GBR"] * n,
        "dolvol": prc * 1000.0 * (g + 1),
    }
    base.update(cols)
    base["ri"] = cols.get("ri", base["prc"] / base["ajexdi"])
    base["me"] = cols.get("me", base["prc"] * base["cshoc"])
    out = pl.DataFrame(base).with_columns(datadate=pl.col("eom"))
    return out.with_columns(pl.col("qunit").fill_nan(None))


def _panel(target: pl.DataFrame) -> pl.DataFrame:
    """59 clean securities plus the target (highest volume)."""
    return pl.concat([_monthly_security(g) for g in range(59)] + [target], how="diagonal")


def _screen(monthly: pl.DataFrame, daily: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
    """Decisions for the target; daily defaults to one row per month (the month-end)."""
    daily = monthly if daily is None else daily
    out = screen_stock_months(daily.lazy(), monthly.lazy(), **kwargs)
    return out.filter(pl.col("gvkey") == TARGET).sort("eom")


def _kept(decisions: pl.DataFrame) -> list[bool]:
    return decisions["keep"].to_list()


def _factors(decisions: pl.DataFrame) -> np.ndarray:
    return decisions["factor"].to_numpy()


class TestMonthlyScreen:
    def test_clean_security_fully_kept(self):
        d = _screen(_panel(_monthly_security(59)))
        assert all(_kept(d)) and len(d) == N_MONTHS
        assert (_factors(d) == 1.0).all()

    def test_low_volume_drops_lowest_volume_security(self):
        panel = _panel(_monthly_security(59))
        d = screen_stock_months(panel.lazy(), panel.lazy()).filter(pl.col("gvkey") == "000000")
        assert not any(_kept(d))

    def test_zero_ajexdi_in_daily_drops_security(self):
        # the drop is per security: the company's other share class (iid 02) is kept
        sibling = _monthly_security(59).with_columns(iid=pl.lit("02"))
        monthly = pl.concat([_panel(_monthly_security(59)), sibling], how="diagonal")
        daily = monthly.with_columns(
            ajexdi=pl.when(
                (pl.col("gvkey") == TARGET)
                & (pl.col("iid") == "01")
                & (pl.col("eom") == _months(5)[4])
            )
            .then(0.0)
            .otherwise(pl.col("ajexdi"))
        )
        d = _screen(monthly, daily)
        assert not any(_kept(d.filter(pl.col("iid") == "01")))
        assert all(_kept(d.filter(pl.col("iid") == "02")))

    def test_qunit_change_with_price_jump_drops_security(self):
        qunit = np.where(np.arange(N_MONTHS) < 30, 1.0, 100.0)
        prc = 10.0 * np.where(np.arange(N_MONTHS) < 30, 1.0, 100.0)
        target = _monthly_security(
            59, qunit=qunit, prc=prc, cshoc=50.0 / prc, ri=np.full(N_MONTHS, 10.0)
        )
        d = _screen(_panel(target))
        assert not any(_kept(d))

    def test_qunit_change_with_currency_change_kept(self):
        qunit = np.where(np.arange(N_MONTHS) < 30, 1.0, 100.0)
        prc = 10.0 * np.where(np.arange(N_MONTHS) < 30, 1.0, 100.0)
        curcdd = ["DEM"] * 30 + ["EUR"] * 30
        target = _monthly_security(
            59, qunit=qunit, prc=prc, cshoc=50.0 / prc, curcdd=curcdd, ri=np.full(N_MONTHS, 10.0)
        )
        assert all(_kept(_screen(_panel(target))))

    def test_initial_low_market_value_drops_security(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[0] = 0.01
        assert not any(_kept(_screen(_panel(_monthly_security(59, cshoc=cshoc)))))

    def test_later_low_market_value_drops_the_months_after(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[40] = 0.01  # month 40 breaches; it is kept, months 41+ are removed
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        assert _kept(d) == [True] * 41 + [False] * (N_MONTHS - 41)

    def test_breach_month_does_not_rescale_the_history(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[40] = 0.01  # a 500x shares drop into the breach month
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        assert (_factors(d)[:41] == 1.0).all()

    def test_breach_cut_applies_to_months_without_a_panel_row(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[40] = 0.01
        monthly = _panel(_monthly_security(59, cshoc=cshoc))
        extra = monthly.filter(pl.col("gvkey") == TARGET).tail(1)
        later = extra.with_columns(eom=date(2005, 3, 31), datadate=date(2005, 3, 31))
        d = _screen(monthly, pl.concat([monthly, later]))
        assert d["eom"].to_list()[-1] == date(2005, 3, 31)
        assert not d["keep"].to_list()[-1]

    def test_gap_of_more_than_eleven_months_drops_the_month_after(self):
        target = pl.concat(
            [
                _monthly_security(59, n=40).head(20),
                _monthly_security(59, n=40).slice(32),  # 12 months missing
            ]
        )
        d = _screen(_panel(target))
        assert _kept(d) == [True] * 20 + [False] + [True] * 7

    def test_gap_of_eleven_months_kept(self):
        target = pl.concat(
            [
                _monthly_security(59, n=40).head(20),
                _monthly_security(59, n=40).slice(31),  # 11 months missing
            ]
        )
        assert all(_kept(_screen(_panel(target))))

    def test_early_shares_jump_deletes_history_through_the_jump(self):
        cshoc = np.where(np.arange(N_MONTHS) < 5, 5.0, 50.0)
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        assert _kept(d) == [False] * 6 + [True] * (N_MONTHS - 6)

    def test_early_window_is_the_lesser_of_24_months_and_20_percent(self):
        # 60 months: 20% = 12 months, so a jump in month 15 is repaired, not deleted
        cshoc = np.where(np.arange(N_MONTHS) < 15, 5.0, 50.0)
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        assert all(_kept(d))
        np.testing.assert_allclose(_factors(d), np.where(np.arange(N_MONTHS) < 15, 1.0, 0.1))

    def test_one_early_window_for_shares_and_market_value_jumps(self):
        # 60 months: the window is the first 12. A shares jump in month 10 deletes months
        # 0-10; the market-value jump in month 15 (x1000, no return) is outside the
        # window, so it is repaired, not deleted, although it falls within 20% of the
        # months left after the cut
        cshoc = np.where(np.arange(N_MONTHS) < 10, 5.0, 50.0)
        sec = _monthly_security(59, cshoc=cshoc)
        me = sec["me"].to_numpy() * np.where(np.arange(N_MONTHS) < 15, 1.0, 1000.0)
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc, me=me)))
        assert _kept(d) == [False] * 11 + [True] * (N_MONTHS - 11)
        np.testing.assert_allclose(
            _factors(d)[11:], np.where(np.arange(11, N_MONTHS) < 15, 1.0, 1e-3), rtol=1e-12
        )

    def test_unpaired_shares_up_jump_scales_the_months_after(self):
        cshoc = np.where(np.arange(N_MONTHS) < 30, 5.0, 50.0)
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        assert all(_kept(d))
        np.testing.assert_allclose(_factors(d), np.where(np.arange(N_MONTHS) < 30, 1.0, 0.1))

    def test_unpaired_shares_down_jump_scales_the_months_before(self):
        cshoc = np.where(np.arange(N_MONTHS) < 30, 50.0, 5.0)
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        np.testing.assert_allclose(_factors(d), np.where(np.arange(N_MONTHS) < 30, 0.1, 1.0))

    def test_china_shares_jump_needs_the_looser_bound(self):
        # shares x10 with market value x3: a shares jump outside China (>= 5x, >= 2.5x),
        # inside China's 50x / 25x bounds; too small for the market-value rule (> 10x)
        after = np.arange(N_MONTHS) >= 30
        cols = {
            "prc": np.where(after, 3.0, 10.0),
            "cshoc": np.where(after, 50.0, 5.0),
            "ri": np.full(N_MONTHS, 10.0),
        }
        other = _screen(_panel(_monthly_security(59, **cols)))
        np.testing.assert_allclose(_factors(other), np.where(after, 0.1, 1.0))
        china = _monthly_security(59, excntry=["CHN"] * N_MONTHS, **cols)
        assert (_factors(_screen(_panel(china))) == 1.0).all()

    def test_paired_shares_jump_takes_the_smaller_end(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[30:35] = 50.0
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        expected = np.ones(N_MONTHS)
        expected[30:35] = 0.1
        np.testing.assert_allclose(_factors(d), expected)

    def test_reversal_beyond_the_pair_limit_is_two_unpaired_jumps(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[20:45] = 50.0
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)), pair_max_months=12)
        # the up-jump scales months 20+ down; the reversal is then a down-jump that
        # scales everything before it down: one consistent level at the lowest value
        expected = np.full(N_MONTHS, 0.1)
        expected[20:45] = 0.01
        np.testing.assert_allclose(_factors(d), expected)
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)), pair_max_months=30)
        expected = np.ones(N_MONTHS)
        expected[20:45] = 0.1
        np.testing.assert_allclose(_factors(d), expected)

    def test_unpaired_market_value_jump_keeps_the_return(self):
        me = 50.0 * np.exp(np.cumsum(np.full(N_MONTHS, 0.001)))
        me[30:] *= 20.0  # market value x20 with flat shares and a 0.1% return
        d = _screen(_panel(_monthly_security(59, me=me)))
        expected = np.where(np.arange(N_MONTHS) < 30, 1.0, 1.0 / 20.0)
        np.testing.assert_allclose(_factors(d), expected, rtol=1e-12)

    def test_paired_market_value_jump_keeps_the_window_returns(self):
        # returns alternate +2% / -1%; market value x20 in months 30-34 with flat shares
        prc = 10.0 * np.exp(np.cumsum(np.where(np.arange(N_MONTHS) % 2 == 0, 0.02, -0.01)))
        me = prc * 5.0
        me[30:35] *= 20.0
        d = _screen(_panel(_monthly_security(59, prc=prc, me=me)))
        assert all(_kept(d))
        # one multiplier for the window, so the repaired market value keeps the returns
        expected = np.ones(N_MONTHS)
        expected[30:35] = 1.0 / 20.0
        np.testing.assert_allclose(_factors(d), expected, rtol=1e-12)

    def test_reversal_to_a_lower_level_repairs_the_step_left(self):
        # shares 5 -> 50 -> 0.25: the window goes to 0.25, leaving 5 -> 0.25 at the jump
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[30:35] = 50.0
        cshoc[35:] = 0.25
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        assert all(_kept(d))
        expected = np.ones(N_MONTHS)
        expected[:30] = 0.05
        expected[30:35] = 0.005
        np.testing.assert_allclose(_factors(d), expected)

    def test_reversal_to_a_higher_level_repairs_the_step_left(self):
        # shares 0.5 -> 50 -> 5: the window goes to 0.5, leaving 0.5 -> 5 at the reversal
        cshoc = np.full(N_MONTHS, 0.5)
        cshoc[30:35] = 50.0
        cshoc[35:] = 5.0
        d = _screen(_panel(_monthly_security(59, cshoc=cshoc)))
        assert all(_kept(d))
        expected = np.ones(N_MONTHS)
        expected[30:35] = 0.01
        expected[35:] = 0.1
        np.testing.assert_allclose(_factors(d), expected)

    def test_reversal_after_a_large_window_return_repairs_the_step_left(self):
        # market value x1000 in month 30 with no return, then +300% over the window and
        # a x0.02 reversal: the window meets the jump, leaving x20 with no return at the
        # reversal, although the whole stretch has a 300% return
        prc = np.full(N_MONTHS, 10.0)
        prc[31:35] = 10.0 * 4.0 ** (np.arange(1, 5) / 4)
        prc[35:] = 40.0
        me = np.full(N_MONTHS, 50.0)
        me[30:35] = 50_000.0 * prc[30:35] / 10.0
        me[35:] = 4_000.0
        d = _screen(_panel(_monthly_security(59, prc=prc, me=me)))
        assert all(_kept(d))
        expected = np.ones(N_MONTHS)
        expected[30:35] = 1e-3
        expected[35:] = 0.05
        np.testing.assert_allclose(_factors(d), expected, rtol=1e-12)

    def test_reversal_step_is_found_at_the_end_left_misaligned(self):
        # market value x1000 in month 30, drifting back down inside the window with no
        # return, then x0.05: end to end (50 to 2.5) it looks like a drop at the jump,
        # but the window meets the jump and leaves x50 at the reversal
        me = np.full(N_MONTHS, 50.0)
        me[30:35] = 50_000.0 * 1000.0 ** (-np.arange(5) / 4)
        me[35:] = 2.5
        d = _screen(_panel(_monthly_security(59, prc=np.full(N_MONTHS, 10.0), me=me)))
        assert all(_kept(d))
        expected = np.ones(N_MONTHS)
        expected[30:35] = 1e-3
        expected[35:] = 0.02
        np.testing.assert_allclose(_factors(d), expected, rtol=1e-12)

    def test_reversal_step_is_found_at_the_jump_when_the_window_drifts_up(self):
        # market value x1000 in month 30, drifting up x100 inside the window with no
        # return, then x0.00005: the window meets the reversal and leaves a x0.05 drop
        # with no return at the jump, although end to end (50 to 250) is only x5
        me = np.full(N_MONTHS, 50.0)
        me[30:35] = 50_000.0 * 100.0 ** (np.arange(5) / 4)
        me[35:] = 250.0
        d = _screen(_panel(_monthly_security(59, prc=np.full(N_MONTHS, 10.0), me=me)))
        assert all(_kept(d))
        expected = np.ones(N_MONTHS)
        expected[:30] = 0.05
        expected[30:35] = 5e-5
        np.testing.assert_allclose(_factors(d), expected, rtol=1e-12)

    def test_initial_ratio_uses_split_adjusted_price(self):
        # a 20:1 split in month 2: raw price ratio 0.05, split-adjusted ratio 1
        prc = np.full(N_MONTHS, 10.0)
        prc[2:] = 0.5
        ajexdi = np.where(np.arange(N_MONTHS) < 2, 1.0, 0.05)
        cshoc = np.where(np.arange(N_MONTHS) < 2, 5.0, 100.0)
        d = _screen(_panel(_monthly_security(59, prc=prc, ajexdi=ajexdi, cshoc=cshoc)))
        assert all(_kept(d))

    def test_daily_rows_follow_the_month_and_the_jump_day(self):
        cshoc = np.where(np.arange(N_MONTHS) < 30, 5.0, 50.0)
        monthly = _panel(_monthly_security(59, cshoc=cshoc))
        jump_month = _months(N_MONTHS)[30]
        before = (
            monthly.filter((pl.col("gvkey") == TARGET) & (pl.col("eom") == jump_month))
            .with_columns(datadate=date(jump_month.year, jump_month.month, 10), cshoc=5.0)
            .with_columns(me=pl.col("prc") * pl.col("cshoc"))
        )
        daily = pl.concat([monthly, before])
        decisions = screen_stock_months(daily.lazy(), monthly.lazy())
        out = (
            apply_screen(daily.lazy(), decisions.lazy())
            .filter((pl.col("gvkey") == TARGET) & (pl.col("eom") == jump_month))
            .sort("datadate")
            .collect()
        )
        # the day before the jump keeps its shares; the month-end is scaled down to them
        assert out["cshoc"].to_list() == [5.0, 5.0]
        assert out.columns == daily.columns

    def test_monthly_rows_follow_the_decisions(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[40] = 0.01
        monthly = _panel(_monthly_security(59, cshoc=cshoc))
        decisions = screen_stock_months(monthly.lazy(), monthly.lazy())
        out = apply_screen(monthly.lazy(), decisions.lazy()).collect()
        assert out.filter(pl.col("gvkey") == TARGET).height == 41

    def test_determinism_rerun(self):
        cshoc = np.full(N_MONTHS, 5.0)
        cshoc[30:35] = 50.0
        panel = _panel(_monthly_security(59, cshoc=cshoc))
        a = screen_stock_months(panel.lazy(), panel.lazy()).sort(GROUP + ["eom"])
        b = screen_stock_months(panel.lazy(), panel.lazy()).sort(GROUP + ["eom"])
        assert a.equals(b)
