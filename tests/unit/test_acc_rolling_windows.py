"""Tests for the rolling-window accounting helpers.

Covers standardized_unexpected, volq, vola, earnings_variability, roe_and_g_exps,
be_and_cd_exps and equity_duration_cd. Frames are monthly panels with a running
``count`` per (gvkey, curcd), as in create_acc_chars. Expected values are computed
from the SAS definitions in char_macros.sas.
"""

from __future__ import annotations

import statistics
from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import (
    be_and_cd_exps,
    earnings_variability,
    equity_duration_cd,
    roe_and_g_exps,
    standardized_unexpected,
    vola,
    volq,
)


def _panel(firms: dict[str, dict[str, list[float | None]]]) -> pl.LazyFrame:
    """One monthly row per index; ``count`` runs 1..n per firm; all rows are USD."""
    frames = []
    for gvkey, cols in firms.items():
        n = len(next(iter(cols.values())))
        frames.append(
            pl.DataFrame(
                {
                    "gvkey": [gvkey] * n,
                    "curcd": ["USD"] * n,
                    "datadate": [date(2000 + k // 12, k % 12 + 1, 1) for k in range(n)],
                    "count": list(range(1, n + 1)),
                    **{
                        name: pl.Series(name, vals, dtype=pl.Float64) for name, vals in cols.items()
                    },
                }
            )
        )
    return pl.concat(frames).lazy()


def _get(df: pl.LazyFrame, gvkey: str, name: str) -> list[float | None]:
    return df.filter(pl.col("gvkey") == gvkey).sort("datadate").collect()[name].to_list()


def _approx_list(actual: list[float | None], expected: list[float | None]) -> None:
    assert len(actual) == len(expected)
    for k, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if e is None:
            assert a is None, f"row {k}: expected null, got {a}"
        else:
            assert a is not None, f"row {k}: expected {e}, got null"
            assert a == pytest.approx(e, rel=1e-9, abs=1e-12), f"row {k}"


def _ramp(n: int, scale: float, offset: float = 0.0) -> list[float | None]:
    """Deterministic non-linear series so that sample std is nonzero."""
    return [
        offset + scale * ((k * 7) % 11) + k * 0.5 + 0.03 * k * k * (1 + k % 3) for k in range(n)
    ]


@pytest.mark.unit
class TestVolq:
    """volq: std over lags 0,3,..,3(qtrs-1); null if count <= (qtrs-1)*3 or n < qtrs_min."""

    def test_count_gate_and_value(self) -> None:
        x = _ramp(12, 1.3)
        out = volq(_panel({"A": {"x": x}}), "x_std", "x", 3, 2)
        got = _get(out, "A", "x_std")
        # qtrs=3: gate count > 6, so index 5 (count 6) is null, index 6 (count 7) is set.
        assert all(v is None for v in got[:6])
        assert got[6] == pytest.approx(statistics.stdev([x[6], x[3], x[0]]))
        assert got[11] == pytest.approx(statistics.stdev([x[11], x[8], x[5]]))
        assert "aux" not in out.collect_schema().names()
        assert "__n" not in out.collect_schema().names()

    def test_min_observations_boundary(self) -> None:
        x = _ramp(12, 1.3)
        x[3] = None
        x[9] = None
        # qtrs_min=3 requires all three window values.
        out3 = volq(_panel({"A": {"x": x}}), "s", "x", 3, 3)
        got3 = _get(out3, "A", "s")
        # index 6 window (6,3,0): n=2 < 3 -> null. index 7 window (7,4,1): n=3 -> value.
        assert got3[6] is None
        assert got3[7] == pytest.approx(statistics.stdev([x[7], x[4], x[1]]))
        # index 9 is null itself; window (9,6,3) has n=1 -> null.
        assert got3[9] is None
        # qtrs_min=2 accepts n=2 exactly; index 6 uses the two non-null values.
        out2 = volq(_panel({"A": {"x": x}}), "s", "x", 3, 2)
        got2 = _get(out2, "A", "s")
        assert got2[6] == pytest.approx(statistics.stdev([x[6], x[0]]))
        # n=1 is below qtrs_min=2 (and std of one value is undefined).
        assert got2[9] is None

    def test_constant_series_gives_zero(self) -> None:
        out = volq(_panel({"A": {"x": [4.0] * 8}}), "s", "x", 3, 2)
        assert _get(out, "A", "s")[7] == pytest.approx(0.0)

    def test_firms_are_separate(self) -> None:
        a = _ramp(10, 1.3)
        b = _ramp(10, 4.0, 100.0)
        out = volq(_panel({"A": {"x": a}, "B": {"x": b}}), "s", "x", 3, 2)
        assert _get(out, "A", "s")[9] == pytest.approx(statistics.stdev([a[9], a[6], a[3]]))
        assert _get(out, "B", "s")[9] == pytest.approx(statistics.stdev([b[9], b[6], b[3]]))
        # Firm B's first rows must not see firm A's tail.
        assert _get(out, "B", "s")[6] == pytest.approx(statistics.stdev([b[6], b[3], b[0]]))


@pytest.mark.unit
class TestVola:
    """vola: std over lags 0,12,..,12(yrs-1); null if count <= (yrs-1)*12 or n < yrs_min."""

    def test_count_gate_and_value(self) -> None:
        x = _ramp(40, 1.3)
        out = vola(_panel({"A": {"x": x}}), "s", "x", 3, 3)
        got = _get(out, "A", "s")
        # yrs=3: gate count > 24, so index 23 (count 24) is null, index 24 (count 25) is set.
        assert got[23] is None
        assert got[24] == pytest.approx(statistics.stdev([x[24], x[12], x[0]]))
        assert got[39] == pytest.approx(statistics.stdev([x[39], x[27], x[15]]))

    def test_min_observations_boundary(self) -> None:
        x = _ramp(40, 1.3)
        x[12] = None
        out3 = vola(_panel({"A": {"x": x}}), "s", "x", 3, 3)
        out2 = vola(_panel({"A": {"x": x}}), "s", "x", 3, 2)
        assert _get(out3, "A", "s")[24] is None  # n=2 < 3
        assert _get(out2, "A", "s")[24] == pytest.approx(statistics.stdev([x[24], x[0]]))
        assert _get(out3, "A", "s")[25] == pytest.approx(statistics.stdev([x[25], x[13], x[1]]))

    def test_firms_are_separate(self) -> None:
        a = _ramp(30, 1.3)
        b = _ramp(30, 3.0, 50.0)
        out = vola(_panel({"A": {"x": a}, "B": {"x": b}}), "s", "x", 3, 3)
        assert _get(out, "A", "s")[29] == pytest.approx(statistics.stdev([a[29], a[17], a[5]]))
        assert _get(out, "B", "s")[24] == pytest.approx(statistics.stdev([b[24], b[12], b[0]]))


def _su_ref(var: list[float | None], qtrs: int, qtrs_min: int) -> list[float | None]:
    """SAS standardized_unexpected written out with plain Python."""
    n = len(var)
    chg: list[float | None] = [
        None if k < 12 or var[k] is None or var[k - 12] is None else var[k] - var[k - 12]  # type: ignore[operator]
        for k in range(n)
    ]
    mean: list[float | None] = []
    std: list[float | None] = []
    for j in range(n):
        w = [chg[j - 3 * i] for i in range(qtrs) if j - 3 * i >= 0 and chg[j - 3 * i] is not None]
        if len(w) > qtrs_min:
            mean.append(statistics.mean(w))  # type: ignore[arg-type]
            std.append(statistics.stdev(w))  # type: ignore[arg-type]
        else:
            mean.append(None)
            std.append(None)
    out: list[float | None] = []
    for k in range(n):
        ok = k + 1 > 12 + 3 * qtrs and k >= 15 and var[k] is not None and var[k - 12] is not None
        m, s = mean[k - 3] if k >= 3 else None, std[k - 3] if k >= 3 else None
        if ok and m is not None and s is not None and s != 0:
            out.append((var[k] - var[k - 12] - m) / s)  # type: ignore[operator]
        else:
            out.append(None)
    return out


@pytest.mark.unit
class TestStandardizedUnexpected:
    """SU = (x - lag12(x) - lag3(mean chg)) / lag3(std chg); mean/std need n > qtrs_min."""

    def test_hand_value_and_gates(self) -> None:
        x = _ramp(34, 2.0)
        out = standardized_unexpected(_panel({"A": {"sale_x": x}}), "sale_x", 3, 2)
        got = _get(out, "A", "sale_su")
        # qtrs=3, qtrs_min=2. chg[k] = x[k]-x[k-12] exists from k=12.
        # Window at j=15 holds chg[15],chg[12]: n=2, not > 2, so SU at k=18 is null.
        assert all(v is None for v in got[:21])
        # Window at j=18: chg[18],chg[15],chg[12] -> n=3 > 2; first SU is k=21 (count 22 > 21).
        chg = {k: x[k] - x[k - 12] for k in (12, 15, 18)}
        vals = [chg[18], chg[15], chg[12]]
        expected = (x[21] - x[9] - statistics.mean(vals)) / statistics.stdev(vals)
        assert got[21] == pytest.approx(expected)
        assert "sale_su" in out.collect_schema().names()
        for helper in ("aux", "__chg", "__chg_mean", "__chg_std", "__chg_n"):
            assert helper not in out.collect_schema().names()

    def test_matches_reference_over_full_series(self) -> None:
        x = _ramp(40, 2.0)
        out = standardized_unexpected(_panel({"A": {"sale_x": x}}), "sale_x", 3, 2)
        _approx_list(_get(out, "A", "sale_su"), _su_ref(x, 3, 2))

    def test_name_strips_x_suffix(self) -> None:
        out = standardized_unexpected(_panel({"A": {"ni_x": _ramp(5, 1.0)}}), "ni_x", 3, 2)
        assert "ni_su" in out.collect_schema().names()

    def test_missing_input_drops_below_min_n(self) -> None:
        x = _ramp(34, 2.0)
        x[17] = None
        out = standardized_unexpected(_panel({"A": {"sale_x": x}}), "sale_x", 3, 2)
        got = _get(out, "A", "sale_su")
        ref = _su_ref(x, 3, 2)
        _approx_list(got, ref)
        # chg[17] null makes window j=20 (20,17,14) n=2, so SU at k=23 is null.
        assert got[23] is None
        assert got[22] is not None

    def test_zero_std_gives_null(self) -> None:
        x: list[float | None] = [2.0 * k for k in range(30)]  # chg is constant 24
        out = standardized_unexpected(_panel({"A": {"sale_x": x}}), "sale_x", 3, 2)
        assert all(v is None for v in _get(out, "A", "sale_su"))

    def test_firms_are_separate(self) -> None:
        a = _ramp(34, 2.0)
        b = _ramp(34, 5.0, 40.0)
        out = standardized_unexpected(
            _panel({"A": {"sale_x": a}, "B": {"sale_x": b}}), "sale_x", 3, 2
        )
        _approx_list(_get(out, "A", "sale_su"), _su_ref(a, 3, 2))
        _approx_list(_get(out, "B", "sale_su"), _su_ref(b, 3, 2))


@pytest.mark.unit
class TestEarningsVariability:
    """std(ROA)/std(CROA) over esm_h years, ROA = ni / lag12(at); null if count <= 12*esm_h."""

    @staticmethod
    def _inputs() -> dict[str, list[float | None]]:
        n = 26
        return {
            "ni_x": [3.0 + ((k * 5) % 7) for k in range(n)],
            "ocf_x": [4.0 + ((k * 3) % 5) for k in range(n)],
            "at_x": [100.0 + 2 * k for k in range(n)],
        }

    def test_hand_value_and_gate(self) -> None:
        d = self._inputs()
        out = earnings_variability(_panel({"A": d}), esm_h=2)
        got = _get(out, "A", "earnings_variability")
        # esm_h=2: gate count > 24; index 23 (count 24) null, index 24 (count 25) set.
        assert got[23] is None
        roa = [d["ni_x"][24] / d["at_x"][12], d["ni_x"][12] / d["at_x"][0]]  # type: ignore[operator]
        croa = [d["ocf_x"][24] / d["at_x"][12], d["ocf_x"][12] / d["at_x"][0]]  # type: ignore[operator]
        assert got[24] == pytest.approx(statistics.stdev(roa) / statistics.stdev(croa))
        roa25 = [d["ni_x"][25] / d["at_x"][13], d["ni_x"][13] / d["at_x"][1]]  # type: ignore[operator]
        croa25 = [d["ocf_x"][25] / d["at_x"][13], d["ocf_x"][13] / d["at_x"][1]]  # type: ignore[operator]
        assert got[25] == pytest.approx(statistics.stdev(roa25) / statistics.stdev(croa25))
        for helper in ("__roa", "__croa", "__roa_std", "__croa_std", "aux1", "aux2"):
            assert helper not in out.collect_schema().names()

    def test_null_in_window_gives_null(self) -> None:
        d = self._inputs()
        d["ni_x"][12] = None  # roa[12] null: n_roa=1 < esm_h at index 24
        d["ocf_x"][13] = None  # croa[13] null: n_croa=1 at index 25
        out = earnings_variability(_panel({"A": d}), esm_h=2)
        got = _get(out, "A", "earnings_variability")
        assert got[24] is None
        assert got[25] is None

    def test_zero_croa_std_gives_null(self) -> None:
        d = self._inputs()
        n = 26
        d["at_x"] = [100.0] * n
        d["ocf_x"] = [5.0] * n  # croa constant -> std 0
        out = earnings_variability(_panel({"A": d}), esm_h=2)
        assert _get(out, "A", "earnings_variability")[24] is None

    def test_zero_lagged_assets_gives_null_roa(self) -> None:
        d = self._inputs()
        d["at_x"][0] = 0.0  # roa[12] and croa[12] undefined
        out = earnings_variability(_panel({"A": d}), esm_h=2)
        assert _get(out, "A", "earnings_variability")[24] is None

    def test_firms_are_separate(self) -> None:
        a = self._inputs()
        b = {
            "ni_x": [1.0 + ((k * 2) % 9) for k in range(26)],
            "ocf_x": [2.0 + ((k * 7) % 6) for k in range(26)],
            "at_x": [50.0 + 3 * k for k in range(26)],
        }
        out = earnings_variability(_panel({"A": a, "B": b}), esm_h=2)
        for gvkey, d in (("A", a), ("B", b)):
            roa = [d["ni_x"][24] / d["at_x"][12], d["ni_x"][12] / d["at_x"][0]]  # type: ignore[operator]
            croa = [d["ocf_x"][24] / d["at_x"][12], d["ocf_x"][12] / d["at_x"][0]]  # type: ignore[operator]
            got = _get(out, gvkey, "earnings_variability")[24]
            assert got == pytest.approx(statistics.stdev(roa) / statistics.stdev(croa))

    def test_default_horizon_is_five_years(self) -> None:
        n = 62
        d: dict[str, list[float | None]] = {
            "ni_x": [3.0 + ((k * 5) % 7) for k in range(n)],
            "ocf_x": [4.0 + ((k * 3) % 5) for k in range(n)],
            "at_x": [100.0 + 2 * k for k in range(n)],
        }
        got = _get(earnings_variability(_panel({"A": d})), "A", "earnings_variability")
        # Gate count > 60: index 59 null; index 60 uses lags 0,12,..,48 -> five values each.
        assert got[59] is None
        idx = [60, 48, 36, 24, 12]
        roa = [d["ni_x"][i] / d["at_x"][i - 12] for i in idx]  # type: ignore[operator]
        croa = [d["ocf_x"][i] / d["at_x"][i - 12] for i in idx]  # type: ignore[operator]
        assert got[60] == pytest.approx(statistics.stdev(roa) / statistics.stdev(croa))


@pytest.mark.unit
class TestProjectionExpressions:
    def test_roe_and_g_exps(self) -> None:
        df = pl.DataFrame({"__g0": [0.10], "__roe0": [0.20]})
        out = df.with_columns(roe_and_g_exps(1, 0.02, 0.5, 0.05, 0.4))
        assert out["__g1"][0] == pytest.approx(0.02 + 0.5 * 0.10)
        assert out["__roe1"][0] == pytest.approx(0.05 + 0.4 * 0.20)

    def test_roe_and_g_exps_null_propagates(self) -> None:
        df = pl.DataFrame(
            {"__g0": [None], "__roe0": [0.2]}, schema={"__g0": pl.Float64, "__roe0": pl.Float64}
        )
        out = df.with_columns(roe_and_g_exps(1, 0.02, 0.5, 0.05, 0.4))
        assert out["__g1"][0] is None
        assert out["__roe1"][0] == pytest.approx(0.13)

    def test_be_and_cd_exps(self) -> None:
        df = pl.DataFrame({"__be0": [100.0], "__g1": [0.07], "__roe1": [0.10]})
        out = df.with_columns(be_and_cd_exps(1))
        assert out["__be1"][0] == pytest.approx(100.0 * 1.07)
        assert out["__cd1"][0] == pytest.approx(100.0 * (0.10 - 0.07))

    def test_be_and_cd_exps_chains_from_previous_period(self) -> None:
        df = pl.DataFrame({"__be1": [107.0], "__g2": [-0.5], "__roe2": [0.1]})
        out = df.with_columns(be_and_cd_exps(2))
        assert out["__be2"][0] == pytest.approx(107.0 * 0.5)
        assert out["__cd2"][0] == pytest.approx(107.0 * 0.6)


@pytest.mark.unit
class TestEquityDurationCd:
    PARAMS = {"horizon": 2, "r": 0.1, "roe_mean": 0.1, "roe_ar1": 0.5, "g_mean": 0.04, "g_ar1": 0.5}

    @staticmethod
    def _firm(
        be0: float = 100.0,
        be12: float = 120.0,
        sale0: float = 200.0,
        sale12: float = 220.0,
        ni12: float = 10.0,
        n: int = 14,
    ) -> dict[str, list[float | None]]:
        be: list[float | None] = [1000.0] * n
        sale: list[float | None] = [1000.0] * n
        ni: list[float | None] = [1.0] * n
        be[0], be[12] = be0, be12
        sale[0], sale[12] = sale0, sale12
        ni[12] = ni12
        return {"be_x": be, "sale_x": sale, "ni_x": ni}

    def test_hand_computed_durations(self) -> None:
        out = equity_duration_cd(_panel({"A": self._firm()}), **self.PARAMS)
        # roe0 = ni/lag12(be) = 10/100 = 0.10; g0 = 220/200 - 1 = 0.10; be0 = 120.
        # roe_c = 0.1*(1-0.5) = 0.05, g_c = 0.04*(1-0.5) = 0.02.
        # t=1: roe1 = 0.05+0.5*0.10 = 0.10; g1 = 0.02+0.5*0.10 = 0.07; be1 = 120*1.07 = 128.4; cd1 = 120*(0.10-0.07) = 3.6
        # t=2: roe2 = 0.05+0.5*0.10 = 0.10; g2 = 0.02+0.5*0.07 = 0.055; be2 = 128.4*1.055; cd2 = 128.4*(0.10-0.055) = 5.778
        ed_cd = 3.6 / 1.1 + 5.778 / 1.21
        ed_cd_w = 1 * 3.6 / 1.1 + 2 * 5.778 / 1.21
        row = out.collect().filter(pl.col("count") == 13)
        assert row["ed_cd"][0] == pytest.approx(ed_cd)
        assert row["ed_cd_w"][0] == pytest.approx(ed_cd_w)
        assert row["ed_constant"][0] == pytest.approx(2 + 1.1 / 0.1)
        assert row["ed_err"][0] == 0.0

    def test_count_gate_and_constant_everywhere(self) -> None:
        out = equity_duration_cd(_panel({"A": self._firm()}), **self.PARAMS)
        df = out.collect()
        # count <= 12 -> roe0, g0 null -> ed_cd null; count 13 (index 12) is the first set.
        assert df["ed_cd"].to_list()[:12] == [None] * 12
        assert df["ed_cd"][12] is not None
        assert df["ed_constant"].to_list() == [pytest.approx(13.0)] * 14
        assert df["ed_err"].to_list()[:12] == [0.0] * 12

    def test_helper_columns_dropped(self) -> None:
        out = equity_duration_cd(_panel({"A": self._firm()}), **self.PARAMS).collect()
        assert not [c for c in out.columns if c.startswith("__")]
        assert {"ed_cd", "ed_cd_w", "ed_constant", "ed_err"} <= set(out.columns)

    def test_small_lagged_denominators_give_null(self) -> None:
        # lag12(be_x) = 1 is not > 1 -> roe0 null -> cd1, cd2 null.
        out = equity_duration_cd(_panel({"A": self._firm(be0=1.0)}), **self.PARAMS).collect()
        assert out["ed_cd"][12] is None
        assert out["ed_cd_w"][12] is None
        # lag12(sale_x) = 1 is not > 1 -> g0 null -> everything downstream null.
        out = equity_duration_cd(_panel({"A": self._firm(sale0=1.0)}), **self.PARAMS).collect()
        assert out["ed_cd"][12] is None
        assert out["ed_err"][12] == 0.0

    def test_negative_book_equity_sets_error_flag(self) -> None:
        # g_ar1=1, g_mean=0: g1 = g0 = 400/100 ... use sale falling: sale12=-200, sale0=100 -> g0=-3.
        params = {**self.PARAMS, "g_mean": 0.0, "g_ar1": 1.0}
        out = equity_duration_cd(
            _panel({"A": self._firm(sale0=100.0, sale12=-200.0)}), **params
        ).collect()
        # be1 = be0*(1+g1) = 120*(1-3) = -240 < 0
        assert out["ed_err"][12] == 1.0

    def test_firms_are_separate(self) -> None:
        a = self._firm()
        b = self._firm(be0=50.0, be12=60.0, sale0=100.0, sale12=130.0, ni12=9.0)
        out = equity_duration_cd(_panel({"A": a, "B": b}), **self.PARAMS)
        # Firm B: roe0 = 9/50 = 0.18; g0 = 0.30; be0 = 60.
        # roe1 = 0.05+0.09 = 0.14; g1 = 0.02+0.15 = 0.17; cd1 = 60*(0.14-0.17) = -1.8; be1 = 70.2
        # roe2 = 0.05+0.07 = 0.12; g2 = 0.02+0.085 = 0.105; cd2 = 70.2*(0.12-0.105) = 1.053
        exp_b = -1.8 / 1.1 + 1.053 / 1.21
        assert _get(out, "B", "ed_cd")[12] == pytest.approx(exp_b)
        exp_a = 3.6 / 1.1 + 5.778 / 1.21
        assert _get(out, "A", "ed_cd")[12] == pytest.approx(exp_a)
        # Firm B's early rows stay null (count <= 12), not filled from firm A's tail.
        assert _get(out, "B", "ed_cd")[:12] == [None] * 12

    def test_default_horizon_runs(self) -> None:
        out = equity_duration_cd(_panel({"A": self._firm()})).collect()
        # horizon=10, r=0.12 -> ed_constant = 10 + 1.12/0.12
        assert out["ed_constant"][0] == pytest.approx(10 + 1.12 / 0.12)
        assert out["ed_cd"][12] is not None
