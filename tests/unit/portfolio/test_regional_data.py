"""Unit tests for ``regional_data`` in ``portfolio.py``."""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.portfolio import regional_data

TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def _month_ends(n_months: int, start_year: int = 2020, start_month: int = 1) -> list[date]:
    """Return ``n_months`` consecutive month-end ``date`` values."""
    out: list[date] = []
    for i in range(n_months):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        out.append(date(year, month, calendar.monthrange(year, month)[1]))
    return out


# =============================================================================
# TestRegionalData (migrated from tests/unit/test_portfolio.py)
# =============================================================================


class TestRegionalData:
    """Tests for ``regional_data()``."""

    @staticmethod
    def _sample_lms(n_countries: int = 5, n_months: int = 24, n_chars: int = 3) -> pl.DataFrame:
        """Build a minimal LMS-shaped frame for regional_data tests."""
        rng = np.random.default_rng(99)
        countries = [f"C{i:02d}" for i in range(n_countries)]
        eoms = _month_ends(n_months)
        chars = [f"factor_{j}" for j in range(n_chars)]
        rows = []
        for c in countries:
            for eom in eoms:
                for ch in chars:
                    rows.append(
                        {
                            "excntry": c,
                            "characteristic": ch,
                            "direction": 1,
                            "eom": eom,
                            "n_stocks": int(rng.integers(5, 50)),
                            "n_stocks_min": int(rng.integers(5, 20)),
                            "signal": float(rng.normal()),
                            "ret_ew": float(rng.normal(0, 0.05)),
                            "ret_vw": float(rng.normal(0, 0.05)),
                            "ret_vw_cap": float(rng.normal(0, 0.05)),
                        }
                    )
        return pl.DataFrame(rows)

    @staticmethod
    def _sample_market(countries: list[str], eoms: list[date]) -> pl.DataFrame:
        rng = np.random.default_rng(100)
        rows = []
        for c in countries:
            for eom in eoms:
                rows.append(
                    {
                        "excntry": c,
                        "eom": eom,
                        "mkt_vw_exc": float(rng.normal(0, 0.04)),
                        "me_lag1": float(np.exp(rng.normal(10, 1))),
                        "stocks": int(rng.integers(50, 500)),
                    }
                )
        return pl.DataFrame(rows)

    def test_n_countries_filter_respected(self):
        lms = self._sample_lms(n_countries=3, n_months=24, n_chars=2)
        countries = lms["excntry"].unique()
        mkt = self._sample_market(countries.to_list(), _month_ends(24))

        result = regional_data(
            data=lms,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=countries,
            weighting="market_cap",
            countries_min=4,  # Higher than the 3 we have
            periods_min=1,
            stocks_min=1,
        )
        # With only 3 countries, nothing passes the countries_min=4 filter
        assert result.height == 0

    def test_periods_min_filter(self):
        lms = self._sample_lms(n_countries=3, n_months=6, n_chars=1)
        countries = lms["excntry"].unique()
        mkt = self._sample_market(countries.to_list(), _month_ends(6))

        result = regional_data(
            data=lms,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=countries,
            weighting="market_cap",
            countries_min=1,
            periods_min=100,  # More months than exist
            stocks_min=1,
        )
        assert result.height == 0

    @pytest.mark.parametrize("weighting", ["market_cap", "ew", "stocks"])
    def test_weighting_modes_run_without_error(self, weighting: str):
        lms = self._sample_lms(n_countries=3, n_months=12, n_chars=1)
        countries = lms["excntry"].unique()
        mkt = self._sample_market(countries.to_list(), _month_ends(12))

        result = regional_data(
            data=lms,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=countries,
            weighting=weighting,
            countries_min=1,
            periods_min=1,
            stocks_min=1,
        )
        assert result.height > 0
        assert "ret_ew" in result.columns


# =============================================================================
# Hand-built fixtures for parity / filter tests
# =============================================================================


_COUNTRIES_3 = ["AAA", "BBB", "CCC"]
_DATES_3 = _month_ends(3)
_CHAR = "f1"


def _build_data(
    *,
    ret_ew_by_country: dict[str, list[float]] | None = None,
    ret_vw_by_country: dict[str, list[float]] | None = None,
    ret_vw_cap_by_country: dict[str, list[float]] | None = None,
    n_stocks_min_by_country: dict[str, list[int]] | None = None,
    countries: list[str] = _COUNTRIES_3,
    dates: list[date] = _DATES_3,
    char: str = _CHAR,
) -> pl.DataFrame:
    """Build a per-(country, date) characteristic frame with explicit values."""
    rows = []
    for c in countries:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "excntry": c,
                    "characteristic": char,
                    "direction": 1,
                    "eom": d,
                    "n_stocks_min": (
                        n_stocks_min_by_country[c][i] if n_stocks_min_by_country else 10
                    ),
                    "ret_ew": (ret_ew_by_country or {}).get(c, [0.01, 0.02, 0.03])[i],
                    "ret_vw": (ret_vw_by_country or {}).get(c, [0.01, 0.02, 0.03])[i],
                    "ret_vw_cap": (ret_vw_cap_by_country or {}).get(c, [0.01, 0.02, 0.03])[i],
                }
            )
    return pl.DataFrame(rows)


def _build_mkt(
    *,
    me_lag1_by_country: dict[str, list[float]] | None = None,
    stocks_by_country: dict[str, list[int]] | None = None,
    mkt_vw_exc_by_country: dict[str, list[float | None]] | None = None,
    countries: list[str] = _COUNTRIES_3,
    dates: list[date] = _DATES_3,
) -> pl.DataFrame:
    """Build a per-(country, date) market frame."""
    default_me = [1e9, 2e9, 3e9]
    default_stocks = [100, 200, 300]
    default_mkt = [0.01, 0.02, 0.03]
    rows = []
    for c in countries:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "excntry": c,
                    "eom": d,
                    "mkt_vw_exc": (mkt_vw_exc_by_country or {}).get(c, default_mkt)[i],
                    "me_lag1": (me_lag1_by_country or {}).get(c, default_me)[i],
                    "stocks": (stocks_by_country or {}).get(c, default_stocks)[i],
                }
            )
    return pl.DataFrame(rows)


def _expected_weighted_mean(
    values_by_country: dict[str, list[float]],
    weights_by_country: dict[str, list[float]],
    dates: list[date],
) -> dict[date, float]:
    """Compute Σ v·w / Σ w per date across countries."""
    out: dict[date, float] = {}
    for i, d in enumerate(dates):
        num = sum(values_by_country[c][i] * weights_by_country[c][i] for c in values_by_country)
        den = sum(weights_by_country[c][i] for c in values_by_country)
        out[d] = num / den
    return out


def _result_by_date(result: pl.DataFrame, col: str) -> dict[date, float]:
    """Map ``eom -> col`` value, sorted by eom."""
    return {row["eom"]: row[col] for row in result.sort("eom").iter_rows(named=True)}


# =============================================================================
# TestRegionalDataWeightingParity
# =============================================================================


class TestRegionalDataWeightingParity:
    """Verify ``regional_data`` matches hand-computed weighted means per mode."""

    def test_weighting_ew_matches_simple_mean(self):
        ret_ew = {
            "AAA": [0.01, 0.02, 0.03],
            "BBB": [0.04, -0.01, 0.02],
            "CCC": [-0.02, 0.03, 0.05],
        }
        ret_vw = {
            "AAA": [0.011, 0.021, 0.031],
            "BBB": [0.041, -0.011, 0.021],
            "CCC": [-0.021, 0.031, 0.051],
        }
        ret_vw_cap = {
            "AAA": [0.012, 0.022, 0.032],
            "BBB": [0.042, -0.012, 0.022],
            "CCC": [-0.022, 0.032, 0.052],
        }
        data = _build_data(
            ret_ew_by_country=ret_ew,
            ret_vw_by_country=ret_vw,
            ret_vw_cap_by_country=ret_vw_cap,
        )
        mkt = _build_mkt()

        result = regional_data(
            data=data,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=pl.Series(_COUNTRIES_3),
            weighting="ew",
            countries_min=1,
            periods_min=1,
            stocks_min=1,
        )

        ones = {c: [1.0, 1.0, 1.0] for c in _COUNTRIES_3}
        for col, src in [("ret_ew", ret_ew), ("ret_vw", ret_vw), ("ret_vw_cap", ret_vw_cap)]:
            expected = _expected_weighted_mean(src, ones, _DATES_3)
            actual = _result_by_date(result, col)
            for d in _DATES_3:
                np.testing.assert_allclose(actual[d], expected[d], **TIGHT)

    def test_weighting_market_cap_matches_me_weighted_mean(self):
        ret_ew = {
            "AAA": [0.01, 0.02, 0.03],
            "BBB": [0.04, -0.01, 0.02],
            "CCC": [-0.02, 0.03, 0.05],
        }
        me = {
            "AAA": [1.0e9, 1.5e9, 2.0e9],
            "BBB": [3.0e9, 2.5e9, 4.0e9],
            "CCC": [0.5e9, 0.8e9, 1.2e9],
        }
        data = _build_data(ret_ew_by_country=ret_ew)
        mkt = _build_mkt(me_lag1_by_country=me)

        result = regional_data(
            data=data,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=pl.Series(_COUNTRIES_3),
            weighting="market_cap",
            countries_min=1,
            periods_min=1,
            stocks_min=1,
        )

        expected = _expected_weighted_mean(ret_ew, me, _DATES_3)
        actual = _result_by_date(result, "ret_ew")
        for d in _DATES_3:
            np.testing.assert_allclose(actual[d], expected[d], **TIGHT)

    def test_weighting_stocks_matches_stocks_weighted_mean(self):
        ret_ew = {
            "AAA": [0.01, 0.02, 0.03],
            "BBB": [0.04, -0.01, 0.02],
            "CCC": [-0.02, 0.03, 0.05],
        }
        stocks = {
            "AAA": [50, 60, 70],
            "BBB": [200, 220, 250],
            "CCC": [10, 15, 20],
        }
        data = _build_data(ret_ew_by_country=ret_ew)
        mkt = _build_mkt(stocks_by_country=stocks)

        result = regional_data(
            data=data,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=pl.Series(_COUNTRIES_3),
            weighting="stocks",
            countries_min=1,
            periods_min=1,
            stocks_min=1,
        )

        stocks_f = {c: [float(s) for s in v] for c, v in stocks.items()}
        expected = _expected_weighted_mean(ret_ew, stocks_f, _DATES_3)
        actual = _result_by_date(result, "ret_ew")
        for d in _DATES_3:
            np.testing.assert_allclose(actual[d], expected[d], **TIGHT)


# =============================================================================
# TestRegionalDataFilters
# =============================================================================


class TestRegionalDataFilters:
    """Verify ``regional_data`` row-filter behaviour."""

    def test_drops_rows_missing_mkt_vw_exc(self):
        # CCC's market return is null on date 1 -> CCC contributes only to dates 0, 2.
        mkt_vw = {
            "AAA": [0.01, 0.02, 0.03],
            "BBB": [0.02, 0.03, 0.04],
            "CCC": [0.03, None, 0.05],
        }
        data = _build_data()
        mkt = _build_mkt(mkt_vw_exc_by_country=mkt_vw)

        result = regional_data(
            data=data,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=pl.Series(_COUNTRIES_3),
            weighting="ew",
            countries_min=1,
            periods_min=1,
            stocks_min=1,
        )

        n_by_date = {row["eom"]: row["n_countries"] for row in result.iter_rows(named=True)}
        assert n_by_date[_DATES_3[0]] == 3
        assert n_by_date[_DATES_3[1]] == 2  # CCC dropped on this date
        assert n_by_date[_DATES_3[2]] == 3

    def test_excludes_countries_outside_set(self):
        data = _build_data()
        mkt = _build_mkt()

        # Restrict to AAA + BBB only.
        result = regional_data(
            data=data,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=pl.Series(["AAA", "BBB"]),
            weighting="ew",
            countries_min=1,
            periods_min=1,
            stocks_min=1,
        )

        assert result["n_countries"].max() == 2
        assert result["n_countries"].min() == 2

    def test_stocks_min_filters_per_country_rows(self):
        # CCC has n_stocks_min below threshold on date 0 -> drop that country-row only.
        n_stocks_min = {
            "AAA": [20, 20, 20],
            "BBB": [20, 20, 20],
            "CCC": [3, 20, 20],
        }
        data = _build_data(n_stocks_min_by_country=n_stocks_min)
        mkt = _build_mkt()

        result = regional_data(
            data=data,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=pl.Series(_COUNTRIES_3),
            weighting="ew",
            countries_min=1,
            periods_min=1,
            stocks_min=10,
        )

        n_by_date = {row["eom"]: row["n_countries"] for row in result.iter_rows(named=True)}
        assert n_by_date[_DATES_3[0]] == 2  # CCC dropped on date 0
        assert n_by_date[_DATES_3[1]] == 3
        assert n_by_date[_DATES_3[2]] == 3

    def test_periods_min_filters_chars(self):
        # Two chars: "f1" has 3 dates, "f2" has only 1 (force null mkt_vw_exc on 2 of 3).
        # Easier: build data with one char having only 1 date present.
        rows = []
        for c in _COUNTRIES_3:
            for d in _DATES_3:
                rows.append(
                    {
                        "excntry": c,
                        "characteristic": "f_long",
                        "direction": 1,
                        "eom": d,
                        "n_stocks_min": 10,
                        "ret_ew": 0.01,
                        "ret_vw": 0.01,
                        "ret_vw_cap": 0.01,
                    }
                )
        # f_short: only date 0
        for c in _COUNTRIES_3:
            rows.append(
                {
                    "excntry": c,
                    "characteristic": "f_short",
                    "direction": 1,
                    "eom": _DATES_3[0],
                    "n_stocks_min": 10,
                    "ret_ew": 0.01,
                    "ret_vw": 0.01,
                    "ret_vw_cap": 0.01,
                }
            )
        data = pl.DataFrame(rows)
        mkt = _build_mkt()

        result = regional_data(
            data=data,
            mkt=mkt,
            date_col="eom",
            char_col="characteristic",
            countries=pl.Series(_COUNTRIES_3),
            weighting="ew",
            countries_min=1,
            periods_min=2,  # f_short has only 1 date -> dropped
            stocks_min=1,
        )

        chars_present = set(result["characteristic"].unique().to_list())
        assert chars_present == {"f_long"}
