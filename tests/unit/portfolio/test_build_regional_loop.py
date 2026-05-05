"""Unit tests for ``_build_regional_loop()``."""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.portfolio import _build_regional_loop


def _month_ends(n_months: int, start_year: int = 2018, start_month: int = 1) -> list[date]:
    out: list[date] = []
    for i in range(n_months):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        out.append(date(year, month, calendar.monthrange(year, month)[1]))
    return out


def _sample_lms(n_countries: int = 4, n_months: int = 12, n_chars: int = 2) -> pl.DataFrame:
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


_OUTPUT_COLS = [
    "region",
    "characteristic",
    "direction",
    "eom",
    "n_countries",
    "ret_ew",
    "ret_vw",
    "ret_vw_cap",
    "mkt_vw_exc",
]


class TestBuildRegionalLoop:
    """Tests for ``_build_regional_loop()``.

    Regression coverage: ``iter_rows(named=True)`` returns Python ``list``
    for list columns, but ``regional_data`` calls ``.implode()`` which only
    exists on ``pl.Series``. The helper must wrap ``country_codes`` in a
    ``pl.Series`` before passing it through.
    """

    @staticmethod
    def _sample_inputs(n_countries: int = 4, n_months: int = 12, n_chars: int = 2):
        lms = _sample_lms(n_countries=n_countries, n_months=n_months, n_chars=n_chars)
        countries = lms["excntry"].unique().to_list()
        mkt = _sample_market(countries, _month_ends(n_months))
        regions = pl.DataFrame(
            {
                "name": ["all", "subset"],
                "country_codes": [countries, countries[:2]],
                "countries_min": [1, 1],
            }
        )
        return lms, mkt, regions, _OUTPUT_COLS

    def test_runs_with_list_country_codes(self):
        """Regression: list-typed country_codes must be wrapped in pl.Series for .implode()."""
        lms, mkt, regions, output_cols = self._sample_inputs()
        # Should NOT raise AttributeError("'list' object has no attribute 'implode'").
        result = _build_regional_loop(
            data=lms,
            mkt=mkt,
            regions=regions,
            date_col="eom",
            char_col="characteristic",
            output_cols=output_cols,
            weighting="market_cap",
            periods_min=1,
            stocks_min=1,
        )
        assert result.height > 0
        assert set(result.columns) == set(output_cols)

    def test_concatenates_one_block_per_region(self):
        lms, mkt, regions, output_cols = self._sample_inputs()
        result = _build_regional_loop(
            data=lms,
            mkt=mkt,
            regions=regions,
            date_col="eom",
            char_col="characteristic",
            output_cols=output_cols,
            weighting="market_cap",
            periods_min=1,
            stocks_min=1,
        )
        assert set(result["region"].unique().to_list()) == {"all", "subset"}

    def test_uses_per_region_countries_min(self):
        """Each region's ``countries_min`` must be applied to its own block."""
        lms, mkt, _, output_cols = self._sample_inputs(n_countries=3)
        regions = pl.DataFrame(
            {
                "name": ["pass", "fail"],
                "country_codes": [
                    lms["excntry"].unique().to_list(),
                    lms["excntry"].unique().to_list(),
                ],
                "countries_min": [1, 99],  # second region must filter to empty
            }
        )
        result = _build_regional_loop(
            data=lms,
            mkt=mkt,
            regions=regions,
            date_col="eom",
            char_col="characteristic",
            output_cols=output_cols,
            weighting="market_cap",
            periods_min=1,
            stocks_min=1,
        )
        assert "pass" in result["region"].unique().to_list()
        assert "fail" not in result["region"].unique().to_list()


class TestBuildRegionalLoopFiveRegions:
    """Five-region scenario tests mirroring the production region set."""

    @staticmethod
    def _five_region_inputs():
        # 8 countries: 2 USA-like, 2 other developed, 2 emerging, 2 frontier.
        countries = ["USA", "CAN", "GBR", "JPN", "BRA", "IND", "VNM", "EGY"]
        n_months = 4
        eoms = _month_ends(n_months)
        chars = ["factor_0", "factor_1"]

        rng = np.random.default_rng(7)
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
        lms = pl.DataFrame(rows)
        mkt = _sample_market(countries, eoms)

        developed = ["USA", "CAN", "GBR", "JPN"]
        emerging = ["BRA", "IND"]
        frontier = ["VNM", "EGY"]
        world = countries
        world_ex_us = [c for c in countries if c != "USA"]

        regions = pl.DataFrame(
            {
                "name": ["developed", "emerging", "frontier", "world", "world_ex_us"],
                "country_codes": [developed, emerging, frontier, world, world_ex_us],
                "countries_min": [1, 1, 1, 1, 1],
            }
        )
        return lms, mkt, regions

    def test_outputs_one_block_per_region(self):
        lms, mkt, regions = self._five_region_inputs()
        result = _build_regional_loop(
            data=lms,
            mkt=mkt,
            regions=regions,
            date_col="eom",
            char_col="characteristic",
            output_cols=_OUTPUT_COLS,
            weighting="market_cap",
            periods_min=1,
            stocks_min=1,
        )
        assert set(result["region"].unique().to_list()) == {
            "developed",
            "emerging",
            "frontier",
            "world",
            "world_ex_us",
        }
        # Each region produces non-zero rows.
        for name in ["developed", "emerging", "frontier", "world", "world_ex_us"]:
            assert result.filter(pl.col("region") == name).height > 0

    def test_region_column_populated(self):
        lms, mkt, regions = self._five_region_inputs()
        result = _build_regional_loop(
            data=lms,
            mkt=mkt,
            regions=regions,
            date_col="eom",
            char_col="characteristic",
            output_cols=_OUTPUT_COLS,
            weighting="market_cap",
            periods_min=1,
            stocks_min=1,
        )
        # Every row's region must be one of the 5 region names; no nulls.
        assert result["region"].null_count() == 0
        assert set(result["region"].unique().to_list()).issubset(
            {"developed", "emerging", "frontier", "world", "world_ex_us"}
        )

    def test_per_region_countries_min_respected(self):
        """countries_min varies per region; each region's filter applies to itself."""
        lms, mkt, _ = self._five_region_inputs()
        # frontier has 2 countries, so countries_min=3 drops it; others survive.
        regions = pl.DataFrame(
            {
                "name": ["developed", "emerging", "frontier", "world", "world_ex_us"],
                "country_codes": [
                    ["USA", "CAN", "GBR", "JPN"],
                    ["BRA", "IND"],
                    ["VNM", "EGY"],
                    ["USA", "CAN", "GBR", "JPN", "BRA", "IND", "VNM", "EGY"],
                    ["CAN", "GBR", "JPN", "BRA", "IND", "VNM", "EGY"],
                ],
                "countries_min": [1, 1, 3, 1, 1],
            }
        )
        result = _build_regional_loop(
            data=lms,
            mkt=mkt,
            regions=regions,
            date_col="eom",
            char_col="characteristic",
            output_cols=_OUTPUT_COLS,
            weighting="market_cap",
            periods_min=1,
            stocks_min=1,
        )
        present = set(result["region"].unique().to_list())
        assert "frontier" not in present
        assert {"developed", "emerging", "world", "world_ex_us"}.issubset(present)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
