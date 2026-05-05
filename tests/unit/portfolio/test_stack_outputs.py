"""Tests for stacking ``portfolios()`` per-country results into combined frames."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jkp.data.portfolio import portfolios
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    make_country_characteristics,
    make_cutoffs,
    make_daily_returns,
    write_synthetic_country,
)

COMMON_KWARGS = {
    "pfs": 3,
    "bps": "non_mc",
    "bp_min_n": 5,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "signals": False,
    "signals_standardize": False,
    "signals_w": "vw_cap",
    "daily_pf": False,
    "ind_pf": True,
    "cmp_key": False,
}


def _run(
    data_path: Path,
    excntry: str,
    chars: list[str],
    eoms,
    **overrides,
) -> dict:
    nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)
    kwargs = {**COMMON_KWARGS, **overrides}
    return portfolios(
        data_path=str(data_path),
        excntry=excntry,
        chars=chars,
        nyse_size_cutoffs=nyse_size_cutoffs,
        ret_cutoffs=ret_cutoffs,
        ret_cutoffs_daily=ret_cutoffs_daily,
        **kwargs,
    )


class TestStackOutputs:
    """Verify per-country ``portfolios()`` results stack cleanly across countries."""

    def test_stack_two_countries_pf_returns(self, tmp_path: Path) -> None:
        chars = SYNTHETIC_CHARS[:2]
        countries = ["USA", "FRA"]
        results = []
        for ex in countries:
            country_dir = tmp_path / ex
            char_df, _ = write_synthetic_country(
                country_dir, ex, chars, seed=42 + countries.index(ex), n_ids=60, n_months=6
            )
            eoms = char_df["eom"].unique().sort().to_list()
            out = _run(country_dir, ex, chars, eoms)
            results.append(out["pf_returns"])

        stacked = pl.concat(results)
        # Schema preserved.
        for r in results:
            assert set(r.columns) == set(stacked.columns)
        # Unique row keys (no collisions across countries).
        keys = stacked.select(["excntry", "characteristic", "pf", "eom"])
        assert keys.n_unique() == stacked.height
        # Both countries present.
        assert set(stacked["excntry"].unique().to_list()) == {"USA", "FRA"}

    def test_stack_single_country(self, tmp_path: Path) -> None:
        chars = SYNTHETIC_CHARS[:1]
        ex = "USA"
        country_dir = tmp_path / ex
        char_df, _ = write_synthetic_country(country_dir, ex, chars, n_ids=60, n_months=4)
        eoms = char_df["eom"].unique().sort().to_list()
        out = _run(country_dir, ex, chars, eoms)
        df = out["pf_returns"]
        stacked = pl.concat([df])
        assert stacked.columns == df.columns
        assert stacked.schema == df.schema
        assert stacked.height == df.height

    def test_ff49_only_for_usa_in_stacked(self, tmp_path: Path) -> None:
        chars = SYNTHETIC_CHARS[:1]
        countries = ["USA", "FRA"]
        ff49_pieces = []
        for ex in countries:
            country_dir = tmp_path / ex
            char_df, _ = write_synthetic_country(
                country_dir, ex, chars, seed=42 + countries.index(ex), n_ids=60, n_months=4
            )
            eoms = char_df["eom"].unique().sort().to_list()
            out = _run(country_dir, ex, chars, eoms)
            if "ff49_returns" in out:
                ff49_pieces.append(out["ff49_returns"])

        # Only USA produced ff49_returns.
        assert len(ff49_pieces) == 1
        stacked = pl.concat(ff49_pieces)
        assert set(stacked["excntry"].unique().to_list()) == {"USA"}

    def test_cmp_only_for_usa_in_stacked(self, tmp_path: Path) -> None:
        chars = SYNTHETIC_CHARS[:1]
        countries = ["USA", "FRA"]
        cmp_pieces = []
        for ex in countries:
            country_dir = tmp_path / ex
            char_df, _ = write_synthetic_country(
                country_dir, ex, chars, seed=42 + countries.index(ex), n_ids=60, n_months=4
            )
            eoms = char_df["eom"].unique().sort().to_list()
            cmp_key = ex.lower() == "usa"
            out = _run(country_dir, ex, chars, eoms, cmp_key=cmp_key)
            if "cmp" in out:
                cmp_pieces.append(out["cmp"])

        assert len(cmp_pieces) == 1
        stacked = pl.concat(cmp_pieces)
        assert set(stacked["excntry"].unique().to_list()) == {"USA"}

    def test_empty_country_yields_empty_or_omitted(self, tmp_path: Path) -> None:
        chars = SYNTHETIC_CHARS[:1]
        ex = "USA"
        country_dir = tmp_path / ex
        char_df, _ = write_synthetic_country(country_dir, ex, chars, n_ids=20, n_months=3)
        eoms = char_df["eom"].unique().sort().to_list()
        # Force every-eom bp_n below threshold so no char survives.
        out = _run(country_dir, ex, chars, eoms, bp_min_n=10**6)

        if "pf_returns" not in out:
            # Key was dropped — acceptable behavior.
            return
        pf_returns = out["pf_returns"]
        assert pf_returns.height == 0, (
            f"expected empty pf_returns when bp_min_n is unreachable, got {pf_returns.height} rows"
        )
        # An empty frame must still concat cleanly with another country's frame.
        ex2 = "FRA"
        ex2_dir = tmp_path / ex2
        char_df2, _ = write_synthetic_country(ex2_dir, ex2, chars, seed=99, n_ids=60, n_months=3)
        eoms2 = char_df2["eom"].unique().sort().to_list()
        out2 = _run(ex2_dir, ex2, chars, eoms2)
        # Align columns then concat.
        common = [c for c in out2["pf_returns"].columns if c in pf_returns.columns]
        stacked = pl.concat(
            [pf_returns.select(common), out2["pf_returns"].select(common)],
            how="vertical_relaxed",
        )
        assert stacked.height == out2["pf_returns"].height


# Silence unused-import warnings for fixtures referenced via conftest indirection.
_ = (make_country_characteristics, make_daily_returns, pytest)
