"""Pairwise matrix smoke tests for ``portfolios()``.

Covers the cartesian of ``(bps, source, signals, cmp_key, wins_ret,
daily_pf, ind_pf, excntry)`` via a handwritten pairwise table sized to
exercise every axis level at least twice while staying under 20 cases.

Each case writes a synthetic country file, calls ``portfolios()``, and
asserts that the returned dict has the expected keys plus the expected
column set on each frame.
"""

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
)

# ---------------------------------------------------------------------------
# Expected schemas
# ---------------------------------------------------------------------------

PF_RETURNS_COLS = {
    "pf",
    "eom",
    "characteristic",
    "n",
    "signal",
    "ret_ew",
    "ret_vw",
    "ret_vw_cap",
    "excntry",
}
PF_DAILY_COLS = {
    "pf",
    "date",
    "characteristic",
    "n",
    "ret_ew",
    "ret_vw",
    "ret_vw_cap",
    "excntry",
}
GICS_RETURNS_COLS = {"gics", "eom", "n", "ret_ew", "ret_vw", "ret_vw_cap", "excntry"}
FF49_RETURNS_COLS = {"ff49", "eom", "n", "ret_ew", "ret_vw", "ret_vw_cap", "excntry"}
GICS_DAILY_COLS = {"gics", "date", "n", "ret_ew", "ret_vw", "ret_vw_cap", "excntry"}
FF49_DAILY_COLS = {"ff49", "date", "n", "ret_ew", "ret_vw", "ret_vw_cap", "excntry"}
CMP_COLS = {
    "size_grp",
    "eom",
    "characteristic",
    "n_stocks",
    "ret_weighted",
    "signal_weighted",
    "excntry",
}


# ---------------------------------------------------------------------------
# Pairwise matrix
# ---------------------------------------------------------------------------
#
# Axes covered (each level appears in >= 2 cases):
#   bps:      non_mc, nyse
#   source:   ["CRSP"], ["COMPUSTAT"], ["CRSP", "COMPUSTAT"]
#   signals:  True (1-char only — multi-char hits a duplicate-w bug), False
#   cmp_key:  True (USA only), False
#   wins_ret: True, False
#   daily_pf: True, False
#   ind_pf:   True, False
#   excntry:  USA, FRA
#
# 14 cases, hand-tuned to cover every level at least twice and to mix axes
# pairwise rather than separately.

_MATRIX_CASES: list[dict] = [
    {
        "id": "non_mc-both-noSig-noCmp-winsT-dailyF-indT-USA",
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": True,
        "daily_pf": False,
        "ind_pf": True,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "nyse-both-noSig-noCmp-winsT-dailyF-indT-USA",
        "bps": "nyse",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": True,
        "daily_pf": False,
        "ind_pf": True,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-CRSP-noSig-noCmp-winsT-dailyF-indF-FRA",
        "bps": "non_mc",
        "source": ["CRSP"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": True,
        "daily_pf": False,
        "ind_pf": False,
        "excntry": "FRA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-COMP-noSig-noCmp-winsT-dailyF-indT-USA",
        "bps": "non_mc",
        "source": ["COMPUSTAT"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": True,
        "daily_pf": False,
        "ind_pf": True,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-both-Sig-noCmp-winsT-dailyF-indF-USA",
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": True,
        "cmp_key": False,
        "wins_ret": True,
        "daily_pf": False,
        "ind_pf": False,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-both-noSig-Cmp-winsT-dailyT-indT-USA",
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": False,
        "cmp_key": True,
        "wins_ret": True,
        "daily_pf": True,
        "ind_pf": True,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-both-noSig-noCmp-winsF-dailyF-indT-USA",
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": False,
        "daily_pf": False,
        "ind_pf": True,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-both-noSig-noCmp-winsT-dailyT-indF-USA",
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": True,
        "daily_pf": True,
        "ind_pf": False,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "nyse-CRSP-noSig-noCmp-winsF-dailyT-indT-USA",
        "bps": "nyse",
        "source": ["CRSP"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": False,
        "daily_pf": True,
        "ind_pf": True,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "nyse-COMP-noSig-noCmp-winsT-dailyT-indF-FRA",
        "bps": "nyse",
        "source": ["COMPUSTAT"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": True,
        "daily_pf": True,
        "ind_pf": False,
        "excntry": "FRA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-CRSP-Sig-noCmp-winsF-dailyT-indT-USA",
        "bps": "non_mc",
        "source": ["CRSP"],
        "signals": True,
        "cmp_key": False,
        "wins_ret": False,
        "daily_pf": True,
        "ind_pf": True,
        "excntry": "USA",
        "n_chars": 1,
    },
    {
        "id": "nyse-both-noSig-Cmp-winsF-dailyF-indF-USA",
        "bps": "nyse",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": False,
        "cmp_key": True,
        "wins_ret": False,
        "daily_pf": False,
        "ind_pf": False,
        "excntry": "USA",
        "n_chars": 2,
    },
    {
        "id": "nyse-COMP-noSig-noCmp-winsF-dailyF-indT-FRA",
        "bps": "nyse",
        "source": ["COMPUSTAT"],
        "signals": False,
        "cmp_key": False,
        "wins_ret": False,
        "daily_pf": False,
        "ind_pf": True,
        "excntry": "FRA",
        "n_chars": 2,
    },
    {
        "id": "non_mc-both-noSig-Cmp-winsT-dailyF-indF-USA",
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "signals": False,
        "cmp_key": True,
        "wins_ret": True,
        "daily_pf": False,
        "ind_pf": False,
        "excntry": "USA",
        "n_chars": 2,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_keys(config: dict) -> set[str]:
    """Compute the expected ``portfolios()`` output keys for a case."""
    keys = {"pf_returns"}
    if config["daily_pf"]:
        keys.add("pf_daily")
    if config["ind_pf"]:
        keys.add("gics_returns")
        if config["excntry"].lower() == "usa":
            keys.add("ff49_returns")
        if config["daily_pf"]:
            keys.add("gics_daily")
            if config["excntry"].lower() == "usa":
                keys.add("ff49_daily")
    if config["cmp_key"]:
        keys.add("cmp")
    if config["signals"]:
        keys.add("signals")
    return keys


def _write_country_inputs(
    tmp_path: Path,
    excntry: str,
    chars: list[str],
    daily: bool,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Write characteristics (+ optional daily) parquets, return cutoff frames."""
    char_df = make_country_characteristics(
        excntry=excntry, chars=chars, n_ids=80, n_months=6, seed=seed
    )
    char_dir = tmp_path / "characteristics"
    char_dir.mkdir(parents=True, exist_ok=True)
    char_df.write_parquet(char_dir / f"{excntry}.parquet")

    if daily:
        daily_dir = tmp_path / "return_data" / "daily_rets_by_country"
        daily_dir.mkdir(parents=True, exist_ok=True)
        daily_df = make_daily_returns(char_df, seed=seed + 1)
        daily_df.write_parquet(daily_dir / f"{excntry}.parquet")

    eoms = char_df["eom"].unique().sort().to_list()
    return make_cutoffs(eoms)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPortfoliosMatrix:
    """Pairwise matrix smoke test for ``portfolios()``."""

    @pytest.mark.parametrize("config", _MATRIX_CASES, ids=lambda c: c["id"])
    def test_portfolios_runs_and_returns_expected_keys(
        self, config: dict, tmp_path: Path, seed: int
    ) -> None:
        chars = SYNTHETIC_CHARS[: config["n_chars"]]
        nyse_cutoffs, ret_cutoffs, ret_cutoffs_daily = _write_country_inputs(
            tmp_path,
            excntry=config["excntry"],
            chars=chars,
            daily=config["daily_pf"],
            seed=seed,
        )

        out = portfolios(
            data_path=str(tmp_path),
            excntry=config["excntry"],
            chars=chars,
            pfs=3,
            bps=config["bps"],
            bp_min_n=2,
            nyse_size_cutoffs=nyse_cutoffs,
            source=config["source"],
            wins_ret=config["wins_ret"],
            cmp_key=config["cmp_key"],
            signals=config["signals"],
            signals_standardize=False,
            signals_w="vw_cap",
            daily_pf=config["daily_pf"],
            ind_pf=config["ind_pf"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
        )

        # Key set matches expectations for this combo.
        assert set(out.keys()) == _expected_keys(config), (
            f"[{config['id']}] keys mismatch: got {sorted(out.keys())}, "
            f"expected {sorted(_expected_keys(config))}"
        )

        # Per-frame schema checks.
        assert set(out["pf_returns"].columns) == PF_RETURNS_COLS
        assert out["pf_returns"].height > 0

        if config["daily_pf"]:
            assert set(out["pf_daily"].columns) == PF_DAILY_COLS
            assert out["pf_daily"].height > 0

        if config["ind_pf"]:
            assert set(out["gics_returns"].columns) == GICS_RETURNS_COLS
            if config["excntry"].lower() == "usa":
                assert set(out["ff49_returns"].columns) == FF49_RETURNS_COLS
            if config["daily_pf"]:
                assert set(out["gics_daily"].columns) == GICS_DAILY_COLS
                if config["excntry"].lower() == "usa":
                    assert set(out["ff49_daily"].columns) == FF49_DAILY_COLS

        if config["cmp_key"]:
            assert set(out["cmp"].columns) == CMP_COLS
            assert out["cmp"].height > 0
