"""Tests for the winsorization branches of ``portfolios()`` (lines 330-370)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from jkp.data.portfolio import portfolios
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    make_country_characteristics,
    make_cutoffs,
    make_daily_returns,
)


def _write_inputs(
    data_root: Path,
    excntry: str,
    char_df: pl.DataFrame,
    daily_df: pl.DataFrame,
) -> None:
    char_dir = data_root / "characteristics"
    daily_dir = data_root / "return_data" / "daily_rets_by_country"
    char_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    char_df.write_parquet(char_dir / f"{excntry}.parquet")
    daily_df.write_parquet(daily_dir / f"{excntry}.parquet")


class TestWinsorization:
    """Cover the monthly + daily winsorization branches at lines 330-370."""

    def test_wins_ret_false_retains_extreme_returns(self, tmp_path: Path) -> None:
        """With ``wins_ret=False`` extreme Compustat returns propagate unclipped."""
        excntry = "SYN"
        data_root = tmp_path / "processed"
        char_df = make_country_characteristics(
            excntry=excntry, chars=SYNTHETIC_CHARS, n_ids=60, n_months=6, seed=11
        )
        # Inject an extreme +5.0 lead1m return on every Compustat row.
        char_df = char_df.with_columns(
            pl.when(pl.col("source_crsp") == 0)
            .then(pl.lit(5.0))
            .otherwise(pl.col("ret_exc_lead1m"))
            .alias("ret_exc_lead1m")
        )
        daily_df = make_daily_returns(char_df, seed=12)
        _write_inputs(data_root, excntry, char_df, daily_df)

        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, _ = make_cutoffs(eoms)

        result = portfolios(
            data_path=str(data_root),
            excntry=excntry,
            chars=[SYNTHETIC_CHARS[0]],
            pfs=3,
            bps="non_mc",
            bp_min_n=1,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=False,
            cmp_key=False,
            signals=False,
            daily_pf=False,
            ind_pf=False,
            ret_cutoffs=ret_cut,
            ret_cutoffs_daily=None,
        )

        pf_ret = result["pf_returns"]
        # 5.0 was injected on Compustat rows; with no winsor, EW aggregate must
        # exceed any plausible-clip threshold (cutoffs in make_cutoffs are 0.5).
        assert pf_ret["ret_ew"].max() > 1.0

    def test_wins_ret_true_clips_compustat_rows_only(self, tmp_path: Path) -> None:
        """``wins_ret=True`` must clip Compustat ret_exc_lead1m above ``p999``."""
        excntry = "SYN"
        data_root = tmp_path / "processed"
        char_df = make_country_characteristics(
            excntry=excntry, chars=SYNTHETIC_CHARS, n_ids=60, n_months=6, seed=21
        )
        # Inject +5.0 on both Compustat and CRSP rows.
        char_df = char_df.with_columns(pl.lit(5.0).alias("ret_exc_lead1m"))
        daily_df = make_daily_returns(char_df, seed=22)
        _write_inputs(data_root, excntry, char_df, daily_df)

        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, _ = make_cutoffs(eoms)
        # ret_cut already supplies p001=-0.5, p999=0.5 from make_cutoffs.

        kwargs = {
            "data_path": str(data_root),
            "excntry": excntry,
            "chars": [SYNTHETIC_CHARS[0]],
            "pfs": 3,
            "bps": "non_mc",
            "bp_min_n": 1,
            "nyse_size_cutoffs": nyse_cut,
            "source": ["CRSP", "COMPUSTAT"],
            "cmp_key": False,
            "signals": False,
            "daily_pf": False,
            "ind_pf": False,
            "ret_cutoffs": ret_cut,
            "ret_cutoffs_daily": None,
        }
        result_win = portfolios(wins_ret=True, **kwargs)
        result_no_win = portfolios(wins_ret=False, **kwargs)

        pf_win = result_win["pf_returns"].sort(["characteristic", "pf", "eom"])
        pf_no = result_no_win["pf_returns"].sort(["characteristic", "pf", "eom"])

        # Winsorized: every value comes from {Compustat clipped to 0.5, CRSP=5.0}.
        # Bounded above by 5.0; if any pure-Compustat portfolio exists,
        # its EW return is exactly 0.5.
        assert pf_win["ret_ew"].max() <= 5.0 + 1e-9
        # Aggregate must change between branches.
        assert not np.allclose(pf_win["ret_ew"].to_numpy(), pf_no["ret_ew"].to_numpy(), atol=1e-12)
        # And the winsorized mean should be strictly lower than the unclipped one.
        assert pf_win["ret_ew"].mean() < pf_no["ret_ew"].mean()

    def test_wins_ret_true_compustat_lower_clip(self, tmp_path: Path) -> None:
        """``wins_ret=True`` must clip Compustat ret_exc_lead1m below ``p001``."""
        excntry = "SYN"
        data_root = tmp_path / "processed"
        char_df = make_country_characteristics(
            excntry=excntry, chars=SYNTHETIC_CHARS, n_ids=60, n_months=6, seed=31
        )
        char_df = char_df.with_columns(
            pl.when(pl.col("source_crsp") == 0)
            .then(pl.lit(-5.0))
            .otherwise(pl.col("ret_exc_lead1m"))
            .alias("ret_exc_lead1m")
        )
        daily_df = make_daily_returns(char_df, seed=32)
        _write_inputs(data_root, excntry, char_df, daily_df)

        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, _ = make_cutoffs(eoms)

        kwargs = {
            "data_path": str(data_root),
            "excntry": excntry,
            "chars": [SYNTHETIC_CHARS[0]],
            "pfs": 3,
            "bps": "non_mc",
            "bp_min_n": 1,
            "nyse_size_cutoffs": nyse_cut,
            "source": ["CRSP", "COMPUSTAT"],
            "cmp_key": False,
            "signals": False,
            "daily_pf": False,
            "ind_pf": False,
            "ret_cutoffs": ret_cut,
            "ret_cutoffs_daily": None,
        }
        result_win = portfolios(wins_ret=True, **kwargs)
        result_no = portfolios(wins_ret=False, **kwargs)

        pf_win = result_win["pf_returns"]
        pf_no = result_no["pf_returns"]

        # Without winsor the minimum EW return should reach toward -5.0;
        # with winsor the Compustat floor of -0.5 lifts the minimum.
        assert pf_no["ret_ew"].min() < -1.0
        assert pf_win["ret_ew"].min() > pf_no["ret_ew"].min()

    def test_wins_ret_true_daily_branch(self, tmp_path: Path) -> None:
        """Daily winsor branch (id > 99999) clips daily Compustat returns."""
        excntry = "SYN"
        data_root = tmp_path / "processed"
        char_df = make_country_characteristics(
            excntry=excntry, chars=SYNTHETIC_CHARS, n_ids=60, n_months=6, seed=41
        )
        # Remap a chunk of Compustat ids above 99999 so the daily-branch
        # ``id > 99999`` predicate fires for them.
        char_df = char_df.with_columns(
            pl.when(pl.col("source_crsp") == 0)
            .then(pl.col("id") + 100000)
            .otherwise(pl.col("id"))
            .alias("id")
        )
        daily_df = make_daily_returns(char_df, seed=42)
        # Inject extreme daily returns on the id > 99999 rows.
        daily_df = daily_df.with_columns(
            pl.when(pl.col("id") > 99999)
            .then(pl.lit(5.0))
            .otherwise(pl.col("ret_exc"))
            .alias("ret_exc")
        )
        _write_inputs(data_root, excntry, char_df, daily_df)

        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)

        kwargs = {
            "data_path": str(data_root),
            "excntry": excntry,
            "chars": [SYNTHETIC_CHARS[0]],
            "pfs": 3,
            "bps": "non_mc",
            "bp_min_n": 1,
            "nyse_size_cutoffs": nyse_cut,
            "source": ["CRSP", "COMPUSTAT"],
            "cmp_key": False,
            "signals": False,
            "daily_pf": True,
            "ind_pf": False,
            "ret_cutoffs": ret_cut,
            "ret_cutoffs_daily": ret_cut_daily,
        }
        result_win = portfolios(wins_ret=True, **kwargs)
        result_no = portfolios(wins_ret=False, **kwargs)

        pf_daily_win = result_win["pf_daily"]
        pf_daily_no = result_no["pf_daily"]

        # Daily cutoffs from make_cutoffs are p001=-0.2, p999=0.2.
        assert pf_daily_no["ret_ew"].max() > 1.0
        assert pf_daily_win["ret_ew"].max() <= pf_daily_no["ret_ew"].max()
        assert not np.allclose(
            pf_daily_win["ret_ew"].to_numpy(),
            pf_daily_no["ret_ew"].to_numpy(),
            atol=1e-12,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
