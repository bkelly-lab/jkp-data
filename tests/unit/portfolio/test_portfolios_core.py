"""Unit tests for ``portfolios()`` core monthly and daily outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from jkp.data.portfolio import portfolios
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    make_cutoffs,
    write_synthetic_country,
)


class TestPortfoliosCore:
    """Tests for the core monthly ``portfolios()`` output."""

    @pytest.fixture()
    def portfolio_result(self, tmp_path: Path, seed: int):
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)
        return portfolios(
            data_path=str(data_root),
            excntry="SYN",
            chars=SYNTHETIC_CHARS,
            pfs=3,
            bps="non_mc",
            bp_min_n=10,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=True,
            cmp_key=False,
            signals=False,
            daily_pf=True,
            ind_pf=True,
            ret_cutoffs=ret_cut,
            ret_cutoffs_daily=ret_cut_daily,
        )

    def test_pf_returns_schema_and_columns(self, portfolio_result):
        pf_ret = portfolio_result["pf_returns"]
        expected_cols = {
            "characteristic",
            "pf",
            "eom",
            "n",
            "signal",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
        }
        assert expected_cols.issubset(set(pf_ret.columns))

    def test_pf_values_in_1_to_pfs_range(self, portfolio_result):
        pf_ret = portfolio_result["pf_returns"]
        pf_vals = pf_ret["pf"].unique().sort().to_list()
        for v in pf_vals:
            assert 1 <= v <= 3

    def test_pf_monotone_signal_medians_within_eom_char(self, portfolio_result):
        """Signal median should be non-decreasing in pf within each
        (eom, characteristic) group — the monotonic quantile invariant.
        """
        pf_ret = portfolio_result["pf_returns"]
        groups = pf_ret.sort(["characteristic", "eom", "pf"]).group_by(
            ["characteristic", "eom"], maintain_order=True
        )
        for _key, grp in groups:
            signals = grp.sort("pf")["signal"].to_list()
            # Filter out None/NaN
            signals = [s for s in signals if s is not None and not np.isnan(s)]
            if len(signals) > 1:
                for i in range(1, len(signals)):
                    assert signals[i] >= signals[i - 1], (
                        f"Signal not monotone: {signals} for group {_key}"
                    )

    def test_ret_exc_lead1m_winsorized_for_compustat(self, tmp_path: Path, seed: int):
        """Compustat rows (source_crsp == 0) should have ret_exc_lead1m
        clipped to [p001, p999] from ret_cutoffs, while CRSP rows remain
        unchanged.
        """
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, _ = make_cutoffs(eoms)

        raw = pl.read_parquet(data_root / "characteristics" / "SYN.parquet")
        compustat_rows = raw.filter(pl.col("source_crsp") == 0)
        crsp_rows = raw.filter(pl.col("source_crsp") == 1)
        assert compustat_rows.height > 0
        assert crsp_rows.height > 0

        # Inject extreme ret_exc_lead1m values that will exceed tight cutoffs.
        raw = raw.with_columns(
            pl.when(pl.col("source_crsp") == 0)
            .then(pl.lit(0.9))
            .otherwise(pl.col("ret_exc_lead1m"))
            .alias("ret_exc_lead1m")
        )
        raw.write_parquet(data_root / "characteristics" / "SYN.parquet")

        # Use tight cutoffs so 0.9 exceeds the upper bound.
        tight_ret_cut = ret_cut.with_columns(
            pl.lit(-0.10).alias("ret_exc_0_1"),
            pl.lit(0.10).alias("ret_exc_99_9"),
        )

        test_char = SYNTHETIC_CHARS[0]

        result_win = portfolios(
            data_path=str(data_root),
            excntry="SYN",
            chars=[test_char],
            pfs=1,
            bps="non_mc",
            bp_min_n=1,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=True,
            cmp_key=False,
            signals=False,
            daily_pf=False,
            ind_pf=False,
            ret_cutoffs=tight_ret_cut,
            ret_cutoffs_daily=None,
        )
        result_no_win = portfolios(
            data_path=str(data_root),
            excntry="SYN",
            chars=[test_char],
            pfs=1,
            bps="non_mc",
            bp_min_n=1,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=False,
            cmp_key=False,
            signals=False,
            daily_pf=False,
            ind_pf=False,
            ret_cutoffs=tight_ret_cut,
            ret_cutoffs_daily=None,
        )

        pf_win = (
            result_win["pf_returns"]
            .filter((pl.col("characteristic") == test_char) & (pl.col("pf") == 1))
            .sort("eom")
        )
        pf_no_win = (
            result_no_win["pf_returns"]
            .filter((pl.col("characteristic") == test_char) & (pl.col("pf") == 1))
            .sort("eom")
        )

        # Winsorized and non-winsorized returns should differ because
        # Compustat ret_exc_lead1m (0.9) exceeds p999 (0.10).
        assert not np.allclose(
            pf_win["ret_ew"].to_numpy(),
            pf_no_win["ret_ew"].to_numpy(),
            atol=1e-12,
        )


class TestPortfoliosDaily:
    """Tests for the daily ``pf_daily`` output of ``portfolios()``."""

    @pytest.fixture()
    def portfolio_result(self, tmp_path: Path, seed: int):
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)
        return portfolios(
            data_path=str(data_root),
            excntry="SYN",
            chars=SYNTHETIC_CHARS,
            pfs=3,
            bps="non_mc",
            bp_min_n=10,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=True,
            cmp_key=False,
            signals=False,
            daily_pf=True,
            ind_pf=False,
            ret_cutoffs=ret_cut,
            ret_cutoffs_daily=ret_cut_daily,
        )

    def test_pf_daily_has_expected_columns(self, portfolio_result):
        pf_daily = portfolio_result["pf_daily"]
        expected = {"characteristic", "pf", "date", "n", "ret_ew", "ret_vw", "ret_vw_cap"}
        assert expected.issubset(set(pf_daily.columns))

    def test_pf_daily_dates_are_weekdays(self, portfolio_result):
        """Daily portfolio dates should be weekdays (Mon-Fri)."""
        pf_daily = portfolio_result["pf_daily"]
        weekdays = pf_daily.with_columns(pl.col("date").dt.weekday().alias("wd"))
        # Polars weekday: 1=Mon .. 7=Sun
        assert weekdays.filter(pl.col("wd") > 5).height == 0
