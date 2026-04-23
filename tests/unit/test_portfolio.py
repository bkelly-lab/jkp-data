"""Tests for the portfolio module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jkp.data.portfolio import portfolios
from tests.unit.test_portfolio_parity import (
    SYNTHETIC_CHARS,
    _assert_frames_parity,
    _make_cutoffs,
    _write_synthetic_country,
)


class TestOutputFormatIntegration:
    """Tests that run_portfolio() forwards output_format to configure_output_format."""

    def test_default_format_is_parquet(self, tmp_path):
        """run_portfolio() defaults to parquet format."""
        from jkp.data.portfolio import run_portfolio

        with patch(
            "jkp.data.portfolio.configure_output_format",
            side_effect=SystemExit("bail"),
        ) as mock_configure:
            with pytest.raises(SystemExit):
                run_portfolio(output_dir=tmp_path)
            mock_configure.assert_called_once_with("parquet")

    def test_csv_format_passed_through(self, tmp_path):
        """run_portfolio(output_format='csv') forwards 'csv' to configure_output_format."""
        from jkp.data.portfolio import run_portfolio

        with patch(
            "jkp.data.portfolio.configure_output_format",
            side_effect=SystemExit("bail"),
        ) as mock_configure:
            with pytest.raises(SystemExit):
                run_portfolio(output_format="csv", output_dir=tmp_path)
            mock_configure.assert_called_once_with("csv")


_TIGHT = {"rtol": 1e-10, "atol": 1e-12}


class TestExcntryGating:
    """Tests that excntry comparison is case-insensitive."""

    def test_usa_gating_case_insensitive(self, tmp_path, seed):
        """portfolios() with excntry='USA' and excntry='usa' produce identical
        ff49_returns and ff49_daily."""
        # macOS filesystem is case-insensitive, so we use separate directories
        # for the two runs to avoid SameFileError.
        upper_root = tmp_path / "upper" / "processed"
        lower_root = tmp_path / "lower" / "processed"

        char_df, _ = _write_synthetic_country(
            data_root=upper_root, excntry="USA", chars=SYNTHETIC_CHARS, seed=seed
        )
        _write_synthetic_country(
            data_root=lower_root, excntry="usa", chars=SYNTHETIC_CHARS, seed=seed
        )

        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)

        shared = {
            "chars": SYNTHETIC_CHARS,
            "pfs": 3,
            "bps": "non_mc",
            "bp_min_n": 10,
            "nyse_size_cutoffs": nyse_cut,
            "source": ["CRSP", "COMPUSTAT"],
            "wins_ret": True,
            "cmp_key": False,
            "signals": False,
            "signals_standardize": True,
            "signals_w": "vw_cap",
            "daily_pf": True,
            "ind_pf": True,
            "ret_cutoffs": ret_cut,
            "ret_cutoffs_daily": ret_cut_daily,
        }

        upper = portfolios(data_path=str(upper_root), excntry="USA", **shared)
        lower = portfolios(data_path=str(lower_root), excntry="usa", **shared)

        for key in ("ff49_returns", "ff49_daily"):
            assert key in upper, f"{key!r} missing from excntry='USA' output"
            assert key in lower, f"{key!r} missing from excntry='usa' output"

        ind_key_cols = ["ff49", "excntry"]
        ind_numeric = {
            "n": _TIGHT,
            "ret_ew": _TIGHT,
            "ret_vw": _TIGHT,
            "ret_vw_cap": _TIGHT,
        }

        _assert_frames_parity(
            upper["ff49_returns"],
            lower["ff49_returns"],
            key_cols=["ff49", "eom", "excntry"],
            numeric_cols=ind_numeric,
            label="ff49_returns case parity",
        )
        _assert_frames_parity(
            upper["ff49_daily"],
            lower["ff49_daily"],
            key_cols=[*ind_key_cols, "date"],
            numeric_cols=ind_numeric,
            label="ff49_daily case parity",
        )
