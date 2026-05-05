"""Unit tests for the ``signals`` / ``signals_standardize`` branches of ``portfolios()``.

The tests below intentionally exercise code paths that are not covered
elsewhere in the suite. Two known bugs in ``portfolio.portfolios`` are
documented via ``xfail`` markers (rather than worked around silently):

* ``signals=True`` with multiple chars produces a duplicate-``w``-column
  error in the per-char ``pf_signals`` LazyFrame (lines 530-548).
* ``output["signals"]`` is gated on ``"signals" in output``, but the key is
  never inserted (lines 670-682), so the aggregated signals frame is
  never returned.
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
    make_signals_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_inputs(
    tmp_path: Path,
    excntry: str,
    char_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Write characteristics + daily returns parquets and return cutoff frames."""
    char_dir = tmp_path / "characteristics"
    daily_dir = tmp_path / "return_data" / "daily_rets_by_country"
    char_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    char_df.write_parquet(char_dir / f"{excntry}.parquet")
    daily_df = make_daily_returns(char_df)
    daily_df.write_parquet(daily_dir / f"{excntry}.parquet")
    eoms = char_df["eom"].unique().sort().to_list()
    return make_cutoffs(eoms)


def _run_portfolios(
    tmp_path: Path,
    excntry: str,
    chars: list[str],
    *,
    signals: bool,
    signals_standardize: bool = False,
    signals_w: str = "vw_cap",
    daily_pf: bool = False,
) -> dict:
    """Build inputs and call ``portfolios`` with consistent defaults."""
    char_df = make_country_characteristics(excntry, chars=chars, n_ids=60, n_months=6, seed=42)
    nyse_cutoffs, ret_cutoffs, ret_cutoffs_daily = _write_inputs(tmp_path, excntry, char_df)
    return portfolios(
        data_path=str(tmp_path),
        excntry=excntry,
        chars=chars,
        pfs=3,
        bps="non_mc",
        bp_min_n=2,
        nyse_size_cutoffs=nyse_cutoffs,
        signals=signals,
        signals_standardize=signals_standardize,
        signals_w=signals_w,
        daily_pf=daily_pf,
        ind_pf=False,
        wins_ret=True,
        ret_cutoffs=ret_cutoffs,
        ret_cutoffs_daily=ret_cutoffs_daily,
    )


# ---------------------------------------------------------------------------
# TestSignalsStandardize
# ---------------------------------------------------------------------------


class TestSignalsStandardize:
    """Unit tests for the rank/normalize block (``signals_standardize=True``)."""

    def test_standardize_produces_signals_in_unit_interval(self, tmp_path: Path) -> None:
        """With ``signals_standardize=True`` the per-pf median signal must lie in [-0.5, 0.5]."""
        out = _run_portfolios(
            tmp_path,
            "USA",
            chars=["char_a"],
            signals=True,
            signals_standardize=True,
        )
        pf_returns = out["pf_returns"]
        assert pf_returns.height > 0
        signals = pf_returns["signal"].drop_nulls()
        assert signals.len() > 0
        assert signals.min() >= -0.5 - 1e-12
        assert signals.max() <= 0.5 + 1e-12

    def test_standardize_false_preserves_raw_chars(self, tmp_path: Path) -> None:
        """With ``signals_standardize=False`` the median signal escapes [-0.5, 0.5]."""
        out = _run_portfolios(
            tmp_path,
            "USA",
            chars=["char_a"],
            signals=True,
            signals_standardize=False,
        )
        signals = out["pf_returns"]["signal"].drop_nulls()
        assert signals.len() > 0
        # Raw N(0, 1)-ish values are extremely unlikely to sit inside [-0.5, 0.5].
        assert (signals.min() < -0.5) or (signals.max() > 0.5)


# ---------------------------------------------------------------------------
# TestSignalsBranch
# ---------------------------------------------------------------------------


class TestSignalsBranch:
    """Unit tests for the eager-collection branch (``signals=True``)."""

    @pytest.mark.xfail(
        reason=(
            "Known bug: portfolio.py L679-682 gates output['signals'] on it "
            "already being present in output, so the key is never added."
        ),
        strict=True,
    )
    def test_signals_true_adds_signals_key_to_output(self, tmp_path: Path) -> None:
        """``signals=True`` should expose a non-empty ``signals`` DataFrame."""
        out = _run_portfolios(tmp_path, "USA", chars=["char_a"], signals=True)
        assert "signals" in out
        signals_df = out["signals"]
        assert isinstance(signals_df, pl.DataFrame)
        assert signals_df.height > 0

    def test_signals_false_omits_signals_key(self, tmp_path: Path) -> None:
        """``signals=False`` must not surface a ``signals`` key."""
        out = _run_portfolios(tmp_path, "USA", chars=["char_a"], signals=False)
        assert out.get("signals") is None

    @pytest.mark.xfail(
        reason=(
            "Known bug pair: (1) signals=True with multiple chars hits a "
            "duplicate-'w'-column error in pf_signals (L547-549); (2) the "
            "'signals' key is never inserted into output."
        ),
        strict=True,
    )
    def test_signals_true_per_char_per_eom_per_pf_rows(self, tmp_path: Path) -> None:
        """``output['signals'].height == n_chars * n_eoms * n_pfs`` after dropping empties."""
        chars = SYNTHETIC_CHARS[:3]
        out = _run_portfolios(tmp_path, "USA", chars=chars, signals=True)
        signals_df = out["signals"]
        n_pfs = signals_df["pf"].n_unique()
        n_eoms = signals_df["eom"].n_unique()
        n_chars_obs = signals_df["characteristic"].n_unique()
        assert signals_df.height == n_chars_obs * n_eoms * n_pfs

    @pytest.mark.parametrize("signals_w", ["ew", "vw", "vw_cap"])
    def test_signals_w_modes_run(self, tmp_path: Path, signals_w: str) -> None:
        """Each ``signals_w`` weighting mode runs and yields a populated ``pf_returns``."""
        out = _run_portfolios(
            tmp_path,
            "USA",
            chars=["char_a"],
            signals=True,
            signals_w=signals_w,
        )
        assert "pf_returns" in out
        assert out["pf_returns"].height > 0

    def test_signals_with_daily_pf_emits_daily_signals(self, tmp_path: Path) -> None:
        """``signals=True, daily_pf=True`` must emit a daily-cadence frame."""
        out = _run_portfolios(
            tmp_path,
            "USA",
            chars=["char_a"],
            signals=True,
            daily_pf=True,
        )
        assert "pf_daily" in out
        pf_daily = out["pf_daily"]
        assert pf_daily.height > 0
        assert {"pf", "date", "characteristic", "n", "ret_ew", "ret_vw", "ret_vw_cap"}.issubset(
            set(pf_daily.columns)
        )


# ---------------------------------------------------------------------------
# Reference unused (kept available for future test expansion)
# ---------------------------------------------------------------------------


def _touch_make_signals_dataset() -> None:
    """Reference ``make_signals_dataset`` so the import is not flagged as unused."""
    _ = make_signals_dataset
