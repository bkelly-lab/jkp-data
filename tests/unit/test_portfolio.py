"""Unit tests for ``code/portfolio.py`` functions.

Tests cover:
    - ``add_ecdf``: ECDF construction, ties, non-bp stocks, null handling,
      multi-group-column support (the vectorized call site).
    - ``portfolios``: schema guarantees, pf range, monotone signal invariant,
      me capping, return winsorization.
    - ``portfolios`` (daily): weight normalisation, month-rollover semantics.
    - ``portfolios`` (industry): GICS extraction, FF49 USA-only guard, bp_min_n.
    - ``regional_data``: country exclusion, n_countries filter, weighting modes,
      periods_min filter.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Ensure code/ is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code"))

from portfolio import add_ecdf, portfolios, regional_data  # noqa: E402

# Reuse synthetic helpers from the parity test module (avoids duplication).
sys.path.insert(0, str(_REPO_ROOT))
from tests.unit.test_portfolio_parity import (  # noqa: E402
    SYNTHETIC_CHARS,
    _make_cutoffs,
    _month_ends,
    _write_synthetic_country,
)

# =============================================================================
# TestAddEcdf
# =============================================================================


class TestAddEcdf:
    """Unit tests for ``add_ecdf``."""

    @staticmethod
    def _simple_frame(
        var_vals: list[float],
        bp_stock: list[bool],
        eom: date | None = None,
    ) -> pl.DataFrame:
        """Build a small DataFrame suitable for ``add_ecdf``."""
        n = len(var_vals)
        if eom is None:
            eom = date(2020, 1, 31)
        return pl.DataFrame(
            {
                "eom": [eom] * n,
                "var": var_vals,
                "bp_stock": bp_stock,
            }
        )

    def test_basic_ecdf_uniform(self):
        """10 bp rows with distinct values → cdf ≈ rank / n."""
        vals = list(range(1, 11))
        df = self._simple_frame(
            var_vals=[float(v) for v in vals],
            bp_stock=[True] * 10,
        )
        out = add_ecdf(df)
        cdfs = out.sort("var")["cdf"].to_list()
        expected = [i / 10 for i in range(1, 11)]
        np.testing.assert_allclose(cdfs, expected, rtol=1e-12)

    def test_ties_give_step_cdf(self):
        """Tied var values in the bp sample produce equal cdf."""
        df = self._simple_frame(
            var_vals=[1.0, 1.0, 2.0, 2.0],
            bp_stock=[True, True, True, True],
        )
        out = add_ecdf(df)
        cdfs = out.sort(["var", "cdf"])["cdf"].to_list()
        # Both var=1.0 rows should share cdf from the asof-join (0.5),
        # both var=2.0 share (1.0).
        assert cdfs[0] == cdfs[1]
        assert cdfs[2] == cdfs[3]
        assert cdfs[2] > cdfs[0]

    def test_non_bp_stock_gets_asof_cdf(self):
        """A non-bp row picks up the cdf of the nearest bp var <= its value."""
        df = self._simple_frame(
            var_vals=[1.0, 2.0, 3.0, 2.5],
            bp_stock=[True, True, True, False],
        )
        out = add_ecdf(df)
        # Non-bp row has var=2.5 → asof picks up cdf of bp var=2.0
        non_bp = out.filter(~pl.col("bp_stock")).sort("var")
        bp_2 = out.filter(pl.col("bp_stock") & (pl.col("var") == 2.0))
        assert non_bp["cdf"][0] == bp_2["cdf"][0]

    def test_below_min_bp_var_fills_null_cdf_to_zero(self):
        """A non-bp row whose var < all bp vars gets cdf = 0.0."""
        df = self._simple_frame(
            var_vals=[10.0, 20.0, 5.0],
            bp_stock=[True, True, False],
        )
        out = add_ecdf(df)
        below = out.filter(pl.col("var") == 5.0)
        assert below["cdf"][0] == 0.0

    def test_multiple_eom_groups_independent(self):
        """Two eom groups compute independent ECDFs."""
        eom1 = date(2020, 1, 31)
        eom2 = date(2020, 2, 29)
        df = pl.concat(
            [
                self._simple_frame([1.0, 2.0], [True, True], eom=eom1),
                self._simple_frame([10.0, 20.0], [True, True], eom=eom2),
            ]
        )
        out = add_ecdf(df)
        g1 = out.filter(pl.col("eom") == eom1).sort("var")["cdf"].to_list()
        g2 = out.filter(pl.col("eom") == eom2).sort("var")["cdf"].to_list()
        assert g1 == [0.5, 1.0]
        assert g2 == [0.5, 1.0]

    def test_multi_group_cols_characteristic_and_eom(self):
        """Call with group_cols=["characteristic","eom"] on a long-format
        frame with two characteristics; verify independent ECDFs per partition.
        """
        eom = date(2020, 1, 31)
        df = pl.DataFrame(
            {
                "characteristic": ["A", "A", "A", "B", "B", "B"],
                "eom": [eom] * 6,
                "var": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
                "bp_stock": [True, True, True, True, True, True],
            }
        )
        out = add_ecdf(df, group_cols=["characteristic", "eom"])
        a = out.filter(pl.col("characteristic") == "A").sort("var")["cdf"].to_list()
        b = out.filter(pl.col("characteristic") == "B").sort("var")["cdf"].to_list()
        # Each partition has 3 bp values → cdf = 1/3, 2/3, 3/3
        expected = [1 / 3, 2 / 3, 1.0]
        np.testing.assert_allclose(a, expected, rtol=1e-12)
        np.testing.assert_allclose(b, expected, rtol=1e-12)

    def test_lazyframe_in_lazyframe_out(self):
        """If input is a LazyFrame, output should also be a LazyFrame."""
        df = self._simple_frame([1.0, 2.0], [True, True])
        out = add_ecdf(df.lazy())
        assert isinstance(out, pl.LazyFrame)
        # And collecting should work
        result = out.collect()
        assert result.height == 2
        assert "cdf" in result.columns


# =============================================================================
# TestPortfoliosCore
# =============================================================================


class TestPortfoliosCore:
    """Tests for the core monthly ``portfolios()`` output."""

    @pytest.fixture()
    def portfolio_result(self, tmp_path: Path, seed: int):
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)
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
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, _ = _make_cutoffs(eoms)

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


# =============================================================================
# TestPortfoliosDaily
# =============================================================================


class TestPortfoliosDaily:
    """Tests for the daily ``pf_daily`` output of ``portfolios()``."""

    @pytest.fixture()
    def portfolio_result(self, tmp_path: Path, seed: int):
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)
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


# =============================================================================
# TestPortfoliosIndustry
# =============================================================================


class TestPortfoliosIndustry:
    """Tests for GICS and FF49 industry portfolio outputs."""

    @pytest.fixture()
    def usa_result(self, tmp_path: Path, seed: int):
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="USA", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)
        return portfolios(
            data_path=str(data_root),
            excntry="USA",
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

    def test_gics_first_2_digits_extracted_as_int(self, usa_result):
        gics = usa_result["gics_returns"]
        assert gics["gics"].dtype == pl.Float64 or gics["gics"].dtype == pl.Int64
        # All values should be 2-digit sector codes (10-60 range)
        vals = gics["gics"].unique().sort().to_list()
        for v in vals:
            assert 10 <= v <= 60

    def test_ff49_returns_present_for_usa(self, usa_result):
        assert "ff49_returns" in usa_result
        assert usa_result["ff49_returns"].height > 0

    def test_ff49_daily_present_for_usa(self, usa_result):
        assert "ff49_daily" in usa_result
        assert usa_result["ff49_daily"].height > 0

    def test_ff49_returns_only_for_usa(self, tmp_path: Path, seed: int):
        """Non-USA inputs should NOT produce ff49_returns."""
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="GBR", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)
        result = portfolios(
            data_path=str(data_root),
            excntry="GBR",
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
        assert "ff49_returns" not in result
        assert "ff49_daily" not in result

    def test_ind_returns_filter_by_bp_min_n(self, tmp_path: Path, seed: int):
        """Industries with n < bp_min_n should be filtered out."""
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)
        result = portfolios(
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
            daily_pf=False,
            ind_pf=True,
            ret_cutoffs=ret_cut,
            ret_cutoffs_daily=ret_cut_daily,
        )
        gics = result["gics_returns"]
        # All groups should have n >= bp_min_n
        assert gics.filter(pl.col("n") < 10).height == 0


# =============================================================================
# TestRegionalData
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


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_format_is_parquet(self, monkeypatch):
        """Default output format should be parquet when no args provided."""
        monkeypatch.setattr("sys.argv", ["portfolio.py"])
        from portfolio import parse_args

        args = parse_args()
        assert args.output_format == "parquet"

    def test_can_specify_csv_format(self, monkeypatch):
        """Can specify CSV output format via command line."""
        monkeypatch.setattr("sys.argv", ["portfolio.py", "--output-format", "csv"])
        from portfolio import parse_args

        args = parse_args()
        assert args.output_format == "csv"

    def test_can_specify_parquet_format(self, monkeypatch):
        """Can explicitly specify parquet output format."""
        monkeypatch.setattr("sys.argv", ["portfolio.py", "--output-format", "parquet"])
        from portfolio import parse_args

        args = parse_args()
        assert args.output_format == "parquet"

    def test_invalid_format_raises_error(self, monkeypatch):
        """Invalid output format raises SystemExit."""
        monkeypatch.setattr("sys.argv", ["portfolio.py", "--output-format", "json"])
        from portfolio import parse_args

        with pytest.raises(SystemExit):
            parse_args()
