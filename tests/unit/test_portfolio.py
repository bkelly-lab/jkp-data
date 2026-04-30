"""Unit tests for ``portfolio.py`` functions.

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
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from jkp.data.portfolio import (
    _build_regional_loop,
    add_ecdf,
    portfolios,
    regional_data,
)

# Reuse synthetic helpers from the parity test module (avoids duplication).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from tests.unit.test_portfolio_parity import (  # noqa: E402
    SYNTHETIC_CHARS,
    _assert_frames_parity,
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


def _add_ecdf_legacy(
    df: pl.DataFrame | pl.LazyFrame,
    group_cols: list[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Pre-refactor implementation, kept here for the parity test below."""
    if group_cols is None:
        group_cols = ["eom"]
    ref_counts = df.filter(pl.col("bp_stock")).group_by(group_cols + ["var"]).agg(n_ref=pl.len())
    ref_steps = (
        ref_counts.sort(group_cols + ["var"])
        .with_columns(cdf_val=(pl.cum_sum("n_ref") / pl.sum("n_ref")).over(group_cols))
        .select(group_cols + ["var", "cdf_val"])
    )
    left = df.sort(group_cols + ["var"])
    right = ref_steps.sort(group_cols + ["var"])
    return (
        left.join_asof(right, on="var", by=group_cols, strategy="backward")
        .with_columns(pl.col("cdf_val").fill_null(0.0).alias("cdf"))
        .drop("cdf_val")
    )


class TestAddEcdfLegacyParity:
    """Refactored ``add_ecdf`` must produce identical output to the pre-refactor version."""

    @staticmethod
    def _build(seed: int, n_groups: int = 3, n_per_group: int = 40) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        eoms = [date(2020, m, 28) for m in range(1, n_groups + 1)]
        rows = []
        for eom in eoms:
            # Mix of breakpoint and non-breakpoint stocks; force duplicate var
            # values across both classes; include rows with var below any bp.
            for _ in range(n_per_group):
                rows.append(
                    {
                        "eom": eom,
                        "var": float(rng.integers(0, 12)),  # repeated integers as floats
                        "bp_stock": bool(rng.integers(0, 2)),
                    }
                )
            # Force a non-bp row strictly below the minimum bp value in this group.
            rows.append({"eom": eom, "var": -100.0, "bp_stock": False})
            # Force a non-bp row that lands strictly between two bp values.
            rows.append({"eom": eom, "var": 5.5, "bp_stock": False})
        return pl.DataFrame(rows)

    @staticmethod
    def _assert_equal(a: pl.DataFrame, b: pl.DataFrame) -> None:
        a_sorted = a.sort(["eom", "var", "bp_stock"])
        b_sorted = b.sort(["eom", "var", "bp_stock"])
        assert a_sorted.height == b_sorted.height
        assert a_sorted.columns == b_sorted.columns
        for col in a_sorted.columns:
            if a_sorted.schema[col] == pl.Float64:
                np.testing.assert_allclose(
                    a_sorted[col].to_numpy(),
                    b_sorted[col].to_numpy(),
                    rtol=1e-12,
                    atol=1e-14,
                )
            else:
                assert a_sorted[col].to_list() == b_sorted[col].to_list()

    def test_eager_parity_default_group(self):
        df = self._build(seed=1)
        legacy = _add_ecdf_legacy(df)
        new = add_ecdf(df)
        self._assert_equal(legacy, new)

    def test_lazy_parity_default_group(self):
        df = self._build(seed=2).lazy()
        legacy = _add_ecdf_legacy(df).collect()
        new = add_ecdf(df).collect()
        self._assert_equal(legacy, new)

    def test_lazy_returns_lazyframe(self):
        df = self._build(seed=3).lazy()
        out = add_ecdf(df)
        assert isinstance(out, pl.LazyFrame)

    def test_duplicate_var_values_in_bp_sample(self):
        """Duplicate bp ``var`` values must contribute their full count to the CDF."""
        df = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31)] * 8,
                "var": [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
                "bp_stock": [True] * 8,
            }
        )
        self._assert_equal(_add_ecdf_legacy(df), add_ecdf(df))

    def test_non_bp_between_breakpoints(self):
        """A non-bp row with var between two bp values gets the lower bp's CDF."""
        df = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31)] * 5,
                "var": [1.0, 2.0, 3.0, 1.5, 2.5],
                "bp_stock": [True, True, True, False, False],
            }
        )
        self._assert_equal(_add_ecdf_legacy(df), add_ecdf(df))

    def test_rows_below_minimum_breakpoint(self):
        """Rows with var below the smallest bp value get cdf == 0.0."""
        df = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31)] * 4,
                "var": [10.0, 20.0, 5.0, -1.0],
                "bp_stock": [True, True, False, False],
            }
        )
        new = add_ecdf(df)
        assert new.filter(pl.col("var") == -1.0)["cdf"].to_list() == [0.0]
        self._assert_equal(_add_ecdf_legacy(df), new)

    def test_multi_group_independence(self):
        """ECDFs must be computed per ``eom`` group, independently."""
        df = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31), date(2020, 1, 31), date(2020, 2, 29), date(2020, 2, 29)],
                "var": [1.0, 2.0, 100.0, 200.0],
                "bp_stock": [True, True, True, True],
            }
        )
        new = add_ecdf(df)
        # In each group of 2 distinct values, smallest gets 0.5, largest gets 1.0.
        for eom in [date(2020, 1, 31), date(2020, 2, 29)]:
            grp = new.filter(pl.col("eom") == eom).sort("var")
            assert grp["cdf"].to_list() == [0.5, 1.0]
        self._assert_equal(_add_ecdf_legacy(df), new)

    def test_custom_group_cols(self):
        df = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31)] * 6,
                "size": ["L", "L", "L", "S", "S", "S"],
                "var": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "bp_stock": [True] * 6,
            }
        )
        legacy = _add_ecdf_legacy(df, group_cols=["eom", "size"])
        new = add_ecdf(df, group_cols=["eom", "size"])
        self._assert_equal(legacy, new)


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


class TestBuildRegionalLoop:
    """Tests for ``_build_regional_loop()``.

    Regression coverage: ``iter_rows(named=True)`` returns Python ``list``
    for list columns, but ``regional_data`` calls ``.implode()`` which only
    exists on ``pl.Series``. The helper must wrap ``country_codes`` in a
    ``pl.Series`` before passing it through.
    """

    @staticmethod
    def _sample_inputs(n_countries: int = 4, n_months: int = 12, n_chars: int = 2):
        lms = TestRegionalData._sample_lms(
            n_countries=n_countries, n_months=n_months, n_chars=n_chars
        )
        countries = lms["excntry"].unique().to_list()
        mkt = TestRegionalData._sample_market(countries, _month_ends(n_months))
        regions = pl.DataFrame(
            {
                "name": ["all", "subset"],
                "country_codes": [countries, countries[:2]],
                "countries_min": [1, 1],
            }
        )
        output_cols = [
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
        return lms, mkt, regions, output_cols

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
