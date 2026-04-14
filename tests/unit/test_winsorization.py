"""
Tests for winsorization-related functions.

Validates that grouping by eom produces correct results and that the
simplification from year/month to eom introduces no regressions.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp_data.aux_functions import (
    add_cutoffs_and_winsorize,
    drop_non_trading_days,
    load_mkt_returns_params,
)

# =============================================================================
# TestLoadMktReturnsParams
# =============================================================================


class TestLoadMktReturnsParams:
    """Tests for load_mkt_returns_params().

    Verifies that group_vars always uses eom and never year/month.
    """

    def test_monthly_group_vars_is_eom(self):
        """Monthly frequency should use eom as group variable."""
        dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols = load_mkt_returns_params("m")
        assert group_vars == ["eom"], f"Expected ['eom'], got {group_vars}"

    def test_daily_group_vars_is_eom(self):
        """Daily frequency should use eom as group variable."""
        dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols = load_mkt_returns_params("d")
        assert group_vars == ["eom"], f"Expected ['eom'], got {group_vars}"

    def test_no_year_month_in_group_vars(self):
        """Neither frequency should return year or month in group_vars."""
        for freq in ["m", "d"]:
            _, _, _, group_vars, _ = load_mkt_returns_params(freq)
            assert "year" not in group_vars, f"'year' found in group_vars for freq={freq}"
            assert "month" not in group_vars, f"'month' found in group_vars for freq={freq}"

    def test_eom_in_comm_stocks_cols(self):
        """Both frequencies should include eom in comm_stocks_cols."""
        for freq in ["m", "d"]:
            _, _, _, _, comm_stocks_cols = load_mkt_returns_params(freq)
            assert "eom" in comm_stocks_cols, f"'eom' missing from comm_stocks_cols for freq={freq}"


# =============================================================================
# TestDropNonTradingDays
# =============================================================================


class TestDropNonTradingDays:
    """Tests for drop_non_trading_days().

    Verifies thin-trading day removal works with eom-based over_vars.
    """

    @pytest.fixture
    def daily_df(self):
        """Create a daily DataFrame with 2 months and varying stock coverage."""
        return pl.LazyFrame(
            {
                "date": [
                    date(2020, 1, 2),
                    date(2020, 1, 3),
                    date(2020, 1, 6),  # low coverage day
                    date(2020, 2, 3),
                    date(2020, 2, 4),
                ],
                "eom": [
                    date(2020, 1, 31),
                    date(2020, 1, 31),
                    date(2020, 1, 31),
                    date(2020, 2, 29),
                    date(2020, 2, 29),
                ],
                "excntry": ["USA"] * 5,
                "stocks": [10, 10, 2, 8, 8],
            }
        )

    def test_drops_low_coverage_day(self, daily_df):
        """Day with coverage < threshold should be removed."""
        result = drop_non_trading_days(daily_df, "stocks", "date", ["excntry", "eom"], 0.25)
        result = result.collect()
        assert len(result) == 4, f"Expected 4 rows, got {len(result)}"
        dates = result["date"].to_list()
        assert date(2020, 1, 6) not in dates, "Low coverage day should be dropped"

    def test_keeps_high_coverage_days(self, daily_df):
        """Days with coverage >= threshold should be kept."""
        result = drop_non_trading_days(
            daily_df, "stocks", "date", ["excntry", "eom"], 0.25
        ).collect()
        dates = result["date"].to_list()
        assert date(2020, 1, 2) in dates
        assert date(2020, 1, 3) in dates
        assert date(2020, 2, 3) in dates
        assert date(2020, 2, 4) in dates

    def test_no_year_month_columns_in_output(self, daily_df):
        """Output should not contain year or month columns."""
        result = drop_non_trading_days(
            daily_df, "stocks", "date", ["excntry", "eom"], 0.25
        ).collect()
        assert "year" not in result.columns, "'year' should not be in output"
        assert "month" not in result.columns, "'month' should not be in output"

    def test_eom_preserved_in_output(self, daily_df):
        """eom column should still be in the output."""
        result = drop_non_trading_days(
            daily_df, "stocks", "date", ["excntry", "eom"], 0.25
        ).collect()
        assert "eom" in result.columns, "'eom' should be preserved in output"


# =============================================================================
# TestAddCutoffsAndWinsorize
# =============================================================================


class TestAddCutoffsAndWinsorize:
    """Tests for add_cutoffs_and_winsorize().

    Verifies winsorization clips Compustat rows and leaves CRSP rows unchanged,
    using eom as the join key.
    """

    @pytest.fixture
    def cutoffs_path(self, tmp_path):
        """Write a mock cutoffs parquet keyed by eom."""
        cutoffs = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31)],
                "n": [100],
                "ret_0_1": [-0.50],
                "ret_1": [-0.20],
                "ret_99": [0.20],
                "ret_99_9": [0.50],
                "ret_local_0_1": [-0.50],
                "ret_local_1": [-0.20],
                "ret_local_99": [0.20],
                "ret_local_99_9": [0.50],
                "ret_exc_0_1": [-0.40],
                "ret_exc_1": [-0.15],
                "ret_exc_99": [0.15],
                "ret_exc_99_9": [0.40],
            }
        )
        path = tmp_path / "cutoffs.parquet"
        cutoffs.write_parquet(path)
        return str(path)

    @pytest.fixture
    def stock_df(self):
        """Create a DataFrame with CRSP and Compustat rows, some with outlier returns."""
        return pl.LazyFrame(
            {
                "id": [1, 2, 100001, 100002],
                "eom": [date(2020, 1, 31)] * 4,
                "date": [date(2020, 1, 15)] * 4,
                "source_crsp": [1, 1, 0, 0],
                "ret": [0.60, 0.05, 0.60, 0.05],
                "ret_local": [0.60, 0.05, 0.60, 0.05],
                "ret_exc": [0.55, 0.03, 0.55, 0.03],
            }
        )

    def test_compustat_outliers_are_clipped(self, stock_df, cutoffs_path, tolerance):
        """Compustat rows with outlier ret_exc should be clipped to bounds."""
        result = add_cutoffs_and_winsorize(stock_df, cutoffs_path, ["eom"], "date").collect()
        comp_outlier = result.filter(pl.col("id") == 100001)
        np.testing.assert_allclose(
            comp_outlier["ret_exc"].to_list(),
            [0.40],
            **tolerance.TIGHT,
            err_msg="Compustat outlier ret_exc should be clipped to 0.40",
        )
        np.testing.assert_allclose(
            comp_outlier["ret"].to_list(),
            [0.50],
            **tolerance.TIGHT,
            err_msg="Compustat outlier ret should be clipped to 0.50",
        )

    def test_compustat_non_outliers_unchanged(self, stock_df, cutoffs_path, tolerance):
        """Compustat rows within bounds should not be modified."""
        result = add_cutoffs_and_winsorize(stock_df, cutoffs_path, ["eom"], "date").collect()
        comp_normal = result.filter(pl.col("id") == 100002)
        np.testing.assert_allclose(
            comp_normal["ret_exc"].to_list(),
            [0.03],
            **tolerance.TIGHT,
            err_msg="Compustat non-outlier ret_exc should be unchanged",
        )

    def test_crsp_rows_unchanged_regardless_of_outliers(self, stock_df, cutoffs_path, tolerance):
        """CRSP rows should never be winsorized, even with outlier values."""
        result = add_cutoffs_and_winsorize(stock_df, cutoffs_path, ["eom"], "date").collect()
        crsp_outlier = result.filter(pl.col("id") == 1)
        np.testing.assert_allclose(
            crsp_outlier["ret_exc"].to_list(),
            [0.55],
            **tolerance.TIGHT,
            err_msg="CRSP outlier ret_exc should NOT be clipped",
        )

    def test_no_year_month_columns_in_output(self, stock_df, cutoffs_path):
        """Output should not contain year or month columns."""
        result = add_cutoffs_and_winsorize(stock_df, cutoffs_path, ["eom"], "date").collect()
        assert "year" not in result.columns, "'year' should not be in output"
        assert "month" not in result.columns, "'month' should not be in output"

    def test_missing_cutoff_rows_preserve_original_returns(self, tmp_path, tolerance):
        """When a stock's eom has no matching cutoff row, returns should be preserved unchanged.

        GREATEST/LEAST in Polars SQL skip NULLs, so missing cutoffs (NULL bounds
        from the LEFT JOIN) must not corrupt or NULL-out the original return values.
        """
        # Cutoffs only for Jan 2020 — no cutoffs for Feb 2020
        cutoffs = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31)],
                "n": [100],
                "ret_0_1": [-0.50],
                "ret_1": [-0.20],
                "ret_99": [0.20],
                "ret_99_9": [0.50],
                "ret_local_0_1": [-0.50],
                "ret_local_1": [-0.20],
                "ret_local_99": [0.20],
                "ret_local_99_9": [0.50],
                "ret_exc_0_1": [-0.40],
                "ret_exc_1": [-0.15],
                "ret_exc_99": [0.15],
                "ret_exc_99_9": [0.40],
            }
        )
        cutoffs_path = tmp_path / "cutoffs_missing.parquet"
        cutoffs.write_parquet(cutoffs_path)

        # Compustat stock in Feb 2020 (no matching cutoff row)
        df = pl.LazyFrame(
            {
                "id": [100001, 100002],
                "eom": [date(2020, 2, 29), date(2020, 2, 29)],
                "date": [date(2020, 2, 15), date(2020, 2, 15)],
                "source_crsp": [0, 0],
                "ret": [0.90, -0.80],
                "ret_local": [0.85, -0.75],
                "ret_exc": [0.80, -0.70],
            }
        )
        result = add_cutoffs_and_winsorize(df, str(cutoffs_path), ["eom"], "date").collect()

        np.testing.assert_allclose(
            result.sort("id")["ret"].to_list(),
            [0.90, -0.80],
            **tolerance.TIGHT,
            err_msg="Returns should be preserved when cutoff row is missing",
        )
        np.testing.assert_allclose(
            result.sort("id")["ret_exc"].to_list(),
            [0.80, -0.70],
            **tolerance.TIGHT,
            err_msg="Excess returns should be preserved when cutoff row is missing",
        )
        np.testing.assert_allclose(
            result.sort("id")["ret_local"].to_list(),
            [0.85, -0.75],
            **tolerance.TIGHT,
            err_msg="Local returns should be preserved when cutoff row is missing",
        )


# =============================================================================
# TestWinsorizeEquivalence (regression)
# =============================================================================


class TestWinsorizeEquivalence:
    """Proves that grouping by eom produces identical cutoffs as year/month.

    This is the key regression test: if eom and year/month produce the same
    quantile bounds, the simplification is safe.
    """

    @pytest.fixture
    def daily_returns(self, seed):
        """Generate daily return data spanning a year boundary and leap/non-leap Februaries.

        Covers Dec 2019 – Mar 2020 (leap year, Feb has 29 days) and
        Dec 2020 – Mar 2021 (non-leap year, Feb has 28 days).
        """
        np.random.seed(seed)
        dates = pl.concat(
            [
                pl.date_range(date(2019, 12, 2), date(2020, 3, 31), "1d", eager=True),
                pl.date_range(date(2020, 12, 1), date(2021, 3, 31), "1d", eager=True),
            ]
        )
        n = len(dates)
        return pl.DataFrame(
            {
                "date": dates,
                "ret_exc": np.random.randn(n) * 0.05,
            }
        ).with_columns(
            eom=pl.col("date").dt.month_end(),
            year=pl.col("date").dt.year(),
            month=pl.col("date").dt.month(),
        )

    def test_quantile_bounds_identical(self, daily_returns, tolerance):
        """QUANTILE_DISC grouped by eom must equal grouping by year+month."""
        by_eom = daily_returns.sql("""
            SELECT
                eom,
                QUANTILE_DISC(ret_exc, 0.001) AS low_eom,
                QUANTILE_DISC(ret_exc, 0.999) AS high_eom
            FROM self
            GROUP BY eom
            ORDER BY eom
        """)

        by_ym = daily_returns.sql("""
            SELECT
                year, month,
                QUANTILE_DISC(ret_exc, 0.001) AS low_ym,
                QUANTILE_DISC(ret_exc, 0.999) AS high_ym
            FROM self
            GROUP BY year, month
            ORDER BY year, month
        """)

        assert len(by_eom) == len(by_ym), (
            f"Group counts differ: {len(by_eom)} (eom) vs {len(by_ym)} (year/month)"
        )

        np.testing.assert_allclose(
            by_eom["low_eom"].to_numpy(),
            by_ym["low_ym"].to_numpy(),
            **tolerance.TIGHT,
            err_msg="Lower bounds differ between eom and year/month grouping",
        )
        np.testing.assert_allclose(
            by_eom["high_eom"].to_numpy(),
            by_ym["high_ym"].to_numpy(),
            **tolerance.TIGHT,
            err_msg="Upper bounds differ between eom and year/month grouping",
        )

    def test_group_count_matches(self, daily_returns):
        """eom grouping and year/month grouping must produce same number of groups."""
        n_eom = daily_returns.select("eom").n_unique()
        n_ym = daily_returns.select(["year", "month"]).unique().height
        assert n_eom == n_ym, f"Group counts differ: {n_eom} (eom) vs {n_ym} (year/month)"


# =============================================================================
# TestPortfolioDailyJoin (regression for portfolio.py)
# =============================================================================


class TestPortfolioDailyJoin:
    """Tests that the portfolio.py daily winsorization join works with eom.

    Replicates the join logic from portfolio.py to verify the eom-based
    approach produces correct winsorization.
    """

    @pytest.fixture
    def daily_data(self):
        """Mock daily returns with CRSP and Compustat stocks."""
        return pl.DataFrame(
            {
                "id": [1, 1, 100001, 100001],
                "date": [
                    date(2020, 1, 15),
                    date(2020, 1, 16),
                    date(2020, 1, 15),
                    date(2020, 1, 16),
                ],
                "ret_exc": [0.10, -0.05, 0.80, -0.70],
            }
        )

    @pytest.fixture
    def cutoffs_daily(self):
        """Mock daily return cutoffs keyed by eom."""
        return pl.DataFrame(
            {
                "eom": [date(2020, 1, 31)],
                "ret_exc_0_1": [-0.50],
                "ret_exc_99_9": [0.50],
            }
        )

    def test_compustat_outliers_clipped(self, daily_data, cutoffs_daily, tolerance):
        """Compustat (id > 99999) outliers should be winsorized via eom join."""
        daily = daily_data.with_columns(pl.col("date").dt.month_end().alias("eom"))
        daily = daily.join(
            cutoffs_daily.select(["eom", "ret_exc_0_1", "ret_exc_99_9"]).rename(
                {"ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}
            ),
            on="eom",
            how="left",
        )
        daily = daily.with_columns(
            pl.when((pl.col("id") > 99999) & (pl.col("ret_exc") > pl.col("p999")))
            .then(pl.col("p999"))
            .when((pl.col("id") > 99999) & (pl.col("ret_exc") < pl.col("p001")))
            .then(pl.col("p001"))
            .otherwise(pl.col("ret_exc"))
            .alias("ret_exc")
        ).drop(["p001", "p999", "eom"])

        comp_rows = daily.filter(pl.col("id") == 100001).sort("date")
        np.testing.assert_allclose(
            comp_rows["ret_exc"].to_list(),
            [0.50, -0.50],
            **tolerance.TIGHT,
            err_msg="Compustat outliers should be clipped to [-0.50, 0.50]",
        )

    def test_crsp_rows_unchanged(self, daily_data, cutoffs_daily, tolerance):
        """CRSP (id <= 99999) rows should not be winsorized."""
        daily = daily_data.with_columns(pl.col("date").dt.month_end().alias("eom"))
        daily = daily.join(
            cutoffs_daily.select(["eom", "ret_exc_0_1", "ret_exc_99_9"]).rename(
                {"ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}
            ),
            on="eom",
            how="left",
        )
        daily = daily.with_columns(
            pl.when((pl.col("id") > 99999) & (pl.col("ret_exc") > pl.col("p999")))
            .then(pl.col("p999"))
            .when((pl.col("id") > 99999) & (pl.col("ret_exc") < pl.col("p001")))
            .then(pl.col("p001"))
            .otherwise(pl.col("ret_exc"))
            .alias("ret_exc")
        ).drop(["p001", "p999", "eom"])

        crsp_rows = daily.filter(pl.col("id") == 1).sort("date")
        np.testing.assert_allclose(
            crsp_rows["ret_exc"].to_list(),
            [0.10, -0.05],
            **tolerance.TIGHT,
            err_msg="CRSP rows should not be modified",
        )

    def test_no_year_month_in_output(self, daily_data, cutoffs_daily):
        """Output should not contain year or month columns."""
        daily = daily_data.with_columns(pl.col("date").dt.month_end().alias("eom"))
        daily = daily.join(
            cutoffs_daily.select(["eom", "ret_exc_0_1", "ret_exc_99_9"]).rename(
                {"ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}
            ),
            on="eom",
            how="left",
        ).drop(["p001", "p999", "eom"])
        assert "year" not in daily.columns
        assert "month" not in daily.columns


# =============================================================================
# TestInlineWinsorization
# =============================================================================


class TestInlineWinsorization:
    """Tests for inline GREATEST/LEAST winsorization used in ap_factors and prep_data_factor_regs.

    Validates that the inline window function approach correctly clips outliers,
    preserves non-outliers, and supports custom thresholds.
    """

    @pytest.fixture
    def returns_df(self, seed):
        """Synthetic returns with known outliers across two months.

        Each month has 2000 rows from a normal distribution plus 2 extreme outliers,
        giving enough data for QUANTILE_DISC at 0.1%/99.9% to produce bounds
        below the outliers.
        """
        np.random.seed(seed)
        n_per_month = 2000

        # Month 1: tight distribution
        ret1 = np.random.randn(n_per_month) * 0.02
        dates1 = [date(2020, 1, 15)] * n_per_month
        eom1 = [date(2020, 1, 31)] * n_per_month

        # Month 2: wider distribution
        ret2 = np.random.randn(n_per_month) * 0.05
        dates2 = [date(2020, 2, 15)] * n_per_month
        eom2 = [date(2020, 2, 29)] * n_per_month

        # Combine and add extreme outliers at known positions
        ret_exc = np.concatenate([ret1, ret2])
        dates_all = dates1 + dates2
        eom_all = eom1 + eom2

        # Inject outliers: indices 0,1 in month 1; 2000,2001 in month 2
        ret_exc[0] = 5.0  # extreme high, month 1
        ret_exc[1] = -5.0  # extreme low, month 1
        ret_exc[2000] = 10.0  # extreme high, month 2
        ret_exc[2001] = -10.0  # extreme low, month 2

        return pl.LazyFrame(
            {
                "excntry": ["USA"] * (n_per_month * 2),
                "id": list(range(1, n_per_month * 2 + 1)),
                "eom": eom_all,
                "date": dates_all,
                "ret_exc": ret_exc.tolist(),
            }
        )

    @staticmethod
    def _winsorize_cte_greatest(lf, lower=0.001, upper=0.999):
        """Apply CTE+JOIN winsorization using GREATEST/LEAST (ap_factors pattern)."""
        return lf.sql(f"""
            WITH bounds AS (
                SELECT eom,
                    QUANTILE_DISC(ret_exc, {lower}) AS low,
                    QUANTILE_DISC(ret_exc, {upper}) AS high
                FROM self
                GROUP BY eom
            )
            SELECT excntry, id, self.eom, date,
                GREATEST(low, LEAST(ret_exc, high)) AS ret_exc
            FROM self
            LEFT JOIN bounds USING (eom)
        """)

    @staticmethod
    def _winsorize_cte_case(lf, lower=0.001, upper=0.999):
        """Apply CTE+JOIN winsorization using CASE (old ap_factors pattern)."""
        return lf.sql(f"""
            WITH bounds AS (
                SELECT eom,
                    QUANTILE_DISC(ret_exc, {lower}) AS low,
                    QUANTILE_DISC(ret_exc, {upper}) AS high
                FROM self
                GROUP BY eom
            )
            SELECT excntry, id, eom, date,
                CASE
                    WHEN ret_exc < low  THEN low
                    WHEN ret_exc > high THEN high
                    ELSE ret_exc
                END AS ret_exc
            FROM self
            LEFT JOIN bounds USING (eom)
        """)

    def test_outliers_clipped_to_bounds(self, returns_df, tolerance):
        """Extreme values should be clipped to the group's percentile bounds."""
        result = self._winsorize_cte_greatest(returns_df).collect()

        # The injected outliers should be clipped (their winsorized value < original)
        for outlier_id, orig_val in [(1, 5.0), (2001, 10.0)]:
            winsorized = result.filter(pl.col("id") == outlier_id)["ret_exc"][0]
            assert winsorized < orig_val, f"High outlier id={outlier_id} not clipped: {winsorized}"

        for outlier_id, orig_val in [(2, -5.0), (2002, -10.0)]:
            winsorized = result.filter(pl.col("id") == outlier_id)["ret_exc"][0]
            assert winsorized > orig_val, f"Low outlier id={outlier_id} not clipped: {winsorized}"

    def test_non_outliers_unchanged(self, returns_df, tolerance):
        """Values within bounds should remain unchanged after winsorization."""
        original = returns_df.collect().sort("id")
        result = self._winsorize_cte_greatest(returns_df).collect().sort("id")

        # Identify rows that were actually clipped (original != result)
        orig_vals = original["ret_exc"].to_numpy()
        result_vals = result["ret_exc"].to_numpy()
        unchanged_mask = np.isclose(orig_vals, result_vals, rtol=1e-10, atol=1e-12)

        # All unchanged rows should be exactly equal
        np.testing.assert_allclose(
            result_vals[unchanged_mask],
            orig_vals[unchanged_mask],
            **tolerance.TIGHT,
            err_msg="Non-outlier values should be unchanged",
        )
        # At least the 4 injected outliers should have been clipped
        assert np.sum(~unchanged_mask) >= 4, "Expected at least 4 clipped values"

    def test_custom_thresholds(self, returns_df):
        """Wider thresholds (5%/95%) should clip more aggressively."""
        narrow = self._winsorize_cte_greatest(returns_df, lower=0.001, upper=0.999).collect()
        wide = self._winsorize_cte_greatest(returns_df, lower=0.05, upper=0.95).collect()

        # Wider thresholds should produce a tighter range
        assert wide["ret_exc"].max() <= narrow["ret_exc"].max()
        assert wide["ret_exc"].min() >= narrow["ret_exc"].min()

    def test_per_group_winsorization(self, returns_df, tolerance):
        """Each eom group should have its own clipping bounds."""
        result = self._winsorize_cte_greatest(returns_df).collect()

        month1 = result.filter(pl.col("eom") == date(2020, 1, 31))
        month2 = result.filter(pl.col("eom") == date(2020, 2, 29))

        # Month 2 has wider distribution (0.05 vs 0.02 std), so bounds should differ
        range1 = month1["ret_exc"].max() - month1["ret_exc"].min()
        range2 = month2["ret_exc"].max() - month2["ret_exc"].min()
        assert range2 > range1, "Month 2 (wider distribution) should have a wider winsorized range"

    def test_single_eom_group(self):
        """Winsorization should work correctly with a single eom group."""
        np.random.seed(123)
        n = 2000
        ret_exc = np.random.randn(n) * 0.03
        ret_exc[0] = 2.0  # outlier
        lf = pl.LazyFrame(
            {
                "excntry": ["USA"] * n,
                "id": list(range(1, n + 1)),
                "eom": [date(2020, 1, 31)] * n,
                "date": [date(2020, 1, 15)] * n,
                "ret_exc": ret_exc.tolist(),
            }
        )
        result = self._winsorize_cte_greatest(lf).collect()
        winsorized = result.filter(pl.col("id") == 1)["ret_exc"][0]
        assert winsorized < 2.0, f"Outlier should be clipped: {winsorized}"

    def test_greatest_least_matches_case(self, returns_df, tolerance):
        """GREATEST/LEAST approach must produce identical results to the CASE approach."""
        greatest_result = self._winsorize_cte_greatest(returns_df).collect().sort("id")
        case_result = self._winsorize_cte_case(returns_df).collect().sort("id")

        np.testing.assert_allclose(
            greatest_result["ret_exc"].to_numpy(),
            case_result["ret_exc"].to_numpy(),
            **tolerance.TIGHT,
            err_msg="GREATEST/LEAST and CASE winsorization should produce identical results",
        )
