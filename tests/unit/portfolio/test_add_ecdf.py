"""Unit tests for ``add_ecdf`` in ``portfolio.py``."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest  # noqa: F401

from jkp.data.portfolio import add_ecdf

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

    def test_empty_bp_stock_group_yields_zero_cdf(self):
        """When ``bp_stock`` is all False for a group, every row gets cdf=0.0."""
        df = self._simple_frame(
            var_vals=[1.0, 2.0, 3.0],
            bp_stock=[False, False, False],
        )
        out = add_ecdf(df)
        assert out["cdf"].to_list() == [0.0, 0.0, 0.0]

    def test_idempotence_on_double_call(self):
        """Calling ``add_ecdf`` twice (dropping ``cdf`` between) matches single call."""
        df = self._simple_frame(
            var_vals=[1.0, 2.0, 3.0, 2.5],
            bp_stock=[True, True, True, False],
        )
        single = add_ecdf(df).sort(["eom", "var"])
        double = add_ecdf(add_ecdf(df).drop("cdf")).sort(["eom", "var"])
        np.testing.assert_allclose(
            single["cdf"].to_list(),
            double["cdf"].to_list(),
            rtol=1e-12,
        )

    def test_cdf_in_unit_interval_on_breakpoint_sample(self):
        """For any ``bp_stock=True`` row, ``0 < cdf <= 1``."""
        df = self._simple_frame(
            var_vals=[1.0, 2.0, 3.0, 4.0, 5.0],
            bp_stock=[True, True, True, True, True],
        )
        out = add_ecdf(df)
        bp_cdfs = out.filter(pl.col("bp_stock"))["cdf"].to_list()
        assert all(0.0 < c <= 1.0 for c in bp_cdfs)
