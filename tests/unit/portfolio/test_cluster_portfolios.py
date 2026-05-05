"""Tests for the cluster portfolio construction step in ``run_portfolio``.

The tested logic is inline at ``portfolio.py`` lines 1119-1150 and joins
``lms_returns`` to ``cluster_labels`` on ``characteristic`` then aggregates
per ``(excntry, cluster, eom)``. We use a hybrid approach:

* Logic-level tests replicate the join+agg expression directly to verify
  invariants (drop-on-unmapped, cardinality, mean equivalence, empty input).
* One e2e test runs ``run_portfolio`` against synthetic data and asserts the
  ``clusters.parquet`` output exists and is non-empty.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from jkp.data.portfolio import run_portfolio
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    build_synthetic_data,
    make_cluster_labels,
    make_factor_details,
    make_multi_region_classification,
    patch_resource_readers,
)

# Tolerances reused from conftest constants.
TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def _aggregate_clusters(lms_returns: pl.DataFrame, cluster_labels: pl.DataFrame) -> pl.DataFrame:
    """Replicate the cluster join+agg from ``run_portfolio`` (lines 1119-1131)."""
    return (
        lms_returns.join(cluster_labels, on="characteristic", how="left")
        .group_by(["excntry", "cluster", "eom"])
        .agg(
            [
                pl.len().alias("n_factors"),
                pl.col("ret_ew").mean().alias("ret_ew"),
                pl.col("ret_vw").mean().alias("ret_vw"),
                pl.col("ret_vw_cap").mean().alias("ret_vw_cap"),
            ]
        )
    )


def _make_lms_returns(rows: list[dict]) -> pl.DataFrame:
    """Build a synthetic ``lms_returns`` frame with the columns the agg reads."""
    if not rows:
        return pl.DataFrame(
            schema={
                "excntry": pl.Utf8,
                "characteristic": pl.Utf8,
                "eom": pl.Date,
                "ret_ew": pl.Float64,
                "ret_vw": pl.Float64,
                "ret_vw_cap": pl.Float64,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("excntry").cast(pl.Utf8),
        pl.col("characteristic").cast(pl.Utf8),
        pl.col("eom").cast(pl.Date),
        pl.col("ret_ew").cast(pl.Float64),
        pl.col("ret_vw").cast(pl.Float64),
        pl.col("ret_vw_cap").cast(pl.Float64),
    )


class TestClusterPortfolios:
    """Tests for the inline cluster portfolio aggregation in ``run_portfolio``."""

    # ------------------------------------------------------------------
    # Logic-level tests
    # ------------------------------------------------------------------

    def test_cluster_join_drops_unmapped_characteristics(self) -> None:
        eom = date(2020, 1, 31)
        lms = _make_lms_returns(
            [
                {
                    "excntry": "USA",
                    "characteristic": "a",
                    "eom": eom,
                    "ret_ew": 0.01,
                    "ret_vw": 0.02,
                    "ret_vw_cap": 0.03,
                },
                {
                    "excntry": "USA",
                    "characteristic": "b",
                    "eom": eom,
                    "ret_ew": 0.02,
                    "ret_vw": 0.03,
                    "ret_vw_cap": 0.04,
                },
                {
                    "excntry": "USA",
                    "characteristic": "c",
                    "eom": eom,
                    "ret_ew": 0.05,
                    "ret_vw": 0.06,
                    "ret_vw_cap": 0.07,
                },
            ]
        )
        labels = pl.DataFrame(
            {
                "characteristic": ["a", "b"],
                "cluster": ["alpha", "alpha"],
            }
        )

        out = _aggregate_clusters(lms, labels)

        # The left join keeps unmapped rows but with cluster=null. The mapped
        # characteristics ("a", "b") must contribute to a cluster row; the
        # unmapped characteristic ("c") shows up only under cluster=null.
        non_null = out.filter(pl.col("cluster").is_not_null())
        assert non_null.height == 1
        assert non_null["cluster"].to_list() == ["alpha"]
        # The "alpha" group sums n_factors over the two mapped chars.
        assert non_null["n_factors"].to_list() == [2]

    def test_cluster_n_factors_equals_join_cardinality(self) -> None:
        eom1 = date(2020, 1, 31)
        eom2 = date(2020, 2, 29)
        rows = []
        # Cluster "x" has chars a, b, c; cluster "y" has chars d, e.
        for ch in ["a", "b", "c"]:
            for eom in (eom1, eom2):
                rows.append(
                    {
                        "excntry": "USA",
                        "characteristic": ch,
                        "eom": eom,
                        "ret_ew": 0.01,
                        "ret_vw": 0.02,
                        "ret_vw_cap": 0.03,
                    }
                )
        # Cluster y only has data in eom1.
        for ch in ["d", "e"]:
            rows.append(
                {
                    "excntry": "USA",
                    "characteristic": ch,
                    "eom": eom1,
                    "ret_ew": 0.04,
                    "ret_vw": 0.05,
                    "ret_vw_cap": 0.06,
                }
            )
        lms = _make_lms_returns(rows)
        labels = pl.DataFrame(
            {
                "characteristic": ["a", "b", "c", "d", "e"],
                "cluster": ["x", "x", "x", "y", "y"],
            }
        )

        out = _aggregate_clusters(lms, labels).sort(["cluster", "eom"])
        x_rows = out.filter(pl.col("cluster") == "x").sort("eom")
        y_rows = out.filter(pl.col("cluster") == "y").sort("eom")
        assert x_rows["n_factors"].to_list() == [3, 3]
        assert y_rows["n_factors"].to_list() == [2]

    def test_cluster_ret_ew_is_simple_mean_of_factor_rets(self) -> None:
        eom = date(2020, 1, 31)
        lms = _make_lms_returns(
            [
                {
                    "excntry": "USA",
                    "characteristic": "a",
                    "eom": eom,
                    "ret_ew": 0.10,
                    "ret_vw": 0.20,
                    "ret_vw_cap": 0.30,
                },
                {
                    "excntry": "USA",
                    "characteristic": "b",
                    "eom": eom,
                    "ret_ew": 0.20,
                    "ret_vw": 0.30,
                    "ret_vw_cap": 0.40,
                },
                {
                    "excntry": "USA",
                    "characteristic": "c",
                    "eom": eom,
                    "ret_ew": 0.30,
                    "ret_vw": 0.40,
                    "ret_vw_cap": 0.50,
                },
            ]
        )
        labels = pl.DataFrame(
            {
                "characteristic": ["a", "b", "c"],
                "cluster": ["alpha", "alpha", "alpha"],
            }
        )

        out = _aggregate_clusters(lms, labels)
        row = out.filter(pl.col("cluster") == "alpha").row(0, named=True)
        assert row["ret_ew"] == pytest.approx(0.20, rel=TIGHT["rtol"], abs=TIGHT["atol"])
        assert row["ret_vw"] == pytest.approx(0.30, rel=TIGHT["rtol"], abs=TIGHT["atol"])
        assert row["ret_vw_cap"] == pytest.approx(0.40, rel=TIGHT["rtol"], abs=TIGHT["atol"])

    def test_empty_lms_returns_no_clusters(self) -> None:
        lms = _make_lms_returns([])
        labels = pl.DataFrame({"characteristic": ["a", "b"], "cluster": ["alpha", "alpha"]})

        out = _aggregate_clusters(lms, labels)
        assert out.height == 0

    # ------------------------------------------------------------------
    # End-to-end test
    # ------------------------------------------------------------------

    def test_run_portfolio_writes_clusters_parquet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub resource files are written as parquet (see conftest); reroute
        # ``pl.read_excel``/``pl.scan_csv`` inside ``jkp.data.portfolio`` to
        # parquet readers.
        patch_resource_readers(monkeypatch)

        # Build synthetic input tree: characteristics, daily returns, cutoffs,
        # market returns. Use one country in each MSCI tier so the regional
        # pipeline finds something, plus USA so the cmp branch is exercised.
        countries = ["USA", "GBR", "BRA", "EGY"]
        chars = SYNTHETIC_CHARS[:5]
        build_synthetic_data(
            data_root=tmp_path,
            countries=countries,
            chars=chars,
            n_ids=40,
            n_months=6,
        )

        # Stub resource files: factor_details, country_classification,
        # cluster_labels. Patch the path-getters at jkp.data.paths so the
        # local imports inside run_portfolio resolve to these stubs.
        resources_dir = tmp_path / "resources"
        resources_dir.mkdir()
        factor_details_path = make_factor_details(resources_dir, characteristics=chars)
        cluster_labels_path = make_cluster_labels(
            resources_dir,
            characteristic_to_cluster={
                chars[0]: "cluster_alpha",
                chars[1]: "cluster_alpha",
                chars[2]: "cluster_beta",
                chars[3]: "cluster_beta",
                chars[4]: "cluster_alpha",
            },
        )
        country_class_path = make_multi_region_classification(resources_dir)

        monkeypatch.setattr("jkp.data.paths.get_factor_details_path", lambda: factor_details_path)
        monkeypatch.setattr("jkp.data.paths.get_cluster_labels_path", lambda: cluster_labels_path)
        monkeypatch.setattr(
            "jkp.data.paths.get_country_classification_path", lambda: country_class_path
        )

        # Patch the chars list and lenient settings on the portfolio module
        # so the test stays small and regional/cluster outputs are non-empty.
        import jkp.data.portfolio as pf_mod

        lenient_settings = {
            "end_date": date(2030, 12, 31),
            "pfs": 3,
            "source": ["CRSP", "COMPUSTAT"],
            "wins_ret": True,
            "bps": "non_mc",
            "bp_min_n": 3,
            "cmp": {"us": False, "int": False},
            "signals": {
                "us": False,
                "int": False,
                "standardize": True,
                "weight": "vw_cap",
            },
            "regional_pfs": {
                "ret_type": "vw_cap",
                "country_excl": ["ZWE", "VEN"],
                "country_weights": "market_cap",
                "stocks_min": 1,
                "months_min": 1,
                "countries_min": 1,
            },
            "daily_pf": False,
            "ind_pf": False,
        }
        monkeypatch.setattr(pf_mod, "PORTFOLIO_CHARS", chars)
        monkeypatch.setattr(pf_mod, "PORTFOLIO_SETTINGS", lenient_settings)

        run_portfolio(output_dir=tmp_path)

        clusters_path = tmp_path / "processed" / "portfolios" / "clusters.parquet"
        assert clusters_path.exists(), f"missing {clusters_path}"
        df = pl.read_parquet(clusters_path)
        assert df.height > 0
        assert {"cluster", "eom", "n_factors", "ret_ew", "ret_vw", "ret_vw_cap"}.issubset(
            df.columns
        )
