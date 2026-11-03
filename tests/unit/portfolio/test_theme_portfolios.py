"""Tests for themes versions and their theme portfolios.

* ``load_themes`` reads v0, v1.0 and edited themes files; ``theme_versions``
  finds the released versions.
* ``_theme_returns`` averages signed factors within subthemes and subtheme
  returns within themes. The v0 cases are the former cluster-portfolio tests:
  in v0 each theme is its own subtheme, so a theme is the mean of its factors,
  as the cluster portfolios were.
* One e2e test runs ``run_portfolio`` against synthetic data and checks the
  ``themes/v0/themes.parquet`` output.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from jkp.data.aux_functions import _theme_returns, load_themes, theme_versions
from jkp.data.config import PORTFOLIO_CHARS
from jkp.data.paths import _resource_path, get_themes_dir
from jkp.data.portfolio import _write_theme_versions, run_portfolio
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    build_synthetic_data,
    make_multi_region_classification,
    make_themes_dir,
)

# Tolerances reused from conftest constants.
TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def _aggregate_clusters(lms_returns: pl.DataFrame, cluster_labels: pl.DataFrame) -> pl.DataFrame:
    """Theme returns of v0-shaped themes, where each cluster is its own subtheme."""
    themes = cluster_labels.select("characteristic", theme="cluster", subtheme="cluster")
    return _theme_returns(lms_returns, themes, "eom")[1].rename({"theme": "cluster"})


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
    """v0 theme returns equal the former cluster portfolios."""

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

        # The unmapped characteristic ("c") is dropped; the "alpha" group
        # counts the two mapped characteristics.
        assert out["cluster"].to_list() == ["alpha"]
        assert out["n_factors"].to_list() == [2]
        assert out["n_subthemes"].to_list() == [1]

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

    def test_run_portfolio_writes_v0_themes_parquet(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        reset_output_writer: None,
    ) -> None:
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

        # Stub resource files: country_classification and the themes folder.
        # Patch the path-getters at jkp.data.paths so the local imports inside
        # run_portfolio resolve to these stubs.
        resources_dir = tmp_path / "resources"
        resources_dir.mkdir()
        themes_dir = make_themes_dir(
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

        monkeypatch.setattr("jkp.data.paths.get_themes_dir", lambda: themes_dir)
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

        themes_path = tmp_path / "processed" / "portfolios" / "themes" / "v0" / "themes.parquet"
        assert themes_path.exists(), f"missing {themes_path}"
        df = pl.read_parquet(themes_path)
        assert set(df["theme"]) == {"cluster_alpha", "cluster_beta"}
        assert {"theme", "eom", "n_factors", "ret_ew", "ret_vw", "ret_vw_cap"}.issubset(df.columns)


def _lms(rets: dict[str, float]) -> pl.DataFrame:
    """One country-month of signed factor returns, the same for ew, vw and vw_cap."""
    return pl.DataFrame(
        {
            "excntry": "USA",
            "characteristic": list(rets),
            "eom": date(2020, 1, 31),
            "ret_ew": list(rets.values()),
            "ret_vw": list(rets.values()),
            "ret_vw_cap": list(rets.values()),
        }
    )


class TestTwoLevelThemeReturns:
    def test_theme_is_mean_of_subtheme_means(self) -> None:
        # Subtheme A has three factors and B one; each subtheme counts equally.
        themes = pl.DataFrame(
            {
                "characteristic": ["a", "b", "c", "d"],
                "theme": "T",
                "subtheme": ["A", "A", "A", "B"],
            }
        )
        subthemes, theme = _theme_returns(
            _lms({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.6}), themes, "eom"
        )

        assert subthemes["subtheme"].to_list() == ["A", "B"]
        assert subthemes["ret_ew"].to_list() == pytest.approx([0.2, 0.6])
        assert subthemes["n_factors"].to_list() == [3, 1]
        row = theme.row(0, named=True)
        assert row["ret_vw"] == pytest.approx(0.4)  # not 0.3, the mean of the four factors
        assert (row["n_subthemes"], row["n_factors"]) == (2, 4)

    def test_null_factor_returns_are_skipped_but_counted(self) -> None:
        themes = pl.DataFrame({"characteristic": ["a", "b"], "theme": "T", "subtheme": "A"})
        lms = _lms({"a": 0.1, "b": 0.0}).with_columns(
            pl.when(pl.col("characteristic") == "b")
            .then(None)
            .otherwise(pl.col("ret_vw"))
            .alias("ret_vw")
        )
        subthemes, _ = _theme_returns(lms, themes, "eom")
        row = subthemes.row(0, named=True)
        assert (row["ret_vw"], row["n_factors"]) == (pytest.approx(0.1), 2)


class TestLoadThemes:
    def test_v0(self) -> None:
        v0 = load_themes(_resource_path("themes/superseded_themes/v0"))
        assert set(v0["characteristic"]) == set(PORTFOLIO_CHARS)
        assert v0["theme"].equals(v0["subtheme"], check_names=False)
        assert v0["theme"].n_unique() == 13
        directions = pl.read_excel(_resource_path("factor_details.xlsx"), sheet_name="details")
        directions = directions.filter(pl.col("abr_jkp").is_not_null()).select(
            characteristic="abr_jkp", factor_details="direction"
        )
        joined = v0.join(directions, on="characteristic")
        assert joined.height == 153
        assert (joined["direction"] == joined["factor_details"]).all()

    def test_v1_direction_is_product_of_signs(self) -> None:
        file = get_themes_dir() / "themes.csv"
        raw = pl.read_csv(file)
        themes = load_themes(file)
        assert themes.schema == pl.Schema(
            {
                "characteristic": pl.String,
                "theme": pl.String,
                "subtheme": pl.String,
                "direction": pl.Int32,
            }
        )
        assert (themes.height, themes["subtheme"].n_unique(), themes["theme"].n_unique()) == (
            402,
            59,
            16,
        )
        product = raw["subtheme_sign"] * raw["theme_sign"] * raw["factor_sign"]
        assert themes["direction"].cast(pl.Int64).equals(product, check_names=False)
        # kz_index buys constrained firms in v0 and unconstrained firms in v1.0.
        v0 = load_themes(get_themes_dir() / "superseded_themes" / "v0")
        changed = v0.join(themes, on="characteristic").filter(
            pl.col("direction") != pl.col("direction_right")
        )
        assert "kz_index" in changed["characteristic"]
        assert changed.height == 9

    def test_edited_file(self, tmp_path: Path) -> None:
        file = tmp_path / "mine.csv"
        pl.DataFrame(
            {
                "characteristic": ["be_me", "ret_12_1"],
                "theme": ["Value", "Momentum"],
                "subtheme": ["Cheapness", "Price momentum"],
                "subtheme_sign": [1, -1],
                "theme_sign": [1, 1],
                "factor_sign": [-1, -1],
            }
        ).write_csv(file)
        assert load_themes(file)["direction"].to_list() == [-1, 1]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"factor_sign": [1, 0]}, "signs other than -1 and 1: \\['b'\\]"),
            ({"characteristic": ["a", "a"]}, "characteristics listed more than once"),
            ({"theme": ["T", "U"]}, "subthemes under more than one theme: \\['S'\\]"),
        ],
    )
    def test_invalid_file(self, tmp_path: Path, change: dict, message: str) -> None:
        file = tmp_path / "mine.csv"
        base = {
            "characteristic": ["a", "b"],
            "theme": ["T", "T"],
            "subtheme": ["S", "S"],
            "subtheme_sign": [1, 1],
            "theme_sign": [1, 1],
            "factor_sign": [1, 1],
        }
        pl.DataFrame({**base, **change}).write_csv(file)
        with pytest.raises(ValueError, match=message):
            load_themes(file)

    def test_missing_columns(self, tmp_path: Path) -> None:
        file = tmp_path / "mine.csv"
        pl.DataFrame({"characteristic": ["a"], "theme": ["T"], "subtheme": ["S"]}).write_csv(file)
        with pytest.raises(ValueError, match="missing columns"):
            load_themes(file)

    def test_theme_versions(self, tmp_path: Path) -> None:
        released = theme_versions(get_themes_dir())
        assert released["v1.0"] == get_themes_dir() / "themes.csv"
        assert released["v0"] == get_themes_dir() / "superseded_themes" / "v0"
        # A superseded v1.0-format version is a folder holding its themes.csv.
        themes_dir = make_themes_dir(tmp_path, {"a": "c"})
        (themes_dir / "superseded_themes" / "v0.5").mkdir()
        pl.read_csv(themes_dir / "themes.csv").write_csv(
            themes_dir / "superseded_themes" / "v0.5" / "themes.csv"
        )
        versions = theme_versions(themes_dir)
        assert sorted(versions) == ["v0", "v0.5", "v1.0"]
        assert load_themes(versions["v0.5"]).equals(load_themes(versions["v1.0"]))

    def test_theme_versions_rejects_a_folder_named_like_the_current_version(
        self, tmp_path: Path
    ) -> None:
        themes_dir = make_themes_dir(tmp_path, {"a": "c"})
        (themes_dir / "superseded_themes" / "v1.0").mkdir()
        with pytest.raises(ValueError, match="superseded_themes/v1.0 has the version name"):
            theme_versions(themes_dir)


class TestWriteThemeVersions:
    def test_missing_factor_stops_before_writing(self, tmp_path: Path) -> None:
        # char "b" has a monthly factor but no daily one. The check runs before
        # anything is written, so the earlier build's folder is left as it was.
        themes = pl.DataFrame(
            {"characteristic": ["a", "b"], "theme": "T", "subtheme": "S", "direction": [1, 1]}
        )
        old = tmp_path / "themes" / "v" / "themes.parquet"
        old.parent.mkdir(parents=True)
        old.touch()
        with pytest.raises(ValueError, match=r"no hml factor for: v \(hml_daily\): \['b'\]"):
            _write_theme_versions(
                portfolios_dir=tmp_path,
                theme_sets={"v": themes},
                hml_series=[
                    ("", pl.DataFrame({"characteristic": ["a", "b"]}), "eom"),
                    ("_daily", pl.DataFrame({"characteristic": ["a"]}), "date"),
                ],
                market=pl.DataFrame(),
                market_daily=None,
                regions=pl.DataFrame(),
                settings={},
            )
        assert [p.name for p in old.parent.iterdir()] == ["themes.parquet"]
