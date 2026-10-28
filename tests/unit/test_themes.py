"""The bundled themes files are a valid two-level grouping with consistent signs and names."""

import polars as pl
import pytest

from jkp.data.config import PORTFOLIO_CHARS
from jkp.data.paths import _resource_path


@pytest.fixture(scope="module")
def themes() -> pl.DataFrame:
    return pl.read_csv(_resource_path("themes/themes.csv"))


@pytest.fixture(scope="module")
def descriptions() -> pl.DataFrame:
    return pl.read_csv(_resource_path("themes/theme_descriptions.csv"))


def not_constant(df: pl.DataFrame, col: str, within: str) -> list[str]:
    """Groups of `within` where `col` takes more than one value."""
    counts = df.group_by(within).agg(pl.col(col).n_unique())
    return sorted(counts.filter(pl.col(col) > 1)[within])


@pytest.mark.unit
def test_grouping_is_a_two_level_partition(themes):
    dup = sorted(set(themes.filter(pl.col("characteristic").is_duplicated())["characteristic"]))
    assert not dup, f"characteristics in more than one subtheme: {dup}"
    split = not_constant(themes, "theme", within="subtheme")
    assert not split, f"subthemes under more than one theme: {split}"


@pytest.mark.unit
def test_signs(themes):
    for col in ["subtheme_sign", "theme_sign", "factor_sign"]:
        assert themes[col].is_in([-1, 1]).all(ignore_nulls=False), f"{col} must be -1 or +1"
    bad = not_constant(themes, "theme_sign", within="subtheme")
    assert not bad, f"theme_sign differs within subthemes: {bad}"
    bad = not_constant(themes, "factor_sign", within="theme")
    assert not bad, f"factor_sign differs within themes: {bad}"


@pytest.mark.unit
def test_naming_rules(themes):
    names = set(themes["theme"]) | set(themes["subtheme"])
    assert not [n for n in names if " & " in n or "(" in n], "names use '&' or parentheses"
    n_subs = themes.group_by("theme").agg(pl.col("subtheme").n_unique())
    multi = set(n_subs.filter(pl.col("subtheme") > 1)["theme"])
    shared = set(themes.filter(pl.col("subtheme") == pl.col("theme"))["theme"]) & multi
    assert not shared, f"a subtheme shares its theme's name: {sorted(shared)}"


@pytest.mark.unit
def test_descriptions_match_grouping(themes, descriptions):
    assert descriptions["level"].is_in(["theme", "subtheme"]).all()
    assert descriptions["description"].str.strip_chars().str.len_chars().fill_null(0).min() > 0
    theme_rows = descriptions.filter(pl.col("level") == "theme")
    assert sorted(theme_rows["name"]) == sorted(themes["theme"].unique()), "theme rows differ"
    assert (theme_rows["name"] == theme_rows["theme"]).all()
    subtheme_rows = descriptions.filter(pl.col("level") == "subtheme").select("name", "theme")
    expected = themes.select(name="subtheme", theme="theme").unique()
    assert subtheme_rows.sort("name").equals(expected.sort("name")), "subtheme rows differ"


@pytest.mark.unit
def test_one_version(themes, descriptions):
    versions = set(themes["version"]) | set(descriptions["version"])
    assert len(versions) == 1, f"more than one version: {sorted(versions)}"


@pytest.mark.unit
def test_portfolio_characteristics_have_a_theme(themes):
    missing = sorted(set(PORTFOLIO_CHARS) - set(themes["characteristic"]))
    assert not missing, f"no theme for: {missing}"
