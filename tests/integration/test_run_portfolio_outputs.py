"""Integration tests for ``run_portfolio`` output files.

Drives the orchestrator end-to-end against a synthetic input tree built from
``tests.unit.portfolio.conftest`` helpers and asserts the expected parquet/CSV
files are written under ``output_dir/processed/portfolios/``.
"""

from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

import polars as pl
import pytest

import jkp.data.portfolio as portfolio_module
from jkp.data.output_writer import configure_output_format
from jkp.data.portfolio import _read_saved, run_portfolio, run_themes
from tests.unit.portfolio.conftest import (
    build_synthetic_data,
    make_multi_region_classification,
    make_themes_dir,
)


@pytest.fixture(autouse=True)
def _allow_format_reconfigure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Always permit reconfiguring the output-format singleton inside ``run_portfolio``."""

    def _force_reset(format_str: str, *, _allow_reset: bool = False) -> None:
        configure_output_format(format_str, _allow_reset=True)

    monkeypatch.setattr("jkp.data.portfolio.configure_output_format", _force_reset)
    configure_output_format("parquet", _allow_reset=True)


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    countries: tuple[str, ...] = ("USA", "FRA"),
    chars: tuple[str, ...] = ("char_a", "char_b"),
    settings_overrides: dict | None = None,
) -> Path:
    build_synthetic_data(tmp_path, list(countries), list(chars))
    cc_path = make_multi_region_classification(tmp_path)
    themes_dir = make_themes_dir(tmp_path, dict.fromkeys(chars, "cluster_x"))

    monkeypatch.setattr("jkp.data.paths.get_country_classification_path", lambda: cc_path)
    monkeypatch.setattr("jkp.data.paths.get_themes_dir", lambda: themes_dir)
    monkeypatch.setattr("jkp.data.portfolio.PORTFOLIO_CHARS", list(chars))

    base_settings = {
        "end_date": _dt.date(2030, 12, 31),
        "pfs": 3,
        "source": ["CRSP", "COMPUSTAT"],
        "wins_ret": True,
        "bps": "non_mc",
        "bp_min_n": 5,
        "cmp": {"us": True, "int": False},
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
        "daily_pf": True,
        "ind_pf": True,
    }
    if settings_overrides:
        base_settings.update(settings_overrides)
    monkeypatch.setattr("jkp.data.portfolio.PORTFOLIO_SETTINGS", base_settings)
    return tmp_path


def _portfolios_dir(tmp_path: Path) -> Path:
    return tmp_path / "processed" / "portfolios"


class TestRunPortfolioOutputs:
    def test_writes_pfs_hml_and_only_the_current_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)
        for name in (
            "pfs.parquet",
            "hml.parquet",
            "themes/v1.0/lms.parquet",
        ):
            path = out / name
            assert path.exists(), f"missing {name}"
            assert pl.read_parquet(path).height > 0, f"{name} is empty"
        assert not (out / "lms.parquet").exists()
        assert [p.name for p in (out / "themes").iterdir()] == ["v1.0"]

    def test_writes_theme_portfolios(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path) / "themes"
        for name in ("v1.0/themes.parquet", "v1.0/subthemes.parquet"):
            assert pl.read_parquet(out / name).height > 0, f"{name} is empty"
        assert list((out / "v1.0" / "regional_subthemes").glob("*.parquet"))

    def test_writes_country_factors_partition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch, countries=("USA", "FRA"))
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        country_dir = _portfolios_dir(tmp_path) / "themes" / "v1.0" / "country_factors"
        assert (country_dir / "USA.parquet").exists()
        assert (country_dir / "FRA.parquet").exists()

    def test_null_country_file_is_not_a_country(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # save_main_data and save_daily_ret write stocks with a null excntry to
        # null_country.parquet; they must not get country portfolios.
        _setup(tmp_path, monkeypatch, countries=("USA", "FRA", "null_country"))
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)
        assert not (out / "themes" / "v1.0" / "country_factors" / "null_country.parquet").exists()
        assert set(pl.read_parquet(out / "pfs.parquet")["excntry"].unique()) == {"USA", "FRA"}

    def test_writes_regional_factors_partition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        regional_dir = _portfolios_dir(tmp_path) / "themes" / "v1.0" / "regional_factors"
        assert regional_dir.exists()
        files = list(regional_dir.glob("*.parquet"))
        assert files, "no regional_factors parquets written"

    def test_writes_industry_files_when_ind_pf_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)
        assert (out / "industry_gics.parquet").exists()
        assert (out / "industry_ff49.parquet").exists()

    def test_no_industry_files_when_ind_pf_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch, settings_overrides={"ind_pf": False})
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)
        assert not (out / "industry_gics.parquet").exists()
        assert not (out / "industry_ff49.parquet").exists()
        assert not (out / "industry_gics_daily.parquet").exists()
        assert not (out / "industry_ff49_daily.parquet").exists()

    def test_no_daily_files_when_daily_pf_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch, settings_overrides={"daily_pf": False})
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)
        daily_files = list(out.rglob("*_daily.parquet"))
        assert not daily_files, f"unexpected daily files: {daily_files}"

    def test_end_date_filter_applied(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cutoff = _dt.date(2020, 4, 30)
        _setup(tmp_path, monkeypatch, settings_overrides={"end_date": cutoff})
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        path = _portfolios_dir(tmp_path) / "pfs.parquet"
        df = pl.read_parquet(path)
        assert df.height > 0
        assert df["eom"].max() <= cutoff

    def test_output_format_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="csv", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)
        csv_files = list(out.rglob("*.csv"))
        assert csv_files, "no csv files produced"
        # All parquet outputs should have been converted away.
        assert not list(out.rglob("*.parquet"))

    def test_country_excl_drops_zwe_from_regional_aggregation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ZWE is present in the synthetic input, classification, and per-country
        # output, but the regional aggregation must not count it because
        # country_excl=["ZWE","VEN"] removes it from country_classification.
        _setup(
            tmp_path,
            monkeypatch,
            countries=("USA", "GBR", "DEU", "ZWE"),
            settings_overrides={
                "regional_pfs": {
                    "country_excl": ["ZWE", "VEN"],
                    "country_weights": "market_cap",
                    "stocks_min": 1,
                    "months_min": 1,
                    "countries_min": 1,
                }
            },
        )
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path) / "themes" / "v1.0"

        # ZWE survives in per-country outputs (country_excl does not gate them).
        assert (out / "country_factors" / "ZWE.parquet").exists()

        # Regional outputs must not count ZWE: n_countries reflects the
        # excluded set. With 4 input countries and ZWE excluded, n_countries
        # in any region containing ZWE must be ≤ 3.
        regional_files = list((out / "regional_factors").glob("*.parquet"))
        assert regional_files, "no regional_factors files written"
        for f in regional_files:
            df = pl.read_parquet(f)
            if df.height == 0:
                continue
            assert df["n_countries"].max() <= 3, (
                f"{f.name}: n_countries exceeds USA+GBR+DEU max of 3 → ZWE leaked in"
            )


def _files(root: Path, suffix: str = "parquet") -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob(f"*.{suffix}")}


def _assert_cmp_signed(signed: pl.DataFrame, unsigned: pl.DataFrame) -> None:
    """Each signed cmp row is the unsigned row times its direction, in the same order."""
    keys = ["characteristic", "size_grp", "eom"]
    assert signed.select(keys).equals(unsigned.select(keys))
    joined = signed.join(unsigned, on=keys, suffix="_unsigned", validate="1:1")
    for col in ("ret_weighted", "signal_weighted"):
        expected = joined.select((pl.col(f"{col}_unsigned") * pl.col("direction")).alias(col))
        assert joined.select(col).equals(expected), col


class TestRunThemes:
    def test_no_option_rebuilds_only_the_current_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        themes_dir = _portfolios_dir(tmp_path) / "themes"
        expected = {rel: pl.read_parquet(themes_dir / rel) for rel in _files(themes_dir)}
        for path in themes_dir.rglob("*.parquet"):
            path.unlink()

        run_themes(output_dir=tmp_path)

        assert _files(themes_dir) == set(expected)
        assert [p.name for p in themes_dir.iterdir()] == ["v1.0"]
        for rel, df in expected.items():
            assert pl.read_parquet(themes_dir / rel).equals(df), rel

    def test_builds_v0_when_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        out = _portfolios_dir(tmp_path) / "themes"
        v1 = {rel: pl.read_parquet(out / "v1.0" / rel) for rel in _files(out / "v1.0")}

        run_themes(output_dir=tmp_path, version="v0")

        for name in ("lms.parquet", "themes.parquet", "lms_daily.parquet"):
            assert pl.read_parquet(out / "v0" / name).height > 0, f"{name} is empty"
        assert list((out / "v0" / "regional_themes").glob("*.parquet"))
        # Every v0 theme is its own subtheme, so v0 has no subtheme files.
        assert not (out / "v0" / "subthemes.parquet").exists()
        # Building v0 leaves the current version's folder as it was.
        for rel, df in v1.items():
            assert pl.read_parquet(out / "v1.0" / rel).equals(df), rel

    def test_cmp_per_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        run_themes(output_dir=tmp_path, version="v0")
        portfolios = _portfolios_dir(tmp_path)
        unsigned = pl.read_parquet(portfolios / "hml_cmp_5.parquet")
        assert unsigned.height > 0
        keys = ["characteristic", "size_grp", "eom"]

        assert unsigned.equals(unsigned.sort("excntry", *keys))
        v0 = pl.read_parquet(portfolios / "themes" / "v0" / "cmp.parquet")
        assert v0.equals(unsigned)
        assert not (portfolios / "themes" / "v0" / "lms_cmp_5.parquet").exists()

        signed = pl.read_parquet(portfolios / "themes" / "v1.0" / "lms_cmp_5.parquet")
        directions = pl.read_parquet(portfolios / "themes" / "v1.0" / "lms.parquet").select(
            "characteristic", "direction"
        )
        assert set(directions["direction"]) == {-1, 1}
        assert (
            signed.select("characteristic", "direction")
            .unique()
            .sort("characteristic")
            .equals(directions.unique().sort("characteristic"))
        )
        _assert_cmp_signed(signed, unsigned)

    def test_cmp_per_version_from_saved_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="csv", output_dir=tmp_path)
        run_themes(output_dir=tmp_path, version="v0")
        run_themes(output_dir=tmp_path)
        portfolios = _portfolios_dir(tmp_path)
        unsigned = pl.read_csv(portfolios / "hml_cmp_5.csv")
        assert unsigned.height > 0
        v0 = pl.read_csv(portfolios / "themes" / "v0" / "cmp.csv")
        assert "direction" not in v0.columns
        assert v0.equals(unsigned)
        _assert_cmp_signed(pl.read_csv(portfolios / "themes" / "v1.0" / "lms_cmp_5.csv"), unsigned)

    def test_cmp_off_removes_an_earlier_cmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        portfolios = _portfolios_dir(tmp_path)
        _read_saved(portfolios / "hml_cmp_5").write_csv(portfolios / "hml_cmp_5.csv")

        monkeypatch.setitem(portfolio_module.PORTFOLIO_SETTINGS, "cmp", {"us": False, "int": False})
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        run_themes(output_dir=tmp_path)

        assert not (portfolios / "hml_cmp_5.parquet").exists()
        assert not (portfolios / "hml_cmp_5.csv").exists()
        assert not (portfolios / "themes" / "v1.0" / "lms_cmp_5.parquet").exists()

    @pytest.mark.parametrize("output_format", ["parquet", "csv"])
    def test_cmp_is_found_in_the_other_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format=output_format, output_dir=tmp_path)
        portfolios = _portfolios_dir(tmp_path)
        cmp = _read_saved(portfolios / "hml_cmp_5")
        (portfolios / f"hml_cmp_5.{output_format}").unlink()
        if output_format == "parquet":
            cmp.write_csv(portfolios / "hml_cmp_5.csv")
        else:
            cmp.write_parquet(portfolios / "hml_cmp_5.parquet")

        run_themes(output_dir=tmp_path)

        assert (portfolios / "themes" / "v1.0" / f"lms_cmp_5.{output_format}").exists()

    def test_cmp_file_is_not_read_as_an_hml_series(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        run_themes(output_dir=tmp_path)

        out = _portfolios_dir(tmp_path) / "themes" / "v1.0"
        assert not list(out.glob("themes_cmp*"))
        assert "ret_weighted" in pl.read_parquet(out / "lms_cmp_5.parquet").columns

    def test_edited_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        edited = tmp_path / "my_themes.csv"
        (
            pl.read_csv(tmp_path / "themes" / "themes.csv")
            .filter(pl.col("characteristic") == "char_a")
            .with_columns(factor_sign=-1)
            .write_csv(edited)
        )

        run_themes(output_dir=tmp_path, file=edited)

        out = _portfolios_dir(tmp_path) / "themes"
        mine = pl.read_parquet(out / "my_themes" / "lms.parquet")
        released = pl.read_parquet(out / "v1.0" / "lms.parquet").filter(
            pl.col("characteristic") == "char_a"
        )
        assert set(mine["characteristic"]) == {"char_a"}
        assert mine["ret_ew"].equals(-released["ret_ew"])
        mine_cmp = pl.read_parquet(out / "my_themes" / "lms_cmp_5.parquet")
        released_cmp = pl.read_parquet(out / "v1.0" / "lms_cmp_5.parquet").filter(
            pl.col("characteristic") == "char_a"
        )
        assert set(mine_cmp["characteristic"]) == {"char_a"}
        assert mine_cmp["ret_weighted"].equals(-released_cmp["ret_weighted"])
        assert mine_cmp["signal_weighted"].equals(-released_cmp["signal_weighted"])

    def test_one_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch, settings_overrides={"daily_pf": False})
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        out = _portfolios_dir(tmp_path) / "themes"
        (out / "v1.0" / "lms.parquet").unlink()
        stale = out / "v0" / "subthemes.parquet"  # v0 has no subthemes
        stale.parent.mkdir()
        stale.touch()

        run_themes(output_dir=tmp_path, version="v0")

        assert not stale.exists()
        assert (out / "v0" / "lms.parquet").exists()
        assert not (out / "v1.0" / "lms.parquet").exists()
        with pytest.raises(ValueError, match="unknown themes version"):
            run_themes(output_dir=tmp_path, version="v9")

    def test_rejects_released_name_and_missing_factor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch, settings_overrides={"daily_pf": False})
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        themes = pl.read_csv(tmp_path / "themes" / "themes.csv")
        themes.write_csv(tmp_path / "v0.csv")
        with pytest.raises(ValueError, match="released themes version"):
            run_themes(output_dir=tmp_path, file=tmp_path / "v0.csv")
        themes.with_columns(characteristic=pl.lit("char_z")).head(1).write_csv(
            tmp_path / "typo.csv"
        )
        with pytest.raises(ValueError, match="char_z"):
            run_themes(output_dir=tmp_path, file=tmp_path / "typo.csv")
        with pytest.raises(ValueError, match="not a .csv file"):
            run_themes(output_dir=tmp_path, file=tmp_path)
        themes.write_csv(tmp_path / "V0.csv")
        with pytest.raises(ValueError, match="released themes version"):
            run_themes(output_dir=tmp_path, file=tmp_path / "V0.csv")
        themes.write_csv(tmp_path / "...csv")
        with pytest.raises(ValueError, match="file name must start"):
            run_themes(output_dir=tmp_path, file=tmp_path / "...csv")

    def test_ignores_unknown_hml_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)
        portfolios = _portfolios_dir(tmp_path)
        shutil.copy(portfolios / "hml.parquet", portfolios / "hml_backup.parquet")
        run_themes(output_dir=tmp_path)

    def test_csv_conversion_runs_when_themes_step_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)

        def fail(**_: object) -> None:
            raise ValueError("themes failed")

        monkeypatch.setattr("jkp.data.portfolio._write_theme_versions", fail)
        with pytest.raises(ValueError, match="themes failed"):
            run_portfolio(output_format="csv", output_dir=tmp_path)
        assert (_portfolios_dir(tmp_path) / "hml.csv").exists()
        assert not list((tmp_path / "processed").rglob("*.parquet"))

    def test_csv_output_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="csv", output_dir=tmp_path)
        themes_dir = _portfolios_dir(tmp_path) / "themes"
        expected = _files(themes_dir, "csv")
        for path in themes_dir.rglob("*.csv"):
            path.unlink()

        run_themes(output_dir=tmp_path)

        assert _files(themes_dir, "csv") == expected
        assert not list(tmp_path.rglob("*.parquet"))

        run_themes(output_dir=tmp_path, version="v0")

        assert (themes_dir / "v0" / "lms.csv").exists()
        assert not list(tmp_path.rglob("*.parquet"))
