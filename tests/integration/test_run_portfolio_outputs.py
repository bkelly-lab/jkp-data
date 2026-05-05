"""Integration tests for ``run_portfolio`` output files.

Drives the orchestrator end-to-end against a synthetic input tree built from
``tests.unit.portfolio.conftest`` helpers and asserts the expected parquet/CSV
files are written under ``output_dir/processed/portfolios/``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import polars as pl
import pytest

from jkp.data.output_writer import configure_output_format
from jkp.data.portfolio import run_portfolio
from tests.unit.portfolio.conftest import (
    build_synthetic_data,
    make_cluster_labels,
    make_factor_details,
    make_multi_region_classification,
    patch_resource_readers,
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
    fd_path = make_factor_details(tmp_path, list(chars))
    cc_path = make_multi_region_classification(tmp_path)
    cl_path = make_cluster_labels(tmp_path, dict.fromkeys(chars, "cluster_x"))

    patch_resource_readers(monkeypatch)
    monkeypatch.setattr("jkp.data.paths.get_factor_details_path", lambda: fd_path)
    monkeypatch.setattr("jkp.data.paths.get_country_classification_path", lambda: cc_path)
    monkeypatch.setattr("jkp.data.paths.get_cluster_labels_path", lambda: cl_path)
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
            "ret_type": "vw_cap",
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
    def test_writes_pfs_and_hml_lms_parquets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)
        for name in ("pfs.parquet", "hml.parquet", "lms.parquet"):
            path = out / name
            assert path.exists(), f"missing {name}"
            assert pl.read_parquet(path).height > 0, f"{name} is empty"

    def test_writes_clusters_parquet(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        path = _portfolios_dir(tmp_path) / "clusters.parquet"
        assert path.exists()
        assert pl.read_parquet(path).height > 0

    def test_writes_country_factors_partition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch, countries=("USA", "FRA"))
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        country_dir = _portfolios_dir(tmp_path) / "country_factors"
        assert (country_dir / "USA.parquet").exists()
        assert (country_dir / "FRA.parquet").exists()

    def test_writes_regional_factors_partition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup(tmp_path, monkeypatch)
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        regional_dir = _portfolios_dir(tmp_path) / "regional_factors"
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
                    "ret_type": "vw_cap",
                    "country_excl": ["ZWE", "VEN"],
                    "country_weights": "market_cap",
                    "stocks_min": 1,
                    "months_min": 1,
                    "countries_min": 1,
                }
            },
        )
        run_portfolio(output_format="parquet", output_dir=tmp_path)

        out = _portfolios_dir(tmp_path)

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
