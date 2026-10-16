"""Equivalence test for the single-pass overnight/intraday portfolio build (#284).

The legacy ``run_portfolio`` reran ``portfolios()`` once per overnight/intraday
component. The current one aggregates every daily return column in the main
pass. Both run end-to-end on the same synthetic inputs and every file under
``processed/portfolios/`` must match: same files, schemas and rows, exact
integer columns, and float columns within tight tolerance.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.output_writer import configure_output_format
from jkp.data.portfolio import run_portfolio
from tests.fixtures import portfolio_oi_legacy
from tests.unit.portfolio.conftest import (
    build_synthetic_data,
    make_cluster_labels,
    make_factor_details,
    make_multi_region_classification,
    patch_resource_readers,
)

pytestmark = pytest.mark.integration

COUNTRIES = ["DEU", "GBR", "USA"]
CHARS = ["char_a", "char_b", "char_c"]
OI_COLS = ["ret_overnight", "ret_intraday", "ret_overnight_local", "ret_intraday_local"]
COMPUSTAT_ID_OFFSET = 200_000
SETTINGS = {
    "end_date": _dt.date(2030, 12, 31),
    "pfs": 3,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "bps": "non_mc",
    "bp_min_n": 5,
    "cmp": {"us": True, "int": False},
    "signals": {"us": False, "int": False, "standardize": True, "weight": "vw_cap"},
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


def _add_oi_columns(processed: Path, excntry: str, seed: int) -> None:
    """Give one country's inputs Compustat ids and O/I return columns with edge cases.

    Description:
        Compustat stocks (``source_crsp == 0``) get ids above 99999 so daily
        winsorization applies to them. The daily file gets the four O/I
        columns with: stock-days where only ``ret_exc`` is present, a date
        where every O/I return is null, stock-days where only the O/I returns
        are present, extra nulls in the local legs only, and extreme Compustat
        returns beyond the daily cutoffs. DEU has the columns but all null.
    """
    char_path = processed / "characteristics" / f"{excntry}.parquet"
    daily_path = processed / "return_data" / "daily_rets_by_country" / f"{excntry}.parquet"
    chars = pl.read_parquet(char_path)
    comp_ids = chars.filter(pl.col("source_crsp") == 0)["id"].unique()
    remap = pl.when(pl.col("id").is_in(comp_ids.implode())).then(pl.col("id") + COMPUSTAT_ID_OFFSET)
    chars.with_columns(remap.otherwise(pl.col("id")).alias("id")).write_parquet(char_path)

    daily = pl.read_parquet(daily_path)
    daily = daily.with_columns(remap.otherwise(pl.col("id")).alias("id"))
    n = daily.height
    rng = np.random.default_rng(seed)
    if excntry == "DEU":
        daily = daily.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in OI_COLS)
        daily.write_parquet(daily_path)
        return

    first_date = daily["date"].min()
    no_open = pl.Series(rng.random(n) < 0.10) | (daily["date"] == first_date)
    no_close = pl.Series(rng.random(n) < 0.03)
    local_extra_null = pl.Series(rng.random(n) < 0.05)
    extreme = pl.Series(rng.random(n) < 0.05) & (daily["id"] > 99999)
    daily = (
        daily.with_columns(
            [pl.Series(c, rng.normal(0.0002, 0.01, n)) for c in OI_COLS],
            no_open=no_open,
            no_close=no_close,
            local_extra_null=local_extra_null,
            extreme=extreme,
        )
        .with_columns(
            pl.when(pl.col("extreme")).then(0.5).otherwise(pl.col("ret_exc")).alias("ret_exc"),
            *[pl.when(pl.col("extreme")).then(-0.4).otherwise(pl.col(c)).alias(c) for c in OI_COLS],
        )
        .with_columns(
            pl.when(~pl.col("no_close")).then(pl.col("ret_exc")).alias("ret_exc"),
            *[pl.when(~pl.col("no_open")).then(pl.col(c)).alias(c) for c in OI_COLS],
        )
        .with_columns(
            pl.when(~pl.col("local_extra_null")).then(pl.col(c)).alias(c)
            for c in ("ret_overnight_local", "ret_intraday_local")
        )
        .drop("no_open", "no_close", "local_extra_null", "extreme")
    )
    daily.write_parquet(daily_path)


def _build_inputs(root: Path) -> None:
    build_synthetic_data(root, COUNTRIES, CHARS, n_ids=80, n_months=10)
    for i, ex in enumerate(COUNTRIES):
        _add_oi_columns(root / "processed", ex, seed=7 + i)


def _patch(monkeypatch: pytest.MonkeyPatch, resources: Path) -> None:
    fd_path = make_factor_details(resources, CHARS)
    cc_path = make_multi_region_classification(resources)
    cl_path = make_cluster_labels(resources, {"char_a": "c1", "char_b": "c1", "char_c": "c2"})
    patch_resource_readers(monkeypatch)
    monkeypatch.setattr("jkp.data.paths.get_factor_details_path", lambda: fd_path)
    monkeypatch.setattr("jkp.data.paths.get_country_classification_path", lambda: cc_path)
    monkeypatch.setattr("jkp.data.paths.get_cluster_labels_path", lambda: cl_path)

    def _force_reset(format_str: str, *, _allow_reset: bool = False) -> None:
        configure_output_format(format_str, _allow_reset=True)

    for mod in ("jkp.data.portfolio", "tests.fixtures.portfolio_oi_legacy"):
        monkeypatch.setattr(f"{mod}.PORTFOLIO_CHARS", CHARS)
        monkeypatch.setattr(f"{mod}.PORTFOLIO_SETTINGS", SETTINGS)
        monkeypatch.setattr(f"{mod}.configure_output_format", _force_reset)
    configure_output_format("parquet", _allow_reset=True)


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    keys = [c for c, t in df.schema.items() if not t.is_float()]
    return df.sort(keys, nulls_last=True)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Run the legacy and the single-pass builds on identical inputs."""
    mp = pytest.MonkeyPatch()
    legacy_root = tmp_path_factory.mktemp("legacy")
    new_root = tmp_path_factory.mktemp("new")
    _build_inputs(legacy_root)
    _build_inputs(new_root)
    _patch(mp, tmp_path_factory.mktemp("resources"))
    try:
        portfolio_oi_legacy.run_portfolio(output_format="parquet", output_dir=legacy_root)
        run_portfolio(output_format="parquet", output_dir=new_root)
    finally:
        mp.undo()
    return legacy_root / "processed" / "portfolios", new_root / "processed" / "portfolios"


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*.parquet")}


class TestOiSinglePassEquivalence:
    def test_same_files(self, outputs: tuple[Path, Path]) -> None:
        legacy, new = outputs
        assert _files(new) == _files(legacy)

    def test_oi_outputs_written(self, outputs: tuple[Path, Path]) -> None:
        _, new = outputs
        files = _files(new)
        for component in ("overnight", "intraday", "overnight_local", "intraday_local"):
            for name in ("pfs", "hml", "lms", "clusters"):
                assert f"{name}_{component}.parquet" in files
            assert any(f.startswith(f"regional_factors_{component}/") for f in files)

    def test_all_outputs_match(self, outputs: tuple[Path, Path], tolerance) -> None:
        legacy, new = outputs
        for rel in sorted(_files(legacy)):
            a = pl.read_parquet(legacy / rel)
            b = pl.read_parquet(new / rel)
            assert b.schema == a.schema, rel
            assert b.height == a.height, rel
            assert_frame_equal(_sorted(b), _sorted(a), check_exact=False, **tolerance.TIGHT), rel

    def test_edge_cases_present(self, outputs: tuple[Path, Path]) -> None:
        """Guard that the fixture exercises the per-leg null handling."""
        _, new = outputs
        std = pl.read_parquet(new / "pfs_daily.parquet")
        ov = pl.read_parquet(new / "pfs_overnight.parquet")
        ov_local = pl.read_parquet(new / "pfs_overnight_local.parquet")
        assert ov.filter(pl.col("excntry") == "DEU").height == 0
        assert ov.height < std.height
        assert ov_local["n"].sum() < ov["n"].sum()
        assert (ov["ret_ew"] != 0).all()
