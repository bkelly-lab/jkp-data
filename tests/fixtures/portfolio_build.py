"""Inputs and patches for running the whole portfolio build in tests.

``build_oi_inputs`` writes synthetic inputs with overnight/intraday returns, and
``patch_oi_build`` points ``run_portfolio`` and ``run_themes`` at stub resources for
them. ``output_files`` lists the files a build wrote. Used by the O/I equivalence, the
determinism and the output-file tests.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from jkp.data.output_writer import configure_output_format
from tests.unit.portfolio.conftest import (
    build_synthetic_data,
    make_multi_region_classification,
    make_themes_dir,
)

OI_COUNTRIES = ["DEU", "GBR", "USA"]
OI_CHARS = ["char_a", "char_b", "char_c"]
OI_COLS = ["ret_overnight", "ret_intraday", "ret_overnight_local", "ret_intraday_local"]
OI_COMPUSTAT_ID_OFFSET = 200_000
OI_SETTINGS = {
    "end_date": date(2030, 12, 31),
    "pfs": 3,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "bps": "non_mc",
    "bp_min_n": 5,
    "cmp": {"us": True, "int": False},
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


def add_oi_columns(processed: Path, excntry: str, seed: int) -> None:
    """Give one country's inputs Compustat ids and O/I return columns with edge cases.

    Description:
        Compustat stocks (``source_crsp == 0``) get ids above 99999 so daily
        winsorization applies to them. The daily file gets the four O/I
        columns with: stock-days where only ``ret_exc`` is present, a date
        where every O/I return is null, stock-days where only the O/I returns
        are present, extra nulls in the local overnight leg only, and extreme
        Compustat returns beyond the daily cutoffs. ``ret_intraday_local`` equals
        ``ret_intraday``, as in the production builders. DEU has the columns but
        all null.
    """
    char_path = processed / "characteristics" / f"{excntry}.parquet"
    daily_path = processed / "return_data" / "daily_rets_by_country" / f"{excntry}.parquet"
    chars = pl.read_parquet(char_path)
    comp_ids = chars.filter(pl.col("source_crsp") == 0)["id"].unique()
    remap = pl.when(pl.col("id").is_in(comp_ids.implode())).then(
        pl.col("id") + OI_COMPUSTAT_ID_OFFSET
    )
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
            pl.when(~pl.col("local_extra_null"))
            .then(pl.col("ret_overnight_local"))
            .alias("ret_overnight_local"),
            ret_intraday_local=pl.col("ret_intraday"),
        )
        .drop("no_open", "no_close", "local_extra_null", "extreme")
    )
    daily.write_parquet(daily_path)


def build_oi_inputs(root: Path) -> None:
    """Write the synthetic inputs with overnight/intraday returns under ``root/processed/``."""
    build_synthetic_data(root, OI_COUNTRIES, OI_CHARS, n_ids=80, n_months=10)
    for i, ex in enumerate(OI_COUNTRIES):
        add_oi_columns(root / "processed", ex, seed=7 + i)


def allow_output_format_reset(format_str: str, *, _allow_reset: bool = False) -> None:
    """``configure_output_format`` that lets one process run several builds."""
    configure_output_format(format_str, _allow_reset=True)


def patch_oi_build(monkeypatch: pytest.MonkeyPatch, resources: Path, settings: dict) -> Path:
    """Point ``run_portfolio`` and ``run_themes`` at stub resources for the O/I inputs.

    Writes the country classification and themes under ``resources`` and returns the
    themes folder.
    """
    cc_path = make_multi_region_classification(resources)
    themes_dir = make_themes_dir(resources, {"char_a": "c1", "char_b": "c1", "char_c": "c2"})
    monkeypatch.setattr("jkp.data.paths.get_country_classification_path", lambda: cc_path)
    monkeypatch.setattr("jkp.data.paths.get_themes_dir", lambda: themes_dir)
    monkeypatch.setattr("jkp.data.portfolio.PORTFOLIO_CHARS", OI_CHARS)
    monkeypatch.setattr("jkp.data.portfolio.PORTFOLIO_SETTINGS", settings)
    monkeypatch.setattr("jkp.data.portfolio.configure_output_format", allow_output_format_reset)
    configure_output_format("parquet", _allow_reset=True)
    return themes_dir


def output_files(root: Path, suffix: str = "parquet") -> set[str]:
    """The ``*.suffix`` files under ``root``, as posix paths relative to it."""
    return {p.relative_to(root).as_posix() for p in root.rglob(f"*.{suffix}")}
