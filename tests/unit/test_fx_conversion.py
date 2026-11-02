"""Tests for the FX table (compustat_fx) and the accounting FX conversion.

Compustat's FX rates start in February 1982, so compustat_fx extends CAD back to
January 1960 with bundled month-end rates. add_fx_and_convert_vars drops accounting
rows that have no FX rate on their datadate instead of keeping them with null values.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import add_fx_and_convert_vars, compustat_fx
from jkp.data.paths import DataPaths, get_fx_cad_backfill_path

CAD_SCHEMA = {"curcdd": pl.String, "datadate": pl.Date, "fx": pl.Float64}


def _backfill() -> pl.DataFrame:
    return pl.read_csv(get_fx_cad_backfill_path(), schema=CAD_SCHEMA)


def _write_fx1(paths: DataPaths, cad_start: date) -> None:
    """Compustat-style __fx1 with two CAD quotes from cad_start and one EUR quote."""
    pl.DataFrame(
        {
            "curcdd": ["CAD", "CAD", "EUR"],
            "datadate": [cad_start, date(1982, 2, 2), date(1999, 1, 1)],
            "fx": [0.5, 0.6, 1.1],
        },
        schema=CAD_SCHEMA,
    ).write_parquet(paths.interim_dir / "raw_data_dfs" / "__fx1.parquet")


def _rate(fx: pl.DataFrame, cur: str, day: date) -> float | None:
    out = fx.filter((pl.col("curcdd") == cur) & (pl.col("datadate") == day))["fx"]
    return out.item() if out.len() else None


class TestCadBackfillFile:
    def test_month_end_cad_rates_1960_to_1982(self):
        df = _backfill()
        assert df.height == 265
        assert df["curcdd"].unique().to_list() == ["CAD"]
        assert df["datadate"].min() == date(1960, 1, 31)
        assert df["datadate"].max() == date(1982, 1, 31)
        assert (df["datadate"] == df["datadate"].dt.month_end()).all()
        assert df["datadate"].is_unique().all()
        # USD per CAD stayed between 0.80 and 1.06 over 1960-1982.
        assert df["fx"].is_between(0.80, 1.06).all()


class TestCompustatFx:
    def test_cad_backfilled_before_compustat_start(self, test_paths: DataPaths):
        _write_fx1(test_paths, cad_start=date(1982, 2, 1))
        fx = compustat_fx(test_paths)
        bf = _backfill()

        # Every month-end in the file gets its own rate, and it carries to the next month-end.
        assert _rate(fx, "CAD", date(1960, 1, 31)) == bf["fx"][0]
        assert _rate(fx, "CAD", date(1965, 6, 15)) == _rate(fx, "CAD", date(1965, 5, 31))
        assert _rate(fx, "CAD", date(1982, 1, 31)) == bf["fx"][-1]
        # Compustat takes over on its first date.
        assert _rate(fx, "CAD", date(1982, 2, 1)) == 0.5
        assert _rate(fx, "CAD", date(1960, 1, 30)) is None
        # Other currencies are unchanged.
        assert _rate(fx, "EUR", date(1999, 1, 1)) == 1.1
        assert _rate(fx, "EUR", date(1998, 12, 31)) is None
        assert fx.select(pl.struct("curcdd", "datadate").is_unique().all()).item()

    def test_compustat_rates_win_where_they_exist(self, test_paths: DataPaths):
        _write_fx1(test_paths, cad_start=date(1975, 1, 1))
        fx = compustat_fx(test_paths)

        assert _rate(fx, "CAD", date(1974, 12, 31)) is not None
        assert _rate(fx, "CAD", date(1975, 1, 1)) == 0.5
        assert _rate(fx, "CAD", date(1980, 6, 30)) == 0.5
        assert fx.select(pl.struct("curcdd", "datadate").is_unique().all()).item()


class TestAddFxAndConvertVars:
    @pytest.fixture
    def fx_df(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "datadate": [date(2000, 12, 31), date(2000, 12, 31), date(2000, 12, 31)],
                "curcdd": ["USD", "CAD", "ITL"],
                "fx": [1.0, 0.5, 0.0005],
            }
        ).lazy()

    @pytest.mark.parametrize(("freq", "cur"), [("annual", "curcd"), ("quarterly", "curcdq")])
    def test_rows_without_fx_are_dropped(self, fx_df: pl.LazyFrame, freq: str, cur: str):
        df = pl.DataFrame(
            {
                "gvkey": ["1", "2", "3", "4", "5", "6"],
                "datadate": [date(2000, 12, 31)] * 5 + [date(2001, 12, 31)],
                # null currency, no FX series (ZWG), no rate on that date (ITL in 2001)
                cur: ["CAD", None, "USD", "ZWG", "ITL", "ITL"],
                "at": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        ).lazy()

        out = add_fx_and_convert_vars(df, fx_df, ["at"], freq).collect()

        assert out["gvkey"].to_list() == ["1", "3", "5"]
        assert out["at"].to_list() == [5.0, 30.0, 0.025]
        assert out[cur].to_list() == ["USD"] * 3
        assert out.columns == df.collect_schema().names()
