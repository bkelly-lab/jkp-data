"""Unit tests for ff_us_mktrf (French-style US market) and its use in ap_factor_model_data."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data import aux_functions
from jkp.data.aux_functions import ap_factor_model_data, ff_us_mktrf
from jkp.data.paths import DataPaths

pytestmark = pytest.mark.unit

JAN, FEB, MAR = date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)


@pytest.fixture(autouse=True)
def _no_exchanges(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_rf_and_exchange_data also builds the Compustat exchange map; not needed here."""
    monkeypatch.setattr(aux_functions, "comp_exchanges", lambda paths: pl.DataFrame())


def _stock(permno: int, permco: int, p: str, rows: list[tuple], **over: str) -> list[dict]:
    """Rows of (date, ret, prc, shrout) for one permno with CIZ universe flags."""
    flags = {
        "sharetype": "NS",
        "securitytype": "EQTY",
        "securitysubtype": "COM",
        "usincflg": "Y",
        "issuertype": "CORP",
        "primaryexch": "N",
    } | over
    return [
        {
            "permno": permno,
            "permco": permco,
            f"{p}caldt": d,
            f"{p}ret": r,
            f"{p}prc": prc,
            "shrout": so,
            **flags,
        }
        for d, r, prc, so in rows
    ]


def _write_rf(paths: DataPaths, mcti: list[tuple], ff: list[tuple]) -> None:
    raw = paths.interim_dir / "raw_data_dfs"
    pl.DataFrame(mcti, schema={"caldt": pl.Date, "t30ret": pl.Float64}, orient="row").write_parquet(
        raw / "crsp_mcti_t30ret.parquet"
    )
    pl.DataFrame(ff, schema={"date": pl.Date, "rf": pl.Float64}, orient="row").write_parquet(
        raw / "ff_factors_monthly.parquet"
    )


def _write_monthly(paths: DataPaths) -> None:
    rows = (
        # permno 1 and 2 share a permco: both count, permno-level weights me=10 and 20
        _stock(
            1, 10, "mth", [(JAN, 0.0, 10.0, 1000), (FEB, 0.10, 10.0, 1000), (MAR, 0.20, 10.0, 1000)]
        )
        + _stock(
            2, 10, "mth", [(JAN, 0.0, -20.0, 1000), (FEB, 0.0, 20.0, 1000), (MAR, 0.05, 20.0, 1000)]
        )
        # excluded: non-US incorporated, non-NS share type
        + _stock(3, 30, "mth", [(JAN, 0.0, 10.0, 1e6), (FEB, 5.0, 10.0, 1e6)], usincflg="N")
        + _stock(4, 40, "mth", [(JAN, 0.0, 10.0, 1e6), (FEB, 5.0, 10.0, 1e6)], sharetype="AD")
        # excluded: not NYSE/AMEX/NASDAQ
        + _stock(6, 60, "mth", [(JAN, 0.0, 10.0, 1e6), (FEB, 5.0, 10.0, 1e6)], primaryexch="X")
        # gap: February row missing, so March has no valid lagged weight
        + _stock(5, 50, "mth", [(JAN, 0.0, 10.0, 1000), (MAR, 0.9, 10.0, 1000)])
    )
    pl.DataFrame(rows).write_parquet(paths.raw_tables_dir / "crsp_msf_v2.parquet")


def test_monthly_universe_weights_gap_and_rf(test_paths: DataPaths) -> None:
    _write_monthly(test_paths)
    _write_rf(
        test_paths,
        mcti=[(FEB, 0.002), (MAR, None)],
        ff=[(FEB, 0.5), (MAR, 0.004)],
    )
    out = ff_us_mktrf(test_paths, "monthly").collect().sort("eom")
    assert out.columns == ["excntry", "eom", "mktrf"]
    assert out["excntry"].to_list() == ["USA", "USA"]
    # January has no lagged weight; February: (0.10*10 + 0*20)/30 - t30ret
    assert out["eom"].to_list() == [FEB, MAR]
    assert out["mktrf"].to_list() == pytest.approx([1 / 30 - 0.002, 0.1 - 0.004])


def test_daily_weights_and_rf_scaled_by_21(test_paths: DataPaths) -> None:
    d1, d2, d3 = date(2020, 2, 3), date(2020, 2, 4), date(2020, 2, 5)
    rows = _stock(
        1, 10, "dly", [(d1, 0.0, 10.0, 1000), (d2, 0.10, 12.0, 1000), (d3, 0.02, 12.0, 1000)]
    ) + _stock(2, 10, "dly", [(d1, 0.0, 20.0, 1000), (d2, 0.0, 20.0, 1000), (d3, 0.05, 20.0, 1000)])
    rows += _stock(3, 30, "dly", [(d1, 0.0, 10.0, 1e6), (d2, 5.0, 10.0, 1e6)], usincflg="N")
    pl.DataFrame(rows).write_parquet(test_paths.raw_tables_dir / "crsp_dsf_v2.parquet")
    _write_rf(test_paths, mcti=[(FEB, 0.021)], ff=[(FEB, 0.5)])
    out = ff_us_mktrf(test_paths, "daily").collect().sort("date")
    assert out.columns == ["excntry", "date", "mktrf"]
    assert out["date"].to_list() == [d2, d3]
    # d2 uses d1 me (10, 20); d3 uses d2 me (12, 20)
    mkt = [(0.10 * 10) / 30, (0.02 * 12 + 0.05 * 20) / 32]
    assert out["mktrf"].to_list() == pytest.approx([m - 0.001 for m in mkt])


def test_ap_factor_model_data_replaces_us_mktrf_only(test_paths: DataPaths) -> None:
    _write_monthly(test_paths)
    d1, d2 = date(2020, 2, 3), date(2020, 2, 4)
    rows = _stock(1, 10, "dly", [(d1, 0.0, 10.0, 1000), (d2, 0.10, 10.0, 1000)])
    pl.DataFrame(rows).write_parquet(test_paths.raw_tables_dir / "crsp_dsf_v2.parquet")
    _write_rf(test_paths, mcti=[(FEB, 0.0), (MAR, 0.0)], ff=[(FEB, 0.0), (MAR, 0.0)])
    interim = test_paths.interim_dir
    mkt_m = pl.DataFrame(
        {
            "excntry": ["USA", "USA", "GBR", "GBR"],
            "eom": [FEB, MAR, FEB, MAR],
            "mkt_vw_exc": [0.5, 0.5, 0.01, 0.02],
        }
    )
    mkt_m.write_parquet(interim / "market_returns.parquet")
    mkt_d = pl.DataFrame({"excntry": ["USA", "GBR"], "date": [d2, d2], "mkt_vw_exc": [0.5, 0.03]})
    mkt_d.write_parquet(interim / "market_returns_daily.parquet")
    fm = interim / "m_in.parquet"
    fd = interim / "d_in.parquet"
    fc = interim / "c_in.parquet"
    pl.DataFrame({"excntry": ["GBR"], "eom": [FEB], "hml_ff": [0.1]}).write_parquet(fm)
    pl.DataFrame({"excntry": ["GBR"], "date": [d2], "hml_ff_d": [0.1]}).write_parquet(fd)
    pl.DataFrame({"id": [1], "eom": [FEB], "x": [1.0]}).write_parquet(fc)
    ap_factor_model_data(test_paths, [fm], [fd], [fc])

    monthly = pl.read_parquet(interim / "ap_factors_monthly.parquet")
    gbr = monthly.filter(pl.col("excntry") == "GBR").sort("eom")
    assert gbr["mktrf"].to_list() == [0.01, 0.02]
    usa = monthly.filter(pl.col("excntry") == "USA").sort("eom")
    assert usa["eom"].to_list() == [FEB, MAR]
    assert 0.5 not in usa["mktrf"].to_list()
    assert usa["mktrf"].to_list() == pytest.approx([1 / 30, 0.1])

    daily = pl.read_parquet(interim / "ap_factors_daily.parquet")
    assert daily.filter(pl.col("excntry") == "GBR")["mktrf"].to_list() == [0.03]
    assert daily.filter(pl.col("excntry") == "USA")["mktrf"].to_list() == pytest.approx([0.10])
