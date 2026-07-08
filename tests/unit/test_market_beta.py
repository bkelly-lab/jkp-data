"""Unit tests for market_beta (rolling CAPM beta + idiosyncratic vol).

These assert the *semantics* of market_beta on small hand-built panels (CAPM
math recovery, min-obs boundary, WHERE-filter gating, mktrf-null handling,
multi-country join, output schema/ordering, horizon parametrization, empty
input) — not golden bytes (see tests/golden/test_market_beta_golden.py for the
byte-level regression).
"""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import market_beta
from jkp.data.paths import DataPaths
from tests.golden.market_beta_inputs import (
    AP_FACTORS_MONTHLY_INPUT_SCHEMA,
    WORLD_MSF_INPUT_SCHEMA,
)


def _eom(i: int) -> date:
    """Month-end date ``i`` months after 2010-01-31."""
    year = 2010 + i // 12
    month = i % 12 + 1
    return date(year, month, calendar.monthrange(year, month)[1])


def _factor_df(rows: list[tuple[str, int, float | None]]) -> pl.DataFrame:
    """Build a factor frame from (excntry, month_index, mktrf) rows."""
    return pl.DataFrame(
        {
            "excntry": [r[0] for r in rows],
            "eom": [_eom(r[1]) for r in rows],
            "mktrf": [r[2] for r in rows],
            "hml": [0.01] * len(rows),
            "smb_ff": [0.01] * len(rows),
        },
        schema=AP_FACTORS_MONTHLY_INPUT_SCHEMA,
    )


def _msf_df(rows: list[dict]) -> pl.DataFrame:
    """Build a world_msf frame from a list of row dicts (id, excntry, i, ...)."""
    return pl.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "excntry": [r["excntry"] for r in rows],
            "eom": [_eom(r["i"]) for r in rows],
            "ret_exc": [r["ret_exc"] for r in rows],
            "ret_local": [
                r.get("ret_local", (r["ret_exc"] + 0.02) if r["ret_exc"] is not None else 0.02)
                for r in rows
            ],
            "ret_lag_dif": [r.get("ret_lag_dif", 1) for r in rows],
        },
        schema=WORLD_MSF_INPUT_SCHEMA,
    )


def _run(
    paths: DataPaths, msf: pl.DataFrame, fac: pl.DataFrame, n: int, min_obs: int
) -> pl.DataFrame:
    """Write inputs, run market_beta, return the output frame."""
    data_path = paths.interim_dir / "world_msf.parquet"
    fcts_path = paths.interim_dir / "ap_factors_monthly.parquet"
    out_path = paths.interim_dir / f"beta_{n}m.parquet"
    msf.write_parquet(data_path)
    fac.write_parquet(fcts_path)
    market_beta(paths, out_path, data_path, fcts_path, n, min_obs)
    return pl.read_parquet(out_path)


@pytest.mark.unit
def test_capm_math_recovery(test_paths, tolerance) -> None:
    """ret_exc = 2*mktrf exactly recovers beta=2.0 and ivol=0 at the full window."""
    mkt = np.random.default_rng(7).normal(0.0, 0.05, 12)
    fac = _factor_df([("USA", i, float(mkt[i])) for i in range(12)])
    msf = _msf_df(
        [{"id": 1, "excntry": "USA", "i": i, "ret_exc": float(2.0 * mkt[i])} for i in range(12)]
    )
    out = _run(test_paths, msf, fac, 12, 6)
    row = out.filter(pl.col("eom") == _eom(11))
    assert row.height == 1
    assert abs(row["beta_12m"][0] - 2.0) < 1e-9
    assert abs(row["ivol_capm_12m"][0]) < 1e-9


@pytest.mark.unit
def test_min_obs_boundary(test_paths) -> None:
    """Firm with exactly 6 obs is emitted; firm with 5 obs is dropped (min_obs=6)."""
    mkt = np.random.default_rng(1).normal(0.0, 0.05, 6)
    fac = _factor_df([("USA", i, float(mkt[i])) for i in range(6)])
    rows = [{"id": 1, "excntry": "USA", "i": i, "ret_exc": float(mkt[i])} for i in range(6)]
    rows += [{"id": 2, "excntry": "USA", "i": i, "ret_exc": float(mkt[i])} for i in range(5)]
    out = _run(test_paths, _msf_df(rows), fac, 12, 6)
    ids = set(out["id"].to_list())
    assert 1 in ids  # 6 obs -> present
    assert 2 not in ids  # 5 obs -> absent


@pytest.mark.unit
def test_where_filters_gate_obs_count(test_paths) -> None:
    """ret_local=0 / ret_lag_dif!=1 / ret_exc=null rows do not count toward min_obs."""
    mkt = np.random.default_rng(2).normal(0.0, 0.05, 9)
    fac = _factor_df([("USA", i, float(mkt[i])) for i in range(9)])

    def firm(fid: int, last: int) -> list[dict]:
        rows: list[dict] = []
        for i in range(last + 1):
            r: dict = {"id": fid, "excntry": "USA", "i": i, "ret_exc": float(mkt[i])}
            if i == 2:
                r["ret_local"] = 0.0
            elif i == 4:
                r["ret_lag_dif"] = 2
            elif i == 6:
                r["ret_exc"] = None
            rows.append(r)
        return rows

    # Firm 1: M0..M8 -> valid {0,1,3,5,7,8} = 6 -> present.
    # Firm 2: M0..M7 -> valid {0,1,3,5,7}   = 5 -> absent.
    out = _run(test_paths, _msf_df(firm(1, 8) + firm(2, 7)), fac, 12, 6)
    ids = set(out["id"].to_list())
    assert 1 in ids
    assert 2 not in ids


@pytest.mark.unit
def test_mktrf_null_factor_month_drops_obs(test_paths) -> None:
    """A factor month with mktrf=null removes that obs from the window count."""
    mkt = np.random.default_rng(3).normal(0.0, 0.05, 6)
    msf = _msf_df([{"id": 1, "excntry": "USA", "i": i, "ret_exc": float(mkt[i])} for i in range(6)])
    full = _factor_df([("USA", i, float(mkt[i])) for i in range(6)])
    holed = _factor_df([("USA", i, None if i == 2 else float(mkt[i])) for i in range(6)])

    present = _run(test_paths, msf, full, 12, 6)
    assert 1 in set(present["id"].to_list())  # 6 obs -> present

    absent = _run(test_paths, msf, holed, 12, 6)
    assert 1 not in set(absent["id"].to_list())  # 5 obs after null -> absent


@pytest.mark.unit
def test_multi_country_join(test_paths) -> None:
    """Each firm's beta is computed against its own country's mktrf series."""
    rng = np.random.default_rng(4)
    mkt_us = rng.normal(0.0, 0.05, 12)
    mkt_ca = rng.normal(0.0, 0.05, 12)
    fac = _factor_df(
        [("USA", i, float(mkt_us[i])) for i in range(12)]
        + [("CAN", i, float(mkt_ca[i])) for i in range(12)]
    )
    msf = _msf_df(
        [{"id": 1, "excntry": "USA", "i": i, "ret_exc": float(1.5 * mkt_us[i])} for i in range(12)]
        + [
            {"id": 2, "excntry": "CAN", "i": i, "ret_exc": float(0.5 * mkt_ca[i])}
            for i in range(12)
        ]
    )
    out = _run(test_paths, msf, fac, 12, 6)
    us = out.filter((pl.col("id") == 1) & (pl.col("eom") == _eom(11)))
    ca = out.filter((pl.col("id") == 2) & (pl.col("eom") == _eom(11)))
    assert abs(us["beta_12m"][0] - 1.5) < 1e-9
    assert abs(ca["beta_12m"][0] - 0.5) < 1e-9


@pytest.mark.unit
def test_output_schema_and_ordering(test_paths) -> None:
    """Output is exactly [id, eom, beta_12m, ivol_capm_12m] sorted by (id, eom)."""
    mkt = np.random.default_rng(5).normal(0.0, 0.05, 8)
    fac = _factor_df([("USA", i, float(mkt[i])) for i in range(8)])
    rows = [
        {"id": fid, "excntry": "USA", "i": i, "ret_exc": float(fid * 0.1 * mkt[i] + mkt[i])}
        for fid in (20, 10)
        for i in range(8)
    ]
    out = _run(test_paths, _msf_df(rows), fac, 12, 6)
    assert out.columns == ["id", "eom", "beta_12m", "ivol_capm_12m"]
    assert out.schema == {
        "id": pl.Int64,
        "eom": pl.Date,
        "beta_12m": pl.Float64,
        "ivol_capm_12m": pl.Float64,
    }
    assert out.select(["id", "eom"]).equals(out.select(["id", "eom"]).sort(["id", "eom"]))


@pytest.mark.unit
def test_horizon_parametrization(test_paths) -> None:
    """__n/__min are honored: columns are named beta_6m / ivol_capm_6m."""
    mkt = np.random.default_rng(6).normal(0.0, 0.05, 8)
    fac = _factor_df([("USA", i, float(mkt[i])) for i in range(8)])
    msf = _msf_df(
        [{"id": 1, "excntry": "USA", "i": i, "ret_exc": float(0.9 * mkt[i])} for i in range(8)]
    )
    out = _run(test_paths, msf, fac, 6, 3)
    assert out.columns == ["id", "eom", "beta_6m", "ivol_capm_6m"]
    assert out.height > 0


@pytest.mark.unit
def test_empty_input(test_paths) -> None:
    """0-row world_msf yields a 0-row output with the correct schema."""
    empty = pl.DataFrame(schema=WORLD_MSF_INPUT_SCHEMA)
    fac = _factor_df([("USA", i, 0.01 * (i + 1)) for i in range(6)])
    out = _run(test_paths, empty, fac, 12, 6)
    assert out.height == 0
    assert out.columns == ["id", "eom", "beta_12m", "ivol_capm_12m"]
