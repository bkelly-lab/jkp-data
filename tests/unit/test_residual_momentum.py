"""Unit tests for ``residual_momentum`` (aux_functions).

These assert SEMANTICS (min-obs gating, both horizon variants, the hml-null
branch, empty input, and a numpy OLS value oracle), not fixture bytes. Inputs
are the shared synthetic builders in ``tests.golden.residual_momentum_inputs``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import residual_momentum
from tests.golden.residual_momentum_inputs import (
    build_ap_factors_monthly_input,
    build_world_msf_input,
    month_grid,
)

_STEM = "resmom_ff3"
_N = 36
_MIN = 24
_TERMINAL_IDX = 39  # m39 = 2016-04-30, terminal month of the fixture grid.
_CHUNK_START_IDX = 4  # 36-month chunk ending at m39 spans m4..m39.


def _write_inputs(paths, world_df: pl.DataFrame | None = None) -> tuple:
    """Write world_msf + ap_factors inputs to interim_dir; return their paths."""
    world = world_df if world_df is not None else build_world_msf_input(seed=42)
    world_path = paths.interim_dir / "world_msf.parquet"
    fcts_path = paths.interim_dir / "ap_factors_monthly.parquet"
    world.write_parquet(world_path)
    build_ap_factors_monthly_input(seed=42).write_parquet(fcts_path)
    return world_path, fcts_path


def _run(paths, incl: int, skip: int, world_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Run residual_momentum for one variant and return the sorted output frame."""
    world_path, fcts_path = _write_inputs(paths, world_df)
    residual_momentum(paths, _STEM, world_path, fcts_path, _N, _MIN, incl, skip)
    return pl.read_parquet(paths.interim_dir / f"{_STEM}_{incl}_{skip}.parquet").sort(["id", "eom"])


def _oracle_resff3(sid: int, country: str, incl: int, skip: int) -> float:
    """Numpy OLS oracle for the terminal-m39 chunk of a single stock.

    Description:
        Replicate ``resff3_{incl}_{skip}`` for the 36-month chunk ending at m39.
    Steps:
        1) Slice ret_exc / mktrf / hml / smb_ff over m4..m39 for (sid, country).
        2) OLS ``ret_exc ~ 1 + mktrf + hml + smb_ff``; residuals = y - X@beta.
        3) Average residuals over aux_date in (T-incl, T-skip]; divide by std(ddof=1).
    Output:
        Expected float value at (sid, m39).
    """
    world = build_world_msf_input(seed=42)
    fcts = build_ap_factors_monthly_input(seed=42)
    grid = month_grid()
    aux = [g.year * 12 + g.month for g in grid]
    terminal_aux = aux[_TERMINAL_IDX]

    chunk_idx = list(range(_CHUNK_START_IDX, _TERMINAL_IDX + 1))
    w = world.filter(pl.col("id") == sid)
    ret_by_eom = dict(zip(w["eom"].to_list(), w["ret_exc"].to_list(), strict=True))
    fc = fcts.filter(pl.col("excntry") == country)
    fac_by_eom = {
        e: (m, h, s)
        for e, m, h, s in zip(
            fc["eom"].to_list(),
            fc["mktrf"].to_list(),
            fc["hml"].to_list(),
            fc["smb_ff"].to_list(),
            strict=True,
        )
    }

    ys, xs, auxs = [], [], []
    for i in chunk_idx:
        eom = grid[i]
        m, h, s = fac_by_eom[eom]
        if eom not in ret_by_eom or h is None or s is None or m is None:
            continue  # mirrors res_mom dropping hml/smb-null rows and prep mktrf gate.
        ys.append(ret_by_eom[eom])
        xs.append([1.0, m, h, s])
        auxs.append(aux[i])

    y = np.array(ys)
    x = np.array(xs)
    auxs = np.array(auxs)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    res = y - x @ beta

    mask = (auxs > terminal_aux - incl) & (auxs <= terminal_aux - skip)
    window = res[mask]
    return float(window.mean() / window.std(ddof=1))


@pytest.mark.unit
@pytest.mark.parametrize(("incl", "skip"), [(12, 1), (6, 1)])
def test_output_schema_and_sort(test_paths, incl: int, skip: int) -> None:
    """Both variants emit [id, eom, resff3_incl_skip] with correct dtypes, sorted."""
    out = _run(test_paths, incl, skip)
    value_col = f"resff3_{incl}_{skip}"
    assert out.columns == ["id", "eom", value_col]
    assert out.schema["id"] == pl.Int64
    assert out.schema["eom"] == pl.Date
    assert out.schema[value_col] == pl.Float64
    assert out.sort(["id", "eom"]).equals(out)


@pytest.mark.unit
def test_value_correctness_12_1(test_paths, tolerance) -> None:
    """id=1 terminal-m39 value for the 12/1 horizon matches the numpy OLS oracle."""
    out = _run(test_paths, 12, 1)
    row = out.filter((pl.col("id") == 1) & (pl.col("eom") == month_grid()[_TERMINAL_IDX]))
    assert row.height == 1
    expected = _oracle_resff3(1, "US", 12, 1)
    np.testing.assert_allclose(row["resff3_12_1"][0], expected, **tolerance.STANDARD)


@pytest.mark.unit
def test_value_correctness_6_1(test_paths, tolerance) -> None:
    """id=1 terminal-m39 value for the 6/1 horizon matches the numpy OLS oracle.

    Same 36-month regression window as 12/1, only the averaging sub-window differs.
    """
    out = _run(test_paths, 6, 1)
    row = out.filter((pl.col("id") == 1) & (pl.col("eom") == month_grid()[_TERMINAL_IDX]))
    assert row.height == 1
    expected = _oracle_resff3(1, "US", 6, 1)
    np.testing.assert_allclose(row["resff3_6_1"][0], expected, **tolerance.STANDARD)


@pytest.mark.unit
@pytest.mark.parametrize(("incl", "skip"), [(12, 1), (6, 1)])
def test_min_obs_exclusion(test_paths, incl: int, skip: int) -> None:
    """id=2 (only 20 months) never reaches __min=24 obs → absent from both outputs."""
    out = _run(test_paths, incl, skip)
    assert 2 not in out["id"].to_list()


@pytest.mark.unit
def test_min_obs_boundary(test_paths) -> None:
    """id=3 (exactly 24 months) passes the __min=24 gate → present at terminal m39."""
    out = _run(test_paths, 12, 1)
    row = out.filter(pl.col("id") == 3)
    assert row.height >= 1
    assert month_grid()[_TERMINAL_IDX] in row["eom"].to_list()
    assert row["resff3_12_1"].is_finite().all()


@pytest.mark.unit
def test_hml_null_branch(test_paths) -> None:
    """id=4 (CA, hml null at m20) still produces finite output; US id=1 is unaffected."""
    out = _run(test_paths, 12, 1)
    ca = out.filter(pl.col("id") == 4)
    assert ca.height >= 1
    assert ca["resff3_12_1"].is_finite().all()
    # The CA-only null must not leak into the US oracle value for id=1.
    us_row = out.filter((pl.col("id") == 1) & (pl.col("eom") == month_grid()[_TERMINAL_IDX]))
    np.testing.assert_allclose(
        us_row["resff3_12_1"][0], _oracle_resff3(1, "US", 12, 1), rtol=1e-6, atol=1e-10
    )


@pytest.mark.unit
def test_horizons_distinct(test_paths) -> None:
    """resff3_12_1 != resff3_6_1 for id=1 at m39 (different averaging windows)."""
    out12 = _run(test_paths, 12, 1)
    out6 = _run(test_paths, 6, 1)
    terminal = month_grid()[_TERMINAL_IDX]
    v12 = out12.filter((pl.col("id") == 1) & (pl.col("eom") == terminal))["resff3_12_1"][0]
    v6 = out6.filter((pl.col("id") == 1) & (pl.col("eom") == terminal))["resff3_6_1"][0]
    assert v12 != pytest.approx(v6)


@pytest.mark.unit
@pytest.mark.parametrize(("incl", "skip"), [(12, 1), (6, 1)])
def test_empty_input(test_paths, incl: int, skip: int) -> None:
    """Empty world_msf → output exists, 0 rows, correct columns/dtypes, no crash."""
    empty_world = build_world_msf_input(empty=True)
    out = _run(test_paths, incl, skip, world_df=empty_world)
    assert out.height == 0
    assert out.columns == ["id", "eom", f"resff3_{incl}_{skip}"]
    assert out.schema["id"] == pl.Int64
    assert out.schema["eom"] == pl.Date
