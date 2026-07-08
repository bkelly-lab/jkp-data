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
    AP_FACTORS_MONTHLY_INPUT_SCHEMA,
    WORLD_MSF_INPUT_SCHEMA,
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


def _oracle_at(
    world: pl.DataFrame,
    fcts: pl.DataFrame,
    sid: int,
    country: str,
    incl: int,
    skip: int,
    terminal_idx: int,
) -> float:
    """Numpy FF3-OLS oracle for the 36-month chunk ending at ``terminal_idx``.

    Description:
        Independently replicate ``resff3_{incl}_{skip}`` for one stock at one
        terminal month, from raw ``world_msf`` / ``ap_factors_monthly`` frames.
    Steps:
        1) Slice ret_exc / mktrf / hml / smb_ff over the 36 months ending at the
           terminal (clamped to grid start) for (sid, country); drop any month
           with a null return or null factor (mirrors the prep mktrf gate and
           res_mom's hml/smb-null filter).
        2) OLS ``ret_exc ~ 1 + mktrf + hml + smb_ff``; residuals = y - X@beta.
        3) Average residuals over aux_date in (T-incl, T-skip]; divide by
           std(ddof=1).
    Output:
        Expected float value at (sid, grid[terminal_idx]).
    """
    grid = month_grid()
    aux = [g.year * 12 + g.month for g in grid]
    terminal_aux = aux[terminal_idx]

    chunk_idx = list(range(max(0, terminal_idx - (_N - 1)), terminal_idx + 1))
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
        m, h, s = fac_by_eom.get(eom, (None, None, None))
        ret = ret_by_eom.get(eom)
        if ret is None or h is None or s is None or m is None:
            continue  # mirrors res_mom dropping hml/smb-null rows and prep mktrf gate.
        ys.append(ret)
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


def _oracle_resff3(sid: int, country: str, incl: int, skip: int) -> float:
    """Numpy OLS oracle for the terminal-m39 chunk of a single golden stock."""
    return _oracle_at(
        build_world_msf_input(seed=42),
        build_ap_factors_monthly_input(seed=42),
        sid,
        country,
        incl,
        skip,
        _TERMINAL_IDX,
    )


def _custom_frames(
    *,
    country: str = "ZZ",
    sid: int = 1000,
    first: int = 16,
    last: int = 39,
    nuke: str | None = None,
    nuke_idx: int = 20,
    seed: int = 7,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build an isolated single-stock world_msf + ap_factors pair over month_grid.

    Description:
        One stock present in ``grid[first..last]`` under ``country``, with FF3
        factors for every grid month so residual_momentum runs end-to-end.
        ``nuke`` disables exactly one chunk month via one mechanism, letting a
        single-month removal cross the 24-obs boundary.
    Steps:
        1) Draw deterministic factors/noise; ret_exc = FF3 model + noise.
        2) Apply the requested ``nuke`` at ``grid[nuke_idx]`` (factor null, prep
           gate, or return null).
    Output:
        (world_df, fcts_df) matching the input schemas, one country only.
    """
    grid = month_grid()
    rng = np.random.default_rng(seed)
    n = len(grid)
    mkt = rng.normal(0.0, 0.04, n)
    hml = rng.normal(0.0, 0.04, n)
    smb = rng.normal(0.0, 0.04, n)
    noise = rng.normal(0.0, 0.02, n)

    frows: list[dict] = []
    for i, eom in enumerate(grid):
        m, h, s = float(mkt[i]), float(hml[i]), float(smb[i])
        if i == nuke_idx and nuke == "mktrf":
            m = None
        if i == nuke_idx and nuke == "hml":
            h = None
        if i == nuke_idx and nuke == "smb":
            s = None
        frows.append({"excntry": country, "eom": eom, "mktrf": m, "hml": h, "smb_ff": s})

    wrows: list[dict] = []
    for i in range(first, last + 1):
        ret = 0.005 + 1.1 * mkt[i] + 0.3 * hml[i] - 0.2 * smb[i] + noise[i]
        rexc: float | None = float(ret)
        rld = 1
        rloc = float(ret) + 1.0
        if i == nuke_idx and nuke == "ret_exc":
            rexc = None
            rloc = 1.0
        if i == nuke_idx and nuke == "ret_lag_dif":
            rld = 2
        if i == nuke_idx and nuke == "ret_local":
            rloc = 0.0
        wrows.append(
            {
                "id": sid,
                "eom": grid[i],
                "excntry": country,
                "ret_exc": rexc,
                "ret_lag_dif": rld,
                "ret_local": rloc,
            }
        )

    world = pl.DataFrame(wrows, schema=WORLD_MSF_INPUT_SCHEMA)
    fcts = pl.DataFrame(frows, schema=AP_FACTORS_MONTHLY_INPUT_SCHEMA)
    return world, fcts


def _run_frames(
    paths, incl: int, skip: int, world_df: pl.DataFrame, fcts_df: pl.DataFrame
) -> pl.DataFrame:
    """Write a custom world_msf + ap_factors pair, run one variant, read sorted output."""
    world_path = paths.interim_dir / "world_msf.parquet"
    fcts_path = paths.interim_dir / "ap_factors_monthly.parquet"
    world_df.write_parquet(world_path)
    fcts_df.write_parquet(fcts_path)
    residual_momentum(paths, _STEM, world_path, fcts_path, _N, _MIN, incl, skip)
    return pl.read_parquet(paths.interim_dir / f"{_STEM}_{incl}_{skip}.parquet").sort(["id", "eom"])


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


@pytest.mark.unit
def test_value_correctness_nonterminal(test_paths, tolerance) -> None:
    """id=1 at a NON-terminal eom (m35) matches the oracle: rolling window is not
    special-cased to the terminal month."""
    idx = 35
    eom = month_grid()[idx]
    out = _run(test_paths, 12, 1)
    row = out.filter((pl.col("id") == 1) & (pl.col("eom") == eom))
    assert row.height == 1
    expected = _oracle_at(
        build_world_msf_input(seed=42), build_ap_factors_monthly_input(seed=42), 1, "US", 12, 1, idx
    )
    np.testing.assert_allclose(row["resff3_12_1"][0], expected, **tolerance.STANDARD)


@pytest.mark.unit
def test_value_correctness_ca_partition(test_paths, tolerance) -> None:
    """id=4 (CA) terminal value matches an oracle fit on CA factors with the m20
    hml-null month dropped: proves per-country partition and that the null row is
    removed from the regression, not merely the averaging window."""
    out = _run(test_paths, 12, 1)
    terminal = month_grid()[_TERMINAL_IDX]
    row = out.filter((pl.col("id") == 4) & (pl.col("eom") == terminal))
    assert row.height == 1
    expected = _oracle_at(
        build_world_msf_input(seed=42),
        build_ap_factors_monthly_input(seed=42),
        4,
        "CA",
        12,
        1,
        _TERMINAL_IDX,
    )
    np.testing.assert_allclose(row["resff3_12_1"][0], expected, **tolerance.STANDARD)


@pytest.mark.unit
def test_min_obs_23_excluded(test_paths) -> None:
    """A stock with exactly 23 chunk months (< __min=24) is absent everywhere."""
    world, fcts = _custom_frames(first=17)  # grid[17..39] = 23 months
    out = _run_frames(test_paths, 12, 1, world, fcts)
    assert out.filter(pl.col("id") == 1000).height == 0


@pytest.mark.unit
def test_min_obs_24_boundary_value(test_paths, tolerance) -> None:
    """A stock with exactly 24 chunk months clears the gate, appears only at the
    terminal month, and its value matches the independent oracle."""
    world, fcts = _custom_frames(first=16, seed=11)  # grid[16..39] = 24 months
    out = _run_frames(test_paths, 12, 1, world, fcts)
    row = out.filter(pl.col("id") == 1000)
    assert row.height == 1  # 24 obs only at terminal m39; earlier terminals have < 24
    assert row["eom"][0] == month_grid()[_TERMINAL_IDX]
    expected = _oracle_at(world, fcts, 1000, "ZZ", 12, 1, _TERMINAL_IDX)
    np.testing.assert_allclose(row["resff3_12_1"][0], expected, **tolerance.STANDARD)


@pytest.mark.unit
@pytest.mark.parametrize(
    "nuke", [None, "mktrf", "ret_exc", "ret_lag_dif", "ret_local", "hml", "smb"]
)
def test_single_month_removal_crosses_min_obs_boundary(test_paths, nuke: str | None) -> None:
    """Nuking one of the 24 chunk months via each drop mechanism pushes the stock
    to 23 obs -> absent; the clean baseline (nuke=None) is present exactly once.

    Covers: prep gates (mktrf-null join, ret_exc-null, ret_lag_dif!=1, ret_local==0),
    the apply_group_filter ret_exc count gate, and res_mom's hml/smb-null filter +
    internal n>=__min gate (hml/smb rows survive prep but are dropped in res_mom).
    """
    world, fcts = _custom_frames(nuke=nuke)
    out = _run_frames(test_paths, 12, 1, world, fcts)
    stock = out.filter(pl.col("id") == 1000)
    if nuke is None:
        assert stock.height == 1
        assert stock["eom"][0] == month_grid()[_TERMINAL_IDX]
        assert stock["resff3_12_1"].is_finite().all()
    else:
        assert stock.height == 0


@pytest.mark.unit
def test_output_unique_id_eom(test_paths) -> None:
    """(id, eom) is a unique key in the output — no duplicate rows per stock-month."""
    out = _run(test_paths, 12, 1)
    assert out.select(["id", "eom"]).n_unique() == out.height


@pytest.mark.unit
def test_determinism_rerun(test_paths) -> None:
    """Re-running the same inputs yields a byte-identical output frame."""
    first = _run(test_paths, 12, 1)
    second = _run(test_paths, 12, 1)
    assert first.equals(second)
