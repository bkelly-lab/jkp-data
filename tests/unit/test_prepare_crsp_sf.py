"""Semantic unit tests for prepare_crsp_sf().

These assert behaviour/edge cases (NASDAQ volume windows, div_tot, delret
imputation, ret backfill, excess-return scale, company ME, delist joins, empty
input, dedup/sort) rather than fixture bytes — the byte-parity check lives in
tests/golden/test_prepare_crsp_sf_golden.py.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import prepare_crsp_sf
from jkp.data.paths import DataPaths
from tests.golden.prepare_crsp_sf_inputs import (
    SCHEMA_CRSP_SF,
    SCHEMA_FF,
    SCHEMA_MCTI,
    SCHEMA_SEDELIST,
    _crsp_row,
    _del_row,
)


def _run(
    paths: DataPaths,
    freq: str,
    crsp_rows: list[dict[str, object]],
    del_rows: list[dict[str, object]] | None = None,
    mcti_rows: list[tuple[date, float]] | None = None,
    ff_rows: list[tuple[date, float]] | None = None,
) -> pl.DataFrame:
    """Write minimal inputs, run prepare_crsp_sf(freq), return the output frame."""
    raw = paths.interim_dir / "raw_data_dfs"
    raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(crsp_rows, schema=SCHEMA_CRSP_SF).write_parquet(raw / f"__crsp_sf_{freq}.parquet")
    pl.DataFrame(del_rows or [], schema=SCHEMA_SEDELIST).write_parquet(
        raw / f"crsp_{freq}sedelist.parquet"
    )
    pl.DataFrame(
        {"caldt": [r[0] for r in (mcti_rows or [])], "t30ret": [r[1] for r in (mcti_rows or [])]},
        schema=SCHEMA_MCTI,
    ).write_parquet(raw / "crsp_mcti_t30ret.parquet")
    pl.DataFrame(
        {"date": [r[0] for r in (ff_rows or [])], "rf": [r[1] for r in (ff_rows or [])]},
        schema=SCHEMA_FF,
    ).write_parquet(raw / "ff_factors_monthly.parquet")
    prepare_crsp_sf(paths, freq)
    return pl.read_parquet(paths.interim_dir / f"crsp_{freq}sf.parquet")


def _val(df: pl.DataFrame, permno: int, d: date, col: str) -> object:
    return df.filter((pl.col("permno") == permno) & (pl.col("date") == d))[col].item()


def _nasdaq_row(permno: int, permco: int, d: date, vol: int, **over: object) -> dict[str, object]:
    """Build a NASDAQ (Q/RW) crsp row with sane defaults and field overrides.

    Description:
        Wrap ``_crsp_row`` (nasdaq=True -> primaryexch="Q", conditionaltype="RW")
        for volume-window / condition-gate cases, then apply ``over`` overrides so
        a case can flip a single field (e.g. conditionaltype) without a new schema.
    Output:
        One SCHEMA_CRSP_SF row dict.
    """
    base = _crsp_row(permno, permco, d, 10.0, 1.0, 0.0, 0.0, vol, 1.0, nasdaq=True)
    base.update(over)
    return base


def test_nasdaq_volume_windows(test_paths: DataPaths) -> None:
    """Daily NASDAQ (Q/RW) volume is divided by 2 / 1.8 / 1.6 across the three
    historic windows and unchanged after 2003-12-31; non-NASDAQ is unchanged."""
    rows = [
        _crsp_row(1, 1, date(2001, 1, 15), 10.0, 1.0, 0.0, 0.0, 1000, 1.0, nasdaq=True),  # /2
        _crsp_row(1, 1, date(2001, 6, 15), 10.0, 1.0, 0.0, 0.0, 900, 1.0, nasdaq=True),  # /1.8
        _crsp_row(1, 1, date(2003, 6, 15), 10.0, 1.0, 0.0, 0.0, 800, 1.0, nasdaq=True),  # /1.6
        _crsp_row(1, 1, date(2004, 6, 15), 10.0, 1.0, 0.0, 0.0, 1234, 1.0, nasdaq=True),  # none
        _crsp_row(2, 2, date(2001, 1, 15), 10.0, 1.0, 0.0, 0.0, 1000, 1.0, nasdaq=False),  # none
    ]
    df = _run(test_paths, "d", rows)
    assert _val(df, 1, date(2001, 1, 15), "vol") == pytest.approx(500.0)
    assert _val(df, 1, date(2001, 6, 15), "vol") == pytest.approx(500.0)
    assert _val(df, 1, date(2003, 6, 15), "vol") == pytest.approx(500.0)
    assert _val(df, 1, date(2004, 6, 15), "vol") == pytest.approx(1234.0)
    assert _val(df, 2, date(2001, 1, 15), "vol") == pytest.approx(1000.0)


def test_monthly_scaling_and_vol_dtype(test_paths: DataPaths) -> None:
    """Monthly vol and dolvol are scaled by 100 and vol is promoted to Float64."""
    rows = [_crsp_row(1, 1, date(2001, 1, 31), 10.0, 1.0, 0.0, 0.0, 1000, 1.0, nasdaq=True)]
    df = _run(test_paths, "m", rows)
    # 1000 vol / 2 (pre-2001-02 NASDAQ window) = 500, then monthly x100
    expected_vol = pl.select(pl.lit(1000.0) / 2 * 100).item()
    assert df.schema["vol"] == pl.Float64
    assert _val(df, 1, date(2001, 1, 31), "vol") == pytest.approx(expected_vol)
    assert _val(df, 1, date(2001, 1, 31), "dolvol") == pytest.approx(10.0 * 500.0 * 100)


def test_div_tot(test_paths: DataPaths) -> None:
    """div_tot is null on the first row per permno and when lagged cfacshr==0,
    else (ret-retx)*prc_prev*(cfacshr/cfacshr_prev)."""
    rows = [
        _crsp_row(1, 1, date(2001, 1, 15), 10.0, 1.0, 0.05, 0.04, 100, 1.0, nasdaq=False),
        _crsp_row(1, 1, date(2001, 1, 16), 20.0, 2.0, 0.06, 0.05, 100, 1.0, nasdaq=False),
        _crsp_row(2, 2, date(2001, 1, 15), 10.0, 0.0, 0.0, 0.0, 100, 1.0, nasdaq=False),
        _crsp_row(2, 2, date(2001, 1, 16), 11.0, 1.0, 0.03, 0.01, 100, 1.0, nasdaq=False),
    ]
    df = _run(test_paths, "d", rows)
    assert _val(df, 1, date(2001, 1, 15), "div_tot") is None
    assert _val(df, 1, date(2001, 1, 16), "div_tot") == pytest.approx(
        (0.06 - 0.05) * 10.0 * (2.0 / 1.0)
    )
    assert _val(df, 2, date(2001, 1, 16), "div_tot") is None  # cfacshr_prev == 0


@pytest.mark.parametrize(
    ("reason", "action", "status", "payment"),
    [("UNAV", "GDR", "VCL", "PRCF"), ("BKPY", "GDR", "VCL", "PRCF")],  # c2, c3
)
def test_bad_delist_imputes_minus_030(
    test_paths: DataPaths, reason: str, action: str, status: str, payment: str
) -> None:
    """Null delret in a c2/c3 bucket with M flag is imputed to -0.30 then compounded."""
    rows = [
        _crsp_row(
            1, 1, date(2001, 1, 31), 10.0, 1.0, 0.04, 0.04, 100, 1.0, nasdaq=False, del_flag="M"
        )
    ]
    dels = [_del_row(1, date(2001, 1, 15), None, action, status, reason, payment)]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, date(2001, 1, 31), "ret") == pytest.approx((0.04 + 1) * (1 - 0.3) - 1)


def test_non_bad_null_delist_not_imputed(test_paths: DataPaths) -> None:
    """A delist with null delret but non-bad codes under M flag is NOT imputed;
    delret stays null, coalesces to 0, so ret is unchanged."""
    rows = [
        _crsp_row(
            1, 1, date(2001, 1, 31), 10.0, 1.0, 0.04, 0.04, 100, 1.0, nasdaq=False, del_flag="M"
        )
    ]
    dels = [_del_row(1, date(2001, 1, 15), None, "MERG", None, None, None)]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, date(2001, 1, 31), "ret") == pytest.approx(0.04)


def test_ret_backfill_when_missing(test_paths: DataPaths) -> None:
    """ret null + delret present under M flag -> ret set to 0 then compounded."""
    rows = [
        _crsp_row(
            1, 1, date(2001, 1, 31), 10.0, 1.0, None, None, 100, 1.0, nasdaq=False, del_flag="M"
        )
    ]
    dels = [_del_row(1, date(2001, 1, 15), -0.2, None, None, None, None)]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, date(2001, 1, 31), "ret") == pytest.approx(-0.2)


def test_no_delist_leaves_ret_unchanged(test_paths: DataPaths) -> None:
    """No matching delist -> delret null, ret compounded with 0 (unchanged)."""
    rows = [_crsp_row(1, 1, date(2001, 1, 31), 10.0, 1.0, 0.07, 0.07, 100, 1.0, nasdaq=False)]
    df = _run(test_paths, "m", rows)
    assert _val(df, 1, date(2001, 1, 31), "ret") == pytest.approx(0.07)


def test_ret_exc_scale_and_coalesce(test_paths: DataPaths) -> None:
    """ret_exc subtracts coalesce(t30ret, rf)/scale: scale 1 monthly, 21 daily;
    t30ret preferred over rf; falls back to rf; both null -> null."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.05, 0.05, 100, 1.0, nasdaq=False)]
    # both present -> t30ret wins, monthly scale 1
    m = _run(test_paths, "m", rows, mcti_rows=[(d, 0.004)], ff_rows=[(d, 0.002)])
    assert _val(m, 1, d, "ret_exc") == pytest.approx(0.05 - 0.004)
    # t30ret absent -> rf fallback
    m2 = _run(test_paths, "m", rows, ff_rows=[(d, 0.002)])
    assert _val(m2, 1, d, "ret_exc") == pytest.approx(0.05 - 0.002)
    # both absent -> null
    m3 = _run(test_paths, "m", rows)
    assert _val(m3, 1, d, "ret_exc") is None
    # daily scale 21
    dd = date(2000, 1, 4)
    drows = [_crsp_row(1, 1, dd, 10.0, 1.0, 0.05, 0.05, 100, 1.0, nasdaq=False)]
    dsf = _run(test_paths, "d", drows, mcti_rows=[(dd, 0.004)])
    assert _val(dsf, 1, dd, "ret_exc") == pytest.approx(0.05 - 0.004 / 21)


def test_me_company(test_paths: DataPaths) -> None:
    """me_company sums me across permnos within permco-date; single permno ->
    its own me; a group whose only me is null -> null."""
    d = date(2001, 1, 31)
    rows = [
        _crsp_row(1, 100, d, 10.0, 1.0, 0.0, 0.0, 100, 60.0, nasdaq=False),
        _crsp_row(2, 100, d, 10.0, 1.0, 0.0, 0.0, 100, 40.0, nasdaq=False),
        _crsp_row(3, 300, d, 10.0, 1.0, 0.0, 0.0, 100, 25.0, nasdaq=False),
        _crsp_row(4, 400, d, 10.0, 1.0, 0.0, 0.0, 100, None, nasdaq=False),
    ]
    df = _run(test_paths, "m", rows)
    assert _val(df, 1, d, "me_company") == pytest.approx(100.0)
    assert _val(df, 2, d, "me_company") == pytest.approx(100.0)
    assert _val(df, 3, d, "me_company") == pytest.approx(25.0)
    assert _val(df, 4, d, "me_company") is None


def test_daily_delist_no_compound(test_paths: DataPaths) -> None:
    """Daily delisting returns are never compounded; DlyRet is used as-is."""
    rows = [
        _crsp_row(1, 1, date(2000, 1, 6), 10.0, 1.0, 0.03, 0.03, 100, 1.0, nasdaq=False),
        _crsp_row(
            1, 1, date(2000, 1, 7), 10.0, 1.0, 0.04, 0.04, 100, 1.0, nasdaq=False, del_flag="Y"
        ),
    ]
    dels = [_del_row(1, date(2000, 1, 7), None, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "d", rows, dels)
    assert _val(df, 1, date(2000, 1, 6), "ret") == pytest.approx(0.03)
    assert _val(df, 1, date(2000, 1, 7), "ret") == pytest.approx(0.04)


@pytest.mark.parametrize("freq", ["m", "d"])
def test_both_freq_filename_and_columns(test_paths: DataPaths, freq: str) -> None:
    """Each freq writes crsp_{freq}sf.parquet with the full computed column set."""
    rows = [_crsp_row(1, 1, date(2001, 1, 15), 10.0, 1.0, 0.01, 0.01, 100, 1.0, nasdaq=False)]
    _run(test_paths, freq, rows)
    out = test_paths.interim_dir / f"crsp_{freq}sf.parquet"
    assert out.exists()
    cols = pl.read_parquet(out).columns
    for extra in ("dolvol", "div_tot", "ret_exc", "me_company"):
        assert extra in cols
    for dropped in ("delret", "del_flag", "rf", "t30ret", "merge_aux", "delistingdt"):
        assert dropped not in cols


def test_empty_input_yields_typed_empty_output(test_paths: DataPaths) -> None:
    """A 0-row typed input produces a 0-row output carrying the full schema."""
    df = _run(test_paths, "m", [])
    assert df.height == 0
    for col in ("permno", "date", "vol", "dolvol", "div_tot", "ret_exc", "me_company"):
        assert col in df.columns


def test_output_unique_and_sorted(test_paths: DataPaths) -> None:
    """Output is deduplicated on (permno, date) and sorted by (permno, date)."""
    rows = [
        _crsp_row(2, 2, date(2001, 2, 15), 10.0, 1.0, 0.0, 0.0, 100, 1.0, nasdaq=False),
        _crsp_row(1, 1, date(2001, 1, 15), 10.0, 1.0, 0.0, 0.0, 100, 1.0, nasdaq=False),
        _crsp_row(1, 1, date(2001, 1, 15), 10.0, 1.0, 0.0, 0.0, 100, 1.0, nasdaq=False),  # dup
    ]
    df = _run(test_paths, "d", rows)
    keys = df.select(["permno", "date"])
    assert keys.is_duplicated().sum() == 0
    assert df["permno"].to_list() == sorted(df["permno"].to_list())


# --- NASDAQ volume window: exact both-sides boundaries -----------------------


# Cutoffs in adj_trd_vol_NASDAQ: <2001-02-01 -> /2; <=2001-12-31 -> /1.8;
# <=2003-12-31 -> /1.6; else unchanged. 2003-12-31 itself is adjusted (/1.6).
@pytest.mark.parametrize(
    ("d", "divisor"),
    [
        (date(2001, 1, 31), 2.0),  # last day of the /2 window
        (date(2001, 2, 1), 1.8),  # first day of the /1.8 window
        (date(2001, 12, 31), 1.8),  # last day of the /1.8 window
        (date(2002, 1, 1), 1.6),  # first day of the /1.6 window
        (date(2003, 12, 30), 1.6),  # inside the /1.6 window
        (date(2003, 12, 31), 1.6),  # inclusive end of the /1.6 window
        (date(2004, 1, 1), 1.0),  # first day after -> unchanged
        (date(2004, 6, 15), 1.0),  # well after -> unchanged
    ],
)
def test_nasdaq_volume_window_boundaries(test_paths: DataPaths, d: date, divisor: float) -> None:
    """Each NASDAQ (Q/RW) volume divisor applies on the exact boundary days;
    2003-12-31 is included in the /1.6 window (Gao–Ritter inclusive end date)."""
    df = _run(test_paths, "d", [_nasdaq_row(1, 1, d, 3600)])
    assert _val(df, 1, d, "vol") == pytest.approx(3600.0 / divisor)


def test_nasdaq_condition_gate(test_paths: DataPaths) -> None:
    """The adjustment needs BOTH primaryexch=='Q' and conditionaltype=='RW';
    Q-but-not-RW and RW-but-not-Q rows are left unadjusted."""
    d = date(2001, 1, 15)  # in the /2 window
    rows = [
        _nasdaq_row(1, 1, d, 1000, conditionaltype="XX"),  # Q, not RW -> no adj
        _nasdaq_row(2, 2, d, 1000, primaryexch="N"),  # RW, not Q -> no adj
        _nasdaq_row(3, 3, d, 1000),  # Q & RW -> /2
    ]
    df = _run(test_paths, "d", rows)
    assert _val(df, 1, d, "vol") == pytest.approx(1000.0)
    assert _val(df, 2, d, "vol") == pytest.approx(1000.0)
    assert _val(df, 3, d, "vol") == pytest.approx(500.0)


def test_daily_vol_not_scaled_by_100(test_paths: DataPaths) -> None:
    """The x100 vol/dolvol rescale is monthly-only; daily keeps raw magnitude."""
    d = date(2004, 6, 15)  # no NASDAQ adjustment
    df = _run(test_paths, "d", [_nasdaq_row(1, 1, d, 777)])
    assert _val(df, 1, d, "vol") == pytest.approx(777.0)
    assert _val(df, 1, d, "dolvol") == pytest.approx(10.0 * 777.0)


# --- delret -0.30 imputation: full c3 reason list + near-misses -------------

C3_REASONS = [
    "MVOT",
    "MTMK",
    "SHLD",
    "LP",
    "INSC",
    "INSF",
    "CORQ",
    "DERE",
    "BKPY",
    "OFFRE",
    "DELQ",
    "FARG",
    "EQRQ",
    "DEEX",
    "FING",
]


@pytest.mark.parametrize("reason", C3_REASONS)
def test_c3_every_reason_code_imputes(test_paths: DataPaths, reason: str) -> None:
    """Every reason in the c3 is_in list (with GDR/VCL/PRCF, null delret, M flag)
    imputes delret=-0.30 and compounds it into ret."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.02, 0.02, 100, 1.0, nasdaq=False, del_flag="M")]
    dels = [_del_row(1, date(2001, 1, 15), None, "GDR", "VCL", reason, "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx((0.02 + 1) * (1 - 0.3) - 1)


def test_c2_near_miss_not_imputed(test_paths: DataPaths) -> None:
    """UNAV with one c2 field wrong (status != VCL) under M flag is neither c2
    nor c3, so null delret is not imputed and ret is unchanged."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.02, 0.02, 100, 1.0, nasdaq=False, del_flag="M")]
    dels = [_del_row(1, date(2001, 1, 15), None, "GDR", "ACT", "UNAV", "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(0.02)


def test_c3_near_miss_wrong_action_not_imputed(test_paths: DataPaths) -> None:
    """A c3 reason (BKPY) with a non-GDR action under M flag fails c3 and is
    not imputed."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.02, 0.02, 100, 1.0, nasdaq=False, del_flag="M")]
    dels = [_del_row(1, date(2001, 1, 15), None, "MERG", "VCL", "BKPY", "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(0.02)


def test_present_delret_in_bad_bucket_not_overwritten(test_paths: DataPaths) -> None:
    """c4 gates on delret being null (c1); a non-null delret in a c2 bucket with
    M flag is kept and compounded, never replaced by -0.30."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.02, 0.02, 100, 1.0, nasdaq=False, del_flag="M")]
    dels = [_del_row(1, date(2001, 1, 15), 0.1, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx((0.02 + 1) * (0.1 + 1) - 1)


# --- delist join misses ------------------------------------------------------


def test_monthly_delist_month_mismatch_no_join(test_paths: DataPaths) -> None:
    """Monthly join is on (permno, MMYY); a delist in a different month than the
    crsp row does not match, so ret is unchanged."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.02, 0.02, 100, 1.0, nasdaq=False)]
    dels = [_del_row(1, date(2001, 2, 15), None, "GDR", "VCL", "UNAV", "PRCF")]  # Feb
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(0.02)


def test_daily_delist_date_mismatch_no_join(test_paths: DataPaths) -> None:
    """Daily join is on the exact (permno, date); a delist on a non-matching day
    does not attach and ret is unchanged."""
    d = date(2000, 1, 6)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.03, 0.03, 100, 1.0, nasdaq=False)]
    dels = [_del_row(1, date(2000, 1, 7), None, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "d", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(0.03)


def test_ret_null_no_delist_stays_null(test_paths: DataPaths) -> None:
    """ret null with no delist stays null: c7 backfill needs delret present, and
    compounding (null+1)*(0+1)-1 propagates null."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, None, None, 100, 1.0, nasdaq=False)]
    df = _run(test_paths, "m", rows)
    assert _val(df, 1, d, "ret") is None
    assert _val(df, 1, d, "ret_exc") is None


# --- me_company null-coalescing within a group ------------------------------


def test_me_company_mixed_null_coalesces_to_zero(test_paths: DataPaths) -> None:
    """A permco group with one null and one non-null me has a non-zero me count,
    so the null is coalesced to 0 in the sum (not propagated to null)."""
    d = date(2001, 1, 31)
    rows = [
        _crsp_row(1, 500, d, 10.0, 1.0, 0.0, 0.0, 100, 60.0, nasdaq=False),
        _crsp_row(2, 500, d, 10.0, 1.0, 0.0, 0.0, 100, None, nasdaq=False),
    ]
    df = _run(test_paths, "m", rows)
    assert _val(df, 1, d, "me_company") == pytest.approx(60.0)
    assert _val(df, 2, d, "me_company") == pytest.approx(60.0)


# --- schema / dtypes ---------------------------------------------------------


def test_output_dtypes(test_paths: DataPaths) -> None:
    """Computed columns are Float64; monthly vol is promoted to Float64 and the
    delist helper columns are dropped."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.01, 0.005, 100, 1.0, nasdaq=False)]
    df = _run(test_paths, "m", rows)
    for c in ("dolvol", "div_tot", "ret_exc", "me_company", "vol"):
        assert df.schema[c] == pl.Float64
    for dropped in (
        "delret",
        "del_flag",
        "delreasontype",
        "delactiontype",
        "delpaymenttype",
        "delstatustype",
    ):
        assert dropped not in df.columns


def test_empty_daily_input_yields_typed_empty(test_paths: DataPaths) -> None:
    """A 0-row daily input (exact-date delist join path) produces a typed 0-row
    output carrying the computed columns."""
    df = _run(test_paths, "d", [])
    assert df.height == 0
    for c in ("permno", "date", "vol", "dolvol", "div_tot", "ret_exc", "me_company"):
        assert c in df.columns


# --- CIZ flag-conditional delisting (Xia 2026, SSRN 7243220) ----------------


@pytest.mark.parametrize("flag", ["A", "P", "V"])
def test_monthly_flag_APV_no_double_count(test_paths: DataPaths, flag: str) -> None:
    """A/P/V flags indicate DelRet is already in MthRet; ret stays as-is even
    when a non-null delret is present from stkdelists."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, -0.40, -0.40, 100, 1.0, nasdaq=False, del_flag=flag)]
    dels = [_del_row(1, date(2001, 1, 15), -0.50, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(-0.40)


def test_monthly_flag_G_uses_mthret_only(test_paths: DataPaths) -> None:
    """G flag (contemporaneous timing): ret stays MthRet, no compounding."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, -0.10, -0.10, 100, 1.0, nasdaq=False, del_flag="G")]
    dels = [_del_row(1, date(2001, 1, 15), -0.50, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(-0.10)


def test_monthly_flag_N_no_adjustment(test_paths: DataPaths) -> None:
    """N flag (no delisting adjustment): ret stays MthRet."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.05, 0.05, 100, 1.0, nasdaq=False, del_flag="N")]
    dels = [_del_row(1, date(2001, 1, 15), -0.30, None, None, None, None)]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(0.05)


def test_monthly_flag_M_preserves_imputation(test_paths: DataPaths) -> None:
    """M flag (unobserved payoff): existing imputation + compounding is
    preserved, matching pre-fix behavior for this flag."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.02, 0.02, 100, 1.0, nasdaq=False, del_flag="M")]
    dels = [_del_row(1, date(2001, 1, 15), None, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx((0.02 + 1) * (1 - 0.3) - 1)


def test_monthly_null_flag_no_compound(test_paths: DataPaths) -> None:
    """Null del_flag (no delisting event in the stock file) means no
    compounding even when a sedelist row happens to join."""
    d = date(2001, 1, 31)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.04, 0.04, 100, 1.0, nasdaq=False)]
    dels = [_del_row(1, date(2001, 1, 15), -0.50, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "m", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(0.04)


def test_daily_dlydelflg_Y_no_compound(test_paths: DataPaths) -> None:
    """DlyDelFlg=Y: delisting return already in DlyRet; no compounding."""
    d = date(2000, 1, 7)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, -0.40, -0.40, 100, 1.0, nasdaq=False, del_flag="Y")]
    dels = [_del_row(1, d, -0.50, "GDR", "VCL", "UNAV", "PRCF")]
    df = _run(test_paths, "d", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(-0.40)


def test_daily_dlydelflg_N_no_compound(test_paths: DataPaths) -> None:
    """DlyDelFlg=N: ordinary return in DlyRet; no compounding."""
    d = date(2000, 1, 7)
    rows = [_crsp_row(1, 1, d, 10.0, 1.0, 0.02, 0.02, 100, 1.0, nasdaq=False, del_flag="N")]
    dels = [_del_row(1, d, -0.30, None, None, None, None)]
    df = _run(test_paths, "d", rows, dels)
    assert _val(df, 1, d, "ret") == pytest.approx(0.02)
