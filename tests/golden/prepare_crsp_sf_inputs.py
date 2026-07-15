"""Shared synthetic-input builders for ``prepare_crsp_sf``.

Used by ``tests/golden/generate_prepare_crsp_sf_golden.py``,
``tests/golden/test_prepare_crsp_sf_golden.py`` and
``tests/unit/test_prepare_crsp_sf.py`` so every input schema is defined once.

All values are invented round literals — no proprietary WRDS/CRSP data. The CIZ
delist code strings (``UNAV``/``GDR``/``PRCF``/``VCL``/``BKPY``/``MERG``) are
enumeration constants hard-coded in ``aux_functions.prepare_crsp_sf`` itself,
not data values.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np
import polars as pl

from jkp.data.paths import DataPaths

# --- Deterministic bulk-panel sizing (Goal 1: scale each golden to ~0.5 MB) --
#
# A large synthetic panel is appended to the hand-crafted edge-case rows so the
# committed golden parquets land around 0.5 MB. Bulk entities live at high id
# ranges (90_000+ monthly, 800_000+ daily) so they never collide with the
# edge-case ids (10_00x monthly, 20_00x daily). Bulk dates live in 1975-1999,
# strictly disjoint from every edge-case month (all 2000+), so appending bulk
# rf/t30ret rows never perturbs the edge-case rf/t30ret null cascade.

FIXED_SEED = 20240707
_BULK_N_ENT_M = 60  # monthly bulk entities
_BULK_T_M = 100  # months per monthly bulk entity (1991-09 .. 1999-12)
_BULK_N_ENT_D = 60  # daily bulk entities
_BULK_T_D = 100  # consecutive days per daily bulk entity (from 1990-01-02)
_BULK_PERMNO0_M = 90_000
_BULK_PERMNO0_D = 800_000
_BULK_DATE0_D = date(1990, 1, 2)

# --- Schemas (derived from the gen_crsp_sf select list + the four
#     collect_and_write blocks in src/jkp/data/aux_functions.py) -------------

SCHEMA_CRSP_SF: dict[str, pl.DataType] = {
    "permno": pl.Int64,
    "permco": pl.Int64,
    "date": pl.Date,
    "bidask": pl.Int32,
    "prc": pl.Float64,
    "prc_open": pl.Float64,
    "prc_close": pl.Float64,
    "shrout": pl.Float64,
    "ret": pl.Float64,
    "retx": pl.Float64,
    "cfacshr": pl.Float64,
    "vol": pl.Int64,
    "prc_high": pl.Float64,
    "prc_low": pl.Float64,
    "common": pl.Int32,
    "primaryexch": pl.Utf8,
    "conditionaltype": pl.Utf8,
    "crsp_nyse": pl.Int32,
    "gvkey": pl.Utf8,
    "iid": pl.Utf8,
    "exch_main": pl.Int32,
    "me": pl.Float64,
    "ticker": pl.Utf8,
}

SCHEMA_SEDELIST: dict[str, pl.DataType] = {
    "delret": pl.Float64,
    "delactiontype": pl.Utf8,
    "delstatustype": pl.Utf8,
    "delreasontype": pl.Utf8,
    "delpaymenttype": pl.Utf8,
    "permno": pl.Int64,
    "delistingdt": pl.Date,
}

SCHEMA_MCTI: dict[str, pl.DataType] = {"caldt": pl.Date, "t30ret": pl.Float64}

SCHEMA_FF: dict[str, pl.DataType] = {"date": pl.Date, "rf": pl.Float64}


def _crsp_row(
    permno: int,
    permco: int,
    d: date,
    prc: float,
    cfacshr: float,
    ret: float | None,
    retx: float | None,
    vol: int,
    me: float | None,
    *,
    nasdaq: bool,
    prc_open: float | None = None,
    prc_close: float | None = None,
) -> dict[str, object]:
    """Build one ``__crsp_sf`` row; pass-through payload columns get stable
    deterministic fillers so golden bytes are reproducible."""
    return {
        "permno": permno,
        "permco": permco,
        "date": d,
        "bidask": 0,
        "prc": prc,
        "prc_open": prc_open,
        "prc_close": prc_close,
        "shrout": 1000.0,
        "ret": ret,
        "retx": retx,
        "cfacshr": cfacshr,
        "vol": vol,
        "prc_high": prc + 1.0,
        "prc_low": prc - 1.0,
        "common": 1,
        "primaryexch": "Q" if nasdaq else "N",
        "conditionaltype": "RW",
        "crsp_nyse": 0 if nasdaq else 1,
        "gvkey": f"{permno:06d}",
        "iid": "01",
        "exch_main": 1,
        "me": me,
        "ticker": "AAA",
    }


def _month_end(y: int, m: int) -> date:
    """Return the calendar month-end date for year ``y`` month ``m``."""
    return date(y, m, calendar.monthrange(y, m)[1])


def _bulk_month_ends() -> list[date]:
    """Last ``_BULK_T_M`` month-ends of 1975-01 .. 1999-12 (monthly bulk dates)."""
    all_months = [_month_end(y, m) for y in range(1975, 2000) for m in range(1, 13)]
    return all_months[-_BULK_T_M:]


def _bulk_daily_dates() -> list[date]:
    """``_BULK_T_D`` consecutive calendar days from ``_BULK_DATE0_D`` (all 1990)."""
    return [_BULK_DATE0_D + timedelta(days=k) for k in range(_BULK_T_D)]


def _build_bulk_rows(freq: str) -> list[dict[str, object]]:
    """Deterministic synthetic bulk panel appended to the edge-case rows.

    Description:
        Generate ``N_ENT`` entities that each span the same date grid, with
        prices positive, returns ~N(0, 0.06), a small dividend wedge in retx,
        constant cfacshr (so div_tot is computed, not null), integer volume and
        positive ME. All draws come from one seeded PCG64 rng, so the panel is
        identical run-to-run and thus safe to pin in a golden fixture.
    Steps:
        1) Pick the date grid and id base for the freq.
        2) Vectorised draw of prc/ret/retx/vol/me over (n_ent, n_per).
        3) Emit one ``_crsp_row`` per (entity, period); unique permco per entity.
    Output:
        list[dict] of SCHEMA_CRSP_SF rows (empty for unknown freq).
    """
    if freq == "m":
        dates = _bulk_month_ends()
        n_ent, permno0, seed = _BULK_N_ENT_M, _BULK_PERMNO0_M, FIXED_SEED
    elif freq == "d":
        dates = _bulk_daily_dates()
        n_ent, permno0, seed = _BULK_N_ENT_D, _BULK_PERMNO0_D, FIXED_SEED + 1
    else:
        return []

    n_per = len(dates)
    rng = np.random.default_rng(seed)
    prc = rng.uniform(2.0, 200.0, size=(n_ent, n_per))
    ret = rng.normal(0.0, 0.06, size=(n_ent, n_per))
    div = rng.uniform(0.0, 0.004, size=(n_ent, n_per))
    vol = rng.integers(1_000, 1_000_000, size=(n_ent, n_per))
    me = rng.uniform(10.0, 5_000.0, size=(n_ent, n_per))

    rows: list[dict[str, object]] = []
    for i in range(n_ent):
        permno = permno0 + i
        nasdaq = (i % 2) == 0
        for j, d in enumerate(dates):
            rows.append(
                _crsp_row(
                    permno,
                    permno,  # unique permco per entity -> me_company == me
                    d,
                    float(prc[i, j]),
                    1.0,
                    float(ret[i, j]),
                    float(ret[i, j] - div[i, j]),
                    int(vol[i, j]),
                    float(me[i, j]),
                    nasdaq=nasdaq,
                )
            )
    return rows


def _bulk_rf_months() -> list[date]:
    """Month-ends 1975-01 .. 1999-12 covering every bulk (monthly + daily) month.

    Daily bulk dates live in 1990, whose months are a subset of this range, so a
    single 1975-1999 month grid supplies rf/t30ret for both bulk panels while
    remaining disjoint from every 2000+ edge-case month.
    """
    return [_month_end(y, m) for y in range(1975, 2000) for m in range(1, 13)]


def build_crsp_sf_input(freq: str) -> pl.DataFrame:
    """Build the ``__crsp_sf_{freq}`` panel exercising every code path.

    Description:
        Monthly (freq="m") rows drive: the four NASDAQ volume windows, div_tot
        (first-row-null, computed, cfacshr_prev==0 -> null), the c2/c3 delret
        imputation buckets, a non-bad delist kept as-is, the c7 ret backfill,
        the ret_exc rf/t30ret coalesce/fallback/null cascade, and me_company
        (multi-permno sum, single-permno self, all-null -> null).
        Daily (freq="d") rows drive the same volume/div_tot/me_company logic
        plus the exact-date delist join and the /21 daily excess-return scale.
    Steps:
        1) Emit hand-written literal rows per permno.
        2) Return a typed DataFrame (0 rows if an unknown freq is passed).
    Output:
        pl.DataFrame with SCHEMA_CRSP_SF columns/dtypes.
    """
    if freq == "m":
        rows = [
            # 10001 permco 100 NASDAQ: four volume windows + div_tot ladder.
            _crsp_row(
                10001, 100, date(2000, 1, 31), 10.0, 1.0, 0.05, 0.04, 1000, 100.0, nasdaq=True
            ),
            _crsp_row(
                10001, 100, date(2001, 6, 30), 20.0, 1.0, 0.03, 0.02, 900, 110.0, nasdaq=True
            ),
            _crsp_row(
                10001, 100, date(2002, 6, 28), 25.0, 2.0, 0.06, 0.05, 800, 120.0, nasdaq=True
            ),
            _crsp_row(
                10001, 100, date(2004, 1, 30), 30.0, 2.0, 0.02, 0.015, 1234, 130.0, nasdaq=True
            ),
            # 10002 permco 100 NYSE, shares permco+date with 10001 r4 -> me_company sum.
            _crsp_row(
                10002, 100, date(2004, 1, 30), 40.0, 1.0, 0.01, 0.01, 500, 70.0, nasdaq=False
            ),
            # 10003 permco 103: r1 cfacshr=0 so r2 div_tot -> null; r2 month delist (c2).
            _crsp_row(
                10003, 103, date(2005, 3, 31), 15.0, 0.0, 0.02, 0.02, 300, 50.0, nasdaq=False
            ),
            _crsp_row(
                10003, 103, date(2005, 4, 29), 16.0, 1.0, 0.04, 0.03, 310, 52.0, nasdaq=False
            ),
            # 10004 permco 104: c3 (reason BKPY) delret imputation.
            _crsp_row(10004, 104, date(2006, 5, 31), 8.0, 1.0, 0.05, 0.05, 200, 30.0, nasdaq=False),
            # 10005 permco 105: delret present, non-bad codes -> kept as-is.
            _crsp_row(
                10005, 105, date(2007, 6, 29), 12.0, 1.0, 0.03, 0.03, 250, 40.0, nasdaq=False
            ),
            # 10006 permco 106: ret null + delret present -> c7 backfill.
            _crsp_row(10006, 106, date(2008, 7, 31), 5.0, 1.0, None, None, 150, 20.0, nasdaq=False),
            # 10007 permco 107: rf-only month, then neither-rf-nor-t30ret + me null.
            _crsp_row(10007, 107, date(2009, 8, 31), 7.0, 1.0, 0.02, 0.02, 100, 25.0, nasdaq=False),
            _crsp_row(10007, 107, date(2009, 9, 30), 8.0, 1.0, 0.01, 0.01, 110, None, nasdaq=False),
        ]
    elif freq == "d":
        rows = [
            # 20001 permco 200 NASDAQ: 3 days in 2000-01 (all /2) + div_tot ladder.
            _crsp_row(
                20001, 200, date(2000, 1, 3), 10.0, 1.0, 0.01, 0.005, 1000, 200.0, nasdaq=True
            ),
            _crsp_row(
                20001, 200, date(2000, 1, 4), 11.0, 1.0, 0.02, 0.01, 1200, 210.0, nasdaq=True
            ),
            _crsp_row(
                20001, 200, date(2000, 1, 5), 12.0, 1.0, 0.015, 0.01, 1400, 220.0, nasdaq=True
            ),
            # 20002 permco 200 NYSE, same day as 20001 d3 -> me_company daily sum.
            _crsp_row(20002, 200, date(2000, 1, 5), 20.0, 1.0, 0.02, 0.02, 500, 80.0, nasdaq=False),
            # 20003 permco 203: delist only on the exact day 2000-01-07 (c2).
            _crsp_row(20003, 203, date(2000, 1, 6), 15.0, 1.0, 0.03, 0.03, 300, 50.0, nasdaq=False),
            _crsp_row(20003, 203, date(2000, 1, 7), 15.0, 1.0, 0.04, 0.04, 310, 52.0, nasdaq=False),
            # 20004 permco 204: t30ret present -> ret_exc = ret - t30ret/21.
            _crsp_row(20004, 204, date(2000, 1, 4), 9.0, 1.0, 0.05, 0.05, 90, 60.0, nasdaq=False),
        ]
    else:
        rows = []
    return pl.DataFrame(rows + _build_bulk_rows(freq), schema=SCHEMA_CRSP_SF)


def _del_row(
    permno: int,
    d: date,
    delret: float | None,
    action: str | None,
    status: str | None,
    reason: str | None,
    payment: str | None,
) -> dict[str, object]:
    return {
        "delret": delret,
        "delactiontype": action,
        "delstatustype": status,
        "delreasontype": reason,
        "delpaymenttype": payment,
        "permno": permno,
        "delistingdt": d,
    }


def build_sedelist_input(freq: str) -> pl.DataFrame:
    """Build ``crsp_{freq}sedelist`` delist rows.

    Description:
        Monthly rows join on (permno, delist-month); daily rows join on the
        exact (permno, delistingdt) day.
    Steps:
        1) Emit c2 (UNAV/GDR/PRCF/VCL), c3 (BKPY), a non-bad (MERG) and a
           plain-delret row for the monthly panel; a single c2 exact-day row
           for the daily panel.
        2) Return a typed DataFrame.
    Output:
        pl.DataFrame with SCHEMA_SEDELIST columns/dtypes.
    """
    if freq == "m":
        rows = [
            _del_row(10003, date(2005, 4, 15), None, "GDR", "VCL", "UNAV", "PRCF"),  # c2
            _del_row(10004, date(2006, 5, 15), None, "GDR", "VCL", "BKPY", "PRCF"),  # c3
            _del_row(10005, date(2007, 6, 15), 0.1, "MERG", None, None, None),  # kept
            _del_row(10006, date(2008, 7, 15), -0.2, None, None, None, None),  # c7 driver
        ]
    elif freq == "d":
        rows = [
            _del_row(20003, date(2000, 1, 7), None, "GDR", "VCL", "UNAV", "PRCF"),  # c2
        ]
    else:
        rows = []
    return pl.DataFrame(rows, schema=SCHEMA_SEDELIST)


def build_mcti_input() -> pl.DataFrame:
    """Build ``crsp_mcti_t30ret`` (shared monthly T-bill returns, one row/month).

    t30ret is present for every month used by the monthly and daily panels
    EXCEPT 2009-08 (rf-only) and 2009-09 (neither), so ret_exc exercises the
    coalesce(t30ret, rf) fallback and the both-null case.
    """
    rows = [
        (date(2000, 1, 31), 0.004),
        (date(2001, 6, 30), 0.005),
        (date(2002, 6, 28), 0.003),
        (date(2004, 1, 30), 0.002),
        (date(2005, 3, 31), 0.001),
        (date(2005, 4, 29), 0.0015),
        (date(2006, 5, 31), 0.002),
        (date(2007, 6, 29), 0.0025),
        (date(2008, 7, 31), 0.001),
    ]
    # Append bulk months (disjoint from the edge-case / rf-null-cascade months).
    rows += [(d, 0.003) for d in _bulk_rf_months()]
    return pl.DataFrame(
        {"caldt": [r[0] for r in rows], "t30ret": [r[1] for r in rows]},
        schema=SCHEMA_MCTI,
    )


def build_ff_input() -> pl.DataFrame:
    """Build ``ff_factors_monthly`` (shared monthly RF, one row/month).

    rf is present for every month EXCEPT 2009-09 (the neither case). 2009-08
    carries rf but no t30ret, so ret_exc there falls back to rf.
    """
    rows = [
        (date(2000, 1, 31), 0.0009),
        (date(2001, 6, 30), 0.0009),
        (date(2002, 6, 28), 0.0009),
        (date(2004, 1, 30), 0.0009),
        (date(2005, 3, 31), 0.0009),
        (date(2005, 4, 29), 0.0009),
        (date(2006, 5, 31), 0.0009),
        (date(2007, 6, 29), 0.0009),
        (date(2008, 7, 31), 0.0009),
        (date(2009, 8, 31), 0.002),  # rf-only fallback month
    ]
    # Append bulk months (disjoint from the edge-case / rf-null-cascade months).
    rows += [(d, 0.0009) for d in _bulk_rf_months()]
    return pl.DataFrame(
        {"date": [r[0] for r in rows], "rf": [r[1] for r in rows]},
        schema=SCHEMA_FF,
    )


def write_all_inputs(paths: DataPaths, freq: str) -> None:
    """Write the six input parquets ``prepare_crsp_sf`` reads for one freq.

    Description:
        Materialize the freq-specific panel + delist and the two shared monthly
        rf tables to the exact interim/raw_data_dfs paths the function scans.
    Steps:
        1) Ensure interim/raw_data_dfs exists.
        2) Write __crsp_sf_{freq}, crsp_{freq}sedelist, crsp_mcti_t30ret,
           ff_factors_monthly.
    Output:
        None (side effect: parquet files on disk).
    """
    raw_dir = paths.interim_dir / "raw_data_dfs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    build_crsp_sf_input(freq).write_parquet(raw_dir / f"__crsp_sf_{freq}.parquet")
    build_sedelist_input(freq).write_parquet(raw_dir / f"crsp_{freq}sedelist.parquet")
    build_mcti_input().write_parquet(raw_dir / "crsp_mcti_t30ret.parquet")
    build_ff_input().write_parquet(raw_dir / "ff_factors_monthly.parquet")


def write_empty_inputs(paths: DataPaths, freq: str) -> None:
    """Write 0-row typed versions of the four input parquets for one freq.

    Description:
        Materialize empty but fully-typed inputs so tests can assert that
        ``prepare_crsp_sf`` tolerates 0-row data without crashing.
    Steps:
        1) Ensure interim/raw_data_dfs exists.
        2) Write each table as an empty DataFrame carrying its schema.
    Output:
        None (side effect: 0-row parquet files on disk).
    """
    raw_dir = paths.interim_dir / "raw_data_dfs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(schema=SCHEMA_CRSP_SF).write_parquet(raw_dir / f"__crsp_sf_{freq}.parquet")
    pl.DataFrame(schema=SCHEMA_SEDELIST).write_parquet(raw_dir / f"crsp_{freq}sedelist.parquet")
    pl.DataFrame(schema=SCHEMA_MCTI).write_parquet(raw_dir / "crsp_mcti_t30ret.parquet")
    pl.DataFrame(schema=SCHEMA_FF).write_parquet(raw_dir / "ff_factors_monthly.parquet")
