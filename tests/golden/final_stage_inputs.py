"""Seeded synthetic inputs for the final merge and save stage.

Imported by ``generate_final_stage_golden.py``, the golden test, and the
integration test, so the schema is defined once.

The panel is 12 ids over 6 month-ends (2020-01..2020-06) and 20 trading days.
Countries: USA (ids 1-5), GBR (6-8), DEU (9-10) and a null ``excntry``
(ids 11-12). Edge cases the joins and filters must handle:

* ids absent from a right-hand table (left joins keep the row, columns null);
* right-hand rows whose keys are not in msf (must not add rows);
* a null ``excntry`` (no match in the qmj join, ``null_country`` daily file);
* null ``source`` rows in the accounting tables (dropped on save);
* MAIN_FILTERS: id 5 never common, id 8 loses exch_main in some months,
  id 12 has a null primary_sec.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from jkp.data.paths import DataPaths

EOMS = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31), date(2020, 4, 30)] + [
    date(2020, 5, 31),
    date(2020, 6, 30),
]
DAYS = [date(2020, 3, 2) + timedelta(days=i) for i in range(20)]
COUNTRY = {
    **dict.fromkeys(range(1, 6), "USA"),
    **dict.fromkeys(range(6, 9), "GBR"),
    **dict.fromkeys((9, 10), "DEU"),
    **dict.fromkeys((11, 12)),
}
IDS = sorted(COUNTRY)
# (id, eom) pairs removed from msf so the panel is not rectangular.
MSF_GAPS = {(2, EOMS[1]), (7, EOMS[3]), (10, EOMS[0]), (12, EOMS[5])}

RET_COLS = [
    "ret_exc",
    "ret",
    "ret_local",
    "ret_exc_wins",
    "ret_intraday",
    "ret_overnight",
    "ret_intraday_local",
    "ret_overnight_local",
    "ret_intraday_lead1m",
    "ret_overnight_lead1m",
    "ret_intraday_local_lead1m",
    "ret_overnight_local_lead1m",
]
OTHER_OUTPUT_FILES = [
    "market_returns.parquet",
    "nyse_cutoffs.parquet",
    "return_cutoffs.parquet",
    "return_cutoffs_daily.parquet",
    "ap_factors_monthly.parquet",
    "ap_factors_daily.parquet",
    "market_returns_daily.parquet",
]


def _filters(i: int, eom: date) -> dict:
    """MAIN_FILTERS columns for one id/date, with the edge cases listed above."""
    return {
        "primary_sec": None if i == 12 else 1,
        "common": 0 if i == 5 else 1,
        "obs_main": 1,
        "exch_main": 0 if i == 8 and eom <= EOMS[2] else 1,
    }


def _gvkey(i: int) -> str | None:
    return None if i == 12 else f"G{i:03d}"


def build_msf(seed: int = 42) -> pl.DataFrame:
    """Monthly stock file with the columns create_world_data_prelim and save_monthly_ret read."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in IDS:
        for eom in EOMS:
            if (i, eom) in MSF_GAPS:
                continue
            r = {
                "id": i,
                "gvkey": _gvkey(i),
                "permco": 100 + i,
                "permno": 1000 + i,
                "excntry": COUNTRY[i],
                "eom": eom,
                "source_crsp": 1 if i <= 5 else 0,
                "source": None if i == 3 else "crsp" if i <= 5 else "comp",
                "div_tot": float(rng.integers(0, 3)),
                "div_cash": float(rng.integers(0, 3)),
                "div_spc": 0.0,
                "me": float(np.round(rng.uniform(50, 5000), 2)),
                **_filters(i, eom),
            }
            for c in RET_COLS:
                r[c] = None if rng.random() < 0.1 else float(np.round(rng.normal(0, 0.08), 6))
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col("eom").cast(pl.Date))


def _keyed(msf: pl.DataFrame, drop_ids: set[int], seed: int, cols: list[str]) -> pl.DataFrame:
    """One row per msf (id, eom) except ``drop_ids``, plus an extra id absent from msf."""
    rng = np.random.default_rng(seed)
    keys = msf.filter(~pl.col("id").is_in(list(drop_ids))).select("id", "eom")
    extra = pl.DataFrame({"id": [99], "eom": [EOMS[0]]}, schema={"id": pl.Int64, "eom": pl.Date})
    keys = pl.concat([keys.cast({"id": pl.Int64}), extra])
    vals = {c: np.round(rng.normal(0, 1, keys.height), 5) for c in cols}
    return keys.with_columns([pl.Series(c, v) for c, v in vals.items()])


def build_market_chars(msf: pl.DataFrame) -> pl.DataFrame:
    return _keyed(msf, {3, 9}, 1, ["market_equity", "mktcorr_x"])


def build_acc_chars(msf: pl.DataFrame) -> pl.DataFrame:
    """Accounting chars keyed (gvkey, public_date); gvkeys G004 and G006 never appear."""
    rng = np.random.default_rng(2)
    rows = []
    for g in (1, 2, 3, 5, 7, 8, 9, 10, 77):
        for eom in EOMS[1:]:
            rows.append(
                {"gvkey": f"G{g:03d}", "public_date": eom, "at_x": float(rng.normal(100, 10))}
            )
    return pl.DataFrame(rows).with_columns(pl.col("public_date").cast(pl.Date))


def build_factor(msf: pl.DataFrame, name: str, seed: int, drop: set[int]) -> pl.DataFrame:
    return _keyed(msf, drop, seed, [name])


def build_firm_age(msf: pl.DataFrame) -> pl.DataFrame:
    df = _keyed(msf, {4}, 8, [])
    return df.with_columns(age=pl.int_range(pl.len()).cast(pl.Int64), age_extra=pl.lit(1.0))


def build_qmj(msf: pl.DataFrame) -> pl.DataFrame:
    """qmj keyed (excntry, id, eom); id 6 missing, id 99 extra, null excntry rows present."""
    rng = np.random.default_rng(9)
    df = msf.filter(pl.col("id") != 6).select("excntry", "id", "eom")
    df = df.with_columns(qmj=pl.Series(np.round(rng.normal(0, 1, df.height), 5)))
    extra = pl.DataFrame(
        {"excntry": ["USA"], "id": [99], "eom": [EOMS[0]], "qmj": [0.5]},
        schema=df.schema,
    )
    return pl.concat([df, extra])


def build_daily(seed: int = 42) -> pl.DataFrame:
    """Daily stock file; ids 1-12 over DAYS, same filter edge cases."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in IDS:
        for d in DAYS:
            r = {
                "id": i,
                "excntry": COUNTRY[i],
                "date": d,
                "me": float(np.round(rng.uniform(50, 5000), 2)),
                **_filters(i, d.replace(day=1)),
            }
            for c in (
                "ret",
                "ret_exc",
                "ret_exc_wins",
                "ret_intraday",
                "ret_overnight",
                "ret_intraday_local",
                "ret_overnight_local",
            ):
                r[c] = None if rng.random() < 0.05 else float(np.round(rng.normal(0, 0.02), 6))
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col("date").cast(pl.Date))


def build_world_data(msf: pl.DataFrame) -> pl.DataFrame:
    """Final characteristic file: msf keys, filters, me and two characteristics."""
    rng = np.random.default_rng(5)
    df = msf.select(
        "id", "excntry", "eom", "me", *["primary_sec", "common", "obs_main", "exch_main"]
    )
    return df.with_columns(
        be_me=pl.Series(np.round(rng.normal(0.5, 0.2, df.height), 5)),
        ret_12_1=pl.Series(np.round(rng.normal(0.1, 0.3, df.height), 5)),
    )


def build_acc_std(kind: str) -> pl.DataFrame:
    """Standardized accounting file; rows with a null ``source`` are dropped on save."""
    rng = np.random.default_rng(11 if kind == "qtr" else 12)
    rows = []
    for g in range(1, 8):
        for k in range(3):
            rows.append(
                {
                    "gvkey": f"G{g:03d}",
                    "datadate": [date(2019, 3, 31), date(2019, 9, 30), date(2020, 3, 31)][k],
                    "source": None if (g + k) % 4 == 0 else "crsp",
                    "at": float(np.round(rng.uniform(10, 1000), 2)),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("datadate").cast(pl.Date))


def build_other_output(name: str) -> pl.DataFrame:
    return pl.DataFrame({"k": [1, 2, 3], "v": [float(len(name)), 2.0, 3.0]})


def write_merge_inputs(paths: DataPaths, seed: int = 42) -> None:
    """Interim inputs for create_world_data_prelim through merge_qmj_to_world_data."""
    d = paths.interim_dir
    d.mkdir(parents=True, exist_ok=True)
    msf = build_msf(seed)
    msf.write_parquet(d / "world_msf.parquet")
    build_market_chars(msf).write_parquet(d / "market_chars_m.parquet")
    build_acc_chars(msf).write_parquet(d / "acc_chars_world.parquet")
    build_factor(msf, "beta_60m", 3, {2}).write_parquet(d / "beta_60m.parquet")
    build_factor(msf, "resmom_12_1", 4, {5}).write_parquet(d / "resmom_ff3_12_1.parquet")
    build_factor(msf, "resmom_6_1", 5, {5, 6}).write_parquet(d / "resmom_ff3_6_1.parquet")
    build_factor(msf, "mp_factor", 6, {7}).write_parquet(d / "mp_factors.parquet")
    build_factor(msf, "rvol_x", 7, {1}).write_parquet(d / "market_chars_d.parquet")
    build_firm_age(msf).write_parquet(d / "firm_age.parquet")
    build_qmj(msf).write_parquet(d / "qmj.parquet")


def write_industry_inputs(paths: DataPaths, seed: int = 42) -> None:
    """``__msf_world``, ``comp_ind`` and ``crsp_ind`` for merge_industry_to_world_msf."""
    d = paths.interim_dir
    d.mkdir(parents=True, exist_ok=True)
    msf = build_msf(seed).select("id", "gvkey", "permco", "permno", "excntry", "eom")
    msf.write_parquet(d / "__msf_world.parquet")
    sched = {1: (100, 10), 2: (None, None), 3: (300, None), 6: (600, 60)}
    comp = [
        {"gvkey": f"G{i:03d}", "date": e, "gics": 5, "sic": s, "naics": n}
        for i, (s, n) in sched.items()
        for e in EOMS
    ]
    pl.DataFrame(
        comp,
        schema={
            "gvkey": pl.Utf8,
            "date": pl.Date,
            "gics": pl.Int64,
            "sic": pl.Int64,
            "naics": pl.Int64,
        },
    ).write_parquet(d / "comp_ind.parquet")
    sched_c = {1: (111, 11), 2: (222, 22), 4: (None, 44), 6: (666, 66), 9: (999, None)}
    crsp = [
        {"permco": 100 + i, "permno": 1000 + i, "date": e, "sic": s, "naics": n}
        for i, (s, n) in sched_c.items()
        for e in EOMS
    ]
    pl.DataFrame(
        crsp,
        schema={
            "permco": pl.Int64,
            "permno": pl.Int64,
            "date": pl.Date,
            "sic": pl.Int64,
            "naics": pl.Int64,
        },
    ).write_parquet(d / "crsp_ind.parquet")


def write_save_inputs(paths: DataPaths, seed: int = 42) -> None:
    """Interim inputs for the filter/save tail (world_dsf/msf/data, acc_std_*, other_output)."""
    d = paths.interim_dir
    d.mkdir(parents=True, exist_ok=True)
    msf = build_msf(seed)
    msf.write_parquet(d / "world_msf.parquet")
    build_daily(seed).write_parquet(d / "world_dsf.parquet")
    build_world_data(msf).write_parquet(d / "world_data.parquet")
    build_acc_std("qtr").write_parquet(d / "acc_std_qtr.parquet")
    build_acc_std("ann").write_parquet(d / "acc_std_ann.parquet")
    for name in OTHER_OUTPUT_FILES:
        build_other_output(name).write_parquet(d / name)


def write_full_inputs(paths: DataPaths, seed: int = 42) -> None:
    """Write every interim input the final merge and save functions need."""
    write_merge_inputs(paths, seed)
    write_industry_inputs(paths, seed)
    write_save_inputs(paths, seed)
