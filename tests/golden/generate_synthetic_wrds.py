"""Generate synthetic WRDS-shaped parquets used by the factor-builder
golden tests (FF3/FF5/UMD, HXZ4, Stambaugh-Yuan mispricing).

License compliance
------------------
No real WRDS cell is ever written. Every value is independently sampled
from a plausible distribution. Identifiers live in synthetic bands
disjoint from real WRDS allocation (permno >= 9_000_000, gvkey >= 900000,
world_id >= 9_000_000_000), so the files cannot be reverse-engineered
to a real WRDS record. This satisfies the WRDS license clause permitting
synthetic data.

Usage
-----
    uv run python -m tests.golden.generate_synthetic_wrds

Deterministic (seed = 42). Re-running overwrites
``tests/golden/fixtures/synthetic_wrds/``.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED = 42
OUT_DIR = Path(__file__).parent / "fixtures" / "synthetic_wrds"

# 2000-2015 window (was 1995-2015): firms are born 1995-2004 (see
# build_firm_registry), so at the start nearly all have full in-window history
# for the momentum/distress warmup, while trimming the daily-fixture footprint
# vs the original 21-year span. A later start (e.g. 2005) thins the per-country
# ROW universe (50 firms each) below the world breakpoint gate and empties the
# ROW mispricing factors, so 2000 is the floor that keeps ROW coverage.
DATE_START = date(2000, 1, 1)
DATE_END = date(2015, 12, 31)

WRITE_OPTS: dict[str, object] = {
    "compression": "zstd",
    "compression_level": 22,
    "statistics": False,
    "row_group_size": 50_000,
}

ID_RANGES = {
    "permno": (9_000_000, 9_999_999),
    "permco": (900_000, 999_999),
    "gvkey_us": (900_000, 949_999),
    "gvkey_row": (950_000, 999_999),
    "world_id_row_base": 9_000_000_000,
}

COUNTRIES = ["USA", "CAN", "GBR", "JPN", "DEU", "AUS"]
ROW_COUNTRIES = ["CAN", "GBR", "JPN", "DEU", "AUS"]
CURRENCY = {"USA": "USD", "CAN": "CAD", "GBR": "GBP", "JPN": "JPY", "DEU": "EUR", "AUS": "AUD"}

US_EXCH_COUNTS = {"N": 300, "A": 100, "Q": 100}
ROW_FIRMS_PER_COUNTRY = 50


def _month_ends(start: date, end: date) -> list[date]:
    out: list[date] = []
    y, m = start.year, start.month
    while True:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        eom = date(ny, nm, 1) - timedelta(days=1)
        if eom > end:
            break
        if eom >= start:
            out.append(eom)
        y, m = ny, nm
    return out


def _weekdays(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _eom_of(d: date) -> date:
    ny, nm = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(ny, nm, 1) - timedelta(days=1)


def build_firm_registry(rng: np.random.Generator) -> pl.DataFrame:
    """One row per synthetic firm. Source of truth for cross-table IDs.

    Each US firm gets `world_id = permno`. ROW firms get
    `world_id >= 9_000_000_000`.
    """
    rows: list[dict] = []

    permno_lo, permno_hi = ID_RANGES["permno"]
    permco_lo, _ = ID_RANGES["permco"]
    gv_us_lo, _ = ID_RANGES["gvkey_us"]
    gv_row_lo, _ = ID_RANGES["gvkey_row"]
    world_row_base = ID_RANGES["world_id_row_base"]

    permno = permno_lo
    permco = permco_lo
    gvkey_us = gv_us_lo
    for exch, n in US_EXCH_COUNTS.items():
        for _ in range(n):
            inception = date(
                int(rng.integers(1995, 2005)),
                int(rng.integers(1, 13)),
                1,
            )
            delisted = bool(rng.random() < 0.15)
            delist_eom = (
                _eom_of(
                    date(
                        int(rng.integers(inception.year + 2, 2016)),
                        int(rng.integers(1, 13)),
                        1,
                    )
                )
                if delisted
                else None
            )
            rows.append(
                {
                    "permno": permno,
                    "permco": permco,
                    "gvkey": f"{gvkey_us:06d}",
                    "world_id": permno,
                    "excntry": "USA",
                    "primaryexch": exch,
                    "sic": int(rng.integers(1000, 9999)),
                    "naics": int(rng.integers(100000, 999999)),
                    "inception_eom": _eom_of(inception),
                    "delisting_eom": delist_eom,
                    "currency": "USD",
                    "source_crsp": 1,
                }
            )
            permno += 1
            permco += 1
            gvkey_us += 1

    world_id = world_row_base
    gvkey_row = gv_row_lo
    for ctry in ROW_COUNTRIES:
        for _ in range(ROW_FIRMS_PER_COUNTRY):
            inception = date(
                int(rng.integers(1995, 2005)),
                int(rng.integers(1, 13)),
                1,
            )
            delisted = bool(rng.random() < 0.15)
            delist_eom = (
                _eom_of(
                    date(
                        int(rng.integers(inception.year + 2, 2016)),
                        int(rng.integers(1, 13)),
                        1,
                    )
                )
                if delisted
                else None
            )
            rows.append(
                {
                    "permno": None,
                    "permco": None,
                    "gvkey": f"{gvkey_row:06d}",
                    "world_id": world_id,
                    "excntry": ctry,
                    "primaryexch": None,
                    "sic": int(rng.integers(1000, 9999)),
                    "naics": int(rng.integers(100000, 999999)),
                    "inception_eom": _eom_of(inception),
                    "delisting_eom": delist_eom,
                    "currency": CURRENCY[ctry],
                    "source_crsp": 0,
                }
            )
            world_id += 1
            gvkey_row += 1

    return pl.DataFrame(rows).with_columns(
        pl.col("permno").cast(pl.Int64),
        pl.col("permco").cast(pl.Int64),
        pl.col("world_id").cast(pl.Int64),
        pl.col("sic").cast(pl.Int64),
        pl.col("naics").cast(pl.Int64),
        pl.col("inception_eom").cast(pl.Date),
        pl.col("delisting_eom").cast(pl.Date),
        pl.col("source_crsp").cast(pl.Int8),
    )


def _firm_monthly_panel(
    reg_us: pl.DataFrame, month_ends: list[date], rng: np.random.Generator
) -> pl.DataFrame:
    """Expand US firms onto month_ends, generating per-firm random-walk
    return/ME paths within [inception_eom, delisting_eom]."""
    rows: list[dict] = []
    for r in reg_us.iter_rows(named=True):
        log_me = float(rng.normal(18.0, 1.5))
        for eom in month_ends:
            if eom < r["inception_eom"]:
                continue
            if r["delisting_eom"] is not None and eom > r["delisting_eom"]:
                continue
            ret = float(rng.normal(0.005, 0.08))
            ret = max(min(ret, 1.0), -0.6)
            log_me += float(np.log1p(ret))
            me = float(np.exp(log_me))
            rows.append(
                {
                    "permno": r["permno"],
                    "permco": r["permco"],
                    "gvkey": r["gvkey"],
                    "primaryexch": r["primaryexch"],
                    "mthcaldt": eom,
                    "mthret": ret,
                    "mthretx": ret - 0.001,
                    "me": me,
                }
            )
    return pl.DataFrame(rows)


def gen_crsp_msf_v2(reg: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Monthly CRSP CIZ panel for US firms."""
    reg_us = reg.filter(pl.col("excntry") == "USA")
    month_ends = _month_ends(DATE_START, DATE_END)
    panel = _firm_monthly_panel(reg_us, month_ends, rng)
    # shrout per firm = (first ME) / target price, so the implied price
    # (mthprc = me / shrout) starts in a realistic $10-$100 band. Drawing
    # shrout independently of ME (the old approach) yielded sub-$1 prices that
    # failed the mispricing penny-stock floor (lag_prc > 5), collapsing the
    # synthetic universe and emptying the mispricing factor goldens.
    first_me = panel.group_by("permno").agg(pl.col("me").first().alias("_me0")).sort("permno")
    target_price = rng.uniform(10.0, 100.0, size=first_me.height)
    shrout_per_firm = first_me.select(
        "permno",
        shrout=(pl.col("_me0") / pl.lit(target_price)).cast(pl.Float64),
    ).join(reg_us.select("permno", siccd=pl.col("sic")), on="permno")
    panel = panel.join(shrout_per_firm, on="permno")
    panel = panel.with_columns(
        mthprc=pl.col("me") / pl.col("shrout"),
        mthcap=pl.col("me"),
        mthcumfacshr=pl.lit(1.0),
        mthvol=pl.col("shrout") * 0.05,
    )
    return panel.with_columns(
        sharetype=pl.lit("NS"),
        securitytype=pl.lit("EQTY"),
        securitysubtype=pl.lit("COM"),
        usincflg=pl.lit("Y"),
        issuertype=pl.lit("CORP"),
        conditionaltype=pl.lit("RW"),
        tradingstatusflg=pl.lit("A"),
    ).select(
        "permno",
        "permco",
        "mthcaldt",
        "mthret",
        "mthretx",
        "mthprc",
        "mthcap",
        "mthvol",
        "mthcumfacshr",
        "shrout",
        "primaryexch",
        "siccd",
        "sharetype",
        "securitytype",
        "securitysubtype",
        "usincflg",
        "issuertype",
        "conditionaltype",
        "tradingstatusflg",
    )


def gen_crsp_dsf_v2(reg: pl.DataFrame, rng: np.random.Generator, msf: pl.DataFrame) -> pl.DataFrame:
    """Daily CRSP CIZ. ME path matches msf at month-ends, daily noise in
    between. Vol/prc derived from same path."""
    weekdays = _weekdays(DATE_START, DATE_END)
    reg_us = reg.filter(pl.col("excntry") == "USA")

    # Per-firm: shrout (constant), monthly anchor ME from msf
    shrout = msf.group_by("permno").agg(pl.col("shrout").first()).sort("permno")

    rows: list[dict] = []
    permno_to_perm = {r["permno"]: r for r in reg_us.iter_rows(named=True)}
    shrout_map = {r["permno"]: r["shrout"] for r in shrout.iter_rows(named=True)}

    n_firms = reg_us.height
    n_days = len(weekdays)
    # Per-firm daily returns matrix
    daily_rets = rng.normal(0.0003, 0.02, size=(n_firms, n_days))
    daily_rets = np.clip(daily_rets, -0.2, 0.2)

    weekday_arr = np.array(weekdays)
    for i, r in enumerate(reg_us.iter_rows(named=True)):
        permno = r["permno"]
        inc = r["inception_eom"]
        delist = r["delisting_eom"]
        mask = (weekday_arr >= inc) & (
            (delist is None) | (weekday_arr <= (delist if delist is not None else weekday_arr[-1]))
        )
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        log_me0 = float(rng.normal(18.0, 1.5))
        cum = np.cumsum(np.log1p(daily_rets[i, idx]))
        log_me_series = log_me0 + cum
        me_series = np.exp(log_me_series)
        shrout_val = float(shrout_map.get(permno, 1e7))
        prc_series = me_series / shrout_val
        for j_idx, d_i in enumerate(idx):
            d = weekdays[d_i]
            ret = float(daily_rets[i, d_i])
            rows.append(
                {
                    "permno": permno,
                    "permco": permno_to_perm[permno]["permco"],
                    "dlycaldt": d,
                    "dlyret": ret,
                    "dlyretx": ret - 0.0001,
                    "dlyprc": float(prc_series[j_idx]),
                    "dlycap": float(me_series[j_idx]),
                    "dlyvol": float(shrout_val * 0.002),
                    "shrout": shrout_val,
                    "primaryexch": permno_to_perm[permno]["primaryexch"],
                    "siccd": permno_to_perm[permno]["sic"],
                }
            )
    df = pl.DataFrame(rows)
    return df.with_columns(
        sharetype=pl.lit("NS"),
        securitytype=pl.lit("EQTY"),
        securitysubtype=pl.lit("COM"),
        usincflg=pl.lit("Y"),
        issuertype=pl.lit("CORP"),
        conditionaltype=pl.lit("RW"),
        tradingstatusflg=pl.lit("A"),
    ).select(
        "permno",
        "permco",
        "dlycaldt",
        "dlyret",
        "dlyretx",
        "dlyprc",
        "dlycap",
        "dlyvol",
        "shrout",
        "primaryexch",
        "siccd",
        "sharetype",
        "securitytype",
        "securitysubtype",
        "usincflg",
        "issuertype",
        "conditionaltype",
        "tradingstatusflg",
    )


def gen_crsp_stksecurityinfohist(reg: pl.DataFrame) -> pl.DataFrame:
    """One row per US firm spanning [inception, delisting or DATE_END]."""
    rows: list[dict] = []
    for r in reg.filter(pl.col("excntry") == "USA").iter_rows(named=True):
        rows.append(
            {
                "permno": r["permno"],
                "secinfostartdt": r["inception_eom"],
                "secinfoenddt": r["delisting_eom"]
                if r["delisting_eom"] is not None
                else date(2099, 12, 31),
                "siccd": r["sic"],
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("permno").cast(pl.Int64),
        pl.col("secinfostartdt").cast(pl.Date),
        pl.col("secinfoenddt").cast(pl.Date),
        pl.col("siccd").cast(pl.Int64),
    )


def gen_crsp_a_indexes_msp500(rng: np.random.Generator) -> pl.DataFrame:
    """Monthly S&P 500 index series. Columns: caldt, sprtrn, totval, vwretd, vwretx.

    Real WRDS crsp_a_indexes.msp500 is monthly (one row per month-end), and the
    distress calculation (``_mp_compute_distress``) joins it on
    ``caldt.dt.month_end()`` assuming one row per eom. Emitting a daily calendar
    here put ~22 rows per eom, fanning out that join so the downstream
    ``unique(keep="first")`` picked a nondeterministic survivor under DuckDB's
    parallel join order (only visible at high core counts) — making the distress
    char and the perf-leg / SMB mispricing factors nonreproducible. One row per
    month-end matches real WRDS and removes the fan-out.
    """
    months = _month_ends(DATE_START, DATE_END)
    n = len(months)
    sprtrn = rng.normal(0.008, 0.045, size=n)  # monthly S&P return magnitudes
    sprtrn = np.clip(sprtrn, -0.3, 0.3)
    totval = np.exp(np.cumsum(sprtrn) + 25.0)
    vwretd = sprtrn + rng.normal(0.0, 0.0005, size=n)
    vwretx = vwretd - 0.0001
    return pl.DataFrame(
        {
            "caldt": months,
            "sprtrn": sprtrn.astype(np.float64),
            "totval": totval.astype(np.float64),
            "vwretd": vwretd.astype(np.float64),
            "vwretx": vwretx.astype(np.float64),
        }
    ).with_columns(pl.col("caldt").cast(pl.Date))


def gen_crsp_a_indexes_acti(rng: np.random.Generator) -> pl.DataFrame:
    """Annual CPI series (December rows + monthly). Columns: caldt, cpiind."""
    # End-of-year + EoM rows; builders only use December rows.
    month_ends = _month_ends(DATE_START, DATE_END)
    n = len(month_ends)
    cpi = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.003, size=n)))
    return pl.DataFrame({"caldt": month_ends, "cpiind": cpi.astype(np.float64)}).with_columns(
        pl.col("caldt").cast(pl.Date)
    )


def _gen_funda(reg: pl.DataFrame, rng: np.random.Generator, is_global: bool) -> pl.DataFrame:
    """Annual Compustat (NA or Global). One row per (gvkey, fiscal_year_end)."""
    if is_global:
        sub = reg.filter(pl.col("excntry") != "USA")
    else:
        sub = reg.filter(pl.col("excntry") == "USA")

    rows: list[dict] = []
    years = list(range(DATE_START.year, DATE_END.year + 1))
    for r in sub.iter_rows(named=True):
        inc_y, delist_y = (
            r["inception_eom"].year,
            (r["delisting_eom"].year if r["delisting_eom"] is not None else DATE_END.year),
        )
        for y in years:
            if y < inc_y or y > delist_y:
                continue
            at = float(rng.lognormal(mean=18.0, sigma=1.5))
            seq = at * float(rng.uniform(0.1, 0.7))
            if rng.random() < 0.03:
                seq = -abs(seq * 0.1)
            ceq = max(seq * float(rng.uniform(0.7, 1.0)), 0.0)
            pstk = at * float(rng.uniform(0.0, 0.05))
            pstkl = pstk * float(rng.uniform(0.8, 1.2))
            pstkrv = pstk * float(rng.uniform(0.9, 1.1))
            txditc = at * float(rng.uniform(0.0, 0.05))
            txdb = txditc * float(rng.uniform(0.5, 1.5))
            lt = at * float(rng.uniform(0.3, 0.8))
            revt = at * float(rng.uniform(0.3, 1.5))
            sale = revt
            cogs = revt * float(rng.uniform(0.4, 0.8))
            xsga = revt * float(rng.uniform(0.05, 0.25))
            xint = revt * float(rng.uniform(0.0, 0.05))
            xopr = cogs + xsga
            oibdp = revt - cogs - xsga
            ebitda = oibdp
            dp = at * float(rng.uniform(0.02, 0.08))
            ib = oibdp - dp - xint
            ib *= float(rng.uniform(0.5, 1.0))
            pi = ib * float(rng.uniform(0.9, 1.1))
            ni = ib * float(rng.uniform(0.85, 1.05))
            ppegt = at * float(rng.uniform(0.3, 0.7))
            invt = at * float(rng.uniform(0.05, 0.2))
            act = at * float(rng.uniform(0.3, 0.6))
            lct = lt * float(rng.uniform(0.3, 0.5))
            che = act * float(rng.uniform(0.1, 0.3))
            dltt = lt * float(rng.uniform(0.3, 0.6))
            dlc = lt * float(rng.uniform(0.05, 0.15))
            mib = at * float(rng.uniform(0.0, 0.02))
            oancf = ib + dp
            csho = float(rng.uniform(5, 500))
            prcc_f = float(rng.uniform(5.0, 200.0))
            ajex = 1.0
            adjex_c = 1.0
            txp = at * float(rng.uniform(0.0, 0.02))

            rec = {
                "gvkey": r["gvkey"],
                "datadate": date(y, 12, 31),
                "fyear": y,
                "fyr": 12,
                "indfmt": "INDL",
                "datafmt": "HIST_STD" if is_global else "STD",
                "popsrc": "I" if is_global else "D",
                "consol": "C",
                "itcb": at * float(rng.uniform(0.0, 0.02)),
                "act": act,
                "adjex_c": adjex_c,
                "ajex": ajex,
                "at": at,
                "ceq": ceq,
                "che": che,
                "cogs": cogs,
                "csho": csho,
                "datafmt_marker": None,
                "dlc": dlc,
                "dltt": dltt,
                "dp": dp,
                "ebitda": ebitda,
                "ib": ib,
                "invt": invt,
                "lct": lct,
                "lt": lt,
                "mib": mib,
                "ni": ni,
                "oancf": oancf,
                "oibdp": oibdp,
                "pi": pi,
                "ppegt": ppegt,
                "prcc_f": prcc_f,
                "pstk": pstk,
                "pstkl": pstkl,
                "pstkrv": pstkrv,
                "revt": revt,
                "sale": sale,
                "seq": seq,
                "txdb": txdb,
                "txditc": txditc,
                "txp": txp,
                "xint": xint,
                "xopr": xopr,
                "xsga": xsga,
            }
            rec["curcd"] = r["currency"]
            if is_global:
                # Global table has narrower column set
                for k in (
                    "act",
                    "adjex_c",
                    "ajex",
                    "csho",
                    "dlc",
                    "dltt",
                    "invt",
                    "itcb",
                    "lct",
                    "mib",
                    "ni",
                    "oancf",
                    "pi",
                    "ppegt",
                    "prcc_f",
                    "pstkl",
                    "pstkrv",
                    "txp",
                ):
                    rec.pop(k, None)
            rec.pop("datafmt_marker", None)
            rows.append(rec)
    return pl.DataFrame(rows).with_columns(pl.col("datadate").cast(pl.Date))


def gen_comp_funda(reg: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    return _gen_funda(reg, rng, is_global=False)


def gen_comp_g_funda(reg: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    return _gen_funda(reg, rng, is_global=True)


def _gen_fundq(reg: pl.DataFrame, rng: np.random.Generator, is_global: bool) -> pl.DataFrame:
    """Quarterly Compustat. One row per (gvkey, fiscal quarter)."""
    if is_global:
        sub = reg.filter(pl.col("excntry") != "USA")
    else:
        sub = reg.filter(pl.col("excntry") == "USA")
    rows: list[dict] = []
    quarter_ends = [(m, _eom_of(date(2000, m, 1))) for m in (3, 6, 9, 12)]
    for r in sub.iter_rows(named=True):
        inc_y, delist_y = (
            r["inception_eom"].year,
            (r["delisting_eom"].year if r["delisting_eom"] is not None else DATE_END.year),
        )
        for y in range(DATE_START.year, DATE_END.year + 1):
            if y < inc_y or y > delist_y:
                continue
            for q, (month, _) in enumerate(quarter_ends, start=1):
                datadate = _eom_of(date(y, month, 1))
                rdq = datadate + timedelta(days=int(rng.integers(45, 90)))
                atq = float(rng.lognormal(18.0, 1.5))
                seqq = atq * float(rng.uniform(0.1, 0.7))
                ceqq = max(seqq * float(rng.uniform(0.7, 1.0)), 0.0)
                pstkq = atq * float(rng.uniform(0.0, 0.05))
                pstkrq = pstkq * float(rng.uniform(0.9, 1.1))
                ibq = atq * float(rng.uniform(-0.01, 0.05))
                saleq = atq * float(rng.uniform(0.1, 0.4))
                revtq = saleq
                niq = ibq * float(rng.uniform(0.85, 1.05))
                cheq = atq * float(rng.uniform(0.05, 0.2))
                ltq = atq * float(rng.uniform(0.3, 0.8))
                dlttq = ltq * float(rng.uniform(0.3, 0.6))
                actq = atq * float(rng.uniform(0.3, 0.6))
                lctq = ltq * float(rng.uniform(0.3, 0.5))
                piq = ibq * float(rng.uniform(0.9, 1.1))
                rec = {
                    "gvkey": r["gvkey"],
                    "datadate": datadate,
                    "fyearq": y,
                    "fqtr": q,
                    "fyr": 12,
                    "indfmt": "INDL",
                    "datafmt": "HIST_STD" if is_global else "STD",
                    "popsrc": "I" if is_global else "D",
                    "consol": "C",
                    "rdq": rdq,
                    "atq": atq,
                    "seqq": seqq,
                    "ceqq": ceqq,
                    "pstkq": pstkq,
                    "pstkrq": pstkrq,
                    "ibq": ibq,
                    "saleq": saleq,
                    "revtq": revtq,
                    "niq": niq,
                    "cheq": cheq,
                    "ltq": ltq,
                    "dlttq": dlttq,
                    "actq": actq,
                    "lctq": lctq,
                    "piq": piq,
                    "cshoq": float(rng.uniform(5, 500)),
                    "ajexq": 1.0,
                    "dvpsxq": float(rng.uniform(0.0, 2.0)),
                    "txditcq": atq * float(rng.uniform(0.0, 0.05)),
                }
                if is_global:
                    for k in (
                        "actq",
                        "ajexq",
                        "cshoq",
                        "dlttq",
                        "dvpsxq",
                        "lctq",
                        "piq",
                        "pstkrq",
                        "revtq",
                        "txditcq",
                    ):
                        rec.pop(k, None)
                rows.append(rec)
    return pl.DataFrame(rows).with_columns(
        pl.col("datadate").cast(pl.Date),
        pl.col("rdq").cast(pl.Date),
        pl.col("fyearq").cast(pl.Int32),
        pl.col("fqtr").cast(pl.Int16),
    )


def gen_comp_fundq(reg: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    return _gen_fundq(reg, rng, is_global=False)


def gen_comp_g_fundq(reg: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    return _gen_fundq(reg, rng, is_global=True)


def gen_ccm_lnkhist(reg: pl.DataFrame) -> pl.DataFrame:
    """CCM link table — one LC/P link per US firm spanning inception→delisting."""
    rows: list[dict] = []
    for r in reg.filter(pl.col("excntry") == "USA").iter_rows(named=True):
        rows.append(
            {
                "gvkey": r["gvkey"],
                "lpermno": r["permno"],
                "linktype": "LC",
                "linkprim": "P",
                "linkdt": r["inception_eom"],
                "linkenddt": r["delisting_eom"],
                "usedflag": 1,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("lpermno").cast(pl.Int64),
        pl.col("linkdt").cast(pl.Date),
        pl.col("linkenddt").cast(pl.Date),
        pl.col("usedflag").cast(pl.Int32),
    )


def gen_world_msf(
    reg: pl.DataFrame, rng: np.random.Generator, msf_us: pl.DataFrame
) -> pl.DataFrame:
    """World monthly panel. US rows mirror msf_us; ROW rows generated."""
    month_ends = _month_ends(DATE_START, DATE_END)
    # US rows from msf
    reg_us = reg.filter(pl.col("excntry") == "USA").select(
        "permno", "world_id", "gvkey", "excntry", "sic", "naics"
    )
    us = (
        msf_us.join(reg_us, on="permno")
        .rename({"mthcaldt": "eom", "mthret": "ret", "mthretx": "retx", "mthprc": "prc"})
        .with_columns(
            id=pl.col("world_id").cast(pl.Int64),
            me=pl.col("mthcap"),
            ret_exc=pl.col("ret") - 0.002,
            common=pl.lit(1, dtype=pl.Int8),
            primary_sec=pl.lit(1, dtype=pl.Int8),
            obs_main=pl.lit(1, dtype=pl.Int8),
            exch_main=pl.lit(1, dtype=pl.Int8),
            source_crsp=pl.lit(1, dtype=pl.Int8),
            size_grp=pl.lit("mega", dtype=pl.String),
        )
        .with_columns(me_lag1=pl.col("me").shift(1).over("permno"))
        .select(
            "id",
            "eom",
            "common",
            "exch_main",
            "excntry",
            "gvkey",
            "me",
            "me_lag1",
            "naics",
            "obs_main",
            "permno",
            "prc",
            "primary_sec",
            "ret",
            "ret_exc",
            "retx",
            "shrout",
            "sic",
            "size_grp",
            "source_crsp",
        )
    )

    # ROW rows
    reg_row = reg.filter(pl.col("excntry") != "USA")
    rows: list[dict] = []
    for r in reg_row.iter_rows(named=True):
        log_me = float(rng.normal(17.5, 1.5))
        # shrout = initial ME / target price so the implied price (me / shrout)
        # starts in a realistic $10-$100 band and clears the mispricing
        # penny-stock floor (lag_prc > 5). An independent shrout draw produced
        # sub-$1 ROW prices that emptied the world mispricing factor goldens.
        shrout = float(np.exp(log_me)) / float(rng.uniform(10.0, 100.0))
        for eom in month_ends:
            if eom < r["inception_eom"]:
                continue
            if r["delisting_eom"] is not None and eom > r["delisting_eom"]:
                continue
            ret = float(rng.normal(0.005, 0.08))
            ret = max(min(ret, 1.0), -0.6)
            log_me += float(np.log1p(ret))
            me = float(np.exp(log_me))
            rows.append(
                {
                    "id": r["world_id"],
                    "eom": eom,
                    "common": 1,
                    "exch_main": 1,
                    "excntry": r["excntry"],
                    "gvkey": r["gvkey"],
                    "me": me,
                    "me_lag1": None,
                    "naics": r["naics"],
                    "obs_main": 1,
                    "permno": None,
                    "prc": me / shrout,
                    "primary_sec": 1,
                    "ret": ret,
                    "ret_exc": ret - 0.002,
                    "retx": ret - 0.001,
                    "shrout": shrout,
                    "sic": r["sic"],
                    "size_grp": "mega",
                    "source_crsp": 0,
                }
            )
    row_df = pl.DataFrame(rows).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col("permno").cast(pl.Int64),
        pl.col("sic").cast(pl.Int64),
        pl.col("naics").cast(pl.Int64),
        pl.col("eom").cast(pl.Date),
        pl.col("common").cast(pl.Int8),
        pl.col("exch_main").cast(pl.Int8),
        pl.col("obs_main").cast(pl.Int8),
        pl.col("primary_sec").cast(pl.Int8),
        pl.col("source_crsp").cast(pl.Int8),
    )
    row_df = row_df.sort(["id", "eom"]).with_columns(me_lag1=pl.col("me").shift(1).over("id"))
    return pl.concat([us, row_df], how="diagonal_relaxed")


def gen_world_dsf(
    reg: pl.DataFrame, rng: np.random.Generator, dsf_us: pl.DataFrame
) -> pl.DataFrame:
    """World daily panel — ROW only.

    Every builder that reads ``world_dsf.parquet`` filters
    ``excntry != USA`` (see ``_mp_world_load_world_dsf``,
    ``ff_load_world_panel``, ``hxz_load_panel_row``). Emitting US rows is
    dead weight that pushes the parquet over GitHub's 100 MB hard limit
    on a synthetic build large enough to exercise ROW breakpoints.
    """
    del dsf_us
    weekdays = _weekdays(DATE_START, DATE_END)

    reg_row = reg.filter(pl.col("excntry") != "USA")
    rows: list[dict] = []
    weekday_arr = np.array(weekdays)
    n_days = len(weekdays)
    n_row = reg_row.height
    daily_rets_row = rng.normal(0.0003, 0.02, size=(n_row, n_days))
    daily_rets_row = np.clip(daily_rets_row, -0.2, 0.2)
    for i, r in enumerate(reg_row.iter_rows(named=True)):
        inc = r["inception_eom"]
        delist = r["delisting_eom"] if r["delisting_eom"] is not None else weekday_arr[-1]
        mask = (weekday_arr >= inc) & (weekday_arr <= delist)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        log_me0 = float(rng.normal(17.5, 1.5))
        # shrout = initial ME / target price → realistic $10-$100 start price
        # (matches gen_world_msf; avoids sub-$1 prices failing lag_prc > 5).
        shrout = float(np.exp(log_me0)) / float(rng.uniform(10.0, 100.0))
        cum = np.cumsum(np.log1p(daily_rets_row[i, idx]))
        me_series = np.exp(log_me0 + cum)
        prc_series = me_series / shrout
        for j_idx, d_i in enumerate(idx):
            d = weekdays[d_i]
            ret = float(daily_rets_row[i, d_i])
            prc = float(prc_series[j_idx])
            rows.append(
                {
                    "id": r["world_id"],
                    "date": d,
                    "eom": _eom_of(d),
                    "excntry": r["excntry"],
                    "common": 1,
                    "primary_sec": 1,
                    "obs_main": 1,
                    "exch_main": 1,
                    "ret": ret,
                    "me": float(me_series[j_idx]),
                    "prc": prc,
                    "prc_high": prc * 1.01,
                    "prc_low": prc * 0.99,
                    "tvol": shrout * 0.002,
                    "bidask": 0.01,
                    "adjfct": 1.0,
                }
            )
    row_df = pl.DataFrame(rows).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col("date").cast(pl.Date),
        pl.col("eom").cast(pl.Date),
        pl.col("common").cast(pl.Int8),
        pl.col("primary_sec").cast(pl.Int8),
        pl.col("obs_main").cast(pl.Int8),
        pl.col("exch_main").cast(pl.Int8),
    )
    return row_df


def gen_world_data(
    reg: pl.DataFrame, rng: np.random.Generator, world_msf: pl.DataFrame
) -> pl.DataFrame:
    """world_data: world_msf rows + extra char columns expected by mispricing.

    Mispricing world stage reads these char columns; populate with plausible
    random values so anomalies / mispricing scores compute.
    """
    n = world_msf.height
    return world_msf.with_columns(
        adjfct=pl.lit(1.0),
        shares=pl.col("me") / pl.col("prc"),
        beme=pl.Series("beme", rng.uniform(0.1, 3.0, size=n)),
        dolvol=pl.col("me") * 0.01,
        mom=pl.Series("mom", rng.normal(0.0, 0.2, size=n)),
        mscore=pl.Series("mscore", rng.normal(-2.5, 1.0, size=n)),
        roe=pl.Series("roe", rng.normal(0.05, 0.1, size=n)),
        rsup=pl.Series("rsup", rng.normal(0.0, 0.02, size=n)),
        issue=pl.Series("issue", rng.normal(0.0, 0.1, size=n)),
        oaccruals_at=pl.Series("oaccruals_at", rng.normal(0.0, 0.05, size=n)),
        at_gr1=pl.Series("at_gr1", rng.normal(0.05, 0.2, size=n)),
        gp_at=pl.Series("gp_at", rng.uniform(0.0, 0.6, size=n)),
        ppeinv_gr1a=pl.Series("ppeinv_gr1a", rng.normal(0.05, 0.15, size=n)),
        noa_at=pl.Series("noa_at", rng.uniform(0.0, 1.5, size=n)),
        o_score=pl.Series("o_score", rng.normal(-1.0, 1.5, size=n)),
        niq_at=pl.Series("niq_at", rng.normal(0.01, 0.02, size=n)),
    ).select(
        "id",
        "permno",
        "gvkey",
        "excntry",
        "eom",
        "me",
        "prc",
        "ret",
        "adjfct",
        "shares",
        "common",
        "primary_sec",
        "obs_main",
        "exch_main",
        "size_grp",
        "beme",
        "dolvol",
        "issue",
        "mom",
        "mscore",
        "roe",
        "rsup",
        "oaccruals_at",
        "at_gr1",
        "gp_at",
        "ppeinv_gr1a",
        "noa_at",
        "o_score",
        "niq_at",
    )


def gen_market_returns(world_msf: pl.DataFrame) -> pl.DataFrame:
    """Per-(excntry, eom) market return + lagged ME aggregate."""
    mr = (
        world_msf.filter(
            (pl.col("common") == 1) & (pl.col("primary_sec") == 1) & (pl.col("obs_main") == 1)
        )
        .group_by(["excntry", "eom"])
        .agg(
            mkt_vw=(pl.col("ret") * pl.col("me_lag1")).sum() / pl.col("me_lag1").sum(),
            me_lag1=pl.col("me_lag1").sum(),
        )
        .with_columns(mkt_vw_exc=pl.col("mkt_vw") - 0.002)
        .sort(["excntry", "eom"])
    )
    return mr


def gen_market_returns_daily(world_dsf: pl.DataFrame) -> pl.DataFrame:
    """Per-(excntry, date) market daily return + lagged ME aggregate."""
    return (
        world_dsf.filter(
            (pl.col("common") == 1) & (pl.col("primary_sec") == 1) & (pl.col("obs_main") == 1)
        )
        .group_by(["excntry", "date"])
        .agg(
            mkt_vw=(pl.col("ret") * pl.col("me")).sum() / pl.col("me").sum(),
            me_lag1=pl.col("me").sum(),
        )
        .with_columns(mkt_vw_exc=pl.col("mkt_vw") - 0.0001)
        .sort(["excntry", "date"])
    )


def gen_fx1(rng: np.random.Generator) -> pl.DataFrame:
    """FX rates per (curcdd, datadate). End-of-month resolution sufficient
    for annual Compustat conversion.

    All currencies share fx=1.0 to sidestep a non-determinism in
    ``ff_load_world_compustat``'s ``join_asof(by="curcd")`` over FX, which
    surfaces when the by-group lookup ties — synthetic data triggers it
    more often than real WRDS data. Using fx=1.0 makes BE-USD identical
    to BE-local and stabilises the regression test.
    """
    del rng
    month_ends = _month_ends(DATE_START, DATE_END)
    rows: list[dict] = []
    for ccy in set(CURRENCY.values()):
        for d in month_ends:
            rows.append({"curcdd": ccy, "datadate": d, "fx": 1.0})
    return pl.DataFrame(rows).with_columns(pl.col("datadate").cast(pl.Date))


def _write(df: pl.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    df.write_parquet(out, **WRITE_OPTS)
    print(
        f"  wrote {out.relative_to(OUT_DIR.parent.parent)} ({out.stat().st_size / 1e6:.2f} MB, {df.height:,} rows)"
    )


def gen_acc_std_ann(funda: pl.DataFrame, g_funda: pl.DataFrame) -> pl.DataFrame:
    """Synthesize jkp's standardized annual accounting panel (``acc_std_ann``).

    The ROW FF/HXZ loaders read book equity (be_x), operating-profit numerator
    (ope_x), assets (at_x -> asset growth) and net income (ni_x) from this panel
    via ``_jkp_annual_acc``. The real builder (``standardized_accounting_data``)
    pulls ~40 Compustat vars this synthetic fixture doesn't carry, so instead we
    reproduce the panel's *shape and the four fields the loaders consume*:

      * monthly-expanded grid per (gvkey, curcd) from the firm's first to last
        fiscal report (so the loader's ``at_x.shift(12)`` sees the prior-year
        report exactly 12 rows back);
      * ``source`` and the ``_x`` fields populated only on report months (Dec),
        null on the fill months in between — matching production, where
        ``_jkp_annual_acc`` filters ``source.is_not_null()``;
      * ``_x`` fields computed with jkp's standardization identities.
    """
    raw_cols = [
        "seq",
        "ceq",
        "pstk",
        "pstkrv",
        "pstkl",
        "txditc",
        "txdb",
        "itcb",
        "at",
        "lt",
        "sale",
        "revt",
        "cogs",
        "xsga",
        "xopr",
        "xint",
        "ebitda",
        "oibdp",
        "ib",
    ]

    def _report_rows(df: pl.DataFrame, source: str) -> pl.DataFrame:
        missing = [c for c in raw_cols if c not in df.columns]
        df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
        pstk_x = pl.coalesce("pstkrv", "pstkl", "pstk")
        txditc_x = pl.coalesce("txditc", pl.col("txdb") + pl.col("itcb"))
        seq_x = pl.coalesce("seq", pl.col("ceq") + pstk_x, pl.col("at") - pl.col("lt"))
        be_x = seq_x + pl.coalesce(txditc_x, pl.lit(0.0)) - pl.coalesce(pstk_x, pl.lit(0.0))
        opex_x = pl.coalesce("xopr", pl.col("cogs") + pl.col("xsga"))
        ebitda_x = pl.coalesce("ebitda", "oibdp", pl.coalesce("sale", "revt") - opex_x)
        return df.select(
            "gvkey",
            "datadate",
            "curcd",
            source=pl.lit(source),
            be_x=pl.when(be_x > 0).then(be_x).otherwise(None),
            ope_x=ebitda_x - pl.col("xint"),
            at_x=pl.col("at"),
            ni_x=pl.col("ib"),
        )

    reports = pl.concat(
        [_report_rows(funda, "NA"), _report_rows(g_funda, "GLOBAL")], how="vertical_relaxed"
    ).unique(["gvkey", "datadate"], keep="first")

    # Monthly grid per gvkey (first->last report), month-ends; report rows land on Dec.
    grid = (
        reports.group_by("gvkey")
        .agg(
            lo=pl.col("datadate").min(), hi=pl.col("datadate").max(), curcd=pl.col("curcd").first()
        )
        .with_columns(datadate=pl.date_ranges("lo", "hi", interval="1mo"))
        .explode("datadate")
        .select("gvkey", "curcd", "datadate")
    )
    return grid.join(reports.drop("curcd"), on=["gvkey", "datadate"], how="left").sort(
        ["gvkey", "curcd", "datadate"]
    )


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR, type=Path)
    ap.add_argument("--seed", default=SEED, type=int)
    args = ap.parse_args()
    OUT_DIR = args.out

    rng = np.random.default_rng(args.seed)
    print("Building firm registry...")
    reg = build_firm_registry(rng)
    print(
        f"  {reg.height} firms ({(reg['excntry'] == 'USA').sum()} US, "
        f"{(reg['excntry'] != 'USA').sum()} ROW)"
    )

    print("Generating crsp_msf_v2...")
    msf = gen_crsp_msf_v2(reg, rng)
    _write(msf, "crsp_msf_v2.parquet")

    print("Generating crsp_dsf_v2...")
    dsf = gen_crsp_dsf_v2(reg, rng, msf)
    _write(dsf, "crsp_dsf_v2.parquet")

    print("Generating crsp_stksecurityinfohist...")
    _write(gen_crsp_stksecurityinfohist(reg), "crsp_stksecurityinfohist.parquet")

    print("Generating crsp_a_indexes_msp500...")
    _write(gen_crsp_a_indexes_msp500(rng), "crsp_a_indexes_msp500.parquet")

    print("Generating crsp_a_indexes_acti...")
    _write(gen_crsp_a_indexes_acti(rng), "crsp_a_indexes_acti.parquet")

    print("Generating comp_funda...")
    funda = gen_comp_funda(reg, rng)
    _write(funda, "comp_funda.parquet")

    print("Generating comp_fundq...")
    _write(gen_comp_fundq(reg, rng), "comp_fundq.parquet")

    print("Generating comp_g_funda...")
    g_funda = gen_comp_g_funda(reg, rng)
    _write(g_funda, "comp_g_funda.parquet")

    print("Generating comp_g_fundq...")
    _write(gen_comp_g_fundq(reg, rng), "comp_g_fundq.parquet")

    print("Generating crsp_ccmxpf_lnkhist...")
    _write(gen_ccm_lnkhist(reg), "crsp_ccmxpf_lnkhist.parquet")

    print("Generating world_msf...")
    world_msf = gen_world_msf(reg, rng, msf)
    _write(world_msf, "world_msf.parquet")

    print("Generating world_dsf...")
    world_dsf = gen_world_dsf(reg, rng, dsf)
    _write(world_dsf, "world_dsf.parquet")

    print("Generating world_data...")
    _write(gen_world_data(reg, rng, world_msf), "world_data.parquet")

    print("Generating market_returns...")
    _write(gen_market_returns(world_msf), "market_returns.parquet")

    print("Generating market_returns_daily...")
    _write(gen_market_returns_daily(world_dsf), "market_returns_daily.parquet")

    print("Generating __fx1...")
    _write(gen_fx1(rng), "__fx1.parquet")

    print("Generating acc_std_ann (jkp standardized accounting)...")
    _write(gen_acc_std_ann(funda, g_funda), "acc_std_ann.parquet")

    total = sum((OUT_DIR / f).stat().st_size for f in OUT_DIR.iterdir() if f.is_file())
    print(f"\nTotal: {total / 1e6:.1f} MB across {len(list(OUT_DIR.iterdir()))} files")


if __name__ == "__main__":
    main()
