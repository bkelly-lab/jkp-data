"""Generate synthetic WRDS-shaped parquets used by the factor-builder
golden test (FF3/FF5/UMD).

License compliance
------------------
No real WRDS cell is ever written. Every value is independently sampled
from a plausible distribution. Identifiers live in synthetic bands
disjoint from real WRDS allocation (permno >= 9_000_000, gvkey >= 900000,
world_id >= 9_000_000_000), so the files cannot be reverse-engineered
to a real WRDS record. This satisfies the WRDS license clause permitting
synthetic data.

Nothing generated here is committed: the golden tests call
``write_synthetic_wrds`` into a temporary directory (session fixture in
``conftest.py``). Values are rounded to the real WRDS scale at generation time,
nulls are injected by sampling the real tables' null patterns (``synthetic_null_patterns.json``,
drawn from a separate RNG so base values do not depend on them), and columns are
cast to the real WRDS dtypes only when written.

Usage
-----
    uv run python -m tests.golden.generate_synthetic_wrds --out DIR [--seed N]

Deterministic for a given seed on a given CPU type.
"""

from __future__ import annotations

import argparse
import functools
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from jkp.data.config import BE_DEFERRED_TAX_END_YEAR

SEED = 42

# 2000-2015 window: firms are born 1995-2004 (see build_firm_registry), so at the
# start nearly all have full in-window history for the momentum warmup.
DATE_START = date(2000, 1, 1)
DATE_END = date(2015, 12, 31)

ID_RANGES = {
    "permno": (9_000_000, 9_999_999),
    "permco": (900_000, 999_999),
    "gvkey_us": (900_000, 949_999),
    "gvkey_row": (950_000, 999_999),
    "world_id_row_base": 9_000_000_000,
}

COUNTRIES = ["USA", "CAN", "GBR", "JPN", "DEU", "AUS"]
CURRENCY = {"USA": "USD", "CAN": "CAD", "GBR": "GBP", "JPN": "JPY", "DEU": "EUR", "AUS": "AUD"}

# Firm counts chosen so the generated composition matches the real 2000-2015 tables: CRSP-sourced
# (US) rows are ~14% of world_msf (permno present), and Compustat North America (US + CAN) is
# ~26% of acc_std_ann rows, the rest Compustat Global.
US_EXCH_COUNTS = {"N": 96, "A": 32, "Q": 32}
ROW_FIRMS = {"CAN": 130, "GBR": 200, "JPN": 200, "DEU": 200, "AUS": 200}
# Compustat North America covers the US and Canada; Global holds every other country.
NA_COUNTRIES = ("USA", "CAN")
NULL_PATTERNS_PATH = Path(__file__).with_name("synthetic_null_patterns.json")

# fmt: off
# Real WRDS dtypes (measured 2026-11-09); generation stays Float64, cast only when writing.
_D = pl.Decimal
REAL_DTYPES: dict[str, dict[str, pl.DataType]] = {
    "crsp_msf_v2": {
        "mthret": _D(10, 6), "mthretx": _D(10, 6), "mthprc": _D(13, 6), "mthcap": _D(13, 2),
        "mthvol": _D(14, 0), "mthcumfacshr": _D(17, 12), "shrout": pl.Int32, "siccd": pl.Int32,
    },
    "crsp_dsf_v2": {
        "dlyret": _D(10, 6), "dlyretx": _D(10, 6), "dlyprc": _D(13, 6), "dlycap": _D(13, 2),
        "dlyvol": _D(14, 0), "shrout": pl.Int32, "siccd": pl.Int32,
    },
    "world_msf": dict.fromkeys(
        ["common", "exch_main", "obs_main", "primary_sec", "source_crsp"], pl.Int32
    ),
    "world_dsf": dict.fromkeys(
        ["common", "primary_sec", "obs_main", "exch_main", "bidask"], pl.Int32
    ),
}
# Columns the real tables lack; generated internally, dropped when written.
DROP_COLS: dict[str, list[str]] = {
    "world_msf": ["me_lag1", "retx", "shrout"],
}
# Compustat items carry Decimal(18,4) except these (Decimal(24,12)).
COMP_WIDE_ITEMS = {"ajex", "adjex_c", "prcc_f", "ajexq"}
COMP_INT_COLS = {"fyear": pl.Int32, "fyearq": pl.Int32, "fyr": pl.Int16, "fqtr": pl.Int16}
# fmt: on


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


def _round_to(df: pl.DataFrame, scales: dict[str, int]) -> pl.DataFrame:
    """Round Float64 columns to the real WRDS decimal scale."""
    return df.with_columns(pl.col(c).round(s) for c, s in scales.items())


def _round_comp(df: pl.DataFrame) -> pl.DataFrame:
    """Round every Compustat item to its real scale (12 decimals for the wide items, else 4)."""
    return _round_to(
        df,
        {c: 12 if c in COMP_WIDE_ITEMS else 4 for c, t in df.schema.items() if t == pl.Float64},
    )


@functools.cache
def _null_patterns() -> dict:
    return json.loads(NULL_PATTERNS_PATH.read_text())


def _apply_null_patterns(
    df: pl.DataFrame,
    table: str,
    nrng: np.random.Generator,
    stratum_mask: np.ndarray | None = None,
    given: tuple[str, ...] = (),
    stratum: str = "all",
) -> pl.DataFrame:
    """Null cells by sampling each row's null pattern from the real table's pattern frequencies.

    ``synthetic_null_patterns.json`` stores, per table and stratum, the share of rows with each
    pattern of null columns (measured on the real table). Patterns are projected onto the
    columns ``df`` has, identical projections merged. ``given`` names columns whose nulls are
    already set by propagation: each row draws from the patterns that agree with its current
    nulls on ``given`` (the nearest ones if none agree) and the other columns are nulled as drawn.
    Each column is then topped up to its real null share (``totals``), which the dropped rare
    patterns leave short.
    Only rows in ``stratum_mask`` (default all) are touched; nulls are only added.
    """
    spec = _null_patterns()[table][stratum]
    keep = [i for i, c in enumerate(spec["columns"]) if c in df.columns]
    cols = [spec["columns"][i] for i in keep]
    bits = np.array([[p[i] == "1" for i in keep] for _, p in spec["patterns"]], dtype=bool)
    bits, inverse = np.unique(bits, axis=0, return_inverse=True)
    share = np.bincount(inverse.ravel(), weights=[w for w, _ in spec["patterns"]])

    rows = np.arange(df.height) if stratum_mask is None else np.flatnonzero(stratum_mask)
    gi = [j for j, c in enumerate(cols) if c in given]
    code = 1 << np.arange(len(gi))
    pattern_code = bits[:, gi] @ code
    current = np.zeros((rows.size, len(gi)), dtype=bool)
    for k, j in enumerate(gi):
        current[:, k] = df[cols[j]].is_null().to_numpy()[rows]
    row_code = current @ code
    pick = np.empty(rows.size, dtype=np.int64)
    for c in np.unique(row_code):
        sel = np.flatnonzero(row_code == c)
        ok = np.flatnonzero(pattern_code == c)
        if ok.size == 0:
            # no pattern agrees: take the nearest ones (fewest differing given columns)
            distance = (bits[:, gi] != current[sel[0]]).sum(axis=1)
            ok = np.flatnonzero(distance == distance.min())
        cum = np.cumsum(share[ok])
        # stratified draws: every pattern gets its share of the group's rows up to rounding
        u = (nrng.permutation(sel.size) + nrng.random(sel.size)) / sel.size
        draw = np.searchsorted(cum, u * cum[-1], side="right")
        pick[sel] = ok[np.minimum(draw, ok.size - 1)]
    new = bits[pick]
    exprs = []
    for j, c in enumerate(cols):
        nul = df[c].is_null().to_numpy()[rows] | new[:, j]
        # patterns seen in few real rows are dropped from the file, which leaves the column
        # short of its real null share: top up from the rows not yet null
        total, cur = spec["totals"][c], nul.mean()
        if cur < total:
            nul |= ~nul & (nrng.random(rows.size) < (total - cur) / (1 - cur))
        if nul.any():
            is_null = np.zeros(df.height, dtype=bool)
            is_null[rows] = nul
            exprs.append(pl.when(pl.Series(is_null)).then(None).otherwise(pl.col(c)).alias(c))
    return df.with_columns(exprs)


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
    for ctry, n_firms in ROW_FIRMS.items():
        for _ in range(n_firms):
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
    # (mthprc = me / shrout) starts in a realistic $10-$100 band.
    first_me = panel.group_by("permno").agg(pl.col("me").first().alias("_me0")).sort("permno")
    target_price = rng.uniform(10.0, 100.0, size=first_me.height)
    shrout_per_firm = first_me.select(
        "permno",
        shrout=(pl.col("_me0") / pl.lit(target_price)).round(0),
    ).join(reg_us.select("permno", siccd=pl.col("sic")), on="permno")
    panel = panel.join(shrout_per_firm, on="permno")
    panel = panel.with_columns(
        mthprc=pl.col("me") / pl.col("shrout"),
        mthcap=pl.col("me"),
        mthcumfacshr=pl.lit(1.0),
        mthvol=pl.col("shrout") * 0.05,
    )
    out = panel.with_columns(
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
    return _round_to(
        out,
        {
            "mthret": 6,
            "mthretx": 6,
            "mthprc": 6,
            "mthcap": 2,
            "mthvol": 0,
            "mthcumfacshr": 12,
        },
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
    out = df.with_columns(
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
    return _round_to(out, {"dlyret": 6, "dlyretx": 6, "dlyprc": 6, "dlycap": 2, "dlyvol": 0})


def _gen_funda(reg: pl.DataFrame, rng: np.random.Generator, is_global: bool) -> pl.DataFrame:
    """Annual Compustat (NA or Global). One row per (gvkey, fiscal_year_end)."""
    sub = reg.filter(pl.col("excntry").is_in(NA_COUNTRIES) != is_global)

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
            # real acc_std_ann: BE <= 0 nulls be_x in ~12% of NA and ~4% of Global report rows
            if rng.random() < (0.03 if is_global else 0.14):
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
                # pre-2009 convention: mib is the total, mibt == mib, so mibn = mibt - mib = 0
                "mibt": mib,
                "mibn": 0.0,
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
                # Global table has a narrower column set
                for k in (
                    "adjex_c",
                    "ajex",
                    "csho",
                    "itcb",
                    "ni",
                    "prcc_f",
                    "pstkl",
                    "pstkrv",
                ):
                    rec.pop(k, None)
            rec.pop("datafmt_marker", None)
            rows.append(rec)
    df = pl.DataFrame(rows).with_columns(pl.col("datadate").cast(pl.Date))
    return _round_comp(df)


def gen_comp_funda(reg: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    return _gen_funda(reg, rng, is_global=False)


def gen_comp_g_funda(reg: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    return _gen_funda(reg, rng, is_global=True)


def gen_world_msf(
    reg: pl.DataFrame, rng: np.random.Generator, msf_us: pl.DataFrame, nrng: np.random.Generator
) -> pl.DataFrame:
    """World monthly panel. US rows mirror msf_us; ROW rows generated.

    Nulls of the CRSP-sourced US rows start from those of msf_us; both strata then draw the
    remaining nulls from the real null patterns of their source.
    """
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
            size_grp=pl.when(pl.col("mthcap").is_not_null()).then(pl.lit("mega")),
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
            "primaryexch",
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
        # starts in a realistic $10-$100 band.
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
    prev_eom = pl.col("eom").shift(1).over("id")
    # ret_lag_dif mirrors the production world_msf column (calendar months since
    # the prior observation; null on the first row) — ff_factor_rows nulls
    # gap-spanning returns off it.
    world = (
        pl.concat([us, row_df], how="diagonal_relaxed")
        .sort(["id", "eom"])
        .with_columns(
            ret_lag_dif=(
                (pl.col("eom").dt.year() - prev_eom.dt.year()) * 12
                + (pl.col("eom").dt.month() - prev_eom.dt.month())
            ).cast(pl.Int64),
            # one security per company in the synthetic panel
            me_company=pl.col("me"),
        )
    )
    us_rows = (world["source_crsp"] == 1).to_numpy()
    # US rows carry the crsp_msf_v2 nulls (me, ret, prc and what is derived from them); the
    # structural permno nulls of ex-US rows are part of the stratum-0 patterns.
    us_given = ("permno", "me", "me_company", "prc", "ret", "ret_exc", "size_grp", "ret_lag_dif")
    world = _apply_null_patterns(world, "world_msf", nrng, us_rows, us_given, "1")
    world = _apply_null_patterns(world, "world_msf", nrng, ~us_rows, ("permno", "ret_lag_dif"), "0")
    # permco joins after the null patterns so it leaves their draws unchanged; CRSP rows
    # carry one, as in the real table, and ex-US rows none
    permco = reg_us.join(reg.select("permno", "permco"), on="permno").select(
        pl.col("world_id").cast(pl.Int64).alias("id"), pl.col("permco").cast(pl.Int64)
    )
    return world.join(permco, on="id", how="left", maintain_order="left").with_columns(
        me_lag1=pl.col("me").shift(1).over("id")
    )


def gen_world_dsf(
    reg: pl.DataFrame, rng: np.random.Generator, dsf_us: pl.DataFrame, nrng: np.random.Generator
) -> pl.DataFrame:
    """World daily panel. US rows mirror dsf_us (as in the real table, whose CRSP-sourced rows
    are ~16%); ROW rows are generated. Builders that read ``world_dsf`` filter ``excntry``.
    """
    weekdays = np.array(_weekdays(DATE_START, DATE_END), dtype="datetime64[D]")
    reg_us = reg.filter(pl.col("excntry") == "USA").select("permno", "world_id")
    us = dsf_us.join(reg_us, on="permno").select(
        id=pl.col("world_id").cast(pl.Int64),
        date=pl.col("dlycaldt"),
        excntry=pl.lit("USA"),
        ret=pl.col("dlyret"),
        me=pl.col("dlycap"),
        prc=pl.col("dlyprc"),
        tvol=pl.col("dlyvol"),
    )

    reg_row = reg.filter(pl.col("excntry") != "USA")
    daily_rets_row = np.clip(
        rng.normal(0.0003, 0.02, size=(reg_row.height, weekdays.size)), -0.2, 0.2
    )
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("id", "excntry", "date", "ret", "me", "prc", "tvol")
    }
    for i, r in enumerate(reg_row.iter_rows(named=True)):
        delist = r["delisting_eom"] if r["delisting_eom"] is not None else DATE_END
        idx = np.flatnonzero(
            (weekdays >= np.datetime64(r["inception_eom"])) & (weekdays <= np.datetime64(delist))
        )
        if idx.size == 0:
            continue
        log_me0 = float(rng.normal(17.5, 1.5))
        # shrout = initial ME / target price → realistic $10-$100 start price
        # (matches gen_world_msf; avoids sub-$1 prices failing lag_prc > 5).
        shrout = float(np.exp(log_me0)) / float(rng.uniform(10.0, 100.0))
        me = np.exp(log_me0 + np.cumsum(np.log1p(daily_rets_row[i, idx])))
        parts["id"].append(np.full(idx.size, r["world_id"], dtype=np.int64))
        parts["excntry"].append(np.full(idx.size, r["excntry"]))
        parts["date"].append(weekdays[idx])
        parts["ret"].append(daily_rets_row[i, idx])
        parts["me"].append(me)
        parts["prc"].append(me / shrout)
        parts["tvol"].append(np.full(idx.size, shrout * 0.002))
    row_df = pl.DataFrame({k: np.concatenate(v) for k, v in parts.items()})

    # ret_lag_dif mirrors production world_dsf (days since the prior
    # observation per id; null on the first row).
    world = (
        pl.concat([us, row_df], how="diagonal_relaxed")
        .with_columns(
            eom=pl.col("date").dt.month_end(),
            common=pl.lit(1, dtype=pl.Int8),
            primary_sec=pl.lit(1, dtype=pl.Int8),
            obs_main=pl.lit(1, dtype=pl.Int8),
            exch_main=pl.lit(1, dtype=pl.Int8),
            prc_high=pl.col("prc") * 1.01,
            prc_low=pl.col("prc") * 0.99,
            bidask=pl.lit(0),
            adjfct=pl.lit(1.0),
        )
        .sort(["id", "date"])
        .with_columns(
            ret_lag_dif=(pl.col("date") - pl.col("date").shift(1).over("id")).dt.total_days()
        )
        .select(
            "id",
            "date",
            "eom",
            "excntry",
            "common",
            "primary_sec",
            "obs_main",
            "exch_main",
            "ret",
            "me",
            "prc",
            "prc_high",
            "prc_low",
            "tvol",
            "bidask",
            "adjfct",
            "ret_lag_dif",
        )
    )
    us_rows = (world["excntry"] == "USA").to_numpy()
    us_given = ("me", "prc", "ret", "tvol", "ret_lag_dif")
    world = _apply_null_patterns(world, "world_dsf", nrng, us_rows, us_given, "1")
    world = _apply_null_patterns(world, "world_dsf", nrng, ~us_rows, (), "0")
    # source_crsp has no nulls in the real table; added after the null draws so the
    # random stream (and every other synthetic value) is unchanged.
    return world.with_columns(source_crsp=pl.Series(us_rows.astype(np.int32)))


def _write(df: pl.DataFrame, out_dir: Path, table: str) -> None:
    """Drop columns the real table lacks, cast to the real dtypes, write ``<table>.parquet``."""
    df = df.drop(DROP_COLS.get(table, []))
    if table.startswith("comp_"):
        casts: dict[str, pl.DataType] = {
            c: pl.Decimal(24, 12) if c in COMP_WIDE_ITEMS else pl.Decimal(18, 4)
            for c, t in df.schema.items()
            if t == pl.Float64
        } | {c: t for c, t in COMP_INT_COLS.items() if c in df.columns}
    else:
        casts = REAL_DTYPES.get(table, {})
    df.with_columns(pl.col(c).cast(t) for c, t in casts.items()).write_parquet(
        out_dir / f"{table}.parquet"
    )


def gen_acc_std_ann(funda: pl.DataFrame, g_funda: pl.DataFrame) -> pl.DataFrame:
    """Synthesize jkp's standardized annual accounting panel (``acc_std_ann``).

    The FF loaders read book equity (be_x), operating profit (ope_x), minority
    interest (mib, mibt, mibn) from this panel via
    ``_jkp_annual_acc``; asset growth comes from ``gen_achars_world``. The real
    builder (``standardized_accounting_data``) pulls ~40 Compustat vars this synthetic
    fixture doesn't carry, so instead we reproduce the panel's *shape and the fields
    the loaders consume*:

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
        "mib",
        "mibt",
        "mibn",
    ]

    def _report_rows(df: pl.DataFrame, source: str) -> pl.DataFrame:
        missing = [c for c in raw_cols if c not in df.columns]
        df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
        pstk_x = pl.coalesce("pstkrv", "pstkl", "pstk")
        txditc_x = pl.coalesce("txditc", pl.col("txdb") + pl.col("itcb"))
        seq_x = pl.coalesce("seq", pl.col("ceq") + pstk_x, pl.col("at") - pl.col("lt"))
        txditc_kept = pl.when(pl.col("datadate").dt.year() < BE_DEFERRED_TAX_END_YEAR).then(
            pl.coalesce(txditc_x, pl.lit(0.0))
        )
        be_x = seq_x + txditc_kept.otherwise(0.0) - pl.coalesce(pstk_x, pl.lit(0.0))
        opex_x = pl.coalesce("xopr", pl.col("cogs") + pl.col("xsga"))
        ebitda_x = pl.coalesce("ebitda", "oibdp", pl.coalesce("sale", "revt") - opex_x)
        return df.select(
            "gvkey",
            "datadate",
            "curcd",
            source=pl.lit(source),
            be_x=pl.when(be_x > 0).then(be_x).otherwise(None),
            ope_x=ebitda_x - pl.coalesce(pl.col("xint"), pl.lit(0.0)),
            mib=pl.col("mib"),
            mibt=pl.col("mibt"),
            mibn=pl.col("mibn"),
            at_x=pl.col("at"),
        )

    reports = pl.concat(
        [_report_rows(funda, "NA"), _report_rows(g_funda, "GLOBAL")], how="vertical_relaxed"
    ).unique(["gvkey", "datadate"], keep="first")

    # Monthly grid per gvkey (first->last report), month-ends; report rows land on Dec.
    grid = (
        reports.group_by("gvkey")
        .agg(
            lo=pl.col("datadate").min(),
            hi=pl.col("datadate").max(),
            curcd=pl.col("curcd").drop_nulls().first(),
        )
        .with_columns(datadate=pl.date_ranges("lo", "hi", interval="1mo"))
        .explode("datadate")
        .select("gvkey", "curcd", "datadate")
    )
    return grid.join(reports.drop("curcd"), on=["gvkey", "datadate"], how="left").sort(
        ["gvkey", "curcd", "datadate"]
    )


def gen_achars_world(acc_std_ann: pl.DataFrame) -> pl.DataFrame:
    """Synthesize the slice of jkp's annual characteristics (``achars_world``) the FF
    loaders read: at_gr1 at each fiscal report, on its first public_date (4 months after
    the fiscal year-end). at_gr1 follows create_acc_chars: at_x over at_x 12 grid rows
    earlier, minus 1, when the (gvkey, curcd) panel has more than 12 rows and the lagged
    at_x is positive.
    """
    lag = pl.col("at_x").shift(12).over(["gvkey", "curcd"])
    return (
        acc_std_ann.sort(["gvkey", "curcd", "datadate"])
        .with_columns(n=pl.col("gvkey").cum_count().over(["gvkey", "curcd"]))
        .filter(pl.col("source").is_not_null())
        .select(
            "source",
            "gvkey",
            "datadate",
            public_date=pl.col("datadate").dt.offset_by("4mo").dt.month_end(),
            at_gr1=pl.when((pl.col("n") > 12) & (lag > 0)).then(pl.col("at_x") / lag - 1),
        )
    )


def write_synthetic_wrds(out_dir: Path, seed: int = SEED) -> None:
    """Generate the synthetic WRDS-shaped parquets into ``out_dir``.

    Base values come from ``default_rng(seed)``; every null is drawn from a separate
    ``default_rng(seed + 1)`` so base values do not depend on the null targets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nrng = np.random.default_rng(seed + 1)
    reg = build_firm_registry(rng)

    msf = _apply_null_patterns(gen_crsp_msf_v2(reg, rng), "crsp_msf_v2", nrng)
    _write(msf, out_dir, "crsp_msf_v2")
    dsf = _apply_null_patterns(gen_crsp_dsf_v2(reg, rng, msf), "crsp_dsf_v2", nrng)
    _write(dsf, out_dir, "crsp_dsf_v2")

    funda = _apply_null_patterns(gen_comp_funda(reg, rng), "comp_funda", nrng)
    _write(funda, out_dir, "comp_funda")
    g_funda = _apply_null_patterns(gen_comp_g_funda(reg, rng), "comp_g_funda", nrng)
    _write(g_funda, out_dir, "comp_g_funda")

    world_msf = gen_world_msf(reg, rng, msf, nrng)
    _write(world_msf, out_dir, "world_msf")
    world_dsf = gen_world_dsf(reg, rng, dsf, nrng)
    _write(world_dsf, out_dir, "world_dsf")
    acc_std_ann = gen_acc_std_ann(funda, g_funda)
    _write(acc_std_ann, out_dir, "acc_std_ann")
    _write(gen_achars_world(acc_std_ann), out_dir, "achars_world")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", default=SEED, type=int)
    args = ap.parse_args()
    write_synthetic_wrds(args.out, args.seed)
    total = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
    print(f"Wrote {len(list(args.out.iterdir()))} files, {total / 1e6:.1f} MB to {args.out}")


if __name__ == "__main__":
    main()
