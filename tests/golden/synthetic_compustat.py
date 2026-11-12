"""
Generate the synthetic Compustat inputs of the accounting-stage tests (issue #303, step 2).

Every value is sampled from the aggregate statistics in compustat_profile.json (written on Yale by
calibration/profile_compustat.py); no WRDS value is copied. gvkeys start at 900001, above the largest
real gvkey in the profile.

Usage: PYTHONPATH=src uv run --no-sync python -m tests.golden.synthetic_compustat OUT_DIR

Deterministic (seed 42). Nothing is committed: the tests generate the inputs once per session
(synthetic_inputs in tests/conftest.py). Writes OUT_DIR in the DataPaths layout:
raw/raw_tables/ holds comp_funda, comp_fundq, comp_g_funda, comp_g_fundq and comp_exrt_dly with the
real column sets and dtypes, and interim/world_msf.parquet holds the market equity the stage joins.
Columns the accounting stage does not read are null.

How values are drawn. Each firm has quarterly latent scores (log size, sales, profitability,
leverage, cash, tangibility): a firm effect plus an AR(1) path, with the profile's correlations and
its autocorrelations one and four quarters apart. Each line item divided by total assets is mapped
from a score through the profile's quantiles and its negative and zero shares. Earnings (ib) come
from the profitability score and the income statement is built up from it (pi = (ib + mii) /
(1 - tax rate), oiadp = pi + xint - nopi - spi, oibdp = oiadp + dp, xsga = gp - oibdp); the balance
sheet follows from identities (seq = at - lt - mib, ...). Quarterly flows come first; the
year-to-date (*y) values cumulate them within the fiscal year and the annual flows are the
fourth-quarter year-to-date values. Values are in USD and converted to the reporting currency at
the rate the stage itself will use.

Random firms (900001 up) follow the profile: source, currency, report format, fiscal year-end and
its changes, life, semiannual and annual-only reporting, quarterly-only months, listing, and the
per-column null rates (empty leading records, plus items the firm never reports: one draw per firm
and item, shared by its annual and quarterly columns).

Persona firms (900101 up), one case each:
    900101  restated duplicate: a second row for one fiscal quarter, dated a month later (bug 0)
    900102  rows without fqtr left on the old calendar after a fiscal year-end change (bug 0)
    900103  a skipped fiscal quarter, so three four-quarter sums are missing (bug 1)
    900104  quarterly reports two years before the first annual report (bug 2)
    900105  DEM reports after the DEM rates end in 2001 (bug 3)
    900106  one fiscal year reported without a currency code (bug 3)
    900107  CAD reports from 1976, before Compustat's first CAD rate (bug 3)
    900108  fiscal year-end change from 12 to 6 with two dates on both calendars (bug 5)
    900109  a fiscal calendar that resumes after a gap over 200 days, next to a newer one (bug 5)
    900110  FS-only firm with dates on two calendars (bug 6)
    900111  INDL and FS records on the same dates and calendar (bug 6)
    900112  in both North America and Global for 2000-2010; the Global rows win
    900113  JPY firm with a March fiscal year-end and a second, non-primary security
    900114  PLN reports before the first PLN rate, across a six-month gap in the rates
    900115  annual reports only
    900116  quarterly reports three years past the last annual report
    900117  zero sales, negative book equity, zero and negative total assets
    900118  three years of history
    900119  six years of history
    900120  twelve years of history
    900121  constant earnings and assets, so ni_ar1 is null
    900122  no reports for three years in the middle of its life
    900123  NA rows the read filters drop (SUMM_STD, FS, consol P, popsrc I)
    900124  Global rows the read filters drop (STD, popsrc D, consol P)
    900125  NA on a December year-end; Global (quarterly only) on a June year-end for 2002-2008,
            on the same dates; the older NA calendar is kept
    900126  Global FS on a December year-end; INDL (quarterly only) on a June year-end for
            2002-2008, on the same dates; the older FS calendar is kept
    900127  fiscal year-end moves from December to November after the first report, so its
            reports fall 12, 24 and 36 months after it
"""

import json
import math
import re
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from jkp.data.paths import DataPaths, get_fx_cad_backfill_path
from tests.golden.calibration.profile_compustat import LIFE_QUANTILES, QUANTILES
from tests.golden.compustat_invariants import TABLES

SEED = 42
PROFILE_PATH = Path(__file__).parent / "compustat_profile.json"

FIRST_FY, LAST_FY = 1990, 2015
N_RANDOM = {"NA": 26, "GLOBAL": 34}
RANDOM_GVKEY, PERSONA_GVKEY = 900001, 900101
TABLE = {(source, freq): table for table, (source, _, freq) in TABLES.items()}
READ_FILTERS = {
    "NA": {"datafmt": "STD", "popsrc": "D", "consol": "C"},
    "GLOBAL": {"datafmt": "HIST_STD", "popsrc": "I", "consol": "C"},
}

# Weekly quotes in USD per unit; GBP is the base of comp_exrt_dly. DEM ends and PLN starts late,
# with a gap, for personas 900105 and 900114.
FX_FIRST, FX_LAST = date(1982, 2, 5), date(2017, 12, 29)
FX_START_LEVEL = {
    "USD": 1.0,
    "GBP": 1.6,
    "CAD": 0.8,
    "EUR": 1.15,
    "JPY": 0.009,
    "AUD": 0.75,
    "DEM": 0.55,
    "PLN": 0.3,
}
FX_WEEKLY_VOL = 0.012
FX_LAST_QUOTE = {"DEM": date(2001, 12, 28)}
FX_FIRST_QUOTE = {"PLN": date(1995, 1, 6)}
FX_QUOTE_GAP = {"PLN": (date(2000, 1, 1), date(2000, 6, 30))}
GLOBAL_CURRENCIES = ["EUR", "JPY", "GBP", "AUD", "USD"]

# The line items standardized_accounting_data reads (avars there), plus dpact for Global ppegtq.
INCOME = [
    "sale", "revt", "gp", "ebitda", "oibdp", "ebit", "oiadp", "pi", "ib", "ni", "mii", "cogs",
    "xsga", "xopr", "xrd", "xad", "xlr", "dp", "xi", "do", "xido", "xint", "spi", "nopi", "txt",
    "dvt",
]  # fmt: skip
CASH_FLOW = [
    "oancf", "ibc", "dpc", "xidoc", "capx", "wcapt", "fincf", "fiao", "txbcof", "ltdch", "dltis",
    "dltr", "dlcch", "purtshr", "prstkc", "sstk", "dv", "dvc",
]  # fmt: skip
BALANCE = [
    "at", "act", "aco", "che", "invt", "rect", "ivao", "ivst", "ppent", "ppegt", "intan", "ao",
    "gdwl", "re", "lt", "lct", "dltt", "dlc", "txditc", "txdb", "itcb", "txp", "ap", "lco", "lo",
    "seq", "ceq", "pstkrv", "pstkl", "pstk", "mib", "icapt",
]  # fmt: skip
FLOWS = INCOME + CASH_FLOW
STOCKS = [*BALANCE, "dpact"]

# Latent scores and the profile variables that calibrate them; the score drives the listed item.
CORE = {
    "size": "log_atq",
    "sale": "saleq/atq",
    "prof": "ibq/atq",
    "lt": "ltq/atq",
    "che": "cheq/atq",
    "ppent": "ppentq/atq",
}
CORE_ITEM = {"sale": "sale", "prof": "ib", "lt": "lt", "che": "che", "ppent": "ppent"}
# Items drawn from their own quantiles, in draw order; LOADINGS ties some to a latent score.
DRAWN = [
    "gp", "dp", "xint", "nopi", "spi", "mii", "xi", "do", "xrd", "xad", "xlr", "dvt", "capx",
    "fincf", "fiao", "txbcof", "dltis", "dltr", "dlcch", "prstkc", "sstk", "wcapt", "ltdch",
    "purtshr", "rect", "invt", "aco", "ivao", "intan", "mib", "pstk", "dlc", "ap", "txp", "lco",
    "dltt", "txditc", "re", "emp",
]  # fmt: skip
LOADINGS = {
    "gp": ("sale", 0.6),
    "dp": ("ppent", 0.6),
    "xint": ("lt", 0.5),
    "spi": ("prof", 0.3),
    "xrd": ("prof", -0.4),
    "dvt": ("prof", 0.5),
    "capx": ("ppent", 0.6),
    "sstk": ("prof", -0.4),
    "rect": ("sale", 0.4),
    "invt": ("sale", 0.3),
    "ap": ("sale", 0.4),
    "dltt": ("lt", 0.6),
    "txditc": ("ppent", 0.3),
    "re": ("prof", 0.6),
    "emp": ("sale", 0.5),
}
# Each score is sqrt(w) x firm effect + sqrt(1 - w) x AR(1). For the latent scores, w and the AR(1)
# coefficient match the profile's autocorrelations one and four quarters apart; for the drawn items,
# which the profile has no autocorrelation for, they are set here.
ITEM_FIRM_SHARE, ITEM_PHI = 0.8, 0.9
SELECTED_NULL_SHARE = 0.9
SIZE_LOG_MEDIAN, SIZE_SD = math.log(150.0), 2.0  # USD millions; median assets in the outputs is 119
SIZE_DRIFT = 0.0125  # log growth per quarter; median at_gr1 in the outputs is 0.05
TAX_RATE, LOSS_TAX_RATE, TAXPAYER_SHARE = 0.3, 0.1, 0.8
ACCRUAL_VOL = 0.03
ASSET_FILL = 0.98  # current assets, PP&E, intangibles and investments fill at most this share of at

# Market equity: log(me / at) has median log(0.68) (at_me median 1.47 in the outputs).
ME_LOG_MEDIAN, ME_FIRM_SD, ME_PROF_LOADING, ME_PHI, ME_VOL = math.log(0.68), 0.8, 0.3, 0.97, 0.08
# Random firms only; with the personas all listed, about a third of reports have no ME (as real).
LISTED_SHARE, LATE_LISTING_SHARE, SECOND_SECURITY_SHARE = 0.5, 0.2, 0.1
# Other random firms start quarterly reports in their second fiscal year, after the first annual.
QUARTERLY_LEAD_SHARE = 0.1  # firms whose quarterly reports start 4-8 quarters before the annual

Spec = dict
Items = dict[str, np.ndarray]


def month_end(year: int, month: int) -> date:
    """Last day of a month; months past 12 or below 1 roll into the next or previous years."""
    year, month = year + (month - 1) // 12, (month - 1) % 12 + 1
    return date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)


def shift_months(d: date, months: int) -> date:
    """Month-end `months` after the month of d."""
    return month_end(d.year, d.month + months)


def fiscal_quarter_end(fy: int, fyr: int, fqtr: int) -> date:
    """Date of a fiscal quarter, with Compustat's rule that years ending January-May belong to the
    previous fiscal year."""
    return month_end(fy + (fyr < 6), fyr - 3 * (4 - fqtr))


def fiscal_labels(fyr: int) -> list[pl.Expr]:
    """fyearq and fqtr of each datadate on the fiscal calendar ending in month fyr."""
    month = pl.col("datadate").dt.month().cast(pl.Int32)
    year_end = pl.col("datadate").dt.year().cast(pl.Int32) + (month > fyr).cast(pl.Int32)
    return [
        (year_end - int(fyr < 6)).alias("fyearq"),
        (4 - ((fyr - month + 12) % 12) // 3).alias("fqtr"),
    ]


def quarter_ends(first: date, last: date) -> list[date]:
    """Every third month-end from first up to last."""
    out, k = [], 0
    while (d := shift_months(first, 3 * k)) <= last:
        out.append(d)
        k += 1
    return out


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def ar1(rng: np.random.Generator, n: int, phi: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """n draws of a stationary AR(1) vector with unit variances and correlation chol @ chol.T."""
    shocks = rng.standard_normal((n, len(phi))) @ chol.T
    z = np.empty_like(shocks)
    z[0] = shocks[0]
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + scale * shocks[t]
    return z


def ratio_knots(
    profile: dict, item: str, source: str, flow: bool
) -> tuple[list[float], list[float]]:
    """
    Description:
        Quantile function of quarterly item / total assets for one source.
    Steps:
        1) Take the source's more complete column, quarterly (itemq) or annual; annual flows are
           divided by 4. If it is more than SELECTED_NULL_SHARE null, it describes a selected
           sample (Compustat Global folds xi into xido, for example), so take the other
           source's column when that is more complete.
        2) Add knots at the column's negative and zero shares, so that draws are negative, zero and
           positive as often as in the profile; interpolating between quantiles alone loses the
           mass at zero.
    Output:
        (levels, values) for np.interp.
    """

    def most_complete(s: str) -> tuple[dict, float] | None:
        found = [
            (stats, 4.0 if flow and freq == "annual" else 1.0)
            for freq, column in (("quarterly", f"{item}q"), ("annual", item))
            if (stats := profile["tables"][TABLE[(s, freq)]]["columns"].get(column, {})).get("q")
        ]
        return min(found, key=lambda f: f[0]["null"], default=None)

    own = most_complete(source)
    other = most_complete("GLOBAL" if source == "NA" else "NA")
    if own is None or (
        own[0]["null"] > SELECTED_NULL_SHARE and other and other[0]["null"] < own[0]["null"]
    ):
        own = other
    if own is None:
        raise KeyError(item)
    stats, scale = own
    low, high = stats["neg"], stats["neg"] + stats["zero"]
    knots = [
        (lv, min(q, 0.0) / scale) for lv, q in zip(QUANTILES, stats["q"], strict=True) if lv < low
    ]
    knots += [(low, 0.0)] + ([(high, 0.0)] if high > low else [])
    knots += [
        (lv, max(q, 0.0) / scale) for lv, q in zip(QUANTILES, stats["q"], strict=True) if lv > high
    ]
    levels, values = zip(*knots, strict=True)
    return list(levels), list(values)


def firm_share_and_phi(r1: float, r4: float) -> tuple[float, float]:
    """
    Description:
        Firm-effect share w and AR(1) coefficient phi of z = sqrt(w) f + sqrt(1 - w) a whose
        autocorrelations one and four quarters apart are r1 and r4.
    Steps:
        1) For w on a grid below min(r1, r4), phi = (r1 - w) / (1 - w) matches r1.
        2) Keep the w whose w + (1 - w) phi^4 is closest to r4.
    Output:
        (w, phi).
    """
    grid = np.linspace(0.0, min(r1, r4) - 1e-6, 2001)
    phi = (r1 - grid) / (1.0 - grid)
    best = int(np.argmin(np.abs(grid + (1.0 - grid) * phi**4 - r4)))
    return float(grid[best]), float(phi[best])


def latent_scores(
    rng: np.random.Generator, n: int, w: np.ndarray, phi: np.ndarray, chol: np.ndarray
) -> np.ndarray:
    """n draws of sqrt(w) x firm effect + sqrt(1 - w) x AR(1), unit variances, correlation chol."""
    effect = chol @ rng.standard_normal(len(phi))
    return np.sqrt(w) * effect + np.sqrt(1.0 - w) * ar1(rng, n, phi, chol)


def source_context(profile: dict, source: str) -> dict:
    """Latent correlations and persistence, and every item's quantiles, for one source."""
    stats = profile["tables"][TABLE[(source, "quarterly")]]["correlations"]
    idx = [stats["vars"].index(v) for v in CORE.values()]
    pearson = 2.0 * np.sin(np.pi * np.asarray(stats["spearman"])[np.ix_(idx, idx)] / 6.0)
    w, v = np.linalg.eigh(pearson)
    pearson = v @ np.diag(np.clip(w, 1e-3, None)) @ v.T
    d = np.sqrt(np.diag(pearson))
    lags = stats["autocorr"]
    w, phi = zip(
        *(firm_share_and_phi(lags["report"][v], lags["year"][v]) for v in CORE.values()),
        strict=True,
    )
    items = [*CORE_ITEM.values(), *DRAWN]
    return {
        "chol": np.linalg.cholesky(pearson / np.outer(d, d)),
        "w": np.array(w),
        "phi": np.array(phi),
        "q": {i: ratio_knots(profile, i, source, i in FLOWS) for i in items},
    }


def firm_items(rng: np.random.Generator, n: int, ctx: dict) -> tuple[Items, Items]:
    """
    Description:
        n quarters of line items in USD for one firm.
    Steps:
        1) Draw the latent scores and total assets.
        2) Map the score of each drawn item to item / at through its quantiles.
        3) Derive the aggregates from identities; scale assets and liabilities to fit at and lt.
    Output:
        (items, latent scores); flows are per quarter, stocks at the quarter end.
    """
    scores = latent_scores(rng, n, ctx["w"], ctx["phi"], ctx["chol"])
    z = dict(zip(CORE, scores.T, strict=True))
    at = np.exp(SIZE_LOG_MEDIAN + SIZE_SD * z["size"] + SIZE_DRIFT * np.arange(n))

    def ratio(item: str, score: np.ndarray) -> np.ndarray:
        return at * np.interp(normal_cdf(score), *ctx["q"][item])

    v = {item: ratio(item, z[k]) for k, item in CORE_ITEM.items()}
    for item in DRAWN:
        own = latent_scores(rng, n, np.array([ITEM_FIRM_SHARE]), np.array([ITEM_PHI]), np.eye(1))
        own = own[:, 0]
        parent, b = LOADINGS.get(item, (None, 0.0))
        v[item] = ratio(item, own if parent is None else b * z[parent] + math.sqrt(1 - b * b) * own)
    taxpayer = float(rng.random() < TAXPAYER_SHARE)
    ivst_share = rng.uniform(0.0, 0.6) * (rng.random() < 0.5)
    depreciated = rng.uniform(0.3, 1.2)
    goodwill_share = rng.uniform(0.2, 0.9) * (rng.random() < 0.5)
    itcb_share = 0.1 * (rng.random() < 0.05)
    accruals = at * ACCRUAL_VOL * ar1(rng, n, np.array([0.5]), np.eye(1))[:, 0]

    sale, ib = v["sale"], v["ib"]
    rate = np.where(ib + v["mii"] > 0, TAX_RATE, LOSS_TAX_RATE) * taxpayer
    pi = (ib + v["mii"]) / (1.0 - rate)
    txt = rate * pi
    oiadp = pi + v["xint"] - v["nopi"] - v["spi"]
    oibdp = oiadp + v["dp"]
    gp = np.maximum(v["gp"], oibdp)
    xsga = gp - oibdp
    cogs = np.maximum(sale - gp, 0.0)
    xido = v["xi"] + v["do"]
    v |= {
        "at": at, "revt": sale, "xsga": xsga, "gp": gp, "cogs": cogs, "xopr": cogs + xsga,
        "oibdp": oibdp, "ebitda": oibdp, "oiadp": oiadp, "ebit": oiadp, "pi": pi, "txt": txt,
        "xido": xido, "ni": ib + xido, "dvc": v["dvt"], "dv": v["dvt"], "ibc": ib, "dpc": v["dp"],
        "xidoc": xido, "oancf": ib + v["dp"] + accruals,
    }  # fmt: skip

    assets = ["che", "rect", "invt", "aco", "ppent", "intan", "ivao"]
    fit = np.minimum(1.0, ASSET_FILL * at / np.maximum(sum(v[a] for a in assets), 1e-12))
    for a in assets:
        v[a] = v[a] * fit
    liabilities = ["dlc", "ap", "txp", "lco", "dltt", "txditc"]
    fit = np.minimum(1.0, ASSET_FILL * v["lt"] / np.maximum(sum(v[x] for x in liabilities), 1e-12))
    for x in liabilities:
        v[x] = v[x] * fit
    v["act"] = v["che"] + v["rect"] + v["invt"] + v["aco"]
    v["ivst"] = v["che"] * ivst_share
    v["ppegt"] = v["ppent"] * (1.0 + depreciated)
    v["dpact"] = v["ppegt"] - v["ppent"]
    v["gdwl"] = v["intan"] * goodwill_share
    v["ao"] = at - v["act"] - v["ppent"] - v["intan"] - v["ivao"]
    v["lct"] = v["dlc"] + v["ap"] + v["txp"] + v["lco"]
    v["txdb"] = v["txditc"] * (1.0 - itcb_share)
    v["itcb"] = v["txditc"] * itcb_share
    v["lo"] = v["lt"] - v["lct"] - v["dltt"] - v["txditc"]
    v["seq"] = at - v["lt"] - v["mib"]
    v["ceq"] = v["seq"] - v["pstk"]
    v["pstkrv"] = v["pstkl"] = v["pstk"]
    v["icapt"] = v["dltt"] + v["seq"] + v["mib"]
    return v, z


def quarterly_rows(rows: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Quarterly table columns; semiannual reporters keep fiscal quarters 2 and 4, without flows."""
    q = rows.select(
        "datadate",
        "fyr",
        "fyearq",
        "fqtr",
        *[pl.col(f).alias(f"{f}q") for f in FLOWS],
        *[pl.col(f"{f}_ytd").alias(f"{f}y") for f in FLOWS],
        *[pl.col(s).alias(f"{s}q") for s in STOCKS],
    )
    if mode == "semiannual":
        q = q.filter(pl.col("fqtr").is_in([2, 4])).with_columns(
            pl.lit(None, pl.Float64).alias(f"{f}q") for f in FLOWS
        )
    return q


def annual_rows(rows: pl.DataFrame) -> pl.DataFrame:
    """Annual table columns at each fourth fiscal quarter: flows are the year-to-date values."""
    return rows.filter(pl.col("fqtr") == 4).select(
        "datadate",
        "fyr",
        pl.col("fyearq").alias("fyear"),
        *[pl.col(f"{f}_ytd").alias(f) for f in FLOWS],
        *STOCKS,
        "emp",
    )


def null_rates(profile: dict, table: str, indfmt: str) -> dict[str, float]:
    """Null share per column of a table; Global tables by report format."""
    by_format = profile["tables"][table].get("null_by_indfmt")
    if by_format:
        return by_format[indfmt]
    return {c: s["null"] for c, s in profile["tables"][table]["columns"].items()}


def mask_random_firm(
    df: pl.DataFrame,
    table: str,
    indfmt: str,
    rng: np.random.Generator,
    profile: dict,
    draws: dict[str, float],
) -> pl.DataFrame:
    """
    Description:
        Give a random firm the table's null rates.
    Steps:
        1) Empty the firm's first k records (all line items null), k ~ Binomial(rows, null share
           of total assets).
        2) Null each other item for the whole firm with its null share beyond that of total assets.
           The firm has one uniform draw per item, shared by its annual and quarterly (q and y)
           columns, so a firm that lacks an item lacks it in every table, as in Compustat.
    Output:
        The firm's rows with nulls.
    """
    rates = null_rates(profile, table, indfmt)
    at_col = "at" if TABLES[table][2] == "annual" else "atq"
    empty = rates.get(at_col, 0.0)
    items = [c for c, t in df.schema.items() if t == pl.Float64]
    missing = [
        c
        for c in items
        if c != at_col
        and draws[c if TABLES[table][2] == "annual" else c[:-1]]
        < max(0.0, (rates.get(c, 0.0) - empty) / (1.0 - empty))
    ]
    leading = pl.int_range(pl.len()) < int(rng.binomial(df.height, empty))
    return (
        df.sort("datadate")
        .with_columns(pl.lit(None, pl.Float64).alias(c) for c in missing)
        .with_columns(pl.when(~leading).then(pl.col(c)).alias(c) for c in items)
    )


def firm_frames(
    spec: Spec, rng: np.random.Generator, ctx: dict, profile: dict
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Description:
        One firm's annual and quarterly rows in USD, and its market equity.
    Steps:
        1) Draw items for every quarter from a year before the first report to the last.
        2) Apply the persona's item edits.
        3) Per fiscal-calendar segment, label the quarters, cumulate year-to-date flows and keep
           the segment's dates; the annual rows are its fourth quarters.
        4) Add identifiers; apply the persona's row edits, or the null rates for a random firm.
        5) Build monthly market equity over the listing period.
    Output:
        (annual, quarterly, world_msf rows).
    """
    segments = spec["segments"]
    first = min(s["first"] for s in segments)
    last = max(s["last"] for s in segments)
    dates = quarter_ends(shift_months(first, -12), last)
    items, scores = firm_items(rng, len(dates), ctx[spec["source"]])
    at_usd = items["at"].copy()
    if "items" in spec:
        spec["items"](items, np.array(dates, dtype="datetime64[D]"))
    usd = pl.DataFrame({"datadate": dates, **items})

    annual, quarterly = [], []
    for seg in segments:
        rows = (
            usd.with_columns(fiscal_labels(seg["fyr"]))
            .with_columns(pl.col(FLOWS).cum_sum().over("fyearq").name.suffix("_ytd"))
            .filter(pl.col("datadate").is_between(seg["first"], seg["last"]))
            .with_columns(fyr=pl.lit(seg["fyr"], pl.Int32))
        )
        if seg.get("quarterly", True) and spec.get("quarterly", "full") != "none":
            quarterly.append(quarterly_rows(rows, spec.get("quarterly", "full")))
        annual.append(annual_rows(rows if seg.get("annual", True) else rows.clear()))
    ids = {"gvkey": spec["gvkey"], "indfmt": spec["indfmt"], **READ_FILTERS[spec["source"]]}
    annual_df = (
        pl.concat(annual)
        .filter(
            pl.col("datadate").is_between(
                spec.get("annual_first", date.min), spec.get("annual_last", date.max)
            )
        )
        .with_columns(**{k: pl.lit(v) for k, v in ids.items()}, currency=pl.lit(spec["currency"]))
        .with_columns(curcd=pl.col("currency"))
    )
    quarterly_df = (
        pl.concat(quarterly) if quarterly else annual_df.head(0).select("datadate")
    ).with_columns(**{k: pl.lit(v) for k, v in ids.items()}, currency=pl.lit(spec["currency"]))
    if quarterly:
        quarterly_df = quarterly_df.filter(
            pl.col("datadate") >= spec.get("quarterly_first", date.min)
        ).with_columns(curcdq=pl.col("currency"))
    if "rows" in spec:
        annual_df, quarterly_df = spec["rows"](annual_df, quarterly_df)
    if spec.get("random"):
        source, indfmt = spec["source"], spec["indfmt"]
        draws = {item: rng.random() for item in [*FLOWS, *STOCKS, "emp"]}
        annual_df = mask_random_firm(
            annual_df, TABLE[(source, "annual")], indfmt, rng, profile, draws
        )
        if quarterly:
            quarterly_df = mask_random_firm(
                quarterly_df, TABLE[(source, "quarterly")], indfmt, rng, profile, draws
            )
    return annual_df, quarterly_df, market_equity(spec, rng, dates, at_usd, scores["prof"])


def market_equity(
    spec: Spec, rng: np.random.Generator, dates: list[date], at_usd: np.ndarray, prof: np.ndarray
) -> pl.DataFrame:
    """Monthly world_msf rows over the firm's listing period: me = at x exp(firm + profit + AR(1))."""
    if spec.get("listed") is None:
        return pl.DataFrame()
    eoms = [shift_months(spec["listed"][0], k) for k in range(_months(*spec["listed"]) + 1)]
    latest = np.searchsorted(
        np.array(dates, dtype="datetime64[D]"), np.array(eoms, dtype="datetime64[D]"), side="right"
    )
    latest = np.clip(latest - 1, 0, None)
    noise = ME_VOL / math.sqrt(1 - ME_PHI**2) * ar1(rng, len(eoms), np.array([ME_PHI]), np.eye(1))
    log_me_at = ME_LOG_MEDIAN + ME_FIRM_SD * rng.standard_normal() + ME_PROF_LOADING * prof[latest]
    me = at_usd[latest] * np.exp(log_me_at + noise[:, 0])
    rows = pl.DataFrame({"gvkey": spec["gvkey"], "eom": eoms, "me_company": me, "primary_sec": 1})
    if spec.get("second_security"):
        rows = pl.concat(
            [rows, rows.with_columns(me_company=pl.col("me_company") * 0.3, primary_sec=0)]
        )
    return rows.with_columns(common=1, obs_main=1)


def _months(first: date, last: date) -> int:
    return (last.year - first.year) * 12 + last.month - first.month


def draw_from(rng: np.random.Generator, shares: dict[str, float], keep: list[str]) -> str:
    """One key of shares, restricted to keep and renormalized."""
    p = np.array([shares[k] for k in keep])
    return keep[int(rng.choice(len(keep), p=p / p.sum()))]


def random_specs(rng: np.random.Generator, profile: dict) -> list[Spec]:
    """
    Description:
        Random firms drawn from the profile's panel shape.
    Steps:
        1) Currency, report format and fiscal year-end from the source's annual shares.
        2) Life from the life quantiles, placed uniformly in FIRST_FY-LAST_FY.
        3) Fiscal year-end changes, annual-only and semiannual reporting, quarterly-only months
           and listing at their rates.
    Output:
        List of firm specs.
    """
    invariants = {(r["rule"], r["table"]): r["rate"] for r in profile["invariants"]}
    specs = []
    sources = ["NA"] * N_RANDOM["NA"] + ["GLOBAL"] * N_RANDOM["GLOBAL"]
    for i, source in enumerate(sources):
        panel = profile["tables"][TABLE[(source, "annual")]]["panel"]
        qpanel = profile["tables"][TABLE[(source, "quarterly")]]["panel"]
        currency = draw_from(
            rng, panel["currency"], ["USD", "CAD"] if source == "NA" else GLOBAL_CURRENCIES
        )
        indfmt = draw_from(rng, panel["indfmt"], list(panel["indfmt"]))
        fyr = int(draw_from(rng, panel["fyr"], [k for k in panel["fyr"] if k != "null"]))
        life = int(np.ceil(np.interp(rng.random(), LIFE_QUANTILES, panel["life_years_q"])))
        life = min(max(life, 1), LAST_FY - FIRST_FY + 1)
        first_fy = int(rng.integers(FIRST_FY, LAST_FY - life + 2))
        last_fy = first_fy + life - 1
        first, last = fiscal_quarter_end(first_fy, fyr, 1), fiscal_quarter_end(last_fy, fyr, 4)
        segments = [{"fyr": fyr, "first": first, "last": last}]
        change = invariants[("fyr_change_firms", TABLE[(source, "annual")])]
        if life >= 4 and rng.random() < change:
            cut = fiscal_quarter_end(int(rng.integers(first_fy + 1, last_fy - 1)), fyr, 4)
            new_fyr = (fyr + 3 * int(rng.integers(1, 4)) - 1) % 12 + 1
            segments = [
                {"fyr": fyr, "first": first, "last": cut},
                {"fyr": new_fyr, "first": shift_months(cut, 3), "last": last},
            ]
        mode = "full"
        if rng.random() < invariants[("annual_only_firms", "annual")]:
            mode = "none"
        elif source == "GLOBAL" and rng.random() < 2 * qpanel["gap_months"]["6"]:
            mode = "semiannual"
        spec = {
            "gvkey": str(RANDOM_GVKEY + i),
            "source": source,
            "currency": currency,
            "indfmt": indfmt,
            "segments": segments,
            "quarterly": mode,
            "random": True,
        }
        if mode != "none" and rng.random() < QUARTERLY_LEAD_SHARE:
            spec["annual_first"] = shift_months(first, 3 * int(rng.integers(4, 9)))
        elif life > 1:
            spec["quarterly_first"] = shift_months(first, 12)
        if rng.random() < LISTED_SHARE:
            start = (
                shift_months(first, int(rng.integers(12, 37)))
                if rng.random() < LATE_LISTING_SHARE
                else shift_months(first, -int(rng.integers(0, 25)))
            )
            spec["listed"] = (start, shift_months(last, int(rng.integers(3, 19))))
            spec["second_security"] = rng.random() < SECOND_SECURITY_SHARE
        specs.append(spec)
    return specs


def _on(dates: np.ndarray, first: date, last: date) -> np.ndarray:
    return (dates >= np.datetime64(first)) & (dates <= np.datetime64(last))


def _copy_rows(frame: pl.DataFrame, when: pl.Expr, scale: float, **changes: object) -> pl.DataFrame:
    """frame plus copies of the rows matching when, values scaled and columns changed."""
    copies = frame.filter(when).with_columns(cs.float() * scale)
    return pl.concat([frame, copies.with_columns(**{k: pl.lit(v) for k, v in changes.items()})])


def _filtered_copies(changes: list[dict[str, str]]) -> Callable:
    """Row edit adding, per change, a copy of every row (values x10) that the read filters drop."""

    def edit(a: pl.DataFrame, q: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        return tuple(
            pl.concat(
                [
                    f,
                    *[
                        f.with_columns(cs.float() * 10.0, **{k: pl.lit(v) for k, v in c.items()})
                        for c in changes
                    ],
                ]
            )
            for f in (a, q)
        )

    return edit


def _zero_sales_negative_equity(v: Items, d: np.ndarray) -> None:
    sales = _on(d, date(1998, 1, 1), date(2001, 12, 31))
    for item in ["sale", "revt", "cogs", "gp"]:
        v[item][sales] = 0.0
    deficit = _on(d, date(2003, 1, 1), date(2005, 12, 31))
    v["lt"][deficit] = 1.3 * v["at"][deficit]
    v["lo"] = v["lt"] - v["lct"] - v["dltt"] - v["txditc"]
    v["seq"] = v["at"] - v["lt"] - v["mib"]
    v["ceq"] = v["seq"] - v["pstk"]
    v["icapt"] = v["dltt"] + v["seq"] + v["mib"]
    v["at"][d == np.datetime64("2007-12-31")] = 0.0
    v["at"][d == np.datetime64("2008-12-31")] = -0.5


def _constant_earnings(v: Items, d: np.ndarray) -> None:
    scale = 100.0 / v["at"]
    for item in v:
        if item != "emp":
            v[item] = v[item] * scale
    for item in ["ib", "ni", "ibc"]:
        v[item] = np.full(len(d), 2.5)
    for item in ["xi", "do", "xido", "xidoc"]:
        v[item] = np.zeros(len(d))


def _year_end_to_november(a: pl.DataFrame, q: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Row edit: every report after December 1995 moves a month earlier onto a November year-end,
    keeping its fiscal year and quarter."""
    later = pl.col("datadate") > date(1995, 12, 31)
    moved = {
        "datadate": pl.when(later)
        .then(pl.col("datadate").dt.offset_by("-1mo").dt.month_end())
        .otherwise(pl.col("datadate")),
        "fyr": pl.when(later).then(pl.lit(11, pl.Int32)).otherwise(pl.col("fyr")),
    }
    return a.with_columns(**moved), q.with_columns(**moved)


def persona_specs() -> list[Spec]:
    """The persona firms listed in the module docstring, in gvkey order."""

    def firm(k: int, first_fy: int = 1995, last_fy: int = 2010, fyr: int = 12, **kw: object):
        gvkey = str(PERSONA_GVKEY + k)
        first, last = fiscal_quarter_end(first_fy, fyr, 1), fiscal_quarter_end(last_fy, fyr, 4)
        return {
            "gvkey": gvkey,
            "source": "NA",
            "currency": "USD",
            "indfmt": "INDL",
            "segments": [{"fyr": fyr, "first": first, "last": last}],
            "listed": (shift_months(first, -12), shift_months(last, 18)),
        } | kw

    def seg(fyr: int, first: date, last: date, **kw: object) -> dict:
        return {"fyr": fyr, "first": first, "last": last} | kw

    def quarterly_edit(edit: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable:
        return lambda a, q: (a, edit(q))

    d = date
    return [
        firm(
            0,
            rows=quarterly_edit(
                lambda q: _copy_rows(
                    q, pl.col("datadate") == d(2005, 12, 31), 1.02, datadate=d(2006, 1, 31)
                )
            ),
        ),
        firm(
            1,
            segments=[
                seg(12, d(1995, 3, 31), d(2005, 12, 31)),
                seg(12, d(2006, 3, 31), d(2006, 6, 30), annual=False),
                seg(6, d(2006, 3, 31), d(2010, 6, 30)),
            ],
            rows=quarterly_edit(
                lambda q: q.with_columns(
                    fqtr=pl.when((pl.col("fyr") == 12) & (pl.col("datadate") > d(2005, 12, 31)))
                    .then(None)
                    .otherwise(pl.col("fqtr"))
                )
            ),
        ),
        firm(2, rows=quarterly_edit(lambda q: q.filter(pl.col("datadate") != d(2005, 6, 30)))),
        firm(3, first_fy=1998, annual_first=d(2000, 12, 31)),
        firm(4, source="GLOBAL", currency="DEM", first_fy=1992, last_fy=2005),
        firm(
            5,
            rows=lambda a, q: (
                a.with_columns(curcd=pl.when(pl.col("fyear") != 2003).then(pl.col("curcd"))),
                q.with_columns(curcdq=pl.when(pl.col("fyearq") != 2003).then(pl.col("curcdq"))),
            ),
        ),
        firm(6, currency="CAD", first_fy=1976, last_fy=1995),
        firm(
            7,
            segments=[
                seg(12, d(1995, 3, 31), d(2005, 12, 31)),
                seg(6, d(2005, 9, 30), d(2010, 6, 30)),
            ],
        ),
        firm(
            8,
            segments=[
                seg(12, d(1995, 3, 31), d(2000, 12, 31)),
                seg(6, d(2001, 3, 31), d(2008, 6, 30)),
                seg(12, d(2002, 3, 31), d(2008, 12, 31), annual=False),
            ],
        ),
        firm(
            9,
            source="GLOBAL",
            currency="EUR",
            indfmt="FS",
            segments=[
                seg(12, d(1996, 3, 31), d(2006, 12, 31)),
                seg(6, d(2006, 9, 30), d(2012, 6, 30)),
            ],
        ),
        firm(
            10,
            source="GLOBAL",
            currency="GBP",
            first_fy=1996,
            rows=lambda a, q: tuple(
                _copy_rows(f, pl.col("datadate").dt.year().is_between(2003, 2005), 0.9, indfmt="FS")
                for f in (a, q)
            ),
        ),
        firm(11, last_fy=2012),
        firm(11, source="GLOBAL", first_fy=2000, listed=None),
        firm(12, source="GLOBAL", currency="JPY", fyr=3, last_fy=2012, second_security=True),
        firm(13, source="GLOBAL", currency="PLN", first_fy=1992),
        firm(14, quarterly="none"),
        firm(15, last_fy=2013, annual_last=d(2010, 12, 31)),
        firm(16, items=_zero_sales_negative_equity),
        firm(17, first_fy=2000, last_fy=2002),
        firm(18, first_fy=2000, last_fy=2005),
        firm(19, first_fy=2000, last_fy=2011),
        firm(20, items=_constant_earnings),
        firm(
            21,
            segments=[
                seg(12, d(1995, 3, 31), d(2001, 12, 31)),
                seg(12, d(2005, 3, 31), d(2010, 12, 31)),
            ],
        ),
        firm(
            22,
            last_fy=2005,
            rows=_filtered_copies(
                [{"datafmt": "SUMM_STD"}, {"indfmt": "FS"}, {"consol": "P"}, {"popsrc": "I"}]
            ),
        ),
        firm(
            23,
            source="GLOBAL",
            currency="EUR",
            last_fy=2005,
            rows=_filtered_copies([{"datafmt": "STD"}, {"popsrc": "D"}, {"consol": "P"}]),
        ),
        firm(24),
        firm(
            24,
            source="GLOBAL",
            segments=[seg(6, d(2002, 9, 30), d(2008, 6, 30), annual=False)],
            listed=None,
        ),
        firm(25, source="GLOBAL", currency="EUR", indfmt="FS", first_fy=1996),
        firm(
            25,
            source="GLOBAL",
            currency="EUR",
            segments=[seg(6, d(2002, 9, 30), d(2008, 6, 30), annual=False)],
            listed=None,
        ),
        firm(26, last_fy=2005, quarterly_first=d(1995, 12, 31), rows=_year_end_to_november),
    ]


def fx_quotes(rng: np.random.Generator) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Description:
        Weekly exchange rates as comp_exrt_dly rows (units per GBP) and as the USD rate per unit
        the stage derives from them.
    Steps:
        1) Random-walk each currency's USD rate from its start level; drop quotes outside each
           currency's window.
        2) Write units per GBP rounded to 8 decimals, the precision of exratd.
        3) Derive USD per unit as gen_fx1 does (USD per GBP / units per GBP), and add the bundled
           CAD rates before the first CAD quote and a USD seed, as compustat_fx does.
    Output:
        (comp_exrt_dly rows, quotes with curcdd, datadate, fx).
    """
    days = pl.date_range(FX_FIRST, FX_LAST, "1w", eager=True)
    walks = {
        c: level * np.exp(np.cumsum(rng.normal(0.0, FX_WEEKLY_VOL * (c != "USD"), len(days))))
        for c, level in FX_START_LEVEL.items()
    }
    usd = pl.DataFrame({"datadate": days, **walks}).unpivot(
        index="datadate", variable_name="tocurd", value_name="usd"
    )
    gbp = usd.filter(pl.col("tocurd") == "GBP").select("datadate", gbp_usd="usd")
    window = pl.lit(True)
    for c, last in FX_LAST_QUOTE.items():
        window &= (pl.col("tocurd") != c) | (pl.col("datadate") <= last)
    for c, first in FX_FIRST_QUOTE.items():
        window &= (pl.col("tocurd") != c) | (pl.col("datadate") >= first)
    for c, (gap_first, gap_last) in FX_QUOTE_GAP.items():
        window &= (pl.col("tocurd") != c) | ~pl.col("datadate").is_between(gap_first, gap_last)
    exrt = (
        usd.join(gbp, on="datadate")
        .filter(window)
        .with_columns(exratd=(pl.col("gbp_usd") / pl.col("usd")).round(8))
        .with_columns(
            exratd=pl.when(pl.col("tocurd") == "GBP").then(1.0).otherwise(pl.col("exratd"))
        )
        .select(
            "tocurd",
            "exratd",
            exrattpd=pl.lit(None, pl.String),
            fromcurd=pl.lit("GBP"),
            datadate="datadate",
        )
        .sort("datadate", "tocurd")
    )
    usd_per_gbp = exrt.filter(pl.col("tocurd") == "USD").select("datadate", b="exratd")
    quotes = exrt.join(usd_per_gbp, on="datadate").select(
        curcdd="tocurd", datadate="datadate", fx=pl.col("b") / pl.col("exratd")
    )
    cad = pl.read_csv(
        get_fx_cad_backfill_path(),
        schema={"curcdd": pl.String, "datadate": pl.Date, "fx": pl.Float64},
    ).filter(pl.col("datadate") < FX_FIRST)
    seed = pl.DataFrame({"curcdd": ["USD"], "datadate": [date(1950, 1, 1)], "fx": [1.0]})
    return exrt, pl.concat([seed, cad, quotes]).sort("curcdd", "datadate")


def to_local_currency(df: pl.DataFrame, quotes: pl.DataFrame, items: list[str]) -> pl.DataFrame:
    """Divide USD items by the latest rate on or before datadate (the first rate before any)."""
    fx = quotes.rename({"curcdd": "currency"}).sort("datadate")
    joined = (
        df.sort("datadate")
        .join_asof(fx, on="datadate", by="currency", strategy="backward", check_sortedness=False)
        .join_asof(
            fx.rename({"fx": "first_fx"}),
            on="datadate",
            by="currency",
            strategy="forward",
            check_sortedness=False,
        )
    )
    return joined.with_columns(
        (pl.col(c) / pl.coalesce("fx", "first_fx")).round(3) for c in items if c in df.columns
    ).drop("fx", "first_fx", "currency")


def polars_dtype(name: str) -> pl.DataType:
    """Parse a dtype as the profile writes it, e.g. 'Decimal(precision=18, scale=4)'."""
    if m := re.fullmatch(r"Decimal\(precision=(\d+), scale=(\d+)\)", name):
        return pl.Decimal(int(m[1]), int(m[2]))
    return getattr(pl, name)()


def cast_to_schema(df: pl.DataFrame, schema: list[list[str]], keys: list[str]) -> pl.DataFrame:
    """The real column set and dtypes, in the real column order; absent columns are null."""
    return df.sort(keys).select(
        pl.col(c).cast(polars_dtype(t))
        if c in df.columns
        else pl.lit(None, polars_dtype(t)).alias(c)
        for c, t in schema
    )


def generate(out_dir: Path, profile_path: Path = PROFILE_PATH) -> dict[str, int]:
    """
    Description:
        Write the synthetic Compustat tables and world_msf.
    Steps:
        1) Draw the FX quotes, then every random and persona firm, in a fixed order.
        2) Convert to the reporting currency and cast to the real schemas.
        3) Write the parquet files.
    Output:
        Row count per written file.
    """
    profile = json.loads(profile_path.read_text())
    rng = np.random.default_rng(SEED)
    exrt, quotes = fx_quotes(rng)
    ctx = {s: source_context(profile, s) for s in ("NA", "GLOBAL")}
    frames: dict[str, list[pl.DataFrame]] = {t: [] for t in TABLES} | {"world_msf": []}
    for spec in random_specs(rng, profile) + persona_specs():
        annual, quarterly, msf = firm_frames(spec, rng, ctx, profile)
        frames[TABLE[(spec["source"], "annual")]].append(annual)
        if quarterly.height:
            frames[TABLE[(spec["source"], "quarterly")]].append(quarterly)
        frames["world_msf"].append(msf)

    paths = DataPaths(out_dir)
    paths.raw_tables_dir.mkdir(parents=True, exist_ok=True)
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for table, (_, _, freq) in TABLES.items():
        items = FLOWS + STOCKS if freq == "annual" else [f"{i}{s}" for i in FLOWS for s in "qy"]
        items = items if freq == "annual" else items + [f"{s}q" for s in STOCKS]
        keys = ["gvkey", "datadate", "indfmt", "datafmt", "consol", "popsrc"]
        keys = keys if freq == "annual" else [*keys, "fyr"]
        df = to_local_currency(pl.concat(frames[table], how="diagonal_relaxed"), quotes, items)
        df = cast_to_schema(df, profile["schemas"][table], keys)
        df.write_parquet(paths.raw_tables_dir / f"{table}.parquet")
        counts[table] = df.height
    exrt = cast_to_schema(exrt, profile["schemas"]["comp_exrt_dly"], ["datadate", "tocurd"])
    exrt.write_parquet(paths.raw_tables_dir / "comp_exrt_dly.parquet")
    msf = (
        pl.concat([f for f in frames["world_msf"] if f.height])
        .select(
            "gvkey",
            "eom",
            pl.col("primary_sec", "common", "obs_main").cast(pl.Int32),
            pl.col("me_company").round(4),
        )
        .sort("gvkey", "eom", "primary_sec")
    )
    msf.write_parquet(paths.interim_dir / "world_msf.parquet")
    return counts | {"comp_exrt_dly": exrt.height, "world_msf": msf.height}


if __name__ == "__main__":
    for name, rows in generate(Path(sys.argv[1])).items():
        print(f"{name}: {rows:,} rows")
