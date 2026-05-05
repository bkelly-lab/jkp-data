"""Shared synthetic-data builders and helpers for portfolio.py tests.

This module exposes plain functions (no fixtures) used across the
``tests/unit/portfolio/`` test suite. Top-level pytest fixtures
(``seed``, ``tolerance``, ``make_dataframe``, ``temp_data_dir``,
``assert_series_equal``) are inherited from ``tests/conftest.py``.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNTHETIC_CHARS: list[str] = [
    "char_a",
    "char_b",
    "char_c",
    "char_d",
    "char_e",
    "char_f",
    "char_g",
    "char_h",
    "char_i",
    "char_j",
]

N_IDS_DEFAULT = 120
N_MONTHS_DEFAULT = 24

TIGHT = {"rtol": 1e-10, "atol": 1e-12}
STANDARD = {"rtol": 1e-6, "atol": 1e-10}
LOOSE = {"rtol": 1e-4, "atol": 1e-8}

NUMERIC_COLS_BY_FILE: dict[str, dict[str, dict]] = {
    "pfs": {
        "n": TIGHT,
        "signal": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "hml": {
        "n_stocks": TIGHT,
        "n_stocks_min": TIGHT,
        "signal": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "lms": {
        "n_stocks": TIGHT,
        "n_stocks_min": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "clusters": {
        "n_factors": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "cmp": {"n_stocks": TIGHT, "ret_weighted": STANDARD, "signal_weighted": STANDARD},
    "industry_gics": {
        "n": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "industry_ff49": {
        "n": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "pfs_daily": {"n": TIGHT, "ret_ew": TIGHT, "ret_vw": STANDARD, "ret_vw_cap": STANDARD},
    "hml_daily": {
        "n_stocks": TIGHT,
        "n_stocks_min": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "lms_daily": {
        "n_stocks": TIGHT,
        "n_stocks_min": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "clusters_daily": {
        "n_factors": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "industry_gics_daily": {
        "n": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "industry_ff49_daily": {
        "n": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
    "regional": {
        "n_countries": TIGHT,
        "ret_ew": STANDARD,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
        "mkt_vw_exc": STANDARD,
    },
    "country": {
        "n_stocks": TIGHT,
        "n_stocks_min": TIGHT,
        "ret_ew": TIGHT,
        "ret_vw": STANDARD,
        "ret_vw_cap": STANDARD,
    },
}


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def month_ends(n_months: int, start_year: int = 2020, start_month: int = 1) -> list[date]:
    """Return ``n_months`` consecutive month-end ``date`` values."""
    out: list[date] = []
    for i in range(n_months):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        out.append(date(year, month, calendar.monthrange(year, month)[1]))
    return out


def next_month_end(d: date) -> date:
    """Return the month-end ``date`` of the month following ``d``."""
    year = d.year + (1 if d.month == 12 else 0)
    month = 1 if d.month == 12 else d.month + 1
    return date(year, month, calendar.monthrange(year, month)[1])


def weekdays_in_month_after(eom: date, n: int = 21) -> list[date]:
    """Return up to ``n`` consecutive weekdays starting the day after ``eom``."""
    out: list[date] = []
    d = eom + timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Synthetic data builders (canonical: parametrized version)
# ---------------------------------------------------------------------------


def make_country_characteristics(
    excntry: str,
    chars: list[str],
    n_ids: int = N_IDS_DEFAULT,
    n_months: int = N_MONTHS_DEFAULT,
    seed: int = 42,
    start_year: int = 2020,
    start_month: int = 1,
) -> pl.DataFrame:
    """Build a synthetic monthly characteristics DataFrame for one country."""
    rng = np.random.default_rng(seed)
    eoms = month_ends(n_months, start_year=start_year, start_month=start_month)
    n_rows = n_ids * n_months

    size_grps_per_id = rng.choice(
        ["mega", "large", "small", "micro", "nano"],
        size=n_ids,
        p=[0.10, 0.25, 0.35, 0.20, 0.10],
    ).tolist()
    source_crsp_per_id: list[int] = rng.choice([0, 1], size=n_ids, p=[0.4, 0.6]).tolist()
    crsp_exchcd_choices = rng.choice([1, 2, 3], size=n_ids).tolist()
    comp_exchg_choices = rng.choice([11, 12, 13], size=n_ids).tolist()
    crsp_exchcd_per_id: list[int | None] = [
        int(crsp_exchcd_choices[i]) if source_crsp_per_id[i] == 1 else None for i in range(n_ids)
    ]
    comp_exchg_per_id: list[int | None] = [
        int(comp_exchg_choices[i]) if source_crsp_per_id[i] == 0 else None for i in range(n_ids)
    ]
    gics_sectors = rng.choice([10, 15, 20, 25, 30, 35], size=n_ids).tolist()
    gics_per_id = [f"{int(s):02d}101010" for s in gics_sectors]
    ff49_per_id = rng.choice([1, 5, 10, 15, 20, 30, 40, 45], size=n_ids).tolist()

    id_col: list[int] = []
    eom_col: list[date] = []
    sg_col: list[str] = []
    sc_col: list[int] = []
    ce_col: list[int | None] = []
    cx_col: list[int | None] = []
    gics_col: list[str] = []
    ff49_col: list[int] = []
    for eom in eoms:
        for i in range(n_ids):
            id_col.append(i + 1)
            eom_col.append(eom)
            sg_col.append(size_grps_per_id[i])
            sc_col.append(int(source_crsp_per_id[i]))
            ce_col.append(crsp_exchcd_per_id[i])
            cx_col.append(comp_exchg_per_id[i])
            gics_col.append(gics_per_id[i])
            ff49_col.append(int(ff49_per_id[i]))

    ret_exc = rng.normal(0.008, 0.08, n_rows)
    ret_exc_lead1m = rng.normal(0.008, 0.08, n_rows)
    me = np.exp(rng.normal(7.0, 1.5, n_rows))

    char_dict: dict[str, pl.Series] = {}
    for j, ch in enumerate(chars):
        vals = rng.normal(0.0, 1.0 + 0.1 * j, n_rows)
        mask = rng.random(n_rows) < (0.05 + 0.01 * j)
        vals[mask] = np.nan
        char_dict[ch] = pl.Series(ch, vals, dtype=pl.Float64)

    return pl.DataFrame(
        {
            "id": pl.Series("id", id_col, dtype=pl.Int64),
            "eom": pl.Series("eom", eom_col, dtype=pl.Date),
            "source_crsp": pl.Series("source_crsp", sc_col, dtype=pl.Int64),
            "comp_exchg": pl.Series("comp_exchg", cx_col, dtype=pl.Int64),
            "crsp_exchcd": pl.Series("crsp_exchcd", ce_col, dtype=pl.Int64),
            "size_grp": pl.Series("size_grp", sg_col, dtype=pl.Utf8),
            "ret_exc": pl.Series("ret_exc", ret_exc, dtype=pl.Float64),
            "ret_exc_lead1m": pl.Series("ret_exc_lead1m", ret_exc_lead1m, dtype=pl.Float64),
            "me": pl.Series("me", me, dtype=pl.Float64),
            "gics": pl.Series("gics", gics_col, dtype=pl.Utf8),
            "ff49": pl.Series("ff49", ff49_col, dtype=pl.Int64),
            "excntry": pl.Series("excntry", [excntry] * n_rows, dtype=pl.Utf8),
            **char_dict,
        }
    )


def make_daily_returns(char_df: pl.DataFrame, seed: int = 43) -> pl.DataFrame:
    """Build a synthetic daily returns DataFrame consistent with ``char_df``."""
    rng = np.random.default_rng(seed)
    pairs = char_df.select(["id", "eom"]).unique().sort(["eom", "id"])
    ids: list[int] = []
    dates: list[date] = []
    rets: list[float] = []
    for row in pairs.iter_rows(named=True):
        for d in weekdays_in_month_after(row["eom"], 21):
            ids.append(int(row["id"]))
            dates.append(d)
            rets.append(float(rng.normal(0.0003, 0.015)))
    return pl.DataFrame(
        {
            "id": pl.Series("id", ids, dtype=pl.Int64),
            "date": pl.Series("date", dates, dtype=pl.Date),
            "ret_exc": pl.Series("ret_exc", rets, dtype=pl.Float64),
        }
    )


def make_cutoffs(eoms: list[date]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build ``nyse_size_cutoffs``, ``ret_cutoffs``, ``ret_cutoffs_daily`` frames."""
    n = len(eoms)
    nyse_size_cutoffs = pl.DataFrame(
        {
            "eom": pl.Series("eom", eoms, dtype=pl.Date),
            "nyse_p80": pl.Series("nyse_p80", [1e12] * n, dtype=pl.Float64),
        }
    )
    eom_lag1 = [date(eom.year, eom.month, 1) - timedelta(days=1) for eom in eoms]
    ret_cutoffs = pl.DataFrame(
        {
            "eom": pl.Series("eom", eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.5] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.5] * n, dtype=pl.Float64),
            "eom_lag1": pl.Series("eom_lag1", eom_lag1, dtype=pl.Date),
        }
    )
    daily_eoms = [next_month_end(eom) for eom in eoms]
    ret_cutoffs_daily = pl.DataFrame(
        {
            "eom": pl.Series("eom", daily_eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.2] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.2] * n, dtype=pl.Float64),
        }
    )
    return nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily


def write_synthetic_country(
    data_root: Path,
    excntry: str,
    chars: list[str],
    seed: int = 42,
    n_ids: int = N_IDS_DEFAULT,
    n_months: int = N_MONTHS_DEFAULT,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Write a synthetic country's characteristics and daily returns parquets."""
    char_dir = data_root / "characteristics"
    daily_dir = data_root / "return_data" / "daily_rets_by_country"
    char_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    char_df = make_country_characteristics(
        excntry=excntry, chars=chars, n_ids=n_ids, n_months=n_months, seed=seed
    )
    char_df.write_parquet(char_dir / f"{excntry}.parquet")
    daily_df = make_daily_returns(char_df, seed=seed + 1)
    daily_df.write_parquet(daily_dir / f"{excntry}.parquet")
    return char_df, daily_df


def build_synthetic_data(
    data_root: Path,
    countries: list[str],
    chars: list[str],
    n_ids: int = 60,
    n_months: int = 8,
    seed: int = 42,
) -> None:
    """Write a complete synthetic input tree under ``data_root/processed/``.

    Includes characteristics, daily returns, nyse cutoffs, return cutoffs,
    market returns, and daily market returns for ``run_portfolio``.
    """
    processed = data_root / "processed"
    char_dir = processed / "characteristics"
    daily_dir = processed / "return_data" / "daily_rets_by_country"
    other_dir = processed / "other_output"
    for d in (char_dir, daily_dir, other_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_eoms: list[date] = []
    for i, excntry in enumerate(countries):
        cdf = make_country_characteristics(
            excntry, chars, n_ids=n_ids, n_months=n_months, seed=seed + i * 100
        )
        cdf.write_parquet(char_dir / f"{excntry}.parquet")
        ddf = make_daily_returns(cdf, seed=seed + 1 + i * 100)
        ddf.write_parquet(daily_dir / f"{excntry}.parquet")
        if not all_eoms:
            all_eoms = cdf["eom"].unique().sort().to_list()

    n = len(all_eoms)

    pl.DataFrame(
        {
            "eom": pl.Series("eom", all_eoms, dtype=pl.Date),
            "nyse_p80": pl.Series("nyse_p80", [1e12] * n, dtype=pl.Float64),
        }
    ).write_parquet(other_dir / "nyse_cutoffs.parquet")

    pl.DataFrame(
        {
            "eom": pl.Series("eom", all_eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.5] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.5] * n, dtype=pl.Float64),
        }
    ).write_parquet(other_dir / "return_cutoffs.parquet")

    daily_eoms = [next_month_end(eom) for eom in all_eoms]
    pl.DataFrame(
        {
            "eom": pl.Series("eom", daily_eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.2] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.2] * n, dtype=pl.Float64),
        }
    ).write_parquet(other_dir / "return_cutoffs_daily.parquet")

    mkt_rows = [(ex, eom) for ex in countries for eom in all_eoms]
    rng = np.random.default_rng(seed + 999)
    pl.DataFrame(
        {
            "excntry": pl.Series("excntry", [r[0] for r in mkt_rows], dtype=pl.Utf8),
            "eom": pl.Series("eom", [r[1] for r in mkt_rows], dtype=pl.Date),
            "mkt_vw_exc": pl.Series(
                "mkt_vw_exc", rng.normal(0.008, 0.04, len(mkt_rows)), dtype=pl.Float64
            ),
            "me_lag1": pl.Series(
                "me_lag1", np.exp(rng.normal(10.0, 1.0, len(mkt_rows))), dtype=pl.Float64
            ),
            "stocks": pl.Series("stocks", rng.integers(20, 60, len(mkt_rows)), dtype=pl.Int64),
        }
    ).write_parquet(other_dir / "market_returns.parquet")

    daily_rows: list[tuple[str, date]] = []
    for ex in countries:
        for eom in all_eoms:
            for d in weekdays_in_month_after(eom, 21):
                daily_rows.append((ex, d))
    rng2 = np.random.default_rng(seed + 998)
    pl.DataFrame(
        {
            "excntry": pl.Series("excntry", [r[0] for r in daily_rows], dtype=pl.Utf8),
            "date": pl.Series("date", [r[1] for r in daily_rows], dtype=pl.Date),
            "mkt_vw_exc": pl.Series(
                "mkt_vw_exc", rng2.normal(0.0003, 0.015, len(daily_rows)), dtype=pl.Float64
            ),
            "me_lag1": pl.Series(
                "me_lag1", np.exp(rng2.normal(10.0, 1.0, len(daily_rows))), dtype=pl.Float64
            ),
            "stocks": pl.Series("stocks", rng2.integers(20, 60, len(daily_rows)), dtype=pl.Int64),
        }
    ).write_parquet(other_dir / "market_returns_daily.parquet")


# ---------------------------------------------------------------------------
# Targeted-shape datasets
# ---------------------------------------------------------------------------


def make_known_monotone_dataset(
    seed: int = 42, n_chars: int = 3, n_eoms: int = 6, n_ids: int = 60
) -> pl.DataFrame:
    """Build a dataset where ``char_*`` ranks are monotone in id within each eom."""
    rng = np.random.default_rng(seed)
    chars = [f"char_{chr(ord('a') + j)}" for j in range(n_chars)]
    eoms = month_ends(n_eoms)
    n_rows = n_ids * n_eoms

    id_col: list[int] = []
    eom_col: list[date] = []
    char_arrays: dict[str, list[float]] = {ch: [] for ch in chars}
    for eom in eoms:
        for i in range(n_ids):
            id_col.append(i + 1)
            eom_col.append(eom)
            for ch in chars:
                char_arrays[ch].append(float(i))

    size_grps = rng.choice(["mega", "large", "small", "nano"], size=n_ids).tolist()
    source_crsp = rng.choice([0, 1], size=n_ids, p=[0.4, 0.6]).tolist()
    crsp_exchcd_raw = rng.choice([1, 2, 3], size=n_ids).tolist()
    comp_exchg_raw = rng.choice([11, 12, 13], size=n_ids).tolist()
    gics_sectors = rng.choice([10, 15, 20, 25], size=n_ids).tolist()
    ff49_per_id = rng.choice([1, 5, 10, 15, 20], size=n_ids).tolist()

    sg_col = [size_grps[(i - 1) % n_ids] for i in id_col]
    sc_col = [int(source_crsp[(i - 1) % n_ids]) for i in id_col]
    ce_col = [
        int(crsp_exchcd_raw[(i - 1) % n_ids]) if sc_col[k] == 1 else None
        for k, i in enumerate(id_col)
    ]
    cx_col = [
        int(comp_exchg_raw[(i - 1) % n_ids]) if sc_col[k] == 0 else None
        for k, i in enumerate(id_col)
    ]
    gics_col = [f"{int(gics_sectors[(i - 1) % n_ids]):02d}101010" for i in id_col]
    ff49_col = [int(ff49_per_id[(i - 1) % n_ids]) for i in id_col]

    me = np.exp(rng.normal(7.0, 1.0, n_rows))
    nyse_p80 = np.full(n_rows, 1e12)
    me_cap = np.minimum(me, nyse_p80)
    ret_exc = rng.normal(0.008, 0.05, n_rows)
    ret_exc_lead1m = rng.normal(0.008, 0.05, n_rows)

    return pl.DataFrame(
        {
            "id": pl.Series("id", id_col, dtype=pl.Int64),
            "eom": pl.Series("eom", eom_col, dtype=pl.Date),
            "excntry": pl.Series("excntry", ["USA"] * n_rows, dtype=pl.Utf8),
            "source_crsp": pl.Series("source_crsp", sc_col, dtype=pl.Int64),
            "size_grp": pl.Series("size_grp", sg_col, dtype=pl.Utf8),
            "me": pl.Series("me", me, dtype=pl.Float64),
            "me_cap": pl.Series("me_cap", me_cap, dtype=pl.Float64),
            "nyse_p80": pl.Series("nyse_p80", nyse_p80, dtype=pl.Float64),
            "comp_exchg": pl.Series("comp_exchg", cx_col, dtype=pl.Int64),
            "crsp_exchcd": pl.Series("crsp_exchcd", ce_col, dtype=pl.Int64),
            "ret_exc": pl.Series("ret_exc", ret_exc, dtype=pl.Float64),
            "ret_exc_lead1m": pl.Series("ret_exc_lead1m", ret_exc_lead1m, dtype=pl.Float64),
            "ff49": pl.Series("ff49", ff49_col, dtype=pl.Int64),
            "gics": pl.Series("gics", gics_col, dtype=pl.Utf8),
            **{ch: pl.Series(ch, vals, dtype=pl.Float64) for ch, vals in char_arrays.items()},
        }
    )


def make_breakpoint_divergent_dataset(seed: int = 42) -> pl.DataFrame:
    """NYSE-flag set and non-microcap set deliberately disjoint."""
    n_ids = 80
    n_months = 4
    rng = np.random.default_rng(seed)
    eoms = month_ends(n_months)
    half = n_ids // 2

    # First half: NYSE-flagged but micro-cap. Second half: non-NYSE but large.
    size_grps = (["micro"] * half) + (["large"] * (n_ids - half))
    source_crsp = ([1] * half) + ([0] * (n_ids - half))
    crsp_exchcd = ([1] * half) + ([None] * (n_ids - half))
    comp_exchg = ([None] * half) + ([12] * (n_ids - half))

    n_rows = n_ids * n_months
    id_col: list[int] = []
    eom_col: list[date] = []
    sg_col: list[str] = []
    sc_col: list[int] = []
    ce_col: list[int | None] = []
    cx_col: list[int | None] = []
    for eom in eoms:
        for i in range(n_ids):
            id_col.append(i + 1)
            eom_col.append(eom)
            sg_col.append(size_grps[i])
            sc_col.append(source_crsp[i])
            ce_col.append(crsp_exchcd[i])
            cx_col.append(comp_exchg[i])

    me = np.exp(rng.normal(7.0, 1.0, n_rows))
    ret_exc = rng.normal(0.0, 0.05, n_rows)
    ret_exc_lead1m = rng.normal(0.0, 0.05, n_rows)
    char_a = rng.normal(0.0, 1.0, n_rows)
    gics_col = ["20101010"] * n_rows
    ff49_col = [1] * n_rows

    return pl.DataFrame(
        {
            "id": pl.Series("id", id_col, dtype=pl.Int64),
            "eom": pl.Series("eom", eom_col, dtype=pl.Date),
            "excntry": pl.Series("excntry", ["USA"] * n_rows, dtype=pl.Utf8),
            "source_crsp": pl.Series("source_crsp", sc_col, dtype=pl.Int64),
            "comp_exchg": pl.Series("comp_exchg", cx_col, dtype=pl.Int64),
            "crsp_exchcd": pl.Series("crsp_exchcd", ce_col, dtype=pl.Int64),
            "size_grp": pl.Series("size_grp", sg_col, dtype=pl.Utf8),
            "me": pl.Series("me", me, dtype=pl.Float64),
            "ret_exc": pl.Series("ret_exc", ret_exc, dtype=pl.Float64),
            "ret_exc_lead1m": pl.Series("ret_exc_lead1m", ret_exc_lead1m, dtype=pl.Float64),
            "ff49": pl.Series("ff49", ff49_col, dtype=pl.Int64),
            "gics": pl.Series("gics", gics_col, dtype=pl.Utf8),
            "char_a": pl.Series("char_a", char_a, dtype=pl.Float64),
        }
    )


def make_signals_dataset(seed: int = 42, null_rate: float = 0.10) -> pl.DataFrame:
    """Like ``make_known_monotone_dataset`` but with intentionally null cells."""
    df = make_known_monotone_dataset(seed=seed, n_chars=3, n_eoms=6, n_ids=60)
    rng = np.random.default_rng(seed + 1)
    char_cols = [c for c in df.columns if c.startswith("char_")]
    n = df.height
    for col in char_cols:
        mask = rng.random(n) < null_rate
        vals = df[col].to_numpy().copy()
        vals[mask] = np.nan
        df = df.with_columns(pl.Series(col, vals, dtype=pl.Float64))
    return df


def make_cmp_dataset(seed: int = 42) -> pl.DataFrame:
    """Includes a ``(size_grp, eom)`` cohort with ``sd_var == 0``."""
    df = make_known_monotone_dataset(seed=seed, n_chars=2, n_eoms=4, n_ids=40)
    # Pick the (size_grp="large", eom=first) cohort and set all char_a to a constant.
    eoms = df["eom"].unique().sort().to_list()
    target_eom = eoms[0]
    return df.with_columns(
        pl.when((pl.col("size_grp") == "large") & (pl.col("eom") == target_eom))
        .then(pl.lit(0.5))
        .otherwise(pl.col("char_a"))
        .alias("char_a"),
    )


def make_market_returns(countries: list[str], dates: list[date]) -> pl.DataFrame:
    """Synthetic ``market_returns`` frame with required columns."""
    rows = [(ex, d) for ex in countries for d in dates]
    rng = np.random.default_rng(123)
    return pl.DataFrame(
        {
            "excntry": pl.Series("excntry", [r[0] for r in rows], dtype=pl.Utf8),
            "eom": pl.Series("eom", [r[1] for r in rows], dtype=pl.Date),
            "mkt_vw_exc": pl.Series(
                "mkt_vw_exc", rng.normal(0.008, 0.04, len(rows)), dtype=pl.Float64
            ),
            "me_lag1": pl.Series(
                "me_lag1", np.exp(rng.normal(10.0, 1.0, len(rows))), dtype=pl.Float64
            ),
            "stocks": pl.Series("stocks", rng.integers(20, 60, len(rows)), dtype=pl.Int64),
        }
    )


# ---------------------------------------------------------------------------
# Stub resource writers (factor_details, country_classification, cluster_labels)
# ---------------------------------------------------------------------------


def make_multi_region_classification(tmp_path: Path) -> Path:
    """Write a stub ``country_classification.xlsx`` covering 3 MSCI tiers.

    Production reads xlsx; the test stub matches the production format.
    Requires ``xlsxwriter`` (test-only dependency).
    """
    out = tmp_path / "country_classification.xlsx"
    df = pl.DataFrame(
        {
            "excntry": [
                "USA",
                "GBR",
                "DEU",
                "JPN",
                "BRA",
                "IND",
                "ZAF",
                "EGY",
                "VNM",
                "ZWE",
                "VEN",
            ],
            "msci_development": [
                "developed",
                "developed",
                "developed",
                "developed",
                "emerging",
                "emerging",
                "emerging",
                "frontier",
                "frontier",
                "frontier",
                "frontier",
            ],
            "region": [
                "north_america",
                "europe",
                "europe",
                "asia",
                "latam",
                "asia",
                "africa",
                "africa",
                "asia",
                "africa",
                "latam",
            ],
        }
    )
    df.write_excel(workbook=out, worksheet="countries")
    return out


def make_factor_details(
    tmp_path: Path,
    characteristics: list[str],
    directions: list[int] | None = None,
) -> Path:
    """Write a stub ``factor_details.xlsx`` (sheet ``details``).

    Production reads xlsx; the test stub matches the production format.
    Requires ``xlsxwriter`` (test-only dependency).
    """
    if directions is None:
        directions = [1] * len(characteristics)
    if len(directions) != len(characteristics):
        raise ValueError("directions length must match characteristics length")
    out = tmp_path / "factor_details.xlsx"
    pl.DataFrame(
        {
            "abr_jkp": pl.Series("abr_jkp", characteristics, dtype=pl.Utf8),
            "direction": pl.Series("direction", directions, dtype=pl.Int64),
        }
    ).write_excel(workbook=out, worksheet="details")
    return out


def make_cluster_labels(tmp_path: Path, characteristic_to_cluster: dict[str, str]) -> Path:
    """Write a stub cluster labels as **parquet** (production uses csv)."""
    out = tmp_path / "cluster_labels.parquet"
    pl.DataFrame(
        {
            "characteristic": pl.Series(
                "characteristic", list(characteristic_to_cluster.keys()), dtype=pl.Utf8
            ),
            "cluster": pl.Series(
                "cluster", list(characteristic_to_cluster.values()), dtype=pl.Utf8
            ),
        }
    ).write_parquet(out)
    return out


def patch_resource_readers(monkeypatch) -> None:
    """Reroute ``pl.read_csv`` / ``pl.scan_csv`` inside ``jkp.data.portfolio``
    to parquet readers so tests can ship the cluster-labels stub as parquet
    rather than csv. xlsx-format stubs (``factor_details``,
    ``country_classification``) match production and need no patch.
    """
    import jkp.data.portfolio as _pmod

    _real_read_parquet = _pmod.pl.read_parquet
    _real_scan_parquet = _pmod.pl.scan_parquet

    def _fake_read_csv(path, **_kw):
        return _real_read_parquet(path)

    def _fake_scan_csv(path, **_kw):
        return _real_scan_parquet(path)

    monkeypatch.setattr(_pmod.pl, "read_csv", _fake_read_csv)
    monkeypatch.setattr(_pmod.pl, "scan_csv", _fake_scan_csv)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def assert_polars_frames_close(
    left: pl.DataFrame,
    right: pl.DataFrame,
    key_cols: list[str],
    numeric_cols: list[str],
    tolerance: dict | None = None,
) -> None:
    """Assert two Polars frames are equal: exact on key cols, allclose on numerics."""
    tol = tolerance or {"rtol": 1e-9, "atol": 1e-12}
    assert left.height == right.height, f"height mismatch: {left.height} vs {right.height}"
    if left.height == 0:
        return
    a = left.sort(key_cols)
    b = right.sort(key_cols)
    for col in key_cols:
        assert a[col].to_list() == b[col].to_list(), f"key column {col!r} mismatch"
    for col in numeric_cols:
        a_np = a[col].to_numpy().astype(np.float64)
        b_np = b[col].to_numpy().astype(np.float64)
        a_nan = np.isnan(a_np)
        b_nan = np.isnan(b_np)
        assert np.array_equal(a_nan, b_nan), f"NaN positions differ in {col!r}"
        mask = ~a_nan
        if mask.any():
            np.testing.assert_allclose(
                a_np[mask],
                b_np[mask],
                err_msg=f"column {col!r} beyond tolerance",
                **tol,
            )


def assert_frames_parity(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    key_cols: list[str],
    numeric_cols: dict[str, dict[str, float]],
    label: str,
) -> None:
    """Compare two frames: key cols exact, numeric cols per per-column tolerance."""
    assert actual.height == expected.height, (
        f"[{label}] height mismatch: {actual.height} vs {expected.height}"
    )
    assert actual.height > 0, f"[{label}] empty frame"
    a = actual.sort(key_cols)
    e = expected.sort(key_cols)
    for col in key_cols:
        assert a[col].to_list() == e[col].to_list(), f"[{label}] key {col!r} mismatch"
    for col, tol in numeric_cols.items():
        a_np = a[col].to_numpy().astype(np.float64)
        e_np = e[col].to_numpy().astype(np.float64)
        a_nan = np.isnan(a_np)
        e_nan = np.isnan(e_np)
        assert np.array_equal(a_nan, e_nan), f"[{label}] NaN positions differ in {col!r}"
        mask = ~a_nan
        if mask.any():
            np.testing.assert_allclose(
                a_np[mask],
                e_np[mask],
                err_msg=f"[{label}] {col!r} beyond tolerance",
                **tol,
            )


def get_numeric_spec(stem: str) -> dict[str, dict]:
    """Return per-column tolerance spec for a parquet file stem."""
    if stem in NUMERIC_COLS_BY_FILE:
        return NUMERIC_COLS_BY_FILE[stem]
    for prefix, spec in NUMERIC_COLS_BY_FILE.items():
        if stem.startswith(prefix):
            return spec
    return NUMERIC_COLS_BY_FILE["regional"]


def sort_key_cols(df: pl.DataFrame) -> list[str]:
    """Heuristic sort-key list for a parquet output frame."""
    date_cols = [c for c in ["eom", "date"] if c in df.columns]
    str_cols = [
        c
        for c in [
            "excntry",
            "region",
            "characteristic",
            "cluster",
            "gics",
            "ff49",
            "pf",
            "size_grp",
        ]
        if c in df.columns
    ]
    return date_cols + str_cols


def compare_parquets(path_a: Path, path_b: Path, label: str) -> list[str]:
    """Compare two parquet files. Return list of failure messages (empty = pass)."""
    failures: list[str] = []
    if not path_a.exists() and not path_b.exists():
        return []
    if not path_a.exists():
        return [f"{label}: missing in A"]
    if not path_b.exists():
        return [f"{label}: missing in B"]

    a = pl.read_parquet(path_a)
    b = pl.read_parquet(path_b)
    if a.height != b.height:
        return [f"{label}: height mismatch {a.height} vs {b.height}"]
    if a.height == 0:
        return []

    keys = sort_key_cols(a)
    try:
        a = a.sort(keys)
        b = b.sort(keys)
    except Exception:
        pass

    spec = get_numeric_spec(path_a.stem)
    for col, tol in spec.items():
        if col not in a.columns or col not in b.columns:
            continue
        a_np = a[col].to_numpy().astype(np.float64)
        b_np = b[col].to_numpy().astype(np.float64)
        a_nan = np.isnan(a_np)
        b_nan = np.isnan(b_np)
        if not np.array_equal(a_nan, b_nan):
            failures.append(f"{label}: NaN positions differ in {col!r}")
            continue
        mask = ~a_nan
        if mask.any():
            try:
                np.testing.assert_allclose(a_np[mask], b_np[mask], **tol)
            except AssertionError as e:
                failures.append(f"{label}: {col!r}: {e!s}")
    return failures
