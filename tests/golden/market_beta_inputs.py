"""Shared synthetic input builders for ``market_beta`` tests.

Used by ``tests/golden/generate_market_beta_golden.py`` and
``tests/golden/test_market_beta_golden.py`` so the input schema and the
deterministic toy panel are defined in exactly one place.

All values are drawn from ``numpy.random.default_rng`` with fixed seeds or are
hand-written literals; no proprietary data is involved.
"""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import polars as pl

# One country, a 72-month month-end calendar (2010-01-31 .. 2015-12-31).
_EXCNTRY = "USA"
_N_MONTHS = 72

# ---------------------------------------------------------------------------
# Bulk panel (golden-size only).
#
# The hand-crafted USA firms above pin specific code paths; the golden must
# also be a realistic *size*. To that end we append a large deterministic
# synthetic panel in its own country ("BLK") on a date range that does NOT
# overlap the 2010-2015 USA calendar. Because winsorization in
# prep_data_factor_regs partitions by eom ALONE (not excntry+eom), a disjoint
# eom range is what guarantees the bulk rows cannot shift the winsorized
# ret_exc of any USA edge-case firm — so the edge-case golden values are byte
# stable regardless of bulk size. All draws come from numpy default_rng with
# fixed seeds; the data is fake.
_BULK_EXCNTRY = "BLK"
_BULK_ID_BASE = 100_000  # far above the 10001-10003 edge-case ids -> no collision
_BULK_START = -420  # month index (rel. 2010-01) -> 1975-01; strictly before month 0
_N_BULK_FIRMS = 340
_N_BULK_MONTHS = 132  # each firm -> (132 - 35) = 97 output windows
_BULK_SEED = 20_260_707

WORLD_MSF_INPUT_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "excntry": pl.String,
    "eom": pl.Date,
    "ret_exc": pl.Float64,
    "ret_local": pl.Float64,
    "ret_lag_dif": pl.Int64,
}

AP_FACTORS_MONTHLY_INPUT_SCHEMA: dict[str, pl.DataType] = {
    "excntry": pl.String,
    "eom": pl.Date,
    "mktrf": pl.Float64,
    "hml": pl.Float64,
    "smb_ff3": pl.Float64,
}


def _month_end(i: int) -> date:
    """Return the month-end date ``i`` months after 2010-01-31."""
    year = 2010 + i // 12
    month = i % 12 + 1
    return date(year, month, calendar.monthrange(year, month)[1])


def _calendar() -> list[date]:
    """Return the 72 month-end dates spanning 2010-01 .. 2015-12."""
    return [_month_end(i) for i in range(_N_MONTHS)]


def _factor_arrays(seed: int) -> tuple[list[date], np.ndarray, np.ndarray, np.ndarray]:
    """Draw the (dates, mktrf, hml, smb_ff3) factor series deterministically.

    Kept private and drawn from a fresh ``default_rng(seed)`` so both the factor
    builder and the return builder reproduce the identical ``mktrf`` series.
    """
    rng = np.random.default_rng(seed)
    dates = _calendar()
    mktrf = rng.normal(0.005, 0.05, _N_MONTHS)
    hml = rng.normal(0.0, 0.03, _N_MONTHS)
    smb_ff3 = rng.normal(0.0, 0.03, _N_MONTHS)
    return dates, mktrf, hml, smb_ff3


def _bulk_calendar() -> list[date]:
    """Return the disjoint month-end calendar for the BLK bulk panel."""
    return [_month_end(_BULK_START + i) for i in range(_N_BULK_MONTHS)]


def _bulk_factor_arrays() -> tuple[list[date], np.ndarray, np.ndarray, np.ndarray]:
    """Draw the (dates, mktrf, hml, smb_ff3) BLK factor series deterministically.

    Drawn from a fixed ``default_rng(_BULK_SEED)`` so the factor builder and the
    return builder reproduce the identical ``mktrf`` series for the bulk panel.
    """
    rng = np.random.default_rng(_BULK_SEED)
    dates = _bulk_calendar()
    mktrf = rng.normal(0.005, 0.05, _N_BULK_MONTHS)
    hml = rng.normal(0.0, 0.03, _N_BULK_MONTHS)
    smb_ff3 = rng.normal(0.0, 0.03, _N_BULK_MONTHS)
    return dates, mktrf, hml, smb_ff3


def build_ap_factors_monthly_input(seed: int = 42) -> pl.DataFrame:
    """Build the synthetic ``ap_factors_monthly`` fixture.

    Description:
        One factor row per (excntry, eom) for the 72-month USA calendar. Only
        ``mktrf`` is consumed by ``capm``; ``hml``/``smb_ff3`` are carried through
        ``prep_data_factor_regs`` but unused, so they are non-null noise.
    Steps:
        1) Draw mktrf ~ N(0.005, 0.05) and hml/smb_ff3 ~ N(0, 0.03) from rng(seed).
        2) Assemble one USA row per month, all non-null (mktrf-IS-NOT-NULL passes).
        3) Append the BLK bulk factor rows (own seed, disjoint eom range).
    Output:
        DataFrame with schema AP_FACTORS_MONTHLY_INPUT_SCHEMA
        (72 USA + _N_BULK_MONTHS BLK rows).
    """
    dates, mktrf, hml, smb_ff3 = _factor_arrays(seed)
    usa = pl.DataFrame(
        {
            "excntry": [_EXCNTRY] * _N_MONTHS,
            "eom": dates,
            "mktrf": mktrf,
            "hml": hml,
            "smb_ff3": smb_ff3,
        },
        schema=AP_FACTORS_MONTHLY_INPUT_SCHEMA,
    )
    b_dates, b_mktrf, b_hml, b_smb = _bulk_factor_arrays()
    bulk = pl.DataFrame(
        {
            "excntry": [_BULK_EXCNTRY] * _N_BULK_MONTHS,
            "eom": b_dates,
            "mktrf": b_mktrf,
            "hml": b_hml,
            "smb_ff3": b_smb,
        },
        schema=AP_FACTORS_MONTHLY_INPUT_SCHEMA,
    )
    return pl.concat([usa, bulk])


def build_world_msf_input(seed: int = 42) -> pl.DataFrame:
    """Build the synthetic ``world_msf`` monthly stock panel fixture.

    Description:
        A 3-firm panel whose excess returns are linear in the shared ``mktrf``
        series plus small idiosyncratic noise, so rolling CAPM betas recover the
        planted slopes.
    Steps:
        1) Rebuild the same mktrf series (via _factor_arrays(seed)) the factor file
           exposes, so ret_exc = beta_true * mktrf + alpha + noise is consistent.
        2) Draw independent noise ~ N(0, 0.01) per firm from a separate rng stream.
        3) Emit rows exercising each code path (see per-firm notes below).
    Output:
        DataFrame with schema WORLD_MSF_INPUT_SCHEMA.

    Per-firm path coverage:
        10001 "A" — 72 clean months, beta_true=1.2. Every 60-month window [E-59,E]
            has >=36 obs for E in [M35, M71] (37 output rows); E=M35 is the exact
            36-obs min-obs boundary (present) vs E=M34 (35 obs, absent).
        10002 "B" — 20 clean months (M0..M19), beta_true=0.8. No window ever
            reaches 36 obs -> firm yields NO output rows (insufficient history).
        10003 "C" — 72 months, beta_true=0.7, with three rows invalidated to
            exercise each prep_data_factor_regs WHERE filter: M2 has ret_local=0
            (dropped by ret_local<>0), M5 has ret_lag_dif=2 (dropped), M8 has
            ret_exc=null (dropped). Those rows never count toward min-obs, so the
            earliest window with >=36 valid obs ends at M38 (34 output rows).
    """
    _, mktrf, _, _ = _factor_arrays(seed)
    dates = _calendar()
    rng = np.random.default_rng(seed + 1000)

    ids: list[int] = []
    excntry: list[str] = []
    eom: list[date] = []
    ret_exc: list[float | None] = []
    ret_local: list[float] = []
    ret_lag_dif: list[int] = []

    def add(firm_id: int, months: range, beta_true: float, invalid: dict[int, str]) -> None:
        noise = rng.normal(0.0, 0.01, len(months))
        for k, m in enumerate(months):
            re: float | None = float(beta_true * mktrf[m] + 0.001 + noise[k])
            rl = re + 0.02
            lag = 1
            flag = invalid.get(m)
            if flag == "ret_local":
                rl = 0.0
            elif flag == "ret_lag_dif":
                lag = 2
            elif flag == "ret_exc":
                re = None
            ids.append(firm_id)
            excntry.append(_EXCNTRY)
            eom.append(dates[m])
            ret_exc.append(re)
            ret_local.append(rl)
            ret_lag_dif.append(lag)

    add(10001, range(0, 72), 1.2, {})
    add(10002, range(0, 20), 0.8, {})
    add(10003, range(0, 72), 0.7, {2: "ret_local", 5: "ret_lag_dif", 8: "ret_exc"})

    edge = pl.DataFrame(
        {
            "id": ids,
            "excntry": excntry,
            "eom": eom,
            "ret_exc": ret_exc,
            "ret_local": ret_local,
            "ret_lag_dif": ret_lag_dif,
        },
        schema=WORLD_MSF_INPUT_SCHEMA,
    )
    return pl.concat([edge, _build_bulk_msf()])


def _build_bulk_msf() -> pl.DataFrame:
    """Build the BLK bulk stock panel that scales the golden fixture.

    Description:
        _N_BULK_FIRMS synthetic firms, each spanning the full _N_BULK_MONTHS
        BLK calendar, with ret_exc = beta_true * mktrf + alpha + noise against
        the shared BLK mktrf series. Every firm has the full history, so each
        yields (_N_BULK_MONTHS - 35) non-null 60m/36-min output rows.
    Steps:
        1) Rebuild the BLK mktrf series (via _bulk_factor_arrays).
        2) Draw one beta_true per firm and per-firm-month noise from a separate
           default_rng stream so returns stay plausible and reproducible.
        3) Emit clean rows (ret_local != 0, ret_exc non-null, ret_lag_dif = 1).
    Output:
        DataFrame with schema WORLD_MSF_INPUT_SCHEMA
        (_N_BULK_FIRMS * _N_BULK_MONTHS rows).
    """
    dates, mktrf, _, _ = _bulk_factor_arrays()
    rng = np.random.default_rng(_BULK_SEED + 5000)
    betas = rng.uniform(0.2, 1.9, _N_BULK_FIRMS)
    total = _N_BULK_FIRMS * _N_BULK_MONTHS

    firm_ids = np.repeat(_BULK_ID_BASE + np.arange(_N_BULK_FIRMS), _N_BULK_MONTHS)
    tiled_mktrf = np.tile(mktrf, _N_BULK_FIRMS)
    tiled_beta = np.repeat(betas, _N_BULK_MONTHS)
    noise = rng.normal(0.0, 0.02, total)
    ret_exc = tiled_beta * tiled_mktrf + 0.001 + noise

    return pl.DataFrame(
        {
            "id": firm_ids.astype(np.int64),
            "excntry": [_BULK_EXCNTRY] * total,
            "eom": dates * _N_BULK_FIRMS,
            "ret_exc": ret_exc,
            "ret_local": ret_exc + 0.02,
            "ret_lag_dif": np.ones(total, dtype=np.int64),
        },
        schema=WORLD_MSF_INPUT_SCHEMA,
    )
