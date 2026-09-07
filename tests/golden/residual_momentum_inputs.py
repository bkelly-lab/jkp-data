"""Shared synthetic-input builders for ``residual_momentum`` fixtures.

Imported by BOTH ``tests/golden/generate_residual_momentum_golden.py`` and the
unit/golden tests so the input schemas are defined exactly once. Every value is
drawn from ``numpy.random.default_rng(42)`` or is a hand-written literal — no
WRDS/CRSP/Compustat data is embedded.

``residual_momentum`` reads two interim parquets:
  * ``world_msf.parquet``          (stock month panel)
  * ``ap_factors_monthly.parquet`` (per-country FF3 factors)
joined on ``(excntry, eom)`` inside ``prep_data_factor_regs``.
"""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import polars as pl

# --- Schemas (only the columns the prep SQL reads are required) --------------
WORLD_MSF_INPUT_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "eom": pl.Date,
    "excntry": pl.Utf8,
    "ret_exc": pl.Float64,
    "ret_lag_dif": pl.Int32,
    "ret_local": pl.Float64,
}

AP_FACTORS_MONTHLY_INPUT_SCHEMA: dict[str, pl.DataType] = {
    "excntry": pl.Utf8,
    "eom": pl.Date,
    "mktrf": pl.Float64,
    "hml_ff": pl.Float64,
    "smb_ff3": pl.Float64,
}

# --- Calendar grid -----------------------------------------------------------
_N_MONTHS = 40  # m0..m39 = 2013-01-31 .. 2016-04-30
_START_YEAR = 2013
_START_MONTH = 1
_CA_HML_NULL_MONTH = 20  # CA hml_ff is null at m20 (hml_ff-null branch)

# Countries and their factor coefficients for the return-generating process.
_COUNTRIES = ("US", "CA")

# (id, excntry, first_month, last_month_inclusive) — each row-map exercises a path.
#   id 1 US m0..m39 (40)  normal rolling stock; value-assertion anchor.
#   id 2 US m0..m19 (20)  never >=24 obs -> absent from both outputs.
#   id 3 US m16..m39 (24) min-obs boundary: terminal chunk has exactly 24.
#   id 4 CA m0..m39 (40)  hml_ff-null branch (CA hml_ff null at m20).
_STOCKS: tuple[tuple[int, str, int, int], ...] = (
    (1, "US", 0, 39),
    (2, "US", 0, 19),
    (3, "US", 16, 39),
    (4, "CA", 0, 39),
)

# --- Bulk panel (Goal 1: scale each golden fixture to ~0.5 MB) ---------------
# The bulk panel is a large deterministic synthetic cross-section appended ONLY
# when ``bulk=True``. It lives on a country and a month range that are BOTH
# disjoint from the hand-crafted edge grid, so:
#   * bulk ids never collide with edge ids 1..4 (bulk ids start at _BULK_ID0);
#   * bulk months never share an ``eom`` with the edge grid, so the per-``eom``
#     winsorization inside ``prep_data_factor_regs`` cannot alter edge values;
#   * bulk factor draws use an independent RNG seed, so the edge rng stream (and
#     therefore every edge fixture value) is byte-for-byte unchanged.
# Sizes are tuned by MEASUREMENT so each committed parquet lands around 0.5 MB.
_BULK_SEED = 2024
_BULK_COUNTRY = "BK"
_BULK_ID0 = 100_000
_BULK_N_ENTITIES = 600
_BULK_N_MONTHS = 132  # consecutive months >> _N (36)/__min (24): every entity fills windows
_BULK_START_YEAR = 1995
_BULK_START_MONTH = 1


def bulk_month_grid() -> list[date]:
    """Return the consecutive month-end dates of the bulk panel.

    Description:
        Build the bulk calendar, disjoint from ``month_grid`` (2013-2016).
    Steps:
        1) Advance month-by-month from (_BULK_START_YEAR, _BULK_START_MONTH).
        2) Snap each to its last calendar day.
    Output:
        List of ``date`` month-ends (length ``_BULK_N_MONTHS``).
    """
    out: list[date] = []
    for i in range(_BULK_N_MONTHS):
        total = _BULK_START_YEAR * 12 + (_BULK_START_MONTH - 1) + i
        year, month = divmod(total, 12)
        month += 1
        out.append(date(year, month, calendar.monthrange(year, month)[1]))
    return out


def _draw_bulk_factors(seed: int = _BULK_SEED) -> dict[str, np.ndarray]:
    """Draw the bulk country's FF3 factor arrays (fixed draw order).

    Description:
        Sample ``mktrf``/``hml_ff``/``smb_ff3`` for ``_BULK_COUNTRY`` from N(0, 0.04).
    Steps:
        1) Seed ``default_rng(seed)`` (independent of the edge stream).
        2) Draw the three factor arrays in a fixed order.
    Output:
        Mapping ``factor -> np.ndarray[_BULK_N_MONTHS]``. Both input builders call
        this so ``world_msf`` and ``ap_factors_monthly`` share identical factors.
    """
    rng = np.random.default_rng(seed)
    return {
        "mktrf": rng.normal(0.0, 0.04, _BULK_N_MONTHS),
        "hml_ff": rng.normal(0.0, 0.04, _BULK_N_MONTHS),
        "smb_ff3": rng.normal(0.0, 0.04, _BULK_N_MONTHS),
    }


def _bulk_ap_rows() -> list[dict]:
    """Build the bulk country's ``ap_factors_monthly`` rows.

    Description:
        One FF3 factor row per bulk month for ``_BULK_COUNTRY``.
    Steps:
        1) Draw shared bulk factors; emit one row per (country, bulk month).
    Output:
        List of dict rows matching ``AP_FACTORS_MONTHLY_INPUT_SCHEMA``.
    """
    grid = bulk_month_grid()
    f = _draw_bulk_factors()
    return [
        {
            "excntry": _BULK_COUNTRY,
            "eom": eom,
            "mktrf": float(f["mktrf"][i]),
            "hml_ff": float(f["hml_ff"][i]),
            "smb_ff3": float(f["smb_ff3"][i]),
        }
        for i, eom in enumerate(grid)
    ]


def _bulk_world_rows() -> list[dict]:
    """Build the bulk ``world_msf`` stock-month rows.

    Description:
        ``_BULK_N_ENTITIES`` stocks, each spanning the full bulk grid, with
        ``ret_exc`` following the shared FF3 model plus per-entity noise.
    Steps:
        1) Reproduce the shared bulk factors (same seed as ``_bulk_ap_rows``).
        2) Draw all idiosyncratic noise from an independent RNG (_BULK_SEED+1).
        3) Emit ``ret_exc = 0.005 + 1.1*mktrf + 0.3*hml_ff - 0.2*smb_ff3 + noise``.
    Output:
        List of dict rows matching ``WORLD_MSF_INPUT_SCHEMA`` (bulk ids only).
    """
    grid = bulk_month_grid()
    f = _draw_bulk_factors()
    noise_rng = np.random.default_rng(_BULK_SEED + 1)
    base = 0.005 + 1.1 * f["mktrf"] + 0.3 * f["hml_ff"] - 0.2 * f["smb_ff3"]
    rows: list[dict] = []
    for e in range(_BULK_N_ENTITIES):
        sid = _BULK_ID0 + e
        ret = base + noise_rng.normal(0.0, 0.02, _BULK_N_MONTHS)
        for i, eom in enumerate(grid):
            r = float(ret[i])
            rows.append(
                {
                    "id": sid,
                    "eom": eom,
                    "excntry": _BULK_COUNTRY,
                    "ret_exc": r,
                    "ret_lag_dif": 1,
                    "ret_local": r + 1.0,
                }
            )
    return rows


def month_grid() -> list[date]:
    """Return the ``_N_MONTHS`` consecutive month-end dates of the fixture.

    Description:
        Build the shared calendar used by both input frames.
    Steps:
        1) Advance month-by-month from (2013, 1).
        2) Snap each to its last calendar day.
    Output:
        List of ``date`` month-ends (length ``_N_MONTHS``).
    """
    out: list[date] = []
    for i in range(_N_MONTHS):
        total = _START_YEAR * 12 + (_START_MONTH - 1) + i
        year, month = divmod(total, 12)
        month += 1
        out.append(date(year, month, calendar.monthrange(year, month)[1]))
    return out


def _draw_country_factors(seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Draw per-country FF3 factor arrays (fixed draw order for determinism).

    Description:
        Sample ``mktrf``/``hml_ff``/``smb_ff3`` for each country from N(0, 0.04).
    Steps:
        1) Seed ``default_rng(seed)``.
        2) For US then CA (fixed order), draw the three factor arrays.
    Output:
        Mapping ``country -> {factor: np.ndarray[_N_MONTHS]}``. The identical
        draw order lets ``world_msf`` (which draws factors first, then noise)
        and ``ap_factors_monthly`` reproduce the same factor values.
    """
    rng = np.random.default_rng(seed)
    factors: dict[str, dict[str, np.ndarray]] = {}
    for country in _COUNTRIES:
        factors[country] = {
            "mktrf": rng.normal(0.0, 0.04, _N_MONTHS),
            "hml_ff": rng.normal(0.0, 0.04, _N_MONTHS),
            "smb_ff3": rng.normal(0.0, 0.04, _N_MONTHS),
        }
    return factors


def build_ap_factors_monthly_input(seed: int = 42, bulk: bool = False) -> pl.DataFrame:
    """Build the synthetic ``ap_factors_monthly`` input frame.

    Description:
        Per-country monthly FF3 factors keyed on ``(excntry, eom)``.
    Steps:
        1) Draw factor arrays via ``_draw_country_factors``.
        2) Emit one row per (country, month); null CA ``hml_ff`` at m20.
        3) When ``bulk``, append the disjoint bulk-country factor rows.
    Output:
        DataFrame with schema ``AP_FACTORS_MONTHLY_INPUT_SCHEMA`` (80 edge rows,
        plus ``_BULK_N_MONTHS`` bulk rows when ``bulk`` is True).
    """
    grid = month_grid()
    factors = _draw_country_factors(seed)
    rows: list[dict] = []
    for country in _COUNTRIES:
        f = factors[country]
        for i, eom in enumerate(grid):
            hml_ff = f["hml_ff"][i]
            if country == "CA" and i == _CA_HML_NULL_MONTH:
                hml_val = None
            else:
                hml_val = float(hml_ff)
            rows.append(
                {
                    "excntry": country,
                    "eom": eom,
                    "mktrf": float(f["mktrf"][i]),
                    "hml_ff": hml_val,
                    "smb_ff3": float(f["smb_ff3"][i]),
                }
            )
    if bulk:
        rows.extend(_bulk_ap_rows())
    return pl.DataFrame(rows, schema=AP_FACTORS_MONTHLY_INPUT_SCHEMA)


def build_world_msf_input(seed: int = 42, empty: bool = False, bulk: bool = False) -> pl.DataFrame:
    """Build the synthetic ``world_msf`` stock-month panel.

    Description:
        Four stocks across two countries; ``ret_exc`` follows a linear FF3
        model plus idiosyncratic noise so residuals are non-degenerate.
    Steps:
        1) Draw factors first (same order as the factor builder), then per-stock
           noise ~ N(0, 0.02).
        2) ``ret_exc = 0.005 + 1.1*mktrf + 0.3*hml_ff - 0.2*smb_ff3 + noise`` over the
           stock's present months; ``ret_local = ret_exc + 1`` (always nonzero);
           ``ret_lag_dif = 1``.
    Output:
        DataFrame with schema ``WORLD_MSF_INPUT_SCHEMA`` (124 rows), or a typed
        0-row frame when ``empty`` is True.
    """
    if empty:
        return pl.DataFrame(schema=WORLD_MSF_INPUT_SCHEMA)

    grid = month_grid()
    rng = np.random.default_rng(seed)
    # Reproduce the factor draws (same order) so ret_exc matches the factor file.
    factors: dict[str, dict[str, np.ndarray]] = {}
    for country in _COUNTRIES:
        factors[country] = {
            "mktrf": rng.normal(0.0, 0.04, _N_MONTHS),
            "hml_ff": rng.normal(0.0, 0.04, _N_MONTHS),
            "smb_ff3": rng.normal(0.0, 0.04, _N_MONTHS),
        }

    rows: list[dict] = []
    for sid, country, first, last in _STOCKS:
        f = factors[country]
        noise = rng.normal(0.0, 0.02, last - first + 1)
        for j, i in enumerate(range(first, last + 1)):
            ret_exc = (
                0.005
                + 1.1 * f["mktrf"][i]
                + 0.3 * f["hml_ff"][i]
                - 0.2 * f["smb_ff3"][i]
                + noise[j]
            )
            rows.append(
                {
                    "id": sid,
                    "eom": grid[i],
                    "excntry": country,
                    "ret_exc": float(ret_exc),
                    "ret_lag_dif": 1,
                    "ret_local": float(ret_exc + 1.0),
                }
            )
    if bulk:
        rows.extend(_bulk_world_rows())
    return pl.DataFrame(rows, schema=WORLD_MSF_INPUT_SCHEMA)
