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
    "hml": pl.Float64,
    "smb_ff": pl.Float64,
}

# --- Calendar grid -----------------------------------------------------------
_N_MONTHS = 40  # m0..m39 = 2013-01-31 .. 2016-04-30
_START_YEAR = 2013
_START_MONTH = 1
_CA_HML_NULL_MONTH = 20  # CA hml is null at m20 (hml-null branch)

# Countries and their factor coefficients for the return-generating process.
_COUNTRIES = ("US", "CA")

# (id, excntry, first_month, last_month_inclusive) — each row-map exercises a path.
#   id 1 US m0..m39 (40)  normal rolling stock; value-assertion anchor.
#   id 2 US m0..m19 (20)  never >=24 obs -> absent from both outputs.
#   id 3 US m16..m39 (24) min-obs boundary: terminal chunk has exactly 24.
#   id 4 CA m0..m39 (40)  hml-null branch (CA hml null at m20).
_STOCKS: tuple[tuple[int, str, int, int], ...] = (
    (1, "US", 0, 39),
    (2, "US", 0, 19),
    (3, "US", 16, 39),
    (4, "CA", 0, 39),
)


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
        Sample ``mktrf``/``hml``/``smb_ff`` for each country from N(0, 0.04).
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
            "hml": rng.normal(0.0, 0.04, _N_MONTHS),
            "smb_ff": rng.normal(0.0, 0.04, _N_MONTHS),
        }
    return factors


def build_ap_factors_monthly_input(seed: int = 42) -> pl.DataFrame:
    """Build the synthetic ``ap_factors_monthly`` input frame.

    Description:
        Per-country monthly FF3 factors keyed on ``(excntry, eom)``.
    Steps:
        1) Draw factor arrays via ``_draw_country_factors``.
        2) Emit one row per (country, month); null CA ``hml`` at m20.
    Output:
        DataFrame with schema ``AP_FACTORS_MONTHLY_INPUT_SCHEMA`` (80 rows).
    """
    grid = month_grid()
    factors = _draw_country_factors(seed)
    rows: list[dict] = []
    for country in _COUNTRIES:
        f = factors[country]
        for i, eom in enumerate(grid):
            hml = f["hml"][i]
            if country == "CA" and i == _CA_HML_NULL_MONTH:
                hml_val = None
            else:
                hml_val = float(hml)
            rows.append(
                {
                    "excntry": country,
                    "eom": eom,
                    "mktrf": float(f["mktrf"][i]),
                    "hml": hml_val,
                    "smb_ff": float(f["smb_ff"][i]),
                }
            )
    return pl.DataFrame(rows, schema=AP_FACTORS_MONTHLY_INPUT_SCHEMA)


def build_world_msf_input(seed: int = 42, empty: bool = False) -> pl.DataFrame:
    """Build the synthetic ``world_msf`` stock-month panel.

    Description:
        Four stocks across two countries; ``ret_exc`` follows a linear FF3
        model plus idiosyncratic noise so residuals are non-degenerate.
    Steps:
        1) Draw factors first (same order as the factor builder), then per-stock
           noise ~ N(0, 0.02).
        2) ``ret_exc = 0.005 + 1.1*mktrf + 0.3*hml - 0.2*smb_ff + noise`` over the
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
            "hml": rng.normal(0.0, 0.04, _N_MONTHS),
            "smb_ff": rng.normal(0.0, 0.04, _N_MONTHS),
        }

    rows: list[dict] = []
    for sid, country, first, last in _STOCKS:
        f = factors[country]
        noise = rng.normal(0.0, 0.02, last - first + 1)
        for j, i in enumerate(range(first, last + 1)):
            ret_exc = (
                0.005 + 1.1 * f["mktrf"][i] + 0.3 * f["hml"][i] - 0.2 * f["smb_ff"][i] + noise[j]
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
    return pl.DataFrame(rows, schema=WORLD_MSF_INPUT_SCHEMA)
