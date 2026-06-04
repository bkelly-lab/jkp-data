"""Shared builders for ``sic_naics_na`` / ``sic_naics_gl`` Compustat input fixtures.

Used by ``tests/unit/test_comp_sic_naics.py`` and
``tests/golden/generate_comp_sic_naics_golden.py`` so the input schema is
defined in one place.
"""

from __future__ import annotations

from datetime import date

import polars as pl

SIC_NAICS_INPUT_SCHEMA: dict[str, pl.DataType] = {
    "gvkey": pl.Utf8,
    "datadate": pl.Date,
    "sic": pl.Int64,
    "naics": pl.Int64,
}


def sic_naics_frame(
    gvkeys: list[str],
    datadates: list[date],
    sics: list[int | None],
    naicses: list[int | None],
) -> pl.DataFrame:
    """Build a ``sic_naics_(na|gl)`` fixture. Empty lists yield a typed 0-row frame."""
    return pl.DataFrame(
        {"gvkey": gvkeys, "datadate": datadates, "sic": sics, "naics": naicses},
        schema=SIC_NAICS_INPUT_SCHEMA,
    )


def empty_sic_naics_frame() -> pl.DataFrame:
    """Return an empty NA/GL frame with the expected schema."""
    return pl.DataFrame(schema=SIC_NAICS_INPUT_SCHEMA)


def build_sic_naics_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build deterministic NA and GL SIC/NAICS fixtures.

    Scenarios covered by the returned (sic_naics_na, sic_naics_gl) pair:
        001000 — NA-only, two datadates (2020-01-01 → 2020-01-04) producing 3 daily
                 fill rows plus a trailing single-date row.
        002000 — GL-only, one datadate (2020-06-15).
        003000 — Both sources on same datadate → single joined row; COALESCE
                 prefers the NA value (6020).
        004000 — NA row with sic=NULL on one datadate; GL row with sic non-null
                 on the same datadate → coalesce keeps non-null SIC.
        175650 — Hard-coded dropped row (datadate=2005-12-31, naics IS NULL); a
                 separate datadate (2006-06-30) for the same gvkey is retained.
        500    — gvkey not zero-padded in input; output must be LPAD to '000500'.
    """
    sic_naics_na = sic_naics_frame(
        gvkeys=[
            "001000",
            "001000",
            "003000",
            "004000",
            "175650",
            "175650",
            "500",
        ],
        datadates=[
            date(2020, 1, 1),
            date(2020, 1, 4),
            date(2018, 5, 1),
            date(2019, 3, 1),
            date(2005, 12, 31),
            date(2006, 6, 30),
            date(2021, 7, 15),
        ],
        sics=[3711, 3713, 6020, None, 1311, 1311, 7372],
        naicses=[
            336111,
            336112,
            522110,
            541110,
            None,  # Triggers the hard-coded drop
            211120,
            511210,
        ],
    )
    sic_naics_gl = sic_naics_frame(
        gvkeys=["002000", "003000", "004000"],
        datadates=[
            date(2020, 6, 15),
            date(2018, 5, 1),
            date(2019, 3, 1),
        ],
        sics=[2834, 6021, 4813],
        naicses=[325412, 522120, 517110],
    )
    return sic_naics_na, sic_naics_gl
