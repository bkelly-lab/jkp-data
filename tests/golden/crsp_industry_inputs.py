"""Shared builders for ``permno0`` CRSP input fixtures.

Used by ``tests/unit/test_crsp_industry.py`` and
``tests/golden/generate_crsp_industry_golden.py`` so the input schema is
defined in one place.
"""

from __future__ import annotations

from datetime import date

import polars as pl

PERMNO0_INPUT_SCHEMA: dict[str, pl.DataType] = {
    "permno": pl.Int64,
    "permco": pl.Int64,
    "secinfostartdt": pl.Date,
    "secinfoenddt": pl.Date,
    "sic": pl.Int64,
    "naics": pl.Int64,
}


def permno0_frame(
    permnos: list[int],
    permcos: list[int],
    starts: list[date],
    ends: list[date],
    sics: list[int | None],
    naicses: list[int | None],
) -> pl.DataFrame:
    """Build a ``permno0`` fixture. Empty lists yield a typed 0-row frame."""
    return pl.DataFrame(
        {
            "permno": permnos,
            "permco": permcos,
            "secinfostartdt": starts,
            "secinfoenddt": ends,
            "sic": sics,
            "naics": naicses,
        },
        schema=PERMNO0_INPUT_SCHEMA,
    )


def build_permno0_input() -> pl.DataFrame:
    """Build a deterministic permno0 fixture exercising every code path in crsp_industry.

    Permnos:
        10001 — two non-overlapping spans (Jan 1-3 with sic=7372, naics=511210;
                Jan 6-8 with sic=7370, naics=511200).
        10002 — single short span (Feb 1-2) with sic=0 (should become null) and
                naics=None (should be preserved).
        10003 — two overlapping spans (Mar 1-4 and Mar 3-6, both same sic/naics)
                to exercise .unique(["permno", "date"]) dedup.
    """
    return permno0_frame(
        permnos=[10001, 10001, 10002, 10003, 10003],
        permcos=[1, 1, 2, 3, 3],
        starts=[
            date(2020, 1, 1),
            date(2020, 1, 6),
            date(2020, 2, 1),
            date(2020, 3, 1),
            date(2020, 3, 3),
        ],
        ends=[
            date(2020, 1, 3),
            date(2020, 1, 8),
            date(2020, 2, 2),
            date(2020, 3, 4),
            date(2020, 3, 6),
        ],
        sics=[7372, 7370, 0, 6020, 6020],
        naicses=[511210, 511200, None, 522110, 522110],
    )
