"""Shared builders for ``firm_age`` input fixtures.

Imported by both ``tests/golden/generate_firm_age_golden.py`` and
``tests/unit/test_firm_age.py`` so every input schema is defined exactly once.

``firm_age(paths, data_path)`` reads six physical parquets and derives three
earliest-date signals per firm, then ages each ``(id, eom)`` row:

* ``crsp_first``   = min ``mthcaldt`` per ``permco`` (from ``crsp_msf_v2_aug``).
* ``comp_ret_first`` = min ``datadate`` over ``comp_secm`` ∪
  ``comp_g_secd[monthend == 1]`` per ``gvkey``, then adjusted to the
  ``Dec-31`` of ``(min_datadate − 1 year).year``.
* ``comp_acc_first`` = min ``datadate`` over ``comp_funda`` ∪ ``comp_g_funda``
  per ``gvkey``, same ``Dec-31`` adjustment.

``first_obs = LEAST(crsp_first, comp_acc_first, comp_ret_first)`` (DuckDB
``LEAST`` ignores NULLs), ``first_alt = MIN(eom) OVER (PARTITION BY id)``, and
``age`` is the whole-month gap between ``eom`` and
``LEAST(first_obs, first_alt)``.

The six-entity synthetic set below drives the golden fixture; each entity
exercises a distinct branch (see ``build_*`` docstrings).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from jkp.data.paths import DataPaths

WORLD_MSF_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "permco": pl.Int64,
    "gvkey": pl.Utf8,
    "eom": pl.Date,
}

COMP_SECM_SCHEMA: dict[str, pl.DataType] = {
    "gvkey": pl.Utf8,
    "datadate": pl.Date,
}

COMP_G_SECD_SCHEMA: dict[str, pl.DataType] = {
    "gvkey": pl.Utf8,
    "datadate": pl.Date,
    "monthend": pl.Int64,
}

COMP_FUNDA_SCHEMA: dict[str, pl.DataType] = {
    "gvkey": pl.Utf8,
    "datadate": pl.Date,
}

COMP_G_FUNDA_SCHEMA: dict[str, pl.DataType] = {
    "gvkey": pl.Utf8,
    "datadate": pl.Date,
}

CRSP_MSF_V2_AUG_SCHEMA: dict[str, pl.DataType] = {
    "permco": pl.Int64,
    "mthcaldt": pl.Date,
}


def empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Return a typed 0-row frame for the given schema.

    Description:
        Build an empty, correctly-typed frame so ``firm_age``'s unconditional
        reads of all six inputs succeed even when a test supplies only some.
    Steps:
        1) Construct an empty DataFrame with the schema's columns and dtypes.
    Output:
        A 0-row ``pl.DataFrame`` matching ``schema``.
    """
    return pl.DataFrame(schema=schema)


def build_world_msf_input() -> pl.DataFrame:
    """Build the six-entity ``world_msf`` fixture (the ``data_path`` input).

    Description:
        One row per ``(id, eom)``; ``permco``/``gvkey`` are constant per id.
    Steps:
        1) A (10001) — CRSP + acc + ret present, CRSP earliest.
        2) B (10002) — all three present, comp_acc earliest, negative month-diff.
        3) C (10003) — null permco, comp_ret only (via comp_g_secd).
        4) D (10004) — CRSP only, age 0 at first eom.
        5) E (10005) — null permco, no comp match; first_obs null → fallback.
        6) F (10006) — comp_g_funda-only accounting wins.
    Output:
        ``pl.DataFrame`` with schema ``WORLD_MSF_SCHEMA``.
    """
    return pl.DataFrame(
        {
            "id": [10001, 10001, 10002, 10003, 10004, 10004, 10005, 10005, 10006],
            "permco": [1001, 1001, 1002, None, 1004, 1004, None, None, 1006],
            "gvkey": [
                "001000",
                "001000",
                "002000",
                "003000",
                "004000",
                "004000",
                "005000",
                "005000",
                "006000",
            ],
            "eom": [
                date(2015, 1, 31),
                date(2015, 2, 28),
                date(2015, 6, 30),
                date(2016, 9, 30),
                date(2018, 2, 28),
                date(2018, 3, 31),
                date(2019, 7, 31),
                date(2019, 8, 31),
                date(2015, 12, 31),
            ],
        },
        schema=WORLD_MSF_SCHEMA,
    )


def build_comp_secm_input() -> pl.DataFrame:
    """Build the US Compustat monthly-returns fixture.

    Description:
        Earliest ``datadate`` per ``gvkey`` feeds the comp_ret union.
    Steps:
        1) A 2012-03-31 (later than its comp_g_secd date — global wins).
        2) B 2012-01-31.
        3) F 2009-06-30.
    Output:
        ``pl.DataFrame`` with schema ``COMP_SECM_SCHEMA``.
    """
    return pl.DataFrame(
        {
            "gvkey": ["001000", "002000", "006000"],
            "datadate": [date(2012, 3, 31), date(2012, 1, 31), date(2009, 6, 30)],
        },
        schema=COMP_SECM_SCHEMA,
    )


def build_comp_g_secd_input() -> pl.DataFrame:
    """Build the global Compustat daily-returns fixture.

    Description:
        Only ``monthend == 1`` rows survive the pre-union filter.
    Steps:
        1) A 2011-01-31 monthend=1 (earlier than A's comp_secm date).
        2) C 2014-08-31 monthend=1 (C's only return signal).
        3) C 2012-05-15 monthend=0 — MUST be excluded; if it leaked it would
           make C's comp_ret_first earlier and change the age.
    Output:
        ``pl.DataFrame`` with schema ``COMP_G_SECD_SCHEMA``.
    """
    return pl.DataFrame(
        {
            "gvkey": ["001000", "003000", "003000"],
            "datadate": [date(2011, 1, 31), date(2014, 8, 31), date(2012, 5, 15)],
            "monthend": [1, 1, 0],
        },
        schema=COMP_G_SECD_SCHEMA,
    )


def build_comp_funda_input() -> pl.DataFrame:
    """Build the US Compustat annual-accounting fixture.

    Description:
        Earliest ``datadate`` per ``gvkey`` feeds the comp_acc union.
    Steps:
        1) A 2011-06-30.
        2) B 2009-06-30 (earliest of B's three signals → wins).
    Output:
        ``pl.DataFrame`` with schema ``COMP_FUNDA_SCHEMA``.
    """
    return pl.DataFrame(
        {
            "gvkey": ["001000", "002000"],
            "datadate": [date(2011, 6, 30), date(2009, 6, 30)],
        },
        schema=COMP_FUNDA_SCHEMA,
    )


def build_comp_g_funda_input() -> pl.DataFrame:
    """Build the global Compustat annual-accounting fixture.

    Description:
        Global-only accounting must contribute to the comp_acc union.
    Steps:
        1) F 2007-12-31 (F has no US funda row → this is F's acc signal).
    Output:
        ``pl.DataFrame`` with schema ``COMP_G_FUNDA_SCHEMA``.
    """
    return pl.DataFrame(
        {
            "gvkey": ["006000"],
            "datadate": [date(2007, 12, 31)],
        },
        schema=COMP_G_FUNDA_SCHEMA,
    )


def build_crsp_msf_v2_aug_input() -> pl.DataFrame:
    """Build the CRSP augmented monthly fixture.

    Description:
        ``crsp_first`` = min ``mthcaldt`` per ``permco``.
    Steps:
        1) A (1001) two months 2010-01-31, 2011-01-31 → min 2010-01-31.
        2) B (1002) 2013-05-31.
        3) D (1004) 2018-02-28 (D's only signal).
        4) F (1006) 2010-01-31.
    Output:
        ``pl.DataFrame`` with schema ``CRSP_MSF_V2_AUG_SCHEMA``.
    """
    return pl.DataFrame(
        {
            "permco": [1001, 1001, 1002, 1004, 1006],
            "mthcaldt": [
                date(2010, 1, 31),
                date(2011, 1, 31),
                date(2013, 5, 31),
                date(2018, 2, 28),
                date(2010, 1, 31),
            ],
        },
        schema=CRSP_MSF_V2_AUG_SCHEMA,
    )


def write_all_inputs(
    paths: DataPaths,
    *,
    world_msf: pl.DataFrame | None = None,
    comp_secm: pl.DataFrame | None = None,
    comp_g_secd: pl.DataFrame | None = None,
    comp_funda: pl.DataFrame | None = None,
    comp_g_funda: pl.DataFrame | None = None,
    crsp: pl.DataFrame | None = None,
) -> None:
    """Write all six ``firm_age`` inputs to the paths it reads.

    Description:
        ``firm_age`` reads all six parquets unconditionally, so every caller
        must materialize all six. Any argument left ``None`` defaults to a
        typed 0-row frame of the right schema.
    Steps:
        1) Coalesce each argument to a supplied frame or a typed empty frame.
        2) Ensure the interim, interim/raw_data_dfs, and raw_tables dirs exist.
        3) Write each frame to the exact path ``firm_age`` reads.
    Output:
        Six parquet files under ``paths`` (returns ``None``).
    """
    world_msf = world_msf if world_msf is not None else empty(WORLD_MSF_SCHEMA)
    comp_secm = comp_secm if comp_secm is not None else empty(COMP_SECM_SCHEMA)
    comp_g_secd = comp_g_secd if comp_g_secd is not None else empty(COMP_G_SECD_SCHEMA)
    comp_funda = comp_funda if comp_funda is not None else empty(COMP_FUNDA_SCHEMA)
    comp_g_funda = comp_g_funda if comp_g_funda is not None else empty(COMP_G_FUNDA_SCHEMA)
    crsp = crsp if crsp is not None else empty(CRSP_MSF_V2_AUG_SCHEMA)

    (paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
    paths.raw_tables_dir.mkdir(parents=True, exist_ok=True)

    world_msf.write_parquet(paths.interim_dir / "world_msf.parquet")
    comp_secm.write_parquet(paths.raw_tables_dir / "comp_secm.parquet")
    comp_g_secd.write_parquet(paths.raw_tables_dir / "comp_g_secd.parquet")
    comp_funda.write_parquet(paths.raw_tables_dir / "comp_funda.parquet")
    comp_g_funda.write_parquet(paths.raw_tables_dir / "comp_g_funda.parquet")
    crsp.write_parquet(paths.interim_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")


def write_full_inputs(paths: DataPaths) -> None:
    """Write the complete six-entity synthetic set used by the golden fixture.

    Description:
        Convenience wrapper writing every ``build_*_input()`` frame.
    Steps:
        1) Delegate to ``write_all_inputs`` with all six builders.
    Output:
        Six parquet files under ``paths`` (returns ``None``).
    """
    write_all_inputs(
        paths,
        world_msf=build_world_msf_input(),
        comp_secm=build_comp_secm_input(),
        comp_g_secd=build_comp_g_secd_input(),
        comp_funda=build_comp_funda_input(),
        comp_g_funda=build_comp_g_funda_input(),
        crsp=build_crsp_msf_v2_aug_input(),
    )
