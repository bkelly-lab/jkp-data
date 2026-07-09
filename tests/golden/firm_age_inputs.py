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

import numpy as np
import polars as pl

from jkp.data.paths import DataPaths

# Deterministic bulk panel that scales the golden fixture to ~0.5 MB.
# The six hand-crafted edge-case entities (ids 10001..10006) pin specific code
# paths; the bulk panel below (ids >= BULK_ID_START, permcos >= BULK_PERMCO_START,
# gvkeys "9xxxxx") is appended purely to grow the committed golden and never
# collides with any edge-case identifier. Every bulk entity carries a CRSP
# signal, so each bulk row produces a non-null, monotonically increasing age.
FIXED_SEED = 20240707
BULK_N_ENTITIES = 22_000
BULK_N_MONTHS = 180
BULK_ID_START = 100_000
BULK_PERMCO_START = 900_000

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


def _bulk_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw the per-entity bulk parameters from a single fixed-seed RNG.

    Description:
        Produce, for each of ``BULK_N_ENTITIES`` synthetic firms, its first-eom
        month index (months since year 0), a CRSP-lead offset (months by which
        the CRSP first date precedes that first eom), and the derived
        first-eom / crsp-first month indices — all deterministic given
        ``FIXED_SEED``.
    Steps:
        1) Seed ``numpy.random.default_rng(FIXED_SEED)``.
        2) Draw start year in [1980, 2016), start month in [1, 13), and CRSP
           lead offset in [0, 60) per entity.
        3) Convert (year, month) to a month index and derive the crsp index.
    Output:
        ``(entity_ids, base_month_idx, crsp_month_idx, lead_offset)`` int64
        arrays, each of length ``BULK_N_ENTITIES``.
    """
    rng = np.random.default_rng(FIXED_SEED)
    n = BULK_N_ENTITIES
    start_year = rng.integers(1980, 2016, n)
    start_month = rng.integers(1, 13, n)
    lead_offset = rng.integers(0, 60, n).astype(np.int64)
    base_month_idx = (start_year * 12 + (start_month - 1)).astype(np.int64)
    crsp_month_idx = base_month_idx - lead_offset
    entity_ids = np.arange(n, dtype=np.int64)
    return entity_ids, base_month_idx, crsp_month_idx, lead_offset


def _month_ends(month_idx: np.ndarray) -> pl.Series:
    """Convert year-0 month indices to month-end ``pl.Date`` values.

    Description:
        Map each integer month index (``year * 12 + (month - 1)``) to the last
        calendar day of that month, vectorised via numpy datetime64.
    Steps:
        1) Rebase the month index onto the 1970 numpy epoch.
        2) Take the first of that month, add one month, subtract one day.
    Output:
        A ``pl.Series`` of dtype ``pl.Date`` (month-end dates).
    """
    months_since_1970 = (month_idx - 1970 * 12).astype("timedelta64[M]")
    first_of_month = np.datetime64("1970-01", "M") + months_since_1970
    month_end = (first_of_month + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(
        1, "D"
    )
    return pl.Series(month_end).cast(pl.Date)


def build_bulk_world_msf_input() -> pl.DataFrame:
    """Build the large deterministic ``world_msf`` bulk panel.

    Description:
        ``BULK_N_ENTITIES`` firms, each with ``BULK_N_MONTHS`` consecutive
        month-end rows starting at its drawn first eom. Ids/permcos/gvkeys are
        in reserved high ranges so they never collide with edge-case entities.
    Steps:
        1) Draw per-entity parameters via ``_bulk_arrays``.
        2) Expand each entity to ``BULK_N_MONTHS`` consecutive month indices.
        3) Materialise ``id``, ``permco``, ``gvkey`` and month-end ``eom``.
    Output:
        ``pl.DataFrame`` with schema ``WORLD_MSF_SCHEMA``.
    """
    entity_ids, base_month_idx, _, _ = _bulk_arrays()
    m = BULK_N_MONTHS
    month_idx = (base_month_idx[:, None] + np.arange(m)).ravel()
    ids = np.repeat(entity_ids + BULK_ID_START, m)
    permco = np.repeat(entity_ids + BULK_PERMCO_START, m)
    gvkey = np.repeat(np.array([f"9{i:05d}" for i in entity_ids], dtype=object), m)
    return pl.DataFrame(
        {
            "id": pl.Series(ids, dtype=pl.Int64),
            "permco": pl.Series(permco, dtype=pl.Int64),
            "gvkey": pl.Series(gvkey, dtype=pl.Utf8),
            "eom": _month_ends(month_idx),
        },
        schema=WORLD_MSF_SCHEMA,
    )


def build_bulk_crsp_input() -> pl.DataFrame:
    """Build the CRSP bulk fixture matching ``build_bulk_world_msf_input``.

    Description:
        One CRSP month-end row per bulk permco, ``lead_offset`` months before
        that firm's first eom, so its age starts at ``lead_offset`` and
        increments by one each subsequent month.
    Steps:
        1) Draw the same per-entity parameters via ``_bulk_arrays``.
        2) Emit ``permco`` and the crsp-first month-end date per entity.
    Output:
        ``pl.DataFrame`` with schema ``CRSP_MSF_V2_AUG_SCHEMA``.
    """
    entity_ids, _, crsp_month_idx, _ = _bulk_arrays()
    return pl.DataFrame(
        {
            "permco": pl.Series(entity_ids + BULK_PERMCO_START, dtype=pl.Int64),
            "mthcaldt": _month_ends(crsp_month_idx),
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
    """Write the six edge-case entities plus the bulk panel for the golden.

    Description:
        The golden fixture is the union of the six hand-crafted edge-case
        entities (which pin distinct code paths) and the large deterministic
        bulk panel (which scales the committed fixture to ~1.5 MB). The bulk
        rows are appended to the ``world_msf`` and CRSP inputs only; the four
        Compustat inputs stay edge-case-only because bulk entities are
        CRSP-only.
    Steps:
        1) Concatenate edge-case + bulk ``world_msf`` and CRSP frames.
        2) Delegate to ``write_all_inputs`` with all six inputs.
    Output:
        Six parquet files under ``paths`` (returns ``None``).
    """
    world_msf = pl.concat([build_world_msf_input(), build_bulk_world_msf_input()])
    crsp = pl.concat([build_crsp_msf_v2_aug_input(), build_bulk_crsp_input()])
    write_all_inputs(
        paths,
        world_msf=world_msf,
        comp_secm=build_comp_secm_input(),
        comp_g_secd=build_comp_g_secd_input(),
        comp_funda=build_comp_funda_input(),
        comp_g_funda=build_comp_g_funda_input(),
        crsp=crsp,
    )
