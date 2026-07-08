"""Unit tests for firm_age() — semantic behaviour and edge cases.

Each test writes minimal (all six) synthetic inputs via ``write_all_inputs``,
runs the real ``firm_age`` on a ``test_paths`` DataPaths, and asserts the exact
``age`` (Int64 whole months) per ``(id, eom)``. Ages are exact integers, so
these use plain equality — never float tolerance.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from jkp.data.aux_functions import firm_age
from jkp.data.paths import DataPaths
from tests.golden.firm_age_inputs import (
    COMP_FUNDA_SCHEMA,
    COMP_G_FUNDA_SCHEMA,
    COMP_G_SECD_SCHEMA,
    COMP_SECM_SCHEMA,
    CRSP_MSF_V2_AUG_SCHEMA,
    WORLD_MSF_SCHEMA,
    build_comp_funda_input,
    build_comp_g_funda_input,
    build_comp_g_secd_input,
    build_comp_secm_input,
    build_crsp_msf_v2_aug_input,
    build_world_msf_input,
    write_all_inputs,
)


def _run(paths: DataPaths, **inputs: pl.DataFrame) -> dict[tuple[int, date], int]:
    """Run firm_age and return an ``{(id, eom): age}`` map.

    Description:
        Write the supplied (and defaulted-empty) inputs, run ``firm_age``, and
        collect its output into a lookup keyed by ``(id, eom)``.
    Steps:
        1) Write all six inputs (missing ones default to typed 0-row frames).
        2) Run ``firm_age`` against the written ``world_msf``.
        3) Read the output parquet and build the ``(id, eom) -> age`` dict.
    Output:
        Dict mapping each ``(id, eom)`` to its computed ``age``.
    """
    write_all_inputs(paths, **inputs)
    firm_age(paths, paths.interim_dir / "world_msf.parquet")
    out = pl.read_parquet(paths.interim_dir / "firm_age.parquet")
    return {(r["id"], r["eom"]): r["age"] for r in out.iter_rows(named=True)}


def _world(id_: int, permco: int | None, gvkey: str, eoms: list[date]) -> pl.DataFrame:
    """Build a single-firm ``world_msf`` frame with one row per ``eom``."""
    return pl.DataFrame(
        {
            "id": [id_] * len(eoms),
            "permco": [permco] * len(eoms),
            "gvkey": [gvkey] * len(eoms),
            "eom": eoms,
        },
        schema=WORLD_MSF_SCHEMA,
    )


def test_all_three_sources_crsp_earliest_wins(test_paths: DataPaths) -> None:
    """Entity A: CRSP is earliest of the three signals; comp_ret union spans US+global."""
    ages = _run(
        test_paths,
        world_msf=_world(10001, 1001, "001000", [date(2015, 1, 31), date(2015, 2, 28)]),
        crsp=pl.DataFrame(
            {"permco": [1001, 1001], "mthcaldt": [date(2010, 1, 31), date(2011, 1, 31)]},
            schema=CRSP_MSF_V2_AUG_SCHEMA,
        ),
        comp_funda=pl.DataFrame(
            {"gvkey": ["001000"], "datadate": [date(2011, 6, 30)]}, schema=COMP_FUNDA_SCHEMA
        ),
        comp_secm=pl.DataFrame(
            {"gvkey": ["001000"], "datadate": [date(2012, 3, 31)]}, schema=COMP_SECM_SCHEMA
        ),
        comp_g_secd=pl.DataFrame(
            {"gvkey": ["001000"], "datadate": [date(2011, 1, 31)], "monthend": [1]},
            schema=COMP_G_SECD_SCHEMA,
        ),
    )
    assert ages[(10001, date(2015, 1, 31))] == 60
    assert ages[(10001, date(2015, 2, 28))] == 61


def test_comp_acc_wins_with_negative_month_diff(test_paths: DataPaths) -> None:
    """Entity B: comp_acc is earliest; eom month < first-obs month gives a negative month term."""
    ages = _run(
        test_paths,
        world_msf=_world(10002, 1002, "002000", [date(2015, 6, 30)]),
        crsp=pl.DataFrame(
            {"permco": [1002], "mthcaldt": [date(2013, 5, 31)]}, schema=CRSP_MSF_V2_AUG_SCHEMA
        ),
        comp_funda=pl.DataFrame(
            {"gvkey": ["002000"], "datadate": [date(2009, 6, 30)]}, schema=COMP_FUNDA_SCHEMA
        ),
        comp_secm=pl.DataFrame(
            {"gvkey": ["002000"], "datadate": [date(2012, 1, 31)]}, schema=COMP_SECM_SCHEMA
        ),
    )
    # first_obs = 2008-12-31; (2015-2008)*12 + (6-12) = 78
    assert ages[(10002, date(2015, 6, 30))] == 78


def test_comp_ret_only_monthend_filter_is_load_bearing(test_paths: DataPaths) -> None:
    """Entity C: comp_g_secd monthend=0 row must be dropped; flipping it changes the age."""
    world = _world(10003, None, "003000", [date(2016, 9, 30)])

    filtered = _run(
        test_paths,
        world_msf=world,
        comp_g_secd=pl.DataFrame(
            {
                "gvkey": ["003000", "003000"],
                "datadate": [date(2014, 8, 31), date(2012, 5, 15)],
                "monthend": [1, 0],
            },
            schema=COMP_G_SECD_SCHEMA,
        ),
    )
    # monthend=0 dropped: first_obs = 2013-12-31; (2016-2013)*12 + (9-12) = 33
    assert filtered[(10003, date(2016, 9, 30))] == 33

    leaked = _run(
        test_paths,
        world_msf=world,
        comp_g_secd=pl.DataFrame(
            {
                "gvkey": ["003000", "003000"],
                "datadate": [date(2014, 8, 31), date(2012, 5, 15)],
                "monthend": [1, 1],
            },
            schema=COMP_G_SECD_SCHEMA,
        ),
    )
    # both kept: min 2012-05-15 -> first_obs 2011-12-31; (2016-2011)*12 + (9-12) = 57
    assert leaked[(10003, date(2016, 9, 30))] == 57


def test_crsp_only_age_zero_at_first_month(test_paths: DataPaths) -> None:
    """Entity D: CRSP-only firm; age is 0 at the first eom and increments monthly."""
    ages = _run(
        test_paths,
        world_msf=_world(10004, 1004, "004000", [date(2018, 2, 28), date(2018, 3, 31)]),
        crsp=pl.DataFrame(
            {"permco": [1004], "mthcaldt": [date(2018, 2, 28)]}, schema=CRSP_MSF_V2_AUG_SCHEMA
        ),
    )
    assert ages[(10004, date(2018, 2, 28))] == 0
    assert ages[(10004, date(2018, 3, 31))] == 1


def test_no_source_falls_back_to_first_alt(test_paths: DataPaths) -> None:
    """Entity E: no signal matches; first_obs is null so age uses MIN(eom) per id."""
    ages = _run(
        test_paths,
        world_msf=_world(10005, None, "005000", [date(2019, 7, 31), date(2019, 8, 31)]),
    )
    assert ages[(10005, date(2019, 7, 31))] == 0
    assert ages[(10005, date(2019, 8, 31))] == 1


def test_comp_g_funda_contributes_to_acc_union(test_paths: DataPaths) -> None:
    """Entity F: global-only accounting (comp_g_funda) wins the acc union."""
    ages = _run(
        test_paths,
        world_msf=_world(10006, 1006, "006000", [date(2015, 12, 31)]),
        crsp=pl.DataFrame(
            {"permco": [1006], "mthcaldt": [date(2010, 1, 31)]}, schema=CRSP_MSF_V2_AUG_SCHEMA
        ),
        comp_secm=pl.DataFrame(
            {"gvkey": ["006000"], "datadate": [date(2009, 6, 30)]}, schema=COMP_SECM_SCHEMA
        ),
        comp_g_funda=pl.DataFrame(
            {"gvkey": ["006000"], "datadate": [date(2007, 12, 31)]}, schema=COMP_G_FUNDA_SCHEMA
        ),
    )
    # comp_acc_first = 2006-12-31 wins; (2015-2006)*12 + (12-12) = 108
    assert ages[(10006, date(2015, 12, 31))] == 108


def test_null_permco_does_not_match_crsp(test_paths: DataPaths) -> None:
    """A null-permco Compustat firm yields null crsp_first but still ages from its comp signal."""
    ages = _run(
        test_paths,
        world_msf=_world(20001, None, "099000", [date(2016, 1, 31)]),
        # a CRSP row exists but the firm's permco is null -> no join match
        crsp=pl.DataFrame(
            {"permco": [1001], "mthcaldt": [date(2005, 1, 31)]}, schema=CRSP_MSF_V2_AUG_SCHEMA
        ),
        comp_funda=pl.DataFrame(
            {"gvkey": ["099000"], "datadate": [date(2011, 6, 30)]}, schema=COMP_FUNDA_SCHEMA
        ),
    )
    # crsp_first null; comp_acc_first = 2010-12-31; (2016-2010)*12 + (1-12) = 61
    assert ages[(20001, date(2016, 1, 31))] == 61


def test_comp_ret_union_min_across_files_is_symmetric(test_paths: DataPaths) -> None:
    """comp_ret_first is the earlier of comp_secm and comp_g_secd regardless of which file holds it."""
    world = _world(30001, 3001, "030000", [date(2016, 1, 31)])

    global_earlier = _run(
        test_paths,
        world_msf=world,
        comp_secm=pl.DataFrame(
            {"gvkey": ["030000"], "datadate": [date(2012, 6, 30)]}, schema=COMP_SECM_SCHEMA
        ),
        comp_g_secd=pl.DataFrame(
            {"gvkey": ["030000"], "datadate": [date(2010, 3, 31)], "monthend": [1]},
            schema=COMP_G_SECD_SCHEMA,
        ),
    )
    us_earlier = _run(
        test_paths,
        world_msf=world,
        comp_secm=pl.DataFrame(
            {"gvkey": ["030000"], "datadate": [date(2010, 3, 31)]}, schema=COMP_SECM_SCHEMA
        ),
        comp_g_secd=pl.DataFrame(
            {"gvkey": ["030000"], "datadate": [date(2012, 6, 30)], "monthend": [1]},
            schema=COMP_G_SECD_SCHEMA,
        ),
    )
    # both: union min 2010-03-31 -> comp_ret_first 2009-12-31; (2016-2009)*12 + (1-12) = 73
    assert global_earlier[(30001, date(2016, 1, 31))] == 73
    assert us_earlier[(30001, date(2016, 1, 31))] == 73


def test_empty_world_msf_yields_empty_typed_output(test_paths: DataPaths) -> None:
    """Empty world_msf produces a 0-row output with the [id, eom, age] schema."""
    write_all_inputs(test_paths)  # all six empty
    firm_age(test_paths, test_paths.interim_dir / "world_msf.parquet")
    out = pl.read_parquet(test_paths.interim_dir / "firm_age.parquet")
    assert out.height == 0
    assert out.columns == ["id", "eom", "age"]
    assert out.schema["id"] == pl.Int64
    assert out.schema["eom"] == pl.Date
    assert out.schema["age"] == pl.Int64


def test_first_alt_is_per_id_not_global(test_paths: DataPaths) -> None:
    """first_alt = MIN(eom) PARTITION BY id: each id ages from its own earliest eom."""
    world = pl.concat(
        [
            _world(40001, None, "040000", [date(2015, 3, 31), date(2015, 4, 30)]),
            _world(40002, None, "040001", [date(2020, 9, 30), date(2020, 10, 31)]),
        ]
    )
    ages = _run(test_paths, world_msf=world)  # no signals -> pure first_alt fallback
    # each id: age 0 at its own min eom, 1 at the next month
    assert ages[(40001, date(2015, 3, 31))] == 0
    assert ages[(40001, date(2015, 4, 30))] == 1
    assert ages[(40002, date(2020, 9, 30))] == 0
    assert ages[(40002, date(2020, 10, 31))] == 1


def test_comp_secm_alone_is_the_minimum_source(test_paths: DataPaths) -> None:
    """Per-source: comp_secm is the sole (and therefore minimum) signal.

    Only the US Compustat monthly-returns file contributes; crsp and both
    accounting sources are null, so LEAST reduces to comp_ret_first alone.
    """
    ages = _run(
        test_paths,
        world_msf=_world(50101, None, "051000", [date(2016, 1, 31)]),
        comp_secm=pl.DataFrame(
            {"gvkey": ["051000"], "datadate": [date(2010, 3, 31)]}, schema=COMP_SECM_SCHEMA
        ),
    )
    # comp_ret_first = 2009-12-31; (2016-2009)*12 + (1-12) = 73
    assert ages[(50101, date(2016, 1, 31))] == 73


def test_first_alt_caps_later_non_null_first_obs(test_paths: DataPaths) -> None:
    """LEAST(first_obs, first_alt) picks first_alt when a non-null first_obs is later.

    The CRSP signal (2015-06-30) post-dates the firm's earliest eom, so the
    per-id MIN(eom) caps it. Without the cap the age would be negative.
    """
    ages = _run(
        test_paths,
        world_msf=_world(50201, 5021, "052000", [date(2015, 1, 31), date(2015, 2, 28)]),
        crsp=pl.DataFrame(
            {"permco": [5021], "mthcaldt": [date(2015, 6, 30)]}, schema=CRSP_MSF_V2_AUG_SCHEMA
        ),
    )
    # first_obs=2015-06-30 (later) is capped by first_alt=2015-01-31.
    assert ages[(50201, date(2015, 1, 31))] == 0
    assert ages[(50201, date(2015, 2, 28))] == 1


def test_first_obs_equal_to_first_alt_tie(test_paths: DataPaths) -> None:
    """When first_obs exactly equals first_alt, LEAST is that shared date; age 0 at it."""
    ages = _run(
        test_paths,
        world_msf=_world(50301, 5031, "053000", [date(2015, 1, 31), date(2015, 2, 28)]),
        crsp=pl.DataFrame(
            {"permco": [5031], "mthcaldt": [date(2015, 1, 31)]}, schema=CRSP_MSF_V2_AUG_SCHEMA
        ),
    )
    # first_obs == first_alt == 2015-01-31.
    assert ages[(50301, date(2015, 1, 31))] == 0
    assert ages[(50301, date(2015, 2, 28))] == 1


def test_comp_acc_union_min_across_files_is_symmetric(test_paths: DataPaths) -> None:
    """comp_acc_first is the earlier of comp_funda and comp_g_funda regardless of file."""
    world = _world(50401, None, "054000", [date(2016, 1, 31)])

    us_earlier = _run(
        test_paths,
        world_msf=world,
        comp_funda=pl.DataFrame(
            {"gvkey": ["054000"], "datadate": [date(2010, 3, 31)]}, schema=COMP_FUNDA_SCHEMA
        ),
        comp_g_funda=pl.DataFrame(
            {"gvkey": ["054000"], "datadate": [date(2012, 6, 30)]}, schema=COMP_G_FUNDA_SCHEMA
        ),
    )
    global_earlier = _run(
        test_paths,
        world_msf=world,
        comp_funda=pl.DataFrame(
            {"gvkey": ["054000"], "datadate": [date(2012, 6, 30)]}, schema=COMP_FUNDA_SCHEMA
        ),
        comp_g_funda=pl.DataFrame(
            {"gvkey": ["054000"], "datadate": [date(2010, 3, 31)]}, schema=COMP_G_FUNDA_SCHEMA
        ),
    )
    # both: union min 2010-03-31 -> comp_acc_first 2009-12-31; (2016-2009)*12 + (1-12) = 73
    assert us_earlier[(50401, date(2016, 1, 31))] == 73
    assert global_earlier[(50401, date(2016, 1, 31))] == 73


def test_crsp_first_is_per_permco_not_global(test_paths: DataPaths) -> None:
    """crsp_first = MIN(mthcaldt) GROUP BY permco: each permco ages from its own min."""
    world = pl.concat(
        [
            _world(50501, 5051, "055000", [date(2015, 1, 31)]),
            _world(50502, 5052, "055001", [date(2020, 1, 31)]),
        ]
    )
    ages = _run(
        test_paths,
        world_msf=world,
        crsp=pl.DataFrame(
            {"permco": [5051, 5052], "mthcaldt": [date(2010, 1, 31), date(2018, 1, 31)]},
            schema=CRSP_MSF_V2_AUG_SCHEMA,
        ),
    )
    # permco 5051 -> 2010-01-31: (2015-2010)*12 = 60
    # permco 5052 -> 2018-01-31: (2020-2018)*12 = 24 (not 120, which a global min would give)
    assert ages[(50501, date(2015, 1, 31))] == 60
    assert ages[(50502, date(2020, 1, 31))] == 24


def test_accounting_adjustment_snaps_to_prior_year_dec31(test_paths: DataPaths) -> None:
    """comp_acc: any datadate in year Y snaps to (Y-1)-12-31; the year boundary is sharp."""
    world_a = _world(50601, None, "056000", [date(2015, 6, 30)])
    world_b = _world(50602, None, "056001", [date(2015, 6, 30)])
    world_c = _world(50603, None, "056002", [date(2015, 6, 30)])

    a = _run(
        test_paths,
        world_msf=world_a,
        comp_funda=pl.DataFrame(
            {"gvkey": ["056000"], "datadate": [date(2010, 3, 31)]}, schema=COMP_FUNDA_SCHEMA
        ),
    )
    b = _run(
        test_paths,
        world_msf=world_b,
        comp_funda=pl.DataFrame(
            {"gvkey": ["056001"], "datadate": [date(2010, 9, 30)]}, schema=COMP_FUNDA_SCHEMA
        ),
    )
    c = _run(
        test_paths,
        world_msf=world_c,
        comp_funda=pl.DataFrame(
            {"gvkey": ["056002"], "datadate": [date(2011, 1, 31)]}, schema=COMP_FUNDA_SCHEMA
        ),
    )
    # 2010-03-31 and 2010-09-30 both snap to 2009-12-31: (2015-2009)*12 + (6-12) = 66
    assert a[(50601, date(2015, 6, 30))] == 66
    assert b[(50602, date(2015, 6, 30))] == 66
    # 2011-01-31 snaps to 2010-12-31 (next year): (2015-2010)*12 + (6-12) = 54
    assert c[(50603, date(2015, 6, 30))] == 54


def test_comp_ret_adjustment_snaps_to_prior_year_dec31(test_paths: DataPaths) -> None:
    """comp_ret path applies the same (Y-1)-12-31 snap as the accounting path."""
    ages = _run(
        test_paths,
        world_msf=_world(50701, None, "057000", [date(2015, 6, 30)]),
        comp_secm=pl.DataFrame(
            {"gvkey": ["057000"], "datadate": [date(2010, 3, 31)]}, schema=COMP_SECM_SCHEMA
        ),
    )
    # 2010-03-31 -> comp_ret_first 2009-12-31; (2015-2009)*12 + (6-12) = 66
    assert ages[(50701, date(2015, 6, 30))] == 66


def test_full_six_entity_set_matches_expected(test_paths: DataPaths) -> None:
    """Sanity check the full shared builder set against the documented ages."""
    ages = _run(
        test_paths,
        world_msf=build_world_msf_input(),
        comp_secm=build_comp_secm_input(),
        comp_g_secd=build_comp_g_secd_input(),
        comp_funda=build_comp_funda_input(),
        comp_g_funda=build_comp_g_funda_input(),
        crsp=build_crsp_msf_v2_aug_input(),
    )
    assert ages == {
        (10001, date(2015, 1, 31)): 60,
        (10001, date(2015, 2, 28)): 61,
        (10002, date(2015, 6, 30)): 78,
        (10003, date(2016, 9, 30)): 33,
        (10004, date(2018, 2, 28)): 0,
        (10004, date(2018, 3, 31)): 1,
        (10005, date(2019, 7, 31)): 0,
        (10005, date(2019, 8, 31)): 1,
        (10006, date(2015, 12, 31)): 108,
    }
