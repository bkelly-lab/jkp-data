"""Acceptance tests for the synthetic Compustat inputs (issue #303, step 4).

The committed inputs in tests/golden/fixtures/synthetic_compustat/ must look like the real data
that tests/golden/compustat_profile.json describes:
    - the validator's hard rules hold, and its soft and cross-table rates are near the real ones;
    - each persona plants the case it exists for;
    - the stage's output is non-degenerate, and for the random firms each characteristic's null
      rate and positive share are near the real output's.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.golden.calibration.profile_compustat import output_stats
from tests.golden.compustat_invariants import check_invariants, hard_violations
from tests.golden.generate_acc_chars_golden import run_stage
from tests.golden.synthetic_compustat import OUT_DIR as INPUT_DIR
from tests.golden.synthetic_compustat import (
    PERSONA_GVKEY,
    PROFILE_PATH,
    generate,
    persona_specs,
)

pytestmark = pytest.mark.regression

PROFILE = json.loads(PROFILE_PATH.read_text())
# Our choice: a soft or cross-table rule's rate on the synthetic inputs is within RATE_BAND of its
# real rate. The personas plant rare cases, so rules that are near 0 in real data sit a little above.
RATE_BAND = 0.1
# From the plan (procedure 3): null rate and positive share within CHAR_BAND of the real output.
CHAR_BAND = 0.25
SIGN_ALLOWLIST = {
    # The AR(1) slope of earnings within a firm. The profile's autocorrelations pool firms, so a
    # firm effect plus fast noise matches them; within a firm that leaves slopes near 0 (and a
    # small-sample negative bias) where real slopes are mostly positive.
    "ni_ar1",
}
# Procedure 5: each characteristic has at least this many non-null values, a null, and this many
# distinct values. No characteristic needs an exception today.
MIN_VALUES, MIN_DISTINCT = 20, 3
DEGENERATE_ALLOWLIST: set[str] = set()
# The validator rule each persona plants (synthetic_compustat.py lists the personas).
PERSONA_RULES = {
    "900101": ("soft", "duplicate_fiscal_quarter", "comp_fundq"),
    "900102": ("soft", "null_fqtr", "comp_fundq"),
    "900103": ("soft", "skipped_quarter", "comp_fundq"),
    "900104": ("cross", "quarterly_before_first_annual", "quarterly"),
    "900105": ("cross", "fx_after_last_rate", "comp_g_funda"),
    "900106": ("cross", "fx_no_currency", "comp_funda"),
    "900108": ("soft", "two_calendars_one_date", "comp_fundq"),
    "900110": ("soft", "fs_on_two_calendars", "comp_g_fundq"),
    "900111": ("soft", "indl_and_fs_one_date", "comp_g_funda"),
    "900112": ("cross", "na_global_overlap", "annual"),
    "900114": ("cross", "fx_before_first_rate", "comp_g_funda"),
    "900115": ("cross", "annual_only_firms", "annual"),
    "900125": ("cross", "two_calendars_one_date", "quarterly"),
    "900126": ("soft", "two_calendars_one_date", "comp_g_fundq"),
    "900127": ("soft", "fyr_change_firms", "comp_fundq"),
}
# The personas whose case no validator rule names; each has its own test below.
PERSONA_TESTS = {
    "900107", "900109", "900113", "900116", "900117", "900118", "900119", "900120", "900121",
    "900122", "900123", "900124",
}  # fmt: skip


@pytest.fixture(scope="module")
def stage(tmp_path_factory: pytest.TempPathFactory) -> DataPaths:
    paths = DataPaths(tmp_path_factory.mktemp("acc_stage"))
    run_stage(paths)
    return paths


@pytest.fixture(scope="module")
def invariants(stage: DataPaths) -> pl.DataFrame:
    return check_invariants(stage)


@pytest.fixture(scope="module")
def chars(stage: DataPaths) -> pl.DataFrame:
    return pl.read_parquet(stage.interim_dir / "acc_chars_world.parquet")


def _violations(results: pl.DataFrame, rule: tuple[str, str, str]) -> int:
    kind, name, table = rule
    return results.filter(kind=kind, rule=name, table=table)["violations"].item()


def _invariants_without(paths: DataPaths, gvkey: str) -> pl.DataFrame:
    """The validator's results on the synthetic inputs with one firm removed."""
    af.setup_folder_structure(paths)
    for src in (INPUT_DIR / "raw" / "raw_tables").glob("*.parquet"):
        df = pl.read_parquet(src)
        if "gvkey" in df.columns:
            df = df.filter(pl.col("gvkey") != gvkey)
        df.write_parquet(paths.raw_tables_dir / src.name)
    pl.read_parquet(INPUT_DIR / "interim" / "world_msf.parquet").filter(
        pl.col("gvkey") != gvkey
    ).write_parquet(paths.interim_dir / "world_msf.parquet")
    af.gen_fx1(paths)
    return check_invariants(paths)


def _input(table: str, gvkey: str) -> pl.DataFrame:
    """One firm's rows of a committed input table."""
    folder = INPUT_DIR / "interim" if table == "world_msf" else INPUT_DIR / "raw" / "raw_tables"
    return pl.read_parquet(folder / f"{table}.parquet").filter(pl.col("gvkey") == gvkey)


def _gaps(df: pl.DataFrame) -> pl.Series:
    """Days between consecutive report dates."""
    return df["datadate"].unique().sort().diff().dt.total_days()


def test_committed_inputs_match_the_generator(tmp_path: Path) -> None:
    """The committed inputs are what the generator writes, so a generator change can't go in
    without regenerating them."""
    generate(tmp_path)
    committed = sorted(p.relative_to(INPUT_DIR) for p in INPUT_DIR.rglob("*.parquet"))
    assert committed == sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.parquet"))
    for path in committed:
        assert_frame_equal(
            pl.read_parquet(INPUT_DIR / path), pl.read_parquet(tmp_path / path), check_exact=True
        )


def test_every_persona_is_checked() -> None:
    assert {spec["gvkey"] for spec in persona_specs()} == set(PERSONA_RULES) | PERSONA_TESTS


def test_hard_rules_hold(invariants: pl.DataFrame) -> None:
    assert hard_violations(invariants).height == 0


def test_rule_rates_near_real(invariants: pl.DataFrame) -> None:
    real = pl.DataFrame(PROFILE["invariants"]).select(
        "kind", "rule", "table", real_rate=pl.col("rate")
    )
    rates = invariants.filter(pl.col("kind") != "hard").join(
        real, on=["kind", "rule", "table"], how="left"
    )
    assert rates["real_rate"].null_count() == 0, "a rule has no real rate in the profile"
    far = rates.filter((pl.col("rate") - pl.col("real_rate")).abs() > RATE_BAND)
    assert far.height == 0, far.select("kind", "rule", "table", "rate", "real_rate")


@pytest.mark.parametrize(("gvkey", "rule"), PERSONA_RULES.items())
def test_persona_plants_its_rule(
    gvkey: str, rule: tuple[str, str, str], invariants: pl.DataFrame, tmp_path: Path
) -> None:
    """The persona's rule fires, and removing the persona lowers its count."""
    with_persona = _violations(invariants, rule)
    assert with_persona > 0
    assert _violations(_invariants_without(DataPaths(tmp_path), gvkey), rule) < with_persona


def test_constant_earnings_persona_has_null_persistence(chars: pl.DataFrame) -> None:
    """900121 reports constant earnings and assets for 16 years, so ni_ar1 is always null."""
    firm = chars.filter(gvkey="900121")
    assert firm.height > 0 and firm["ni_ar1"].is_null().all()
    assert chars.filter(pl.col("gvkey").cast(pl.Int64) < PERSONA_GVKEY)["ni_ar1"].count() > 0


def test_cad_persona_reports_before_first_cad_rate(chars: pl.DataFrame) -> None:
    """900107 reports in CAD from 1976; the OECD backfill gives those reports a rate."""
    early = chars.filter(gvkey="900107").filter(pl.col("public_date").dt.year() < 1982)
    assert early.height > 0 and early["assets"].count() > 0


def test_resumed_calendar_persona() -> None:
    """900109's December calendar stops for more than 200 days and resumes, while its newer June
    calendar reports in the gap."""
    quarterly = _input("comp_fundq", "900109")
    december = quarterly.filter(pl.col("fyr") == 12)["datadate"].sort()
    gaps = december.diff().dt.total_days()
    i = gaps.arg_max()
    assert gaps[i] > 200
    in_gap = pl.col("datadate").is_between(december[i - 1], december[i], closed="none")
    assert quarterly.filter((pl.col("fyr") == 6) & in_gap).height > 0


def test_jpy_persona_has_a_march_year_end_and_a_second_security() -> None:
    """900113 reports in JPY on a March year-end, and world_msf has a non-primary security in the
    months of its primary one."""
    annual = _input("comp_g_funda", "900113")
    assert annual.height > 0
    assert set(annual["curcd"]) == {"JPY"} and set(annual["fyr"]) == {3}
    msf = _input("world_msf", "900113")
    primary = msf.filter(pl.col("primary_sec") == 1)
    assert msf.filter(pl.col("primary_sec") == 0).join(primary, on="eom").height > 0


def test_late_quarterly_persona() -> None:
    """900116's quarterly reports run three years past its last annual report."""
    last_annual = _input("comp_funda", "900116")["datadate"].max()
    last_quarter = _input("comp_fundq", "900116")["datadate"].max()
    assert last_quarter.year - last_annual.year >= 3


def test_zero_sales_negative_equity_persona() -> None:
    """900117 has years with zero sales, with negative book equity, and with zero and with
    negative total assets."""
    annual = _input("comp_funda", "900117").select(pl.col("sale", "seq", "at").cast(pl.Float64))
    assert (annual["sale"] == 0).any() and (annual["seq"] < 0).any()
    assert (annual["at"] == 0).any() and (annual["at"] < 0).any()


@pytest.mark.parametrize(("gvkey", "years"), [("900118", 3), ("900119", 6), ("900120", 12)])
def test_short_history_personas(gvkey: str, years: int) -> None:
    """900118, 900119 and 900120 have 3, 6 and 12 fiscal years of annual reports."""
    assert _input("comp_funda", gvkey)["fyear"].n_unique() == years


def test_reporting_gap_persona() -> None:
    """900122 has no annual or quarterly report for three years in the middle of its life."""
    for table in ["comp_funda", "comp_fundq"]:
        assert _gaps(_input(table, "900122")).max() > 3 * 365, table


@pytest.mark.parametrize(
    ("gvkey", "tables", "dropped"),
    [
        (
            "900123",
            ["comp_funda", "comp_fundq"],
            {"datafmt": "SUMM_STD", "indfmt": "FS", "consol": "P", "popsrc": "I"},
        ),
        (
            "900124",
            ["comp_g_funda", "comp_g_fundq"],
            {"datafmt": "STD", "consol": "P", "popsrc": "D"},
        ),
    ],
)
def test_filtered_row_personas(gvkey: str, tables: list[str], dropped: dict[str, str]) -> None:
    """900123 (NA) and 900124 (Global) have rows with each value the read filters drop."""
    for table in tables:
        df = _input(table, gvkey)
        for column, value in dropped.items():
            assert (df[column] == value).any(), (table, column, value)


def test_output_is_non_degenerate(chars: pl.DataFrame) -> None:
    """Procedure 5: every characteristic has non-null values, a null, and distinct values."""
    names = [c for c in af.acc_chars_list() if c not in DEGENERATE_ALLOWLIST]
    finite = {c: pl.col(c).cast(pl.Float64).fill_nan(None) for c in names}
    stats = chars.select(
        *(finite[c].count().alias(f"{c}|values") for c in names),
        *(finite[c].null_count().alias(f"{c}|nulls") for c in names),
        *(finite[c].drop_nulls().n_unique().alias(f"{c}|distinct") for c in names),
    ).row(0, named=True)
    bad = [
        c
        for c in names
        if stats[f"{c}|values"] < MIN_VALUES
        or stats[f"{c}|nulls"] == 0
        or stats[f"{c}|distinct"] < MIN_DISTINCT
    ]
    assert not bad, bad


def test_random_firms_match_real_output(chars: pl.DataFrame, tmp_path: Path) -> None:
    """Procedure 3, on the random firms (the personas are unusual on purpose)."""
    path = tmp_path / "random.parquet"
    chars.filter(pl.col("gvkey").cast(pl.Int64) < PERSONA_GVKEY).write_parquet(path)
    synthetic = output_stats(path)["chars"]
    real = PROFILE["outputs"]["chars"]
    assert set(synthetic) == set(real)
    null_far = {
        c: (round(s["null"] + s["nan"], 3), round(real[c]["null"] + real[c]["nan"], 3))
        for c, s in synthetic.items()
        if abs(s["null"] + s["nan"] - real[c]["null"] - real[c]["nan"]) > CHAR_BAND
    }
    sign_far = {
        c: (round(s["pos"], 3), round(real[c]["pos"], 3))
        for c, s in synthetic.items()
        if c not in SIGN_ALLOWLIST and abs(s["pos"] - real[c]["pos"]) > CHAR_BAND
    }
    assert not null_far, null_far
    assert not sign_far, sign_far
