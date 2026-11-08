"""Tests for the Compustat input rules in ``tests/golden/compustat_invariants.py``."""

import copy
import json
from collections.abc import Callable
from datetime import date

import polars as pl
import pytest

from jkp.data.paths import DataPaths
from tests.golden.calibration.profile_compustat import profile
from tests.golden.compustat_invariants import TABLES, check_invariants, hard_violations

READ_FILTERS = {
    "NA": {"datafmt": "STD", "popsrc": "D", "consol": "C"},
    "GLOBAL": {"datafmt": "HIST_STD", "popsrc": "I", "consol": "C"},
}
ANNUAL = {
    "gvkey": pl.String,
    "datadate": pl.Date,
    "indfmt": pl.String,
    "curcd": pl.String,
    "fyr": pl.Int16,
}
QUARTERLY = {
    "gvkey": pl.String,
    "datadate": pl.Date,
    "indfmt": pl.String,
    "curcdq": pl.String,
    "fyr": pl.Int16,
    "fyearq": pl.Int32,
    "fqtr": pl.Int16,
}
QUARTER_ENDS = [date(2011, 3, 31), date(2011, 6, 30), date(2011, 9, 30), date(2011, 12, 31)]


def _clean() -> dict[str, list[tuple]]:
    """Two firms per source, 2010-2011, every rule at 0: NA firm 001 in USD, Global firm 101 in EUR."""
    years = [date(2010, 12, 31), date(2011, 12, 31)]
    month_ends = pl.date_range(
        date(2010, 1, 1), date(2012, 12, 1), "1mo", eager=True
    ).dt.month_end()
    return {
        "comp_funda": [("001", d, "INDL", "USD", 12) for d in years],
        "comp_g_funda": [("101", d, "INDL", "EUR", 12) for d in years],
        "comp_fundq": [
            ("001", d, "INDL", "USD", 12, 2011, q) for q, d in enumerate(QUARTER_ENDS, 1)
        ],
        "comp_g_fundq": [
            ("101", d, "INDL", "EUR", 12, 2011, q) for q, d in enumerate(QUARTER_ENDS, 1)
        ],
        "fx1": [
            ("USD", date(2010, 1, 1), 1.0),
            ("USD", date(2012, 12, 31), 1.0),
            ("EUR", date(2010, 1, 1), 1.3),
            ("EUR", date(2012, 12, 31), 1.2),
        ],
        "msf": [(g, d) for g in ["001", "101"] for d in month_ends],
    }


def _run(paths: DataPaths, data: dict[str, list[tuple]]) -> pl.DataFrame:
    """Write the tables where the stage reads them and run every rule."""
    for table, schema, source, at in [
        ("comp_funda", ANNUAL, "NA", "at"),
        ("comp_g_funda", ANNUAL, "GLOBAL", "at"),
        ("comp_fundq", QUARTERLY, "NA", "atq"),
        ("comp_g_fundq", QUARTERLY, "GLOBAL", "atq"),
    ]:
        pl.DataFrame(data[table], schema=schema, orient="row").with_columns(
            **{k: pl.lit(v) for k, v in READ_FILTERS[source].items()},
            **{at: pl.lit(100.0), at.replace("at", "ib"): pl.lit(10.0)},
        ).write_parquet(paths.raw_tables_dir / f"{table}.parquet")
    pl.DataFrame(
        data["fx1"],
        schema={"curcdd": pl.String, "datadate": pl.Date, "fx": pl.Float64},
        orient="row",
    ).write_parquet(paths.interim_dir / "raw_data_dfs" / "__fx1.parquet")
    pl.DataFrame(
        data["msf"], schema={"gvkey": pl.String, "eom": pl.Date}, orient="row"
    ).with_columns(
        primary_sec=pl.lit(1), common=pl.lit(1), obs_main=pl.lit(1), me_company=pl.lit(50.0)
    ).write_parquet(paths.interim_dir / "world_msf.parquet")
    return check_invariants(paths)


def _violations(results: pl.DataFrame, kind: str, rule: str, table: str) -> int:
    return results.filter(kind=kind, rule=rule, table=table)["violations"].item()


def _add(table: str, *rows: tuple) -> Callable[[dict], None]:
    return lambda data: data[table].extend(rows)


def _drop(table: str, keep: Callable[[tuple], bool]) -> Callable[[dict], None]:
    return lambda data: data.update({table: [r for r in data[table] if keep(r)]})


CASES = [
    # (change to the clean data, kind, rule, table, violations)
    (_add("comp_funda", ("001", date(2009, 6, 15), "INDL", "USD", 12)), "hard", "month_end_datadate", "comp_funda", 1),
    (_add("comp_funda", ("001", date(2011, 12, 31), "INDL", "USD", 12)), "hard", "unique_key", "comp_funda", 1),
    (_add("comp_fundq", ("001", date(2012, 3, 31), "INDL", "USD", 12, 2012, 5)), "hard", "fiscal_keys", "comp_fundq", 1),
    (_add("fx1", ("GBP", date(2011, 1, 1), 0.0)), "hard", "fx_positive", "__fx1", 1),
    (_add("comp_fundq", ("001", date(2010, 12, 31), "INDL", "USD", 12, 2010, None)), "soft", "null_fqtr", "comp_fundq", 1),
    # The leftover and the regular Q4 share one fiscal quarter: both rows are flagged.
    (_add("comp_fundq", ("001", date(2012, 1, 31), "INDL", "USD", 12, 2011, 4)), "soft", "duplicate_fiscal_quarter", "comp_fundq", 2),
    (_drop("comp_fundq", lambda r: r[6] != 2), "soft", "skipped_quarter", "comp_fundq", 1),
    (_add("comp_fundq", ("001", date(2012, 6, 30), "INDL", "USD", 6, 2012, 4)), "soft", "fyr_change_firms", "comp_fundq", 1),
    (_add("comp_g_fundq", ("101", date(2011, 12, 31), "INDL", "EUR", 6, 2012, 2)), "soft", "two_calendars_one_date", "comp_g_fundq", 1),
    (_add("comp_g_fundq", ("101", date(2011, 12, 31), "INDL", "EUR", 6, 2012, 2)), "cross", "two_calendars_one_date", "quarterly", 1),
    (_add("comp_g_funda", ("101", date(2011, 12, 31), "FS", "EUR", 12)), "soft", "indl_and_fs_one_date", "comp_g_funda", 1),
    (
        _add("comp_g_fundq", ("102", date(2011, 12, 31), "FS", "EUR", 12, 2011, 4), ("102", date(2011, 12, 31), "FS", "EUR", 6, 2012, 2)),
        "soft", "fs_on_two_calendars", "comp_g_fundq", 1,
    ),
    (_add("comp_funda", ("001", date(2009, 12, 31), "INDL", None, 12)), "cross", "fx_no_currency", "comp_funda", 1),
    (_add("comp_g_funda", ("101", date(2009, 12, 31), "INDL", "ZWG", 12)), "cross", "fx_currency_without_rates", "comp_g_funda", 1),
    (_add("comp_g_funda", ("101", date(2009, 12, 31), "INDL", "EUR", 12)), "cross", "fx_before_first_rate", "comp_g_funda", 1),
    (_add("comp_g_funda", ("101", date(2013, 12, 31), "INDL", "EUR", 12)), "cross", "fx_after_last_rate", "comp_g_funda", 1),
    (_add("comp_g_funda", ("001", date(2011, 12, 31), "INDL", "USD", 12)), "cross", "na_global_overlap", "annual", 1),
    (_add("comp_g_fundq", ("001", date(2011, 12, 31), "INDL", "USD", 12, 2011, 4)), "cross", "na_global_overlap", "quarterly", 1),
    (_add("comp_fundq", ("001", date(2010, 9, 30), "INDL", "USD", 12, 2010, 3)), "cross", "quarterly_before_first_annual", "quarterly", 1),
    (_add("comp_funda", ("003", date(2011, 12, 31), "INDL", "USD", 12)), "cross", "annual_only_firms", "annual", 1),
    (_drop("msf", lambda r: r != ("001", date(2011, 12, 31))), "cross", "me_missing_at_datadate", "annual", 1),
    (_drop("msf", lambda r: r != ("001", date(2011, 12, 31))), "cross", "me_missing_at_datadate", "quarterly", 1),
    (_drop("msf", lambda r: r != ("001", date(2011, 4, 30))), "cross", "me_missing_4_months_later", "annual", 1),
]  # fmt: skip


@pytest.mark.unit
def test_clean_inputs_have_no_violations(test_paths: DataPaths) -> None:
    results = _run(test_paths, _clean())
    assert results.filter(pl.col("violations") > 0).height == 0
    assert hard_violations(results).height == 0


@pytest.mark.unit
@pytest.mark.parametrize(("change", "kind", "rule", "table", "violations"), CASES)
def test_rule_counts_planted_violation(
    test_paths: DataPaths,
    change: Callable[[dict], None],
    kind: str,
    rule: str,
    table: str,
    violations: int,
) -> None:
    data = copy.deepcopy(_clean())
    change(data)
    assert _violations(_run(test_paths, data), kind, rule, table) == violations


@pytest.mark.unit
def test_missing_required_column_raises(test_paths: DataPaths) -> None:
    _run(test_paths, _clean())
    path = test_paths.raw_tables_dir / "comp_fundq.parquet"
    pl.read_parquet(path).drop("fqtr").write_parquet(path)
    with pytest.raises(ValueError, match="fqtr"):
        check_invariants(test_paths)


def _strings(obj: object) -> set[str]:
    """Every string in a JSON-like object, keys included."""
    if isinstance(obj, dict):
        return set(obj) | {s for v in obj.values() for s in _strings(v)}
    if isinstance(obj, list):
        return {s for v in obj for s in _strings(v)}
    return {obj} if isinstance(obj, str) else set()


@pytest.mark.unit
def test_profile_holds_only_aggregates(test_paths: DataPaths) -> None:
    """The profile serializes to JSON, covers every table, and carries no firm identifier."""
    data = _clean()
    _run(test_paths, data)
    for table in TABLES:
        path = test_paths.raw_tables_dir / f"{table}.parquet"
        pl.read_parquet(path).with_columns(
            pl.col(pl.Float64).cast(pl.Decimal(18, 4))
        ).write_parquet(path)
    pl.read_parquet(test_paths.raw_tables_dir / "comp_funda.parquet").head(0).write_parquet(
        test_paths.raw_tables_dir / "comp_exrt_dly.parquet"
    )
    chars = test_paths.interim_dir / "acc_chars_world.parquet"
    pl.DataFrame(
        {
            "gvkey": ["001", "101"],
            "public_date": [date(2012, 4, 30)] * 2,
            "at_gr1": [0.1, float("nan")],
        }
    ).write_parquet(chars)

    result = json.loads(json.dumps(profile(test_paths.base_dir, chars, "test")))

    assert set(result["tables"]) == set(TABLES)
    assert result["tables"]["comp_funda"]["columns"]["at"]["pos"] == 1.0
    assert result["outputs"]["chars"]["at_gr1"]["nan"] == 0.5
    assert result["max_gvkey"] == 101
    assert not {r[0] for r in data["msf"]} & _strings(result)
