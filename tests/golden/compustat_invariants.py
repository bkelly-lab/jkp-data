"""
Invariant rules for the Compustat inputs of the accounting stage.

The same rules run on the real WRDS tables (on Yale, through calibration/profile_compustat.py) and on
the synthetic tables (in CI). Each rule counts violations among the rows it checks:
    hard:  must be 0 in the synthetic data (the stage assumes them);
    soft:  rates the synthetic data reproduce (each one is a case the stage must handle);
    cross: coverage between tables (NA vs Global, accounting vs FX and market equity).
Rules see the rows the stage sees: each table is read with `load_raw_fund_table_and_filter`, FX with
`compustat_fx` and market equity with `load_mkt_equity_data`.

Usage: python -m tests.golden.compustat_invariants DATA_DIR
"""

import sys
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import (
    compustat_fx,
    load_mkt_equity_data,
    load_raw_fund_table_and_filter,
)
from jkp.data.config import ACCOUNTING_START_DATE
from jkp.data.paths import DataPaths

TABLES = {
    "comp_funda": ("NA", 2, "annual"),
    "comp_g_funda": ("GLOBAL", 1, "annual"),
    "comp_fundq": ("NA", 2, "quarterly"),
    "comp_g_fundq": ("GLOBAL", 1, "quarterly"),
}
FILTER_COLUMNS = ["gvkey", "datadate", "indfmt", "datafmt", "popsrc", "consol"]
REQUIRED_COLUMNS = {
    "annual": [*FILTER_COLUMNS, "fyr", "curcd", "at", "ib"],
    "quarterly": [*FILTER_COLUMNS, "fyr", "fyearq", "fqtr", "curcdq", "atq", "ibq"],
}
RESULT_COLUMNS = ["kind", "rule", "table", "violations", "rows"]


def _result(
    lf: pl.LazyFrame, kind: str, rule: str, table: str, violations: pl.Expr, rows: pl.Expr
) -> pl.LazyFrame:
    """One result row: the rule's violations and the rows it checked, both aggregated over lf."""
    return lf.select(
        kind=pl.lit(kind),
        rule=pl.lit(rule),
        table=pl.lit(table),
        violations=violations.cast(pl.Int64),
        rows=rows.cast(pl.Int64),
    )


def _flag_share(lf: pl.LazyFrame, flag: pl.Expr, kind: str, rule: str, table: str) -> pl.LazyFrame:
    """Result where each row of lf is one checked unit and flag marks a violation."""
    return _result(lf.select(flag=flag), kind, rule, table, pl.col("flag").sum(), pl.len())


def load_tables(paths: DataPaths) -> dict[str, pl.LazyFrame]:
    """
    Description:
        Read the four Compustat fundamentals tables with the stage's own read filters.
    Steps:
        1) Raise if a table lacks a column the stage filters or keys on.
        2) Load each table with load_raw_fund_table_and_filter from ACCOUNTING_START_DATE.
    Output:
        Dict of table name to filtered LazyFrame.
    """
    tables = {}
    for table, (source, mode, freq) in TABLES.items():
        file = paths.raw_tables_dir / f"{table}.parquet"
        missing = set(REQUIRED_COLUMNS[freq]) - set(pl.scan_parquet(file).collect_schema().names())
        if missing:
            raise ValueError(f"{table} lacks required columns {sorted(missing)}")
        tables[table] = load_raw_fund_table_and_filter(file, ACCOUNTING_START_DATE, source, mode)
    return tables


def table_rules(lf: pl.LazyFrame, table: str) -> list[pl.LazyFrame]:
    """
    Description:
        Rules on one fundamentals table: date format, keys, fiscal quarters and calendars.
    Steps:
        1) Hard: month-end datadate; unique key ((gvkey, datadate[, fyr], indfmt)); valid fyr, fyearq, fqtr.
        2) Soft: firms with an fyr change; quarterly null fqtr, duplicate and skipped fiscal quarters,
           dates on two calendars; Global dates with INDL and FS, and FS dates on two calendars.
    Output:
        List of one-row result LazyFrames.
    """
    quarterly = TABLES[table][2] == "quarterly"
    keys = ["gvkey", "datadate", "fyr", "indfmt"] if quarterly else ["gvkey", "datadate", "indfmt"]
    out = [
        _flag_share(
            lf,
            pl.col("datadate") != pl.col("datadate").dt.month_end(),
            "hard",
            "month_end_datadate",
            table,
        ),
        _result(lf, "hard", "unique_key", table, pl.len() - pl.struct(keys).n_unique(), pl.len()),
        _flag_share(
            lf.group_by("gvkey").agg(pl.col("fyr").n_unique()),
            pl.col("fyr") > 1,
            "soft",
            "fyr_change_firms",
            table,
        ),
    ]
    if quarterly:
        fiscal_quarter = ["gvkey", "fyr", "fyearq", "fqtr", "indfmt"]
        with_fqtr = lf.filter(pl.col("fqtr").is_not_null())
        quarter_steps = (
            with_fqtr.select("gvkey", "fyr", "indfmt", q=pl.col("fyearq") * 4 + pl.col("fqtr"))
            .unique()
            .sort(["gvkey", "fyr", "indfmt", "q"])
            .select(step=pl.col("q").diff().over(["gvkey", "fyr", "indfmt"]))
            .drop_nulls()
        )
        out += [
            _flag_share(
                lf,
                pl.col("fyr").is_null()
                | pl.col("fyearq").is_null()
                | ~pl.col("fyr").is_between(1, 12)
                | (pl.col("fqtr").is_not_null() & ~pl.col("fqtr").is_between(1, 4)),
                "hard",
                "fiscal_keys",
                table,
            ),
            _flag_share(lf, pl.col("fqtr").is_null(), "soft", "null_fqtr", table),
            _flag_share(
                with_fqtr,
                pl.len().over(fiscal_quarter) > 1,
                "soft",
                "duplicate_fiscal_quarter",
                table,
            ),
            _flag_share(quarter_steps, pl.col("step") > 1, "soft", "skipped_quarter", table),
            _flag_share(
                lf.group_by(["gvkey", "datadate"]).agg(pl.col("fyr").n_unique()),
                pl.col("fyr") > 1,
                "soft",
                "two_calendars_one_date",
                table,
            ),
        ]
    if TABLES[table][0] == "GLOBAL":
        date_keys = ["gvkey", "datadate", "fyr"] if quarterly else ["gvkey", "datadate"]
        out.append(
            _flag_share(
                lf.group_by(date_keys).agg(pl.col("indfmt").n_unique()),
                pl.col("indfmt") > 1,
                "soft",
                "indl_and_fs_one_date",
                table,
            )
        )
        if quarterly:
            out.append(
                _flag_share(
                    lf.filter(pl.col("indfmt") == "FS")
                    .group_by(["gvkey", "datadate"])
                    .agg(pl.col("fyr").n_unique()),
                    pl.col("fyr") > 1,
                    "soft",
                    "fs_on_two_calendars",
                    table,
                )
            )
    return out


def fx_rules(tables: dict[str, pl.LazyFrame], fx: pl.LazyFrame) -> list[pl.LazyFrame]:
    """
    Description:
        FX coverage: why a record finds no rate in compustat_fx (the stage drops such records).
    Steps:
        1) Hard: every rate is positive.
        2) Per table, split records without a rate into: no currency, a currency without rates, a date
           before the currency's first rate, a date after its last rate (compustat_fx has no gaps inside).
    Output:
        List of one-row result LazyFrames.
    """
    ranges = fx.group_by("curcdd").agg(
        first=pl.col("datadate").min(), last=pl.col("datadate").max()
    )
    out = [_flag_share(fx, ~(pl.col("fx") > 0).fill_null(False), "hard", "fx_positive", "__fx1")]
    for table, lf in tables.items():
        cur = "curcdq" if TABLES[table][2] == "quarterly" else "curcd"
        joined = lf.select(cur, "datadate").join(ranges, left_on=cur, right_on="curcdd", how="left")
        has_rates = pl.col(cur).is_not_null() & pl.col("first").is_not_null()
        out += [
            _flag_share(joined, pl.col(cur).is_null(), "cross", "fx_no_currency", table),
            _flag_share(
                joined,
                pl.col(cur).is_not_null() & pl.col("first").is_null(),
                "cross",
                "fx_currency_without_rates",
                table,
            ),
            _flag_share(
                joined,
                has_rates & (pl.col("datadate") < pl.col("first")),
                "cross",
                "fx_before_first_rate",
                table,
            ),
            _flag_share(
                joined,
                has_rates & (pl.col("datadate") > pl.col("last")),
                "cross",
                "fx_after_last_rate",
                table,
            ),
        ]
    return out


def cross_rules(tables: dict[str, pl.LazyFrame], me: pl.LazyFrame) -> list[pl.LazyFrame]:
    """
    Description:
        Rules across tables: NA/Global overlap, quarterly and annual coverage, market equity coverage.
    Steps:
        1) NA/Global overlap of firm-periods (annual: gvkey, datadate; quarterly: gvkey, fyr, fyearq, fqtr).
        2) Quarterly dates on two calendars across both sources; quarterly dates before a firm's first
           annual report (or with none); firms with annual but no quarterly records.
        3) Records without market equity at datadate and four months later (the first public month).
    Output:
        List of one-row result LazyFrames.
    """
    out = []
    for freq, keys in [
        ("annual", ["gvkey", "datadate"]),
        ("quarterly", ["gvkey", "fyr", "fyearq", "fqtr"]),
    ]:
        na, gl = (
            tables[t].select(keys).unique().with_columns(src=pl.lit(t))
            for t in TABLES
            if TABLES[t][2] == freq
        )
        out.append(
            _flag_share(
                pl.concat([na, gl]).group_by(keys).agg(pl.col("src").n_unique()),
                pl.col("src") > 1,
                "cross",
                "na_global_overlap",
                freq,
            )
        )
    ann = pl.concat([tables[t].select("gvkey", "datadate") for t in ["comp_funda", "comp_g_funda"]])
    qtr = pl.concat(
        [tables[t].select("gvkey", "datadate", "fyr") for t in ["comp_fundq", "comp_g_fundq"]]
    )
    qtr_dates = qtr.select("gvkey", "datadate").unique()
    first_annual = ann.group_by("gvkey").agg(first_annual=pl.col("datadate").min())
    out += [
        _flag_share(
            qtr.group_by(["gvkey", "datadate"]).agg(pl.col("fyr").n_unique()),
            pl.col("fyr") > 1,
            "cross",
            "two_calendars_one_date",
            "quarterly",
        ),
        _flag_share(
            qtr_dates.join(first_annual, on="gvkey", how="left"),
            pl.col("first_annual").is_null() | (pl.col("datadate") < pl.col("first_annual")),
            "cross",
            "quarterly_before_first_annual",
            "quarterly",
        ),
        _flag_share(
            ann.select("gvkey")
            .unique()
            .join(
                qtr.select("gvkey").unique().with_columns(has_qtr=pl.lit(True)),
                on="gvkey",
                how="left",
            ),
            pl.col("has_qtr").is_null(),
            "cross",
            "annual_only_firms",
            "annual",
        ),
    ]
    for freq, dates in [("annual", ann.unique()), ("quarterly", qtr_dates)]:
        for rule, offset in [("me_missing_at_datadate", 0), ("me_missing_4_months_later", 4)]:
            eom = pl.col("datadate").dt.offset_by(f"{offset}mo").dt.month_end()
            out.append(
                _flag_share(
                    dates.with_columns(eom=eom).join(me, on=["gvkey", "eom"], how="left"),
                    pl.col("me_fiscal").is_null(),
                    "cross",
                    rule,
                    freq,
                )
            )
    return out


def check_invariants(paths: DataPaths) -> pl.DataFrame:
    """
    Description:
        Run every rule on the accounting-stage inputs under paths.
    Steps:
        1) Load the four fundamentals tables (stage filters), FX (compustat_fx, needs
           interim/raw_data_dfs/__fx1.parquet) and market equity (interim/world_msf.parquet).
        2) Evaluate table, FX and cross-table rules in one collect.
    Output:
        DataFrame [kind, rule, table, violations, rows, rate], sorted by kind, rule and table.
    """
    tables = load_tables(paths)
    fx = compustat_fx(paths).lazy()
    me = load_mkt_equity_data(paths.interim_dir / "world_msf.parquet")
    rules = [r for table, lf in tables.items() for r in table_rules(lf, table)]
    rules += fx_rules(tables, fx) + cross_rules(tables, me)
    return (
        pl.concat(pl.collect_all(rules))
        .select(RESULT_COLUMNS)
        .with_columns(rate=pl.col("violations") / pl.col("rows"))
        .sort(["kind", "rule", "table"])
    )


def hard_violations(results: pl.DataFrame) -> pl.DataFrame:
    """Hard rules with at least one violation."""
    return results.filter((pl.col("kind") == "hard") & (pl.col("violations") > 0))


if __name__ == "__main__":
    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        print(check_invariants(DataPaths(Path(sys.argv[1]).expanduser())))
