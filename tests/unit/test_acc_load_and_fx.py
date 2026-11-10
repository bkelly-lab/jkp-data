"""Tests for the accounting-stage loaders and FX helpers.

Covers load_raw_fund_table_and_filter, gen_fx1, compustat_fx, add_fx_and_convert_vars,
load_mkt_equity_data, expand and ttm.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import (
    add_fx_and_convert_vars,
    compustat_fx,
    expand,
    gen_fx1,
    load_mkt_equity_data,
    load_raw_fund_table_and_filter,
    ttm,
)
from jkp.data.paths import DataPaths

# ---------------------------------------------------------------------------
# load_raw_fund_table_and_filter
# ---------------------------------------------------------------------------


def _fund_rows(path: Path, rows: list[tuple]) -> None:
    pl.DataFrame(
        rows,
        schema={
            "gvkey": pl.String,
            "indfmt": pl.String,
            "datafmt": pl.String,
            "popsrc": pl.String,
            "consol": pl.String,
            "datadate": pl.Date,
        },
        orient="row",
    ).write_parquet(path)


@pytest.mark.unit
def test_load_raw_fund_mode_1_keeps_indl_and_fs_hist_std(tmp_path: Path) -> None:
    f = tmp_path / "fund.parquet"
    d = date(2000, 12, 31)
    _fund_rows(
        f,
        [
            ("A", "INDL", "HIST_STD", "I", "C", d),  # kept
            ("B", "FS", "HIST_STD", "I", "C", d),  # kept: FS allowed in mode 1
            ("C", "INDL", "STD", "I", "C", d),  # wrong datafmt
            ("D", "INDL", "HIST_STD", "D", "C", d),  # wrong popsrc
            ("E", "INDL", "HIST_STD", "I", "N", d),  # not consolidated
            ("F", "INDL", "HIST_STD", "I", "C", date(1999, 12, 31)),  # before start
            ("G", "XXX", "HIST_STD", "I", "C", d),  # other indfmt
        ],
    )
    out = load_raw_fund_table_and_filter(f, date(2000, 1, 1), "GLOBAL", 1).collect()
    assert out["gvkey"].to_list() == ["A", "B"]
    assert out["source"].to_list() == ["GLOBAL", "GLOBAL"]
    assert out["n"].to_list() == [0, 1]  # row index of the raw file, before filtering


@pytest.mark.unit
def test_load_raw_fund_other_mode_keeps_only_indl_std_d(tmp_path: Path) -> None:
    f = tmp_path / "fund.parquet"
    d = date(2000, 12, 31)
    _fund_rows(
        f,
        [
            ("A", "INDL", "STD", "D", "C", d),  # kept
            ("B", "FS", "STD", "D", "C", d),  # FS not allowed
            ("C", "INDL", "HIST_STD", "D", "C", d),  # wrong datafmt
            ("D", "INDL", "STD", "I", "C", d),  # wrong popsrc
            ("E", "INDL", "STD", "D", "C", date(2000, 1, 1)),  # datadate == start kept
        ],
    )
    out = load_raw_fund_table_and_filter(f, date(2000, 1, 1), "gl", 2).collect()
    assert out["gvkey"].to_list() == ["A", "E"]
    assert out["source"].to_list() == ["gl", "gl"]


# ---------------------------------------------------------------------------
# gen_fx1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gen_fx1_cross_rates_via_gbp(test_paths: DataPaths) -> None:
    (test_paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
    d1, d2 = date(2000, 1, 3), date(2000, 1, 4)
    pl.DataFrame(
        [
            ("GBP", "USD", d1, 1.6),
            ("GBP", "EUR", d1, 1.5),
            ("GBP", "GBP", d1, 1.0),
            ("GBP", "USD", d2, 1.5),
            ("GBP", "EUR", d2, 1.25),
            ("GBP", "EUR", d2, 1.25),  # exact duplicate -> DISTINCT
            ("EUR", "USD", d1, 9.0),  # fromcurd != GBP: ignored
            ("GBP", "JPY", date(2000, 1, 5), 150.0),  # no USD quote that day: no output
        ],
        schema={
            "fromcurd": pl.String,
            "tocurd": pl.String,
            "datadate": pl.Date,
            "exratd": pl.Float64,
        },
        orient="row",
    ).write_parquet(test_paths.raw_tables_dir / "comp_exrt_dly.parquet")

    gen_fx1(test_paths)

    out = pl.read_parquet(test_paths.interim_dir / "raw_data_dfs" / "__fx1.parquet").sort(
        "curcdd", "datadate"
    )
    expected = pl.DataFrame(
        [
            ("EUR", d1, 1.6 / 1.5),
            ("EUR", d2, 1.5 / 1.25),
            ("GBP", d1, 1.6 / 1.0),
            ("USD", d1, 1.0),
            ("USD", d2, 1.0),
        ],
        schema={"curcdd": pl.String, "datadate": pl.Date, "fx": pl.Float64},
        orient="row",
    )
    # GBP on d2 has no GBP->GBP row in the input, so it is absent.
    assert_frame_equal(out, expected, check_exact=False)


# ---------------------------------------------------------------------------
# compustat_fx
# ---------------------------------------------------------------------------


def _write_fx1(test_paths: DataPaths, rows: list[tuple]) -> None:
    (test_paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={"curcdd": pl.String, "datadate": pl.Date, "fx": pl.Float64},
        orient="row",
    ).write_parquet(test_paths.interim_dir / "raw_data_dfs" / "__fx1.parquet")


@pytest.mark.unit
def test_compustat_fx_carries_each_quote_forward_and_last_covers_one_day(
    test_paths: DataPaths,
) -> None:
    _write_fx1(
        test_paths,
        [
            ("EUR", date(2000, 1, 3), 0.9),
            ("EUR", date(2000, 1, 6), 1.0),
            ("EUR", date(2000, 1, 7), 1.1),  # last quote: its own day only
            ("GBP", date(2000, 1, 4), 1.6),  # different currency: separate .over()
        ],
    )
    out = compustat_fx(test_paths)

    eur = out.filter(pl.col("curcdd") == "EUR")
    assert eur.select("datadate", "fx").rows() == [
        (date(2000, 1, 3), 0.9),
        (date(2000, 1, 4), 0.9),
        (date(2000, 1, 5), 0.9),
        (date(2000, 1, 6), 1.0),
        (date(2000, 1, 7), 1.1),
    ]
    gbp = out.filter(pl.col("curcdd") == "GBP")
    assert gbp.rows() == [(date(2000, 1, 4), "GBP", 1.6)]
    assert out.columns == ["datadate", "curcdd", "fx"]


@pytest.mark.unit
def test_compustat_fx_usd_seed_runs_to_first_usd_quote(test_paths: DataPaths) -> None:
    _write_fx1(test_paths, [("USD", date(1982, 2, 1), 1.0)])
    usd = compustat_fx(test_paths).filter(pl.col("curcdd") == "USD")
    # Seed row 1950-01-01 is expanded day by day up to the day before the first quote.
    assert usd["datadate"].min() == date(1950, 1, 1)
    assert usd["datadate"].max() == date(1982, 2, 1)
    assert usd.height == (date(1982, 2, 1) - date(1950, 1, 1)).days + 1
    assert usd["fx"].unique().to_list() == [1.0]


@pytest.mark.unit
def test_compustat_fx_seed_alone_covers_only_its_own_day(test_paths: DataPaths) -> None:
    _write_fx1(test_paths, [("EUR", date(2000, 1, 3), 0.9)])
    usd = compustat_fx(test_paths).filter(pl.col("curcdd") == "USD")
    assert usd.rows() == [(date(1950, 1, 1), "USD", 1.0)]


@pytest.mark.unit
def test_compustat_fx_cad_backfill_before_first_compustat_quote(test_paths: DataPaths) -> None:
    # Bundled month-end rates: 1981-12-31 = 0.843525938422606, 1982-01-31 = 0.834724540901502.
    _write_fx1(
        test_paths,
        [("CAD", date(1982, 2, 26), 0.8), ("CAD", date(1982, 2, 27), 0.81)],
    )
    cad = compustat_fx(test_paths).filter(pl.col("curcdd") == "CAD").sort("datadate")

    assert cad["datadate"].min() == date(1960, 1, 31)  # start of the bundled series
    # Dec month-end rate holds until the day before the next month-end, Jan 30 inclusive.
    dec = cad.filter(pl.col("datadate") == date(1982, 1, 30))["fx"].item()
    assert dec == pytest.approx(0.843525938422606)
    # Last backfilled month-end covers up to the day before Compustat's first CAD quote.
    jan = cad.filter(pl.col("datadate") == date(1982, 2, 25))["fx"].item()
    assert jan == pytest.approx(0.834724540901502)
    assert cad.filter(pl.col("datadate") == date(1982, 1, 31))["fx"].item() == pytest.approx(
        0.834724540901502
    )
    # Compustat's own quotes take over from their first date; no row is dated on/after it twice.
    assert cad.filter(pl.col("datadate") == date(1982, 2, 26))["fx"].item() == 0.8
    assert cad.filter(pl.col("datadate") == date(1982, 2, 27))["fx"].item() == 0.81
    assert cad["datadate"].is_unique().all()


@pytest.mark.unit
def test_compustat_fx_backfill_rows_on_or_after_first_quote_are_dropped(
    test_paths: DataPaths,
) -> None:
    # First Compustat CAD quote predates the end of the bundled series (1982-01-31):
    # bundled month-ends dated >= 1982-01-15 must not be used.
    _write_fx1(test_paths, [("CAD", date(1982, 1, 15), 0.5)])
    cad = compustat_fx(test_paths).filter(pl.col("curcdd") == "CAD")
    assert cad.filter(pl.col("datadate") == date(1982, 1, 15))["fx"].item() == 0.5
    assert cad["datadate"].max() == date(1982, 1, 15)


@pytest.mark.unit
def test_compustat_fx_without_cad_quotes_has_no_backfill(test_paths: DataPaths) -> None:
    _write_fx1(test_paths, [("EUR", date(2000, 1, 3), 0.9)])
    assert compustat_fx(test_paths).filter(pl.col("curcdd") == "CAD").is_empty()


# ---------------------------------------------------------------------------
# add_fx_and_convert_vars
# ---------------------------------------------------------------------------


def _fx_df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "datadate": [date(2000, 12, 31), date(2000, 12, 31), date(2001, 12, 31)],
            "curcdd": ["EUR", "USD", "EUR"],
            "fx": [0.9, 1.0, 1.1],
        }
    )


@pytest.mark.unit
def test_add_fx_annual_converts_vars_and_drops_unmatched_rows() -> None:
    df = pl.LazyFrame(
        {
            "gvkey": ["B", "A", "C", "D", "E"],
            "datadate": [date(2000, 12, 31)] * 3 + [date(2002, 12, 31), date(2000, 12, 31)],
            "curcd": ["EUR", "USD", "JPY", "EUR", None],
            "sale": [100.0, 50.0, 7.0, 8.0, 9.0],
            "at": [10.0, None, 1.0, 1.0, 1.0],
            "other": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = add_fx_and_convert_vars(df, _fx_df(), ["sale", "at"], "annual").collect()
    # JPY (no rate), EUR on a date without a rate and null currency are dropped;
    # remaining rows keep their input order (B before A).
    expected = pl.DataFrame(
        {
            "gvkey": ["B", "A"],
            "datadate": [date(2000, 12, 31)] * 2,
            "curcd": ["USD", "USD"],
            "sale": [100.0 * 0.9, 50.0 * 1.0],
            "at": [10.0 * 0.9, None],
            "other": [1.0, 2.0],
        }
    )
    assert_frame_equal(out, expected)


@pytest.mark.unit
def test_add_fx_quarterly_uses_curcdq() -> None:
    df = pl.LazyFrame(
        {
            "gvkey": ["A", "A"],
            "datadate": [date(2000, 12, 31), date(2001, 12, 31)],
            "curcdq": ["EUR", "EUR"],
            "saleq": [10.0, 10.0],
        }
    )
    out = add_fx_and_convert_vars(df, _fx_df(), ["saleq"], "quarterly").collect()
    assert out["saleq"].to_list() == pytest.approx([10.0 * 0.9, 10.0 * 1.1])
    assert out["curcdq"].to_list() == ["USD", "USD"]
    assert "fx" not in out.columns


# ---------------------------------------------------------------------------
# load_mkt_equity_data
# ---------------------------------------------------------------------------


def _mkt_file(path: Path) -> None:
    e1, e2 = date(2000, 1, 31), date(2000, 2, 29)
    pl.DataFrame(
        [
            ("A", e1, 1, 1, 1, 10.0),
            ("A", e1, 1, 1, 1, 30.0),  # same (gvkey, eom): max kept
            ("A", e1, 1, 1, 1, 20.0),
            ("A", e2, 1, 1, 1, 5.0),
            ("B", e1, 1, 1, 1, 7.0),  # per-gvkey separation
            (None, e1, 1, 1, 1, 99.0),  # null gvkey
            ("C", e1, 0, 1, 1, 99.0),  # not primary
            ("D", e1, 1, 0, 1, 99.0),  # not common
            ("E", e1, 1, 1, 0, 99.0),  # not obs_main
            ("F", e1, 1, 1, 1, None),  # null me_company
        ],
        schema={
            "gvkey": pl.String,
            "eom": pl.Date,
            "primary_sec": pl.Int64,
            "common": pl.Int64,
            "obs_main": pl.Int64,
            "me_company": pl.Float64,
        },
        orient="row",
    ).write_parquet(path)


@pytest.mark.unit
@pytest.mark.parametrize(("alias", "name"), [(True, "me_fiscal"), (False, "me_company")])
def test_load_mkt_equity_filters_and_takes_max_per_gvkey_eom(
    tmp_path: Path, alias: bool, name: str
) -> None:
    f = tmp_path / "mkt.parquet"
    _mkt_file(f)
    out = load_mkt_equity_data(f, alias=alias).collect().sort("gvkey", "eom")
    expected = pl.DataFrame(
        {
            "gvkey": ["A", "A", "B"],
            "eom": [date(2000, 1, 31), date(2000, 2, 29), date(2000, 1, 31)],
            name: [30.0, 5.0, 7.0],
        }
    )
    assert_frame_equal(out, expected)


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_expand_day_is_inclusive_of_both_ends_and_per_id() -> None:
    data = pl.DataFrame(
        {
            "id": ["B", "A"],
            "start": [date(2000, 1, 5), date(2000, 1, 1)],
            "end": [date(2000, 1, 5), date(2000, 1, 3)],
        }
    )
    out = expand(data, ["id"], "start", "end", freq="day")
    expected = pl.DataFrame(
        {
            "id": ["A", "A", "A", "B"],
            "date": [date(2000, 1, 1), date(2000, 1, 2), date(2000, 1, 3), date(2000, 1, 5)],
        }
    )
    assert_frame_equal(out, expected)


@pytest.mark.unit
def test_expand_month_snaps_to_month_end_and_dedups() -> None:
    data = pl.DataFrame(
        {
            "id": ["A", "A"],
            "start": [date(2000, 1, 31), date(2000, 3, 15)],
            "end": [date(2000, 3, 31), date(2000, 3, 31)],
        }
    )
    # Row 1: Jan 31, Feb 29 (Jan 31 + 1mo clips), Mar 31; row 2: Mar 15 -> Mar 31.
    # Month-end snapping makes the Mar entries collide; they are de-duplicated.
    out = expand(data, ["id"], "start", "end", freq="month", new_date_name="eom")
    expected = pl.DataFrame(
        {
            "id": ["A", "A", "A"],
            "eom": [date(2000, 1, 31), date(2000, 2, 29), date(2000, 3, 31)],
        }
    )
    assert_frame_equal(out, expected)


@pytest.mark.unit
def test_expand_empty_range_drops_row() -> None:
    data = pl.DataFrame({"id": ["A"], "start": [date(2000, 1, 5)], "end": [date(2000, 1, 1)]})
    out = expand(data, ["id"], "start", "end", freq="day")
    assert out.is_empty()


# ---------------------------------------------------------------------------
# ttm
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ttm_sums_current_and_three_lags_with_null_for_short_history() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 10.0]})
    out = df.select(ttm("x").alias("t"))["t"].to_list()
    assert out == [None, None, None, 1 + 2 + 3 + 4, 2 + 3 + 4 + 10]


@pytest.mark.unit
def test_ttm_null_input_propagates_to_four_windows() -> None:
    df = pl.DataFrame({"x": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    out = df.select(ttm("x").alias("t"))["t"].to_list()
    # Windows containing the null (rows 1..4) are null; row 5 onwards is clean.
    assert out == [None, None, None, None, None, 3 + 4 + 5 + 6, 4 + 5 + 6 + 7, 5 + 6 + 7 + 8]


@pytest.mark.unit
def test_ttm_separates_groups_only_with_over() -> None:
    # ttm is a plain shift expression: grouping is the caller's job via .over().
    df = pl.DataFrame({"g": ["A"] * 4 + ["B"] * 4, "x": [1.0] * 4 + [10.0] * 4})
    bare = df.select(t=ttm("x"))["t"].to_list()
    grouped = df.with_columns(t=ttm("x").over("g"))["t"].to_list()
    assert bare == [None, None, None, 4.0, 13.0, 22.0, 31.0, 40.0]  # windows run from A into B
    assert grouped == [None, None, None, 4.0, None, None, None, 40.0]
