"""Tests for the four-quarter (TTM) sums built by ``cumulate_4q``."""

import random

import polars as pl
import pytest
from polars import col
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import cumulate_4q, fl_none, ttm

QK = ["gvkey", "fyr", "fyearq", "fqtr"]
SCHEMA = {
    "gvkey": pl.String,
    "fyr": pl.Int64,
    "fyearq": pl.Int64,
    "fqtr": pl.Int64,
    "curcdq": pl.String,
    "saleq": pl.Float64,
    "saley": pl.Float64,
}


def _panel(rows: list[tuple]) -> pl.LazyFrame:
    """rows: (fyearq, fqtr, saleq, saley) for one firm with fyr 12 and USD, or full tuples in SCHEMA order."""
    full = [
        r if len(r) == len(SCHEMA) else ("001", 12, r[0], r[1], "USD", r[2], r[3]) for r in rows
    ]
    return pl.LazyFrame(full, schema=SCHEMA, orient="row").sort(QK)


def _sale(df: pl.LazyFrame) -> list:
    return cumulate_4q(df, ["saleq"]).collect()["sale"].to_list()


@pytest.mark.unit
class TestCumulate4q:
    def test_one_fiscal_year(self) -> None:
        """Q4 keeps the four-quarter sum when it is valid, even if the year-to-date value differs."""
        df = _panel(
            [
                (2010, 1, 10.0, 10.0),
                (2010, 2, 20.0, 30.0),
                (2010, 3, 30.0, 60.0),
                (2010, 4, 40.0, 101.0),
            ]
        )
        assert _sale(df) == [None, None, None, 100.0]

    def test_across_fiscal_years(self) -> None:
        df = _panel(
            [
                (2010, 3, 30.0, 60.0),
                (2010, 4, 40.0, 100.0),
                (2011, 1, 15.0, 15.0),
                (2011, 2, 25.0, 40.0),
            ]
        )
        assert _sale(df) == [None, 100.0, None, 110.0]

    def test_twenty_year_gap_is_missing(self) -> None:
        """FY1994 Q3, Q4 and FY2016 Q1, Q2: fqtr sums to 10, but the quarters are not consecutive."""
        df = _panel(
            [
                (1994, 3, 100.0, 300.0),
                (1994, 4, 100.0, 400.0),
                (2016, 1, 200.0, 200.0),
                (2016, 2, 200.0, 400.0),
            ]
        )
        assert _sale(df) == [None, 400.0, None, None]

    def test_one_missing_fiscal_year_is_missing(self) -> None:
        df = _panel(
            [
                (2010, 3, 30.0, 60.0),
                (2010, 4, 40.0, 100.0),
                (2012, 1, 15.0, 15.0),
                (2012, 2, 25.0, 40.0),
            ]
        )
        assert _sale(df) == [None, 100.0, None, None]

    def test_gap_ending_in_q4_uses_year_to_date(self) -> None:
        df = _panel(
            [
                (1994, 2, 100.0, 200.0),
                (1994, 3, 100.0, 300.0),
                (2016, 1, 200.0, 200.0),
                (2016, 4, 200.0, 900.0),
            ]
        )
        assert _sale(df) == [None, None, None, 900.0]

    def test_fyr_change_is_missing(self) -> None:
        """Consecutive quarters with fqtr summing to 10, but the fiscal year-end moves from June to December."""
        df = _panel(
            [
                ("001", 6, 2010, 1, "USD", 10.0, 10.0),
                ("001", 6, 2010, 2, "USD", 10.0, 20.0),
                ("001", 12, 2010, 3, "USD", 10.0, 30.0),
                ("001", 12, 2010, 4, "USD", 10.0, 99.0),
            ]
        )
        assert _sale(df) == [None, None, None, 99.0]

    def test_currency_change_is_missing(self) -> None:
        df = _panel(
            [
                ("001", 12, 2010, 2, "CAD", 10.0, 20.0),
                ("001", 12, 2010, 3, "USD", 10.0, 30.0),
                ("001", 12, 2010, 4, "USD", 10.0, 40.0),
                ("001", 12, 2011, 1, "USD", 10.0, 10.0),
            ]
        )
        assert _sale(df) == [None, None, 40.0, None]

    def test_missing_input_is_missing(self) -> None:
        df = _panel(
            [
                (2010, 2, None, 30.0),
                (2010, 3, 30.0, 60.0),
                (2010, 4, 40.0, 100.0),
                (2011, 1, 15.0, 15.0),
            ]
        )
        assert _sale(df) == [None, None, 100.0, None]


def _legacy_cumulate_4q(df, var_list):
    """Verbatim copy of cumulate_4q from main (daacd5e), the reference for the equivalence tests."""
    var_yrl_name_list = [var[:-1] for var in var_list]
    df = (
        df.with_columns(
            [
                ttm(var_yrl).alias(var_yrl_name)
                for var_yrl, var_yrl_name in zip(var_list, var_yrl_name_list, strict=True)
            ]
            + [
                (
                    (col("gvkey") == col("gvkey").shift(3))
                    & (col("fyr") == col("fyr").shift(3))
                    & (col("curcdq") == col("curcdq").shift(3))
                    & (ttm("fqtr") == 10)
                ).alias("not_null_flag")
            ]
        )
        .with_columns(
            [
                pl.when(col("not_null_flag"))
                .then(col(var_yrl_name))
                .otherwise(fl_none())
                .alias(var_yrl_name)
                for var_yrl_name in var_yrl_name_list
            ]
        )
        .with_columns(
            [
                pl.when(col(var_yrl_name).is_null() & (col("fqtr") == 4))
                .then(col(f"{var_yrl_name}y"))
                .otherwise(col(var_yrl_name))
                .alias(var_yrl_name)
                for var_yrl_name in var_yrl_name_list
            ]
        )
        .drop(
            ["not_null_flag"]
            + var_list
            + [f"{var_yrl_name}y" for var_yrl_name in var_yrl_name_list]
        )
    )
    return df


def _random_panel(seed: int, n_firms: int = 40) -> pl.LazyFrame:
    """Firms with gaps in their fiscal quarters, fiscal year-end and currency switches, and null values."""
    rng = random.Random(seed)
    rows = []
    for g in range(n_firms):
        quarters = sorted(rng.sample(range(8000, 8048), rng.randint(1, 30)))
        switch_fyr, switch_cur = rng.choice(quarters), rng.choice(quarters + [10**9])
        for q in quarters:
            fyearq, fqtr = (q - 1) // 4, (q - 1) % 4 + 1
            vals = [None if rng.random() < 0.1 else float(rng.randint(-50, 500)) for _ in range(4)]
            rows.append(
                (
                    f"{g:03d}",
                    6 if q < switch_fyr and g % 3 == 0 else 12,
                    fyearq,
                    fqtr,
                    "CAD" if q >= switch_cur else "USD",
                    *vals,
                )
            )
    schema = {**{k: SCHEMA[k] for k in QK}, "curcdq": pl.String}
    schema |= {"saleq": pl.Float64, "saley": pl.Float64, "ibq": pl.Float64, "iby": pl.Float64}
    return pl.LazyFrame(rows, schema=schema, orient="row").sort(QK)


def _skipped_quarters() -> pl.Expr:
    """Rows the old check accepts although the four quarters are not consecutive."""
    q = 4 * pl.col("fyearq") + pl.col("fqtr")
    old_check = (
        (pl.col("gvkey") == pl.col("gvkey").shift(3))
        & (pl.col("fyr") == pl.col("fyr").shift(3))
        & (pl.col("curcdq") == pl.col("curcdq").shift(3))
        & (ttm("fqtr") == 10)
    )
    return (old_check & (q - q.shift(3) != 3)).fill_null(False)


@pytest.mark.unit
class TestCumulate4qEquivalence:
    """cumulate_4q matches main's version except where main summed non-consecutive quarters."""

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_legacy_except_skipped_quarters(self, seed: int) -> None:
        df = _random_panel(seed).with_columns(
            skipped=_skipped_quarters(), ytd_sale=pl.col("saley"), ytd_ib=pl.col("iby")
        )
        new = cumulate_4q(df, ["saleq", "ibq"]).collect()
        old = _legacy_cumulate_4q(df, ["saleq", "ibq"]).collect()
        assert_frame_equal(new.filter(~pl.col("skipped")), old.filter(~pl.col("skipped")))
        skipped = new.filter("skipped")
        assert skipped.height > 0
        assert_frame_equal(
            skipped.select("sale", "ib"),
            skipped.select(
                pl.when(pl.col("fqtr") == 4).then(pl.col(f"ytd_{v}")).alias(v)
                for v in ("sale", "ib")
            ),
        )
