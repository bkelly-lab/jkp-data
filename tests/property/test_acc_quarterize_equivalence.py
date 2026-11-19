"""Equivalence of quarterize with its previous implementation on generated quarterly panels.

The reference is a verbatim copy of quarterize from 22df051, the version before quarterize was
rewritten as one expression per variable. The panels mix the cases quarterize has to resolve: the same
fiscal quarter reported twice, missing quarters, fiscal years that start after Q1, rows without a
fiscal quarter (alone on their date, sharing a date, or inside a fiscal year with regular quarters),
fiscal year-end switches, missing values and an arbitrary row order.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from polars import col
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import fl_none, quarterize

VARS = ["saley", "capxy"]


def _legacy_quarterize(df: pl.LazyFrame, var_list: list[str]) -> pl.LazyFrame:
    """Verbatim copy of quarterize from 22df051, the reference for the equivalence test."""
    list_aux1 = [col("gvkey").cum_count().over(["gvkey", "fyr", "fyearq"]).alias("count_aux")] + [
        col(var).cast(pl.Float64).diff().alias(var + "_q") for var in var_list
    ]
    c1 = (col("fqtr") != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col("fqtr").diff() != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    list_aux2 = [
        pl.when(col("count_aux") == 1).then(col(var)).otherwise(col(var + "_q")).alias(var + "_q")
        for var in var_list
    ] + [pl.when(col("count_aux") == 1).then(c1).otherwise(c2).alias("del")]
    list_aux3 = [
        pl.when(col("del")).then(fl_none()).otherwise(col(var + "_q")).alias(var + "_q")
        for var in var_list
    ]
    transition = (
        col("fqtr").is_null()
        & (pl.len().over("gvkey", "datadate") == 1)
        & col("fqtr").is_null().all().over("gvkey", "fyr", "fyearq")
    )
    regular = (
        df.filter(col("fqtr").is_not_null())
        .sort(["gvkey", "fyr", "fyearq", "fqtr", "datadate"])
        .unique(["gvkey", "fyr", "fyearq", "fqtr"], keep="first", maintain_order=True)
        .with_columns(list_aux1)
        .sort(["gvkey", "fyr", "fyearq", "fqtr"])
        .with_columns(list_aux2)
        .with_columns(list_aux3)
        .drop(["del", "count_aux"])
        .unique(["gvkey", "fyr", "fyearq", "fqtr"])
    )
    transition_rows = df.filter(transition).with_columns(
        [fl_none().alias(var + "_q") for var in var_list]
    )
    return pl.concat([regular, transition_rows]).sort(
        ["gvkey", "fyr", "fyearq", "fqtr", "datadate"]
    )


@st.composite
def quarterly_panels(draw) -> pl.DataFrame:
    """Rows drawn from a small grid of firms, fiscal year-ends, fiscal years, fiscal quarters and
    month-ends, so duplicates, gaps, shared dates and rows without fqtr all occur. No two rows share
    (gvkey, fyr, fyearq, fqtr, datadate): which of two such rows is kept is arbitrary in both versions."""
    value = st.one_of(st.none(), st.integers(-50, 500).map(float))
    keys = draw(
        st.lists(
            st.tuples(
                st.sampled_from(["001", "002", "003"]),
                st.sampled_from([3, 12]),
                st.integers(2010, 2012),
                st.one_of(st.none(), st.integers(1, 4)),
                st.integers(0, 11),
            ),
            min_size=1,
            max_size=40,
            unique=True,
        )
    )
    rows = [
        (g, date(fy, month + 1, 28), fyr, fy, q, draw(value), draw(value))
        for g, fyr, fy, q, month in keys
    ]
    schema = {
        "gvkey": pl.String,
        "datadate": pl.Date,
        "fyr": pl.Int64,
        "fyearq": pl.Int64,
        "fqtr": pl.Int64,
        "saley": pl.Float64,
        "capxy": pl.Float64,
    }
    return pl.DataFrame(draw(st.permutations(rows)), schema=schema, orient="row")


class TestQuarterizeEquivalence:
    @given(df=quarterly_panels())
    @settings(max_examples=500, deadline=None)
    def test_matches_legacy(self, df: pl.DataFrame) -> None:
        new = quarterize(df.lazy(), VARS).collect()
        old = _legacy_quarterize(df.lazy(), VARS).collect()
        assert_frame_equal(new, old, check_exact=True)
