"""Tests for the NA/Global stacking, quarterization and one-row-per-date steps of ``standardized_accounting_data``."""

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import one_row_per_date, prefer_global, quarterize

QK = ["gvkey", "fyr", "fyearq", "fqtr"]


def _fundq(rows: list[tuple], source: str) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema={
            "gvkey": pl.String,
            "datadate": pl.Date,
            "fyr": pl.Int16,
            "fyearq": pl.Int32,
            "fqtr": pl.Int16,
            "atq": pl.Float64,
        },
        orient="row",
    ).with_columns(source=pl.lit(source))


@pytest.mark.unit
class TestPreferGlobal:
    def test_overlap_keeps_global_row(self) -> None:
        g = _fundq([("001", date(2010, 3, 31), 12, 2010, 1, 10.0)], "GLOBAL")
        na = _fundq([("001", date(2010, 3, 31), 12, 2010, 1, 11.0)], "NA")
        out = prefer_global(g, na, QK).collect()
        assert out.select("source", "atq").rows() == [("GLOBAL", 10.0)]

    def test_same_source_duplicates_are_kept(self) -> None:
        """Two NA rows sharing a key (no Global row) both survive; quarterize picks one later."""
        na = _fundq(
            [
                ("001", date(2012, 12, 31), 9, 2013, None, 1.0),
                ("001", date(2013, 3, 31), 9, 2013, None, 2.0),
            ],
            "NA",
        )
        g = _fundq([("002", date(2012, 12, 31), 9, 2013, 1, 5.0)], "GLOBAL")
        out = prefer_global(g, na, QK).collect()
        assert out.filter(pl.col("gvkey") == "001").height == 2
        assert out.filter(pl.col("gvkey") == "002").height == 1

    def test_global_pair_kept_and_na_dropped_in_mixed_group(self) -> None:
        g = _fundq(
            [
                ("001", date(2021, 8, 31), 9, 2021, 4, 1.0),
                ("001", date(2021, 9, 30), 9, 2021, 4, 1.0),
            ],
            "GLOBAL",
        )
        na = _fundq([("001", date(2021, 9, 30), 9, 2021, 4, 2.0)], "NA")
        out = prefer_global(g, na, QK).collect()
        assert out["source"].to_list() == ["GLOBAL", "GLOBAL"]

    def test_null_key_values_match_across_sources(self) -> None:
        g = _fundq([("001", date(2013, 3, 31), 9, 2013, None, 1.0)], "GLOBAL")
        na = _fundq([("001", date(2013, 6, 30), 9, 2013, None, 2.0)], "NA")
        out = prefer_global(g, na, QK).collect()
        assert out["source"].to_list() == ["GLOBAL"]

    def test_annual_key(self) -> None:
        keys = ["gvkey", "datadate"]
        g = pl.LazyFrame(
            {
                "gvkey": ["001"],
                "datadate": [date(1996, 12, 31)],
                "at": [353.8],
                "source": ["GLOBAL"],
            }
        )
        na = pl.LazyFrame(
            {
                "gvkey": ["001", "001"],
                "datadate": [date(1996, 12, 31), date(1997, 12, 31)],
                "at": [354.3, 357.1],
                "source": ["NA", "NA"],
            }
        )
        out = prefer_global(g, na, keys).sort("datadate").collect()
        assert out.select("source", "at").rows() == [("GLOBAL", 353.8), ("NA", 357.1)]


@pytest.mark.unit
class TestQuarterize:
    def test_ytd_to_quarterly_flows(self) -> None:
        df = pl.LazyFrame(
            {
                "gvkey": ["001"] * 4,
                "datadate": [
                    date(2010, 3, 31),
                    date(2010, 6, 30),
                    date(2010, 9, 30),
                    date(2010, 12, 31),
                ],
                "fyr": pl.Series([12] * 4, dtype=pl.Int16),
                "fyearq": pl.Series([2010] * 4, dtype=pl.Int32),
                "fqtr": pl.Series([1, 2, 3, 4], dtype=pl.Int16),
                "saley": [10.0, 25.0, 45.0, 70.0],
            }
        )
        out = quarterize(df, ["saley"]).collect()
        assert out["saley_q"].to_list() == [10.0, 15.0, 20.0, 25.0]

    def test_duplicate_fiscal_quarter_keeps_earliest_datadate(self) -> None:
        """Same fiscal quarter reported twice: the earliest datadate is kept, whatever the input order."""
        df = pl.LazyFrame(
            {
                "gvkey": ["001"] * 3,
                "datadate": [date(2010, 3, 31), date(2010, 7, 31), date(2010, 6, 30)],
                "fyr": pl.Series([12] * 3, dtype=pl.Int16),
                "fyearq": pl.Series([2010] * 3, dtype=pl.Int32),
                "fqtr": pl.Series([1, 2, 2], dtype=pl.Int16),
                "saley": [10.0, 99.0, 25.0],
            }
        )
        out = quarterize(df, ["saley"]).collect()
        assert out.select("datadate", "saley_q").rows() == [
            (date(2010, 3, 31), 10.0),
            (date(2010, 6, 30), 15.0),
        ]

    def test_missing_fqtr_group_keeps_earliest_row(self) -> None:
        df = pl.LazyFrame(
            {
                "gvkey": ["001"] * 3,
                "datadate": [date(2013, 3, 31), date(2012, 12, 31), date(2013, 6, 30)],
                "fyr": pl.Series([9] * 3, dtype=pl.Int16),
                "fyearq": pl.Series([2013] * 3, dtype=pl.Int32),
                "fqtr": pl.Series([None] * 3, dtype=pl.Int16),
                "saley": [2.0, 1.0, 3.0],
            }
        )
        out = quarterize(df, ["saley"]).collect()
        assert out["datadate"].to_list() == [date(2012, 12, 31)]
        assert out["saley_q"].to_list() == [None]


@pytest.mark.unit
class TestOneRowPerDate:
    def test_regular_quarter_beats_leftover_with_higher_fyr(self) -> None:
        """Year-end moved from 12 to 9: the old-calendar row (fqtr missing) must not replace the real quarter."""
        df = _fundq(
            [
                ("001", date(2014, 3, 31), 9, 2014, 2, 10.0),
                ("001", date(2014, 3, 31), 12, 2014, None, 99.0),
            ],
            "NA",
        )
        out = one_row_per_date(df).collect()
        assert out.select("fyr", "fqtr", "atq").rows() == [(9, 2, 10.0)]

    def test_both_regular_keeps_highest_fyr(self) -> None:
        df = _fundq(
            [
                ("001", date(2012, 9, 30), 6, 2013, 1, 10.0),
                ("001", date(2012, 9, 30), 12, 2012, 3, 11.0),
            ],
            "NA",
        )
        out = one_row_per_date(df).collect()
        assert out.select("fyr", "atq").rows() == [(12, 11.0)]

    def test_leftover_alone_on_its_date_is_kept(self) -> None:
        df = _fundq(
            [
                ("001", date(2012, 12, 31), 9, 2013, None, 1.0),
                ("001", date(2013, 3, 31), 12, 2013, 1, 2.0),
            ],
            "NA",
        )
        out = one_row_per_date(df).sort("datadate").collect()
        assert out.select("datadate", "atq").rows() == [
            (date(2012, 12, 31), 1.0),
            (date(2013, 3, 31), 2.0),
        ]
