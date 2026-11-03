"""Tests for the NA/Global stacking, quarterization and one-row-per-date steps of ``standardized_accounting_data``."""

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import one_row_per_date, prefer_global, quarterize

QK = ["gvkey", "fyr", "fyearq", "fqtr"]


def _fundq(rows: list[tuple], source: str) -> pl.LazyFrame:
    return (
        pl.LazyFrame(
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
        )
        .with_columns(source=pl.lit(source))
        .with_row_index("n")
    )


def _calendar(fyr: int, ends: list[date]) -> list[tuple]:
    """Quarter-end rows of one fiscal calendar for `_fundq`; fyearq and fqtr are placeholders."""
    return [("001", d, fyr, d.year, 1, float(fyr)) for d in ends]


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

    def test_rows_without_fqtr_are_dropped(self) -> None:
        """A leftover from the old fiscal calendar (fqtr missing) is dropped and does not break Q1's flow."""
        df = pl.LazyFrame(
            {
                "gvkey": ["001"] * 3,
                "datadate": [date(2014, 3, 31), date(2014, 6, 30), date(2014, 9, 30)],
                "fyr": pl.Series([12] * 3, dtype=pl.Int16),
                "fyearq": pl.Series([2014] * 3, dtype=pl.Int32),
                "fqtr": pl.Series([1, 2, None], dtype=pl.Int16),
                "capxy": [10.0, 25.0, 40.0],
            }
        )
        out = quarterize(df, ["capxy"]).collect()
        assert out.select("datadate", "capxy_q").rows() == [
            (date(2014, 3, 31), 10.0),
            (date(2014, 6, 30), 15.0),
        ]


@pytest.mark.unit
class TestOneRowPerDate:
    @pytest.mark.parametrize(("old", "new"), [(3, 12), (12, 6)])
    def test_overlap_keeps_established_calendar(self, old: int, new: int) -> None:
        """Both calendars report Sep and Dec 2015; the old calendar's run started first, so it is kept
        whichever fyr is higher."""
        old_ends = [
            date(2014, 9, 30),
            date(2014, 12, 31),
            date(2015, 3, 31),
            date(2015, 6, 30),
            date(2015, 9, 30),
            date(2015, 12, 31),
        ]
        new_ends = [date(2015, 9, 30), date(2015, 12, 31), date(2016, 3, 31)]
        df = _fundq(_calendar(old, old_ends) + _calendar(new, new_ends), "NA")
        out = one_row_per_date(df).sort("datadate").collect()
        assert out["fyr"].to_list() == [old] * 6 + [new]

    def test_gap_restarts_calendar_run(self) -> None:
        """A calendar that returns after a gap of more than 100 days starts a new run, so its earlier
        quarters don't make it the established calendar."""
        df = _fundq(
            _calendar(12, [date(2010, 3, 31), date(2010, 6, 30)])
            + _calendar(3, [date(2014, 9, 30), date(2014, 12, 31), date(2015, 3, 31)])
            + _calendar(12, [date(2015, 3, 31), date(2015, 6, 30)]),
            "NA",
        )
        out = one_row_per_date(df).filter(pl.col("datadate") == date(2015, 3, 31)).collect()
        assert out["fyr"].to_list() == [3]

    def test_same_run_start_keeps_highest_fyr(self) -> None:
        """Both calendars' runs start on the same date: the higher fyr is kept."""
        df = _fundq(
            [
                ("001", date(2012, 9, 30), 6, 2013, 1, 10.0),
                ("001", date(2012, 9, 30), 12, 2012, 3, 11.0),
            ],
            "NA",
        )
        out = one_row_per_date(df).collect()
        assert out.select("fyr", "atq").rows() == [(12, 11.0)]

    def test_same_fyr_keeps_global_row(self) -> None:
        """NA and Global report the same date and fyr under different fiscal quarters: the Global row is kept."""
        df = pl.concat(
            [
                _fundq([("001", date(2014, 3, 31), 12, 2014, 1, 10.0)], "GLOBAL"),
                _fundq([("001", date(2014, 3, 31), 12, 2013, 4, 99.0)], "NA"),
            ]
        )
        out = one_row_per_date(df).collect()
        assert out.select("source", "atq").rows() == [("GLOBAL", 10.0)]
