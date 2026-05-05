from datetime import date

import polars as pl
import pytest


class TestEomOffset1MoMonthEnd:
    @pytest.mark.parametrize(
        "input_eom, expected_eom",
        [
            (date(2020, 1, 31), date(2020, 2, 29)),  # Jan 31 -> Feb 29 (leap)
            (date(2021, 1, 31), date(2021, 2, 28)),  # Jan 31 -> Feb 28 (non-leap)
            (date(2020, 2, 29), date(2020, 3, 31)),  # Feb 29 -> Mar 31
            (date(2020, 12, 31), date(2021, 1, 31)),  # Dec -> Jan
            (date(2020, 4, 30), date(2020, 5, 31)),  # Apr 30 -> May 31
            (date(2020, 5, 31), date(2020, 6, 30)),  # May 31 -> Jun 30
        ],
    )
    def test_offset_one_month_then_month_end(self, input_eom, expected_eom):
        df = pl.DataFrame({"eom": pl.Series("eom", [input_eom], dtype=pl.Date)})
        result = df.with_columns(
            pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("next_eom")
        )["next_eom"][0]
        assert result == expected_eom

    def test_dec_to_jan_year_increment(self):
        df = pl.DataFrame(
            {"eom": pl.Series("eom", [date(2020, 12, 31)], dtype=pl.Date)}
        )
        result = df.with_columns(
            pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("next_eom")
        )["next_eom"][0]
        assert result == date(2021, 1, 31)


class TestEomLag1FromDate:
    @pytest.mark.parametrize(
        "input_date, expected_eom_lag1",
        [
            (date(2020, 1, 15), date(2019, 12, 31)),
            (date(2020, 3, 1), date(2020, 2, 29)),
            (date(2021, 3, 1), date(2021, 2, 28)),
            (date(2020, 1, 1), date(2019, 12, 31)),
        ],
    )
    def test_eom_lag1_via_month_start_minus_one_day(
        self, input_date, expected_eom_lag1
    ):
        df = pl.DataFrame({"date": pl.Series("date", [input_date], dtype=pl.Date)})
        result = df.with_columns(
            (pl.col("date").dt.month_start() - pl.duration(days=1)).alias("eom_lag1")
        )["eom_lag1"][0]
        assert result == expected_eom_lag1
