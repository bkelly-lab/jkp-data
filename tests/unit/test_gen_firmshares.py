"""Tests for gen_firmshares().

The function stacks Compustat NA quarterly (FUNDQ) and annual (FUNDA) shares outstanding
and adjustment factors, and keeps one record per (gvkey, datadate). When both files report
the same date, FUNDQ wins, matching the SAS union order and stable nodupkey sort. When a
fiscal year-end change leaves two FUNDQ records on one date, the one with a fiscal quarter
wins, then the highest fyr, the same rule the quarterly accounting panel uses.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import gen_firmshares
from jkp.data.paths import DataPaths
from tests.conftest import assert_unique_keys

_STD = {"indfmt": "INDL", "datafmt": "STD", "popsrc": "D", "consol": "C"}
_FILTER_SCHEMA = {
    "gvkey": pl.Utf8,
    "datadate": pl.Date,
    "indfmt": pl.Utf8,
    "datafmt": pl.Utf8,
    "popsrc": pl.Utf8,
    "consol": pl.Utf8,
    "fyr": pl.Int64,
}


def _fundq(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [{**_STD, "fqtr": 4, **r} for r in rows],
        schema={**_FILTER_SCHEMA, "fqtr": pl.Int64, "cshoq": pl.Float64, "ajexq": pl.Float64},
    )


def _funda(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [{**_STD, "at": 1.0, **r} for r in rows],
        schema={**_FILTER_SCHEMA, "csho": pl.Float64, "ajex": pl.Float64, "at": pl.Float64},
    )


class TestGenFirmshares:
    """Tests for ``gen_firmshares``."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths

    def _run(self, fundq: pl.DataFrame, funda: pl.DataFrame) -> pl.DataFrame:
        fundq.write_parquet(self.paths.raw_tables_dir / "comp_fundq.parquet")
        funda.write_parquet(self.paths.raw_tables_dir / "comp_funda.parquet")
        gen_firmshares(self.paths)
        return pl.read_parquet(
            self.paths.interim_dir / "raw_data_dfs" / "__firm_shares1.parquet"
        ).sort("gvkey", "datadate")

    def test_fundq_wins_over_funda(self) -> None:
        """FUNDQ and FUNDA disagree on the same (gvkey, datadate): the FUNDQ value is kept."""
        d = date(2020, 12, 31)
        result = self._run(
            _fundq([{"gvkey": "001000", "datadate": d, "cshoq": 10.0, "ajexq": 1.0, "fyr": 12}]),
            _funda([{"gvkey": "001000", "datadate": d, "csho": 12.0, "ajex": 2.0, "fyr": 12}]),
        )
        assert result.height == 1
        row = result.row(0, named=True)
        assert (row["csho_fund"], row["ajex_fund"]) == (10.0, 1.0), f"Expected FUNDQ values: {row}"

    def test_fiscal_year_end_change_keeps_highest_fyr(self) -> None:
        """Two FUNDQ records on one date (fyr 6 and 12): the fyr 12 record is kept.

        The fyr 12 row is written second so that a keep-first rule would fail.
        """
        d = date(2021, 6, 30)
        result = self._run(
            _fundq(
                [
                    {"gvkey": "001000", "datadate": d, "cshoq": 5.0, "ajexq": 1.0, "fyr": 6},
                    {"gvkey": "001000", "datadate": d, "cshoq": 7.0, "ajexq": 1.0, "fyr": 12},
                ]
            ),
            _funda([{"gvkey": "001000", "datadate": d, "csho": 9.0, "ajex": 1.0, "fyr": 6}]),
        )
        assert result.height == 1
        assert result["csho_fund"].item() == 7.0, f"Expected the fyr 12 FUNDQ value: {result}"

    def test_fiscal_year_end_change_prefers_record_with_fiscal_quarter(self) -> None:
        """Two FUNDQ records on one date: one with a fiscal quarter beats a higher-fyr one without."""
        d = date(2021, 6, 30)
        result = self._run(
            _fundq(
                [
                    {"gvkey": "001000", "datadate": d, "cshoq": 5.0, "ajexq": 1.0, "fyr": 6},
                    {
                        "gvkey": "001000",
                        "datadate": d,
                        "cshoq": 7.0,
                        "ajexq": 1.0,
                        "fyr": 12,
                        "fqtr": None,
                    },
                ]
            ),
            _funda([]),
        )
        assert result["csho_fund"].to_list() == [5.0], f"Expected the record with fqtr: {result}"

    def test_untied_rows_pass_through(self) -> None:
        """Dates reported by only one file keep that file's values."""
        result = self._run(
            _fundq(
                [
                    {
                        "gvkey": "001000",
                        "datadate": date(2020, 3, 31),
                        "cshoq": 1.0,
                        "ajexq": 1.0,
                        "fyr": 12,
                    }
                ]
            ),
            _funda(
                [
                    {
                        "gvkey": "001000",
                        "datadate": date(2019, 12, 31),
                        "csho": 2.0,
                        "ajex": 3.0,
                        "fyr": 12,
                    }
                ]
            ),
        )
        assert result.select("datadate", "csho_fund", "ajex_fund").rows() == [
            (date(2019, 12, 31), 2.0, 3.0),
            (date(2020, 3, 31), 1.0, 1.0),
        ]

    def test_filters_drop_non_standard_and_null_rows(self) -> None:
        """Rows outside INDL/STD/D/C, or with null shares or adjustment factor, are dropped."""
        d = date(2020, 12, 31)
        result = self._run(
            _fundq(
                [
                    {
                        "gvkey": "001000",
                        "datadate": d,
                        "cshoq": 1.0,
                        "ajexq": 1.0,
                        "fyr": 12,
                        "indfmt": "FS",
                    },
                    {"gvkey": "001001", "datadate": d, "cshoq": None, "ajexq": 1.0, "fyr": 12},
                ]
            ),
            _funda(
                [
                    {"gvkey": "001002", "datadate": d, "csho": 1.0, "ajex": None, "fyr": 12},
                    {
                        "gvkey": "001003",
                        "datadate": d,
                        "csho": 1.0,
                        "ajex": 1.0,
                        "fyr": 12,
                        "consol": "P",
                    },
                    {"gvkey": "001004", "datadate": d, "csho": 4.0, "ajex": 1.0, "fyr": 12},
                ]
            ),
        )
        assert result["gvkey"].to_list() == ["001004"]

    def test_output_unique_on_gvkey_datadate(self) -> None:
        """Every (gvkey, datadate) appears once, whatever mix of sources reports it."""
        d1, d2 = date(2020, 6, 30), date(2020, 12, 31)
        result = self._run(
            _fundq(
                [
                    {"gvkey": "001000", "datadate": d1, "cshoq": 1.0, "ajexq": 1.0, "fyr": 6},
                    {"gvkey": "001000", "datadate": d1, "cshoq": 2.0, "ajexq": 1.0, "fyr": 12},
                    {"gvkey": "001000", "datadate": d2, "cshoq": 3.0, "ajexq": 1.0, "fyr": 12},
                ]
            ),
            _funda(
                [
                    {"gvkey": "001000", "datadate": d1, "csho": 4.0, "ajex": 1.0, "fyr": 6},
                    {"gvkey": "001000", "datadate": d2, "csho": 5.0, "ajex": 1.0, "fyr": 12},
                ]
            ),
        )
        assert_unique_keys(result, ["gvkey", "datadate"])
        assert result["csho_fund"].to_list() == [2.0, 3.0]
