"""Tests for combine_ann_qtr_chars (annual + quarterly accounting characteristics)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import combine_ann_qtr_chars
from jkp.data.paths import DataPaths


def _run(test_paths: DataPaths, ann: pl.DataFrame, qtr: pl.DataFrame) -> pl.DataFrame:
    ann_path = test_paths.interim_dir / "achars.parquet"
    qtr_path = test_paths.interim_dir / "qchars.parquet"
    ann.write_parquet(ann_path)
    qtr.write_parquet(qtr_path)
    combine_ann_qtr_chars(test_paths, ann_path, qtr_path, ["assets"], "_qitem")
    return pl.read_parquet(test_paths.interim_dir / "acc_chars_world.parquet")


def _ann(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "source": pl.String,
            "gvkey": pl.String,
            "public_date": pl.Date,
            "datadate": pl.Date,
            "assets": pl.Float64,
        },
        orient="row",
    )


def _qtr(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "source": pl.String,
            "gvkey": pl.String,
            "public_date": pl.Date,
            "datadate": pl.Date,
            "assets_qitem": pl.Float64,
        },
        orient="row",
    )


def _out(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "source": pl.String,
            "gvkey": pl.String,
            "public_date": pl.Date,
            "assets": pl.Float64,
        },
        orient="row",
    )


@pytest.mark.unit
def test_quarterly_only_month_is_kept(test_paths: DataPaths) -> None:
    """A month before the first annual report keeps its quarterly characteristics."""
    ann = _ann([("NA", "001", date(2016, 4, 30), date(2015, 12, 31), 100.0)])
    qtr = _qtr(
        [
            ("NA", "001", date(2015, 10, 31), date(2015, 6, 30), 80.0),
            ("NA", "001", date(2016, 4, 30), date(2015, 12, 31), 90.0),
        ]
    )

    expected = _out(
        [
            ("NA", "001", date(2015, 10, 31), 80.0),
            ("NA", "001", date(2016, 4, 30), 100.0),
        ]
    )
    assert_frame_equal(_run(test_paths, ann, qtr), expected)


@pytest.mark.unit
def test_shared_and_annual_only_months_unchanged(test_paths: DataPaths) -> None:
    """On shared months the newer quarterly value wins, source stays annual; annual-only months pass through."""
    ann = _ann(
        [
            ("NA", "001", date(2016, 4, 30), date(2015, 12, 31), 100.0),
            ("NA", "001", date(2016, 8, 31), date(2015, 12, 31), 100.0),
            ("GLOBAL", "002", date(2016, 4, 30), date(2015, 12, 31), 50.0),
        ]
    )
    qtr = _qtr(
        [
            ("NA", "001", date(2016, 4, 30), date(2015, 12, 31), 90.0),
            ("GLOBAL", "001", date(2016, 8, 31), date(2016, 3, 31), 110.0),
        ]
    )

    expected = _out(
        [
            ("NA", "001", date(2016, 4, 30), 100.0),
            ("NA", "001", date(2016, 8, 31), 110.0),
            ("GLOBAL", "002", date(2016, 4, 30), 50.0),
        ]
    )
    assert_frame_equal(_run(test_paths, ann, qtr), expected)


def _run_two_vars(test_paths: DataPaths, ann: pl.DataFrame, qtr: pl.DataFrame) -> pl.DataFrame:
    """Run combine_ann_qtr_chars on the two characteristics ``assets`` and ``sales``."""
    ann_path = test_paths.interim_dir / "achars2.parquet"
    qtr_path = test_paths.interim_dir / "qchars2.parquet"
    ann.write_parquet(ann_path)
    qtr.write_parquet(qtr_path)
    combine_ann_qtr_chars(test_paths, ann_path, qtr_path, ["assets", "sales"], "_qitem")
    return pl.read_parquet(test_paths.interim_dir / "acc_chars_world.parquet")


_ROW_SCHEMA_A = {
    "source": pl.String,
    "gvkey": pl.String,
    "public_date": pl.Date,
    "datadate": pl.Date,
    "assets": pl.Float64,
    "sales": pl.Float64,
}
_ROW_SCHEMA_Q = {
    "source": pl.String,
    "gvkey": pl.String,
    "public_date": pl.Date,
    "datadate": pl.Date,
    "assets_qitem": pl.Float64,
    "sales_qitem": pl.Float64,
}
_ROW_SCHEMA_OUT = {
    "source": pl.String,
    "gvkey": pl.String,
    "public_date": pl.Date,
    "assets": pl.Float64,
    "sales": pl.Float64,
}
_PD = date(2016, 8, 31)


@pytest.mark.unit
def test_precedence_per_variable(test_paths: DataPaths) -> None:
    """Each characteristic picks annual or quarterly on its own, never mixing the two within one variable.

    Annual datadate is 2015-12-31 in every row.
    - 001: quarterly newer (2016-03-31) and present: quarterly for both vars.
    - 002: quarterly datadate equal to annual: strictly-newer rule keeps annual.
    - 003: quarterly older (2015-06-30): annual.
    - 004: quarterly newer but null for assets, present for sales: assets stays annual, sales quarterly.
    - 005: annual assets null, quarterly older: the null annual is filled by the older quarterly value;
      annual sales also null and quarterly null: stays null.
    """
    newer, same, older = date(2016, 3, 31), date(2015, 12, 31), date(2015, 6, 30)
    ann = pl.DataFrame(
        [
            ("NA", "001", _PD, same, 100.0, 10.0),
            ("NA", "002", _PD, same, 100.0, 10.0),
            ("NA", "003", _PD, same, 100.0, 10.0),
            ("NA", "004", _PD, same, 100.0, 10.0),
            ("NA", "005", _PD, same, None, None),
        ],
        schema=_ROW_SCHEMA_A,
        orient="row",
    )
    qtr = pl.DataFrame(
        [
            ("NA", "001", _PD, newer, 90.0, 9.0),
            ("NA", "002", _PD, same, 90.0, 9.0),
            ("NA", "003", _PD, older, 90.0, 9.0),
            ("NA", "004", _PD, newer, None, 9.0),
            ("NA", "005", _PD, older, 90.0, None),
        ],
        schema=_ROW_SCHEMA_Q,
        orient="row",
    )
    expected = pl.DataFrame(
        [
            ("NA", "001", _PD, 90.0, 9.0),
            ("NA", "002", _PD, 100.0, 10.0),
            ("NA", "003", _PD, 100.0, 10.0),
            ("NA", "004", _PD, 100.0, 9.0),
            ("NA", "005", _PD, 90.0, None),
        ],
        schema=_ROW_SCHEMA_OUT,
        orient="row",
    )
    assert_frame_equal(_run_two_vars(test_paths, ann, qtr), expected)


@pytest.mark.unit
def test_null_annual_datadate_keeps_annual_value(test_paths: DataPaths) -> None:
    """With an unknown annual datadate the 'quarterly is more recent' test is null, so a present annual value stays.

    SAS would take the quarterly value (a missing date sorts first). The stage never builds such a
    row: every annual row has a datadate.
    """
    ann = pl.DataFrame([("NA", "001", _PD, None, 100.0, 10.0)], schema=_ROW_SCHEMA_A, orient="row")
    qtr = pl.DataFrame(
        [("NA", "001", _PD, date(2016, 3, 31), 90.0, 9.0)], schema=_ROW_SCHEMA_Q, orient="row"
    )
    expected = pl.DataFrame([("NA", "001", _PD, 100.0, 10.0)], schema=_ROW_SCHEMA_OUT, orient="row")
    assert_frame_equal(_run_two_vars(test_paths, ann, qtr), expected)


@pytest.mark.unit
def test_output_sorted_and_firms_kept_separate(test_paths: DataPaths) -> None:
    """Unsorted inputs come out sorted by (gvkey, public_date); the same month for two firms stays two rows."""
    d1, d2 = date(2016, 7, 31), date(2016, 8, 31)
    old = date(2015, 12, 31)
    ann = pl.DataFrame(
        [
            ("NA", "002", d2, old, 20.0, 2.0),
            ("NA", "001", d2, old, 10.0, 1.0),
            ("NA", "002", d1, old, 21.0, 2.1),
        ],
        schema=_ROW_SCHEMA_A,
        orient="row",
    )
    qtr = pl.DataFrame([("NA", "001", d1, old, 11.0, 1.1)], schema=_ROW_SCHEMA_Q, orient="row")
    expected = pl.DataFrame(
        [
            ("NA", "001", d1, 11.0, 1.1),
            ("NA", "001", d2, 10.0, 1.0),
            ("NA", "002", d1, 21.0, 2.1),
            ("NA", "002", d2, 20.0, 2.0),
        ],
        schema=_ROW_SCHEMA_OUT,
        orient="row",
    )
    assert_frame_equal(_run_two_vars(test_paths, ann, qtr), expected)


@pytest.mark.unit
def test_annual_source_wins_on_shared_month(test_paths: DataPaths) -> None:
    """The source label comes from the annual row when both exist, from the quarterly row when only it exists."""
    ann = pl.DataFrame(
        [("NA", "001", _PD, date(2015, 12, 31), 100.0, 10.0)], schema=_ROW_SCHEMA_A, orient="row"
    )
    qtr = pl.DataFrame(
        [
            ("GLOBAL", "001", _PD, date(2016, 3, 31), 90.0, 9.0),
            ("GLOBAL", "002", _PD, date(2016, 3, 31), 80.0, 8.0),
        ],
        schema=_ROW_SCHEMA_Q,
        orient="row",
    )
    out = _run_two_vars(test_paths, ann, qtr)
    assert out["gvkey"].to_list() == ["001", "002"]
    assert out["source"].to_list() == ["NA", "GLOBAL"]
