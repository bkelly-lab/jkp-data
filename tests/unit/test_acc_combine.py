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
