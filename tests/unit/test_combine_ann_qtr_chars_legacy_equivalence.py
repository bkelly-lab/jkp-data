"""Equivalence tests: the DuckDB combine_ann_qtr_chars must reproduce the legacy Ibis output.

combine_ann_qtr_chars changed in two ways. A full outer join replaced the left join on the
annual panel, so firm-months with only quarterly characteristics are now kept, and one DuckDB
COPY replaced the Ibis query. These tests run the legacy Ibis code (verbatim) on the same random
inputs and assert that every month the legacy code wrote is unchanged, and that every added
month equals its quarterly input row.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import ibis
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import combine_ann_qtr_chars
from jkp.data.paths import DataPaths

CHAR_VARS = ["c1", "c2", "c3"]
Q_SUFFIX = "_qitem"
KEY = ["gvkey", "public_date"]


def _legacy_combine(interim_dir: Path, ann_df_path, qtr_df_path, char_vars, q_suffix) -> None:
    """Pre-rewrite combine_ann_qtr_chars, verbatim except that paths.interim_dir is interim_dir."""
    (interim_dir / "aux_aqtr_chars.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(str(interim_dir / "aux_aqtr_chars.ddb"), threads=os.cpu_count())
    con.create_table(
        "ann",
        con.read_parquet(ann_df_path).mutate(n1=ibis.row_number()),
        overwrite=True,
    )
    con.create_table(
        "qtr",
        con.read_parquet(qtr_df_path)
        .mutate(n2=ibis.row_number())
        .rename({"datadate_qitem": "datadate", "source_qitem": "source"}),
        overwrite=True,
    )
    ann = con.table("ann")
    qtr = con.table("qtr")
    combined = ann.left_join(qtr, [ann.gvkey == qtr.gvkey, ann.public_date == qtr.public_date])
    drop_columns = [
        "datadate",
        f"datadate{q_suffix}",
        "gvkey_right",
        "public_date_right",
        "source_qitem",
    ] + [f"{ann_var}{q_suffix}" for ann_var in char_vars]
    subs = {}
    for ann_var in char_vars:
        qtr_var = f"{ann_var}{q_suffix}"
        subs[ann_var] = ibis.ifelse(
            combined[ann_var].isnull()
            | (combined[qtr_var].notnull() & (combined[f"datadate{q_suffix}"] > combined.datadate)),
            combined[qtr_var],
            combined[ann_var],
        )
    combined = (
        combined.mutate(subs)
        .drop(drop_columns)
        .order_by(["gvkey", "public_date", "n1", "n2"])
        .distinct(on=["gvkey", "public_date"], keep="first")
        .drop(["n1", "n2"])
        .order_by(["gvkey", "public_date"])
    )
    combined.to_parquet(interim_dir / "acc_chars_world.parquet")
    con.disconnect()
    (interim_dir / "aux_aqtr_chars.ddb").unlink(missing_ok=True)


def _random_side(
    rng: np.random.Generator, grid: pl.DataFrame, char_cols: list[str]
) -> pl.DataFrame:
    """Random ~60% of the firm-month grid with source, datadate and characteristic columns.

    Characteristics are ~20% null and ~5% NaN. datadate is public_date minus 120, 210 or 300
    days, so annual and quarterly datadates tie on about a third of shared months.
    """
    df = grid.filter(pl.Series(rng.random(grid.height) < 0.6))
    n = df.height
    chars = {}
    for c in char_cols:
        u = rng.random(n)
        chars[c] = pl.Series(
            [
                None if x < 0.2 else float("nan") if x < 0.25 else v
                for v, x in zip(rng.normal(size=n).tolist(), u, strict=True)
            ],
            dtype=pl.Float64,
        )
    return df.select(
        source=pl.Series(rng.choice(["NA", "GLOBAL"], n)),
        gvkey="gvkey",
        public_date="public_date",
        datadate=pl.col("public_date")
        - pl.duration(days=pl.Series(rng.choice([120, 210, 300], n))),
        **chars,
    )


def _outputs(test_paths: DataPaths, tmp_path: Path) -> tuple[pl.DataFrame, ...]:
    """Write random inputs, run the new and the legacy code, return (ann, qtr, new, legacy)."""
    rng = np.random.default_rng(0)
    months = pl.date_range(date(2015, 1, 1), date(2017, 12, 1), "1mo", eager=True).dt.month_end()
    grid = pl.DataFrame({"gvkey": [f"{i:06d}" for i in range(40)]}).join(
        pl.DataFrame({"public_date": months}), how="cross"
    )
    ann = _random_side(rng, grid, CHAR_VARS)
    qtr = _random_side(rng, grid, [f"{v}{Q_SUFFIX}" for v in CHAR_VARS])
    ann_path, qtr_path = tmp_path / "achars.parquet", tmp_path / "qchars.parquet"
    ann.write_parquet(ann_path)
    qtr.write_parquet(qtr_path)

    combine_ann_qtr_chars(test_paths, ann_path, qtr_path, CHAR_VARS, Q_SUFFIX)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    _legacy_combine(legacy_dir, ann_path, qtr_path, CHAR_VARS, Q_SUFFIX)
    return (
        ann,
        qtr,
        pl.read_parquet(test_paths.interim_dir / "acc_chars_world.parquet"),
        pl.read_parquet(legacy_dir / "acc_chars_world.parquet"),
    )


@pytest.mark.unit
class TestCombineAnnQtrCharsEquivalence:
    """combine_ann_qtr_chars must match the legacy Ibis code on every month it wrote."""

    def test_annual_months_match_legacy(self, test_paths: DataPaths, tmp_path: Path) -> None:
        ann, _, new, legacy = _outputs(test_paths, tmp_path)
        assert_frame_equal(
            new.join(ann.select(KEY), on=KEY, how="semi").sort(KEY), legacy.sort(KEY)
        )

    def test_quarterly_only_months_equal_quarterly_input(
        self, test_paths: DataPaths, tmp_path: Path
    ) -> None:
        ann, qtr, new, _ = _outputs(test_paths, tmp_path)
        expected = (
            qtr.join(ann.select(KEY), on=KEY, how="anti")
            .select("source", *KEY, *[pl.col(f"{v}{Q_SUFFIX}").alias(v) for v in CHAR_VARS])
            .sort(KEY)
        )
        assert expected.height > 0
        assert_frame_equal(new.join(ann.select(KEY), on=KEY, how="anti").sort(KEY), expected)
