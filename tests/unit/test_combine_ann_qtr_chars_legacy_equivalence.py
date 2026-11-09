"""Equivalence tests: the DuckDB combine_ann_qtr_chars must reproduce the legacy output.

combine_ann_qtr_chars changed in two ways. A full outer join replaced the left join on the
annual panel, so firm-months with only quarterly characteristics are now kept, and one DuckDB
COPY replaced the old query. These tests run the legacy SQL on the same random inputs and assert
that every month the legacy code wrote is unchanged, and that every added month equals its
quarterly input row.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import combine_ann_qtr_chars
from jkp.data.paths import DataPaths

CHAR_VARS = ["c1", "c2", "c3"]
Q_SUFFIX = "_qitem"
KEY = ["gvkey", "public_date"]


def _legacy_combine(interim_dir: Path, ann_df_path: Path, qtr_df_path: Path) -> None:
    """Pre-rewrite combine_ann_qtr_chars: the legacy SQL for CHAR_VARS and Q_SUFFIX.

    Only the view names became read_parquet() calls and the table names ann/qtr.
    """
    ann_path, qtr_path = ann_df_path.as_posix(), qtr_df_path.as_posix()
    out = (interim_dir / "acc_chars_world.parquet").as_posix()
    (interim_dir / "aux_aqtr_chars.ddb").unlink(missing_ok=True)
    with duckdb.connect(str(interim_dir / "aux_aqtr_chars.ddb")) as con:
        con.execute(f"""
        CREATE TABLE ann AS
        SELECT "t0"."source", "t0"."gvkey", "t0"."public_date", "t0"."datadate",
          "t0"."c1", "t0"."c2", "t0"."c3",
          ROW_NUMBER() OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "n1"
        FROM read_parquet('{ann_path}') AS "t0";

        CREATE TABLE qtr AS
        SELECT "t1"."source" AS "source_qitem", "t1"."gvkey", "t1"."public_date",
          "t1"."datadate" AS "datadate_qitem", "t1"."c1_qitem", "t1"."c2_qitem", "t1"."c3_qitem",
          "t1"."n2"
        FROM (
          SELECT "t0"."source", "t0"."gvkey", "t0"."public_date", "t0"."datadate",
            "t0"."c1_qitem", "t0"."c2_qitem", "t0"."c3_qitem",
            ROW_NUMBER() OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "n2"
          FROM read_parquet('{qtr_path}') AS "t0"
        ) AS "t1";

        COPY (
        SELECT * FROM (
          SELECT "t7".* EXCLUDE ("n1", "n2") FROM (
            SELECT "t6"."source", "t6"."gvkey", "t6"."public_date", "t6"."c1", "t6"."c2",
              "t6"."c3", "t6"."n1", "t6"."n2"
            FROM (
              SELECT "t5"."gvkey", "t5"."public_date",
                FIRST("t5"."source") FILTER(WHERE "t5"."source" IS NOT NULL) AS "source",
                FIRST("t5"."c1") FILTER(WHERE "t5"."c1" IS NOT NULL) AS "c1",
                FIRST("t5"."c2") FILTER(WHERE "t5"."c2" IS NOT NULL) AS "c2",
                FIRST("t5"."c3") FILTER(WHERE "t5"."c3" IS NOT NULL) AS "c3",
                FIRST("t5"."n1") FILTER(WHERE "t5"."n1" IS NOT NULL) AS "n1",
                FIRST("t5"."n2") FILTER(WHERE "t5"."n2" IS NOT NULL) AS "n2"
              FROM (
                SELECT "t4"."source", "t4"."gvkey", "t4"."public_date",
                  CASE WHEN ("t4"."c1" IS NULL) OR (("t4"."c1_qitem" IS NOT NULL)
                    AND ("t4"."datadate_qitem" > "t4"."datadate"))
                    THEN "t4"."c1_qitem" ELSE "t4"."c1" END AS "c1",
                  CASE WHEN ("t4"."c2" IS NULL) OR (("t4"."c2_qitem" IS NOT NULL)
                    AND ("t4"."datadate_qitem" > "t4"."datadate"))
                    THEN "t4"."c2_qitem" ELSE "t4"."c2" END AS "c2",
                  CASE WHEN ("t4"."c3" IS NULL) OR (("t4"."c3_qitem" IS NOT NULL)
                    AND ("t4"."datadate_qitem" > "t4"."datadate"))
                    THEN "t4"."c3_qitem" ELSE "t4"."c3" END AS "c3",
                  "t4"."n1", "t4"."n2"
                FROM (
                  SELECT "t2"."source", "t2"."gvkey", "t2"."public_date", "t2"."datadate",
                    "t2"."c1", "t2"."c2", "t2"."c3", "t2"."n1", "t3"."source_qitem",
                    "t3"."gvkey" AS "gvkey_right", "t3"."public_date" AS "public_date_right",
                    "t3"."datadate_qitem", "t3"."c1_qitem", "t3"."c2_qitem", "t3"."c3_qitem",
                    "t3"."n2"
                  FROM "ann" AS "t2"
                  LEFT OUTER JOIN "qtr" AS "t3"
                    ON "t2"."gvkey" = "t3"."gvkey" AND "t2"."public_date" = "t3"."public_date"
                ) AS "t4"
                ORDER BY "t4"."gvkey" ASC, "t4"."public_date" ASC, "t4"."n1" ASC, "t4"."n2" ASC
              ) AS "t5"
              GROUP BY 1, 2
            ) AS "t6"
          ) AS "t7"
        ) AS "t8"
        ORDER BY "t8"."gvkey" ASC, "t8"."public_date" ASC
        ) TO '{out}' (FORMAT 'parquet')
        """)
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


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[pl.DataFrame, ...]:
    """Write random inputs, run the new and the legacy code once, return (ann, qtr, new, legacy)."""
    tmp_path = tmp_path_factory.mktemp("combine")
    test_paths = DataPaths(tmp_path / "data")
    test_paths.interim_dir.mkdir(parents=True)
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
    _legacy_combine(legacy_dir, ann_path, qtr_path)
    return (
        ann,
        qtr,
        pl.read_parquet(test_paths.interim_dir / "acc_chars_world.parquet"),
        pl.read_parquet(legacy_dir / "acc_chars_world.parquet"),
    )


@pytest.mark.unit
class TestCombineAnnQtrCharsEquivalence:
    """combine_ann_qtr_chars must match the legacy SQL on every month it wrote."""

    def test_annual_months_match_legacy(self, outputs: tuple[pl.DataFrame, ...]) -> None:
        ann, _, new, legacy = outputs
        assert_frame_equal(
            new.join(ann.select(KEY), on=KEY, how="semi").sort(KEY), legacy.sort(KEY)
        )

    def test_quarterly_only_months_equal_quarterly_input(
        self, outputs: tuple[pl.DataFrame, ...]
    ) -> None:
        ann, qtr, new, _ = outputs
        expected = (
            qtr.join(ann.select(KEY), on=KEY, how="anti")
            .select("source", *KEY, *[pl.col(f"{v}{Q_SUFFIX}").alias(v) for v in CHAR_VARS])
            .sort(KEY)
        )
        assert expected.height > 0
        assert_frame_equal(new.join(ann.select(KEY), on=KEY, how="anti").sort(KEY), expected)
