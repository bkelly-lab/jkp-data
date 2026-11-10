"""Equivalence test: key-level add_primary_sec must reproduce the legacy DuckDB output.

add_primary_sec used to range-join the primary-issue histories onto the full rows, remove
the join fan-out with SELECT DISTINCT and DISTINCT ON, and stage the result in a .ddb
table. It now computes primary_sec on the key columns and joins it back. This test runs
the legacy SQL (verbatim) on the same inputs and asserts identical output, row order
included. Keys (gvkey, iid, datadate) are unique in the pipeline (census job 4187674 on
the Nov-1 data); the fixture adds rows with a NULL iid and a NULL datadate, which must
survive as the legacy SQL kept them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import add_primary_sec
from jkp.data.paths import DataPaths

# Verbatim SQL from the pre-rewrite add_primary_sec (connection moved to in-memory duckdb).
LEGACY_SQL = """
        CREATE OR REPLACE TABLE __data2 AS
        WITH data AS (
            SELECT * FROM read_parquet('{data_path_posix}')
            ORDER BY gvkey, iid, datadate
        ),
        prihistrow AS (
            SELECT * FROM read_parquet('{prihistrow_path}')
            ORDER BY gvkey, effdate
        ),
        prihistusa AS (
            SELECT * FROM read_parquet('{prihistusa_path}')
            ORDER BY gvkey, effdate
        ),
        prihistcan AS (
            SELECT * FROM read_parquet('{prihistcan_path}')
            ORDER BY gvkey, effdate
        ),
        header AS (
            SELECT * FROM read_parquet('{header_path}')
            ORDER BY gvkey
        ),
        __data1 AS (
            SELECT DISTINCT
                a.*,
                COALESCE(b.prihistrow, e.prirow)  AS prihistrow,
                COALESCE(c.prihistusa, e.priusa)  AS prihistusa,
                COALESCE(d.prihistcan, e.prican)  AS prihistcan
            FROM data AS a
            LEFT JOIN prihistrow AS b
                ON a.gvkey = b.gvkey
                AND a.{datevar} BETWEEN b.effdate AND COALESCE(b.thrudate, DATE '2262-04-11')
            LEFT JOIN prihistusa AS c
                ON a.gvkey = c.gvkey
                AND a.{datevar} BETWEEN c.effdate AND COALESCE(c.thrudate, DATE '2262-04-11')
            LEFT JOIN prihistcan AS d
                ON a.gvkey = d.gvkey
                AND a.{datevar} BETWEEN d.effdate AND COALESCE(d.thrudate, DATE '2262-04-11')
            LEFT JOIN header AS e
                ON a.gvkey = e.gvkey
        )
        SELECT
            * EXCLUDE (prihistrow, prihistusa, prihistcan),
            CASE
                WHEN iid IS NOT NULL
                    AND COALESCE(
                        (iid = prihistrow) OR (iid = prihistusa) OR (iid = prihistcan),
                        FALSE
                        )
                THEN 1
                ELSE 0
            END AS primary_sec
        FROM __data1
        ORDER BY gvkey, iid, datadate;

        COPY (
            SELECT DISTINCT ON (gvkey, iid, {datevar}) *
            FROM __data2
            ORDER BY gvkey, iid, {datevar}, primary_sec DESC
        ) TO '{file_name_posix}' (FORMAT parquet);
"""

DATES = [
    date(2019, 12, 31),
    date(2020, 2, 28),
    date(2020, 3, 31),
    date(2020, 5, 29),
    date(2020, 8, 31),
]


def _history(name: str, rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"gvkey": pl.Utf8, name: pl.Utf8, "effdate": pl.Date, "thrudate": pl.Date},
        orient="row",
    )


def _write_fixtures(paths: DataPaths) -> Path:
    """Histories, header and a shuffled security file covering every path of the rule:
    000001 has two overlapping ROW records with conflicting iids (both primary where they
    overlap), an open-ended record, a USA record with a NULL value (header fallback) and
    dates no ROW record covers (header fallback); 000002 has only a header with NULL
    entries; 000003 has a CAN record but no header; 000004 has neither."""
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    _history(
        "prihistrow",
        [
            ("000001", "01", date(2020, 1, 1), date(2020, 6, 30)),
            ("000001", "02", date(2020, 4, 1), None),
        ],
    ).write_parquet(raw_data_dfs / "__prihistrow.parquet")
    _history("prihistusa", [("000001", None, date(2020, 1, 1), date(2020, 12, 31))]).write_parquet(
        raw_data_dfs / "__prihistusa.parquet"
    )
    _history("prihistcan", [("000003", "01", date(2020, 3, 1), date(2020, 3, 31))]).write_parquet(
        raw_data_dfs / "__prihistcan.parquet"
    )
    pl.DataFrame(
        {
            "gvkey": ["000001", "000002"],
            "prirow": ["01", None],
            "priusa": ["03W", "01"],
            "prican": [None, None],
        }
    ).write_parquet(raw_data_dfs / "__header.parquet")

    securities = [
        ("000001", "01"),
        ("000001", "02"),
        ("000001", "03W"),
        ("000002", "01"),
        ("000002", "02"),
        ("000003", "01"),
        ("000004", "01"),
    ]
    rows = [(g, i, d) for g, i in securities for d in DATES]
    rows += [("000001", None, DATES[0]), ("000004", "01", None)]
    data = pl.DataFrame(
        rows, schema={"gvkey": pl.Utf8, "iid": pl.Utf8, "datadate": pl.Date}, orient="row"
    ).with_columns(
        prc=pl.int_range(pl.len()).cast(pl.Float64) * 1.5,
        curcdd=pl.lit("USD"),
        # columns add_primary_sec's duplicate-key rule reads
        prc_local=pl.lit(1.0),
        prcstd=pl.lit(10),
        exch_main=pl.lit(1),
        dolvol=pl.lit(1.0),
    )
    data = data.with_columns(prc=pl.when(pl.col("iid") == "02").then(None).otherwise("prc"))
    data_path = paths.interim_dir / "__comp_sf2.parquet"
    data.sample(fraction=1.0, shuffle=True, seed=7).write_parquet(data_path)
    return data_path


def _legacy_add_primary_sec(paths: DataPaths, data_path: Path, file_name: Path) -> None:
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    sql = LEGACY_SQL.format(
        data_path_posix=data_path.as_posix(),
        prihistrow_path=(raw_data_dfs / "__prihistrow.parquet").as_posix(),
        prihistusa_path=(raw_data_dfs / "__prihistusa.parquet").as_posix(),
        prihistcan_path=(raw_data_dfs / "__prihistcan.parquet").as_posix(),
        header_path=(raw_data_dfs / "__header.parquet").as_posix(),
        file_name_posix=file_name.as_posix(),
        datevar="datadate",
    )
    with duckdb.connect() as con:
        con.execute(sql)


def test_matches_legacy_sql(test_paths: DataPaths) -> None:
    data_path = _write_fixtures(test_paths)
    legacy_path = test_paths.interim_dir / "legacy.parquet"
    new_path = test_paths.interim_dir / "new.parquet"
    _legacy_add_primary_sec(test_paths, data_path, legacy_path)
    add_primary_sec(test_paths, data_path, "datadate", new_path)

    legacy, new = pl.read_parquet(legacy_path), pl.read_parquet(new_path)
    assert_frame_equal(new, legacy)
    # rows with a NULL key column are kept
    assert new["iid"].null_count() == 1
    assert new["datadate"].null_count() == 1
    # the fixtures reach both flag values and the overlap
    assert set(new["primary_sec"]) == {0, 1}
    overlap = new.filter((pl.col("gvkey") == "000001") & (pl.col("datadate") == date(2020, 5, 29)))
    assert overlap.filter(pl.col("iid").is_in(["01", "02"]))["primary_sec"].to_list() == [1, 1]
