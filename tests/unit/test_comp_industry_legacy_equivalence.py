"""Equivalence tests: pure-Polars comp_industry must reproduce the legacy DuckDB output.

comp_industry was rewritten from a disk-backed DuckDB SQL pipeline to lazy Polars
for speed. These tests run the legacy SQL (verbatim) against the same intermediate
inputs and assert the new implementation produces identical rows, including the
null-coded interior gap days the legacy gap-fill emitted.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import comp_industry
from jkp.data.paths import DataPaths

# Verbatim SQL from the pre-rewrite comp_industry (only table creation moved to
# plain duckdb from ibis).
LEGACY_SQL = """
            DROP TABLE IF EXISTS join_table;
            CREATE TABLE join_table AS
            SELECT          *,
                            COALESCE( LEAD(date) OVER (PARTITION BY gvkey ORDER BY date) - INTERVAL '1 day', date )::DATE AS aux_date
            FROM            comp_gics
            FULL OUTER JOIN comp_other
            USING           (gvkey, date);

            DROP TABLE IF EXISTS gap_dates;
            CREATE TABLE gap_dates AS
            SELECT *
            FROM join_table
            WHERE date <> aux_date;

            DROP TABLE IF EXISTS gaps;
            CREATE TABLE gaps AS
            WITH full_span AS (
            SELECT
                j.gvkey, gs.gap_date::DATE AS date,
                FROM gap_dates as j
                CROSS JOIN LATERAL
                generate_series(j.date, j.aux_date, INTERVAL '1 day') AS gs(gap_date)
                ORDER BY gvkey, date
            )
            SELECT
            fs.gvkey, fs.date, gd.gics, gd.sic, gd.naics
            FROM full_span fs
            LEFT JOIN gap_dates gd
            ON gd.gvkey = fs.gvkey
            AND gd.date  = fs.date
            ORDER BY fs.gvkey, fs.date;

            DROP TABLE IF EXISTS continuous;
            CREATE TABLE continuous AS
            SELECT *
            FROM join_table
            WHERE date = aux_date;

            DROP TABLE IF EXISTS merged_data;
            CREATE TABLE merged_data AS
            SELECT gvkey, date, gics, sic, naics FROM continuous
            UNION
            SELECT gvkey, date, gics, sic, naics FROM gaps;

            DROP TABLE IF EXISTS comp_industry;
            CREATE TABLE comp_industry AS
            SELECT DISTINCT ON (gvkey, date)
                *
            FROM merged_data
            ORDER BY (gvkey, date);
"""


def _legacy_comp_industry(interim_dir: Path) -> pl.DataFrame:
    """Run the pre-rewrite DuckDB pipeline on the intermediate parquets."""
    con = duckdb.connect()
    try:
        gics_path = (interim_dir / "comp_hgics.parquet").as_posix()
        other_path = (interim_dir / "comp_other.parquet").as_posix()
        con.execute(f"CREATE TABLE comp_gics AS SELECT * FROM read_parquet('{gics_path}')")
        con.execute(f"CREATE TABLE comp_other AS SELECT * FROM read_parquet('{other_path}')")
        con.execute(LEGACY_SQL)
        return con.execute("SELECT * FROM comp_industry").pl()
    finally:
        con.close()


def _write_raw_fixtures(raw_data_dfs: Path) -> None:
    """Synthetic GICS histories and SIC/NAICS records exercising the gap-fill paths.

    Covers: multi-interval gvkeys with gaps, open-ended (null indthru) terminal
    records, gvkeys present in only one source, misaligned gics vs sic dates, and
    single-record gvkeys. Intervals within each library do not overlap, since
    overlapping (gvkey, date) keys are resolved nondeterministically (unique
    keep="any") by both old and new code.
    """
    pl.DataFrame(
        {
            "gvkey": ["000001", "000001", "000003", "000006"],
            "indfrom": [date(2020, 1, 1), date(2020, 1, 20), date(2020, 3, 1), None],
            "indthru": [date(2020, 1, 10), date(2020, 2, 5), None, date(2020, 5, 1)],
            "gics": [10101010, 20202020, None, 50505050],
        }
    ).write_parquet(raw_data_dfs / "comp_hgics_na.parquet")
    pl.DataFrame(
        {
            "gvkey": ["000001", "000004"],
            "indfrom": [date(2020, 1, 5), date(2020, 2, 10)],
            "indthru": [date(2020, 1, 8), date(2020, 2, 12)],
            "gics": [30303030, 40404040],
        }
    ).write_parquet(raw_data_dfs / "comp_hgics_gl.parquet")
    pl.DataFrame(
        {
            "gvkey": ["000001", "000001", "000002"],
            "datadate": [date(2020, 1, 5), date(2020, 1, 25), date(2020, 2, 1)],
            "sic": [3711, None, 6020],
            "naics": [336111, 336112, 522110],
        }
    ).write_parquet(raw_data_dfs / "sic_naics_na.parquet")
    pl.DataFrame(
        {
            "gvkey": ["000002", "000005"],
            "datadate": [date(2020, 2, 3), date(2020, 4, 15)],
            "sic": [6021, 2834],
            "naics": [522120, 325412],
        }
    ).write_parquet(raw_data_dfs / "sic_naics_gl.parquet")


@pytest.fixture(scope="module")
def comp_ind_paths(tmp_path_factory: pytest.TempPathFactory) -> DataPaths:
    """Write the synthetic raw fixtures and run comp_industry once for the module."""
    paths = DataPaths(tmp_path_factory.mktemp("comp_ind"))
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    raw_data_dfs.mkdir(parents=True)
    _write_raw_fixtures(raw_data_dfs)
    comp_industry(paths)
    return paths


@pytest.mark.unit
class TestCompIndustryEquivalence:
    """comp_industry rewrite must match the legacy DuckDB SQL row-for-row."""

    def test_matches_legacy_duckdb_output(self, comp_ind_paths: DataPaths) -> None:
        new = pl.read_parquet(comp_ind_paths.interim_dir / "comp_ind.parquet")
        legacy = _legacy_comp_industry(comp_ind_paths.interim_dir)

        sort_cols = ["gvkey", "date"]
        assert_frame_equal(
            new.sort(sort_cols).select(legacy.columns),
            legacy.sort(sort_cols),
            check_dtypes=False,
        )

    def test_no_duplicate_keys(self, comp_ind_paths: DataPaths) -> None:
        """DISTINCT ON in the legacy SQL was a no-op; the rewrite must not introduce dups."""
        new = pl.read_parquet(comp_ind_paths.interim_dir / "comp_ind.parquet")
        assert new.height == new.unique(["gvkey", "date"]).height
