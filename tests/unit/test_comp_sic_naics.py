"""Tests for ``comp_sic_naics`` (Issue #155).

Covers the daily SIC/NAICS expansion from Compustat NA + Global histories:

- Full outer join between ``sic_naics_na`` and ``sic_naics_gl`` on ``(gvkey, datadate)``
- Coalesce precedence: non-null SIC wins (DuckDB ``DISTINCT ON ... ORDER BY ... sic``)
- Hard-coded filter removing ``gvkey='175650'`` / ``datadate=2005-12-31`` / null naics
- ``LPAD(gvkey, 6, '0')`` zero-padding
- Daily expansion of [datadate, next datadate) gaps with ``closed="left"``
- Dedup + sort by ``(gvkey, date)``
- A regression golden fixture locking the output bit-for-bit.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import comp_sic_naics
from jkp.data.paths import DataPaths

GOLDEN_DIR = Path(__file__).parent.parent / "golden" / "fixtures" / "comp_sic_naics"


def _write_inputs(paths: DataPaths, na: pl.DataFrame, gl: pl.DataFrame) -> None:
    """Persist NA/GL fixtures in the locations ``comp_sic_naics`` reads from."""
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    raw_data_dfs.mkdir(parents=True, exist_ok=True)
    na.write_parquet(raw_data_dfs / "sic_naics_na.parquet")
    gl.write_parquet(raw_data_dfs / "sic_naics_gl.parquet")


def _empty_input() -> pl.DataFrame:
    """Return an empty NA/GL frame with the expected schema."""
    return pl.DataFrame(
        schema={
            "gvkey": pl.Utf8,
            "datadate": pl.Date,
            "sic": pl.Int64,
            "naics": pl.Int64,
        }
    )


class TestCompSicNaics:
    """Tests for ``comp_sic_naics``."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths
        self.output_path = self.paths.interim_dir / "comp_other.parquet"

    def test_na_only_gvkey(self) -> None:
        """A gvkey present only in NA appears in the output with its NA codes."""
        na = pl.DataFrame(
            {
                "gvkey": ["001000"],
                "datadate": [date(2020, 1, 1)],
                "sic": [7372],
                "naics": [511210],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, na, _empty_input())

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.height == 1
        row = result.row(0, named=True)
        assert row["gvkey"] == "001000"
        assert row["sic"] == 7372
        assert row["naics"] == 511210

    def test_gl_only_gvkey(self) -> None:
        """A gvkey present only in GL appears in the output with its GL codes."""
        gl = pl.DataFrame(
            {
                "gvkey": ["002000"],
                "datadate": [date(2020, 6, 15)],
                "sic": [2834],
                "naics": [325412],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, _empty_input(), gl)

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.height == 1
        row = result.row(0, named=True)
        assert row["sic"] == 2834
        assert row["naics"] == 325412

    def test_coalesce_prefers_non_null_sic(self) -> None:
        """When NA has null SIC and GL has a real one, the non-null value wins."""
        na = pl.DataFrame(
            {
                "gvkey": ["004000"],
                "datadate": [date(2019, 3, 1)],
                "sic": [None],
                "naics": [541110],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        gl = pl.DataFrame(
            {
                "gvkey": ["004000"],
                "datadate": [date(2019, 3, 1)],
                "sic": [4813],
                "naics": [517110],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, na, gl)

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.height == 1
        row = result.row(0, named=True)
        # Non-null SIC from GL wins because DISTINCT ON sorts sic DESC and
        # COALESCE(sica, sicb) brings the GL value in when NA is null.
        assert row["sic"] == 4813

    def test_hardcoded_175650_row_dropped(self) -> None:
        """The hard-coded NA filter removes ``(175650, 2005-12-31, naics=NULL)``."""
        na = pl.DataFrame(
            {
                "gvkey": ["175650", "175650"],
                "datadate": [date(2005, 12, 31), date(2006, 6, 30)],
                "sic": [1311, 1311],
                "naics": [None, 211120],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, na, _empty_input())

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        # The 2005-12-31 row was dropped; only the 2006-06-30 row remains and
        # (since it is the only/last row per gvkey) is kept as a single-date row.
        assert result.height == 1
        row = result.row(0, named=True)
        assert row["date"] == date(2006, 6, 30)
        assert row["naics"] == 211120

    def test_gvkey_zero_padded_to_six_chars(self) -> None:
        """Unpadded gvkey inputs are ``LPAD``-ed to width 6 in the output."""
        na = pl.DataFrame(
            {
                "gvkey": ["500"],
                "datadate": [date(2021, 7, 15)],
                "sic": [7372],
                "naics": [511210],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, na, _empty_input())

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.height == 1
        assert result["gvkey"].to_list() == ["000500"]

    def test_daily_expansion_closed_left(self) -> None:
        """Two consecutive datadates fill the half-open interval [d1, d2)."""
        na = pl.DataFrame(
            {
                "gvkey": ["001000", "001000"],
                "datadate": [date(2020, 1, 1), date(2020, 1, 4)],
                "sic": [3711, 3713],
                "naics": [336111, 336112],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, na, _empty_input())

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path).sort("date")
        # First span Jan-1 -> Jan-4 (closed="left") = Jan-1, Jan-2, Jan-3 (3 rows)
        # plus the trailing single-date row at Jan-4 = 4 rows total.
        assert result.height == 4
        assert result["date"].to_list() == [
            date(2020, 1, 1),
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 4),
        ]
        # First three rows carry the Jan-1 codes; the Jan-4 row carries Jan-4 codes.
        first_three = result.filter(pl.col("date") < date(2020, 1, 4))
        assert first_three["sic"].unique().to_list() == [3711]
        last = result.filter(pl.col("date") == date(2020, 1, 4)).row(0, named=True)
        assert last["sic"] == 3713

    def test_distinct_on_picks_lowest_sic(self) -> None:
        """When duplicate (gvkey, datadate) rows have different SICs, the lowest wins.

        The SQL uses ``DISTINCT ON (gvkey, date) ... ORDER BY gvkey, date, sic``
        (ascending), so among rows sharing the same (gvkey, date) after
        COALESCE, the row with the smallest SIC is retained.
        """
        na = pl.DataFrame(
            {
                "gvkey": ["005000", "005000"],
                "datadate": [date(2020, 1, 1), date(2020, 1, 1)],
                "sic": [9000, 1000],
                "naics": [999999, 111111],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, na, _empty_input())

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.height == 1
        row = result.row(0, named=True)
        assert row["sic"] == 1000, (
            "DISTINCT ON ... ORDER BY sic (ascending) must keep the lowest SIC"
        )
        assert row["naics"] == 111111

    def test_dedup_and_sort_invariants(self) -> None:
        """Output has unique ``(gvkey, date)`` keys and is sorted ascending."""
        na = pl.DataFrame(
            {
                "gvkey": ["002000", "001000"],
                "datadate": [date(2020, 6, 15), date(2020, 1, 1)],
                "sic": [2834, 7372],
                "naics": [325412, 511210],
            },
            schema={
                "gvkey": pl.Utf8,
                "datadate": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )
        _write_inputs(self.paths, na, _empty_input())

        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.unique(["gvkey", "date"]).height == result.height
        gvkeys = result["gvkey"].to_list()
        assert gvkeys == sorted(gvkeys)

    @pytest.mark.regression
    def test_comp_sic_naics_golden_fixture(self) -> None:
        """Bit-identical match against the locked golden fixture."""
        from tests.golden.generate_comp_sic_naics_golden import build_sic_naics_inputs

        na, gl = build_sic_naics_inputs(seed=42)
        _write_inputs(self.paths, na, gl)
        comp_sic_naics(self.paths)

        result = pl.read_parquet(self.output_path)
        golden = pl.read_parquet(GOLDEN_DIR / "comp_other.parquet")
        assert_frame_equal(result, golden, check_exact=True)
