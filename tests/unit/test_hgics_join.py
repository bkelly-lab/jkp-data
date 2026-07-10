"""Tests for hgics_join() (Issue #155).

Covers the merge of national + global Compustat GICS daily panels:

- Coalesce precedence: national GICS wins over global on shared ``(gvkey, date)``
- National-only and global-only gvkeys are both preserved
- The ``-999`` sentinel produced by ``comp_hgics`` for null gics survives the join
- Date-range expansion through the upstream ``comp_hgics`` call
- Dedup + sort by ``(gvkey, date)``
- Wall-clock independence (``@pytest.mark.regression``): two runs with different
  patched ``date.today()`` must produce bit-identical output, guarding the
  END_DATE-driven open-ended ``indthru`` behavior inherited from ``comp_hgics``
  (Issue #131 regression).
- A regression golden fixture locking the output bit-for-bit.
"""

from __future__ import annotations

import datetime as _dt
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import jkp.data.aux_functions as aux_functions
from jkp.data.aux_functions import hgics_join
from jkp.data.paths import DataPaths
from tests.conftest import assert_sorted_by_keys, assert_unique_keys

GOLDEN_DIR = Path(__file__).parent.parent / "golden" / "fixtures" / "hgics_join"


def _write_inputs(paths: DataPaths, na: pl.DataFrame, gl: pl.DataFrame) -> None:
    """Persist NA/GL GICS fixtures in the locations ``comp_hgics`` reads from."""
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    raw_data_dfs.mkdir(parents=True, exist_ok=True)
    na.write_parquet(raw_data_dfs / "comp_hgics_na.parquet")
    gl.write_parquet(raw_data_dfs / "comp_hgics_gl.parquet")


def _hgics_frame(
    gvkeys: list[str],
    indfroms: list[_dt.date],
    indthrus: list[_dt.date | None],
    gicses: list[int | None],
) -> pl.DataFrame:
    """Build a comp_hgics_(na|gl) fixture with the expected schema."""
    return pl.DataFrame(
        {
            "gvkey": gvkeys,
            "indfrom": indfroms,
            "indthru": indthrus,
            "gics": gicses,
        },
        schema={
            "gvkey": pl.Utf8,
            "indfrom": pl.Date,
            "indthru": pl.Date,
            "gics": pl.Int64,
        },
    )


def _empty_frame() -> pl.DataFrame:
    """Typed 0-row frame for the unused side in single-source tests."""
    return _hgics_frame([], [], [], [])


def _date_subclass_returning(today_value: _dt.date) -> type:
    """Build a ``date`` subclass whose ``today()`` returns a fixed value.

    Mirrors the pattern used in tests/unit/test_comp_hgics.py: subclassing
    keeps ``pl.lit(date.today())`` happy because Polars still sees a
    ``datetime.date`` instance.
    """

    class _FixedDate(_dt.date):
        @classmethod
        def today(cls) -> _dt.date:
            return today_value

    return _FixedDate


class TestHgicsJoin:
    """Tests for ``hgics_join``."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths
        self.output_path = self.paths.interim_dir / "comp_hgics.parquet"

    def test_national_only_gvkey_preserved(self) -> None:
        """A gvkey present only in NA carries its NA gics through the join."""
        na = _hgics_frame(["100000"], [date(2020, 1, 1)], [date(2020, 1, 2)], [10101010])
        gl = _empty_frame()
        _write_inputs(self.paths, na, gl)

        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path).sort(["gvkey", "date"])
        assert result.height == 2, f"Expected 2 rows (2-day span), got {result.height}"
        assert result["gvkey"].unique().to_list() == ["100000"]
        assert result["gics"].unique().to_list() == [10101010], (
            f"Expected NA gics=10101010 preserved, got {result['gics'].unique().to_list()}"
        )

    def test_global_only_gvkey_preserved(self) -> None:
        """A gvkey present only in GL carries its GL gics through the join."""
        na = _empty_frame()
        gl = _hgics_frame(["200000"], [date(2020, 2, 1)], [date(2020, 2, 2)], [20202020])
        _write_inputs(self.paths, na, gl)

        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path).sort(["gvkey", "date"])
        assert result.height == 2, f"Expected 2 rows (2-day span), got {result.height}"
        assert result["gvkey"].unique().to_list() == ["200000"]
        assert result["gics"].unique().to_list() == [20202020], (
            f"Expected GL gics=20202020 preserved, got {result['gics'].unique().to_list()}"
        )

    def test_national_wins_coalesce(self) -> None:
        """When both NA and GL have a row, national gics wins via coalesce."""
        na = _hgics_frame(["300000"], [date(2020, 3, 1)], [date(2020, 3, 2)], [30303030])
        gl = _hgics_frame(["300000"], [date(2020, 3, 1)], [date(2020, 3, 2)], [39393939])
        _write_inputs(self.paths, na, gl)

        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path).sort("date")
        assert result.height == 2, f"Expected 2 rows (2-day span), got {result.height}"
        assert result["gics"].unique().to_list() == [30303030], (
            f"National gics must take precedence over global gics, "
            f"got {result['gics'].unique().to_list()}"
        )

    def test_minus_999_sentinel_passthrough(self) -> None:
        """Null gics in input → -999 in ``comp_hgics`` output → -999 after join."""
        na = _hgics_frame(["400000"], [date(2020, 4, 1)], [date(2020, 4, 2)], [None])
        gl = _empty_frame()
        _write_inputs(self.paths, na, gl)

        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.height == 2, f"Expected 2 rows (2-day span), got {result.height}"
        assert result["gics"].unique().to_list() == [-999], (
            f"Expected -999 sentinel for null gics, got {result['gics'].unique().to_list()}"
        )

    def test_date_range_expansion(self) -> None:
        """``indfrom`` → ``indthru`` is exploded into one row per calendar day."""
        na = _hgics_frame(["100000"], [date(2020, 1, 1)], [date(2020, 1, 5)], [10101010])
        gl = _empty_frame()
        _write_inputs(self.paths, na, gl)

        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path).sort("date")
        assert result.height == 5, f"Expected 5 rows (5-day span), got {result.height}"
        expected_dates = [
            date(2020, 1, 1),
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 4),
            date(2020, 1, 5),
        ]
        assert result["date"].to_list() == expected_dates, (
            f"Expected contiguous dates {expected_dates}, got {result['date'].to_list()}"
        )

    def test_dedup_and_sort_invariants(self) -> None:
        """Output has unique ``(gvkey, date)`` keys and is sorted ascending."""
        na = _hgics_frame(
            ["100000", "100000"],
            [date(2020, 1, 1), date(2020, 1, 3)],
            [date(2020, 1, 2), date(2020, 1, 4)],
            [10101010, 10101011],
        )
        gl = _hgics_frame(["200000"], [date(2020, 2, 1)], [date(2020, 2, 2)], [20202020])
        _write_inputs(self.paths, na, gl)

        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path)
        assert_unique_keys(result, ["gvkey", "date"])
        assert_sorted_by_keys(result, "gvkey", "date")

    def test_dedup_with_duplicate_expanded_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Duplicate (gvkey, date) rows from upstream are collapsed by .unique().

        In production, ``comp_hgics`` deduplicates its own output, but if that
        guarantee were violated the full join could produce duplicate
        (gvkey, date) rows.  We bypass ``comp_hgics`` and write pre-expanded
        intermediates with a deliberate duplicate to verify that ``hgics_join``'s
        own ``.unique()`` catches it.
        """
        monkeypatch.setattr(aux_functions, "comp_hgics", lambda _paths, _lib: None)

        na_expanded = pl.DataFrame(
            {
                "gvkey": ["100000", "100000"],
                "date": [date(2020, 1, 1), date(2020, 1, 1)],
                "gics": [10101010, 10101010],
            },
            schema={"gvkey": pl.Utf8, "date": pl.Date, "gics": pl.Int64},
        )
        gl_expanded = pl.DataFrame(
            {
                "gvkey": ["200000"],
                "date": [date(2020, 2, 1)],
                "gics": [20202020],
            },
            schema={"gvkey": pl.Utf8, "date": pl.Date, "gics": pl.Int64},
        )
        self.paths.interim_dir.mkdir(parents=True, exist_ok=True)
        na_expanded.write_parquet(self.paths.interim_dir / "na_hgics.parquet")
        gl_expanded.write_parquet(self.paths.interim_dir / "g_hgics.parquet")

        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path)
        assert_unique_keys(result, ["gvkey", "date"])

    @pytest.mark.regression
    def test_independent_of_wall_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``hgics_join`` output must not depend on ``date.today()``.

        ``hgics_join`` calls ``comp_hgics`` (twice), and ``comp_hgics`` reads
        ``END_DATE`` to fill open-ended ``indthru``. Issue #131 fixed an earlier
        ``date.today()`` dependency in ``comp_hgics``; this test guards
        ``hgics_join`` against any re-introduction by patching the wall-clock
        ``date.today()`` to two distant values and asserting bit-identical output.
        """
        na = _hgics_frame(
            ["100000", "100000"],
            [date(2010, 1, 1), date(2015, 6, 1)],
            [date(2015, 5, 31), None],  # Open-ended → triggers END_DATE path
            [10101010, 10101020],
        )
        gl = _hgics_frame(["200000"], [date(2018, 3, 1)], [None], [20202020])
        _write_inputs(self.paths, na, gl)

        monkeypatch.setattr(aux_functions, "date", _date_subclass_returning(_dt.date(2024, 1, 15)))
        hgics_join(self.paths)
        first = pl.read_parquet(self.output_path)

        self.output_path.unlink()

        monkeypatch.setattr(aux_functions, "date", _date_subclass_returning(_dt.date(2030, 7, 15)))
        hgics_join(self.paths)
        second = pl.read_parquet(self.output_path)

        assert first.equals(second), (
            "hgics_join output must not depend on the wall-clock date; two runs "
            "with different patched date.today() values produced different output, "
            "which indicates a wall-clock dependency has been re-introduced "
            "(regression of #131 via the comp_hgics sub-calls)."
        )

    @pytest.mark.regression
    def test_hgics_join_golden_fixture(self) -> None:
        """Bit-identical match against the locked golden fixture."""
        from tests.golden.generate_hgics_join_golden import build_hgics_inputs

        na, gl = build_hgics_inputs()
        _write_inputs(self.paths, na, gl)
        hgics_join(self.paths)

        result = pl.read_parquet(self.output_path)
        golden = pl.read_parquet(GOLDEN_DIR / "comp_hgics.parquet")
        assert_frame_equal(result, golden, check_exact=True)
