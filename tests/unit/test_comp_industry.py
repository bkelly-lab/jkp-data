"""Tests for comp_industry() (Issue #155).

``comp_industry`` merges the daily SIC/NAICS panel (``comp_other``) and the
daily GICS panel (``comp_hgics``) into a single daily Compustat industry file.
Its DuckDB SQL does the following, per ``gvkey``:

1. ``FULL OUTER JOIN`` ``comp_gics`` and ``comp_other`` on ``(gvkey, date)``.
2. ``aux_date = LEAD(date) OVER (PARTITION BY gvkey ORDER BY date) - 1 day``,
   with ``COALESCE(..., date)`` so the *last* row of each gvkey gets
   ``aux_date = date``.
3. Rows with ``date <> aux_date`` are "gap" anchors; ``generate_series(date,
   aux_date)`` expands them into a contiguous daily axis. The expanded rows are
   ``LEFT JOIN``-ed back to the anchors on ``(gvkey, date)``, so **only the
   anchor date keeps its codes — the in-between days carry NULL codes** (the
   axis is made continuous, but codes are intentionally not forward-filled).
4. Rows with ``date = aux_date`` (the terminal row of every gvkey, plus any
   single-date gvkey) pass through unchanged via the ``continuous`` branch.
5. ``continuous`` UNION ``gaps`` → ``SELECT DISTINCT ON (gvkey, date)`` ordered
   by ``(gvkey, date)``.

Coverage here, keyed to the behaviors Issue #155 calls out:

- Gap-fill continuity, including multi-span chaining within one gvkey.
- Per-gvkey isolation of the ``PARTITION BY`` window (one gvkey's gap range
  must not bleed into another's rows on the same calendar date).
- Full-outer-join shape when a date is present in only one source.
- Same-(gvkey, date) coalesce: a date present in *both* sources collapses to a
  single row carrying GICS *and* SIC/NAICS.
- ``COALESCE(LEAD..., date)`` terminal-row handling for single-date gvkeys
  (present in both sources, and present in only one source).
- Terminal-row preservation for multi-date gvkeys.
- Dedup to unique ``(gvkey, date)``.
- Sort by ``(gvkey, date)``.
- Output schema (column names + dtypes), to catch silent dtype drift.
- Robust cleanup of the transient ``aux_comp_ind.ddb`` file.
- A regression golden fixture locking the output bit-for-bit.

To exercise only ``comp_industry``'s SQL we stub its two upstream sub-calls
(``comp_sic_naics``, ``hgics_join``) to no-ops — see
``tests.golden.comp_industry_stubs`` — and write ``comp_other.parquet`` /
``comp_hgics.parquet`` directly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import comp_industry
from jkp.data.paths import DataPaths
from tests.conftest import assert_sorted_by_keys, assert_unique_keys
from tests.golden.comp_industry_stubs import (
    CompIndustryUpstreamStubs,
    patch_comp_industry_upstream_stubs,
)

GOLDEN_DIR = Path(__file__).parent.parent / "golden" / "fixtures" / "comp_industry"

# The exact output contract of comp_industry: column order and dtypes.
EXPECTED_SCHEMA: dict[str, pl.DataType] = {
    "gvkey": pl.Utf8,
    "date": pl.Date,
    "gics": pl.Int64,
    "sic": pl.Int64,
    "naics": pl.Int64,
}


def _write_intermediates(
    paths: DataPaths, comp_other: pl.DataFrame, comp_hgics: pl.DataFrame
) -> None:
    """Persist comp_other / comp_hgics fixtures where ``comp_industry`` reads them."""
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    comp_other.write_parquet(paths.interim_dir / "comp_other.parquet")
    comp_hgics.write_parquet(paths.interim_dir / "comp_hgics.parquet")


def _other_frame(
    gvkeys: list[str],
    dates: list[date],
    sics: list[int | None],
    naicses: list[int | None],
) -> pl.DataFrame:
    """Build a ``comp_other`` (SIC/NAICS) frame. Empty lists yield a typed 0-row frame."""
    return pl.DataFrame(
        {"gvkey": gvkeys, "date": dates, "sic": sics, "naics": naicses},
        schema={
            "gvkey": pl.Utf8,
            "date": pl.Date,
            "sic": pl.Int64,
            "naics": pl.Int64,
        },
    )


def _gics_frame(gvkeys: list[str], dates: list[date], gicses: list[int | None]) -> pl.DataFrame:
    """Build a ``comp_hgics`` (GICS) frame. Empty lists yield a typed 0-row frame."""
    return pl.DataFrame(
        {"gvkey": gvkeys, "date": dates, "gics": gicses},
        schema={
            "gvkey": pl.Utf8,
            "date": pl.Date,
            "gics": pl.Int64,
        },
    )


class TestCompIndustry:
    """Tests for ``comp_industry``."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bind paths and stub upstream calls so only ``comp_industry``'s SQL runs."""
        self.paths = test_paths
        self.output_path = self.paths.interim_dir / "comp_ind.parquet"
        self.ddb_path = self.paths.interim_dir / "aux_comp_ind.ddb"
        self.upstream_stubs: CompIndustryUpstreamStubs = patch_comp_industry_upstream_stubs(
            monkeypatch
        )

    def _run(self, comp_other: pl.DataFrame, comp_hgics: pl.DataFrame) -> pl.DataFrame:
        """Write intermediates, run ``comp_industry``, and return the parquet output."""
        _write_intermediates(self.paths, comp_other, comp_hgics)
        comp_industry(self.paths)
        self.upstream_stubs.assert_called()
        return pl.read_parquet(self.output_path)

    # ------------------------------------------------------------------
    # Gap-fill continuity
    # ------------------------------------------------------------------

    def test_gap_fill_continuity(self) -> None:
        """Sparse dates expand to a contiguous daily axis with stable output schema.

        ``aux_date`` on the Jan-1 row is Jan-4 (``LEAD(Jan-5) - 1 day``), so
        ``generate_series(Jan-1, Jan-4)`` yields Jan 1-4; the LEFT JOIN back to
        the anchors keeps codes only on Jan-1. Jan-5 is the terminal row and
        flows through the ``continuous`` branch with its codes intact.

        Also asserts the output column names and dtypes — e.g. nullable gap rows
        must not coerce integer code columns to ``Float64`` through the union.
        """
        comp_other = _other_frame(
            ["100000", "100000"],
            [date(2020, 1, 1), date(2020, 1, 5)],
            [7372, 7372],
            [511210, 511210],
        )
        comp_hgics = _gics_frame(
            ["100000", "100000"],
            [date(2020, 1, 1), date(2020, 1, 5)],
            [10101010, 10101010],
        )
        result = self._run(comp_other, comp_hgics).sort("date")

        assert result.height == 5, f"Expected 5 contiguous rows from gap-fill, got {result.height}"
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
        anchors = result.filter(pl.col("date").is_in([date(2020, 1, 1), date(2020, 1, 5)]))
        intermediates = result.filter(
            pl.col("date").is_in([date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)])
        )
        assert anchors["sic"].to_list() == [7372, 7372], (
            f"Anchor rows should carry sic=7372, got {anchors['sic'].to_list()}"
        )
        assert anchors["naics"].to_list() == [511210, 511210], (
            f"Anchor rows should carry naics=511210, got {anchors['naics'].to_list()}"
        )
        assert anchors["gics"].to_list() == [10101010, 10101010], (
            f"Anchor rows should carry gics=10101010, got {anchors['gics'].to_list()}"
        )
        assert intermediates["sic"].null_count() == intermediates.height, (
            f"Intermediate rows should have null sic, got "
            f"{intermediates.height - intermediates['sic'].null_count()} non-null"
        )
        assert intermediates["naics"].null_count() == intermediates.height, (
            f"Intermediate rows should have null naics, got "
            f"{intermediates.height - intermediates['naics'].null_count()} non-null"
        )
        assert intermediates["gics"].null_count() == intermediates.height, (
            f"Intermediate rows should have null gics, got "
            f"{intermediates.height - intermediates['gics'].null_count()} non-null"
        )
        assert result.columns == list(EXPECTED_SCHEMA), (
            f"Column order mismatch: expected {list(EXPECTED_SCHEMA)}, got {result.columns}"
        )
        assert dict(result.schema) == EXPECTED_SCHEMA, (
            f"Schema mismatch: expected {EXPECTED_SCHEMA}, got {dict(result.schema)}"
        )
        )

    def test_gap_fill_multi_span_chaining(self) -> None:
        """Three anchors (Jan 1, 3, 6) chain into contiguous, non-overlapping spans.

        Each anchor carries *distinct* codes, so this also checks that gap-fill
        attaches each anchor's codes to the correct date rather than smearing a
        neighbour's values across the span:

            Jan 1 (aux=Jan 2) -> series Jan 1-2   (Jan 1 keeps codes)
            Jan 3 (aux=Jan 5) -> series Jan 3-5   (Jan 3 keeps codes)
            Jan 6 (aux=Jan 6) -> terminal/continuous (keeps codes)
        """
        comp_other = _other_frame(
            ["100000", "100000", "100000"],
            [date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 6)],
            [1000, 3000, 6000],
            [11, 33, 66],
        )
        comp_hgics = _gics_frame(
            ["100000", "100000", "100000"],
            [date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 6)],
            [111, 333, 666],
        )
        result = self._run(comp_other, comp_hgics).sort("date")

        expected_dates = [date(2020, 1, d) for d in (1, 2, 3, 4, 5, 6)]
        assert result["date"].to_list() == expected_dates, (
            f"Expected contiguous dates {expected_dates}, got {result['date'].to_list()}"
        )

        def row(d: int) -> dict:
            return result.filter(pl.col("date") == date(2020, 1, d)).row(0, named=True)

        assert (row(1)["sic"], row(1)["naics"], row(1)["gics"]) == (1000, 11, 111), (
            f"Jan-1 anchor codes wrong, got {(row(1)['sic'], row(1)['naics'], row(1)['gics'])}"
        )
        assert (row(3)["sic"], row(3)["naics"], row(3)["gics"]) == (3000, 33, 333), (
            f"Jan-3 anchor codes wrong, got {(row(3)['sic'], row(3)['naics'], row(3)['gics'])}"
        )
        assert (row(6)["sic"], row(6)["naics"], row(6)["gics"]) == (6000, 66, 666), (
            f"Jan-6 anchor codes wrong, got {(row(6)['sic'], row(6)['naics'], row(6)['gics'])}"
        )
        for d in (2, 4, 5):
            assert row(d)["sic"] is None, (
                f"Jan-{d} gap row should have null sic, got {row(d)['sic']}"
            )
            assert row(d)["naics"] is None, (
                f"Jan-{d} gap row should have null naics, got {row(d)['naics']}"
            )
            assert row(d)["gics"] is None, (
                f"Jan-{d} gap row should have null gics, got {row(d)['gics']}"
            )

    def test_gap_fill_partitioned_by_gvkey(self) -> None:
        """One gvkey's gap range must not contaminate another gvkey's rows.

        gvkey A (100000) spans Jan 1-5 and produces NULL-code intermediate rows
        on Jan 2-4. gvkey B (200000) has a single real observation on Jan 3 with
        its own codes. Because ``aux_date`` is computed ``PARTITION BY gvkey``,
        B's Jan-3 row is a terminal/continuous row that keeps its codes — it is
        *not* overwritten by A's NULL Jan-3 gap row. A missing ``PARTITION BY``
        would corrupt this.
        """
        comp_other = _other_frame(
            ["100000", "100000", "200000"],
            [date(2020, 1, 1), date(2020, 1, 5), date(2020, 1, 3)],
            [1000, 5000, 2000],
            [11, 55, 22],
        )
        comp_hgics = _gics_frame(
            ["100000", "100000", "200000"],
            [date(2020, 1, 1), date(2020, 1, 5), date(2020, 1, 3)],
            [111, 555, 222],
        )
        result = self._run(comp_other, comp_hgics).sort(["gvkey", "date"])

        # gvkey A: 5 daily rows; gvkey B: its single Jan-3 row, codes intact.
        a = result.filter(pl.col("gvkey") == "100000")
        b = result.filter(pl.col("gvkey") == "200000")
        assert a.height == 5, f"gvkey A should have 5 daily rows, got {a.height}"
        assert b.height == 1, f"gvkey B should have 1 row, got {b.height}"
        b_row = b.row(0, named=True)
        assert b_row["date"] == date(2020, 1, 3), (
            f"gvkey B's row should be Jan 3, got {b_row['date']}"
        )
        assert (b_row["sic"], b_row["naics"], b_row["gics"]) == (2000, 22, 222), (
            f"gvkey B's codes should be (2000, 22, 222), "
            f"got {(b_row['sic'], b_row['naics'], b_row['gics'])}"
        )

    # ------------------------------------------------------------------
    # Full outer join + coalesce
    # ------------------------------------------------------------------

    def test_full_outer_join_disjoint_dates(self) -> None:
        """A GICS-only date and a SIC-only date each produce one row with nulls.

        comp_hgics has Jun-15 (GICS only); comp_other has Jun-16 (SIC only).
        Jun-15's ``aux_date`` is Jun-15 (``LEAD(Jun-16) - 1 day``), so both rows
        are terminal/continuous and no gap expansion occurs.
        """
        comp_other = _other_frame(["200000"], [date(2020, 6, 16)], [3711], [336111])
        comp_hgics = _gics_frame(["200000"], [date(2020, 6, 15)], [20202020])
        result = self._run(comp_other, comp_hgics).sort("date")

        assert result.height == 2, f"Expected 2 rows (one per disjoint date), got {result.height}"

        row_15 = result.filter(pl.col("date") == date(2020, 6, 15)).row(0, named=True)
        assert row_15["gics"] == 20202020, (
            f"Jun-15 (GICS-only) should have gics=20202020, got {row_15['gics']}"
        )
        assert row_15["sic"] is None, (
            f"Jun-15 (GICS-only) should have null sic, got {row_15['sic']}"
        )
        assert row_15["naics"] is None, (
            f"Jun-15 (GICS-only) should have null naics, got {row_15['naics']}"
        )

        row_16 = result.filter(pl.col("date") == date(2020, 6, 16)).row(0, named=True)
        assert row_16["gics"] is None, (
            f"Jun-16 (SIC-only) should have null gics, got {row_16['gics']}"
        )
        assert row_16["sic"] == 3711, f"Jun-16 (SIC-only) should have sic=3711, got {row_16['sic']}"
        assert row_16["naics"] == 336111, (
            f"Jun-16 (SIC-only) should have naics=336111, got {row_16['naics']}"
        )

    def test_same_date_coalesces_both_sources(self) -> None:
        """A (gvkey, date) present in *both* sources collapses to a single row.

        The ``FULL OUTER JOIN ... USING (gvkey, date)`` coalesces the join keys,
        so the output row carries GICS (from comp_hgics) *and* SIC/NAICS (from
        comp_other) together. This is the join-precedence case Issue #155 flags.
        """
        comp_other = _other_frame(["400000"], [date(2020, 3, 15)], [1234], [567890])
        comp_hgics = _gics_frame(["400000"], [date(2020, 3, 15)], [45678900])
        result = self._run(comp_other, comp_hgics)

        assert result.height == 1, f"Expected 1 coalesced row, got {result.height}"
        row = result.row(0, named=True)
        assert row["gvkey"] == "400000", f"Expected gvkey='400000', got {row['gvkey']}"
        assert row["date"] == date(2020, 3, 15), f"Expected date=2020-03-15, got {row['date']}"
        assert row["gics"] == 45678900, f"Expected gics=45678900, got {row['gics']}"
        assert row["sic"] == 1234, f"Expected sic=1234, got {row['sic']}"
        assert row["naics"] == 567890, f"Expected naics=567890, got {row['naics']}"

    # ------------------------------------------------------------------
    # COALESCE(LEAD..., date) terminal-row handling
    # ------------------------------------------------------------------

    def test_single_date_gvkey_both_sources(self) -> None:
        """A gvkey with one date in both sources produces one continuous row."""
        comp_other = _other_frame(["300000"], [date(2021, 12, 31)], [4813], [517110])
        comp_hgics = _gics_frame(["300000"], [date(2021, 12, 31)], [50505050])
        result = self._run(comp_other, comp_hgics)

        assert result.height == 1, (
            f"Expected 1 continuous row for single-date gvkey, got {result.height}"
        )
        row = result.row(0, named=True)
        assert row["date"] == date(2021, 12, 31), f"Expected date=2021-12-31, got {row['date']}"
        assert row["sic"] == 4813, f"Expected sic=4813, got {row['sic']}"
        assert row["naics"] == 517110, f"Expected naics=517110, got {row['naics']}"
        assert row["gics"] == 50505050, f"Expected gics=50505050, got {row['gics']}"

    def test_single_date_gvkey_one_source_only(self) -> None:
        """A single-date gvkey present in only one source still flows through.

        comp_hgics is empty, so the gvkey exists only in comp_other. ``LEAD`` is
        NULL → ``COALESCE(..., date)`` makes ``aux_date = date`` → continuous
        branch. The missing GICS stays NULL.
        """
        comp_other = _other_frame(["500000"], [date(2020, 7, 1)], [2222], [333333])
        comp_hgics = _gics_frame([], [], [])
        result = self._run(comp_other, comp_hgics)

        assert result.height == 1, (
            f"Expected 1 row for single-date gvkey in one source, got {result.height}"
        )
        row = result.row(0, named=True)
        assert row["date"] == date(2020, 7, 1), f"Expected date=2020-07-01, got {row['date']}"
        assert row["sic"] == 2222, f"Expected sic=2222, got {row['sic']}"
        assert row["naics"] == 333333, f"Expected naics=333333, got {row['naics']}"
        assert row["gics"] is None, (
            f"Expected null gics (absent from comp_hgics), got {row['gics']}"
        )

    def test_terminal_row_preserved_for_multi_date_gvkey(self) -> None:
        """The last date of a multi-date gvkey survives via the continuous branch.

        With dates Jan 1 and Jan 5, the Jan-5 terminal row (``date = aux_date``)
        must appear exactly once with its codes — it is the only source of the
        Jan-5 row (gap expansion of the Jan-1 anchor stops at Jan-4).
        """
        comp_other = _other_frame(
            ["100000", "100000"],
            [date(2020, 1, 1), date(2020, 1, 5)],
            [7372, 9999],
            [511210, 999999],
        )
        comp_hgics = _gics_frame(
            ["100000", "100000"],
            [date(2020, 1, 1), date(2020, 1, 5)],
            [10101010, 90909090],
        )
        result = self._run(comp_other, comp_hgics)

        terminal = result.filter(pl.col("date") == date(2020, 1, 5))
        assert terminal.height == 1, (
            f"Expected exactly 1 terminal row for Jan-5, got {terminal.height}"
        )
        row = terminal.row(0, named=True)
        assert row["sic"] == 9999, f"Terminal row sic should be 9999, got {row['sic']}"
        assert row["naics"] == 999999, f"Terminal row naics should be 999999, got {row['naics']}"
        assert row["gics"] == 90909090, f"Terminal row gics should be 90909090, got {row['gics']}"

    # ------------------------------------------------------------------
    # Dedup + sort + schema invariants
    # ------------------------------------------------------------------

    def test_unique_gvkey_date_invariant(self) -> None:
        """Output is uniquely keyed on ``(gvkey, date)`` on realistic input."""
        comp_other = _other_frame(
            ["100000", "100000", "200000"],
            [date(2020, 1, 1), date(2020, 1, 5), date(2020, 6, 16)],
            [7372, 7372, 3711],
            [511210, 511210, 336111],
        )
        comp_hgics = _gics_frame(
            ["100000", "100000", "200000"],
            [date(2020, 1, 1), date(2020, 1, 5), date(2020, 6, 15)],
            [10101010, 10101010, 20202020],
        )
        result = self._run(comp_other, comp_hgics)
        assert_unique_keys(result, ["gvkey", "date"])

    def test_sort_invariant(self) -> None:
        """Output is sorted by ``(gvkey, date)`` ascending."""
        comp_other = _other_frame(
            ["200000", "100000"],
            [date(2020, 6, 16), date(2020, 1, 1)],
            [3711, 7372],
            [336111, 511210],
        )
        comp_hgics = _gics_frame(
            ["200000", "100000"],
            [date(2020, 6, 15), date(2020, 1, 1)],
            [20202020, 10101010],
        )
        result = self._run(comp_other, comp_hgics)
        assert_sorted_by_keys(result, "gvkey", "date")

    # ------------------------------------------------------------------
    # Operational: transient DuckDB file
    # ------------------------------------------------------------------

    def test_runs_despite_stale_aux_ddb(self) -> None:
        """A stale ``aux_comp_ind.ddb`` from a prior run must not break the call.

        The guarantee in the code is the ``unlink(missing_ok=True)`` *before*
        connecting: a leftover file is removed so a fresh DuckDB database is
        created cleanly. We plant non-DuckDB bytes at that path and assert the
        run still succeeds and produces correct output (which it cannot do if
        the stale file poisoned the new connection).
        """
        comp_other = _other_frame(["100000"], [date(2020, 1, 1)], [7372], [511210])
        comp_hgics = _gics_frame(["100000"], [date(2020, 1, 1)], [10101010])
        _write_intermediates(self.paths, comp_other, comp_hgics)

        self.ddb_path.write_bytes(b"not a valid duckdb file")

        comp_industry(self.paths)  # must not raise
        self.upstream_stubs.assert_called()

        result = pl.read_parquet(self.output_path)
        assert result.height == 1, (
            f"Expected 1 output row after stale-ddb recovery, got {result.height}"
        )
        row = result.row(0, named=True)
        assert (row["sic"], row["naics"], row["gics"]) == (7372, 511210, 10101010), (
            f"Expected codes (7372, 511210, 10101010), "
            f"got {(row['sic'], row['naics'], row['gics'])}"
        )
        # The stale bytes were replaced by a real DuckDB database.
        assert self.ddb_path.exists(), "Expected aux_comp_ind.ddb to exist after a successful run"
        assert self.ddb_path.read_bytes()[:23] != b"not a valid duckdb file", (
            "Stale DuckDB file was not replaced by a valid database"
        )

    # ------------------------------------------------------------------
    # Golden regression
    # ------------------------------------------------------------------

    @pytest.mark.regression
    def test_comp_industry_golden_fixture(self) -> None:
        """Bit-identical match against the locked golden fixture.

        Regenerate the fixture with::

            uv run python -m tests.golden.generate_comp_industry_golden
        """
        from tests.golden.generate_comp_industry_golden import build_comp_industry_inputs

        comp_other, comp_hgics = build_comp_industry_inputs()
        result = self._run(comp_other, comp_hgics)

        golden = pl.read_parquet(GOLDEN_DIR / "comp_ind.parquet")
        assert_frame_equal(result, golden, check_exact=True)
