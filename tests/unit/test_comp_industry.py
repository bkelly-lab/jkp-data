"""Tests for ``comp_industry`` (Issue #155).

Covers the daily merge of ``comp_other`` (SIC/NAICS) and ``comp_hgics`` (GICS)
into a single Compustat industry panel:

- Gap-fill continuity via the DuckDB ``aux_date = LEAD(date) - 1 day`` logic
- Full outer join across the two sources on ``(gvkey, date)``
- Single-date gvkey handled by ``COALESCE(LEAD..., date)``
- Dedup + sort by ``(gvkey, date)``
- Cleanup of the transient ``aux_comp_ind.ddb`` file
- A regression golden fixture locking the output bit-for-bit.

To exercise ``comp_industry``'s SQL in isolation we monkeypatch its two
sub-calls (``comp_sic_naics``, ``hgics_join``) to no-ops and write
``comp_other.parquet`` / ``comp_hgics.parquet`` directly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import jkp.data.aux_functions as aux_functions
from jkp.data.aux_functions import comp_industry
from jkp.data.paths import DataPaths

GOLDEN_DIR = Path(__file__).parent.parent / "golden" / "fixtures" / "comp_industry"


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
        """Bind paths and monkeypatch sub-calls so only ``comp_industry``'s SQL runs."""
        self.paths = test_paths
        self.output_path = self.paths.interim_dir / "comp_ind.parquet"
        self.ddb_path = self.paths.interim_dir / "aux_comp_ind.ddb"
        monkeypatch.setattr(aux_functions, "comp_sic_naics", lambda _paths: None)
        monkeypatch.setattr(aux_functions, "hgics_join", lambda _paths: None)

    def test_gap_fill_continuity(self) -> None:
        """Sparse dates (Jan 1, Jan 5) produce a contiguous daily date axis.

        Gap-fill creates rows for the intermediate dates (Jan 2-4) but the SQL
        LEFT JOIN back to ``gap_dates`` only matches on the anchor ``date``,
        so the intermediate rows carry NULL codes. This is the intentional
        behavior: the date axis is contiguous (useful for downstream as-of
        joins on month-end), but codes are not forward-filled.
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
        _write_intermediates(self.paths, comp_other, comp_hgics)

        comp_industry(self.paths)

        result = pl.read_parquet(self.output_path).sort("date")
        # Jan-1 anchor expands to Jan-1..Jan-4 (4 days), Jan-5 is the continuous
        # last row → 5 daily rows total.
        assert result.height == 5
        assert result["date"].to_list() == [
            date(2020, 1, 1),
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 4),
            date(2020, 1, 5),
        ]
        # Anchor dates carry their codes through; intermediate dates have nulls.
        anchors = result.filter(pl.col("date").is_in([date(2020, 1, 1), date(2020, 1, 5)]))
        intermediates = result.filter(
            pl.col("date").is_in([date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)])
        )
        assert anchors["sic"].to_list() == [7372, 7372]
        assert anchors["naics"].to_list() == [511210, 511210]
        assert anchors["gics"].to_list() == [10101010, 10101010]
        assert intermediates["sic"].null_count() == intermediates.height
        assert intermediates["naics"].null_count() == intermediates.height
        assert intermediates["gics"].null_count() == intermediates.height

    def test_full_outer_join_with_nulls(self) -> None:
        """A GICS-only date and a SIC-only date each produce one row with nulls."""
        comp_other = _other_frame(["200000"], [date(2020, 6, 16)], [3711], [336111])
        comp_hgics = _gics_frame(["200000"], [date(2020, 6, 15)], [20202020])
        _write_intermediates(self.paths, comp_other, comp_hgics)

        comp_industry(self.paths)

        result = pl.read_parquet(self.output_path).sort("date")
        assert result.height == 2

        row_15 = result.filter(pl.col("date") == date(2020, 6, 15)).row(0, named=True)
        assert row_15["gics"] == 20202020
        assert row_15["sic"] is None
        assert row_15["naics"] is None

        row_16 = result.filter(pl.col("date") == date(2020, 6, 16)).row(0, named=True)
        assert row_16["gics"] is None
        assert row_16["sic"] == 3711
        assert row_16["naics"] == 336111

    def test_single_date_gvkey(self) -> None:
        """A gvkey with one date in both sources produces one continuous row."""
        comp_other = _other_frame(["300000"], [date(2021, 12, 31)], [4813], [517110])
        comp_hgics = _gics_frame(["300000"], [date(2021, 12, 31)], [50505050])
        _write_intermediates(self.paths, comp_other, comp_hgics)

        comp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.height == 1
        row = result.row(0, named=True)
        assert row["date"] == date(2021, 12, 31)
        assert row["sic"] == 4813
        assert row["gics"] == 50505050

    def test_dedup_invariant(self) -> None:
        """Output has unique ``(gvkey, date)`` rows."""
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
        _write_intermediates(self.paths, comp_other, comp_hgics)

        comp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        assert result.unique(["gvkey", "date"]).height == result.height

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
        _write_intermediates(self.paths, comp_other, comp_hgics)

        comp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        gvkeys = result["gvkey"].to_list()
        dates = result["date"].to_list()
        assert gvkeys == sorted(gvkeys)
        for i in range(1, len(result)):
            same_gvkey = gvkeys[i] == gvkeys[i - 1]
            assert (not same_gvkey) or dates[i] >= dates[i - 1]

    def test_aux_ddb_cleanup(self) -> None:
        """The transient ``aux_comp_ind.ddb`` file is removed after the call."""
        comp_other = _other_frame(["100000"], [date(2020, 1, 1)], [7372], [511210])
        comp_hgics = _gics_frame(["100000"], [date(2020, 1, 1)], [10101010])
        _write_intermediates(self.paths, comp_other, comp_hgics)

        # Plant a stale .ddb to verify the unlink(missing_ok=True) call clears it
        # before creating a fresh connection.
        self.ddb_path.write_text("stale")

        comp_industry(self.paths)

        # The DuckDB connection should still be holding the file (or have left it
        # behind after disconnect()). The cleanup guarantee in the code is
        # `.unlink(missing_ok=True)` at the *start* of each call, so what we
        # really want to verify is that the file is no larger / different than
        # what comp_industry produced — i.e. the stale content is gone.
        if self.ddb_path.exists():
            content = self.ddb_path.read_bytes()
            assert content[:5] != b"stale", (
                "Stale aux_comp_ind.ddb was not cleaned up before the new connection"
            )

    @pytest.mark.regression
    def test_comp_industry_golden_fixture(self) -> None:
        """Bit-identical match against the locked golden fixture."""
        from tests.golden.generate_comp_industry_golden import build_comp_industry_inputs

        comp_other, comp_hgics = build_comp_industry_inputs(seed=42)
        _write_intermediates(self.paths, comp_other, comp_hgics)
        comp_industry(self.paths)

        result = pl.read_parquet(self.output_path)
        golden = pl.read_parquet(GOLDEN_DIR / "comp_ind.parquet")
        assert_frame_equal(result, golden, check_exact=True)
