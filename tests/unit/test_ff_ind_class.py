"""Tests for Fama-French industry classification functions."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from aux_functions import _build_sic_ff_mapping, _parse_siccodes_file, ff_ind_class

# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_SICCODES = """\
 1 Agric  Agriculture
          0100-0199 Agricultural production - crops
          0200-0299 Agricultural production - livestock

 2 Food   Food Products
          2000-2009 Food and kindred products
          2010-2019 Meat products
"""


@pytest.fixture
def siccodes_file(tmp_path: Path) -> Path:
    """Write a small synthetic Siccodes file and return its path."""
    p = tmp_path / "Siccodes_test.txt"
    p.write_text(SAMPLE_SICCODES, encoding="utf-8")
    return p


# =============================================================================
# TestParseSiccodesFile — single-file parser
# =============================================================================


class TestParseSiccodesFile:
    """Tests for _parse_siccodes_file()."""

    def test_parses_correct_columns(self, siccodes_file: Path):
        """Result has exactly 'sic' and the given label column."""
        df = _parse_siccodes_file(str(siccodes_file), label="ff_test")
        assert set(df.columns) == {"sic", "ff_test"}

    def test_sic_range_expansion(self, siccodes_file: Path):
        """SIC ranges are expanded to individual codes."""
        df = _parse_siccodes_file(str(siccodes_file), label="ff_test")
        # Category 1: 0100-0199 (100 codes) + 0200-0299 (100 codes) = 200
        cat1 = df.filter(pl.col("ff_test") == 1)
        assert len(cat1) == 200

    def test_category_assignment(self, siccodes_file: Path):
        """SIC codes are assigned to the correct category."""
        df = _parse_siccodes_file(str(siccodes_file), label="ff_test")
        row = df.filter(pl.col("sic") == 150).row(0, named=True)
        assert row["ff_test"] == 1

        row = df.filter(pl.col("sic") == 2015).row(0, named=True)
        assert row["ff_test"] == 2

    def test_no_duplicate_sics(self, siccodes_file: Path):
        """Each SIC code appears at most once."""
        df = _parse_siccodes_file(str(siccodes_file), label="ff_test")
        assert df["sic"].n_unique() == len(df)

    def test_output_dtypes(self, siccodes_file: Path):
        """SIC is Int64 and the label column is Int32."""
        df = _parse_siccodes_file(str(siccodes_file), label="ff_test")
        assert df["sic"].dtype == pl.Int64
        assert df["ff_test"].dtype == pl.Int32


# =============================================================================
# TestBuildSicFfMapping — full mapping builder (uses real Siccodes files)
# =============================================================================


class TestBuildSicFfMapping:
    """Tests for _build_sic_ff_mapping()."""

    @pytest.fixture(autouse=True)
    def _chdir_to_repo_root(self, monkeypatch: pytest.MonkeyPatch):
        """Ensure tests run with CWD set to the repo root for any code using relative paths."""
        repo_root = Path(__file__).parent.parent.parent
        monkeypatch.chdir(repo_root)

    def test_has_all_columns(self):
        """Result contains sic and all eight FF classification columns."""
        df = _build_sic_ff_mapping().collect()
        expected = {"sic", "ff5", "ff10", "ff12", "ff17", "ff30", "ff38", "ff48", "ff49"}
        assert set(df.columns) == expected

    def test_no_duplicate_sics(self):
        """Each SIC code appears at most once in the mapping."""
        df = _build_sic_ff_mapping().collect()
        assert df["sic"].n_unique() == len(df)

    def test_known_sic_mapping(self):
        """Spot-check: SIC 2011 (Meat packing) -> ff49 category 2 (Food)."""
        df = _build_sic_ff_mapping().collect()
        row = df.filter(pl.col("sic") == 2011).row(0, named=True)
        assert row["ff49"] == 2

    def test_unmapped_sics_are_null(self):
        """SIC codes not listed in a scheme should not appear (or have null)."""
        df = _build_sic_ff_mapping().collect()
        # SIC 0050 is not in any FF classification
        rows = df.filter(pl.col("sic") == 50)
        if len(rows) > 0:
            row = rows.row(0, named=True)
            # If it exists, all FF columns should be null
            for c in ["ff5", "ff10", "ff12", "ff17", "ff30", "ff38", "ff48", "ff49"]:
                assert row[c] is None

    def test_output_dtypes(self):
        """All FF columns should be Int32."""
        df = _build_sic_ff_mapping().collect()
        for c in ["ff5", "ff10", "ff12", "ff17", "ff30", "ff38", "ff48", "ff49"]:
            assert df[c].dtype == pl.Int32


# =============================================================================
# TestFFIndClass — full pipeline function
# =============================================================================


class TestFFIndClass:
    """Tests for ff_ind_class()."""

    @pytest.fixture(autouse=True)
    def _setup_workdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Set up working directory with Siccodes files symlinked from the repo."""
        repo_root = Path(__file__).parent.parent.parent
        # Create data/raw/ in the temp directory and symlink the real Siccodes files
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        for f in (repo_root / "data" / "raw").glob("Siccodes*.txt"):
            (raw_dir / f.name).symlink_to(f)
        monkeypatch.chdir(tmp_path)
        self._tmp = tmp_path

    def test_mapped_sic_gets_classification(self):
        """A known SIC code should receive FF classification values."""
        input_df = pl.DataFrame({"sic": [2011], "dummy": [1.0]})
        input_path = str(self._tmp / "input.parquet")
        input_df.write_parquet(input_path)

        ff_ind_class(input_path)

        result = pl.read_parquet("__msf_world3.parquet")
        row = result.row(0, named=True)
        assert row["ff49"] == 2  # Food Products

    def test_unmapped_sic_gets_null(self):
        """A SIC code not in any mapping should have null FF columns."""
        input_df = pl.DataFrame({"sic": [50], "dummy": [1.0]})
        input_path = str(self._tmp / "input.parquet")
        input_df.write_parquet(input_path)

        ff_ind_class(input_path)

        result = pl.read_parquet("__msf_world3.parquet")
        row = result.row(0, named=True)
        for c in ["ff5", "ff10", "ff12", "ff17", "ff30", "ff38", "ff48", "ff49"]:
            assert row[c] is None

    def test_null_sic_gets_null(self):
        """A null SIC should result in null FF columns."""
        input_df = pl.DataFrame({"sic": [None]}, schema={"sic": pl.Int64})
        input_path = str(self._tmp / "input.parquet")
        input_df.write_parquet(input_path)

        ff_ind_class(input_path)

        result = pl.read_parquet("__msf_world3.parquet")
        row = result.row(0, named=True)
        for c in ["ff5", "ff10", "ff12", "ff17", "ff30", "ff38", "ff48", "ff49"]:
            assert row[c] is None

    def test_preserves_all_input_rows(self):
        """Output should have the same number of rows as input."""
        input_df = pl.DataFrame({"sic": [100, 2011, 50, None, 3714]})
        input_path = str(self._tmp / "input.parquet")
        input_df.write_parquet(input_path)

        ff_ind_class(input_path)

        result = pl.read_parquet("__msf_world3.parquet")
        assert len(result) == len(input_df)

    def test_preserves_existing_columns(self):
        """Non-FF columns from input should be retained."""
        input_df = pl.DataFrame({"sic": [2011], "price": [42.5], "ticker": ["ACME"]})
        input_path = str(self._tmp / "input.parquet")
        input_df.write_parquet(input_path)

        ff_ind_class(input_path)

        result = pl.read_parquet("__msf_world3.parquet")
        assert "price" in result.columns
        assert "ticker" in result.columns
        assert result["price"][0] == 42.5
        assert result["ticker"][0] == "ACME"
