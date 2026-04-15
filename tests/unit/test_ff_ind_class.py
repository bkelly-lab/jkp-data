"""Tests for Fama-French industry classification functions."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jkp.data.aux_functions import _parse_siccodes_file, ff_ind_class

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
# TestFFIndClass — full pipeline function
# =============================================================================


class TestFFIndClass:
    """Tests for ff_ind_class()."""

    @pytest.fixture(autouse=True)
    def _setup_workdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Set working directory to temp path for parquet output."""
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
        """A SIC code not in any mapping should have null ff49."""
        input_df = pl.DataFrame({"sic": [50], "dummy": [1.0]})
        input_path = str(self._tmp / "input.parquet")
        input_df.write_parquet(input_path)

        ff_ind_class(input_path)

        result = pl.read_parquet("__msf_world3.parquet")
        row = result.row(0, named=True)
        assert row["ff49"] is None

    def test_null_sic_gets_null(self):
        """A null SIC should result in null ff49."""
        input_df = pl.DataFrame({"sic": [None]}, schema={"sic": pl.Int64})
        input_path = str(self._tmp / "input.parquet")
        input_df.write_parquet(input_path)

        ff_ind_class(input_path)

        result = pl.read_parquet("__msf_world3.parquet")
        row = result.row(0, named=True)
        assert row["ff49"] is None

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
