"""Tests for load_dff_be (Davis-Fama-French hand-collected book equity loader)."""

import polars as pl
import pytest

from jkp.data.aux_functions import load_dff_be
from jkp.data.paths import get_dff_be_path


@pytest.fixture
def synthetic_dff_file(tmp_path):
    """Two-permno file with 4 BE columns (years 1926-1929), one missing value."""
    lines = [
        "  10006 1926 1929      67.743      71.245     -99.990      70.139",
        "  10014 1927 1929     -99.990       5.500       6.250       7.000",
    ]
    path = tmp_path / "dff_be.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


class TestLoadDffBeSynthetic:
    def test_long_shape_and_columns(self, synthetic_dff_file):
        df = load_dff_be(synthetic_dff_file)
        assert df.columns == ["permno", "year", "be"]
        assert df.shape == (8, 3)  # 2 permnos x 4 years

    def test_dtypes(self, synthetic_dff_file):
        df = load_dff_be(synthetic_dff_file)
        assert df["permno"].dtype == pl.Int64
        assert df["year"].dtype == pl.Int32
        assert df["be"].dtype == pl.Float64

    def test_year_range_derived_from_file(self, synthetic_dff_file):
        df = load_dff_be(synthetic_dff_file)
        assert df["year"].min() == 1926
        assert df["year"].max() == 1929

    def test_missing_sentinel_becomes_null(self, synthetic_dff_file):
        df = load_dff_be(synthetic_dff_file)
        assert df.filter((pl.col("permno") == 10006) & (pl.col("year") == 1928))["be"][0] is None
        assert df.filter((pl.col("permno") == 10014) & (pl.col("year") == 1926))["be"][0] is None
        assert df["be"].null_count() == 2

    def test_values_parsed(self, synthetic_dff_file):
        df = load_dff_be(synthetic_dff_file)
        assert df.filter((pl.col("permno") == 10006) & (pl.col("year") == 1926))["be"][0] == 67.743
        assert df.filter((pl.col("permno") == 10014) & (pl.col("year") == 1929))["be"][0] == 7.000

    def test_ragged_records_raise(self, tmp_path):
        path = tmp_path / "ragged.txt"
        path.write_text("10006 1926 1929 1.0 2.0 3.0 4.0\n10014 1927 1929 1.0 2.0\n")
        with pytest.raises(ValueError, match="ragged records"):
            load_dff_be(path)


class TestLoadDffBeBundled:
    def test_bundled_resource_exists(self):
        assert get_dff_be_path().exists()

    def test_bundled_file_loads(self):
        df = load_dff_be()
        n_years = df["year"].max() - df["year"].min() + 1
        assert df["year"].min() == 1926
        assert df["year"].max() == 2001
        assert df.height == df["permno"].n_unique() * n_years

    def test_known_value(self):
        # First record of the file: permno 10006, BE 1926 = 67.743
        df = load_dff_be()
        assert df.filter((pl.col("permno") == 10006) & (pl.col("year") == 1926))["be"][0] == 67.743
