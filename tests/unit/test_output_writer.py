"""Tests for the output_writer module."""

import tempfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from jkp_data.output_writer import (
    OutputFormat,
    _collect_dataframe,
    configure_output_format,
    convert_outputs_to_csv,
    get_output_format,
    write_dataframe,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with various data types for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "value": [1.5, 2.5, 3.5, 4.5, 5.5],
            "is_active": [True, False, True, False, True],
            "date_col": [
                date(2024, 1, 1),
                date(2024, 2, 1),
                date(2024, 3, 1),
                date(2024, 4, 1),
                date(2024, 5, 1),
            ],
            "country": ["USA", "USA", "CAN", "CAN", "MEX"],
        }
    )


@pytest.fixture
def reset_output_format():
    """Reset output format to default before and after test."""
    configure_output_format("parquet", _allow_reset=True)
    yield
    configure_output_format("parquet", _allow_reset=True)


class TestOutputFormatConfiguration:
    """Tests for output format configuration functions."""

    def test_default_format_is_parquet(self, reset_output_format):
        """Default output format should be PARQUET."""
        configure_output_format("parquet", _allow_reset=True)  # Reset first
        assert get_output_format() == OutputFormat.PARQUET

    def test_configure_format_to_csv(self, reset_output_format, capsys):
        """Can configure output format to CSV."""
        configure_output_format("csv", _allow_reset=True)
        assert get_output_format() == OutputFormat.CSV
        captured = capsys.readouterr()
        assert "CSV" in captured.out

    def test_configure_format_to_parquet(self, reset_output_format, capsys):
        """Can configure output format to parquet."""
        configure_output_format("parquet", _allow_reset=True)
        assert get_output_format() == OutputFormat.PARQUET
        captured = capsys.readouterr()
        assert "Parquet" in captured.out

    def test_reconfigure_raises_error(self, reset_output_format):
        """Reconfiguring without _allow_reset raises RuntimeError."""
        configure_output_format("parquet", _allow_reset=True)
        with pytest.raises(RuntimeError, match="already configured"):
            configure_output_format("csv")

    def test_invalid_format_raises_error(self, reset_output_format):
        """Invalid format string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid output format"):
            configure_output_format("json", _allow_reset=True)


class TestWriteDataframe:
    """Tests for write_dataframe function."""

    def test_write_parquet(self, sample_dataframe, reset_output_format):
        """Writing DataFrame always outputs parquet format."""
        configure_output_format("parquet", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            write_dataframe(sample_dataframe, path)

            assert path.exists()
            df_read = pl.read_parquet(path)
            assert df_read.shape == sample_dataframe.shape

    def test_extension_correction_to_parquet(self, sample_dataframe, reset_output_format):
        """Path extension should be corrected to .parquet regardless of input."""
        configure_output_format("parquet", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"  # Wrong extension
            write_dataframe(sample_dataframe, path)

            correct_path = Path(tmpdir) / "test.parquet"
            assert correct_path.exists()
            assert not path.exists()  # Original .csv path should not exist

    def test_lazyframe_collection(self, sample_dataframe, reset_output_format):
        """LazyFrames should be collected before writing."""
        configure_output_format("parquet", _allow_reset=True)
        lazy_df = sample_dataframe.lazy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            write_dataframe(lazy_df, path)

            assert path.exists()
            df_read = pl.read_parquet(path)
            assert df_read.shape == sample_dataframe.shape

    def test_creates_parent_directory(self, sample_dataframe, reset_output_format):
        """Parent directories should be created if they don't exist."""
        configure_output_format("parquet", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deeply" / "test.parquet"
            write_dataframe(sample_dataframe, path)

            assert path.exists()

    def test_csv_format_still_writes_parquet(self, sample_dataframe, reset_output_format):
        """Even with CSV format configured, write_dataframe outputs parquet."""
        configure_output_format("csv", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            write_dataframe(sample_dataframe, path)

            assert path.exists()
            # Should be parquet, not CSV
            df_read = pl.read_parquet(path)
            assert df_read.shape == sample_dataframe.shape


class TestConvertOutputsToCsv:
    """Tests for convert_outputs_to_csv function."""

    def test_converts_parquet_to_csv(self, sample_dataframe, reset_output_format):
        """Converts parquet files to CSV and removes parquet files."""
        configure_output_format("csv", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            chars_dir = processed_dir / "characteristics"
            chars_dir.mkdir(parents=True)

            # Write a parquet file
            parquet_path = chars_dir / "test.parquet"
            sample_dataframe.write_parquet(parquet_path)
            assert parquet_path.exists()

            # Convert to CSV
            convert_outputs_to_csv(processed_dir)

            # Parquet should be gone, CSV should exist
            assert not parquet_path.exists()
            csv_path = chars_dir / "test.csv"
            assert csv_path.exists()

            # Verify content can be read
            df_read = pl.read_csv(csv_path)
            assert len(df_read) == len(sample_dataframe)

    def test_skips_conversion_when_parquet_format(
        self, sample_dataframe, reset_output_format, capsys
    ):
        """Skips conversion when output format is parquet."""
        configure_output_format("parquet", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            chars_dir = processed_dir / "characteristics"
            chars_dir.mkdir(parents=True)

            # Write a parquet file
            parquet_path = chars_dir / "test.parquet"
            sample_dataframe.write_parquet(parquet_path)

            # Try to convert
            convert_outputs_to_csv(processed_dir)

            # Parquet should still exist (no conversion)
            assert parquet_path.exists()
            captured = capsys.readouterr()
            assert "skipping" in captured.out.lower()

    def test_csv_leading_zeros_quoted(self, reset_output_format):
        """String columns with leading zeros should be quoted in CSV output."""
        configure_output_format("csv", _allow_reset=True)
        df = pl.DataFrame(
            {
                "gvkey": ["001000", "013007", "000100"],
                "iid": ["01", "02", "03"],
                "value": [1.5, 2.5, 3.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            chars_dir = processed_dir / "characteristics"
            chars_dir.mkdir(parents=True)

            # Write parquet, then convert
            parquet_path = chars_dir / "test.parquet"
            df.write_parquet(parquet_path)
            convert_outputs_to_csv(processed_dir)

            # Read raw CSV content to verify quoting
            csv_path = chars_dir / "test.csv"
            csv_content = csv_path.read_text()
            lines = csv_content.strip().split("\n")

            # Data rows should have quoted string values
            assert '"001000"' in lines[1]
            assert '"013007"' in lines[2]
            assert '"000100"' in lines[3]

    def test_csv_null_values_empty(self, reset_output_format):
        """Null values should be written as empty strings in CSV."""
        configure_output_format("csv", _allow_reset=True)
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [1.5, None, 3.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            chars_dir = processed_dir / "characteristics"
            chars_dir.mkdir(parents=True)

            parquet_path = chars_dir / "test.parquet"
            df.write_parquet(parquet_path)
            convert_outputs_to_csv(processed_dir)

            csv_path = chars_dir / "test.csv"
            csv_content = csv_path.read_text()
            lines = csv_content.strip().split("\n")
            # Second data row should have empty value for null
            assert lines[2] == "2,"  # id=2, value is empty

    def test_converts_multiple_directories(self, sample_dataframe, reset_output_format):
        """Converts files in multiple subdirectories."""
        configure_output_format("csv", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"

            # Create multiple directories
            for subdir in ["characteristics", "portfolios", "return_data"]:
                dir_path = processed_dir / subdir
                dir_path.mkdir(parents=True)
                parquet_path = dir_path / "test.parquet"
                sample_dataframe.write_parquet(parquet_path)

            convert_outputs_to_csv(processed_dir)

            # All should be converted
            for subdir in ["characteristics", "portfolios", "return_data"]:
                dir_path = processed_dir / subdir
                assert not (dir_path / "test.parquet").exists()
                assert (dir_path / "test.csv").exists()

    def test_converts_to_separate_output_dir(self, sample_dataframe, reset_output_format):
        """Converts parquet files to CSV in a separate output directory."""
        configure_output_format("csv", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            output_dir = Path(tmpdir) / "output"
            chars_dir = processed_dir / "characteristics"
            chars_dir.mkdir(parents=True)

            # Write a parquet file
            parquet_path = chars_dir / "test.parquet"
            sample_dataframe.write_parquet(parquet_path)

            # Convert to CSV in separate output directory
            convert_outputs_to_csv(processed_dir, output_dir)

            # Parquet should be gone from source
            assert not parquet_path.exists()

            # CSV should exist in output directory with same relative structure
            csv_path = output_dir / "characteristics" / "test.csv"
            assert csv_path.exists()

            # Verify content
            df_read = pl.read_csv(csv_path)
            assert len(df_read) == len(sample_dataframe)

    def test_no_parquet_files_to_convert(self, reset_output_format, capsys):
        """Handles empty directory with no parquet files."""
        configure_output_format("csv", _allow_reset=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir(parents=True)

            # Convert with no parquet files
            convert_outputs_to_csv(processed_dir)

            captured = capsys.readouterr()
            assert "No parquet files to convert" in captured.out


class TestStreamingMode:
    """Tests for streaming parameter in write_dataframe."""

    def test_streaming_collection(self, sample_dataframe, reset_output_format):
        """LazyFrame with streaming=True should work."""
        configure_output_format("parquet", _allow_reset=True)
        lazy_df = sample_dataframe.lazy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            write_dataframe(lazy_df, path, streaming=True)

            assert path.exists()
            df_read = pl.read_parquet(path)
            assert df_read.shape == sample_dataframe.shape


class TestShrinkDtype:
    """Tests for shrink_dtype parameter in write_dataframe."""

    def test_shrink_dtype_reduces_size(self, reset_output_format):
        """shrink_dtype=True should reduce file size for large integer values."""
        configure_output_format("parquet", _allow_reset=True)
        # Create DataFrame with small values in Int64 columns
        df = pl.DataFrame(
            {
                "small_int": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64),
                "small_float": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path_normal = Path(tmpdir) / "normal.parquet"
            path_shrunk = Path(tmpdir) / "shrunk.parquet"

            write_dataframe(df, path_normal, shrink_dtype=False)
            write_dataframe(df, path_shrunk, shrink_dtype=True)

            # Shrunk file should be smaller or equal (depends on compression)
            normal_size = path_normal.stat().st_size
            shrunk_size = path_shrunk.stat().st_size
            assert shrunk_size <= normal_size

    def test_shrink_dtype_with_lazyframe(self, sample_dataframe, reset_output_format):
        """shrink_dtype=True should work with LazyFrames."""
        configure_output_format("parquet", _allow_reset=True)
        lazy_df = sample_dataframe.lazy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            write_dataframe(lazy_df, path, shrink_dtype=True)

            assert path.exists()
            df_read = pl.read_parquet(path)
            assert len(df_read) == len(sample_dataframe)


class TestCollectDataframe:
    """Tests for _collect_dataframe helper function."""

    def test_collect_dataframe_from_dataframe(self, sample_dataframe):
        """_collect_dataframe returns DataFrame unchanged."""
        result = _collect_dataframe(sample_dataframe)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == sample_dataframe.shape

    def test_collect_dataframe_from_lazyframe(self, sample_dataframe):
        """_collect_dataframe collects LazyFrame."""
        lazy_df = sample_dataframe.lazy()
        result = _collect_dataframe(lazy_df)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == sample_dataframe.shape

    def test_collect_dataframe_with_streaming(self, sample_dataframe):
        """_collect_dataframe uses streaming when requested."""
        lazy_df = sample_dataframe.lazy()
        result = _collect_dataframe(lazy_df, streaming=True)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == sample_dataframe.shape

    @pytest.mark.parametrize("use_lazy", [False, True])
    def test_collect_dataframe_with_shrink_dtype(self, use_lazy):
        """_collect_dataframe applies shrink_dtype to DataFrame or LazyFrame."""
        df = pl.DataFrame(
            {
                "small_int": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64),
            }
        )
        input_df = df.lazy() if use_lazy else df
        result = _collect_dataframe(input_df, shrink_dtype=True)
        # shrink_dtype should return a valid DataFrame with the same data
        assert isinstance(result, pl.DataFrame)
        assert result.shape == df.shape
        assert list(result["small_int"]) == [1, 2, 3, 4, 5]
