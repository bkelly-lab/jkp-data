"""Tests for `_write_filtered` and `_write_split_by_key` in `portfolio.py`."""

from datetime import date
from pathlib import Path

import polars as pl

from jkp.data.portfolio import _write_filtered, _write_split_by_key


class TestWriteFiltered:
    def test_filters_rows_after_end_date(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)],
                "value": [1.0, 2.0, 3.0],
            }
        )
        path = tmp_path / "out.parquet"
        _write_filtered(df, str(path), "eom", date(2020, 2, 29))

        result = pl.read_parquet(path)
        assert result["eom"].to_list() == [date(2020, 1, 31), date(2020, 2, 29)]
        assert result["value"].to_list() == [1.0, 2.0]

    def test_preserves_schema_and_dtypes(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "eom": [date(2020, 1, 31), date(2020, 2, 29)],
                "key": ["a", "b"],
                "value": [1.5, 2.5],
                "count": [10, 20],
            }
        )
        path = tmp_path / "out.parquet"
        _write_filtered(df, str(path), "eom", date(2020, 12, 31))

        result = pl.read_parquet(path)
        assert result.columns == df.columns
        assert result.schema == df.schema

    def test_empty_input_writes_empty_parquet(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            schema={"eom": pl.Date, "value": pl.Float64},
        )
        path = tmp_path / "out.parquet"
        _write_filtered(df, str(path), "eom", date(2020, 12, 31))

        assert path.exists()
        result = pl.read_parquet(path)
        assert result.height == 0
        assert result.columns == ["eom", "value"]


class TestWriteSplitByKey:
    def test_one_file_per_unique_key(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "key": ["a", "b", "c", "a"],
                "eom": [date(2020, 1, 31)] * 4,
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        folder = tmp_path / "split"
        _write_split_by_key(df, str(folder), "key", "eom", date(2020, 12, 31))

        files = sorted(p.name for p in folder.glob("*.parquet"))
        assert files == ["a.parquet", "b.parquet", "c.parquet"]

        a_rows = pl.read_parquet(folder / "a.parquet")
        assert a_rows.height == 2
        assert sorted(a_rows["value"].to_list()) == [1.0, 4.0]

    def test_drops_null_keys(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "key": ["a", None, "b"],
                "eom": [date(2020, 1, 31)] * 3,
                "value": [1.0, 2.0, 3.0],
            }
        )
        folder = tmp_path / "split"
        _write_split_by_key(df, str(folder), "key", "eom", date(2020, 12, 31))

        files = sorted(p.name for p in folder.glob("*.parquet"))
        assert files == ["a.parquet", "b.parquet"]
        assert not (folder / "None.parquet").exists()

    def test_drops_empty_string_keys(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "key": ["a", "", "b"],
                "eom": [date(2020, 1, 31)] * 3,
                "value": [1.0, 2.0, 3.0],
            }
        )
        folder = tmp_path / "split"
        _write_split_by_key(df, str(folder), "key", "eom", date(2020, 12, 31))

        files = sorted(p.name for p in folder.glob("*.parquet"))
        assert files == ["a.parquet", "b.parquet"]
        assert not (folder / ".parquet").exists()

    def test_filters_rows_after_end_date(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "key": ["a", "a", "b"],
                "eom": [date(2020, 1, 31), date(2021, 1, 31), date(2020, 6, 30)],
                "value": [1.0, 2.0, 3.0],
            }
        )
        folder = tmp_path / "split"
        _write_split_by_key(df, str(folder), "key", "eom", date(2020, 12, 31))

        a_rows = pl.read_parquet(folder / "a.parquet")
        assert a_rows["eom"].to_list() == [date(2020, 1, 31)]
        assert a_rows["value"].to_list() == [1.0]

        b_rows = pl.read_parquet(folder / "b.parquet")
        assert b_rows["value"].to_list() == [3.0]

    def test_empty_input_creates_folder_no_files(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            schema={"key": pl.String, "eom": pl.Date, "value": pl.Float64},
        )
        folder = tmp_path / "split"
        _write_split_by_key(df, str(folder), "key", "eom", date(2020, 12, 31))

        assert folder.exists()
        assert folder.is_dir()
        assert list(folder.glob("*.parquet")) == []

    def test_keeps_numeric_zero_keys(self, tmp_path: Path) -> None:
        """Numeric keys including 0 must be kept (production never uses these,
        but the helper's falsy-skip footgun is locked down here)."""
        df = pl.DataFrame(
            {
                "key": [0, 1, 2],
                "eom": [date(2020, 1, 31)] * 3,
                "value": [1.0, 2.0, 3.0],
            }
        )
        folder = tmp_path / "split"
        _write_split_by_key(df, str(folder), "key", "eom", date(2020, 12, 31))

        files = sorted(p.name for p in folder.glob("*.parquet"))
        assert files == ["0.parquet", "1.parquet", "2.parquet"]

    def test_filename_is_flat_not_hive(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "key": ["a", "b"],
                "eom": [date(2020, 1, 31), date(2020, 2, 29)],
                "value": [1.0, 2.0],
            }
        )
        folder = tmp_path / "split"
        _write_split_by_key(df, str(folder), "key", "eom", date(2020, 12, 31))

        assert (folder / "a.parquet").is_file()
        assert (folder / "b.parquet").is_file()
        assert not (folder / "key=a").exists()
        assert not (folder / "key=b").exists()
