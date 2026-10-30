"""Unit tests for the final save functions in aux_functions.py."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import polars as pl
import polars.testing
import pytest

from jkp.data.aux_functions import save_daily_ret, save_main_data, setup_folder_structure

DAILY_RET_COLS = [
    "id",
    "date",
    "me",
    "ret",
    "ret_exc",
    "ret_exc_wins",
    "ret_intraday",
    "ret_overnight",
    "ret_intraday_local",
    "ret_overnight_local",
]


def _daily_frame(countries: list[str], n_per_country: int) -> pl.DataFrame:
    """world_dsf_output-shaped frame with rows interleaved across countries."""
    n = len(countries) * n_per_country
    return pl.DataFrame(
        {
            "excntry": countries * n_per_country,
            "id": list(range(n)),
            "date": [date(2020, 1, 1) + timedelta(days=i % 365) for i in range(n)],
            **{c: [float(i) for i in range(n)] for c in DAILY_RET_COLS[2:]},
        }
    )


def _world_data_frame(countries: list[str], n_per_country: int) -> pl.DataFrame:
    """world_data_output-shaped frame with rows interleaved across countries."""
    n = len(countries) * n_per_country
    return pl.DataFrame(
        {
            "excntry": countries * n_per_country,
            "id": list(range(n)),
            "eom": [date(2020, 1, 31)] * n,
            "me": [float(i) for i in range(n)],
            "be_me": [i / 7 for i in range(n)],
        }
    )


# (function, interim input file, processed output dir, input builder, published sort key)
COUNTRY_SPLIT_WRITERS = [
    pytest.param(
        save_daily_ret,
        "world_dsf_output.parquet",
        "return_data/daily_rets_by_country",
        _daily_frame,
        ["id", "date"],
        id="save_daily_ret",
    ),
    pytest.param(
        save_main_data,
        "world_data_output.parquet",
        "characteristics",
        _world_data_frame,
        ["id", "eom"],
        id="save_main_data",
    ),
]


def _read_country_files(out_dir) -> dict[str, pl.DataFrame]:
    return {f.stem: pl.read_parquet(f) for f in sorted(out_dir.glob("*.parquet"))}


@pytest.mark.parametrize(
    ("writer", "input_name", "out_subdir", "build", "keys"), COUNTRY_SPLIT_WRITERS
)
class TestCountrySplitWriters:
    """save_daily_ret and save_main_data split a file by excntry via DuckDB PARTITION_BY."""

    def test_rows_conserved_when_duckdb_splits_a_country(
        self, test_paths, monkeypatch, writer, input_name, out_subdir, build, keys
    ):
        """No rows are lost when DuckDB writes one country's partition as several files.

        DuckDB closes and reopens partition files once more partitions are open than
        partitioned_write_max_open_files allows; each reopen starts a new part file.
        """
        real_connect = duckdb.connect

        def connect_with_tiny_partition_limits(*args, **kwargs):
            con = real_connect(*args, **kwargs)
            con.execute("SET partitioned_write_max_open_files = 1")
            con.execute("SET partitioned_write_flush_threshold = 1")
            return con

        monkeypatch.setattr(duckdb, "connect", connect_with_tiny_partition_limits)
        setup_folder_structure(test_paths)
        data = build(["USA", "CAN", "GBR"], n_per_country=1000)
        data.write_parquet(test_paths.interim_dir / input_name)

        writer(test_paths)

        published = _read_country_files(test_paths.processed_dir / out_subdir)
        assert sum(df.height for df in published.values()) == data.height
        ids = pl.concat(df.select("id") for df in published.values())["id"].sort()
        assert ids.to_list() == data["id"].sort().to_list()

    def test_country_files_equal_input_rows(
        self, test_paths, writer, input_name, out_subdir, build, keys
    ):
        """Each country file holds exactly that country's rows, values and dtypes intact."""
        setup_folder_structure(test_paths)
        data = build(["USA", "CAN", "GBR"], n_per_country=50)
        data.write_parquet(test_paths.interim_dir / input_name)

        writer(test_paths)

        # save_main_data rewrites its input (adds me_lag1, shrinks dtypes) before splitting.
        expected_all = pl.read_parquet(test_paths.interim_dir / input_name)
        published = _read_country_files(test_paths.processed_dir / out_subdir)
        assert set(published) == {"USA", "CAN", "GBR"}
        for country, actual in published.items():
            expected = expected_all.filter(pl.col("excntry") == country)
            expected = expected.select(actual.columns)
            polars.testing.assert_frame_equal(
                actual.sort(keys), expected.sort(keys), check_exact=True
            )
