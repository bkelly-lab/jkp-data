"""Unit tests for the final save functions in aux_functions.py."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import polars as pl
import polars.testing
import pytest

from jkp.data import aux_functions as af
from jkp.data.aux_functions import (
    save_accounting_data,
    save_daily_ret,
    save_full_files_and_cleanup,
    save_main_data,
    save_monthly_ret,
    save_output_files,
    setup_folder_structure,
)

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

        real_flatten = af._flatten_country_partitions
        part_counts: list[int] = []

        def count_parts_then_flatten(out_dir):
            part_counts.extend(len(list(d.glob("*.parquet"))) for d in out_dir.glob("excntry=*"))
            real_flatten(out_dir)

        monkeypatch.setattr(duckdb, "connect", connect_with_tiny_partition_limits)
        monkeypatch.setattr(af, "_flatten_country_partitions", count_parts_then_flatten)
        setup_folder_structure(test_paths)
        data = build(["USA", "CAN", "GBR"], n_per_country=1000)
        data.write_parquet(test_paths.interim_dir / input_name)

        writer(test_paths)

        assert max(part_counts) > 1, "DuckDB wrote one file per country; nothing was merged"
        published = _read_country_files(test_paths.processed_dir / out_subdir)
        assert sum(df.height for df in published.values()) == data.height
        ids = pl.concat(df.select("id") for df in published.values())["id"].sort()
        assert ids.to_list() == data["id"].sort().to_list()

    def test_null_country_goes_to_null_country_file(
        self, test_paths, writer, input_name, out_subdir, build, keys
    ):
        setup_folder_structure(test_paths)
        data = build(["USA", "CAN"], n_per_country=5)
        data = data.with_columns(
            pl.when(pl.col("id") < 3).then(None).otherwise(pl.col("excntry")).alias("excntry")
        )
        data.write_parquet(test_paths.interim_dir / input_name)

        writer(test_paths)

        published = _read_country_files(test_paths.processed_dir / out_subdir)
        assert set(published) == {"USA", "CAN", "null_country"}
        assert sorted(published["null_country"]["id"].to_list()) == [0, 1, 2]

    def test_country_files_equal_input_rows(
        self, test_paths, writer, input_name, out_subdir, build, keys
    ):
        """Each country file holds exactly that country's rows, values and dtypes intact."""
        setup_folder_structure(test_paths)
        data = build(["USA", "CAN", "GBR"], n_per_country=50)
        data.write_parquet(test_paths.interim_dir / input_name)

        writer(test_paths)

        # Compare with the input, not save_main_data's rewritten copy of it; me_lag1 and
        # the exact column lists are tested per writer.
        published = _read_country_files(test_paths.processed_dir / out_subdir)
        assert set(published) == {"USA", "CAN", "GBR"}
        for country, actual in published.items():
            expected = data.filter(pl.col("excntry") == country)
            shared = [c for c in actual.columns if c in expected.columns]
            polars.testing.assert_frame_equal(
                actual.select(shared).sort(keys),
                expected.select(shared).sort(keys),
                check_exact=True,
            )

    def test_rerun_removes_files_from_the_previous_run(
        self, test_paths, writer, input_name, out_subdir, build, keys
    ):
        """A rerun with fewer countries leaves no stale country files (COPY ... OVERWRITE)."""
        setup_folder_structure(test_paths)
        build(["USA", "CAN", "GBR"], n_per_country=5).write_parquet(
            test_paths.interim_dir / input_name
        )
        writer(test_paths)
        data = build(["USA", "CAN"], n_per_country=3)
        data.write_parquet(test_paths.interim_dir / input_name)

        writer(test_paths)

        published = _read_country_files(test_paths.processed_dir / out_subdir)
        assert set(published) == {"USA", "CAN"}
        assert sum(df.height for df in published.values()) == data.height


class TestSaveMainData:
    def test_me_lag1_is_previous_month_me_within_id(self, test_paths):
        """me_lag1 is the same id's me one month earlier; null at a first month or after a gap."""
        setup_folder_structure(test_paths)
        dec, jan, feb, mar, may = (
            date(2019, 12, 31),
            date(2020, 1, 31),
            date(2020, 2, 29),
            date(2020, 3, 31),
            date(2020, 5, 31),
        )
        # Rows shuffled. id 1 skips April; id 2 crosses a year boundary.
        data = pl.DataFrame(
            {
                "excntry": ["USA"] * 6,
                "id": [1, 2, 1, 1, 2, 1],
                "eom": [may, jan, jan, mar, dec, feb],
                "me": [50.0, 6.0, 10.0, 30.0, 5.0, 20.0],
            }
        )
        data.write_parquet(test_paths.interim_dir / "world_data_output.parquet")

        save_main_data(test_paths)

        out = pl.read_parquet(test_paths.processed_dir / "characteristics" / "USA.parquet")
        assert out.sort("id", "eom")["me_lag1"].to_list() == [None, 10.0, 20.0, None, None, 5.0]

    def test_country_files_columns_exact(self, test_paths):
        """Each file has the input columns plus me_lag1, excntry included and single-valued."""
        setup_folder_structure(test_paths)
        data = _world_data_frame(["USA", "CAN"], n_per_country=5)
        data.write_parquet(test_paths.interim_dir / "world_data_output.parquet")

        save_main_data(test_paths)

        published = _read_country_files(test_paths.processed_dir / "characteristics")
        assert set(published) == {"USA", "CAN"}
        for country, df in published.items():
            assert df.columns == [*data.columns, "me_lag1"]
            assert df["excntry"].unique().to_list() == [country]


OTHER_OUTPUT_FILES = [
    "market_returns.parquet",
    "market_returns_daily.parquet",
    "nyse_cutoffs.parquet",
    "return_cutoffs.parquet",
    "return_cutoffs_daily.parquet",
    "ap_factors_monthly.parquet",
    "ap_factors_daily.parquet",
]

MONTHLY_RET_COLS = [
    "excntry",
    "id",
    "source_crsp",
    "eom",
    "me",
    "ret_exc",
    "ret",
    "ret_local",
    "ret_exc_wins",
    "ret_intraday",
    "ret_overnight",
    "ret_intraday_local",
    "ret_overnight_local",
    "ret_intraday_lead1m",
    "ret_overnight_lead1m",
    "ret_intraday_local_lead1m",
    "ret_overnight_local_lead1m",
]


def _monthly_frame(n: int = 20) -> pl.DataFrame:
    """world_msf_output-shaped frame with an extra column that must not be published."""
    return pl.DataFrame(
        {
            "excntry": ["USA", "CAN"] * (n // 2),
            "id": list(range(n)),
            "source_crsp": [1, 0] * (n // 2),
            "eom": [date(2020, 1, 31)] * n,
            "me": [float(i) * 1000.5 for i in range(n)],
            **{c: [i / 100 for i in range(n)] for c in MONTHLY_RET_COLS[5:]},
            "ret_local": [i / 50 for i in range(n)],
            "extra_col": ["x"] * n,
        }
    )


class TestSaveOutputFiles:
    def test_files_copied_byte_identical_and_interim_kept(self, test_paths):
        setup_folder_structure(test_paths)
        for i, name in enumerate(OTHER_OUTPUT_FILES):
            pl.DataFrame({"a": [i, i + 1]}).write_parquet(test_paths.interim_dir / name)

        save_output_files(test_paths)

        for name in OTHER_OUTPUT_FILES:
            src = test_paths.interim_dir / name
            dst = test_paths.processed_dir / "other_output" / name
            assert src.exists()
            assert dst.read_bytes() == src.read_bytes()

    def test_missing_source_raises(self, test_paths):
        setup_folder_structure(test_paths)
        for name in OTHER_OUTPUT_FILES[:-1]:
            pl.DataFrame({"a": [1]}).write_parquet(test_paths.interim_dir / name)

        with pytest.raises(FileNotFoundError):
            save_output_files(test_paths)


class TestSaveDailyRet:
    def _run(self, test_paths, data: pl.DataFrame) -> None:
        data.write_parquet(test_paths.interim_dir / "world_dsf_output.parquet")
        save_daily_ret(test_paths)

    def test_no_partition_folders_remain_and_columns_exact(self, test_paths):
        setup_folder_structure(test_paths)
        data = _daily_frame(["USA", "CAN"], n_per_country=5).with_columns(extra=pl.lit("x"))

        self._run(test_paths, data)

        out_dir = test_paths.processed_dir / "return_data/daily_rets_by_country"
        assert list(out_dir.glob("excntry=*")) == []
        for df in _read_country_files(out_dir).values():
            assert df.columns == DAILY_RET_COLS


class TestSaveMonthlyRet:
    def _out(self, test_paths) -> pl.DataFrame:
        return pl.read_parquet(
            test_paths.processed_dir / "return_data" / "world_ret_monthly.parquet"
        )

    def test_selects_exact_columns_in_order(self, test_paths):
        setup_folder_structure(test_paths)
        _monthly_frame().write_parquet(test_paths.interim_dir / "world_msf_output.parquet")

        save_monthly_ret(test_paths)

        assert self._out(test_paths).columns == MONTHLY_RET_COLS

    def test_missing_required_column_raises(self, test_paths):
        setup_folder_structure(test_paths)
        _monthly_frame().drop("ret_local").write_parquet(
            test_paths.interim_dir / "world_msf_output.parquet"
        )

        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            save_monthly_ret(test_paths)

    def test_values_and_dtypes_equal_input(self, test_paths):
        setup_folder_structure(test_paths)
        data = _monthly_frame()
        data.write_parquet(test_paths.interim_dir / "world_msf_output.parquet")

        save_monthly_ret(test_paths)

        polars.testing.assert_frame_equal(
            self._out(test_paths), data.select(MONTHLY_RET_COLS), check_exact=True
        )


class TestSaveAccountingData:
    def test_null_source_rows_removed(self, test_paths):
        setup_folder_structure(test_paths)
        for name, offset in (("acc_std_qtr", 0), ("acc_std_ann", 100)):
            pl.DataFrame(
                {
                    "gvkey": [f"{offset + i:06d}" for i in range(6)],
                    "source": ["a", None, "b", None, "c", "d"],
                    "at": [float(i) for i in range(6)],
                }
            ).write_parquet(test_paths.interim_dir / f"{name}.parquet")

        save_accounting_data(test_paths)

        for name, out in (("acc_std_qtr", "quarterly"), ("acc_std_ann", "annual")):
            src = pl.read_parquet(test_paths.interim_dir / f"{name}.parquet")
            expected = src.filter(pl.col("source").is_not_null())
            actual = pl.read_parquet(
                test_paths.processed_dir / "accounting_data" / f"{out}.parquet"
            )
            assert actual.height == 4
            polars.testing.assert_frame_equal(actual, expected, check_exact=True)


class TestSaveFullFilesAndCleanup:
    def _setup(self, test_paths) -> tuple[pl.DataFrame, pl.DataFrame]:
        setup_folder_structure(test_paths)
        dsf = _daily_frame(["USA", "CAN"], n_per_country=5)
        wd = _world_data_frame(["USA", "CAN"], n_per_country=5)
        dsf.write_parquet(test_paths.interim_dir / "world_dsf_output.parquet")
        wd.write_parquet(test_paths.interim_dir / "world_data_output.parquet")
        (test_paths.interim_dir / "sub").mkdir()
        (test_paths.interim_dir / "sub" / "f.txt").write_text("x")
        (test_paths.raw_dir / "sub2").mkdir(parents=True, exist_ok=True)
        (test_paths.raw_dir / "sub2" / "g.txt").write_text("y")
        (test_paths.raw_dir / "loose.txt").write_text("z")
        return dsf, wd

    def test_full_files_match_inputs(self, test_paths):
        dsf, wd = self._setup(test_paths)

        save_full_files_and_cleanup(test_paths, clear_interim=False)

        for expected, path in (
            (dsf, test_paths.processed_dir / "return_data" / "world_dsf.parquet"),
            (wd, test_paths.processed_dir / "characteristics" / "world_data.parquet"),
        ):
            polars.testing.assert_frame_equal(pl.read_parquet(path), expected, check_exact=True)

    def test_clear_interim_true_empties_but_keeps_folders(self, test_paths):
        self._setup(test_paths)
        before = sorted(p.name for p in test_paths.processed_dir.rglob("*"))

        save_full_files_and_cleanup(test_paths, clear_interim=True)

        assert test_paths.interim_dir.is_dir()
        assert test_paths.raw_dir.is_dir()
        assert list(test_paths.interim_dir.iterdir()) == []
        assert list(test_paths.raw_dir.iterdir()) == []
        after = {p.name for p in test_paths.processed_dir.rglob("*")}
        assert set(before) <= after
        assert {"world_dsf.parquet", "world_data.parquet"} <= after

    def test_clear_interim_false_leaves_folders_intact(self, test_paths):
        self._setup(test_paths)

        save_full_files_and_cleanup(test_paths, clear_interim=False)

        assert (test_paths.interim_dir / "world_dsf_output.parquet").exists()
        assert (test_paths.interim_dir / "sub" / "f.txt").exists()
        assert (test_paths.raw_dir / "sub2" / "g.txt").exists()
        assert (test_paths.raw_dir / "loose.txt").exists()
