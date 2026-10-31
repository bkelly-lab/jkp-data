"""Integration tests for the final merge chain and the save tail.

Drives the functions in the order main.py runs them, against the seeded
synthetic inputs from tests.golden.final_stage_inputs.
"""

from __future__ import annotations

import polars as pl
import pytest

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.golden.final_stage_inputs import OTHER_OUTPUT_FILES, write_full_inputs

pytestmark = pytest.mark.integration

EXPECTED_PROCESSED = {
    "accounting_data/annual.parquet",
    "accounting_data/quarterly.parquet",
    "characteristics/DEU.parquet",
    "characteristics/GBR.parquet",
    "characteristics/USA.parquet",
    "characteristics/null_country.parquet",
    "characteristics/world_data.parquet",
    "return_data/daily_rets_by_country/DEU.parquet",
    "return_data/daily_rets_by_country/GBR.parquet",
    "return_data/daily_rets_by_country/USA.parquet",
    "return_data/daily_rets_by_country/null_country.parquet",
    "return_data/world_dsf.parquet",
    "return_data/world_ret_monthly.parquet",
    *(f"other_output/{n}" for n in OTHER_OUTPUT_FILES),
}
INPUT_FILES = {
    "market_chars_m": "market_chars_m.parquet",
    "acc_chars": "acc_chars_world.parquet",
    "beta": "beta_60m.parquet",
    "rm12": "resmom_ff3_12_1.parquet",
    "rm6": "resmom_ff3_6_1.parquet",
    "mp": "mp_factors.parquet",
    "mcd": "market_chars_d.parquet",
    "age": "firm_age.parquet",
    "qmj": "qmj.parquet",
}


@pytest.fixture
def merge_outputs(test_paths: DataPaths) -> dict:
    """Run create_world_data_prelim -> merge_world_data_prelim -> merge_qmj_to_world_data."""
    write_full_inputs(test_paths)
    i = test_paths.interim_dir
    af.create_world_data_prelim(
        test_paths,
        i / "world_msf.parquet",
        i / "market_chars_m.parquet",
        i / "acc_chars_world.parquet",
        i / "world_data_prelim.parquet",
    )
    af.merge_world_data_prelim(test_paths)
    af.merge_qmj_to_world_data(test_paths)
    return {
        "msf": pl.read_parquet(i / "world_msf.parquet"),
        "prelim": pl.read_parquet(i / "world_data_prelim.parquet"),
        "world_data_-1": pl.read_parquet(i / "world_data_-1.parquet"),
        "world_data": pl.read_parquet(i / "world_data.parquet"),
        "inputs": {k: pl.read_parquet(i / f) for k, f in INPUT_FILES.items()},
    }


class TestMergeChain:
    def test_key_unique_and_row_count_equals_msf(self, merge_outputs: dict) -> None:
        """Every stage keeps msf's rows; checked before merge_qmj's unique() can hide a fan-out."""
        msf = merge_outputs["msf"]
        msf_keys = set(msf.select("id", "eom").rows())
        for stage in ("prelim", "world_data_-1", "world_data"):
            df = merge_outputs[stage]
            assert df.height == msf.height, stage
            assert df.select(["id", "eom"]).is_duplicated().sum() == 0, stage
            assert set(df.select("id", "eom").rows()) == msf_keys, stage

    def test_every_input_column_present(self, merge_outputs: dict) -> None:
        cols = set(merge_outputs["world_data"].columns)
        for name, df in merge_outputs["inputs"].items():
            missing = set(df.columns) - cols - {"public_date", "age_extra"}
            assert not missing, f"{name}: {missing} missing from world_data"
        # create_world_data_prelim drops these msf columns
        assert not cols & {"div_tot", "div_cash", "div_spc", "source"}
        # merge_world_data_prelim keeps only (id, eom, age) from firm_age
        assert "age_extra" not in cols

    def test_unmatched_keys_give_nulls_not_rows(self, merge_outputs: dict) -> None:
        wd = merge_outputs["world_data"]
        assert wd.filter(pl.col("id") == 99).is_empty()
        # id 3 is absent from market_chars_m
        assert (
            wd.filter(pl.col("id") == 3)["market_equity"].null_count()
            == wd.filter(pl.col("id") == 3).height
        )
        # null excntry rows never match qmj (null != null in a join) but survive
        nulls = wd.filter(pl.col("excntry").is_null())
        assert nulls.height > 0
        assert nulls["qmj"].null_count() == nulls.height

    def test_sorted_by_id_eom(self, merge_outputs: dict) -> None:
        wd = merge_outputs["world_data"]
        assert wd.equals(wd.sort(["id", "eom"]))


@pytest.fixture
def save_tail(test_paths: DataPaths) -> dict:
    """Run main.py's tail up to, but not including, save_full_files_and_cleanup."""
    af.setup_folder_structure(test_paths)
    write_full_inputs(test_paths)
    for fn in (
        af.filter_dsf,
        af.filter_msf,
        af.filter_world,
        af.save_main_data,
        af.save_daily_ret,
        af.save_monthly_ret,
        af.save_accounting_data,
        af.save_output_files,
    ):
        fn(test_paths)
    i = test_paths.interim_dir
    counts = {
        n: pl.scan_parquet(i / f"{n}_output.parquet").select(pl.len()).collect().item()
        for n in ("world_dsf", "world_msf", "world_data")
    }
    return {"paths": test_paths, "counts": counts}


def _processed_files(paths: DataPaths) -> set[str]:
    return {
        p.relative_to(paths.processed_dir).as_posix()
        for p in paths.processed_dir.rglob("*")
        if p.is_file()
    }


class TestSaveTail:
    def test_filters_drop_non_main_rows(self, save_tail: dict) -> None:
        i = save_tail["paths"].interim_dir
        for n in ("world_dsf", "world_msf", "world_data"):
            full = pl.read_parquet(i / f"{n}.parquet").height
            assert 0 < save_tail["counts"][n] < full

    def test_rows_conserved_into_published_files(self, save_tail: dict) -> None:
        paths, counts = save_tail["paths"], save_tail["counts"]
        # each assertion runs after the last step that touches that file
        monthly = pl.read_parquet(paths.processed_dir / "return_data" / "world_ret_monthly.parquet")
        assert monthly.height == counts["world_msf"]
        daily = pl.scan_parquet(
            paths.processed_dir / "return_data" / "daily_rets_by_country" / "*.parquet"
        )
        assert daily.select(pl.len()).collect().item() == counts["world_dsf"]
        chars = pl.scan_parquet(
            [
                p
                for p in (paths.processed_dir / "characteristics").glob("*.parquet")
                if p.stem != "world_data"
            ]
        )
        assert chars.select(pl.len()).collect().item() == counts["world_data"]

    def test_cleanup_layout_and_counts(self, save_tail: dict) -> None:
        paths, counts = save_tail["paths"], save_tail["counts"]
        af.save_full_files_and_cleanup(paths, clear_interim=True)

        assert _processed_files(paths) == EXPECTED_PROCESSED
        assert list(paths.interim_dir.iterdir()) == []
        assert list(paths.raw_dir.iterdir()) == []
        rd = paths.processed_dir / "return_data"
        assert pl.read_parquet(rd / "world_dsf.parquet").height == counts["world_dsf"]
        assert (
            pl.read_parquet(paths.processed_dir / "characteristics" / "world_data.parquet").height
            == counts["world_data"]
        )

    def test_accounting_drops_only_null_source(self, save_tail: dict) -> None:
        paths = save_tail["paths"]
        for kind, short in (("qtr", "quarterly"), ("ann", "annual")):
            src = pl.read_parquet(paths.interim_dir / f"acc_std_{kind}.parquet")
            out = pl.read_parquet(paths.processed_dir / "accounting_data" / f"{short}.parquet")
            assert out.height == src.filter(pl.col("source").is_not_null()).height
            assert out["source"].null_count() == 0

    def test_null_excntry_daily_file_is_null_country(self, save_tail: dict) -> None:
        paths = save_tail["paths"]
        f = paths.processed_dir / "return_data" / "daily_rets_by_country" / "null_country.parquet"
        assert sorted(pl.read_parquet(f)["id"].unique().to_list()) == [
            11
        ]  # id 12 has a null primary_sec

    def test_null_excntry_characteristics_file_named_null_country(self, save_tail: dict) -> None:
        chars = save_tail["paths"].processed_dir / "characteristics"
        assert (chars / "null_country.parquet").exists()
