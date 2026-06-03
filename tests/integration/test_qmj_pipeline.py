"""Integration tests for the quality_minus_junk → merge_qmj_to_world_data pipeline chain.

Drives both functions end-to-end against deterministic synthetic data
(seed=42, from tests.golden.generate_qmj_golden) and asserts output shape,
column presence, left-join semantics, sort order, and value parity with the
standalone qmj.parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.testing
import pytest

from jkp.data.aux_functions import merge_qmj_to_world_data, quality_minus_junk
from jkp.data.paths import DataPaths
from tests.golden.generate_qmj_golden import build_qmj_world_data_input

# ---------------------------------------------------------------------------
# Tolerance (mirrors ToleranceSpec.STANDARD)
# ---------------------------------------------------------------------------
_RTOL = 1e-6
_ATOL = 1e-10

# Columns that qmj adds to world_data
_QMJ_COLS = ["qmj_prof", "qmj_growth", "qmj_safety", "qmj"]


# ---------------------------------------------------------------------------
# Class-scoped fixture: run the full chain once for tests 1-5
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def pipeline_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run quality_minus_junk → merge_qmj_to_world_data once; return artefacts."""
    base = tmp_path_factory.mktemp("qmj_pipeline")
    _setup_dirs(base)
    paths = DataPaths(base_dir=base)

    df_input = build_qmj_world_data_input(seed=42)
    input_path = paths.interim_dir / "world_data_-1.parquet"
    df_input.write_parquet(input_path)

    quality_minus_junk(paths, input_path, 10)
    merge_qmj_to_world_data(paths)

    world_data = pl.read_parquet(paths.interim_dir / "world_data.parquet")
    qmj = pl.read_parquet(paths.interim_dir / "qmj.parquet")

    return {
        "df_input": df_input,
        "world_data": world_data,
        "qmj": qmj,
        "paths": paths,
    }


def _setup_dirs(base: Path) -> None:
    (base / "interim" / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
    (base / "raw" / "raw_tables").mkdir(parents=True, exist_ok=True)
    (base / "processed" / "characteristics").mkdir(parents=True, exist_ok=True)
    (base / "processed" / "return_data").mkdir(parents=True, exist_ok=True)
    (base / "processed" / "other_output").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tests 1-5: use the class-scoped fixture
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pipeline_outputs")
class TestQmjPipeline:
    def test_world_data_has_qmj_cols(self, pipeline_outputs: dict) -> None:
        """world_data.parquet contains all input cols plus qmj_prof/growth/safety/qmj."""
        world_data = pipeline_outputs["world_data"]
        df_input = pipeline_outputs["df_input"]

        for col in df_input.columns:
            assert col in world_data.columns, f"input col {col!r} missing from world_data"

        for col in _QMJ_COLS:
            assert col in world_data.columns, f"qmj col {col!r} missing from world_data"

    def test_gate_failing_rows_present_with_null_qmj(self, pipeline_outputs: dict) -> None:
        """Left-join: gate-failing rows are present in world_data with null qmj columns."""
        df_input = pipeline_outputs["df_input"]
        world_data = pipeline_outputs["world_data"]

        # Identify gate-failing rows from the input
        c1 = (
            (pl.col("common") == 1)
            & (pl.col("primary_sec") == 1)
            & (pl.col("obs_main") == 1)
            & (pl.col("exch_main") == 1)
            & pl.col("ret_exc").is_not_null()
            & pl.col("me").is_not_null()
        )
        failing = df_input.filter(~c1)
        assert len(failing) > 0, "test requires at least one gate-failing row"

        for row in failing.iter_rows(named=True):
            matched = world_data.filter((pl.col("id") == row["id"]) & (pl.col("eom") == row["eom"]))
            assert len(matched) >= 1, (
                f"gate-failing row id={row['id']} eom={row['eom']} not in world_data"
            )
            for qcol in _QMJ_COLS:
                assert matched[qcol].is_null().all(), (
                    f"expected null {qcol} for gate-failing id={row['id']}"
                )

    def test_height_equals_unique_id_eom(self, pipeline_outputs: dict) -> None:
        """Output height equals number of unique (id, eom) pairs in the input."""
        df_input = pipeline_outputs["df_input"]
        world_data = pipeline_outputs["world_data"]

        n_unique = df_input.select(["id", "eom"]).unique().height
        assert world_data.height == n_unique, (
            f"world_data height {world_data.height} != unique (id,eom) count {n_unique}"
        )

    def test_output_sorted_by_id_eom(self, pipeline_outputs: dict) -> None:
        """world_data is sorted by [id, eom]."""
        world_data = pipeline_outputs["world_data"]
        sorted_wd = world_data.sort(["id", "eom"])
        polars.testing.assert_frame_equal(world_data, sorted_wd, check_row_order=True)

    def test_qmj_values_match_standalone(self, pipeline_outputs: dict) -> None:
        """Matched (excntry, id, eom) rows have qmj cols equal to standalone qmj.parquet."""
        world_data = pipeline_outputs["world_data"]
        qmj = pipeline_outputs["qmj"]

        # Join on all three keys; keep only rows present in qmj (inner)
        merged = world_data.join(
            qmj.rename({c: f"ref_{c}" for c in _QMJ_COLS}),
            on=["excntry", "id", "eom"],
            how="inner",
        )
        assert len(merged) > 0, "no matched rows between world_data and qmj"

        for col in _QMJ_COLS:
            ref_col = f"ref_{col}"
            polars.testing.assert_series_equal(
                merged[col].rename(col),
                merged[ref_col].rename(col),
                rtol=_RTOL,
                atol=_ATOL,
                check_names=True,
            )


# ---------------------------------------------------------------------------
# Test 6: duplicate (id, eom) input collapses to unique count
# ---------------------------------------------------------------------------


def test_duplicate_id_eom_collapses(tmp_path: Path) -> None:
    """When world_data_-1 has a duplicated (id, eom) row, world_data height == unique count.

    The contract is:
    - quality_minus_junk reads world_data_-1 (requires unique (excntry,eom,id) triples).
    - merge_qmj_to_world_data reads world_data_-1 and calls .unique(["id","eom"]),
      so it is what deduplicates the world_data side.
    We therefore run quality_minus_junk on clean data, then overwrite world_data_-1
    with a duplicated version before calling merge_qmj_to_world_data.  Keep is
    non-deterministic; we assert only on height.
    """
    _setup_dirs(tmp_path)
    paths = DataPaths(base_dir=tmp_path)

    df_base = build_qmj_world_data_input(seed=42)
    input_path = paths.interim_dir / "world_data_-1.parquet"

    # Step 1: run quality_minus_junk on clean (duplicate-free) input
    df_base.write_parquet(input_path)
    quality_minus_junk(paths, input_path, 10)

    # Step 2: overwrite world_data_-1 with a version containing one duplicate row
    dup_row = df_base.filter(
        (pl.col("common") == 1) & pl.col("ret_exc").is_not_null() & pl.col("me").is_not_null()
    ).head(1)
    df_with_dup = pl.concat([df_base, dup_row])
    df_with_dup.write_parquet(input_path)

    # Step 3: merge — this is what deduplicates on (id, eom)
    merge_qmj_to_world_data(paths)

    world_data = pl.read_parquet(paths.interim_dir / "world_data.parquet")
    n_unique = df_with_dup.select(["id", "eom"]).unique().height

    assert world_data.height == n_unique, (
        f"expected {n_unique} rows after dedup, got {world_data.height}"
    )
