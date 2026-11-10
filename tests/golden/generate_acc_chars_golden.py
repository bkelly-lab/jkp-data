"""Generate the accounting-stage golden fingerprint (issue #303, step 4).

Run with:
    PYTHONPATH=src:. uv run --no-sync python -m tests.golden.generate_acc_chars_golden [out_dir]
or with --regen-golden on tests/golden/test_acc_chars_golden.py.

Generates the synthetic Compustat inputs (synthetic_compustat.py), runs the accounting stage on
them as main.py does, and writes the fingerprint of its output to
tests/golden/fixtures/acc_chars/fingerprint.parquet (or out_dir): per output column over all rows,
its dtype, non-null count, null count, sum, min and max, in full precision. The oracle
(tests/unit/test_acc_chars_oracle.py) checks the values row by row, so no rows are stored.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl

from jkp.data import aux_functions as af
from jkp.data.config import ACCOUNTING_START_DATE
from jkp.data.paths import DataPaths
from tests.golden.synthetic_compustat import generate

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "acc_chars"
KEYS = ["gvkey", "public_date"]


def run_stage(paths: DataPaths, inputs: DataPaths) -> None:
    """
    Description:
        Run the accounting stage on the synthetic inputs under inputs, in main.py's order.
    Steps:
        1) Create the folder tree; copy the inputs' raw tables and world_msf into it.
        2) Build __fx1 from comp_exrt_dly (gen_fx1, which the download step runs in main.py).
        3) standardized_accounting_data, create_acc_chars (annual, then quarterly),
           combine_ann_qtr_chars, with main.py's arguments.
    Output:
        The stage's files in paths.interim_dir, ending with acc_chars_world.parquet.
    """
    af.setup_folder_structure(paths)
    shutil.copytree(inputs.raw_tables_dir, paths.raw_tables_dir, dirs_exist_ok=True)
    shutil.copy(inputs.interim_dir / "world_msf.parquet", paths.interim_dir)
    af.gen_fx1(paths)
    i = paths.interim_dir
    msf = i / "world_msf.parquet"
    chars = af.acc_chars_list()
    af.standardized_accounting_data(paths, "world", 1, msf, 1, ACCOUNTING_START_DATE)
    af.create_acc_chars(
        paths, i / "acc_std_ann.parquet", i / "achars_world.parquet", 4, 18, chars, msf, ""
    )
    af.create_acc_chars(
        paths, i / "acc_std_qtr.parquet", i / "qchars_world.parquet", 4, 18, chars, msf, "_qitem"
    )
    af.combine_ann_qtr_chars(
        paths, i / "achars_world.parquet", i / "qchars_world.parquet", chars, "_qitem"
    )


def fingerprint(df: pl.DataFrame) -> pl.DataFrame:
    """One row per column: dtype, non-null count, null count, and sum, min and max if numeric."""

    def stats(column: str, dtype: pl.DataType) -> pl.Expr:
        x = pl.col(column).cast(pl.Float64)
        none = pl.lit(None, pl.Float64)
        numeric = dtype.is_numeric()
        return pl.struct(
            column=pl.lit(column),
            dtype=pl.lit(str(dtype)),
            count=pl.col(column).count(),
            nulls=pl.col(column).null_count(),
            sum=x.sum() if numeric else none,
            min=x.min() if numeric else none,
            max=x.max() if numeric else none,
        )

    return (
        df.select(pl.concat_list(stats(c, t) for c, t in df.schema.items()).alias("stats"))
        .explode("stats")
        .unnest("stats")
    )


def stage_fingerprint(paths: DataPaths) -> pl.DataFrame:
    """The fingerprint of acc_chars_world.parquet in paths, sorted on KEYS first."""
    return fingerprint(pl.read_parquet(paths.interim_dir / "acc_chars_world.parquet").sort(KEYS))


def write_golden(golden: pl.DataFrame, out_dir: Path = GOLDEN_DIR) -> None:
    """Write the fingerprint to out_dir/fingerprint.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    golden.write_parquet(out_dir / "fingerprint.parquet", compression="zstd", compression_level=22)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        inputs, stage = DataPaths(Path(tmp) / "inputs"), DataPaths(Path(tmp) / "stage")
        generate(inputs.base_dir)
        run_stage(stage, inputs)
        write_golden(
            stage_fingerprint(stage), Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_DIR
        )
