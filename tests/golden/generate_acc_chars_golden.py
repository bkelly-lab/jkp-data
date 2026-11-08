"""Generate the accounting-stage golden fixtures (issue #303, step 4).

Run with:
    PYTHONPATH=src:. uv run --no-sync python -m tests.golden.generate_acc_chars_golden [out_dir]
or with --regen-golden on tests/golden/test_acc_chars_golden.py.

Runs the accounting stage, as main.py does, on the committed synthetic Compustat inputs in
tests/golden/fixtures/synthetic_compustat/ (written by synthetic_compustat.py) and writes to
tests/golden/fixtures/acc_chars/ (or out_dir):
    personas     every output row of the persona firms (gvkey 900101 up)
    random       the June and December public dates of the random firms
    fingerprint  per output column over all rows: dtype, non-null count, null count, sum, min, max

personas and random store Float64 columns as Float32, which halves their size; the fingerprint is
computed in full precision, so it still locks the dtypes and small changes anywhere.
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
from tests.golden.synthetic_compustat import OUT_DIR as INPUT_DIR
from tests.golden.synthetic_compustat import PERSONA_GVKEY

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "acc_chars"
GOLDEN_NAMES = ["personas", "random", "fingerprint"]
KEYS = ["gvkey", "public_date"]


def run_stage(paths: DataPaths) -> None:
    """
    Description:
        Run the accounting stage on the synthetic inputs, in main.py's order.
    Steps:
        1) Create the folder tree; copy the synthetic raw tables and world_msf into it.
        2) Build __fx1 from comp_exrt_dly (gen_fx1, which the download step runs in main.py).
        3) standardized_accounting_data, create_acc_chars (annual, then quarterly),
           combine_ann_qtr_chars, with main.py's arguments.
    Output:
        The stage's files in paths.interim_dir, ending with acc_chars_world.parquet.
    """
    af.setup_folder_structure(paths)
    shutil.copytree(INPUT_DIR / "raw" / "raw_tables", paths.raw_tables_dir, dirs_exist_ok=True)
    shutil.copy(INPUT_DIR / "interim" / "world_msf.parquet", paths.interim_dir)
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


def compute_outputs(paths: DataPaths) -> dict[str, pl.DataFrame]:
    """
    Description:
        The golden frames of one stage run.
    Steps:
        1) Run the stage and read acc_chars_world sorted on (gvkey, public_date).
        2) Keep every persona row and the June and December rows of the random firms, with
           Float64 columns as Float32, and the fingerprint of all rows.
    Output:
        Dict of golden name to DataFrame.
    """
    run_stage(paths)
    chars = pl.read_parquet(paths.interim_dir / "acc_chars_world.parquet").sort(KEYS)
    persona = pl.col("gvkey").cast(pl.Int64) >= PERSONA_GVKEY
    rows = chars.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    return {
        "personas": rows.filter(persona),
        "random": rows.filter(~persona & pl.col("public_date").dt.month().is_in([6, 12])),
        "fingerprint": fingerprint(chars),
    }


def write_goldens(outputs: dict[str, pl.DataFrame], out_dir: Path = GOLDEN_DIR) -> None:
    """Write each golden frame to out_dir/<name>.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.write_parquet(out_dir / f"{name}.parquet", compression="zstd", compression_level=22)


if __name__ == "__main__":
    pl.Config.set_engine_affinity("in-memory")  # as main.py's run_pipeline and tests/conftest.py
    with tempfile.TemporaryDirectory() as tmp:
        outputs = compute_outputs(DataPaths(Path(tmp)))
    write_goldens(outputs, Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_DIR)
