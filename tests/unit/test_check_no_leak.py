"""Tests for the leak check in ``tests/golden/calibration/check_no_leak.py``."""

from pathlib import Path

import polars as pl

from jkp.data.paths import DataPaths
from tests.golden.calibration.check_no_leak import ITEMS, leaks
from tests.golden.synthetic_compustat import OUT_DIR as SYNTHETIC_DIR

SYNTHETIC = DataPaths(SYNTHETIC_DIR)


def test_synthetic_data_matches_itself() -> None:
    counts = leaks(SYNTHETIC, SYNTHETIC)
    assert counts["gvkeys"] > 0
    for table, items in ITEMS.items():
        df = pl.read_parquet(SYNTHETIC.raw_tables_dir / f"{table}.parquet")
        assert counts[table] == df.drop_nulls(list(items)).height


def test_changed_gvkeys_and_assets_match_nothing(tmp_path: Path) -> None:
    real = DataPaths(tmp_path)
    real.raw_tables_dir.mkdir(parents=True)
    real.interim_dir.mkdir(parents=True)
    for table, (at, _) in ITEMS.items():
        df = pl.read_parquet(SYNTHETIC.raw_tables_dir / f"{table}.parquet")
        changed = df.with_columns(pl.col(at) + 1, gvkey="1" + pl.col("gvkey"))
        changed.write_parquet(real.raw_tables_dir / f"{table}.parquet")
    msf = pl.read_parquet(SYNTHETIC.interim_dir / "world_msf.parquet")
    msf.with_columns(gvkey="1" + pl.col("gvkey")).write_parquet(
        real.interim_dir / "world_msf.parquet"
    )
    assert leaks(real, SYNTHETIC) == dict.fromkeys(["gvkeys", *ITEMS], 0)
