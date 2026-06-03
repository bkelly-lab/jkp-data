"""Generate golden fixture for crsp_industry().

Run with:
    uv run python -m tests.golden.generate_crsp_industry_golden

Writes:
    tests/golden/fixtures/crsp_industry/crsp_ind.parquet
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import crsp_industry
from jkp.data.paths import DataPaths

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "crsp_industry"


def build_permno0_input(seed: int = 42) -> pl.DataFrame:
    """Build a deterministic permno0 fixture exercising every code path in crsp_industry.

    Permnos:
        10001 — two non-overlapping spans (Jan 1-3 with sic=7372, naics=511210;
                Jan 6-8 with sic=7370, naics=511200).
        10002 — single short span (Feb 1-2) with sic=0 (should become null) and
                naics=None (should be preserved).
        10003 — two overlapping spans (Mar 1-4 and Mar 3-6, both same sic/naics)
                to exercise .unique(["permno", "date"]) dedup.
    """
    # `seed` is unused for this hand-built fixture; kept in the signature to
    # match the build_*_input(seed) convention used by other generators.
    del seed
    return pl.DataFrame(
        {
            "permno": [10001, 10001, 10002, 10003, 10003],
            "permco": [1, 1, 2, 3, 3],
            "secinfostartdt": [
                date(2020, 1, 1),
                date(2020, 1, 6),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 3, 3),
            ],
            "secinfoenddt": [
                date(2020, 1, 3),
                date(2020, 1, 8),
                date(2020, 2, 2),
                date(2020, 3, 4),
                date(2020, 3, 6),
            ],
            "sic": [7372, 7370, 0, 6020, 6020],
            "naics": [511210, 511200, None, 522110, 522110],
        },
        schema={
            "permno": pl.Int64,
            "permco": pl.Int64,
            "secinfostartdt": pl.Date,
            "secinfoenddt": pl.Date,
            "sic": pl.Int64,
            "naics": pl.Int64,
        },
    )


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        (paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
        build_permno0_input().write_parquet(paths.interim_dir / "raw_data_dfs" / "permno0.parquet")
        crsp_industry(paths)
        out_path = GOLDEN_DIR / "crsp_ind.parquet"
        pl.read_parquet(paths.interim_dir / "crsp_ind.parquet").write_parquet(out_path)
        n_rows = pl.read_parquet(out_path).height
        print(f"crsp_ind.parquet: {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
