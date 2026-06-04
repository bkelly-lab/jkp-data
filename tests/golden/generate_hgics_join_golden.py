"""Generate golden fixture for hgics_join().

Run with:
    uv run python -m tests.golden.generate_hgics_join_golden

Writes:
    tests/golden/fixtures/hgics_join/comp_hgics.parquet
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import hgics_join
from jkp.data.paths import DataPaths

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "hgics_join"


def build_hgics_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build deterministic NA and GL GICS history fixtures.

    Scenarios covered by the returned (comp_hgics_na, comp_hgics_gl) pair:
        100000 — NA-only, 3-day span (2020-01-01 to 2020-01-03), gics=10101010.
        200000 — GL-only, 2-day span (2020-02-01 to 2020-02-02), gics=20202020.
        300000 — Present in both NA and GL on the same 2-day span (2020-03-01
                 to 2020-03-02) with different gics; national (30303030) must
                 win the coalesce over global (39393939).
        400000 — NA-only, 2-day span (2020-04-01 to 2020-04-02), gics=None;
                 comp_hgics() rewrites null to -999, which must propagate
                 through hgics_join unchanged.
    """
    comp_hgics_na = pl.DataFrame(
        {
            "gvkey": ["100000", "300000", "400000"],
            "indfrom": [date(2020, 1, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "indthru": [date(2020, 1, 3), date(2020, 3, 2), date(2020, 4, 2)],
            "gics": [10101010, 30303030, None],
        },
        schema={
            "gvkey": pl.Utf8,
            "indfrom": pl.Date,
            "indthru": pl.Date,
            "gics": pl.Int64,
        },
    )

    comp_hgics_gl = pl.DataFrame(
        {
            "gvkey": ["200000", "300000"],
            "indfrom": [date(2020, 2, 1), date(2020, 3, 1)],
            "indthru": [date(2020, 2, 2), date(2020, 3, 2)],
            "gics": [20202020, 39393939],
        },
        schema={
            "gvkey": pl.Utf8,
            "indfrom": pl.Date,
            "indthru": pl.Date,
            "gics": pl.Int64,
        },
    )
    return comp_hgics_na, comp_hgics_gl


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        (paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
        comp_hgics_na, comp_hgics_gl = build_hgics_inputs()
        comp_hgics_na.write_parquet(paths.interim_dir / "raw_data_dfs" / "comp_hgics_na.parquet")
        comp_hgics_gl.write_parquet(paths.interim_dir / "raw_data_dfs" / "comp_hgics_gl.parquet")
        hgics_join(paths)
        out_path = GOLDEN_DIR / "comp_hgics.parquet"
        pl.read_parquet(paths.interim_dir / "comp_hgics.parquet").write_parquet(out_path)
        n_rows = pl.read_parquet(out_path).height
        print(f"comp_hgics.parquet: {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
