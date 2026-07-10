"""Generate golden fixture for comp_industry().

Run with:
    uv run python -m tests.golden.generate_comp_industry_golden

Writes:
    tests/golden/fixtures/comp_industry/comp_ind.parquet
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import comp_industry
from jkp.data.paths import DataPaths
from tests.golden.comp_industry_stubs import comp_industry_upstream_stubs

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "comp_industry"


def build_comp_industry_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build deterministic (comp_other, comp_hgics) inputs for comp_industry.

    Scenarios:
        100000 — Sparse dates in both sources (2020-01-01 and 2020-01-05) with
                 identical codes on each anchor. Exercises gap-fill: aux_date on
                 the Jan-1 row equals Jan-4 (LEAD(Jan-5)-1), triggering
                 generate_series(Jan-1, Jan-4); intermediate dates (Jan 2-4)
                 carry NULL codes because the SQL only LEFT JOINs the gap rows
                 back to ``gap_dates`` on the anchor ``date``. The Jan-5 row
                 goes to the continuous branch and keeps its codes.
                 Expected: 5 daily rows (2 with codes, 3 with all nulls).
        200000 — Full-outer-join boundary: comp_hgics has 2020-06-15 (GICS-only)
                 and comp_other has 2020-06-16 (SIC-only). After the outer
                 join, each date sits one day before the other so the aux_date
                 check classifies both as continuous; nulls survive on the
                 missing source.
                 Expected: 2 daily rows.
        300000 — Single-date in both sources (2021-12-31). LEAD is null, so
                 COALESCE(..., date) makes aux_date = date → continuous branch.
                 Expected: 1 daily row.
    """
    comp_other = pl.DataFrame(
        {
            "gvkey": ["100000", "100000", "200000", "300000"],
            "date": [
                date(2020, 1, 1),
                date(2020, 1, 5),
                date(2020, 6, 16),
                date(2021, 12, 31),
            ],
            "sic": [7372, 7372, 3711, 4813],
            "naics": [511210, 511210, 336111, 517110],
        },
        schema={
            "gvkey": pl.Utf8,
            "date": pl.Date,
            "sic": pl.Int64,
            "naics": pl.Int64,
        },
    )

    comp_hgics = pl.DataFrame(
        {
            "gvkey": ["100000", "100000", "200000", "300000"],
            "date": [
                date(2020, 1, 1),
                date(2020, 1, 5),
                date(2020, 6, 15),
                date(2021, 12, 31),
            ],
            "gics": [10101010, 10101010, 20202020, 50505050],
        },
        schema={
            "gvkey": pl.Utf8,
            "date": pl.Date,
            "gics": pl.Int64,
        },
    )
    return comp_other, comp_hgics


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        paths.interim_dir.mkdir(parents=True, exist_ok=True)
        comp_other, comp_hgics = build_comp_industry_inputs()
        comp_other.write_parquet(paths.interim_dir / "comp_other.parquet")
        comp_hgics.write_parquet(paths.interim_dir / "comp_hgics.parquet")

        with comp_industry_upstream_stubs() as stubs:
            comp_industry(paths)
            stubs.assert_called()

        out_path = GOLDEN_DIR / "comp_ind.parquet"
        pl.read_parquet(paths.interim_dir / "comp_ind.parquet").write_parquet(out_path)
        n_rows = pl.read_parquet(out_path).height
        print(f"comp_ind.parquet: {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
