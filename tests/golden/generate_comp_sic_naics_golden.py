"""Generate golden fixture for comp_sic_naics().

Run with:
    uv run python -m tests.golden.generate_comp_sic_naics_golden

Writes:
    tests/golden/fixtures/comp_sic_naics/comp_other.parquet
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import comp_sic_naics
from jkp.data.paths import DataPaths

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "comp_sic_naics"


def build_sic_naics_inputs(seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build deterministic NA and GL SIC/NAICS fixtures.

    Scenarios covered by the returned (sic_naics_na, sic_naics_gl) pair:
        001000 — NA-only, two datadates (2020-01-01 → 2020-01-04) producing 3 daily
                 fill rows plus a trailing single-date row.
        002000 — GL-only, one datadate (2020-06-15).
        003000 — Both sources on same datadate → single joined row; COALESCE
                 prefers the NA value (6020).
        004000 — NA row with sic=NULL on one datadate; GL row with sic non-null
                 on the same datadate → coalesce keeps non-null SIC.
        175650 — Hard-coded dropped row (datadate=2005-12-31, naics IS NULL); a
                 separate datadate (2006-06-30) for the same gvkey is retained.
        500    — gvkey not zero-padded in input; output must be LPAD to '000500'.
    """
    del seed
    sic_naics_na = pl.DataFrame(
        {
            "gvkey": [
                "001000",
                "001000",
                "003000",
                "004000",
                "175650",
                "175650",
                "500",
            ],
            "datadate": [
                date(2020, 1, 1),
                date(2020, 1, 4),
                date(2018, 5, 1),
                date(2019, 3, 1),
                date(2005, 12, 31),
                date(2006, 6, 30),
                date(2021, 7, 15),
            ],
            "sic": [3711, 3713, 6020, None, 1311, 1311, 7372],
            "naics": [
                336111,
                336112,
                522110,
                541110,
                None,  # Triggers the hard-coded drop
                211120,
                511210,
            ],
        },
        schema={
            "gvkey": pl.Utf8,
            "datadate": pl.Date,
            "sic": pl.Int64,
            "naics": pl.Int64,
        },
    )

    sic_naics_gl = pl.DataFrame(
        {
            "gvkey": [
                "002000",
                "003000",
                "004000",
            ],
            "datadate": [
                date(2020, 6, 15),
                date(2018, 5, 1),
                date(2019, 3, 1),
            ],
            "sic": [2834, 6021, 4813],
            "naics": [325412, 522120, 517110],
        },
        schema={
            "gvkey": pl.Utf8,
            "datadate": pl.Date,
            "sic": pl.Int64,
            "naics": pl.Int64,
        },
    )
    return sic_naics_na, sic_naics_gl


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        paths = DataPaths(base_dir=Path(td))
        (paths.interim_dir / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
        sic_naics_na, sic_naics_gl = build_sic_naics_inputs()
        sic_naics_na.write_parquet(paths.interim_dir / "raw_data_dfs" / "sic_naics_na.parquet")
        sic_naics_gl.write_parquet(paths.interim_dir / "raw_data_dfs" / "sic_naics_gl.parquet")
        comp_sic_naics(paths)
        out_path = GOLDEN_DIR / "comp_other.parquet"
        pl.read_parquet(paths.interim_dir / "comp_other.parquet").write_parquet(out_path)
        n_rows = pl.read_parquet(out_path).height
        print(f"comp_other.parquet: {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
