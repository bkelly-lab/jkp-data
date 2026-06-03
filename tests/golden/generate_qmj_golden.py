"""Generate golden fixture for quality_minus_junk.

Run with:
    PYTHONPATH=src uv run --no-sync python -m tests.golden.generate_qmj_golden
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from jkp.data.aux_functions import quality_minus_junk
from jkp.data.paths import DataPaths

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "qmj"

# Float z-var columns (16, but __evol is derived; we supply the raw inputs)
_FLOAT_VARS = [
    "gp_at",
    "ni_be",
    "ni_at",
    "ocf_at",
    "gp_sale",
    "oaccruals_at",
    "gpoa_ch5",
    "roe_ch5",
    "roa_ch5",
    "cfoa_ch5",
    "gmar_ch5",
    "betabab_1260d",
    "debt_at",
    "o_score",
    "z_score",
    "roeq_be_std",
    "roe_be_std",
]

_EOMS = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
_COUNTRIES = ["USA", "FRA"]
# USA ids: 1..200, FRA ids: 201..400 — disjoint to ensure unique (id, eom)
_USA_ID_START = 1
_FRA_ID_START = 201
_N_PER_COUNTRY_MONTH = 40  # well above min_stks=10


def build_qmj_world_data_input(seed: int = 42) -> pl.DataFrame:
    """Build deterministic synthetic input for quality_minus_junk.

    Shape: 2 countries x 3 month-ends x ~40 stocks = 240 valid rows plus
    a handful of gate-failing and corner-case rows.

    USA ids are in [1, 200]; FRA ids in [201, 400] — disjoint so every
    (id, eom) pair in the output is unique, which is required for
    integration-test determinism.
    """
    rng = np.random.default_rng(seed)

    rows: list[dict] = []

    for country in _COUNTRIES:
        id_start = _USA_ID_START if country == "USA" else _FRA_ID_START
        for eom in _EOMS:
            for i in range(_N_PER_COUNTRY_MONTH):
                row: dict = {
                    "id": id_start + i,
                    "eom": eom,
                    "excntry": country,
                    "common": 1,
                    "primary_sec": 1,
                    "obs_main": 1,
                    "exch_main": 1,
                    "ret_exc": float(rng.standard_normal()),
                    "me": float(rng.uniform(10.0, 1e6)),
                }
                for var in _FLOAT_VARS:
                    row[var] = float(rng.standard_normal())
                rows.append(row)

    # --- Corner case: a few NaN values in single z-vars ---
    # Use id=100 (USA) / id=300 (FRA); they already appear above — we
    # overwrite individual float cols to NaN via separate "NaN injection"
    # rows using ids outside the base range.
    nan_id_usa = 150
    nan_id_fra = 350
    for eom in _EOMS[:1]:
        for nan_id, country in [(nan_id_usa, "USA"), (nan_id_fra, "FRA")]:
            row = {
                "id": nan_id,
                "eom": eom,
                "excntry": country,
                "common": 1,
                "primary_sec": 1,
                "obs_main": 1,
                "exch_main": 1,
                "ret_exc": float(rng.standard_normal()),
                "me": float(rng.uniform(10.0, 1e6)),
            }
            for var in _FLOAT_VARS:
                row[var] = float(rng.standard_normal())
            # Inject NaN into two vars to test partial-null z-rank behaviour
            row["gp_at"] = float("nan")
            row["roe_ch5"] = float("nan")
            rows.append(row)

    # --- Corner case: roeq_be_std null → evol falls back to roe_be_std ---
    evol_fallback_id_usa = 151
    evol_fallback_id_fra = 351
    for eom in _EOMS[:1]:
        for fb_id, country in [(evol_fallback_id_usa, "USA"), (evol_fallback_id_fra, "FRA")]:
            row = {
                "id": fb_id,
                "eom": eom,
                "excntry": country,
                "common": 1,
                "primary_sec": 1,
                "obs_main": 1,
                "exch_main": 1,
                "ret_exc": float(rng.standard_normal()),
                "me": float(rng.uniform(10.0, 1e6)),
            }
            for var in _FLOAT_VARS:
                row[var] = float(rng.standard_normal())
            row["roeq_be_std"] = None  # force evol = roe_be_std
            rows.append(row)

    # --- Corner case: gate-failing rows (filtered by c1) ---
    gate_fail_configs = [
        {"common": 0},
        {"primary_sec": 0},
        {"ret_exc": None},
        {"me": None},
    ]
    gate_fail_id_start = 160
    for offset, overrides in enumerate(gate_fail_configs):
        for country, id_base in [("USA", gate_fail_id_start), ("FRA", gate_fail_id_start + 100)]:
            row = {
                "id": id_base + offset,
                "eom": _EOMS[0],
                "excntry": country,
                "common": 1,
                "primary_sec": 1,
                "obs_main": 1,
                "exch_main": 1,
                "ret_exc": float(rng.standard_normal()),
                "me": float(rng.uniform(10.0, 1e6)),
            }
            for var in _FLOAT_VARS:
                row[var] = float(rng.standard_normal())
            row.update(overrides)
            rows.append(row)

    # --- Corner case: stocks with ALL safety z-vars null → null safety → null qmj ---
    null_safety_id_usa = 170
    null_safety_id_fra = 370
    for ns_id, country in [(null_safety_id_usa, "USA"), (null_safety_id_fra, "FRA")]:
        for eom in _EOMS[:1]:
            row = {
                "id": ns_id,
                "eom": eom,
                "excntry": country,
                "common": 1,
                "primary_sec": 1,
                "obs_main": 1,
                "exch_main": 1,
                "ret_exc": float(rng.standard_normal()),
                "me": float(rng.uniform(10.0, 1e6)),
            }
            for var in _FLOAT_VARS:
                row[var] = float(rng.standard_normal())
            # All safety inputs null → mean_horizontal → null safety → null qmj
            for null_var in [
                "betabab_1260d",
                "debt_at",
                "o_score",
                "z_score",
                "roeq_be_std",
                "roe_be_std",
            ]:
                row[null_var] = None
            rows.append(row)

    df = pl.DataFrame(
        rows,
        schema={
            "id": pl.Int64,
            "eom": pl.Date,
            "excntry": pl.Utf8,
            "common": pl.Int64,
            "primary_sec": pl.Int64,
            "obs_main": pl.Int64,
            "exch_main": pl.Int64,
            "ret_exc": pl.Float64,
            "me": pl.Float64,
            **dict.fromkeys(_FLOAT_VARS, pl.Float64),
        },
    )
    return df


def _make_paths(base_dir: str) -> DataPaths:
    """Create a DataPaths rooted at base_dir with required subdirs."""
    base = Path(base_dir)
    (base / "interim" / "raw_data_dfs").mkdir(parents=True, exist_ok=True)
    (base / "raw" / "raw_tables").mkdir(parents=True, exist_ok=True)
    (base / "processed" / "characteristics").mkdir(parents=True, exist_ok=True)
    (base / "processed" / "return_data").mkdir(parents=True, exist_ok=True)
    (base / "processed" / "other_output").mkdir(parents=True, exist_ok=True)
    return DataPaths(base_dir=base)


def main() -> None:
    """Generate and write the qmj golden fixture."""
    df = build_qmj_world_data_input(seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = _make_paths(tmpdir)
        input_path = paths.interim_dir / "world_data_-1.parquet"
        df.write_parquet(input_path)

        quality_minus_junk(paths, input_path, 10)

        result = pl.read_parquet(paths.interim_dir / "qmj.parquet").sort(["excntry", "id", "eom"])

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLDEN_DIR / "qmj.parquet"
    result.write_parquet(out_path)
    print(f"qmj.parquet: {len(result)} rows -> {out_path}")


if __name__ == "__main__":
    main()
