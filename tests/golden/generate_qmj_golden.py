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

# --- Calibration -------------------------------------------------------------
# Derived from SUMMARY STATISTICS of the real interim world_data_-1.parquet
# (Yale HPC, June 2026): per-column medians, IQR-based scales, null rates,
# gate pass rates, and (excntry, eom) group sizes. Only aggregates were used —
# no WRDS rows are embedded in this file or the fixture. quality_minus_junk is
# rank-based, so the output depends on orderings, null patterns, and group
# sizes rather than exact distribution families; normal/lognormal samplers
# matched at the median/IQR are sufficient. The real data contains NO NaN
# values in the z-var columns (only nulls); the NaN injections below are a
# synthetic-only corner to lock NaN-handling behaviour.
#
# var: (median, scale ~= IQR/1.35, null_rate) — sampled as median + scale*N(0,1)
_NORMAL_CALIBRATION: dict[str, tuple[float, float, float]] = {
    "gp_at": (0.22, 0.21, 0.27),
    "ni_be": (0.065, 0.11, 0.18),
    "ni_at": (0.024, 0.054, 0.16),
    "ocf_at": (0.040, 0.082, 0.17),
    "gp_sale": (0.31, 0.23, 0.29),
    "oaccruals_at": (-0.026, 0.078, 0.29),
    "gpoa_ch5": (-0.008, 0.096, 0.47),
    "roe_ch5": (-0.012, 0.117, 0.41),
    "roa_ch5": (-0.003, 0.056, 0.38),
    "cfoa_ch5": (0.0, 0.086, 0.39),
    "gmar_ch5": (0.0, 0.077, 0.48),
    "o_score": (-2.73, 2.24, 0.33),
    "z_score": (2.80, 2.34, 0.31),
    # debt_at is sampled the same way, then clipped to [0, 1]
    "debt_at": (0.19, 0.23, 0.17),
}
# var: (mu, sigma, null_rate) — sampled as exp(mu + sigma*N(0,1)); positive vars
_LOGNORMAL_CALIBRATION: dict[str, tuple[float, float, float]] = {
    "betabab_1260d": (-0.036, 0.42, 0.41),
    "roeq_be_std": (-3.54, 1.28, 0.54),
    "roe_be_std": (-2.83, 1.24, 0.38),
}
_ME_LOGNORMAL = (4.49, 2.37)  # median ~89.5, q75 ~443
_RET_EXC_SCALE = 0.115  # IQR/1.35; null rate below
_RET_EXC_NULL = 0.12
_GATE_PASS = {"common": 0.84, "primary_sec": 0.88, "obs_main": 0.84, "exch_main": 0.82}

# Countries: USA/FRA are full-size groups; ITA is deliberately below
# min_stks=10 (real q5 group size is ~5) so the golden locks the
# thin-group → all-null-qmj behaviour. Id ranges disjoint so every
# (id, eom) pair is unique, required for integration-test determinism.
_COUNTRY_SPECS = [("USA", 1, 80), ("FRA", 201, 80), ("ITA", 401, 6)]


def _sample_values(rng: np.random.Generator, var: str, n: int) -> np.ndarray:
    """Draw n calibrated, non-null values for var."""
    if var in _LOGNORMAL_CALIBRATION:
        mu, sigma, _ = _LOGNORMAL_CALIBRATION[var]
        return np.exp(mu + sigma * rng.standard_normal(n))
    median, scale, _ = _NORMAL_CALIBRATION[var]
    vals = median + scale * rng.standard_normal(n)
    if var == "debt_at":
        vals = np.clip(vals, 0.0, 1.0)
    return vals


def _null_rate(var: str) -> float:
    cal = _LOGNORMAL_CALIBRATION.get(var) or _NORMAL_CALIBRATION[var]
    return cal[2]


def _corner_row_floats(rng: np.random.Generator) -> dict:
    """Calibrated, fully non-null float values for a hand-built corner row."""
    row = {var: float(_sample_values(rng, var, 1)[0]) for var in _FLOAT_VARS}
    row["ret_exc"] = float(_RET_EXC_SCALE * rng.standard_normal())
    row["me"] = float(np.exp(_ME_LOGNORMAL[0] + _ME_LOGNORMAL[1] * rng.standard_normal()))
    return row


def build_qmj_world_data_input(seed: int = 42) -> pl.DataFrame:
    """Build deterministic synthetic input for quality_minus_junk.

    Shape: (80 USA + 80 FRA + 6 ITA) stocks x 3 month-ends = 498 base rows
    plus a handful of hand-built corner-case rows. Values, null rates, and
    gate pass rates are calibrated to summary statistics of the real
    world_data_-1.parquet (see module comments); no WRDS data is embedded.

    USA ids are in [1, 200]; FRA in [201, 400]; ITA in [401, 500] — disjoint
    so every (id, eom) pair in the output is unique, which is required for
    integration-test determinism.
    """
    rng = np.random.default_rng(seed)

    rows: list[dict] = []

    for country, id_start, n in _COUNTRY_SPECS:
        for eom in _EOMS:
            flags = {g: rng.random(n) < p for g, p in _GATE_PASS.items()}
            ret_exc = _RET_EXC_SCALE * rng.standard_normal(n)
            ret_null = rng.random(n) < _RET_EXC_NULL
            me = np.exp(_ME_LOGNORMAL[0] + _ME_LOGNORMAL[1] * rng.standard_normal(n))
            values = {var: _sample_values(rng, var, n) for var in _FLOAT_VARS}
            nulls = {var: rng.random(n) < _null_rate(var) for var in _FLOAT_VARS}
            for i in range(n):
                row: dict = {
                    "id": id_start + i,
                    "eom": eom,
                    "excntry": country,
                    "common": int(flags["common"][i]),
                    "primary_sec": int(flags["primary_sec"][i]),
                    "obs_main": int(flags["obs_main"][i]),
                    "exch_main": int(flags["exch_main"][i]),
                    "ret_exc": None if ret_null[i] else float(ret_exc[i]),
                    "me": float(me[i]),
                }
                for var in _FLOAT_VARS:
                    row[var] = None if nulls[var][i] else float(values[var][i])
                rows.append(row)

    # --- Corner case: a few NaN values in single z-vars ---
    # Use id=150 (USA) / id=350 (FRA) — ids outside the base ranges
    # ([1, 80] and [201, 280]) so these "NaN injection" rows don't
    # collide with the base rows above.
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
                **_corner_row_floats(rng),
            }
            # Inject NaN into two vars to test partial-null z-rank behaviour
            # (the real data has no NaNs; synthetic-only corner)
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
                **_corner_row_floats(rng),
            }
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
        for country, id_base in [("USA", gate_fail_id_start), ("FRA", gate_fail_id_start + 200)]:
            row = {
                "id": id_base + offset,
                "eom": _EOMS[0],
                "excntry": country,
                "common": 1,
                "primary_sec": 1,
                "obs_main": 1,
                "exch_main": 1,
                **_corner_row_floats(rng),
            }
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
                **_corner_row_floats(rng),
            }
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
            # flags are Int32 in the real world_data_-1 schema
            "common": pl.Int32,
            "primary_sec": pl.Int32,
            "obs_main": pl.Int32,
            "exch_main": pl.Int32,
            "ret_exc": pl.Float64,
            "me": pl.Float64,
            **dict.fromkeys(_FLOAT_VARS, pl.Float64),
        },
    )
    return df


def _make_paths(base_dir: str) -> DataPaths:
    """Create a DataPaths rooted at base_dir with the interim dir (all we write to)."""
    paths = DataPaths(base_dir=Path(base_dir))
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    return paths


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
