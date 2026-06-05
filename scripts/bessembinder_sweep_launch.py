"""Launch the Section 8 threshold sweep as parallel cluster jobs.

Description:
    One sbatch job per grid variant, each in an isolated sandbox (shared
    read-only inputs symlinked in; scratch files per-variant, since parallel
    jobs in one sandbox collide on spill files). All jobs are submitted at
    once; slurm schedules them across whatever nodes are free.
Steps:
    1) Per variant: build sandbox_<tag>/ under SWEEP_ROOT with symlinks to
       the shared raw tables and reusable interim inputs.
    2) Write the overrides JSON ({"tag", "s8": {...}}).
    3) sbatch submit_bessembinder_validation.slurm bessembinder <json>
       with BESS_VALIDATION_BASE pointing at the sandbox.
Output:
    Per-variant outputs under sandbox_<tag>/interim/ (comparison files,
    corrected parquet, corrections log). Aggregate with
    bessembinder_sweep_collect.py once all jobs finish.

Usage (on the cluster, from the branch checkout):
    PYTHONPATH=src python scripts/bessembinder_sweep_launch.py [--dry-run]
"""

import json
import subprocess
import sys
from pathlib import Path

SHARED = Path.home() / "bessembinder_validation" / "data"
SWEEP_ROOT = Path.home() / "bessembinder_sweep"
REPO = Path.home() / "jkp-data-bess"

# Shared inputs reused read-only by every variant (built by previous runs)
SHARED_INTERIM_FILES = [
    "__firm_shares2.parquet",
    "__comp_dsf_uncorrected.parquet",
    "crsp_comparison_before_cache.parquet",
]

# One-at-a-time grid around the paper defaults (see Section8Params)
GRID: dict[str, dict[str, float]] = {
    "g_ret_05": {"g_ret": 0.5},
    "g_ret_065": {"g_ret": 0.65},
    "g_ret_10": {"g_ret": 1.0},
    "g_ret_12": {"g_ret": 1.2},
    "g_me_03": {"g_me_change": 0.3},
    "g_me_07": {"g_me_change": 0.7},
    "e_up_3_15": {"e_up_jump": 3.0, "e_up_confirm": 1.5},
    "e_up_8_4": {"e_up_jump": 8.0, "e_up_confirm": 4.0},
    "e_down_03_05": {"e_down_jump": 0.3, "e_down_confirm": 0.5},
    "f_up_5_1": {"f_up_ratio": 5.0, "f_up_ret": 1.0},
    "f_up_20_4": {"f_up_ratio": 20.0, "f_up_ret": 4.0},
    "f_down_02_m03": {"f_down_ratio": 0.2, "f_down_ret": -0.3},
    "early_252_01": {"early_obs": 252.0, "early_frac": 0.1},
    "early_1008_04": {"early_obs": 1008.0, "early_frac": 0.4},
    "h_obs_5": {"h_max_obs": 5.0},
    "h_obs_10": {"h_max_obs": 10.0},
    "h_band_5_02": {"h_ratio_hi": 5.0, "h_ratio_lo": 0.2},
    # baseline through the identical sweep machinery, for apples-to-apples
    "default": {},
}


def build_sandbox(tag: str) -> Path:
    base = SWEEP_ROOT / f"sandbox_{tag}" / "data"
    interim = base / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    # Raw tables and prepared raw frames: read-only, share via dir symlinks
    raw = base / "raw"
    raw.mkdir(exist_ok=True)
    for name, target in [
        (raw / "raw_tables", SHARED / "raw" / "raw_tables"),
        (interim / "raw_data_dfs", SHARED / "interim" / "raw_data_dfs"),
    ]:
        if not name.exists():
            name.symlink_to(target, target_is_directory=True)
    for fname in SHARED_INTERIM_FILES:
        link = interim / fname
        if not link.exists():
            link.symlink_to(SHARED / "interim" / fname)
    return base


def main() -> None:
    dry = "--dry-run" in sys.argv
    SWEEP_ROOT.mkdir(exist_ok=True)
    for tag, s8 in GRID.items():
        base = build_sandbox(tag)
        spec = base.parent / "overrides.json"
        spec.write_text(json.dumps({"tag": tag, "s8": s8}))
        cmd = [
            "sbatch",
            f"--export=ALL,BESS_VALIDATION_BASE={base}",
            f"--job-name=sweep_{tag}",
            str(REPO / "slurm" / "submit_bessembinder_validation.slurm"),
            "bessembinder",
            str(spec),
        ]
        if dry:
            print(" ".join(cmd))
        else:
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=True)
            print(f"{tag}: {out.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
