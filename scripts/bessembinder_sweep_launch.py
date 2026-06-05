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

# One-at-a-time grid around the paper defaults (see Section8Params).
# Round 1: near-default (~2x steps). Round 2: log-spaced extremes (~5-10x)
# to bracket where the response surface stops being flat.
GRID: dict[str, dict[str, float]] = {
    # ---- round 1 ----
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
    # ---- round 2: extremes ----
    "g_ret_02": {"g_ret": 0.2},
    "g_ret_04": {"g_ret": 0.4},
    "g_ret_16": {"g_ret": 1.6},
    "g_ret_30": {"g_ret": 3.0},
    "g_me_01": {"g_me_change": 0.1},
    "g_me_09": {"g_me_change": 0.9},
    "e_up_2_1": {"e_up_jump": 2.0, "e_up_confirm": 1.0},
    "e_up_20_10": {"e_up_jump": 20.0, "e_up_confirm": 10.0},
    "e_up_100_50": {"e_up_jump": 100.0, "e_up_confirm": 50.0},
    "e_down_005_01": {"e_down_jump": 0.05, "e_down_confirm": 0.1},
    "e_down_05_07": {"e_down_jump": 0.5, "e_down_confirm": 0.7},
    "f_up_3_05": {"f_up_ratio": 3.0, "f_up_ret": 0.5},
    "f_up_50_10": {"f_up_ratio": 50.0, "f_up_ret": 10.0},
    "f_down_002_m08": {"f_down_ratio": 0.02, "f_down_ret": -0.8},
    "f_down_03_m02": {"f_down_ratio": 0.3, "f_down_ret": -0.2},
    "early_63_002": {"early_obs": 63.0, "early_frac": 0.02},
    "early_126_005": {"early_obs": 126.0, "early_frac": 0.05},
    "early_2520_06": {"early_obs": 2520.0, "early_frac": 0.6},
    "h_band_3_033": {"h_ratio_hi": 3.0, "h_ratio_lo": 0.33},
    "h_band_50_002": {"h_ratio_hi": 50.0, "h_ratio_lo": 0.02},
    "h_obs_0": {"h_max_obs": 0.0},
    "h_obs_20": {"h_max_obs": 20.0},
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
        # is_symlink, not exists: exists() follows the link and returns False
        # for a link whose target is missing, then symlink_to crashes
        if not (name.is_symlink() or name.exists()):
            name.symlink_to(target, target_is_directory=True)
    for fname in SHARED_INTERIM_FILES:
        link = interim / fname
        if not (link.is_symlink() or link.exists()):
            link.symlink_to(SHARED / "interim" / fname)
    return base


def main() -> None:
    dry = "--dry-run" in sys.argv
    SWEEP_ROOT.mkdir(exist_ok=True)
    for tag, s8 in GRID.items():
        base = build_sandbox(tag)
        # already-completed variants (earlier rounds) are skipped
        done = base / "interim" / f"bessembinder_{tag}" / "crsp_comparison_after.parquet"
        if done.exists():
            print(f"{tag}: done, skipping", flush=True)
            continue
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
