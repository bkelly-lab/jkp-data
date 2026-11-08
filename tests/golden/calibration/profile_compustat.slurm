#!/bin/bash
#SBATCH -J acc_profile
#SBATCH -o slurm/acc_profile_%j.out
#SBATCH -e slurm/acc_profile_%j.err
#SBATCH --partition=build,cpunormal,default_queue,gpunormal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Read-only profile of the accounting stage's real inputs and outputs (issue #303, step 1).
# Submit from the root of a fresh clone of the branch, after `uv sync --group test`:
#     sbatch tests/golden/calibration/profile_compustat.slurm
# Override the inputs with DATA_DIR, CHARS and VINTAGE in the environment.
# Writes tests/golden/compustat_profile.json in the clone; `time -v` in the .err file gives the
# measured wall time and peak memory for the next run's limits.

set -euo pipefail

DATA_DIR=${DATA_DIR:-$HOME/jkp_factors_rep_10142026/data}
CHARS=${CHARS:-$HOME/acc_ab/det/data_main1107_t32/interim/acc_chars_world.parquet}
VINTAGE=${VINTAGE:-2026-10-15}

export PYTHONPATH=src:.
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

/usr/bin/time -v .venv/bin/python -m tests.golden.calibration.profile_compustat \
    "$DATA_DIR" "$CHARS" tests/golden/compustat_profile.json --vintage "$VINTAGE"
