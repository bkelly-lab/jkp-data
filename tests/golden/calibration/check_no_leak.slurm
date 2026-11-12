#!/bin/bash
#SBATCH -J acc_leak
#SBATCH -o slurm/acc_leak_%j.out
#SBATCH -e slurm/acc_leak_%j.err
#SBATCH --partition=build,cpunormal,default_queue,gpunormal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Read-only check that the synthetic Compustat inputs copy no real gvkey or row (issue #303, step 6).
# Submit from the root of a fresh clone of the branch, after `uv sync --group test`:
#     DATA_DIR=<build>/data sbatch tests/golden/calibration/check_no_leak.slurm
# Set DATA_DIR (a build's data/ folder) in the environment; there is no default.
# Prints counts only; exits 1 on a match.
# The limits come from the profile job 4187475 (3:22 wall, 18 GB peak), which read every column
# of the same tables; this job reads four columns of each.

set -euo pipefail

: "${DATA_DIR:?set DATA_DIR}"

export PYTHONPATH=src:.
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

/usr/bin/time -v .venv/bin/python -m tests.golden.calibration.check_no_leak "$DATA_DIR"
