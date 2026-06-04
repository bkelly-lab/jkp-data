#!/bin/bash
#SBATCH -J bess_valid
#SBATCH -o slurm/output_bess_%j.txt
#SBATCH -e slurm/error_bess_%j.txt
#SBATCH --partition=gpunormal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH --time=15:00:00
#SBATCH --mail-type=ALL

set -e

echo '-------------------------------'
cd "${SLURM_SUBMIT_DIR}"
echo "${SLURM_SUBMIT_DIR}"
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "SLURM_NODES are ${SLURM_NODELIST}"
echo '-------------------------------'
echo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Reuse the clean repo's venv (same deps); run this branch's code via PYTHONPATH
source "${HOME}/jkp-data/.venv/bin/activate"

PYTHONPATH=src python scripts/bessembinder_us_validation.py
