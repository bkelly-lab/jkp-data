#!/bin/bash
#SBATCH -J dhs_factors
#SBATCH -o ff_data_comp/output_dhs_%j.txt
#SBATCH -e ff_data_comp/error_dhs_%j.txt
#SBATCH --partition=gpunormal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL

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

# Script reads pipeline parquets from cwd; run from data/interim via uv
cd data/interim
uv run --project ../.. python ../../ff_data_comp/build_dhs_factors.py
