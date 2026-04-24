#!/bin/bash
#SBATCH -J jkpfactors
#SBATCH -o slurm/output_jkp_%j.txt
#SBATCH -e slurm/error_jkp_%j.txt
#SBATCH --partition=gpunormal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=450G
#SBATCH --time=15:00:00
#SBATCH --mail-type=ALL

echo '-------------------------------'
cd "${SLURM_SUBMIT_DIR}"
echo "${SLURM_SUBMIT_DIR}"
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "SLURM_NODES are ${SLURM_NODELIST}"
echo '-------------------------------'
echo

# (Optional) Make threaded libs respect Slurm CPU allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Activate the project's virtual environment so the jkp CLI is on PATH
source .venv/bin/activate

# Create char data
# Set PERSISTENT_WRDS_CONNECTION=1 to use a single WRDS connection (reduces MFA on NAT-rotated networks)
if [ "${PERSISTENT_WRDS_CONNECTION:-0}" = "1" ]; then
    uv run jkp build data --persistent-connection
else
    uv run jkp build data
fi

# Create portfolio data
uv run jkp portfolio data
