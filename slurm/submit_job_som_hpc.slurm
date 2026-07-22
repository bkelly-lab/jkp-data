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

# Abort the job if any command fails (e.g. `jkp build`) instead of
# proceeding to `jkp portfolio` against incomplete data.
set -e

echo '-------------------------------'
cd "${SLURM_SUBMIT_DIR}"
echo "${SLURM_SUBMIT_DIR}"
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "SLURM_NODES are ${SLURM_NODELIST}"
echo '-------------------------------'
echo

# Output directory for the pipeline, taken from the first script argument.
# Relative paths resolve against the submit directory. Defaults to "data":
#
#     sbatch slurm/submit_job_som_hpc.slurm /path/to/output
#
OUTPUT_DIR="${1:-data}"

# (Optional) Make threaded libs respect Slurm CPU allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Activate the project's virtual environment so the jkp CLI is on PATH
source .venv/bin/activate

# WRDS credentials. Compute nodes have no system keyring daemon, so provision
# credentials ONCE on a login node (which has an interactive terminal) BEFORE
# submitting this job -- $HOME is shared with the compute nodes, so the batch
# job sees the file:
#
#     jkp connect          # prompts, writes ~/.pgpass (mode 600)
#
# The pipeline reads the password from ~/.pgpass via libpq; jkp never stores a
# separate copy. Point PGPASSFILE at a non-default location if you prefer.
#
# Fully non-interactive alternative (no login-node terminal): export
# WRDS_USERNAME / WRDS_PASSWORD.

# Create char data
# Set PERSISTENT_WRDS_CONNECTION=1 to use a single WRDS connection (reduces MFA on NAT-rotated networks)
# Use --force because a batch job has no TTY to answer the interactive overwrite prompt, and
# resubmitting the job implies intent to regenerate the data.
if [ "${PERSISTENT_WRDS_CONNECTION:-0}" = "1" ]; then
    jkp build "${OUTPUT_DIR}" --force --persistent-connection
else
    jkp build "${OUTPUT_DIR}" --force
fi

# Create portfolio data
jkp portfolio "${OUTPUT_DIR}"
