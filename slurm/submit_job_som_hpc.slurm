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

# WRDS credentials. Compute nodes don't have a system keyring daemon, so the
# pipeline needs an alternate source. The default below opts in to the
# file-backed keyring (a mode-600 file under ~/.local/share/python_keyring/);
# populate it once on the login node by running:
#
#     JKP_ALLOW_PLAINTEXT_KEYRING=1 jkp connect
#
# Alternative (for shared service accounts or container deployments): unset
# JKP_ALLOW_PLAINTEXT_KEYRING and export WRDS_USERNAME / WRDS_PASSWORD instead,
# e.g. by sourcing a mode-600 ~/.wrds_secrets file with `export WRDS_USERNAME=...`.
export JKP_ALLOW_PLAINTEXT_KEYRING=1

# Create char data
# Set PERSISTENT_WRDS_CONNECTION=1 to use a single WRDS connection (reduces MFA on NAT-rotated networks)
if [ "${PERSISTENT_WRDS_CONNECTION:-0}" = "1" ]; then
    jkp build data --persistent-connection
else
    jkp build data
fi

# Create portfolio data
jkp portfolio data
