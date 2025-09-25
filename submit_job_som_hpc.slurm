#!/bin/bash
#SBATCH -J jkpfactors
#SBATCH -o output_jkp.txt
#SBATCH -e error_jkp.txt
#SBATCH --partition=gpunormal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
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

# Activate the conda environment
conda init
conda activate jkp_factors

# Run the Python script
python main.py
