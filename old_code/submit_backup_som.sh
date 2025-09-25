#!/bin/bash

#SBATCH -J jkpfactors                           # Job name

#SBATCH -o output_%j.txt                # Output file (%j will be replaced by the job ID)

#SBATCH -e error_%j.txt                 # Error file (%j will be replaced by the job ID)

#SBATCH --ntasks=1                      # Number of tasks (processes)

#SBATCH --cpus-per-task=40              # Number of CPU cores per task

#SBATCH --mem=450G                      # Memory per node

#SBATCH --partition=normal

#SBATCH --time=15:00:00                # HH:MM:SS

#SBATCH --mail-type=ALL                 # Send email on start, end and fail



echo '-------------------------------'

cd ${SLURM_SUBMIT_DIR}                  # Change directory to slurm submit directory

echo ${SLURM_SUBMIT_DIR}

echo Running on host $(hostname)

echo Time is $(date)

echo SLURM_NODES are $(echo ${SLURM_NODELIST})

echo '-------------------------------'

echo -e '\n\n'


# Load the necessary module
module load miniconda

# Activate the conda environment
conda activate jkp_factors

# Navigate to the Build_database directory
cd Build_database

# Run the Python script
python main.py
