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

# Option parsing requires the enhanced (util-linux) getopt(1). It is present in
# the base install of every mainstream Linux distribution, including this
# cluster's login and compute nodes, and is deliberately NOT a dependency of the
# `jkp` package: these batch scripts are worked examples of running the pipeline
# under Slurm, not part of the installed package. The --test guard below fails
# loudly rather than misparsing if only the legacy getopt is available.

usage() {
    cat <<'USAGE'
usage: sbatch slurm/submit_job_som_hpc.slurm [--output-dir DIR]

  --output-dir DIR  Directory for pipeline output (raw, interim and processed
                    data). Relative paths resolve against the submit directory.
                    Defaults to "data".
  -h, --help        Show this message and exit.
USAGE
}

# `getopt --test` exits 4 on the enhanced version, so it cannot be run bare
# under `set -e`.
GETOPT_TEST_RC=0
getopt --test >/dev/null || GETOPT_TEST_RC=$?
if [ "${GETOPT_TEST_RC}" -ne 4 ]; then
    echo "error: this script requires the enhanced (util-linux) getopt" >&2
    exit 1
fi

PARSED=$(getopt -o 'h' --long 'output-dir:,help' -n 'submit_job_som_hpc.slurm' -- "$@") || {
    usage >&2
    exit 1
}
eval set -- "${PARSED}"

OUTPUT_DIR=data
while true; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) echo "error: unhandled option: $1" >&2; exit 1 ;;
    esac
done

if [ "$#" -ne 0 ]; then
    echo "error: unexpected argument: $1" >&2
    usage >&2
    exit 1
fi

# Resolved after the cd so a relative --output-dir reports the same path the
# pipeline will actually write to.
echo '-------------------------------'
cd "${SLURM_SUBMIT_DIR}"
echo "${SLURM_SUBMIT_DIR}"
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "SLURM_NODES are ${SLURM_NODELIST}"
echo "Output directory is $(realpath -m "${OUTPUT_DIR}")"
echo '-------------------------------'
echo

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
