# Global Factor, Stock, and Firm data

This repo contains Python code to generate the global dataset of factor returns, stock returns, and firm characteristics from ["Is there a Replication Crisis in Finance?"](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249) by Jensen, Kelly, and Pedersen (Journal of Finance, 2023).

## Prerequisites

- Active [WRDS](https://wrds-www.wharton.upenn.edu/) account with access to CRSP and Compustat.
- Python 3.11 or later.

## Installation

Install with your preferred Python package manager:

**pip**
```bash
pip install jkp-data
```

**conda / mamba**
```bash
pip install jkp-data
```
(Run inside your active conda environment.)

**uv**
```bash
uv add jkp-data
```

> **uv tip:** Activate the virtual environment (`source <venv-path>/bin/activate`) and run `jkp` commands directly — no need to prefix every command with `uv run`. Tab completion also works when the environment is activated.

After installation, the `jkp` command is available from any directory.

## Setup: WRDS credentials

Run the following command once to save your WRDS credentials:

```bash
jkp connect
```

Follow the prompts. To update your credentials later, run `jkp connect --reset`.

## Running the pipeline

The full pipeline requires approximately 450 GB of RAM and 128 CPU cores, which for most users means running on a high-performance computing cluster. Interactive use on a workstation is possible for testing, but the full global dataset will exceed the memory available on most desktop machines.

**Build stock returns and firm characteristics:**

```bash
jkp build <output-dir>
```

**Build factor returns** (run after `jkp build`):

```bash
jkp portfolio <output-dir>
```

`<output-dir>` is any directory of your choice (e.g. `~/data/jkp`). Final output is written to `<output-dir>/processed/`.

**Note:** When the pipeline first connects to WRDS, you may receive a two-factor authentication prompt (e.g. a Duo push notification). Approve it promptly — the pipeline will stall until you do.

### Running on a Slurm cluster

Most HPC clusters use Slurm. Below is a template job script — adapt the partition, account, and environment activation to match your cluster:

```bash
#!/bin/bash
#SBATCH --job-name=jkpfactors
#SBATCH --partition=<your-partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=450G
#SBATCH --time=15:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL

# Make threaded libraries respect the Slurm CPU allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Activate your environment so that jkp is on PATH
source /path/to/your/venv/bin/activate

# Set PERSISTENT_WRDS_CONNECTION=1 to use a single WRDS connection (reduces MFA on NAT-rotated networks)
if [ "${PERSISTENT_WRDS_CONNECTION:-0}" = "1" ]; then
    jkp build ~/data/jkp --persistent-connection
else
    jkp build ~/data/jkp
fi

jkp portfolio ~/data/jkp
```

Submit with:

```bash
sbatch jkp_job.slurm
```

## Notes

- **Output format:** Files default to Parquet. To write CSV instead (with quoted strings to preserve leading zeros in identifiers like `gvkey`), pass `--output-format csv`:

  ```bash
  jkp portfolio <output-dir> --output-format csv
  ```

- **End date:** The default end date is 2025-12-31. Support for overriding this without editing source is planned for a future release.

- **Persistent WRDS connection:** On HPC clusters with NAT IP rotation, each database query may trigger a separate MFA prompt because the outbound IP changes between connections. Pass `--persistent-connection` to maintain a single connection throughout the download, reducing approximately 26 prompts to one. The Slurm template above supports this via the `PERSISTENT_WRDS_CONNECTION` environment variable.

- **Documentation:** See our [documentation](https://jkpfactors.s3.amazonaws.com/documents/Documentation.pdf) for a full description of output files and a comparison with the original SAS/R codebase.

- **Pre-built data:** Factor returns are available at [jkpfactors.com](https://jkpfactors.com). Stock returns and firm characteristics are available through [WRDS](https://wrds-www.wharton.upenn.edu/pages/get-data/contributed-data-forms/global-factor-data/).

- **Original codebase:** The original SAS/R codebase is available at [github.com/bkelly-lab/ReplicationCrisis](https://github.com/bkelly-lab/ReplicationCrisis), but we recommend this Python codebase for future work.

## License

Code in this repository is released under the [MIT License](LICENSE).

Data distributed in this repository is licensed under [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](DATA_LICENSE).

See [LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE) for details.
