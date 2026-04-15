# Global Factor, Stock, and Firm data

This repo contains Python code to generate the global dataset of factor returns, stock returns, and firm characteristics from [“Is there a Replication Crisis in Finance?”](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249) by Jensen, Kelly, and Pedersen (Journal of Finance, 2023).

## Instructions

### Prerequisites

- Obtain your WRDS credentials.
- Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) installed on your system.

### Steps

1. **Clone the repo**

   - Clone the folder to your local machine by running the following command from your terminal:
     ```sh
     git clone https://github.com/bkelly-lab/jkp-data.git
     ```
2. **Input WRDS credentials**

   - To save your WRDS credentials, navigate to the `jkp-data/` folder and run:
     ```sh
     jkp connect
     ```
     Kindly follow the prompts.

     Note: If you need to change your password or credentials, run `jkp connect --reset` and then `jkp connect`

3. **Run the script**

   - We run the code via a Slurm scheduler, but we also show how to run it in an interactive Python session.

   - Before running the following commands, make sure you are in `jkp-data/`

   - On a cluster with a Slurm scheduler, run:
     ```sh
     sbatch slurm/submit_job_som_hpc.slurm
     ```
     to create the factor returns, stock returns, and firm characteristics.

     In an interactive session, run:
     ```sh
     jkp build
     ```
     to create the stock returns and firm characteristics, and
     ```sh
     jkp portfolio
     ```
     to create the factor returns.

   **IMPORTANT:** When starting the code, you may be prompted to grant access to WRDS using two-factor authentication, for example via a Duo notification. You need to approve this request, as the program will otherwise fail. After a few seconds or minutes, you should see data being created in `data/raw`. If that is not the case, please check your internet connection or credentials.

When the code is finished, you can find the output in:
```
data/processed/
```
Please see the release notes (`documentation/release_notes.html`) for a description of the output files and a comparison between the output of the SAS/R codebase and the new Python codebase.

## Notes
- By default, output files are written in Parquet format. To output CSV files instead (with quoted strings to preserve leading zeros in identifiers like `gvkey`), run:
  ```sh
  jkp portfolio --output-format csv
  ```

- By default, the end date for the data in the code is 2025-12-31, which you can change by editing the `end_date` assignment in `src/jkp/data/config.py`. For example, for May 6, 1992, use: `END_DATE = date(1992, 5, 6)`.

- **Persistent WRDS Connection**: If you're running on an HPC cluster with NAT IP rotation (such as Yale's Bouchet cluster), you may receive many MFA prompts during data download. This happens because each database query creates a new TCP connection, and the NAT gateway assigns a random outbound IP to each connection. WRDS sees these as connections from different locations and triggers MFA for each.

  To avoid this, use the `--persistent-connection` flag, which maintains a single database connection throughout the download process:
  ```sh
  # Interactive session
  jkp build --persistent-connection

  # Slurm job (set environment variable)
  sbatch --export=ALL,PERSISTENT_WRDS_CONNECTION=1 slurm/submit_job_som_hpc.slurm
  ```
  This reduces MFA prompts from ~26 (one per table) to just 1 (at connection time).

- To run the code, we utilize a high performance computing cluster, where we request 450 GB RAM and 128 CPU cores. Running the routine takes about 6 hours.

- To understand the data, please refer to our [documentation](https://jkpfactors.s3.amazonaws.com/documents/Documentation.pdf).

- We distribute the global factor returns generated from this codebase at [jkpfactors.com](https://jkpfactors.com) and the stock returns and firm characteristics at [wrds-www.wharton.upenn.edu/pages/get-data/contributed-data-forms/global-factor-data/](https://wrds-www.wharton.upenn.edu/pages/get-data/contributed-data-forms/global-factor-data/).

- The original SAS/R codebase is still available at [github.com/bkelly-lab/ReplicationCrisis](https://github.com/bkelly-lab/ReplicationCrisis), but we recommend using this new Python codebase for future work.

## License

Code in this repository is released under the [MIT License](LICENSE).

Data distributed in this repository is licensed under [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](DATA_LICENSE).

See [LICENSE](LICENSE) and [DATA_LICENSE](DATA_LICENSE) for details.
