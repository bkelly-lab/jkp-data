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
     git clone git@github.com:bkelly-lab/jkp-data.git
     ```
2. **Input WRDS credentials**

   - To save your WRDS credentials, navigate to the `jkp-data/` folder and run:
     ```sh
     uv run python code/wrds_credentials.py
     ```
     Kindly follow the prompts.  

     Note: If you need to change your password or credentials, run `uv run python code/wrds_credentials.py --reset` and then `uv run python code/wrds_credentials.py`

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
     uv run python code/main.py
     ```
     to create the stock returns and firm characteristics, and 
     ```sh
     uv run python code/portfolio.py
     ```
     to create the factor returns.

   **IMPORTANT:** When starting the code, you may be prompted to grant access to WRDS using two-factor authentication, for example via a Duo notification. You need to approve this request, as the program will otherwise fail. After a few seconds or minutes, you should see data being created in `data/raw`. If that is not the case, please check your internet connection or credentials.

When the code is finished, you can find the output in:
```
data/processed/
```
Please see the release notes (`documentation/release_notes.html`) for a description of the output files and a comparison between the output of the SAS/R codebase and the new Python codebase.

## Notes
- By default, the end date for the data in the code is 2024-12-31, which you can change by editing line 4 of the `code/main.py` file. For example, for May 6, 1992, use: `end_date = pl.datetime(1992, 5, 6)`.

- To run the code, we utilize a high performance computing cluster, where we request 450 GB RAM and 128 CPU cores. Running the routine takes about 6 hours.

- To understand the data, please refer to our [documentation](https://jkpfactors.s3.amazonaws.com/documents/Documentation.pdf). 

- We distribute the global factor returns generated from this codebase at [jkpfactors.com](https://jkpfactors.com) and the stock returns and firm characteristics at [wrds-www.wharton.upenn.edu/pages/get-data/contributed-data-forms/global-factor-data/](https://wrds-www.wharton.upenn.edu/pages/get-data/contributed-data-forms/global-factor-data/).

- The original SAS/R codebase is still available at [github.com/bkelly-lab/ReplicationCrisis](https://github.com/bkelly-lab/ReplicationCrisis), but we recommend using this new Python codebase for future work.



