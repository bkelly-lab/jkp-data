# Global Stock And Factors Database

This document provides instructions for creating a dataset based on the paper *“Is There a Replication Crisis in Finance?”* by Jensen, Kelly, and Pedersen (*Journal of Finance*, 2023) as well as portfolio returns.

## Instructions

### Prerequisites

- Obtain your WRDS credentials.
- Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) installed on your system.

### Steps

1. **Download the `SAS-Python-Migrate` folder**

   - Download the folder to your local machine by running the following command:
     ```sh
     git clone git@github.com:bkelly-lab/SAS-Python-Migrate.git
     ```
2. **Input WRDS Credentials**

   - To save your WRDS credentials, run:
     ```sh
     uv run python code/wrds_credentials.py
     ```
     Kindly follow the prompts.  

     Note: If you need to change your password or credentials, run `uv run python code/wrds_credentials.py --reset` and then `uv run python code/wrds_credentials.py`

4. **Run the Script**

   - A sample Slurm script is provided to run the Python routine on a cluster with a Slurm scheduler.

   - Run:
     ```sh
     sbatch slurm/submit_job_som_hpc.slurm
     ```
     (This will create the characteristics and the portfolio datasets).

     In interactive mode, run:
     ```sh
     uv run python code/main.py
     ```
     This will create the characteristics dataset at stock-level. To get the portfolio return series, run:
     ```sh
     uv run python code/portfolio.py
     ```

   During the initial execution, you may be prompted to grant access to WRDS using two-factor authentication (2FA), such as a Duo notification.  
   It is crucial to approve this request for the program to function correctly.

   After a few seconds or minutes, you should see files being created in `build_database/code/raw_table`.  
   If that is not the case, please check your internet connection or credentials.

At the end of the routine, you will find the output in:
```
data/processed/
```
Please see the release notes (`documentation/release_notes.html`) for a description of the output files.

### Notes

By default, the end date for the data in the code is **2024-12-31**.  
You can change it by editing line 4 of the `main.py` file.  

The date should be in the format `'Year, Month, Day'` and entered as integers.  
For example, for May 6, 1992, use:

```python
end_date = pl.datetime(1992, 5, 6)
```

A wide array of options for portfolios is available in the source code. For example, characteristic managed portfolios. Please refer to the SAS version of the code for more extensive documentation of the portfolio code since the Python version replicates the R code and there are no major changes in the structure of the code. 

To regenerate the release notes `html` file:
1. Navigate to `documentation/` and run:
    ```sh
     conda create --name jkp_factors python=3.11.11 -y
     conda activate jkp_factors
     conda install -c conda-forge postgresql jupyter deno pandoc quarto -y
     pip install -r requirements.txt
     conda activate jkp_factors
     ```
2. Run: 
```sh
  quarto render jkp_factors_migration.qmd --embed-resources
```

## Hardware Requirements

We use a server with **450 GB RAM** and **128 CPU cores**. Running the routine takes about five and a half hours.
