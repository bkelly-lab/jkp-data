# Global Stock And Factors Database

This document provides instructions for creating a dataset based on the paper *“Is There a Replication Crisis in Finance?”* by Jensen, Kelly, and Pedersen (*Journal of Finance*, 2023) as well as portfolio returns.

## Instructions

### Prerequisites

- Obtain your WRDS credentials.
- Ensure you have Conda or Miniconda installed on your system.

### Steps

1. **Download the `SAS-Python-Migrate` folder**

   - Download the folder to your local machine by running the following command:
     ```sh
     git clone git@github.com:bkelly-lab/SAS-Python-Migrate.git
     ```
     
2. **Set up a Conda environment and install the packages in the `requirements.txt` file**

   The sample instructions below may vary depending on your Conda installation.

   - Navigate to `SAS-Python-Migrate/`
   - Run the following commands:
     ```sh
     conda create --name jkp_factors python=3.11.11 -y
     conda activate jkp_factors
     conda install -c conda-forge postgresql jupyter deno pandoc quarto -y
     pip install -r requirements.txt
     ```
3. **Input WRDS Credentials**

   - Navigate to `SAS-Python-Migrate/build_database/code` and run:
     ```sh
     python jkp_credentials.py
     ```
     Kindly follow the prompts.  

     Note: If you need to change your password or credentials, run `python jkp_credentials.py --reset` and then `python jkp_credentials.py`

4. **Run the Script**

   - Make sure you have activated your Conda environment.  
     A sample Slurm script is provided to run the Python routine on a cluster with a Slurm scheduler.

   - Navigate to the `SAS-Python-Migrate` directory and run:
     ```sh
     sbatch build_database/slurm/submit_job_som_hpc.slurm
     ```
     (This will create the characteristics and the portfolio datasets).

     In interactive mode, navigate to `SAS-Python-Migrate` and run:
     ```sh
     python build_database/code/main.py
     ```
     This will create the characteristics dataset at stock-level. To get the portfolio return series, run:
     ```sh
     python build_database/code/portfolio.py
     ```

   During the initial execution, you may be prompted to grant access to WRDS using two-factor authentication (2FA), such as a Duo notification.  
   It is crucial to approve this request for the program to function correctly.

   After a few seconds or minutes, you should see files being created in `build_database/code/raw_table`.  
   If that is not the case, please check your internet connection or credentials.

   After execution, deactivate the Conda environment:
   ```sh
   conda deactivate
   ```

At the end of the routine, you will find the output in:
```
SAS-Python-Migrate/build_database/data/processed/
```
Please see the release notes (`release_notes.html`) for a description of the output files.

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
1. Activate the `jkp_factors` conda environment
2. Navigate do `SAS-Python-Migrate` and run: 
```sh
  quarto render release_notes_files/jkp_factors_migration.qmd --embed-resources && mv release_notes_files/jkp_factors_migration.html release_notes.html
```

## Hardware Requirements

We use a server with **450 GB RAM** and **128 CPU cores**. Running the routine takes about five and a half hours.
