# Global Stock Returns and Stock Characteristics Database

This document provides instructions for creating a dataset based on the paper *“Is There a Replication Crisis in Finance?”* by Jensen, Kelly, and Pedersen (*Journal of Finance*, 2023).

## Instructions

### Prerequisites

- Obtain your WRDS credentials.  
- Ensure you have Conda or Miniconda installed on your system.

### Steps

1. **Set up a Conda environment and install the packages in the `requirements.txt` file**

   The sample instructions below may vary depending on your Conda installation.

   - Download the `requirements.txt` file and navigate to the directory where it is stored.
   - Run the following commands:
     ```sh
     conda create --name jkp_factors python=3.11.11
     conda activate jkp_factors
     conda install -c conda-forge postgresql
     pip install -r requirements.txt
     ```

2. **Download the `Build_database` Folder**

   - Download the folder to your local machine by running the following command:
     ```sh
     git clone https://github.com/bkelly-lab/SAS-Python-Migrate.git
     ```

3. **Input WRDS Credentials**

   - Navigate to `SAS-Python-Migrate/build_database/code` and run the `jkp_credentials.py` script.  
     You will need to input your credentials to access WRDS.  

     If you need to change your password or credentials, run:
     ```sh
     python jkp_credentials.py --reset
     ```
     and then re-run:
     ```sh
     python jkp_credentials.py
     ```

4. **Run the Script**

   - Make sure you have activated your Conda environment.  
     A sample Slurm script is provided to run the Python routine on a cluster with a Slurm scheduler.

   - Navigate to the `SAS-Python-Migrate` directory and run:
     ```sh
     sbatch build_database/slurm/submit_job_som_hpc.slurm
     ```

     In interactive mode, navigate to `SAS-Python-Migrate` and run:
     ```sh
     python build_database/slurm/main.py
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

### Notes

By default, the end date for the data in the code is **2024-12-31**.  
You can change it by editing line 4 of the `main.py` file.  

The date should be in the format `'Year, Month, Day'` and entered as integers.  
For example, for May 6, 1992, use:

```python
end_date = pl.datetime(1992, 5, 6)
```

## Hardware Requirements

We use a server with **450 GB RAM** and **128 CPU cores**. Running the routine takes about five and a half hours.
