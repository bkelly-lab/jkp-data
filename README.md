# Global Stock Returns and Stock Characteristics Database

This document provides instructions for creating a dataset based on the paper "Is There a Replication Crisis in Finance?" by Jensen, Kelly, and Pedersen (Journal of Finance, 2023).

## Instructions

### Prerequisites

- Ensure you have Python installed on your system.
- Obtain your WRDS credentials.
- Ensure you have conda or miniconda installed on your system.

### Steps

1. **Setup a conda environment for the script and install the packages in the `requirements.txt` file:**

   A sample set of instructions is provided below, however, the specific instructions may vary depending on your installation of conda.
   
   - Download the `requirements.txt` file and navigate to the directory where it's stored.
   - Run the following commands:
     ```sh
     conda create --name jkp_factors python=3.9
     conda activate jkp_factors
     pip install -r requirements.txt
     ```

2. **Download the 'Build_database' Folder**

   - Download the 'Build_database' folder to your local machine by running the following command:
     ```sh
     git clone https://github.com/bkelly-lab/SAS-Python-Migrate.git
     ```

3. **Input WRDS Credentials**

   - Open the `main.py` file inside the 'Build_database' folder and input your WRDS credentials on line 3.

4. **Set Desired End Date**

   - Choose your desired end date and input it on line 4 of the `main.py` file. The date should be in the format of 'Year, Month, Day' and entered as integers. For example, for May 6th, 1992, the input would be:
     ```python
     end_date = pl.datetime(1992, 5, 6)
     ```

5. **Run the Script**

   - Make sure you have activated your new conda environment.
   - Navigate to the 'Build_database' directory and run the following command from the command line:
     ```sh
     python main.py
     ```

   In the initial moments of execution, you may be prompted to grant access to WRDS using two-factor authentication (2FA), such as a Duo notification. It is crucial to grant this access for the program to function. After a few seconds/minutes, you should see a confirmation message from the Python code indicating that a table has been downloaded. If this message does not appear, please check your Internet connection. Should the code generate a timeout error, restart the code and attempt to download the tables again, as this error typically indicates that access to WRDS was not properly granted.

   After execution, you can deactivate the conda environment with:
   ```sh
   conda deactivate
   ```

At the end of the routine, 4 folders should appear: `World_Ret_Monthly`, `World_Data`, `Daily_Returns`, and `Characteristics`. They contain Global Stock Returns and Stock Characteristics.

### Important Notice

Do not unpack the folder into another directory. All files with the `.ft` extension in the folder where `main.py` is executed will be deleted. To avoid any unintended data loss, ensure that you run the script from within the 'Build_database' folder.

## Hardware Requirements

We use a server with 503 GB RAM and 128 CPU cores. We recommend having at least 2 TB of free storage to run this code.


