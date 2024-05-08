#!/bin/bash
mkdir -p Characteristics
mkdir -p Raw\ data
mkdir -p World_Ret_Monthly
mkdir -p Daily_Returns
python WRDS_SQL_queries_Fernando.py
python trial_full_run.py
