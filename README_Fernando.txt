================================================================================
README
================================================================================

1. `wrds_Fernando.py`:
--------------------------------------------------------------------------------
- This is a refactored version of the `wrds` python library aimed to offer faster download speeds.
- Ensure this file is in the same directory as all other relevant code files.
- Upon importing this library in other python files, a compiled version will automatically be created.
- This is the latest stable release, but it is a shorter version. Expect more functionalities in future updates.

2. `WRDS_SQL_queries_Fernando.py`:
--------------------------------------------------------------------------------
- This file makes calls to the WRDS API to fetch raw data which is then stored in feather format.
- Before running this code, create a directory named 'Raw data'. The data will be downloaded to 
this directory, which will then be used for further processing.
- This is an ongoing project. More API queries will be added in subsequent versions.

3. `prepare_comp_crsp_Fernando.py`:
--------------------------------------------------------------------------------
- Executes the prepare_comp_sf and prepare_crsp_sf functions for both the daily, and monthly options.
- The code first defines functions and subroutines necessary for getting the stock files.
- Output feather format files with the stock files.

4. Short Instructions:
--------------------------------------------------------------------------------
- Download all code files and place them in a single directory.
- Create a subdirectory named 'Raw data'.
- First, execute `WRDS_SQL_queries_Fernando.py`.
- Followed by `prepare_comp_crsp_Fernando.py`.
- The final output will be extensive files in feather format.


A sample of the compustat output files, the full crsp output files, and the ‘Raw Data’ directory is available on:
https://www.dropbox.com/scl/fo/wrd688slicwko7iucn3v3/h?rlkey=vjwfi2xl3swc2aazbcoaga6lb&dl=0

Functions already implememented:
prepare_comp_sf
populate_own
compustat_fx
comp_exchanges
add_primary_sec
prepare_crsp_sf

To do: Comment code and put some references. Test against SAS. Rest of the functions.
================================================================================

