def populate_own(inset, idvar, datevar, datename, forward_max, period):
    # sort the input according to id and date respectively and remove duplicates: arg: inset=, datevar=, idvar
    temp_df = inset.sort([idvar, datevar]).unique([idvar, datevar], keep='last')

    if period == 'month':

        # add a new date column that has month end dates: arg: datename=:
        temp_df = temp_df.with_columns(pl.col(datevar).dt.month_end().alias(datename))

        # create an empty dataframe
        output = temp_df.clear()

        # sort into groups according to ids to forward fill:
        for idx, group in temp_df.group_by(idvar):
            min_date = group.select(pl.min(datename))[0,0]
            max_date = group.select(pl.max(datename))[0,0] + relativedelta(months=forward_max)
            max_date = max_date.replace(day=1)
            max_date = max_date - datetime.timedelta(days=1)

            all_dates = []
            current_date = min_date

            while current_date <= max_date:
                all_dates.append(current_date)
                current_date += relativedelta(months=1)

            all_dates_df = pl.DataFrame({datename: all_dates})
            all_dates_df = all_dates_df.select(pl.col(datename).dt.month_end())
            all_dates_df = all_dates_df.join(group, on=datename, how="left")
            all_dates_df = all_dates_df.select(pl.all().forward_fill(limit=forward_max))
            all_dates_df = all_dates_df.select(output.columns)

            output = output.vstack(all_dates_df)

        return output

    else:

        # add a new date column that has daily dates: arg: datename=:
        temp_df = temp_df.with_columns(pl.col(datevar).alias(datename))

        # create an empty dataframe
        output = temp_df.clear()

        # sort into groups according to ids to forward fill:
        for idx, group in temp_df.group_by(idvar):
            min_date = group.select(pl.min(datename))[0,0]
            max_date = group.select(pl.max(datename))[0,0] + relativedelta(days=forward_max)
            # max_date = max_date.replace(day=1)
            # max_date = max_date - datetime.timedelta(days=1)

            all_dates = []
            current_date = min_date

            while current_date <= max_date:
                all_dates.append(current_date)
                current_date += relativedelta(days=1)
            
            all_dates_df = pl.DataFrame({datename: all_dates})
            all_dates_df = all_dates_df.join(group, on=datename, how="left")
            all_dates_df = all_dates_df.select(pl.all().forward_fill(limit=forward_max))
            all_dates_df = all_dates_df.select(output.columns)

            output = output.vstack(all_dates_df)

        return output




def compustat_fx(exrt_dly):
    usd_curcdd = pl.DataFrame({
        'curcdd': ['USD'],
        'datadate': [19500101],
        'fx': [1.00]
    })

    
    a = exrt_dly.filter(pl.col('fromcurd') == 'GBP')
    b = exrt_dly.filter(pl.col('tocurd') == 'USD')

    # Convert the 'datadate' column to Date type
    usd_curcdd = usd_curcdd.with_columns(
        pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
    )

    fx1 = (
        a.join(b, on=['fromcurd', 'datadate'])
        .with_columns((pl.col('exratd') / pl.col('exratd_right')).alias('fx'))
        .select([pl.col('tocurd').alias('curcdd'), 'datadate', 'fx'])
    )

    # Step 4: Merging USD Base Data with Extracted FX Data
    fx2 = fx1.vstack(usd_curcdd)

    # Step 5: Sorting the Data
    fx2 = fx2.sort(['curcdd', 'datadate'], descending=[False, False])
    fx3 = populate_own(inset=fx2, idvar='curcdd', datevar='datadate', datename='date', forward_max=12, period='daily')
    fx3 = fx3.filter(~pl.all_horizontal(pl.col("curcdd").is_null()))
    fx3=fx3.drop("datadate").select([pl.col('curcdd'), pl.col('date').alias('datadate'), pl.col("fx")])

   

    
    # Step 7: Final Sorting and Deduplication
    output = fx3.sort(['curcdd', 'datadate']).unique(['curcdd', 'datadate'])

    return output


def parse_dates(df, column_name="datadate", specified_format=None):
    """
    Parse a date column in the given DataFrame using Polars API with a specified format or multiple default formats.
    
    Parameters:
    - df: The input Polars DataFrame.
    - column_name: The name of the column to parse. Default is "datadate".
    - specified_format: A specific date format to try first. If not provided, defaults will be used.
    
    Returns:
    - A Polars DataFrame with the parsed date column.
    """
    formats = ["%Y%m%d", "%Y-%m-%d", "%d-%m-%Y"]
    
    # If a specified format is provided, prioritize it
    if specified_format:
        formats.insert(0, specified_format)
    
    for fmt in formats:
        try:
            # Attempt to parse the date column with the current format
            df = df.with_columns(
                pl.col(column_name).cast(pl.Utf8).str.strptime(pl.Date, format=fmt).alias(column_name)
            )
            # If parsing succeeds without errors, break out of the loop
            break
        except:
            # If parsing fails, continue to the next format
            continue
    
    return df

def CRSP_INDUSTRY():
    """
    Create daily historical SIC and NAICS industry identifiers dataset from CRSP data 
    
    Parameters:
    - None

    Returns:
    - Historical SIC and NAICS industry identifiers
    """
    
    # Load the CSV file into a DataFrame
    CRSP_DSENAMES = pl.read_csv("path", ignore_errors=True)
    
    # Convert date columns from string to date format
    CRSP_DSENAMES = CRSP_DSENAMES.with_columns(
        pl.col("NAMEDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("NAMEDT"),
        pl.col("NAMEENDT").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("NAMEENDT")
    )

    # Select relevant columns, rename, sort, and remove duplicates
    permno0 = CRSP_DSENAMES.select(
        pl.col("PERMNO"),
        pl.col("PERMCO"),
        pl.col("NAMEDT"),
        pl.col("NAMEENDT"),
        pl.col("SICCD").alias("SIC"),
        pl.col("NAICS").cast(pl.Int32).alias("NAICS"),
    ).unique().sort(["PERMNO", "NAMEDT", "NAMEENDT"])

    # Replace missing or zero values in SIC and NAICS with -999
    permno1 = permno0.with_columns(
        pl.when(pl.col("SIC").is_null() | (pl.col("SIC") == 0))
        .then(-999)
        .otherwise(pl.col("SIC"))
        .alias("SIC")
    ).with_columns(
        pl.when(pl.col("NAICS").is_null())
        .then(-999)
        .otherwise(pl.col("NAICS"))
        .alias("NAICS")        
    )
    
    # Calculate duration between NAMEENDT and NAMEDT dates
    permno2 = permno1.with_columns(
        (pl.col("NAMEENDT") - pl.col("NAMEDT")).alias("PERMNO_DIFF")
    )

    # Expand rows for each date in the range between NAMEDT and NAMEENDT
    permno3 = permno2.with_columns(pl.date_ranges("NAMEDT", "NAMEENDT")).explode("date_range")
    permno3 = permno3.select(["PERMNO", "PERMCO", "SIC", "NAICS", pl.col("date_range").alias("NAMEDT")])

    # Convert the placeholder value -999 back to None for SIC and NAICS columns and rename the NAMEDT column to DATE
    permno4 = permno3.with_columns(
        pl.when(pl.col("SIC") == -999)
        .then(pl.lit(None).cast(pl.Int64))
        .otherwise(pl.col("SIC"))
        .alias("SIC")
    ).with_columns(
        pl.when(pl.col("NAICS") == -999)
        .then(pl.lit(None).cast(pl.Int64))
        .otherwise(pl.col("NAICS"))
        .alias("NAICS")
    ).rename({"NAMEDT": "DATE"})

    # Sort the DataFrame by PERMNO and DATE and remove duplicates based on these columns
    out = permno4.sort(["PERMNO", "DATE"]).unique(["PERMNO", "DATE"])
    
    return out

def COMP_SIC_NAICS():
    """
    Create a daily historical SIC and NAICS industry identifiers dataset using NA and global annual reports
    
    Parameters:
    - None

    Returns:
    - Historical SIC and NAICS industry identifiers dataset
    """
    
    # Load the COMP_FUNDA CSV file into a DataFrame and format relevant columns
    COMP_FUNDA = pl.read_csv("path", infer_schema_length=int(1e13), ignore_errors=True)
    COMP_FUNDA = COMP_FUNDA.with_columns(
        pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
    )
    COMP_FUNDA = COMP_FUNDA.with_columns(pl.col("gvkey").cast(pl.Utf8).alias("gvkey"))

    # Load the COMP_G_FUNDA CSV file into a DataFrame and format relevant columns
    COMP_G_FUNDA = pl.read_csv("path", infer_schema_length=int(1e10), ignore_errors=True)
    COMP_G_FUNDA = COMP_G_FUNDA.with_columns(
        pl.col("datadate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("datadate")
    )
    COMP_G_FUNDA = COMP_G_FUNDA.with_columns(pl.col("gvkey").cast(pl.Utf8).alias("gvkey"))

    # Select relevant columns from COMP_FUNDA and remove duplicates
    comp1 = COMP_FUNDA.select(["gvkey", "datadate", pl.col("sich").alias("sic"), pl.col("naicsh").alias("naics")]).unique()

    # Filter out a specific set of data based on given conditions
    comp2 = comp1.filter(~((pl.col("gvkey") == "175650") & 
                           (pl.col("datadate") == pl.date(2005,12,31)) & 
                           (pl.col("naics").is_null())))

    # Select relevant columns from COMP_G_FUNDA and remove duplicates
    comp3 = COMP_G_FUNDA.select(["gvkey", "datadate", pl.col("sich").alias("sic"), pl.col("naicsh").alias("naics")]).unique()

    # Outer join comp2 and comp3 DataFrames on 'gvkey' and 'datadate'
    comp4 = comp2.join(comp3, on=["gvkey", "datadate"], how="outer")

    # Create a DataFrame selecting and re-assigning columns based on conditions
    comp5 = comp4.select([
        "gvkey", 
        pl.col("datadate").alias("date"), 
        pl.when(pl.col("sic").is_null()).then(pl.col("sic_right")).otherwise(pl.col("sic")).alias("sic"),
        pl.when(pl.col("naics").is_null()).then(pl.col("naics_right")).otherwise(pl.col("naics")).alias("naics")
    ])

    # Sort DataFrame by 'gvkey' and 'date', and prepare for date range calculations
    comp6 = comp5.sort(["gvkey", "date"])
    comp6 = comp6.with_columns(
        pl.when(pl.col('gvkey') != pl.col('gvkey').shift(-1))
        .then(True)
        .otherwise(False)
        .alias("mask")
    )
    comp6 = comp6.with_columns(pl.col("date").shift(-1).alias("end_date"))
    comp6 = comp6.with_columns(
        pl.when(pl.col('mask'))
        .then(pl.lit(None))
        .otherwise(pl.col("end_date"))
        .alias("updated_end_date")
    )
    comp6 = comp6.with_columns(
        pl.when(pl.col('updated_end_date').is_null())
        .then(pl.col("date"))
        .otherwise(pl.col("updated_end_date"))
        .alias("valid_to")
    )

    # Expand rows for each date in the range between 'date' and 'valid_to'
    comp7 = comp6.with_columns(pl.date_ranges("date", "valid_to")).explode("date_range")

    # Select and rename the necessary columns
    comp8 = comp7.select(["gvkey", "date_range", "sic", "naics"])
    comp8 = comp8.rename({"date_range": "date"})

    # Remove duplicates based on 'gvkey' and 'date' and sort the DataFrame
    out = comp8.unique(subset=["gvkey", "date"], keep="last").sort(["gvkey", "date"])

    return out

def COMP_HGICS(lib):
    """
    Create a daily historical gics dataset from COMPUSTAT, either from the NA or global dataset 
    
    Parameters:
    - lib: A string, if 'global' will process COMP_G_CO_HGIC dataset, otherwise will process COMP_CO_HGIC dataset.

    Returns:
    - Daily historical gics dataset from COMPUSTAT
    """
    
    # If lib parameter is 'global', read COMP_G_CO_HGIC dataset, else read COMP_CO_HGIC dataset
    if lib == "global":
        COMP_G_CO_HGIC = pl.read_csv("path", infer_schema_length=int(1e13), ignore_errors=True)
        COMP_G_CO_HGIC = COMP_G_CO_HGIC.with_columns(
            pl.col("indfrom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indfrom"),
            pl.col("indthru").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indthru"),
            pl.col("gvkey").cast(pl.Utf8),
            pl.col("indtype").cast(pl.Utf8),
            pl.col("ggroup").cast(pl.Utf8),
            pl.col("gind").cast(pl.Utf8),
            pl.col("gsector").cast(pl.Utf8),
            pl.col("gsubind").cast(pl.Utf8)
        )
        data = COMP_G_CO_HGIC
    else:
        COMP_CO_HGIC = pl.read_csv("path", infer_schema_length=int(1e13), ignore_errors=True)
        COMP_CO_HGIC = COMP_CO_HGIC.with_columns(
            pl.col("indfrom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indfrom"),
            pl.col("indthru").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("indthru"),
            pl.col("gvkey").cast(pl.Utf8),
            pl.col("indtype").cast(pl.Utf8),
            pl.col("ggroup").cast(pl.Utf8),
            pl.col("gind").cast(pl.Utf8),
            pl.col("gsector").cast(pl.Utf8),
            pl.col("gsubind").cast(pl.Utf8)
        )
        data = COMP_CO_HGIC
        

    # Filter rows where 'gvkey' is not null, select relevant columns, and remove duplicates
    gic1 = data.filter(pl.col("gvkey").is_not_null()).select([
        "gvkey", "indfrom", "indthru", pl.col("gsubind").alias("gics")]
    ).unique().sort(["gvkey", "indfrom"])

    # Replace null 'gics' values with -999
    gic2 = gic1.with_columns(
        pl.when(pl.col("gics").is_null()).then(-999).otherwise(pl.col("gics")).alias("gics")
    )

    # Handle missing 'indthru' values
    gic3 = gic2.with_columns(
        pl.when((pl.col("indthru").is_null()) & (pl.col("gvkey") != pl.col("gvkey").shift(-1)))
        .then(pl.lit(date.today()))
        .otherwise(pl.col("indthru"))
        .alias("indthru")
    )

    # Handle missing 'indthru' for the last row
    gic3 = gic3.with_row_count().with_columns(
        pl.when((pl.col("row_nr") == (gic3.height-1)) & (pl.col("indthru").is_null()))
        .then(pl.lit(date.today()))
        .otherwise(pl.col("indthru"))
        .alias("indthru")
    ).drop("row_nr")

    # Expand rows for each date in the range between 'indfrom' and 'indthru'
    gic4 = gic3.with_columns(pl.date_ranges("indfrom", "indthru")).explode("date_range")

    # Select and rename the necessary columns
    gic5 = gic4.select(["gvkey", "gics", pl.col("date_range").alias("date")])

    # Remove duplicates based on 'gvkey' and 'date' and sort the DataFrame
    out = gic5.sort(["gvkey", "date"]).unique(subset=["gvkey", "date"], keep="last")

    return out



def HGICS_JOIN():
    """
    Join NA and global daily historical gics data from COMPUSTAT.

    Parameters:
    - None

    Returns:
    - A Polars DataFrame with joined and cleaned GICS and date information.
    """
    g_hgics = COMP_HGICS(lib="global")
    na_hgics = COMP_HGICS(lib="local")

    # Join the local_data and global_data based on 'gvkey' and 'date' using an outer join
    gjoin1 = local_data.join(global_data, on=["gvkey", "date"], how="outer")

    # Handle missing 'gics' values. If 'gics' from local_data is missing, use 'gics' from global_data
    gjoin2 = gjoin1.with_columns(
        pl.when(pl.col("gics").is_null()).then(pl.col("gics_right")).otherwise(pl.col("gics")).alias("gics")
    ).select(["gvkey", "date", "gics"])

    # Remove duplicates based on 'gvkey' and 'date' and sort the DataFrame
    out = gjoin2.sort(["gvkey", "date"]).unique(subset=["gvkey", "date"], keep="last")

    return out



def comp_industry():
    """
    Joins SIC and NAICS industry identifiers to GICS identifiers constructed from COMPUSTAT data

    Parameters:
    - None

    Returns:
    - Combined SIC, NAICS, and GICS identifiers constructed from COMPUSTAT data
    """
    comp_other= COMP_SIC_NAICS()
    comp_gics=HGICS_JOIN()

    # Step 1: Perform an outer join between gics_table and other_table based on 'gvkey' and 'date' and sort the resulting DataFrame
    join1 = gics_table.join(other_table, on=["gvkey", "date"], how="outer")
    join1 = join1.sort(["gvkey", "date"])

    # Step 2: Create a mask column to identify the last row for each 'gvkey' group
    join2 = join1.with_columns(
        pl.when(pl.col('gvkey') != pl.col('gvkey').shift(-1))
        .then(True)
        .otherwise(False)
        .alias("mask")
    )

    # Step 3: Shift the 'date' column to compute end dates
    join3 = join2.with_columns(pl.col("date").shift(-1).alias("end_date"))

    # Step 4: Update the end date based on the mask created in Step 2
    join4 = join3.with_columns(
        pl.when(pl.col('mask'))
        .then(pl.col("date"))
        .otherwise(pl.col("end_date")) 
        .alias("updated_end_date")
    )

    # Step 5: Handle the end date for the very last row
    join5 = join4.with_row_count().with_columns(
        pl.when((pl.col("row_nr") == (join4.height-1)) & (pl.col("updated_end_date").is_null()))
        .then(pl.col("date"))
        .otherwise(pl.col("updated_end_date"))
        .alias("final_end_date")
    )

    # Step 6: Generate a range of dates between the 'date' and 'final_end_date'
    join6 = join5.with_columns(pl.date_ranges("date", "final_end_date", closed="left"))

    # Step 7: Explode the DataFrame based on the date range
    join7 = join6.explode("date_range")

    # Step 8: Create a mask to identify rows with a change in 'gvkey'
    join8 = join7.with_columns(
        pl.when(pl.col('row_nr') != pl.col('row_nr').shift(1))
        .then(False)
        .otherwise(True)
        .alias("mask_orig")
    ).drop("row_nr")

    # Step 9: Adjust the mask for the first row
    join9 = join8.with_row_count().with_columns(
        pl.when(pl.col("row_nr") == 0).then(False)
        .otherwise(pl.col("mask_orig"))
        .alias("mask_orig")
    ).drop("row_nr")

    # Step 10: Reset industry classification columns based on the mask
    join10 = join9.with_columns([
        pl.when(pl.col("mask_orig"))
        .then(pl.lit(None))
        .otherwise(pl.col("gics"))
        .alias("gics"),
        
        pl.when(pl.col("mask_orig"))
        .then(pl.lit(None))
        .otherwise(pl.col("sic"))
        .alias("sic"),
        
        pl.when(pl.col("mask_orig"))
        .then(pl.lit(None))
        .otherwise(pl.col("naics"))
        .alias("naics"),
    ])

    # Step 11: Rename the 'date_range' column and filter the relevant columns
    join11 = join10.select(["gvkey", pl.col("date_range").alias("date"), "gics", "sic", "naics"])

    # Step 12: Remove duplicates and sort the final DataFrame
    join12 = join11.unique(["gvkey", "date"], keep="first")
    out = join12.sort(["gvkey", "date"])

    return out



def ff_ind_class(data, ff_grps):
    if ff_grps==38:
        df1 =   data.with_columns(pl.when(pl.col("sic").is_between(100, 999)).then(1)
        .when(pl.col("sic").is_between(1000, 1299)).then(2)
        .when(pl.col("sic").is_between(1300, 1399)).then(3)
        .when(pl.col("sic").is_between(1400, 1499)).then(4)
        .when(pl.col("sic").is_between(1500, 1799)).then(5)
        .when(pl.col("sic").is_between(2000, 2099)).then(6)
        .when(pl.col("sic").is_between(2100, 2199)).then(7)
        .when(pl.col("sic").is_between(2200, 2299)).then(8)
        .when(pl.col("sic").is_between(2300, 2399)).then(9)
        .when(pl.col("sic").is_between(2400, 2499)).then(10)
        .when(pl.col("sic").is_between(2500, 2599)).then(11)
        .when(pl.col("sic").is_between(2600, 2661)).then(12)
        .when(pl.col("sic").is_between(2700, 2799)).then(13)
        .when(pl.col("sic").is_between(2800, 2899)).then(14)
        .when(pl.col("sic").is_between(2900, 2999)).then(15)
        .when(pl.col("sic").is_between(3000, 3099)).then(16)
        .when(pl.col("sic").is_between(3100, 3199)).then(17)
        .when(pl.col("sic").is_between(3200, 3299)).then(18)
        .when(pl.col("sic").is_between(3300, 3399)).then(19)
        .when(pl.col("sic").is_between(3400, 3499)).then(20)
        .when(pl.col("sic").is_between(3500, 3599)).then(21)
        .when(pl.col("sic").is_between(3600, 3699)).then(22)
        .when(pl.col("sic").is_between(3700, 3799)).then(23)
        .when(pl.col("sic").is_between(3800, 3879)).then(24)
        .when(pl.col("sic").is_between(3900, 3999)).then(25)
        .when(pl.col("sic").is_between(4000, 4799)).then(26)
        .when(pl.col("sic").is_between(4800, 4829)).then(27)
        .when(pl.col("sic").is_between(4830, 4899)).then(28)
        .when(pl.col("sic").is_between(4900, 4949)).then(29)
        .when(pl.col("sic").is_between(4950, 4959)).then(30)
        .when(pl.col("sic").is_between(4960, 4969)).then(31)
        .when(pl.col("sic").is_between(4970, 4979)).then(32)
        .when(pl.col("sic").is_between(5000, 5199)).then(33)
        .when(pl.col("sic").is_between(5200, 5999)).then(34)
        .when(pl.col("sic").is_between(6000, 6999)).then(35)
        .when(pl.col("sic").is_between(7000, 8999)).then(36)
        .when(pl.col("sic").is_between(9000, 9999)).then(37)
        .otherwise(pl.lit(None)) # If you have a default value, put it here. I've added 38 assuming it's the next number.
        .alias("ff38")
    )
        return df1
        
    else:
        df1= data.with_columns(pl.when(pl.col("sic").is_in([2048, 
                                *range(100, 299+1), 
                                *range(700, 799+1), 
                                *range(910, 919+1)])
           ).then(1)
    
    .when(pl.col("sic").is_in([2095, 2098, 2099, 
                               *range(2000, 2046+1), 
                               *range(2050, 2063+1), 
                               *range(2070, 2079+1), 
                               *range(2090, 2092+1)])
          ).then(2)
    
    .when(pl.col("sic").is_in([2086, 2087, 2096, 2097, 
                               *range(2064, 2068+1)])
          ).then(3)

    .when(pl.col("sic").is_in([2080, 
                               *range(2082, 2085+1)])
          ).then(4)

    .when(pl.col("sic").is_in([*range(2100, 2199+1)])
          ).then(5)

    .when(pl.col("sic").is_in([3732, 3930, 3931, 
                               *range(920, 999+1),
                               *range(3650, 3652+1),
                               *range(3940, 3949+1)])
          ).then(6)

    .when(pl.col("sic").is_in([7840, 7841, 7900, 7910, 7911, 7980,
                               *range(7800, 7833+1),
                               *range(7920, 7933+1),
                               *range(7940, 7949+1),
                               *range(7990, 7999+1)])
          ).then(7)

    .when(pl.col("sic").is_in([2770, 2771,
                               *range(2700, 2749+1),
                               *range(2780, 2799+1)])
          ).then(8)

    .when(pl.col("sic").is_in([2047, 2391, 2392, 3160, 3161, 3229, 3260, 3262, 3263, 3269, 3230, 3231, 3750, 3751, 3800, 3860, 
                               3861, 3910, 3911, 3914, 3915, 3991, 3995,
                               *range(2510, 2519+1),
                               *range(2590, 2599+1),
                               *range(2840, 2844+1),
                               *range(3170, 3172+1),
                               *range(3190, 3199+1),
                               *range(3630, 3639+1),
                               *range(3870, 3873+1),
                               *range(3960, 3962+1)])
          ).then(9)

    .when(pl.col("sic").is_in([3020, 3021, 3130, 3131, 3150, 3151,
                               *range(2300, 2390+1),
                               *range(3100, 3111+1),
                               *range(3140, 3149+1),
                               *range(3963, 3965+1)])
          ).then(10)

     .when(pl.col("sic").is_in([*range(8000, 8099+1)])
          ).then(11)

    .when(pl.col("sic").is_in([3693, 3850, 3851,
                               *range(3840, 3849+1)])
          ).then(12)

    .when(pl.col("sic").is_in([2830, 2831,
                               *range(2833, 2836+1)])
          ).then(13)

    .when(pl.col("sic").is_in([*range(2800, 2829+1),
                               *range(2850, 2879+1),
                               *range(2890, 2899+1)])
          ).then(14)

    .when(pl.col("sic").is_in([3031, 3041,
                               *range(3050, 3053+1),
                               *range(3060, 3099+1)])
          ).then(15)

    .when(pl.col("sic").is_in([*range(2200, 2284+1),
                               *range(2290, 2295+1),
                               *range(2297, 2299+1),
                               *range(2393, 2395+1),
                               *range(2397, 2399+1)])
          ).then(16)

    .when(pl.col("sic").is_in([2660, 2661, 3200, 3210, 3211, 3240, 3241, 3261, 3264, 3280, 3281, 3446, 3996,
                               *range(800, 899+1),
                               *range(2400, 2439+1),
                               *range(2450, 2459+1),
                               *range(2490, 2499+1),
                               *range(2950, 2952+1),
                               *range(3250, 3259+1),
                               *range(3270, 3275+1),
                               *range(3290, 3293+1),
                               *range(3295, 3299+1),
                               *range(3420, 3429+1),
                               *range(3430, 3433+1),
                               *range(3440, 3442+1),
                               *range(3448, 3452+1),
                               *range(3490, 3499+1)])
          ).then(17)

    .when(pl.col("sic").is_in([*range(1500, 1511+1),
                               *range(1520, 1549+1),
                               *range(1600, 1799+1)])
          ).then(18)

    .when(pl.col("sic").is_in([3300, 
                               *range(3310, 3317+1),
                               *range(3320, 3325+1),
                               *range(3330, 3341+1),
                               *range(3350, 3357+1),
                               *range(3360, 3379+1),
                               *range(3390, 3399+1)])
          ).then(19)

    .when(pl.col("sic").is_in([3400, 3443, 3444,
                               *range(3460, 3479+1)])
          ).then(20).when(pl.col("sic").is_in([3538, 3585, 3586,
                               *range(3510, 3536+1),
                               *range(3540, 3569+1),
                               *range(3580, 3582+1),
                               *range(3589, 3599+1)])
          ).then(21)

    .when(pl.col("sic").is_in([3600, 3620, 3621, 3648, 3649, 3660, 3699,
                               *range(3610, 3613+1),
                               *range(3623, 3629+1),
                               *range(3640, 3646+1),
                               *range(3690, 3692+1)])
          ).then(22)

    .when(pl.col("sic").is_in([2296, 2396, 3010, 3011, 3537, 3647, 3694, 3700, 3710, 3711, 3799,
                               *range(3713, 3716+1),
                               *range(3790, 3792+1)])
          ).then(23)

    .when(pl.col("sic").is_in([3720, 3721, 3728, 3729,
                               *range(3723, 3725+1)])
          ).then(24)

    .when(pl.col("sic").is_in([3730, 3731,
                               *range(3740, 3743+1)])
          ).then(25)

    .when(pl.col("sic").is_in([3795,
                               *range(3760, 3769+1),
                               *range(3480, 3489+1)])
          ).then(26)

    .when(pl.col("sic").is_in([*range(1040, 1049+1)])
          ).then(27)

    .when(pl.col("sic").is_in([*range(1000, 1039+1),
                               *range(1050, 1119+1),
                               *range(1400, 1499+1)])
          ).then(28)

    .when(pl.col("sic").is_in([*range(1200, 1299+1)])
          ).then(29)

    .when(pl.col("sic").is_in([1300, 1389,
                               *range(1310, 1339+1),
                               *range(1370, 1382+1),
                               *range(2900, 2912+1),
                               *range(2990, 2999+1)])
          ).then(30)

    .when(pl.col("sic").is_in([4900, 4910, 4911, 4939,
                               *range(4920, 4925+1),
                               *range(4930, 4932+1),
                               *range(4940, 4942+1)])
          ).then(31)

    .when(pl.col("sic").is_in([4800, 4899,
                               *range(4810, 4813+1),
                               *range(4820, 4822+1),
                               *range(4830, 4841+1),
                               *range(4880, 4892+1)])
          ).then(32)

    .when(pl.col("sic").is_in([7020, 7021, 7200, 7230, 7231, 7240, 7241, 7250, 7251, 7395, 7500, 7600, 7620, 7622, 7623, 7640, 
                               7641,
                               *range(7030, 7033+1),
                               *range(7210, 7212+1),
                               *range(7214, 7217+1),
                               *range(7219, 7221+1),
                               *range(7260, 7299+1),
                               *range(7520, 7549+1),
                               *range(7629, 7631+1),
                               *range(7690, 7699+1),
                               *range(8100, 8499+1),
                               *range(8600, 8699+1),
                               *range(8800, 8899+1),
                               *range(7510, 7515+1)])
          ).then(33)

    .when(pl.col("sic").is_in([3993, 7218, 7300, 7374, 7396, 7397, 7399, 7519, 8700, 8720, 8721,
                               *range(2750, 2759+1),
                               *range(7310, 7342+1),
                               *range(7349, 7353+1),
                               *range(7359, 7369+1),
                               *range(7376, 7385+1),
                               *range(7389, 7394+1),
                               *range(8710, 8713+1),
                               *range(8730, 8734+1),
                               *range(8740, 8748+1),
                               *range(8900, 8911+1),
                               *range(8920, 8999+1),
                               *range(4220, 4229+1)])
          ).then(34)

    .when(pl.col("sic").is_in([3695,
                               *range(3570, 3579+1),
                               *range(3680, 3689+1)])
          ).then(35)

    .when(pl.col("sic").is_in([7375,
                               *range(7370, 7373+1)])
          ).then(36)

    .when(pl.col("sic").is_in([3622, 3810, 3812,
                               *range(3661, 3666+1),
                               *range(3669, 3679+1)])
          ).then(37)

    .when(pl.col("sic").is_in([3811,
                               *range(3820, 3827+1),
                               *range(3829, 3839+1)])
          ).then(38)

    .when(pl.col("sic").is_in([2760, 2761,
                               *range(2520, 2549+1),
                               *range(2600, 2639+1),
                               *range(2670, 2699+1),
                               *range(3950, 3955+1)])
          ).then(39)

    .when(pl.col("sic").is_in([3220, 3221,
                               *range(2440, 2449+1),
                               *range(2640, 2659+1),
                               *range(3410, 3412+1)])
          ).then(40)

    .when(pl.col("sic").is_in([4100, 4130, 4131, 4150, 4151, 4230, 4231, 4780, 4789,
                               *range(4000, 4013+1),
                               *range(4040, 4049+1),
                               *range(4110, 4121+1),
                               *range(4140, 4142+1),
                               *range(4170, 4173+1),
                               *range(4190, 4200+1),
                               *range(4210, 4219+1),
                               *range(4240, 4249+1),
                               *range(4400, 4700+1),
                               *range(4710, 4712+1),
                               *range(4720, 4749+1),
                               *range(4782, 4785+1)])
          ).then(41)

    .when(pl.col("sic").is_in([5000, 5099, 5100,
                               *range(5010, 5015+1),
                               *range(5020, 5023+1),
                               *range(5030, 5060+1),
                               *range(5063, 5065+1),
                               *range(5070, 5078+1),
                               *range(5080, 5088+1),
                               *range(5090, 5094+1),
                               *range(5110, 5113+1),
                               *range(5120, 5122+1),
                               *range(5130, 5172+1),
                               *range(5180, 5182+1),
                               *range(5190, 5199+1)])
          ).then(42)

    .when(pl.col("sic").is_in([5200, 5250, 5251, 5260, 5261, 5270, 5271, 5300, 5310, 5311, 5320, 5330, 5331, 5334, 5900, 5999,
                               *range(5210, 5231+1),
                               *range(5340, 5349+1),
                               *range(5390, 5400+1),
                               *range(5410, 5412+1),
                               *range(5420, 5469+1),
                               *range(5490, 5500+1),
                               *range(5510, 5579+1),
                               *range(5590, 5700+1),
                               *range(5710, 5722+1),
                               *range(5730, 5736+1),
                               *range(5750, 5799+1),
                               *range(5910, 5912+1),
                               *range(5920, 5932+1),
                               *range(5940, 5990+1),
                               *range(5992, 5995+1)])
          ).then(43)

    .when(pl.col("sic").is_in([7000, 7213,
                               *range(5800, 5829+1),
                               *range(5890, 5899+1),
                               *range(7010, 7019+1),
                               *range(7040, 7049+1)])
          ).then(44)

    .when(pl.col("sic").is_in([6000,
                               *range(6010, 6036+1),
                               *range(6040, 6062+1),
                               *range(6080, 6082+1),
                               *range(6090, 6100+1),
                               *range(6110, 6113+1),
                               *range(6120, 6179+1),
                               *range(6190, 6199+1)])
          ).then(45)

    .when(pl.col("sic").is_in([6300, 6350, 6351, 6360, 6361,
                               *range(6310, 6331+1),
                               *range(6370, 6379+1),
                               *range(6390, 6411+1)])
          ).then(46)

    .when(pl.col("sic").is_in([6500, 6510, 6540, 6541, 6610, 6611,
                               *range(6512, 6515+1),
                               *range(6517, 6532+1),
                               *range(6550, 6553+1),
                               *range(6590, 6599+1)])
          ).then(47)
                           
    .when(pl.col("sic").is_in([6700, 6798, 6799,
                               *range(6200, 6299+1),
                               *range(6710, 6726+1),
                               *range(6730, 6733+1),
                               *range(6740, 6779+1),
                               *range(6790, 6795+1)])
          ).then(48)

    .when(pl.col("sic").is_in([4970, 4971, 4990, 4991,
                               *range(4950, 4961+1)])
          ).then(49)
    .otherwise(pl.lit(None)) 
        .alias("ff49")
    )

        return df1



def nyse_size_cutoffs(data):
    """
    Computes the 1st, 20th, 50th, and 80th percentiles of the market equity'me 'column based on NYSE stocks.
    The percentiles are calculated using the SAS percentile method 5.

    Parameters:
    - data: Input dataset containing stock information.

    Returns:
    - A DataFrame with columns 'eom' (end of month), 'n' (number of observations), and percentiles of 'me' column.
    """
    
    def sas_percentile_method_5(series, p):
        """
        Calculates the given percentile using the SAS 5th method, which is the default SAS method and was used in our SAS code.

        """
        n = len(series)
        rank = p * n
    
        if rank.is_integer():
             return (series[int(rank) - 1] + series[int(rank)]) / 2
        else:
            return series[int(rank)]

    
    # Filter the data for NYSE stocks based on specific criteria
    nyse_stocks = data.filter(
        (pl.col('crsp_exchcd') == 1) &         # NYSE exchange code
        (pl.col('obs_main') == 1) &            # Main observation flag
        (pl.col('exch_main') == 1) &           # Main exchange flag
        (pl.col('primary_sec') == 1) &         # Primary security flag
        (pl.col('common') == 1) &              # Common stock flag
        pl.col('me').is_not_null()             # Ensure market equity (me) is not null
    )

    # Sort the filtered data by 'eom' (end of month) for grouping
    nyse_stocks = nyse_stocks.sort('eom')

    # Group the data by 'eom' and calculate percentiles on the 'me' column
    grouped = nyse_stocks.group_by('eom').agg(
        [
            pl.col('me').sort().count().alias('n'),   # Count of 'me' values
            pl.col('me').sort().apply(lambda series: sas_percentile_method_5(series, 0.01)).alias('nyse_p1'),   # 1st percentile
            pl.col('me').sort().apply(lambda series: sas_percentile_method_5(series, 0.20)).alias('nyse_p20'),  # 20th percentile
            pl.col('me').sort().apply(lambda series: sas_percentile_method_5(series, 0.50)).alias('nyse_p50'),  # 50th percentile
            pl.col('me').sort().apply(lambda series: sas_percentile_method_5(series, 0.80)).alias('nyse_p80')   # 80th percentile
        ]
    )

    return grouped




def return_cutoffs(data, freq, crsp_only):
    """
    Computes various percentiles of the 'ret', 'ret_local', and 'ret_exc' columns based on specific criteria.
    Can be used for both monthly or daily data.

    Parameters:
    - data: Input dataset containing stock return information.
    - freq (str): The frequency at which to group data, either 'm' (monthly) or 'd' (daily).
    - crsp_only (int): A flag to filter data based on the "source_crsp" column. 1 means use CRSP data only, 0 means use all data.

    Returns:
    - A DataFrame with date, 'n' (number of observations), and percentiles of 'me' columns.
    """

    def sas_percentile_method_5(series, p):
        """
        Calculates the given percentile using the SAS 5th method, which is the default SAS method and was used in our SAS code.
        
        """
        n = len(series)
        rank = p * n
    
        if rank.is_integer():
            return (series[int(rank) - 1] + series[int(rank)]) / 2
        else:
            return series[int(rank)]

    # Filter data based on provided criteria. If 'crsp_only' is 1, filter for CRSP data only.
    if crsp_only == 1:
        base = (data.filter(
                    (pl.col("source_crsp") == 1) & 
                    (pl.col("common") == 1) & 
                    (pl.col("obs_main") == 1) & 
                    (pl.col("exch_main") == 1) & 
                    (pl.col("primary_sec") == 1) & 
                    (pl.col("excntry") != 'ZWE') & 
                    (pl.col("ret_exc").is_not_null()))
                .sort("eom"))
    else:
        base = (data.filter(
                    (pl.col("common") == 1) & 
                    (pl.col("obs_main") == 1) & 
                    (pl.col("exch_main") == 1) & 
                    (pl.col("primary_sec") == 1) & 
                    (pl.col("excntry") != 'ZWE') & 
                    (pl.col("ret_exc").is_not_null()))
                .sort("eom"))

    # If frequency is monthly, group by 'eom' (end of month)
    if freq == 'm':
        grouped = base.group_by('eom').agg(
            [
            pl.col("ret").sort().count().alias('n'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_0_1'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_1'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_99'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_99_9'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_local_0_1'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_local_1'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_local_99'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_local_99_9'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_exc_0_1'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_exc_1'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_exc_99'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_exc_99_9')
            ]
        )
    # If frequency is daily, group by year and month. We get one output per unique year and month combination considering all the relevant observations
    else:
        # Extract year and month from the 'date' column for grouping
        base = base.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month")
        ])

        grouped = base.group_by(["year", "month"]).agg(
            [
            pl.col("ret").sort().count().alias('n'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_0_1'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_1'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_99'),
            pl.col('ret').sort().apply(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_99_9'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_local_0_1'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_local_1'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_local_99'),
            pl.col('ret_local').sort().apply(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_local_99_9'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_exc_0_1'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_exc_1'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_exc_99'),
            pl.col('ret_exc').sort().apply(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_exc_99_9')
            ]
        )

    return grouped