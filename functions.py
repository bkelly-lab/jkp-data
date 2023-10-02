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