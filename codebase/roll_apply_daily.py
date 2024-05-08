import wrds_Fernando
import polars as pl
import numpy as np
import time
from tqdm.notebook import tqdm
import datetime
from datetime import date
from math import sqrt, exp
import gc
import functools
import sys
def clear_locals_and_collect(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        func.__dict__.clear()
        gc.collect()
        return result
    return wrapper
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"{func.__name__} started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", flush=True)
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}", flush=True)
        # Calculate total seconds
        total_seconds = end_time - start_time
        # Calculate minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        print(f"{func.__name__} execution time: \t{minutes} minutes and {seconds:.5f} seconds", flush=True)
        return result
    return wrapper
def create_sublists(lst, k):
    sublists_1 = [lst[:i] for i in range(1, k)]
    sublists_2 = [lst[i:i+k] for i in range(len(lst) - k + 1)]
    return sublists_1 + sublists_2
def gen_date_partition(df, sublist):
    date_condition = (pl.col('aux_date') == sublist[0])
    for i in range(1, len(sublist)): date_condition |= (pl.col('aux_date') == sublist[i])
    df_partition = df.filter(date_condition)
    return df_partition
def gen_partitions(df, sublist_size):
    dataframes = []
    date_idx = datetime.datetime.today().month + datetime.datetime.today().year * 12
    sublists = create_sublists([i for i in range(23113, date_idx+1)], sublist_size)
    dataframes = [gen_date_partition(df, sublist) for sublist in sublists]
    return dataframes, sublists
def safe_div(num, den, name, mode = 1):
    if mode == 1:  return pl.when(pl.col(den) != 0).then(pl.col(num)/pl.col(den)).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
    if mode == 2:  return pl.when(pl.col(den) != 0).then(pl.col(num)/(pl.col(den).abs())).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
    if mode == 3:  return pl.when(pl.col(den) > 0).then(pl.col(num)/pl.col(den)).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
    if mode == 4:
        cond1 = pl.col('count') > 12
        cond2 = (pl.col(den).shift(12) > 0).over(['gvkey', 'curcd'])
        col_exp = (pl.col(num) / pl.col(den).shift(12)).over(['gvkey', 'curcd'])
        return pl.when(cond1 & cond2).then(col_exp).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
def regression_3vars_total(y, x1, x2, x3):
    den = (-((pl.col(x1).var()) * (pl.col(x2).var()) * (pl.col(x3).var())) + (pl.col(x1).var()) * (pl.cov(x2, x3))**2 + (pl.col(x2).var()) * (pl.cov(x1, x3))**2 - 2 * (pl.cov(x1, x3)) * (pl.cov(x2, x3)) * (pl.cov(x1, x2)) + (pl.col(x3).var()) * (pl.cov(x1, x2))**2)
    beta1 = ((pl.col(x2).var()) * (pl.cov(x1, x3)) * (pl.cov(x3, y)) - (pl.col(x2).var()) * ((pl.col(x3).var()) * (pl.cov(x1, y))) - (pl.cov(x1, x3)) * (pl.cov(x2, x3)) * (pl.cov(x2, y)) + (pl.cov(x2, x3))**2 * (pl.cov(x1, y)) - (pl.cov(x2, x3)) * (pl.cov(x1, x2)) * (pl.cov(x3, y)) + (pl.col(x3).var()) * (pl.cov(x1, x2)) * (pl.cov(x2, y))) / den
    beta2 = ((pl.col(x1).var()) * (pl.cov(x2, x3)) * (pl.cov(x3, y)) - (pl.col(x1).var()) * ((pl.col(x3).var()) * (pl.cov(x2, y))) + (pl.cov(x1, x3))**2 * (pl.cov(x2, y)) - (pl.cov(x1, x3)) * (pl.cov(x2, x3)) * (pl.cov(x1, y)) - (pl.cov(x1, x3)) * (pl.cov(x1, x2)) * (pl.cov(x3, y)) + (pl.col(x3).var()) * (pl.cov(x1, y)) * (pl.cov(x1, x2))) / den
    beta3 = (-((pl.col(x1).var()) * (pl.col(x2).var()) * (pl.cov(x3, y))) + (pl.col(x1).var()) * (pl.cov(x2, x3)) * (pl.cov(x2, y)) + (pl.col(x2).var()) * (pl.cov(x1, x3)) * (pl.cov(x1, y)) - (pl.cov(x1, x3)) * (pl.cov(x1, x2)) * (pl.cov(x2, y)) - (pl.cov(x2, x3)) * (pl.cov(x1, y)) * (pl.cov(x1, x2)) + (pl.cov(x1, x2))**2 * (pl.cov(x3, y))) / den
    return beta1, beta2, beta3
def regression_4vars_total(y, x1, x2, x3, x4):
    b1 = (pl.cov(x1,x2) * pl.cov(x3,x4)**2 * pl.cov(y,x2) - pl.cov(x1,x2) * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.cov(y,x3) - pl.cov(x3,x4)**2 * pl.cov(y,x1) * pl.var(x2) + pl.cov(x1,x4) * pl.cov(x3,x4) * pl.cov(y,x3) * pl.var(x2) - pl.cov(x2,x4)**2 * pl.cov(y,x1) * pl.var(x3) + pl.cov(x1,x4) * pl.cov(x2,x4) * pl.cov(y,x2) * pl.var(x3) + pl.cov(x1,x2) * pl.cov(x2,x4) * pl.cov(y,x4) * pl.var(x3) -pl.cov(x1,x4) * pl.cov(y,x4) * pl.var(x2) * pl.var(x3) - pl.cov(x1,x2) * pl.cov(y,x2) * pl.var(x3) * pl.var(x4) + pl.cov(y,x1) * pl.var(x2) * pl.var(x3) * pl.var(x4) + pl.cov(x2,x3)**2 * (pl.cov(x1,x4) * pl.cov(y,x4) - pl.cov(y,x1) * pl.var(x4)) - pl.cov(x2,x3) * (-2  * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.cov(y,x1) + pl.cov(x1,x4) * pl.cov(x3,x4) * pl.cov(y,x2) + pl.cov(x1,x4) * pl.cov(x2,x4) * pl.cov(y,x3) + pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(y,x4) + pl.cov(x1,x2) * pl.cov(x3,x4) * pl.cov(y,x4) - (pl.cov(x1,x3) * pl.cov(y,x2) + pl.cov(x1,x2) * pl.cov(y,x3)) * pl.var(x4)) + pl.cov(x1,x3) * (-pl.cov(x2,x4) * pl.cov(x3,x4) * pl.cov(y,x2) + pl.cov(x2,x4)**2 * pl.cov(y,x3) + pl.cov(x3,x4) * pl.cov(y,x4) * pl.var(x2) - pl.cov(y,x3) * pl.var(x2) * pl.var(x4)))/(pl.cov(x1,x3)**2 * pl.cov(x2,x4)**2 - 2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) + pl.cov(x1,x2)**2 * pl.cov(x3,x4)**2 + 2 * pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.var(x1) - pl.cov(x3,x4)**2 * pl.var(x1) * pl.var(x2) - pl.cov(x2,x4)**2 * pl.var(x1) * pl.var(x3) + 2 * pl.cov(x1,x4) * (-pl.cov(x1,x3) * pl.cov(x2,x3) * pl.cov(x2,x4) - pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(x3,x4) + pl.cov(x1,x3) * pl.cov(x3,x4) * pl.var(x2) + pl.cov(x1,x2) * pl.cov(x2,x4) * pl.var(x3)) + pl.cov(x1,x4)**2 * (pl.cov(x2,x3)**2 - pl.var(x2) * pl.var(x3)) - (-2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x3) + pl.cov(x2,x3)**2 * pl.var(x1) + pl.cov(x1,x3)**2 * pl.var(x2) + pl.cov(x1,x2)**2 * pl.var(x3) - pl.var(x1) * pl.var(x2) * pl.var(x3)) * pl.var(x4))
    b2 =  (-pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.cov(y,x1) + pl.cov(x1,x2) * pl.cov(x3,x4)**2 * pl.cov(y,x1) + pl.cov(x1,x3)**2 * pl.cov(x2,x4) * pl.cov(y,x4) - pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x3,x4) * pl.cov(y,x4) - pl.cov(x1,x4) * (pl.cov(x2,x3) * pl.cov(x3,x4) * pl.cov(y,x1) - 2 * pl.cov(x1,x3) * pl.cov(x3,x4) * pl.cov(y,x2) + pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(y,x3) + pl.cov(x1,x2) * pl.cov(x3,x4) * pl.cov(y,x3) + pl.cov(x1,x3) * pl.cov(x2,x3) * pl.cov(y,x4)) - pl.cov(x3,x4)**2 * pl.cov(y,x2) * pl.var(x1) + pl.cov(x2,x4) * pl.cov(x3,x4) * pl.cov(y,x3) * pl.var(x1) + pl.cov(x2,x3) * pl.cov(x3,x4) * pl.cov(y,x4) * pl.var(x1) + pl.cov(x1,x4) * (pl.cov(x2,x4) * pl.cov(y,x1) + pl.cov(x1,x2) * pl.cov(y,x4)) * pl.var(x3) - pl.cov(x2,x4) * pl.cov(y,x4) * pl.var(x1) * pl.var(x3) + pl.cov(x1,x4)**2 * (pl.cov(x2,x3) * pl.cov(y,x3) - pl.cov(y,x2) * pl.var(x3)) + (pl.cov(x1,x3) * pl.cov(x2,x3) * pl.cov(y,x1) - pl.cov(x1,x3)**2 * pl.cov(y,x2) + pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(y,x3) - pl.cov(x2,x3) * pl.cov(y,x3) * pl.var(x1) - pl.cov(x1,x2) * pl.cov(y,x1) * pl.var(x3) + pl.cov(y,x2) * pl.var(x1) * pl.var(x3)) * pl.var(x4))/(pl.cov(x1,x3)**2 * pl.cov(x2,x4)**2 - 2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) + pl.cov(x1,x2)**2 * pl.cov(x3,x4)**2 + 2 * pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.var(x1) - pl.cov(x3,x4)**2 * pl.var(x1) * pl.var(x2) - pl.cov(x2,x4)**2 * pl.var(x1) * pl.var(x3) + 2 * pl.cov(x1,x4) * (-pl.cov(x1,x3) * pl.cov(x2,x3) * pl.cov(x2,x4) - pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(x3,x4) + pl.cov(x1,x3) * pl.cov(x3,x4) * pl.var(x2) + pl.cov(x1,x2) * pl.cov(x2,x4) * pl.var(x3)) + pl.cov(x1,x4)**2 * (pl.cov(x2,x3)**2 - pl.var(x2) * pl.var(x3)) - (-2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x3) + pl.cov(x2,x3)**2 * pl.var(x1) + pl.cov(x1,x3)**2 * pl.var(x2) + pl.cov(x1,x2)**2 * pl.var(x3) - pl.var(x1) * pl.var(x2) * pl.var(x3)) * pl.var(x4))
    b3 = (pl.cov(x1,x3) * pl.cov(x2,x4)**2 * pl.cov(y,x1) - pl.cov(x1,x2) * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.cov(y,x1) - pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(y,x4) + pl.cov(x1,x2)**2 * pl.cov(x3,x4) * pl.cov(y,x4) - pl.cov(x1,x4) * (pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(y,x1) + pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(y,x2) + pl.cov(x1,x2) * pl.cov(x3,x4) * pl.cov(y,x2) - 2 * pl.cov(x1,x2) * pl.cov(x2,x4) * pl.cov(y,x3) + pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(y,x4)) + pl.cov(x2,x4) * pl.cov(x3,x4) * pl.cov(y,x2) * pl.var(x1) - pl.cov(x2,x4)**2 * pl.cov(y,x3) * pl.var(x1) + pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(y,x4) * pl.var(x1) + pl.cov(x1,x4) * (pl.cov(x3,x4) * pl.cov(y,x1) + pl.cov(x1,x3) * pl.cov(y,x4)) * pl.var(x2) - pl.cov(x3,x4) * pl.cov(y,x4) * pl.var(x1) * pl.var(x2) + pl.cov(x1,x4)**2 * (pl.cov(x2,x3) * pl.cov(y,x2) - pl.cov(y,x3) * pl.var(x2)) + (pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(y,x1) + pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(y,x2) - pl.cov(x1,x2)**2 * pl.cov(y,x3) - pl.cov(x2,x3) * pl.cov(y,x2) * pl.var(x1) - pl.cov(x1,x3) * pl.cov(y,x1) * pl.var(x2) + pl.cov(y,x3) * pl.var(x1) * pl.var(x2)) * pl.var(x4))/(pl.cov(x1,x3)**2 * pl.cov(x2,x4)**2 - 2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) + pl.cov(x1,x2)**2 * pl.cov(x3,x4)**2 + 2 * pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.var(x1) - pl.cov(x3,x4)**2 * pl.var(x1) * pl.var(x2) - pl.cov(x2,x4)**2 * pl.var(x1) * pl.var(x3) + 2 * pl.cov(x1,x4) * (-pl.cov(x1,x3) * pl.cov(x2,x3) * pl.cov(x2,x4) - pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(x3,x4) + pl.cov(x1,x3) * pl.cov(x3,x4) * pl.var(x2) + pl.cov(x1,x2) * pl.cov(x2,x4) * pl.var(x3)) + pl.cov(x1,x4)**2 * (pl.cov(x2,x3)**2 - pl.var(x2) * pl.var(x3)) - (-2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x3) + pl.cov(x2,x3)**2 * pl.var(x1) + pl.cov(x1,x3)**2 * pl.var(x2) + pl.cov(x1,x2)**2 * pl.var(x3) - pl.var(x1) * pl.var(x2) * pl.var(x3)) * pl.var(x4))
    b4 = (-pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(x3,x4) * pl.cov(y,x1) + pl.cov(x1,x2)**2 * pl.cov(x3,x4) * pl.cov(y,x3) + pl.cov(x2,x3) * pl.cov(x3,x4) * pl.cov(y,x2) * pl.var(x1) + pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(y,x3) * pl.var(x1) - pl.cov(x2,x3)**2 * pl.cov(y,x4) * pl.var(x1) - pl.cov(x3,x4) * pl.cov(y,x3) * pl.var(x1) * pl.var(x2) - pl.cov(x1,x3) * (pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(y,x1) + pl.cov(x1,x2) * pl.cov(x3,x4) * pl.cov(y,x2) + pl.cov(x1,x2) * pl.cov(x2,x4) * pl.cov(y,x3) - 2 * pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(y,x4) - pl.cov(x3,x4) * pl.cov(y,x1) * pl.var(x2)) + pl.cov(x1,x3)**2 * (pl.cov(x2,x4) * pl.cov(y,x2) - pl.cov(y,x4) * pl.var(x2)) + pl.cov(x1,x2) * pl.cov(x2,x4) * pl.cov(y,x1) * pl.var(x3) - pl.cov(x1,x2)**2 * pl.cov(y,x4) * pl.var(x3) - pl.cov(x2,x4) * pl.cov(y,x2) * pl.var(x1) * pl.var(x3) + pl.cov(y,x4) * pl.var(x1) * pl.var(x2) * pl.var(x3) + pl.cov(x1,x4) * (pl.cov(x2,x3)**2 * pl.cov(y,x1) - pl.cov(x2,x3) * (pl.cov(x1,x3) * pl.cov(y,x2) + pl.cov(x1,x2) * pl.cov(y,x3)) + pl.cov(x1,x3) * pl.cov(y,x3) * pl.var(x2) + pl.cov(x1,x2) * pl.cov(y,x2) * pl.var(x3) - pl.cov(y,x1) * pl.var(x2) * pl.var(x3)))/(pl.cov(x1,x3)**2 * pl.cov(x2,x4)**2 - 2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) + pl.cov(x1,x2)**2 * pl.cov(x3,x4)**2 + 2 * pl.cov(x2,x3) * pl.cov(x2,x4) * pl.cov(x3,x4) * pl.var(x1) - pl.cov(x3,x4)**2 * pl.var(x1) * pl.var(x2) - pl.cov(x2,x4)**2 * pl.var(x1) * pl.var(x3) + 2 * pl.cov(x1,x4) * (-pl.cov(x1,x3) * pl.cov(x2,x3) * pl.cov(x2,x4) - pl.cov(x1,x2) * pl.cov(x2,x3) * pl.cov(x3,x4) + pl.cov(x1,x3) * pl.cov(x3,x4) * pl.var(x2) + pl.cov(x1,x2) * pl.cov(x2,x4) * pl.var(x3)) + pl.cov(x1,x4)**2 * (pl.cov(x2,x3)**2 - pl.var(x2) * pl.var(x3)) - (-2 * pl.cov(x1,x2) * pl.cov(x1,x3) * pl.cov(x2,x3) + pl.cov(x2,x3)**2 * pl.var(x1) + pl.cov(x1,x3)**2 * pl.var(x2) + pl.cov(x1,x2)**2 * pl.var(x3) - pl.var(x1) * pl.var(x2) * pl.var(x3)) * pl.var(x4))
    return b1, b2, b3, b4
@measure_time
@clear_locals_and_collect
def roll_apply_daily(output_path_prefix,__min, base_data_path, sfx, stats):
    print(f'Processing stat {stats[0]}. Time period: {sfx[1:]}')
    parameter_mapping = {"_21d": 1,"_126d": 6,"_252d": 12,"_1260d": 60}
    columns_base_data = ['id','aux_date','ret_exc', 'ret', 'dolvol_d','prc_adj', 'mktrf','smb_ff','hml','smb_hxz','roe','inv', 'shares', 'tvol']
    base_data = pl.scan_ipc(base_data_path).with_columns((12*pl.col('eom').dt.year() + pl.col('eom').dt.month()).alias('aux_date')).filter((pl.col('ret_exc').is_not_null()) & (pl.col('zero_obs') < 10))#.select(columns_base_data)
    partitions, sublists = gen_partitions(base_data, parameter_mapping[sfx])
    #partitions = [i.with_columns(pl.count('ret_exc').over('id').alias('n')).filter(pl.col('n') >= __min).select(columns_base_data) for i in partitions]
    partitions = [i.with_columns(pl.count('ret_exc').over('id').alias('n')).filter(pl.col('n') >= __min) for i in partitions]
    #Filtering groups doesn't really affect performance (~1.7% time improvement)
    if 'rvol' in stats: rvol = pl.concat([partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), pl.col('ret_exc').std().alias(f'rvol{sfx}'), pl.count('ret_exc').alias('n')]).filter(pl.col('n') >= __min).select(['id','aux_date', f'rvol{sfx}']) for i in range(len(partitions))])
    if 'rmax' in stats: rmax = pl.concat([partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), (pl.col('ret').top_k(5)).alias(f'rmax5{sfx}'), pl.col('ret').max().alias(f'rmax1{sfx}')]).with_columns(pl.col(f'rmax5{sfx}').list.mean()) for i in range(len(partitions))])
    if 'skew' in stats: skew = pl.concat([partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), (pl.col('ret_exc').skew(bias = False)).alias(f'rskew{sfx}'), pl.count('ret_exc').alias('n')]).filter(pl.col('n') >= __min).select(['id','aux_date', f'rskew{sfx}']) for i in range(len(partitions))])
    if 'prc_to_high' in stats: prc_to_high = pl.concat([partitions[i].sort(['id','aux_date']).group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), (pl.col('prc_adj').last()/ pl.col('prc_adj').max()).alias(f'prc_highprc{sfx}'), pl.count('prc_adj').alias('n')]).filter(pl.col('n') >= __min).select(['id','aux_date', f'prc_highprc{sfx}']) for i in range(len(partitions))])
    if 'turnover' in stats:
        c1 = (pl.col('tvol').is_not_null())
        col_exp = pl.when(pl.col('shares') != 0).then(pl.col('tvol')/(pl.col('shares')*1e6)).otherwise(pl.lit(None).cast(pl.Float64))
        col_exp2 = pl.col('turnover_d').list.mean()
        col_exp3 = pl.when(pl.col('turnover_d').list.mean() != 0).then(pl.col('turnover_d').list.std()/pl.col('turnover_d').list.mean()).otherwise(pl.lit(None).cast(pl.Float64))
        base_data_turnover = pl.scan_ipc(base_data_path).with_columns((12*pl.col('eom').dt.year() + pl.col('eom').dt.month()).alias('aux_date')).filter(c1)
        base_data_turnover.collect().write_ipc('aux.ft')
        base_data_turnover = pl.scan_ipc('aux.ft')
        partitions_turnover, sublists_turnover = gen_partitions(base_data_turnover, parameter_mapping[sfx])
        turnover = [partitions_turnover[i].group_by(['id']).agg([pl.lit(sublists_turnover[i][-1]).alias('aux_date'), col_exp.alias('turnover_d')]).with_columns(col_exp2.alias(f'turnover{sfx}'), col_exp3.alias(f'turnover_var{sfx}'), (pl.col('turnover_d').list.len()).alias('n')) for i in range(len(partitions_turnover))]
        turnover = pl.concat(turnover).filter(pl.col('n') >= __min).select(['id','aux_date',f'turnover{sfx}',f'turnover_var{sfx}'])
    if 'ami' in stats:
        aux = pl.when(pl.col('dolvol_d') == 0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('dolvol_d'))
        ami = pl.concat([partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), (pl.col('ret').abs()/aux * 1e6).mean().alias(f'ami{sfx}'), pl.count('dolvol_d').alias('n')]).filter(pl.col('n') >= __min).select(['id','aux_date', f'ami{sfx}']) for i in range(len(partitions))])
    if 'capm' in stats:
        beta_col = (pl.cov('ret_exc', 'mktrf')/pl.var('mktrf'))
        residual_col = (pl.col('ret_exc')- (pl.col('mktrf')*(beta_col * pl.lit(1.))))
        capm = pl.concat([partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), pl.count('ret_exc').alias('n'), beta_col.cast(pl.Float64).alias(f'beta{sfx}'),residual_col.cast(pl.Float64).alias(f'ivol_capm{sfx}')]).filter(pl.col('n')>=__min) for i in range(len(partitions))])
        capm = capm.with_columns(pl.col(f'ivol_capm{sfx}').list.eval(pl.element().std()).alias(f"ivol_capm{sfx}")).select(['id','aux_date', f'beta{sfx}', f'ivol_capm{sfx}']).explode(f'ivol_capm{sfx}')
    if 'capm_ext' in stats:
        beta_col = (pl.cov('ret_exc', 'mktrf')/pl.var('mktrf'))
        alpha_col = pl.mean('ret_exc') - beta_col * pl.mean('mktrf')
        residual_col = (pl.col('ret_exc') - (alpha_col*pl.lit(1.) + pl.col('mktrf') * beta_col))
        exp_coskew1 = (pl.col('residuals').cast(pl.Float64) * (pl.col('aux_mkt')**2).cast(pl.Float64)).mean()
        exp_coskew2 = (pl.col('residuals').cast(pl.Float64)**2).mean()**0.5 * (pl.col('aux_mkt').cast(pl.Float64)**2).mean()
        capm_ext = [partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), pl.count('ret_exc').alias('n'), beta_col.cast(pl.Float64).alias(f'beta_capm{sfx}'), residual_col.cast(pl.Float64).alias('residuals'), (pl.col('mktrf') - pl.col('mktrf').mean()*pl.lit(1.)).alias('aux_mkt')]).filter(pl.col('n')>=__min) for i in range(len(partitions))]
        capm_ext = [i.with_columns(pl.col('residuals').list.eval(pl.element().skew(bias = False)).alias(f'iskew_capm{sfx}')).explode(f'iskew_capm{sfx}').explode(['residuals', 'aux_mkt']).group_by(['id']).agg([pl.col('aux_date').last(), pl.col(f'beta_capm{sfx}').last(), pl.col(f'iskew_capm{sfx}').last(), pl.std('residuals').alias(f'ivol_capm{sfx}'), exp_coskew1.alias('aux1'), exp_coskew2.alias('aux2')]) for i in capm_ext]
        capm_ext = pl.concat([i.with_columns((pl.col('aux1')/pl.col('aux2')).alias(f'coskew{sfx}')).select(['id', 'aux_date', f'beta_capm{sfx}', f'ivol_capm{sfx}', f'iskew_capm{sfx}', f'coskew{sfx}']) for i in capm_ext])
    if 'ff3' in stats:
        b1_col, b2_col, b3_col = regression_3vars_total('ret_exc', 'mktrf', 'smb_ff', 'hml')
        ff3_aux = [i.filter(pl.col('ret_exc').is_not_null() & pl.col('smb_ff').is_not_null() & pl.col('hml').is_not_null()) for i in partitions]
        res_exp = pl.col('ret_exc').cast(pl.Float64) - ( ((b1_col * pl.lit(1.)) * pl.col('mktrf')).cast(pl.Float64) + ((b2_col * pl.lit(1.)) * pl.col('smb_ff')).cast(pl.Float64) + ((b3_col * pl.lit(1.)) * pl.col('hml')).cast(pl.Float64) )
        aux_1 = [pl.count('ret_exc').alias('n'), res_exp.cast(pl.Float64).alias('residuals')]
        aux_2 = [pl.col('residuals').list.eval(pl.element().std(ddof = 3)).alias(f'ivol_ff3{sfx}'),pl.col('residuals').list.eval(pl.element().skew(bias = False)).alias(f'iskew_ff3{sfx}')]
        ff3 = pl.concat([ff3_aux[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), *aux_1]).filter(pl.col('n')>=__min).with_columns(aux_2).select(['id','aux_date',f'ivol_ff3{sfx}', f'iskew_ff3{sfx}']) for i in range(len(ff3_aux))]).explode([f'ivol_ff3{sfx}', f'iskew_ff3{sfx}'])
    if 'hxz4' in stats:
        b1_col, b2_col, b3_col, b4_col = regression_4vars_total('ret_exc', 'mktrf', 'smb_hxz', 'roe', 'inv')
        c1 = pl.col('ret_exc').is_not_null()
        c2 = pl.col('smb_hxz').is_not_null()
        c3 = pl.col('roe').is_not_null()
        c4 = pl.col('inv').is_not_null()
        res_exp = pl.col('ret_exc').cast(pl.Float64) -(((b1_col * pl.lit(1.)) * pl.col('mktrf')).cast(pl.Float64) + ((b2_col * pl.lit(1.)) * pl.col('smb_hxz')).cast(pl.Float64) + ((b3_col * pl.lit(1.)) * pl.col('roe')).cast(pl.Float64) + ((b4_col * pl.lit(1.)) * pl.col('inv')).cast(pl.Float64))
        aux_1 = [pl.count('ret_exc').alias('n'), res_exp.cast(pl.Float64).alias('residuals')]
        aux_2 = [pl.col('residuals').list.eval(pl.element().std(ddof = 4)).alias(f'ivol_hxz4{sfx}'),pl.col('residuals').list.eval(pl.element().skew(bias = False)).alias(f'iskew_hxz4{sfx}')]
        hxz4 = [i.filter(c1 & c2 & c3 & c4) for i in partitions]
        hxz4 = pl.concat([partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), *aux_1]).filter(pl.col('n')>=__min).with_columns(aux_2).select(['id', 'aux_date', f'ivol_hxz4{sfx}', f'iskew_hxz4{sfx}']) for i in range(len(partitions))]).explode([f'ivol_hxz4{sfx}', f'iskew_hxz4{sfx}'])
    if 'downbeta' in stats:
        beta_col = (pl.cov('ret_exc', 'mktrf')/pl.var('mktrf')).cast(pl.Float64)
        c1 = pl.col('ret_exc').is_not_null()
        c2 = pl.col('mktrf') < 0
        downbeta = pl.concat([partitions[i].filter(c1 & c2).group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), pl.count('ret_exc').alias('n'), beta_col.alias(f'betadown{sfx}')]).filter(pl.col('n')>=__min/2).select(['id', 'aux_date', f'betadown{sfx}']) for i in range(len(partitions))])
    if 'dolvol' in stats:
        base_data_dolvol = pl.scan_ipc(base_data_path, memory_map = False).with_columns((12*pl.col('eom').dt.year() + pl.col('eom').dt.month()).alias('aux_date')).filter(pl.col('dolvol_d').is_not_null())
        base_data_dolvol.collect().write_ipc('aux.ft')
        base_data_dolvol = pl.scan_ipc('aux.ft')
        partitions_dolvol, sublists_dolvol = gen_partitions(base_data_dolvol, parameter_mapping[sfx])
        dolvol = pl.concat([partitions_dolvol[i].group_by(['id']).agg([pl.lit(sublists_dolvol[i][-1]).alias('aux_date'), pl.count('dolvol_d').alias('n'), pl.col('dolvol_d').mean().alias(f'dolvol{sfx}'), pl.col('dolvol_d').std().alias(f'dolvol_var{sfx}')]).filter(pl.col('n') >= __min) for i in range(len(partitions_dolvol))])
        dolvol = dolvol.with_columns(safe_div(f'dolvol_var{sfx}', f'dolvol{sfx}', f'dolvol_var{sfx}')).select(['id', 'aux_date', f'dolvol{sfx}', f'dolvol_var{sfx}'])
    if 'mktcorr' in stats:
        c1 = pl.col('ret_exc_3l').is_not_null()
        c2 = pl.col('zero_obs') < 10
        c3 = pl.col('n1')>=__min
        c4 = pl.col('n2')>=__min
        mktcorr_data = pl.scan_ipc('corr_data.ft').filter(c1 & c2).rename({'eom': 'aux_date'}).with_columns(aux_date = 12*pl.col('aux_date').dt.year() + pl.col('aux_date').dt.month())
        mktcorr_data = mktcorr_data.select(['id', 'aux_date', 'ret_exc_3l','mkt_exc_3l'])
        mktcorr_partitions, sublists_mktcorr = gen_partitions(mktcorr_data, parameter_mapping[sfx])
        mktcorr = pl.concat([mktcorr_partitions[i].group_by(['id']).agg([pl.lit(sublists_mktcorr[i][-1]).alias('aux_date'), pl.count('ret_exc_3l').alias('n1'), pl.count('mkt_exc_3l').alias('n2'), pl.corr('ret_exc_3l', 'mkt_exc_3l').alias(f'corr{sfx}')]).filter(c3 & c4).select('id','aux_date', f'corr{sfx}') for i in range(len(mktcorr_partitions))])
    if 'zero_trades' in stats:
        base_data_zero_trades = pl.scan_ipc(base_data_path).with_columns((12*pl.col('eom').dt.year() + pl.col('eom').dt.month()).alias('aux_date')).filter(pl.col('ret_exc').is_not_null())#tvol
        base_data_zero_trades.collect().write_ipc('aux.ft')
        base_data_zero_trades = pl.scan_ipc('aux.ft')
        partitions_zero_trades, sublists_zero_trades = gen_partitions(base_data_zero_trades, parameter_mapping[sfx])
        aux_1 = (pl.col('tvol') == 0).mean() * 21
        aux_2 = pl.when(pl.col('shares') != 0).then(pl.col('tvol')/(pl.col('shares')*1e6)).otherwise(pl.lit(None).cast(pl.Float64))
        c2 = (pl.col('n')>=__min) & (pl.col('zero_trades').is_not_null()) & (pl.col('turnover').is_not_null())
        zero_trades = [partitions_zero_trades[i].group_by(['id']).agg([pl.lit(sublists_zero_trades[i][-1]).alias('aux_date'), aux_1.alias('zero_trades'), aux_2.alias('turnover'), pl.count('tvol').alias('n')]).filter(c2).with_columns(pl.col('turnover').list.drop_nulls()).with_columns(pl.col('turnover').list.mean()) for i in range(len(partitions_zero_trades))]
        aux_3 = (pl.col('turnover').rank(descending=True, method='average')/pl.count('turnover')).over('aux_date')
        aux_4 = (aux_3 / 100) + pl.col('zero_trades')
        zero_trades = pl.concat([i.with_columns(aux_4.alias(f'zero_trades{sfx}')) for i in zero_trades]).select(['id', 'aux_date', f'zero_trades{sfx}'])
    if 'mktvol' in stats:  mktvol = pl.concat([partitions[i].group_by(['id']).agg([pl.lit(sublists[i][-1]).alias('aux_date'), pl.col('mktrf').std().alias(f'__mktvol{sfx}'), pl.count('ret_exc').alias('n')]).filter(pl.col('n') >= __min).select(['id','aux_date', f'__mktvol{sfx}']) for i in range(len(partitions))])
    if 'dimsonbeta' in stats:
        dimson_data = pl.scan_ipc(base_data_path).filter((pl.col('ret_exc').is_not_null()) & (pl.col('zero_obs') < 10)).select(['excntry','id','eom','date','ret_exc','mktrf'])
        lead_lag = pl.scan_ipc('mkt_lead_lag.ft').select(['excntry','date', 'mktrf_lg1', 'mktrf_ld1'])
        dimson_data = dimson_data.join(lead_lag, on = ['excntry', 'date']).filter(pl.col('mktrf_lg1').is_not_null() & pl.col('mktrf_ld1').is_not_null()).with_columns(n = pl.len().over(['id','eom'])).filter(pl.col('n') >= (__min - 1))
        dimson_data = dimson_data.select(['id', (12*pl.col('eom').dt.year() + pl.col('eom').dt.month()).alias('aux_date'), 'ret_exc', 'mktrf', 'mktrf_lg1', 'mktrf_ld1'])
        dimson_data.collect().write_ipc('dimson_data.ft')
        dimson_data = pl.scan_ipc('dimson_data.ft')
        partitions_dimson, sublists_dimson = gen_partitions(dimson_data, parameter_mapping[sfx])
        partitions_dimson = [i.with_columns(pl.count('ret_exc').over('id').alias('n')).filter(pl.col('n') >= __min).select(['id', 'aux_date', 'ret_exc', 'mktrf', 'mktrf_lg1', 'mktrf_ld1']) for i in partitions_dimson]
        b1_col, b2_col, b3_col = regression_3vars_total('ret_exc', 'mktrf', 'mktrf_ld1', 'mktrf_lg1')
        dimsonbeta = pl.concat([partitions_dimson[i].group_by(['id']).agg([pl.lit(sublists_dimson[i][-1]).alias('aux_date'), (b1_col + b2_col + b3_col).alias(f'beta_dimson{sfx}')]) for i in range(len(partitions_dimson))])
    result_df = locals()[stats[0]]
    result_df.collect().write_ipc(output_path_prefix + sfx + '_' + stats[0] + '.ft')

if __name__ == "__main__":
    arg1 = int(sys.argv[1])
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    roll_apply_daily('__roll',arg1, 'dsf1.ft', arg2, [arg3])
