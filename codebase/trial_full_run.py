import wrds_Fernando
import polars as pl
import numpy as np
import time
import datetime
from datetime import date
from math import sqrt, exp
import gc
import subprocess
import os
from polars import col
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
@measure_time
def prepare_comp_sf(freq):
    populate_own('Raw data/__firm_shares1.ft', 'gvkey', 'datadate', 'ddate')
    gen_comp_dsf()
    if freq == 'both':
        process_comp_sf1('d')
        process_comp_sf1('m')
    else: process_comp_sf1(freq)
@measure_time
def populate_own(inset, idvar, datevar, datename):
    inset = pl.scan_ipc(inset)
    inset = inset.unique([idvar, datevar])
    inset_flags = inset.select([col(idvar), col(datevar).alias(datename), pl.lit(1).alias('flag_orig')])
    inset = inset.with_columns(col(datevar).alias(datename))
    inset = inset.sort([idvar, datevar]).with_columns(following = (col('datadate').shift(-1)).over(idvar))
    inset = inset.with_columns(col(datevar).dt.offset_by('12mo').dt.month_end().alias('forward_max'))
    inset = inset.with_columns(pl.min_horizontal('following','forward_max').dt.offset_by('-1d').alias('n')).drop(['following','forward_max'])
    inset = inset.with_columns(pl.date_ranges('ddate', 'n')).explode('ddate')
    inset = inset.join(inset_flags, on = [idvar, datename], how = 'left')
    inset = inset.filter(~((col('flag_orig').is_null()) & (col(datename) > col('n')))).drop(['n','flag_orig'])
    inset = inset.select(['ddate','gvkey','datadate', 'csho_fund', 'ajex_fund']).sort(['gvkey','datadate'])
    inset.collect().write_ipc('__firm_shares2.ft')
def compustat_fx():
    __fx1 = pl.scan_ipc('Raw data/__fx1.ft')
    aux = pl.DataFrame({'curcdd': 'USD','datadate': '1950-01-01','fx': 1.0}).with_columns(col('datadate').str.to_date('%Y-%m-%d')).lazy()
    __fx1 = pl.concat([aux, __fx1]).sort(['curcdd', 'datadate'])
    __fx1 = __fx1.with_columns(aux = (col('datadate').shift(-1).over('curcdd')).dt.offset_by(f'-1d'))
    __fx1 = __fx1.with_columns(date_range = pl.coalesce([pl.date_ranges(start='datadate', end='aux', interval='1d'), 'datadate']))
    __fx1 = __fx1.select([col('date_range').alias('datadate'),'curcdd', 'fx']).explode('datadate')
    __fx1 = __fx1.unique(['curcdd','datadate']).sort(['curcdd', 'datadate'])
    return __fx1.collect()
@measure_time
def gen_comp_dsf():
    __comp_dsf = pl.scan_ipc('Raw data/__comp_dsf_na.ft')
    aux = pl.scan_ipc('__firm_shares2.ft').select(['ddate','gvkey','csho_fund','ajex_fund'])
    __comp_dsf = __comp_dsf.join(aux, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'ddate'])
    __comp_dsf = __comp_dsf.with_columns(col('cshoc').fill_null(col('csho_fund')*col('ajex_fund')/col('ajexdi')))
    __comp_dsf = __comp_dsf.with_columns(adj_trd_vol_NASDAQ('datadate', 'cshtrd', 'exchg', 14)).drop(['csho_fund','ajex_fund'])
    __comp_dsf_global = pl.scan_ipc('Raw data/__comp_dsf_global.ft')
    __comp_dsf = pl.concat([__comp_dsf, __comp_dsf_global.select(__comp_dsf.columns)])
    fx = compustat_fx().lazy()
    fx_div = fx.clone().with_columns(col('fx').alias('fx_div')).drop('fx').lazy()
    __comp_dsf =  __comp_dsf.join(fx, how = 'left', on = ['datadate', 'curcdd']) #fx datadate in our terminology is the same as date in the SAS code
    __comp_dsf = __comp_dsf.join(fx_div, how = 'left', left_on = ['datadate', 'curcddv'], right_on = ['datadate', 'curcdd'])
    __comp_dsf = __comp_dsf.with_columns(prc = col('prc_local')*col('fx'),prc_high = col('prc_high_lcl')*col('fx'),prc_low = col('prc_low_lcl')*col('fx'),ri = col('ri_local')*col('fx'))
    __comp_dsf = __comp_dsf.with_columns(me = col('prc')*col('cshoc'),dolvol = col('prc')*col('cshtrd'),div_tot = (col('div').fill_null(strategy='zero'))*col('fx_div'),div_cash = (col('divd').fill_null(strategy='zero'))*col('fx_div'),div_spc = (col('divsp').fill_null(strategy='zero'))*col('fx_div'),eom = col('datadate').dt.month_end()).drop(['div', 'divd', 'divsp', 'fx_div', 'curcddv', 'prc_high_lcl', 'prc_low_lcl'])
    __comp_dsf.collect(streaming=True).write_ipc('__comp_dsf.ft')
def adj_trd_vol_NASDAQ(datevar, col_to_adjust, exchg_var, exchg_val):
    c1 = col(exchg_var) == exchg_val
    c2 = col(datevar) < pl.datetime(2001, 2, 1)
    c3 = col(datevar) <= pl.datetime(2001, 12, 31)
    c4 = col(datevar) < pl.datetime(2003, 12, 31)
    adj_trd_vol = (pl.when(c1 & c2).then(col(col_to_adjust) / 2)
                        .when(c1 & c3).then(col(col_to_adjust) / 1.8)
                        .when(c1 & c4).then(col(col_to_adjust) / 1.6)
                        .otherwise(col(col_to_adjust))).alias(col_to_adjust)
    return adj_trd_vol
@measure_time
def gen_comp_msf():
    col_names_aux = ['prc_highm','prc_lowm','div_totm','div_cashm','div_spcm','cshtrm']
    set_aux_cols = [pl.when(col('ajexdi') != 0).then(pl.max_horizontal('prc', 'prc_high') / col('ajexdi')).otherwise(pl.lit(None).cast(pl.Float64)).alias('aux1'),
                    pl.when(col('ajexdi') != 0).then(pl.max_horizontal('prc', 'prc_low')  / col('ajexdi')).otherwise(pl.lit(None).cast(pl.Float64)).alias('aux2')]
    set_aux_cols += [safe_div(var, 'ajexdi', f'aux{i+1}') for var, i in zip(['div_tot', 'div_cash', 'div_spc', 'cshtrd'], range(2, 6))]
    set_aux_cols2 = [col('aux1').max().over(['gvkey', 'iid', 'eom']), col('aux2').min().over(['gvkey', 'iid', 'eom'])]
    set_aux_cols2 += [col(f'aux{i+1}').sum().over(['gvkey', 'iid', 'eom']) for i in range(2, 6)]
    set_aux_cols3 = [pl.sum('dolvol').over(['gvkey', 'iid', 'eom']).alias('dolvolm')]
    set_aux_cols3 += [(col(f'aux{j+1}') * col('ajexdi')).alias(i) for i, j in zip(col_names_aux, range(6))]
    cols_to_drop_aux = ['cshtrd', 'div_tot', 'div_cash','div_spc', 'dolvol', 'prc_high', 'prc_low']  + [f'aux{i+1}' for i in range(6)]
    dict_aux = {'div_totm': 'div_tot', 'div_cashm': 'div_cash', 'div_spcm': 'div_spc', 'dolvolm': 'dolvol', 'prc_highm': 'prc_high', 'prc_lowm': 'prc_low'}
    __comp_msf = (pl.scan_ipc('__comp_dsf.ft')
                    .with_columns(set_aux_cols)
                    .with_columns(set_aux_cols2)
                    .with_columns(set_aux_cols3)
                    .drop(cols_to_drop_aux)
                    .filter((col('prc_local').is_not_null()) & (col('curcdd').is_not_null()) & (col('prcstd').is_in([3, 4, 10])))
                    .rename(dict_aux)
                    .sort(['gvkey', 'iid', 'eom', 'datadate'])
                    .group_by(['gvkey', 'iid', 'eom']).last()
                    .with_columns(source    = pl.lit(1).cast(pl.Int32),
                                  exchg     = col('exchg').cast(pl.Int32),
                                  prcstd    = col('prcstd').cast(pl.Int32)))
    aux_ajexm_exp = pl.when(col('ajexm') != 0).then(col('csho_fund') * col('ajex_fund') / col('ajexm')).otherwise(pl.lit(None).cast(pl.Float64))
    __comp_secm = pl.scan_ipc('Raw data/__comp_secm1.ft')
    fx = compustat_fx().lazy()
    fx_div = fx.clone().with_columns(col('fx').alias('fx_div')).drop('fx')
    aux = pl.scan_ipc('__firm_shares2.ft')
    __comp_secm = (__comp_secm.join(aux, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'ddate'])
                              .join(fx, how = 'left', on = ['datadate', 'curcdd']) #fx datadate in our terminology is the same as date in Theis' code
                              .join(fx_div, how = 'left', left_on = ['datadate', 'curcddvm'], right_on = ['datadate', 'curcdd'])
                              .with_columns(eom      = col('datadate').dt.month_end(),
                                            cshoc    = pl.coalesce(col('cshom') / 1e6, col('csfsm') / 1e3, col('cshoq'), aux_ajexm_exp),
                                            chstrm   = adj_trd_vol_NASDAQ('datadate', 'cshtrm', 'exchg', 14),
                                            fx       = pl.when(col('curcdd') == 'USD').then(pl.lit(1)).otherwise(col('fx')),
                                            fx_div   = pl.when(col('curcddvm') == 'USD').then(pl.lit(1)).otherwise(col('fx_div')).alias('fx_div'))
                              .with_columns(prc      = col('prc_local') * col('fx'),
                                            prc_high = col('prc_high') * col('fx'),
                                            prc_low  = col('prc_low') * col('fx'),
                                            ri       = col('ri_local') * col('fx'),
                                            div_tot  = col('dvpsxm') * col('fx_div'))
                              .with_columns(me       = col('prc') * col('cshoc'),
                                            dolvol   = col('prc') * col('cshtrm'),
                                            div_cash = pl.lit(None).cast(pl.Float64),
                                            div_spc  = pl.lit(None).cast(pl.Float64),
                                            prcstd   = pl.lit(10).cast(pl.Int32),
                                            source   = pl.lit(2).cast(pl.Int32),
                                            exchg    = col('exchg').cast(pl.Int32)))
    common_vars = ['gvkey', 'iid', 'datadate', 'eom', 'tpci', 'exchg', 'curcdd', 'prc_local', 'prc_high', 'prc_low', 'ajexdi', 'cshoc', 'ri_local', 'fx', 'prc', 'me', 'cshtrm', 'dolvol', 'ri', 'div_tot', 'div_cash', 'div_spc', 'prcstd', 'source']
    __comp_msf = (pl.concat([__comp_msf.select(common_vars), __comp_secm.select(common_vars)])
                    .sort('source')
                    .unique(subset = ['gvkey', 'iid', 'eom'], keep = 'first')
                    .drop('source')
                    .sort(['gvkey', 'iid', 'eom']))
    __comp_msf.collect().write_ipc('__comp_msf.ft')
@measure_time
def comp_exchanges():
    __ex_country = pl.read_ipc('Raw data/__ex_country1.ft')
    special_exchanges = [
        0, 1, 2, 3, 4, 15, 16, 17, 18, 21,
        13, 19, 20, 127, 150, 157, 229, 263, 269, 281,
        283, 290, 320, 326, 341, 342, 347, 348, 349, 352]
    #15, 16, 17, 18, 21 US exchanges not in NYSE, Amex and NASDAQ
    #150 AIAF Mercado De Renta Fija --> Spanish exchange for trading debt securities https://practiceguides.chambers.com/practice-guides/capital-markets-debt-2019/spain/1-debt-marketsexchanges
    #349 BATS Chi-X Europe --> Trades stocks from various european exchanges. Should we keep it?
    #352 CHI-X Australia --> Only Trades securities listed on ASX (exchg=106). Should we keep it?
    __ex_country = pl.SQLContext(frame=__ex_country).execute(
        """
        SELECT DISTINCT exchg,
            CASE
                WHEN COUNT(DISTINCT excntry) > 1 THEN 'multi national'
                ELSE MAX(excntry)
            END AS excntry
        FROM frame
        WHERE excntry IS NOT NULL AND exchg IS NOT NULL
        GROUP BY exchg
        """).collect().sort('exchg')
    comp_r_ex_codes = pl.read_ipc('Raw data/comp_r_ex_codes.ft')
    __ex_country = __ex_country.join(comp_r_ex_codes, how = 'left', left_on = 'exchg', right_on = 'exchgcd')
    __ex_country = __ex_country.with_columns(col('exchg').cast(pl.Int64))
    exch_main = pl.when((col('excntry') != 'multi national') & col('exchg').is_in(special_exchanges).not_()).then(pl.lit(1)).otherwise(pl.lit(0)).alias('exch_main')
    __ex_country = __ex_country.with_columns(exch_main)
    return __ex_country
@measure_time
def add_primary_sec(data_path, datevar, file_name):
    __prihistrow = pl.read_ipc('Raw data/__prihistrow.ft')
    __prihistusa = pl.read_ipc('Raw data/__prihistusa.ft')
    __prihistcan = pl.read_ipc('Raw data/__prihistcan.ft')
    c1 = col(datevar) >= col('effdate')
    c2 = col(datevar) <= col('thrudate')
    c3 = col('thrudate').is_null()
    aux_data = pl.read_ipc(data_path,columns = ['gvkey', datevar]).unique()
    __prihistrow_join = aux_data.join(__prihistrow, how = 'left', on = 'gvkey').filter(c1).filter(c2|c3)
    cols_to_drop = ['effdate','thrudate']
    cols_to_keep = [aux_var for aux_var in __prihistrow_join.columns if aux_var not in cols_to_drop]
    __prihistrow_join = __prihistrow_join.select(cols_to_keep)
    __prihistusa_join = aux_data.join(__prihistusa, how = 'left', on = 'gvkey').filter(c1).filter(c2|c3)
    cols_to_drop = ['effdate','thrudate']
    cols_to_keep = [aux_var for aux_var in __prihistusa_join.columns if aux_var not in cols_to_drop]
    __prihistusa_join = __prihistusa_join.select(cols_to_keep)
    __prihistcan_join = aux_data.join(__prihistcan, how = 'left', on = 'gvkey').filter(c1).filter(c2|c3)
    cols_to_drop = ['effdate','thrudate']
    cols_to_keep = [aux_var for aux_var in __prihistcan_join.columns if aux_var not in cols_to_drop]
    __prihistcan_join = __prihistcan_join.select(cols_to_keep)
    del aux_data, __prihistrow, __prihistusa, __prihistcan
    collected = gc.collect()
    data = pl.read_ipc(data_path)
    for df in [__prihistrow_join, __prihistusa_join, __prihistcan_join]: data = data.join(df, how = 'left', on = ['gvkey', datevar])
    data = data.join(pl.read_ipc('Raw data/__header.ft').unique('gvkey', keep = 'first'), how = 'left', on = 'gvkey')#Header has duplicates
    del __prihistrow_join, __prihistusa_join, __prihistcan_join
    data = data.with_columns([pl.coalesce(['prihistrow', 'prirow']).alias('prihistrow'),pl.coalesce(['prihistusa', 'priusa']).alias('prihistusa'),pl.coalesce(['prihistcan', 'prican']).alias('prihistcan')])
    c1 = col('iid').is_not_null()
    c2 = col('iid') == col('prihistrow')
    c3 = col('iid') == col('prihistusa')
    c4 = col('iid') == col('prihistcan')
    primary_sec = pl.when(c1 & (c2 | c3 | c4)).then(pl.lit(1)).otherwise(pl.lit(0)).alias('primary_sec')
    data = data.with_columns(pl.coalesce(primary_sec))
    cols_to_drop = ['prihistrow','prihistusa','prihistcan', 'prirow', 'priusa', 'prican']
    cols_to_keep = [aux_var for aux_var in data.columns if aux_var not in cols_to_drop]
    data.select(cols_to_keep).unique(['gvkey', 'iid', 'datadate']).write_ipc(file_name)
@measure_time
def process_comp_sf1(freq):
    if freq == 'm':
        gen_comp_msf()
        base = pl.read_ipc('__comp_msf.ft')
    else:
        base = pl.read_ipc('__comp_dsf.ft')
    __returns = base.filter(col('ri').is_not_null()).filter(col('prcstd').is_in([3, 4, 10])).select(['gvkey','iid','datadate','ri','ri_local','prcstd','curcdd']).unique(['gvkey','iid','datadate']).sort(['gvkey', 'iid', 'datadate'])
    if freq == 'd':
        __returns = __returns.with_columns(ret = col('ri').pct_change().over(['gvkey','iid']),
                                            ret_local = col('ri_local').pct_change().over(['gvkey','iid']),
                                            ret_lag_dif = (col('datadate')-col('datadate').shift(1)).over(['gvkey','iid']),
                                            lagged_iid = col('iid').shift(1).over(['gvkey','iid']),
                                            lagged_curcdd = col('curcdd').shift(1).over(['gvkey','iid']))
    else:
        months_date = col('datadate').dt.year()*12 + col('datadate').dt.month().cast(pl.Int32)
        months_date_lag = col('datadate').shift(1).dt.year()*12 + col('datadate').shift(1).dt.month().cast(pl.Int32)
        ret_lag_dif = (months_date - months_date_lag).alias('ret_lag_dif')
        __returns = __returns.with_columns(ret = col('ri').pct_change().over(['gvkey','iid']),
                                            ret_local = col('ri_local').pct_change().over(['gvkey','iid']),
                                            ret_lag_dif = (months_date - months_date_lag).over(['gvkey','iid']),
                                            lagged_iid = col('iid').shift(1).over(['gvkey','iid']),
                                            lagged_curcdd = col('curcdd').shift(1).over(['gvkey','iid']))
    c1 = col('iid') == col('lagged_iid')
    c2 = col('curcdd') != col('lagged_curcdd')
    __returns = __returns.with_columns(ret_local = pl.when(c1 & c2).then(col('ret')).otherwise(col('ret_local')))
    __returns = __returns.select(['gvkey','iid','datadate','ret','ret_local','ret_lag_dif'])
    __delist = __returns.filter(col('ret_local').is_not_null()).filter(col('ret_local') != 0.)
    __delist = __delist.select(['gvkey', 'iid', 'datadate']).sort(['gvkey', 'iid', 'datadate']).unique(['gvkey', 'iid'], keep = 'last')
    __sec_info = pl.read_ipc('Raw data/__sec_info.ft')
    __delist = __delist.join(__sec_info, how = 'left', on = ['gvkey', 'iid']).rename({'datadate': 'date_delist'})
    del __sec_info
    __delist = __delist.filter(col('secstat') == 'I').with_columns(pl.when(col('dlrsni').is_in(['02', '03'])).then(pl.lit(-0.3)).otherwise(pl.lit(0.)).alias('dlret'))
    __delist = __delist.select(['gvkey','iid','date_delist','dlret'])
    __comp_sf2 = base.join(__returns, how = 'left', on = ['gvkey','iid', 'datadate'])
    del __returns
    collected = gc.collect()
    __comp_sf2 = __comp_sf2.join(__delist, how = 'left', on = ['gvkey', 'iid'])
    del __delist
    __comp_sf2 = __comp_sf2.filter((col('datadate') <= col('date_delist')) | (col('date_delist').is_null()))
    __comp_sf2 = __comp_sf2.with_columns([pl.when(col('datadate') == col('date_delist')).then((1+col('ret'))*(1+col('dlret')) - 1).otherwise(col('ret')).alias('ret'),
                                            pl.when(col('datadate') == col('date_delist')).then((1+col('ret_local'))*(1+col('dlret')) - 1).otherwise(col('ret_local')).alias('ret_local')])
    cols_to_drop = ['ri', 'ri_local', 'date_delist', 'dlret']
    cols_to_keep = [aux_var for aux_var in __comp_sf2.columns if aux_var not in cols_to_drop]
    __comp_sf2 = __comp_sf2.select(cols_to_keep)
    crsp_mcti = pl.read_ipc('Raw data/crsp_mcti_t30ret.ft').with_columns(merge_aux = col('caldt').dt.year()*12 + col('caldt').dt.month()).drop('caldt')
    ff_factors_monthly = pl.read_ipc('Raw data/ff_factors_monthly.ft').with_columns(merge_aux = col('date').dt.year()*12 + col('date').dt.month()).drop('date')
    scale = 1 if (freq == 'm') else 21
    __comp_sf2 = __comp_sf2.with_columns(merge_aux = col('datadate').dt.year()*12 + col('datadate').dt.month())
    __comp_sf2 = __comp_sf2.join(crsp_mcti, how = 'left', on = 'merge_aux').join(ff_factors_monthly, how = 'left', on = 'merge_aux')
    __comp_sf2 = __comp_sf2.with_columns(ret_exc = col('ret') - pl.coalesce(['t30ret','rf'])/scale)
    cols_to_drop = ['merge_aux', 'rf', 't30ret']
    cols_to_keep = [aux_var for aux_var in __comp_sf2.columns if aux_var not in cols_to_drop]
    __comp_sf2 = __comp_sf2.select(cols_to_keep)
    __exchanges = comp_exchanges()
    __comp_sf2 = __comp_sf2.with_columns(col('exchg').cast(pl.Int64))
    __comp_sf2.join(__exchanges, how = 'left', on = ['exchg']).write_ipc('__comp_sf2.ft')
    del __comp_sf2
    collected = gc.collect()
    add_primary_sec('__comp_sf2.ft', 'datadate',f'comp_{freq}sf.ft')
    collected = gc.collect()
def gen_MMYY_column(var): return col(var).dt.year()*12 + col(var).dt.month()
def add_MMYY_column_drop_original(df, var): return df.with_columns(merge_aux = gen_MMYY_column(var)).drop(var)
@measure_time
def prepare_crsp_sf(freq):
    __crsp_sf = pl.scan_ipc(f'Raw data/__crsp_sf_{freq}.ft')
    dolvol_exp = col('prc').abs() * col('vol')
    c_div_tot = col('cfacshr').shift(1).over('permno') != 0
    div_tot_aux = ( (col('ret') - col('retx')) * col('prc').shift(1) * (col('cfacshr') / col('cfacshr').shift(1)) ).over('permno')
    div_tot_exp = pl.when(c_div_tot).then(div_tot_aux).otherwise(pl.lit(None).cast(pl.Float64))
    permno_exp = col('permno').cast(pl.Int64)
    __crsp_sf = __crsp_sf.with_columns(adj_trd_vol_NASDAQ('date', 'vol', 'exchcd', 3))
    __crsp_sf = __crsp_sf.sort(['permno', 'date']).with_columns(dolvol = dolvol_exp,
                                                                div_tot = div_tot_exp,
                                                                permno = permno_exp,
                                                                merge_aux = gen_MMYY_column('date'))
    crsp_sedelist_aux_col = [gen_MMYY_column('dlstdt').alias('merge_aux')] if (freq == 'm') else [col('dlstdt').alias('date')]
    crsp_sedelist = pl.scan_ipc(f'Raw data/crsp_{freq}sedelist.ft').with_columns(crsp_sedelist_aux_col)
    merge_vars = ['permno', 'merge_aux'] if (freq == 'm') else ['permno', 'date']
    __crsp_sf = __crsp_sf.join(crsp_sedelist, how = 'left', on = merge_vars)
    c1 = col('dlret').is_null()
    c2 = col('dlstcd') == 500
    c3 = col('dlstcd').is_between(520, 584)
    c4 = (c1 & (c2 | c3))
    c5 = col('ret').is_null()
    c6 = col('dlret').is_not_null()
    c7 = (c5 & c6)
    __crsp_sf = __crsp_sf.with_columns(dlret = pl.when(c4).then(pl.lit(-0.3)).otherwise(col('dlret')),
                                       ret = pl.when(c7).then(pl.lit(0.)).otherwise(col('ret')))
    __crsp_sf = __crsp_sf.with_columns(ret = (((col('ret') + 1) * (pl.coalesce(['dlret', 0.]) + 1)) -1))
    raw_rf_data = [pl.scan_ipc('Raw data/crsp_mcti_t30ret.ft'), pl.scan_ipc('Raw data/ff_factors_monthly.ft')]
    crsp_mcti = add_MMYY_column_drop_original(raw_rf_data[0], 'caldt')
    ff_factors_monthly = add_MMYY_column_drop_original(raw_rf_data[1], 'date')
    for df in [crsp_mcti, ff_factors_monthly]: __crsp_sf = __crsp_sf.join(df, how = 'left', on = 'merge_aux')
    scale = 1 if (freq == 'm') else 21
    c1 = pl.count('me').over(['permco', 'date']) != 0
    me_company_aux = pl.coalesce(['me',0.]).sum().over(['permco', 'date'])
    me_company_exp = pl.when(c1).then(me_company_aux).otherwise(pl.lit(None).cast(pl.Float64))
    ret_exc_exp = col('ret') - pl.coalesce(['t30ret','rf'])/scale
    __crsp_sf = __crsp_sf.with_columns(ret_exc = ret_exc_exp,
                                       me_company = me_company_exp)
    if freq == 'm': __crsp_sf = __crsp_sf.with_columns([(col(var) * 100).alias(var) for var in ['vol', 'dolvol']])
    __crsp_sf = __crsp_sf.drop(['rf', 't30ret','merge_aux','dlret', 'dlstcd', 'dlstdt'])
    __crsp_sf = __crsp_sf.unique(['permno', 'date']).sort(['permno', 'date'])
    __crsp_sf.collect(streaming=True).write_ipc(f'crsp_{freq}sf.ft')
@measure_time
def combine_crsp_comp_sf():
    crsp_msf = pl.scan_ipc('crsp_msf.ft')
    comp_msf = pl.scan_ipc('comp_msf.ft')
    cols_to_keep = ['id','permno','permco','gvkey', 'iid','excntry', 'exch_main', 'common', 'primary_sec', 'bidask','crsp_shrcd','crsp_exchcd','comp_tpci','comp_exchg', 'curcd', 'fx', 'date', 'eom','adjfct','shares', 'me', 'me_company', 'prc','prc_local', 'prc_high', 'prc_low', 'dolvol','tvol', 'ret','ret_local','ret_exc','ret_lag_dif', 'div_tot','div_cash','div_spc','source_crsp']
    crsp_msf = crsp_msf.with_columns([
                            col('exch_main').cast(pl.Int32),
                            col('bidask').cast(pl.Int32),
                            col('permno').alias('id'),
                            pl.lit('USA').alias('excntry'),
                            #col('shrcd').is_in([10, 11, 12]).cast(pl.Int32).alias('common'),
                            (col('shrcd').is_in([10, 11, 12]).fill_null(pl.lit(False).cast(pl.Boolean))).cast(pl.Int32).alias('common'),
                            pl.lit(1).alias('primary_sec'),
                            pl.lit('').alias('comp_tpci'),
                            pl.lit(None).cast(pl.Int64).alias('comp_exchg'),
                            pl.lit('USD').alias('curcd'),
                            pl.lit(1.).alias('fx'),
                            col('date').dt.month_end().alias('eom'),
                            col('prc').alias('prc_local'),
                            col('vol').alias('tvol'),
                            col('ret').alias('ret_local'),
                            pl.lit(1).cast(pl.Int64).alias('ret_lag_dif'),
                            pl.lit(None).cast(pl.Float64).alias('div_cash'),
                            pl.lit(None).cast(pl.Float64).alias('div_spc'),
                            pl.lit(1).alias('source_crsp')
                            ]).rename({'shrcd': 'crsp_shrcd', 'exchcd': 'crsp_exchcd', 'cfacshr': 'adjfct', 'shrout': 'shares'}).select(cols_to_keep)
    c1 = col('iid').str.contains('W')
    exp1 = pl.lit('3') + col('gvkey').cast(pl.Utf8) + col('iid').str.slice(0,2)
    c2 = col('iid').str.contains('C')
    exp2 = pl.lit('2') + col('gvkey').cast(pl.Utf8) + col('iid').str.slice(0,2)
    exp3 = pl.lit('1') + col('gvkey').cast(pl.Utf8) + col('iid').str.slice(0,2)
    exp4 = pl.when(c1).then(exp1).when(c2).then(exp2).otherwise(exp3)
    comp_msf = comp_msf.with_columns([
                                        exp4.cast(pl.Int64).alias('id'),
                                        pl.lit(None).cast(pl.Int64).alias('permno'),
                                        pl.lit(None).cast(pl.Int64).alias('permco'),
                                        pl.when(col('tpci') == '0').then(pl.lit(1)).otherwise(pl.lit(0)).alias('common'),
                                        pl.when(col('prcstd') == 4).then(pl.lit(1)).otherwise(pl.lit(0)).alias('bidask'),
                                        pl.lit(None).cast(pl.Float64).alias('crsp_shrcd'),
                                        pl.lit(None).cast(pl.Float64).alias('crsp_exchcd'),
                                        col('me').alias('me_company'),
                                        pl.lit(0).alias('source_crsp'),
                                        col('ret_lag_dif').cast(pl.Int64).alias('ret_lag_dif')
                                    ]).rename({'tpci': 'comp_tpci', 'exchg': 'comp_exchg', 'curcdd': 'curcd', 'datadate': 'date', 'ajexdi': 'adjfct', 'cshoc': 'shares', 'cshtrm': 'tvol'}).select(cols_to_keep)

    __msf_world = pl.concat([crsp_msf, comp_msf], how="vertical_relaxed")
    __msf_world = __msf_world.sort(['id', 'eom']).with_columns([col('ret_exc').shift(-1).over('id').alias('ret_exc_lead1m'), col('id').shift(-1).alias('id_lead1m'), col('ret_lag_dif').shift(-1).over('id').alias('ret_lag_dif_lead1m')])
    c1 = (col('id_lead1m') != col('id'))
    c2 = (col('ret_lag_dif') != 1)
    c3 = (c1 | c2)
    __msf_world = __msf_world.sort(['id', 'eom']).with_columns(pl.when(c3).then(pl.lit(None).cast(pl.Float64)).otherwise(col('ret_exc_lead1m')).alias('ret_exc_lead1m'))
    counts = __msf_world.group_by(['gvkey', 'iid', 'eom']).agg(pl.len().alias('count'))
    __obs_main = __msf_world.select(['id','source_crsp','gvkey','iid','eom']).join(counts, on=['gvkey', 'iid', 'eom'], how='left').with_columns(pl.when((col('count') > 1) & (col('source_crsp') == 0)).then(0).otherwise(1).alias('obs_main')).drop(['count','iid','gvkey','source_crsp'])
    #c1 = col('count').is_in([0,1])
    #c2 = (col('count') > 1) & (col('source_crsp') == 1)
    #__obs_main = __msf_world.select(['id','source_crsp','gvkey','iid','eom']).join(counts, on=['gvkey', 'iid', 'eom'], how='left').with_columns((pl.when(c1|c2).then(1).otherwise(0)).alias('obs_main')).drop(['count','iid','gvkey','source_crsp'])
    crsp_dsf = pl.scan_ipc('crsp_dsf.ft')
    comp_dsf = pl.scan_ipc('comp_dsf.ft')
    cols_to_keep = ['id','excntry','exch_main','common','primary_sec','bidask','curcd','fx','date','eom','adjfct','shares','me','dolvol','tvol','prc','prc_high','prc_low','ret_local','ret','ret_exc','ret_lag_dif','source_crsp']
    crsp_dsf = crsp_dsf.with_columns([
                                        col('permno').alias('id'),
                                        pl.lit('USA').alias('excntry'),
                                        #col('shrcd').is_in([10, 11, 12]).cast(pl.Int32).alias('common'),
                                        (col('shrcd').is_in([10, 11, 12]).fill_null(pl.lit(False).cast(pl.Boolean))).cast(pl.Int32).alias('common'),
                                        pl.lit(1).alias('primary_sec'),
                                        pl.lit('USD').alias('curcd'),
                                        pl.lit(1.).alias('fx'),
                                        col('date').dt.month_end().alias('eom'),
                                        col('ret').alias('ret_local'),
                                        pl.lit(1).cast(pl.Int64).alias('ret_lag_dif'),
                                        col('exch_main').cast(pl.Int32).alias('exch_main'),
                                        col('bidask').cast(pl.Int32).alias('bidask'),
                                        pl.lit(1).alias('source_crsp')
                                    ]).rename({'cfacshr': 'adjfct', 'shrout': 'shares', 'vol': 'tvol'}).select(cols_to_keep)
    c1 = col('iid').str.contains('W')
    exp1 = pl.lit('3') + col('gvkey').cast(pl.Utf8) + col('iid').str.slice(0,2)
    c2 = col('iid').str.contains('C')
    exp2 = pl.lit('2') + col('gvkey').cast(pl.Utf8) + col('iid').str.slice(0,2)
    exp3 = pl.lit('1') + col('gvkey').cast(pl.Utf8) + col('iid').str.slice(0,2)
    exp4 = pl.when(c1).then(exp1).when(c2).then(exp2).otherwise(exp3)
    comp_dsf = comp_dsf.with_columns([
                                        exp4.cast(pl.Int64).alias('id'),
                                        pl.when(col('tpci') == '0').then(pl.lit(1)).otherwise(pl.lit(0)).alias('common'),
                                        pl.when(col('prcstd') == 4).then(pl.lit(1)).otherwise(pl.lit(0)).alias('bidask'),
                                        col('datadate').dt.month_end().alias('eom'),
                                        pl.lit(0).alias('source_crsp'),
                                        (col('ret_lag_dif')/86_400_000).cast(pl.Int64).alias('ret_lag_dif')
                                    ]).rename({'curcdd': 'curcd', 'datadate': 'date', 'ajexdi': 'adjfct', 'cshoc': 'shares', 'cshtrd': 'tvol'}).select(cols_to_keep)
    __dsf_world = pl.concat([crsp_dsf, comp_dsf], how="vertical_relaxed")
    __msf_world = __msf_world.join(__obs_main, on=['id','eom'],how="left").unique(['id','eom']).sort(['id','eom'])
    __dsf_world = __dsf_world.join(__obs_main, on=['id','eom'],how="left").unique(['id','date']).sort(['id','date'])
    __msf_world.collect(streaming = True).write_ipc('__msf_world.ft')
    #__dsf_world.sink_ipc('out.ft')
    __dsf_world.collect(streaming = True).write_ipc('world_dsf.ft')
@measure_time
def crsp_industry():
    permno0 = pl.scan_ipc('Raw data/permno0.ft')
    permno0 = (permno0.with_columns(sic = pl.when(col('sic') == 0).then(pl.lit(None).cast(pl.Int64)).otherwise(col('sic')))
                      .with_columns(date = pl.date_ranges('namedt', 'nameendt'))
                      .explode('date')
                      .select(['permno', 'permco', 'date', 'sic', 'naics'])
                      .unique(['permno', 'date'])
                      .sort(['permno', 'date']))
    permno0.collect().write_ipc('crsp_ind.ft')
@measure_time
def COMP_HGICS(lib):
    paths = {'raw data': {'national': 'Raw data/comp_hgics_na.ft', 'global': 'Raw data/comp_hgics_gl.ft'},
             'output' : {'national': 'na_hgics.ft', 'global': 'g_hgics.ft'}}
    data = pl.read_ipc(paths['raw data'][lib]).sort(['gvkey', 'indfrom'])
    data = data.with_columns(gics = pl.when(col('gics').is_null()).then(-999).otherwise(col('gics')),
                             n = pl.len().over('gvkey'),
                             n_aux = pl.cum_count('gvkey').over('gvkey'))
    indthru_date = pl.lit(data[['indfrom']].max()[0,0]) if data[['indfrom']].max()[0,0] > date.today() else pl.lit(date.today())
    c1 = col('n') == col('n_aux')
    c2 = col('indthru').is_null()
    data = (data.with_columns(indthru = pl.when(c1 & c2).then(indthru_date).otherwise(col('indthru')))
                .select(['gvkey', pl.date_ranges('indfrom', 'indthru').alias('date'), 'gics'])
                .explode('date')
                .unique(subset = ['gvkey', 'date'])
                .sort(['gvkey', 'date']))
    data.write_ipc(paths['output'][lib])
@measure_time
def HGICS_JOIN():
    COMP_HGICS('global')
    COMP_HGICS('national')
    global_data = pl.scan_ipc('g_hgics.ft')
    local_data = pl.scan_ipc('na_hgics.ft')
    gjoin = local_data.join(global_data, on = ['gvkey', 'date'], how = 'outer_coalesce')
    gjoin = (gjoin.with_columns(gics = pl.coalesce(['gics', 'gics_right']))
                  .select(['gvkey', 'date', 'gics'])
                  .unique(['gvkey','date'])
                  .sort(['gvkey','date']))
    gjoin.collect().write_ipc('comp_hgics.ft')
@measure_time
def COMP_SIC_NAICS():
    funda_data  = pl.scan_ipc('Raw data/sic_naics_na.ft')
    gfunda_data = pl.scan_ipc('Raw data/sic_naics_gl.ft')
    funda_data = funda_data.filter(~( (col('gvkey') == 175650) & (col('datadate') == pl.date(2005,12,31)) & (col('naics').is_null() )))
    comp = funda_data.join(gfunda_data, on = ['gvkey', 'datadate'], how = 'outer_coalesce').sort(['gvkey','datadate'])
    comp = (comp.with_columns(sic = pl.coalesce(['sic', 'sic_right']), naics = pl.coalesce(['naics', 'naics_right']), end_date = col('datadate').shift(-1).over('gvkey'))
                .with_columns(end_date = pl.when(col('end_date').is_null()).then(col('datadate')).otherwise(col('end_date')))
                .with_columns(date = pl.date_ranges('datadate', 'end_date', closed = 'left'))
                .explode('date')
                .with_columns(date = pl.when(col('datadate') == col('end_date')).then(col('datadate')).otherwise(col('date')))
                .select(['gvkey', 'date', 'sic', 'naics'])
                .unique(['gvkey', 'date'])
                .sort(['gvkey', 'date']))
    comp.collect().write_ipc('comp_other.ft')
@measure_time
def comp_industry():
    COMP_SIC_NAICS()
    HGICS_JOIN()
    comp_other = pl.read_ipc('comp_other.ft')
    comp_gics = pl.read_ipc('comp_hgics.ft')
    join = comp_gics.join(comp_other, on=['gvkey', 'date'], how='outer_coalesce').sort(['gvkey', 'date'])
    join = (join.with_columns(aux_date = (col('date').shift(-1).dt.offset_by('-1d')).over('gvkey'))
                .with_columns(aux_date = pl.when(col('aux_date').is_null()).then(col('date')).otherwise('aux_date')))
    gaps = (join.filter(col('date') != col('aux_date'))
                .select(['gvkey', pl.date_ranges('date','aux_date', closed = 'right').alias('date')])
                .explode('date'))
    join = join.filter(col('date') == col('aux_date')).drop('aux_date')
    join = pl.concat([join, gaps], how = 'diagonal').unique(['gvkey', 'date']).sort(['gvkey', 'date'])
    join.write_ipc('comp_ind.ft')
@measure_time
def ff_ind_class(data, ff_grps):
    if ff_grps==38:
        lower_bounds = [1000, 1300, 1400, 1500, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4800, 4830, 4900, 4950, 4960, 4970, 5000, 5200, 6000, 7000, 9000]
        upper_bounds = [1299, 1399, 1499, 1799, 2099, 2199, 2299, 2399, 2499, 2599, 2661, 2799, 2899, 2999, 3099, 3199, 3299, 3399, 3499, 3599, 3699, 3799, 3879, 3999, 4799, 4829, 4899, 4949, 4959, 4969, 4979, 5199, 5999, 6999, 8999, 9999]
        classification = [i + 2 for i in range(len(lower_bounds))]
        ff38_exp = pl.when(col('sic').is_between(100, 999)).then(1)
        for i, j, k in zip(lower_bounds, upper_bounds, classification): ff38_exp = ff38_exp.when(col('sic').is_between(i, j)).then(k)
        ff38_exp = (ff38_exp.otherwise(pl.lit(None))).alias('ff38')
        df1 = data.with_columns(ff38_exp)
    else:
        df1= data.with_columns(pl.when(col("sic").is_in([2048,*range(100, 299+1), *range(700, 799+1), *range(910, 919+1)])).then(1)
                                .when(col("sic").is_in([2095, 2098, 2099, *range(2000, 2046+1), *range(2050, 2063+1), *range(2070, 2079+1), *range(2090, 2092+1)])).then(2)
                                .when(col("sic").is_in([2086, 2087, 2096, 2097, *range(2064, 2068+1)])).then(3)
                                .when(col("sic").is_in([2080, *range(2082, 2085+1)])).then(4)
                                .when(col("sic").is_in([*range(2100, 2199+1)])).then(5)
                                .when(col("sic").is_in([3732, 3930, 3931, *range(920, 999+1),*range(3650, 3652+1),*range(3940, 3949+1)])).then(6)
                                .when(col("sic").is_in([7840, 7841, 7900, 7910, 7911, 7980,*range(7800, 7833+1),*range(7920, 7933+1),*range(7940, 7949+1),*range(7990, 7999+1)])).then(7)
                                .when(col("sic").is_in([2770, 2771,*range(2700, 2749+1),*range(2780, 2799+1)])).then(8)
                                .when(col("sic").is_in([2047, 2391, 2392, 3160, 3161, 3229, 3260, 3262, 3263, 3269, 3230, 3231, 3750, 3751, 3800, 3860, 3861, 3910, 3911, 3914, 3915, 3991, 3995,*range(2510, 2519+1),*range(2590, 2599+1),*range(2840, 2844+1),*range(3170, 3172+1),*range(3190, 3199+1),*range(3630, 3639+1),*range(3870, 3873+1),*range(3960, 3962+1)])).then(9)
                                .when(col("sic").is_in([3020, 3021, 3130, 3131, 3150, 3151,*range(2300, 2390+1),*range(3100, 3111+1),*range(3140, 3149+1),*range(3963, 3965+1)])).then(10)
                                .when(col("sic").is_in([*range(8000, 8099+1)])).then(11)
                                .when(col("sic").is_in([3693, 3850, 3851,*range(3840, 3849+1)])).then(12)
                                .when(col("sic").is_in([2830, 2831,*range(2833, 2836+1)])).then(13)
                                .when(col("sic").is_in([*range(2800, 2829+1),*range(2850, 2879+1),*range(2890, 2899+1)])).then(14)
                                .when(col("sic").is_in([3031, 3041,*range(3050, 3053+1),*range(3060, 3099+1)])).then(15)
                                .when(col("sic").is_in([*range(2200, 2284+1),*range(2290, 2295+1),*range(2297, 2299+1),*range(2393, 2395+1),*range(2397, 2399+1)])).then(16)
                                .when(col("sic").is_in([2660, 2661, 3200, 3210, 3211, 3240, 3241, 3261, 3264, 3280, 3281, 3446, 3996,*range(800, 899+1),*range(2400, 2439+1),*range(2450, 2459+1),*range(2490, 2499+1),*range(2950, 2952+1),*range(3250, 3259+1),*range(3270, 3275+1),*range(3290, 3293+1),*range(3295, 3299+1),*range(3420, 3429+1),*range(3430, 3433+1),*range(3440, 3442+1),*range(3448, 3452+1),*range(3490, 3499+1)])).then(17)
                                .when(col("sic").is_in([*range(1500, 1511+1),*range(1520, 1549+1),*range(1600, 1799+1)])).then(18)
                                .when(col("sic").is_in([3300,*range(3310, 3317+1),*range(3320, 3325+1),*range(3330, 3341+1),*range(3350, 3357+1),*range(3360, 3379+1),*range(3390, 3399+1)])).then(19)
                                .when(col("sic").is_in([3400, 3443, 3444,*range(3460, 3479+1)])).then(20)
                                .when(col("sic").is_in([3538, 3585, 3586,*range(3510, 3536+1),*range(3540, 3569+1),*range(3580, 3582+1),*range(3589, 3599+1)])).then(21)
                                .when(col("sic").is_in([3600, 3620, 3621, 3648, 3649, 3660, 3699,*range(3610, 3613+1),*range(3623, 3629+1),*range(3640, 3646+1),*range(3690, 3692+1)])).then(22)
                                .when(col("sic").is_in([2296, 2396, 3010, 3011, 3537, 3647, 3694, 3700, 3710, 3711, 3799,*range(3713, 3716+1),*range(3790, 3792+1)])).then(23)
                                .when(col("sic").is_in([3720, 3721, 3728, 3729,*range(3723, 3725+1)])).then(24)
                                .when(col("sic").is_in([3730, 3731,*range(3740, 3743+1)])).then(25)
                                .when(col("sic").is_in([3795,*range(3760, 3769+1),*range(3480, 3489+1)])).then(26)
                                .when(col("sic").is_in([*range(1040, 1049+1)])).then(27)
                                .when(col("sic").is_in([*range(1000, 1039+1),*range(1050, 1119+1),*range(1400, 1499+1)])).then(28)
                                .when(col("sic").is_in([*range(1200, 1299+1)])).then(29)
                                .when(col("sic").is_in([1300, 1389,*range(1310, 1339+1),*range(1370, 1382+1),*range(2900, 2912+1),*range(2990, 2999+1)])).then(30)
                                .when(col("sic").is_in([4900, 4910, 4911, 4939,*range(4920, 4925+1),*range(4930, 4932+1),*range(4940, 4942+1)])).then(31)
                                .when(col("sic").is_in([4800, 4899,*range(4810, 4813+1),*range(4820, 4822+1),*range(4830, 4841+1),*range(4880, 4892+1)])).then(32)
                                .when(col("sic").is_in([7020, 7021, 7200, 7230, 7231, 7240, 7241, 7250, 7251, 7395, 7500, 7600, 7620, 7622, 7623, 7640, 7641,*range(7030, 7033+1),*range(7210, 7212+1),*range(7214, 7217+1),*range(7219, 7221+1),*range(7260, 7299+1),*range(7520, 7549+1),*range(7629, 7631+1),*range(7690, 7699+1),*range(8100, 8499+1),*range(8600, 8699+1),*range(8800, 8899+1),*range(7510, 7515+1)])).then(33)
                                .when(col("sic").is_in([3993, 7218, 7300, 7374, 7396, 7397, 7399, 7519, 8700, 8720, 8721,*range(2750, 2759+1),*range(7310, 7342+1),*range(7349, 7353+1),*range(7359, 7369+1),*range(7376, 7385+1),*range(7389, 7394+1),*range(8710, 8713+1),*range(8730, 8734+1),*range(8740, 8748+1),*range(8900, 8911+1),*range(8920, 8999+1),*range(4220, 4229+1)])).then(34)
                                .when(col("sic").is_in([3695,*range(3570, 3579+1),*range(3680, 3689+1)])).then(35)
                                .when(col("sic").is_in([7375,*range(7370, 7373+1)])).then(36)
                                .when(col("sic").is_in([3622, 3810, 3812,*range(3661, 3666+1),*range(3669, 3679+1)])).then(37)
                                .when(col("sic").is_in([3811,*range(3820, 3827+1),*range(3829, 3839+1)])).then(38)
                                .when(col("sic").is_in([2760, 2761,*range(2520, 2549+1),*range(2600, 2639+1),*range(2670, 2699+1),*range(3950, 3955+1)])).then(39)
                                .when(col("sic").is_in([3220, 3221,*range(2440, 2449+1),*range(2640, 2659+1),*range(3410, 3412+1)])).then(40)
                                .when(col("sic").is_in([4100, 4130, 4131, 4150, 4151, 4230, 4231, 4780, 4789,*range(4000, 4013+1),*range(4040, 4049+1),*range(4110, 4121+1),*range(4140, 4142+1),*range(4170, 4173+1),*range(4190, 4200+1),*range(4210, 4219+1),*range(4240, 4249+1),*range(4400, 4700+1),*range(4710, 4712+1),*range(4720, 4749+1),*range(4782, 4785+1)])).then(41)
                                .when(col("sic").is_in([5000, 5099, 5100,*range(5010, 5015+1),*range(5020, 5023+1),*range(5030, 5060+1),*range(5063, 5065+1),*range(5070, 5078+1),*range(5080, 5088+1),*range(5090, 5094+1),*range(5110, 5113+1),*range(5120, 5122+1),*range(5130, 5172+1),*range(5180, 5182+1),*range(5190, 5199+1)])).then(42)
                                .when(col("sic").is_in([5200, 5250, 5251, 5260, 5261, 5270, 5271, 5300, 5310, 5311, 5320, 5330, 5331, 5334, 5900, 5999,
                                                        *range(5210, 5231+1),*range(5340, 5349+1),*range(5390, 5400+1),*range(5410, 5412+1),*range(5420, 5469+1),*range(5490, 5500+1),*range(5510, 5579+1),*range(5590, 5700+1),*range(5710, 5722+1),*range(5730, 5736+1),*range(5750, 5799+1),*range(5910, 5912+1),*range(5920, 5932+1),*range(5940, 5990+1),*range(5992, 5995+1)])).then(43)
                                .when(col("sic").is_in([7000, 7213,*range(5800, 5829+1),*range(5890, 5899+1),*range(7010, 7019+1),*range(7040, 7049+1)])).then(44)
                                .when(col("sic").is_in([6000,*range(6010, 6036+1),*range(6040, 6062+1),*range(6080, 6082+1),*range(6090, 6100+1),*range(6110, 6113+1),*range(6120, 6179+1),*range(6190, 6199+1)])).then(45)
                                .when(col("sic").is_in([6300, 6350, 6351, 6360, 6361,*range(6310, 6331+1),*range(6370, 6379+1),*range(6390, 6411+1)])).then(46)
                                .when(col("sic").is_in([6500, 6510, 6540, 6541, 6610, 6611,*range(6512, 6515+1),*range(6517, 6532+1),*range(6550, 6553+1),*range(6590, 6599+1)])).then(47)
                                .when(col("sic").is_in([6700, 6798, 6799,*range(6200, 6299+1),*range(6710, 6726+1),*range(6730, 6733+1),*range(6740, 6779+1),*range(6790, 6795+1)])).then(48)
                                .when(col("sic").is_in([4970, 4971, 4990, 4991,*range(4950, 4961+1)])).then(49)
                                .otherwise(pl.lit(None)).alias("ff49")
                                )

    return df1
def sas_percentile_method_5(series, p):
    """
    Calculates the given percentile using the SAS 5th method, which is the default SAS method and was used in our SAS code.
    """
    n = len(series)
    rank = p * n
    if rank.is_integer():return (series[int(rank) - 1] + series[int(rank)]) / 2
    else: return series[int(rank)]
@measure_time
def nyse_size_cutoffs(data_path):
    """
    Computes the 1st, 20th, 50th, and 80th percentiles of the market equity'me 'column based on NYSE stocks.
    The percentiles are calculated using the SAS percentile method 5.

    Parameters:
    - data: Input dataset containing stock information.

    Returns:
    - A DataFrame with columns 'eom' (end of month), 'n' (number of observations), and percentiles of 'me' column.
    """
    nyse_stocks = pl.scan_ipc(data_path)
    # Filter the data for NYSE stocks based on specific criteria
    nyse_stocks = nyse_stocks.filter(
        (col('crsp_exchcd') == 1) &         # NYSE exchange code
        (col('obs_main') == 1) &            # Main observation flag
        (col('exch_main') == 1) &           # Main exchange flag
        (col('primary_sec') == 1) &         # Primary security flag
        (col('common') == 1) &              # Common stock flag
        col('me').is_not_null()             # Ensure market equity (me) is not null
    )
    # Group the data by 'eom' and calculate percentiles on the 'me' column
    grouped = nyse_stocks.sort(['eom','me']).group_by('eom').agg([
            col('me').count().alias('n'),   # Count of 'me' values
            col('me').map_elements(lambda series: sas_percentile_method_5(series, 0.01)).alias('nyse_p1'),   # 1st percentile
            col('me').map_elements(lambda series: sas_percentile_method_5(series, 0.20)).alias('nyse_p20'),  # 20th percentile
            col('me').map_elements(lambda series: sas_percentile_method_5(series, 0.50)).alias('nyse_p50'),  # 50th percentile
            col('me').map_elements(lambda series: sas_percentile_method_5(series, 0.80)).alias('nyse_p80')   # 80th percentile
        ])
    grouped.collect().write_ipc('nyse_cutoffs.ft')
@measure_time
def classify_stocks_size_groups():
    nyse_cutoffs = pl.scan_ipc('nyse_cutoffs.ft')
    __msf_world = pl.scan_ipc('__msf_world3.ft')
    world_msf = __msf_world.join(nyse_cutoffs, how = 'left', on = 'eom')
    size_grp_column = (pl.when(col('me').is_null()).then(pl.lit(''))
                       .when(col('me') >= col('nyse_p80')).then(pl.lit('mega'))
                       .when(col('me') >= col('nyse_p50')).then(pl.lit('large'))
                       .when(col('me') >= col('nyse_p20')).then(pl.lit('small'))
                       .when(col('me') >= col('nyse_p1')).then(pl.lit('micro'))
                       .otherwise(pl.lit('nano')))
                       #pl.when(col('me').is_null()).then(pl.lit(None).cast(pl.Utf8))
    world_msf = world_msf.with_columns(size_grp = size_grp_column).drop([i for i in nyse_cutoffs.columns if i not in ['eom', 'n']])
    world_msf.collect().write_ipc('world_msf.ft')
@measure_time
def return_cutoffs(freq, crsp_only):
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
    # Filter data based on provided criteria. If 'crsp_only' is 1, filter for CRSP data only.
    data = pl.scan_ipc(f'world_{freq}sf.ft')
    c1 = (col("common") == 1) & (col("obs_main") == 1) & (col("exch_main") == 1) & (col("primary_sec") == 1) & (col("excntry") != 'ZWE') & (col("ret_exc").is_not_null())
    c2 = (col("source_crsp") == 1)
    base = data.filter(c1)
    if crsp_only == 1: base = base.filter(c2)#.sort('eom')
    #base = base.sort('eom')
    # If frequency is monthly, group by 'eom' (end of month)
    if freq == 'm':
        grouped = base.group_by('eom').agg([
            col("ret").count().alias('n'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_0_1'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_1'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_99'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_99_9'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_local_0_1'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_local_1'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_local_99'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_local_99_9'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_exc_0_1'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_exc_1'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_exc_99'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_exc_99_9')
            ])
    # If frequency is daily, group by year and month. We get one output per unique year and month combination considering all the relevant observations
    else:
        # Extract year and month from the 'date' column for grouping
        base = base.with_columns(year = col("date").dt.year().alias("year"),month = col("date").dt.month())
        grouped = base.group_by(["year", "month"]).agg([
            col("ret").count().alias('n'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_0_1'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_1'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_99'),
            col('ret').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_99_9'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_local_0_1'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_local_1'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_local_99'),
            col('ret_local').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_local_99_9'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.001)).alias('ret_exc_0_1'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.01)).alias('ret_exc_1'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.99)).alias('ret_exc_99'),
            col('ret_exc').sort().map_elements(lambda series: sas_percentile_method_5(series, 0.999)).alias('ret_exc_99_9')
            ])
    if freq == 'm': grouped.collect().sort('eom').write_ipc('return_cutoffs.ft')
    else: grouped.collect().sort(['year','month']).write_ipc('return_cutoffs_daily.ft')
def winsorize_mkt_ret(var, cutoff, comparison):
    if comparison == '>': c1 = (col(var) > col(cutoff)) & (col('source_crsp') == 0) & (col(var).is_not_null())
    else: c1 = ((col(var) < col(cutoff)).fill_null(pl.lit(False).cast(pl.Boolean))) & (col('source_crsp') == 0) & (col(var).is_not_null())
    return (pl.when(c1).then(col(cutoff)).otherwise(col(var))).alias(var)
@measure_time
def market_returns(data_path, freq, wins_comp, wins_data_path):
    if freq == 'm': dt_col, max_date_lag = 'eom', 1
    else: dt_col, max_date_lag = 'date', 14
    cols = ["source_crsp","id","date","eom","excntry","obs_main","exch_main","primary_sec","common","ret_lag_dif","me","dolvol","ret","ret_local","ret_exc"]
    __common_stocks = pl.scan_ipc(data_path).select(cols).unique().sort(['id',dt_col])
    __common_stocks = __common_stocks.with_columns(me_lag1 = col('me').shift(1).over('id'), dolvol_lag1 = col('dolvol').shift(1).over('id'))
    if wins_comp == 1:
        if freq =='m':
            wins_data = pl.scan_ipc(wins_data_path).select(['eom','ret_exc_0_1', 'ret_exc_99_9', 'ret_0_1', 'ret_99_9', 'ret_local_0_1', 'ret_local_99_9'])
            __common_stocks = __common_stocks.join(wins_data, how = 'left', on = 'eom')
        else:
            wins_data = pl.scan_ipc(wins_data_path).select(['year', 'month','ret_exc_0_1', 'ret_exc_99_9', 'ret_0_1', 'ret_99_9', 'ret_local_0_1', 'ret_local_99_9'])
            __common_stocks = __common_stocks.with_columns(year = col(dt_col).dt.year(),month = col(dt_col).dt.month())
            __common_stocks = __common_stocks.join(wins_data, how = 'left', on = ['year','month'])#.drop(['year','month'])
            __common_stocks = __common_stocks.select([i for i in __common_stocks.columns if i not in ['year','month']])
        __common_stocks = __common_stocks.with_columns([winsorize_mkt_ret('ret', 'ret_99_9', '>'), winsorize_mkt_ret('ret_local', 'ret_local_99_9', '>'), winsorize_mkt_ret('ret_exc', 'ret_exc_99_9', '>')])
        __common_stocks = __common_stocks.with_columns([winsorize_mkt_ret('ret', 'ret_0_1', '<'), winsorize_mkt_ret('ret_local', 'ret_local_0_1', '<'),winsorize_mkt_ret('ret_exc', 'ret_exc_0_1', '<')])
    c1 = (col('obs_main') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c2 = (col('exch_main') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c3 = (col('primary_sec') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c4 = (col('common') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c5 = (col('ret_lag_dif') <= max_date_lag).fill_null(pl.lit(False).cast(pl.Boolean))
    c6 = (col('me_lag1').is_not_null()).fill_null(pl.lit(False).cast(pl.Boolean))
    c7 = (col('ret_local').is_not_null()).fill_null(pl.lit(False).cast(pl.Boolean))
    __common_stocks = __common_stocks.filter(c7).filter(c6).filter(c5).filter(c4).filter(c3).filter(c2).filter(c1)
    __common_stocks = __common_stocks.with_columns(aux1 = col('ret_local')*col('me_lag1'), aux2 = col('ret')*col('me_lag1'), aux3 = col('ret_exc')*col('me_lag1'))
    __common_stocks = __common_stocks.group_by(['excntry', dt_col]).agg(stocks = pl.len(),
                                                                        me_lag1 = pl.sum('me_lag1'),
                                                                        dolvol_lag1 = pl.sum('dolvol_lag1'),
                                                                        mkt_vw_lcl = pl.sum('aux1')/pl.sum('me_lag1'),
                                                                        mkt_ew_lcl = pl.mean('ret_local'),
                                                                        mkt_vw = pl.sum('aux2')/pl.sum('me_lag1'),
                                                                        mkt_ew = pl.mean('ret'),
                                                                        mkt_vw_exc = pl.sum('aux3')/pl.sum('me_lag1'),
                                                                        mkt_ew_exc = pl.mean('ret_exc'))
    if freq == 'd':
        __common_stocks = __common_stocks.with_columns(year = col(dt_col).dt.year(), month = col(dt_col).dt.month())
        __common_stocks = __common_stocks.with_columns(max_stocks = pl.max('stocks').over(['excntry','year','month']))
        __common_stocks = __common_stocks.filter((col('stocks')/col('max_stocks')) >= 0.25)#.drop(['year','month','max_stocks'])
        __common_stocks = __common_stocks.select([i for i in __common_stocks.columns if i not in ['year','month','max_stocks']])
        __common_stocks.sort(['excntry',dt_col]).collect(streaming = True).write_ipc('market_returns_daily.ft')
    else:
        __common_stocks.sort(['excntry',dt_col]).collect(streaming = True).write_ipc('market_returns.ft')
@measure_time
def quarterize(df, var_list):
    #Check duplicates
    quarterized_df = df.clone().sort(['gvkey','fyr','fyearq','fqtr']).unique(['gvkey','fyr','fyearq','fqtr'], keep = 'first').sort(['gvkey','fyr','fyearq','fqtr'])
    list_aux = [col(var).cast(pl.Float64).diff().alias(var + '_q') for var in var_list]
    exp_count_aux = [col('gvkey').cum_count().over(['gvkey','fyr','fyearq']).alias('count_aux')]
    quarterized_df = quarterized_df.with_columns(list_aux + exp_count_aux)
    aux_ytd = [pl.when(col('count_aux') == 1).then(col(var)).otherwise(col(var + '_q')).alias(var + '_q') for var in var_list]
    c1 = (col('fqtr') != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col('fqtr').diff() != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    exp_del_aux = [pl.when(col('count_aux') == 1).then(c1).otherwise(c2).alias('del')]
    quarterized_df = quarterized_df.sort(['gvkey','fyr','fyearq','fqtr']).with_columns(aux_ytd + exp_del_aux)
    quarterized_df = quarterized_df.with_columns([pl.when(col('del')).then(pl.lit(None).cast(pl.Float64)).otherwise(col(var + '_q')).alias(var + '_q') for var in var_list]).drop(['del', 'count_aux'])
    quarterized_df = quarterized_df.sort(['gvkey','fyr','fyearq','fqtr']).unique(['gvkey','fyr','fyearq','fqtr'], keep = 'first').sort(['gvkey','fyr','fyearq','fqtr'])
    return quarterized_df
def ttm(var): return col(var) + col(var).shift(1) + col(var).shift(2) + col(var).shift(3)
def cumulate_4q(var_yrl, mode = 'add'):
    var_yrl_name = var_yrl[:-1]
    c1 = (col('gvkey') != col('gvkey').shift(3)).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col('fyr') != col('fyr').shift(3)).fill_null(pl.lit(True).cast(pl.Boolean))
    c3 = (col('curcdq') != col('curcdq').shift(3)).fill_null(pl.lit(True).cast(pl.Boolean))
    c4 = (ttm('fqtr') != 10).fill_null(pl.lit(True).cast(pl.Boolean))
    c5 = (c1 | c2 | c3 | c4)
    c6 = ttm(var_yrl).is_null()
    c7 = col('fqtr') == 4
    c8 = (c6 & c7)
    if mode == 'add': return pl.when(c5).then(pl.lit(None).cast(pl.Float64)).when(c8).then(col(f'{var_yrl_name}y')).otherwise(ttm(var_yrl)).alias(var_yrl_name)
    else: return [var_yrl, f'{var_yrl_name}y']
@measure_time
def standardized_accounting_data(coverage,convert_to_usd, me_data_path,include_helpers_vars, start_date):
    print('Beginning exploratory query into WRDS database g_fundq', flush=True)
    g_fundq_cols = wrds_session.raw_sql_polars_uri(f"""
                        SELECT *
                        FROM comp.g_fundq
                        LIMIT 10
                        """).columns
    print('Finished', flush=True)
    print('Beginning exploratory query into WRDS database fundq', flush=True)
    fundq_cols = wrds_session.raw_sql_polars_uri(f"""
                        SELECT *
                        FROM comp.fundq
                        LIMIT 10
                        """).columns
    print('Finished', flush=True)
    print('Begin standardized_accounting_data', flush=True)
    #Compustat Accounting Vars to Extract
    avars_inc = ["sale", "revt", "gp", "ebitda", "oibdp", "ebit", "oiadp", "pi", "ib", "ni", "mii","cogs", "xsga", "xopr", "xrd", "xad", "xlr", "dp", "xi", "do", "xido", "xint", "spi", "nopi", "txt","dvt"]
    avars_cf = ["oancf", "ibc", "dpc", "xidoc", "capx", "wcapt", # Operating
    "fincf", "fiao", "txbcof", "ltdch", "dltis", "dltr", "dlcch", "purtshr", "prstkc", "sstk","dv", "dvc"] # Financing
    avars_bs = ["at", "act", "aco", "che", "invt", "rect", "ivao", "ivst", "ppent", "ppegt", "intan", "ao", "gdwl", "re", # Assets
    "lt", "lct", "dltt", "dlc", "txditc", "txdb", "itcb", "txp", "ap", "lco", "lo", "seq", "ceq", "pstkrv", "pstkl", "pstk", "mib", "icapt"] # Liabilities
    # Variables in avars_other are not measured in currency units, and only available in annual data
    avars_other = ["emp"]
    avars = avars_inc + avars_cf + avars_bs
    print(f"INCOME STATEMENT: {len(avars_inc)} || CASH FLOW STATEMENT: {len(avars_cf)} || BALANCE SHEET: {len(avars_bs)} || OTHER: {len(avars_other)}", flush=True)
    #finding which variables of interest are available in the quarterly data
    combined_columns = g_fundq_cols + fundq_cols
    qvars_q = list({aux_var for aux_var in combined_columns if aux_var[:-1].lower() in avars and aux_var.endswith("q")}) #different from above to get only unique values
    qvars_y = list({aux_var for aux_var in combined_columns if aux_var[:-1].lower() in avars and aux_var.endswith("y")})
    qvars = qvars_q + qvars_y
    if coverage in ['global', 'world']:
        #preparing global data
        #annual
        query_vars = ''
        for i in avars + avars_other:
            # Add double quotes around 'do' or any other reserved keywords
            if i == 'do': query_vars += '"' + i + '", '
            elif i in ('gp', 'pstkrv', 'pstkl', 'itcb', 'xad', 'txbcof', 'ni'): pass
            else: query_vars += i + ', '
        query_vars = query_vars.rstrip(', ')
        g_funda = wrds_session.raw_sql_polars_uri(f"""
            SELECT gvkey, datadate, indfmt, curcd, 'GLOBAL' AS source, ib + COALESCE(xi, 0) + COALESCE("do", 0) AS ni, NULL AS gp, NULL AS pstkrv, NULL AS pstkl, NULL AS itcb, NULL AS xad, NULL AS txbcof, {query_vars}
            FROM comp.g_funda
            WHERE indfmt IN ('INDL', 'FS') AND datafmt = 'HIST_STD' AND popsrc = 'I' AND consol = 'C' AND datadate >= {start_date}
            """)
        g_funda = g_funda.with_columns([col('gvkey').cast(pl.Int64)] + [col(col_name).cast(pl.Float64) for col_name in ['gp', 'pstkrv', 'pstkl', 'itcb', 'xad', 'txbcof', 'ni']])
        g_funda.write_ipc('Raw data/g_funda.ft')
        del g_funda
        __gfunda =  pl.scan_ipc('Raw data/g_funda.ft').with_columns(pl.len().over(['gvkey', 'datadate']).alias('count_indfmt'))
        __gfunda = __gfunda.filter((col('count_indfmt') == 1) | ((col('count_indfmt') == 2) & (col('indfmt') == 'INDL'))).drop(['indfmt', 'count_indfmt'])
        #quarterly:
        query_vars = ''
        for i in qvars:
            if i in ('icaptq','niy','txditcq','txpq','xidoq','xidoy','xrdq','xrdy','txbcofy', 'niq', 'ppegtq', 'doq', 'doy'): pass
            else:
                query_vars += i + ', '
        query_vars = query_vars.rstrip(', ')
        g_fundq = wrds_session.raw_sql_polars_uri(f"""
            SELECT gvkey, datadate, indfmt, fyr, fyearq, fqtr, curcdq, 'GLOBAL' AS source, ibq + COALESCE(xiq, 0) AS niq, (ppentq + dpactq) AS ppegtq, NULL AS icaptq, NULL AS niy, NULL AS txditcq, NULL AS txpq, NULL AS xidoq, NULL AS xidoy, NULL AS xrdq, NULL AS xrdy, NULL AS txbcofy, {query_vars}
            FROM comp.g_fundq
            WHERE indfmt IN ('INDL', 'FS') AND datafmt = 'HIST_STD' AND popsrc = 'I' AND consol = 'C' AND datadate >= {start_date}
            """)
        g_fundq = g_fundq.with_columns([col('gvkey').cast(pl.Int64)] + [col(col_name).cast(pl.Float64) for col_name in ['icaptq','niy','txditcq','txpq','xidoq','xidoy','xrdq','xrdy','txbcofy', 'niq', 'ppegtq']])
        g_fundq.write_ipc('Raw data/g_fundq.ft')
        del g_fundq
        __gfundq =  pl.scan_ipc('Raw data/g_fundq.ft').with_columns(pl.len().over(['gvkey', 'datadate']).alias('count_indfmt'))
        __gfundq = __gfundq.filter((col('count_indfmt') == 1) | ((col('count_indfmt') == 2) & (col('indfmt') == 'INDL'))).drop(['indfmt', 'count_indfmt'])
    if coverage in ['na', 'world']:
        #preparing north american data
        #annual
        query_vars = ''
        for i in avars + avars_other:
            # Add double quotes around 'do' or any other reserved keywords
            if i == 'do': query_vars += '"' + i + '", '
            elif i in ('wcapt', 'ltdch', 'purtshr'): pass
            else: query_vars += i + ', '
        query_vars = query_vars.rstrip(', ')
        funda = wrds_session.raw_sql_polars_uri(f"""
                SELECT gvkey, datadate, curcd, 'NA' AS source, {query_vars}
                FROM comp.funda
                WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' AND datadate >= {start_date}
                """)
        funda = funda.with_columns(col('gvkey').cast(pl.Int64))
        funda.write_ipc('Raw data/funda.ft')
        del funda
        __funda =  pl.scan_ipc('Raw data/funda.ft')
        #quarterly
        query_vars = ''
        for i in qvars:
            if i in ('dvtq','gpq','dvty','gpy','ltdchy','purtshry','wcapty'): pass
            else:
                query_vars += i + ', '
        query_vars = query_vars.rstrip(', ')
        fundq = wrds_session.raw_sql_polars_uri(f"""
            SELECT gvkey, datadate, fyr, fyearq, fqtr, curcdq, 'NA' AS source, {query_vars}
            FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' AND datadate >= {start_date}
            """)
        fundq = fundq.with_columns(col('gvkey').cast(pl.Int64))
        fundq.write_ipc('Raw data/fundq.ft')
        del fundq
        __fundq =  pl.scan_ipc('Raw data/fundq.ft')
    if coverage == 'world':
        __wfunda = pl.concat([__gfunda, __funda], how = 'diagonal')
        __wfundq = pl.concat([__gfundq, __fundq], how = 'diagonal')
    else: pass
    if coverage == 'na':
        aname = __funda
        casting = [col(i).cast(pl.Float64) for i in ['wcapt', 'ltdch', 'purtshr']]
        aname = aname.with_columns(casting)
        qname = __fundq
        casting = [col(i).cast(pl.Float64) for i in ['dvtq','gpq','dvty','gpy','ltdchy','purtshry','wcapty']]
        qname = qname.with_columns(casting)
    elif coverage == 'global':
        aname = __gfunda
        casting = [col(i).cast(pl.Float64) for i in ['gp', 'pstkrv', 'pstkl', 'itcb', 'xad', 'txbcof']]
        aname = aname.with_columns(casting)
        qname = __gfundq
        casting = [col(i).cast(pl.Float64) for i in ['icaptq','niy','txditcq','txpq','xidoq','xidoy', 'xrdq','xrdy','txbcofy','doq','doy']]
        qname = qname.with_columns(casting)
    else:
        aname = __wfunda
        qname = __wfundq
    #converting to usd if required
    if convert_to_usd == 1:
        fx = compustat_fx().lazy()
        __tempa = aname.join(fx, left_on=["datadate", "curcd"], right_on=["datadate", "curcdd"], how="left").select(aname.columns+ ["fx"])
        __tempq = qname.join(fx, left_on=["datadate", "curcdq"], right_on=["datadate", "curcdd"], how="left").select(qname.columns+ ["fx"])
        __tempa = __tempa.with_columns([(col(col_name) * col('fx')).alias(col_name) for col_name in avars] + [pl.lit("USD").alias("curcd")]).drop('fx')
        __tempq = __tempq.with_columns([(col(col_name) * col('fx')).alias(col_name) for col_name in qvars] + [pl.lit("USD").alias("curcdq")]).drop('fx')
        # Rename final dataframes after conversion
        __compa1 = __tempa.clone()
        __compq1 = __tempq.clone()
    else: # Rename the dataframes
        __compa1 = aname.clone()
        __compq1 = qname.clone()
    __compq1 = __compq1.with_columns([col(column).cast(pl.Int64) for column in ["fyr", "fyearq", "fqtr"] if column in __compq1.columns])
    #quarterizing year-to-date data:
    __compq2 = quarterize(df=__compq1, var_list=qvars_y)
    __compq2 = __compq2.sort(['gvkey', 'fyr', 'fyearq', 'fqtr']).unique(['gvkey', 'fyr', 'fyearq', 'fqtr'], keep='first').sort(['gvkey', 'fyr', 'fyearq', 'fqtr'])
    #we quarterized some variables that were already available quarterized. Now just updating them if they have missing values
    remove_missing_vals = [pl.coalesce([f'{var[:-1]}q', f'{var[:-1]}y_q']).alias(f'{var[:-1]}q') for var in qvars_y if f'{var[:-1]}q' in __compq2.columns]
    rename_cols = [col(f'{var[:-1]}y_q').alias(f'{var[:-1]}q') for var in qvars_y if f'{var[:-1]}q' not in __compq2.columns]
    drop_cols = [f'{var[:-1]}y_q' for var in qvars_y]

    __compq3 = __compq2.with_columns(remove_missing_vals + rename_cols).drop(drop_cols)
    __compq3 = __compq3.with_columns(ni_qtr = col("ibq"),
                                     sale_qtr = col("saleq"),
                                     ocf_qtr = pl.coalesce(["oancfq", (col('ibq') + col('dpq') - pl.coalesce([col('wcaptq'), 0]))]))
    yrl_vars = ['cogsq', 'xsgaq', 'xintq', 'dpq', 'txtq', 'xrdq', 'dvq', 'spiq', 'saleq', 'revtq', 'xoprq', 'oibdpq', 'oiadpq', 'ibq', 'niq', 'xidoq', 'nopiq', 'miiq', 'piq', 'xiq','xidocq', 'capxq', 'oancfq', 'ibcq', 'dpcq', 'wcaptq','prstkcq', 'sstkq', 'purtshrq','dsq', 'dltrq', 'ltdchq', 'dlcchq','fincfq', 'fiaoq', 'txbcofq', 'dvtq']
    bs_vars = ['seqq', 'ceqq', 'pstkq', 'icaptq', 'mibq', 'gdwlq', 'req','atq', 'actq', 'invtq', 'rectq', 'ppegtq', 'ppentq', 'aoq', 'acoq', 'intanq', 'cheq', 'ivaoq', 'ivstq', 'ltq', 'lctq', 'dlttq', 'dlcq', 'txpq', 'apq', 'lcoq', 'loq', 'txditcq', 'txdbq']
    exps = [cumulate_4q(var_yrl) for var_yrl in yrl_vars]
    drop_cols = [a for b in [cumulate_4q(var_yrl, 'drop_cols') for var_yrl in yrl_vars] for a in b]
    new_names_bs_vars = {**dict(zip(bs_vars, list(aux[:-1] for aux in bs_vars))), **{'curcdq': 'curcd'}}
    __compq3 = __compq3.with_columns(pl.lit(None).cast(pl.Float64).alias('dsy'), pl.lit(None).cast(pl.Float64).alias('dsq'))
    __compq3 = __compq3.sort(['gvkey', 'fyr', 'fyearq', 'fqtr']).with_columns(exps).drop(drop_cols).rename(new_names_bs_vars)
    __compq4 = __compq3.sort(['gvkey', 'datadate', 'fyr', 'fqtr']).unique(['gvkey','datadate', 'fyr'], keep='first')
    #Maybe there's something weird here (missing 2 observations with respect to SAS) Flag for revision!
    __compq4 = __compq4.sort(['gvkey', 'datadate', 'fyr','fqtr']).unique(['gvkey', 'datadate'], keep='last').drop(['fyr', 'fyearq', 'fqtr'])
    __compa2 = __compa1.with_columns(ni_qtr = pl.lit(None).cast(pl.Float64),
                                     sale_qtr = pl.lit(None).cast(pl.Float64),
                                     ocf_qtr = pl.lit(None).cast(pl.Float64))
    #preparing market equity data:
    __me_data = pl.scan_ipc(me_data_path)
    c1 = col('gvkey').is_not_null()
    c2 = col('primary_sec') == 1
    c3 = col('me_company').is_not_null()
    c4 = col('common') == 1
    c5 = col('obs_main') == 1
    __me_data = __me_data.filter(c1 & c2 & c3 & c4 & c5).select(['gvkey', 'eom', col('me_company').alias('me_fiscal')]).group_by(['gvkey', 'eom']).agg(col('me_fiscal').max())
    #adding market equity data to accounting data:
    __compa3 = __compa2.join(__me_data, how = 'left', left_on=['gvkey', 'datadate'], right_on=['gvkey', 'eom'])
    __compq5 = __compq4.join(__me_data, how = 'left', left_on=['gvkey', 'datadate'], right_on=['gvkey', 'eom'])
    #standardizing annual and quarterly data
    __compa3 = __compa3.with_columns([pl.lit(None).alias('fqtr'),
                                      pl.lit(None).alias('fyearq'),
                                      pl.lit(None).alias('fyr')])
    __compq5 = __compq5.with_columns([pl.lit(None).cast(pl.Float64).alias('gp'),
                                      pl.lit(None).cast(pl.Float64).alias('dltis'),
                                      pl.lit(None).cast(pl.Float64).alias('do'),
                                      pl.lit(None).cast(pl.Float64).alias('dvc'),
                                      pl.lit(None).cast(pl.Float64).alias('ebit'),
                                      pl.lit(None).cast(pl.Float64).alias('ebitda'),
                                      pl.lit(None).cast(pl.Float64).alias('itcb'),
                                      pl.lit(None).cast(pl.Float64).alias('pstkl'),
                                      pl.lit(None).cast(pl.Float64).alias('pstkrv'),
                                      pl.lit(None).cast(pl.Float64).alias('xad'),
                                      pl.lit(None).cast(pl.Float64).alias('xlr'),
                                      pl.lit(None).cast(pl.Float64).alias('emp')])
    #adding helper variables if required:
    if include_helpers_vars==1:
        __compq6 = add_helper_vars(data=__compq5)
        __compa4 = add_helper_vars(data=__compa3)
    else:
        __compq6 = __compq5.clone()
        __compa4 = __compa3.clone()
    acc_std_ann = __compa4.unique(['gvkey', 'datadate']).sort(['gvkey', 'datadate'])
    acc_std_qtr = __compq6.unique(['gvkey', 'datadate']).sort(['gvkey', 'datadate'])
    #Do not use streaming here, gives an error. Normal collect
    acc_std_ann.collect().write_ipc('acc_std_ann.ft')
    acc_std_qtr.collect().write_ipc('acc_std_qtr.ft')
@measure_time
def expand(data, id_vars, start_date, end_date, freq='day', new_date_name='date'):
    if freq =='day': __expanded = data.with_columns(date_range = pl.date_ranges(start=start_date, end=end_date, interval='1d')).rename({"date_range": new_date_name}).explode(new_date_name).drop([start_date, end_date])
    elif freq == 'month': __expanded = data.with_columns(date_range = pl.date_ranges(start=start_date, end=end_date, interval='1mo')).rename({"date_range": new_date_name}).explode(new_date_name).with_columns(col(new_date_name).dt.month_end()).drop([start_date, end_date])
    __expanded = __expanded.unique(id_vars + [new_date_name]).sort(id_vars + [new_date_name])
    return __expanded
def sum_sas(col1, col2):
    c1 = col(col1).is_not_null()
    c2 = col(col2).is_not_null()
    return pl.when(c1 | c2).then(pl.coalesce([col1, 0.]) + pl.coalesce([col2, 0.])).otherwise(pl.lit(None).cast(pl.Float64))
def sub_sas(col1, col2):
    c1 = col(col1).is_not_null()
    c2 = col(col2).is_not_null()
    return pl.when(c1 | c2).then(pl.coalesce([col1, 0.]) - pl.coalesce([col2, 0.])).otherwise(pl.lit(None).cast(pl.Float64))
@measure_time
def add_helper_vars(data):
    __comp_dates1 = data.select(['gvkey', 'curcd', 'datadate']).group_by(["gvkey", "curcd"]).agg(col("datadate").min().alias('start_date'),col("datadate").max().alias('end_date'))
    __comp_dates2 = expand(data=__comp_dates1, id_vars=['gvkey'], start_date='start_date', end_date='end_date', freq='month', new_date_name='datadate')
    temp_data = data.with_columns((col('gvkey').is_not_null()).cast(pl.Float64).alias('data_available'))
    __helpers1 = __comp_dates2.join(temp_data, left_on=["gvkey", "curcd", "datadate"], right_on=["gvkey", "curcd", "datadate"], how="left").with_columns(col("data_available").fill_null(strategy="zero")).select(temp_data.columns)
    __helpers1 = __helpers1.sort(["gvkey", "curcd", "datadate"]).unique(["gvkey", "curcd", "datadate"], keep = 'first')
    __helpers2 = __helpers1.sort(["gvkey", "curcd", "datadate"]).with_columns(col('curcd').cum_count().over(['gvkey', 'curcd']).alias('count'))
    output = __helpers2.with_columns([pl.when(col(var)>=0).then(col(var)).otherwise(pl.lit(None).cast(pl.Float64)).alias(var) for var in ['at', 'sale', 'revt', 'dv', 'che']])
    data_with_helper_variables = output.clone()
    data_with_helper_variables = (data_with_helper_variables
    .with_columns(pl.coalesce([col("sale"), col("revt")]).alias("sale_x"))
    .with_columns([pl.coalesce([col("gp"), col("sale_x") - col("cogs")]).alias("gp_x"), pl.coalesce([col("xopr"), col("cogs") + col("xsga")]).alias("opex_x")])
    .with_columns(pl.coalesce([col("ebitda"), col("oibdp"), col("sale_x") - col("opex_x"), col("gp_x") - col("xsga")]).alias("ebitda_x"))
    .with_columns(pl.coalesce([col("ebit"), col("oiadp"), col("ebitda_x") - col("dp")]).alias("ebit_x"))
    .with_columns([(col("ebitda_x") + pl.coalesce([col("xrd"), pl.lit(0.)])).alias("op_x"), (col("ebitda_x") - col("xint")).alias("ope_x"), pl.coalesce([col("pi"), (col("ebit_x") - col("xint") + pl.coalesce([col("spi"), pl.lit(0.)]) + pl.coalesce([col("nopi"), pl.lit(0.)]))]).alias("pi_x"), pl.coalesce([col("xido"), (col("xi") + pl.coalesce([col("do"), pl.lit(0)]))]).alias("xido_x")])
    .with_columns(pl.coalesce([col("ib"), (col("ni") - col("xido_x")), (col("pi_x") - col("txt") - pl.coalesce([col("mii"), pl.lit(0)]))]).alias("ni_x"))
    .with_columns(pl.coalesce([col("ni"), (col("ni_x") + pl.coalesce([col("xido_x"), pl.lit(0.)])), (col("ni_x") + col("xi") + col("do"))]).alias("nix_x"))
    .with_columns([(col("nix_x") + col("xint")).alias("fi_x"), pl.coalesce(['dvt', 'dv']).alias("div_x")]))
    data_with_helper_variables = (data_with_helper_variables
    .with_columns([sum_sas('prstkc', 'purtshr').alias("eqbb_x"), col("sstk").alias("eqis_x")])
    .with_columns([sub_sas('eqis_x', 'eqbb_x').alias("eqnetis_x"),(col("div_x") + col("eqbb_x")).alias("eqpo_x")])
    .with_columns((col("div_x") - col("eqnetis_x")).alias("eqnpo_x"))
    .sort(['gvkey','curcd','datadate']).with_columns([pl.when((col("dltis").is_null()) & (col("dltr").is_null()) & (col("ltdch").is_null()) & (col("count") <= 12)).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.coalesce([sub_sas('dltis', 'dltr'), col("ltdch"), (col("dltt") - col("dltt").shift(12)).over(['gvkey','curcd'])])).alias("dltnetis_x"),
                                                     pl.when(col("dlcch").is_null() & (col("count") <= 12)).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.coalesce([col("dlcch"), (col("dlc") - col("dlc").shift(12)).over(['gvkey','curcd'])])).alias("dstnetis_x")])
    .with_columns(sum_sas('dstnetis_x', 'dltnetis_x').alias("dbnetis_x"))
    .with_columns((col("eqnetis_x") + col("dbnetis_x")).alias("netis_x"))
    .with_columns(pl.coalesce([col("fincf"), (col("netis_x") - col("dv") + pl.coalesce(["fiao", 0.]) + pl.coalesce(["txbcof", 0.]))]).alias("fincf_x")))
    data_with_helper_variables = (data_with_helper_variables
    .with_columns([sum_sas('dltt', 'dlc').alias("debt_x"), pl.coalesce([col("pstkrv"), col("pstkl"), col("pstk")]).alias("pstk_x")])
    .with_columns(pl.coalesce([col("seq"), col("ceq") + pl.coalesce([col("pstk_x"), pl.lit(0.)]), col("at") - col("lt")]).alias("seq_x"))
    .with_columns([pl.coalesce([col("at"), col("seq_x") + col("dltt") + pl.coalesce(col("lct"), pl.lit(0)) + pl.coalesce(col("lo"), pl.lit(0)) + pl.coalesce(col("txditc"), pl.lit(0))]).alias("at_x"),pl.coalesce([col("act"), col("rect") + col("invt") + col("che") + col("aco")]).alias("ca_x")])
    .with_columns([(col("at_x") - col("ca_x")).alias("nca_x"),pl.coalesce([col("lct"), col("ap") + col("dlc") + col("txp") + col("lco")]).alias("cl_x")])
    .with_columns((col("lt") - col("cl_x")).alias("ncl_x"))
    .with_columns([(col("debt_x") - pl.coalesce([col("che"), pl.lit(0.)])).alias("netdebt_x"),pl.coalesce([col("txditc"), sum_sas('txdb', 'itcb')]).alias("txditc_x")])
    .with_columns([(col("seq_x") + pl.coalesce([col("txditc_x"),pl.lit(0)]) - pl.coalesce([col("pstk_x"),pl.lit(0)])).alias("be_x"), pl.coalesce([col("icapt") + pl.coalesce(col("dlc"), pl.lit(0)) - pl.coalesce(col("che"), pl.lit(0)), col("netdebt_x") + col("seq_x") + pl.coalesce(col("mib"), pl.lit(0))]).alias("bev_x")])
    .with_columns([(col("ca_x") - col("che")).alias("coa_x"),(col("cl_x") - pl.coalesce(col("dlc"), pl.lit(0))).alias("col_x")])
    .with_columns((col("coa_x") - col("col_x")).alias("cowc_x"))
    .with_columns([(col("at_x") - col("ca_x") - pl.coalesce(col("ivao"), pl.lit(0))).alias("ncoa_x"), (col("lt") - col("cl_x") - col("dltt")).alias("ncol_x")])
    .with_columns((col("ncoa_x") - col("ncol_x")).alias("nncoa_x"))
    .with_columns([(pl.coalesce([col("ivst"), pl.lit(0)]) + pl.coalesce([col("ivao"), pl.lit(0)])).alias("fna_x"), (col("debt_x") + pl.coalesce([col("pstk_x"), pl.lit(0)])).alias("fnl_x")])
    .with_columns((col("fna_x") - col("fnl_x")).alias("nfna_x"))
    .with_columns([(col("coa_x") + col("ncoa_x")).alias("oa_x"), (col("col_x") + col("ncol_x")).alias("ol_x")])
    .with_columns([(col("oa_x") - col("ol_x")).alias("noa_x"), (col("ppent") + col("intan") + col("ao") - col("lo") + col("dp")).alias("lnoa_x")])
    .with_columns([pl.coalesce([col("ca_x") - col("invt"), col("che") + col("rect")]).alias("caliq_x"), (col("ca_x") - col("cl_x")).alias("nwc_x"),(col("ppegt") + col("invt")).alias("ppeinv_x"),(col("che") + 0.75 * col("coa_x") + 0.5 * (col("at_x") - col("ca_x") - pl.coalesce([col("intan"), pl.lit(0.)]))).alias("aliq_x")]))
    data_with_helper_variables = data_with_helper_variables.with_columns([pl.when(col(var)>=0).then(var).otherwise(pl.lit(None).cast(pl.Float64)).alias(var) for var in ['be_x', 'bev_x']])
    data_with_helper_variables = (data_with_helper_variables
    .sort(['gvkey','curcd','datadate']).with_columns(pl.when(col("count") <= 12).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.coalesce([col("ni_x") - col("oancf"), col("cowc_x").diff(n=12) + col("nncoa_x").diff(n=12)])).alias("oacc_x"))
    .sort(['gvkey','curcd','datadate']).with_columns(pl.when(col("count") <= 12).then(pl.lit(None).cast(pl.Float64)).otherwise(col("oacc_x") + col("nfna_x").diff(n=12)).alias("tacc_x"))
    .with_columns(pl.coalesce([col("oancf"), col("ni_x") - col("oacc_x"), col("ni_x") + col("dp") - pl.coalesce(["wcapt", 0.])]).alias("ocf_x"))
    .with_columns([(col("ocf_x") - col("capx")).alias("fcf_x"), (col("ebitda_x") + pl.coalesce([col("xrd"), pl.lit(0)]) - col("oacc_x")).alias("cop_x")]))
    data_with_helper_variables = data_with_helper_variables.drop('count')
    return data_with_helper_variables
def var_growth(var_gr, horizon):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    name_gr = f"{name_gr}_gr{int(horizon/12)}"
    name_gr_exp  = (col(var_gr) / col(var_gr).shift(horizon)) - 1
    c1 = (col(var_gr).shift(horizon) > 0) & (col("count") > horizon)
    name_gr_col = pl.when(c1).then(name_gr_exp).otherwise(pl.lit(None).cast(pl.Float64)).alias(name_gr)
    return name_gr_col
def chg_to_assets(var_gr, horizon):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    name_gr = f"{name_gr}_gr{int(horizon/12)}a"
    name_gr_exp = ((col(var_gr) - col(var_gr).shift(horizon))/col('at_x'))
    c1 = (col('at_x') > 0) & (col("count") > horizon)
    name_gr_col = pl.when(c1).then(name_gr_exp).otherwise(pl.lit(None).cast(pl.Float64)).alias(name_gr)
    return name_gr_col
def chg_to_lagassets(var_gr):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    name_gr = f"{name_gr}_gr1a"
    # Calculating the growth rate
    name_gr_exp = (col(var_gr) - col(var_gr).shift(12))/col('at_x').shift(12)
    c1 = (col('at_x').shift(12) > 0) & (col("count") > 12)
    return pl.when(c1).then(name_gr_exp).otherwise(pl.lit(None).cast(pl.Float64)).alias(name_gr)
def chg_to_exp(var):
    new_name = var.replace('_x', '')
    new_name = f"{new_name}_ce"
    c1 = (col(var).shift(12) + col(var).shift(24)) > 0
    c2 = col('count') > 24
    num = col(var)
    den = (col(var).shift(12) + col(var).shift(24)) / 2
    return pl.when(c1 & c2).then(num/den - 1).otherwise(pl.lit(None).cast(pl.Float64)).alias(new_name)
def chg_to_avgassets(var):
    new_name = var.replace('_x', '')
    new_name = f"{new_name}_gr1a"
    c1 = (col('at_x') + col('at_x').shift(12)) > 0
    c2 = col('count') > 12
    num = col(var) - col(var).shift(12)
    den = col('at_x') + col('at_x').shift(12)
    return pl.when(c1 & c2).then(num/den).otherwise(pl.lit(None).cast(pl.Float64)).alias(new_name)
def standardized_unexpected(df, var, qtrs, qtrs_min):
    name = var.replace('_x', '')
    name = f"{name}_su"
    df = df.sort(['gvkey','curcd','datadate']).with_columns((col(var) - col(var).shift(12)).over(['gvkey','curcd']).alias('__chg'))
    df = df.sort(['gvkey','curcd','datadate']).with_columns(aux = pl.concat_list([col('__chg').shift(i).over(['gvkey','curcd']) for i in range(0, (3*qtrs), 3)]).list.drop_nulls())
    df = df.with_columns(__chg_mean = col('aux').list.mean(), __chg_n = col("aux").list.len(), __chg_std = col("aux").list.eval(pl.element().std())).explode('__chg_std')
    c1 = (col('__chg_n') <= qtrs_min).fill_null(pl.lit(True).cast(pl.Boolean))
    df = df.with_columns([(pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(col("__chg_mean"))).alias('__chg_mean'), (pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(col("__chg_std"))).alias('__chg_std')])
    aux_std = pl.when(col('__chg_std').shift(3) == 0).then(pl.lit(None).cast(pl.Float64)).otherwise(col('__chg_std').shift(3))
    df = df.sort(['gvkey','curcd','datadate']).with_columns(((col(var) - col(var).shift(12) - col('__chg_mean').shift(3)) /aux_std).over(['gvkey','curcd']).alias(name))
    df = df.with_columns(pl.when(col('count') <= (12 + qtrs*3)).then(pl.lit(None).cast(pl.Float64)).otherwise(col(name)).alias(name))
    df = df.drop(['__chg', '__chg_mean', '__chg_std', '__chg_n','aux'])
    return df
def volq(df, name, var, qtrs, qtrs_min):
    df = df.sort(['gvkey','curcd','datadate']).with_columns(aux = pl.concat_list([col(var).shift(i).over(['gvkey','curcd']) for i in range(0, (3*qtrs), 3)]).list.drop_nulls())
    df = df.with_columns([col('aux').list.std().alias(name), col('aux').list.len().alias('__n')])
    df = df.with_columns(pl.when((col('count') <= ((qtrs-1)*3)) | (col('__n') < qtrs_min)).then(pl.lit(None).cast(pl.Float64)).otherwise(col(name)).alias(name)).drop(['__n','aux'])
    return df
def vola(df, name, var, yrs, yrs_min):
    df = df.sort(['gvkey','curcd','datadate']).with_columns(aux = pl.concat_list([col(var).shift(i).over(['gvkey','curcd']) for i in range(0, (12*yrs), 12)]).list.drop_nulls())
    df = df.with_columns([col('aux').list.std().alias(name), col('aux').list.len().alias('__n')])
    df = df.with_columns(pl.when((col('count') <= ((yrs-1)*12)) | (col('__n') < yrs_min)).then(pl.lit(None).cast(pl.Float64)).otherwise(col(name)).alias(name)).drop(['__n','aux'])
    return df
@measure_time
def earnings_variability(df, esm_h):
    #creating helping variables
    df = df.sort(['gvkey','curcd','datadate']).with_columns([(col('ni_x')/col('at_x').shift(12)).over(['gvkey','curcd']).alias('__roa'),(col('ocf_x')/col('at_x').shift(12)).over(['gvkey','curcd']).alias('__croa')])
    df = df.sort(['gvkey','curcd','datadate']).with_columns(aux = pl.concat_list([col('__roa').shift(i).over(['gvkey','curcd']) for i in range(0, (12*esm_h), 12)]).list.drop_nulls(), aux2 = pl.concat_list([col('__croa').shift(i).over(['gvkey','curcd']) for i in range(0, (12*esm_h), 12)]).list.drop_nulls())
    df = df.with_columns([col('aux').list.eval(pl.element().std()).alias('__roa_std'),col('aux2').list.eval(pl.element().std()).alias('__croa_std'), col("aux").list.len().alias('__roa_n'),col("aux2").list.len().alias('__croa_n')]).explode('__roa_std').explode('__croa_std')
    #calculating earning variability:
    df = df.with_columns((col('__roa_std')/col('__croa_std')).alias('earnings_variability'))
    #dealing with corner cases and deleting helping variables
    df = df.with_columns(pl.when((col('count') <= (12*esm_h)) | (col('__croa_std') <= 0) | (col('__roa_n') < esm_h) | (col('__croa_n') < esm_h)).then(pl.lit(None).cast(pl.Float64)).otherwise(col('earnings_variability')).alias('earnings_variability'))
    df = df.drop(['__roa', '__croa', '__roa_n', '__croa_n', '__roa_std', '__croa_std','aux','aux2'])
    return df
@measure_time
def equity_duration_cd(df, horizon, r, roe_mean, roe_ar1, g_mean, g_ar1):
    c1 = (col('count') <= 12).fill_null(pl.lit(False).cast(pl.Boolean))
    c2 = (col('be_x').shift(12) <=1).over(['gvkey','curcd']).fill_null(pl.lit(False).cast(pl.Boolean))
    roe0_exp = pl.when(c1|c2).then(pl.lit(None).cast(pl.Float64)).otherwise((col('ni_x')/col('be_x').shift(12)).over(['gvkey','curcd']))
    c3 = (col('count') <= 12).fill_null(pl.lit(False).cast(pl.Boolean))
    c4 = (col('sale_x').shift(12) <=1).over(['gvkey','curcd']).fill_null(pl.lit(False).cast(pl.Boolean))
    g0_exp = pl.when(c3|c4).then(pl.lit(None).cast(pl.Float64)).otherwise(((col('sale_x')/col('sale_x').shift(12))-1).over(['gvkey','curcd']))
    be0_exp = col('be_x')
    df = df.sort(['gvkey','curcd','datadate']).with_columns(__roe0 = roe0_exp, __g0 = g0_exp, __be0 = be0_exp)
    roe_c = roe_mean*(1 - roe_ar1)
    g_c = g_mean*(1 - g_ar1)
    for i in range(1, horizon+1): df = df.with_columns([(g_c + g_ar1*col(f'__g{i-1}')).alias(f'__g{i}'),(roe_c + roe_ar1*col(f'__roe{i-1}')).alias(f'__roe{i}')])
    for i in range(1, horizon+1): df = df.with_columns([(col(f'__be{i-1}') * (1 + col(f'__g{i}'))).alias(f'__be{i}'), (col(f'__be{i-1}') * (col(f'__roe{i}') - col(f'__g{i}'))).alias(f'__cd{i}')])
    df = df.with_columns([(pl.lit(horizon) + ((1 + r) / r)).alias('ed_constant'), pl.lit(0.).alias('ed_cd_w'), pl.lit(0.).alias('ed_cd'), pl.lit(0.).alias('ed_err')])
    for t in range(1, horizon+1):
         df = df.with_columns([(col('ed_cd_w') + t * col(f'__cd{t}') / ((1 + r)**t)).alias('ed_cd_w'),
                               (col('ed_cd') + col(f'__cd{t}') / ((1 + r)**t)).alias('ed_cd'),
                               pl.when(col(f'__be{t}') < 0).then(pl.lit(1.)).otherwise(col('ed_err')).alias('ed_err')])
    cols_to_drop = []
    for x in [[f'__roe{i}', f'__g{i}',f'__be{i}',f'__cd{i}'] for i in range(0,horizon+1)]:
        for y in x: cols_to_drop.append(y)
    df = df.drop(cols_to_drop)
    return df
@measure_time
def pitroski_f(df, name):
    c1 = (col('count') <= 12)
    c2 = ((col('at_x').shift(12).over(['gvkey', 'curcd']) <= 0)).fill_null(pl.lit(True).cast(pl.Boolean))
    col_exp_roa = (pl.when(c1 | c2).then(pl.lit(None).cast(pl.Float64)).otherwise((col('ni_x') / col('at_x').shift(12)).over(['gvkey', 'curcd'])).alias('__f_roa'))
    col_exp_croa = (pl.when(c1 | c2).then(pl.lit(None).cast(pl.Float64)).otherwise((col('ocf_x') / col('at_x').shift(12)).over(['gvkey', 'curcd']))).alias('__f_croa')
    col_exp_droa = (pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise((col('__f_roa') - col('__f_roa').shift(12)).over(['gvkey', 'curcd']))).alias('__f_droa')
    col_exp_acc = (col('__f_croa') - col('__f_roa')).alias('__f_acc')
    c3 = (col('at_x') <= 0).fill_null(pl.lit(False).cast(pl.Boolean))
    c4 = (col('at_x').shift(12).over(['gvkey', 'curcd']) <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    col_exp_lev = (pl.when(c1|c3|c4).then(pl.lit(None).cast(pl.Float64)).otherwise((col('dltt') / col('at_x') - (col('dltt') / col('at_x')).shift(12)).over(['gvkey', 'curcd']))).alias('__f_lev')
    c5 = (col('cl_x') <= 0).fill_null(pl.lit(False).cast(pl.Boolean))
    c6 = (col('cl_x').shift(12).over(['gvkey', 'curcd']) <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    col_exp_liq = (pl.when(c1|c5|c6).then(pl.lit(None).cast(pl.Float64)).otherwise((col('ca_x') / col('cl_x') - (col('ca_x') / col('cl_x')).shift(12)).over(['gvkey', 'curcd']))).alias('__f_liq')
    col_exp_eqis = col('eqis_x').alias('__f_eqis')
    c7 = (col('sale_x') <= 0).fill_null(pl.lit(False).cast(pl.Boolean))
    c8 = (col('sale_x').shift(12).over(['gvkey', 'curcd']) <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    col_exp_gm = (pl.when(c1|c7|c8).then(pl.lit(None).cast(pl.Float64)).otherwise((col('gp_x') / col('sale_x') - (col('gp_x') / col('sale_x')).shift(12)).over(['gvkey', 'curcd']))).alias('__f_gm')
    c9 = col('count')<=24
    c10 = (col('at_x').shift(12).over(['gvkey', 'curcd']) <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    c11 = (col('at_x').shift(24).over(['gvkey', 'curcd']) <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    col_exp_aturn = (pl.when(c9|c10|c11).then(pl.lit(None).cast(pl.Float64)).otherwise(((col('sale_x') / col('at_x').shift(12)) - (col('sale_x').shift(12) / col('at_x').shift(24))).over(['gvkey', 'curcd']))).alias('__f_aturn')
    df = df.sort(['gvkey','curcd']).with_columns([col_exp_roa, col_exp_croa])
    df = df.sort(['gvkey','curcd']).with_columns([col_exp_droa, col_exp_acc, col_exp_lev, col_exp_liq, col_exp_eqis, col_exp_gm, col_exp_aturn])
    df = (df
          .with_columns(((col('__f_roa') > 0).cast(pl.Int32) + (col('__f_croa') > 0).cast(pl.Int32) + (col('__f_droa') > 0).cast(pl.Int32) + (col('__f_acc') > 0).cast(pl.Int32) + (col('__f_lev') < 0).cast(pl.Int32) + (col('__f_liq') > 0).cast(pl.Int32) + (pl.coalesce([col('__f_eqis'), 0]) == 0).cast(pl.Int32) + (col('__f_gm') > 0).cast(pl.Int32) + (col('__f_aturn') > 0).cast(pl.Int32)).alias(name))
          .with_columns(pl.when(col('__f_roa').is_null() |col('__f_croa').is_null() |col('__f_droa').is_null() |col('__f_acc').is_null() |col('__f_lev').is_null() |col('__f_liq').is_null() |col('__f_gm').is_null() |col('__f_aturn').is_null()).then(pl.lit(None).cast(pl.Float64)).otherwise(col(name)).alias(name)))
    df = df.drop(['__f_roa', '__f_croa', '__f_droa', '__f_acc', '__f_lev', '__f_liq', '__f_eqis', '__f_gm', '__f_aturn'])
    return df
@measure_time
def ohlson_o(df, name):
    col1 = (pl.when((col('at_x') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))).then(pl.lit(None).cast(pl.Float64)).otherwise(col('at_x').log())).alias('__o_lat')
    col2 = (pl.when((col('at_x') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))).then(pl.lit(None).cast(pl.Float64)).otherwise(col('debt_x') / col('at_x'))).alias('__o_lev')
    col3 = (pl.when((col('at_x') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))).then(pl.lit(None).cast(pl.Float64)).otherwise((col('ca_x') - col('cl_x')) / col('at_x'))).alias('__o_wc')
    col4 = (pl.when((col('at_x') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))).then(pl.lit(None).cast(pl.Float64)).otherwise(col('nix_x') / col('at_x'))).alias('__o_roe')
    col5 = (pl.when((col('ca_x') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))).then(pl.lit(None).cast(pl.Float64)).otherwise(col('cl_x') / col('ca_x'))).alias('__o_cacl')
    col6 = (pl.when((col('lt')   <= 0).fill_null(pl.lit(True).cast(pl.Boolean))).then(pl.lit(None).cast(pl.Float64)).otherwise((col('pi_x') + col('dp')) / col('lt'))).alias('__o_ffo')
    col7 = (pl.when((col('lt').is_null()) | (col('at_x').is_null())).then(pl.lit(None).cast(pl.Float64)).otherwise((col('lt') > col('at_x')).cast(pl.Int32))).alias('__o_neg_eq')
    c1 = (col('count') <= 12)
    c2 = col('nix_x').is_null()
    c3 = (col('nix_x').shift(12).is_null()).over(['gvkey','curcd'])
    exp_aux = ((col('nix_x') < 0).fill_null(pl.lit(True).cast(pl.Boolean)) & (col('nix_x').shift(12) < 0).fill_null(pl.lit(True).cast(pl.Boolean))).cast(pl.Float64)
    col8 = (pl.when(c1|c2|c3).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux)).alias('__o_neg_earn')
    c4 = (col('count') <= 12)
    c5 = (((col('nix_x').abs() + col('nix_x').shift(12).abs()) == 0).over(['gvkey','curcd'])).fill_null(pl.lit(True).cast(pl.Boolean))
    exp_aux2 = ((col('nix_x') - col('nix_x').shift(12)) / (col('nix_x').abs() + col('nix_x').shift(12).abs())).over(['gvkey','curcd'])
    col9 = (pl.when(c4|c5).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux2)).alias('__o_nich')
    df = df.sort(['gvkey','curcd','datadate']).with_columns([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    df = df.with_columns((-1.32 - 0.407 * col('__o_lat') + 6.03 * col('__o_lev') + 1.43 * col('__o_wc') + 0.076 * col('__o_cacl') - 1.72 * col('__o_neg_eq') - 2.37 * col('__o_roe') - 1.83 * col('__o_ffo') + 0.285 * col('__o_neg_earn') - 0.52 * col('__o_nich')).alias(name))
    return df
@measure_time
def altman_z(df, name):
    c1 = (col('at_x') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col('lt') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    df = (df.with_columns([pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise((col('ca_x') - col('cl_x'))/col('at_x')).alias('__z_wc'),
                           pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise((col('re'))/col('at_x')).alias('__z_re'),
                           pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise((col('ebitda_x'))/col('at_x')).alias('__z_eb'),
                           pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise((col('sale_x'))/col('at_x')).alias('__z_sa'),
                           pl.when(c2).then(pl.lit(None).cast(pl.Float64)).otherwise((col('me_fiscal'))/col('lt')).alias('__z_me')])
           .with_columns((1.2 * col('__z_wc') + 1.4 * col('__z_re') + 3.3 * col('__z_eb') + 0.6 * col('__z_me') + 1.0 * col('__z_sa')).alias(name))).drop(['__z_wc', '__z_re', '__z_eb', '__z_sa', '__z_me'])
    return df
@measure_time
def intrinsic_value(df, name, r):
    c1 = col('nix_x') <= 0
    col_exp1 = (pl.when(c1).then((col('div_x')/ (col('at_x') * 0.06))).otherwise((col('div_x')/ col('nix_x')))).alias('__iv_po')
    c2 = col('count') <= 12
    c3 = ((col('be_x') + col('be_x').shift(12)).over(['gvkey','curcd']) <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    aux = (col('nix_x')/ ((col('be_x') + col('be_x').shift(12))/2)).over(['gvkey','curcd'])
    col_exp2 = pl.when(c2|c3).then(pl.lit(None).cast(pl.Float64)).otherwise(aux).alias('__iv_roe')
    df = df.sort(['gvkey','curcd','datadate']).with_columns([col_exp1, col_exp2]).with_columns(((1 + (1 - col('__iv_po')) * col('__iv_roe')) * col('be_x')).alias('__iv_be1'))
    df = df.with_columns(( col('be_x') + (((col('__iv_roe') - r)/(1+ r)) * col('be_x')) + (((col('__iv_roe') - r)/((1+ r) * r)) * col('__iv_be1'))).alias(name))
    df = df.with_columns(pl.when(col(name) <= 0).then(pl.lit(None).cast(pl.Float64)).otherwise(col(name)).alias(name)).drop(['__iv_po', '__iv_roe', '__iv_be1'])
    return df
@measure_time
def kz_index(df, name):
# Assume that __chars5 is your initial DataFrame and you have added the appropriate columns.
    c1 = (col('count') <= 12)
    c2 = ((col('ppent').shift(12) <= 0).over(['gvkey','curcd'])).fill_null(pl.lit(True).cast(pl.Boolean))
    exp_aux = ((col('ni_x') + col('dp')) / col('ppent').shift(12)).over(['gvkey','curcd'])
    col1 = (pl.when(c1|c2).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux)).alias('__kz_cf')
    c3 = (col('at_x') <= 0).fill_null(pl.lit(True).cast(pl.Boolean))
    exp_aux2 = (col('at_x') + col('me_fiscal') - col('be_x')) / col('at_x')
    col2 = (pl.when(c3).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux2)).alias('__kz_q')
    c4 = ((col('debt_x') + col('seq_x')) == 0).fill_null(pl.lit(True).cast(pl.Boolean))
    exp_aux3 = (col('debt_x') / (col('debt_x') + col('seq_x')))
    col3 = (pl.when(c4).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux3)).alias('__kz_db')
    c5 = (col('count') <= 12)
    c6 =  (col('ppent').shift(12) <= 0).over(['gvkey','curcd']).fill_null(pl.lit(True).cast(pl.Boolean))
    exp_aux4 = (col('div_x') / col('ppent').shift(12)).over(['gvkey','curcd'])
    col4 = (pl.when(c5|c6).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux4)).alias('__kz_dv')
    c7 = (col('count') <= 12)
    c8 = (col('ppent').shift(12) <= 0).over(['gvkey','curcd']).fill_null(pl.lit(True).cast(pl.Boolean))
    exp_aux5 = (col('che') / col('ppent').shift(12)).over(['gvkey','curcd'])
    col5 = (pl.when(c7|c8).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux5)).alias('__kz_cs')
    df = df.sort(['gvkey','curcd','datadate']).with_columns([col1,col2,col3,col4,col5])
    df = df.with_columns((- 1.002 * col('__kz_cf') + 0.283 * col('__kz_q') + 3.139 * col('__kz_db') - 39.368 * col('__kz_dv') - 1.315 * col('__kz_cs')).alias(name))
    return df
def chg_var1_to_var2(df, name, var1, var2, horizon):
    df = df.with_columns(__x = pl.when(col(var2) > 0).then(col(var1) / col(var2)).otherwise(pl.lit(None).cast(pl.Float64)))
    c1 = col('count') > horizon
    exp_aux = col('__x') - col('__x').shift(horizon)
    exp_chg = pl.when(c1).then(exp_aux).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
    df = df.sort(['gvkey','curcd','datadate']).with_columns(exp_chg)
    df = df.drop('__x')
    return df
@measure_time
def earnings_persistence(data, __n, __min):
    aux = data.sort(['gvkey', 'curcd', 'datadate']).with_columns(count = pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).over(['gvkey','curcd']), __ni_at = col('ni_x')/col('at_x'),__ni_at_l1 = (col('ni_x')/col('at_x')).shift(12).over(['gvkey','curcd']), month = col('datadate').dt.month())
    x ='__ni_at_l1'
    y = '__ni_at'
    beta = pl.rolling_cov(x,y,window_size=__n, min_periods = __min)/col(x).rolling_var(window_size=__n, min_periods = __min)
    alpha = col(y).rolling_mean(window_size=__n, min_periods = __min) - beta * col(x).rolling_mean(window_size=__n, min_periods = __min)
    exp1 = col(y).rolling_var(window_size=__n, min_periods = __min)
    exp2 = col(x).rolling_var(window_size=__n, min_periods = __min)
    aux = aux.sort(['gvkey','curcd','month','datadate']).with_columns([beta.over(['gvkey','curcd']).alias(f'ni_ar1'),alpha.over(['gvkey','curcd']).alias('alpha')])
    aux = aux.sort(['gvkey','curcd','month','datadate']).with_columns(((exp1 - (col('ni_ar1')**2) * exp2) ** (1/2)).over(['gvkey','curcd']).alias(('ni_ivol'))).filter(col('ni_ar1').is_not_null())
    return aux.select(['gvkey','curcd','datadate','ni_ar1','ni_ivol'])
def scale_me(var):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    # Appending '_me' to the name
    name = f'{name}_me'
    col_aux = (col(var) * col('fx'))/col('me_company')
    return pl.when(col('me_company') != 0).then(col_aux).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
def scale_mev(var):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    # Appending '_me' to the name
    name = f'{name}_mev'
    col_aux = (col(var) * col('fx'))/col('mev')
    return pl.when(col('mev') != 0).then(col_aux).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
def mean_year(var):
    return (pl.when(col(var).is_not_null() & (col(var).shift(12).over(['gvkey','curcd'])).is_not_null()).then((col(var) + col(var).shift(12)).over(['gvkey','curcd'])/2)
            .when(col(var).is_not_null()).then(col(var))
            .when((col(var).shift(12).over(['gvkey','curcd'])).is_not_null()).then(col(var).shift(12).over(['gvkey','curcd']))
            .otherwise(pl.lit(None).cast(pl.Float64)))
@measure_time
def temp_liq_rat(col_avg, den, alias):
    col1 = (365 * mean_year(col_avg)/col(den))
    c1 = col('count') > 12
    c2 = col(den) != 0
    return pl.when(c1 & c2).then(col1).otherwise(pl.lit(None).cast(pl.Float64)).alias(alias)
@measure_time
def temp_rat_other(num, den, alias):
    col_expr = (col(num) / mean_year(den))
    c1 = col('count') > 12
    c2 = mean_year(den) != 0
    return pl.when(c1 & c2).then(col_expr).otherwise(pl.lit(None).cast(pl.Float64)).alias(alias)
@measure_time
def temp_rat_other_spc():
    num_expr = col('cogs') + col('invt') - col('invt').shift(12)
    col_expr = (num_expr.over(['gvkey','curcd']) / mean_year('ap'))
    c1 = col('count') > 12
    c2 = mean_year('ap') != 0
    return pl.when(c1 & c2).then(col_expr).otherwise(pl.lit(None).cast(pl.Float64)).alias('ap_turnover')
def safe_div(num, den, name, mode = 1):
    if mode == 1:  return pl.when(col(den) != 0).then(col(num)/col(den)).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
    if mode == 2:  return pl.when(col(den) != 0).then(col(num)/(col(den).abs())).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
    if mode == 3:  return pl.when(col(den) > 0).then(col(num)/col(den)).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
    if mode == 4:
        cond1 = col('count') > 12
        cond2 = (col(den).shift(12) > 0).over(['gvkey', 'curcd'])
        col_exp = (col(num) / col(den).shift(12)).over(['gvkey', 'curcd'])
        return pl.when(cond1 & cond2).then(col_exp).otherwise(pl.lit(None).cast(pl.Float64)).alias(name)
def update_ni_inc_and_decrease(df, lag):
    c1 = (col('ni_inc').shift(lag) == 1) & (col('no_decrease') == 1)
    ni_inc8q_updated_exp = pl.when(c1).then(col('ni_inc8q') + 1).otherwise(col('ni_inc8q')).alias('ni_inc8q')
    no_decrease_updated_exp = pl.when(c1).then(col('no_decrease')).otherwise(pl.lit(0)).alias('no_decrease')
    return df.sort(['gvkey','curcd','datadate']).with_columns([ni_inc8q_updated_exp, no_decrease_updated_exp])
@measure_time
def calculate_consecutive_earnings_increases(df):
    ni_inc_exp = (pl.when(col('ni_x') > col('ni_x').shift(12)).then(pl.lit(1).cast(pl.Int64))
                    .when(col('ni_x').is_null()).then(pl.lit(None).cast(pl.Int64))
                    .when(col('ni_x').shift(12).is_null()).then(pl.lit(None).cast(pl.Int64))
                    .otherwise(pl.lit(0).cast(pl.Int64)).alias('ni_inc'))
    ni_inc8q_exp = pl.lit(0).alias('ni_inc8q')
    no_decrease_exp = pl.lit(1).alias('no_decrease')
    df = df.sort(['gvkey','curcd','datadate']).with_columns([ni_inc_exp, ni_inc8q_exp, no_decrease_exp])
    for i in range(8): df = update_ni_inc_and_decrease(df, 3*i)
    n_ni_inc_exp = col('ni_inc').is_not_null()
    for i in range(1, 8): n_ni_inc_exp += col('ni_inc').shift(3*i).is_not_null()
    df = df.sort(['gvkey','curcd','datadate']).with_columns(n_ni_inc = n_ni_inc_exp)
    c1 = col('ni_inc').is_null()
    c2 = (col('n_ni_inc') != 8).fill_null(pl.lit(False).cast(pl.Boolean))
    c3 = col('count') < 33
    ni_inc8q_exp_final = pl.when(c1 | c2 | c3).then(pl.lit(None)).otherwise(col('ni_inc8q'))
    df = df.with_columns(ni_inc8q = ni_inc8q_exp_final).drop(['ni_inc','no_decrease', 'n_ni_inc'])
    return df
def compute_capex_abn(df):
    df = df.with_columns(safe_div('capx', 'sale_x', '__capex_sale', 3))
    c1 = (col('__capex_sale').shift(12) + col('__capex_sale').shift(24) + col('__capex_sale').shift(36)) != 0
    c2 = col('count') > 36
    num = col('__capex_sale')
    den = (col('__capex_sale').shift(12) + col('__capex_sale').shift(24) + col('__capex_sale').shift(36)) / 3
    capex_abn_exp = pl.when(c1 & c2).then(num/den - 1).otherwise(pl.lit(None).cast(pl.Float64)).alias('capex_abn')
    df = df.sort(['gvkey', 'curcd', 'datadate']).with_columns(capex_abn_exp).drop('__capex_sale')
    return df
@measure_time
def create_acc_chars(data_path, output_path, lag_to_public, max_data_lag, __keep_vars, me_data_path, suffix):
    data = pl.scan_ipc(data_path)
    #sorting the data
    __chars3 = data.sort(['gvkey', 'curcd', 'datadate'])
    #adding a count column that keeps a count of the number of the obs for a given gvkey (and curcd)
    __chars4 = __chars3.sort(['gvkey', 'curcd', 'datadate']).with_columns(col('gvkey').cum_count().over(['gvkey', 'curcd']).alias('count'))
    #accounting based size measures
    __chars5 = __chars4.with_columns([col("at_x").alias("assets"),col("sale_x").alias("sales"),col("be_x").alias("book_equity"),col("ni_x").alias("net_income")])
    #growth characteristics
    growth_vars = [
    "at_x", "ca_x", "nca_x",                 # Assets - Aggregated
    "lt", "cl_x", "ncl_x",                   # Liabilities - Aggregated
    "be_x", "pstk_x", "debt_x",              # Financing Book Values
    "sale_x", "cogs", "xsga", "opex_x",      # Sales and Operating Costs
    "capx", "invt"]
    ch_asset_vars = [
    "che", "invt", "rect", "ppegt", "ivao", "ivst", "intan",# Assets - Individual Items
    "dlc", "ap", "txp", "dltt", "txditc",# Liabilities - Individual Items
    "coa_x", "col_x", "cowc_x", "ncoa_x", "ncol_x", "nncoa_x", "oa_x", "ol_x",# Operating Assets/Liabilities
    "fna_x", "fnl_x", "nfna_x",# Financial Assets/Liabilities
    "gp_x", "ebitda_x", "ebit_x", "ope_x", "ni_x", "nix_x", "dp",# Income Statement
    "fincf_x", "ocf_x", "fcf_x", "nwc_x",# Aggregated Cash Flow
    "eqnetis_x", "dltnetis_x", "dstnetis_x", "dbnetis_x", "netis_x", "eqnpo_x",# Financing Cash Flow
    "txt",# Tax Change
    "eqbb_x", "eqis_x", "div_x", "eqpo_x",# Financing Cash Flow
    "capx", "be_x"]# Other
    #1-yr growth,  3-yr growth, 1yr Change Scaled by Assets & 3yr Change Scaled by Assets
    grt1 = [var_growth(i, 12) for i in growth_vars]
    grt3 = [var_growth(i, 36) for i in growth_vars]
    chg_at1 = [chg_to_assets(i, 12) for i in ch_asset_vars]
    chg_at3 = [chg_to_assets(i, 36) for i in ch_asset_vars]
    #Investment Measure & Non-Recurring Items & Profitability margins
    c_at_sale = [safe_div('capx', 'at_x', 'capx_at'),
                 safe_div('xrd', 'at_x', 'rd_at'),
                 safe_div('spi', 'at_x', 'spi_at'),
                 safe_div('xido_x', 'at_x', 'xido_at'),
                 pl.when(col('at_x') != 0).then((col('spi') + col('xido_x'))/col('at_x')).otherwise(pl.lit(None).cast(pl.Float64)).alias('nri_at'),
                 safe_div('gp_x', 'sale_x', 'gp_sale'),
                 safe_div('ebitda_x', 'sale_x', 'ebitda_sale'),
                 safe_div('ebit_x', 'sale_x', 'ebit_sale'),
                 safe_div('pi_x', 'sale_x', 'pi_sale'),
                 safe_div('ni_x', 'sale_x', 'ni_sale'),
                 safe_div('ni', 'sale_x', 'nix_sale'),
                 safe_div('ocf_x', 'sale_x', 'ocf_sale'),
                 safe_div('fcf_x', 'sale_x', 'fcf_sale')]
    #Return on assets:
    c_ret_at = [safe_div(f'{i}_x', 'at_x', f'{i}_at') for i in ['gp', 'ebitda', 'ebit', 'fi', 'cop', 'ni']]
    #Return on book equity:
    c_ret_be = [safe_div(f'{i}_x', 'be_x', f'{i}_be') for i in ['ope', 'ni', 'nix', 'ocf', 'fcf']]
    #Return on invested book capital:
    c_ret_bev = [safe_div(f'{i}_x', 'bev_x', f'{i}_bev') for i in ['gp', 'ebitda', 'ebit', 'fi', 'cop']]
    #Return on Physical Capital:
    c_ret_ppent = [safe_div(f'{i}_x', 'ppent', f'{i}_ppen') for i in ['gp', 'ebitda', 'fcf']]
    #Issuance Variables & Equity Payout
    aux_iss_eqp = ['fincf', 'netis', 'eqnetis', 'eqis', 'dbnetis', 'dltnetis', 'dstnetis', 'eqnpo', 'eqbb', 'div']
    c_iss_eqp = [safe_div(f'{i}_x', 'at_x', f'{i}_at') for i in aux_iss_eqp]
    #Solvency Ratios: Debt-to-assets, debt to shareholders' equity ratio, interest coverage ratio
    c_solv_rat = [safe_div('debt_x', 'at_x', 'debt_at'), safe_div('debt_x', 'be_x', 'debt_be'), safe_div('ebit_x', 'xint', 'ebit_int')]
    #Capitalization/Leverage Ratios Book:
    c_cap_lev = [safe_div(f'{i}_x', 'bev_x', f'{i}_bev') for i in ['be', 'debt', 'pstk']] +\
                [safe_div('che', 'bev_x', 'cash_bev'), safe_div('dltt', 'bev_x', 'debtlt_bev'),safe_div('dlc', 'bev_x', 'debtst_bev')]
    acc_columns = grt1 + grt3 + chg_at1 + chg_at3 + c_at_sale + c_ret_at + c_ret_be + c_ret_bev + c_ret_ppent + c_iss_eqp + c_solv_rat + c_cap_lev
    __chars5 = __chars5.sort(['gvkey', 'curcd', 'datadate']).with_columns(acc_columns)
    #accruals:
    __chars5 = (__chars5
            .with_columns([safe_div('oacc_x', 'at_x', 'oaccruals_at'), safe_div('tacc_x', 'at_x', 'taccruals_at'),
                           safe_div('oacc_x', 'nix_x', 'oaccruals_ni', 2), safe_div('tacc_x', 'nix_x', 'taccruals_ni', 2)])
            .sort(['gvkey', 'curcd', 'datadate']).with_columns(((col("noa_x")/col('at_x').shift(12)).over(['gvkey','curcd'])).alias("noa_at")) #Net Operating Asset to Total Assets
            .sort(['gvkey', 'curcd', 'datadate']).with_columns(
                pl.when((col("count") <= 12) | ((col("at_x").shift(12)).over(['gvkey','curcd']) <= 0).fill_null(pl.lit(True).cast(pl.Boolean)))
                .then(pl.lit(None).cast(pl.Float64)).otherwise(col("noa_at")).alias("noa_at")))

    #Financial Soundness Ratios:
    c_fin_s_rat = [safe_div('xint', 'debt_x', 'int_debt'), safe_div('ocf_x', 'debt_x', 'ocf_debt'), safe_div('ebitda_x', 'debt_x', 'ebitda_debt'), safe_div('dlc', 'debt_x', 'debtst_debt'), safe_div('dltt', 'debt_x', 'debtlt_debt'),\
                   safe_div('xint', 'dltt', 'int_debtlt'), safe_div('ebitda_x', 'cl_x', 'profit_cl'), safe_div('ocf_x', 'cl_x', 'ocf_cl'), safe_div('che', 'lt', 'cash_lt'), safe_div('cl_x', 'lt', 'cl_lt'),\
                   safe_div('invt', 'act', 'inv_act'), safe_div('rect', 'act', 'rec_act'), safe_div('opex_x', 'at_x', 'opex_at'), safe_div('nwc_x', 'at_x', 'nwc_at'),\
                   safe_div('lt', 'ppent', 'lt_ppen'), safe_div('dltt', 'be_x', 'debtlt_be'), safe_div('fcf_x', 'ocf_x', 'fcf_ocf', 3)]
    __chars5 = __chars5.with_columns(c_fin_s_rat)
    #Liquidity Ratios:
    #Days Inventory Outstanding, Days Sales Outstanding, Days Accounts Payable Outstanding
    c_days = [temp_liq_rat('invt','cogs','inv_days'), temp_liq_rat('rect','sale_x','rec_days'), temp_liq_rat('ap','cogs','ap_days')]
    #Cash, quick, and current ratios; cash Conversion Cycle
    cond_cash = (col('inv_days') + col('rec_days') - col('ap_days')) > 0
    exp_cash = (col('inv_days') + col('rec_days') - col('ap_days'))
    c_liq_rat = [safe_div('che', 'cl_x', 'cash_cl', 3), safe_div('caliq_x', 'cl_x', 'caliq_cl', 3), safe_div('ca_x', 'cl_x', 'ca_cl', 3),\
                 pl.when(cond_cash).then(exp_cash).otherwise(pl.lit(None).cast(pl.Float64)).alias('cash_conversion')]
    __chars5 = __chars5.sort(['gvkey','curcd','datadate']).with_columns(c_days)
    __chars5 = __chars5.with_columns(c_liq_rat)
    #Activity/Efficiency Ratios:
    __chars5 = __chars5.sort(['gvkey','curcd','datadate']).with_columns([temp_rat_other('cogs', 'invt', 'inv_turnover'),
                                                                         temp_rat_other('sale_x', 'at_x', 'at_turnover'),
                                                                         temp_rat_other('sale_x', 'rect', 'rec_turnover'),
                                                                         temp_rat_other_spc()])

    #Miscellaneous Ratios
    c1 = (pl.coalesce('nix_x', 'ni_x') > 0.) & (col('nix_x') != 0)
    div_ni_exp = col("div_x") / col("nix_x")
    __chars5 = __chars5.with_columns([safe_div('xad','sale_x','adv_sale'),
                                       safe_div('xlr','sale_x','staff_sale'),
                                       safe_div('sale_x', 'bev_x', 'sale_bev'),
                                       safe_div('xrd', 'sale_x', 'rd_sale'),
                                       safe_div('sale_x', 'be_x', 'sale_be'),
                                       safe_div('sale_x', 'nwc_x', 'sale_nwc', 3),
                                       safe_div('txt', 'pi_x', 'tax_pi', 3),
                                       pl.when(c1).then(div_ni_exp).otherwise(pl.lit(None).cast(pl.Float64)).alias("div_ni")])

    #New variables, emp not available in quarterly data created by sas code.
    __chars5 = __chars5.with_columns([safe_div('che', 'at_x', 'cash_at', 3),
                                      safe_div('ni_x', 'emp', 'ni_emp', 3),
                                      safe_div('sale_x', 'emp', 'sale_emp', 3)])

    aux_col = ((col('sale_emp') / col('sale_emp').shift(12)) - 1).over(['gvkey','curcd'])
    c1 = (col('count') > 12) & (col('sale_emp').shift(12).over(['gvkey','curcd']) > 0)
    aux_col2 = pl.when(c1).then(aux_col).otherwise(pl.lit(None).cast(pl.Float64)).alias('sale_emp_gr1')
    __chars5 = __chars5.sort(['gvkey', 'curcd', 'datadate']).with_columns(aux_col2)

    col_expr = ((col('emp') - col('emp').shift(12)) / (0.5 * col('emp') + 0.5 * col('emp').shift(12))).over(['gvkey','curcd'])
    c1 = (col('count') <= 12)
    c2 = (col_expr == 0)
    c3 = ((0.5 * col('emp') + 0.5 * col('emp').shift(12)) == 0).over(['gvkey','curcd']).fill_null(pl.lit(True).cast(pl.Boolean))
    col_expr_2 = pl.when(c1|c2|c3).then(pl.lit(None).cast(pl.Boolean)).otherwise(col_expr).alias('emp_gr1')
    if data_path == 'acc_std_qtr.ft': __chars5 = __chars5.with_columns(pl.lit(None).cast(pl.Float64).alias('emp_gr1'))
    else: __chars5 = __chars5.sort(['gvkey', 'curcd', 'datadate']).with_columns(col_expr_2)

    aux_col = ((col('ni_x') > col('ni_x').shift(12)).over(['gvkey','curcd']).fill_null(pl.lit(False).cast(pl.Boolean))).cast(pl.Int64)
    #Number of Consecutive Earnings Increases:
    __chars5 = calculate_consecutive_earnings_increases(__chars5)
    c_ch_asset_lag_vars = [chg_to_lagassets(i) for i in ['noa_x', 'ppeinv_x']]      #1yr Change Scaled by Lagged Assets
    c_ch_asset_avg_vars = [chg_to_avgassets(i) for i in  ['lnoa_x']]                #1yr Change Scaled by Average Assets
    c_capex_vars = [var_growth(var_gr='capx', horizon=24)]                          #CAPEX growth over 2 years
    __chars5 = __chars5.sort(['gvkey', 'curcd', 'datadate']).with_columns(c_ch_asset_lag_vars + c_ch_asset_avg_vars + c_capex_vars)
    c1 = (col('count') <= 12)
    c2 = (col('sale_qtr').shift(12) <= 0).over(['gvkey','curcd'])
    c3 = (col('count') <= 3)
    c4 = (col('be_x').shift(3) <= 0).over(['gvkey','curcd'])
    c5 = (col('count') <= 3)
    c6 = (col('at_x').shift(3) <= 0).over(['gvkey','curcd'])
    exp_aux1 = ((col('sale_qtr') / col('sale_qtr').shift(12)) - 1).over(['gvkey','curcd'])
    exp_aux2 = (col('ni_qtr') / col('be_x').shift(3)).over(['gvkey','curcd'])
    exp_aux3 = (col('ni_qtr') / col('at_x').shift(3)).over(['gvkey','curcd'])
    exp_aux4 = (col('niq_be') - col('niq_be').shift(12)).over(['gvkey','curcd'])
    exp_aux5 = (col('niq_at') - col('niq_at').shift(12)).over(['gvkey','curcd'])
    #Quarterly profitability measures:
    __chars5 = (__chars5
               .sort(['gvkey', 'curcd', 'datadate']).with_columns([pl.when(c1 | c2).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux1).alias('saleq_gr1'),
                                                                   pl.when(c3 | c4).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux2).alias('niq_be'),
                                                                   pl.when(c5 | c6).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux3).alias('niq_at')])
               .sort(['gvkey', 'curcd', 'datadate']).with_columns([pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux4).alias('niq_be_chg1'),
                                                                   pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(exp_aux5).alias('niq_at_chg1')]))
    #R&D capital-to-assets
    __chars5 = (__chars5
               .sort(['gvkey', 'curcd', 'datadate']).with_columns(((col('xrd') + col('xrd').shift(12) * 0.8 + col('xrd').shift(24) * 0.6 + col('xrd').shift(36) * 0.4 + col('xrd').shift(48) * 0.2) / col('at_x')).over(['gvkey','curcd']).alias('rd5_at'))
               .with_columns(pl.when((col('count') <= 48) | (col('at_x') <= 0)).then(pl.lit(None).cast(pl.Float64)).otherwise(col('rd5_at')).alias('rd5_at')))
    #Abarbanell and Bushee (1998)
    ch_asset_AandB = ['sale_x', 'invt', 'rect', 'gp_x', 'xsga']
    __chars5 = __chars5.sort(['gvkey', 'curcd', 'datadate']).with_columns([chg_to_exp(i) for i in ch_asset_AandB])
    __chars5 = (__chars5
               .with_columns((col('sale_ce') - col('invt_ce')).alias('dsale_dinv'))
               .with_columns((col('sale_ce') - col('rect_ce')).alias('dsale_drec'))
               .with_columns((col('gp_ce') - col('sale_ce')).alias('dgp_dsale'))
               .with_columns((col('sale_ce') - col('xsga_ce')).alias('dsale_dsga'))
              ).drop(['sale_ce', 'invt_ce', 'rect_ce', 'gp_ce', 'xsga_ce'])
    #Earnings and Revenue 'Surprise'
    __chars5 = standardized_unexpected(df=__chars5, var='sale_qtr', qtrs=8, qtrs_min=6)
    __chars5 = standardized_unexpected(df=__chars5, var='ni_qtr', qtrs=8, qtrs_min=6)
    #Abnormal Corporate Investment
    __chars5 = compute_capex_abn(__chars5)
    #Profit scaled by lagged
    __chars5 = __chars5.sort(['gvkey', 'curcd', 'datadate']).with_columns([safe_div('op_x', 'at_x', 'op_atl1', 4),
                                                                           safe_div('gp_x', 'at_x', 'gp_atl1', 4),
                                                                           safe_div('ope_x', 'be_x', 'ope_bel1', 4),
                                                                           safe_div('cop_x', 'at_x', 'cop_atl1', 4)])
    #Profitability Measures
    __chars5 = (__chars5
               .with_columns((col('pi_x') / col('nix_x')).alias('pi_nix'))
               .with_columns(pl.when((col('pi_x') <= 0) | (col('nix_x') <= 0)).then(pl.lit(None).cast(pl.Float64)).otherwise(col('pi_nix')).alias('pi_nix'))
               .with_columns([safe_div('ocf_x', 'at_x', 'ocf_at'), safe_div('op_x', 'at_x', 'op_at', 3)])
               .sort(['gvkey', 'curcd', 'datadate']).with_columns((col('ocf_at') - col('ocf_at').shift(12)).over(['gvkey','curcd']).alias('ocf_at_chg1'))
               .with_columns(pl.when(col('count') <= 12).then(pl.lit(None).cast(pl.Float64)).otherwise(col('ocf_at_chg1')).alias('ocf_at_chg1')))
    #Book Leverage & Volatility Quarterly Items
    c_bl_volq = [safe_div('at_x', 'be_x', 'at_be'), safe_div('ocf_qtr', 'sale_qtr', '__ocfq_saleq', 3), safe_div('ni_qtr', 'sale_qtr', '__niq_saleq', 3), safe_div('ni_qtr', 'be_x', '__roeq', 3),]
    __chars5 = __chars5.with_columns(c_bl_volq)
    __chars5 = volq(df=__chars5, name='ocfq_saleq_std', var='__ocfq_saleq', qtrs=16, qtrs_min=8)
    __chars5 = volq(df=__chars5, name='niq_saleq_std', var='__niq_saleq', qtrs=16, qtrs_min=8)
    __chars5 = volq(df=__chars5, name='roeq_be_std', var='__roeq', qtrs=20, qtrs_min=12)
    __chars5 = __chars5.drop(['__ocfq_saleq', '__niq_saleq', '__roeq'])
    #Volatility Annual Items:
    __chars5 = __chars5.with_columns(safe_div('ni_x', 'be_x', '__roe', 3))
    __chars5 = vola(df=__chars5, name='roe_be_std', var='__roe', yrs=5, yrs_min=5).drop('__roe')
    __chars5 = __chars5.with_columns(((col('che') + 0.715 * col('rect') + 0.547 * col('invt') + 0.535 * col('ppegt'))/ col('at_x')).alias('tangibility'))
    #Earnings Smoothness
    __chars5 = earnings_variability(df=__chars5, esm_h=5)
    #Asset Liquidity:
    __chars5 = (__chars5
               .sort(['gvkey','curcd','datadate']).with_columns((col('aliq_x') / col('at_x').shift(12)).over(['gvkey','curcd']).alias('aliq_at'))
               .sort(['gvkey','curcd','datadate']).with_columns(pl.when((col('count') <= 12) | ((col('at_x').shift(12) <=0).over(['gvkey','curcd']))).then(pl.lit(None).cast(pl.Float64)).otherwise(col('aliq_at')).alias('aliq_at')))
    #Equity Duration
    __chars5 = equity_duration_cd(df=__chars5, horizon=10, r=0.12, roe_mean=0.12, roe_ar1=0.57, g_mean=0.06, g_ar1=0.24)
    #F-score
    __chars5 = pitroski_f(df=__chars5, name='f_score')
    #O-score
    __chars5 = ohlson_o(df=__chars5, name='o_score')
    #Z-score
    __chars5 = altman_z(df=__chars5, name='z_score')
    #Invtrinsics value
    __chars5 = intrinsic_value(df= __chars5, name ='intrinsic_value', r=0.12)
    #Kz-index
    __chars5 = kz_index(df= __chars5, name ='kz_index')
    #5 year ratio change (For quality minus junk variables)
    __chars5 = chg_var1_to_var2(df=__chars5, name='gpoa_ch5', var1='gp_x', var2='at_x', horizon=60)
    __chars5 = chg_var1_to_var2(df=__chars5, name='roe_ch5', var1='ni_x', var2='be_x', horizon=60)
    __chars5 = chg_var1_to_var2(df=__chars5, name='roa_ch5', var1='ni_x', var2='at_x', horizon=60)
    __chars5 = chg_var1_to_var2(df=__chars5, name='cfoa_ch5', var1='ocf_x', var2='at_x', horizon=60)
    __chars5 = chg_var1_to_var2(df=__chars5, name='gmar_ch5', var1='gp_x', var2='sale_x', horizon=60)

    #delete helper variables
    __chars5 = __chars5.drop('count')
    #Earning's persistence
    earnings_pers = earnings_persistence(data=__chars5.select(['gvkey', 'curcd', 'datadate', 'ni_x', 'at_x']), __n=5, __min=5).sort(['gvkey','datadate','curcd'])
    __chars6 = __chars5.join(earnings_pers, on=['gvkey', 'curcd', 'datadate'], how='left')
    #Keep only dates with accounting data
    __chars7  = __chars6.filter(col('data_available')==1).sort(["gvkey",'curcd', "datadate"]).unique(['gvkey','curcd','datadate'], keep = 'first').sort(["gvkey",'curcd', "datadate"])#Needed to remove duplicates for expand to work
    #lagging for public availability of data
    __chars8 =  __chars7.with_columns(col('datadate').dt.offset_by(f'{lag_to_public}mo').dt.month_end().alias('start_date')).sort(["gvkey", "datadate"])
    __chars8 =  __chars8.with_columns(col('start_date').shift(-1).over(['gvkey']).alias('next_start_date'))
    __chars8 =  __chars8.with_columns(pl.min_horizontal((col('next_start_date').dt.offset_by('-1mo').dt.month_end()),(col('datadate').dt.offset_by(f'{max_data_lag}mo').dt.month_end())).alias('end_date'))
    __chars8 = __chars8.drop('next_start_date')
    __chars9 = expand(data=__chars8, id_vars=['gvkey'], start_date='start_date', end_date='end_date', freq='month', new_date_name='public_date')

    #fx datadate in the new code is the same as date in SAS code
    fx = compustat_fx().rename({'datadate': 'date'}).lazy()
    #Convert All Raw (non-scaled) Variables to USD
    __chars10 = __chars9.join(fx, left_on=['curcd', 'public_date'], right_on=['curcdd', 'date'], how='left').select(__chars9.columns + ['fx'])
    var_raw = ['assets', 'sales', 'book_equity', 'net_income']
    __chars11 = __chars10.with_columns([(col(i)*col('fx')).alias(i) for i in var_raw]).drop('curcd')

    #adding and filtering market return data
    me_data = pl.scan_ipc(me_data_path)
    __me_data1 = me_data.filter(col("gvkey").is_not_null()).filter(col("primary_sec") == 1).filter(col("me_company").is_not_null()).filter(col("common") == 1).filter(col("obs_main") == 1).select(['gvkey', 'eom', 'me_company']).unique().group_by(["gvkey", "eom"]).agg(col("me_company").max())
    __chars12 = __chars11.join(__me_data1, left_on=['gvkey', 'public_date'], right_on=['gvkey', 'eom'], how='left').select(__chars11.columns + ['me_company'])
    __chars13 = __chars12.sort(['gvkey', 'public_date']).unique(['gvkey', 'public_date'], keep = 'first')

    #Create Ratios using both Accounting and Market Value
    __chars14 = __chars13.with_columns(mev = col('me_company') + col('netdebt_x') * col('fx'),
                                       mat = col('at_x') * col('fx') - col('be_x') * col('fx') + col('me_company'))
    __chars14 = __chars14.with_columns(mev = pl.when(col('mev') > 0).then(col('mev')).otherwise(pl.lit(None).cast(pl.Float64)),
                                       me_company = pl.when(col('me_company') > 0).then(col('me_company')).otherwise(pl.lit(None).cast(pl.Float64)),
                                       mat = pl.when(col('mat') > 0).then(col('mat')).otherwise(pl.lit(None).cast(pl.Float64)))

    #Characteristics Scaled by Market Equity
    me_vars = ["at_x", "be_x", "debt_x", "netdebt_x", "che", "sale_x", "gp_x", "ebitda_x","ebit_x", "ope_x", "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "div_x","eqbb_x", "eqis_x", "eqpo_x", "eqnpo_x", "eqnetis_x", "xrd"]
    __chars14 = __chars14.with_columns([scale_me(i) for i in me_vars])
    __chars14 = (__chars14.with_columns(((col('intrinsic_value') * col('fx')) / (col('me_company'))).alias('ival_me')))
    #Characteristics Scaled by Market Enterprise Value
    mev_vars = ["at_x", "bev_x", "ppent", "be_x", "che", "sale_x", "gp_x", "ebitda_x","ebit_x", "ope_x", "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "debt_x","pstk_x", "dltt", "dlc", "dltnetis_x", "dstnetis_x", "dbnetis_x", "netis_x", "fincf_x"]
    __chars14 = __chars14.with_columns([scale_mev(i) for i in mev_vars])
    #Characteristics Scaled by Market Assets
    __chars14 = (__chars14
        .sort(['gvkey','public_date']).with_columns(((col('aliq_x') * col('fx')) / (col('mat').shift(12))).over(['gvkey']).alias('aliq_mat'))
        .sort(['gvkey','public_date']).with_columns(pl.when((col('gvkey') != col('gvkey').shift(12)).over(['gvkey']).fill_null(pl.lit(False).cast(pl.Boolean))).then(pl.lit(None).cast(pl.Float64)).otherwise(col('aliq_mat')).alias('aliq_mat')))
    #Size Measure
    __chars14 = __chars14.with_columns((col('mev')).alias('enterprise_value'))
    #Equity Duration
    __chars14 = (__chars14.with_columns(((col('ed_cd_w') * col('fx')) / (col('me_company')) + col('ed_constant') * (col('me_company') - col('ed_cd') * col('fx'))/col('me_company')).alias('eq_dur'))
                .with_columns(pl.when((col('ed_err') ==1) | (col('eq_dur') <=0)).then(None).otherwise(col('eq_dur')).alias('eq_dur')))
    #renaming columns:
    __chars15 = __chars14
    rename_dict = {"xrd": "rd","xsga": "sga","dlc": "debtst","dltt": "debtlt","oancf": "ocf","ppegt": "ppeg","ppent": "ppen","che": "cash","invt": "inv","rect": "rec","txt": "tax","ivao": "lti","ivst": "sti","sale_qtr": "saleq","ni_qtr": "niq","ocf_qtr": "ocfq"}
    new_names = {}
    for  i in sorted(__chars15.columns):
        col_name = i
        for a, b in rename_dict.items():
            col_name = col_name.replace(a,b,1)
        new_names[i] = col_name
    __chars15 = __chars15.rename(new_names)
    #selecting variable columns of interest
    __chars16 = __chars15.select(['source', 'gvkey', 'public_date', 'datadate'] + __keep_vars)
    #addinf sufiix if mentioned
    if suffix is None: __chars16 = __chars16
    else: __chars16 = __chars16.rename({i: (i+suffix) for i in __keep_vars})
    output = __chars16.sort(['gvkey', 'public_date']).unique(['gvkey', 'public_date'],keep = 'first').sort(['gvkey', 'public_date'])
    output.collect().write_ipc(output_path)
@measure_time
def combine_ann_qtr_chars(ann_df_path, qtr_df_path, char_vars, q_suffix):
    ann_df = pl.scan_ipc(ann_df_path)
    qtr_df = pl.scan_ipc(qtr_df_path)
    combined_df = ann_df.join(qtr_df, on=['gvkey', 'public_date'], how='left', suffix=q_suffix)
    updated_cols = []
    drop_cols = []
    # Define the logic to update annual data with quarterly data if it is more recent
    for char_var in char_vars:
        c1 = col(char_var).is_null()
        c2 = (col(f"{char_var}{q_suffix}").is_not_null()) & (col(f"datadate{q_suffix}") > col('datadate'))
        updated_cols.append(pl.when(c1|c2).then(col(f"{char_var}{q_suffix}")).otherwise(col(char_var)).alias(char_var))
        drop_cols.append(f"{char_var}{q_suffix}")
    # Drop the quarterly variable after the update and the no longer needed 'datadate' fields
    combined_df = combined_df.with_columns(updated_cols).drop(drop_cols)
    combined_df = combined_df.drop(['datadate', f'datadate{q_suffix}'])
    # Remove duplicates based on 'gvkey' and 'public_date' and sort the DataFrame
    combined_df = combined_df.unique(subset=['gvkey', 'public_date']).sort(['gvkey', 'public_date'])
    combined_df.collect(streaming = True).write_ipc('acc_chars_world.ft')
@measure_time
def seasonality(df, ret_x, start_year, end_year):
    data=df
    #return over all lags
    for i in range((start_year-1)*12, (end_year*12)):
        if i == (start_year-1)*12: data = data.sort(['id','eom']).with_columns(col(ret_x).shift(i).over('id').alias('__all_ret')).sort(['id','eom'])
        else: data = data.with_columns((col('__all_ret') + col(ret_x).shift(i).over('id')).alias('__all_ret')).sort(['id','eom'])
    #return over annual lags
    for i in range((start_year*12-1), (end_year*12), 12):
        if i == (start_year*12-1): data = data.sort(['id','eom']).with_columns(col(ret_x).shift(i).over('id').alias('__an_ret')).sort(['id','eom'])
        else: data = data.with_columns((col('__an_ret') + col(ret_x).shift(i).over('id')).alias('__an_ret')).sort(['id','eom'])
    data = data.with_columns((col('__all_ret') - col('__an_ret')).alias('__na_ret'))
    #creating variables:
    data = data.with_columns([(col('__an_ret')/len(range((start_year*12-1), (end_year*12), 12))).alias(f'seas_{start_year}_{end_year}an'),
                             (col('__na_ret')/(len(range((start_year-1)*12, (end_year*12))) - len(range((start_year*12-1), (end_year*12), 12)))).alias(f'seas_{start_year}_{end_year}na')])
    #correcting for corner cases
    data = data.with_columns([pl.when(col("count") <= (end_year*12)).then(pl.lit(None).cast(pl.Float64)).otherwise(col(f'seas_{start_year}_{end_year}an')).alias(f'seas_{start_year}_{end_year}an'),
                              pl.when(col("count") <= (end_year*12)).then(pl.lit(None).cast(pl.Float64)).otherwise(col(f'seas_{start_year}_{end_year}na')).alias(f'seas_{start_year}_{end_year}na')])
    return data
def eqnpo_col(lag):
    c1 = (col('ri').shift(lag).over('id').is_not_null()) & (col('ri').shift(lag).over('id') != 0)
    c2 = (col('me').shift(lag).over('id').is_not_null()) & (col('me').shift(lag).over('id') != 0)
    eqnpo_col_exp = ((col('ri')/col('ri').shift(lag)).log() - (col('me')/col('me').shift(lag)).log()).over('id')
    return pl.when(c1 & c2).then(eqnpo_col_exp).otherwise(pl.lit(None).cast(pl.Float64))
@measure_time
def market_chars_monthly(data_path, market_ret_path, local_currency):
    data = pl.scan_ipc(data_path)
    market_ret = pl.scan_ipc(market_ret_path)
    ret_var = 'ret_local' if (local_currency == 1) else 'ret'
    data = data.join(market_ret, on = ['excntry', 'eom'])
    c1 = (col('ret_local') == 0).fill_null(pl.lit(False).cast(pl.Boolean))
    data = data.with_columns(pl.when(c1).then(pl.lit(1)).otherwise(pl.lit(0)).alias('ret_zero'))
    data = data.select(['id','date','eom','me','shares','adjfct','prc','ret','ret_local',col(ret_var).alias('ret_x'), 'div_tot', 'div_cash', 'div_spc', 'dolvol','ret_lag_dif','ret_zero','ret_exc','mkt_vw_exc'])
    __stock_coverage = data.group_by('id').agg(start_date = pl.min('eom'),end_date = pl.max('eom')).sort(['id','start_date'])
    __full_range = expand(data=__stock_coverage, id_vars=['id'], start_date='start_date', end_date='end_date', freq='month', new_date_name='eom')
    data = __full_range.join(data, how = 'left', on = ['id', 'eom'])
    data = data.select(['id', 'eom', 'me', 'shares', 'adjfct', 'prc', 'ret','ret_local','ret_x','ret_lag_dif','div_tot','div_cash','div_spc','dolvol','ret_zero','ret_exc','mkt_vw_exc'])
    data = data.sort(['id', 'eom']).with_columns([((1 + col('ret').fill_null(0)).cum_prod().over('id')).alias('ri'), ((1 + col('ret_x').fill_null(0)).cum_prod().over('id')).alias('ri_x'), (col('id').cum_count()).over('id').alias('count')])
    c1 = col('ret_x').is_null()
    c2 = (col('ret_lag_dif') != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    data = data.with_columns(pl.when(c1|c2).then(pl.lit(1)).otherwise(pl.lit(0)).alias('ret_miss'))
    c1 = (col('ret_miss') == 1)
    data = data.with_columns([
        (pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(col('ret_x'))).alias('ret_x'),
        (pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(col('ret'))).alias('ret'),
        (pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(col('ret_local'))).alias('ret_local'),
        (pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(col('ret_exc'))).alias('ret_exc'),
        (pl.when(c1).then(pl.lit(None).cast(pl.Float64)).otherwise(col('mkt_vw_exc'))).alias('mkt_vw_exc')]).drop(['ret_zero', 'ret_lag_dif']).unique(['id','eom'])
    data = data.with_columns(market_equity = col('me'),
                            div1m_me = (col('div_tot').fill_null(0)*col('shares')),
                            divspc1m_me = (col('div_spc').fill_null(0)*col('shares')),
                            aux = col('shares')*col('adjfct'))
    div_range = [1,3,6,12]#[1,3,6,12,24,36]
    div_spc_range = [1,12]
    chcsho_lags = [1,3,6,12]
    eqnpo_lags = [1,3,6,12]
    mom_rev_lags = [[0, 1],[0, 2],[0, 3],[1, 3],[0, 6],[1, 6],[0, 9],[1, 9],[0, 12],[1, 12],[7, 12],[1, 18],[1, 24],[12, 24],[1, 36],[12, 36],[12, 48],[1, 48],[1, 60],[12, 60],[36, 60]]
    div_cols1 = [(col('div1m_me').rolling_sum(window_size=i)).over('id').alias(f'div{i}m_me') for i in div_range[1:]]
    div_spc_cols1 = [(col('divspc1m_me').rolling_sum(window_size=i)).over('id').alias(f'divspc{i}m_me') for i in div_spc_range[1:]]
    chcsho_cols1 = [(((col('aux')/(col('aux').shift(i)))-1).over('id')).alias(f'chcsho_{i}m') for i in chcsho_lags]
    eqnpo_cols1 = [eqnpo_col(i).alias(f'eqnpo_{i}m') for i in eqnpo_lags]
    mom_rev_cols1 = [((col('ri_x').shift(i)/col('ri_x').shift(j)) - 1).over('id').alias(f'ret_{j}_{i}') for i,j in mom_rev_lags]
    div_cols2 = [(pl.when(col('count') < i).then(pl.lit(None).cast(pl.Float64)).otherwise(col(f'div{i}m_me')/col('me'))).alias(f'div{i}m_me') for i in div_range]
    div_spc_cols2 = [(pl.when(col('count') < i).then(pl.lit(None).cast(pl.Float64)).otherwise(col(f'divspc{i}m_me')/col('me'))).alias(f'divspc{i}m_me') for i in div_spc_range]
    chcsho_cols2 = [(pl.when(col('count')<=i).then(pl.lit(None).cast(pl.Float64)).otherwise(col(f'chcsho_{i}m'))).alias(f'chcsho_{i}m') for i in chcsho_lags]
    eqnpo_cols2 = [(pl.when(col('count')<=i).then(pl.lit(None).cast(pl.Float64)).otherwise(col(f'eqnpo_{i}m'))).alias(f'eqnpo_{i}m') for i in eqnpo_lags]
    mom_rev_cols2 = [(pl.when((col('ret_x').shift(i).over('id').is_null()) | (col('count') <= j)).then(pl.lit(None).cast(pl.Float64)).otherwise(col(f'ret_{j}_{i}'))).alias(f'ret_{j}_{i}') for i,j in mom_rev_lags]
    data = data.sort(['id','eom']).with_columns([*div_cols1, *div_spc_cols1, *chcsho_cols1, *eqnpo_cols1, *mom_rev_cols1]).drop('aux')
    data = data.sort(['id','eom']).with_columns([*div_cols2, *div_spc_cols2, *chcsho_cols2, *eqnpo_cols2, *mom_rev_cols2])
    data = seasonality(data, 'ret_x', 1, 1)
    data = seasonality(data, 'ret_x', 2, 5)
    data = seasonality(data, 'ret_x', 6, 10)
    data = seasonality(data, 'ret_x', 11, 15)
    data = seasonality(data, 'ret_x', 16, 20)
    data = data.drop(['me','shares','adjfct', 'adjfct', 'prc', 'ret','ret_local','ret_x', 'div_tot', 'div_cash', 'div_spc', 'dolvol', 'ret_exc', 'mkt_vw_exc','ret_miss', 'ri_x', 'ri', 'count'])
    data = data.sort(['id','eom'])
    #DO NOT USE STREAMING HERE
    data.collect().write_ipc('market_chars_m.ft')
@measure_time
def firm_age(data_path):
    data = pl.scan_ipc(data_path)
    crsp_age = pl.scan_ipc('Raw data/crsp_age.ft')
    comp_acc_age = pl.scan_ipc('Raw data/comp_acc_age.ft')
    comp_ret_age = pl.scan_ipc('Raw data/comp_ret_age.ft')
    comp_acc_age = comp_acc_age.group_by('gvkey').agg(col('datadate').min().alias('comp_acc_first')).sort('gvkey')
    comp_acc_age = comp_acc_age.with_columns(col('comp_acc_first').dt.offset_by('-1y').dt.month_end().alias('comp_acc_first'))
    comp_acc_age = comp_acc_age.with_columns((col('comp_acc_first').dt.year().cast(pl.Utf8) + pl.lit('-12-31')).str.strptime(pl.Date))
    comp_ret_age = comp_ret_age.group_by('gvkey').agg(col('datadate').min().alias('comp_ret_first')).sort('gvkey')
    comp_ret_age = comp_ret_age.with_columns(col('comp_ret_first').dt.offset_by('-1y').dt.month_end().alias('comp_ret_first'))
    comp_ret_age = comp_ret_age.with_columns((col('comp_ret_first').dt.year().cast(pl.Utf8) + pl.lit('-12-31')).str.strptime(pl.Date))
    data = data.select(['permco','gvkey','id','eom'])
    data = data.join(crsp_age, on = 'permco', how = 'left')
    data = data.join(comp_acc_age, on = 'gvkey', how = 'left')
    data = data.join(comp_ret_age, on = 'gvkey', how = 'left')
    data = data.with_columns(pl.min_horizontal('crsp_first', 'comp_acc_first','comp_ret_first').alias('first_obs')).drop(['permco','gvkey','crsp_first','comp_acc_first', 'comp_ret_first'])
    data = data.with_columns(pl.min('eom').over(['id']).alias('first_alt'))
    data = data.with_columns(pl.min_horizontal('first_obs', 'first_alt').alias('aux'))
    exp1 = (col('eom').dt.year() * 12 + col('eom').dt.month().cast(pl.Int32))
    exp2 = (col('aux').dt.year() * 12 + col('aux').dt.month().cast(pl.Int32))
    exp3 = (exp1 - exp2).alias('age')
    data = data.with_columns(exp3).drop('aux').sort(['id','eom'])
    data.collect(streaming = True).write_ipc('firm_age.ft')
def winsorize_own(data, sort_vars, wins_var, perc_low, perc_high):
    aux = data.group_by(sort_vars).agg(col(wins_var))
    aux = aux.with_columns(low = col(wins_var).list.sort().map_elements(lambda x: sas_percentile_method_5(x, perc_low)),
                           high = col(wins_var).list.sort().map_elements(lambda x: sas_percentile_method_5(x, perc_high)))
    return aux.select([*sort_vars, 'low','high'])
@measure_time
def sort_ff_style(char, freq, min_stocks_bp, min_stocks_pf, date_col, data, sf):
    print(f'Executing sort_ff_style for {char}', flush=True)
    c1 = ((col('size_grp_l').is_in(['small', 'large', 'mega'])).fill_null(pl.lit(False).cast(pl.Boolean)) & (col('excntry_l') != 'USA').fill_null(pl.lit(False).cast(pl.Boolean)))
    c2 = (((col('crsp_exchcd_l') == 1).fill_null(pl.lit(False).cast(pl.Boolean)) | (col('comp_exchg_l') == 11).fill_null(pl.lit(False).cast(pl.Boolean))) & (col('excntry_l') == 'USA').fill_null(pl.lit(False).cast(pl.Boolean)))
    c3 = col(f'{char}_l').is_not_null()
    bp_stocks = data.sort(['eom','excntry_l','id']).filter((c1 | c2) & c3)#.sort(['eom', 'excntry_l'])
    bp_stocks = bp_stocks.group_by(['eom', 'excntry_l']).agg([pl.len().alias('n'), col(f'{char}_l').alias(f'{char}_l')])
    bp_stocks = bp_stocks.with_columns(bp_p30 = col(f'{char}_l').list.sort().map_elements(lambda x: sas_percentile_method_5(x, 0.3)),
                                       bp_p70 = col(f'{char}_l').list.sort().map_elements(lambda x: sas_percentile_method_5(x, 0.7)))
    bp_stocks = bp_stocks.select(['eom', 'excntry_l', 'n', 'bp_p30','bp_p70'])
    data = data.join(bp_stocks, how = 'left', on = ['excntry_l', 'eom']).filter(col('n') >= min_stocks_bp).filter(col(f'{char}_l').is_not_null()).filter((col('size_pf') != '').fill_null(pl.lit(True).cast(pl.Boolean))).drop('n')
    char_pf_col = (pl.when(col(f'{char}_l') >= col('bp_p70')).then(pl.lit('high')).when(col(f'{char}_l') >= col('bp_p30')).then(pl.lit('mid')).otherwise(pl.lit('low'))).alias('char_pf')
    data = data.with_columns(char_pf_col).select(['excntry_l','id','eom','size_pf', 'me_l', 'char_pf'])
    aux_w_exp = pl.when((pl.sum('me_l') != 0).over('excntry_l','size_pf', 'char_pf', 'eom')).then((col('me_l')/pl.sum('me_l')).over('excntry_l','size_pf', 'char_pf', 'eom')).otherwise(pl.lit(None).cast(pl.Float64))
    data = data.with_columns(w = aux_w_exp, n = pl.len().over('excntry_l','size_pf', 'char_pf', 'eom'))
    data = data.filter(col('n') >= min_stocks_pf).drop('n')
    returns = sf.join(data, how = 'inner', left_on= ['id','eom','excntry'],right_on= ['id','eom','excntry_l'])
    returns = returns.with_columns((col('ret_exc')*col('w')).alias('ret_exc'))
    returns = returns.group_by(['excntry', 'size_pf', 'char_pf', date_col]).agg(ret_exc = pl.sum('ret_exc'))
    returns = returns.with_columns(characteristic = pl.lit(char), combined_pf = (col('size_pf') + '_' + col('char_pf'))).drop(['size_pf', 'char_pf'])
    returns = returns.collect()
    returns = returns.pivot(values='ret_exc',index=['excntry', date_col],columns='combined_pf')
    lms = ((col('small_high') + col('big_high')) / 2 - (col('small_low') + col('big_low')) / 2).alias('lms')
    smb = ((col('small_high') + col('small_mid') + col('small_low')) / 3 - (col('big_high') + col('big_mid') + col('big_low')) / 3).alias('smb')
    returns = returns.with_columns([lms, smb]).select(['excntry', date_col, 'lms', 'smb']).sort(['excntry', date_col])
    return returns
@measure_time
def ap_factors(output_path, freq, sf_path, mchars_path, mkt_path, min_stocks_bp, min_stocks_pf):
    print(f'Executing AP factors with frequency {freq}', flush=True)
    sf = pl.scan_ipc(sf_path)
    mchars = pl.scan_ipc(mchars_path)
    mkt = pl.scan_ipc(mkt_path)
    date_col = {'m': 'eom','d': 'date'}
    if freq == 'd': world_sf1 = sf.select(['excntry','id', 'date','eom', 'ret_exc','ret_lag_dif']).filter((col('ret_lag_dif') <= 5) & (col('ret_exc').is_not_null())).drop('ret_lag_dif')
    else: world_sf1 = sf.select(['excntry','id','eom', 'ret_exc','ret_lag_dif']).filter((col('ret_lag_dif') == 1) & (col('ret_exc').is_not_null())).drop('ret_lag_dif')
    wins_data = winsorize_own(world_sf1,['eom'],'ret_exc',0.1/100,99.9/100)
    world_sf2 = world_sf1.join(wins_data, how = 'left', on = 'eom').with_columns(ret_exc = pl.when(col('ret_exc')<= col('low')).then(col('low')).when(col('ret_exc')>= col('high')).then(col('high')).otherwise(col('ret_exc'))).drop(['high','low'])
    base = mchars.select(['id','eom','size_grp','excntry','me','market_equity','be_me','at_gr1','niq_be','source_crsp','exch_main','obs_main','common','comp_exchg','crsp_exchcd','primary_sec','ret_lag_dif']).rename({'comp_exchg': 'comp_exchg_l', 'crsp_exchcd': 'crsp_exchcd_l', 'exch_main': 'exch_main_l', 'obs_main': 'obs_main_l', 'common': 'common_l', 'primary_sec': 'primary_sec_l', 'excntry': 'excntry_l', 'size_grp': 'size_grp_l', 'me': 'me_l', 'be_me': 'be_me_l', 'at_gr1': 'at_gr1_l', 'niq_be': 'niq_be_l'})
    cols_to_lag = ['comp_exchg_l', 'crsp_exchcd_l', 'exch_main_l', 'obs_main_l', 'common_l', 'primary_sec_l', 'excntry_l', 'size_grp_l', 'me_l', 'be_me_l', 'at_gr1_l', 'niq_be_l']
    base = base.sort(['id', 'eom']).with_columns([col(i).shift(1).over(['id']).alias(i) for i in cols_to_lag])
    c1 = (col('id') != col('id').shift(1)).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col('source_crsp') != col('source_crsp').shift(1)).fill_null(pl.lit(True).cast(pl.Boolean))
    col1 = (col('eom').dt.year() - col('eom').shift(1).dt.year()).over(['id'])
    col2 = (col('eom').dt.month().cast(pl.Int32) - col('eom').shift(1).dt.month().cast(pl.Int32)).over(['id'])
    c3 = ((12*col1 + col2) != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    base = base.sort(['id', 'eom']).with_columns([pl.when(c1 | c2 | c3).then(pl.lit(None).cast(base.schema[i])).otherwise(i).alias(i) for i in cols_to_lag])#.sort(['id', 'eom'])
    c1 = (col("obs_main_l") == 1)
    c2 = (col("exch_main_l") == 1)
    c3 = (col("common_l") == 1)
    c4 = (col("primary_sec_l") == 1)
    c5 = (col("ret_lag_dif") == 1)
    c6 = col('me_l').is_not_null()
    base = base.filter(c1 & c2 & c3 & c4 & c5 & c6)
    size_pf_column = (pl.when(col('size_grp_l').is_null()).then(pl.lit('')).when(col('size_grp_l').is_in(['large', 'mega'])).then(pl.lit('big')).otherwise(pl.lit('small'))).alias('size_pf')
    base = base.with_columns(size_pf_column).sort(['excntry_l', 'size_grp_l', 'eom'])
    ff = sort_ff_style('be_me', freq, min_stocks_bp,min_stocks_pf,date_col[freq], base, world_sf2).rename({'lms' : 'hml', 'smb' : 'smb_ff'})
    asset_growth = sort_ff_style('at_gr1', freq, min_stocks_bp,min_stocks_pf,date_col[freq], base, world_sf2).rename({'lms': 'at_gr1_lms', 'smb': 'at_gr1_smb'})
    roeq = sort_ff_style('niq_be', freq, min_stocks_bp,min_stocks_pf,date_col[freq], base, world_sf2).rename({'lms': 'niq_be_lms', 'smb': 'niq_be_smb'})
    hxz = asset_growth.join(roeq, how = 'left', on = ['excntry',date_col[freq]])
    hxz = hxz.select(['excntry', date_col[freq], (-1*col('at_gr1_lms')).alias('inv'), col('niq_be_lms').alias('roe'), ((col('at_gr1_smb') + col('niq_be_smb'))/2).alias('smb_hxz')])
    output = mkt.select(['excntry', date_col[freq], col('mkt_vw_exc').alias('mktrf')]).collect()
    output = output.join(ff, how = 'left', on = ['excntry', date_col[freq]])
    output = output.join(hxz, how = 'left', on = ['excntry', date_col[freq]])
    output.write_ipc(output_path)
@measure_time
def market_beta(output_path, data_path, fcts_path, __n , __min):
    data = pl.scan_ipc(data_path)
    fcts = pl.scan_ipc(fcts_path)
    c1 = col('ret_local')!= 0.
    c2 = col('ret_exc').is_not_null()
    c3 = col('ret_lag_dif') == 1
    c4 = col('mktrf').is_not_null()
    data = data.filter(c1).filter(c2).filter(c3).select(['id','eom','ret_exc','excntry'])
    fcts = fcts.filter(c4).select(['mktrf', 'excntry', 'eom'])
    __msf = data.join(fcts, how = 'left', on = ['excntry', 'eom']).drop('excntry')
    wins_data = winsorize_own(__msf,['eom'],'ret_exc',0.1/100,99.9/100)
    __msf = __msf.join(wins_data, how = 'left', on = 'eom')
    __msf = __msf.with_columns(ret_exc = pl.when(col('ret_exc')<= col('low')).then(col('low')).when(col('ret_exc')>= col('high')).then(col('high')).otherwise(col('ret_exc'))).select(['id','eom','ret_exc','mktrf'])
    x ='mktrf'
    y = 'ret_exc'
    beta = pl.rolling_cov(x,y,window_size=__n, min_periods = __min)/col(x).rolling_var(window_size=__n, min_periods = __min)
    alpha = col(y).rolling_mean(window_size=__n, min_periods = __min) - beta * col(x).rolling_mean(window_size=__n, min_periods = __min)
    exp1 = col(y).rolling_var(window_size=__n, min_periods = __min)
    exp2 = col(x).rolling_var(window_size=__n, min_periods = __min)
    __msf2 = __msf.select(['id','eom']).sort(['id','eom']).with_columns(aux = (col('eom').shift(-1).over('id')).dt.offset_by(f'-1mo').dt.month_end())
    __msf2 = __msf2.with_columns(date_range = pl.date_ranges(start='eom', end='aux', interval='1mo')).with_columns(date_range = pl.coalesce(['date_range', 'eom'])).select(['id','date_range']).explode('date_range').rename({'date_range': 'eom'})
    __msf = __msf2.join(__msf, on = ['id','eom'], how = 'left')
    __msf = __msf.sort(['id','eom']).with_columns([beta.over('id').alias(f'beta_{__n}m'),alpha.over('id').alias('alpha')])
    __msf = __msf.sort(['id','eom']).with_columns(((exp1 - (col(f'beta_{__n}m')**2) * exp2) ** (1/2)).over('id').alias((f'ivol_capm_{__n}m')))
    __msf = __msf.select(['id','eom',f'beta_{__n}m',f'ivol_capm_{__n}m']).unique(['id', 'eom'])
    __msf = __msf.sort(['id','eom']).with_columns(beta_60m = col('beta_60m').fill_null(strategy='forward').over('id'),
                                                  ivol_capm_60m = col('ivol_capm_60m').fill_null(strategy='forward').over('id'))
    __msf.sort(['id','eom']).collect().write_ipc(output_path)
@measure_time
def prepare_daily(data_path, fcts_path):
    cols_to_select = ['excntry', 'id', 'date', 'eom', 'prc', 'adjfct', 'ret', 'ret_exc', 'dolvol', 'shares', 'tvol', 'ret_lag_dif', 'bidask', 'ret_local']
    cols_to_select2 = ['ret_exc','id','date','mktrf','eom','zero_obs']
    zero_obs_exp = pl.when(col('ret_local') == 0).then(1).otherwise(0).alias('zero_obs')
    c1 = col('ret_lag_dif') <= 14
    data = pl.scan_ipc(data_path)
    fcts = pl.scan_ipc(fcts_path)
    dsf1 = (data.select(cols_to_select)
                .join(fcts, how = 'left', on = ['excntry', 'date'])
                .filter(col('mktrf').is_not_null())
                .with_columns(zero_obs_exp)
                .with_columns(zero_obs = pl.sum('zero_obs').over(['id', 'eom']),
                              ret_exc  = pl.when(c1).then(col('ret_exc')).otherwise(pl.lit(None).cast(pl.Float64)),
                              ret      = pl.when(c1).then(col('ret')).otherwise(pl.lit(None).cast(pl.Float64)),
                              dolvol_d = col('dolvol'),
                              prc_adj  = safe_div('prc', 'adjfct', 'prc_adj'))
                .drop(['ret_lag_dif', 'bidask', 'ret_local', 'adjfct','prc', 'dolvol']))
    dsf1.collect(streaming=True).write_ipc('dsf1.ft')

    mkt_lead_lag = (fcts.select(['excntry', 'date', 'mktrf', col('date').dt.month_end().alias('eom')])
                        .sort(['excntry','date'])
                        .with_columns(mktrf_ld1 = col('mktrf').shift(-1).over(['excntry','eom']),
                                      mktrf_lg1 = col('mktrf').shift(1).over(['excntry','eom'])))
    mkt_lead_lag.collect(streaming=True).write_ipc('mkt_lead_lag.ft')

    corr_data = (pl.scan_ipc('dsf1.ft')
                   .select(cols_to_select2)
                   .sort(['id','date'])
                   .with_columns(ret_exc_3l = (col('ret_exc') + col('ret_exc').shift(1) + col('ret_exc').shift(2)).over(['id']),
                                 mkt_exc_3l = (col('mktrf') + col('mktrf').shift(1) + col('mktrf').shift(2)).over(['id']))
                   .select(['id', 'eom', 'zero_obs', 'ret_exc_3l', 'mkt_exc_3l']))
    corr_data.collect(streaming = True).write_ipc('corr_data.ft')
@measure_time
def mispricing_factors(data_path, min_stks, min_fcts = 3):
    chars = {}
    data = pl.scan_ipc(data_path)
    chars1 = data.filter((col('common') == 1) & (col('primary_sec') == 1) & (col('obs_main') == 1) & (col('exch_main') == 1) & (col('ret_exc').is_not_null()) & (col('me').is_not_null()))
    chars1 = chars1.select(['id', 'eom', 'excntry', 'chcsho_12m', 'eqnpo_12m', 'oaccruals_at', 'noa_at','at_gr1', 'ppeinv_gr1a', 'o_score', 'ret_12_1', 'gp_at', 'niq_at']).sort(['excntry', 'eom'])
    chars['1'] = chars1
    __direction = [True, False, True, True, True, True, True, False, False, False]
    __vars = ['chcsho_12m','eqnpo_12m','oaccruals_at','noa_at','at_gr1','ppeinv_gr1a','o_score','ret_12_1','gp_at','niq_at']
    index = [1,2,3,4,5,6,7,8,9,10]
    for __d, __v, i in zip(__direction,__vars, index):
        subset = chars1.select(['id','eom','excntry',__v])
        subset = subset.with_columns(count = col(__v).count().over(['excntry','eom'])).filter(col('count')>min_stks)
        subset = subset.with_columns(col(__v).rank(descending = __d).over(['excntry','eom']).alias(f'rank_{__v}')).drop(__v)
        subset = subset.with_columns((col(f'rank_{__v}')-pl.min(f'rank_{__v}')).over(['excntry','eom']).alias(f'rank_{__v}'))
        subset = subset.with_columns((col(f'rank_{__v}')/col('count')).alias(f'rank_{__v}')).drop(['excntry','count'])
        #chars[f'{i+1}'] = chars[f'{i}'].join(subset, on = ['id', 'eom'], how = 'left')
        chars[f'{(i+1)%2}'] = chars[f'{i%2}'].join(subset, on = ['id', 'eom'], how = 'left')
    c1 = (col('rank_o_score').is_null() + col('rank_ret_12_1').is_null() + col('rank_gp_at').is_null() + col('rank_niq_at').is_null()) > min_fcts
    c2 = (col('rank_chcsho_12m').is_null() + col('rank_eqnpo_12m').is_null() + col('rank_oaccruals_at').is_null() + col('rank_noa_at').is_null() + col('rank_at_gr1').is_null() + col('rank_ppeinv_gr1a').is_null()) > min_fcts
    exp1 = pl.mean_horizontal('rank_o_score', 'rank_ret_12_1', 'rank_gp_at', 'rank_niq_at')
    exp2 = pl.mean_horizontal('rank_chcsho_12m', 'rank_eqnpo_12m', 'rank_oaccruals_at', 'rank_noa_at', 'rank_at_gr1', 'rank_ppeinv_gr1a')
    exp3 = pl.lit(None).cast(pl.Float64)
    chars['1'] = chars['1'].with_columns([(pl.when(c1).then(exp3).otherwise(exp1)).alias('mispricing_perf'),(pl.when(c2).then(exp3).otherwise(exp2)).alias('mispricing_mgmt')])
    chars['1'] = chars['1'].select(['id', 'eom', 'mispricing_perf', 'mispricing_mgmt'])
    chars['1'].collect().write_ipc('mp_factors.ft')
@measure_time
def regression_3vars(y, x1, x2, x3, __n, __min):
    den = (-((col(x1).rolling_var(window_size=__n, min_periods=__min)) * (col(x2).rolling_var(window_size=__n, min_periods=__min)) * (col(x3).rolling_var(window_size=__n, min_periods=__min))) +
       (col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min))**2 +
       (col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min))**2 -
       2 * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) +
       (col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))**2)
    beta1 = ((col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) -
         (col(x2).rolling_var(window_size=__n, min_periods=__min)) * ((col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min))) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) +
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) +
         (col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min))) / den
    beta2 = ((col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) -
         (col(x1).rolling_var(window_size=__n, min_periods=__min)) * ((col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min))) +
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) +
         (col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))) / den
    beta3 = (-((col(x1).rolling_var(window_size=__n, min_periods=__min)) * (col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min))) +
         (col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) +
         (col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) +
         (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min))) / den
    alpha = col(y).rolling_mean(window_size=__n, min_periods = __min) - beta1 * col(x1).rolling_mean(window_size=__n, min_periods = __min) - beta2 * col(x2).rolling_mean(window_size=__n, min_periods = __min) - beta3 * col(x3).rolling_mean(window_size=__n, min_periods = __min)
    return alpha, beta1, beta2, beta3
@measure_time
def get_rolling_residuals(a,b1,b2,b3,x1,x2,x3,y,w):
    squared_res_sum = w*(col(a)**2) + 2 * col(a) * col(b1) * col(x1).rolling_sum(window_size=w) + (col(b1)**2) * (col(x1)**2).rolling_sum(window_size=w) + 2 * col(a) * col(b2)* col(x2).rolling_sum(window_size=w) + 2 * col(b1) * col(b2) * (col(x1) * col(x2)).rolling_sum(window_size=w) + (col(b2)**2) * (col(x2)**2).rolling_sum(window_size=w) + 2 * col(a) * col(b3) * col(x3).rolling_sum(window_size=w) + 2 * col(b1) * col(b3) * (col(x1) * col(x3)).rolling_sum(window_size=w) + 2 * col(b2) * col(b3) * (col(x2) *  col(x3)).rolling_sum(window_size=w) + (col(b3)**2) * (col(x3)**2).rolling_sum(window_size=w) - 2 * col(a) * col(y).rolling_sum(window_size=w) -  2 * col(b1) * (col(x1) * col(y)).rolling_sum(window_size=w) - 2 * col(b2) * (col(x2) * col(y)).rolling_sum(window_size=w) - 2 * col(b3) * (col(x3) * col(y)).rolling_sum(window_size=w) + (col(y)**2).rolling_sum(window_size=w)
    res_mean = (col(y).rolling_mean(window_size=w) - col(a) - col(b1) * col(x1).rolling_mean(window_size=w) - col(b2) * col(x2).rolling_mean(window_size=w) - col(b3) * col(x3).rolling_mean(window_size=w))
    var = (squared_res_sum - w*(res_mean**2))/(w-1)
    return (res_mean/(var**0.5))
@measure_time
def residual_momentum(output_path, data_path, fcts_path, __n, __min, incl, skip):
    w = incl - skip
    data = pl.scan_ipc(data_path)
    fcts = pl.scan_ipc(fcts_path).select(['excntry', 'eom','mktrf', 'hml', 'smb_ff'])
    c1 = col('ret_local') != 0
    c2 = col('ret_exc').is_not_null()
    c3 = col('ret_lag_dif') == 1
    c4 = col('mktrf').is_not_null()
    data = data.filter(c1).filter(c2).filter(c3).select(['excntry','id','eom','ret_exc'])
    fcts = fcts.filter(c4).select(['excntry', 'eom','mktrf', 'hml', 'smb_ff'])
    __msf = data.join(fcts, how = 'left', on = ['excntry', 'eom']).drop('excntry')
    wins_data = winsorize_own(__msf,['eom'],'ret_exc',0.1/100,99.9/100)
    __msf = __msf.join(wins_data, how = 'left', on = 'eom')
    __msf = __msf.with_columns(ret_exc = pl.when(col('ret_exc')<= col('low')).then(col('low')).when(col('ret_exc')>= col('high')).then(col('high')).otherwise(col('ret_exc'))).select(['id','eom','ret_exc','mktrf','hml','smb_ff'])
    __msf2 = __msf.select(['id','eom']).sort(['id','eom']).with_columns(aux = (col('eom').shift(-1).over('id')).dt.offset_by(f'-1mo').dt.month_end())
    __msf2 = __msf2.with_columns(date_range = pl.date_ranges(start='eom', end='aux', interval='1mo')).with_columns(date_range = pl.coalesce(['date_range', 'eom'])).select(['id','date_range']).explode('date_range').rename({'date_range': 'eom'})
    __msf = __msf2.join(__msf, on = ['id','eom'], how = 'left')
    alpha, beta1, beta2, beta3 = regression_3vars('ret_exc','mktrf','smb_ff','hml',__n,__min)
    __msf = __msf.sort(['id','eom']).with_columns([alpha.over('id').alias('alpha'),beta1.over('id').alias('beta1'),beta2.over('id').alias('beta2'),beta3.over('id').alias('beta3')])
    residual_exp = lambda i:  col('ret_exc').shift(i) - (col('alpha')*pl.lit(1.) + col('beta1') * col('mktrf').shift(i) + col('beta2') * col('smb_ff').shift(i) + col('beta3') * col('hml').shift(i))
    __msf = __msf.sort(['id','eom']).with_columns(std_res = pl.concat_list([residual_exp(i).over(['id']) for i in range(1, w+1)]).list.drop_nulls()).select(['id','eom','std_res'])
    __msf = __msf.collect()
    __msf = __msf.with_columns(std_res =  col('std_res').list.eval(pl.element().mean()/pl.element().std())).explode('std_res')
    __msf = __msf.select(['id','eom',col('std_res').alias(f'resff3_{incl}_{skip}')])
    __msf.sort(['id','eom']).write_ipc(output_path + f'_{incl}_{skip}.ft')
    del __msf
@measure_time
def bidask_hl(output_path, data_path, market_returns_daily_path ,__min_obs):
    data = pl.scan_ipc(data_path)
    market_returns_daily = pl.scan_ipc(market_returns_daily_path)
    __dsf = data.join(market_returns_daily, how = 'left', on=['excntry', 'date']).filter(col('mkt_vw_exc').is_not_null())
    __dsf = __dsf.with_columns(prc = (col('prc')/col('adjfct')),
                               prc_high = (col('prc_high')/col('adjfct')),
                               prc_low = (col('prc_low')/col('adjfct')))
    __dsf = __dsf.select(['id', 'date', 'eom', 'bidask', 'tvol', 'prc', 'prc_high', 'prc_low'])
    c1 = col('bidask') == 1
    c2 = col('prc_low') == col('prc_high')
    c3 = col('prc_low') <= 0
    c4 = col('prc_high') <= 0
    c5 = col('tvol') == 0
    __dsf = __dsf.with_columns(prc_high = (pl.when(c1 | c2 | c3 | c4 | c5).then(pl.lit(None).cast(pl.Float64)).otherwise(col('prc_high'))),
                               prc_low = (pl.when(c1 | c2 | c3 | c4 | c5).then(pl.lit(None).cast(pl.Float64)).otherwise(col('prc_low'))))
    ind_exp = ((0 <col('prc_low')) & (col('prc_low') <col('prc_high'))).cast(pl.Int64).fill_null(0)
    id_exp = (col('id').shift(1).over('id').is_null()).cast(pl.Int64)
    __dsf = __dsf.sort(['id','date']).with_columns(prc_low_r = col('prc_low').shift(1).over('id'),prc_high_r = col('prc_high').shift(1).over('id'), ind = ind_exp, id_flag = id_exp)
    __dsf = __dsf.sort(['id','date']).with_columns(prc_low_r = col('prc_low_r').fill_null(strategy = 'forward').over('id'),prc_high_r = col('prc_high_r').fill_null(strategy = 'forward').over('id'))
    c1 = col('id_flag') == 0
    c2 = col('ind') == 0
    c3 = (col('prc_low_r') <= col('prc')) & (col('prc') <= col('prc_high_r'))
    c4 = col('prc') < col('prc_low_r')
    c5 = col('prc') > col('prc_high_r')
    c6 = c1 & c2 & c3
    c7 = c1 & c2 & c4
    c8 = c1 & c2 & c5
    __dsf = __dsf.with_columns(prc_low = pl.when(c6).then(col('prc_low_r')).when(c7).then(col('prc')).when(c8).then(col('prc_low_r') + col('prc') - col('prc_high_r')).otherwise(col('prc_low')),
                               prc_high = pl.when(c6).then(col('prc_high_r')).when(c7).then(col('prc_high_r') - (col('prc_low_r') - col('prc'))).when(c8).then(col('prc')).otherwise(col('prc_high')))
    c9 = (col('prc_low')!= 0.) & ((col('prc_high')/col('prc_low')) > 8)
    __dsf = __dsf.with_columns(prc_low = pl.when(c9).then(pl.lit(None).cast(pl.Float64)).otherwise(col('prc_low')), prc_high = pl.when(c9).then(pl.lit(None).cast(pl.Float64)).otherwise(col('prc_high')))
    __dsf = __dsf.select(['id','date','eom','prc', 'prc_high','prc_low'])
    __dsf = __dsf.sort(['id','date']).with_columns(prc_low_t = col('prc_low'), prc_high_t = col('prc_high'), prc_low_l1 = col('prc_low').shift(1).over('id'), prc_high_l1 = col('prc_high').shift(1).over('id'), prc_l1 = col('prc').shift(1).over('id'))
    c1 = (col('prc_l1') < col('prc_low')) & (col('prc_l1') > 0)
    c2 = (col('prc_l1') > col('prc_high')) & (col('prc_l1') > 0)
    # Apply the expressions to the DataFrame
    __dsf = __dsf.with_columns(prc_high_t = pl.when(c1).then(col('prc_high') - (col('prc_low') - col('prc_l1'))).otherwise(col('prc_high_t')),
                               prc_low_t = pl.when(c1).then(col('prc_l1')).otherwise(col('prc_low_t')))
    __dsf = __dsf.with_columns(prc_high_t = pl.when(c2).then(col('prc_l1')).otherwise(col('prc_high_t')).alias('prc_high_t'),
                            prc_low_t = pl.when(c2).then(col('prc_low') + (col('prc_l1') - col('prc_high'))).otherwise(col('prc_low_t')).alias('prc_low_t'))
    pi = 3.141592653589793
    k2 = sqrt(8 / pi)
    const = 3 - 2 * sqrt(2)
    prc_high_2d_expr = pl.max_horizontal('prc_high_t', 'prc_high_l1').alias('prc_high_2d')
    prc_low_2d_expr = pl.min_horizontal('prc_low_t', 'prc_low_l1').alias('prc_low_2d')
    c1 = (col('prc_low_t') > 0) & (col('prc_low_l1') > 0)
    beta_expr = (pl.when(c1).then(((col('prc_high_t') / col('prc_low_t')).log() ** 2) + ((col('prc_high_l1') / col('prc_low_l1')).log() ** 2)).otherwise(pl.lit(None).cast(pl.Float64))).alias('beta')
    c2 = (col('prc_low_2d') > 0)
    gamma_expr = (pl.when(c2).then(((col('prc_high_2d') / col('prc_low_2d')).log() ** 2)).otherwise(pl.lit(None).cast(pl.Float64))).alias('gamma')
    alpha_expr = ( ((sqrt(2) - 1) * col('beta').sqrt()) / const - (col('gamma') / const).sqrt()).alias('alpha')
    spread_expr = (2 * (pl.lit(exp(1)).pow(col('alpha')) - 1) / (1 + pl.lit(exp(1)).pow(col('alpha')))).alias('spread')
    spread_0_expr = (pl.when(col('spread') <0).then(pl.lit(0.)).otherwise(col('spread'))).alias('spread_0')
    spread_0_expr2 = (pl.when(col('spread').is_null()).then(pl.lit(None).cast(pl.Float64)).otherwise(col('spread_0'))).alias('spread_0')
    sigma_expr = (( ((col('beta')/ 2).sqrt() - col('beta').sqrt()) / (k2 * const)) + (col('gamma') / (k2 * k2 * const)).sqrt()).alias('sigma')
    sigma_0_expr = (pl.when(col('sigma') <0).then(pl.lit(0.)).otherwise(col('sigma'))).alias('sigma_0')
    sigma_0_expr2 = (pl.when(col('sigma').is_null()).then(pl.lit(None).cast(pl.Float64)).otherwise(col('sigma_0'))).alias('sigma_0')
    __dsf = __dsf.with_columns([prc_high_2d_expr,prc_low_2d_expr]).with_columns([beta_expr,gamma_expr]).with_columns(alpha_expr)
    __dsf = __dsf.with_columns([spread_expr, sigma_expr]).with_columns([spread_0_expr, sigma_0_expr]).with_columns([spread_0_expr2, sigma_0_expr2])
    __dsf = __dsf.select(['id','date','eom','spread_0','sigma_0'])
    __dsf = __dsf.with_columns(count = col('spread_0').is_not_null().cast(pl.Int8))
    __dsf = __dsf.group_by(['id','eom']).agg(bidaskhl_21d = pl.mean('spread_0'), rvolhl_21d = pl.mean('sigma_0'), count = pl.sum('count')).filter(col('count') > __min_obs).drop('count').sort(['id','eom'])
    __dsf.collect().write_ipc(output_path)
@measure_time
def create_world_data_prelim(msf_path, market_chars_monthly_path, acc_chars_world_path, output_path):
    a = pl.scan_ipc(msf_path)
    b = pl.scan_ipc(market_chars_monthly_path)
    c = pl.scan_ipc(acc_chars_world_path)
    world_data_prelim = a.join(b, how = 'left', on = ['id','eom'])
    world_data_prelim = world_data_prelim.join(c, how = 'left', left_on = ['gvkey','eom'], right_on = ['gvkey', 'public_date'])
    world_data_prelim = world_data_prelim.drop(['div_tot', 'div_cash', 'div_spc', 'public_date', 'source'])
    world_data_prelim.collect().write_ipc(output_path)
    #Streaming can be used here if needed
    #world_data_prelim.collect(streaming = True).write_ipc('world_data_prelim.ft')

acc_chars = [
    # Accounting Based Size Measures
    "assets", "sales", "book_equity", "net_income", "enterprise_value",
    # 1yr Growth
    "at_gr1", "ca_gr1", "nca_gr1", "lt_gr1", "cl_gr1", "ncl_gr1", "be_gr1", "pstk_gr1", "debt_gr1", "sale_gr1", "cogs_gr1", "sga_gr1", "opex_gr1",
    # 3yr Growth
    "at_gr3", "ca_gr3", "nca_gr3", "lt_gr3", "cl_gr3", "ncl_gr3", "be_gr3", "pstk_gr3", "debt_gr3", "sale_gr3", "cogs_gr3", "sga_gr3", "opex_gr3",
    # 1yr Growth Scaled by Assets
    "cash_gr1a", "inv_gr1a", "rec_gr1a", "ppeg_gr1a", "lti_gr1a", "intan_gr1a", "debtst_gr1a", "ap_gr1a", "txp_gr1a", "debtlt_gr1a", "txditc_gr1a",
    "coa_gr1a", "col_gr1a", "cowc_gr1a", "ncoa_gr1a", "ncol_gr1a", "nncoa_gr1a", "oa_gr1a", "ol_gr1a", "noa_gr1a", "fna_gr1a", "fnl_gr1a",
    "nfna_gr1a", "gp_gr1a", "ebitda_gr1a", "ebit_gr1a", "ope_gr1a", "ni_gr1a", "nix_gr1a", "dp_gr1a", "ocf_gr1a", "fcf_gr1a", "nwc_gr1a",
    "eqnetis_gr1a", "dltnetis_gr1a", "dstnetis_gr1a", "dbnetis_gr1a", "netis_gr1a", "fincf_gr1a", "eqnpo_gr1a", "tax_gr1a", "div_gr1a",
    "eqbb_gr1a", "eqis_gr1a", "eqpo_gr1a", "capx_gr1a",
    # 3yr Growth Scaled by Assets
    "cash_gr3a", "inv_gr3a", "rec_gr3a", "ppeg_gr3a", "lti_gr3a", "intan_gr3a", "debtst_gr3a", "ap_gr3a", "txp_gr3a", "debtlt_gr3a", "txditc_gr3a",
    "coa_gr3a", "col_gr3a", "cowc_gr3a", "ncoa_gr3a", "ncol_gr3a", "nncoa_gr3a", "oa_gr3a", "ol_gr3a", "fna_gr3a", "fnl_gr3a", "nfna_gr3a",
    "gp_gr3a", "ebitda_gr3a", "ebit_gr3a", "ope_gr3a", "ni_gr3a", "nix_gr3a", "dp_gr3a", "ocf_gr3a", "fcf_gr3a", "nwc_gr3a",
    "eqnetis_gr3a", "dltnetis_gr3a", "dstnetis_gr3a", "dbnetis_gr3a", "netis_gr3a", "fincf_gr3a", "eqnpo_gr3a", "tax_gr3a", "div_gr3a",
    "eqbb_gr3a", "eqis_gr3a", "eqpo_gr3a", "capx_gr3a",
    # Investment
    "capx_at", "rd_at",
    # Profitability
    "gp_sale", "ebitda_sale", "ebit_sale", "pi_sale", "ni_sale", "nix_sale", "ocf_sale", "fcf_sale",
    # Return on Assets
    "gp_at", "ebitda_at", "ebit_at", "fi_at", "cop_at",
    # Return on Book Equity
    "ope_be", "ni_be", "nix_be", "ocf_be", "fcf_be",
    # Return on Invested Capital
    "gp_bev", "ebitda_bev", "ebit_bev", "fi_bev", "cop_bev",
    # Return on Physical Capital
    "gp_ppen", "ebitda_ppen", "fcf_ppen",
    # Issuance
    "fincf_at", "netis_at", "eqnetis_at", "eqis_at", "dbnetis_at", "dltnetis_at", "dstnetis_at",
    # Equity Payout
    "eqnpo_at", "eqbb_at", "div_at",
    # Accruals
    "oaccruals_at", "oaccruals_ni", "taccruals_at", "taccruals_ni", "noa_at",
    # Capitalization/Leverage Ratios
    "be_bev", "debt_bev", "cash_bev", "pstk_bev", "debtlt_bev", "debtst_bev", "debt_mev", "pstk_mev", "debtlt_mev", "debtst_mev",
    # Financial Soundness Ratios
    "int_debtlt", "int_debt", "cash_lt", "inv_act", "rec_act", "ebitda_debt", "debtst_debt", "cl_lt", "debtlt_debt", "profit_cl", "ocf_cl",
    "ocf_debt", "lt_ppen", "debtlt_be", "fcf_ocf", "opex_at", "nwc_at",
    # Solvency Ratios
    "debt_at", "debt_be", "ebit_int",
    # Liquidity Ratios
    "cash_cl", "caliq_cl", "ca_cl", "inv_days", "rec_days", "ap_days", "cash_conversion",
    # Activity/Efficiency Ratio
    "inv_turnover", "at_turnover", "rec_turnover", "ap_turnover",
    # Non-Recurring Items
    "spi_at", "xido_at", "nri_at",
    # Miscellaneous
    "adv_sale", "staff_sale", "rd_sale", "div_ni", "sale_bev", "sale_be", "sale_nwc", "tax_pi",
    # Balance Sheet Fundamentals to Market Equity
    "be_me", "at_me", "cash_me",
    # Income Fundamentals to Market Equity
    "gp_me", "ebitda_me", "ebit_me", "ope_me", "ni_me", "nix_me", "sale_me", "ocf_me", "fcf_me", "cop_me", "rd_me",
    # Equity Payout/issuance to Market Equity
    "div_me", "eqbb_me", "eqis_me", "eqpo_me", "eqnpo_me", "eqnetis_me",
    # Debt Issuance to Market Enterprise Value
    "dltnetis_mev", "dstnetis_mev", "dbnetis_mev",
    # Firm Payout/issuance to Market Enterprise Value
    "netis_mev",
    # Balance Sheet Fundamentals to Market Enterprise Value
    "at_mev", "be_mev", "bev_mev", "ppen_mev", "cash_mev",
    # Income/CF Fundamentals to Market Enterprise Value
    "gp_mev", "ebitda_mev", "ebit_mev", "cop_mev", "sale_mev", "ocf_mev", "fcf_mev", "fincf_mev",
    # New Variables from HXZ
    "ni_inc8q", "ppeinv_gr1a", "lnoa_gr1a", "capx_gr1", "capx_gr2", "capx_gr3", "sti_gr1a",
    "niq_at", "niq_at_chg1", "niq_be", "niq_be_chg1", "saleq_gr1", "rd5_at",
    "dsale_dinv", "dsale_drec", "dgp_dsale", "dsale_dsga",
    "saleq_su", "niq_su", "debt_me", "netdebt_me", "capex_abn", "inv_gr1", "be_gr1a",
    "op_at", "pi_nix", "op_atl1", "gp_atl1", "ope_bel1", "cop_atl1",
    "at_be", "ocfq_saleq_std", "aliq_at", "aliq_mat", "tangibility",
    "eq_dur", "f_score", "o_score", "z_score", "kz_index", "intrinsic_value", "ival_me",
    "sale_emp_gr1", "emp_gr1", "cash_at", "earnings_variability",
    # New Variables not in HXZ
    "niq_saleq_std", "ni_emp", "sale_emp", "ni_at", "ocf_at", "ocf_at_chg1",
    "roeq_be_std", "roe_be_std", "gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5"
]

########################################################################################################################################################################################################################
@measure_time
def finish_daily_chars(output_path):
    bidask = pl.scan_ipc('corwin_schultz.ft')
    r1 = pl.scan_ipc('roll_apply_daily.ft')
    daily_chars = bidask.join(r1, how = 'outer_coalesce', on=['id','eom'])
    daily_chars = daily_chars.with_columns(betabab_1260d = col('corr_1260d') * col('rvol_252d')/ col('__mktvol_252d'), rmax5_rvol_21d = col('rmax5_21d') / col('rvol_252d')).drop('__mktvol_252d')
    daily_chars.collect().write_ipc(output_path)
@measure_time
def z_ranks(data, var, min, sort):
    order = False if sort == 'ascending' else True
    __subset = data.select(['excntry','id','eom',var]).with_columns(count = (col(var).is_not_null().sum()).over(['excntry','eom'])).filter(col('count') > min).drop('count')
    __subset = __subset.with_columns(col(var).rank(descending = order).over(['excntry','eom']).alias(f'rank_{var}')).filter(col(f'rank_{var}').is_not_null())
    __subset = __subset.with_columns(((col(f'rank_{var}') - pl.mean(f'rank_{var}'))/pl.std(f'rank_{var}').over(['excntry','eom'])).alias(f'z_{var}'))
    __subset = __subset.select(['excntry','id','eom',f'z_{var}'])
    return __subset
@measure_time
def quality_minus_junk(data, min_stks):
    cols = ['id', 'eom', 'excntry', 'gp_at', 'ni_be', 'ni_at', 'ocf_at', 'gp_sale', 'oaccruals_at', 'gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5', 'betabab_1260d', 'debt_at', 'o_score', 'z_score', 'roeq_be_std', 'roe_be_std']
    c1 = (col('common') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c2 = (col('primary_sec') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c3 = (col('obs_main') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c4 = (col('exch_main') == 1).fill_null(pl.lit(False).cast(pl.Boolean))
    c5 = col('ret_exc').is_not_null()
    c6 = col('me').is_not_null()
    qmj = data.filter(c1 & c2 & c3 & c4 & c5 & c6).select(cols)
    qmj = qmj.with_columns(pl.when(col('roeq_be_std').is_not_null()).then(col('roeq_be_std') * 2).otherwise(col('roe_be_std')).alias('__evol')).sort(['excntry', 'eom'])
    z_vars = ['gp_at', 'ni_be', 'ni_at', 'ocf_at', 'gp_sale', 'oaccruals_at','gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5','betabab_1260d', 'debt_at', 'o_score', 'z_score', '__evol']
    direction = ['ascending', 'ascending', 'ascending', 'ascending', 'ascending', 'descending','ascending', 'ascending', 'ascending', 'ascending', 'ascending','descending', 'descending', 'descending', 'ascending', 'descending']
    for var_z,dir in zip(z_vars,direction):
        __z = z_ranks(qmj, var_z, min_stks, dir)
        qmj = qmj.join(__z, how = 'left', on = ['excntry','eom','id'])
    qmj = qmj.select([col('excntry'), col('id'), col('eom'),(pl.sum_horizontal('z_gp_at', 'z_ni_be', 'z_ni_at', 'z_ocf_at', 'z_gp_sale', 'z_oaccruals_at')/6).alias('__prof'),(pl.sum_horizontal('z_gpoa_ch5', 'z_roe_ch5', 'z_roa_ch5', 'z_cfoa_ch5', 'z_gmar_ch5')/5).alias('__growth'),(pl.sum_horizontal('z_betabab_1260d', 'z_debt_at', 'z_o_score', 'z_z_score', 'z___evol')/5).alias('__safety')])
    __prof = z_ranks(qmj, '__prof', min_stks, 'ascending').rename({'z___prof': 'qmj_prof'})#.with_columns(qmj_prof = pl.when(col('qmj_prof')<-99999).then(pl.lit(None).cast(pl.Float64)).otherwise(col('qmj_prof')))
    __growth = z_ranks(qmj, '__growth', min_stks, 'ascending').rename({'z___growth': 'qmj_growth'})#.with_columns(qmj_growth = pl.when(col('qmj_growth')<-99999).then(pl.lit(None).cast(pl.Float64)).otherwise(col('qmj_growth')))
    __safety = z_ranks(qmj, '__safety', min_stks, 'ascending').rename({'z___safety': 'qmj_safety'})#.with_columns(qmj_safety = pl.when(col('qmj_safety')<-99999).then(pl.lit(None).cast(pl.Float64)).otherwise(col('qmj_safety')))
    qmj = qmj.select(['excntry', 'id', 'eom'])
    qmj = qmj.join(__prof, how = 'left', on = ['excntry','id','eom'])
    qmj = qmj.join(__growth, how = 'left', on = ['excntry','id','eom'])
    qmj = qmj.join(__safety, how = 'left', on = ['excntry','id','eom'])
    qmj = qmj.with_columns(__qmj = (pl.sum_horizontal('qmj_prof', 'qmj_growth', 'qmj_safety')/3))
    qmj = qmj.with_columns(__qmj = pl.when((col('qmj_prof').is_null()) | (col('qmj_growth').is_null()) |(col('qmj_safety').is_null())).then(pl.lit(None).cast(pl.Float64)).otherwise(col('__qmj')))
    __qmj = z_ranks(qmj, '__qmj', min_stks, 'ascending').rename({'z___qmj': 'qmj'})
    qmj = qmj.join(__qmj, how = 'left', on = ['excntry', 'id', 'eom'])#.with_columns(qmj = pl.when(col('qmj')<-99999).then(pl.lit(None).cast(pl.Float64)).otherwise(col('qmj')))
    qmj.write_ipc('qmj.ft')
@measure_time
def save_main_data(end_date):
    months_exp = (col('eom').dt.year() * 12 + col('eom').dt.month()).cast(pl.Int64)
    data = pl.scan_ipc('world_data.ft').with_columns( dif_aux = months_exp).sort(['id','eom'])
    data = data.with_columns(me_lag1 = col('me').shift(1).over('id'), dif_aux = (col('dif_aux') - col('dif_aux').shift(1)).over('id')).with_columns(me_lag1 = pl.when(col('dif_aux')!= 1).then(pl.lit(None).cast(pl.Float64)).otherwise(col('me_lag1'))).drop('dif_aux')
    data = data.filter(col('primary_sec') == 1).filter(col('common') == 1).filter(col('obs_main') == 1).filter(col('exch_main') == 1).filter(col('eom')<= end_date)
    data.collect(streaming=True).write_ipc('world_data_filtered.ft')
    countries = pl.scan_ipc('world_data_filtered.ft').select('excntry').unique().collect().to_numpy().flatten()
    for i in countries:
        print(f'Filtering data for country {i}', flush=True)
        data = pl.scan_ipc('world_data_filtered.ft').filter(col('excntry') == i)
        data.collect().write_ipc(f'Characteristics/{i}.ft', compression='zstd')
@measure_time
def save_daily_ret():
    data = pl.scan_ipc('world_dsf.ft').select(['excntry', 'id', 'date', 'me', 'ret', 'ret_exc'])
    countries = pl.scan_ipc('world_dsf.ft').select('excntry').unique().collect().to_numpy().flatten()
    for i in countries:
        if i == None:
            print(f'Filtering data for null country', flush=True)
            data.filter(col('excntry').is_null()).collect().write_ipc(f'Daily_Returns/null_country.ft',compression='zstd')
        else:
            print(f'Filtering data for country {i}', flush=True)
            data.filter(col('excntry') == i).collect().write_ipc(f'Daily_Returns/{i}.ft',compression='zstd')
@measure_time
def save_monthly_ret():
    data = pl.scan_ipc('world_msf.ft').select(['excntry', 'id', 'source_crsp', 'eom', 'me', 'ret_exc', 'ret', 'ret_local'])
    data.collect().write_ipc(f'World_Ret_Monthly/world_ret_monthly.ft',compression='zstd')
@measure_time
def merge_roll_apply_daily_results():
    date_idx = datetime.datetime.today().month + datetime.datetime.today().year * 12
    df_dates = pl.DataFrame({'aux_date': [i+1 for i in range(23112, date_idx+1)],'eom': [f'{i//12}-{i%12+1}-1' for i in range(23112, date_idx+1)]})
    df_dates = df_dates.with_columns(col('eom').str.strptime(pl.Date, "%Y-%m-%d").dt.month_end().alias('eom'),col('aux_date').cast(pl.Int64))
    file_paths = [i for i in os.listdir() if i.startswith('__roll')]
    if len(file_paths) != 1:
        joint_file = pl.scan_ipc(file_paths[0])
        for i in file_paths[1:]:
            df_aux = pl.scan_ipc(i)
            joint_file = joint_file.join(df_aux, how = 'outer_coalesce', on = ['id','aux_date'])
        joint_file = joint_file.with_columns(col('aux_date').cast(pl.Int64))
        joint_file.with_columns(col('aux_date').cast(pl.Int64)).join(df_dates.lazy(), how = 'left', on = 'aux_date').drop('aux_date').collect().write_ipc('roll_apply_daily.ft')
    else:
        pl.scan_ipc(file_paths[0]).with_columns(col('aux_date').cast(pl.Int64)).join(df_dates.lazy(), how = 'left', on = 'aux_date').drop('aux_date').collect().write_ipc('roll_apply_daily.ft')
###############################################################################################################################################################################
wrds_session = wrds_Fernando.Connection(wrds_username= "your_username_here", wrds_password="your_password_here")
end_date = pl.datetime(2023, 12, 31)
prepare_comp_sf('both')
prepare_crsp_sf('m')
prepare_crsp_sf('d')
combine_crsp_comp_sf()
crsp_industry()
comp_industry()
__msf_world = pl.scan_ipc('__msf_world.ft')
comp_ind = pl.scan_ipc('comp_ind.ft')
crsp_ind = pl.scan_ipc('crsp_ind.ft').rename({'sic': 'sic_crsp', 'naics': 'naics_crsp'})
__msf_world = __msf_world.join(comp_ind, how = 'left', left_on = ['gvkey', 'eom'], right_on = ['gvkey','date'])
__msf_world = __msf_world.join(crsp_ind, how = 'left', left_on = ['permco', 'permno', 'eom'], right_on = ['permco','permno','date'])
__msf_world = __msf_world.with_columns(sic = pl.coalesce(['sic','sic_crsp']).alias('sic'), naics = pl.coalesce(['naics','naics_crsp']).alias('naics')).drop(['sic_crsp','naics_crsp'])
__msf_world.collect(streaming = True).write_ipc('__msf_world2.ft')
__msf_world = pl.scan_ipc('__msf_world2.ft')
ff_ind_class(__msf_world, 49).collect().write_ipc('__msf_world3.ft')
del __msf_world
nyse_size_cutoffs('__msf_world3.ft')
classify_stocks_size_groups()
return_cutoffs('m', 0)
return_cutoffs('d', 0)
market_returns('world_dsf.ft', 'd', 1, 'return_cutoffs_daily.ft')
market_returns('world_msf.ft', 'm', 1, 'return_cutoffs.ft')
standardized_accounting_data('world',1, 'world_msf.ft',1, "'1949-12-31'")
create_acc_chars('acc_std_ann.ft', 'achars_world.ft', 4, 18, acc_chars, 'world_msf.ft', '')
create_acc_chars('acc_std_qtr.ft', 'qchars_world.ft', 4, 18, acc_chars, 'world_msf.ft', '_qitem')
combine_ann_qtr_chars('achars_world.ft', 'qchars_world.ft', acc_chars, '_qitem')
market_chars_monthly('world_msf.ft', 'market_returns.ft', 0)
create_world_data_prelim('world_msf.ft', 'market_chars_m.ft', 'acc_chars_world.ft', 'world_data_prelim.ft')
ap_factors('ap_factors_daily.ft','d', 'world_dsf.ft', 'world_data_prelim.ft', 'market_returns_daily.ft', 10, 3)
ap_factors('ap_factors_monthly.ft','m', 'world_msf.ft', 'world_data_prelim.ft', 'market_returns.ft', 10, 3)
firm_age('world_msf.ft')
mispricing_factors('world_data_prelim.ft', 10, min_fcts = 3)
market_beta('beta_60m.ft','world_msf.ft','ap_factors_monthly.ft',60,36)
residual_momentum('resmom_ff3', 'world_msf.ft', 'ap_factors_monthly.ft', 36, 24, 12, 1)
residual_momentum('resmom_ff3', 'world_msf.ft', 'ap_factors_monthly.ft', 36, 24, 6, 1)
bidask_hl('corwin_schultz.ft', 'world_dsf.ft', 'market_returns_daily.ft', 10)
prepare_daily('world_dsf.ft', 'ap_factors_daily.ft')
os.system('./call_roll_apply_daily.sh')
merge_roll_apply_daily_results()
finish_daily_chars('market_chars_d.ft')
a = pl.scan_ipc('world_data_prelim.ft')
b = pl.scan_ipc('beta_60m.ft')
c = pl.scan_ipc('resmom_ff3_12_1.ft')
d = pl.scan_ipc('resmom_ff3_6_1.ft')
e = pl.scan_ipc('mp_factors.ft')
f = pl.scan_ipc('market_chars_d.ft')
g = pl.scan_ipc('firm_age.ft').select(['id','eom','age'])
world_data = a.join(b, how = 'left', on = ['id','eom']).join(c, how = 'left', on = ['id','eom']).join(d, how = 'left', on = ['id','eom']).join(e, how = 'left', on = ['id','eom']).join(f, how = 'left', on = ['id','eom']).join(g, how = 'left', on = ['id','eom'])
world_data.collect().write_ipc('world_data_-1.ft')
quality_minus_junk(pl.read_ipc('world_data_-1.ft'),10)
a = pl.scan_ipc('world_data_-1.ft')
b = pl.scan_ipc('qmj.ft')
a.join(b, how = 'left', on = ['excntry','id','eom']).unique(['id','eom']).sort(['id','eom']).collect(streaming=True).write_ipc('world_data.ft')
save_main_data(end_date)
save_daily_ret()
save_monthly_ret()
