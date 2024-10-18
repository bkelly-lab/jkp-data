import polars as pl
import numpy as np
import polars_ols as pls
import time
import datetime
import ibis
import pyarrow as pa
import pyarrow.parquet as pq 
import pyarrow.feather as pf
import os
from datetime import date
from math import sqrt, exp
from functools import reduce
from polars import col
def fl_none(): return pl.lit(None).cast(pl.Float64)
def bo_false(): return pl.lit(False).cast(pl.Boolean)
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function       : {func.__name__.upper()}", flush=True)
        print(f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", flush=True)
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}", flush=True)
        # Calculate total seconds
        total_seconds = end_time - start_time
        # Calculate minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        print(f"Execution time : {minutes} minutes and {seconds:.2f} seconds", flush=True)
        print()
        return result
    return wrapper

@measure_time
def setup_folder_structure():
    os.system('mkdir -p Raw_tables Raw_data_dfs Characteristics World_Ret_Monthly Daily_Returns World_Data Accounting_Data Output')

def collect_and_write(df, filename, collect_streaming = False):
    df.collect(streaming = collect_streaming).write_parquet(filename)

def write_parquet_batches(data, output_path, chunk_size = 100_000, verbose = False):
    
    writer = None
    
    for batch in data.to_pyarrow_batches(chunk_size = chunk_size):
        table = pa.Table.from_batches([batch])
        if writer == None: writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
    
    if writer is not None: writer.close()
        
    if verbose: print(f'Finished writing file to {output_path}')
        
def load_firmshares_aux(filename):
    csho_var = 'cshoq' if 'fundq' in filename else 'csho'
    ajex_var = 'ajexq' if 'fundq' in filename else 'ajex'
    df = (pl.scan_parquet(filename)
            .filter((col('indfmt')  == 'INDL')    &
                    (col('datafmt') == 'STD')     &
                    (col('popsrc')  == 'D')       &
                    (col('consol')  == 'C')       &
                    (col(csho_var).is_not_null()) &
                    (col(ajex_var).is_not_null()))
            .select(['gvkey', 'datadate', col(csho_var).alias('csho_fund'), col(ajex_var).alias('ajex_fund')]))
    return df
def sic_naics_aux(filename):
    df = (pl.scan_parquet(filename)
            .select(['gvkey', 'datadate', col('sich').alias('sic'), col('naicsh').alias('naics')])
            .unique())
    return df
def load_age_aux(filename, filter_monthend = False):
    df = pl.scan_parquet(filename)
    if filter_monthend: df = df.filter(col('monthend') == 1)
    return df.select(['gvkey', 'datadate'])
def comp_hgics_aux(filename):
    df = (pl.scan_parquet(filename)
            .filter(col('gvkey').is_not_null())
            .select(['gvkey', 'indfrom', 'indthru', col('gsubind').alias('gics')])
            .unique())
    return df
def sec_info_aux(filename):
    df = (pl.scan_parquet(filename)
            .select(['gvkey', 'iid', 'secstat', 'dlrsni']))
    return df
def ex_country_aux(filename):
    df = (pl.scan_parquet(filename)
            .select(['exchg', 'excntry'])
            .unique())
    return df
def header_aux(filename):
    df = (pl.scan_parquet(filename)
            .select(['gvkey', 'prirow', 'priusa', 'prican'])
            .unique())
    return df
def prihist_aux(filename, alias_itemvalue):
    df = (pl.scan_parquet(filename)
            .filter(col('item') == alias_itemvalue.upper())
            .select(['gvkey', col('itemvalue').alias(alias_itemvalue), 'effdate', 'thrudate']))
    return df
def comp_dsf_aux(filename):
    if 'comp_secd' in filename:
        aux_exps = [pl.when(col('ajexdi') != 0).then(col('prccd') / col('ajexdi') * col('trfd')).otherwise(fl_none()).alias('ri_local'),
                    col('prccd').alias('prc_local'),
                    pl.when((col('prcstd') != 5)).then(col('prchd')).otherwise(fl_none()).alias('prc_high_lcl'),
                    pl.when((col('prcstd') != 5)).then(col('prcld')).otherwise(fl_none()).alias('prc_low_lcl')]
    if 'comp_g_secd' in filename:
        aux_exps = [pl.when((col('ajexdi') != 0) & (col('qunit') != 0)).then((col('prccd') / col('qunit')) / col('ajexdi') * col('trfd')).otherwise(fl_none()).alias('ri_local'),
                    pl.when(col('qunit') != 0).then(col('prccd') / col('qunit')).otherwise(fl_none()).alias('prc_local'),
                    pl.when((col('prcstd') != 5) & (col('qunit') != 0)).then(col('prchd')/col('qunit')).otherwise(fl_none()).alias('prc_high_lcl'),
                    pl.when((col('prcstd') != 5) & (col('qunit') != 0)).then(col('prcld')/col('qunit')).otherwise(fl_none()).alias('prc_low_lcl')]
    df = (pl.scan_parquet(filename)
            .select(['gvkey', 'iid', 'datadate', 'tpci', 'exchg', 'prcstd', 'curcdd',
                     'ajexdi', 'cshtrd', 'curcddv', 'div', 'divd', 'divsp',
                     (col('cshoc')/1e6).alias('cshoc'), *aux_exps]))
    return df
@measure_time
def gen_raw_data_dfs():
    __firm_shares1 = pl.concat([load_firmshares_aux('Raw_tables/comp_funda.parquet'), load_firmshares_aux('Raw_tables/comp_fundq.parquet')])
    collect_and_write(__firm_shares1, 'Raw_data_dfs/__firm_shares1.parquet')
    sic_naics_na = sic_naics_aux('Raw_tables/comp_funda.parquet')
    collect_and_write(sic_naics_na, 'Raw_data_dfs/sic_naics_na.parquet')
    sic_naics_gl = sic_naics_aux('Raw_tables/comp_g_funda.parquet')
    collect_and_write(sic_naics_gl, 'Raw_data_dfs/sic_naics_gl.parquet')
    comp_acc_age = pl.concat([load_age_aux('Raw_tables/comp_funda.parquet'), load_age_aux('Raw_tables/comp_g_funda.parquet')])
    collect_and_write(comp_acc_age, 'Raw_data_dfs/comp_acc_age.parquet')
    comp_ret_age = pl.concat([load_age_aux('Raw_tables/comp_secm.parquet'), load_age_aux('Raw_tables/comp_g_secd.parquet', True)])
    collect_and_write(comp_ret_age, 'Raw_data_dfs/comp_ret_age.parquet')
    permno0 = (pl.scan_parquet('Raw_tables/crsp_dsenames.parquet')
                .select([col('permno').cast(pl.Int64), col('permco').cast(pl.Int64), 'namedt',
                         'nameendt', col('siccd').cast(pl.Int64).alias('sic'), col('naics').cast(pl.Int64)])
                .unique()
                .sort(['permno', 'namedt', 'nameendt']))
    collect_and_write(permno0, 'Raw_data_dfs/permno0.parquet')
    comp_hgics_na = comp_hgics_aux('Raw_tables/comp_co_hgic.parquet')
    collect_and_write(comp_hgics_na, 'Raw_data_dfs/comp_hgics_na.parquet')
    comp_hgics_gl = comp_hgics_aux('Raw_tables/comp_g_co_hgic.parquet')
    collect_and_write(comp_hgics_gl, 'Raw_data_dfs/comp_hgics_gl.parquet')
    crsp_dsedelist = (pl.scan_parquet('Raw_tables/crsp_dsedelist.parquet')
                        .select(['dlret', 'dlstcd', col('permno').cast(pl.Int64), 'dlstdt']))
    collect_and_write(crsp_dsedelist, 'Raw_data_dfs/crsp_dsedelist.parquet')
    crsp_msedelist = (pl.scan_parquet('Raw_tables/crsp_msedelist.parquet')
                        .select(['dlret', 'dlstcd', col('permno').cast(pl.Int64), 'dlstdt']))
    collect_and_write(crsp_msedelist, 'Raw_data_dfs/crsp_msedelist.parquet')
    __sec_info = pl.concat([sec_info_aux('Raw_tables/comp_security.parquet'), sec_info_aux('Raw_tables/comp_g_security.parquet')])
    collect_and_write(__sec_info, 'Raw_data_dfs/__sec_info.parquet')
    crsp_age = (pl.scan_parquet('Raw_tables/crsp_msf.parquet')
                .with_columns(col('permco').cast(pl.Int64))
                .group_by('permco')
                .agg(crsp_first = pl.min('date')))
    collect_and_write(crsp_age, 'Raw_data_dfs/crsp_age.parquet')
    crsp_mcti_t30ret = (pl.scan_parquet('Raw_tables/crsp_mcti.parquet')
                        .select(['caldt','t30ret']))
    collect_and_write(crsp_mcti_t30ret, 'Raw_data_dfs/crsp_mcti_t30ret.parquet')
    ff_factors_monthly = (pl.scan_parquet('Raw_tables/ff_factors_monthly.parquet')
                            .select(['date', 'rf']))
    collect_and_write(ff_factors_monthly, 'Raw_data_dfs/ff_factors_monthly.parquet')
    comp_r_ex_codes = (pl.scan_parquet('Raw_tables/comp_r_ex_codes.parquet')
                            .select(['exchgdesc', 'exchgcd']))
    collect_and_write(comp_r_ex_codes, 'Raw_data_dfs/comp_r_ex_codes.parquet')
    __ex_country1 = pl.concat([ex_country_aux('Raw_tables/comp_g_security.parquet'), ex_country_aux('Raw_tables/comp_security.parquet')])
    collect_and_write(__ex_country1, 'Raw_data_dfs/__ex_country1.parquet')
    __header = pl.concat([header_aux('Raw_tables/comp_company.parquet'), header_aux('Raw_tables/comp_g_company.parquet')])
    collect_and_write(__header, 'Raw_data_dfs/__header.parquet')
    __prihistcan = prihist_aux('Raw_tables/comp_sec_history.parquet', 'prihistcan')
    collect_and_write(__prihistcan, 'Raw_data_dfs/__prihistcan.parquet')
    __prihistusa = prihist_aux('Raw_tables/comp_sec_history.parquet', 'prihistusa')
    collect_and_write(__prihistusa, 'Raw_data_dfs/__prihistusa.parquet')
    __prihistrow = prihist_aux('Raw_tables/comp_g_sec_history.parquet', 'prihistrow')
    collect_and_write(__prihistrow, 'Raw_data_dfs/__prihistrow.parquet')
    __comp_secm1 = (pl.scan_parquet('Raw_tables/comp_secm.parquet')
                    .select(['gvkey', 'iid', 'datadate', 'tpci', 'exchg', col('curcdm').alias('curcdd'),
                            col('prccm').alias('prc_local'), col('prchm').alias('prc_high'), col('prclm').alias('prc_low'),
                            col('ajexm').alias('ajexdi'), 'cshom', 'csfsm', 'cshoq', 'ajexm', 'dvpsxm', 'cshtrm', 'curcddvm',
                            pl.when(col('trfm') != 0).then(col('prccm') / col('ajexm') * col('trfm')).otherwise(fl_none()).alias('ri_local')]))
    collect_and_write(__comp_secm1, 'Raw_data_dfs/__comp_secm1.parquet')
    __comp_dsf_na = comp_dsf_aux('Raw_tables/comp_secd.parquet')
    collect_and_write(__comp_dsf_na, 'Raw_data_dfs/__comp_dsf_na.parquet')
    __comp_dsf_global = comp_dsf_aux('Raw_tables/comp_g_secd.parquet')
    collect_and_write(__comp_dsf_global, 'Raw_data_dfs/__comp_dsf_global.parquet')
    a = pl.scan_parquet('Raw_tables/comp_exrt_dly.parquet').filter(col('fromcurd') == 'GBP')
    b = pl.scan_parquet('Raw_tables/comp_exrt_dly.parquet').filter(col('tocurd') == 'USD')
    __fx1 = (a.join(b, how = 'inner', on = ['fromcurd', 'datadate'], suffix = '_b')
              .select([col('tocurd').alias('curcdd'), 'datadate', (col('exratd_b')/col('exratd')).alias('fx')])
              .unique()
              .sort(['datadate', 'curcdd']))
    collect_and_write(__fx1, 'Raw_data_dfs/__fx1.parquet')
    write_parquet_batches(gen_crsp_sf('m'), 'Raw_data_dfs/aux.parquet')
    pl.scan_parquet('Raw_data_dfs/aux.parquet').collect(streaming=True).write_parquet('Raw_data_dfs/__crsp_sf_m.parquet')
    write_parquet_batches(gen_crsp_sf('d'), 'Raw_data_dfs/aux.parquet')
    pl.scan_parquet('Raw_data_dfs/aux.parquet').collect(streaming=True).write_parquet('Raw_data_dfs/__crsp_sf_d.parquet')
    os.system('rm Raw_data_dfs/aux.parquet')

def gen_crsp_sf(freq):
    con = ibis.duckdb.connect(threads = os.cpu_count())
    sf = con.read_parquet(f'Raw_tables/crsp_{freq}sf.parquet')
    senames = con.read_parquet(f'Raw_tables/crsp_{freq}senames.parquet')
    ccmxpf_lnkhist = con.read_parquet('Raw_tables/crsp_ccmxpf_lnkhist.parquet')
    sf_senames_join = sf.join(senames, 
                              how = 'left',
                              predicates = [(sf.permno == senames.permno  ), 
                                            (sf.date   >= senames.namedt  ),
                                            (sf.date   <= senames.nameendt)])
    
    full_join = sf_senames_join.join(ccmxpf_lnkhist,
                                     how='left',
                                     predicates=[(sf.permno == ccmxpf_lnkhist.lpermno  ),
                                                 ((sf.date  >= ccmxpf_lnkhist.linkdt   ) | ccmxpf_lnkhist.linkdt.isnull()   ),
                                                 ((sf.date  <= ccmxpf_lnkhist.linkenddt) | ccmxpf_lnkhist.linkenddt.isnull()),
                                                 ccmxpf_lnkhist.linktype.isin(['LC', 'LU', 'LS'])])
    
    
    result = (full_join.mutate(bidask    = (sf.prc < 0).cast('int32'),
                               prc       = sf.prc.abs(),
                               shrout    = (sf.shrout / 1000),
                               me        = (sf.prc.abs() * (sf.shrout / 1000)),
                               prc_high  = ibis.case()
                                               .when((sf.prc > 0) & (sf.askhi > 0), sf.askhi)
                                               .else_(ibis.null())
                                               .end(),
                               prc_low   = ibis.case()
                                               .when((sf.prc > 0) & (sf.bidlo > 0), sf.bidlo)
                                               .else_(ibis.null())
                                               .end(),
                               iid       = ccmxpf_lnkhist.liid,
                               exch_main = senames.exchcd.isin([1, 2, 3]).cast('int32'))
                        .select(['permno','permco','date'   ,'bidask'   ,'prc'     ,'shrout' ,
                                'ret'    ,'retx'  ,'cfacshr','vol'      ,'prc_high','prc_low',
                                'exchcd' ,'gvkey' ,'iid'    ,'exch_main','shrcd'   ,'me'      ]))
    return result

def gen_wrds_connection_object(user, password):
    # Connect to WRDS using Ibis
    con = ibis.postgres.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        user=user,
        password=password,
        database="wrds")
    return con

def download_wrds_table(conn_obj, table_name, filename, cols = None):
    print('Downloading table:' , table_name)

    lib, table = table_name.split('.')
    t = conn_obj.table(database = lib, name = table)
    if cols: t = t.select(cols)

    casting_cols = [t[var].cast('int64').name(var) for var in ['permno', 'permco', 'sic', 'sich'] if var in t.columns]
    if casting_cols: t = t.mutate(*casting_cols)
        
    write_parquet_batches(t, filename)
    del t
    print('Finished')

def check_and_reset_connection(wrds_session, start_time, username, password):
    elapsed_time = time.time() - start_time
    if elapsed_time >= 45 * 60:
        wrds_session.disconnect()
        print('The connection to WRDS server needs to be reset to avoid exceeding time limits.')
        for attempt in range(1, 6):  # Attempts 1 to 5
            # Wait 60 seconds with messages every 10 seconds
            for remaining in range(60, 0, -10):
                print(f"Attempting to reconnect in {remaining} seconds. You might be sent a Duo authentication request.")
                time.sleep(10)
            try:
                wrds_session = gen_wrds_connection_object(username, password)
                print('Connection established. Continuing downloads.')
                start_time = time.time()
                break  # Exit the loop if connection is successful
            except Exception as e:
                print(f"Failed to establish connection (Attempt {attempt} of 5): {e}")
                if attempt < 5:
                    print("Retrying in 60 seconds.")
                else:
                    print("Maximum connection attempts reached. Exiting.")
                    raise e  # Or handle the failure accordingly
    return wrds_session, start_time

@measure_time
def download_raw_data_tables(username, password):
    table_names = ['comp.exrt_dly'  , 'ff.factors_monthly', 'comp.g_security' , 'comp.security'      ,
                   'comp.r_ex_codes', 'comp.g_sec_history', 'comp.sec_history', 'comp.company'       ,
                   'comp.g_company' , 'crsp.msenames'     , 'crsp.dsenames'   , 'crsp.ccmxpf_lnkhist',
                   'comp.funda'     , 'comp.fundq'        , 'crsp.dsedelist'  , 'crsp.msedelist'     ,
                   'comp.secm'      , 'crsp.mcti'         , 'crsp.msf'        , 'comp.g_co_hgic'     ,
                   'crsp.dsf'       , 'comp.g_funda'      , 'comp.co_hgic'    , 'comp.g_fundq']
    
    wrds_session = gen_wrds_connection_object(username, password)
    start_time = time.time()

    for table in table_names: 
        download_wrds_table(wrds_session, table, 'Raw_tables/' + table.replace('.', '_') + '.parquet')
        wrds_session, start_time = check_and_reset_connection(wrds_session, start_time, username, password)

    cols_comp_secd = ['gvkey','iid', 'datadate', 'tpci', 'exchg', 'prcstd', 'curcdd', 'prccd', 'ajexdi', 'cshoc', 'prchd', 'prcld', 'cshtrd', 'trfd', 'curcddv', 'div', 'divd', 'divsp']
    cols_comp_g_secd = ['gvkey', 'iid', 'datadate', 'tpci', 'exchg', 'prcstd','curcdd', 'prccd', 'qunit', 'ajexdi', 'cshoc', 'prchd', 'prcld', 'cshtrd','trfd', 'curcddv', 'div', 'divd', 'divsp', 'monthend']

    download_wrds_table(wrds_session, 'comp.secd', 'Raw_tables/comp_secd.parquet', cols_comp_secd)
    wrds_session, start_time = check_and_reset_connection(wrds_session, start_time, username, password)
    download_wrds_table(wrds_session, 'comp.g_secd', 'Raw_tables/comp_g_secd.parquet', cols_comp_g_secd)

    wrds_session.disconnect()

@measure_time
def prepare_comp_sf(freq):
    populate_own('Raw_data_dfs/__firm_shares1.parquet', 'gvkey', 'datadate', 'ddate')
    gen_comp_dsf()
    if freq == 'both':
        process_comp_sf1('d')
        process_comp_sf1('m')
    else: process_comp_sf1(freq)

@measure_time
def populate_own(inset_path, idvar, datevar, datename):
    inset = (pl.scan_parquet(inset_path)
               .unique([idvar, datevar])
               .with_columns(col(datevar).alias(datename))
               .sort([idvar, datevar])
               .with_columns(n = pl.min_horizontal((col('datadate').shift(-1)).over(idvar), col(datevar).dt.offset_by('12mo').dt.month_end()).dt.offset_by('-1d'))
               .with_columns(pl.date_ranges('ddate', 'n'))
               .explode('ddate')
               .select(['ddate','gvkey','datadate', 'csho_fund', 'ajex_fund'])
               .sort(['gvkey','datadate']))
    inset.collect().write_parquet('__firm_shares2.parquet')
def compustat_fx():
    aux = pl.DataFrame({'curcdd': 'USD','datadate': '1950-01-01','fx': 1.0}).with_columns(col('datadate').str.to_date('%Y-%m-%d')).lazy()
    __fx1 = pl.scan_parquet('Raw_data_dfs/__fx1.parquet')
    __fx1 = (pl.concat([aux, __fx1], how = 'vertical_relaxed')
               .sort(['curcdd', 'datadate'])
               .with_columns(aux = col('datadate').shift(-1).over('curcdd'))
               .with_columns(datadate = pl.coalesce([pl.date_ranges(start='datadate', end='aux', interval='1d', closed = 'left'), pl.concat_list([col('datadate')])]))
               .select(['datadate','curcdd', 'fx'])
               .explode('datadate')
               .unique(['curcdd','datadate'])
               .sort(['curcdd', 'datadate']))
    return __fx1.collect()

@measure_time
def gen_comp_dsf():
    fx = compustat_fx().lazy()
    fx_div = fx.clone().rename({'fx': 'fx_div'}).lazy()
    __comp_dsf_global = pl.scan_parquet('Raw_data_dfs/__comp_dsf_global.parquet')
    aux = pl.scan_parquet('__firm_shares2.parquet').select(['ddate','gvkey','csho_fund','ajex_fund'])
    __comp_dsf = (pl.scan_parquet('Raw_data_dfs/__comp_dsf_na.parquet')
                    .join(aux, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'ddate'])
                    .with_columns(cshoc  = pl.coalesce(['cshoc', pl.when(col('ajexdi') != 0).then(col('csho_fund')*col('ajex_fund')/col('ajexdi')).otherwise(fl_none())]),
                                  cshtrd = adj_trd_vol_NASDAQ('datadate', 'cshtrd', 'exchg', 14))
                    .drop(['csho_fund','ajex_fund']))
    __comp_dsf = (pl.concat([__comp_dsf, __comp_dsf_global.select(__comp_dsf.collect_schema().names())], how = 'vertical_relaxed')
                    .join(fx, how = 'left', on = ['datadate', 'curcdd']) #fx datadate in our terminology is the same as date in the SAS code
                    .join(fx_div, how = 'left', left_on = ['datadate', 'curcddv'], right_on = ['datadate', 'curcdd'])
                    .with_columns(prc      = col('prc_local')    * col('fx'),
                                  prc_high = col('prc_high_lcl') * col('fx'),
                                  prc_low  = col('prc_low_lcl')  * col('fx'),
                                  ri       = col('ri_local')     * col('fx'))
                    .with_columns(me       = col('prc') * col('cshoc'),
                                  dolvol   = col('prc') * col('cshtrd'),
                                  div_tot  = pl.coalesce(['div'  , 0]) * col('fx_div'),
                                  div_cash = pl.coalesce(['divd' , 0]) * col('fx_div'),
                                  div_spc  = pl.coalesce(['divsp', 0]) * col('fx_div'),
                                  eom      = col('datadate').dt.month_end())
                    .drop(['div', 'divd', 'divsp', 'fx_div', 'curcddv', 'prc_high_lcl', 'prc_low_lcl']))
    __comp_dsf.collect(streaming=True).write_parquet('__comp_dsf.parquet')
def adj_trd_vol_NASDAQ(datevar, col_to_adjust, exchg_var, exchg_val):
    c1 = col(exchg_var) == exchg_val
    c2 = col(datevar)   <  pl.datetime(2001, 2, 1)
    c3 = col(datevar)   <= pl.datetime(2001, 12, 31)
    c4 = col(datevar)   <  pl.datetime(2003, 12, 31)
    adj_trd_vol = (pl.when(c1 & c2).then(col(col_to_adjust) / 2)
                     .when(c1 & c3).then(col(col_to_adjust) / 1.8)
                     .when(c1 & c4).then(col(col_to_adjust) / 1.6)
                     .otherwise(col(col_to_adjust))).alias(col_to_adjust)
    return adj_trd_vol

@measure_time
def gen_comp_msf():
    col_names_aux  = ['prc_highm','prc_lowm','div_totm','div_cashm','div_spcm','cshtrm']
    set_aux_cols   = [pl.when(col('ajexdi') != 0).then(pl.max_horizontal('prc', 'prc_high') / col('ajexdi')).otherwise(fl_none()).alias('aux1'),
                      pl.when(col('ajexdi') != 0).then(pl.max_horizontal('prc', 'prc_low')  / col('ajexdi')).otherwise(fl_none()).alias('aux2')]
    set_aux_cols  += [safe_div(var, 'ajexdi', f'aux{i+1}') for var, i in zip(['div_tot', 'div_cash', 'div_spc', 'cshtrd'], range(2, 6))]
    set_aux_cols2  = [col('aux1').max().over(['gvkey', 'iid', 'eom']), col('aux2').min().over(['gvkey', 'iid', 'eom'])]
    set_aux_cols2 += [col(f'aux{i+1}').sum().over(['gvkey', 'iid', 'eom']) for i in range(2, 6)]
    set_aux_cols3  = [pl.sum('dolvol').over(['gvkey', 'iid', 'eom']).alias('dolvolm')]
    set_aux_cols3 += [(col(f'aux{j+1}') * col('ajexdi')).alias(i) for i, j in zip(col_names_aux, range(6))]
    cols_to_drop_aux = ['cshtrd', 'div_tot', 'div_cash','div_spc', 'dolvol', 'prc_high', 'prc_low']  + [f'aux{i+1}' for i in range(6)]
    dict_aux = {'div_totm': 'div_tot', 'div_cashm': 'div_cash', 'div_spcm': 'div_spc', 'dolvolm': 'dolvol', 'prc_highm': 'prc_high', 'prc_lowm': 'prc_low'}
    __comp_msf = (pl.scan_parquet('__comp_dsf.parquet')
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
    aux_ajexm_exp = pl.when(col('ajexm') != 0).then(col('csho_fund') * col('ajex_fund') / col('ajexm')).otherwise(fl_none())
    __comp_secm = pl.scan_parquet('Raw_data_dfs/__comp_secm1.parquet')
    fx = compustat_fx().lazy()
    fx_div = fx.clone().with_columns(col('fx').alias('fx_div')).drop('fx')
    aux = pl.scan_parquet('__firm_shares2.parquet')
    __comp_secm = (__comp_secm.join(aux, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'ddate'])
                              .join(fx, how = 'left', on = ['datadate', 'curcdd']) #fx datadate in our terminology is the same as date in Theis' code
                              .join(fx_div, how = 'left', left_on = ['datadate', 'curcddvm'], right_on = ['datadate', 'curcdd'])
                              .with_columns(eom      = col('datadate').dt.month_end(),
                                            cshoc    = pl.coalesce(col('cshom') / 1e6, col('csfsm') / 1e3, col('cshoq'), aux_ajexm_exp),
                                            chstrm   = adj_trd_vol_NASDAQ('datadate', 'cshtrm', 'exchg', 14),
                                            fx       = pl.when(col('curcdd') == 'USD').then(pl.lit(1)).otherwise(col('fx')),
                                            fx_div   = pl.when(col('curcddvm') == 'USD').then(pl.lit(1)).otherwise(col('fx_div')).alias('fx_div'))
                              .with_columns(prc      = col('prc_local') * col('fx'),
                                            prc_high = col('prc_high')  * col('fx'),
                                            prc_low  = col('prc_low')   * col('fx'),
                                            ri       = col('ri_local')  * col('fx'),
                                            div_tot  = col('dvpsxm')    * col('fx_div'))
                              .with_columns(me       = col('prc')       * col('cshoc'),
                                            dolvol   = col('prc')       * col('cshtrm'),
                                            div_cash = fl_none(),
                                            div_spc  = fl_none(),
                                            prcstd   = pl.lit(10).cast(pl.Int32),
                                            source   = pl.lit(2).cast(pl.Int32),
                                            exchg    = col('exchg').cast(pl.Int32)))
    common_vars = ['gvkey', 'iid', 'datadate', 'eom', 'tpci', 'exchg', 'curcdd', 'prc_local', 'prc_high', 'prc_low', 'ajexdi', 'cshoc', 'ri_local', 'fx', 'prc', 'me', 'cshtrm', 'dolvol', 'ri', 'div_tot', 'div_cash', 'div_spc', 'prcstd', 'source']
    __comp_msf = (pl.concat([__comp_msf.select(common_vars), __comp_secm.select(common_vars)], how = 'vertical_relaxed')
                    .sort('source')
                    .unique(subset = ['gvkey', 'iid', 'eom'], keep = 'first')
                    .drop('source')
                    .sort(['gvkey', 'iid', 'eom']))
    __comp_msf.collect().write_parquet('__comp_msf.parquet')

@measure_time
def comp_exchanges():
    special_exchanges = [
        0, 1, 2, 3, 4, 15, 16, 17, 18, 21,
        13, 19, 20, 127, 150, 157, 229, 263, 269, 281,
        283, 290, 320, 326, 341, 342, 347, 348, 349, 352]
    #15, 16, 17, 18, 21 US exchanges not in NYSE, Amex and NASDAQ
    #150 AIAF Mercado De Renta Fija --> Spanish exchange for trading debt securities https://practiceguides.chambers.com/practice-guides/capital-markets-debt-2019/spain/1-debt-marketsexchanges
    #349 BATS Chi-X Europe --> Trades stocks from various european exchanges. Should we keep it?
    #352 CHI-X Australia --> Only Trades securities listed on ASX (exchg=106). Should we keep it?
    SQL_query = """
        SELECT DISTINCT exchg,
            CASE
                WHEN COUNT(DISTINCT excntry) > 1 THEN 'multi national'
                ELSE MAX(excntry)
            END AS excntry
        FROM frame
        WHERE excntry IS NOT NULL AND exchg IS NOT NULL
        GROUP BY exchg
        """
    exch_exp = pl.when((col('excntry') != 'multi national') & (col('exchg').is_in(special_exchanges).not_()))\
                 .then(pl.lit(1))\
                 .otherwise(pl.lit(0))\
                 .alias('exch_main')
    comp_r_ex_codes = pl.read_parquet('Raw_data_dfs/comp_r_ex_codes.parquet')
    __ex_country = pl.read_parquet('Raw_data_dfs/__ex_country1.parquet')
    __ex_country = (pl.SQLContext(frame = __ex_country)
                      .execute(SQL_query)
                      .collect()
                      .sort('exchg')
                      .join(comp_r_ex_codes, how = 'left', left_on = 'exchg', right_on = 'exchgcd')
                      .with_columns(col('exchg').cast(pl.Int64))
                      .with_columns(exch_exp))
    return __ex_country
def gen_prihist_df(path_aux_data, geo, datevar):
    c1 = col(datevar) >= col('effdate')
    c2 = col(datevar) <= col('thrudate')
    c3 = col('thrudate').is_null()
    aux_data = pl.read_parquet(path_aux_data, columns = ['gvkey', datevar]).unique()
    prihistr = pl.read_parquet(f'Raw_data_dfs/__prihist{geo}.parquet')
    prihistr = (aux_data.join(prihistr, how = 'left', on = 'gvkey')
                        .filter(c1 & (c2|c3))
                        .drop(['effdate','thrudate']))
    return prihistr
@measure_time
def add_primary_sec(data_path, datevar, file_name):
    c1 = (col('iid').is_not_null()) & ((col('iid') == col('prihistrow')) | (col('iid') == col('prihistusa')) | (col('iid') == col('prihistcan')))
    data = pl.read_parquet(data_path)
    for df in [gen_prihist_df(data_path, 'row', datevar), gen_prihist_df(data_path, 'usa', datevar), gen_prihist_df(data_path, 'can', datevar)]:
        data = data.join(df, how = 'left', on = ['gvkey', datevar])
    data = (data.join(pl.read_parquet('Raw_data_dfs/__header.parquet').unique('gvkey', keep = 'first'), how = 'left', on = 'gvkey')#Header has duplicates
                .with_columns(prihistrow  = pl.coalesce(['prihistrow', 'prirow']),
                              prihistusa  = pl.coalesce(['prihistusa', 'priusa']),
                              prihistcan  = pl.coalesce(['prihistcan', 'prican']))
                .with_columns(primary_sec = pl.when(c1).then(pl.lit(1)).otherwise(pl.lit(0)))
                .drop(['prihistrow','prihistusa','prihistcan', 'prirow', 'priusa', 'prican'])
                .unique(['gvkey', 'iid', 'datadate']))
    data.write_parquet(file_name)
def load_rf_and_exchange_data():
    crsp_mcti = (pl.read_parquet('Raw_data_dfs/crsp_mcti_t30ret.parquet')
                   .with_columns(merge_aux = gen_MMYY_column('caldt'))
                   .drop('caldt'))
    ff_factors_monthly = (pl.read_parquet('Raw_data_dfs/ff_factors_monthly.parquet')
                            .with_columns(merge_aux = gen_MMYY_column('date'))
                            .drop('date'))
    __exchanges = comp_exchanges()
    return crsp_mcti, ff_factors_monthly, __exchanges
def gen_returns_df(freq):
    ret_lag_dif_exp = (gen_MMYY_column('datadate') - gen_MMYY_column('datadate', 1)).over(['gvkey','iid']) if freq == 'm' else (col('datadate') -col('datadate').shift(1)).over(['gvkey','iid'])
    base = pl.scan_parquet(f'__comp_{freq}sf.parquet')
    __returns = (base.filter((col('ri').is_not_null()) & (col('prcstd').is_in([3, 4, 10])))
                     .select(['gvkey','iid','datadate','ri','ri_local','prcstd','curcdd'])
                     .unique(['gvkey','iid','datadate'])
                     .sort(['gvkey', 'iid', 'datadate'])
                     .with_columns(ret           = col('ri').pct_change().over(['gvkey','iid']),
                                   ret_local     = col('ri_local').pct_change().over(['gvkey','iid']),
                                   ret_lag_dif   = ret_lag_dif_exp,
                                   lagged_iid    = col('iid').shift(1).over(['gvkey','iid']),
                                   lagged_curcdd = col('curcdd').shift(1).over(['gvkey','iid']))
                     .with_columns(ret_local = pl.when((col('iid') == col('lagged_iid')) & (col('curcdd') != col('lagged_curcdd'))).then(col('ret')).otherwise(col('ret_local'))))
    return __returns.collect()
def gen_delist_df(__returns):
    __sec_info = pl.read_parquet('Raw_data_dfs/__sec_info.parquet')
    __delist = (__returns.filter((col('ret_local').is_not_null()) & (col('ret_local') != 0.))
                         .select(['gvkey', 'iid', 'datadate'])
                         .sort(['gvkey', 'iid', 'datadate'])
                         .unique(['gvkey', 'iid'], keep = 'last')
                         .join(__sec_info, how = 'left', on = ['gvkey', 'iid'])
                         .rename({'datadate': 'date_delist'})
                         .filter(col('secstat') == 'I')
                         .with_columns(dlret = pl.when(col('dlrsni').is_in(['02', '03'])).then(pl.lit(-0.3)).otherwise(pl.lit(0.)))
                         .select(['gvkey','iid','date_delist','dlret']))
    return __delist
def gen_temporary_sf(freq, __returns, __delist):
    base = pl.read_parquet(f'__comp_{freq}sf.parquet')
    temp_sf = (base.join(__returns, how = 'left', on = ['gvkey','iid', 'datadate'])
                   .join(__delist, how = 'left', on = ['gvkey', 'iid'])
                   .filter((col('datadate') <= col('date_delist')) | (col('date_delist').is_null()))
                   .with_columns(ret       = pl.when(col('datadate') == col('date_delist')).then((1+col('ret'))*(1+col('dlret')) - 1).otherwise(col('ret')),
                                 ret_local = pl.when(col('datadate') == col('date_delist')).then((1+col('ret_local'))*(1+col('dlret')) - 1).otherwise(col('ret_local')))
                   .drop(['ri', 'ri_local', 'date_delist', 'dlret']))
    return temp_sf
def add_rf_and_exchange_data_to_temporary_sf(freq, temp_sf):
    crsp_mcti, ff_factors_monthly, __exchanges = load_rf_and_exchange_data()
    scale = 1 if (freq == 'm') else 21
    temp_sf = (temp_sf.with_columns(merge_aux = gen_MMYY_column('datadate'))
                      .join(crsp_mcti, how = 'left', on = 'merge_aux')
                      .join(ff_factors_monthly, how = 'left', on = 'merge_aux')
                      .with_columns(ret_exc = col('ret') - pl.coalesce(['t30ret','rf'])/scale)
                      .drop(['merge_aux', 'rf', 't30ret'])
                      .with_columns(col('exchg').cast(pl.Int64))
                      .join(__exchanges, how = 'left', on = ['exchg']))
    return temp_sf

@measure_time
def process_comp_sf1(freq):
    #Eager mode is faster here
    if freq == 'm': gen_comp_msf()
    __returns  = gen_returns_df(freq)
    __delist   = gen_delist_df(__returns)
    __comp_sf2 = gen_temporary_sf(freq, __returns, __delist)
    __comp_sf2 = add_rf_and_exchange_data_to_temporary_sf(freq, __comp_sf2)
    __comp_sf2.write_parquet('__comp_sf2.parquet')
    del __comp_sf2
    add_primary_sec('__comp_sf2.parquet', 'datadate',f'comp_{freq}sf.parquet')

def gen_MMYY_column(var, shift = None):
    if shift == None:
        return (col(var).dt.year()*12 + col(var).dt.month()).cast(pl.Int32)
    else:
        return (col(var).shift(1).dt.year()*12 + col(var).shift(1).dt.month()).cast(pl.Int32)
def add_MMYY_column_drop_original(df, var): return df.with_columns(merge_aux = gen_MMYY_column(var)).drop(var)

@measure_time
def prepare_crsp_sf(freq):
    merge_vars = ['permno', 'merge_aux'] if (freq == 'm') else ['permno', 'date']
    __crsp_sf = (pl.scan_parquet(f'Raw_data_dfs/__crsp_sf_{freq}.parquet')
                   .with_columns(adj_trd_vol_NASDAQ('date', 'vol', 'exchcd', 3))
                   .sort(['permno', 'date'])
                   .with_columns(dolvol = col('prc').abs() * col('vol'),
                                 div_tot = pl.when(col('cfacshr').shift(1).over('permno') != 0).then( ( (col('ret') - col('retx')) * col('prc').shift(1) * (col('cfacshr') / col('cfacshr').shift(1)) ).over('permno')).otherwise(fl_none()),
                                 permno = col('permno').cast(pl.Int64),
                                 merge_aux = gen_MMYY_column('date')))
    c1 = col('dlret').is_null()
    c2 = col('dlstcd') == 500
    c3 = col('dlstcd').is_between(520, 584)
    c4 = (c1 & (c2 | c3))
    c5 = col('ret').is_null()
    c6 = col('dlret').is_not_null()
    c7 = (c5 & c6)
    crsp_sedelist_aux_col = [gen_MMYY_column('dlstdt').alias('merge_aux')] if (freq == 'm') else [col('dlstdt').alias('date')]
    crsp_sedelist = pl.scan_parquet(f'Raw_data_dfs/crsp_{freq}sedelist.parquet').with_columns(crsp_sedelist_aux_col)
    crsp_mcti = add_MMYY_column_drop_original(pl.scan_parquet('Raw_data_dfs/crsp_mcti_t30ret.parquet'), 'caldt')
    ff_factors_monthly = add_MMYY_column_drop_original(pl.scan_parquet('Raw_data_dfs/ff_factors_monthly.parquet'), 'date')
    me_company_exp = (pl.when(pl.count('me').over(['permco', 'date']) != 0)
                        .then(pl.coalesce(['me',0.]).sum().over(['permco', 'date']))
                        .otherwise(fl_none()))
    scale = 1 if (freq == 'm') else 21
    ret_exc_exp = col('ret') - pl.coalesce(['t30ret','rf'])/scale
    __crsp_sf = (__crsp_sf.join(crsp_sedelist, how = 'left', on = merge_vars)
                          .with_columns(dlret = pl.when(c4).then(pl.lit(-0.3)).otherwise(col('dlret')),
                                        ret = pl.when(c7).then(pl.lit(0.)).otherwise(col('ret')))
                          .with_columns(ret = (((col('ret') + 1) * (pl.coalesce(['dlret', 0.]) + 1)) -1))
                          .join(crsp_mcti, how = 'left', on = 'merge_aux')
                          .join(ff_factors_monthly, how = 'left', on = 'merge_aux')
                          .with_columns(ret_exc = ret_exc_exp,
                                       me_company = me_company_exp))
    if freq == 'm': __crsp_sf = __crsp_sf.with_columns([(col(var) * 100).alias(var) for var in ['vol', 'dolvol']])
    __crsp_sf = (__crsp_sf.drop(['rf', 't30ret','merge_aux','dlret', 'dlstcd', 'dlstdt'])
                          .unique(['permno', 'date'])
                          .sort(['permno', 'date']))
    __crsp_sf.collect(streaming=True).write_parquet(f'crsp_{freq}sf.parquet')

def prepare_crsp_sfs_for_merging():
    crsp_msf = (pl.scan_parquet('crsp_msf.parquet')
                  .with_columns(exch_main   = col('exch_main').cast(pl.Int32),
                                bidask      = col('bidask').cast(pl.Int32),
                                id          = col('permno'),
                                excntry     = pl.lit('USA'),
                                common      = (col('shrcd').is_in([10, 11, 12]).fill_null(bo_false())).cast(pl.Int32),
                                primary_sec = pl.lit(1),
                                comp_tpci   = pl.lit(''),
                                comp_exchg  = pl.lit(None).cast(pl.Int64),
                                curcd       = pl.lit('USD'),
                                fx          = pl.lit(1.),
                                eom         = col('date').dt.month_end(),
                                prc_local   = col('prc'),
                                tvol        = col('vol'),
                                ret_local   = col('ret'),
                                ret_lag_dif = pl.lit(1).cast(pl.Int64),
                                div_cash    = fl_none(),
                                div_spc     = fl_none(),
                                source_crsp = pl.lit(1))
                  .rename({'shrcd'  : 'crsp_shrcd',
                           'exchcd' : 'crsp_exchcd',
                           'cfacshr': 'adjfct',
                           'shrout' : 'shares'}))

    crsp_dsf = (pl.scan_parquet('crsp_dsf.parquet')
                  .with_columns(id          = col('permno'),
                                excntry     = pl.lit('USA'),
                                common      = (col('shrcd').is_in([10, 11, 12]).fill_null(bo_false())).cast(pl.Int32),
                                primary_sec = pl.lit(1),
                                curcd       = pl.lit('USD'),
                                fx          = pl.lit(1.),
                                eom         = col('date').dt.month_end(),
                                ret_local   = col('ret'),
                                ret_lag_dif = pl.lit(1).cast(pl.Int64),
                                exch_main   = col('exch_main').cast(pl.Int32),
                                bidask      = col('bidask').cast(pl.Int32),
                                source_crsp = pl.lit(1))
                  .rename({'cfacshr': 'adjfct',
                           'shrout' : 'shares',
                           'vol'    : 'tvol'}))
    return crsp_msf, crsp_dsf
def prepare_comp_sfs_for_merging():
    id_exp = (pl.when(col('iid').str.contains('W')).then(pl.lit('3') + col('gvkey') + col('iid').str.slice(0,2))
                .when(col('iid').str.contains('C')).then(pl.lit('2') + col('gvkey') + col('iid').str.slice(0,2))
                .otherwise(pl.lit('1') + col('gvkey') + col('iid').str.slice(0,2))).cast(pl.Int64)

    comp_msf = (pl.scan_parquet('comp_msf.parquet')
                  .with_columns(id          = id_exp,
                                permno      = pl.lit(None).cast(pl.Int64),
                                permco      = pl.lit(None).cast(pl.Int64),
                                common      = pl.when(col('tpci') == '0').then(pl.lit(1)).otherwise(pl.lit(0)),
                                bidask      = pl.when(col('prcstd') == 4).then(pl.lit(1)).otherwise(pl.lit(0)),
                                crsp_shrcd  = fl_none(),
                                crsp_exchcd = fl_none(),
                                me_company  = col('me'),
                                source_crsp = pl.lit(0),
                                ret_lag_dif = col('ret_lag_dif').cast(pl.Int64))
                  .rename({'tpci'    : 'comp_tpci',
                           'exchg'   : 'comp_exchg',
                           'curcdd'  : 'curcd',
                           'datadate': 'date',
                           'ajexdi'  : 'adjfct',
                           'cshoc'   : 'shares',
                           'cshtrm'  : 'tvol'}))

    comp_dsf = (pl.scan_parquet('comp_dsf.parquet')
                  .with_columns(id          = id_exp,
                                common      = pl.when(col('tpci') == '0').then(pl.lit(1)).otherwise(pl.lit(0)),
                                bidask      = pl.when(col('prcstd') == 4).then(pl.lit(1)).otherwise(pl.lit(0)),
                                eom         = col('datadate').dt.month_end(),
                                source_crsp = pl.lit(0),
                                ret_lag_dif = (col('ret_lag_dif')/86_400_000).cast(pl.Int64))
                  .rename({'curcdd'  : 'curcd',
                           'datadate': 'date',
                           'ajexdi'  : 'adjfct',
                           'cshoc'   : 'shares',
                           'cshtrd'  : 'tvol'}))
    return comp_msf, comp_dsf
def gen_temp_sf(freq, crsp_df, comp_df):
    cols_to_keep = ['id','permno','permco','gvkey', 'iid','excntry', 'exch_main', 'common', 'primary_sec', 'bidask','crsp_shrcd',\
                    'crsp_exchcd','comp_tpci','comp_exchg', 'curcd', 'fx', 'date', 'eom','adjfct','shares', 'me', 'me_company',\
                    'prc','prc_local', 'prc_high', 'prc_low', 'dolvol','tvol', 'ret','ret_local','ret_exc','ret_lag_dif',\
                    'div_tot','div_cash','div_spc','source_crsp']\
                   if freq == 'm' else\
                   ['id','excntry','exch_main','common','primary_sec','bidask','curcd','fx','date','eom','adjfct','shares',\
                    'me','dolvol','tvol','prc','prc_high','prc_low','ret_local','ret','ret_exc','ret_lag_dif','source_crsp']
    sf_world = pl.concat([crsp_df.select(cols_to_keep), comp_df.select(cols_to_keep)], how = 'vertical_relaxed')
    if freq == 'm':
        c1 = (col('ret_lag_dif') == 1) & (col('id_lead1m') == col('id'))
        sf_world = (sf_world.sort(['id', 'eom'])
                            .with_columns(ret_exc_lead1m     = col('ret_exc').shift(-1).over('id'),
                                          id_lead1m          = col('id').shift(-1),
                                          ret_lag_dif_lead1m = col('ret_lag_dif').shift(-1).over('id'))
                            .with_columns(ret_exc_lead1m     = pl.when(c1).then(col('ret_exc_lead1m')).otherwise(fl_none()))
                            .drop(['id_lead1m', 'ret_lag_dif_lead1m']))
    return sf_world
def add_obs_main_to_sf_and_write_file(freq, sf_df, obs_main):
    file_path = '__msf_world.parquet' if freq == 'm' else 'world_dsf.parquet'
    sort_vars = ['id','eom'] if freq == 'm' else ['id','date']
    (sf_df.join(obs_main, on = ['id','eom'], how = 'left')
          .unique(sort_vars)
          #.sort(sort_vars)
          .collect(streaming = True)
          .write_parquet(file_path))
    
@measure_time
def combine_crsp_comp_sf():
    crsp_msf, crsp_dsf = prepare_crsp_sfs_for_merging()
    comp_msf, comp_dsf = prepare_comp_sfs_for_merging()
    __msf_world = gen_temp_sf('m', crsp_msf, comp_msf)
    __dsf_world = gen_temp_sf('d', crsp_dsf, comp_dsf)
    obs_main = (__msf_world.select(['id','source_crsp','gvkey','iid','eom'])
                           .with_columns(count    = pl.count('gvkey').over(['gvkey', 'iid', 'eom']))
                           .with_columns(obs_main = pl.when((col('count').is_in([0, 1])) | ((col('count') > 1) & (col('source_crsp') == 1))).then(1).otherwise(0))
                           .drop(['count','iid','gvkey','source_crsp']))
    add_obs_main_to_sf_and_write_file('m', __msf_world, obs_main)
    add_obs_main_to_sf_and_write_file('d', __dsf_world, obs_main)

@measure_time
def crsp_industry():
    permno0 = pl.scan_parquet('Raw_data_dfs/permno0.parquet')
    permno0 = (permno0.with_columns(sic = pl.when(col('sic') == 0).then(pl.lit(None).cast(pl.Int64)).otherwise(col('sic')))
                      .with_columns(date = pl.date_ranges('namedt', 'nameendt'))
                      .explode('date')
                      .select(['permno', 'permco', 'date', 'sic', 'naics'])
                      .unique(['permno', 'date'])
                      .sort(['permno', 'date']))
    permno0.collect().write_parquet('crsp_ind.parquet')

@measure_time
def comp_hgics(lib):
    paths = {'raw data': {'national': 'Raw_data_dfs/comp_hgics_na.parquet', 'global': 'Raw_data_dfs/comp_hgics_gl.parquet'},
             'output' : {'national': 'na_hgics.parquet', 'global': 'g_hgics.parquet'}}
    data = pl.read_parquet(paths['raw data'][lib]).sort(['gvkey', 'indfrom'])
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
    data.write_parquet(paths['output'][lib])

@measure_time
def hgics_join():
    comp_hgics('global')
    comp_hgics('national')
    global_data = pl.scan_parquet('g_hgics.parquet')
    local_data = pl.scan_parquet('na_hgics.parquet')
    gjoin = local_data.join(global_data, on = ['gvkey', 'date'], how = 'outer_coalesce')
    gjoin = (gjoin.with_columns(gics = pl.coalesce(['gics', 'gics_right']))
                  .select(['gvkey', 'date', 'gics'])
                  .unique(['gvkey','date'])
                  .sort(['gvkey','date']))
    gjoin.collect().write_parquet('comp_hgics.parquet')
    
@measure_time
def comp_sic_naics():
    funda_data  = pl.scan_parquet('Raw_data_dfs/sic_naics_na.parquet')
    gfunda_data = pl.scan_parquet('Raw_data_dfs/sic_naics_gl.parquet')
    funda_data = funda_data.filter(~( (col('gvkey') == '175650') & (col('datadate') == pl.date(2005,12,31)) & (col('naics').is_null() )))
    comp = funda_data.join(gfunda_data, on = ['gvkey', 'datadate'], how = 'outer_coalesce').sort(['gvkey','datadate'])
    comp = (comp.with_columns(sic = pl.coalesce(['sic', 'sic_right']), naics = pl.coalesce(['naics', 'naics_right']), end_date = col('datadate').shift(-1).over('gvkey'))
                .with_columns(end_date = pl.when(col('end_date').is_null()).then(col('datadate')).otherwise(col('end_date')))
                .with_columns(date = pl.date_ranges('datadate', 'end_date', closed = 'left'))
                .explode('date')
                .with_columns(date = pl.when(col('datadate') == col('end_date')).then(col('datadate')).otherwise(col('date')))
                .select(['gvkey', 'date', 'sic', 'naics'])
                .unique(['gvkey', 'date'])
                .sort(['gvkey', 'date']))
    comp.collect().write_parquet('comp_other.parquet')

@measure_time
def comp_industry():
    comp_sic_naics()
    hgics_join()
    comp_other = pl.read_parquet('comp_other.parquet')
    comp_gics = pl.read_parquet('comp_hgics.parquet')
    join = comp_gics.join(comp_other, on=['gvkey', 'date'], how='outer_coalesce').sort(['gvkey', 'date'])
    join = (join.with_columns(aux_date = (col('date').shift(-1).dt.offset_by('-1d')).over('gvkey'))
                .with_columns(aux_date = pl.when(col('aux_date').is_null()).then(col('date')).otherwise('aux_date')))
    gaps = (join.filter(col('date') != col('aux_date'))
                .select(['gvkey', pl.date_ranges('date','aux_date', closed = 'right').alias('date')])
                .explode('date'))
    join = join.filter(col('date') == col('aux_date')).drop('aux_date')
    join = pl.concat([join, gaps], how = 'diagonal').unique(['gvkey', 'date']).sort(['gvkey', 'date'])
    join.write_parquet('comp_ind.parquet')

@measure_time
def ff_ind_class(data_path, ff_grps):
    data = pl.scan_parquet(data_path)
    if ff_grps==38:
        lower_bounds = [1000, 1300, 1400, 1500, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4800, 4830, 4900, 4950, 4960, 4970, 5000, 5200, 6000, 7000, 9000]
        upper_bounds = [1299, 1399, 1499, 1799, 2099, 2199, 2299, 2399, 2499, 2599, 2661, 2799, 2899, 2999, 3099, 3199, 3299, 3399, 3499, 3599, 3699, 3799, 3879, 3999, 4799, 4829, 4899, 4949, 4959, 4969, 4979, 5199, 5999, 6999, 8999, 9999]
        classification = [i + 2 for i in range(len(lower_bounds))]
        ff38_exp = pl.when(col('sic').is_between(100, 999)).then(1)
        for i, j, k in zip(lower_bounds, upper_bounds, classification): ff38_exp = ff38_exp.when(col('sic').is_between(i, j)).then(k)
        ff38_exp = (ff38_exp.otherwise(pl.lit(None))).alias('ff38')
        data = data.with_columns(ff38_exp)
    else:
        data = data.with_columns(pl.when(col("sic").is_in([2048,*range(100, 299+1), *range(700, 799+1), *range(910, 919+1)])).then(1)
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

    data.collect().write_parquet('__msf_world3.parquet')
def perc_method(series, p):
    """
    Calculates the given percentile using the SAS 5th method, which is the default SAS method and was used in our SAS code.
    """
    n = len(series)
    rank = p * n
    if rank.is_integer():return (series[int(rank) - 1] + series[int(rank)]) / 2
    else: return series[int(rank)]
def perc_exp(var, perc_function, list = False, type = 'float'):
    if list: return col(var).list.sort().map_elements(perc_function, return_dtype = pl.Float64)
    else: return col(var).sort().map_elements(perc_function, return_dtype = pl.Float64)
def winsorize_var(df, sort_vars, wins_var, perc_low, perc_high):
    aux = (df.group_by(sort_vars)
                .agg(col(wins_var))
                .with_columns(low  = perc_exp(wins_var, lambda x: perc_method(x, perc_low), True),
                              high = perc_exp(wins_var, lambda x: perc_method(x, perc_high), True))
                .select([*sort_vars, 'low','high']))
    wins_df = (df.join(aux, how = 'left', on = sort_vars)
                    .with_columns(ret_exc = (pl.when(col(wins_var) < col('low' )).then(col('low' ))
                                            .when(col(wins_var) > col('high')).then(col('high'))
                                            .otherwise(col(wins_var))))
                    .drop(['low', 'high']))
    return wins_df

@measure_time
def nyse_size_cutoffs(data_path):
    nyse_sf = (pl.scan_parquet(data_path)
                 .filter((col('crsp_exchcd') == 1) &         # NYSE exchange code
                         (col('obs_main') == 1)    &         # Main observation flag
                         (col('exch_main') == 1)   &         # Main exchange flag
                         (col('primary_sec') == 1) &         # Primary security flag
                         (col('common') == 1)      &         # Common stock flag
                         (col('me').is_not_null()))          # Ensure market equity (me) is not null
                 .group_by('eom')
                 .agg(n        = col('me').count(),
                      nyse_p1  = perc_exp('me', lambda x: perc_method(x, 0.01)),
                      nyse_p20 = perc_exp('me', lambda x: perc_method(x, 0.20)),
                      nyse_p50 = perc_exp('me', lambda x: perc_method(x, 0.50)),
                      nyse_p80 = perc_exp('me', lambda x: perc_method(x, 0.80))))
    nyse_sf.collect().write_parquet('nyse_cutoffs.parquet')

@measure_time
def classify_stocks_size_groups():
    nyse_cutoffs = pl.scan_parquet('nyse_cutoffs.parquet')
    __msf_world = pl.scan_parquet('__msf_world3.parquet')
    world_msf = (__msf_world.join(nyse_cutoffs, how = 'left', on = 'eom')
                            .with_columns(size_grp = pl.when(col('me').is_null()).then(pl.lit(''))
                                                       .when(col('me') >= col('nyse_p80')).then(pl.lit('mega'))
                                                       .when(col('me') >= col('nyse_p50')).then(pl.lit('large'))
                                                       .when(col('me') >= col('nyse_p20')).then(pl.lit('small'))
                                                       .when(col('me') >= col('nyse_p1')).then(pl.lit('micro'))
                                                       .otherwise(pl.lit('nano')))
                            .drop([i for i in nyse_cutoffs.collect_schema().names() if i not in ['eom']]))#['eom', 'n']])
    world_msf.collect().write_parquet('world_msf.parquet')

@measure_time
def return_cutoffs(freq, crsp_only):
    # Filter data based on provided criteria. If 'crsp_only' is 1, filter for CRSP data only.
    group_vars = ['eom'] if freq == 'm' else ['year', 'month']
    res_path = 'return_cutoffs.parquet' if freq == 'm' else 'return_cutoffs_daily.parquet'
    data = pl.scan_parquet(f'world_{freq}sf.parquet')
    if crsp_only == 1: data = data.filter(col('source_crsp') == 1)
    data = data.filter((col('common')      == 1     ) &
                       (col('obs_main')    == 1     ) &
                       (col('exch_main')   == 1     ) &
                       (col('primary_sec') == 1     ) &
                       (col('excntry')     != 'ZWE' ) &
                       (col('ret_exc').is_not_null())  )
    data = (data.with_columns(year  = col('date').dt.year(),
                              month = col('date').dt.month())
                .group_by(group_vars)
                .agg(n = col('ret').count(),
                     ret_0_1        = perc_exp('ret'      , lambda x: perc_method(x, 0.001)),
                     ret_1          = perc_exp('ret'      , lambda x: perc_method(x, 0.01)),
                     ret_99         = perc_exp('ret'      , lambda x: perc_method(x, 0.99)),
                     ret_99_9       = perc_exp('ret'      , lambda x: perc_method(x, 0.999)),
                     ret_local_0_1  = perc_exp('ret_local', lambda x: perc_method(x, 0.001)),
                     ret_local_1    = perc_exp('ret_local', lambda x: perc_method(x, 0.01)),
                     ret_local_99   = perc_exp('ret_local', lambda x: perc_method(x, 0.99)),
                     ret_local_99_9 = perc_exp('ret_local', lambda x: perc_method(x, 0.999)),
                     ret_exc_0_1    = perc_exp('ret_exc'  , lambda x: perc_method(x, 0.001)),
                     ret_exc_1      = perc_exp('ret_exc'  , lambda x: perc_method(x, 0.01)),
                     ret_exc_99     = perc_exp('ret_exc'  , lambda x: perc_method(x, 0.99)),
                     ret_exc_99_9   = perc_exp('ret_exc'  , lambda x: perc_method(x, 0.999))))
    data.sort(group_vars).collect().write_parquet(res_path)
def winsorize_mkt_ret(var, cutoff, comparison):
    if comparison == '>': c1 = (col(var) > col(cutoff)) & (col('source_crsp') == 0) & (col(var).is_not_null())
    else: c1 = ((col(var) < col(cutoff)).fill_null(bo_false())) & (col('source_crsp') == 0) & (col(var).is_not_null())
    return (pl.when(c1).then(col(cutoff)).otherwise(col(var))).alias(var)
def load_mkt_returns_params(freq):
    dt_col           = 'date' if freq == 'd' else 'eom'
    max_date_lag     = 14 if freq == 'd' else 1
    path_aux         = '_daily' if freq == 'd' else ''
    group_vars       = ['year', 'month'] if freq == 'd' else ['eom']
    comm_stocks_cols = ['source_crsp', 'id', 'date', 'eom', 'excntry', 'obs_main', 'exch_main', 'primary_sec', 'common', 'ret_lag_dif', 'me', 'dolvol', 'ret', 'ret_local', 'ret_exc']
    return dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols
def add_cutoffs_and_winsorize(df, wins_data_path, group_vars, dt_col):
    wins_data = (pl.scan_parquet(wins_data_path)
                   .select(group_vars + ['ret_exc_0_1', 'ret_exc_99_9', 'ret_0_1', 'ret_99_9', 'ret_local_0_1', 'ret_local_99_9']))
    df = (df.with_columns(year   = col(dt_col).dt.year(),
                          month  = col(dt_col).dt.month())
            .join(wins_data, how = 'left', on = group_vars)
            .with_columns([winsorize_mkt_ret(i, f'{i}_99_9', '>') for i in ['ret', 'ret_local', 'ret_exc']])
            .with_columns([winsorize_mkt_ret(i, f'{i}_0_1' , '<') for i in ['ret', 'ret_local', 'ret_exc']]))
    return df
def apply_stock_filter_and_compute_indexes(df, dt_col, max_date_lag):
    c1 = (col('obs_main') == 1) & (col('exch_main') == 1) & (col('primary_sec') == 1) & (col('common') == 1) & (col('ret_lag_dif') <= max_date_lag) & (col('me_lag1').is_not_null()) & (col('ret_local').is_not_null())
    df = (df.filter(c1)
            .with_columns(aux1 = col('ret_local') * col('me_lag1'),
                          aux2 = col('ret')       * col('me_lag1'),
                          aux3 = col('ret_exc')   * col('me_lag1'))
            .group_by(['excntry', dt_col])
            .agg(stocks        = pl.len(),
                 me_lag1       = pl.sum('me_lag1'),
                 dolvol_lag1   = pl.sum('dolvol_lag1'),
                 mkt_vw_lcl    = pl.sum('aux1')/pl.sum('me_lag1'),
                 mkt_ew_lcl    = pl.mean('ret_local'),
                 mkt_vw        = pl.sum('aux2')/pl.sum('me_lag1'),
                 mkt_ew        = pl.mean('ret'),
                 mkt_vw_exc    = pl.sum('aux3')/pl.sum('me_lag1'),
                 mkt_ew_exc    = pl.mean('ret_exc')))
    return df
def drop_non_trading_days(df, n_col, dt_col, over_vars, thresh_fraction):
    df = (df.with_columns(year  = col(dt_col).dt.year(),
                          month = col(dt_col).dt.month())
             .with_columns(max_stocks = pl.max(n_col).over(over_vars))
             .filter((col(n_col)/col('max_stocks')) >= thresh_fraction)
             .drop(['year','month','max_stocks']))
    return df

@measure_time
def market_returns(data_path, freq, wins_comp, wins_data_path):
    dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols = load_mkt_returns_params(freq)
    __common_stocks = (pl.scan_parquet(data_path)
                         .select(comm_stocks_cols)
                         .unique()
                         .sort(['id', dt_col])
                         .with_columns(me_lag1     = col('me').shift(1).over('id'),
                                       dolvol_lag1 = col('dolvol').shift(1).over('id')))
    if wins_comp == 1: __common_stocks = add_cutoffs_and_winsorize(__common_stocks, wins_data_path, group_vars, dt_col)
    __common_stocks = apply_stock_filter_and_compute_indexes(__common_stocks, dt_col, max_date_lag)
    if freq == 'd': __common_stocks = drop_non_trading_days(__common_stocks, 'stocks', dt_col, ['excntry','year','month'], 0.25)
    __common_stocks.sort(['excntry', dt_col]).collect(streaming = True).write_parquet(f'market_returns{path_aux}.parquet')

@measure_time
def quarterize(df, var_list):
    list_aux1 = [col('gvkey').cum_count().over(['gvkey','fyr','fyearq']).alias('count_aux')] +\
                [col(var).cast(pl.Float64).diff().alias(var + '_q') for var in var_list]
    c1 = (col('fqtr') != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col('fqtr').diff() != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    list_aux2 = [pl.when(col('count_aux') == 1).then(col(var)).otherwise(col(var + '_q')).alias(var + '_q') for var in var_list] +\
                [pl.when(col('count_aux') == 1).then(c1).otherwise(c2).alias('del')]
    list_aux3 = [pl.when(col('del')).then(fl_none()).otherwise(col(var + '_q')).alias(var + '_q') for var in var_list]
    df = (df.sort(['gvkey','fyr','fyearq','fqtr'])
            .unique(['gvkey','fyr','fyearq','fqtr'], keep = 'first')
            .sort(['gvkey','fyr','fyearq','fqtr'])
            .with_columns(list_aux1)
            .sort(['gvkey','fyr','fyearq','fqtr'])
            .with_columns(list_aux2)
            .with_columns(list_aux3)
            .drop(['del', 'count_aux'])
            .sort(['gvkey','fyr','fyearq','fqtr'])
            .unique(['gvkey','fyr','fyearq','fqtr'], keep = 'first')
            .sort(['gvkey','fyr','fyearq','fqtr']))
    return df
def ttm(var): return col(var) + col(var).shift(1) + col(var).shift(2) + col(var).shift(3)
def cumulate_4q(var_yrl, mode = 'add'):
    var_yrl_name = var_yrl[:-1]
    c1 = (col('gvkey')  != col('gvkey').shift(3)).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col('fyr')    != col('fyr').shift(3)).fill_null(pl.lit(True).cast(pl.Boolean))
    c3 = (col('curcdq') != col('curcdq').shift(3)).fill_null(pl.lit(True).cast(pl.Boolean))
    c4 = (ttm('fqtr')   != 10).fill_null(pl.lit(True).cast(pl.Boolean))
    c5 = (c1 | c2 | c3 | c4)
    c6 = ttm(var_yrl).is_null()
    c7 = col('fqtr') == 4
    c8 = (c6 & c7)
    if mode == 'add': return pl.when(c5).then(fl_none()).when(c8).then(col(f'{var_yrl_name}y')).otherwise(ttm(var_yrl)).alias(var_yrl_name)
    else: return [var_yrl, f'{var_yrl_name}y']
def load_raw_fund_table_and_filter(filename, start_date, source_str, mode):
    c1 = (col('indfmt').is_in(['INDL', 'FS'])) if mode == 1 else (col('indfmt') == 'INDL')
    datafmt_val = 'HIST_STD' if mode == 1 else 'STD'
    popsrc_val  = 'I'        if mode == 1 else 'D'
    df = (pl.scan_parquet(filename)
            .filter( c1                              &
                    (col('datafmt') == datafmt_val)  &
                    (col('popsrc')  == popsrc_val)   &
                    (col('consol')  == 'C')          &
                    (col('datadate') >= start_date))
            .with_columns(source = pl.lit(source_str)))
    return df
def apply_indfmt_filter(df):
    df = (df.with_columns(count_indfmt = pl.len().over(['gvkey', 'datadate']))
            .filter((col('count_indfmt') == 1) | ((col('count_indfmt') == 2) & (col('indfmt') == 'INDL')))
            .drop(['indfmt', 'count_indfmt']))
    return df
def add_fx_and_convert_vars(df, fx_df, vars, freq):
    fx_var = 'curcd' if freq == 'annual' else 'curcdq'
    aux = (df.join(fx_df, left_on = ['datadate', fx_var], right_on = ['datadate', 'curcdd'], how = 'left')
                .select(df.collect_schema().names() + ['fx'])
                .with_columns([(col(var) * col('fx')).alias(var) for var in vars] + [pl.lit('USD').alias(fx_var)])
                .drop('fx'))
    return aux
def load_mkt_equity_data(filename, alias = True):
    col_name = 'me_fiscal' if alias else 'me_company'
    df = (pl.scan_parquet(filename)
            .filter((col('gvkey').is_not_null())      &
                    (col('primary_sec') == 1)         &
                    (col('me_company').is_not_null()) &
                    (col('common') == 1)              &
                    (col('obs_main') == 1))
            .select(['gvkey', 'eom', col('me_company').alias(col_name)])
            .group_by(['gvkey', 'eom'])
            .agg(col(col_name).max()))
    return df

@measure_time
def standardized_accounting_data(coverage,convert_to_usd, me_data_path,include_helpers_vars, start_date):
    g_fundq_cols = pl.scan_parquet('Raw_tables/comp_g_fundq.parquet').collect_schema().names()
    fundq_cols = pl.scan_parquet('Raw_tables/comp_fundq.parquet').collect_schema().names()
    #Compustat Accounting Vars to Extract
    avars_inc = ['sale', 'revt', 'gp', 'ebitda', 'oibdp', 'ebit', 'oiadp', 'pi', 'ib', 'ni', 'mii','cogs', 'xsga', 'xopr', 'xrd', 'xad', 'xlr', 'dp', 'xi', 'do', 'xido', 'xint', 'spi', 'nopi', 'txt','dvt']
    avars_cf = ['oancf', 'ibc', 'dpc', 'xidoc', 'capx', 'wcapt', # Operating
    'fincf', 'fiao', 'txbcof', 'ltdch', 'dltis', 'dltr', 'dlcch', 'purtshr', 'prstkc', 'sstk','dv', 'dvc'] # Financing
    avars_bs = ['at', 'act', 'aco', 'che', 'invt', 'rect', 'ivao', 'ivst', 'ppent', 'ppegt', 'intan', 'ao', 'gdwl', 're', # Assets
    'lt', 'lct', 'dltt', 'dlc', 'txditc', 'txdb', 'itcb', 'txp', 'ap', 'lco', 'lo', 'seq', 'ceq', 'pstkrv', 'pstkl', 'pstk', 'mib', 'icapt'] # Liabilities
    # Variables in avars_other are not measured in currency units, and only available in annual data
    avars_other = ['emp']
    avars = avars_inc + avars_cf + avars_bs
    print(f"INCOME STATEMENT: {len(avars_inc)} || CASH FLOW STATEMENT: {len(avars_cf)} || BALANCE SHEET: {len(avars_bs)} || OTHER: {len(avars_other)}", flush=True)
    #finding which variables of interest are available in the quarterly data
    combined_columns = g_fundq_cols + fundq_cols
    qvars_q = list({aux_var for aux_var in combined_columns if aux_var[:-1].lower() in avars and aux_var.endswith('q')}) #different from above to get only unique values
    qvars_y = list({aux_var for aux_var in combined_columns if aux_var[:-1].lower() in avars and aux_var.endswith('y')})
    qvars = qvars_q + qvars_y
    if coverage in ['global', 'world']:
        #Annual global data:
        vars_not_in_query = ['gp', 'pstkrv', 'pstkl', 'itcb', 'xad', 'txbcof', 'ni']
        query_vars = [var for var in (avars + avars_other) if var not in vars_not_in_query]
        g_funda = load_raw_fund_table_and_filter('Raw_tables/comp_g_funda.parquet', start_date, 'GLOBAL', 1)
        __gfunda = (g_funda.with_columns(ni = (col('ib') + pl.coalesce('xi', 0) + pl.coalesce('do', 0)).cast(pl.Float64))
                           .select(['gvkey', 'datadate', 'indfmt', 'curcd', 'source', 'ni'] +\
                                   [fl_none().alias(i) for i in ['gp', 'pstkrv', 'pstkl', 'itcb', 'xad', 'txbcof']] +\
                                   query_vars)
                           .pipe(apply_indfmt_filter))
        #Quarterly global data:
        vars_not_in_query = ['icaptq','niy','txditcq','txpq','xidoq','xidoy','xrdq','xrdy','txbcofy', 'niq', 'ppegtq', 'doq', 'doy']
        query_vars = [var for var in qvars if var not in vars_not_in_query]
        g_fundq = load_raw_fund_table_and_filter('Raw_tables/comp_g_fundq.parquet', start_date, 'GLOBAL', 1)
        __gfundq = (g_fundq.with_columns(niq    = (col('ibq') + pl.coalesce('xiq', 0.)).cast(pl.Float64),
                                         ppegtq = (col('ppentq') + col('dpactq')).cast(pl.Float64))
                           .select(['gvkey', 'datadate', 'indfmt', 'fyr', 'fyearq', 'fqtr', 'curcdq', 'source', 'niq', 'ppegtq'] +\
                                   [fl_none().alias(i) for i in ['icaptq', 'niy', 'txditcq', 'txpq', 'xidoq', 'xidoy', 'xrdq', 'xrdy', 'txbcofy']] +\
                                   query_vars)
                           .pipe(apply_indfmt_filter))
    if coverage in ['na', 'world']:
        #Annual north american data:
        vars_not_in_query = ['wcapt', 'ltdch', 'purtshr']
        query_vars = [var for var in (avars + avars_other) if var not in vars_not_in_query]
        funda = load_raw_fund_table_and_filter('Raw_tables/comp_funda.parquet', start_date, 'NA', 2)
        __funda = funda.select(['gvkey', 'datadate', 'curcd', 'source'] +\
                               [fl_none().alias(i) for i in ['wcapt', 'ltdch', 'purtshr']] +\
                               query_vars)
        #Quarterly north american data:
        vars_not_in_query = ['dvtq','gpq','dvty','gpy','ltdchy','purtshry','wcapty']
        query_vars = [var for var in qvars if var not in vars_not_in_query]
        fundq = load_raw_fund_table_and_filter('Raw_tables/comp_fundq.parquet', start_date, 'NA', 2)
        __fundq = fundq.select(['gvkey', 'datadate', 'fyr', 'fyearq', 'fqtr', 'curcdq', 'source'] +\
                               [fl_none().alias(i) for i in ['dvtq','gpq','dvty','gpy','ltdchy','purtshry','wcapty']] +\
                               query_vars)
    if coverage == 'world': __wfunda, __wfundq = pl.concat([__gfunda, __funda], how = 'diagonal_relaxed'), pl.concat([__gfundq, __fundq], how = 'diagonal_relaxed')
    else: pass
    if coverage == 'na': aname, qname= __funda, __fundq
    elif coverage == 'global': aname, qname = __gfunda, __gfundq
    else: aname, qname = __wfunda, __wfundq
    #converting to usd if required
    if convert_to_usd == 1:
        fx = compustat_fx().lazy()
        __compa = add_fx_and_convert_vars(aname, fx, avars, 'annual')
        __compq = add_fx_and_convert_vars(qname, fx, qvars, 'quarterly')
    else: __compa, __compq = aname, qname
    __me_data = load_mkt_equity_data(me_data_path)

    yrl_vars = ['cogsq', 'xsgaq', 'xintq', 'dpq', 'txtq', 'xrdq', 'dvq', 'spiq', 'saleq', 'revtq', 'xoprq', 'oibdpq', 'oiadpq', 'ibq', 'niq', 'xidoq', 'nopiq', 'miiq', 'piq', 'xiq','xidocq', 'capxq', 'oancfq', 'ibcq', 'dpcq', 'wcaptq','prstkcq', 'sstkq', 'purtshrq','dsq', 'dltrq', 'ltdchq', 'dlcchq','fincfq', 'fiaoq', 'txbcofq', 'dvtq']
    bs_vars = ['seqq', 'ceqq', 'pstkq', 'icaptq', 'mibq', 'gdwlq', 'req','atq', 'actq', 'invtq', 'rectq', 'ppegtq', 'ppentq', 'aoq', 'acoq', 'intanq', 'cheq', 'ivaoq', 'ivstq', 'ltq', 'lctq', 'dlttq', 'dlcq', 'txpq', 'apq', 'lcoq', 'loq', 'txditcq', 'txdbq']
    __compq = (__compq.with_columns([col(var).cast(pl.Int64) for var in ['fyr', 'fyearq', 'fqtr'] if var in __compq.collect_schema().names()])
                      .pipe(quarterize, var_list = qvars_y)
                      .sort(['gvkey', 'fyr', 'fyearq', 'fqtr'])
                      .unique(['gvkey', 'fyr', 'fyearq', 'fqtr'], keep='first')
                      .sort(['gvkey', 'fyr', 'fyearq', 'fqtr']))
    __compq = (__compq.with_columns([pl.coalesce([f'{var[:-1]}q', f'{var[:-1]}y_q']).alias(f'{var[:-1]}q') for var in qvars_y if f'{var[:-1]}q' in __compq.collect_schema().names()] +\
                                    [col(f'{var[:-1]}y_q').alias(f'{var[:-1]}q') for var in qvars_y if f'{var[:-1]}q' not in __compq.collect_schema().names()])
                      .drop([f'{var[:-1]}y_q' for var in qvars_y])
                      .with_columns(ni_qtr   = col('ibq'),
                                    sale_qtr = col('saleq'),
                                    ocf_qtr  = pl.coalesce(['oancfq', (col('ibq') + col('dpq') - pl.coalesce([col('wcaptq'), 0]))]),
                                    dsy      = fl_none(),
                                    dsq      = fl_none())
                      .sort(['gvkey', 'fyr', 'fyearq', 'fqtr'])
                      .with_columns([cumulate_4q(var_yrl) for var_yrl in yrl_vars])
                      .drop([a for b in [cumulate_4q(var_yrl, 'drop_cols') for var_yrl in yrl_vars] for a in b])
                      .rename({**dict(zip(bs_vars, list(aux[:-1] for aux in bs_vars))), **{'curcdq': 'curcd'}})
                      .sort(['gvkey', 'datadate', 'fyr', 'fqtr'])
                      .unique(['gvkey','datadate', 'fyr'], keep='first')
                      .sort(['gvkey', 'datadate', 'fyr','fqtr'])
                      .unique(['gvkey', 'datadate'], keep='last')
                      .drop(['fyr', 'fyearq', 'fqtr'])
                      .join(__me_data, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'eom'])
                      .with_columns([fl_none().alias(i) for i in ['gp', 'dltis', 'do', 'dvc', 'ebit', 'ebitda', 'itcb', 'pstkl', 'pstkrv', 'xad', 'xlr', 'emp']]))

    __compa = (__compa.with_columns(ni_qtr    = fl_none(),
                                     sale_qtr = fl_none(),
                                     ocf_qtr  = fl_none(),
                                     fqtr     = pl.lit(None),
                                     fyearq   = pl.lit(None),
                                     fyr      = pl.lit(None))
                       .join(__me_data, how = 'left', left_on = ['gvkey', 'datadate'], right_on = ['gvkey', 'eom']))

    if include_helpers_vars==1:
        __compq = add_helper_vars(__compq)
        __compa = add_helper_vars(__compa)
    #Do not use streaming here, gives an error. Normal collect
    __compa.unique(['gvkey', 'datadate']).sort(['gvkey', 'datadate']).collect().write_parquet('acc_std_ann.parquet')
    __compq.unique(['gvkey', 'datadate']).sort(['gvkey', 'datadate']).collect().write_parquet('acc_std_qtr.parquet')

@measure_time
def expand(data, id_vars, start_date, end_date, freq='day', new_date_name='date'):
    freq_range = '1d' if (freq == 'day') else '1mo'
    expanded_df = (data.with_columns(pl.date_ranges(start=start_date, end=end_date, interval=freq_range).alias(new_date_name))
                       .explode(new_date_name)
                       .drop([start_date, end_date]))
    if freq == 'month': expanded_df = expanded_df.with_columns(col(new_date_name).dt.month_end())
    expanded_df = (expanded_df.unique(id_vars + [new_date_name])
                              .sort(id_vars + [new_date_name]))
    return expanded_df
def sum_sas(col1, col2):
    c1 = col(col1).is_not_null()
    c2 = col(col2).is_not_null()
    return pl.when(c1 | c2).then(pl.coalesce([col1, 0.]) + pl.coalesce([col2, 0.])).otherwise(fl_none())
def sub_sas(col1, col2):
    c1 = col(col1).is_not_null()
    c2 = col(col2).is_not_null()
    return pl.when(c1 | c2).then(pl.coalesce([col1, 0.]) - pl.coalesce([col2, 0.])).otherwise(fl_none())
@measure_time
def add_helper_vars(data):
    c1 = (col('dltis').is_null()) & (col('dltr').is_null()) & (col('ltdch').is_null()) & (col('count') <= 12)
    c2 = (col('dlcch').is_null()) & (col('count') <= 12)
    sort_vars = ['gvkey', 'curcd', 'datadate']
    over_vars = ['gvkey', 'curcd']
    dates_df = (data.select(sort_vars)
                    .group_by(over_vars)
                    .agg(start_date = col('datadate').min(),
                         end_date   = col('datadate').max()))
    dates_df = expand(dates_df, ['gvkey'], 'start_date', 'end_date', 'month', 'datadate')
    temp_data = data.with_columns(data_available = (col('gvkey').is_not_null()).cast(pl.Float64))
    base = (dates_df.join(temp_data, left_on = sort_vars, right_on = sort_vars, how = 'left')
                    .with_columns(col('data_available').fill_null(strategy = 'zero'))
                    .select(temp_data.collect_schema().names())
                    .sort(sort_vars)
                    .unique(sort_vars, keep = 'first')
                    .sort(sort_vars)
                    .with_columns(count = col('curcd').cum_count().over(over_vars))
                    .with_columns([pl.when(col(var) >= 0).then(col(var)).otherwise(fl_none()).alias(var) for var in ['at', 'sale', 'revt', 'dv', 'che']]))
    helpers = (base.with_columns(sale_x     = pl.coalesce(['sale', 'revt']),
                                 debt_x     = sum_sas('dltt', 'dlc'),
                                 pstk_x     = pl.coalesce(['pstkrv', 'pstkl', 'pstk']),
                                 opex_x     = pl.coalesce(['xopr', col('cogs') + col('xsga')]),
                                 eqis_x     = col('sstk'),
                                 div_x      = pl.coalesce(['dvt', 'dv']),
                                 eqbb_x     = sum_sas('prstkc', 'purtshr'),
                                 xido_x     = pl.coalesce(['xido', (col('xi') + pl.coalesce(['do', 0.]))]),
                                 ca_x       = pl.coalesce(['act', col('rect') + col('invt') + col('che') + col('aco')]),
                                 cl_x       = pl.coalesce([col('lct'), col('ap') + col('dlc') + col('txp') + col('lco')]),
                                 fna_x      = pl.coalesce(['ivst', 0.]) + pl.coalesce(['ivao', 0.]),
                                 ppeinv_x   = col('ppegt') + col('invt'),
                                 lnoa_x     = col('ppent') + col('intan') + col('ao') - col('lo') + col('dp'),
                                 txditc_x   = pl.coalesce(['txditc', sum_sas('txdb', 'itcb')]))
                   .with_columns(gp_x       = pl.coalesce(['gp', col('sale_x') - col('cogs')]),
                                 eqnetis_x  = sub_sas('eqis_x', 'eqbb_x'),
                                 eqpo_x     = col('div_x') + col('eqbb_x'),
                                 seq_x      = pl.coalesce(['seq', col('ceq') + pl.coalesce(['pstk_x', 0]), col('at') - col('lt')]),
                                 ncl_x      = col('lt') - col('cl_x'),
                                 coa_x      = col('ca_x') - col('che'),
                                 col_x      = col('cl_x') - pl.coalesce(['dlc', 0.]),
                                 ncol_x     = col('lt') - col('cl_x') - col('dltt'),
                                 fnl_x      = col('debt_x') + pl.coalesce(['pstk_x', 0.]),
                                 nwc_x      = col('ca_x') - col('cl_x'),
                                 caliq_x    = pl.coalesce([col('ca_x') - col('invt'), col('che') + col('rect')]),
                                 netdebt_x  = col('debt_x') - pl.coalesce([col('che'), pl.lit(0.)]))
                   .with_columns(ebitda_x   = pl.coalesce(['ebitda', 'oibdp', col('sale_x') - col('opex_x'), col('gp_x') - col('xsga')]),
                                 eqnpo_x    = col('div_x') - col('eqnetis_x'),
                                 at_x       = pl.coalesce(['at', col('seq_x') + col('dltt') + pl.coalesce(['lct', 0.]) + pl.coalesce(['lo', 0.]) + pl.coalesce(['txditc', 0.])]),
                                 cowc_x     = col('coa_x') - col('col_x'),
                                 ol_x       = col('col_x') + col('ncol_x'),
                                 nfna_x     = col('fna_x') - col('fnl_x'),
                                 be_x       = col('seq_x') + pl.coalesce([col('txditc_x'),pl.lit(0)]) - pl.coalesce([col('pstk_x'),pl.lit(0)]),
                                 bev_x      = pl.coalesce([col('icapt') + pl.coalesce(['dlc', 0.]) - pl.coalesce(['che', 0.]), col('netdebt_x') + col('seq_x') + pl.coalesce(['mib', 0.])]))
                   .with_columns(ebit_x     = pl.coalesce(['ebit', 'oiadp', col('ebitda_x') - col('dp')]),
                                 op_x       = col('ebitda_x') + pl.coalesce(['xrd', 0.]),
                                 ope_x      = col('ebitda_x') - col('xint'),
                                 nca_x      = col('at_x') - col('ca_x'),
                                 ncoa_x     = col('at_x') - col('ca_x') - pl.coalesce(['ivao', 0]),
                                 aliq_x     = col('che') + 0.75 * col('coa_x') + 0.5 * (col('at_x') - col('ca_x') - pl.coalesce(['intan', 0.])),
                                 be_x       = pl.when(col('be_x') >= 0).then('be_x').otherwise(fl_none()),
                                 bev_x      = pl.when(col('bev_x') >= 0).then('bev_x').otherwise(fl_none()))
                   .with_columns(pi_x       = pl.coalesce(['pi', (col('ebit_x') - col('xint') + pl.coalesce(['spi', 0.]) + pl.coalesce(['nopi', 0.]))]),
                                 oa_x       = col('coa_x') + col('ncoa_x'),
                                 nncoa_x    = col('ncoa_x') - col('ncol_x'))
                   .with_columns(ni_x       = pl.coalesce(['ib', (col('ni') - col('xido_x')), (col('pi_x') - col('txt') - pl.coalesce(['mii', 0.]))]),
                                 noa_x      = col('oa_x') - col('ol_x'))
                   .sort(sort_vars)
                   .with_columns(nix_x      = pl.coalesce(['ni', (col('ni_x') + pl.coalesce(['xido_x', 0.])), col('ni_x') + col('xi') + col('do')]),
                                 oacc_x     = pl.when(col('count') > 12).then(pl.coalesce([col('ni_x') - col('oancf'), col('cowc_x').diff(n=12) + col('nncoa_x').diff(n=12)])).otherwise(fl_none()),
                                 dltnetis_x = pl.when(c1).then(fl_none()).otherwise(pl.coalesce([sub_sas('dltis', 'dltr'), 'ltdch', col('dltt').diff(n=12).over(over_vars)])),
                                 dstnetis_x = pl.when(c2).then(fl_none()).otherwise(pl.coalesce(['dlcch', col('dlc').diff(n=12).over(over_vars)])))
                   .sort(sort_vars)
                   .with_columns(fi_x       = col('nix_x') + col('xint'),
                                 tacc_x     = pl.when(col('count') > 12).then(col('oacc_x') + col('nfna_x').diff(n=12)).otherwise(fl_none()),
                                 ocf_x      = pl.coalesce(['oancf', col('ni_x') - col('oacc_x'), col('ni_x') + col('dp') - pl.coalesce(['wcapt', 0.])]),
                                 cop_x      = col('ebitda_x') + pl.coalesce(['xrd', 0.]) - col('oacc_x'),
                                 dbnetis_x  = sum_sas('dstnetis_x', 'dltnetis_x'))
                   .with_columns(netis_x    = col('eqnetis_x') + col('dbnetis_x'),
                                 fcf_x      = col('ocf_x') - col('capx'))
                   .with_columns(fincf_x    = pl.coalesce(['fincf', (col('netis_x') - col('dv') + pl.coalesce(['fiao', 0.]) + pl.coalesce(['txbcof', 0.]))]))
                   .drop('count')
              )
    return helpers
def var_growth(var_gr, horizon):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    name_gr = f"{name_gr}_gr{int(horizon/12)}"
    name_gr_exp  = (col(var_gr) / col(var_gr).shift(horizon)) - 1
    c1 = (col(var_gr).shift(horizon) > 0) & (col("count") > horizon)
    name_gr_col = pl.when(c1).then(name_gr_exp).otherwise(fl_none()).alias(name_gr)
    return name_gr_col
def chg_to_assets(var_gr, horizon):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    name_gr = f"{name_gr}_gr{int(horizon/12)}a"
    name_gr_exp = ((col(var_gr) - col(var_gr).shift(horizon))/col('at_x'))
    c1 = (col('at_x') > 0) & (col("count") > horizon)
    name_gr_col = pl.when(c1).then(name_gr_exp).otherwise(fl_none()).alias(name_gr)
    return name_gr_col
def chg_to_lagassets(var_gr):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    name_gr = f"{name_gr}_gr1a"
    # Calculating the growth rate
    name_gr_exp = (col(var_gr) - col(var_gr).shift(12))/col('at_x').shift(12)
    c1 = (col('at_x').shift(12) > 0) & (col("count") > 12)
    return pl.when(c1).then(name_gr_exp).otherwise(fl_none()).alias(name_gr)
def chg_to_exp(var):
    new_name = var.replace('_x', '')
    new_name = f"{new_name}_ce"
    c1 = (col(var).shift(12) + col(var).shift(24)) > 0
    c2 = col('count') > 24
    num = col(var)
    den = (col(var).shift(12) + col(var).shift(24)) / 2
    return pl.when(c1 & c2).then(num/den - 1).otherwise(fl_none()).alias(new_name)
def chg_to_avgassets(var):
    new_name = var.replace('_x', '')
    new_name = f"{new_name}_gr1a"
    c1 = (col('at_x') + col('at_x').shift(12)) > 0
    c2 = col('count') > 12
    num = col(var) - col(var).shift(12)
    den = col('at_x') + col('at_x').shift(12)
    return pl.when(c1 & c2).then(num/den).otherwise(fl_none()).alias(new_name)
def standardized_unexpected(df, var, qtrs, qtrs_min):
    name = var.replace('_x', '')
    name = f'{name}_su'
    c1 = col('__chg_n') > qtrs_min
    aux_std = pl.when(col('__chg_std').shift(3) != 0).then(col('__chg_std').shift(3)).otherwise(fl_none())
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns((col(var) - col(var).shift(12)).over(['gvkey','curcd']).alias('__chg'))
            .sort(['gvkey','curcd','datadate'])
            .with_columns(aux = pl.concat_list([col('__chg').shift(i).over(['gvkey','curcd']) for i in range(0, (3*qtrs), 3)]).list.drop_nulls())
            .with_columns(__chg_mean = col('aux').list.mean(),
                          __chg_n    = col("aux").list.len(),
                          __chg_std  = col("aux").list.eval(pl.element().std()))
            .explode('__chg_std')
            .with_columns(__chg_mean = pl.when(c1).then(col('__chg_mean')).otherwise(fl_none()),
                          __chg_std  = pl.when(c1).then(col('__chg_std' )).otherwise(fl_none()))
            .sort(['gvkey','curcd','datadate'])
            .with_columns(((col(var) - col(var).shift(12) - col('__chg_mean').shift(3)) / aux_std).over(['gvkey','curcd']).alias(name))
            .with_columns(pl.when(col('count') > (12 + qtrs*3)).then(col(name)).otherwise(fl_none()).alias(name))
            .drop(['__chg', '__chg_mean', '__chg_std', '__chg_n','aux']))
    return df
def volq(df, name, var, qtrs, qtrs_min):
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns(aux = pl.concat_list([col(var).shift(i).over(['gvkey','curcd']) for i in range(0, (3*qtrs), 3)]).list.drop_nulls())
            .with_columns([col('aux').list.std().alias(name), col('aux').list.len().alias('__n')])
            .with_columns(pl.when((col('count') > ((qtrs-1)*3)) & (col('__n') >= qtrs_min)).then(col(name)).otherwise(fl_none()).alias(name))
            .drop(['__n','aux']))
    return df
def vola(df, name, var, yrs, yrs_min):
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns(aux = pl.concat_list([col(var).shift(i).over(['gvkey','curcd']) for i in range(0, (12*yrs), 12)]).list.drop_nulls())
            .with_columns([col('aux').list.std().alias(name), col('aux').list.len().alias('__n')])
            .with_columns(pl.when((col('count') > ((yrs-1)*12)) & (col('__n') >= yrs_min)).then(col(name)).otherwise(fl_none()).alias(name))
            .drop(['__n','aux']))
    return df

@measure_time
def earnings_variability(df, esm_h = 5):
    c1 = (col('count') > (12*esm_h)) & (col('__croa_std') > 0) & (col('__roa_n') >= esm_h) & (col('__croa_n') >= esm_h)
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns([safe_div('ni_x', 'at_x', '__roa', 6), safe_div('ocf_x', 'at_x', '__croa',6)])
            .sort(['gvkey','curcd','datadate'])
            .with_columns(aux1 = pl.concat_list([col('__roa').shift(i).over(['gvkey','curcd']) for i in range(0, (12*esm_h), 12)]).list.drop_nulls(),
                          aux2 = pl.concat_list([col('__croa').shift(i).over(['gvkey','curcd']) for i in range(0, (12*esm_h), 12)]).list.drop_nulls())
            .with_columns(__roa_std  = col('aux1').list.eval(pl.element().std()),
                          __roa_n    = col('aux1').list.len(),
                          __croa_std = col('aux2').list.eval(pl.element().std()),
                          __croa_n   = col('aux2').list.len())
            .explode(['__roa_std','__croa_std'])
            .with_columns(safe_div('__roa_std', '__croa_std','earnings_variability'))
            .with_columns(earnings_variability = pl.when(c1).then(col('earnings_variability')).otherwise(fl_none()))
            .drop(['__roa', '__croa', '__roa_n', '__croa_n', '__roa_std', '__croa_std','aux1','aux2']))
    return df
def roe_and_g_exps(i, g_c, g_ar1, roe_c, roe_ar1): 
    return [(g_c + g_ar1 * col(f'__g{i-1}')).alias(f'__g{i}'), (roe_c + roe_ar1 * col(f'__roe{i-1}')).alias(f'__roe{i}')]
def be_and_cd_exps(i): 
    return [(col(f'__be{i-1}') * (1 + col(f'__g{i}'))).alias(f'__be{i}'), (col(f'__be{i-1}') * (col(f'__roe{i}') - col(f'__g{i}'))).alias(f'__cd{i}')]

@measure_time
def equity_duration_cd(df, horizon=10, r=0.12, roe_mean=0.12, roe_ar1=0.57, g_mean=0.06, g_ar1=0.24):
    c1 = (col('count') > 12) & (col('be_x').shift(12) > 1)
    c2 = (col('count') > 12) & (col('sale_x').shift(12) > 1)
    roe_c = roe_mean * (1 - roe_ar1)
    g_c = g_mean * (1 - g_ar1)
    roe0_exp = pl.when(c1).then(col('ni_x')/col('be_x').shift(12)).otherwise(fl_none())
    g0_exp = pl.when(c2).then(col('sale_x')/col('sale_x').shift(12) - 1).otherwise(fl_none())
    be0_exp = col('be_x')
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns(__roe0 = roe0_exp,
                          __g0   = g0_exp,
                          __be0  = be0_exp))
    for t in range(1, horizon+1): df = df.with_columns(roe_and_g_exps(t, g_c, g_ar1, roe_c, roe_ar1))
    for t in range(1, horizon+1): df = df.with_columns(be_and_cd_exps(t))

    ed_cd_w_exp = sum(t * col(f'__cd{t}') / ((1 + r)**t) for t in range(1, horizon + 1))
    ed_cd_exp = sum(col(f'__cd{t}') / ((1 + r)**t) for t in range(1, horizon + 1))
    c_aux = reduce(lambda a, b: a | b, (col(f'__be{t}') < 0. for t in range(1, horizon + 1)))
    ed_err_exp = pl.when(c_aux).then(pl.lit(1.)).otherwise(pl.lit(0.))
    df = df.with_columns(ed_constant = (pl.lit(horizon) + ((1 + r) / r)),
                         ed_cd_w = ed_cd_w_exp,
                         ed_cd   = ed_cd_exp,
                         ed_err  = ed_err_exp)

    cols_to_drop = [y for x in [[f'__roe{i}', f'__g{i}', f'__be{i}', f'__cd{i}'] for i in range(0, horizon+1)] for y in x]
    cols_to_drop.remove('__cd0')
    df = df.drop(cols_to_drop)
    return df

@measure_time
def pitroski_f(df, name = 'f_score'):
    c1      = (col('count') > 12)
    c2      = col('at_x').shift(12) > 0
    c3      = col('at_x') > 0
    c4      = col('cl_x') > 0
    c5      = col('cl_x').shift(12) > 0
    c6      = col('sale_x') > 0
    c7      = col('sale_x').shift(12) > 0
    c8      = col('count') > 24
    c9      = col('at_x').shift(24) > 0
    col_exp = (pl.coalesce([col('__f_eqis'), 0]) == 0).cast(pl.Int32) + (col('__f_lev') < 0).cast(pl.Int32)
    for var_name in ['__f_roa', '__f_croa', '__f_droa', '__f_acc', '__f_liq', '__f_gm', '__f_aturn']: col_exp += (col(var_name) > 0).cast(pl.Int32)
    df = (df.sort(['gvkey','curcd'])
            .with_columns(__f_roa   = pl.when(c1 & c2).then(col('ni_x') / col('at_x').shift(12)).otherwise(fl_none()),
                          __f_croa  = pl.when(c1 & c2).then(col('ocf_x') / col('at_x').shift(12)).otherwise(fl_none()))
            .sort(['gvkey','curcd'])
            .with_columns(__f_droa  = pl.when(c1).then(col('__f_roa') - col('__f_roa').shift(12)).otherwise(fl_none()),
                          __f_acc   = col('__f_croa') - col('__f_roa'),
                          __f_lev   = pl.when(c1 & c2 & c3).then(col('dltt') / col('at_x') - (col('dltt') / col('at_x')).shift(12)).otherwise(fl_none()),
                          __f_liq   = pl.when(c1 & c4 & c5).then(col('ca_x') / col('cl_x') - (col('ca_x') / col('cl_x')).shift(12)).otherwise(fl_none()),
                          __f_eqis  = col('eqis_x'),
                          __f_gm    = pl.when(c1 & c6 & c7).then(col('gp_x') / col('sale_x') - (col('gp_x') / col('sale_x')).shift(12)).otherwise(fl_none()),
                          __f_aturn = pl.when(c2 & c8 & c9).then((col('sale_x') / col('at_x').shift(12)) - (col('sale_x').shift(12) / col('at_x').shift(24))).otherwise(fl_none()))
            .with_columns(col_exp.alias(name))
            .drop(['__f_roa', '__f_croa', '__f_droa', '__f_acc', '__f_lev', '__f_liq', '__f_eqis', '__f_gm', '__f_aturn']))
    return df

@measure_time
def ohlson_o(df, name = 'o_score'):
    c1 = (col('count') > 12) & (col('nix_x').is_not_null()) & (col('nix_x').shift(12).is_not_null())
    c2 = (col('count') > 12) & (col('nix_x').abs() + col('nix_x').shift(12).abs() != 0)
    exp_aux1 = ((col('nix_x') < 0) & (col('nix_x').shift(12) < 0)).cast(pl.Float64)
    exp_aux2 = (col('nix_x') - col('nix_x').shift(12)) / (col('nix_x').abs() + col('nix_x').shift(12).abs())
    col1 = (pl.when(c1).then(exp_aux1).otherwise(fl_none())).alias('__o_neg_earn')
    col2 = (pl.when(c2).then(exp_aux2).otherwise(fl_none())).alias('__o_nich')
    col3 = safe_div('debt_x', 'at_x', '__o_lev' , 3)
    col4 = safe_div('nix_x' , 'at_x', '__o_roe' , 3)
    col5 = safe_div('cl_x'  , 'ca_x', '__o_cacl', 3)
    col6 = (pl.when(col('at_x') > 0).then(col('at_x').log()                        ).otherwise(fl_none())).alias('__o_lat')
    col7 = (pl.when(col('at_x') > 0).then((col('ca_x') - col('cl_x')) / col('at_x')).otherwise(fl_none())).alias('__o_wc')
    col8 = (pl.when(col('lt')   > 0).then((col('pi_x') + col('dp'))  / col('lt')   ).otherwise(fl_none())).alias('__o_ffo')
    col9 = (pl.when((col('lt').is_not_null()) & (col('at_x').is_not_null())).then((col('lt') > col('at_x')).cast(pl.Int32)).otherwise(fl_none())).alias('__o_neg_eq')
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns([col1, col2, col3, col4, col5, col6, col7, col8, col9])
            .with_columns((-1.32 - 0.407 * col('__o_lat') + 6.03 * col('__o_lev') + 1.43 * col('__o_wc') + 0.076 * col('__o_cacl') - 1.72 * col('__o_neg_eq') - 2.37 * col('__o_roe') - 1.83 * col('__o_ffo') + 0.285 * col('__o_neg_earn') - 0.52 * col('__o_nich')).alias(name)))
    return df

@measure_time
def altman_z(df, name = 'z_score'):
    df = (df.with_columns([pl.when(col('at_x') > 0).then((col('ca_x') - col('cl_x'))/col('at_x')).otherwise(fl_none()).alias('__z_wc'),
                           safe_div('re'       ,'at_x' , '__z_re', 3),
                           safe_div('ebitda_x' , 'at_x', '__z_eb', 3),
                           safe_div('me_fiscal', 'lt'  , '__z_me', 3),
                           safe_div('sale_x'   , 'at_x', '__z_sa', 3)])
           .with_columns((1.2 * col('__z_wc') + 1.4 * col('__z_re') + 3.3 * col('__z_eb') + 0.6 * col('__z_me') + 1.0 * col('__z_sa')).alias(name))
           .drop(['__z_wc', '__z_re', '__z_eb', '__z_sa', '__z_me']))
    return df

@measure_time
def intrinsic_value(df, name = 'intrinsic_value', r = 0.12):
    c1 = col('count') > 12
    c2 = (col('be_x') + col('be_x').shift(12) > 0)
    iv_roe_exp = (pl.when(c1 & c2).then(col('nix_x') / ((col('be_x') + col('be_x').shift(12)) / 2))
                    .otherwise(fl_none())).alias('__iv_roe')
    iv_po_exp  = (pl.when(col('nix_x') > 0  ).then(col('div_x') / col('nix_x'))
                    .when(col('at_x') != 0).then(col('div_x')/ (col('at_x') * 0.06))
                    .otherwise(fl_none())).alias('__iv_po')
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns([iv_roe_exp, iv_po_exp])
            .with_columns(__iv_be1 = ((1 + (1 - col('__iv_po')) * col('__iv_roe')) * col('be_x')))
            .with_columns(( col('be_x') + (((col('__iv_roe') - r)/(1+ r)) * col('be_x')) + (((col('__iv_roe') - r)/((1+ r) * r)) * col('__iv_be1'))).alias(name))
            .with_columns(pl.when(col(name) > 0).then(col(name)).otherwise(fl_none()).alias(name))
            .drop(['__iv_po', '__iv_roe', '__iv_be1']))
    return df

@measure_time
def kz_index(df, name ='kz_index'):
    c1   = (col('count') > 12) & (col('ppent').shift(12) > 0)
    c2   = (col('at_x') > 0)
    c3   = (col('debt_x') + col('seq_x')) != 0
    col1 = (pl.when(c1).then((col('ni_x') + col('dp')) / col('ppent').shift(12)).otherwise(fl_none())).alias('__kz_cf')
    col2 = (pl.when(c1).then(col('div_x') / col('ppent').shift(12)).otherwise(fl_none())).alias('__kz_dv')
    col3 = (pl.when(c1).then(col('che') / col('ppent').shift(12)).otherwise(fl_none())).alias('__kz_cs')
    col4 = (pl.when(c2).then((col('at_x') + col('me_fiscal') - col('be_x')) / col('at_x')).otherwise(fl_none())).alias('__kz_q')
    col5 = (pl.when(c3).then(col('debt_x') / (col('debt_x') + col('seq_x'))).otherwise(fl_none())).alias('__kz_db')
    df   = (df.sort(['gvkey','curcd','datadate'])
              .with_columns([col1,col2,col3,col4,col5])
              .with_columns((- 1.002 * col('__kz_cf') + 0.283 * col('__kz_q') + 3.139 * col('__kz_db') - 39.368 * col('__kz_dv') - 1.315 * col('__kz_cs')).alias(name)))
    return df
def chg_var1_to_var2(df, name, var1, var2, horizon):
    df = (df.with_columns(safe_div(var1, var2, '__x', 3))
            .sort(['gvkey','curcd','datadate'])
            .with_columns(pl.when(col('count') > horizon).then(col('__x') - col('__x').shift(horizon)).otherwise(fl_none()).alias(name))
            .drop('__x'))
    return df

@measure_time
def earnings_persistence(data, __n, __min):
    x, y = '__ni_at_l1', '__ni_at'
    beta = pl.rolling_cov(x,y,window_size = __n, min_periods = __min)/col(x).rolling_var(window_size = __n, min_periods = __min)
    alpha_exp = col(y).rolling_mean(window_size = __n, min_periods = __min) - beta * col(x).rolling_mean(window_size = __n, min_periods = __min)
    exp1 = col(y).rolling_var(window_size = __n, min_periods = __min)
    exp2 = col(x).rolling_var(window_size = __n, min_periods = __min)
    df = (data.select(['gvkey', 'curcd', 'datadate', 'ni_x', 'at_x'])
              .clone()
              .sort(['gvkey', 'curcd', 'datadate'])
              .with_columns(count      = pl.cum_count('gvkey').over(['gvkey','curcd']),
                            __ni_at    = safe_div('ni_x', 'at_x', '__ni_at'),
                            __ni_at_l1 = safe_div('ni_x', 'at_x', '__ni_at_l1', 7))
              .sort(['gvkey','curcd','datadate'])
              .with_columns(ni_ar1     = beta.over(['gvkey','curcd']),
                            alpha      = alpha_exp.over(['gvkey','curcd']))
              .sort(['gvkey','curcd','datadate'])
              .with_columns(ni_ivol    = ((exp1 - (col('ni_ar1')**2) * exp2) ** (1/2)).over(['gvkey','curcd']))
              .filter(col('ni_ar1').is_not_null())
              .select(['gvkey','curcd','datadate','ni_ar1','ni_ivol'])
              .sort(['gvkey','datadate','curcd']))
    return df
def scale_me(var):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    # Appending '_me' to the name
    name = f'{name}_me'
    col_aux = (col(var) * col('fx'))/col('me_company')
    return pl.when(col('me_company') != 0).then(col_aux).otherwise(fl_none()).alias(name)
def scale_mev(var):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    # Appending '_me' to the name
    name = f'{name}_mev'
    col_aux = (col(var) * col('fx'))/col('mev')
    return pl.when(col('mev') != 0).then(col_aux).otherwise(fl_none()).alias(name)
def mean_year(var):
    return (pl.when(col(var).is_not_null() & (col(var).shift(12).over(['gvkey','curcd'])).is_not_null()).then((col(var) + col(var).shift(12)).over(['gvkey','curcd'])/2)
            .when(col(var).is_not_null()).then(col(var))
            .when((col(var).shift(12).over(['gvkey','curcd'])).is_not_null()).then(col(var).shift(12).over(['gvkey','curcd']))
            .otherwise(fl_none()))

@measure_time
def temp_liq_rat(col_avg, den, alias):
    col1 = (365 * mean_year(col_avg)/col(den))
    c1 = col('count') > 12
    c2 = col(den) != 0
    return pl.when(c1 & c2).then(col1).otherwise(fl_none()).alias(alias)

def temp_rat_other(num, den, alias):
    col_expr = (col(num) / mean_year(den))
    c1 = col('count') > 12
    c2 = mean_year(den) != 0
    return pl.when(c1 & c2).then(col_expr).otherwise(fl_none()).alias(alias)

def temp_rat_other_spc():
    num_expr = col('cogs') + col('invt') - col('invt').shift(12)
    col_expr = (num_expr.over(['gvkey','curcd']) / mean_year('ap'))
    c1 = col('count') > 12
    c2 = mean_year('ap') != 0
    return pl.when(c1 & c2).then(col_expr).otherwise(fl_none()).alias('ap_turnover')
def safe_div(num, den, name, mode = 1):
    if mode == 1: return pl.when(col(den) != 0).then(col(num)/col(den)).otherwise(fl_none()).alias(name)
    if mode == 2: return pl.when(col(den) != 0).then(col(num)/(col(den).abs())).otherwise(fl_none()).alias(name)
    if mode == 3: return pl.when(col(den) > 0).then(col(num)/col(den)).otherwise(fl_none()).alias(name)
    if mode == 4:
        cond1 = col('count') > 12
        cond2 = (col(den).shift(12) > 0).over(['gvkey', 'curcd'])
        col_exp = (col(num) / col(den).shift(12)).over(['gvkey', 'curcd'])
        return pl.when(cond1 & cond2).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 5: return pl.when(col(den) != 0).then(col(num) * col('fx')/col(den)).otherwise(fl_none()).alias(name)
    if mode == 6:
        cond1 = (col(den).shift(12) != 0).over(['gvkey','curcd'])
        col_exp = (col(num)/col(den).shift(12)).over(['gvkey','curcd'])
        return pl.when(cond1).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 7:
        cond1 = (col(den).shift(12) > 0).over(['gvkey', 'curcd'])
        col_exp = (col(num) / col(den)).shift(12).over(['gvkey', 'curcd'])
        return pl.when(cond1).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 8:
        cond1 = col(num) > 0
        cond2 = col(den) > 0
        return pl.when(cond1 & cond2).then(col(num)/col(den)).otherwise(fl_none()).alias(name)
    if mode == 9:
        cond1 = col('count') > 3
        cond2 = col(den).shift(3) > 0
        col_exp = col(num) / col(den).shift(3)
        return pl.when(cond1 & cond2).then(col_exp).otherwise(fl_none()).alias(name)
def update_ni_inc_and_decrease(df, lag):
    c1 = (col('ni_inc').shift(lag) == 1) & (col('no_decrease') == 1)
    ni_inc8q_updated_exp = pl.when(c1).then(col('ni_inc8q') + 1).otherwise(col('ni_inc8q')).alias('ni_inc8q')
    no_decrease_updated_exp = pl.when(c1).then(col('no_decrease')).otherwise(pl.lit(0)).alias('no_decrease')
    return df.sort(['gvkey','curcd','datadate']).with_columns([ni_inc8q_updated_exp, no_decrease_updated_exp])

@measure_time
def calculate_consecutive_earnings_increases(df):
    ni_inc_exp = (pl.when(col('ni_x') > col('ni_x').shift(12)).then(pl.lit(1).cast(pl.Int64))
                    .when((col('ni_x').is_null()) | (col('ni_x').shift(12).is_null())).then(pl.lit(None).cast(pl.Int64))
                    .otherwise(pl.lit(0).cast(pl.Int64)).alias('ni_inc'))
    ni_inc8q_exp = pl.lit(0).alias('ni_inc8q')
    no_decrease_exp = pl.lit(1).alias('no_decrease')
    c1 = (col('ni_inc').is_not_null()) & (col('n_ni_inc') == 8) & (col('count') >= 33)
    ni_inc8q_exp_final = pl.when(c1).then(col('ni_inc8q')).otherwise(pl.lit(None))
    n_ni_inc_exp = col('ni_inc').is_not_null()
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns([ni_inc_exp, ni_inc8q_exp, no_decrease_exp]))
    for i in range(8):
        df = update_ni_inc_and_decrease(df, 3*i)
        if i > 0: n_ni_inc_exp += col('ni_inc').shift(3*i).is_not_null()
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns(n_ni_inc = n_ni_inc_exp)
            .with_columns(ni_inc8q = ni_inc8q_exp_final)
            .drop(['ni_inc','no_decrease', 'n_ni_inc']))
    return df
def compute_capex_abn(df):
    c1  = (col('__capex_sale').shift(12) + col('__capex_sale').shift(24) + col('__capex_sale').shift(36)) != 0
    c2  = col('count') > 36
    num = col('__capex_sale')
    den = (col('__capex_sale').shift(12) + col('__capex_sale').shift(24) + col('__capex_sale').shift(36)) / 3
    capex_abn_exp = pl.when(c1 & c2).then(num/den - 1).otherwise(fl_none()).alias('capex_abn')
    df = (df.with_columns(safe_div('capx', 'sale_x', '__capex_sale', 3))
            .sort(['gvkey', 'curcd', 'datadate'])
            .with_columns(capex_abn_exp)
            .drop('__capex_sale'))
    return df
def tangibility():
    c1 = pl.col('at_x') != 0
    div_exp = (col('che') + 0.715 * col('rect') + 0.547 * col('invt') + 0.535 * col('ppegt'))/ col('at_x')
    return pl.when(c1).then(div_exp).otherwise(fl_none()).alias('tangibility')
def emp_gr(path):
    if path == 'acc_std_qtr.parquet':
        col_expr = fl_none().alias('emp_gr1')
    else:
        c1 = (col('count') > 12)
        c2 = (col('emp') - col('emp').shift(12)) / (0.5 * col('emp') + 0.5 * col('emp').shift(12)) != 0
        c3 = (0.5 * col('emp') + 0.5 * col('emp').shift(12)) != 0
        col_expr = (pl.when(c1 & c2 & c3)\
                        .then((col('emp') - col('emp').shift(12)) / (0.5 * col('emp') + 0.5 * col('emp').shift(12)))\
                        .otherwise(fl_none())\
                        .alias('emp_gr1'))
    return col_expr

def add_accounting_misc_cols_1(df):
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
                 pl.when(col('at_x') != 0).then((col('spi') + col('xido_x'))/col('at_x')).otherwise(fl_none()).alias('nri_at'),
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
    #Accrual ratios
    c_accruals = [safe_div('oacc_x', 'at_x', 'oaccruals_at'), safe_div('tacc_x', 'at_x', 'taccruals_at'),
                  safe_div('oacc_x', 'nix_x', 'oaccruals_ni', 2), safe_div('tacc_x', 'nix_x', 'taccruals_ni', 2)]
    c_noa_at = [safe_div('noa_x', 'at_x', 'noa_at', 4)]
    acc_columns = grt1 + grt3 + chg_at1 + chg_at3 + c_at_sale + c_ret_at + c_ret_be + c_ret_bev + c_ret_ppent +\
                  c_iss_eqp + c_solv_rat + c_cap_lev + c_accruals + c_noa_at
    return df.sort(['gvkey', 'curcd', 'datadate']).with_columns(acc_columns)
def add_accounting_misc_cols_2(df):
    #Volatility items
    funcs_vol = [volq, volq, volq, vola]
    names_col = ['ocfq_saleq_std', 'niq_saleq_std', 'roeq_be_std', 'roe_be_std']
    vars_vol  = ['__ocfq_saleq', '__niq_saleq', '__roeq', '__roe']
    t1_col    = [16, 16, 20, 5]
    t2_col    = [8, 8, 12, 5]
    for df_function, n_col, var_vol, t1, t2 in zip(funcs_vol, names_col, vars_vol, t1_col, t2_col): df = df_function(df, n_col, var_vol, t1, t2)
    for df_function in [earnings_variability, equity_duration_cd, pitroski_f, ohlson_o, altman_z, intrinsic_value, kz_index]: df = df.pipe(df_function)
    #5 year ratio change (For quality minus junk variables)
    names = ['gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5']
    vars1 = ['gp_x', 'ni_x', 'ni_x', 'ocf_x', 'gp_x']
    vars2 = ['at_x', 'be_x', 'at_x', 'at_x', 'sale_x']
    for i, j, k in zip(names, vars1, vars2): df = df.pipe(chg_var1_to_var2, name = i, var1 = j, var2 = k, horizon = 60)
    return df.drop(['count', '__ocfq_saleq', '__niq_saleq', '__roeq', '__roe'])
def add_liq_and_efficiency_ratios(df):
    #Liquidity Ratios:
    #Days Inventory Outstanding, Days Sales Outstanding, Days Accounts Payable Outstanding
    c_days = [temp_liq_rat('invt','cogs','inv_days'), temp_liq_rat('rect','sale_x','rec_days'), temp_liq_rat('ap','cogs','ap_days')]
    #Cash, quick, and current ratios; cash Conversion Cycle
    c_liq_rat = [safe_div('che', 'cl_x', 'cash_cl', 3), safe_div('caliq_x', 'cl_x', 'caliq_cl', 3), safe_div('ca_x', 'cl_x', 'ca_cl', 3),\
                pl.when((col('inv_days') + col('rec_days') - col('ap_days')) > 0).then(col('inv_days') + col('rec_days') - col('ap_days')).otherwise(fl_none()).alias('cash_conversion')]
    df = (df.sort(['gvkey','curcd','datadate'])
            .with_columns(c_days)
            .with_columns(c_liq_rat)
            .sort(['gvkey','curcd','datadate'])
            #Activity/Efficiency Ratios:
            .with_columns([temp_rat_other('cogs', 'invt', 'inv_turnover'),
                            temp_rat_other('sale_x', 'at_x', 'at_turnover'),
                            temp_rat_other('sale_x', 'rect', 'rec_turnover'),
                            temp_rat_other_spc()]))
    return df
def add_profit_scaled_by_lagged_vars(df):
    df = (df.sort(['gvkey', 'curcd', 'datadate'])
            .with_columns([safe_div('op_x', 'at_x', 'op_atl1', 4),
                            safe_div('gp_x', 'at_x', 'gp_atl1', 4),
                            safe_div('ope_x', 'be_x', 'ope_bel1', 4),
                            safe_div('cop_x', 'at_x', 'cop_atl1', 4)]))
    return df
def add_earnings_persistence_and_expand(df, lag_to_pub, max_lag):
    earnings_pers = earnings_persistence(df, 5, 5)
    df = (df.join(earnings_pers, on = ['gvkey', 'curcd', 'datadate'], how = 'left')
            .filter(col('data_available')==1)
            .sort(['gvkey','curcd', 'datadate'])
            .unique(['gvkey','curcd','datadate'], keep = 'first')
            .sort(['gvkey','curcd', 'datadate'])
            .with_columns(start_date = col('datadate').dt.offset_by(f'{lag_to_pub}mo').dt.month_end())
            .sort(['gvkey', 'datadate'])
            .with_columns(next_start_date = col('start_date').shift(-1).over(['gvkey']))
            .with_columns(end_date = pl.min_horizontal((col('next_start_date').dt.offset_by('-1mo').dt.month_end()),(col('datadate').dt.offset_by(f'{max_lag}mo').dt.month_end())))
            .drop('next_start_date'))
    return expand(data=df, id_vars=['gvkey'], start_date='start_date', end_date='end_date', freq='month', new_date_name='public_date')
def add_me_data_and_compute_me_mev_mat_eqdur_vars(df, me_df):
    #Characteristics Scaled by Market Equity
    me_vars = ['at_x', 'be_x', 'debt_x', 'netdebt_x', 'che', 'sale_x', 'gp_x', 'ebitda_x','ebit_x', 'ope_x', 'ni_x', 'nix_x', 'cop_x', 'ocf_x', 'fcf_x', 'div_x','eqbb_x', 'eqis_x', 'eqpo_x', 'eqnpo_x', 'eqnetis_x', 'xrd']
    #Characteristics Scaled by Market Enterprise Value
    mev_vars = ['at_x', 'bev_x', 'ppent', 'be_x', 'che', 'sale_x', 'gp_x', 'ebitda_x','ebit_x', 'ope_x', 'ni_x', 'nix_x', 'cop_x', 'ocf_x', 'fcf_x', 'debt_x','pstk_x', 'dltt', 'dlc', 'dltnetis_x', 'dstnetis_x', 'dbnetis_x', 'netis_x', 'fincf_x']
    c_misc = [col('mev').alias('enterprise_value'),
              (pl.when((col('gvkey') == col('gvkey').shift(12)) & (col('mat').shift(12) != 0)).then(col('aliq_x') * col('fx') / col('mat').shift(12)).otherwise(fl_none())).alias('aliq_mat'),
              ((col('ed_cd_w') * col('fx')) / (col('me_company')) + col('ed_constant') * (col('me_company') - col('ed_cd') * col('fx'))/col('me_company')).alias('eq_dur')]
    df = (df.join(me_df, left_on=['gvkey', 'public_date'], right_on=['gvkey', 'eom'], how='left')
            .select(df.collect_schema().names() + ['me_company'])
            .sort(['gvkey', 'public_date'])
            .unique(['gvkey', 'public_date'], keep = 'first')
            .with_columns(mev = col('me_company') + col('netdebt_x') * col('fx'),
                          mat = col('at_x') * col('fx') - col('be_x') * col('fx') + col('me_company'),
                          me_company = pl.when(col('me_company') > 0).then(col('me_company')).otherwise(fl_none()))
            .with_columns(mev = pl.when(col('mev') > 0).then(col('mev')).otherwise(fl_none()),
                          mat = pl.when(col('mat') > 0).then(col('mat')).otherwise(fl_none()))
            .sort(['gvkey','public_date'])
            .with_columns([scale_me(i) for i in me_vars] +\
                          [safe_div('intrinsic_value', 'me_company', 'ival_me', 5)] +\
                          [scale_mev(i) for i in mev_vars] +\
                          c_misc)
            .with_columns(pl.when((col('ed_err') ==1) | (col('eq_dur') <=0) | (col('me_company') == 0)).then(None).otherwise(col('eq_dur')).alias('eq_dur')))
    return df
def rename_cols_and_select_keep_vars(df, rename_dict, vars_to_keep, suffix):
    new_names = {}
    for i in sorted(df.collect_schema().names()):
        col_name = i
        for a, b in rename_dict.items():
            col_name = col_name.replace(a, b, 1)
        new_names[i] = col_name
    df = (df.rename(new_names)
            .select(['source', 'gvkey', 'public_date', 'datadate'] + vars_to_keep))
    if suffix is None: return df
    else: return df.rename({i: (i+suffix) for i in vars_to_keep})
def convert_raw_vars_to_usd(df):
    fx = compustat_fx().rename({'datadate': 'date'}).lazy()
    cols_for_new_df = df.collect_schema().names()
    df = (df.join(fx, left_on = ['curcd', 'public_date'], right_on = ['curcdd', 'date'], how='left')
            .select(cols_for_new_df + ['fx'])
            .with_columns([(col(i) * col('fx')).alias(i) for i in ['assets', 'sales', 'book_equity', 'net_income']])
            .drop('curcd'))
    return df
def financial_soundness_and_misc_ratios_exps():
    #Financial Soundness Ratios:
    c_fin_s_rat = [safe_div('xint', 'debt_x', 'int_debt') , safe_div('ocf_x', 'debt_x', 'ocf_debt')  , safe_div('ebitda_x', 'debt_x', 'ebitda_debt'), safe_div('dlc', 'debt_x', 'debtst_debt'), safe_div('dltt', 'debt_x', 'debtlt_debt'),\
                    safe_div('xint', 'dltt', 'int_debtlt'), safe_div('ebitda_x', 'cl_x', 'profit_cl'), safe_div('ocf_x', 'cl_x', 'ocf_cl')          , safe_div('che', 'lt', 'cash_lt')        , safe_div('cl_x', 'lt', 'cl_lt'),\
                    safe_div('invt', 'act', 'inv_act')    , safe_div('rect', 'act', 'rec_act')       , safe_div('opex_x', 'at_x', 'opex_at')        , safe_div('nwc_x', 'at_x', 'nwc_at')     , safe_div('lt', 'ppent', 'lt_ppen'),\
                    safe_div('dltt', 'be_x', 'debtlt_be') , safe_div('fcf_x', 'ocf_x', 'fcf_ocf', 3)]
    c_misc_rat  = [safe_div('xad','sale_x','adv_sale')  , safe_div('xlr','sale_x','staff_sale'), safe_div('sale_x', 'bev_x', 'sale_bev')   ,\
                    safe_div('xrd', 'sale_x', 'rd_sale'), safe_div('sale_x', 'be_x', 'sale_be'), safe_div('sale_x', 'nwc_x', 'sale_nwc', 3),\
                    safe_div('txt', 'pi_x', 'tax_pi', 3), safe_div('che', 'at_x', 'cash_at', 3), safe_div('ni_x', 'emp', 'ni_emp', 3)      ,\
                    safe_div('sale_x', 'emp', 'sale_emp', 3), pl.when((pl.coalesce('nix_x', 'ni_x') > 0.) & (col('nix_x') != 0)).then(col("div_x") / col("nix_x")).otherwise(fl_none()).alias('div_ni')]
    return c_fin_s_rat + c_misc_rat

@measure_time
def create_acc_chars(data_path, output_path, lag_to_public, max_data_lag, __keep_vars, me_data_path, suffix):
    #fx datadate in the new code is the same as date in SAS code
    fx = compustat_fx().rename({'datadate': 'date'}).lazy()
    #adding and filtering market return data
    me_data = load_mkt_equity_data(me_data_path, False)

    chars_df = pl.scan_parquet(data_path)

    chars_df = (chars_df.sort(['gvkey', 'curcd', 'datadate'])
                        .with_columns(count       = col('gvkey').cum_count().over(['gvkey', 'curcd']),
                                      assets      = col('at_x'),
                                      sales       = col('sale_x'),
                                      book_equity = col('be_x'),
                                      net_income  = col('ni_x'))
                        .pipe(add_accounting_misc_cols_1)
                        .with_columns(financial_soundness_and_misc_ratios_exps())
                        .pipe(add_liq_and_efficiency_ratios)
                        .sort(['gvkey', 'curcd', 'datadate'])
                        .with_columns(sale_emp_gr1 = pl.when((col('count') > 12) & (col('sale_emp').shift(12) > 0))\
                                                       .then(col('sale_emp') / col('sale_emp').shift(12) - 1)\
                                                       .otherwise(fl_none()))
                        .sort(['gvkey', 'curcd', 'datadate'])
                        .with_columns(emp_gr(data_path))
                        .pipe(calculate_consecutive_earnings_increases)
                        .sort(['gvkey', 'curcd', 'datadate'])
                        .with_columns([chg_to_lagassets(i) for i in ['noa_x', 'ppeinv_x']] +  #1yr Change Scaled by Lagged Assets)
                                      [chg_to_avgassets(i) for i in  ['lnoa_x']]           +  #1yr Change Scaled by Average Assets
                                      [var_growth(var_gr='capx', horizon=24)])                #CAPEX growth over 2 years
                        .sort(['gvkey', 'curcd', 'datadate'])
                        #Quarterly profitability measures:
                        .with_columns([pl.when((col('count') > 12) & (col('sale_qtr').shift(12) > 0)).then(col('sale_qtr') / col('sale_qtr').shift(12) - 1).otherwise(fl_none()).alias('saleq_gr1'),
                                       safe_div('ni_qtr', 'be_x', 'niq_be', 9),
                                       safe_div('ni_qtr', 'at_x', 'niq_at', 9)])
                        .sort(['gvkey', 'curcd', 'datadate'])
                        .with_columns([pl.when(col('count') > 12).then(col('niq_be') - col('niq_be').shift(12)).otherwise(fl_none()).alias('niq_be_chg1'),
                                       pl.when(col('count') > 12).then(col('niq_at') - col('niq_at').shift(12)).otherwise(fl_none()).alias('niq_at_chg1')])
                        .sort(['gvkey', 'curcd', 'datadate'])
                        #R&D capital-to-assets
                        .with_columns(pl.when((col('count') > 48) & (col('at_x') > 0))\
                                        .then((col('xrd') + col('xrd').shift(12) * 0.8 + col('xrd').shift(24) * 0.6 + col('xrd').shift(36) * 0.4 + col('xrd').shift(48) * 0.2) / col('at_x'))\
                                        .otherwise(fl_none())\
                                        .alias('rd5_at'))
                        .sort(['gvkey', 'curcd', 'datadate'])
                        .with_columns([chg_to_exp(i) for i in ['sale_x', 'invt', 'rect', 'gp_x', 'xsga']])#Abarbanell and Bushee (1998)
                        .with_columns(dsale_dinv = col('sale_ce') - col('invt_ce'),
                                      dsale_drec = col('sale_ce') - col('rect_ce'),
                                      dgp_dsale  = col('gp_ce')   - col('sale_ce'),
                                      dsale_dsga = col('sale_ce') - col('xsga_ce'))
                        .drop(['sale_ce', 'invt_ce', 'rect_ce', 'gp_ce', 'xsga_ce'])
                        .pipe(standardized_unexpected, var='sale_qtr', qtrs=8, qtrs_min=6)
                        .pipe(standardized_unexpected, var='ni_qtr', qtrs=8, qtrs_min=6)
                        .pipe(compute_capex_abn)
                        .pipe(add_profit_scaled_by_lagged_vars)
                        .with_columns(pi_nix       = safe_div('pi_x', 'nix_x', 'pi_nix', 8),
                                      ocf_at       = safe_div('ocf_x', 'at_x', 'ocf_at'),
                                      op_at        = safe_div('op_x', 'at_x', 'op_at', 3),
                                      at_be        = safe_div('at_x', 'be_x', 'at_be'),
                                      __ocfq_saleq = safe_div('ocf_qtr', 'sale_qtr', '__ocfq_saleq', 3),
                                      __niq_saleq  = safe_div('ni_qtr', 'sale_qtr', '__niq_saleq', 3),
                                      __roeq       = safe_div('ni_qtr', 'be_x', '__roeq', 3),
                                      __roe        = safe_div('ni_x', 'be_x', '__roe', 3),
                                      tangibility  = tangibility(),
                                      aliq_at = safe_div('aliq_x', 'at_x', 'aliq_at', 4))
                        .sort(['gvkey', 'curcd', 'datadate'])
                        .with_columns(ocf_at_chg1  = pl.when(col('count') > 12).then(col('ocf_at') - col('ocf_at').shift(12)).otherwise(fl_none()))
                        .pipe(add_accounting_misc_cols_2)
                        .pipe(add_earnings_persistence_and_expand, lag_to_pub = lag_to_public, max_lag = max_data_lag)
                        .pipe(convert_raw_vars_to_usd)
                        .pipe(add_me_data_and_compute_me_mev_mat_eqdur_vars, me_df = me_data))

    rename_dict = {"xrd": "rd","xsga": "sga","dlc": "debtst","dltt": "debtlt","oancf": "ocf","ppegt": "ppeg","ppent": "ppen","che": "cash","invt": "inv","rect": "rec","txt": "tax","ivao": "lti","ivst": "sti","sale_qtr": "saleq","ni_qtr": "niq","ocf_qtr": "ocfq"}
    rename_cols_and_select_keep_vars(chars_df, rename_dict, __keep_vars, suffix)\
                                    .sort(['gvkey', 'public_date'])\
                                    .unique(['gvkey', 'public_date'],keep = 'first')\
                                    .sort(['gvkey', 'public_date'])\
                                    .collect()\
                                    .write_parquet(output_path)
def combine_qtr_ann_vars(var, suffix):
    c1 = col(var).is_null()
    c2 = (col(f"{var}{suffix}").is_not_null()) & (col(f"datadate{suffix}") > col('datadate'))
    return pl.when(c1|c2).then(col(f"{var}{suffix}")).otherwise(col(var)).alias(var)
@measure_time
def combine_ann_qtr_chars(ann_df_path, qtr_df_path, char_vars, q_suffix):
    ann_df = pl.scan_parquet(ann_df_path)
    qtr_df = pl.scan_parquet(qtr_df_path)
    combined_df = (ann_df.join(qtr_df, on=['gvkey', 'public_date'], how='left', suffix=q_suffix)
                        .select(['source', 'gvkey', 'public_date'] + 
                                [combine_qtr_ann_vars(char_var, q_suffix) for char_var in char_vars])
                        .unique(['gvkey', 'public_date'])
                        .sort(['gvkey', 'public_date']))
    combined_df.collect().write_parquet('acc_chars_world.parquet')
def seasonality(data, ret_x, start_year, end_year):
    all_r = pl.lit(0.)
    ann_r = pl.lit(0.)
    for i in range((start_year-1)*12, (end_year*12)): all_r += col(ret_x).shift(i)
    for i in range((start_year*12-1), (end_year*12), 12): ann_r += col(ret_x).shift(i)
    c1 = col('count') > (end_year*12)
    seas_an_exp = ann_r / len(range((start_year*12-1), (end_year*12), 12))
    seas_na_exp = (all_r - ann_r) / (len(range((start_year-1)*12, (end_year*12))) - len(range((start_year*12-1), (end_year*12), 12)))
    data = (data.sort(['id','eom'])
                .with_columns([pl.when(c1).then(seas_an_exp).otherwise(fl_none()).alias(f'seas_{start_year}_{end_year}an'),
                               pl.when(c1).then(seas_na_exp).otherwise(fl_none()).alias(f'seas_{start_year}_{end_year}na')]))
    return data
def mom_rev_cols(i, j):
    c1 = col('ri_x').shift(j) != 0
    c2 = col('count') > j
    c3 = (col('ret_x').shift(i)).is_not_null()
    return (pl.when(c1 & c2 & c3).then(col('ri_x').shift(i)/col('ri_x').shift(j) - 1).otherwise(fl_none())).alias(f'ret_{j}_{i}')
def chcsho_cols(i):
    c1 = col('aux').shift(i) != 0
    c2 = col('count') > i
    return (pl.when(c1 & c2).then(col('aux')/col('aux').shift(i) - 1).otherwise(fl_none())).alias(f'chcsho_{i}m')
def eqnpo_cols(lag):
    c1 = (col('ri') > 0) & (col('ri').shift(lag) > 0)
    c2 = (col('me') > 0) & (col('me').shift(lag) > 0)
    c3 = col('count') > lag
    eqnpo_col_exp = (col('ri')/col('ri').shift(lag)).log() - (col('me')/col('me').shift(lag)).log()
    return (pl.when(c1 & c2 & c3).then(eqnpo_col_exp).otherwise(fl_none())).alias(f'eqnpo_{lag}m')
def div_cols(i, spc = False):
    div_var = 'div' if (not spc) else 'divspc'
    num = col(f'{div_var}1m_me') if (i == 1) else col(f'{div_var}1m_me').rolling_sum(window_size = i, min_periods = 1)
    return (pl.when((col('count') >= i) & (col('me') != 0)).then(num / col('me')).otherwise(fl_none())).alias(f'{div_var}{i}m_me')
@measure_time
def market_chars_monthly(data_path, market_ret_path, local_currency):
    div_range = [1,3,6,12]#[1,3,6,12,24,36]
    div_spc_range = [1,12]
    chcsho_lags = [1,3,6,12]
    eqnpo_lags = [1,3,6,12]
    mom_rev_lags = [[0, 1],[0, 2],[0, 3],[1, 3],[0, 6],[1, 6],[0, 9],[1, 9],[0, 12],[1, 12],[7, 12],[1, 18],[1, 24],[12, 24],[1, 36],[12, 36],[12, 48],[1, 48],[1, 60],[12, 60],[36, 60]]
    ret_var = 'ret_local' if (local_currency == 1) else 'ret'
    market_ret = pl.scan_parquet(market_ret_path)
    data = (pl.scan_parquet(data_path)
              .join(market_ret, how = 'left', on = ['excntry', 'eom'])
              .with_columns(pl.when(col('ret_local') == 0).then(pl.lit(1)).otherwise(pl.lit(0)).alias('ret_zero'))
              .select(['id','date','eom','me','shares','adjfct','prc','ret','ret_local', 'div_tot', 'div_cash', 'div_spc', 'dolvol','ret_lag_dif','ret_zero','ret_exc','mkt_vw_exc', col(ret_var).alias('ret_x')]))
    __stock_coverage = (data.group_by('id')
                            .agg(start_date = pl.min('eom'),
                                 end_date   = pl.max('eom'))
                            .sort(['id','start_date']))
    __full_range = expand(__stock_coverage, ['id'], 'start_date', 'end_date', 'month', 'eom')
    data = (__full_range.join(data, how = 'left', on = ['id', 'eom'])
                        .select(['id', 'eom', 'me', 'shares', 'adjfct', 'prc', 'ret','ret_local','ret_x','ret_lag_dif','div_tot','div_cash','div_spc','dolvol','ret_zero','ret_exc','mkt_vw_exc'])
                        .sort(['id', 'eom'])
                        .with_columns(ri       = ((1 + pl.coalesce(['ret',   0])).cum_prod()).over('id'),
                                      ri_x     = ((1 + pl.coalesce(['ret_x', 0])).cum_prod()).over('id'),
                                      count    = (col('id').cum_count()).over('id'),
                                      ret_miss = pl.when((col('ret_x').is_not_null()) & (col('ret_lag_dif') == 1)).then(pl.lit(0)).otherwise(pl.lit(1)))
                        .with_columns([pl.when(col('ret_miss') == 1).then(fl_none()).otherwise(i).alias(i) for i in ['ret_x', 'ret', 'ret_local', 'ret_exc', 'mkt_vw_exc']])
                        .drop(['ret_zero', 'ret_lag_dif'])
                        .unique(['id','eom'])
                        .with_columns(market_equity = col('me'),
                                      div1m_me      = col('div_tot') * col('shares'),
                                      divspc1m_me   = col('div_spc') * col('shares'),
                                      aux           = col('shares')  * col('adjfct'))
                        .sort(['id', 'eom'])
                        .with_columns([div_cols(i, spc = False) for i in div_range] +\
                                      [div_cols(i, spc = True) for i in div_spc_range] +\
                                      [eqnpo_cols(i) for i in eqnpo_lags] +\
                                      [chcsho_cols(i) for i in chcsho_lags] +\
                                      [mom_rev_cols(i,j) for i,j in mom_rev_lags]))
    for i in [[1,1], [2,5], [6, 10], [11, 15], [16, 20]]: data = seasonality(data, 'ret_x', i[0], i[1])
    data = (data.drop(['me','shares','adjfct', 'adjfct', 'prc', 'ret','ret_local','ret_x', 'div_tot', 'div_cash', 'div_spc', 'dolvol', 'ret_exc', 'mkt_vw_exc','ret_miss', 'ri_x', 'ri', 'count', 'aux'])
                .sort(['id','eom']))
    #DO NOT USE STREAMING HERE
    data.collect().write_parquet('market_chars_m.parquet')
def load_age_data_and_process_dates(mode):
    df = (pl.scan_parquet(f'Raw_data_dfs/comp_{mode}_age.parquet')
            .group_by('gvkey')
            .agg(col('datadate').min().alias(f'comp_{mode}_first'))
            .sort('gvkey')
            .with_columns(col(f'comp_{mode}_first').dt.offset_by('-1y').dt.month_end())
            .with_columns((col(f'comp_{mode}_first').dt.year().cast(pl.Utf8) + pl.lit('-12-31')).str.strptime(pl.Date)))
    return df

@measure_time
def firm_age(data_path):
    crsp_age = pl.scan_parquet('Raw_data_dfs/crsp_age.parquet')
    comp_acc_age = load_age_data_and_process_dates('acc')
    comp_ret_age = load_age_data_and_process_dates('ret')
    data = (pl.scan_parquet(data_path)
              .select(['permco','gvkey','id','eom'])
              .join(crsp_age, on = 'permco', how = 'left')
              .join(comp_acc_age, on = 'gvkey', how = 'left')
              .join(comp_ret_age, on = 'gvkey', how = 'left')
              .with_columns(first_obs = pl.min_horizontal('crsp_first', 'comp_acc_first','comp_ret_first'))
              .drop(['permco','gvkey','crsp_first','comp_acc_first', 'comp_ret_first'])
              .with_columns(first_alt = pl.min('eom').over('id'))
              .with_columns(aux = pl.min_horizontal('first_obs', 'first_alt'))
              .with_columns(age = gen_MMYY_column('eom') - gen_MMYY_column('aux'))
              .drop('aux')
              .sort(['id','eom']))
    data.collect(streaming = True).write_parquet('firm_age.parquet')
def char_pf_rets():
    lms = ((col('small_high') + col('big_high')) / 2 - (col('small_low') + col('big_low')) / 2).alias('lms')
    smb = ((col('small_high') + col('small_mid') + col('small_low')) / 3 - (col('big_high') + col('big_mid') + col('big_low')) / 3).alias('smb')
    return [lms, smb]

@measure_time
def sort_ff_style(char, freq, min_stocks_bp, min_stocks_pf, date_col, data, sf):
    print(f'Executing sort_ff_style for {char}', flush=True)
    c1 = (((col('size_grp_l').is_in(['small', 'large', 'mega'])) & (col('excntry_l') != 'USA')) | ((col('crsp_exchcd_l') == 1) | (col('comp_exchg_l') == 11) & (col('excntry_l') == 'USA'))) &\
         col(f'{char}_l').is_not_null()
    char_pf_exp = (pl.when(col(f'{char}_l') >= col('bp_p70')).then(pl.lit('high'))\
                     .when(col(f'{char}_l') >= col('bp_p30')).then(pl.lit('mid'))\
                     .otherwise(pl.lit('low'))).alias('char_pf')
    over_vars = ['excntry_l','size_pf', 'char_pf', 'eom']
    bp_stocks = (data.sort(['eom','excntry_l','id'])
                     .filter(c1)
                     .group_by(['eom', 'excntry_l'])
                     .agg(n   = pl.len().alias('n'),
                          aux = col(f'{char}_l'))
                     .with_columns(bp_p30 = perc_exp('aux', lambda x: perc_method(x, 0.3), True),
                                   bp_p70 = perc_exp('aux', lambda x: perc_method(x, 0.7), True))
                     .drop('aux'))
    data = (data.join(bp_stocks, how = 'left', on = ['excntry_l', 'eom'])
                .filter((col('n') >= min_stocks_bp) & (col(f'{char}_l').is_not_null()) & (col('size_pf') != '').fill_null(pl.lit(True).cast(pl.Boolean)))
                .select(['excntry_l','id','eom','size_pf', 'me_l', char_pf_exp])
                .with_columns(w = pl.when((pl.sum('me_l') != 0).over(over_vars)).then((col('me_l')/pl.sum('me_l')).over(over_vars)).otherwise(fl_none()),
                              n = pl.len().over('excntry_l','size_pf', 'char_pf', 'eom'))
                .filter(col('n') >= min_stocks_pf)
                .drop('n'))
    returns = sf.join(data, how = 'inner', left_on = ['id','eom','excntry'], right_on = ['id','eom','excntry_l'])
    returns = (returns.with_columns(ret_exc = col('ret_exc') * col('w'))
                      .group_by(['excntry', 'size_pf', 'char_pf', date_col])
                      .agg(ret_exc = pl.sum('ret_exc'))
                      .with_columns(characteristic = pl.lit(char),
                                    combined_pf = (col('size_pf') + '_' + col('char_pf')))
                      .drop(['size_pf', 'char_pf'])
                      .collect()
                      .pivot(values='ret_exc',index=['excntry', date_col],columns='combined_pf')
                      .select(['excntry', date_col, *char_pf_rets()]).sort(['excntry', date_col]))
    return returns

@measure_time
def ap_factors(output_path, freq, sf_path, mchars_path, mkt_path, min_stocks_bp, min_stocks_pf):
    date_col    = 'eom' if freq == 'm' else 'date'
    sf_cols     = {'m': ['excntry','id','eom', 'ret_exc','ret_lag_dif'], 'd': ['excntry','id', 'date','eom', 'ret_exc','ret_lag_dif']}
    sf_cond     = {'m': (col('ret_lag_dif') == 1) & (col('ret_exc').is_not_null()), 'd': (col('ret_lag_dif') <= 5) & (col('ret_exc').is_not_null())}
    copied_cols = ['id','eom','market_equity','source_crsp','ret_lag_dif']
    lag_vars    = ['comp_exchg', 'crsp_exchcd', 'exch_main', 'obs_main', 'common', 'primary_sec', 'excntry', 'size_grp', 'me', 'be_me', 'at_gr1', 'niq_be']
    lagged_cols = [col(i).shift(1).over(['id', 'source_crsp']).alias(i + '_l') for i in lag_vars]

    print(f'Executing AP factors with frequency {freq}', flush=True)
    world_sf1 = (pl.scan_parquet(sf_path)
                   .select(pl.all().shrink_dtype())
                   .select(sf_cols[freq])
                   .filter(sf_cond[freq])
                   .drop('ret_lag_dif'))
    world_sf2 =  winsorize_var(world_sf1, ['eom'], 'ret_exc', 0.1/100, 99.9/100)

    base = (pl.scan_parquet(mchars_path)
              .select(pl.all().shrink_dtype())
              .sort(['id', 'eom'])
              .select(copied_cols + lagged_cols)
              .sort(['id', 'eom'])
              .with_columns([pl.when(((12* (col('eom').dt.year() - col('eom').shift(1).dt.year()) +\
                                       (col('eom').dt.month() - col('eom').shift(1).dt.month()).cast(pl.Int32)).over('id') != 1)).then(pl.lit(None))
                               .otherwise(i + '_l').alias(i + '_l') for i in lag_vars])
              .filter((col('obs_main_l') == 1)    &\
                      (col('exch_main_l') == 1)   &\
                      (col('common_l') == 1)      &\
                      (col('primary_sec_l') == 1) &\
                      (col('ret_lag_dif') == 1)   &\
                      col('me_l').is_not_null())
              .with_columns(size_pf = (pl.when(col('size_grp_l').is_null()).then(pl.lit(''))\
                                         .when(col('size_grp_l').is_in(['large', 'mega'])).then(pl.lit('big'))\
                                         .otherwise(pl.lit('small'))))
              .sort(['excntry_l', 'size_grp_l', 'eom']))

    ff           = sort_ff_style('be_me' , freq, min_stocks_bp,min_stocks_pf, date_col, base, world_sf2).rename({'lms': 'hml'       , 'smb': 'smb_ff'})

    asset_growth = sort_ff_style('at_gr1', freq, min_stocks_bp,min_stocks_pf, date_col, base, world_sf2).rename({'lms': 'at_gr1_lms', 'smb': 'at_gr1_smb'})
    roeq         = sort_ff_style('niq_be', freq, min_stocks_bp,min_stocks_pf, date_col, base, world_sf2).rename({'lms': 'niq_be_lms', 'smb': 'niq_be_smb'})
    hxz          = (asset_growth.join(roeq, how = 'left', on = ['excntry',date_col])
                                .select(['excntry', date_col, (-1*col('at_gr1_lms')).alias('inv'),\
                                         col('niq_be_lms').alias('roe'), ((col('at_gr1_smb') + col('niq_be_smb'))/2).alias('smb_hxz')]))

    output = (pl.scan_parquet(mkt_path)
                .select(['excntry', date_col, col('mkt_vw_exc').alias('mktrf')])
                .collect()
                .join(ff , how = 'left', on = ['excntry', date_col])
                .join(hxz, how = 'left', on = ['excntry', date_col]))
    output.write_parquet(output_path)
def get_beta_and_ivol_exp(x, y, __n, __min, beta_var):
    beta_exp  = pl.rolling_cov(x,y,window_size=__n, min_periods = __min)/col(x).rolling_var(window_size=__n, min_periods = __min)
    ivol_exp  = ((col(y).rolling_var(window_size=__n, min_periods = __min) - (col(beta_var) ** 2) * col(x).rolling_var(window_size=__n, min_periods = __min)) ** (1/2)).fill_nan(None)
    return beta_exp, ivol_exp
def merge_sf_and_fcts(data_path, fcts_path):
    fcts = (pl.scan_parquet(fcts_path)
              .filter(col('mktrf').is_not_null())
              .select(['excntry', 'eom','mktrf', 'hml', 'smb_ff']))
    data = (pl.scan_parquet(data_path)
              .filter((col('ret_local') != 0) & (col('ret_exc').is_not_null()) & (col('ret_lag_dif') == 1))
              .select(['excntry','id','eom','ret_exc']))
    sf = (data.join(fcts, how = 'left', on = ['excntry', 'eom'])
              .drop('excntry'))
    return sf
def gen_resampled_dates(df, id_vars, time_var, freq):
    df = (df.select([*id_vars, time_var])
            .sort([*id_vars, time_var])
            .with_columns(aux = (col(time_var).shift(-1).over(id_vars)).dt.offset_by('-' + freq).dt.month_end())
            .with_columns((pl.coalesce([pl.date_ranges(start= time_var, end = 'aux', interval = freq), pl.concat_list([col(time_var)])])).alias(time_var))
            .explode(time_var)
            .select([*id_vars, time_var])
            .unique())
    return df
def gen_sf_for_regression(data_path, fcts_path, id_vars, time_var, freq, wins_var, perc_low, perc_high):
    __msf = merge_sf_and_fcts(data_path, fcts_path)
    __msf2 = gen_resampled_dates(__msf, id_vars, time_var, freq)
    __msf = winsorize_var(__msf, [time_var], wins_var, perc_low, perc_high)
    __msf = __msf2.join(__msf, on = [*id_vars, time_var], how = 'left')
    return __msf

@measure_time
def market_beta(output_path, data_path, fcts_path, __n , __min):
    beta_var, ivol_var = f'beta_{__n}m', f'ivol_capm_{__n}m'
    beta_exp, ivol_exp = get_beta_and_ivol_exp('mktrf', 'ret_exc', __n, __min, beta_var)
    __msf = gen_sf_for_regression(data_path, fcts_path, ['id'], 'eom', '1mo', 'ret_exc', 0.1/100, 99.9/100)
    __msf = (__msf.sort(['id','eom'])
                  .with_columns(beta_exp.over('id').alias(beta_var))
                  .sort(['id','eom'])
                  .with_columns(ivol_exp.over('id').alias(ivol_var))
                  .unique(['id', 'eom'])
                  .select(['id','eom', beta_var, ivol_var])
                  .sort(['id','eom'])
                  .with_columns([col(beta_var).fill_null(strategy = 'forward').over('id').alias(beta_var),
                                 col(ivol_var).fill_null(strategy = 'forward').over('id').alias(ivol_var)]))
    __msf.sort(['id','eom']).collect().write_parquet(output_path)
    
@measure_time
def prepare_daily(data_path, fcts_path):
    data = pl.scan_parquet(data_path)
    fcts = pl.scan_parquet(fcts_path)
    dsf1 = (data.select(['excntry', 'id', 'date', 'eom', 'prc', 'adjfct', 'ret', 'ret_exc', 'dolvol', 'shares', 'tvol', 'ret_lag_dif', 'ret_local'])
                .join(fcts, how = 'left', on = ['excntry', 'date'])
                .filter(col('mktrf').is_not_null())
                .with_columns(zero_obs = pl.when(col('ret_local') == 0).then(1).otherwise(0),
                              id_int = pl.col('id').rank(method = 'min').cast(pl.Int64))
                .with_columns(zero_obs = pl.sum('zero_obs').over(['id_int', 'eom']),
                              ret_exc  = pl.when(col('ret_lag_dif') <= 14).then(col('ret_exc')).otherwise(fl_none()),
                              ret      = pl.when(col('ret_lag_dif') <= 14).then(col('ret')).otherwise(fl_none()),
                              dolvol_d = col('dolvol'),
                              prc_adj  = safe_div('prc', 'adjfct', 'prc_adj'))
                .drop(['ret_lag_dif', 'ret_local', 'adjfct','prc', 'dolvol']))
    dsf1.select(pl.all().shrink_dtype()).sort('eom').collect(streaming=True).write_parquet('dsf1.parquet')

    id_int_key = (pl.scan_parquet('dsf1.parquet')
                    .select(['id', 'id_int'])
                    .unique())
    id_int_key.collect(streaming=True).write_parquet('id_int_key.parquet')

    mkt_lead_lag = (fcts.select(['excntry', 'date', 'mktrf', col('date').dt.month_end().alias('eom')])
                        .sort(['excntry','date'])
                        .with_columns(mktrf_ld1 = col('mktrf').shift(-1).over(['excntry','eom']),
                                      mktrf_lg1 = col('mktrf').shift(1).over(['excntry'])))
    mkt_lead_lag.collect(streaming=True).write_parquet('mkt_lead_lag.parquet')

    corr_data = (pl.scan_parquet('dsf1.parquet')
                   .select(['ret_exc', 'id', 'id_int', 'date', 'mktrf', 'eom', 'zero_obs'])
                   .sort(['id_int','date'])
                   .with_columns(ret_exc_3l = (col('ret_exc') + col('ret_exc').shift(1) + col('ret_exc').shift(2)).over(['id_int']),
                                 mkt_exc_3l = (col('mktrf')   + col('mktrf').shift(1)   + col('mktrf').shift(2)).over(['id_int']))
                   .select(['id_int', 'eom', 'zero_obs', 'ret_exc_3l', 'mkt_exc_3l']))
    corr_data.collect(streaming = True).write_parquet('corr_data.parquet')

def gen_ranks_and_normalize(df, id_vars, geo_vars, time_vars, desc_flag, var, min_stks):
    by_vars = geo_vars + time_vars
    ranked_var = f'rank_{var}'
    var_ranks = (df.select([*id_vars, *by_vars, var])
                   .with_columns(count = pl.count(var).over(by_vars))
                   .filter(col('count') > min_stks)
                   .with_columns(col(var).rank(descending = desc_flag).over(by_vars).alias(ranked_var))
                   .with_columns(((col(ranked_var)-pl.min(ranked_var))/col('count')).over(by_vars).alias(ranked_var))
                   .drop([*geo_vars, var, 'count']))
    return var_ranks
def gen_misp_exp(var_list, min_fcts):
    sum = col('rank_' + var_list[0]).is_null().cast(pl.Int32)
    for i in var_list[1:]: sum += col('rank_' + i).is_null().cast(pl.Int32)
    c1 = (sum > min_fcts)
    return pl.when(c1).then(fl_none()).otherwise(pl.mean_horizontal(['rank_' + f'{i}' for i in var_list]))

@measure_time
def mispricing_factors(data_path, min_stks, min_fcts = 3):
    vars_mgmt = ['chcsho_12m','eqnpo_12m','oaccruals_at','noa_at','at_gr1','ppeinv_gr1a']
    vars_perf = ['o_score','ret_12_1','gp_at','niq_at']
    direction = [True, False, True, True, True, True, True, False, False, False]
    index = [1,2,3,4,5,6,7,8,9,10]
    aux_df = (pl.scan_parquet(data_path)
                .filter((col('common')      == 1)      &\
                        (col('primary_sec') == 1)      &\
                        (col('obs_main')    == 1)      &\
                        (col('exch_main')   == 1)      &\
                        (col('ret_exc').is_not_null()) &\
                        (col('me').is_not_null()))
                .select(['id', 'eom', 'excntry', *(vars_mgmt + vars_perf)])
                .sort(['excntry', 'eom']))
    chars = {'1': aux_df}
    for __d, __v, i in zip(direction, vars_mgmt + vars_perf, index):
        subset = gen_ranks_and_normalize(aux_df, ['id'], ['excntry'], ['eom'], __d, __v, min_stks)
        chars[f'{(i+1)%2}'] = chars[f'{i%2}'].join(subset, on = ['id', 'eom'], how = 'left')
    chars['1'] = (chars['1'].with_columns(mispricing_mgmt = gen_misp_exp(vars_mgmt, min_fcts),
                                          mispricing_perf = gen_misp_exp(vars_perf, min_fcts))
                            .select(['id', 'eom', 'mispricing_perf', 'mispricing_mgmt']))
    chars['1'].collect().write_parquet('mp_factors.parquet')

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

def residual_momentum(output_path, data_path, fcts_path, __n, __min, incl, skip):
    w = incl - skip
    alpha_exp, beta1_exp, beta2_exp, beta3_exp = regression_3vars('ret_exc', 'mktrf', 'smb_ff', 'hml', __n, __min)
    residual_exp = lambda i:  col('ret_exc').shift(i) - (col('alpha') * pl.lit(1.) + col('beta1') * col('mktrf').shift(i) + col('beta2') * col('smb_ff').shift(i) + col('beta3') * col('hml').shift(i))
    res_var = f'resff3_{incl}_{skip}'
    __msf = gen_sf_for_regression(data_path, fcts_path, ['id'], 'eom', '1mo', 'ret_exc', 0.1/100, 99.9/100)
    __msf = (__msf.sort(['id','eom'])
                  .with_columns(alpha = alpha_exp.over('id'),
                                beta1 = beta1_exp.over('id'),
                                beta2 = beta2_exp.over('id'),
                                beta3 = beta3_exp.over('id'))
                  .sort(['id','eom'])
                  .with_columns(std_res = pl.concat_list([residual_exp(i).over(['id']) for i in range(1, w+1)]).list.drop_nulls())
                  .select(['id','eom','std_res'])
                  .with_columns(std_res = col('std_res').list.eval(pl.element().mean()/pl.element().std()),
                                den     = col('std_res').list.eval(pl.element().std()))
                  .explode(['std_res','den'])
                  .with_columns((pl.when(col('den') != 0).then(col('std_res')).otherwise(fl_none())).alias(res_var))
                  .select(['id','eom', res_var])
                  .sort(['id','eom']))
    __msf.collect().write_parquet(output_path + f'_{incl}_{skip}.parquet')
    del __msf

@measure_time
def bidask_hl(output_path, data_path, market_returns_daily_path, __min_obs):
    pi = 3.141592653589793
    k2 = sqrt(8 / pi)
    const = 3 - 2 * sqrt(2)
    c1 = (col('bidask') != 1)                &\
         (col('prc_low') != col('prc_high')) &\
         (col('prc_low') > 0)                &\
         (col('prc_high') > 0)               &\
         (col('tvol') != 0)
    c2 = (col('id_flag') == 0) & (col('ind') == 0) & (col('prc_low_r') <= col('prc')) & (col('prc') <= col('prc_high_r'))
    c3 = (col('id_flag') == 0) & (col('ind') == 0) & (col('prc') < col('prc_low_r'))
    c4 = (col('id_flag') == 0) & (col('ind') == 0) & (col('prc') > col('prc_high_r'))
    c5 = (col('prc_low')!= 0.) & (col('prc_high')/col('prc_low') > 8)
    c6 = (col('prc_l1') < col('prc_low')) & (col('prc_l1') > 0)
    c7 = (col('prc_l1') > col('prc_high')) & (col('prc_l1') > 0)
    market_returns_daily = pl.scan_parquet(market_returns_daily_path)
    __dsf = (pl.scan_parquet(data_path)
               .join(market_returns_daily, how = 'left', on=['excntry', 'date'])
               .filter(col('mkt_vw_exc').is_not_null())
               .with_columns([safe_div(var, 'adjfct', var) for var in ['prc', 'prc_high', 'prc_low']])
               .select(['id', 'date', 'eom', 'bidask', 'tvol', 'prc', 'prc_high', 'prc_low'])
               .with_columns(prc_high         = pl.when(c1).then(col('prc_high')).otherwise(fl_none()),
                             prc_low          = pl.when(c1).then(col('prc_low')).otherwise(fl_none()))
               .sort(['id','date'])
               .with_columns(prc_low_r        = col('prc_low').shift(1).over('id'),
                             prc_high_r       = col('prc_high').shift(1).over('id'),
                             ind              = ((0 < col('prc_low')) & (col('prc_low') < col('prc_high'))).cast(pl.Int64).fill_null(0),
                             id_flag          = (col('id').shift(1).over('id').is_null()).cast(pl.Int64))
               .sort(['id','date'])
               .with_columns(prc_low_r        = col('prc_low_r' ).fill_null(strategy = 'forward').over('id'),
                             prc_high_r       = col('prc_high_r').fill_null(strategy = 'forward').over('id'))
               .with_columns(prc_low          = (pl.when(c2).then(col('prc_low_r'))
                                                   .when(c3).then(col('prc'))
                                                   .when(c4).then(col('prc_low_r') + col('prc') - col('prc_high_r'))
                                                   .otherwise(col('prc_low'))),
                             prc_high         = (pl.when(c2).then(col('prc_high_r'))
                                                   .when(c3).then(col('prc_high_r') - (col('prc_low_r') - col('prc')))
                                                   .when(c4).then(col('prc'))
                                                   .otherwise(col('prc_high'))))
               .with_columns(prc_low          = pl.when(c5).then(fl_none()).otherwise(col('prc_low')),
                             prc_high         = pl.when(c5).then(fl_none()).otherwise(col('prc_high')))
               .select(['id','date','eom','prc', 'prc_high','prc_low'])
               .sort(['id','date'])
               .with_columns(prc_low_t        = col('prc_low'),
                             prc_high_t       = col('prc_high'),
                             prc_low_l1       = col('prc_low').shift(1).over('id'),
                             prc_high_l1      = col('prc_high').shift(1).over('id'),
                             prc_l1           = col('prc').shift(1).over('id'))
               .with_columns(prc_high_t       = pl.when(c6).then(col('prc_high') - (col('prc_low') - col('prc_l1'))).otherwise(col('prc_high_t')),
                             prc_low_t        = pl.when(c6).then(col('prc_l1')).otherwise(col('prc_low_t')))
               .with_columns(prc_high_t       = pl.when(c7).then(col('prc_l1')).otherwise(col('prc_high_t')),
                             prc_low_t        = pl.when(c7).then(col('prc_low') + (col('prc_l1') - col('prc_high'))).otherwise(col('prc_low_t')))
               .with_columns(prc_high_2d      = pl.max_horizontal('prc_high_t', 'prc_high_l1'),
                             prc_low_2d       = pl.min_horizontal('prc_low_t', 'prc_low_l1'))
               .with_columns(beta             = (pl.when((col('prc_low_t') > 0) & (col('prc_low_l1') > 0))
                                                   .then(((col('prc_high_t') / col('prc_low_t')).log() ** 2) + ((col('prc_high_l1') / col('prc_low_l1')).log() ** 2))
                                                   .otherwise(fl_none())),
                             gamma            = pl.when(col('prc_low_2d') > 0).then(((col('prc_high_2d') / col('prc_low_2d')).log() ** 2)).otherwise(fl_none()))
               .with_columns(alpha            = ((sqrt(2) - 1) * col('beta').sqrt()) / const - (col('gamma') / const).sqrt())
               .with_columns(spread           = 2 * (pl.lit(exp(1)).pow(col('alpha')) - 1) / (1 + pl.lit(exp(1)).pow(col('alpha'))),
                             sigma            = (((col('beta')/ 2).sqrt() - col('beta').sqrt()) / (k2 * const)) + (col('gamma') / (k2 * k2 * const)).sqrt())
               .with_columns(spread_0         = pl.when(col('spread') < 0).then(pl.lit(0.)).otherwise(col('spread')),
                             sigma_0          = pl.when(col('sigma') < 0).then(pl.lit(0.)).otherwise(col('sigma')))
               .select(['id','date','eom','spread_0','sigma_0'])
               .group_by(['id','eom'])
               .agg(bidaskhl_21d = pl.mean('spread_0'),
                    rvolhl_21d   = pl.mean('sigma_0'),
                    count        = pl.count('spread_0'))
                     #count = pl.sum('count'))
               .filter(col('count') > __min_obs)
               .drop('count')
               .sort(['id','eom']))
    __dsf.collect().write_parquet(output_path)

@measure_time
def create_world_data_prelim(msf_path, market_chars_monthly_path, acc_chars_world_path, output_path):
    a = pl.scan_parquet(msf_path)
    b = pl.scan_parquet(market_chars_monthly_path)
    c = pl.scan_parquet(acc_chars_world_path)
    world_data_prelim = (a.join(b, how = 'left', on = ['id','eom'])
                          .join(c, how = 'left', left_on = ['gvkey','eom'], right_on = ['gvkey', 'public_date'])
                          .drop(['div_tot', 'div_cash', 'div_spc', 'source']))
    world_data_prelim.collect().write_parquet(output_path)
    #Streaming can be used here if needed
    #world_data_prelim.collect(streaming = True).write_parquet('world_data_prelim.parquet')
def acc_chars_list():
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
    "sale_emp_gr1", "emp_gr1", "cash_at", "earnings_variability", "ni_ar1", "ni_ivol",
    # New Variables not in HXZ
    "niq_saleq_std", "ni_emp", "sale_emp", "ni_at", "ocf_at", "ocf_at_chg1",
    "roeq_be_std", "roe_be_std", "gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5"
    ]
    return acc_chars

@measure_time
def finish_daily_chars(output_path):
    bidask = pl.scan_parquet('corwin_schultz.parquet')
    r1 = pl.scan_parquet('roll_apply_daily.parquet').with_columns(col('id').cast(pl.Int64))
    daily_chars = bidask.join(r1, how = 'outer_coalesce', on=['id','eom'])
    daily_chars = daily_chars.with_columns(betabab_1260d = col('corr_1260d') * col('rvol_252d')/ col('__mktvol_252d'), rmax5_rvol_21d = col('rmax5_21d') / col('rvol_252d')).drop('__mktvol_252d')
    daily_chars.collect().write_parquet(output_path)

def z_ranks(data, var, min, sort):
    order = False if sort == 'ascending' else True
    c1 = pl.std('rank_aux').over(['excntry','eom']) != 0
    __subset = (data.select(['excntry','id','eom',var])
                    .with_columns(count = (col(var).is_not_null().sum()).over(['excntry','eom']))
                    .filter(col('count') > min).drop('count')
                    .with_columns(col(var).rank(descending = order).over(['excntry','eom']).alias('rank_aux'))
                    .filter(col('rank_aux').is_not_null())
                    .with_columns(zvar = pl.when(c1).then(((col('rank_aux') - pl.mean('rank_aux'))/pl.std('rank_aux')).over(['excntry','eom'])).otherwise(fl_none()))
                    .select(['excntry','id','eom', col('zvar').alias(f'z_{var}')]))
    return __subset

@measure_time
def quality_minus_junk(data_path, min_stks):
    z_vars = ['gp_at', 'ni_be', 'ni_at', 'ocf_at', 'gp_sale', 'oaccruals_at','gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5','betabab_1260d', 'debt_at', 'o_score', 'z_score', '__evol']
    direction = ['ascending', 'ascending', 'ascending', 'ascending', 'ascending', 'descending','ascending', 'ascending', 'ascending', 'ascending', 'ascending','descending', 'descending', 'descending', 'ascending', 'descending']
    cols = ['id', 'eom', 'excntry', 'gp_at', 'ni_be', 'ni_at', 'ocf_at', 'gp_sale', 'oaccruals_at', 'gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5', 'betabab_1260d', 'debt_at', 'o_score', 'z_score', 'roeq_be_std', 'roe_be_std']
    c1 = (col('common') == 1) & (col('primary_sec') == 1) & (col('obs_main') == 1) & (col('exch_main') == 1) & (col('ret_exc').is_not_null()) & (col('me').is_not_null())
    qmj = (pl.read_parquet(data_path, columns = cols + ['common', 'primary_sec', 'obs_main', 'exch_main', 'ret_exc', 'me'])
             .filter(c1)
             .select(cols)
             .with_columns(__evol = pl.coalesce(2 * col('roeq_be_std'), 'roe_be_std'))
             .sort(['excntry', 'eom']))
    for var_z,dir in zip(z_vars,direction):
        __z = z_ranks(qmj, var_z, min_stks, dir)
        qmj = qmj.join(__z, how = 'left', on = ['excntry','eom','id'])
    qmj = (qmj.with_columns(__prof   = pl.mean_horizontal('z_gp_at', 'z_ni_be', 'z_ni_at', 'z_ocf_at', 'z_gp_sale', 'z_oaccruals_at'),
                            __growth = pl.mean_horizontal('z_gpoa_ch5', 'z_roe_ch5', 'z_roa_ch5', 'z_cfoa_ch5', 'z_gmar_ch5'),
                            __safety = pl.mean_horizontal('z_betabab_1260d', 'z_debt_at', 'z_o_score', 'z_z_score', 'z___evol'))
              .select(['excntry', 'id', 'eom','__prof','__growth','__safety']))
    ranks = {i: z_ranks(qmj, f'__{i}'  , min_stks, 'ascending').rename({f'z___{i}'  : f'qmj_{i}'}) for i in ['prof', 'growth', 'safety']}
    qmj = (qmj.select(['excntry', 'id', 'eom'])
              .join(ranks['prof'], how = 'left', on = ['excntry','id','eom'])
              .join(ranks['growth'], how = 'left', on = ['excntry','id','eom'])
              .join(ranks['safety'], how = 'left', on = ['excntry','id','eom'])
              .with_columns(__qmj = col('qmj_prof') + col('qmj_growth') + col('qmj_safety')))
    __qmj = z_ranks(qmj, '__qmj', min_stks, 'ascending').rename({'z___qmj': 'qmj'})
    qmj = qmj.join(__qmj, how = 'left', on = ['excntry', 'id', 'eom']).drop('__qmj')
    qmj.write_parquet('qmj.parquet')

@measure_time
def save_main_data(end_date):
    months_exp = (col('eom').dt.year() * 12 + col('eom').dt.month()).cast(pl.Int64)
    data = (pl.scan_parquet('world_data.parquet')
              .with_columns(dif_aux = months_exp)
              .sort(['id','eom'])
              .with_columns(me_lag1 = col('me').shift(1).over('id'),
                            dif_aux = (col('dif_aux') - col('dif_aux').shift(1)).over('id'))
              .with_columns(me_lag1 = pl.when(col('dif_aux') == 1).then(col('me_lag1')).otherwise(fl_none()))
              .drop('dif_aux')
              .filter((col('primary_sec') == 1) & (col('common') == 1) & (col('obs_main') == 1) & (col('exch_main') == 1) & (col('eom') <= end_date)))
    data.select(pl.all().shrink_dtype()).collect(streaming=True).write_parquet('world_data_filtered.parquet')
    countries = pl.scan_parquet('world_data_filtered.parquet').select('excntry').unique().collect().to_numpy().flatten()
    for i in countries:
        print(f'Filtering data for country {i}', flush=True)
        data = pl.scan_parquet('world_data_filtered.parquet').filter(col('excntry') == i)
        data.collect().write_parquet(f'Characteristics/{i}.parquet', compression='zstd', compression_level = 11, statistics = False)

@measure_time
def save_output_files():
    os.system('mv market_returns.parquet Output/')
    os.system('mv market_returns_daily.parquet Output/')
    os.system('mv nyse_cutoffs.parquet Output/')
    os.system('mv return_cutoffs.parquet Output/')
    os.system('mv return_cutoffs_daily.parquet Output/')

@measure_time
def save_daily_ret():
    data = pl.scan_parquet('world_dsf.parquet').select(['excntry', 'id', 'date', 'me', 'ret', 'ret_exc'])
    countries = pl.scan_parquet('world_dsf.parquet').select('excntry').unique().collect().to_numpy().flatten()
    for i in countries:
        if i == None:
            print(f'Filtering data for null country', flush=True)
            data.select(pl.all().shrink_dtype()).filter(col('excntry').is_null()).collect().write_parquet(f'Daily_Returns/null_country.parquet', compression='zstd', compression_level = 11, statistics = False)
        else:
            print(f'Filtering data for country {i}', flush=True)
            data.select(pl.all().shrink_dtype()).filter(col('excntry') == i).collect().write_parquet(f'Daily_Returns/{i}.parquet', compression='zstd', compression_level = 11, statistics = False)
@measure_time
def save_accounting_data():
    pl.scan_parquet('acc_std_qtr.parquet').filter(col('source').is_not_null()).collect().write_parquet('Accounting_Data/Quarterly.parquet')
    pl.scan_parquet('acc_std_ann.parquet').filter(col('source').is_not_null()).collect().write_parquet('Accounting_Data/Annual.parquet')

@measure_time
def save_full_files_and_cleanup():
    pl.scan_parquet('world_dsf.parquet').select(pl.all().shrink_dtype()).collect(streaming = True).write_parquet(f'Daily_Returns/world_dsf.parquet', compression='zstd', compression_level = 11, statistics = False)
    pl.scan_parquet('world_data.parquet').select(pl.all().shrink_dtype()).collect(streaming = True).write_parquet(f'World_Data/world_data.parquet', compression='zstd', compression_level = 11, statistics = False)
    pl.scan_parquet('world_data_filtered.parquet').select(pl.all().shrink_dtype()).collect(streaming = True).write_parquet(f'World_Data/world_data_filtered.parquet', compression='zstd', compression_level = 11, statistics = False)
    os.system('rm *.parquet')
    os.system('rm -rf Raw_tables')
    os.system('rm -rf Raw_data_dfs')

@measure_time
def save_monthly_ret():
    data = pl.scan_parquet('world_msf.parquet').select(['excntry', 'id', 'source_crsp', 'eom', 'me', 'ret_exc', 'ret', 'ret_local'])
    data.select(pl.all().shrink_dtype()).collect().write_parquet(f'World_Ret_Monthly/world_ret_monthly.parquet', compression='zstd', compression_level = 11, statistics = False)

@measure_time
def merge_roll_apply_daily_results():
    date_idx = datetime.datetime.today().month + datetime.datetime.today().year * 12
    df_dates = pl.DataFrame({'aux_date': [i+1 for i in range(23112, date_idx+1)],'eom': [f'{i//12}-{i%12+1}-1' for i in range(23112, date_idx+1)]})
    df_dates = df_dates.with_columns(col('eom').str.strptime(pl.Date, "%Y-%m-%d").dt.month_end().alias('eom'), col('aux_date').cast(pl.Int64))
    df_id    = pl.scan_parquet('id_int_key.parquet')
    file_paths = [i for i in os.listdir() if i.startswith('__roll')]
    if len(file_paths) != 1:
        joint_file = pl.scan_parquet(file_paths[0])
        for i in file_paths[1:]:
            df_aux = pl.scan_parquet(i)
            joint_file = joint_file.join(df_aux, how = 'outer_coalesce', on = ['id_int','aux_date'])
        joint_file.with_columns(col('aux_date').cast(pl.Int64))\
                  .join(df_dates.lazy(), how = 'left', on = 'aux_date')\
                  .join(df_id, how = 'left', on = 'id_int')\
                  .drop(['aux_date', 'id_int'])\
                  .collect()\
                  .write_parquet('roll_apply_daily.parquet')
    else: joint_file = pl.scan_parquet(file_paths[0])

    joint_file.with_columns(col('aux_date').cast(pl.Int64))\
              .join(df_dates.lazy().with_columns(col('aux_date').cast(pl.Int64)), how = 'left', on = 'aux_date')\
              .join(df_id, how = 'left', on = 'id_int')\
              .drop(['aux_date', 'id_int'])\
              .collect()\
              .write_parquet('roll_apply_daily.parquet')

@measure_time
def merge_world_data_prelim():
    a = pl.scan_parquet('world_data_prelim.parquet')
    b = pl.scan_parquet('beta_60m.parquet')
    c = pl.scan_parquet('resmom_ff3_12_1.parquet')
    d = pl.scan_parquet('resmom_ff3_6_1.parquet')
    e = pl.scan_parquet('mp_factors.parquet')
    f = pl.scan_parquet('market_chars_d.parquet')
    g = pl.scan_parquet('firm_age.parquet').select(['id','eom','age'])
    world_data = (a.join(b, how = 'left', on = ['id','eom'])
                   .join(c, how = 'left', on = ['id','eom'])
                   .join(d, how = 'left', on = ['id','eom'])
                   .join(e, how = 'left', on = ['id','eom'])
                   .join(f, how = 'left', on = ['id','eom'])
                   .join(g, how = 'left', on = ['id','eom']))
    world_data.collect().write_parquet('world_data_-1.parquet')

@measure_time
def merge_qmj_to_world_data():
    a = pl.scan_parquet('world_data_-1.parquet')
    b = pl.scan_parquet('qmj.parquet')
    result = (a.join(b, how = 'left', on = ['excntry','id','eom'])
               .unique(['id','eom'])
               .sort(['id','eom']))
    result.collect(streaming=True).write_parquet('world_data.parquet')

@measure_time
def merge_industry_to_world_msf():
    __msf_world = pl.scan_parquet('__msf_world.parquet')
    comp_ind = pl.scan_parquet('comp_ind.parquet')
    crsp_ind = pl.scan_parquet('crsp_ind.parquet').rename({'sic': 'sic_crsp', 'naics': 'naics_crsp'})
    __msf_world = (__msf_world.join(comp_ind, how = 'left', left_on = ['gvkey', 'eom'], right_on = ['gvkey','date'])
                              .join(crsp_ind, how = 'left', left_on = ['permco', 'permno', 'eom'], right_on = ['permco','permno','date'])
                              .with_columns(sic   = pl.coalesce(['sic','sic_crsp']),
                                            naics = pl.coalesce(['naics','naics_crsp']))
                              .drop(['sic_crsp','naics_crsp']))
    __msf_world.collect(streaming = True).write_parquet('__msf_world2.parquet')

@measure_time
def roll_apply_daily(stats, sfx, __min):
    print(f"Processing {stats} - {sfx.replace('_', '')} - {__min}", flush=True)
    aux_maps = gen_aux_maps(sfx)
    base_data = prepare_base_data(stat = stats)
    results = pl.concat([process_map_chunks(base_data, mapping, stats, sfx, __min) for mapping in aux_maps])
    results.collect().write_parquet(f'__roll{sfx}_{stats}.parquet')

def gen_consecutive_lists(input_list, k): return [input_list[i:i+k] for i in range(0, len(input_list), k) if len(input_list[i:i+k]) == k]
    
def build_groups(input_list, k): return [gen_consecutive_lists(input_list[offset:], k) for offset in range(k)]
    
def group_mapping_dfs(input_list, k):
    groups =  build_groups(input_list, k)
    dfs = [pl.DataFrame({'aux_date': group})
             .with_columns(group_number = pl.cum_count('aux_date'), 
                           new_date     = col('aux_date').list.max())
              for group in groups]
    return [{'group_map': df.explode('aux_date').select([col('aux_date').cast(pl.Int32), 'group_number']).lazy(), 
             'date_map' : df.select(['group_number', col('new_date').alias('aux_date')]).unique().sort(['group_number']).lazy()}for df in dfs]

def base_data_filter_exp(stat):
    if   stat == 'zero_trades': return col('tvol').is_not_null()
    elif stat == 'dolvol'     : return col('dolvol_d').is_not_null()
    elif stat == 'turnover'   : return col('tvol').is_not_null()
    elif stat == 'mktcorr'    : return (col('ret_exc_3l').is_not_null()) & (col('zero_obs') < 10)
    else                      : return (col('ret_exc').is_not_null()) & (col('zero_obs') < 10)

def prepare_base_data(stat):
    base_data_path = 'corr_data.parquet' if stat == 'mktcorr' else 'dsf1.parquet'
    base_data = (pl.scan_parquet(base_data_path)
                .with_columns(aux_date = gen_MMYY_column('eom'))
                .filter(base_data_filter_exp(stat)))

    if stat == 'dimsonbeta':
        lead_lag = pl.scan_parquet('mkt_lead_lag.parquet').drop(['eom', 'mktrf'])
        base_data = base_data.join(lead_lag, how = 'inner', on = ['excntry', 'date'])
        
    return base_data

def apply_group_filter(df, stat, min_obs):
    if stat == 'turnover' or stat == 'mktcorr': 
        pass
    elif stat == 'dimsonbeta':
        df = (df.with_columns(n1 = pl.len().over(['id_int','eom']),
                            n2 = pl.count('ret_exc').over(['id_int', 'group_number']))
                .filter((col('n1') >= min_obs - 1) & (col('n2') >= min_obs) & (col('mktrf_lg1').is_not_null()) & (col('mktrf_ld1').is_not_null())))
    else:
        if stat == 'zero_trades': filter_var = 'tvol' 
        elif stat == 'dolvol':  filter_var = 'dolvol_d'
        else: filter_var = 'ret_exc'
        df = (df.with_columns(n = pl.count(filter_var).over(['id_int', 'group_number']))
                .filter(col('n') >= min_obs))
    return df

def process_map_chunks(base_data, mapping, stats, sfx, __min):

    funcs = {'rvol'       : rvol, 
             'rmax'       : rmax, 
             'skew'       : skew, 
             'prc_to_high': prc_to_high, 
             'capm'       : capm, 
             'ami'        : ami, 
             'downbeta'   : downbeta, 
             'mktrf_vol'  : mktrf_vol, 
             'capm_ext'   : capm_ext, 
             'ff3'        : ff3, 
             'hxz4'       : hxz4, 
             'zero_trades': zero_trades, 
             'dolvol'     : dolvol, 
             'turnover'   : turnover, 
             'mktcorr'    : mktcorr, 
             'mktvol'     : mktrf_vol,
             'dimsonbeta' : dimsonbeta}
    
    df = (base_data.join(mapping['group_map'], how = 'inner', on = 'aux_date')
                .pipe(apply_group_filter, stat = stats, min_obs = __min)
                .pipe(funcs[stats], sfx = sfx, __min = __min)
                .join(mapping['date_map'], how = 'left', on = 'group_number')
                .drop('group_number'))
    
    return df
    
def gen_aux_maps(sfx):
    parameter_mapping = {"_21d": 1,"_126d": 6,"_252d": 12,"_1260d": 60}
    date_aux = datetime.datetime.today().month + datetime.datetime.today().year * 12
    date_idx = [i for i in range(23113 - parameter_mapping[sfx], date_aux+1)]
    aux_maps = group_mapping_dfs(date_idx, parameter_mapping[sfx])
    return aux_maps

def rvol(df, sfx, __min):
    df = (df.group_by(['id_int', 'group_number'])
            .agg(col('ret_exc').std().alias(f'rvol{sfx}')))
    return df

def rmax(df, sfx, __min):
    df = (df.group_by(['id_int', 'group_number'])
            .agg([col('ret').top_k(5).mean().alias(f'rmax5{sfx}'), 
                  col('ret').max().alias(f'rmax1{sfx}')]))
    return df

def skew(df, sfx, __min):
    df = (df.group_by(['id_int', 'group_number'])
            .agg(col('ret_exc').skew(bias = False).alias(f'rskew{sfx}')))
    return df

def prc_to_high(df, sfx, __min):
    df = (df.sort(['id_int', 'date'])
            .group_by(['id_int', 'group_number'])
            .agg([(col('prc_adj').last()/ col('prc_adj').max()).alias(f'prc_highprc{sfx}'), 
                pl.count('prc_adj').alias('n')])
            .filter(col('n') >= __min)
            .drop('n'))
    return df
def capm(df, sfx, __min):
    df = (df.group_by(['id_int', 'group_number'])
            .agg([(pl.cov('ret_exc', 'mktrf')/pl.var('mktrf')).alias(f'beta{sfx}'),
                (col('ret_exc') - col('mktrf') * (pl.cov('ret_exc', 'mktrf')/pl.var('mktrf'))).std().alias(f'ivol_capm{sfx}')]))
    return df

def ami(df, sfx, __min):
    aux_1 = pl.when(col('dolvol_d') == 0).then(fl_none()).otherwise(col('dolvol_d'))
    df = (df.group_by(['id_int', 'group_number'])
            .agg([(col('ret').abs()/aux_1 * 1e6).mean().alias(f'ami{sfx}'), 
                  pl.count('dolvol_d').alias('n')])
            .filter(col('n') >= __min)
            .drop('n'))
    return df

def downbeta(df, sfx, __min):
    df = (df.filter(col('mktrf') < 0)
            .group_by(['id_int', 'group_number'])
            .agg([(pl.cov('ret_exc', 'mktrf')/pl.var('mktrf')).alias(f'betadown{sfx}'), 
                pl.count('ret_exc').alias('n')])
            .filter(col('n')>=__min/2)
            .drop('n'))
    return df

def mktrf_vol(df, sfx, __min):
    df = (df.group_by(['id_int', 'group_number'])
            .agg(col('mktrf').std().alias(f'__mktvol{sfx}')))
    return df

def capm_ext(df, sfx, __min):
    beta_col     = (pl.cov('ret_exc', 'mktrf')/pl.var('mktrf'))
    alpha_col    = pl.mean('ret_exc') - beta_col * pl.mean('mktrf')
    residual_col = (col('ret_exc') - (alpha_col + col('mktrf') * beta_col))
    exp_mkt      =  (col('mktrf') - col('mktrf').mean())
    exp_coskew1  = (residual_col * (exp_mkt**2)).mean()
    exp_coskew2  = (residual_col**2).mean()**0.5 * (exp_mkt**2).mean()

    df = (df.group_by(['id_int', 'group_number'])
            .agg([beta_col.cast(pl.Float64).alias(f'beta_{sfx}'),
                  residual_col.std().alias(f'ivol_capm{sfx}'),
                  residual_col.skew(bias = False).alias(f'iskew_capm{sfx}'), 
                  (exp_coskew1/exp_coskew2).alias(f'coskew{sfx}')]))
    return df

def ff3(df, sfx, __min):
    res_exp = pl.col('ret_exc').least_squares.ols('mktrf', 'smb_ff', 'hml', add_intercept = True, mode = 'residuals')
    df = (df.filter(col('smb_ff').is_not_null() & col('hml').is_not_null())
            .group_by(['id_int', 'group_number'])
            .agg(res_exp.std(ddof = 3).alias(f'ivol_ff3{sfx}'),
                 res_exp.skew(bias = False).alias(f'iskew_ff3{sfx}')))
    return df

def hxz4(df, sfx, __min):
    res_exp = pl.col('ret_exc').least_squares.ols('mktrf', 'smb_hxz', 'roe', 'inv', add_intercept = True, mode = 'residuals')
    df = (df.filter(col('smb_hxz').is_not_null() & col('roe').is_not_null() & col('inv').is_not_null())
            .group_by(['id_int', 'group_number'])
            .agg(res_exp.std(ddof = 4).alias(f'ivol_hxz4{sfx}'), 
                res_exp.skew(bias = False).alias(f'iskew_hxz4{sfx}')))
    return df

def zero_trades(df, sfx, __min):
    aux_1 = (pl.col('tvol') == 0).mean() * 21
    aux_2 = pl.when(pl.col('shares') != 0).then(pl.col('tvol')/(pl.col('shares')*1e6)).otherwise(pl.lit(None).cast(pl.Float64))
    aux_3 = (pl.col('turnover').rank(descending=True, method='average')/pl.count('turnover')).over('group_number')
    aux_4 = (aux_3 / 100) + pl.col('zero_trades')
    df = (df.group_by(['id_int', 'group_number'])
            .agg([aux_1.alias('zero_trades'),
                aux_2.alias('turnover')])
            .filter(pl.col('zero_trades').is_not_null() & pl.col('turnover').is_not_null())
            .with_columns(pl.col('turnover').list.mean())
            .with_columns(aux_4.alias(f'zero_trades{sfx}'))
            .select(['id_int', 'group_number', f'zero_trades{sfx}']))
    return df

def dolvol(df, sfx, __min):
    df = (df.group_by(['id_int', 'group_number'])
            .agg([col('dolvol_d').mean().alias(f'dolvol{sfx}'),
                pl.when(col('dolvol_d').mean() != 0)
                    .then(col('dolvol_d').std()/col('dolvol_d').mean())
                    .otherwise(fl_none()).alias(f'dolvol_var{sfx}')]))
    return df

def turnover(df, sfx, __min):
    aux_1 = pl.when(col('turnover_d').list.mean() != 0).then(col('turnover_d').list.std()/col('turnover_d').list.mean()).otherwise(fl_none())
    df = (df.group_by(['id_int', 'group_number'])
            .agg([pl.when(col('shares') != 0).then(col('tvol')/(col('shares')*1e6)).otherwise(fl_none()).alias('turnover_d')])
            .with_columns([col('turnover_d').list.mean().alias(f'turnover{sfx}'), 
                        aux_1.alias(f'turnover_var{sfx}'), 
                        (col('turnover_d').list.len()).alias('n')])
            .filter(col('n') >= __min)
            .drop(['n', 'turnover_d']))
    return df

def mktcorr(df, sfx, __min):
    df = (df.group_by(['id_int', 'group_number'])
            .agg([pl.count('ret_exc_3l').alias('n1'), 
                pl.count('mkt_exc_3l').alias('n2'), 
                pl.corr('ret_exc_3l', 'mkt_exc_3l').alias(f'corr{sfx}')])
            .filter((col('n1')>=__min) & (col('n2')>=__min))
            .drop(['n1', 'n2']))
    return df

def dimsonbeta(df, sfx, __min):
    b1_col, b2_col, b3_col = regression_3vars_total('ret_exc', 'mktrf', 'mktrf_ld1', 'mktrf_lg1')
    df = (df.group_by(['id_int', 'group_number'])
            .agg((b1_col + b2_col + b3_col).alias(f'beta_dimson{sfx}')))
    return df
