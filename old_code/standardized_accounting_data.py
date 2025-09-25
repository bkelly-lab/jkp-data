import polars as pl
import numpy as np
import polars_ols as pls
import time
import datetime
import ibis
from ibis import _
import os
from datetime import date
from math import sqrt, exp
from functools import reduce
from polars import col
import math
import polars_readstat

#helper functions:
def quarterize(df: pl.DataFrame, var_list: list[str]) -> pl.DataFrame:
    df = df.sort(['gvkey', 'fyr', 'fyearq', 'fqtr', 'source']) \
           .unique(subset=['gvkey', 'fyr', 'fyearq', 'fqtr'], keep='first') \
           .sort(['gvkey', 'fyr', 'fyearq', 'fqtr'])

    for var in var_list:
        lag_val = pl.col(var).shift(1).over(['gvkey', 'fyr', 'fyearq'])
        lag_fqtr = pl.col("fqtr").shift(1).over(['gvkey', 'fyr', 'fyearq'])
        first_fqtr = pl.col("fqtr") == 1
        valid_diff = (pl.col("fqtr") - lag_fqtr) == 1

        df = df.with_columns(
            pl.when(first_fqtr)
              .then(pl.col(var))
              .when(valid_diff)
              .then(pl.col(var) - lag_val)
              .otherwise(None)
              .alias(var + "_q")
        )

    return df




def add_helper_vars(data):
    
    __comp_dates1 = data.select(['gvkey', 'curcd', 'datadate']).group_by(
    ["gvkey", "curcd"]).agg(
    pl.col("datadate").min().alias('start_date'),
    pl.col("datadate").max().alias('end_date'))

    __comp_dates2 = expand(data=__comp_dates1, id_vars=['gvkey'], start_date='start_date', end_date='end_date', freq='month', new_date_name='datadate')

    
    temp_data = data.with_columns(pl.lit(1).cast(pl.Float64).alias('data_available'))
    __helpers1 = __comp_dates2.join(temp_data, left_on=["gvkey", "curcd", "datadate"], right_on=["gvkey", "curcd", "datadate"], how="left").with_columns(pl.col("data_available").fill_null(strategy="zero")).select(temp_data.columns)
    __helpers1 = __helpers1.sort(["gvkey", "curcd", "datadate", 'source']).unique(["gvkey", "curcd", "datadate"], keep='last') 


    __helpers2  = __helpers1.sort(["gvkey","curcd","datadate"]).with_columns(
        pl.col("datadate")
          .rank(method="ordinal")
          .over(["gvkey","curcd"])
          .cast(pl.Int32)
          .alias("count")
    )
    output = __helpers2.clone()
    var_pos= ['at', 'sale', 'revt', 'dv', 'che']
    for var in var_pos:
        output = output.with_columns(pl.when(pl.col(var)<0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(var)).alias(var))

    data_with_helper_variables = output.clone()


    for i in ['ebit', 'ebitda', 'pstkrv', 'pstkl', 'itcb','gp', 'dltis', 'do']:
        if i not in data_with_helper_variables.columns:     
            data_with_helper_variables=data_with_helper_variables.with_columns([
                pl.lit(None).cast(pl.Float64).alias(i)       
            ])

    data_with_helper_variables = (data_with_helper_variables
    .with_columns(pl.coalesce([pl.col("sale"), pl.col("revt")]).alias("sale_x"))
    .with_columns(pl.coalesce([pl.col("gp"), pl.col("sale_x") - pl.col("cogs")]).alias("gp_x"))
    .with_columns(pl.coalesce([pl.col("xopr"), pl.col("cogs") + pl.col("xsga")]).alias("opex_x"))
    .with_columns(pl.coalesce([pl.col("ebitda"), pl.col("oibdp"), pl.col("sale_x") - pl.col("opex_x"), pl.col("gp_x") - pl.col("xsga")]).alias("ebitda_x"))
    .with_columns(pl.coalesce([pl.col("ebit"), pl.col("oiadp"), pl.col("ebitda_x") - pl.col("dp")]).alias("ebit_x"))
    .with_columns((pl.col("ebitda_x") + pl.coalesce([pl.col("xrd"), pl.lit(0)])).alias("op_x"))
    .with_columns((pl.col("ebitda_x") - pl.col("xint")).alias("ope_x"))
    .with_columns(pl.coalesce([pl.col("pi"), (pl.col("ebit_x") - pl.col("xint") + pl.coalesce([pl.col("spi"), pl.lit(0)]) + pl.coalesce([pl.col("nopi"), pl.lit(0)]))]).alias("pi_x"))
    .with_columns(pl.coalesce([pl.col("xido"), (pl.col("xi") + pl.coalesce([pl.col("do"), pl.lit(0)]))]).alias("xido_x"))
    .with_columns(pl.coalesce([pl.col("ib"), (pl.col("ni") - pl.col("xido_x")), (pl.col("pi_x") - pl.col("txt") - pl.coalesce([pl.col("mii"), pl.lit(0)]))]).alias("ni_x"))
    .with_columns(pl.coalesce([pl.col("ni"), (pl.col("ni_x") + pl.coalesce([pl.col("xido_x"), pl.lit(0)])), (pl.col("ni_x") + pl.col("xi") + pl.col("do"))]).alias("nix_x"))
    .with_columns((pl.col("nix_x") + pl.col("xint")).alias("fi_x"))
    .with_columns(pl.coalesce([pl.col("dvt"), pl.col("dv")]).alias("div_x"))
)


    data_with_helper_variables = (data_with_helper_variables
  .with_columns(
    pl.when(
        pl.col("prstkc").is_null() & pl.col("purtshr").is_null()
    ).then(
        None
    ).otherwise(
        pl.sum_horizontal(["prstkc", "purtshr"])
    ).alias("eqbb_x")
)
    .with_columns(pl.col("sstk").alias("eqis_x"))
    .with_columns(
    pl.when(
        pl.col("eqis_x").is_null() & pl.col("eqbb_x").is_null()
    ).then(
        None
    ).otherwise(
        pl.sum_horizontal([
            pl.col("eqis_x"),
            (pl.col("eqbb_x") * -1)
        ])
    ).alias("eqnetis_x")
)
    .with_columns((pl.col("div_x") + pl.col("eqbb_x")).alias("eqpo_x"))
    .with_columns((pl.col("div_x") - pl.col("eqnetis_x")).alias("eqnpo_x"))
    # .with_columns(pl.when((pl.col("dltis").is_null()) & (pl.col("dltr").is_null()) & (pl.col("ltdch").is_null()) & (pl.col("count") <= 12)).then(pl.lit(None))
    #               .otherwise(pl.coalesce([(pl.col("dltis") - pl.col("dltr")), pl.col("ltdch"), (pl.col("dltt") - pl.col("dltt").shift(12))])).alias("dltnetis_x"))
    .sort(["gvkey", "curcd", "datadate"]).with_columns(
    pl.coalesce([
        # sum(dltis,-dltr), returns null if both missing
        pl.when(
            pl.col("dltis").is_null() & pl.col("dltr").is_null()
        ).then(None).otherwise(
            pl.sum_horizontal([pl.col("dltis"), (pl.col("dltr") * -1)])
        ),
        # fallback: ltdch
        pl.col("ltdch"),
        # fallback: 12‑month change in dltt
        pl.col("dltt") - pl.col("dltt").shift(12).over(["gvkey","curcd"])
    ]).alias("dltnetis_x")
).with_columns(
    pl.when(
        pl.col("dltis").is_null()
        & pl.col("dltr").is_null()
        & pl.col("ltdch").is_null()
        & (pl.col("count") <= 12)
    ).then(None).otherwise(pl.col("dltnetis_x")).alias("dltnetis_x")
)
    .with_columns(
    pl.coalesce([
        pl.col("dlcch"),
        # 12‑period change in dlc
        pl.col("dlc") - pl.col("dlc").shift(12).over(["gvkey","curcd"])
    ]).alias("dstnetis_x")
).with_columns(
    pl.when(
        pl.col("dlcch").is_null() & (pl.col("count") <= 12)
    ).then(
        None
    ).otherwise(
        pl.col("dstnetis_x")
    ).alias("dstnetis_x")
)

    .with_columns(
    pl.when(
        pl.col("dstnetis_x").is_null() & pl.col("dltnetis_x").is_null()
    ).then(
        None
    ).otherwise(
        pl.sum_horizontal(["dstnetis_x", "dltnetis_x"])
    ).alias("dbnetis_x")
)
    .with_columns((pl.col("eqnetis_x") + pl.col("dbnetis_x")).alias("netis_x"))
    .with_columns(pl.coalesce([pl.col("fincf"), (pl.col("netis_x") - pl.col("dv") + pl.coalesce([pl.col("fiao"), pl.lit(0)]) + pl.coalesce([pl.col("txbcof"), pl.lit(0)]))]).alias("fincf_x"))
)


    data_with_helper_variables = (data_with_helper_variables
    .with_columns(
    pl.when(
        pl.col("dltt").is_null() & pl.col("dlc").is_null()
    ).then(
        None
    ).otherwise(
        pl.sum_horizontal(["dltt", "dlc"])
    ).alias("debt_x")
)
    .with_columns(pl.coalesce([pl.col("pstkrv"), pl.col("pstkl"), pl.col("pstk")]).alias("pstk_x"))
    .with_columns(
    pl.coalesce([
        pl.col("seq"),
        # ceq + pstk_x (treat missing pstk_x as 0)
        pl.col("ceq") + pl.coalesce([pl.col("pstk_x"), pl.lit(0)]),
        # fallback to at – lt
        pl.col("at") - pl.col("lt"),
    ]).alias("seq_x")
)
                                  
    .with_columns(pl.coalesce([pl.col("at"), pl.col("seq_x") + pl.col("dltt") + pl.coalesce(pl.col("lct"), pl.lit(0)) + pl.coalesce(pl.col("lo"), pl.lit(0)) + pl.coalesce(pl.col("txditc"), pl.lit(0))]).alias("at_x"))
    .with_columns(pl.coalesce([pl.col("act"), pl.col("rect") + pl.col("invt") + pl.col("che") + pl.col("aco")]).alias("ca_x"))
    .with_columns(pl.coalesce([pl.col("lct"), pl.col("ap") + pl.col("dlc") + pl.col("txp") + pl.col("lco")]).alias("cl_x"))
    .with_columns((pl.col("at_x") - pl.col("ca_x")).alias("nca_x"))
    .with_columns((pl.col("lt") - pl.col("cl_x")).alias("ncl_x"))
    .with_columns((pl.col("debt_x") - pl.coalesce([pl.col("che"), pl.lit(0)])).alias("netdebt_x"))
    # .with_columns(pl.coalesce([pl.col("txditc"), pl.col("txdb") + pl.col("itcb")]).alias("txditc_x"))
     .with_columns(
    pl.coalesce([
        # 1) if txditc exists, use it
        pl.col("txditc"),
        # 2) otherwise, sum txdb+itcb, but return None if both are null
        pl.when(
            pl.col("txdb").is_null() & pl.col("itcb").is_null()
        ).then(
            None
        ).otherwise(
            pl.sum_horizontal(["txdb", "itcb"])
        )
    ]).alias("txditc_x")
)
    .with_columns(
    (
        pl.col("seq_x")
        + pl.coalesce([pl.col("txditc_x"), pl.lit(0)])
        - pl.coalesce([pl.col("pstk_x"), pl.lit(0)])
    ).alias("be_x")
)
    .with_columns(pl.coalesce([pl.col("icapt") + pl.coalesce(pl.col("dlc"), pl.lit(0)) - pl.coalesce(pl.col("che"), pl.lit(0)), pl.col("netdebt_x") + pl.col("seq_x") + pl.coalesce(pl.col("mib"), pl.lit(0))]).alias("bev_x"))
    .with_columns((pl.col("ca_x") - pl.col("che")).alias("coa_x"))
    .with_columns((pl.col("cl_x") - pl.coalesce(pl.col("dlc"), pl.lit(0))).alias("col_x"))
    .with_columns((pl.col("coa_x") - pl.col("col_x")).alias("cowc_x"))
    .with_columns((pl.col("at_x") - pl.col("ca_x") - pl.coalesce(pl.col("ivao"), pl.lit(0))).alias("ncoa_x"))
    .with_columns((pl.col("lt") - pl.col("cl_x") - pl.col("dltt")).alias("ncol_x"))
    .with_columns((pl.col("ncoa_x") - pl.col("ncol_x")).alias("nncoa_x"))
    .with_columns((pl.coalesce(pl.col("ivst"), pl.lit(0)) + pl.coalesce(pl.col("ivao"), pl.lit(0))).alias("fna_x"))
    .with_columns((pl.col("debt_x") + pl.coalesce(pl.col("pstk_x"), pl.lit(0))).alias("fnl_x"))
    .with_columns((pl.col("fna_x") - pl.col("fnl_x")).alias("nfna_x"))
    .with_columns((pl.col("coa_x") + pl.col("ncoa_x")).alias("oa_x"))
    .with_columns((pl.col("col_x") + pl.col("ncol_x")).alias("ol_x"))
    .with_columns((pl.col("oa_x") - pl.col("ol_x")).alias("noa_x"))
    .with_columns((pl.col("ppent") + pl.col("intan") + pl.col("ao") - pl.col("lo") + pl.col("dp")).alias("lnoa_x"))
    .with_columns(pl.coalesce([pl.col("ca_x") - pl.col("invt"), pl.col("che") + pl.col("rect")]).alias("caliq_x"))
    .with_columns((pl.col("ca_x") - pl.col("cl_x")).alias("nwc_x"))
    .with_columns((pl.col("ppegt") + pl.col("invt")).alias("ppeinv_x"))
    .with_columns((pl.col("che") + 0.75 * pl.col("coa_x") + 0.5 * (pl.col("at_x") - pl.col("ca_x") - pl.coalesce(pl.col("intan"), 0))).alias("aliq_x")))

    var_bs= ['be_x', 'bev_x']
    for var in var_bs:
        data_with_helper_variables = data_with_helper_variables.with_columns(pl.when(pl.col(var)<0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(var)).alias(var))

    data_with_helper_variables = (data_with_helper_variables
    # .with_columns(pl.when(pl.col("count") <= 12).then(None).otherwise(pl.coalesce([pl.col("ni_x") - pl.col("oancf"), pl.col("cowc_x") - pl.col("cowc_x").shift(12) + pl.col("nncoa_x") - pl.col("nncoa_x").shift(12)])).alias("oacc_x"))
    .with_columns(
    pl.coalesce([
        # candidate 1: earnings accrual
        pl.col("ni_x") - pl.col("oancf"),
        # candidate 2: change in working capital + change in non-current operating assets
        (pl.col("cowc_x") - pl.col("cowc_x").shift(12).over(["gvkey","curcd"]))
        + (pl.col("nncoa_x") - pl.col("nncoa_x").shift(12).over(["gvkey","curcd"]))
    ]).alias("oacc_temp")
).with_columns(
    pl.when(pl.col("count") <= 12)
      .then(None)
      .otherwise(pl.col("oacc_temp"))
      .alias("oacc_x")
).drop("oacc_temp")
    .with_columns(pl.when(pl.col("count") <= 12).then(None).otherwise(pl.col("oacc_x") + pl.col("nfna_x") - pl.col("nfna_x").shift(12)).alias("tacc_x"))
    .with_columns(pl.coalesce([pl.col("oancf"), pl.col("ni_x") - pl.col("oacc_x"), pl.col("ni_x") + pl.col("dp") - pl.coalesce([pl.col("wcapt"), pl.lit(0)])]).alias("ocf_x"))
    .with_columns((pl.col("ocf_x") - pl.col("capx")).alias("fcf_x"))
    .with_columns((pl.col("ebitda_x") + pl.coalesce([pl.col("xrd"), pl.lit(0)]) - pl.col("oacc_x")).alias("cop_x"))
    )

    data_with_helper_variables = data_with_helper_variables#.drop('count')

    return data_with_helper_variables


def expand(data, id_vars, start_date, end_date, freq='day', new_date_name='date'):
    if freq =='day':
        __expanded = data.with_columns(pl.date_ranges(start=start_date, end=end_date, interval='1d')).rename({"date_range": new_date_name}).explode(new_date_name).drop([start_date, end_date])
    elif freq == 'month':
         __expanded = data.with_columns(pl.date_ranges(start=start_date, end=end_date, interval='1mo').alias('date_range')).rename({"date_range": new_date_name}).explode(new_date_name).with_columns(pl.col(new_date_name).dt.month_end()).drop([start_date, end_date])


    __expanded = __expanded.sort(id_vars + [new_date_name]).unique(id_vars + [new_date_name])
    return __expanded


#function:
def standardized_accounting_data(coverage, convert_to_usd, me_data_path, include_helpers_vars, start_date):


    g_fundq_cols = pl.scan_parquet('Raw_tables/comp_g_fundq.parquet').collect_schema().names()
    fundq_cols   = pl.scan_parquet('Raw_tables/comp_fundq.parquet'  ).collect_schema().names()
    
    #Compustat Accounting Vars to Extract
    avars_inc = ['sale', 'revt', 'gp', 'ebitda', 'oibdp', 'ebit', 'oiadp', 'pi', 'ib', 'ni', 'mii','cogs', 'xsga', 'xopr', 'xrd', 'xad', 'xlr', 'dp', 'xi', 'do', 'xido', 'xint', 'spi', 'nopi', 'txt','dvt']
    avars_cf  = ['oancf', 'ibc', 'dpc', 'xidoc', 'capx', 'wcapt', # Operating
    'fincf', 'fiao', 'txbcof', 'ltdch', 'dltis', 'dltr', 'dlcch', 'purtshr', 'prstkc', 'sstk','dv', 'dvc'] # Financing
    avars_bs  = ['at', 'act', 'aco', 'che', 'invt', 'rect', 'ivao', 'ivst', 'ppent', 'ppegt', 'intan', 'ao', 'gdwl', 're', # Assets
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
                           .select(['gvkey', 'datadate', 'n', 'indfmt', 'curcd', 'source', 'ni'] +\
                                   [fl_none().alias(i) for i in ['gp', 'pstkrv', 'pstkl', 'itcb', 'xad', 'txbcof']] +\
                                   query_vars)
                           .pipe(apply_indfmt_filter))
        #Quarterly global data:
        vars_not_in_query = ['icaptq','niy','txditcq','txpq','xidoq','xidoy','xrdq','xrdy','txbcofy', 'niq', 'ppegtq', 'doq', 'doy']
        query_vars = [var for var in qvars if var not in vars_not_in_query]
        g_fundq = load_raw_fund_table_and_filter('Raw_tables/comp_g_fundq.parquet', start_date, 'GLOBAL', 1)
        __gfundq = (g_fundq.with_columns(niq    = (col('ibq') + pl.coalesce('xiq', 0.)).cast(pl.Float64),
                                         ppegtq = (col('ppentq') + col('dpactq')).cast(pl.Float64))
                           .select(['gvkey', 'datadate', 'n', 'indfmt', 'fyr', 'fyearq', 'fqtr', 'curcdq', 'source', 'niq', 'ppegtq'] +\
                                   [fl_none().alias(i) for i in ['icaptq', 'niy', 'txditcq', 'txpq', 'xidoq', 'xidoy', 'xrdq', 'xrdy', 'txbcofy']] +\
                                   query_vars)
                           .pipe(apply_indfmt_filter))
    
    
    
    if coverage in ['na', 'world']:
        #Annual north american data:
        vars_not_in_query = ['wcapt', 'ltdch', 'purtshr']
        query_vars = [var for var in (avars + avars_other) if var not in vars_not_in_query]
        funda = load_raw_fund_table_and_filter('Raw_tables/comp_funda.parquet', start_date, 'NA', 2)
        __funda = funda.select(['gvkey', 'datadate', 'n', 'curcd', 'source'] +\
                               [fl_none().alias(i) for i in ['wcapt', 'ltdch', 'purtshr']] +\
                               query_vars)
        #Quarterly north american data:
        vars_not_in_query = ['dvtq','gpq','dvty','gpy','ltdchy','purtshry','wcapty']
        query_vars = [var for var in qvars if var not in vars_not_in_query]
        fundq = load_raw_fund_table_and_filter('Raw_tables/comp_fundq.parquet', start_date, 'NA', 2)
        __fundq = fundq.select(['gvkey', 'datadate', 'n', 'fyr', 'fyearq', 'fqtr', 'curcdq', 'source'] +\
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
        __compa = add_fx_and_convert_vars(df=aname, fx_df=fx, vars=avars, freq ='annual')
        __compq = add_fx_and_convert_vars(qname, fx, qvars, 'quarterly')
    else: __compa, __compq = aname, qname
    
    
    __me_data = load_mkt_equity_data(me_data_path).collect()
    
    yrl_vars = ['cogsq', 'xsgaq', 'xintq', 'dpq', 'txtq', 'xrdq', 'dvq', 'spiq', 'saleq', 'revtq', 'xoprq', 'oibdpq', 'oiadpq', 'ibq', 'niq', 'xidoq', 'nopiq', 'miiq', 'piq', 'xiq','xidocq', 'capxq', 'oancfq', 'ibcq', 'dpcq', 'wcaptq','prstkcq', 'sstkq', 'purtshrq','dsq', 'dltrq', 'ltdchq', 'dlcchq','fincfq', 'fiaoq', 'txbcofq', 'dvtq']
    __compq =  quarterize(df=__compq, var_list=qvars_y).collect()
    __compq2 = __compq.clone()
    
    
    #we quarterized some variables that were already available quarterized. Now just updating them if they have missing values
    __compq3 = __compq
    for var_ytd in qvars_y:
        var = var_ytd[:-1]
        if (var + "q") in qvars_q:
            __compq3 = __compq3.with_columns(
            pl.col(var + "q").fill_null(pl.col(var_ytd + "_q"))
        ).drop(var_ytd + "_q")
        else:
            __compq3 = __compq3.rename({var_ytd + "_q": var + "q"})
    
    
    #creating some variables that need in quarterly forms
    __compq3 = __compq3.with_columns([
        pl.col("ibq").alias("ni_qtr"),
        pl.col("saleq").alias("sale_qtr"),
        (
        pl.coalesce([
            pl.col("oancfq"),
            pl.col("ibq") + pl.col("dpq") - pl.col("wcaptq").fill_null(0)
        ])
    ).alias("ocf_qtr")
    ])
    
    
    #replaing quarterly variables with ttm:
    
    yrl_vars = [
        "cogsq", "xsgaq", "xintq", "dpq", "txtq", "xrdq", "dvq", "spiq", "saleq", "revtq",
        "xoprq", "oibdpq", "oiadpq", "ibq", "niq", "xidoq", "nopiq", "miiq", "piq", "xiq",
        "xidocq", "capxq", "oancfq", "ibcq", "dpcq", "wcaptq",
        "prstkcq", "sstkq", "purtshrq",
        # "dsq",
        "dltrq", "ltdchq", "dlcchq",
        "fincfq", "fiaoq", "txbcofq", "dvtq",
        #we are missingk three variables here
        #adding here
        # "gpq", "doq", "dltisq"
    ]
    
    
    for var_yrl in yrl_vars:
        var_yrl_name = var_yrl[:-1]
    
        has_all_values = (
            pl.col(var_yrl).is_not_null() &
            pl.col(var_yrl).shift(1).is_not_null() &
            pl.col(var_yrl).shift(2).is_not_null() &
            pl.col(var_yrl).shift(3).is_not_null()
        )
    
        bad_history = (
            (((pl.col("gvkey") != pl.col("gvkey").shift(3)) | pl.col("gvkey").shift(3).is_null()) |
            ((pl.col("fyr") != pl.col("fyr").shift(3)) | pl.col("fyr").shift(3).is_null()) |
            ((pl.col("curcdq") != pl.col("curcdq").shift(3)) | pl.col("curcdq").shift(3).is_null()) |
            (((pl.col("fqtr") + pl.col("fqtr").shift(1) + pl.col("fqtr").shift(2) + pl.col("fqtr").shift(3)) != 10) |
             (pl.col("fqtr").shift(1).is_null()) |
             (pl.col("fqtr").shift(2).is_null()) |
             (pl.col("fqtr").shift(3).is_null()))) 
            | (~has_all_values)
        ) 
    
    
        ttm_value = (
            pl.col(var_yrl) +
            pl.col(var_yrl).shift(1) +
            pl.col(var_yrl).shift(2) +
            pl.col(var_yrl).shift(3)
        )
    
        __compq3 = __compq3.with_columns(
            pl.when((pl.col("fqtr") == 4) & bad_history )
              .then(pl.col(var_yrl_name + 'y'))
            .when(((pl.col("fqtr") != 4) & bad_history))
              .then(pl.lit(None).cast(pl.Float64))
            .otherwise(ttm_value)
            .alias(var_yrl_name)
        )
    
    
    
    
    #renaming bs variables by removing the q suffix
    bs_vars = [
        "seqq", "ceqq", "pstkq", "icaptq", "mibq", "gdwlq", "req",
        "atq", "actq", "invtq", "rectq", "ppegtq", "ppentq", "aoq", "acoq", 
        "intanq", "cheq", "ivaoq", "ivstq", "ltq", "lctq", "dlttq", 
        "dlcq", "txpq", "apq", "lcoq", "loq", "txditcq", "txdbq"
    ]
    
    bs_vars_updated = list(col[:-1] for col in bs_vars)
    
    __compq3 = __compq3.rename(dict(zip(bs_vars, bs_vars_updated)))
    __compq3 = __compq3.rename({"curcdq": "curcd"})
    __compq4 = (
        __compq3
        .sort(["gvkey", "datadate", "fyr", "fyearq", "fqtr", "source"])  # match SAS sort order
        .group_by(["gvkey", "datadate"], maintain_order=True)
        .tail(1)  # keep last row per group
        .drop(["fyr", "fyearq", "fqtr"])
    )
    
    __compa2 = __compa.with_columns(
                pl.lit(None).cast(pl.Float64).alias('ni_qtr'),
                pl.lit(None).cast(pl.Float64).alias('sale_qtr'),
                pl.lit(None).cast(pl.Float64).alias('ocf_qtr') 
        )
    

    
    __compa3 = __compa2.collect().join(__me_data, left_on=["gvkey", "datadate"], right_on=["gvkey", "eom"], how="left").select(__compa2.columns+ ["me_fiscal"])
    __compq5 = __compq4.join(__me_data, left_on=["gvkey", "datadate"], right_on=["gvkey", "eom"], how="left").select(__compq4.columns+ ["me_fiscal"])
    

    
    include_helpers_vars=1
    if include_helpers_vars==1:
        __compq6 = add_helper_vars(data=__compq5)
        __compa4 = add_helper_vars(data=__compa3)
    else:
        __compq6 = __compq5.clone()
        __compa4 = __compa3.clone()

    return __compa4, __compq6