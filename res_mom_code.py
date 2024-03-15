def regression_3vars(y, x1, x2, x3, __n, __min):
    den __msf(pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.col(x3).rolling_var(window_size=__n, min_periods=__min))) +
       (pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min))**2 +
       (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min))**2 -
       2 * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) +
       (pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))**2)
    beta1 = ((pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) -
         (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * ((pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min))) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) +
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) +
         (pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min))) / den
    beta2 = ((pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) -
         (pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * ((pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min))) +
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min)) +
         (pl.col(x3).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))) / den
    beta3 = (-((pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min))) +
         (pl.col(x1).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) +
         (pl.col(x2).rolling_var(window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x1, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x2, y, window_size=__n, min_periods=__min)) -
         (pl.rolling_cov(x2, x3, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, y, window_size=__n, min_periods=__min)) * (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min)) +
         (pl.rolling_cov(x1, x2, window_size=__n, min_periods=__min))**2 * (pl.rolling_cov(x3, y, window_size=__n, min_periods=__min))) / den
    alpha = pl.col(y).rolling_mean(window_size=__n, min_periods = __min) - beta1 * pl.col(x1).rolling_mean(window_size=__n, min_periods = __min) - beta2 * pl.col(x2).rolling_mean(window_size=__n, min_periods = __min) - beta3 * pl.col(x3).rolling_mean(window_size=__n, min_periods = __min)
    return alpha, beta1, beta2, beta3
def get_rolling_residuals(a,b1,b2,b3,x1,x2,x3,y,w):
    squared_res_sum = w*(pl.col(a)**2) + 2 * pl.col(a) * pl.col(b1) * pl.col(x1).rolling_sum(window_size=w) + (pl.col(b1)**2) * (pl.col(x1)**2).rolling_sum(window_size=w) + 2 * pl.col(a) * pl.col(b2)* pl.col(x2).rolling_sum(window_size=w) + 2 * pl.col(b1) * pl.col(b2) * (pl.col(x1) * pl.col(x2)).rolling_sum(window_size=w) + (pl.col(b2)**2) * (pl.col(x2)**2).rolling_sum(window_size=w) + 2 * pl.col(a) * pl.col(b3) * pl.col(x3).rolling_sum(window_size=w) + 2 * pl.col(b1) * pl.col(b3) * (pl.col(x1) * pl.col(x3)).rolling_sum(window_size=w) + 2 * pl.col(b2) * pl.col(b3) * (pl.col(x2) *  pl.col(x3)).rolling_sum(window_size=w) + (pl.col(b3)**2) * (pl.col(x3)**2).rolling_sum(window_size=w) - 2 * pl.col(a) * pl.col(y).rolling_sum(window_size=w) -  2 * pl.col(b1) * (pl.col(x1) * pl.col(y)).rolling_sum(window_size=w) - 2 * pl.col(b2) * (pl.col(x2) * pl.col(y)).rolling_sum(window_size=w) - 2 * pl.col(b3) * (pl.col(x3) * pl.col(y)).rolling_sum(window_size=w) + (pl.col(y)**2).rolling_sum(window_size=w)
    res_mean = (pl.col(y).rolling_mean(window_size=w) - pl.col(a) - pl.col(b1) * pl.col(x1).rolling_mean(window_size=w) - pl.col(b2) * pl.col(x2).rolling_mean(window_size=w) - pl.col(b3) * pl.col(x3).rolling_mean(window_size=w))
    var = (squared_res_sum - w*(res_mean**2))/(w-1)
    return (res_mean/(var**0.5))
def residual_momentum(output_path, data_path, fcts_path, __n, __min, incl, skip):
    w = incl - skip
    data = pl.scan_ipc(data_path)
    fcts = pl.scan_ipc(fcts_path).select(['excntry', 'eom','mktrf', 'hml', 'smb_ff'])
    c1 = pl.col('ret_local') != 0
    c2 = pl.col('ret_exc').is_not_null()
    c3 = pl.col('ret_lag_dif') == 1
    c4 = pl.col('mktrf').is_not_null()
    data = data.filter(c1).filter(c2).filter(c3).select(['excntry','id','eom','ret_exc'])
    fcts = fcts.filter(c4).select(['excntry', 'eom','mktrf', 'hml', 'smb_ff'])
    __msf = data.join(fcts, how = 'left', on = ['excntry', 'eom']).drop('excntry')
    wins_data = winsorize_own(__msf,['eom'],'ret_exc',0.1/100,99.9/100)
    __msf = __msf.join(wins_data, how = 'left', on = 'eom')
    __msf = __msf.with_columns(ret_exc = pl.when(pl.col('ret_exc')<= pl.col('low')).then(pl.col('low')).when(pl.col('ret_exc')>= pl.col('high')).then(pl.col('high')).otherwise(pl.col('ret_exc'))).select(['id','eom','ret_exc','mktrf','hml','smb_ff'])
    __msf2 = __msf.select(['id','eom']).sort(['id','eom']).with_columns(aux = (pl.col('eom').shift(-1).over('id')).dt.offset_by(f'-1mo').dt.month_end())
    __msf2 = __msf2.with_columns(date_range = pl.date_ranges(start='eom', end='aux', interval='1mo')).with_columns(date_range = pl.coalesce(['date_range', 'eom'])).select(['id','date_range']).explode('date_range').rename({'date_range': 'eom'})
    __msf = __msf2.join(__msf, on = ['id','eom'], how = 'left')
    alpha, beta1, beta2, beta3 = regression_3vars('ret_exc','mktrf','smb_ff','hml',__n,__min)
    __msf = __msf.sort(['id','eom']).with_columns([alpha.over('id').alias('alpha'),beta1.over('id').alias('beta1'),beta2.over('id').alias('beta2'),beta3.over('id').alias('beta3')])
    residual_exp = lambda i:  pl.col('ret_exc').shift(i) - (pl.col('alpha')*pl.lit(1.) + pl.col('beta1') * pl.col('mktrf').shift(i) + pl.col('beta2') * pl.col('smb_ff').shift(i) + pl.col('beta3') * pl.col('hml').shift(i))
    __msf = __msf.sort(['id','eom']).with_columns(std_res = pl.concat_list([residual_exp(i).over(['id']) for i in range(1, w+1)]).list.drop_nulls()).select(['id','eom','std_res'])
    __msf = __msf.collect()
    __msf = __msf.with_columns(std_res =  pl.col('std_res').list.eval(pl.element().mean()/pl.element().std())).explode('std_res')
    __msf = __msf.select(['id','eom',pl.col('std_res').alias(f'resff3_{incl}_{skip}')])
    __msf.sort(['id','eom']).write_ipc(output_path + f'_{incl}_{skip}.ft')
    del __msf
def sas_percentile_method_5(series, p):
    """
    Calculates the given percentile using the SAS 5th method, which is the default SAS method and was used in our SAS code.
    """
    n = len(series)
    rank = p * n
    if rank.is_integer():return (series[int(rank) - 1] + series[int(rank)]) / 2
    else: return series[int(rank)]
def winsorize_own(data, sort_vars, wins_var, perc_low, perc_high):
    aux = data.group_by(sort_vars).agg(pl.col(wins_var))
    aux = aux.with_columns(low = pl.col(wins_var).list.sort().map_elements(lambda x: sas_percentile_method_5(x, perc_low)),
                            high = pl.col(wins_var).list.sort().map_elements(lambda x: sas_percentile_method_5(x, perc_high)))
    return aux.select([*sort_vars, 'low','high'])
def sort_ff_style(char, freq, min_stocks_bp, min_stocks_pf, date_col, data, sf):
    print(f'Executing sort_ff_style for {char}', flush=True)
    c1 = ((pl.col('size_grp_l').is_in(['small', 'large', 'mega'])).fill_null(pl.lit(False).cast(pl.Boolean)) & (pl.col('excntry_l') != 'USA').fill_null(pl.lit(False).cast(pl.Boolean)))
    c2 = (((pl.col('crsp_exchcd_l') == 1).fill_null(pl.lit(False).cast(pl.Boolean)) | (pl.col('comp_exchg_l') == 11).fill_null(pl.lit(False).cast(pl.Boolean))) & (pl.col('excntry_l') == 'USA').fill_null(pl.lit(False).cast(pl.Boolean)))
    c3 = pl.col(f'{char}_l').is_not_null()
    bp_stocks = data.sort(['eom','excntry_l','id']).filter((c1 | c2) & c3)#.sort(['eom', 'excntry_l'])
    bp_stocks = bp_stocks.group_by(['eom', 'excntry_l']).agg([pl.len().alias('n'), pl.col(f'{char}_l').alias(f'{char}_l')])
    bp_stocks = bp_stocks.with_columns(bp_p30 = pl.col(f'{char}_l').list.sort().map_elements(lambda x: sas_percentile_method_5(x, 0.3)),
                                        bp_p70 = pl.col(f'{char}_l').list.sort().map_elements(lambda x: sas_percentile_method_5(x, 0.7)))
    #bp_stocks = bp_stocks.filter(pl.col('n') >= min_stocks_bp).select(['eom', 'excntry_l', 'bp_p30','bp_p70'])
    #data = data.filter(pl.col(f'{char}_l').is_not_null()).filter((pl.col('size_pf') != '').fill_null(pl.lit(True).cast(pl.Boolean))).join(bp_stocks, how = 'left', on = ['excntry_l', 'eom'])
    bp_stocks = bp_stocks.select(['eom', 'excntry_l', 'n', 'bp_p30','bp_p70'])
    data = data.join(bp_stocks, how = 'left', on = ['excntry_l', 'eom']).filter(pl.col('n') >= min_stocks_bp).filter(pl.col(f'{char}_l').is_not_null()).filter((pl.col('size_pf') != '').fill_null(pl.lit(True).cast(pl.Boolean))).drop('n')
    char_pf_col = (pl.when(pl.col(f'{char}_l') >= pl.col('bp_p70')).then(pl.lit('high')).when(pl.col(f'{char}_l') >= pl.col('bp_p30')).then(pl.lit('mid')).otherwise(pl.lit('low'))).alias('char_pf')
    data = data.with_columns(char_pf_col).select(['excntry_l','id','eom','size_pf', 'me_l', 'char_pf'])
    data = data.with_columns(w = (pl.col('me_l')/pl.sum('me_l')).over('excntry_l','size_pf', 'char_pf', 'eom'), n = pl.len().over('excntry_l','size_pf', 'char_pf', 'eom'))
    data = data.filter(pl.col('n') >= min_stocks_pf).drop('n')
    returns = sf.join(data, how = 'inner', left_on= ['id','eom','excntry'],right_on= ['id','eom','excntry_l'])
    returns = returns.with_columns((pl.col('ret_exc')*pl.col('w')).alias('ret_exc'))
    returns = returns.group_by(['excntry', 'size_pf', 'char_pf', date_col]).agg(ret_exc = pl.sum('ret_exc'))
    returns = returns.with_columns(characteristic = pl.lit(char), combined_pf = (pl.col('size_pf') + '_' + pl.col('char_pf'))).drop(['size_pf', 'char_pf'])
    returns = returns.collect()
    returns = returns.pivot(values='ret_exc',index=['excntry', date_col],columns='combined_pf')
    lms = ((pl.col('small_high') + pl.col('big_high')) / 2 - (pl.col('small_low') + pl.col('big_low')) / 2).alias('lms')
    smb = ((pl.col('small_high') + pl.col('small_mid') + pl.col('small_low')) / 3 - (pl.col('big_high') + pl.col('big_mid') + pl.col('big_low')) / 3).alias('smb')
    returns = returns.with_columns([lms, smb]).select(['excntry', date_col, 'lms', 'smb']).sort(['excntry', date_col])
    return returns
def ap_factors(output_path, freq, sf_path, mchars_path, mkt_path, min_stocks_bp, min_stocks_pf):
    print(f'Executing AP factors with frequency {freq}', flush=True)
    sf = pl.scan_ipc(sf_path)
    mchars = pl.scan_ipc(mchars_path)
    mkt = pl.scan_ipc(mkt_path)
    date_col = {'m': 'eom','d': 'date'}
    if freq == 'd': world_sf1 = sf.select(['excntry','id', 'date','eom', 'ret_exc','ret_lag_dif']).filter((pl.col('ret_lag_dif') <= 5) & (pl.col('ret_exc').is_not_null())).drop('ret_lag_dif')
    else: world_sf1 = sf.select(['excntry','id','eom', 'ret_exc','ret_lag_dif']).filter((pl.col('ret_lag_dif') == 1) & (pl.col('ret_exc').is_not_null())).drop('ret_lag_dif')
    wins_data = winsorize_own(world_sf1,['eom'],'ret_exc',0.1/100,99.9/100)
    world_sf2 = world_sf1.join(wins_data, how = 'left', on = 'eom').with_columns(ret_exc = pl.when(pl.col('ret_exc')<= pl.col('low')).then(pl.col('low')).when(pl.col('ret_exc')>= pl.col('high')).then(pl.col('high')).otherwise(pl.col('ret_exc'))).drop(['high','low'])
    base = mchars.select(['id','eom','size_grp','excntry','me','market_equity','be_me','at_gr1','niq_be','source_crsp','exch_main','obs_main','common','comp_exchg','crsp_exchcd','primary_sec','ret_lag_dif']).rename({'comp_exchg': 'comp_exchg_l', 'crsp_exchcd': 'crsp_exchcd_l', 'exch_main': 'exch_main_l', 'obs_main': 'obs_main_l', 'common': 'common_l', 'primary_sec': 'primary_sec_l', 'excntry': 'excntry_l', 'size_grp': 'size_grp_l', 'me': 'me_l', 'be_me': 'be_me_l', 'at_gr1': 'at_gr1_l', 'niq_be': 'niq_be_l'})
    cols_to_lag = ['comp_exchg_l', 'crsp_exchcd_l', 'exch_main_l', 'obs_main_l', 'common_l', 'primary_sec_l', 'excntry_l', 'size_grp_l', 'me_l', 'be_me_l', 'at_gr1_l', 'niq_be_l']
    base = base.sort(['id', 'eom']).with_columns([pl.col(i).shift(1).over(['id']).alias(i) for i in cols_to_lag])
    c1 = (pl.col('id') != pl.col('id').shift(1)).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (pl.col('source_crsp') != pl.col('source_crsp').shift(1)).fill_null(pl.lit(True).cast(pl.Boolean))
    col1 = (pl.col('eom').dt.year() - pl.col('eom').shift(1).dt.year()).over(['id'])
    col2 = (pl.col('eom').dt.month().cast(pl.Int32) - pl.col('eom').shift(1).dt.month().cast(pl.Int32)).over(['id'])
    c3 = ((12*col1 + col2) != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    base = base.sort(['id', 'eom']).with_columns([pl.when(c1 | c2 | c3).then(pl.lit(None).cast(base.schema[i])).otherwise(i).alias(i) for i in cols_to_lag])#.sort(['id', 'eom'])
    c1 = (pl.col("obs_main_l") == 1)
    c2 = (pl.col("exch_main_l") == 1)
    c3 = (pl.col("common_l") == 1)
    c4 = (pl.col("primary_sec_l") == 1)
    c5 = (pl.col("ret_lag_dif") == 1)
    c6 = pl.col('me_l').is_not_null()
    base = base.filter(c1 & c2 & c3 & c4 & c5 & c6)
    size_pf_column = (pl.when(pl.col('size_grp_l').is_null()).then(pl.lit('')).when(pl.col('size_grp_l').is_in(['large', 'mega'])).then(pl.lit('big')).otherwise(pl.lit('small'))).alias('size_pf')
    base = base.with_columns(size_pf_column).sort(['excntry_l', 'size_grp_l', 'eom'])
    ff = sort_ff_style('be_me', freq, min_stocks_bp,min_stocks_pf,date_col[freq], base, world_sf2).rename({'lms' : 'hml', 'smb' : 'smb_ff'})
    asset_growth = sort_ff_style('at_gr1', freq, min_stocks_bp,min_stocks_pf,date_col[freq], base, world_sf2).rename({'lms': 'at_gr1_lms', 'smb': 'at_gr1_smb'})
    roeq = sort_ff_style('niq_be', freq, min_stocks_bp,min_stocks_pf,date_col[freq], base, world_sf2).rename({'lms': 'niq_be_lms', 'smb': 'niq_be_smb'})
    hxz = asset_growth.join(roeq, how = 'left', on = ['excntry',date_col[freq]])
    hxz = hxz.select(['excntry', date_col[freq], (-1*pl.col('at_gr1_lms')).alias('inv'), pl.col('niq_be_lms').alias('roe'), ((pl.col('at_gr1_smb') + pl.col('niq_be_smb'))/2).alias('smb_hxz')])
    output = mkt.select(['excntry', date_col[freq], pl.col('mkt_vw_exc').alias('mktrf')]).collect()
    output = output.join(ff, how = 'left', on = ['excntry', date_col[freq]])
    output = output.join(hxz, how = 'left', on = ['excntry', date_col[freq]])
    output.write_ipc(output_path)
