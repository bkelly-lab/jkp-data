# Data Changelog
This change log keeps track of changes to the underlying data set. In brackets, we highlight versions of importance. The version with _factor data set_ is the basis of the factor portfolios we upload at [https://jkpfactors.com/](https://jkpfactors.com/). The version with _paper data set_ is the basis of [Jensen, Kelly and Pedersen (2023)](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249).

This repository ports the original SAS pipeline ([ReplicationCrisis](https://github.com/bkelly-lab/ReplicationCrisis)) to Python using Polars. Entries up to and including 05-03-2025 are from the original change log.

## 01-11-2026
__Changes__:
- Fixed the de-duplication of Compustat quarterly records ([#306](https://github.com/bkelly-lab/jkp-data/pull/306)). When a firm changes its fiscal year-end, Compustat keeps some quarters from the old fiscal calendar with a missing fiscal quarter (`fqtr`): 347 records in 179 North America firms. These records are now dropped before the year-to-date flows are converted to quarterly flows. Before, such a record could fill a month that has no regular quarter (46 months), replace the regular quarter on the same `datadate` (8 quarters), or set the year-to-date flows of the fiscal year's first quarter to missing (3 fiscal years). Months that were filled by a dropped record now carry the previous regular quarter forward, and the quarterly data lose these 46 rows. When one source reports the same fiscal quarter twice (5 cases in Compustat Global), the record with the earliest `datadate` is kept; before, the choice depended on row order. When two records share a `datadate`, the one with the highest `fyr` is kept as before, then the Compustat Global record. The annual data are unchanged. In the quarterly data, 2,852 of 754,108,988 values change. In the accounting characteristics, 23,698 of 6,137,845,592 values change in 51 firms (1,150 newly filled, 7,571 newly missing), with no change in rows, and every characteristic's correlation with the old values is at least 0.99999.
- Four-quarter sums of quarterly income and cash-flow items (trailing twelve months, TTM) now require four consecutive fiscal quarters ([#308](https://github.com/bkelly-lab/jkp-data/pull/308)). The previous check, ported from SAS, accepted four rows of a firm with the same fiscal year-end and currency whenever their fiscal-quarter numbers summed to 10, so a firm with a gap in its quarterly data could get a sum of quarters years apart (6,556 quarterly records in 4,294 firms, most from 2000 onward). These sums are now missing, except in the fourth fiscal quarter, which takes the year-to-date value. The annual data are unchanged. In the quarterly data, 37,771 of 754,108,988 values change, in 1,606 firm-quarters of 866 firms. In the accounting characteristics, 219,590 of 6,137,845,592 values change, in 803 firms (15 newly filled, 68,943 newly missing), with no change in rows; 8,802 of the changed firm-months are in the main JKP universe, mostly micro and nano caps. Every characteristic's correlation with the old values is at least 0.99998.
- Firm-months with quarterly but no annual accounting data now get accounting characteristics ([#312](https://github.com/bkelly-lab/jkp-data/pull/312)). The annual and quarterly characteristics were combined with a left join on the annual data, so these months had none; they now take the quarterly characteristics. Existing accounting characteristic values are unchanged. 285,629 security-months gain accounting characteristics, 111,002 of them in the main JKP universe: 0.76% of its 14,551,988 stock-months and 0.73% of its market cap, in 14,563 firms, mostly micro and nano caps (89%). Of these, 75% are months between a firm's listing and its first annual report; the rest fall in gaps in or after the firm's annual history, or in 97 firms with no annual data. The added rows are thin: the median has 30 of the 315 accounting characteristics, and 64% have `be_me`. Because more stocks enter the cross-sectional sorts, values that depend on them change for other stocks too: the FF3 and HXZ4 factors behind `resff3_12_1`, `resff3_6_1`, `ivol_ff3_21d`, `iskew_ff3_21d`, `ivol_hxz4_21d` and `iskew_hxz4_21d`, the rank-based `qmj*` and `mispricing_*` scores, and the factor portfolios. These changes were not measured.
- Fixed the conversion of accounting data to US dollars ([#313](https://github.com/bkelly-lab/jkp-data/pull/313)). Compustat's exchange rates start in February 1982, so Canadian-dollar reports before then had no rate and all their values were missing. They now use OECD month-end CAD/USD rates from January 1960 (OECD Main Economic Indicators, via FRED), which gives values to about 8,200 annual reports of 547 firms. Monthly Canadian-dollar price records before 1982 (483) also get an exchange rate, so a few Canadian stocks a month can now enter the stock universe before 1982; since country market returns have no minimum number of stocks, Canada's market return before 1982 may rest on very few stocks (not measured, as the full pipeline was not rerun). Records that still have no exchange rate (no currency code, Zimbabwe dollars, Canadian dollars before 1960, or a date outside the currency's rate history: 708 annual and 204 quarterly records) used to be kept with all values missing, which stopped the firm's previous report from being used. They are now dropped, and the previous report stays in use for up to 18 months as usual. In the main JKP universe, accounting characteristics change in 33,041 of 14,551,988 stock-months (0.23%), mostly in firms listed in the US or Canada before 1990. In 982 of them `assets` becomes missing because the dropped report was the firm's only annual report; #312 keeps such months with their quarterly data.

## 28-10-2026
__Changes__:
- Set `ni_ar1` to null instead of NaN when the lagged NI/AT does not vary over the regression window, where DuckDB's `regr_slope` returns NaN ([#294](https://github.com/bkelly-lab/jkp-data/pull/294)). This affects 694 stock-months in 10 countries, 48 of them in the US. Polars sorts NaN above every number, so these stocks were placed in the top `ni_ar1` portfolio. 672 are now null and left out of the sort. In the other 22, the annual NaN blocked the fallback to the quarterly value that all accounting characteristics use; they now take the quarterly `ni_ar1`, from the same regression as the quarterly `ni_ivol` they already had. As a safeguard, factor portfolio construction now treats any NaN characteristic value as missing.

## 15-10-2026
__Changes__:
- Nulled rolling daily characteristics over constant or near-dead windows ([#269](https://github.com/bkelly-lab/jkp-data/pull/269)). Return-based characteristics (`rvol_*`, `ivol_capm_*`, `ivol_ff3_*`, `ivol_hxz4_*`, `beta_*`, `betadown_*`, `beta_dimson_*`, `rmax*`, `rskew_*`, `iskew_*`, `coskew_*`, `corr_*`, `ami_*`) are nulled when raw `ret` is constant over the window or zero on at least 90% of its days. The check uses raw `ret` rather than `ret_exc`, whose monthly risk-free steps would otherwise hide constant returns over multi-month windows (`_252d`, `_1260d`). `prc_highprc_*`, `dolvol_var_*` and `turnover_var_*` are nulled only when the adjusted price, dollar volume or turnover is exactly constant over the window. The derived `betabab_1260d` and `rmax5_rvol_21d` become null with their inputs. Residual NaN/inf values are scrubbed in `finish_daily_chars`.

## 02-09-2026
__Changes__:
- Fixed double-counting of CRSP CIZ monthly delisting returns. `MthRet` already includes the delisting return when `MthDelFlg` is `"A"`, `"P"` or `"V"`, but the pipeline compounded `delret` into it again. The pipeline now carries `MthDelFlg`/`DlyDelFlg` through as `del_flag` and skips the compounding for those flags. Flags `"M"` (delisting return missing) and `"G"` (delisting return more than 10 days from the delisting date) are still compounded, with `delret` imputed at −0.30 for performance-related delistings when it is missing. Published daily returns are unchanged: CRSP's own delisting row (`DlyDelFlg = "Y"`, dated `deldlydt`) is off the main exchange and never reached the published `world_dsf`, so `delret` (or the −0.30 imputation) is still compounded into the last trading day's return. The pipeline now drops that row before monthly overnight/intraday compounding. Because it has no open/close prices, it previously set `ret_intraday`, `ret_overnight`, `ret_intraday_local`, `ret_overnight_local` and their `_lead1m` leads to null for the delisting month; about 10,000 US stock-months, mostly from 1992 on, now have values, and no existing value changes. This affects `world_ret_monthly.parquet` and downstream monthly characteristics and factor portfolios. See Xia (2026, SSRN 7243220) for background.

## 30-07-2026
__Changes__:
- Added daily overnight/intraday factor portfolios in both USD and local currency under `portfolios/`, parallel to the standard excess-return daily factors. Sorting and breakpoints use the usual `ret_exc_lead1m` characteristic sorts; only the daily return leg changes. For each component in `{overnight, intraday, overnight_local, intraday_local}` (using `ret_overnight`, `ret_intraday`, `ret_overnight_local`, `ret_intraday_local` respectively), the pipeline writes `pfs_{component}.parquet`, `hml_{component}.parquet`, `lms_{component}.parquet`, `clusters_{component}.parquet`, plus `regional_factors_{component}/`, `regional_clusters_{component}/`, and `country_factors_{component}/`. O/I portfolio returns are not excess returns (no risk-free subtraction). Winsorization of the daily return leg applies only to the standard `ret_exc` path; O/I legs are aggregated as-is. Industry and characteristics-managed portfolios are not rebuilt for O/I components. Monthly O/I factor portfolios are not produced.

## 23-07-2026
__Changes__:
- Added local-currency overnight/intraday return columns (`ret_intraday_local`, `ret_overnight_local`) alongside the existing USD versions, following the Lou, Polk, and Skouras (2019) decomposition:
  - `ret_intraday = P_close / P_open − 1`
  - `ret_overnight = (1 + ret) / (1 + ret_intraday) − 1`
  - `ret_intraday_local = ret_intraday` (FX cancels in the open/close ratio)
  - `ret_overnight_local = (1 + ret_local) / (1 + ret_intraday_local) − 1`
  so that `(1 + ret_intraday)(1 + ret_overnight) = (1 + ret)` and `(1 + ret_intraday_local)(1 + ret_overnight_local) = (1 + ret_local)`. The overnight residual against `ret_local` instead of `ret` means daily FX moves no longer appear in `ret_overnight_local`. New columns are propagated through monthly compounding (`ret_intraday_local`, `ret_overnight_local`), one-month-ahead leads (`ret_intraday_local_lead1m`, `ret_overnight_local_lead1m`), daily country files, monthly returns, and daily factor portfolios. Also added the previously missing `_lead1m` columns for the USD variants to the monthly returns output.

## 15-07-2026
__Changes__:
- Applied the daily `zero_obs < 10` quality screen only to Compustat-sourced rows; CRSP stock-months with many zero-return days are no longer dropped from daily return-based characteristics and market correlation inputs
- Made the Gao–Ritter NASDAQ dealer-volume `/1.6` window inclusive of 2003-12-31 (CRSP Polars helper and Compustat SECD/SECM SQL), matching the documented schedule

## 10-07-2026
__Changes__:
- Applied an adapted version of the Bessembinder, Chen, Choi, and Wei (2023) data corrections to the Compustat daily security files (North America and Global). Decimal-shift price errors are detected and rescaled (single- and multi-period spike-and-reversal); observations that cannot be reliably repaired are dropped (low trading volume, zero AJEXDI, sub-threshold price or market equity, long data gaps, implausible shares-outstanding and market-equity jumps, return/market-equity mismatches, and initial-observation ratio errors). The multi-period search is adapted for daily data (symmetric windows over a priority set of 1, 2, 3, 5, 10, and 21 trading days). For more details, please look at the [release notes](documentation/compustat_correction/compustat_correction.html)
- Recovered returns for never-dividend securities by substituting `trfd = 1` in `gen_comp_dsf` when a security never pays a dividend, so its price return equals its total return instead of being dropped as a missing total-return factor
- Added a post-correction return screen: Compustat returns above 1000% (`ret` > 10) that survive the price correction are set to null in `gen_returns_df` as residual, unrepairable data errors
- The impact concentrates outside the United States; US factor series are largely unchanged because US observations are sourced primarily from CRSP rather than Compustat

## 04-07-2026
__Changes__:
- Replaced `crsp_shrcd` and `crsp_exchcd` with `primaryexch` and `conditionaltype` in the characteristics output ([#205](https://github.com/bkelly-lab/jkp-data/pull/205))

## 27-04-2026
__Changes__:
- Added `mkt_vw_cap_exc` (cap-weighted excess market return) to market returns output ([#81](https://github.com/bkelly-lab/jkp-data/pull/81))

## 22-04-2026
__Changes__:
- Fixed a case-sensitivity bug in the Fama-French 49 industry classification that left some stocks unclassified ([#78](https://github.com/bkelly-lab/jkp-data/pull/78))
- Fixed non-deterministic deduplication in the Compustat preparation step, which could cause output to differ slightly between runs ([#83](https://github.com/bkelly-lab/jkp-data/pull/83))

## 14-04-2026
__Changes__:
- Default screening filters are now applied before saving output, consistent with the documentation ([#68](https://github.com/bkelly-lab/jkp-data/pull/68))

## 05-03-2025 [Factor data set]
__Changes__:
- Added 2024 data
- moved world_data_prelim from scratch to work folder
- updated market returns macro to add a capped value option
- corrected error in o-score calculation
- added end_date filter in macros saving daily and monthly returns 

## 11-03-2024 
__Changes__:
- Added 2023 data
- Updated the country classification according the latest MSCI market classification

## 03-03-2023 
__Changes__:
- Added 2022 data
- Added 'me' (market equity) and 'ret' (total return) and removed 'source_crsp' from daily return files

__Impact__:
- Replication rate: 83.2% 

## 30-06-2022 [Paper data set] 

__Changes__:
- Changed name of "Skewness" cluster to "Short-Term Reversal"

__Impact__:
- Replication rate: 82.4% 

## 08-02-2022 

__Changes__:
- Fix error in the construction of intrinsic_value. Previously, we failed to scale intrinsic_value by market equity as done in Frankel and Lee (1998). We call the new characteristic ival_me and keep intrinsic_value in the data set. The alpha of the new factor based on ival_me is significantly different from zero, while the factor based on intrinsic_value is insignificant.

__Impact__:
- Replication rate: 82.4% (added 2020 data)


## 16-11-2021 

__Changes__:
- Changed return cutoffs to depend on all stocks, instead of only stocks from CRSP.
- Added monthly and daily returns to the output folder. 
- Changed the 'source' (character) column to 'source_crsp' (integer),. source_crsp is 1 if CRSP is the return data source.
- Changed the 'id' column from character to integer. For stocks from CRSP, the id is just their permno. For stocks from Compustat, the first digits is 1 if the stocks is traded on a US exchange, 2 if it's traded on a Canadian exchange, and 3 otherwise. The next two digits are the IID from Compustat, and the remaining six digits are the gvkey.  
- Adapted the primary_sec column such that all observations from CRSP have primary_sec=1. 
- Previously, we treated a zero return as a missing observation. Now, we have removed this screen, such that a zero return is treated like any other return. 
- Previously, we winsorized daily returns, market equity, and dollar volume, before creating charactersitics based on daily stock market data. Now, we have removed this winsorization, and daily characteristics are based on the raw data. 
- Added the option to create daily factor return in the portfolios.R code.
- Added the option to create industry returns in the portfolios.R code.

__Impact__:
- Replication rate: 83.2%

## 27-08-2021 

__Changes__:
- Fixed a bug regarding how daily delisting returns from CRSP is incorporated.
- Added indfmt='FS' to the international accounting data. 

__Impact__:
- Replication rate: 83.2%

## 14-06-2021 

__Changes__:

- We changed the winsorization scheme. First, we removed the 0.01%/99.9% winsorization of market equity in all countries. Second, we removed the winsorization of returns from the CRSP database. For Compustat returns, we set returns above (below) the 99.9% (0.01%) of CRSP returns in the same month, to that level. In other words, we base our winsorization of Compustat data on CRSP data from the same month. 
- We made several changes to the code for easier usability. Notably, the updated `main.sas` file returns a zip folder called "output" in the scratch folder, which contains all data neccesary to re-produce the results in the paper.  

__Impact__:

- Replication rate: 83.2%
- The revisions impacted all factors slightly, but the overall results are qualitatively very similar. 

## 02-19-2021 

__Changes__:

- Previously we did not exclude securities that are only traded over the counter. In the new version of the data set, we include an indicator column "exch_main" to exclude non-standard exchanges. In the US, the main exchanges are AMEX, NASDAQ and NYSE. Outside of the US, we exclude over the counter exchanges, stock connect exchanges in China and cross-country exchanges such as BATS Chi-X Europe. The documentation includes a full list of the excluded exchanges.  
- Included SIC, NAICS and GICS industry codes.

__Impact__:

- Replication rate: 84.0%. 
- Excluding non-standard exchanges mainly affected the US. By December 2019, the number of stocks in the US dropped from 5,256 to 4,102 (-22%) after adding the new 'exch_main' screen. The excluded securities are mainly tiny stocks traded over the counter, so the aggregate market cap only dropped by 2%. The change also mostly affected post 2000 data, because over the counter observations in Compustat are very rare before this point in time.    
- The change had a small effect outside of the US, because of our 'primary_sec' screen. It's very rare for Compustat to identify a security traded on a non-standard exchange as the primary security of a firm. 
- Because the changes mainly affected tiny stocks, our results did not change much. Across the 153 factors in the US, Developed and Emerging regions, the change in posterior monthly alpha ranged from -0.06% to +0.07.

## 02-15-2021

__Changes__:

- A bug caused _ivol_ff3_21d_, _iskew_ff3_21d_, _ivol_hxz4_21d_ and _iskew_hxz4_21d_ to require 17 (ff3) and 18 (hxz) observations for a valid estimate. Consistent with our original intent, we now require at least 15 observations for a valid estimate.    

__Impact__:

- Replication Rate: 84.0%. 
- The changes had a negligible effect on the affected factors.

## 02-01-2021 

__Changes__:

- Fixed a small bug in the bidask_hl() macro.
- When creating asset pricing factors (FF and HXZ), we previously required at least 5 stocks in a sub-portfolio (e.g. small stocks with high BM) for the observation to be valid. This led to missing observation in the 1950's for small stocks with low bm. We lowered this requirement to at least 3 stocks. Furthermore, when creating asset pricing factors, we changed the breakpoints to be based on NYSE stocks in the US instead of non-microcap stocks. Outside of the US, breakpoints are still based on non-microcap stocks.
  
__Impact__:

- Replication Rate: 84.0% 
- The _bidaskhl_21d_ factor changed slightly but is still significantly negative in all regions. The US factor IR changed from -0.11 to -0.09.
- The change in asset pricing factor generally didn't affect the results much.

## 01-25-2021
__Changes__:

- Changed residual momentum characteristics (resff3_12_1 & resff3_6_1) to be scaled with the standard deviation of residuals consistent with Blitz, Huij and Mertens (2011). 
- Fixed error in creating _qmj_prof_. The issue was that the _oaccruals_at_ used the value instead of the z-score of ranks. This effectively meant that accruals didn't impact the profitability score. 
- Fixed error for annual seasonality characteristics (factor names starting with seas_ and ending with _an). There was a bug in the screening procedure which meant that the characteristic for one stock could use information from an unrelated stock. 
- Rounding issues when converting a .csv file to an excel file, caused the zero_trades_* variables to not have any decimals which made the turnover tie-breaker ineffective.
- Standardized unexpected earnings (niq_su) and sales (saleq_su) is computed as the actual value minus the expected value (standardized by the standard deviation of this change). Before, the expected value was computed as the mean yearly change over the last 8 quarters added to the last quarterly value. Now the expected value is the same mean yearly change, but added to the quarterly value 4 quarters ago consistent with Jegadeesh and Livnat (2006).
  
__Impact__:

- Replication Rate: 84.0%
- The change to the residual momentum variables, made them slightly weaker. As an example, the monthly OLS information ratio of the US resff3_12_1 factor dropped from 0.33 to 0.28. 
- The _qmj_prof_ change made _qmj_prof_ and _qmj_ slightly stronger. As an example, the monthly OLS information ratio of the US _qmj_prof_ factor increased from 0.16 to 0.22.  
- The seasonality fix didn't have a large qualitative impact for the US factors, but did have a large positive effect outside of the US. As an example, the OLS IR of the developed market _seas_11_15an_ factor changed from -0.06 to 0.11.   
- The zero_trades where missing in the developed market because of too few non-missing observations. The developed market zero trades factors are generally strong and the IR ranges from 0.07 to 0.20. Similarly, The Emerging market zero trades factors where slightly negative before. After, the factors are strong with IRs ranging from 0.14 to 0.17. The US market zero trades factors improved slightly. The IR of zero_trades_21d has the most notable increase from 0.05 to 0.09.   
- The standardized unexpected sales (saleq_su) variable went from a significant IR of 0.12 to an insignificant IR of 0.05. This explains the drop in the replication rate. On the other hand, niq_su increased from 0.11 to 0.19.

## 01-15-2021 
__Changes__:

  - Base data set used in the first online version of Jensen, Kelly and Pedersen (2021).
  
__Impact__:

- Replication Rate: 84.9%
