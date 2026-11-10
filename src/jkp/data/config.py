from datetime import date

import polars as pl

# Last calendar date kept in pipeline outputs.
END_DATE = date(2025, 12, 31)

# Earliest fiscal-period-end date kept when building the standardized
# accounting panel; rows with `datadate` before this are dropped.
ACCOUNTING_START_DATE = pl.datetime(1949, 12, 31)

# Longest gap, in days, between two reports on one fiscal calendar before the
# calendar's run restarts, used by `one_row_per_date` to choose between two
# calendars on one date. Consecutive quarters are 89-92 days apart, semiannual
# reports or one missing quarter 181-184, and two missing quarters 273-276.
CALENDAR_RUN_MAX_GAP_DAYS = 200

# CRSP MSF / DSF row filters: 1 keeps the row, 0 drops it.
MAIN_FILTERS = {
    "primary_sec": 1,
    "common": 1,
    "obs_main": 1,
    "exch_main": 1,
}

# Number of portfolios characteristics are sorted into (e.g. 3 -> tertiles).
PORTFOLIO_PFS = 3

# Minimum number of stocks per (industry, month) group required for the
# group to contribute to characteristic breakpoints.
PORTFOLIO_BP_MIN_N = 10

# Minimum number of stocks a country-month needs to be eligible for the
# regional aggregation step.
REGIONAL_STOCKS_MIN = 5

# Minimum months of history (in months; multiplied by 21 trading days for
# daily aggregation) required for a characteristic to enter regional pfs.
REGIONAL_MONTHS_MIN = 5 * 12

# Minimum number of countries required for a regional portfolio to be
# reported.
REGIONAL_COUNTRIES_MIN = 3

# ISO-3 country codes excluded from regional aggregation (extreme
# inflation / hyperinflation regimes that distort returns).
REGIONAL_COUNTRY_EXCL = ("ZWE", "VEN")

# Themes version that keeps the old `cmp` file: the unsigned characteristics-managed
# portfolios (#297). Every other version writes `lms_cmp_{n}`, signed as `lms` signs `hml`.
UNSIGNED_CMP_VERSION = "v0"
# That version's file keeps its layout from before #323: US rows, five size groups.
UNSIGNED_CMP_SIZE_GROUPS = 5

# Size groupings of the characteristics-managed portfolios (#323), one file each:
# everything (`all`), `big` and `small`, or the five groups of `size_grp`.
CMP_SIZE_GROUPS = (1, 2, 5)

# A characteristics-managed group has a row only if the sample standard deviation of its
# characteristic is above this. Below it, the ranks would mostly order rounding noise. The value
# is our choice (#324), it is absolute, and it applies to every characteristic.
CMP_MIN_STD = 1e-10

# The 153 characteristics of JKP (2023) (themes v0), sorted first. `jkp portfolio`
# also sorts every other characteristic in a released themes version, and the
# themes files decide which factors are signed. Production default;
# tests override via `monkeypatch.setattr(..., PORTFOLIO_CHARS, ...)`.
PORTFOLIO_CHARS = [
    "age",
    "aliq_at",
    "aliq_mat",
    "ami_126d",
    "at_be",
    "at_gr1",
    "at_me",
    "at_turnover",
    "be_gr1a",
    "be_me",
    "beta_60m",
    "beta_dimson_21d",
    "betabab_1260d",
    "betadown_252d",
    "bev_mev",
    "bidaskhl_21d",
    "capex_abn",
    "capx_gr1",
    "capx_gr2",
    "capx_gr3",
    "cash_at",
    "chcsho_12m",
    "coa_gr1a",
    "col_gr1a",
    "cop_at",
    "cop_atl1",
    "corr_1260d",
    "coskew_21d",
    "cowc_gr1a",
    "dbnetis_at",
    "debt_gr3",
    "debt_me",
    "dgp_dsale",
    "div12m_me",
    "dolvol_126d",
    "dolvol_var_126d",
    "dsale_dinv",
    "dsale_drec",
    "dsale_dsga",
    "earnings_variability",
    "ebit_bev",
    "ebit_sale",
    "ebitda_mev",
    "emp_gr1",
    "eq_dur",
    "eqnetis_at",
    "eqnpo_12m",
    "eqnpo_me",
    "eqpo_me",
    "f_score",
    "fcf_me",
    "fnl_gr1a",
    "gp_at",
    "gp_atl1",
    "ival_me",
    "inv_gr1",
    "inv_gr1a",
    "iskew_capm_21d",
    "iskew_ff3_21d",
    "iskew_hxz4_21d",
    "ivol_capm_21d",
    "ivol_capm_252d",
    "ivol_ff3_21d",
    "ivol_hxz4_21d",
    "kz_index",
    "lnoa_gr1a",
    "lti_gr1a",
    "market_equity",
    "mispricing_mgmt",
    "mispricing_perf",
    "ncoa_gr1a",
    "ncol_gr1a",
    "netdebt_me",
    "netis_at",
    "nfna_gr1a",
    "ni_ar1",
    "ni_be",
    "ni_inc8q",
    "ni_ivol",
    "ni_me",
    "niq_at",
    "niq_at_chg1",
    "niq_be",
    "niq_be_chg1",
    "niq_su",
    "nncoa_gr1a",
    "noa_at",
    "noa_gr1a",
    "o_score",
    "oaccruals_at",
    "oaccruals_ni",
    "ocf_at",
    "ocf_at_chg1",
    "ocf_me",
    "ocfq_saleq_std",
    "op_at",
    "op_atl1",
    "ope_be",
    "ope_bel1",
    "opex_at",
    "pi_nix",
    "ppeinv_gr1a",
    "prc",
    "prc_highprc_252d",
    "qmj",
    "qmj_growth",
    "qmj_prof",
    "qmj_safety",
    "rd_me",
    "rd_sale",
    "rd5_at",
    "resff3_12_1",
    "resff3_6_1",
    "ret_1_0",
    "ret_12_1",
    "ret_12_7",
    "ret_3_1",
    "ret_6_1",
    "ret_60_12",
    "ret_9_1",
    "rmax1_21d",
    "rmax5_21d",
    "rmax5_rvol_21d",
    "rskew_21d",
    "rvol_21d",
    "sale_bev",
    "sale_emp_gr1",
    "sale_gr1",
    "sale_gr3",
    "sale_me",
    "saleq_gr1",
    "saleq_su",
    "seas_1_1an",
    "seas_1_1na",
    "seas_11_15an",
    "seas_11_15na",
    "seas_16_20an",
    "seas_16_20na",
    "seas_2_5an",
    "seas_2_5na",
    "seas_6_10an",
    "seas_6_10na",
    "sti_gr1a",
    "taccruals_at",
    "taccruals_ni",
    "tangibility",
    "tax_gr1a",
    "turnover_126d",
    "turnover_var_126d",
    "z_score",
    "zero_trades_126d",
    "zero_trades_21d",
    "zero_trades_252d",
]

# (window_suffix, min_obs, variables) triples driving roll_apply_daily.
# min_obs is the per-group minimum observation count for each window.
# Variables can repeat across windows (e.g. zero_trades runs at 21d/126d/252d).
ROLLING_DAILY_SPECS: list[tuple[str, int, list[str]]] = [
    ("_21d", 15, ["rvol", "rmax", "skew", "capm_ext", "ff3", "hxz4", "dimsonbeta", "zero_trades"]),
    ("_126d", 60, ["zero_trades", "turnover", "dolvol", "ami"]),
    ("_252d", 120, ["rvol", "capm", "downbeta", "zero_trades", "prc_to_high", "mktvol"]),
    ("_1260d", 750, ["mktcorr"]),
]

# ============================================================================
# FF factor pipeline (gen_ff_data)
# ============================================================================

# ISO-3 code for the United States; sentinel for excntry-branched filters in
# the unified FF pipeline (US-strict NYSE breakpoint pool, US-bypassed
# min-stocks gates).
US_EXCNTRY = "USA"

# Ex-US minimums of the factor builders (FF and HXZ): min firms per
# (excntry, formation date) for breakpoints, and per (excntry, date, bucket) for
# the VW legs. US bypassed.
FACTOR_MIN_STOCKS_BP = 10
FACTOR_MIN_STOCKS_PF = 3

# CRSP -99.0 missing-return sentinel: treat as 0 return when compounding the
# UMD prior-(2-12) signal, per FF convention.
FF_MISSING_RET_CODE = -99.0
FF_MISSING_RET_TOL = 1e-6

# UMD signal window lengths (trading days for daily, calendar months for monthly).
FF_UMD_DAILY_SKIP = 21
FF_UMD_DAILY_LOOKBACK = 251
FF_UMD_MONTHLY_LOOKBACK = 12
FF_UMD_MONTHLY_SKIP = 1

# Non-US UMD halt tolerance: min rows in the momentum window (monthly t-12..t-2 =
# 11 months; daily t-250..t-21 = 230 trading days). Compustat Global records a
# halt as missing rows, not French's -99.0. Endpoints stay strict; US needs all.
FF_UMD_ROW_MIN_MONTHS = 6
FF_UMD_ROW_MIN_DAYS = 115

# DFF (Davis-Fama-French) hand-collected Moody's BE splice into the US FF path.
# When True, union DFF BE (permno-keyed, bundled resource) with Compustat BE;
# DFF fills (permno, formation year) cells Compustat lacks, extending SMB/HML
# back to 1926-07 (and UMD to ~1927-01 via the longer CRSP panel). Per DFF
# (2000) the two BE sets are disjoint by construction: "we collect book common
# equity (BE) from 1925 to 1996 for all NYSE industrial firms that do not have
# BE data on Compustat."
FF_USE_DFF_BE = True

# Strict DFF (2000) replication gate: "beginning in 1954 we merge the
# hand-collected data ... with the Compustat data". True drops Compustat BE
# from formations before June FF_DFF_MERGE_YEAR prior to the union (pre-1954
# Compustat rows are backfill French never used). Default False: union
# whatever Compustat exists.
FF_DFF_GATE_COMPUSTAT_PRE1954 = False
FF_DFF_MERGE_YEAR = 1954

# On a (permno, year) collision the BE value is coalesced DFF-first
# (be = coalesce(dff.be, comp.be)) while the Compustat row keeps its
# accounting fields: most collisions are Compustat rows with NULL BE (early
# balance-sheet fields absent from the current vintage), and where both are
# non-null the sources agree (median |log ratio| ~ 0).

# CCM link-window backdating rescue for the early-1960s Compustat expansion.
# Compustat added ~200 (mostly small) NYSE/AMEX firms per year around 1964-66
# and backfilled their fy1962-64 statements; CCM stamps linkdt at coverage
# start, so those fiscal years sit BEFORE the link window and the standard
# linkdt <= June(t) filter rejects them. French uses them (his ME x OP
# portfolio counts exceed ours by ~200/yr in 1963-65 and match from 1966;
# diagnosed 2026-06: 208/206/192 of the misses are window-only failures with
# a valid L*/P-C link and a fy(t-1) funda row, median linkdt 2.6y after the
# formation). Rescue: for June formations in [FROM, THROUGH], accept a link
# whose linkdt is after the formation when it is the permno's first link and
# starts within FF_CCM_BACKDATE_MAX_YEARS of the formation (kills recycled-
# permno misattribution, e.g. a 2001 link claiming a 1963 row).
FF_CCM_BACKDATE_FROM = 1963
FF_CCM_BACKDATE_THROUGH = 1966
FF_CCM_BACKDATE_MAX_YEARS = 5

# ============================================================================
# HXZ q-factor pipeline (gen_hxz_data)
# ============================================================================

# First month of the HXZ factors. US: rdq coverage starts in 1972 (the pre-1972
# Roe proxy of Hou, Xue and Zhang 2019 is not used). Ex US: 1967.
HXZ_US_START = date(1972, 1, 1)
HXZ_ROW_START = date(1967, 1, 1)

# US quarterly Roe is usable from the month after its announcement (rdq) to this
# many months after the fiscal quarter end.
HXZ_US_ROE_MAX_LAG_MONTHS = 6

# Ex-US annual Roe is usable from MIN to MAX months after the fiscal year end.
HXZ_ROW_ROE_MIN_LAG_MONTHS = 6
HXZ_ROW_ROE_MAX_LAG_MONTHS = 18

# Clean-surplus BEQ imputation reaches back at most this many quarters (HXZ A.4).
HXZ_BEQ_IMPUTE_MAX_QUARTERS = 4

# Floor on the standard deviation of raw USD `ret` over a rolling window. At or
# below it the window counts as constant and the return-based statistics
# (betas, residual vols and skews, correlations, residual momentum) are null.
# Our choice (#338): on the 2026-11 data, 1e-6 nulls no daily window beyond the
# exact-constant / 90%-zero rule and only the 11 residual-momentum windows per
# horizon whose returns vary by about 1e-9.
RET_MIN_STD = 1e-6

# Floor on the standard deviation of lagged NI/AT, the regressor of earnings
# persistence. At or below it `ni_ar1` and `ni_ivol` are null. Our choice (#338):
# on the 2026-11 data the windows below it vary by at most 3e-8 and give
# slopes near 4e5; the next windows vary by 3e-6 or more.
NI_AT_MIN_STD = 1e-6

PORTFOLIO_SETTINGS = {
    "end_date": END_DATE,
    "pfs": PORTFOLIO_PFS,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "bps": "non_mc",
    "bp_min_n": PORTFOLIO_BP_MIN_N,
    "cmp": {"us": True, "int": True},
    "regional_pfs": {
        "country_excl": list(REGIONAL_COUNTRY_EXCL),
        "country_weights": "market_cap",
        "stocks_min": REGIONAL_STOCKS_MIN,
        "months_min": REGIONAL_MONTHS_MIN,
        "countries_min": REGIONAL_COUNTRIES_MIN,
    },
    "daily_pf": True,
    "ind_pf": True,
}

# Compustat return corrections (Bessembinder et al. 2023 Data Appendix)

# Compression for the scratch spill files written during the correction array
# passes; read once then deleted, so favor encode speed over size.
CORRECTION_SPILL_COMPRESSION = "lz4"

# Valid decimal-correction methods. The floor variants gate the price
# correction to the divide direction on >=$1 prices (handoff recommendation).
DECIMAL_CORRECTION_METHODS = ("multiplier", "interpolation", "floor", "floor_interp")

# Default multi-period detection windows (trading days), covering common error
# durations 1 day to ~1 month; ~10x fewer iterations than an exhaustive search.
DECIMAL_DETECTION_WINDOWS = [1, 2, 3, 5, 10, 21]

# Countries with a $0.001 minimum-price threshold instead of $0.01.
LOW_PRICE_COUNTRIES = ["BRA", "IDN", "NGA", "TUR"]

# Drop the first observation after a calendar gap wider than this many trading
# days (~11 months; converted to calendar days at 365/252).
FILTER_GAP_TRADING_DAYS = 231
