from datetime import date

import polars as pl

# Last calendar date kept in pipeline outputs.
END_DATE = date(2025, 12, 31)

# Earliest fiscal-period-end date kept when building the standardized
# accounting panel; rows with `datadate` before this are dropped.
ACCOUNTING_START_DATE = pl.datetime(1949, 12, 31)

# CRSP MSF / DSF row filters: 1 keeps the row, 0 drops it.
MAIN_FILTERS = {
    "primary_sec": 1,
    "common": 1,
    "obs_main": 1,
    "exch_main": 1,
}

# Number of per-characteristic lazy pipelines collected concurrently per
# chunk in `portfolios()`' chunked `collect_all` logic (bounds peak memory
# of sort buffers / join hash tables when running over hundreds of
# characteristics).
COLLECT_CHUNK_SIZE = 20

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

# Canonical list of stock characteristics that `portfolios()` constructs
# breakpoints / portfolio sorts / HML-LMS factors for. Production default;
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
    ("_21d", 15, ["ff3", "hxz4"]),
    # ("_126d", 60, ["zero_trades", "turnover", "dolvol", "ami"]),
    # ("_252d", 120, ["rvol", "capm", "downbeta", "zero_trades", "prc_to_high", "mktvol"]),
    # ("_1260d", 750, ["mktcorr"]),
]

# ============================================================================
# FF factor pipeline (gen_ff_data)
# ============================================================================

# ISO-3 code for the United States; sentinel for excntry-branched filters in
# the unified FF pipeline (US-strict NYSE breakpoint pool, US-bypassed
# min-stocks gates).
US_EXCNTRY = "USA"

# Raw CRSP parquet + column-prefix per frequency. US-only.
FF_CRSP_SPECS = {
    "monthly": ("crsp_msf_v2.parquet", "mth"),
    "daily": ("crsp_dsf_v2.parquet", "dly"),
}

# port_year = formation year y for any date t in Jul(y)..Jun(y+1).
FF_PORT_YEAR = (
    pl.when(pl.col("date").dt.month() >= 7)
    .then(pl.col("date").dt.year())
    .otherwise(pl.col("date").dt.year() - 1)
    .alias("port_year")
)

# Per-sort definitions: value col, NYSE 30/70 break column names, bucket
# labels (ladder), port column, eligibility flag, bucket label space, and
# the SN/BN rename to disambiguate across OP/INV legs.
FF_SORT_SPECS: dict[str, dict] = {
    "bm": {
        "value": "beme",
        "breaks": ["beme30", "beme70"],
        "labels": ["L", "M", "H"],
        "port": "btmport",
        "flag": "positivebeme",
        "buckets": ["SL", "SM", "SH", "BL", "BM", "BH"],
        "rename": {},
    },
    "op": {
        "value": "op",
        "breaks": ["op30", "op70"],
        "labels": ["W", "N", "R"],
        "port": "opport",
        "flag": "nonmiss_op",
        "buckets": ["SW", "SN", "SR", "BW", "BN", "BR"],
        "rename": {"SN": "SN_op", "BN": "BN_op"},
    },
    "inv": {
        "value": "inv",
        "breaks": ["inv30", "inv70"],
        "labels": ["C", "N", "A"],
        "port": "invport",
        "flag": "nonmiss_inv",
        "buckets": ["SC", "SN", "SA", "BC", "BN", "BA"],
        "rename": {"SN": "SN_inv", "BN": "BN_inv"},
    },
    "mom": {
        "value": "mom_2_12",
        "breaks": ["mom30", "mom70"],
        "labels": ["L", "N", "H"],
        "port": "momport",
        "flag": "nonmiss_mom",
        "buckets": ["SL", "SN", "SH", "BL", "BN", "BH"],
        "rename": {},
    },
}

# Portfolio assignment columns broadcast from June(y) to the daily/monthly
# panel.
FF_PORT_COLS = (
    "sizeport",
    "btmport",
    "positivebeme",
    "opport",
    "nonmiss_op",
    "invport",
    "nonmiss_inv",
)

# Min firms per (excntry, June(y)) for ROW breakpoints; US bypassed.
FF_MIN_STOCKS_BP = 10

# Min firms per (excntry, date, bucket) for ROW VW; US bypassed.
FF_MIN_STOCKS_PF = 3

# HXZ q-factor ROW thresholds (same values as FF; separate names for clarity).
HXZ_MIN_STOCKS_BP = 10
HXZ_MIN_STOCKS_PF = 3

# CRSP -99.0 missing-return sentinel: treat as 0 return when compounding the
# UMD prior-(2-12) signal, per FF convention.
FF_MISSING_RET_CODE = -99.0
FF_MISSING_RET_TOL = 1e-6

# UMD signal window lengths (trading days for daily, calendar months for monthly).
FF_UMD_DAILY_SKIP = 21
FF_UMD_DAILY_LOOKBACK = 251
FF_UMD_MONTHLY_LOOKBACK = 12
FF_UMD_MONTHLY_SKIP = 1


# ============================================================================
# Mispricing Factors (Stambaugh-Yuan) — ported from MisprProject
# ============================================================================

# anomallist.sas7bdat (alphabetical, 1-indexed to match SAS macro args)
MP_ANOMALY_LIST = [
    "ACCRUAL_ADJ",      # 1
    "ASSET_GROWTH",     # 2
    "COMPOSITE_ISSUE",  # 3
    "DISTRESS",         # 4
    "GP_ADJ",           # 5
    "INVASSET",         # 6
    "MOMENTUM",         # 7
    "NOA",              # 8
    "OSCORE",           # 9
    "ROA",              # 10
    "STOCK_ISSUE",      # 11
]
MP_POSITIVE_ANOMALIES = {"GP_ADJ", "MOMENTUM", "ROA"}  # bucketed descending

# Cal_M4 macro args. SAS final cols rename umo1->MGMT, umo2->PERF.
# WARNING: SAS dump filenames are SWAPPED relative to leg content. Preserve verbatim.
MP_MGMT_IDX = [6, 1, 3, 8, 2, 11]   # INVASSET, ACCRUAL_ADJ, COMPOSITE_ISSUE, NOA, ASSET_GROWTH, STOCK_ISSUE
MP_PERF_IDX = [4, 9, 10, 7, 5]      # DISTRESS, OSCORE, ROA, MOMENTUM, GP_ADJ

# CHS distress (Campbell-Hilscher-Szilagyi) coefficient set
MP_DISTRESS_BETAS = {
    "intercept": -9.164,
    "NIMTAAVG": -20.264,
    "TLMTA": 1.416,
    "EXRETAVG": -7.129,
    "SIGMA": 1.411,
    "RSIZE": -0.045,
    "CASHMTA": -2.132,
    "MB": 0.075,
    "PRICE": -0.058,
}

MP_LAG_M = 4
MP_START_FACTOR_EOM = date(1963, 1, 31)

# World extension constants (per-country breakpoint pool gates, mirrors jkp ROW gen_ff_data)
MP_MAIN_FILTERS_COLS = ["primary_sec", "common", "obs_main", "exch_main"]
MP_MIN_STKS_BP_WORLD = 10  # min stocks per (excntry, eom) for percentile rank inclusion
MP_MIN_OBS_PF_WORLD = 3    # min stocks per (excntry, eom, bucket) for portfolio VW


PORTFOLIO_SETTINGS = {
    "end_date": END_DATE,
    "pfs": PORTFOLIO_PFS,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "bps": "non_mc",
    "bp_min_n": PORTFOLIO_BP_MIN_N,
    "cmp": {"us": True, "int": False},
    "signals": {"us": False, "int": False, "standardize": True, "weight": "vw_cap"},
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
