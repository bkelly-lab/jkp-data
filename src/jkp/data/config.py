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
