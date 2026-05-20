# Factor Replication Documentation

This document describes exactly how each of the 9 thesis factors is replicated at the
individual stock level in `src/jkp/data/thesis_factors.py`. It is intended to be read
by AI agents auditing the replication methodology against original papers.

**Output file:** `data/processed/thesis_factor_weights.parquet`
**Columns:** `eom, id, w_MktRF, w_SMB, w_HML, w_MOM, w_RMW, w_CMA, w_ROE, w_IA, w_BAB`
**Convention:** Long-leg weights sum to +1 per month; short-leg weights sum to −1 per month.
One row per (eom, id) with at least one non-zero weight.

**Benchmark factors for comparison (VMP.R Part 3):**
- MktRF, SMB, HML, MOM, RMW, CMA: Ken French Data Library (FF5 + MOM)
- ROE, IA: Hou-Xue-Zhang q5 factors (`r_roe`, `r_ia` from global-q.org)
- BAB: Frazzini-Pedersen (2014) / AQR

---

## Pipeline Infrastructure

### Characteristics source
All characteristics come from `data/processed/characteristics/USA.parquet`, produced by
`jkp build`. The file is filtered to US common stocks on major exchanges
(`primary_sec=1, common=1, obs_main=1, exch_main=1, me>0`).

### Publication lags (`main.py`)
JKP's original pipeline applies a **uniform 4-month publication lag** to all accounting
data (both quarterly and annual) via the `lag_to_public` parameter in `create_acc_chars`.

This repo changes the lags to match the q5 convention:
```python
create_acc_chars("acc_std_ann.parquet", ..., lag_to_public=6, ...)  # annual: 4 → 6 months
create_acc_chars("acc_std_qtr.parquet", ..., lag_to_public=3, ...)  # quarterly: 4 → 3 months
```

**Effect on ROE:** In any calendar month, JKP's original used quarterly earnings from
4 months prior; q5 uses 3 months prior. The 1-month gap meant JKP was often one full
fiscal quarter staler than q5.

**Effect on IA:** JKP's original started using annual asset growth 4 months after fiscal
year end; q5 waits 6 months (standard FF timing). JKP was 2 months earlier, causing a
2-month divergence window each year where portfolio compositions differ.

### NYSE breakpoint identification
```python
_NYSE = (
    ((pl.col("crsp_exchcd") == 1) & pl.col("comp_exchg").is_null())
    | ((pl.col("comp_exchg") == 11) & pl.col("crsp_exchcd").is_null())
)
```

### Financial firm exclusion (`_FIN_FF49`)
FF49 industries 45–48 (Banks, Insurance, Real Estate, Finance/Trading) are used as a
proxy for SIC 6000–6999. Applied to CMA, RMW, ROE, and IA per French (2015) and
Hou-Xue-Zhang q5 conventions. Note: FF49 is not an exact SIC 6000-6999 equivalent —
this is a known approximation.

### June rebalancing year mapping
```python
_reb_year(): Jul–Dec → same calendar year; Jan–Jun → prior calendar year
```
This ensures portfolios formed in June year Y are held July Y through June Y+1.

---

## Factor-by-Factor Description

---

### MktRF — Market Excess Return

**Benchmark:** Ken French Data Library

**Construction:**
- Universe: all stocks passing the main filter
- Weight: value-weighted by market cap each month
  `w_MktRF = me_i / Σme` within each `eom`

**Characteristic used:** `me` (market equity, from CRSP)

**Current replication quality:** Very high correlation with French MktRF.

**History / what was tried:**
- No significant issues. The VW market portfolio is mechanically straightforward.
- The excess return in VMP.R is computed using `ret_exc` from `world_ret_monthly.parquet`
  which already subtracts the risk-free rate.

---

### SMB — Small Minus Big

**Benchmark:** Ken French FF5 Data Library

**Construction (Fama-French 2015):**
SMB is the average of the small-minus-big components from three independent annual
2×3 sorts: on `be_me`, `ope_be`, and `at_gr1`. Each sort contributes a per-stock
weight `w_smb = +(1/3)×vw` for small-cap stocks and `−(1/3)×vw` for large-cap stocks.
The final `w_SMB = (w_bm + w_op + w_inv) / 3`.

**Breakpoints:** NYSE stocks only, June of each year, 50th percentile of ME for size split.
No financial exclusion for SMB (matches French).

**Rebalancing:** Annual, frozen July–June.

**History / what was tried:**
- No significant issues. Implemented directly per FF (2015).

---

### HML — High Minus Low (Value)

**Benchmark:** Ken French FF5 Data Library

**Construction (Fama-French 1993, B/M denominator per French website):**
Annual 2×3 sort. Size split at NYSE ME p50; B/M split at NYSE p30/p70.

**B/M ratio construction — critical detail:**
French's B/M uses **December prior-year market equity** as the denominator, not June ME.
JKP's `be_me` characteristic uses June ME implicitly. The fix:

```
bm_french = (be_me_june × me_june) / me_dec
```

where `me_dec` is the ME from December of year Y, matched to portfolios held
July Y+1 through June Y+2 (i.e., `dec_year + 1 = reb_yr`).

- Breakpoints are computed from `bm_french` for NYSE June stocks with valid December ME.
- All stocks with valid `be_me > 0` and a non-null `bm_french` are assigned to portfolios.
- Portfolios held July–June (annual rebalancing).
- No financial exclusion (matches French HML original).

**Weight formula:**
```
w_HML = +0.5 × vw  (char_pf = H, high B/M = value)
w_HML = −0.5 × vw  (char_pf = L, low B/M = growth)
```

**Current replication quality:** High correlation after the December ME fix.

**History / what was tried:**
- **Initial version:** Used JKP `be_me` directly as the sort characteristic, which
  implicitly uses June ME as the B/M denominator. This caused a significant magnitude
  gap vs. French HML even though the sign/direction was correct.
- **Fix:** Implemented French B/M with December ME denominator. Substantial improvement.
  The key insight is that French freezes December ME for the denominator but updates
  the book value from fiscal year-end accounting data.

---

### MOM — Momentum

**Benchmark:** Ken French Momentum Factor (Data Library)

**Construction (Jegadeesh-Titman 1993 with skip-month):**
Monthly 2×3 sort. Size split at NYSE ME p50; momentum split at NYSE p30/p70.

**Momentum signal:**
```
ret_12_2 = (1 + ret_12_1) / (1 + ret_exc_{t-1}) − 1
```
`ret_12_1` is the 12-month cumulative return through last month (already in JKP).
Dividing out `ret_exc_{t-1}` (one-month-lagged excess return) produces the skip-month
momentum signal (months t−12 through t−2).

**Rebalancing:** Monthly.

**Weight formula:**
```
w_MOM = +0.5 × vw  (char_pf = H, winners)
w_MOM = −0.5 × vw  (char_pf = L, losers)
```

**Current replication quality:** High correlation.

**History / what was tried:**
- No significant issues. Monthly rebalancing with skip-month correction implemented
  directly. The `ret_12_1` characteristic from JKP already provides the cumulative return.

---

### RMW — Robust Minus Weak (Profitability)

**Benchmark:** Ken French FF5 Data Library

**Construction (Fama-French 2015):**
Annual 2×3 sort on `ope_be` (operating profitability / book equity).
Size split at NYSE ME p50; ope_be split at NYSE p30/p70.
**Financial firms excluded** from both portfolio universe and breakpoints (FF49 45–48).

**Rebalancing:** Annual, frozen July–June.

**Weight formula:**
```
w_RMW = +0.5 × vw  (char_pf = H, Robust = high ope_be)
w_RMW = −0.5 × vw  (char_pf = L, Weak = low ope_be)
```

**Current replication quality:** Moderate-to-good correlation.

**History / what was tried:**
- **Initial version:** No financial exclusion. The breakpoint code path already had
  an `excl_financials` option but the portfolio universe was not filtered.
- **Fix:** Added financial exclusion from the portfolio universe (pre-filter `df_sort`
  by `~ff49.is_in([45,46,47,48])`). Marginal improvement in correlation.
- Note: financial exclusion from breakpoints was already present via `excl_financials=True`
  in `_june_breakpoints`.

---

### CMA — Conservative Minus Aggressive (Investment)

**Benchmark:** Ken French FF5 Data Library

**Construction (Fama-French 2015):**
Annual 2×3 sort on `at_gr1` (annual asset growth).
Size split at NYSE ME p50; at_gr1 split at NYSE p30/p70.
**Financial firms excluded** from both portfolio universe and breakpoints (FF49 45–48).

**June at_gr1 freeze — critical detail:**
JKP's `at_gr1` characteristic is updated monthly as new annual filings arrive
(propagated forward with the 4→6 month lag). Without freezing, a stock could
change sort bucket mid-year as its annual accounting data becomes available,
violating the FF annual-rebalancing convention. Fix: the June `at_gr1` value
is captured and held constant for the entire July–June holding year:

```python
june_at = df_sort.filter(eom.dt.month() == 6) → at_gr1_june per (id, reb_yr)
# All months in holding year use at_gr1_june for portfolio assignment
```

**Rebalancing:** Annual, frozen July–June (using June `at_gr1` value).

**Weight formula:**
```
w_CMA = +0.5 × vw  (char_pf = L, Conservative = low investment)
w_CMA = −0.5 × vw  (char_pf = H, Aggressive = high investment)
```

**Current replication quality:** Good correlation after June freeze + financial exclusion.

**History / what was tried:**
- **Initial version:** Used monthly `at_gr1` directly, causing mid-year re-sorting as
  new annual filings arrived in JKP's monthly-expanded characteristic file. This
  produced higher turnover and portfolio compositions that diverged from French.
- **Fix 1:** June freeze (at_gr1_june carried forward). Significant improvement.
- **Fix 2:** Financial exclusion added. Additional marginal improvement.

---

### ROE — Return on Equity (q-factor)

**Benchmark:** Hou-Xue-Zhang q5 (`r_roe` from global-q.org)

**Construction (Hou-Xue-Zhang 2015, q5):**
Monthly 2×3 sort on `niq_be`.
Size split at NYSE ME p50; niq_be split at NYSE p30/p70.
**Financial firms excluded** from both portfolio universe and breakpoints (FF49 45–48).

**Characteristic `niq_be` in JKP pipeline:**
```
niq_be = ibq / be_x.shift(3)   [safe_div mode 9]
```
`ibq` = income before extraordinary items (Compustat quarterly).
`be_x.shift(3)` = book equity from 3 months prior in the monthly panel
(gives one-quarter-lagged book equity for December FY firms).
**This is `ibq`, not `niq` (net income)** — JKP already uses IB, matching q5 exactly.

**Publication lag:** 3 months for quarterly data (`create_acc_chars` in `main.py`).
This means `niq_be` for a quarter ending month Q becomes available at month Q+3,
matching the q5 convention. (JKP original used 4 months.)

**Rebalancing:** Monthly (portfolios sort on the latest available quarterly earnings).

**Weight formula:**
```
w_ROE = +0.5 × vw  (char_pf = H, high ROE)
w_ROE = −0.5 × vw  (char_pf = L, low ROE)
```

**Current replication quality:** Correlation ~0.889 with q5 r_roe.

**History / what was tried:**
- **Initial version:** Annual June rebalancing (same as FF factors). Wrong — q5 rebalances monthly.
- **Fix 1:** Changed to monthly rebalancing. Improvement.
- **Fix 2:** Financial exclusion added. Slight worsening (0.838→0.835). The imperfect
  FF49→SIC mapping may be causing slight divergence.
- **Investigated IB vs NI:** Initially hypothesized JKP uses `niq` (net income including
  extraordinary items) while q5 uses `ibq`. Code investigation revealed JKP already uses
  `ibq` via `ni_qtr = col("ibq")` in `aux_functions.py`. No fix needed or possible.
- **Fix 3 (pipeline-level):** Changed publication lag from 4 to 3 months for quarterly
  data in `main.py`. This required a full `jkp build` re-run. Improved from 0.835→0.889.

**Remaining gap:** The FF49 45–48 proxy for SIC 6000-6999 financial exclusion is
imperfect — some financial firms may be included/excluded differently than q5.

---

### IA — Investment (q-factor)

**Benchmark:** Hou-Xue-Zhang q5 (`r_ia` from global-q.org)

**Construction (Hou-Xue-Zhang 2015, q5):**
Monthly 2×3 sort on `at_gr1` (annual asset growth = (at_t − at_{t-1}) / at_{t-1}).
Size split at NYSE ME p50; at_gr1 split at NYSE p30/p70.
**Financial firms excluded** from both portfolio universe and breakpoints (FF49 45–48).

**Key distinction from CMA:** IA rebalances **monthly** using the most recently available
`at_gr1` (no June freeze). CMA uses annual June freeze. Both factors use the same
underlying characteristic (`at_gr1`) but with different rebalancing conventions, making
them conceptually distinct despite sharing the same raw data.

**Publication lag:** 6 months for annual data (`create_acc_chars` in `main.py`).
This means `at_gr1` for a fiscal year ending in December Y becomes available in June Y+1,
matching the q5 convention and standard FF timing. (JKP original used 4 months, making
data available 2 months too early in April Y+1.)

**Rebalancing:** Monthly.

**Weight formula:**
```
w_IA = +0.5 × vw  (char_pf = L, Conservative = low investment)
w_IA = −0.5 × vw  (char_pf = H, Aggressive = high investment)
```

**Current replication quality:** Correlation ~0.736 with q5 r_ia. Visual alignment
improved after publication lag fix even though correlation slightly decreased (0.747→0.736).

**History / what was tried:**
- **Initial version:** Annual June rebalancing (wrong — q5 uses monthly sort).
  Also no financial exclusion.
- **Fix 1:** Monthly rebalancing. Improvement.
- **Fix 2:** Financial exclusion added. No measurable effect on correlation.
- **Fix 3 (pipeline-level):** Changed publication lag from 4 to 6 months for annual
  data in `main.py`. Required full `jkp build` re-run. Correlation slightly worsened
  (0.747→0.736) but visual cumulative return alignment improved.

**Remaining gap:** IA has the lowest correlation of all 9 factors. Possible remaining
sources of divergence:
1. FF49 45–48 proxy imperfectly captures q5's SIC 6000-6999 exclusion.
2. Fiscal year end distribution differences between JKP and q5 universe.
3. Universe differences (JKP uses additional quality filters not in q5).

---

### BAB — Betting Against Beta

**Benchmark:** Frazzini-Pedersen (2014), AQR

**Construction (Frazzini-Pedersen 2014):**
Rank-weighted long-short portfolio where each leg is scaled to unit beta.
The net beta of the combined long-short position is zero by construction.

**Beta measure:** `betabab_1260d` (raw FP beta, computed in JKP pipeline):
```
beta_raw = corr(r_i, r_mkt)_{1260d} × (rvol_252d / mktvol_252d)
```
Uses 1260-day (5-year) rolling correlation and 252-day rolling volatility ratio.

**Vasicek shrinkage** (applied in `thesis_factors.py`):
```
beta_shrunk = 0.6 × beta_raw + 0.4
```
Shrinks toward the cross-sectional mean of 1.0, as in FP (2014).

**Rank weighting:**
Stocks are ranked by `beta_shrunk` each month. The centered rank is:
```
z_i = rank_i − (n+1)/2
```
Stocks with `z < 0` (low beta) go long; stocks with `z > 0` (high beta) go short.
Raw weights proportional to |z|, normalized within each leg.

**Beta scaling (unit-beta legs):**
```
w_long_final  = w_long_raw  / Σ(w_long_raw  × beta_shrunk)   → long leg has beta=1
w_short_final = −w_short_raw / Σ(w_short_raw × beta_shrunk)  → short leg has beta=−1
net beta = 0
```

**Universe:** All stocks with non-null `betabab_1260d`. No size or financial exclusion —
BAB spans the entire universe by design.

**Current replication quality:** Good correlation with FP BAB. Closest to original among
all 9 factors.

**History / what was tried:**
- **Unit test bug:** `test_net_beta_zero` was initially testing net raw beta = 0.
  The zero-net-beta property holds for *shrunk* beta, not raw beta. Fixed by computing
  `beta_shrunk` in the test before asserting net beta = 0.
- No other significant methodology issues. The FP construction is well-defined and
  implemented directly from the paper.

---

## Summary Table

| Factor | Characteristic | Sort type | Rebalancing | Financial excl. | Benchmark | Corr (approx.) |
|--------|---------------|-----------|-------------|-----------------|-----------|----------------|
| MktRF  | me            | VW all    | Monthly     | No              | French    | ~1.00          |
| SMB    | be_me+ope_be+at_gr1 | FF5 2×3 avg | Annual June | No        | French FF5 | High          |
| HML    | bm_french (Dec ME denom) | FF 2×3 | Annual June | No    | French FF5 | High          |
| MOM    | ret_12_2      | FF 2×3    | Monthly     | No              | French MOM | High          |
| RMW    | ope_be        | FF 2×3    | Annual June | Yes (FF49 45-48) | French FF5 | Moderate-good |
| CMA    | at_gr1_june (frozen) | FF 2×3 | Annual June | Yes (FF49 45-48) | French FF5 | Good        |
| ROE    | niq_be (ibq/be_x) | FF 2×3 | Monthly    | Yes (FF49 45-48) | q5 r_roe | ~0.889        |
| IA     | at_gr1        | FF 2×3    | Monthly     | Yes (FF49 45-48) | q5 r_ia  | ~0.736        |
| BAB    | betabab_1260d | Rank-weighted | Monthly | No             | FP (2014) | Good          |

---

## Key Design Decisions and Known Limitations

1. **FF49 vs SIC 6000-6999:** French and q5 exclude financial firms defined as SIC
   6000-6999. JKP's pipeline assigns FF49 industry codes; industries 45-48 (Banks,
   Insurance, Real Estate, Finance/Trading) are used as a proxy. This is not a perfect
   mapping and may cause small systematic differences in the financial exclusion.

2. **`at_gr1` shared by CMA and IA:** Both factors use the same underlying characteristic
   but with fundamentally different construction rules — CMA uses annual June-frozen values
   with French 2×3 convention; IA uses monthly-updated values with q5 convention. The
   replication code maintains separate implementations.

3. **Publication lag change scope:** The `lag_to_public` change in `main.py` affects ALL
   characteristics derived from annual or quarterly Compustat data, not just `at_gr1`
   and `niq_be`. Other factors using accounting characteristics (HML, RMW, CMA via
   `be_me`, `ope_be`, `at_gr1`) will also be slightly affected, though these are
   benchmarked against French (not q5) and the effect is small relative to the HML/RMW/CMA
   correlations.

4. **`niq_be` uses `ibq` not `niq`:** Despite the variable name ("net income quarterly"),
   JKP computes `niq_be = ibq / be_x.shift(3)` using income *before* extraordinary items
   (`ibq`), matching q5 exactly. This was verified in `aux_functions.py` line 3623:
   `ni_qtr = col("ibq")`.

5. **SMB financial exclusion:** SMB is computed without financial exclusion even though
   its constituent sorts (ope_be, at_gr1) are used with financial exclusion in RMW/CMA.
   This matches French — SMB itself is not filtered for financials.
