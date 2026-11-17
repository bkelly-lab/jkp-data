# Fama-French factors: methodology

`gen_ff_data` (src/jkp/data/aux_functions.py) builds FF3, FF5 and UMD factor
returns for every country, monthly and daily. This note records the rules the
code follows and where they differ from Fama and French.

## References

- Fama and French (1993), "Common risk factors in the returns on stocks and
  bonds", Journal of Financial Economics 33(1), 3-56.
- Fama and French (2015), "A five-factor asset pricing model", Journal of
  Financial Economics 116(1), 1-22.
- WRDS, "Fama-French Factors (CIZ Format)":
  https://wrds-www.wharton.upenn.edu/login/?next=/pages/wrds-research/applications/risk-factors-and-industry-benchmarks/fama-french-factors-ciz-format/
- Historical book equity from Ken French's data library:
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#BookEquity

## Structure

One code path serves every country. The functions appear in build order in
aux_functions.py: the firm panel and its weights, the annual characteristics,
the breakpoints, the portfolio assignments, the momentum signal and the factor
returns, then `gen_ff_data`. The universe and weight rules that differ between the US and other
countries are row columns set in `load_ff_universe`; the CRSP-only rules (the DFF
BE and the early-link rule in `link_accounting_to_securities`, the permco unit in
`compute_ff_annual_characteristics`) sit in those functions. Each
characteristic is built once, in `compute_ff_annual_characteristics`, for the
June sorts; the sample screens decide only which stocks enter a sort.

## Rules

- **US:** CRSP CIZ common shares of US companies on NYSE/AMEX/NASDAQ, NYSE
  breakpoints. The unit is the company, the CRSP permco: its share classes
  combine as securities in a portfolio (`build_ff_firm_panel`), with ME = JKP's
  `me_company` and the return weighted by class ME at t-1, without French's hand
  splits of a permco. A company is NYSE if any class is. Daily weights use the
  CIZ sum over the classes.
- **Elsewhere:** JKP's main sample, per-country breakpoints.
- **Minimums, every country, the US included:** `FACTOR_MIN_STOCKS_BP` pool
  stocks for a breakpoint and `FACTOR_MIN_STOCKS_PF` stocks for a portfolio
  return.
- **Returns, every country:** JKP's world_msf/world_dsf, so US returns include
  CRSP's delisting return (-30% where a bad delisting has none).
- **Weights, every country:** buy-and-hold from June, as Fama and French hold
  the portfolios from July to June. A stock's weight is its end-of-June ME grown
  by its total return since then (`add_ff_buy_and_hold_weights`); UMD, re-sorted
  every period, weights by ME at t-1.
- **Formation, every country:** June ME from the June row and Dec ME from the
  December t-1 row, with no fallback to earlier months, so a stock halted over
  June or December leaves that year's sorts. The B/M, OP and INV sorts also need
  the stock in the universe in December t-1.
- **Size median, every country:** one per country and date, shared by the B/M,
  OP and INV sorts, over every breakpoint-pool stock with ME > 0, whatever its
  sort eligibility (`compute_ff_breakpoints`).
- **No minimum number of years on Compustat, in any country.** Fama and French
  (1993) required two, against the backfill bias of firms Compustat adds with
  past years of data; their later descriptions of the factors and French's data
  library state no such rule.
- **Fiscal records, every country:** JKP's annual records, Global over North
  America at the same datadate, the latest per calendar year
  (`select_ff_fiscal_records`). A US CRSP security reaches its record through
  the gvkey JKP attaches to it in world_msf, on its June row, with an early-link
  rule for the June 1963-66 formations (`link_accounting_to_securities`). For a US
  CRSP security the DFF BE fills years with no Compustat BE, as in Davis, Fama
  and French (2000) (`add_dff_book_equity`).
- **BE, every country:** JKP's `be_x` at the fiscal year t-1 record, in USD; a
  stock with BE <= 0 is not eligible for the B/M and OP sorts. Like French's, it
  has no deferred taxes for fiscal years ending in 1993 or later.
- **OP, every country:** JKP's `ope_x` (EBITDA minus interest expense, a missing
  XINT counting as 0) over `be_x` plus minority interest, at the same record
  (`load_jkp_annual_accounting`). French's numerator is revenue minus the cost
  of goods sold, SG&A and interest expense.
- **INV, every country:** JKP's `at_gr1` (`at_x` over `at_x` 12 months earlier,
  minus 1) at the fiscal year t-1 record. French's US rule grows assets over
  consecutive fiscal years whatever their spacing; `at_gr1` is null after a
  change of fiscal year-end and is growth in USD.
- **Daily FF3/FF5** hold the monthly June portfolios; only returns and weights
  are daily.
- **UMD, every country:** sorts on the compounded t-12..t-2 return (the
  company's own return in the US) under French's eligibility rules, with one
  change: a stock needs `FF_UMD_MIN_MONTHS`/`FF_UMD_MIN_DAYS` rows in the
  window, where French requires it listed throughout
  (`compute_ff_momentum_signal`); daily UMD uses French's daily signal.
