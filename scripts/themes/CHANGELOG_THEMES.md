# Themes changelog

Changes to the economic themes, newest first. See [README.md](README.md) for the versioning rules.

## v1.0 (15-10-2026)

A new grouping by economic meaning, replacing the statistically derived v0 themes. It covers
402 characteristics (v0: 153) in 58 subthemes and 16 themes, and adds a second level: each
characteristic belongs to a subtheme, and each subtheme to a theme. Every characteristic with
a v0 theme has a v1.0 theme.

**Themes:** Value, Profitability, Fundamental momentum, Earnings quality, Investment,
Intangibles, Operating efficiency, Financing, Leverage, Risk, Size, Short-term returns,
Medium-term returns, Long-term returns, Seasonal returns, Composites. See
`theme_descriptions.csv` for the subthemes.

**Names.** Theme and subtheme names describe the characteristics; `factor_name` in
`theme_descriptions.csv` names the long-short factor and shows which side it goes long: the
name itself for high values, and "Low" plus the name for low values (for example Low risk,
Low leverage, Low asset growth), with plain antonyms where they read better: Small,
Young, Illiquidity, Short duration and Stable earnings). Past returns keep their established
factor names: Short-term returns (factor: Short-term reversal), Medium-term returns (Momentum,
with subthemes Returns 2-6 and Returns 7-12 for Momentum 2-6 and Momentum 7-12), Long-term
returns (Long-term reversal) and Seasonal returns (Seasonality).

### Signs

Three signs, one per level: `subtheme_sign` (per characteristic) is +1 if a high value means
more of what the subtheme's name says; `theme_sign` (per subtheme) is +1 if more of the
subtheme means more of what the theme's name says; `factor_sign` (per theme) is +1 if more of
the theme earns higher returns in the literature. A characteristic's factor goes long the side
given by the product of the three. v0 instead signed each factor by the direction of its return
in the original paper; signing characteristics that way would point characteristics in the same
subtheme in opposite directions in 8 subthemes: in Leverage, for example, `debt_me` would be
+1 and `at_be` −1, although the two correlate positively.

- 35 characteristics have `subtheme_sign` −1, for example `ami_126d` in Liquidity,
  `z_score` in Distress and net payout in Equity issuance. The data agree with every sign except
  for four characteristics that barely correlate with their subtheme (`coskew_21d` and the three
  non-annual seasonality measures for years 6–20); these are +1 by definition.
- 8 subthemes have `theme_sign` −1: Equity duration (Value), Equity payout (Financing),
  Balance sheet liquidity and Debt maturity (Leverage), Accruals and Special items (Earnings
  quality), Capital intensity and Labor intensity (Operating efficiency). Where the data say
  little, these are economic choices: Debt maturity (−1) and Cost of debt (+1) in Leverage, and
  the Earnings quality and Operating efficiency signs.
- `factor_sign` is +1 for Value, Profitability, Fundamental momentum, Earnings quality,
  Intangibles, Operating efficiency, Medium-term returns, Seasonal returns and Composites, and
  −1 for Investment, Financing, Leverage, Risk, Size, Short-term returns and Long-term returns.

For 144 of the 153 v0 factors, the product of the three signs equals the direction used in
Jensen, Kelly and Pedersen (2023). For nine it is the opposite, because the characteristic sits
with its subtheme in the data while its own published direction pointed the other way:
`aliq_mat` (Value), `dsale_drec` (Working capital growth), `sti_gr1a` (Financial asset growth),
`debt_me` (Leverage), `aliq_at` (Balance sheet liquidity), `kz_index` (Financial constraints),
`ni_ivol` (Earnings volatility), `dolvol_var_126d` and `turnover_var_126d` (Liquidity).
The factor portfolios produced by `jkp portfolio` still use the published directions.

### Main decisions

- **Naming.** The broad groups are called themes, as in the original JKP paper; the finer
  groups are subthemes. Names have no "&" or parentheses; a group that needed "&" was split.
- **Past returns.** Medium-term returns are split by window: months 2–6 (Returns 2-6),
  months 7–12 (Returns 7-12), and windows of 9 months or longer plus the 52-week high
  (Medium-term returns). Returns over months 2–6 and 7–12 correlate only 0.07. Last month alone
  is Short-term returns; the 2- and 3-month windows that include it go to Returns 2-6.
  Multi-year windows go to Long-term returns.
- **Seasonal returns.** Same-month returns are seasonal at every lag. Averages over the other
  months go with the return window they overlap: year 1 to Medium-term returns, years 2–20 to
  Long-term returns. Their correlations with everything are close to zero, and their published
  direction matches long-term reversal.
- **Investment.** Asset growth is split by what grows: total and long-lived assets, capex,
  working capital and book equity. Individual balance-sheet items go to the subtheme of their
  item. Long-term investments and financial assets are separate subthemes.
- **Intangibles.** R&D and advertising intensity form their own theme rather than sitting in
  Investment. They are spending on intangible capital, not balance-sheet growth: R&D
  correlates close to zero with the growth subthemes and goes with low earnings yield, high
  cash and long duration, and R&D-intensive firms earn higher returns while fast-growing firms
  earn lower ones.
- **Profitability.** Margins and cash-flow profitability are separate from profitability. All
  profit-growth horizons form one subtheme. The profitability and growth parts of QMJ go to
  Profitability and Profit growth.
- **Value.** Gross profit to price is an earnings yield; cash-based operating profit to price
  is a cash flow yield. Equity duration is its own subtheme.
- **Financing.** Changes in equity and debt issuance join their issuance subthemes. Net total
  financing is External financing. Debt terms are split into maturity and cost. Growth in total
  liabilities, which includes debt, is Liability growth; growth in payables, accrued items and
  taxes stays with Investment as Operating payables growth.
- **Risk.** Lottery is separate from Volatility despite a correlation of 0.72. Market
  correlation sits with Beta.
- **Composites.** Scores that combine signals from several themes (QMJ, QMJ safety, F-score,
  both Stambaugh–Yuan mispricing factors) form their own theme. Scores that combine inputs
  toward a single concept (O-score, Z-score, Kaplan–Zingales index) stay in that concept's
  subtheme.
- **Fundamental momentum** is the theme for earnings surprises and profit growth, the usual
  meaning of the term.
- **Operating costs** over assets (Novy-Marx's operating leverage) stay in Asset turnover;
  they correlate 0.87 with asset turnover.

### Exceptions kept for economic reasons

These characteristics fit another subtheme clearly better in the data (by at least 0.15, at
0.40 or more) but stay where they are. The composite scores are not listed: by construction,
each fits its main ingredient better.

| Characteristic | Subtheme | Fits better with | Why it stays |
|---|---|---|---|
| `seas_1_1an` | Seasonal returns | Returns 7-12 | A same-month return; its overlap with months 7–12 is mechanical. |
| `cl_lt` | Debt maturity | Leverage | Measures the maturity of liabilities. |
| `corr_1260d` | Beta | Liquidity | A component of beta; it co-moves with size in the data. |
| `bidaskhl_21d` | Liquidity | Volatility | A spread estimate; volatile, low-priced stocks have wide spreads. |
| `fcf_gr3a` | Profit growth | Cash flow profitability | A growth measure, so it belongs with growth. |
| `o_score` | Distress | Composites | Two composites use the O-score as an input, which causes the overlap. |
| `div_gr3a` | Equity payout | Dividend yield | Dividend growth is a payout-policy measure. |
| `debtlt_gr3a` | Debt issuance | Liability growth | A debt measure; total liabilities include debt. |

### Statistics at release

From `theme_diagnostics.py`, on U.S. stocks from January 1972 to December 2025, using a jkp-data
run of 3 October 2026 (commit 666e896, WRDS data downloaded that day):

- Median of the mean absolute correlation within subthemes with more than one
  characteristic: 0.56.
- Subtheme pairs kept apart although their averages correlate 0.70 or more in absolute
  value: Size and Liquidity, Size and Share price, Share price and Liquidity,
  Cash flow yield and Cash flow profitability, Earnings yield and Equity duration,
  Profitability and Profit margins, Volatility and Lottery, Distress and Composites,
  Profitability and Composites. Each pair measures a different economic idea.
- As a one-off check, the subthemes were compared with a purely statistical grouping
  (Ward clustering into 58 groups on one minus the absolute rank correlation, plus one
  group for characteristics that correlate weakly with all others). Adjusted Rand
  index: 0.48. The statistical clustering is not part of this repository.

## v0 (2023)

The 13 themes of Jensen, Kelly and Pedersen (2023), "Is There a Replication Crisis in
Finance?", *Journal of Finance*, for 153 factors, derived by clustering factor returns. Archived
in `superseded_themes/v0/cluster_labels.csv`. Still used by the theme portfolios in
`jkp portfolio`.
