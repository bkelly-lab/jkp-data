# Themes changelog

Changes to the economic themes, newest first. See [README.md](README.md) for the versioning rules.

## Planned

- Readable names for individual characteristics, following the naming principles in the
  README (for example, Book-to-market for `be_me`). Until then,
  characteristics are identified by their JKP codes.
- Theme portfolios in `jkp portfolio` built from these themes; they still use v0.
- Short codes for themes and subthemes (for example, three letters such as VAL or MOM) for
  regression tables and figures.

## v1.0 (15-10-2026)

A new grouping by economic meaning, replacing the statistically derived v0 themes. It covers
402 characteristics (v0: 153) in 59 subthemes and 16 themes, and adds a second level: each
characteristic belongs to a subtheme, and each subtheme to a theme. Every characteristic with
a v0 theme has a v1.0 theme.

**Themes:** Value, Profitability, Fundamental momentum, Earnings quality, Investment,
Intangibles, Operating efficiency, Issuance, Low leverage, Low risk, Size, Short-term reversal,
Momentum, Long-term reversal, Seasonality, Quality. See `theme_descriptions.csv` for the
subthemes.

**Names.** Each theme and subtheme has one name, used for both its characteristic and its
factor: the name the asset pricing literature uses for the factor. A high value means more of
what the name says, and the signs give the side the factor buys; the Size factor, for example,
buys small firms. Four names state the side, as the literature does: Low risk, Low leverage,
Short-term reversal and Long-term reversal. All 13 v0 theme names are kept, ten as themes and
Accruals, Debt issuance and Profit growth as subthemes, but not always with the same members. A
subtheme shares the theme's name only when it is the theme's only subtheme.

### Signs

Three signs, one per level: `subtheme_sign` (per characteristic) is +1 if a high value means
more of what the subtheme's name says; `theme_sign` (per subtheme) is +1 if more of the
subtheme means more of what the theme's name says; `factor_sign` (per theme) is +1 if more of
the theme earns higher returns in the literature. A characteristic's factor goes long the side
given by the product of the three. A subtheme characteristic averages the cross-sectional
ranks of its characteristics, each multiplied by `subtheme_sign`; a theme characteristic
averages its subtheme characteristics, each multiplied by `theme_sign`, so every subtheme
counts equally. v0 instead signed each factor by the direction of its return
in the original paper; signing characteristics that way would point characteristics in the same
subtheme in opposite directions in 6 subthemes: in Market leverage, for example, `debt_me`
would be +1 and `netdebt_me` −1, although the two correlate 0.83.

- 48 characteristics have `subtheme_sign` −1, for example `dolvol_126d` in Illiquidity,
  `z_score` in Distress, net payout in Equity issuance and the past returns in Short-term
  reversal and Long-term reversal. The data agree with every sign except for four
  characteristics that barely correlate with their subtheme (`coskew_21d` and the three
  non-annual seasonality measures for years 6-20); these keep the sign their definition implies.
- 17 subthemes have `theme_sign` −1: Equity duration (Value), Accruals (Earnings quality),
  Capital intensity and Labor intensity (Operating efficiency), Equity payout (Issuance),
  Illiquidity (Size), all five subthemes of Low risk, and all of Low leverage except Debt
  maturity and Balance sheet liquidity. Where the data say little, these are economic choices:
  Debt maturity (+1) and Cost of debt (−1) in Low leverage, and the Earnings quality and
  Operating efficiency signs.
- Special items have `theme_sign` +1: they are mostly write-downs and other one-off charges,
  so a high (less negative) value means fewer charges and higher earnings quality. In the data
  they correlate weakly and negatively with Accruals and Tax burden (−0.19 and −0.16 once
  signed).
- `factor_sign` is −1 for Investment, Issuance and Size, and +1 for the other 13 themes.

For 144 of the 153 v0 factors, the product of the three signs equals the direction used in
Jensen, Kelly and Pedersen (2023). For nine it is the opposite, because the characteristic
sits with its subtheme in the data while its own published direction pointed the other way:
`aliq_mat` (Cheapness), `dsale_drec` (Working capital growth), `sti_gr1a` (Financial asset
growth), `debt_me` (Market leverage), `aliq_at` (Balance sheet liquidity), `kz_index`
(Financial constraints), `ni_ivol` (Earnings volatility), `dolvol_var_126d` and
`turnover_var_126d` (Illiquidity).
The factor portfolios produced by `jkp portfolio` still use the published directions.

### Main decisions

- **Naming.** The broad groups are called themes, as in the original JKP paper; the finer
  groups are subthemes. Each group has one name, its factor's name in the literature; separate
  names for the characteristic and the factor were dropped, because two names for one group
  invite confusion. Names have no "&" or parentheses; a group that needed "&" was split.
- **Past returns.** Momentum is split by window: months 2-6 (Momentum 2-6), months 7-12
  (Momentum 7-12), and months 2-12 with similar windows of 9 to 24 months and the 52-week high
  (Momentum 2-12). Returns over months 2-6 and 7-12 correlate only 0.07. Last month alone is
  Short-term reversal; the 2- and 3-month windows that include it go to Momentum 2-6.
  Multi-year windows go to Long-term reversal.
- **Seasonality.** Same-month returns are seasonal at every lag. Averages over the other months
  go with the return window they overlap: year 1 to Momentum, years 2-20 to Long-term reversal.
  Their correlations with everything are close to zero, and their published direction matches
  long-term reversal.
- **Investment.** Asset growth is split by what grows: total and long-lived assets, capex,
  working capital and book equity. Individual balance-sheet items go to the subtheme of their
  item. Long-term investments and financial assets are separate subthemes.
- **Intangibles.** R&D and advertising intensity form their own theme rather than sitting in
  Investment. They are spending on intangible capital, not balance-sheet growth: R&D
  correlates close to zero with the growth subthemes and goes with low earnings yield, high
  cash and long duration, and R&D-intensive firms earn higher returns while fast-growing firms
  earn lower ones.
- **Profitability.** Income profitability (earnings-based profits relative to assets, equity or
  liabilities), Profit margins (profits relative to sales or employees) and Cash flow
  profitability are separate subthemes. Cash-based measures go to Cash flow profitability
  whatever their denominator, including cash-based operating profitability, just as
  cash-based operating profit to price is a cash flow yield. Free cash flow to operating cash
  flow measures how much cash flow goes to capital spending, so it sits in Capital intensity.
  All profit-growth horizons form one subtheme.
- **Value.** Book equity, assets and sales relative to market value form Cheapness. Gross profit
  to price is an earnings yield; cash-based operating profit to price is a cash flow yield.
  Equity duration is its own subtheme.
- **Low leverage.** Book and market leverage are separate subthemes. They correlate 0.87, but
  market leverage also moves with value (0.47, against 0.14 for book leverage), because the
  share price is in the denominator. This mirrors the split between yields and profitability.
  Debt terms are split into maturity and cost.
- **Issuance.** Most of the theme is net issuance of equity, debt or both; payout counts as
  the reverse. Changes in equity and debt issuance join their issuance subthemes. Net total
  financing is External financing. Growth in total liabilities, which includes debt, is
  Liability growth; growth in payables, accrued items and taxes stays with Investment as
  Operating payables growth.
- **Low risk.** Lottery is separate from Volatility despite a correlation of 0.72. Market
  correlation sits with Beta.
- **Composite scores.** A score that combines several signals goes to the subtheme that best
  matches what it combines. QMJ, the Piotroski F-score and the Stambaugh–Yuan performance
  mispricing score form Quality. QMJ's parts go where they fit best: profitability to Income
  profitability, growth to Profit growth and safety to Distress. The Stambaugh–Yuan
  management score goes to Asset growth: it is built mostly from issuance, accruals and
  asset-growth signals, and a high score means low growth. A separate Mispricing subtheme was
  rejected because the two Stambaugh–Yuan scores barely correlate (−0.05).
- **Fundamental momentum** is the theme for earnings surprises and profit growth, the usual
  meaning of the term.
- **Operating costs** over assets (Novy-Marx's operating leverage) stay in Asset turnover;
  they correlate 0.86 with asset turnover.

### Exceptions kept for economic reasons

These characteristics fit another subtheme clearly better in the data (by at least 0.15, at
0.40 or more) but stay where they are.

| Characteristic | Subtheme | Fits better with | Why it stays |
|---|---|---|---|
| `fcf_gr3a` | Profit growth | Cash flow profitability | A growth measure, so it belongs with growth. |
| `div_gr3a` | Equity payout | Dividend yield | Dividend growth is a payout-policy measure. |
| `debtlt_gr3a` | Debt issuance | Liability growth | A debt measure; total liabilities include debt. |
| `cl_lt` | Debt maturity | Book leverage | Measures the maturity of liabilities. |
| `corr_1260d` | Beta | Illiquidity | A component of beta; in the data it moves with size and liquidity. |
| `bidaskhl_21d` | Illiquidity | Volatility | A spread estimate; volatile, low-priced stocks have wide spreads. |
| `seas_1_1an` | Seasonality | Momentum 7-12 | A same-month return; its overlap with months 7-12 is mechanical. |
| `mispricing_perf` | Quality | Income profitability | It combines profitability with safety and momentum, as quality scores do. |

### Statistics at release

From `theme_diagnostics.py`, on U.S. stocks from January 1972 to December 2025, using a jkp-data
run of 3 October 2026 (commit 666e896, WRDS data downloaded that day):

- Median of the mean absolute correlation within subthemes with more than one
  characteristic: 0.58.
- Subtheme pairs kept apart although their averages correlate 0.70 or more in absolute
  value: Firm size and Illiquidity, Firm size and Share price, Share price and Illiquidity,
  Cash flow yield and Cash flow profitability, Earnings yield and Equity duration,
  Income profitability and Profit margins, Income profitability and Quality, Volatility and
  Lottery, Book leverage and Market leverage. Each pair measures a different economic idea.
- As a one-off check, the subthemes were compared with a purely statistical grouping
  (Ward clustering into 58 groups on one minus the absolute rank correlation, plus one
  group for characteristics that correlate weakly with all others). Adjusted Rand
  index: 0.48. The statistical clustering is not part of this repository.

## v0 (2023)

The 13 themes of Jensen, Kelly and Pedersen (2023), "Is There a Replication Crisis in
Finance?", *Journal of Finance*, for 153 factors, derived by clustering factor returns. Archived
in `superseded_themes/v0/cluster_labels.csv`. Still used by the theme portfolios in
`jkp portfolio`.
