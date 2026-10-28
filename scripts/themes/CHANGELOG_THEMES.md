# Themes changelog

Changes to the economic themes, newest first. See [README.md](README.md) for the versioning rules.

## Planned

- Readable names for individual characteristics, following the naming principles in the
  README (for example, Book-to-market for `be_me`). Until then,
  characteristics are identified by their JKP codes.
- Theme portfolios in `jkp portfolio` built from these themes; they still use v0.
- Short codes for themes and subthemes (for example, three letters such as VAL or MOM) for
  regression tables and figures.

## v1.0 (unreleased)

A new grouping by economic meaning. It replaces the v0 themes, which were derived from
statistics alone. It covers 402 characteristics (v0: 153) in 59 subthemes and 16 themes. The
subthemes are a new, second level: each characteristic belongs to a subtheme, and each
subtheme to a theme. Every characteristic with a v0 theme has a v1.0 theme. A `note` column
in both files gives the reason for signs that may surprise and for the exceptions below.

**Themes:** Value, Profitability, Fundamental momentum, Earnings quality, Investment,
Intangibles, Operating efficiency, Issuance, Financial strength, Low risk, Size, Short-term
reversal, Momentum, Long-term reversal, Seasonality, Quality. See `theme_descriptions.csv` for
the subthemes.

**Names.** Each theme and subtheme has one name, used for both its characteristic and its
factor. It is the name the asset pricing literature uses for the factor. A high value means
more of what the name says, and the signs say which side the factor buys. The Size factor, for
example, buys small firms. Three names also give the side, as in the literature: Low risk,
Short-term reversal and Long-term reversal. Twelve of the 13 v0 theme names are still used,
nine as themes and three as subthemes (Accruals, Debt issuance and Profit growth), but some now
hold different characteristics. The thirteenth, Low Leverage, is replaced by Financial
strength. A subtheme has its theme's name only if it is the theme's only subtheme.

### Signs

There is one sign per level:

- `subtheme_sign` (per characteristic) is +1 if a high value means more of what the subtheme's
  name says.
- `theme_sign` (per subtheme) is +1 if more of the subtheme means more of what the theme's name
  says.
- `factor_sign` (per theme) is +1 if more of the theme earns higher returns in the literature.

A characteristic's factor buys the side given by the product of the three signs. A subtheme
characteristic averages the cross-sectional ranks of its characteristics, each multiplied by
`subtheme_sign`. A theme characteristic averages its subtheme characteristics, each multiplied
by `theme_sign`, so every subtheme counts equally.

- 48 characteristics have `subtheme_sign` −1, for example `dolvol_126d` in Illiquidity,
  `z_score` in Distress, net payout in Equity issuance and the past returns in Short-term
  reversal and Long-term reversal. The data agree with every sign except for four
  characteristics that barely correlate with their subtheme: `coskew_21d` and the three
  non-annual seasonality measures for years 6-20. These keep the sign their definition implies.
- 17 subthemes have `theme_sign` −1: Equity duration (Value), Accruals (Earnings quality),
  Capital intensity and Labor intensity (Operating efficiency), Equity payout (Issuance),
  Illiquidity (Size), all five subthemes of Low risk, and all of Financial strength except
  Debt maturity and Balance sheet liquidity. Where the data say little, we chose the sign on
  economic grounds: Debt maturity (+1) and Cost of debt (−1) in Financial strength, and the
  signs in Earnings quality and Operating efficiency.
- The subtheme Special items has `theme_sign` +1. Special items are mostly write-downs and other
  one-off charges, so a high (less negative) value means fewer charges and higher earnings
  quality. After signing, they correlate weakly and negatively with Accruals and Tax burden
  (−0.19 and −0.16).
- `factor_sign` is −1 for Investment, Issuance and Size, and +1 for the other 13 themes.

In v0, each factor was signed by the direction of its return in the original paper. Signing
characteristics that way would make characteristics in the same subtheme point in opposite
directions in 6 subthemes. In Market leverage, for example, `debt_me` would be +1 and
`netdebt_me` −1, although the two correlate 0.83.

For 144 of the 153 v0 factors, the product of the three signs equals the direction used in
Jensen, Kelly and Pedersen (2023). For nine it is the opposite. Each of the nine moves with its
subtheme in the data, but its published direction pointed the other way: `aliq_mat`
(Cheapness), `dsale_drec` (Working capital growth), `sti_gr1a` (Financial asset growth),
`debt_me` (Market leverage), `aliq_at` (Balance sheet liquidity), `kz_index` (Financial
constraints), `ni_ivol` (Earnings volatility), `dolvol_var_126d` and `turnover_var_126d`
(Illiquidity). The factor portfolios produced by `jkp portfolio` still use the published
directions.

### Main decisions

- **Naming.** The broad groups are called themes, as in the original JKP paper, and the finer
  groups subthemes. Each group has one name, the name of its factor in the literature. We
  dropped separate names for the characteristic and the factor, because two names for one group
  are confusing. Names don't join two groups with "&", and have no parentheses. A group that
  needed "&" was split. The theme with leverage, debt terms, liquidity and distress is called
  Financial strength, because only two of its eight subthemes are leverage ratios. So the v0
  name Low Leverage is no longer a theme name. Only 5 of the 11 v0 Low Leverage characteristics
  are in Financial strength. The others moved to Intangibles, Low risk, Size and Value. Of the
  41 characteristics in Financial strength, 31 were not in v0.
- **Past returns.** Momentum is split by window: months 2-6 (Momentum 2-6), months 7-12
  (Momentum 7-12), and months 2-12 together with similar windows of 9 to 24 months and the
  52-week high (Momentum 2-12). Returns over months 2-6 and 7-12 correlate only 0.07. Last month
  alone is Short-term reversal. The 2- and 3-month windows that include last month go to
  Momentum 2-6. Multi-year windows go to Long-term reversal.
- **Seasonality.** Returns in the same calendar month go to Seasonality, whatever the lag.
  Average returns in the other months go with the return window they overlap: year 1 to
  Momentum, years 2-20 to Long-term reversal. They barely correlate with any other
  characteristic, and their published direction matches long-term reversal.
- **Investment.** Asset growth is split by what grows: total and long-lived assets, capex,
  working capital and book equity. Growth in a single balance sheet item goes to that item's
  subtheme. Long-term investments and financial assets are separate subthemes.
- **Intangibles.** R&D and advertising intensity form their own theme. They measure spending on
  intangible capital, while the Investment subthemes measure balance sheet growth. R&D barely
  correlates with the growth subthemes and goes with low earnings yield, high cash and long
  duration. R&D-intensive firms also earn higher returns, while fast-growing firms earn lower
  ones.
- **Profitability.** Income profitability (earnings-based profits relative to assets, equity or
  liabilities), Profit margins (profits relative to sales or employees) and Cash flow
  profitability are separate subthemes. Cash-based measures, including cash-based operating
  profitability, go to Cash flow profitability whatever their denominator. In the same way,
  cash-based operating profit to price is a cash flow yield. Free cash flow to operating cash
  flow measures how much cash flow goes to capital spending, so it is in Capital intensity. All
  profit-growth horizons form one subtheme.
- **Value.** Book equity, assets and sales relative to market value form Cheapness. Gross profit
  to price is an earnings yield, and cash-based operating profit to price is a cash flow yield.
  Equity duration is its own subtheme.
- **Financial strength.** Book and market leverage are separate subthemes. They correlate
  0.87, but market leverage also moves with value (0.47, against 0.14 for book leverage),
  because the share price is in the denominator. Yields and profitability are split for the
  same reason. Debt terms are split into maturity and cost.
- **Issuance.** Most of the theme is net issuance of equity, debt or both. Payout counts as
  negative issuance. Changes in equity and debt issuance join their issuance subthemes. Net
  total financing is External financing. Growth in total liabilities, which include debt, is
  Liability growth. Growth in payables, accrued items and taxes stays in Investment as
  Operating payables growth.
- **Low risk.** Lottery is separate from Volatility, although the two correlate 0.72. Market
  correlation is in Beta.
- **Composite scores.** A score that combines several signals goes to the subtheme that best
  matches what it combines. QMJ, the Piotroski F-score and the Stambaugh–Yuan performance
  mispricing score form Quality. QMJ's parts go where they fit best: profitability to Income
  profitability, growth to Profit growth and safety to Distress. The Stambaugh–Yuan management
  score goes to Asset growth. It is built mostly from issuance, accruals and asset-growth
  signals, and a high score means low growth. We did not make a separate Mispricing subtheme,
  because the two Stambaugh–Yuan scores barely correlate (−0.05).
- **Fundamental momentum** is the theme for earnings surprises and profit growth, the usual
  meaning of the term.
- **Operating costs** over assets (Novy-Marx's operating leverage) stay in Asset turnover. They
  correlate 0.86 with asset turnover.

### Exceptions kept for economic reasons

These characteristics fit another subtheme clearly better in the data: their fit there is 0.40
or more, and at least 0.15 higher than with their own subtheme. They stay where they are, and
the reason is also in their `note` in `themes.csv`.

| Characteristic | Subtheme | Fits better with | Why it stays |
|---|---|---|---|
| `fcf_gr3a` | Profit growth | Cash flow profitability | It measures growth, not the level of cash flow. |
| `div_gr3a` | Equity payout | Dividend yield | Dividend growth is a payout-policy measure. |
| `debtlt_gr3a` | Debt issuance | Liability growth | It measures debt growth. Total liabilities include debt, so the overlap is expected. |
| `cl_lt` | Debt maturity | Book leverage | It measures the maturity of liabilities. |
| `corr_1260d` | Beta | Illiquidity | A component of beta. In the data it moves with size and liquidity. |
| `bidaskhl_21d` | Illiquidity | Volatility | A bid-ask spread estimate. Volatile, low-priced stocks have wide spreads. |
| `seas_1_1an` | Seasonality | Momentum 7-12 | Last year's return in the same month, which is also one of months 7-12. |
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
  Lottery, Book leverage and Market leverage. The two subthemes in each pair measure different
  things.
- As a one-off check, the subthemes were compared with a statistical grouping (Ward clustering
  into 58 groups on one minus the absolute rank correlation, plus one group for characteristics
  that correlate weakly with all others). Adjusted Rand index: 0.48. The statistical clustering
  is not part of this repository.

## v0 (2023)

The 13 themes of Jensen, Kelly and Pedersen (2023), "Is There a Replication Crisis in
Finance?", *Journal of Finance*, for 153 factors, derived by clustering factor returns. Archived
in `superseded_themes/v0/`. `cluster_labels.csv` gives each factor's theme, and
`factor_directions.csv` gives the direction each factor was signed by, copied from
`factor_details.xlsx`. Still used by the theme portfolios in `jkp portfolio`.
