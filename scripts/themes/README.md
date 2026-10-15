# Economic themes

The themes group the JKP characteristics by economic meaning. Each characteristic belongs to
one **subtheme** (for example *Earnings yield*), and each subtheme to one **theme** (for
example *Value*).

`themes.csv` has one row per characteristic, with three signs, one per level:

| Column | Constant within | Captures |
|---|---|---|
| `subtheme_sign` | (per characteristic) | +1 if a high value means more of what the subtheme's name says |
| `theme_sign` | subtheme | +1 if more of the subtheme means more of what the theme's name says |
| `factor_sign` | theme | +1 if more of the theme earns higher returns in the literature |

To orient a characteristic, multiply it by:

- `subtheme_sign`, to point with its subtheme;
- `subtheme_sign × theme_sign`, to point with its theme;
- `subtheme_sign × theme_sign × factor_sign`, to point toward higher returns (the long side of
  its long-short factor).

Example: `ami_126d` (Amihud illiquidity) in Liquidity, theme Size. `subtheme_sign` is
−1 (high `ami_126d` is low liquidity), `theme_sign` is +1 (liquid stocks are large), and
`factor_sign` is −1 (small stocks earn more). The product is +1: the factor goes long illiquid
stocks.

- `economic_themes.py` is the source of truth: the grouping, the signs, the descriptions and
  the version. Running it writes `themes.csv` and `theme_descriptions.csv` to
  `src/jkp/data/resources/themes/`.
- `theme_diagnostics.py` checks how well each subtheme holds together in the U.S. data.
- `CHANGELOG_THEMES.md` records what changed in each version, and why.
- Earlier versions are kept in `src/jkp/data/resources/themes/superseded_themes/vX.Y/`.
  `v0/` holds the original themes of Jensen, Kelly and Pedersen (2023), which the theme
  portfolios in `jkp portfolio` still use.

## Naming principles

The data can be used at three levels: theme, subtheme and individual characteristic. At each
level there are two objects:

- **A characteristic:** a value for each stock and month. At the bottom level it is the
  characteristic itself, such as `ami_126d`. A subtheme characteristic averages its
  characteristics after multiplying each by `subtheme_sign`. A theme characteristic averages
  the characteristics of all its subthemes after multiplying each by
  `subtheme_sign × theme_sign`.
- **A factor:** a long-short portfolio that sorts stocks on that characteristic and buys the
  side that earns more in the literature. For a theme characteristic that side is set by
  `factor_sign`, for a subtheme characteristic by `theme_sign × factor_sign`, and for an
  individual characteristic by `subtheme_sign × theme_sign × factor_sign`.

Each object has its own name:

- **The name** (`theme`, `subtheme`) says what a high value of the characteristic measures, so
  it reads directly in a regression. For example, suppose you regress future returns on the
  Short-term returns characteristic and the coefficient is negative. That means stocks with
  high returns last month tend to have lower returns next month.
- **The factor name** (`factor_name`) says which stocks the factor buys. It is the name itself
  when the factor buys high values, and "Low" plus the name when it buys low values (Low risk,
  Low asset growth). The only exceptions are established names from the literature (Momentum,
  Short-term reversal, Long-term reversal, Seasonality) and plain antonyms that read better
  (Small, Young, Illiquidity, Short duration, Stable earnings).

For example, `ami_126d` belongs to the subtheme Liquidity, in the theme Size. A high value of
the Liquidity characteristic means a liquid stock, and its factor, Illiquidity, buys illiquid
stocks. A high value of the Size characteristic means a large firm, and its factor, Small, buys
small firms. In short: names describe what is measured; factor names describe what is bought.

## Versions

Versions are numbered `vMAJOR.MINOR`; cite them as, for example, "JKP themes v1.0". A released
version is never edited.

- **Minor** (v1.0 to v1.1): characteristics are added to existing subthemes or moved between
  them, a sign is changed, or descriptions are reworded.
- **Major** (v1.x to v2.0): a theme or subtheme is added, removed, renamed, split or merged.

A minor version needs one co-author's approval on the pull request; a major version needs all
co-authors.

## Making a new version

1. Copy the current `themes.csv` and `theme_descriptions.csv` to `superseded_themes/vX.Y/`.
2. Edit `economic_themes.py`, including `VERSION`.
3. Run `uv run python scripts/themes/theme_diagnostics.py` (it reads
   `data/processed/characteristics/USA.parquet`; set `RERUN_CORR = True` after a new data
   run). Move clear misfits, or note in the changelog why they stay. Check the signs the data
   disagree with.
4. Run `uv run python scripts/themes/economic_themes.py`.
5. Add a changelog entry and open a pull request.

## Grouping principles

Economics decides: a subtheme collects characteristics that measure the same thing. The data
check it: clear misfits are moved. Where the two disagree, the economic grouping stays and the
changelog records the exception. Names are one plain word or phrase, without "&" or
parentheses.
