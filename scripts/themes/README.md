# Economic themes

The themes group the JKP characteristics by economic meaning. Each characteristic belongs to
one **sub-theme** (for example *Earnings yield*), and each sub-theme to one **theme** (for
example *Value*).

`themes.csv` has one row per characteristic, with three signs, one per level:

| Column | Constant within | Captures |
|---|---|---|
| `sub_theme_sign` | (per characteristic) | +1 if a high value means more of what the sub-theme's name says |
| `theme_sign` | sub-theme | +1 if more of the sub-theme means more of what the theme's name says |
| `factor_sign` | theme | +1 if more of the theme earns higher returns in the literature |

To orient a characteristic, multiply it by:

- `sub_theme_sign`, to point with its sub-theme;
- `sub_theme_sign × theme_sign`, to point with its theme;
- `sub_theme_sign × theme_sign × factor_sign`, to point toward higher returns (the long side of
  its long-short factor).

Example: `ami_126d` (Amihud illiquidity) in Liquidity, theme Size. `sub_theme_sign` is
−1 (high `ami_126d` is low liquidity), `theme_sign` is +1 (liquid stocks are large), and
`factor_sign` is −1 (small stocks earn more). The product is +1: the factor goes long illiquid
stocks.

The first two signs describe what a characteristic measures; only `factor_sign` is about
returns. In the same spirit, theme and sub-theme names describe the characteristics, and
`theme_descriptions.csv` gives each one's `factor_name`, which says which side its long-short
factor goes long: the name itself when it goes long high values, and "Low" plus the name when
it goes long low values (Low risk, Low asset growth), with a few plain antonyms instead (Small, Young,
Illiquidity, Short duration, Stable earnings). Past returns keep
their established names: Short-term returns, for example, is the characteristic behind the
Short-term reversal factor.

- `economic_themes.py` is the source of truth: the grouping, the signs, the descriptions and
  the version. Running it writes `themes.csv` and `theme_descriptions.csv` to
  `src/jkp/data/resources/themes/`.
- `theme_diagnostics.py` checks how well each sub-theme holds together in the U.S. data.
- `CHANGELOG_THEMES.md` records what changed in each version, and why.
- Earlier versions are kept in `src/jkp/data/resources/themes/superseded_themes/vX.Y/`.
  `v0/` holds the original themes of Jensen, Kelly and Pedersen (2023), which the theme
  portfolios in `jkp portfolio` still use.

## Versions

Versions are numbered `vMAJOR.MINOR`; cite them as, for example, "JKP themes v1.0". A released
version is never edited.

- **Minor** (v1.0 to v1.1): characteristics are added to existing sub-themes or moved between
  them, a sign is changed, or descriptions are reworded.
- **Major** (v1.x to v2.0): a theme or sub-theme is added, removed, renamed, split or merged.

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

## Principles

Economics decides: a sub-theme collects characteristics that measure the same thing. The data
check it: clear misfits are moved. Where the two disagree, the economic grouping stays and the
changelog records the exception. Names are one plain word or phrase, without "&" or
parentheses.
