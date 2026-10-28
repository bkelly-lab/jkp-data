# Economic themes

The themes group the JKP characteristics by economic meaning. Each characteristic belongs to
one **subtheme** (for example *Earnings yield*), and each subtheme to one **theme** (for
example *Value*).

## Signs

Characteristics in the same group can point in opposite directions. A high dollar volume means
a liquid stock, but a high Amihud measure means an illiquid one. So `themes.csv` gives each
characteristic three signs, one per level. Each sign answers a yes-or-no question: +1 for yes,
−1 for no.

| Column | Same for all | Question |
|---|---|---|
| `subtheme_sign` | (set per characteristic) | Does a high value mean more of the subtheme? |
| `theme_sign` | characteristics in a subtheme | Does more of the subtheme mean more of the theme? |
| `factor_sign` | characteristics in a theme | Does more of the theme earn higher returns in the literature? |

Multiplying a characteristic by its signs changes what a high value means:

| Multiply by | A high value then means |
|---|---|
| `subtheme_sign` | more of the subtheme |
| `subtheme_sign × theme_sign` | more of the theme |
| `subtheme_sign × theme_sign × factor_sign` | higher returns in the literature: the side the factor buys |

Example: `ami_126d`, the Amihud illiquidity measure, in subtheme Illiquidity and theme Size.

| Sign | Value | Why |
|---|---|---|
| `subtheme_sign` | +1 | a high `ami_126d` means an illiquid stock |
| `theme_sign` | −1 | illiquid stocks are small |
| `factor_sign` | −1 | small stocks earn more |
| Product | +1 | the factor buys stocks with high `ami_126d`: illiquid stocks |

## Naming principles

You can use the grouping at three levels: theme, subtheme and individual characteristic. Each
level has a characteristic and a factor:

- **Characteristic:** a value for each stock and month. At the individual level, it is the
  characteristic itself, such as `ami_126d`. A subtheme characteristic averages the
  cross-sectional ranks of its characteristics, each multiplied by `subtheme_sign`. A theme
  characteristic averages its subtheme characteristics, each multiplied by `theme_sign`, so
  every subtheme counts equally, no matter how many characteristics it has.
- **Factor:** a long-short portfolio that sorts stocks on the characteristic and buys the side
  that earns more in the literature. That side is set by `factor_sign` for a theme, by
  `theme_sign × factor_sign` for a subtheme, and by `subtheme_sign × theme_sign × factor_sign`
  for an individual characteristic.

The characteristic and the factor share one name, the one the asset pricing literature uses
for the factor, such as Value, Investment or Momentum. A high value of the characteristic
means more of what the name says, and the signs say which side the factor buys. As in Fama and
French, the Size factor buys small firms, and the Investment factor buys firms that invest
little. Most names say only what the factor sorts on. Three also give the side, because that
is how the literature names these factors: Low risk, Short-term reversal and Long-term
reversal. For these three, a high value means low risk or a low past return.

For example, `ami_126d` is in the subtheme Illiquidity, which is in the theme Size. A high
value of the Illiquidity characteristic means an illiquid stock, and the Illiquidity factor
buys illiquid stocks. A high value of the Size characteristic means a large firm, and the Size
factor buys small firms.

Names have no "&" or parentheses. A subtheme has its theme's name only if it is the theme's
only subtheme. `economic_themes.py` checks these rules every time it runs.

## Grouping principles

Each subtheme holds characteristics that measure the same thing. We check the grouping in the
data and move clear misfits. When the data disagree with the economics, we keep the economic
grouping and list the exception in the changelog. A score that combines several signals goes
to the subtheme that best matches what it combines.

## Files

- `economic_themes.py` defines the grouping, the signs, the descriptions and the version.
  Running it writes `themes.csv` and `theme_descriptions.csv` to
  `src/jkp/data/resources/themes/`.
- `theme_diagnostics.py` checks the grouping against correlations in the U.S. data.
- `CHANGELOG_THEMES.md` records what changed in each version, and why.
- Earlier versions are kept in `src/jkp/data/resources/themes/superseded_themes/vX.Y/`.
  `v0/` holds the original themes of Jensen, Kelly and Pedersen (2023), which the theme
  portfolios in `jkp portfolio` still use.

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
4. Run `uv run python scripts/themes/economic_themes.py`. It checks the grouping, the signs and
   the naming rules before writing the files.
5. Add a changelog entry and open a pull request.
