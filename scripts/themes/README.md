# Economic themes

The themes group the JKP characteristics by economic meaning. Each characteristic belongs to
one **subtheme** (for example *Earnings yield*), and each subtheme to one **theme** (for
example *Value*).

## Signs

Characteristics in the same group do not all point the same way: a high dollar volume means a
liquid stock, but a high Amihud measure means an illiquid one. `themes.csv` therefore gives
each characteristic three signs, one per level. Each answers a yes-or-no question, with +1 for
yes and −1 for no:

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

Example: `ami_126d`, the Amihud illiquidity measure, in subtheme Liquidity and theme Size.

| Sign | Value | Why |
|---|---|---|
| `subtheme_sign` | −1 | a high `ami_126d` means low liquidity |
| `theme_sign` | +1 | liquid stocks are large |
| `factor_sign` | −1 | small stocks earn more |
| Product | +1 | the factor buys stocks with high `ami_126d`: illiquid stocks |

## Naming principles

The data can be used at three levels: theme, subtheme and individual characteristic. At each
level there are two objects:

- **A characteristic:** a value for each stock and month. At the bottom level it is the
  characteristic itself, such as `ami_126d`. A subtheme characteristic averages the
  cross-sectional ranks of its characteristics, each multiplied by `subtheme_sign`. A theme
  characteristic averages its subtheme characteristics, each multiplied by `theme_sign`, so
  every subtheme counts equally however many characteristics it has.
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
  Short-term reversal, Long-term reversal and Seasonality, plus momentum windows such as
  Momentum 2-6) and plain antonyms that read better (Small, Small firms, Young, Illiquidity,
  Short duration and Stable earnings).

For example, `ami_126d` belongs to the subtheme Liquidity, in the theme Size. A high value of
the Liquidity characteristic means a liquid stock, and its factor, Illiquidity, buys illiquid
stocks. A high value of the Size characteristic means a large firm, and its factor, Small, buys
small firms. In short: names describe what is measured; factor names describe what is bought.

Names are plain words or phrases, without "&" or parentheses. A subtheme shares its theme's
name, and its factor the theme factor's name, only when it is the theme's only subtheme.
`economic_themes.py` checks these rules every time it runs.

## Grouping principles

Economics decides: a subtheme collects characteristics that measure the same thing. The data
check it: clear misfits are moved. Where the two disagree, the economic grouping stays and the
changelog records the exception. A score that combines several signals goes to the subtheme
that best matches what it combines.

## Files

- `economic_themes.py` is the source of truth: the grouping, the signs, the descriptions and
  the version. Running it writes `themes.csv` and `theme_descriptions.csv` to
  `src/jkp/data/resources/themes/`.
- `theme_diagnostics.py` checks how well each subtheme holds together in the U.S. data.
- `CHANGELOG_THEMES.md` records what changed in each version, and why.
- Earlier versions are kept in `src/jkp/data/resources/themes/superseded_themes/vX.Y/`.
  `v0/` holds the original themes of Jensen, Kelly and Pedersen (2023), which the theme
  portfolios in `jkp portfolio` still use.

## Versions

Versions are numbered `vMAJOR.MINOR`; cite them as, for example, "JKP themes v1.0". A released
version is never edited.

- **Minor** (v1.0 to v1.1): characteristics are added to existing subthemes or moved between
  them, a sign is changed, or descriptions are reworded.
- **Major** (v1.x to v2.0): a theme or subtheme is added, removed, renamed (including its factor
  name), split or merged.

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
