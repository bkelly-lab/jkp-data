# Data Directory

Data files in this directory authored by Jensen, Kelly, and Pedersen are
licensed under [CC BY-NC 4.0](https://github.com/bkelly-lab/jkp-data/blob/main/DATA_LICENSE)
(Creative Commons Attribution-NonCommercial 4.0). When this file appears in
your output directory, it accompanies generated factor portfolios, stock
returns, and firm characteristics; those outputs are also subject to your
WRDS and vendor license terms.

## Usage

You are free to share and adapt this data for non-commercial purposes,
provided you give appropriate credit.

## Citation

If you use this data, please cite:

> Jensen, T. I., Kelly, B. T., & Pedersen, L. H. (2023).
> Is There a Replication Crisis in Finance?
> *Journal of Finance*. https://doi.org/10.1111/jofi.13249

## Bundled reference files (source distribution)

The `jkp-data` package source ships with the following reference
files under `src/jkp/data/resources/`. None are derived from
WRDS, CRSP, Compustat, or IBES vendor data:

- **`Siccodes49.txt`** — Fama-French 49-industry classification mapping SIC
  codes to industry groups. Sourced from Ken French's
  [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html);
  subject to its own publicly-available terms.
- **`characteristics.csv`** — The names of the characteristics in the
  `characteristics/` output files, one per row. The other columns in those
  files are not characteristics: for example identifiers (`id`, `gvkey`), the
  month's returns (`ret`, `ret_exc`) and industry codes (`sic`, `ff49`).
  Authored by Jensen, Kelly, and Pedersen.
- **`fx_cad_1960_1982.csv`** — Month-end Canadian dollar exchange rates (USD
  per CAD) from January 1960 to January 1982, used for Canadian-dollar values
  dated before Compustat's exchange rates begin in February 1982. Source:
  OECD Main Economic Indicators, retrieved from FRED, Federal Reserve Bank of
  St. Louis ([series CCUSSP01CAM650N](https://fred.stlouisfed.org/series/CCUSSP01CAM650N)).
  FRED dates each month by its first day; the file dates it by its last day.
  Copyright OECD; subject to the OECD's terms of use.
- **`cluster_labels.csv`** — Maps firm characteristics to thematic clusters
  (e.g., "Low Leverage", "Investment", "Size", "Low Risk"). Authored by
  Jensen, Kelly, and Pedersen.
- **`themes/themes.csv`** and **`themes/theme_descriptions.csv`** — The
  current version of the economic themes: each characteristic's theme,
  subtheme and signs, a description for each theme and subtheme, and notes
  explaining signs and placements that may surprise. Earlier versions are kept in
  `themes/superseded_themes/`. Authored by Jensen, Kelly, and Pedersen. See
  `scripts/themes/` in the repository for the principles, versioning rules and
  changelog.
- **`country_classification.xlsx`** — Country-to-region mapping used to
  group outputs. Authored by Jensen, Kelly, and Pedersen.
- **`factor_details.xlsx`** — Metadata describing factor portfolios
  (cluster membership, descriptions). Authored by Jensen, Kelly, and
  Pedersen.
