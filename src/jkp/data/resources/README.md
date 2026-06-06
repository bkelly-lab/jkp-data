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
classification files under `src/jkp/data/resources/`. None are derived from
WRDS, CRSP, Compustat, or IBES vendor data:

- **`Siccodes49.txt`** — Fama-French 49-industry classification mapping SIC
  codes to industry groups. Sourced from Ken French's
  [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html);
  subject to its own publicly-available terms.
- **`cluster_labels.csv`** — Maps firm characteristics to thematic clusters
  (e.g., "Low Leverage", "Investment", "Size", "Low Risk"). Authored by
  Jensen, Kelly, and Pedersen.
- **`country_classification.xlsx`** — Country-to-region mapping used to
  group outputs. Authored by Jensen, Kelly, and Pedersen.
- **`factor_details.xlsx`** — Metadata describing factor portfolios
  (cluster membership, descriptions). Authored by Jensen, Kelly, and
  Pedersen.
