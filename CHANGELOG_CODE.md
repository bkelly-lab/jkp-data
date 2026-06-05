# Code Changelog
This change log tracks software changes to jkp-data — infrastructure, tooling, API, performance improvements, and bug fixes that do not affect the underlying data set. For changes that affect the data, see [CHANGELOG.md](CHANGELOG.md).

## 04-06-2026
__Changes__:
- Add Claude Code GitHub Actions workflow ([#190](https://github.com/bkelly-lab/jkp-data/pull/190))

## 03-06-2026
__Changes__:
- Add Slurm `set -e` to abort job on failed `jkp build` ([#183](https://github.com/bkelly-lab/jkp-data/pull/183))

## 29-05-2026
__Changes__:
- Use `pds.lin_reg` for Dimson β estimation ([#148](https://github.com/bkelly-lab/jkp-data/pull/148))
- Add unit tests for scaling/ratio helpers ([#149](https://github.com/bkelly-lab/jkp-data/pull/149))

## 27-05-2026
__Changes__:
- Consolidate `portfolio.py` helpers into `aux_functions.py` ([#146](https://github.com/bkelly-lab/jkp-data/pull/146))

## 17-05-2026
__Changes__:
- Remove `os.chdir`/`os.system`; thread `DataPaths` through pipeline for reproducible path handling ([#132](https://github.com/bkelly-lab/jkp-data/pull/132))

## 12-05-2026
__Changes__:
- Simplify `prc_to_high`, `turnover`, and `zero_trades` implementations; add parity tests ([#119](https://github.com/bkelly-lab/jkp-data/pull/119))

## 11-05-2026
__Changes__:
- Add comprehensive regression test suite for portfolio construction and pipeline outputs (494 tests: unit, integration, property, golden) ([#140](https://github.com/bkelly-lab/jkp-data/pull/140))
- Fix signals output never written (`output["signals"]` gated on wrong condition), duplicate `w` column in `pf_signals`, and `UnboundLocalError` for `ret_cutoffs_daily`/`market_daily` when `daily_pf=False` ([#140](https://github.com/bkelly-lab/jkp-data/pull/140))
- Extract `portfolio.py` helpers into named functions; vectorize `cmp_key` and `_build_industry_daily_returns`; hoist `PORTFOLIO_CHARS`, `PORTFOLIO_SETTINGS`, and `ROLLING_DAILY_SPECS` into `config.py` ([#140](https://github.com/bkelly-lab/jkp-data/pull/140))
- Add `hypothesis` and `xlsxwriter` to test dependencies; combine unit and integration/property test runs into a single CI job ([#140](https://github.com/bkelly-lab/jkp-data/pull/140))

## 06-05-2026
__Changes__:
- Define public API via `__all__` with lazy module `__getattr__` ([#134](https://github.com/bkelly-lab/jkp-data/pull/134))
- Add `__version__` and CLI `--version` flag ([#127](https://github.com/bkelly-lab/jkp-data/pull/127))
- Add `py.typed` marker for PEP 561 type information distribution ([#126](https://github.com/bkelly-lab/jkp-data/pull/126))

## 05-05-2026
__Changes__:
- Bump `pyproject.toml` lower bounds to match `uv.lock`; validate bounds in CI ([#139](https://github.com/bkelly-lab/jkp-data/pull/139))

## 04-05-2026
__Changes__:
- Add WRDS data-usage disclaimer to README; fix broken `DATA_LICENSE` relative link in `resources/README.md` ([#125](https://github.com/bkelly-lab/jkp-data/pull/125))

## 03-05-2026
__Changes__:
- Fix wall-clock date dependency in `aux_functions` that caused non-reproducible results when run on different calendar days ([#133](https://github.com/bkelly-lab/jkp-data/pull/133))

## 27-04-2026
__Changes__:
- Replace polars-ds Dimson β with pure-Polars closed-form ([#114](https://github.com/bkelly-lab/jkp-data/pull/114))
- Fix duplicate file write and nondeterministic processing order in `merge_roll_apply_daily_results` ([#124](https://github.com/bkelly-lab/jkp-data/pull/124))
- Unify pipeline parameters in `config.py` ([#122](https://github.com/bkelly-lab/jkp-data/pull/122))

## 22-04-2026
__Changes__:
- Speed up `portfolio.py` with lazy evaluation and parallel collection ([#79](https://github.com/bkelly-lab/jkp-data/pull/79))
