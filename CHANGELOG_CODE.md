# Code Changelog
This change log tracks software changes to jkp-data — infrastructure, tooling, API, performance improvements, and bug fixes that do not affect the underlying data set. For changes that affect the data, see [CHANGELOG.md](CHANGELOG.md).

## Unreleased

### Infrastructure
- Add Claude Code GitHub Actions workflow ([#190](https://github.com/bkelly-lab/jkp-data/pull/190))
- Add Slurm `set -e` to abort job on failed `jkp build` ([#183](https://github.com/bkelly-lab/jkp-data/pull/183))

### Tests
- Add unit tests for scaling/ratio helpers ([#149](https://github.com/bkelly-lab/jkp-data/pull/149))
- Add comprehensive regression test suite for portfolio construction and pipeline outputs ([#140](https://github.com/bkelly-lab/jkp-data/pull/140))
- Add parity tests for `prc_to_high`, `turnover`, and `zero_trades` ([#119](https://github.com/bkelly-lab/jkp-data/pull/119))

### Refactoring
- Use `pds.lin_reg` for Dimson β estimation ([#148](https://github.com/bkelly-lab/jkp-data/pull/148))
- Consolidate `portfolio.py` helpers into `aux_functions.py` ([#146](https://github.com/bkelly-lab/jkp-data/pull/146))
- Remove `os.chdir`/`os.system`; thread `DataPaths` through pipeline for reproducible path handling ([#132](https://github.com/bkelly-lab/jkp-data/pull/132))
- Simplify `prc_to_high`, `turnover`, and `zero_trades` implementations ([#119](https://github.com/bkelly-lab/jkp-data/pull/119))
- Replace polars-ds Dimson β with pure-Polars closed-form ([#114](https://github.com/bkelly-lab/jkp-data/pull/114))
- Unify pipeline parameters in `config.py` ([#122](https://github.com/bkelly-lab/jkp-data/pull/122))

### API
- Define public API via `__all__` with lazy module `__getattr__` ([#134](https://github.com/bkelly-lab/jkp-data/pull/134))
- Add `__version__` and CLI `--version` flag ([#127](https://github.com/bkelly-lab/jkp-data/pull/127))
- Add `py.typed` marker for PEP 561 type information distribution ([#126](https://github.com/bkelly-lab/jkp-data/pull/126))

### Performance
- Speed up `portfolio.py` with lazy evaluation and parallel collection ([#79](https://github.com/bkelly-lab/jkp-data/pull/79))

### Bug Fixes
- Fix wall-clock date dependency in `aux_functions` that caused non-reproducible results when run on different calendar days ([#133](https://github.com/bkelly-lab/jkp-data/pull/133))
- Fix duplicate file write and nondeterministic processing order in `merge_roll_apply_daily_results` ([#124](https://github.com/bkelly-lab/jkp-data/pull/124))

### Dependencies / CI
- Bump `pyproject.toml` lower bounds to match `uv.lock`; validate bounds in CI ([#139](https://github.com/bkelly-lab/jkp-data/pull/139))

### Documentation
- Add WRDS data-usage disclaimer to README ([#125](https://github.com/bkelly-lab/jkp-data/pull/125))
