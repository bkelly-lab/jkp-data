# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo generates the Global Factor Data: 406 stock characteristics and their associated factor portfolios, based on "Is there a Replication Crisis in Finance?" by Jensen, Kelly, and Pedersen (Journal of Finance, 2023). It downloads data from WRDS (Wharton Research Data Services), computes characteristics from CRSP and Compustat sources, and constructs factor portfolios.

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run linting
uv run ruff check code/ tests/
uv run ruff format --check code/ tests/

# Run type checking (informational, not blocking in CI)
uv run pyright code/

# Run all unit tests
uv run pytest tests/unit/

# Run a specific test file or class
uv run pytest tests/unit/test_expressions.py
uv run pytest tests/unit/test_expressions.py::TestSumSas

# Run tests with coverage
uv run pytest

# Run the full pipeline (requires WRDS credentials and ~450 GB RAM)
uv run python code/main.py          # stock returns and firm characteristics
uv run python code/portfolio.py     # factor returns (run after main.py)
```

## Architecture

The pipeline has two entry points that run sequentially:

**`code/main.py`** produces stock returns and firm characteristics:
1. Download raw data from WRDS (CRSP, Compustat)
2. Prepare and merge data sources (augmented monthly stock file, market cap/trading info)
3. Classify stocks by industry (Fama-French 49) and size (NYSE quintile cutoffs)
4. Compute characteristics from accounting and market data
5. Calculate rolling daily metrics (volatility, beta, skewness) across 21d/126d/252d/1260d windows
6. Save outputs as parquet files to `data/processed/`

**`code/portfolio.py`** constructs factor portfolios from the characteristics output by `main.py`, using ECDF ranking and value-weighting by market cap.

### Key source files

- `code/main.py` — Pipeline orchestration; calls functions from `aux_functions` in sequence
- `code/aux_functions.py` — Core library: all characteristic calculations, data transformations, and I/O utilities
- `code/portfolio.py` — Standalone factor portfolio construction script
- `code/wrds_credentials.py` — Keyring-based WRDS credential management

### Data flow

Raw WRDS data → `data/raw/` → intermediate processing in `data/interim/` → final outputs in `data/processed/` (subdirectories: `characteristics/`, `portfolios/`, `return_data/`, `accounting_data/`, `other_output/`).

Static reference data (`data/cluster_labels.csv`, `data/country_classification.xlsx`, `data/factor_details.xlsx`) is checked into the repo and used by the pipeline.

## Code Conventions

- **Polars, not Pandas.** The codebase uses Polars with lazy evaluation throughout. Always use `import polars as pl` and Polars APIs.
- **`pl.col()`, not bare `col()`.** Use `pl.col(...)` in new code. Existing code uses `from polars import col` with bare `col(...)`, but the standard Polars convention is the namespaced form.
- **Line length:** 100 characters (ruff configured)
- **Ruff rules:** E, W, F, I (isort), B (bugbear), C4, UP, SIM — with E501, B008, SIM102, SIM108 ignored
- **Python target:** 3.11+

## Development Guidelines

**Modularity & structure:**
- New pipeline functions should be added to `aux_functions.py` and called from `main.py`
- Use the `@measure_time` decorator on pipeline-level functions
- Use `collect_and_write()` for the lazy→eager→parquet workflow

**Docstring format** (the codebase uses a consistent three-part format):
```
Description:
    What the function does.
Steps:
    1) ...
Output:
    What is written/returned.
```

**Polars patterns:**
- Use `pl.scan_parquet()` (lazy) for reading, not `pl.read_parquet()` unless full data is needed eagerly
- Return Polars expressions from helper functions (like `safe_div`)
- Use `safe_div()` for any division — never raw `/`
- **Do not use `sum_sas` or `sub_sas` in new code.** These are legacy functions that replicate SAS null semantics (treat null as 0 when any input is non-null). They exist for backward compatibility in existing characteristic calculations. New code should use standard Polars null propagation, or explicit `pl.coalesce()` when null-as-zero behavior is needed.
- Prefer Polars over DuckDB/Ibis for new code. Existing code uses DuckDB via Ibis for some complex SQL aggregations, but new work should use Polars expressions and lazy evaluation where possible.

**Type hints:**
- Add type annotations to new functions (parameters and return types), even though older code in `aux_functions.py` lacks them

**Testing:**
- New functions should have corresponding unit tests in `tests/unit/`; follow existing tests as examples for patterns and conventions

**DRY / reuse:**
- Check `aux_functions.py` for existing helpers before writing new ones (especially `safe_div`, `fl_none`, `bo_false`)
- Don't duplicate characteristic calculations that already exist

## Testing

- Tests live in `tests/unit/` with shared fixtures in `tests/conftest.py`
- Markers: `unit`, `integration`, `methodology`, `regression`, `expensive`, `wrds`
- `ToleranceSpec` class provides named tolerance levels (TIGHT, STANDARD, LOOSE, VERY_LOOSE) for numerical comparisons
- `assert_series_equal()` fixture handles NaN-aware Polars series comparison
- `make_dataframe` factory fixture creates test DataFrames; `temp_data_dir` provides a temp directory with standard pipeline subdirectory structure
- CI runs lint first, then unit tests on Python 3.11 and 3.12

## Development Workflow

Follow these steps when making code changes:

1. **Find or open an issue** — Search existing GitHub issues (`gh issue list`) for a matching issue. If none exists, create one (`gh issue create`) describing the problem or feature before starting work.

2. **Create a branch** — Branch from `main` with a descriptive name:
   ```bash
   git checkout -b <topic>  # e.g. fix-beta-calculation, add-momentum-char
   ```

3. **Implement the change** — Follow the conventions in this file (Code Conventions, Development Guidelines).

4. **Verify before opening a PR:**
   - All existing tests pass: `uv run pytest tests/unit/`
   - New functions have corresponding unit tests (use the test-scaffolder agent if needed)
   - Code passes lint and format checks: `uv run ruff check code/ tests/ && uv run ruff format --check code/ tests/`
   - Changed code follows project conventions: run `/review-code`

5. **Open a pull request** — Reference the issue in the PR description. The PR template will guide you through the required sections.
