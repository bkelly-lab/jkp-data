# Contributing

This document explains how to set up a development environment and contribute code to the jkp-data repository.

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/bkelly-lab/jkp-data.git
   cd jkp-data
   ```

2. **Install dependencies**

   Install all development dependencies using your preferred Python package manager. With [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync --group dev --group test --group lint
   ```

   This installs the base dependencies plus tools for testing (pytest, pytest-cov) and linting (ruff, pyright).

## Running Tests

Tests live in the `tests/` directory. Run them with:

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_expressions.py
```

Coverage is enabled by default. After each run, you'll see which lines of code were exercised by tests.

For detailed guidance on writing tests, see [tests/README.md](tests/README.md).

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Before submitting a PR:

```bash
# Check for lint errors
ruff check src/jkp_data/ tests/

# Auto-fix what can be fixed
ruff check --fix src/jkp_data/ tests/

# Check formatting
ruff format --check src/jkp_data/ tests/

# Auto-format
ruff format src/jkp_data/ tests/
```

## Type Checking

We use [pyright](https://github.com/microsoft/pyright) for static type checking:

```bash
pyright src/jkp_data/
```

Pyright runs in CI but does not currently block PRs. It surfaces potential bugs and type inconsistencies. If you see errors related to your changes, consider fixing them.

## Pull Request Process

1. **Create a branch** for your changes

2. **Write tests** for new functionality or bug fixes. See [tests/README.md](tests/README.md) for guidance.

3. **Run tests locally** before pushing:
   ```bash
   pytest
   ```

4. **Run the linter** and fix any issues:
   ```bash
   ruff check src/jkp_data/ tests/
   ruff format --check src/jkp_data/ tests/
   ```

5. **Push and open a PR**. The CI pipeline will run automatically.

6. **All checks must pass** before merging. This includes:
   - Ruff lint and format checks
   - All unit tests passing on Python 3.11 and 3.12

## Project Structure

```
jkp-data/
├── src/
│   └── jkp_data/            # Main source package
│       ├── __init__.py
│       ├── cli.py           # CLI entry point (jkp command)
│       ├── aux_functions.py # Core utility functions and characteristics
│       ├── main.py          # Pipeline orchestration
│       ├── portfolio.py     # Factor portfolio construction
│       ├── config.py        # Pipeline configuration
│       └── wrds_credentials.py # WRDS credential management
├── tests/                   # Test suite
│   ├── conftest.py          # Shared fixtures
│   └── unit/                # Unit tests
├── data/                    # Data directory (not in git)
└── documentation/           # Release notes and docs
```

## Questions

If you're unsure about something, open an issue to discuss before investing significant effort.
