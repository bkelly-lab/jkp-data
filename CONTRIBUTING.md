# Contributing

This document explains how to set up a development environment and contribute code to the jkp-data repository.

## Development Setup

1. **Clone the repository**

   ```bash
   git clone git@github.com:bkelly-lab/jkp-data.git
   cd jkp-data
   ```

2. **Install dependencies**

   We use [uv](https://docs.astral.sh/uv/) for dependency management. Install all development dependencies:

   ```bash
   uv sync --group dev --group test --group lint
   ```

   This installs the base dependencies plus tools for testing (pytest, pytest-cov) and linting (ruff, pyright).

## Running Tests

Tests live in the `tests/` directory. Run them with:

```bash
# Run all tests with coverage
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/unit/test_expressions.py
```

Coverage is enabled by default. After each run, you'll see which lines of code were exercised by tests.

For detailed guidance on writing tests, see [tests/README.md](tests/README.md).

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Before submitting a PR:

```bash
# Check for lint errors
uv run ruff check code/ tests/

# Auto-fix what can be fixed
uv run ruff check --fix code/ tests/

# Check formatting
uv run ruff format --check code/ tests/

# Auto-format
uv run ruff format code/ tests/
```

## Type Checking

We use [pyright](https://github.com/microsoft/pyright) for static type checking:

```bash
uv run pyright code/
```

Pyright runs in CI but does not currently block PRs. It surfaces potential bugs and type inconsistencies. If you see errors related to your changes, consider fixing them.

## Pull Request Process

1. **Create a branch** for your changes

2. **Write tests** for new functionality or bug fixes. See [tests/README.md](tests/README.md) for guidance.

3. **Run tests locally** before pushing:
   ```bash
   uv run pytest
   ```

4. **Run the linter** and fix any issues:
   ```bash
   uv run ruff check code/ tests/
   uv run ruff format --check code/ tests/
   ```

5. **Push and open a PR**. The CI pipeline will run automatically.

6. **All checks must pass** before merging. This includes:
   - Ruff lint and format checks
   - All unit tests passing on Python 3.11 and 3.12

## Project Structure

```
jkp-data/
├── code/                # Main source code
│   ├── aux_functions.py # Core utility functions and characteristics
│   ├── main.py          # Pipeline entry point
│   └── portfolio.py     # Factor portfolio construction
├── tests/               # Test suite
│   ├── conftest.py      # Shared fixtures
│   └── unit/            # Unit tests
├── data/                # Data directory (not in git)
└── documentation/       # Release notes and docs
```

## Questions

If you're unsure about something, open an issue to discuss before investing significant effort.
