# Tests

This directory contains the test suite for the jkp-data repository, which replicates the factor data from Jensen, Kelly, and Pedersen (2023), "Is There a Replication Crisis in Finance?"

Testing is particularly important for this codebase because it implements financial calculations where silent numerical errors can propagate invisibly. A small error in portfolio weights or return calculations might not cause the code to crash, but it could invalidate research conclusions. The tests in this repository are designed to catch these subtle issues before they affect results.

## Running Tests

The test suite uses [pytest](https://docs.pytest.org/), a widely-used Python testing framework. All commands should be run from the repository root directory.

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest tests/unit/

# Run with verbose output (shows each test name as it runs)
uv run pytest -v

# Run a specific test file
uv run pytest tests/unit/test_expressions.py

# Run a specific test class
uv run pytest tests/unit/test_expressions.py::TestSumSas

# Run a specific test method
uv run pytest tests/unit/test_expressions.py::TestSumSas::test_sum_sas_both_non_null
```

If tests fail, pytest will show you which assertions failed and why. The `-v` (verbose) flag is helpful when debugging because it shows each test name as it executes.

**Code coverage** is enabled by default. After each test run, you'll see a coverage report showing which lines of code were exercised by the tests. The "Missing" column shows line numbers that weren't covered. These are opportunities for additional tests.

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── __init__.py              # Makes tests/ a Python package
├── README.md                # This file
└── unit/                    # Unit tests (fast, isolated)
    ├── __init__.py
    ├── test_expressions.py  # Core Polars expression utilities
    └── test_accounting_formulas.py  # Scoring functions (F-score, Z-score, etc.)
```

The `conftest.py` file is special to pytest. It contains fixtures (reusable test setup code) and configuration that are automatically available to all tests in the directory and its subdirectories.

## Writing New Tests

This section explains how to add tests when you implement new features or fix bugs. Even if you're not familiar with pytest, the patterns below should help you write effective tests.

### Why Write Tests?

When you change code in this repository, tests help you verify that:

1. **Your new code works correctly** - Tests document the expected behavior and verify it automatically.

2. **You haven't broken existing functionality** - Running the full test suite catches unintended side effects.

3. **Future changes don't break your code** - Your tests protect against regressions when others modify the codebase.

For financial code, tests are especially valuable because many bugs don't cause crashes. Instead, they just produce slightly wrong numbers. A test that checks expected outputs will catch these silent errors.

### Step 1: Choose the Right Location

Currently, all tests go in `tests/unit/`. Unit tests are:

- **Fast** - Each test should run in milliseconds, not seconds
- **Isolated** - Tests don't depend on external resources (databases, files, network)
- **Focused** - Each test checks one specific behavior

If you're testing a function from `aux_functions.py`, add your tests to an existing test file or create a new one following the naming pattern `test_<module_name>.py`.

### Step 2: Understand the Test File Structure

Test files follow a consistent organization:

```python
"""
Tests for <module_name>.

This module tests <brief description of what's being tested>.
Each test class corresponds to a function or closely related group of functions.
"""

import polars as pl
import numpy as np
import pytest

# Import the functions you're testing
from aux_functions import my_function, another_function


class TestMyFunction:
    """Tests for my_function().

    This function does X and is used for Y. These tests verify
    the core behavior and edge cases.
    """

    def test_basic_case(self):
        """Test the most common use case with typical inputs."""
        # Arrange: Set up test data
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })

        # Act: Call the function being tested
        result = my_function(df)

        # Assert: Verify the result is correct
        assert "output" in result.columns, (
            f"Expected 'output' column in result, got columns: {result.columns}"
        )
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"

    def test_handles_nulls(self):
        """Test that null values are handled correctly."""
        df = pl.DataFrame({
            "x": [1.0, None, 3.0],
            "y": [4.0, 5.0, None],
        })
        result = my_function(df)
        # Document WHY we expect this behavior
        # (e.g., "nulls propagate because..." or "nulls are replaced because...")
        assert result["output"][1] is None, (
            f"Expected null at row 1 (null input), got {result['output'][1]}"
        )

    def test_empty_dataframe(self):
        """Test behavior with empty input."""
        df = pl.DataFrame({"x": [], "y": []})
        result = my_function(df)
        assert len(result) == 0, f"Empty input should produce empty output, got {len(result)} rows"

    def test_single_row(self):
        """Test with minimal valid input."""
        df = pl.DataFrame({"x": [1.0], "y": [2.0]})
        result = my_function(df)
        assert len(result) == 1, f"Single-row input should produce single-row output, got {len(result)}"
```

### Step 3: Follow Naming Conventions

Consistent naming makes the codebase easier to navigate:

- **Test files**: `test_<module_name>.py` - The `test_` prefix is required for pytest to discover the file.

- **Test classes**: `TestFunctionName` - Group related tests together. The class name should clearly indicate what's being tested.

- **Test methods**: `test_<what>_<condition>()` - Describe what you're testing and under what conditions. Examples:
 - `test_sum_sas_both_non_null()` - Testing `sum_sas` when both inputs are non-null
 - `test_altman_z_negative_equity()` - Testing `altman_z` with negative equity
 - `test_safe_div_zero_denominator()` - Testing `safe_div` when denominator is zero

Good test names read like documentation. Someone should be able to understand what's being tested without reading the test body.

### Step 4: Write Clear Assertions

Assertions are the heart of a test. They verify that the code behaves correctly. pytest will report a failure if any assertion is false.

```python
# Simple equality (with diagnostic message)
assert result == expected, f"Expected {expected}, got {result}"

# Checking for None/null
assert result is None, "Result should be None for invalid input"
assert result is not None, "Result should not be None for valid input"

# Checking collections
assert "column_name" in df.columns, "Missing expected column 'column_name'"
assert len(result) == 10, f"Expected 10 rows, got {len(result)}"

# Checking conditions
assert result > 0, f"Result should be positive, got {result}"
assert 0 <= score <= 9, f"F-score must be 0-9, got {score}"
```

The second argument to `assert` is a message that appears when the assertion fails. Always include a diagnostic message that explains what went wrong and shows the actual value. This makes debugging much faster.

For numerical comparisons, see [Testing Numerical Results](#testing-numerical-results) below.

### Step 5: Test Edge Cases

Financial data has many edge cases that can cause subtle bugs. Good tests cover:

- **Null/missing values** - How should the function handle nulls in each input column?
- **Zero values** - Especially important for denominators in ratios
- **Negative values** - Negative equity, negative earnings, etc.
- **Empty input** - What happens with zero rows?
- **Single row** - Some calculations need multiple rows (e.g., lags)
- **Extreme values** - Very large or very small numbers

Example of testing edge cases:

```python
class TestSafeDiv:
    """Tests for safe_div() with various edge cases."""

    def test_normal_division(self):
        """Basic division with positive numbers."""
        result = safe_div(10.0, 2.0)
        assert result == 5.0, f"10/2 should equal 5, got {result}"

    def test_zero_denominator_returns_null(self):
        """Division by zero should return null, not raise an error."""
        result = safe_div(10.0, 0.0)
        assert result is None, f"Division by zero should return None, got {result}"

    def test_null_numerator(self):
        """Null numerator propagates to result."""
        result = safe_div(None, 2.0)
        assert result is None, f"Null numerator should return None, got {result}"

    def test_null_denominator(self):
        """Null denominator returns null."""
        result = safe_div(10.0, None)
        assert result is None, f"Null denominator should return None, got {result}"

    def test_negative_denominator(self):
        """Negative denominators are handled based on mode."""
        result = safe_div(10.0, -2.0, mode=1)
        assert result == -5.0, f"10/-2 should equal -5, got {result}"
```

## Testing Numerical Results

Financial calculations require special care because floating-point arithmetic isn't exact. Two calculations that should mathematically produce the same result might differ by tiny amounts due to rounding.

### The Problem with Exact Equality

```python
# This looks like it should work, but it fails!
>>> 0.1 + 0.2 == 0.3
False

>>> 0.1 + 0.2
0.30000000000000004
```

This isn't a bug - it's how floating-point numbers work in all programming languages. For tests, we need to compare "close enough" rather than "exactly equal."

### Using Tolerances

The `numpy.testing.assert_allclose()` function compares values within a tolerance:

```python
import numpy as np

np.testing.assert_allclose(
    actual,      # The value your code computed
    expected,    # The value you expect
    rtol=1e-6,   # Relative tolerance (proportional to magnitude)
    atol=1e-10,  # Absolute tolerance (fixed amount)
)
```

The comparison passes if: `|actual - expected| <= atol + rtol * |expected|`

- **Relative tolerance (rtol)** matters for large values. An rtol of 1e-6 means values can differ by 0.0001%.
- **Absolute tolerance (atol)** matters for small values near zero.

### Choosing the Right Tolerance

Different calculations need different tolerances. A simple ratio has less accumulated error than a complex formula with many operations.

**Rule of thumb**: Count how many arithmetic operations stand between your inputs and outputs.

| Operations | Tolerance Level | Example Calculations |
|------------|-----------------|---------------------|
| 1-2 operations | TIGHT (rtol=1e-10) | Simple returns, weight sums |
| 3-10 operations | STANDARD (rtol=1e-6) | Financial ratios (ROE, B/M) |
| 10+ operations | LOOSE (rtol=1e-4) | Composite scores (F-score, Z-score) |
| Statistical estimation | VERY_LOOSE (rtol=0.01) | Beta, correlation, volatility |

### Using the ToleranceSpec Fixture

The `conftest.py` file provides a `tolerance` fixture with pre-calibrated values:

```python
class TestMyCalculation:
    def test_simple_return(self, tolerance):
        """Single division operation uses tight tolerance."""
        actual = (price_t1 - price_t0) / price_t0
        expected = 0.05
        np.testing.assert_allclose(actual, expected, **tolerance.TIGHT)

    def test_roe(self, tolerance):
        """ROE is net_income / equity, a simple ratio."""
        actual = compute_roe(net_income=100, equity=500)
        expected = 0.20
        np.testing.assert_allclose(actual, expected, **tolerance.FINANCIAL_RATIOS)

    def test_altman_z(self, tolerance):
        """Z-score combines 5 weighted ratios."""
        actual = altman_z(firm_data)
        expected = 2.5
        np.testing.assert_allclose(actual, expected, **tolerance.COMPOSITE_SCORES)
```

The available tolerance levels and their aliases:

| Core Level | rtol | atol | Domain Alias |
|------------|------|------|--------------|
| `TIGHT` | 1e-10 | 1e-12 | `SIMPLE_ARITHMETIC` |
| `STANDARD` | 1e-6 | 1e-10 | `FINANCIAL_RATIOS` |
| `LOOSE` | 1e-4 | 1e-6 | `COMPOSITE_SCORES` |
| `VERY_LOOSE` | 0.01 | 0.001 | `STATISTICAL_ESTIMATES` |

You can use either the core level name or the domain alias - they're equivalent. Use whichever makes your test's intent clearer.

### Discrete Values Don't Need Tolerances

Some financial measures are discrete (integer) values. For these, use exact comparison:

```python
def test_piotroski_f_score_range(self):
    """F-score must be an integer from 0 to 9."""
    result = piotroski_f(firm_data)
    score = result["f_score"][0]

    # F-score is discrete, so exact comparison is fine
    assert isinstance(score, int)
    assert 0 <= score <= 9
```

## Test Markers

pytest allows you to "mark" tests with labels, which helps organize and selectively run tests.

Tests in this repository are automatically marked based on their location:

- Tests in `tests/unit/` get the `@pytest.mark.unit` marker

To run only tests with a specific marker, use the `-m` flag:

```bash
# Run only unit tests
uv run pytest -m unit

# Run everything except slow tests (if we had them)
uv run pytest -m "not slow"
```

To manually mark a test:

```python
import pytest

@pytest.mark.slow
def test_something_that_takes_a_while():
    """This test takes several seconds to run."""
    pass
```

## CI Integration

Every pull request automatically runs the test suite via GitHub Actions. This happens before your code can be merged.

The CI pipeline includes:

1. **Lint** - Ruff checks that code follows style guidelines
2. **Unit Tests** - All tests in `tests/unit/` must pass

If tests fail, the PR cannot be merged. This protects the main branch from broken code.

To avoid surprises, **always run tests locally before pushing**:

```bash
uv run pytest tests/unit/ -v
```

## Common Patterns

### Testing with Polars DataFrames

Most functions in this codebase operate on Polars DataFrames. Here's how to set up test data:

```python
import polars as pl

def test_my_dataframe_function(self):
    # Create a small DataFrame with known values
    df = pl.DataFrame({
        "permno": [10001, 10001, 10002, 10002],
        "date": ["2020-01-31", "2020-02-28", "2020-01-31", "2020-02-28"],
        "ret": [0.05, -0.02, 0.03, 0.01],
        "mktcap": [1000.0, 1050.0, 500.0, 505.0],
    }).with_columns(pl.col("date").str.to_date())

    result = my_function(df)

    # Check specific values you can compute by hand
    assert result["output"][0] == expected_value
```

### Testing Functions That Return Expressions

Many utility functions return Polars expressions rather than DataFrames. Test these within a DataFrame context:

```python
from aux_functions import safe_div

def test_safe_div_in_dataframe(self):
    df = pl.DataFrame({
        "numerator": [10.0, 20.0, 30.0],
        "denominator": [2.0, 0.0, 5.0],
    })

    result = df.select(
        safe_div("numerator", "denominator").alias("ratio")
    )

    assert result["ratio"][0] == 5.0   # 10/2
    assert result["ratio"][1] is None  # 20/0 -> null
    assert result["ratio"][2] == 6.0   # 30/5
```

### Testing Expected Failures

Sometimes you want to verify that a function raises an error for invalid input:

```python
import pytest

def test_invalid_input_raises_error(self):
    """Function should raise ValueError for negative counts."""
    with pytest.raises(ValueError, match="count must be positive"):
        my_function(count=-1)
```

### Using Fixtures for Repeated Setup

If multiple tests need the same setup, define a fixture:

```python
import pytest
import polars as pl

@pytest.fixture
def sample_returns():
    """A small returns DataFrame for testing."""
    return pl.DataFrame({
        "permno": [10001, 10001, 10001],
        "date": ["2020-01-31", "2020-02-28", "2020-03-31"],
        "ret": [0.05, -0.02, 0.03],
    }).with_columns(pl.col("date").str.to_date())


class TestReturnCalculations:
    def test_cumulative_return(self, sample_returns):
        result = cumulative_return(sample_returns)
        # ...

    def test_volatility(self, sample_returns):
        result = calculate_volatility(sample_returns)
        # ...
```

Fixtures defined in `conftest.py` are available to all tests automatically. Fixtures defined in a test file are only available within that file.

## Troubleshooting

### "Module not found" errors

The test configuration adds the `code/` directory to the Python path. If you still get import errors:

```bash
# Make sure you're running from the repository root
cd /path/to/jkp-data
uv run pytest tests/unit/
```

### Tests pass locally but fail in CI

This usually means:
- You have uncommitted changes that tests depend on
- Your local environment has packages not in `pyproject.toml`
- A test relies on a local resource (database, credentials, file) that doesn't exist in GitHub Actions and needs to be mocked
- There's a platform-specific issue (rare with pure Python code)

Always commit and push all related changes together.

### Floating-point comparison failures

If you see errors like `AssertionError: Arrays are not almost equal`, the tolerance might be too tight. Review the "Choosing the Right Tolerance" section above.

### Tests are slow

Unit tests should be fast (< 1 second each). If a test is slow:
- Reduce the size of test data
- Mock expensive operations
- Consider if it's really a unit test or should be an integration test

## Understanding Code Coverage

After running tests, you'll see a coverage report like this:

```
Name                         Stmts   Miss Branch BrPart  Cover   Missing
------------------------------------------------------------------------
code/aux_functions.py         1373   1071    136      5    21%   41-62, 79-83, ...
```

The columns mean:
- **Stmts** - Total number of executable statements in the file
- **Miss** - Statements not executed by any test
- **Branch** - Total conditional branches (if/else, loops)
- **BrPart** - Branches only partially covered
- **Cover** - Percentage of statements covered
- **Missing** - Line numbers not covered by tests

### Interpreting Coverage

Coverage tells you which code paths tests exercise, but **high coverage doesn't guarantee correctness**. A test that runs code without checking results adds coverage but catches no bugs.

That said, low coverage does indicate risk. Uncovered code has no automated verification.

For this repository:
- Core utility functions (`fl_none`, `safe_div`, etc.) have high coverage
- Scoring functions (`altman_z`, `piotroski_f`, etc.) have high coverage
- Pipeline orchestration code has low coverage (harder to unit test)

### Generating HTML Coverage Reports

For a detailed, browsable coverage report:

```bash
uv run pytest --cov-report=html

# Open the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

The HTML report lets you click into each file and see exactly which lines are covered (green) or missing (red).

## Getting Help

If you're unsure how to test something:

1. Look at existing tests for similar functions
2. Start with the simplest test case that exercises the code
3. Add edge cases one at a time
4. Ask for review - other contributors can suggest cases you might have missed
