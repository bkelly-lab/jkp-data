"""
Shared pytest fixtures for jkp-data tests.

This module provides common test fixtures used across the test suite, including:
- Helper utilities for numerical comparison with appropriate tolerances
- Pytest configuration for test markers

Paper Reference: Jensen, Kelly, Pedersen (2023), "Is There a Replication Crisis in Finance?"
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

# Add the code directory to the path so we can import from it
CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (module interactions)")
    config.addinivalue_line("markers", "methodology: Paper methodology validation tests")
    config.addinivalue_line("markers", "regression: Golden output regression tests")
    config.addinivalue_line("markers", "expensive: Expensive tests requiring significant resources")
    config.addinivalue_line("markers", "wrds: Tests requiring WRDS credentials")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark tests based on their location."""
    for item in items:
        test_path = Path(item.fspath)
        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "methodology" in test_path.parts:
            item.add_marker(pytest.mark.methodology)
        elif "regression" in test_path.parts:
            item.add_marker(pytest.mark.regression)
        elif "wrds" in test_path.parts:
            item.add_marker(pytest.mark.wrds)


# =============================================================================
# Numerical Tolerance Helpers
# =============================================================================


class ToleranceSpec:
    """Specification for numerical tolerances in financial calculations.

    Core Precision Levels (use these for general cases):
        TIGHT       - Simple arithmetic (single operation, minimal error)
        STANDARD    - Accumulated operations (ratios, chained calculations)
        LOOSE       - Complex formulas (multi-step scores, indices)
        VERY_LOOSE  - Statistical estimates (regression coefficients, volatility)

    Domain Aliases (for discoverability):
        SIMPLE_ARITHMETIC    -> TIGHT       (returns, weight sums)
        FINANCIAL_RATIOS     -> STANDARD    (ROE, B/M, leverage)
        COMPOSITE_SCORES     -> LOOSE       (F-score, Z-score, KZ index)
        STATISTICAL_ESTIMATES -> VERY_LOOSE (beta, correlation, volatility)

    Usage:
        np.testing.assert_allclose(actual, expected, **ToleranceSpec.STANDARD)
        np.testing.assert_allclose(actual, expected, **ToleranceSpec.FINANCIAL_RATIOS)
    """

    # Core precision levels
    TIGHT = {"rtol": 1e-10, "atol": 1e-12}
    STANDARD = {"rtol": 1e-6, "atol": 1e-10}
    LOOSE = {"rtol": 1e-4, "atol": 1e-6}
    VERY_LOOSE = {"rtol": 0.01, "atol": 0.001}

    # Domain aliases (map to core levels)
    SIMPLE_ARITHMETIC = TIGHT
    FINANCIAL_RATIOS = STANDARD
    COMPOSITE_SCORES = LOOSE
    STATISTICAL_ESTIMATES = VERY_LOOSE

    # Default for when you're unsure
    DEFAULT = STANDARD


@pytest.fixture
def tolerance() -> ToleranceSpec:
    """Provide tolerance specifications for numerical assertions."""
    return ToleranceSpec()


def assert_series_equal(
    actual: pl.Series,
    expected: pl.Series,
    rtol: float = 1e-6,
    atol: float = 1e-10,
    check_names: bool = True,
) -> None:
    """Assert two Polars series are equal within tolerance, handling NaNs properly.

    Args:
        actual: Computed series
        expected: Expected series
        rtol: Relative tolerance
        atol: Absolute tolerance
        check_names: Whether to check series names match
    """
    if check_names:
        assert actual.name == expected.name, (
            f"Series names differ: {actual.name} vs {expected.name}"
        )

    assert len(actual) == len(expected), f"Series lengths differ: {len(actual)} vs {len(expected)}"

    # Convert to numpy for comparison
    actual_np = actual.to_numpy()
    expected_np = expected.to_numpy()

    # Handle NaN comparison: NaN == NaN should be True
    both_nan = np.isnan(actual_np) & np.isnan(expected_np)
    actual_nan = np.isnan(actual_np)
    expected_nan = np.isnan(expected_np)

    # Check that NaNs are in the same positions
    if not np.array_equal(actual_nan, expected_nan):
        nan_diff_count = np.sum(actual_nan != expected_nan)
        raise AssertionError(
            f"NaN positions differ in {nan_diff_count} locations. "
            f"Actual NaNs: {np.sum(actual_nan)}, Expected NaNs: {np.sum(expected_nan)}"
        )

    # Compare non-NaN values
    mask = ~both_nan
    if np.any(mask):
        np.testing.assert_allclose(
            actual_np[mask],
            expected_np[mask],
            rtol=rtol,
            atol=atol,
            err_msg=f"Values differ beyond tolerance (rtol={rtol}, atol={atol})",
        )


# =============================================================================
# Test Data Helpers
# =============================================================================


@pytest.fixture
def seed() -> int:
    """Provide a fixed random seed for reproducibility."""
    return 42


def make_test_dataframe(
    schema: dict[str, pl.DataType],
    n_rows: int = 10,
    seed: int = 42,
) -> pl.DataFrame:
    """Create a test DataFrame with given schema and random data.

    Args:
        schema: Dictionary mapping column names to Polars data types
        n_rows: Number of rows to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with random values matching the schema
    """
    np.random.seed(seed)
    data = {}

    for col_name, dtype in schema.items():
        if dtype == pl.Float64:
            data[col_name] = np.random.randn(n_rows)
        elif dtype == pl.Int64:
            data[col_name] = np.random.randint(0, 1000, n_rows)
        elif dtype == pl.Utf8:
            data[col_name] = [f"{col_name}_{i}" for i in range(n_rows)]
        elif dtype == pl.Date:
            start = date(2020, 1, 31)
            data[col_name] = [
                date(start.year, (start.month + i - 1) % 12 + 1, 28) for i in range(n_rows)
            ]
        elif dtype == pl.Boolean:
            data[col_name] = np.random.choice([True, False], n_rows)
        else:
            data[col_name] = [None] * n_rows

    return pl.DataFrame(data)


@pytest.fixture
def make_dataframe():
    """Factory fixture for creating test DataFrames."""
    return make_test_dataframe


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide a temporary directory for test data files.

    Creates the standard subdirectory structure expected by the pipeline.
    """
    # Create expected subdirectories
    (tmp_path / "raw_data_dfs").mkdir()
    (tmp_path / "raw" / "raw_tables").mkdir(parents=True)
    (tmp_path / "processed" / "characteristics").mkdir(parents=True)
    (tmp_path / "processed" / "return_data").mkdir(parents=True)
    (tmp_path / "processed" / "other_output").mkdir(parents=True)

    yield tmp_path
