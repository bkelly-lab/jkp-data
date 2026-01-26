"""
Tests for core Polars expression utilities in aux_functions.py.

These are fundamental building blocks used throughout the factor construction
pipeline. Tests verify correct handling of:
- Null values (SAS-style semantics)
- Division by zero protection
- Type consistency

Paper Reference: Jensen, Kelly, Pedersen (2023), "Is There a Replication Crisis in Finance?"
"""

from __future__ import annotations

# Import the functions we're testing
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from aux_functions import bo_false, fl_none, safe_div, sub_sas, sum_sas

# safe_div modes:
#   1 = protect zero denominator (den != 0)
#   2 = divide by |den| (absolute value)
#   3 = only when den > 0
#   8 = only when num > 0 and den > 0


class TestFlNone:
    """Tests for fl_none() - null float literal expression."""

    def test_fl_none_returns_null(self):
        """fl_none() should create a null Float64 literal."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = df.select(fl_none().alias("null_col"))

        assert result["null_col"].dtype == pl.Float64, (
            f"Expected Float64 dtype, got {result['null_col'].dtype}"
        )
        assert result["null_col"].is_null().all(), (
            f"Expected all nulls, got {result['null_col'].to_list()}"
        )

    def test_fl_none_in_when_expression(self):
        """fl_none() should work correctly in conditional expressions."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = df.select(
            pl.when(pl.col("x") > 3).then(pl.col("x")).otherwise(fl_none()).alias("result")
        )

        expected = pl.Series("result", [None, None, None, 4.0, 5.0])
        assert result["result"].to_list() == expected.to_list(), (
            f"Expected {expected.to_list()}, got {result['result'].to_list()}"
        )

    def test_fl_none_dtype_consistency(self):
        """fl_none() should always produce Float64 regardless of context."""
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "str_col": ["a", "b", "c"],
            }
        )

        result = df.select(
            fl_none().alias("null1"),
            fl_none().alias("null2"),
        )

        assert result["null1"].dtype == pl.Float64, f"Expected Float64, got {result['null1'].dtype}"
        assert result["null2"].dtype == pl.Float64, f"Expected Float64, got {result['null2'].dtype}"


class TestBoFalse:
    """Tests for bo_false() - false boolean literal expression."""

    def test_bo_false_returns_false(self):
        """bo_false() should create a False Boolean literal."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = df.select(bo_false().alias("false_col"))

        assert result["false_col"].dtype == pl.Boolean, (
            f"Expected Boolean dtype, got {result['false_col'].dtype}"
        )
        assert (~result["false_col"]).all(), (
            f"Expected all False values, got {result['false_col'].to_list()}"
        )

    def test_bo_false_in_filter(self):
        """bo_false() should work in filter expressions."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = df.filter(bo_false())

        assert len(result) == 0, f"Expected 0 rows after filtering by False, got {len(result)}"


class TestSumSas:
    """Tests for sum_sas() - SAS-style sum with null handling.

    SAS sum behavior:
    - If ANY input is non-null, treat nulls as 0 and sum
    - If ALL inputs are null, return null
    """

    def test_sum_sas_both_non_null(self, tolerance):
        """When both values are non-null, sum normally."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [10.0, 20.0, 30.0],
            }
        )
        result = df.select(sum_sas("a", "b").alias("result"))

        expected = [11.0, 22.0, 33.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_sum_sas_first_null(self, tolerance):
        """When first value is null, treat as 0 and use second value."""
        df = pl.DataFrame(
            {
                "a": [None, None, 3.0],
                "b": [10.0, 20.0, 30.0],
            }
        )
        result = df.select(sum_sas("a", "b").alias("result"))

        expected = [10.0, 20.0, 33.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_sum_sas_second_null(self, tolerance):
        """When second value is null, treat as 0 and use first value."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [None, None, 30.0],
            }
        )
        result = df.select(sum_sas("a", "b").alias("result"))

        expected = [1.0, 2.0, 33.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_sum_sas_both_null(self):
        """When both values are null, return null."""
        df = pl.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "b": [10.0, None, 30.0],
            }
        )
        result = df.select(sum_sas("a", "b").alias("result"))

        assert result["result"][0] == 11.0, f"Row 0: expected 11.0, got {result['result'][0]}"
        assert result["result"][1] is None, f"Row 1: expected None, got {result['result'][1]}"
        assert result["result"][2] == 33.0, f"Row 2: expected 33.0, got {result['result'][2]}"

    def test_sum_sas_with_zeros(self, tolerance):
        """Zero should be treated as a valid value, not as null."""
        df = pl.DataFrame(
            {
                "a": [0.0, 0.0, None],
                "b": [0.0, None, 0.0],
            }
        )
        result = df.select(sum_sas("a", "b").alias("result"))

        expected = [0.0, 0.0, 0.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_sum_sas_with_negative_values(self, tolerance):
        """Negative values should be handled correctly."""
        df = pl.DataFrame(
            {
                "a": [-1.0, -2.0, None],
                "b": [10.0, None, -30.0],
            }
        )
        result = df.select(sum_sas("a", "b").alias("result"))

        expected = [9.0, -2.0, -30.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )


class TestSubSas:
    """Tests for sub_sas() - SAS-style subtraction with null handling.

    Same semantics as sum_sas but for subtraction (col1 - col2).
    """

    def test_sub_sas_both_non_null(self, tolerance):
        """When both values are non-null, subtract normally."""
        df = pl.DataFrame(
            {
                "a": [10.0, 20.0, 30.0],
                "b": [1.0, 2.0, 3.0],
            }
        )
        result = df.select(sub_sas("a", "b").alias("result"))

        expected = [9.0, 18.0, 27.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_sub_sas_first_null(self, tolerance):
        """When first value is null, treat as 0: 0 - b = -b."""
        df = pl.DataFrame(
            {
                "a": [None, None, 30.0],
                "b": [1.0, 2.0, 3.0],
            }
        )
        result = df.select(sub_sas("a", "b").alias("result"))

        expected = [-1.0, -2.0, 27.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_sub_sas_second_null(self, tolerance):
        """When second value is null, treat as 0: a - 0 = a."""
        df = pl.DataFrame(
            {
                "a": [10.0, 20.0, 30.0],
                "b": [None, None, 3.0],
            }
        )
        result = df.select(sub_sas("a", "b").alias("result"))

        expected = [10.0, 20.0, 27.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_sub_sas_both_null(self):
        """When both values are null, return null."""
        df = pl.DataFrame(
            {
                "a": [10.0, None, 30.0],
                "b": [1.0, None, 3.0],
            }
        )
        result = df.select(sub_sas("a", "b").alias("result"))

        assert result["result"][0] == 9.0, f"Row 0: expected 9.0, got {result['result'][0]}"
        assert result["result"][1] is None, f"Row 1: expected None, got {result['result'][1]}"
        assert result["result"][2] == 27.0, f"Row 2: expected 27.0, got {result['result'][2]}"


class TestSafeDivMode1:
    """Tests for safe_div() mode 1 - basic division with zero protection."""

    def test_safe_div_mode1_normal_division(self, tolerance):
        """Normal division when denominator is non-zero."""
        df = pl.DataFrame(
            {
                "num": [10.0, 20.0, 30.0],
                "den": [2.0, 4.0, 5.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=1))

        expected = [5.0, 5.0, 6.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_safe_div_mode1_zero_denominator(self):
        """Returns null when denominator is zero."""
        df = pl.DataFrame(
            {
                "num": [10.0, 20.0, 30.0],
                "den": [2.0, 0.0, 5.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=1))

        assert result["result"][0] == 5.0, f"Row 0: expected 5.0, got {result['result'][0]}"
        assert result["result"][1] is None, (
            f"Row 1: expected None for div by zero, got {result['result'][1]}"
        )
        assert result["result"][2] == 6.0, f"Row 2: expected 6.0, got {result['result'][2]}"

    def test_safe_div_mode1_negative_denominator(self, tolerance):
        """Negative denominators should work (only zero is protected)."""
        df = pl.DataFrame(
            {
                "num": [10.0, -20.0, 30.0],
                "den": [-2.0, -4.0, -5.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=1))

        expected = [-5.0, 5.0, -6.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )


class TestSafeDivMode2:
    """Tests for safe_div() mode 2 - division by absolute value."""

    def test_safe_div_mode2_uses_absolute_denominator(self, tolerance):
        """Mode 2 divides by |den|."""
        df = pl.DataFrame(
            {
                "num": [10.0, 20.0, 30.0],
                "den": [-2.0, -4.0, 5.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=2))

        expected = [5.0, 5.0, 6.0]
        np.testing.assert_allclose(
            result["result"].to_list(),
            expected,
            **tolerance.TIGHT,
            err_msg=f"Expected {expected}, got {result['result'].to_list()}",
        )

    def test_safe_div_mode2_zero_protected(self):
        """Mode 2 still protects against zero denominator."""
        df = pl.DataFrame(
            {
                "num": [10.0, 20.0],
                "den": [0.0, 2.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=2))

        assert result["result"][0] is None, (
            f"Row 0: expected None for div by zero, got {result['result'][0]}"
        )
        assert result["result"][1] == 10.0, f"Row 1: expected 10.0, got {result['result'][1]}"


class TestSafeDivMode3:
    """Tests for safe_div() mode 3 - division only when den > 0."""

    def test_safe_div_mode3_positive_denominator(self):
        """Mode 3 only divides when denominator is strictly positive."""
        df = pl.DataFrame(
            {
                "num": [10.0, 20.0, 30.0, 40.0],
                "den": [2.0, 0.0, -5.0, 5.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=3))

        assert result["result"][0] == 5.0, f"Row 0: expected 5.0, got {result['result'][0]}"
        assert result["result"][1] is None, (
            f"Row 1: expected None for zero, got {result['result'][1]}"
        )
        assert result["result"][2] is None, (
            f"Row 2: expected None for negative, got {result['result'][2]}"
        )
        assert result["result"][3] == 8.0, f"Row 3: expected 8.0, got {result['result'][3]}"


class TestSafeDivMode8:
    """Tests for safe_div() mode 8 - division only when both num > 0 and den > 0."""

    def test_safe_div_mode8_both_positive(self):
        """Mode 8 only divides when both numerator and denominator are positive."""
        df = pl.DataFrame(
            {
                "num": [10.0, -20.0, 30.0, -40.0, 50.0],
                "den": [2.0, 4.0, -5.0, -5.0, 5.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=8))

        assert result["result"][0] == 5.0, (
            f"Row 0 (both positive): expected 5.0, got {result['result'][0]}"
        )
        assert result["result"][1] is None, (
            f"Row 1 (num negative): expected None, got {result['result'][1]}"
        )
        assert result["result"][2] is None, (
            f"Row 2 (den negative): expected None, got {result['result'][2]}"
        )
        assert result["result"][3] is None, (
            f"Row 3 (both negative): expected None, got {result['result'][3]}"
        )
        assert result["result"][4] == 10.0, (
            f"Row 4 (both positive): expected 10.0, got {result['result'][4]}"
        )


class TestSafeDivEdgeCases:
    """Edge case tests for safe_div()."""

    def test_safe_div_null_numerator(self):
        """Null numerator should propagate as null result."""
        df = pl.DataFrame(
            {
                "num": [None, 20.0, None],
                "den": [2.0, 4.0, 0.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=1))

        assert result["result"][0] is None, (
            f"Row 0: expected None for null numerator, got {result['result'][0]}"
        )
        assert result["result"][1] == 5.0, f"Row 1: expected 5.0, got {result['result'][1]}"
        assert result["result"][2] is None, f"Row 2: expected None, got {result['result'][2]}"

    def test_safe_div_null_denominator(self):
        """Null denominator should result in null."""
        df = pl.DataFrame(
            {
                "num": [10.0, 20.0],
                "den": [None, 4.0],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=1))

        assert result["result"][0] is None, (
            f"Row 0: expected None for null denominator, got {result['result'][0]}"
        )
        assert result["result"][1] == 5.0, f"Row 1: expected 5.0, got {result['result'][1]}"

    def test_safe_div_very_small_denominator(self):
        """Very small denominators should not cause overflow issues."""
        df = pl.DataFrame(
            {
                "num": [1.0],
                "den": [1e-300],
            }
        )
        result = df.select(safe_div("num", "den", "result", mode=1))

        # Should produce a very large number, not an error
        assert result["result"][0] is not None, "Result should not be None for small denominator"
        assert result["result"][0] > 1e290, f"Expected very large number, got {result['result'][0]}"

    def test_safe_div_output_column_name(self):
        """safe_div should correctly name the output column."""
        df = pl.DataFrame(
            {
                "a": [10.0],
                "b": [2.0],
            }
        )
        result = df.select(safe_div("a", "b", "my_custom_ratio", mode=1))

        assert "my_custom_ratio" in result.columns, (
            f"Expected 'my_custom_ratio' column, got {result.columns}"
        )
        assert result["my_custom_ratio"][0] == 5.0, (
            f"Expected 5.0, got {result['my_custom_ratio'][0]}"
        )


class TestExpressionChaining:
    """Tests for chaining expression utilities."""

    def test_sum_sas_chained_with_safe_div(self, tolerance):
        """sum_sas result can be divided using safe_div."""
        df = pl.DataFrame(
            {
                "a": [10.0, None, 30.0],
                "b": [5.0, 15.0, None],
                "c": [3.0, 5.0, 0.0],
            }
        )

        # First compute sum_sas, then divide by c using safe_div
        result = df.select(
            sum_sas("a", "b").alias("sum_ab"),
            pl.col("c"),
        ).select(safe_div("sum_ab", "c", "ratio", mode=1))

        # sum_ab = [15, 15, 30] (SAS semantics: null treated as 0)
        # ratio = [15/3, 15/5, null] = [5.0, 3.0, null] (div by zero protected)
        assert result["ratio"][0] == 5.0, f"Row 0: expected 5.0, got {result['ratio'][0]}"
        assert result["ratio"][1] == 3.0, f"Row 1: expected 3.0, got {result['ratio'][1]}"
        assert result["ratio"][2] is None, (
            f"Row 2: expected None for div by zero, got {result['ratio'][2]}"
        )

    def test_multiple_expressions_in_select(self):
        """Multiple expression utilities should work in the same select."""
        df = pl.DataFrame(
            {
                "x": [10.0, 20.0],
                "y": [5.0, None],
                "z": [2.0, 4.0],
            }
        )

        result = df.select(
            sum_sas("x", "y").alias("sum_xy"),
            sub_sas("x", "y").alias("diff_xy"),
            safe_div("x", "z", "ratio_xz", mode=1),
        )

        assert result["sum_xy"][0] == 15.0, f"sum_xy[0]: expected 15.0, got {result['sum_xy'][0]}"
        assert result["diff_xy"][0] == 5.0, f"diff_xy[0]: expected 5.0, got {result['diff_xy'][0]}"
        assert result["ratio_xz"][0] == 5.0, (
            f"ratio_xz[0]: expected 5.0, got {result['ratio_xz'][0]}"
        )
