"""Tests for the growth and change helpers of the accounting stage.

SAS reference: char_macros.sas (var_growth, chg_to_assets, chg_var1_to_var2, chg_to_exp)
and accounting_chars.sas (emp_gr1, tangibility). The helpers shift by row position, so
each test frame holds one firm with one row per month, and ``count`` is the number of
observations seen so far.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import (
    chg_to_assets,
    chg_to_avgassets,
    chg_to_exp,
    chg_to_lagassets,
    chg_var1_to_var2,
    emp_gr,
    tangibility,
    var_growth,
)


def _frame(n: int, **cols: list) -> pl.DataFrame:
    """One firm, n monthly rows, ``count`` = 1..n unless given."""
    cols.setdefault("count", list(range(1, n + 1)))
    return pl.DataFrame(cols)


def _out(df: pl.DataFrame, expr: pl.Expr, name: str) -> list[float | None]:
    return df.select(expr)[name].to_list()


@pytest.mark.unit
class TestVarGrowth:
    def test_one_year(self) -> None:
        """Growth = x / x_lag12 - 1, null for the first 12 rows (count <= 12)."""
        n = 15
        x = [2.0 * (i + 1) for i in range(n)]
        df = _frame(n, sale_x=x)
        out = _out(df, var_growth("sale_x", 12), "sale_gr1")
        assert out[:12] == [None] * 12
        assert out[12] == pytest.approx(26.0 / 2.0 - 1)
        assert out[14] == pytest.approx(30.0 / 6.0 - 1)

    def test_three_years_name_and_horizon(self) -> None:
        n = 40
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)])
        out = _out(df, var_growth("sale_x", 36), "sale_gr3")
        assert out[:36] == [None] * 36
        assert out[36] == pytest.approx(37.0 / 1.0 - 1)
        assert out[39] == pytest.approx(40.0 / 4.0 - 1)

    def test_two_years_name(self) -> None:
        n = 26
        df = _frame(n, capx=[float(i + 1) for i in range(n)])
        out = _out(df, var_growth("capx", 24), "capx_gr2")
        assert out[:24] == [None] * 24
        assert out[24] == pytest.approx(25.0 / 1.0 - 1)

    def test_count_gate_both_sides(self) -> None:
        """Lag is available at row 12, but count == 12 nulls it while count == 13 keeps it."""
        n = 14
        x = [float(i + 1) for i in range(n)]
        count = list(range(1, n + 1))
        count[12] = 12
        df = _frame(n, sale_x=x, count=count)
        out = _out(df, var_growth("sale_x", 12), "sale_gr1")
        assert out[12] is None
        assert out[13] == pytest.approx(14.0 / 2.0 - 1)

    def test_nonpositive_lag_is_null(self) -> None:
        n = 14
        x = [1.0] * n
        x[0] = 0.0
        x[1] = -5.0
        x[2] = 4.0
        x[14 - 2] = 8.0
        df = _frame(n, sale_x=x)
        out = _out(df, var_growth("sale_x", 12), "sale_gr1")
        assert out[12] is None  # lag 0
        assert out[13] is None  # lag -5

    def test_null_input_propagates(self) -> None:
        n = 14
        x: list[float | None] = [float(i + 1) for i in range(n)]
        x[12] = None
        df = _frame(n, sale_x=x)
        out = _out(df, var_growth("sale_x", 12), "sale_gr1")
        assert out[12] is None
        assert out[13] == pytest.approx(14.0 / 2.0 - 1)


@pytest.mark.unit
class TestChgToAssets:
    def test_normal(self) -> None:
        """(x - x_lag12) / at_x, with the current-period assets."""
        n = 14
        x = [float(i + 1) for i in range(n)]
        at = [100.0] * n
        at[12] = 50.0
        df = _frame(n, sale_x=x, at_x=at)
        out = _out(df, chg_to_assets("sale_x", 12), "sale_gr1a")
        assert out[:12] == [None] * 12
        assert out[12] == pytest.approx((13.0 - 1.0) / 50.0)
        assert out[13] == pytest.approx((14.0 - 2.0) / 100.0)

    def test_horizon_36(self) -> None:
        n = 38
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=[10.0] * n)
        out = _out(df, chg_to_assets("sale_x", 36), "sale_gr3a")
        assert out[:36] == [None] * 36
        assert out[36] == pytest.approx((37.0 - 1.0) / 10.0)

    def test_nonpositive_assets_null(self) -> None:
        n = 14
        at = [10.0] * n
        at[12] = 0.0
        at[13] = -1.0
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=at)
        out = _out(df, chg_to_assets("sale_x", 12), "sale_gr1a")
        assert out[12] is None
        assert out[13] is None

    def test_count_gate_both_sides(self) -> None:
        n = 14
        count = list(range(1, n + 1))
        count[12] = 12
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=[10.0] * n, count=count)
        out = _out(df, chg_to_assets("sale_x", 12), "sale_gr1a")
        assert out[12] is None
        assert out[13] == pytest.approx((14.0 - 2.0) / 10.0)


@pytest.mark.unit
class TestChgToLagAssets:
    def test_normal(self) -> None:
        n = 14
        at = [100.0 + i for i in range(n)]
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=at)
        out = _out(df, chg_to_lagassets("sale_x"), "sale_gr1a")
        assert out[:12] == [None] * 12
        assert out[12] == pytest.approx((13.0 - 1.0) / 100.0)
        assert out[13] == pytest.approx((14.0 - 2.0) / 101.0)

    def test_nonpositive_lag_assets_null(self) -> None:
        n = 14
        at = [10.0] * n
        at[0] = 0.0
        at[1] = -3.0
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=at)
        out = _out(df, chg_to_lagassets("sale_x"), "sale_gr1a")
        assert out[12] is None
        assert out[13] is None

    def test_count_gate_both_sides(self) -> None:
        n = 14
        count = list(range(1, n + 1))
        count[12] = 12
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=[10.0] * n, count=count)
        out = _out(df, chg_to_lagassets("sale_x"), "sale_gr1a")
        assert out[12] is None
        assert out[13] == pytest.approx((14.0 - 2.0) / 10.0)


@pytest.mark.unit
class TestChgToExp:
    def test_normal(self) -> None:
        """x / mean(x_lag12, x_lag24) - 1; needs count > 24."""
        n = 26
        x = [float(i + 1) for i in range(n)]
        df = _frame(n, sale_x=x)
        out = _out(df, chg_to_exp("sale_x"), "sale_ce")
        assert out[:24] == [None] * 24
        assert out[24] == pytest.approx(25.0 / ((13.0 + 1.0) / 2) - 1)
        # row 25: x=26, lag12 = 14, lag24 = 2
        assert out[25] == pytest.approx(26.0 / ((14.0 + 2.0) / 2) - 1)

    def test_count_gate_both_sides(self) -> None:
        n = 27
        count = list(range(1, n + 1))
        count[24] = 24
        df = _frame(n, invt=[float(i + 1) for i in range(n)], count=count)
        out = _out(df, chg_to_exp("invt"), "invt_ce")
        assert out[24] is None  # lags exist, count == 24
        assert out[25] == pytest.approx(26.0 / ((14.0 + 2.0) / 2) - 1)

    def test_nonpositive_expectation_null(self) -> None:
        n = 26
        x = [float(i + 1) for i in range(n)]
        x[13] = -14.0  # lag12 of row 25
        x[1] = -2.0  # lag24 of row 25: sum = -16 -> not > 0
        x[14] = 14.0
        df = _frame(n, sale_x=x)
        out = _out(df, chg_to_exp("sale_x"), "sale_ce")
        assert out[25] is None
        # row 24: lag12 = x[12] = 13, lag24 = x[0] = 1
        assert out[24] == pytest.approx(25.0 / ((13.0 + 1.0) / 2) - 1)

    def test_zero_sum_null(self) -> None:
        n = 26
        x = [1.0] * n
        x[13] = 3.0
        x[1] = -3.0
        df = _frame(n, sale_x=x)
        out = _out(df, chg_to_exp("sale_x"), "sale_ce")
        assert out[25] is None


@pytest.mark.unit
class TestChgToAvgAssets:
    def test_normal(self) -> None:
        """(x - x_lag12) / (at + at_lag12), no halving of the denominator."""
        n = 14
        at = [10.0] * n
        at[12] = 30.0
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=at)
        out = _out(df, chg_to_avgassets("sale_x"), "sale_gr1a")
        assert out[:12] == [None] * 12
        assert out[12] == pytest.approx((13.0 - 1.0) / (30.0 + 10.0))
        assert out[13] == pytest.approx((14.0 - 2.0) / 20.0)

    def test_nonpositive_denominator_null(self) -> None:
        n = 14
        at = [10.0] * n
        at[0] = -10.0  # row 12: 10 + -10 = 0
        at[1] = -20.0  # row 13: 10 + -20 < 0
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=at)
        out = _out(df, chg_to_avgassets("sale_x"), "sale_gr1a")
        assert out[12] is None
        assert out[13] is None

    def test_count_gate_both_sides(self) -> None:
        n = 14
        count = list(range(1, n + 1))
        count[12] = 12
        df = _frame(n, sale_x=[float(i + 1) for i in range(n)], at_x=[10.0] * n, count=count)
        out = _out(df, chg_to_avgassets("sale_x"), "sale_gr1a")
        assert out[12] is None
        assert out[13] == pytest.approx((14.0 - 2.0) / 20.0)


@pytest.mark.unit
class TestChgVar1ToVar2:
    @staticmethod
    def _df(var1: list[float], var2: list[float], count: list[int] | None = None) -> pl.DataFrame:
        n = len(var1)
        return pl.DataFrame(
            {
                "gvkey": ["001"] * n,
                "curcd": ["USD"] * n,
                "datadate": list(range(n)),
                "count": count if count is not None else list(range(1, n + 1)),
                "a": var1,
                "b": var2,
            }
        )

    def test_change_in_ratio(self) -> None:
        n = 14
        a = [float(i + 1) for i in range(n)]
        b = [2.0] * n
        out = chg_var1_to_var2(self._df(a, b).lazy(), "d", "a", "b", 12).collect()
        assert "__x" not in out.columns
        assert out["d"].to_list()[:12] == [None] * 12
        assert out["d"][12] == pytest.approx(13.0 / 2.0 - 1.0 / 2.0)
        assert out["d"][13] == pytest.approx(14.0 / 2.0 - 2.0 / 2.0)

    def test_nonpositive_var2_nulls_ratio(self) -> None:
        """A var2 <= 0 nulls the ratio, so every change touching it is null."""
        n = 26
        a = [float(i + 1) for i in range(n)]
        b = [2.0] * n
        b[12] = 0.0
        b[13] = -1.0
        out = chg_var1_to_var2(self._df(a, b).lazy(), "d", "a", "b", 12).collect()["d"].to_list()
        assert out[12] is None  # current ratio null
        assert out[13] is None
        assert out[24] is None  # lagged ratio null (row 12)
        assert out[25] is None  # lagged ratio null (row 13)
        assert out[14] == pytest.approx(15.0 / 2.0 - 3.0 / 2.0)

    def test_horizon_24_and_count_gate(self) -> None:
        n = 26
        a = [float(i + 1) for i in range(n)]
        b = [1.0] * n
        count = list(range(1, n + 1))
        count[24] = 24
        out = chg_var1_to_var2(self._df(a, b, count).lazy(), "d", "a", "b", 24).collect()["d"]
        assert out[24] is None  # lag exists, count == horizon
        assert out[25] == pytest.approx(26.0 - 2.0)

    def test_sorts_by_key(self) -> None:
        """Rows given out of date order are sorted before the lag is taken."""
        df = self._df([float(i + 1) for i in range(14)], [1.0] * 14).reverse()
        out = chg_var1_to_var2(df.lazy(), "d", "a", "b", 12).collect()
        assert out["datadate"].to_list() == list(range(14))
        assert out["d"][12] == pytest.approx(13.0 - 1.0)


@pytest.mark.unit
class TestEmpGr:
    def test_normal(self) -> None:
        """(emp - emp_lag12) / (0.5 emp + 0.5 emp_lag12)."""
        n = 14
        emp = [10.0] * n
        emp[12] = 30.0
        df = _frame(n, emp=emp)
        out = _out(df, emp_gr("acc_std_ann.parquet"), "emp_gr1")
        assert out[:12] == [None] * 12
        assert out[12] == pytest.approx((30.0 - 10.0) / (0.5 * 30.0 + 0.5 * 10.0))

    def test_zero_growth_is_null(self) -> None:
        """SAS line 861: emp_gr1 = 0 is set to missing."""
        n = 14
        df = _frame(n, emp=[10.0] * n)
        out = _out(df, emp_gr("acc_std_ann.parquet"), "emp_gr1")
        assert out[12] is None

    def test_zero_average_is_null(self) -> None:
        n = 14
        emp = [0.0] * n
        df = _frame(n, emp=emp)
        out = _out(df, emp_gr("acc_std_ann.parquet"), "emp_gr1")
        assert out[12] is None

    def test_count_gate_both_sides(self) -> None:
        n = 14
        count = list(range(1, n + 1))
        count[12] = 12
        emp = [10.0] * n
        emp[12] = 30.0
        emp[13] = 30.0
        df = _frame(n, emp=emp, count=count)
        out = _out(df, emp_gr("acc_std_ann.parquet"), "emp_gr1")
        assert out[12] is None
        assert out[13] == pytest.approx((30.0 - 10.0) / 20.0)

    def test_quarterly_file_is_null(self) -> None:
        n = 14
        df = _frame(n, emp=[float(i + 1) for i in range(n)])
        out = df.with_columns(emp_gr("data/interim/acc_std_qtr.parquet"))["emp_gr1"].to_list()
        assert out == [None] * n

    def test_null_employment(self) -> None:
        n = 14
        emp: list[float | None] = [10.0] * n
        emp[12] = None
        df = _frame(n, emp=emp)
        out = _out(df, emp_gr("acc_std_ann.parquet"), "emp_gr1")
        assert out[12] is None


@pytest.mark.unit
class TestTangibility:
    def test_formula(self) -> None:
        df = pl.DataFrame(
            {
                "che": [10.0],
                "rect": [20.0],
                "invt": [30.0],
                "ppegt": [40.0],
                "at_x": [200.0],
            }
        )
        expected = pl.DataFrame(
            {"tangibility": [(10.0 + 0.715 * 20.0 + 0.547 * 30.0 + 0.535 * 40.0) / 200.0]}
        )
        assert_frame_equal(df.select(tangibility()), expected)

    def test_zero_assets_null_negative_assets_kept(self) -> None:
        df = pl.DataFrame(
            {
                "che": [10.0, 10.0],
                "rect": [0.0, 0.0],
                "invt": [0.0, 0.0],
                "ppegt": [0.0, 0.0],
                "at_x": [0.0, -5.0],
            }
        )
        out = df.select(tangibility())["tangibility"].to_list()
        assert out[0] is None
        assert out[1] == pytest.approx(10.0 / -5.0)

    def test_null_component_propagates(self) -> None:
        df = pl.DataFrame(
            {
                "che": [10.0],
                "rect": [None],
                "invt": [1.0],
                "ppegt": [1.0],
                "at_x": [100.0],
            },
            schema_overrides={"rect": pl.Float64},
        )
        assert df.select(tangibility())["tangibility"][0] is None
