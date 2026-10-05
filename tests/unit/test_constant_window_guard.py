"""Unit tests for constant-window guards and the non-finite backstop in finish_daily_chars.

Verifies that rolling daily characteristics produce null (not NaN, inf, fake 0.0,
or fake 1.0) when the input window is constant or near-dead, and that
finish_daily_chars scrubs any residual non-finite values.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import (
    CONSTANT_MAX_ZERO_FRAC,
    _guard_constant,
    ami,
    capm,
    capm_ext,
    dimsonbeta,
    dolvol,
    downbeta,
    ff3,
    finish_daily_chars,
    hxz4,
    mktcorr,
    prc_to_high,
    rmax,
    rvol,
    skew,
    turnover,
)

# ---------------------------------------------------------------------------
# Helper: _guard_constant expression
# ---------------------------------------------------------------------------


class TestGuardConstantHelper:
    """Tests for the _guard_constant expression helper."""

    def test_constant_series_returns_none(self):
        df = pl.DataFrame({"x": [5.0, 5.0, 5.0], "grp": [1, 1, 1]})
        result = df.group_by("grp").agg(
            _guard_constant("x", pl.col("x").std(), max_zero_frac=None).alias("val")
        )
        assert result["val"][0] is None

    def test_varying_series_returns_value(self, tolerance):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "grp": [1, 1, 1]})
        result = df.group_by("grp").agg(
            _guard_constant("x", pl.col("x").std(), max_zero_frac=None).alias("val")
        )
        np.testing.assert_allclose(result["val"][0], 1.0, **tolerance.STANDARD)

    def test_all_null_returns_none(self):
        df = pl.DataFrame({"x": pl.Series([None, None, None], dtype=pl.Float64), "grp": [1, 1, 1]})
        result = df.group_by("grp").agg(
            _guard_constant("x", pl.col("x").std(), max_zero_frac=None).alias("val")
        )
        assert result["val"][0] is None

    def test_single_element_returns_none(self):
        df = pl.DataFrame({"x": [7.0], "grp": [1]})
        result = df.group_by("grp").agg(
            _guard_constant("x", pl.col("x").std(), max_zero_frac=None).alias("val")
        )
        assert result["val"][0] is None

    def test_near_dead_zero_fraction_nulls(self):
        """One nonzero in 20 days (~0.95 zeros) exceeds CONSTANT_MAX_ZERO_FRAC."""
        n = 20
        vals = [0.0] * (n - 1) + [0.05]
        df = pl.DataFrame({"x": vals, "grp": [1] * n})
        result = df.group_by("grp").agg(_guard_constant("x", pl.col("x").std()).alias("val"))
        assert result["val"][0] is None
        assert (n - 1) / n >= CONSTANT_MAX_ZERO_FRAC


# ---------------------------------------------------------------------------
# Per-stat constant-window regression tests
# ---------------------------------------------------------------------------

N = 20


def _const_ret_df(value: float = 0.0, n: int = N) -> pl.DataFrame:
    """DataFrame with constant ret_exc (and ret) for one (id_int, group_number)."""
    return pl.DataFrame(
        {
            "id_int": [1] * n,
            "group_number": [10] * n,
            "ret_exc": [value] * n,
            "ret": [value] * n,
        }
    )


def _rf_stepped_dead_df(n_months: int = 12, days_per_month: int = 21) -> pl.DataFrame:
    """Dead stock (ret==0) whose ret_exc varies with monthly RF steps.

    Mimics multi-month windows where guarding on ret_exc would incorrectly pass.
    """
    rows: list[dict] = []
    for m in range(n_months):
        rf_daily = 0.001 * (m + 1) / 21.0
        for d in range(days_per_month):
            rows.append(
                {
                    "id_int": 1,
                    "group_number": 10,
                    "ret": 0.0,
                    "ret_exc": -rf_daily,
                    "mktrf": 0.001 * ((d % 7) - 3),
                    "aux_date": date(2020, 1, 1),
                }
            )
    return pl.DataFrame(rows)


class TestConstantWindowRvol:
    def test_constant_returns_give_null_rvol(self):
        result = rvol(_const_ret_df(), "_21d", __min=15)
        assert result["rvol_21d"][0] is None

    def test_varying_returns_give_finite_rvol(self):
        df = _const_ret_df()
        df = df.with_columns(
            ret_exc=pl.lit(0.01) * pl.int_range(pl.len()),
            ret=pl.lit(0.01) * pl.int_range(pl.len()),
        )
        result = rvol(df, "_21d", __min=15)
        assert result["rvol_21d"][0] is not None
        assert np.isfinite(result["rvol_21d"][0])

    def test_rf_stepped_dead_stock_nulls_rvol_252d(self):
        """ret constant, ret_exc stepped by month → still null (guard keys on ret)."""
        result = rvol(_rf_stepped_dead_df(), "_252d", __min=120)
        assert result["rvol_252d"][0] is None


class TestConstantWindowRmax:
    def test_constant_returns_give_null_rmax(self):
        result = rmax(_const_ret_df(), "_21d", __min=15)
        assert result["rmax5_21d"][0] is None
        assert result["rmax1_21d"][0] is None


class TestConstantWindowSkew:
    def test_constant_returns_give_null_rskew(self):
        result = skew(_const_ret_df(), "_21d", __min=15)
        assert result["rskew_21d"][0] is None


class TestConstantWindowCapm:
    def _df(self, const_ret: float = 0.0, n: int = N) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "mktrf": [0.001 * ((i % 7) - 3) for i in range(n)],
                "ret_exc": [const_ret] * n,
                "ret": [const_ret] * n,
                "aux_date": [date(2020, 1, 1 + i) for i in range(n)],
            }
        )

    def test_constant_ret_gives_null_ivol(self):
        result = capm(self._df(), "_21d", __min=15)
        assert result["ivol_capm_21d"][0] is None

    def test_constant_ret_nulls_beta(self):
        result = capm(self._df(), "_21d", __min=15)
        assert result["beta_21d"][0] is None

    def test_rf_stepped_dead_stock_nulls_capm_252d(self):
        df = _rf_stepped_dead_df()
        result = capm(df, "_252d", __min=120)
        assert result["beta_252d"][0] is None
        assert result["ivol_capm_252d"][0] is None


class TestConstantWindowCapmExt:
    def _df(self, n: int = N) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "mktrf": [0.001 * ((i % 7) - 3) for i in range(n)],
                "ret_exc": [0.0] * n,
                "ret": [0.0] * n,
            }
        )

    def test_constant_ret_nulls_ivol_iskew_coskew(self):
        result = capm_ext(self._df(), "_21d", __min=15)
        assert result["ivol_capm_21d"][0] is None
        assert result["iskew_capm_21d"][0] is None
        assert result["coskew_21d"][0] is None

    def test_constant_ret_nulls_beta(self):
        result = capm_ext(self._df(), "_21d", __min=15)
        assert result["beta_21d"][0] is None


class TestConstantWindowFf3:
    def _df(self, n: int = N) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "mktrf": [0.001 * ((i % 7) - 3) for i in range(n)],
                "smb_ff": [0.002 * ((i % 5) - 2) for i in range(n)],
                "hml": [0.001 * ((i % 3) - 1) for i in range(n)],
                "ret_exc": [0.0] * n,
                "ret": [0.0] * n,
            }
        )

    def test_constant_ret_gives_null_ivol_and_iskew(self):
        result = ff3(self._df(), "_21d", __min=15)
        assert result["ivol_ff3_21d"][0] is None
        assert result["iskew_ff3_21d"][0] is None


class TestConstantWindowHxz4:
    def _df(self, n: int = N) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "mktrf": [0.001 * ((i % 7) - 3) for i in range(n)],
                "smb_hxz": [0.002 * ((i % 5) - 2) for i in range(n)],
                "roe": [0.001 * ((i % 3) - 1) for i in range(n)],
                "inv": [0.001 * ((i % 4) - 2) for i in range(n)],
                "ret_exc": [0.0] * n,
                "ret": [0.0] * n,
            }
        )

    def test_constant_ret_gives_null_ivol_and_iskew(self):
        result = hxz4(self._df(), "_21d", __min=15)
        assert result["ivol_hxz4_21d"][0] is None
        assert result["iskew_hxz4_21d"][0] is None


class TestConstantWindowPrcToHigh:
    def test_constant_price_gives_null(self):
        n = 20
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "date": [date(2024, 1, 1 + i) for i in range(n)],
                "prc_adj": [50.0] * n,
            }
        )
        result = prc_to_high(df, "_252d", __min=1)
        assert result["prc_highprc_252d"][0] is None

    def test_varying_price_gives_value(self, tolerance):
        df = pl.DataFrame(
            {
                "id_int": [1, 1, 1, 1, 1],
                "group_number": [10, 10, 10, 10, 10],
                "date": [date(2024, 1, d) for d in range(1, 6)],
                "prc_adj": [10.0, 12.0, 11.0, 15.0, 13.0],
            }
        )
        result = prc_to_high(df, "_252d", __min=1)
        np.testing.assert_allclose(result["prc_highprc_252d"][0], 13.0 / 15.0, **tolerance.STANDARD)


class TestConstantWindowMktcorr:
    def test_constant_ret_gives_null_corr(self):
        n = 800
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "ret": [0.0] * n,
                "ret_exc_3l": [0.0] * n,
                "mkt_exc_3l": [0.001 * ((i % 7) - 3) for i in range(n)],
            }
        )
        result = mktcorr(df, "_1260d", __min=750)
        assert result["corr_1260d"][0] is None

    def test_rf_stepped_dead_stock_nulls_corr(self):
        """ret==0 every day but ret_exc_3l varies with RF → still null."""
        n = 800
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "ret": [0.0] * n,
                "ret_exc_3l": [-0.001 * ((i // 21) + 1) for i in range(n)],
                "mkt_exc_3l": [0.001 * ((i % 7) - 3) for i in range(n)],
            }
        )
        result = mktcorr(df, "_1260d", __min=750)
        assert result["corr_1260d"][0] is None


class TestConstantWindowAmi:
    def test_constant_zero_returns_with_nonzero_dolvol(self):
        """Dead stock with volume must null ami (not rank as most liquid at 0.0)."""
        n = 80
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "ret": [0.0] * n,
                "dolvol_d": [1e6] * n,
            }
        )
        result = ami(df, "_126d", __min=60)
        assert result["ami_126d"][0] is None

    def test_all_zero_dolvol_gives_null(self):
        n = 80
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "ret": [0.01 * ((i % 5) - 2) for i in range(n)],
                "dolvol_d": [0.0] * n,
            }
        )
        result = ami(df, "_126d", __min=60)
        assert result["ami_126d"][0] is None


class TestConstantWindowDolvolVar:
    def test_constant_dolvol_nulls_var(self):
        n = 80
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "dolvol_d": [1e6] * n,
            }
        )
        result = dolvol(df, "_126d", __min=60)
        assert result["dolvol_var_126d"][0] is None


class TestConstantWindowTurnoverVar:
    def test_constant_turnover_nulls_var(self):
        n = 80
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "tvol": [100.0] * n,
                "shares": [1.0] * n,
            }
        )
        result = turnover(df, "_126d", __min=60)
        assert result["turnover_var_126d"][0] is None


class TestConstantWindowDownbeta:
    def test_constant_ret_nulls_betadown(self):
        n = 40
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "mktrf": [-0.01] * n,
                "ret_exc": [0.0] * n,
                "ret": [0.0] * n,
            }
        )
        result = downbeta(df, "_252d", __min=20)
        assert len(result) == 1
        assert result["betadown_252d"][0] is None


class TestConstantWindowDimsonbeta:
    def test_constant_ret_nulls_dimson(self):
        n = 20
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "mktrf": [0.001 * ((i % 7) - 3) for i in range(n)],
                "mktrf_ld1": [0.001 * ((i % 5) - 2) for i in range(n)],
                "mktrf_lg1": [0.001 * ((i % 3) - 1) for i in range(n)],
                "ret_exc": [0.0] * n,
                "ret": [0.0] * n,
            }
        )
        result = dimsonbeta(df, "_21d", __min=15)
        assert len(result) == 0  # filter drops null beta


# ---------------------------------------------------------------------------
# Backstop: finish_daily_chars non-finite scrub
# ---------------------------------------------------------------------------


class TestFinishDailyCharsBackstop:
    def test_inf_and_nan_scrubbed_to_null(self, test_paths):
        """Non-finite values in rolled metrics should become null after backstop."""
        cs_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "eom": [date(2024, 1, 31)] * 3,
                "bidaskhl_21d": [0.01, 0.02, 0.03],
            }
        )
        roll_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "eom": [date(2024, 1, 31)] * 3,
                "rvol_21d": [0.05, float("nan"), float("inf")],
                "rvol_252d": [0.10, 0.20, 0.0],
                "rmax5_21d": [0.02, 0.03, 0.04],
                "corr_1260d": [0.5, float("-inf"), 0.8],
                "__mktvol_252d": [0.15, 0.15, 0.15],
            }
        )

        cs_df.write_parquet(test_paths.interim_dir / "corwin_schultz.parquet")
        roll_df.write_parquet(test_paths.interim_dir / "roll_apply_daily.parquet")

        out = test_paths.interim_dir / "market_chars_d.parquet"
        finish_daily_chars(test_paths, out)
        result = pl.read_parquet(out)

        float_cols = [name for name, dtype in result.schema.items() if dtype.is_float()]
        for c in float_cols:
            series = result[c]
            non_null = series.drop_nulls()
            assert non_null.is_finite().all(), (
                f"Column {c} has non-finite values after backstop: {non_null.to_list()}"
            )

    def test_finite_values_preserved(self, test_paths):
        """Legitimate finite values pass through the backstop unchanged."""
        cs_df = pl.DataFrame(
            {
                "id": [1],
                "eom": [date(2024, 1, 31)],
                "bidaskhl_21d": [0.01],
            }
        )
        roll_df = pl.DataFrame(
            {
                "id": [1],
                "eom": [date(2024, 1, 31)],
                "rvol_21d": [0.05],
                "rvol_252d": [0.10],
                "rmax5_21d": [0.02],
                "corr_1260d": [0.5],
                "__mktvol_252d": [0.15],
            }
        )

        cs_df.write_parquet(test_paths.interim_dir / "corwin_schultz.parquet")
        roll_df.write_parquet(test_paths.interim_dir / "roll_apply_daily.parquet")

        out = test_paths.interim_dir / "market_chars_d.parquet"
        finish_daily_chars(test_paths, out)
        result = pl.read_parquet(out)

        assert result["rvol_21d"][0] == pytest.approx(0.05)
        assert result["corr_1260d"][0] == pytest.approx(0.5)
