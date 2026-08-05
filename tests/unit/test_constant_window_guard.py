"""Unit tests for constant-window guards and the non-finite backstop in finish_daily_chars.

Verifies that rolling daily characteristics produce null (not NaN, inf, fake 0.0,
or fake 1.0) when the input window is constant, and that finish_daily_chars scrubs
any residual non-finite values.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from jkp.data.aux_functions import (
    _guard_constant,
    ami,
    capm,
    capm_ext,
    ff3,
    finish_daily_chars,
    hxz4,
    mktcorr,
    prc_to_high,
    rmax,
    rvol,
    skew,
)

# ---------------------------------------------------------------------------
# Helper: _guard_constant expression
# ---------------------------------------------------------------------------


class TestGuardConstantHelper:
    """Tests for the _guard_constant expression helper."""

    def test_constant_series_returns_none(self):
        df = pl.DataFrame({"x": [5.0, 5.0, 5.0], "grp": [1, 1, 1]})
        result = df.group_by("grp").agg(_guard_constant("x", pl.col("x").std()).alias("val"))
        assert result["val"][0] is None

    def test_varying_series_returns_value(self, tolerance):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "grp": [1, 1, 1]})
        result = df.group_by("grp").agg(_guard_constant("x", pl.col("x").std()).alias("val"))
        np.testing.assert_allclose(result["val"][0], 1.0, **tolerance.STANDARD)

    def test_all_null_returns_none(self):
        df = pl.DataFrame({"x": pl.Series([None, None, None], dtype=pl.Float64), "grp": [1, 1, 1]})
        result = df.group_by("grp").agg(_guard_constant("x", pl.col("x").std()).alias("val"))
        assert result["val"][0] is None

    def test_single_element_returns_none(self):
        df = pl.DataFrame({"x": [7.0], "grp": [1]})
        result = df.group_by("grp").agg(_guard_constant("x", pl.col("x").std()).alias("val"))
        assert result["val"][0] is None


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


class TestConstantWindowRvol:
    def test_constant_returns_give_null_rvol(self):
        result = rvol(_const_ret_df(), "_21d", __min=15)
        assert result["rvol_21d"][0] is None

    def test_varying_returns_give_finite_rvol(self):
        df = _const_ret_df()
        df = df.with_columns(ret_exc=pl.lit(0.01) * pl.int_range(pl.len()))
        result = rvol(df, "_21d", __min=15)
        assert result["rvol_21d"][0] is not None
        assert np.isfinite(result["rvol_21d"][0])


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
                "aux_date": [date(2020, 1, 1 + i) for i in range(n)],
            }
        )

    def test_constant_ret_exc_gives_null_ivol(self):
        result = capm(self._df(), "_21d", __min=15)
        assert result["ivol_capm_21d"][0] is None

    def test_constant_ret_exc_keeps_beta(self):
        """Beta divides by market variance; it remains defined."""
        result = capm(self._df(), "_21d", __min=15)
        assert result["beta_21d"][0] is not None


class TestConstantWindowCapmExt:
    def _df(self, n: int = N) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "mktrf": [0.001 * ((i % 7) - 3) for i in range(n)],
                "ret_exc": [0.0] * n,
            }
        )

    def test_constant_ret_exc_nulls_ivol_iskew_coskew(self):
        result = capm_ext(self._df(), "_21d", __min=15)
        assert result["ivol_capm_21d"][0] is None
        assert result["iskew_capm_21d"][0] is None
        assert result["coskew_21d"][0] is None

    def test_constant_ret_exc_keeps_beta(self):
        result = capm_ext(self._df(), "_21d", __min=15)
        assert result["beta_21d"][0] is not None


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
            }
        )

    def test_constant_ret_exc_gives_null_ivol_and_iskew(self):
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
            }
        )

    def test_constant_ret_exc_gives_null_ivol_and_iskew(self):
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
    def test_constant_ret_exc_3l_gives_null_corr(self):
        n = 800
        df = pl.DataFrame(
            {
                "id_int": [1] * n,
                "group_number": [10] * n,
                "ret_exc_3l": [0.0] * n,
                "mkt_exc_3l": [0.001 * ((i % 7) - 3) for i in range(n)],
            }
        )
        result = mktcorr(df, "_1260d", __min=750)
        assert result["corr_1260d"][0] is None


class TestConstantWindowAmi:
    def test_constant_zero_returns_with_nonzero_dolvol(self):
        """All-zero returns give |ret|/dolvol = 0 for every day → mean = 0.0.

        The mean itself is finite, but the ami for a dead stock is misleading;
        the non-finite backstop in finish_daily_chars handles any edge-case inf.
        """
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
        val = result["ami_126d"][0]
        assert val is None or val == 0.0

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
