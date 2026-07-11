"""
Tests for LPS (2019) overnight/intraday return decomposition.

Validates the formulas:
    ret_intraday  = P_close / P_open - 1
    ret_overnight = (1 + ret) / (1 + ret_intraday) - 1

and the identity:
    (1 + ret_intraday)(1 + ret_overnight) = (1 + ret)
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _compute_oi_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the same LPS return decomposition logic used in the pipeline."""
    return df.with_columns(
        ret_intraday=pl.when(
            pl.col("prc_close").is_not_null()
            & pl.col("prc_open").is_not_null()
            & (pl.col("prc_open") > 0)
        )
        .then(pl.col("prc_close") / pl.col("prc_open") - 1)
        .otherwise(None),
    ).with_columns(
        ret_overnight=pl.when(pl.col("ret_intraday").is_not_null() & pl.col("ret").is_not_null())
        .then((1 + pl.col("ret")) / (1 + pl.col("ret_intraday")) - 1)
        .otherwise(None),
    )


class TestLPSReturnDecomposition:
    """Validate LPS (2019) overnight/intraday return decomposition."""

    def test_known_values(self) -> None:
        """close=110, open=105, ret=0.10 → known intraday and overnight."""
        df = pl.DataFrame({"prc_close": [110.0], "prc_open": [105.0], "ret": [0.10]})
        result = _compute_oi_returns(df)

        expected_intra = 110.0 / 105.0 - 1
        expected_overnight = (1 + 0.10) / (1 + expected_intra) - 1

        np.testing.assert_allclose(result["ret_intraday"][0], expected_intra, rtol=1e-12)
        np.testing.assert_allclose(result["ret_overnight"][0], expected_overnight, rtol=1e-12)

    def test_identity_holds(self) -> None:
        """(1 + ret_intraday)(1 + ret_overnight) == (1 + ret) for various values."""
        df = pl.DataFrame(
            {
                "prc_close": [110.0, 95.0, 50.0, 200.0, 100.0],
                "prc_open": [105.0, 100.0, 48.0, 190.0, 100.0],
                "ret": [0.10, -0.05, 0.04, 0.06, 0.02],
            }
        )
        result = _compute_oi_returns(df)

        lhs = (1 + result["ret_intraday"].to_numpy()) * (1 + result["ret_overnight"].to_numpy())
        rhs = 1 + result["ret"].to_numpy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_null_when_open_missing(self) -> None:
        """Both component returns are null when prc_open is null."""
        df = pl.DataFrame({"prc_close": [110.0], "prc_open": [None], "ret": [0.10]})
        result = _compute_oi_returns(df)
        assert result["ret_intraday"][0] is None
        assert result["ret_overnight"][0] is None

    def test_null_when_open_zero(self) -> None:
        """Both component returns are null when prc_open is zero."""
        df = pl.DataFrame({"prc_close": [110.0], "prc_open": [0.0], "ret": [0.10]})
        result = _compute_oi_returns(df)
        assert result["ret_intraday"][0] is None
        assert result["ret_overnight"][0] is None

    def test_null_when_open_negative(self) -> None:
        """Both component returns are null when prc_open is negative."""
        df = pl.DataFrame({"prc_close": [110.0], "prc_open": [-5.0], "ret": [0.10]})
        result = _compute_oi_returns(df)
        assert result["ret_intraday"][0] is None
        assert result["ret_overnight"][0] is None

    def test_null_when_close_missing(self) -> None:
        """Both component returns are null when prc_close is null."""
        df = pl.DataFrame({"prc_close": [None], "prc_open": [105.0], "ret": [0.10]})
        result = _compute_oi_returns(df)
        assert result["ret_intraday"][0] is None
        assert result["ret_overnight"][0] is None

    def test_null_when_ret_missing(self) -> None:
        """ret_intraday is computed but ret_overnight is null when ret is null."""
        df = pl.DataFrame({"prc_close": [110.0], "prc_open": [105.0], "ret": [None]})
        result = _compute_oi_returns(df)
        assert result["ret_intraday"][0] is not None
        assert result["ret_overnight"][0] is None

    def test_zero_intraday(self) -> None:
        """When close == open, ret_intraday = 0 and ret_overnight = ret."""
        df = pl.DataFrame({"prc_close": [100.0], "prc_open": [100.0], "ret": [0.05]})
        result = _compute_oi_returns(df)
        np.testing.assert_allclose(result["ret_intraday"][0], 0.0, atol=1e-15)
        np.testing.assert_allclose(result["ret_overnight"][0], 0.05, rtol=1e-12)

    def test_negative_intraday(self) -> None:
        """Price fell during the day: close < open."""
        df = pl.DataFrame({"prc_close": [95.0], "prc_open": [100.0], "ret": [-0.02]})
        result = _compute_oi_returns(df)

        expected_intra = 95.0 / 100.0 - 1  # -0.05
        expected_overnight = (1 - 0.02) / (1 + expected_intra) - 1

        np.testing.assert_allclose(result["ret_intraday"][0], expected_intra, rtol=1e-12)
        np.testing.assert_allclose(result["ret_overnight"][0], expected_overnight, rtol=1e-12)
        lhs = (1 + result["ret_intraday"][0]) * (1 + result["ret_overnight"][0])
        np.testing.assert_allclose(lhs, 1 - 0.02, rtol=1e-12)

    def test_batch_with_mixed_nulls(self) -> None:
        """Batch of rows with some valid and some null open prices."""
        df = pl.DataFrame(
            {
                "prc_close": [110.0, None, 50.0, 100.0],
                "prc_open": [105.0, 100.0, None, 0.0],
                "ret": [0.10, 0.02, 0.04, 0.05],
            }
        )
        result = _compute_oi_returns(df)

        # Row 0: valid
        assert result["ret_intraday"][0] is not None
        assert result["ret_overnight"][0] is not None
        lhs = (1 + result["ret_intraday"][0]) * (1 + result["ret_overnight"][0])
        np.testing.assert_allclose(lhs, 1.10, rtol=1e-12)

        # Row 1: close is null
        assert result["ret_intraday"][1] is None

        # Row 2: open is null
        assert result["ret_intraday"][2] is None

        # Row 3: open is zero
        assert result["ret_intraday"][3] is None


class TestCompustatReturnDecomposition:
    """Validate LPS decomposition using prc (close) and prc_open as Compustat does."""

    def _compute_comp_oi(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compustat-style computation using prc (close) instead of prc_close."""
        return df.with_columns(
            ret_intraday=pl.when(
                pl.col("prc_open").is_not_null()
                & (pl.col("prc_open") > 0)
                & pl.col("prc").is_not_null()
            )
            .then(pl.col("prc") / pl.col("prc_open") - 1)
            .otherwise(None),
        ).with_columns(
            ret_overnight=pl.when(
                pl.col("ret_intraday").is_not_null() & pl.col("ret").is_not_null()
            )
            .then((1 + pl.col("ret")) / (1 + pl.col("ret_intraday")) - 1)
            .otherwise(None),
        )

    def test_identity_holds_compustat(self) -> None:
        """Identity holds using prc as close price."""
        df = pl.DataFrame(
            {
                "prc": [110.0, 95.0, 50.0],
                "prc_open": [105.0, 100.0, 48.0],
                "ret": [0.10, -0.05, 0.04],
            }
        )
        result = self._compute_comp_oi(df)
        lhs = (1 + result["ret_intraday"].to_numpy()) * (1 + result["ret_overnight"].to_numpy())
        rhs = 1 + result["ret"].to_numpy()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_null_open_compustat(self) -> None:
        df = pl.DataFrame({"prc": [110.0], "prc_open": [None], "ret": [0.10]})
        result = self._compute_comp_oi(df)
        assert result["ret_intraday"][0] is None
        assert result["ret_overnight"][0] is None
