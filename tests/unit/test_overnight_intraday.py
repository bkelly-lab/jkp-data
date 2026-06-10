"""
Tests for overnight and intraday return decomposition.

Validates the Lou, Polk, and Skouras (2019) return decomposition:
    r_intraday  = P_close / P_open  - 1
    r_overnight = (1 + r) / (1 + r_intraday) - 1

And the identity: (1 + r_intraday)(1 + r_overnight) = (1 + r).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from tests.conftest import ToleranceSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_prices() -> pl.DataFrame:
    """Five-day panel for two stocks with known prices."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
            ],
            "prc": [100.0, 110.0, 105.0, 108.0, 112.0, 50.0, 52.0, 48.0, 51.0, 53.0],
            "prc_open": [98.0, 108.0, 107.0, 104.0, 109.0, 49.0, 51.0, 50.0, 47.0, 52.0],
            "ret": [
                None,
                0.10,
                -0.04545454545,
                0.02857142857,
                0.03703703704,
                None,
                0.04,
                -0.07692307692,
                0.0625,
                0.03921568627,
            ],
            "eom": [
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 1, 31),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Daily return decomposition
# ---------------------------------------------------------------------------


class TestDailyReturnDecomposition:
    """Test daily intraday and overnight return computation."""

    def test_intraday_formula(self, daily_prices: pl.DataFrame):
        """ret_intraday = P_close / P_open - 1."""
        result = daily_prices.with_columns(
            ret_intraday=pl.when(pl.col("prc_open") > 0)
            .then(pl.col("prc") / pl.col("prc_open") - 1)
            .otherwise(None)
        )

        expected = daily_prices["prc"].to_numpy() / daily_prices["prc_open"].to_numpy() - 1
        np.testing.assert_allclose(
            result["ret_intraday"].to_numpy(), expected, **ToleranceSpec.TIGHT
        )

    def test_overnight_formula(self, daily_prices: pl.DataFrame):
        """ret_overnight = (1 + ret) / (1 + ret_intraday) - 1."""
        result = daily_prices.with_columns(
            ret_intraday=pl.when(pl.col("prc_open") > 0)
            .then(pl.col("prc") / pl.col("prc_open") - 1)
            .otherwise(None)
        ).with_columns(
            ret_overnight=pl.when(
                pl.col("ret_intraday").is_not_null() & pl.col("ret").is_not_null()
            )
            .then((1 + pl.col("ret")) / (1 + pl.col("ret_intraday")) - 1)
            .otherwise(None)
        )

        # For the second row of stock 1: ret=0.10, prc_open=108, prc=110
        # ret_intraday = 110/108 - 1 = 0.01851851...
        # ret_overnight = 1.10/1.01851851... - 1 = 0.08
        row = result.filter((pl.col("id") == 1) & (pl.col("date") == date(2024, 1, 3)))
        np.testing.assert_allclose(row["ret_intraday"].item(), 110 / 108 - 1, **ToleranceSpec.TIGHT)
        np.testing.assert_allclose(
            row["ret_overnight"].item(), 1.10 / (110 / 108) - 1, **ToleranceSpec.TIGHT
        )

    def test_decomposition_identity(self, daily_prices: pl.DataFrame):
        """(1 + r_intraday)(1 + r_overnight) == (1 + ret) for all non-null rows."""
        result = (
            daily_prices.with_columns(
                ret_intraday=pl.when(pl.col("prc_open") > 0)
                .then(pl.col("prc") / pl.col("prc_open") - 1)
                .otherwise(None)
            )
            .with_columns(
                ret_overnight=pl.when(
                    pl.col("ret_intraday").is_not_null() & pl.col("ret").is_not_null()
                )
                .then((1 + pl.col("ret")) / (1 + pl.col("ret_intraday")) - 1)
                .otherwise(None)
            )
            .filter(pl.col("ret").is_not_null())
        )

        recomposed = (1 + result["ret_intraday"].to_numpy()) * (
            1 + result["ret_overnight"].to_numpy()
        )
        np.testing.assert_allclose(recomposed, 1 + result["ret"].to_numpy(), **ToleranceSpec.TIGHT)


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


class TestNullHandling:
    """Test that missing open prices produce null returns."""

    def test_null_open_gives_null_intraday(self):
        """When prc_open is null, ret_intraday must be null."""
        df = pl.DataFrame(
            {
                "prc": [100.0, 110.0],
                "prc_open": [None, 105.0],
                "ret": [0.05, 0.10],
            }
        )
        result = df.with_columns(
            ret_intraday=pl.when(pl.col("prc_open") > 0)
            .then(pl.col("prc") / pl.col("prc_open") - 1)
            .otherwise(None)
        )
        assert result["ret_intraday"][0] is None
        assert result["ret_intraday"][1] is not None

    def test_null_open_gives_null_overnight(self):
        """When prc_open is null, ret_overnight must also be null."""
        df = pl.DataFrame(
            {
                "prc": [100.0, 110.0],
                "prc_open": [None, 105.0],
                "ret": [0.05, 0.10],
            }
        )
        result = df.with_columns(
            ret_intraday=pl.when(pl.col("prc_open") > 0)
            .then(pl.col("prc") / pl.col("prc_open") - 1)
            .otherwise(None)
        ).with_columns(
            ret_overnight=pl.when(
                pl.col("ret_intraday").is_not_null() & pl.col("ret").is_not_null()
            )
            .then((1 + pl.col("ret")) / (1 + pl.col("ret_intraday")) - 1)
            .otherwise(None)
        )
        assert result["ret_overnight"][0] is None
        assert result["ret_overnight"][1] is not None

    def test_zero_open_gives_null(self):
        """Open price of zero should not produce intraday return."""
        df = pl.DataFrame({"prc": [100.0], "prc_open": [0.0], "ret": [0.05]})
        result = df.with_columns(
            ret_intraday=pl.when(pl.col("prc_open") > 0)
            .then(pl.col("prc") / pl.col("prc_open") - 1)
            .otherwise(None)
        )
        assert result["ret_intraday"][0] is None

    def test_null_ret_gives_null_overnight(self):
        """When ret is null, ret_overnight must be null (even if open exists)."""
        df = pl.DataFrame({"prc": [100.0], "prc_open": [98.0], "ret": [None]})
        result = df.with_columns(
            ret_intraday=pl.when(pl.col("prc_open") > 0)
            .then(pl.col("prc") / pl.col("prc_open") - 1)
            .otherwise(None)
        ).with_columns(
            ret_overnight=pl.when(
                pl.col("ret_intraday").is_not_null() & pl.col("ret").is_not_null()
            )
            .then((1 + pl.col("ret")) / (1 + pl.col("ret_intraday")) - 1)
            .otherwise(None)
        )
        assert result["ret_intraday"][0] is not None
        assert result["ret_overnight"][0] is None


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------


class TestPortfolioAggregation:
    """Test value-weighted portfolio return aggregation with OI returns."""

    def test_vw_aggregation(self):
        """Value-weighted portfolio return = sum(w_i * r_i)."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "me": [100.0, 200.0, 300.0],
                "ret_intraday": [0.02, 0.04, -0.01],
                "ret_overnight": [0.01, -0.02, 0.03],
                "pf": [1, 1, 1],
                "eom": [date(2024, 1, 31)] * 3,
            }
        )
        total_me = df["me"].sum()
        expected_intraday_vw = (
            df["me"].to_numpy() * df["ret_intraday"].to_numpy()
        ).sum() / total_me
        expected_overnight_vw = (
            df["me"].to_numpy() * df["ret_overnight"].to_numpy()
        ).sum() / total_me

        result = df.group_by(["pf", "eom"]).agg(
            ret_intraday_vw=((pl.col("ret_intraday") * pl.col("me")).sum() / pl.col("me").sum()),
            ret_overnight_vw=((pl.col("ret_overnight") * pl.col("me")).sum() / pl.col("me").sum()),
        )

        np.testing.assert_allclose(
            result["ret_intraday_vw"].item(), expected_intraday_vw, **ToleranceSpec.TIGHT
        )
        np.testing.assert_allclose(
            result["ret_overnight_vw"].item(),
            expected_overnight_vw,
            **ToleranceSpec.TIGHT,
        )

    def test_ew_aggregation(self):
        """Equal-weighted portfolio return = mean(r_i)."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "ret_intraday": [0.02, 0.04, -0.01],
                "pf": [1, 1, 1],
                "eom": [date(2024, 1, 31)] * 3,
            }
        )
        expected_ew = np.mean([0.02, 0.04, -0.01])
        result = df.group_by(["pf", "eom"]).agg(
            ret_intraday_ew=pl.mean("ret_intraday"),
        )
        np.testing.assert_allclose(
            result["ret_intraday_ew"].item(), expected_ew, **ToleranceSpec.TIGHT
        )
