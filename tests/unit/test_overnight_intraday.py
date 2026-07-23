"""Unit tests for LPS (2019) overnight/intraday return decomposition.

Tests cover:
- Daily return formulas (ret_intraday, ret_overnight)
- Return identity: (1 + ret_intraday)(1 + ret_overnight) == (1 + ret)
- Null handling: missing open/close prices yield null components
- Monthly compounding via compound_overnight_intraday
- Column propagation through prepare_crsp_sf (daily and monthly)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from tests.conftest import ToleranceSpec

# ---------------------------------------------------------------------------
# Pure-formula tests (no pipeline dependency)
# ---------------------------------------------------------------------------


class TestReturnFormulas:
    """Verify LPS (2019) decomposition formulas on known values."""

    def test_basic_decomposition(self) -> None:
        """close=110, open=105, ret=0.10 -> known intraday and overnight."""
        prc_close = 110.0
        prc_open = 105.0
        ret = 0.10

        ret_intraday = prc_close / prc_open - 1
        ret_overnight = (1 + ret) / (1 + ret_intraday) - 1

        assert ret_intraday == pytest.approx(110.0 / 105.0 - 1)
        assert ret_overnight == pytest.approx((1.10) / (1 + ret_intraday) - 1)

    def test_identity_holds(self) -> None:
        """(1 + ret_intraday) * (1 + ret_overnight) == (1 + ret) exactly."""
        prc_close = 110.0
        prc_open = 105.0
        ret = 0.10

        ret_intraday = prc_close / prc_open - 1
        ret_overnight = (1 + ret) / (1 + ret_intraday) - 1

        product = (1 + ret_intraday) * (1 + ret_overnight)
        np.testing.assert_allclose(product, 1 + ret, **ToleranceSpec.TIGHT)

    @pytest.mark.parametrize(
        ("prc_close", "prc_open", "ret"),
        [
            (100.0, 100.0, 0.0),
            (50.0, 55.0, -0.05),
            (200.0, 180.0, 0.15),
            (10.0, 9.5, 0.08),
        ],
    )
    def test_identity_parametrized(self, prc_close: float, prc_open: float, ret: float) -> None:
        """Identity holds for a variety of price/return combinations."""
        ret_intraday = prc_close / prc_open - 1
        ret_overnight = (1 + ret) / (1 + ret_intraday) - 1
        product = (1 + ret_intraday) * (1 + ret_overnight)
        np.testing.assert_allclose(product, 1 + ret, **ToleranceSpec.TIGHT)

    def test_negative_return(self) -> None:
        """Identity holds when close-to-close return is negative."""
        prc_close = 95.0
        prc_open = 100.0
        ret = -0.08

        ret_intraday = prc_close / prc_open - 1
        ret_overnight = (1 + ret) / (1 + ret_intraday) - 1

        assert ret_intraday < 0
        product = (1 + ret_intraday) * (1 + ret_overnight)
        np.testing.assert_allclose(product, 1 + ret, **ToleranceSpec.TIGHT)


# ---------------------------------------------------------------------------
# Monthly compounding tests
# ---------------------------------------------------------------------------


class TestMonthlyCompounding:
    """Test compound_overnight_intraday by calling the real function."""

    @staticmethod
    def _setup_and_run(
        test_paths,
        dsf_rows: list[dict],
        msf_rows: list[dict],
    ) -> pl.DataFrame:
        """Write input parquets, run compound_overnight_intraday, return result."""
        from jkp.data.aux_functions import compound_overnight_intraday

        dsf_schema = {
            "id": pl.Int64,
            "eom": pl.Date,
            "ret_intraday": pl.Float64,
            "ret_overnight": pl.Float64,
        }
        msf_schema = {
            "id": pl.Int64,
            "eom": pl.Date,
            "ret_lag_dif": pl.Int64,
        }

        pl.DataFrame(dsf_rows, schema=dsf_schema).write_parquet(
            test_paths.interim_dir / "world_dsf.parquet"
        )
        pl.DataFrame(msf_rows, schema=msf_schema).write_parquet(
            test_paths.interim_dir / "__msf_world.parquet"
        )
        compound_overnight_intraday(test_paths)
        return pl.read_parquet(test_paths.interim_dir / "__msf_world.parquet")

    def test_normal_compounding(self, test_paths) -> None:
        """All-valid daily returns compound to prod(1 + r_d) - 1."""
        dsf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.02, "ret_overnight": 0.01},
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.03, "ret_overnight": -0.005},
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": -0.01, "ret_overnight": 0.015},
        ]
        msf = [{"id": 1, "eom": date(2020, 1, 31), "ret_lag_dif": 1}]

        result = self._setup_and_run(test_paths, dsf, msf)
        row = result.filter(pl.col("id") == 1).row(0, named=True)

        expected_intra = (1.02) * (1.03) * (0.99) - 1
        expected_over = (1.01) * (0.995) * (1.015) - 1
        assert row["ret_intraday"] == pytest.approx(expected_intra)
        assert row["ret_overnight"] == pytest.approx(expected_over)

    def test_null_intraday_propagates(self, test_paths) -> None:
        """A single null ret_intraday day nullifies the monthly intraday,
        but ret_overnight (all valid) still compounds normally."""
        dsf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.01, "ret_overnight": 0.005},
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": None, "ret_overnight": 0.01},
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.02, "ret_overnight": 0.003},
        ]
        msf = [{"id": 1, "eom": date(2020, 1, 31), "ret_lag_dif": 1}]

        result = self._setup_and_run(test_paths, dsf, msf)
        row = result.row(0, named=True)

        assert row["ret_intraday"] is None
        expected_over = (1.005) * (1.01) * (1.003) - 1
        assert row["ret_overnight"] == pytest.approx(expected_over)

    def test_null_overnight_propagates(self, test_paths) -> None:
        """A single null ret_overnight day nullifies the monthly overnight,
        but ret_intraday (all valid) still compounds normally."""
        dsf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.05, "ret_overnight": None},
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": -0.02, "ret_overnight": 0.01},
        ]
        msf = [{"id": 1, "eom": date(2020, 1, 31), "ret_lag_dif": 1}]

        result = self._setup_and_run(test_paths, dsf, msf)
        row = result.row(0, named=True)

        expected_intra = (1.05) * (0.98) - 1
        assert row["ret_intraday"] == pytest.approx(expected_intra)
        assert row["ret_overnight"] is None

    def test_lead_columns_consecutive_months(self, test_paths) -> None:
        """Lead columns are populated when the next month has ret_lag_dif == 1."""
        dsf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.04, "ret_overnight": 0.02},
            {"id": 1, "eom": date(2020, 2, 29), "ret_intraday": 0.01, "ret_overnight": 0.03},
        ]
        msf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_lag_dif": 1},
            {"id": 1, "eom": date(2020, 2, 29), "ret_lag_dif": 1},
        ]

        result = self._setup_and_run(test_paths, dsf, msf).sort(["id", "eom"])
        jan = result.filter(pl.col("eom") == date(2020, 1, 31)).row(0, named=True)
        feb = result.filter(pl.col("eom") == date(2020, 2, 29)).row(0, named=True)

        # Jan lead should equal Feb's compounded values (single-day months here)
        assert jan["ret_intraday_lead1m"] == pytest.approx(0.01)
        assert jan["ret_overnight_lead1m"] == pytest.approx(0.03)
        # Feb has no successor → leads are null
        assert feb["ret_intraday_lead1m"] is None
        assert feb["ret_overnight_lead1m"] is None

    def test_lead_columns_non_consecutive(self, test_paths) -> None:
        """Lead columns are null when the next month has ret_lag_dif != 1."""
        dsf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.04, "ret_overnight": 0.02},
            {"id": 1, "eom": date(2020, 3, 31), "ret_intraday": 0.01, "ret_overnight": 0.03},
        ]
        msf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_lag_dif": 1},
            {"id": 1, "eom": date(2020, 3, 31), "ret_lag_dif": 2},
        ]

        result = self._setup_and_run(test_paths, dsf, msf).sort(["id", "eom"])
        jan = result.filter(pl.col("eom") == date(2020, 1, 31)).row(0, named=True)

        # Gap month → leads are null even though Mar data exists
        assert jan["ret_intraday_lead1m"] is None
        assert jan["ret_overnight_lead1m"] is None

    def test_msf_id_not_in_dsf_gets_nulls(self, test_paths) -> None:
        """An id present in msf but absent from dsf gets null return columns."""
        dsf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_intraday": 0.02, "ret_overnight": 0.01},
        ]
        msf = [
            {"id": 1, "eom": date(2020, 1, 31), "ret_lag_dif": 1},
            {"id": 99, "eom": date(2020, 1, 31), "ret_lag_dif": 1},
        ]

        result = self._setup_and_run(test_paths, dsf, msf)
        missing = result.filter(pl.col("id") == 99).row(0, named=True)

        assert missing["ret_intraday"] is None
        assert missing["ret_overnight"] is None


# ---------------------------------------------------------------------------
# Pipeline integration tests (prepare_crsp_sf)
# ---------------------------------------------------------------------------


class TestPrepareCrspSfIntegration:
    """Test that prepare_crsp_sf produces ret_intraday/ret_overnight."""

    @staticmethod
    def _run(
        paths,
        freq: str,
        crsp_rows: list[dict],
    ) -> pl.DataFrame:
        from jkp.data.aux_functions import prepare_crsp_sf
        from tests.golden.prepare_crsp_sf_inputs import (
            SCHEMA_CRSP_SF,
            SCHEMA_FF,
            SCHEMA_MCTI,
            SCHEMA_SEDELIST,
        )

        raw = paths.interim_dir / "raw_data_dfs"
        raw.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(crsp_rows, schema=SCHEMA_CRSP_SF).write_parquet(
            raw / f"__crsp_sf_{freq}.parquet"
        )
        pl.DataFrame([], schema=SCHEMA_SEDELIST).write_parquet(raw / f"crsp_{freq}sedelist.parquet")
        pl.DataFrame(schema=SCHEMA_MCTI).write_parquet(raw / "crsp_mcti_t30ret.parquet")
        pl.DataFrame(schema=SCHEMA_FF).write_parquet(raw / "ff_factors_monthly.parquet")
        prepare_crsp_sf(paths, freq)
        return pl.read_parquet(paths.interim_dir / f"crsp_{freq}sf.parquet")

    def test_daily_has_return_columns(self, test_paths) -> None:
        """Daily output includes ret_intraday and ret_overnight columns."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        rows = [
            _crsp_row(
                1,
                1,
                date(2020, 1, 6),
                110.0,
                1.0,
                0.10,
                0.10,
                100,
                50.0,
                nasdaq=False,
                prc_open=105.0,
                prc_close=110.0,
            )
        ]
        df = self._run(test_paths, "d", rows)
        assert "ret_intraday" in df.columns
        assert "ret_overnight" in df.columns

    def test_daily_return_identity(self, test_paths) -> None:
        """(1 + ret_intraday) * (1 + ret_overnight) == (1 + ret) across rows."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        cases = [
            (1, date(2020, 1, 6), 110.0, 105.0, 0.10),
            (2, date(2020, 1, 7), 95.0, 100.0, -0.08),
            (3, date(2020, 1, 8), 200.0, 180.0, 0.15),
            (4, date(2020, 1, 9), 50.0, 55.0, -0.05),
        ]
        rows = [
            _crsp_row(
                permno,
                permno,
                d,
                prc_close,
                1.0,
                ret,
                ret,
                100,
                50.0,
                nasdaq=False,
                prc_open=prc_open,
                prc_close=prc_close,
            )
            for permno, d, prc_close, prc_open, ret in cases
        ]
        df = self._run(test_paths, "d", rows)
        product = (1 + df["ret_intraday"]) * (1 + df["ret_overnight"])
        np.testing.assert_allclose(
            product.to_numpy(), (1 + df["ret"]).to_numpy(), **ToleranceSpec.TIGHT
        )

    def test_daily_intraday_formula(self, test_paths) -> None:
        """ret_intraday = prc_close / prc_open - 1 (uses dlyclose, not dlyprc)."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        prc_open = 100.0
        prc_close = 105.0
        rows = [
            _crsp_row(
                1,
                1,
                date(2020, 1, 6),
                999.0,
                1.0,
                0.10,
                0.10,
                100,
                50.0,
                nasdaq=False,
                prc_open=prc_open,
                prc_close=prc_close,
            )
        ]
        df = self._run(test_paths, "d", rows)
        expected = prc_close / prc_open - 1
        assert df["ret_intraday"][0] == pytest.approx(expected)

    def test_daily_null_when_no_open(self, test_paths) -> None:
        """When prc_open is null, ret_intraday and ret_overnight are null."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        rows = [
            _crsp_row(
                1,
                1,
                date(2020, 1, 6),
                110.0,
                1.0,
                0.10,
                0.10,
                100,
                50.0,
                nasdaq=False,
                prc_open=None,
                prc_close=110.0,
            )
        ]
        df = self._run(test_paths, "d", rows)
        assert df["ret_intraday"][0] is None
        assert df["ret_overnight"][0] is None

    def test_daily_null_when_no_close(self, test_paths) -> None:
        """When prc_close is null, both components are null."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        rows = [
            _crsp_row(
                1,
                1,
                date(2020, 1, 6),
                110.0,
                1.0,
                0.10,
                0.10,
                100,
                50.0,
                nasdaq=False,
                prc_open=105.0,
                prc_close=None,
            )
        ]
        df = self._run(test_paths, "d", rows)
        assert df["ret_intraday"][0] is None
        assert df["ret_overnight"][0] is None

    def test_daily_null_ret(self, test_paths) -> None:
        """When ret is null, ret_intraday is computed but ret_overnight is null."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        rows = [
            _crsp_row(
                1,
                1,
                date(2020, 1, 6),
                110.0,
                1.0,
                None,
                None,
                100,
                50.0,
                nasdaq=False,
                prc_open=105.0,
                prc_close=110.0,
            )
        ]
        df = self._run(test_paths, "d", rows)
        assert df["ret_intraday"][0] == pytest.approx(110.0 / 105.0 - 1)
        assert df["ret_overnight"][0] is None

    def test_daily_zero_open_price(self, test_paths) -> None:
        """prc_open == 0 is treated as missing (guard: prc_open > 0)."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        rows = [
            _crsp_row(
                1,
                1,
                date(2020, 1, 6),
                110.0,
                1.0,
                0.10,
                0.10,
                100,
                50.0,
                nasdaq=False,
                prc_open=0.0,
                prc_close=110.0,
            )
        ]
        df = self._run(test_paths, "d", rows)
        assert df["ret_intraday"][0] is None
        assert df["ret_overnight"][0] is None

    def test_daily_negative_open_price(self, test_paths) -> None:
        """Negative prc_open is treated as missing."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        rows = [
            _crsp_row(
                1,
                1,
                date(2020, 1, 6),
                110.0,
                1.0,
                0.10,
                0.10,
                100,
                50.0,
                nasdaq=False,
                prc_open=-5.0,
                prc_close=110.0,
            )
        ]
        df = self._run(test_paths, "d", rows)
        assert df["ret_intraday"][0] is None
        assert df["ret_overnight"][0] is None

    def test_monthly_no_return_columns(self, test_paths) -> None:
        """Monthly output does not contain ret_intraday/ret_overnight
        (they are compounded from daily in a separate step)."""
        from tests.golden.prepare_crsp_sf_inputs import _crsp_row

        rows = [_crsp_row(1, 1, date(2020, 1, 31), 110.0, 1.0, 0.10, 0.10, 100, 50.0, nasdaq=False)]
        df = self._run(test_paths, "m", rows)
        assert "ret_intraday" not in df.columns
        assert "ret_overnight" not in df.columns
