"""
Tests for combine_crsp_comp_sf DuckDB implementation.

Validates each sub-operation of the CRSP/Compustat merge pipeline independently,
then runs end-to-end comparison with a Polars reference implementation on HUGE
toy datasets.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Toy data generators
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)

# Monthly date grid: 120 months (10 years)
_MONTHLY_DATES = pl.Series(
    "date",
    pl.date_range(date(2010, 1, 31), date(2019, 12, 31), "1mo", eager=True),
)

# Daily date grid: ~2500 business days
_DAILY_DATES = pl.Series(
    "date",
    [
        date(2010, 1, 4) + timedelta(days=i)
        for i in range(365 * 10)
        if (date(2010, 1, 4) + timedelta(days=i)).weekday() < 5
    ],
)


def _random_float_col(n: int, null_rate: float = 0.05) -> list[float | None]:
    vals = _RNG.randn(n).tolist()
    for i in _RNG.choice(n, int(n * null_rate), replace=False):
        vals[i] = None
    return vals


def _make_crsp_msf(tmp: Path, n_permnos: int = 500) -> None:
    """Generate crsp_msf.parquet with n_permnos x 120 months."""
    permnos = list(range(10001, 10001 + n_permnos))
    n_dates = len(_MONTHLY_DATES)
    n = n_permnos * n_dates
    permno = np.repeat(permnos, n_dates).tolist()
    dates = _MONTHLY_DATES.to_list() * n_permnos
    gvkeys = [f"{p % 100000:06d}" if _RNG.rand() > 0.3 else None for p in permno]
    iids = ["01" if _RNG.rand() > 0.1 else None for _ in range(n)]

    df = pl.DataFrame(
        {
            "permno": permno,
            "permco": [p + 10000 for p in permno],
            "gvkey": gvkeys,
            "iid": iids,
            "exch_main": _RNG.choice([1, 2, 3], n).tolist(),
            "bidask": _RNG.choice([0, 1], n).tolist(),
            "shrcd": _RNG.choice([10, 11, 12, 14, 31, None], n).tolist(),
            "exchcd": _RNG.choice([1, 2, 3, 4], n).tolist(),
            "date": dates,
            "cfacshr": _random_float_col(n, 0.02),
            "shrout": _random_float_col(n, 0.02),
            "me": _random_float_col(n),
            "me_company": _random_float_col(n),
            "prc": _random_float_col(n),
            "prc_high": _random_float_col(n),
            "prc_low": _random_float_col(n),
            "dolvol": _random_float_col(n),
            "vol": _random_float_col(n),
            "ret": _random_float_col(n),
            "ret_exc": _random_float_col(n),
            "div_tot": _random_float_col(n, 0.8),
        }
    ).cast(
        {
            "permno": pl.Int64,
            "permco": pl.Int64,
            "gvkey": pl.Utf8,
            "iid": pl.Utf8,
            "exch_main": pl.Int64,
            "bidask": pl.Int64,
            "shrcd": pl.Int64,
            "exchcd": pl.Int64,
            "date": pl.Date,
        }
    )
    df.write_parquet(tmp / "crsp_msf.parquet")


def _make_comp_msf(
    tmp: Path,
    n_gvkeys: int = 500,
    *,
    overlap_permnos: list[int] | None = None,
) -> None:
    """Generate comp_msf.parquet with n_gvkeys x 120 months.

    If overlap_permnos is provided, those gvkey values will match CRSP permnos
    to create obs_main duplicates.
    """
    n_dates = len(_MONTHLY_DATES)
    gvkeys: list[str] = []
    iids: list[str] = []
    for i in range(n_gvkeys):
        gk = f"{200000 + i:06d}"
        # Generate varied iid patterns for ID prefix testing
        if i % 20 == 0:
            iid = "01W"
        elif i % 15 == 0:
            iid = "01C"
        else:
            iid = "01"
        gvkeys.extend([gk] * n_dates)
        iids.extend([iid] * n_dates)

    n = n_gvkeys * n_dates
    dates = _MONTHLY_DATES.to_list() * n_gvkeys
    eoms = list(dates)  # Already month-end dates

    # Insert gaps: ~10% of rows get ret_lag_dif != 1
    ret_lag_dif = [1] * n
    for i in _RNG.choice(n, int(n * 0.1), replace=False):
        ret_lag_dif[i] = _RNG.choice([0, 2, 3])

    tpci_vals = ["0", "0", "0", "A", "F"]
    prcstd_vals = [1, 2, 3, 4, 4]

    df = pl.DataFrame(
        {
            "gvkey": gvkeys,
            "iid": iids,
            "excntry": _RNG.choice(["USA", "GBR", "JPN", "DEU", "CAN"], n).tolist(),
            "exch_main": _RNG.choice([1, 2, 3, 7, 15], n).tolist(),
            "primary_sec": _RNG.choice([0, 1, 1, 1], n).tolist(),
            "tpci": [tpci_vals[_RNG.randint(0, len(tpci_vals))] for _ in range(n)],
            "prcstd": [prcstd_vals[_RNG.randint(0, len(prcstd_vals))] for _ in range(n)],
            "exchg": _RNG.choice([11, 12, 13, 14, 15], n).tolist(),
            "curcdd": _RNG.choice(["USD", "GBP", "JPY", "EUR", "CAD"], n).tolist(),
            "fx": _random_float_col(n, 0.02),
            "datadate": dates,
            "eom": eoms,
            "ajexdi": _random_float_col(n, 0.02),
            "cshoc": _random_float_col(n),
            "me": _random_float_col(n),
            "prc": _random_float_col(n),
            "prc_local": _random_float_col(n),
            "prc_high": _random_float_col(n),
            "prc_low": _random_float_col(n),
            "dolvol": _random_float_col(n),
            "cshtrm": _random_float_col(n),
            "ret": _random_float_col(n),
            "ret_local": _random_float_col(n),
            "ret_exc": _random_float_col(n),
            "ret_lag_dif": ret_lag_dif,
            "div_tot": _random_float_col(n, 0.8),
            "div_cash": _random_float_col(n, 0.8),
            "div_spc": _random_float_col(n, 0.9),
        }
    ).cast(
        {
            "gvkey": pl.Utf8,
            "iid": pl.Utf8,
            "excntry": pl.Utf8,
            "exch_main": pl.Int64,
            "primary_sec": pl.Int64,
            "prcstd": pl.Int64,
            "exchg": pl.Int64,
            "curcdd": pl.Utf8,
            "datadate": pl.Date,
            "eom": pl.Date,
            "ret_lag_dif": pl.Int64,
        }
    )
    df.write_parquet(tmp / "comp_msf.parquet")


def _make_crsp_dsf(tmp: Path, n_permnos: int = 200) -> None:
    """Generate crsp_dsf.parquet with n_permnos x ~2500 trading days."""
    n_dates = len(_DAILY_DATES)
    permnos = list(range(10001, 10001 + n_permnos))
    n = n_permnos * n_dates
    permno = np.repeat(permnos, n_dates).tolist()
    dates = _DAILY_DATES.to_list() * n_permnos

    df = pl.DataFrame(
        {
            "permno": permno,
            "exch_main": _RNG.choice([1, 2, 3], n).tolist(),
            "bidask": _RNG.choice([0, 1], n).tolist(),
            "shrcd": _RNG.choice([10, 11, 12, 14, 31], n).tolist(),
            "date": dates,
            "cfacshr": _random_float_col(n, 0.02),
            "shrout": _random_float_col(n, 0.02),
            "me": _random_float_col(n),
            "dolvol": _random_float_col(n),
            "vol": _random_float_col(n),
            "prc": _random_float_col(n),
            "prc_high": _random_float_col(n),
            "prc_low": _random_float_col(n),
            "ret": _random_float_col(n),
            "ret_exc": _random_float_col(n),
        }
    ).cast(
        {
            "permno": pl.Int64,
            "exch_main": pl.Int64,
            "bidask": pl.Int64,
            "shrcd": pl.Int64,
            "date": pl.Date,
        }
    )
    df.write_parquet(tmp / "crsp_dsf.parquet")


def _make_comp_dsf(tmp: Path, n_gvkeys: int = 200) -> None:
    """Generate comp_dsf.parquet with n_gvkeys x ~2500 trading days."""
    n_dates = len(_DAILY_DATES)
    gvkeys: list[str] = []
    iids: list[str] = []
    for i in range(n_gvkeys):
        gk = f"{200000 + i:06d}"
        iid = "01W" if i % 20 == 0 else ("01C" if i % 15 == 0 else "01")
        gvkeys.extend([gk] * n_dates)
        iids.extend([iid] * n_dates)

    n = n_gvkeys * n_dates
    dates = _DAILY_DATES.to_list() * n_gvkeys

    ret_lag_dif = [1] * n
    for i in _RNG.choice(n, int(n * 0.1), replace=False):
        ret_lag_dif[i] = _RNG.choice([0, 2, 3])

    df = pl.DataFrame(
        {
            "gvkey": gvkeys,
            "iid": iids,
            "excntry": _RNG.choice(["USA", "GBR", "JPN"], n).tolist(),
            "exch_main": _RNG.choice([1, 2, 3, 7], n).tolist(),
            "primary_sec": _RNG.choice([0, 1, 1, 1], n).tolist(),
            "tpci": _RNG.choice(["0", "0", "A"], n).tolist(),
            "prcstd": _RNG.choice([1, 2, 4], n).tolist(),
            "curcdd": _RNG.choice(["USD", "GBP", "JPY"], n).tolist(),
            "fx": _random_float_col(n, 0.02),
            "datadate": dates,
            "ajexdi": _random_float_col(n, 0.02),
            "cshoc": _random_float_col(n),
            "me": _random_float_col(n),
            "dolvol": _random_float_col(n),
            "cshtrd": _random_float_col(n),
            "prc": _random_float_col(n),
            "prc_high": _random_float_col(n),
            "prc_low": _random_float_col(n),
            "ret_local": _random_float_col(n),
            "ret": _random_float_col(n),
            "ret_exc": _random_float_col(n),
            "ret_lag_dif": ret_lag_dif,
        }
    ).cast(
        {
            "gvkey": pl.Utf8,
            "iid": pl.Utf8,
            "exch_main": pl.Int64,
            "primary_sec": pl.Int64,
            "prcstd": pl.Int64,
            "datadate": pl.Date,
            "ret_lag_dif": pl.Int64,
        }
    )
    df.write_parquet(tmp / "comp_dsf.parquet")


def _write_all_fixtures(tmp: Path) -> None:
    """Write all 4 input parquet fixtures to tmp."""
    _make_crsp_msf(tmp)
    _make_comp_msf(tmp)
    _make_crsp_dsf(tmp)
    _make_comp_dsf(tmp)


# ---------------------------------------------------------------------------
# Polars reference implementation (mirrors the old helpers exactly)
# ---------------------------------------------------------------------------


def _polars_combine_crsp_comp_sf(tmp: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run a Polars-only reference implementation equivalent to the old code.

    Returns (monthly_df, daily_df).
    """
    from polars import col

    def fl_none():
        return pl.lit(None).cast(pl.Float64)

    def bo_false():
        return pl.lit(False).cast(pl.Boolean)

    # --- Normalize CRSP monthly ---
    crsp_msf = (
        pl.scan_parquet(str(tmp / "crsp_msf.parquet"))
        .with_columns(
            exch_main=col("exch_main").cast(pl.Int32),
            bidask=col("bidask").cast(pl.Int32),
            id=col("permno"),
            excntry=pl.lit("USA"),
            common=(col("shrcd").is_in([10, 11, 12]).fill_null(bo_false())).cast(pl.Int32),
            primary_sec=pl.lit(1),
            comp_tpci=pl.lit(None).cast(pl.Utf8),
            comp_exchg=pl.lit(None).cast(pl.Int64),
            curcd=pl.lit("USD"),
            fx=pl.lit(1.0),
            eom=col("date").dt.month_end(),
            prc_local=col("prc"),
            tvol=col("vol"),
            ret_local=col("ret"),
            ret_lag_dif=pl.lit(1).cast(pl.Int64),
            div_cash=fl_none(),
            div_spc=fl_none(),
            source_crsp=pl.lit(1),
        )
        .rename(
            {
                "shrcd": "crsp_shrcd",
                "exchcd": "crsp_exchcd",
                "cfacshr": "adjfct",
                "shrout": "shares",
            }
        )
    )

    # --- Normalize Compustat monthly ---
    id_exp = (
        pl.when(col("iid").str.contains("W"))
        .then(pl.lit("3") + col("gvkey") + col("iid").str.slice(0, 2))
        .when(col("iid").str.contains("C"))
        .then(pl.lit("2") + col("gvkey") + col("iid").str.slice(0, 2))
        .otherwise(pl.lit("1") + col("gvkey") + col("iid").str.slice(0, 2))
    ).cast(pl.Int64)

    comp_msf = (
        pl.scan_parquet(str(tmp / "comp_msf.parquet"))
        .with_columns(
            id=id_exp,
            permno=pl.lit(None).cast(pl.Int64),
            permco=pl.lit(None).cast(pl.Int64),
            common=pl.when(col("tpci") == "0").then(pl.lit(1)).otherwise(pl.lit(0)),
            bidask=pl.when(col("prcstd") == 4).then(pl.lit(1)).otherwise(pl.lit(0)),
            crsp_shrcd=fl_none(),
            crsp_exchcd=fl_none(),
            me_company=col("me"),
            source_crsp=pl.lit(0),
            ret_lag_dif=col("ret_lag_dif").cast(pl.Int64),
        )
        .rename(
            {
                "tpci": "comp_tpci",
                "exchg": "comp_exchg",
                "curcdd": "curcd",
                "datadate": "date",
                "ajexdi": "adjfct",
                "cshoc": "shares",
                "cshtrm": "tvol",
            }
        )
    )

    m_cols = [
        "id",
        "permno",
        "permco",
        "gvkey",
        "iid",
        "excntry",
        "exch_main",
        "common",
        "primary_sec",
        "bidask",
        "crsp_shrcd",
        "crsp_exchcd",
        "comp_tpci",
        "comp_exchg",
        "curcd",
        "fx",
        "date",
        "eom",
        "adjfct",
        "shares",
        "me",
        "me_company",
        "prc",
        "prc_local",
        "prc_high",
        "prc_low",
        "dolvol",
        "tvol",
        "ret",
        "ret_local",
        "ret_exc",
        "ret_lag_dif",
        "div_tot",
        "div_cash",
        "div_spc",
        "source_crsp",
    ]

    __msf_world = pl.concat(
        [crsp_msf.select(m_cols), comp_msf.select(m_cols)],
        how="vertical_relaxed",
    )
    __msf_world = __msf_world.sort(["id", "eom"]).with_columns(
        ret_exc_lead1m=pl.when(col("ret_lag_dif").shift(-1).over("id") != 1)
        .then(None)
        .otherwise(col("ret_exc").shift(-1).over("id"))
    )

    obs_main = (
        __msf_world.select(["id", "source_crsp", "gvkey", "iid", "eom"])
        .with_columns(count=pl.count("gvkey").over(["gvkey", "iid", "eom"]))
        .with_columns(
            obs_main=pl.when(
                (col("count").is_in([0, 1])) | ((col("count") > 1) & (col("source_crsp") == 1))
            )
            .then(1)
            .otherwise(0)
        )
        .drop(["count", "iid", "gvkey", "source_crsp"])
    )

    msf_out = (
        __msf_world.join(obs_main, on=["id", "eom"], how="left")
        .sort(["id", "eom", "source_crsp"], descending=[False, False, True])
        .unique(["id", "eom"], keep="first")
        .sort(["id", "eom"])
        .collect()
    )

    # --- Daily ---
    crsp_dsf = (
        pl.scan_parquet(str(tmp / "crsp_dsf.parquet"))
        .with_columns(
            id=col("permno"),
            excntry=pl.lit("USA"),
            common=(col("shrcd").is_in([10, 11, 12]).fill_null(bo_false())).cast(pl.Int32),
            primary_sec=pl.lit(1),
            curcd=pl.lit("USD"),
            fx=pl.lit(1.0),
            eom=col("date").dt.month_end(),
            ret_local=col("ret"),
            ret_lag_dif=pl.lit(1).cast(pl.Int64),
            exch_main=col("exch_main").cast(pl.Int32),
            bidask=col("bidask").cast(pl.Int32),
            source_crsp=pl.lit(1),
        )
        .rename({"cfacshr": "adjfct", "shrout": "shares", "vol": "tvol"})
    )

    comp_dsf = (
        pl.scan_parquet(str(tmp / "comp_dsf.parquet"))
        .with_columns(
            id=id_exp,
            common=pl.when(col("tpci") == "0").then(pl.lit(1)).otherwise(pl.lit(0)),
            bidask=pl.when(col("prcstd") == 4).then(pl.lit(1)).otherwise(pl.lit(0)),
            eom=col("datadate").dt.month_end(),
            source_crsp=pl.lit(0),
        )
        .rename(
            {
                "curcdd": "curcd",
                "datadate": "date",
                "ajexdi": "adjfct",
                "cshoc": "shares",
                "cshtrd": "tvol",
            }
        )
    )

    d_cols = [
        "id",
        "excntry",
        "exch_main",
        "common",
        "primary_sec",
        "bidask",
        "curcd",
        "fx",
        "date",
        "eom",
        "adjfct",
        "shares",
        "me",
        "dolvol",
        "tvol",
        "prc",
        "prc_high",
        "prc_low",
        "ret_local",
        "ret",
        "ret_exc",
        "ret_lag_dif",
        "source_crsp",
    ]

    __dsf_world = pl.concat(
        [crsp_dsf.select(d_cols), comp_dsf.select(d_cols)],
        how="vertical_relaxed",
    )

    dsf_out = (
        __dsf_world.join(obs_main, on=["id", "eom"], how="left")
        .sort(["id", "date", "source_crsp"], descending=[False, False, True])
        .unique(["id", "date"], keep="first")
        .sort(["id", "date"])
        .collect()
    )

    return msf_out, dsf_out


# ---------------------------------------------------------------------------
# DuckDB implementation runner
# ---------------------------------------------------------------------------


def _duckdb_combine_crsp_comp_sf(tmp: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the DuckDB implementation and return (monthly_df, daily_df)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))
    from aux_functions import combine_crsp_comp_sf

    orig_dir = os.getcwd()
    os.chdir(str(tmp))
    try:
        combine_crsp_comp_sf()
        msf = pl.read_parquet(str(tmp / "__msf_world.parquet"))
        dsf = pl.read_parquet(str(tmp / "world_dsf.parquet"))
    finally:
        os.chdir(orig_dir)
    return msf, dsf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create HUGE toy datasets once for the whole module."""
    tmp = tmp_path_factory.mktemp("combine_sf")
    _write_all_fixtures(tmp)
    return tmp


@pytest.fixture(scope="module")
def duckdb_output(toy_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _duckdb_combine_crsp_comp_sf(toy_dir)


@pytest.fixture(scope="module")
def polars_output(toy_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _polars_combine_crsp_comp_sf(toy_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cte_on_parquet(tmp: Path, sql: str) -> pl.DataFrame:
    """Execute a SQL query reading parquet files from tmp and return as Polars DF."""
    orig = os.getcwd()
    os.chdir(str(tmp))
    try:
        con = duckdb.connect()
        result = con.execute(sql).pl()
        con.close()
    finally:
        os.chdir(orig)
    return result


# =========================================================================
# Test Class 1: CRSP Normalization
# =========================================================================


class TestCrspNormalization:
    """Validate the CRSP normalization CTEs in isolation."""

    _CRSP_MSF_CTE = """
        SELECT
            permno AS id, permno, permco, gvkey, iid,
            'USA' AS excntry,
            exch_main::INT AS exch_main,
            CASE WHEN shrcd IN (10, 11, 12) THEN 1 ELSE 0 END AS common,
            1 AS primary_sec,
            bidask::INT AS bidask,
            shrcd::DOUBLE AS crsp_shrcd,
            exchcd::DOUBLE AS crsp_exchcd,
            NULL::VARCHAR AS comp_tpci,
            NULL::BIGINT AS comp_exchg,
            'USD' AS curcd,
            1.0 AS fx,
            date,
            last_day(date) AS eom,
            cfacshr AS adjfct, shrout AS shares,
            me, me_company, prc,
            prc AS prc_local,
            prc_high, prc_low, dolvol,
            vol AS tvol,
            ret,
            ret AS ret_local,
            ret_exc,
            1::BIGINT AS ret_lag_dif,
            div_tot,
            NULL::DOUBLE AS div_cash,
            NULL::DOUBLE AS div_spc,
            1 AS source_crsp
        FROM read_parquet('crsp_msf.parquet')
    """

    def test_crsp_msf_id_equals_permno(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        assert (df["id"] == df["permno"]).all()

    def test_crsp_msf_excntry_is_usa(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        assert (df["excntry"] == "USA").all()

    def test_crsp_msf_common_flag(self, toy_dir: Path) -> None:
        raw = pl.read_parquet(toy_dir / "crsp_msf.parquet")
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        expected = raw["shrcd"].is_in([10, 11, 12]).fill_null(False).cast(pl.Int32)
        assert (df["common"] == expected).all()

    def test_crsp_msf_eom_is_month_end(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        assert (df["eom"] == df["date"].dt.month_end()).all()

    def test_crsp_msf_constants(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        assert (df["curcd"] == "USD").all()
        assert (df["fx"] == 1.0).all()
        assert (df["ret_lag_dif"] == 1).all()
        assert (df["source_crsp"] == 1).all()
        assert (df["primary_sec"] == 1).all()

    def test_crsp_msf_prc_local_equals_prc(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        mask = df["prc"].is_not_null()
        assert (df.filter(mask)["prc_local"] == df.filter(mask)["prc"]).all()

    def test_crsp_msf_ret_local_equals_ret(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        mask = df["ret"].is_not_null()
        assert (df.filter(mask)["ret_local"] == df.filter(mask)["ret"]).all()

    def test_crsp_msf_null_columns(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        assert df["div_cash"].is_null().all()
        assert df["div_spc"].is_null().all()
        assert df["comp_tpci"].is_null().all()
        assert df["comp_exchg"].is_null().all()

    def test_crsp_msf_column_renames(self, toy_dir: Path) -> None:
        raw = pl.read_parquet(toy_dir / "crsp_msf.parquet")
        df = _run_cte_on_parquet(toy_dir, self._CRSP_MSF_CTE)
        np.testing.assert_allclose(
            df["crsp_shrcd"].to_numpy(),
            raw["shrcd"].cast(pl.Float64).to_numpy(),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            df["adjfct"].to_numpy(),
            raw["cfacshr"].cast(pl.Float64).to_numpy(),
            equal_nan=True,
        )

    def test_crsp_dsf_normalization(self, toy_dir: Path) -> None:
        sql = """
            SELECT
                permno AS id, 'USA' AS excntry,
                exch_main::INT AS exch_main,
                CASE WHEN shrcd IN (10, 11, 12) THEN 1 ELSE 0 END AS common,
                1 AS primary_sec, bidask::INT AS bidask,
                'USD' AS curcd, 1.0 AS fx,
                date, last_day(date) AS eom,
                cfacshr AS adjfct, shrout AS shares,
                me, dolvol, vol AS tvol,
                prc, prc_high, prc_low,
                ret AS ret_local, ret, ret_exc,
                1::BIGINT AS ret_lag_dif,
                1 AS source_crsp
            FROM read_parquet('crsp_dsf.parquet')
        """
        df = _run_cte_on_parquet(toy_dir, sql)
        assert (df["id"] == df.select(pl.col("id")).to_series()).all()
        assert (df["excntry"] == "USA").all()
        assert (df["source_crsp"] == 1).all()
        assert df.shape[0] > 0
        assert len(df.columns) == 23


# =========================================================================
# Test Class 2: Compustat Normalization
# =========================================================================


class TestCompNormalization:
    """Validate the Compustat normalization CTEs."""

    _COMP_MSF_CTE = """
        SELECT
            CAST(
                CASE
                    WHEN iid LIKE '%W%' THEN '3' || gvkey || SUBSTRING(iid, 1, 2)
                    WHEN iid LIKE '%C%' THEN '2' || gvkey || SUBSTRING(iid, 1, 2)
                    ELSE '1' || gvkey || SUBSTRING(iid, 1, 2)
                END AS BIGINT
            ) AS id,
            gvkey, iid,
            NULL::BIGINT AS permno,
            NULL::BIGINT AS permco,
            excntry,
            exch_main::INT AS exch_main,
            CASE WHEN tpci = '0' THEN 1 ELSE 0 END AS common,
            primary_sec::INT AS primary_sec,
            CASE WHEN prcstd = 4 THEN 1 ELSE 0 END AS bidask,
            tpci AS comp_tpci,
            exchg::BIGINT AS comp_exchg,
            curcdd AS curcd,
            0 AS source_crsp
        FROM read_parquet('comp_msf.parquet')
    """

    def test_comp_msf_id_prefix_common(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        common = df.filter(~pl.col("iid").str.contains("W") & ~pl.col("iid").str.contains("C"))
        if common.height > 0:
            id_strs = common["id"].cast(pl.Utf8)
            assert (id_strs.str.starts_with("1")).all()

    def test_comp_msf_id_prefix_adr(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        adr = df.filter(pl.col("iid").str.contains("C") & ~pl.col("iid").str.contains("W"))
        if adr.height > 0:
            id_strs = adr["id"].cast(pl.Utf8)
            assert (id_strs.str.starts_with("2")).all()

    def test_comp_msf_id_prefix_when_issued(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        wi = df.filter(pl.col("iid").str.contains("W"))
        if wi.height > 0:
            id_strs = wi["id"].cast(pl.Utf8)
            assert (id_strs.str.starts_with("3")).all()

    def test_comp_msf_id_construction(self, toy_dir: Path) -> None:
        """Verify id = int(prefix + gvkey + iid[0:2])."""
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        sample = df.head(100)
        for row in sample.iter_rows(named=True):
            iid = row["iid"]
            gvkey = row["gvkey"]
            prefix = "3" if "W" in iid else ("2" if "C" in iid else "1")
            expected = int(prefix + gvkey + iid[:2])
            assert row["id"] == expected, f"id mismatch: {row['id']} != {expected}"

    def test_comp_msf_common_flag(self, toy_dir: Path) -> None:
        raw = pl.read_parquet(toy_dir / "comp_msf.parquet")
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        expected = (raw["tpci"] == "0").cast(pl.Int32)
        assert (df["common"] == expected).all()

    def test_comp_msf_bidask_flag(self, toy_dir: Path) -> None:
        raw = pl.read_parquet(toy_dir / "comp_msf.parquet")
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        expected = (raw["prcstd"] == 4).cast(pl.Int32)
        assert (df["bidask"] == expected).all()

    def test_comp_msf_null_columns(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        assert df["permno"].is_null().all()
        assert df["permco"].is_null().all()

    def test_comp_msf_source_crsp_is_zero(self, toy_dir: Path) -> None:
        df = _run_cte_on_parquet(toy_dir, self._COMP_MSF_CTE)
        assert (df["source_crsp"] == 0).all()

    def test_comp_dsf_eom_computed(self, toy_dir: Path) -> None:
        sql = """
            SELECT datadate, last_day(datadate) AS eom
            FROM read_parquet('comp_dsf.parquet')
        """
        df = _run_cte_on_parquet(toy_dir, sql)
        expected = df["datadate"].dt.month_end()
        assert (df["eom"] == expected).all()

    def test_comp_dsf_tvol_from_cshtrd(self, toy_dir: Path) -> None:
        sql = """
            SELECT CAST(cshtrd AS DOUBLE) AS tvol
            FROM read_parquet('comp_dsf.parquet')
        """
        raw = pl.read_parquet(toy_dir / "comp_dsf.parquet")
        df = _run_cte_on_parquet(toy_dir, sql)
        np.testing.assert_allclose(
            df["tvol"].to_numpy(),
            raw["cshtrd"].cast(pl.Float64).to_numpy(),
            equal_nan=True,
        )


# =========================================================================
# Test Class 3: UNION ALL + Lead
# =========================================================================


class TestUnionAndLead:
    """Validate UNION ALL and ret_exc_lead1m computation."""

    def test_union_row_count(
        self, toy_dir: Path, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        msf, _ = duckdb_output
        crsp_rows = pl.read_parquet(toy_dir / "crsp_msf.parquet").height
        comp_rows = pl.read_parquet(toy_dir / "comp_msf.parquet").height
        # After dedup, row count <= sum; but unique (id,eom) keys should cover both
        assert msf.height <= crsp_rows + comp_rows
        assert msf.height > 0

    def test_union_preserves_source_crsp(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        msf, _ = duckdb_output
        assert set(msf["source_crsp"].unique().to_list()).issubset({0, 1})
        assert (msf.filter(pl.col("source_crsp") == 1)["permno"].is_not_null()).all()

    def test_union_column_set_monthly(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        msf, _ = duckdb_output
        expected = {
            "id",
            "permno",
            "permco",
            "gvkey",
            "iid",
            "excntry",
            "exch_main",
            "common",
            "primary_sec",
            "bidask",
            "crsp_shrcd",
            "crsp_exchcd",
            "comp_tpci",
            "comp_exchg",
            "curcd",
            "fx",
            "date",
            "eom",
            "adjfct",
            "shares",
            "me",
            "me_company",
            "prc",
            "prc_local",
            "prc_high",
            "prc_low",
            "dolvol",
            "tvol",
            "ret",
            "ret_local",
            "ret_exc",
            "ret_lag_dif",
            "div_tot",
            "div_cash",
            "div_spc",
            "source_crsp",
            "ret_exc_lead1m",
            "obs_main",
        }
        assert set(msf.columns) == expected

    def test_union_column_set_daily(self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        _, dsf = duckdb_output
        expected = {
            "id",
            "excntry",
            "exch_main",
            "common",
            "primary_sec",
            "bidask",
            "curcd",
            "fx",
            "date",
            "eom",
            "adjfct",
            "shares",
            "me",
            "dolvol",
            "tvol",
            "prc",
            "prc_high",
            "prc_low",
            "ret_local",
            "ret",
            "ret_exc",
            "ret_lag_dif",
            "source_crsp",
            "obs_main",
        }
        assert set(dsf.columns) == expected

    def test_ret_exc_lead1m_contiguous(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """When next row is contiguous (ret_lag_dif=1), lead should equal next ret_exc."""
        msf, _ = duckdb_output
        msf_sorted = msf.sort(["id", "eom"])
        for _id in msf_sorted["id"].unique().head(50).to_list():
            sub = msf_sorted.filter(pl.col("id") == _id)
            if sub.height < 2:
                continue
            for i in range(sub.height - 1):
                next_rld = sub["ret_lag_dif"][i + 1]
                lead_val = sub["ret_exc_lead1m"][i]
                next_ret_exc = sub["ret_exc"][i + 1]
                if next_rld == 1:
                    if next_ret_exc is None:
                        assert lead_val is None
                    else:
                        assert lead_val is not None
                        np.testing.assert_allclose(lead_val, next_ret_exc, rtol=1e-10)

    def test_ret_exc_lead1m_gap(self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """When next row has ret_lag_dif != 1, lead should be null."""
        msf, _ = duckdb_output
        msf_sorted = msf.sort(["id", "eom"])
        # Sample from Compustat IDs (source_crsp=0) which have ret_lag_dif gaps
        found_gap = False
        comp_ids = msf_sorted.filter(pl.col("source_crsp") == 0)["id"].unique().head(50).to_list()
        for _id in comp_ids:
            sub = msf_sorted.filter(pl.col("id") == _id)
            for i in range(sub.height - 1):
                next_rld = sub["ret_lag_dif"][i + 1]
                if next_rld is not None and next_rld != 1:
                    assert sub["ret_exc_lead1m"][i] is None, (
                        f"id={_id}, row={i}: expected null lead when gap"
                    )
                    found_gap = True
        assert found_gap, "No gaps found in test data — test is vacuous"

    def test_ret_exc_lead1m_last_row_per_id(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        msf, _ = duckdb_output
        msf_sorted = msf.sort(["id", "eom"])
        for _id in msf_sorted["id"].unique().head(50).to_list():
            sub = msf_sorted.filter(pl.col("id") == _id)
            assert sub["ret_exc_lead1m"][-1] is None

    def test_ret_exc_lead1m_not_in_daily(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        _, dsf = duckdb_output
        assert "ret_exc_lead1m" not in dsf.columns

    def test_ret_exc_lead1m_cross_id_boundary(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Lead must not bleed across different ids."""
        msf, _ = duckdb_output
        msf_sorted = msf.sort(["id", "eom"])
        ids = msf_sorted["id"].unique().sort().to_list()
        if len(ids) < 2:
            return
        # The last row of any id must have null ret_exc_lead1m
        last_rows = msf_sorted.group_by("id").last()
        assert last_rows["ret_exc_lead1m"].is_null().all()


# =========================================================================
# Test Class 4: obs_main
# =========================================================================


class TestObsMain:
    """Validate the obs_main derivation logic."""

    def test_obs_main_values_are_binary(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        msf, dsf = duckdb_output
        assert set(msf["obs_main"].drop_nulls().unique().to_list()).issubset({0, 1})
        assert set(dsf["obs_main"].drop_nulls().unique().to_list()).issubset({0, 1})

    def test_obs_main_null_gvkey_gets_one(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """CRSP rows with null gvkey → count(gvkey)=0 → obs_main=1."""
        msf, _ = duckdb_output
        crsp_null_gvkey = msf.filter((pl.col("source_crsp") == 1) & pl.col("gvkey").is_null())
        if crsp_null_gvkey.height > 0:
            assert (crsp_null_gvkey["obs_main"] == 1).all()

    def test_obs_main_single_observation_gets_one(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """When only one obs per (gvkey, iid, eom), obs_main should be 1."""
        msf, _ = duckdb_output
        # For source_crsp=0 rows with unique (gvkey, iid, eom) across whole dataset
        comp_rows = msf.filter(pl.col("source_crsp") == 0)
        if comp_rows.height == 0:
            return
        # Count how many rows share each (gvkey, iid, eom)
        counts = (
            msf.filter(pl.col("gvkey").is_not_null())
            .group_by(["gvkey", "iid", "eom"])
            .agg(pl.len().alias("cnt"))
        )
        singletons = counts.filter(pl.col("cnt") == 1)
        if singletons.height == 0:
            return
        # Join back to get the obs_main for singletons
        singleton_rows = msf.join(
            singletons.select(["gvkey", "iid", "eom"]),
            on=["gvkey", "iid", "eom"],
            how="inner",
        )
        assert (singleton_rows["obs_main"] == 1).all()

    def test_obs_main_applied_to_daily(
        self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Daily obs_main comes from monthly derivation via (id, eom) join."""
        msf, dsf = duckdb_output
        # For a sample of daily rows, verify obs_main matches monthly
        msf_obs = msf.select(["id", "eom", "obs_main"]).unique(["id", "eom"])
        sample_dsf = dsf.head(1000)
        joined = sample_dsf.join(msf_obs, on=["id", "eom"], how="left", suffix="_msf")
        matched = joined.filter(pl.col("obs_main_msf").is_not_null())
        if matched.height > 0:
            assert (matched["obs_main"] == matched["obs_main_msf"]).all()


# =========================================================================
# Test Class 5: Dedup Determinism
# =========================================================================


class TestDedupDeterminism:
    """Validate ROW_NUMBER()-based deterministic dedup."""

    def test_no_duplicates_monthly(self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        msf, _ = duckdb_output
        assert msf.select(["id", "eom"]).is_duplicated().sum() == 0

    def test_no_duplicates_daily(self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        _, dsf = duckdb_output
        assert dsf.select(["id", "date"]).is_duplicated().sum() == 0

    def test_dedup_monthly_sorted(self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        msf, _ = duckdb_output
        assert msf.sort(["id", "eom"]).equals(msf)

    def test_dedup_daily_sorted(self, duckdb_output: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        _, dsf = duckdb_output
        assert dsf.sort(["id", "date"]).equals(dsf)


# =========================================================================
# Test Class 6: End-to-End Comparison
# =========================================================================


@pytest.mark.expensive
class TestEndToEnd:
    """Compare DuckDB output with Polars reference on HUGE toy data."""

    def test_monthly_schema_matches(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        duck_msf, _ = duckdb_output
        pol_msf, _ = polars_output
        assert set(duck_msf.columns) == set(pol_msf.columns)

    def test_daily_schema_matches(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        _, duck_dsf = duckdb_output
        _, pol_dsf = polars_output
        assert set(duck_dsf.columns) == set(pol_dsf.columns)

    def test_monthly_row_count_matches(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        duck_msf, _ = duckdb_output
        pol_msf, _ = polars_output
        assert duck_msf.height == pol_msf.height

    def test_daily_row_count_matches(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        _, duck_dsf = duckdb_output
        _, pol_dsf = polars_output
        assert duck_dsf.height == pol_dsf.height

    def test_monthly_key_set_matches(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        duck_msf, _ = duckdb_output
        pol_msf, _ = polars_output
        duck_keys = set(duck_msf.select(["id", "eom"]).iter_rows())
        pol_keys = set(pol_msf.select(["id", "eom"]).iter_rows())
        assert duck_keys == pol_keys

    def test_daily_key_set_matches(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        _, duck_dsf = duckdb_output
        _, pol_dsf = polars_output
        duck_keys = set(duck_dsf.select(["id", "date"]).iter_rows())
        pol_keys = set(pol_dsf.select(["id", "date"]).iter_rows())
        assert duck_keys == pol_keys

    def test_monthly_numerical_values_match(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        duck_msf, _ = duckdb_output
        pol_msf, _ = polars_output
        # Sort both identically and compare float columns
        duck = duck_msf.sort(["id", "eom"])
        pol = pol_msf.sort(["id", "eom"]).select(duck.columns)
        float_cols = [c for c in duck.columns if duck[c].dtype == pl.Float64]
        for c in float_cols:
            d = duck[c].to_numpy()
            p = pol[c].to_numpy()
            both_nan = np.isnan(np.where(d is None, np.nan, d)) & np.isnan(
                np.where(p is None, np.nan, p)
            )
            d_filled = np.where(np.isnan(d), 0, d)
            p_filled = np.where(np.isnan(p), 0, p)
            mask = ~both_nan & np.isfinite(d_filled) & np.isfinite(p_filled)
            if np.any(mask):
                np.testing.assert_allclose(
                    d_filled[mask],
                    p_filled[mask],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"Monthly column {c} mismatch",
                )

    def test_monthly_categorical_values_match(
        self,
        duckdb_output: tuple[pl.DataFrame, pl.DataFrame],
        polars_output: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        duck_msf, _ = duckdb_output
        pol_msf, _ = polars_output
        duck = duck_msf.sort(["id", "eom"])
        pol = pol_msf.sort(["id", "eom"]).select(duck.columns)
        cat_cols = [
            c
            for c in duck.columns
            if duck[c].dtype in (pl.Utf8, pl.String)
            or c
            in (
                "id",
                "obs_main",
                "source_crsp",
                "common",
                "primary_sec",
                "bidask",
                "ret_lag_dif",
            )
        ]
        for c in cat_cols:
            if duck[c].dtype != pol[c].dtype:
                # Cast to common type for comparison
                d = duck[c].cast(pl.Utf8)
                p = pol[c].cast(pl.Utf8)
            else:
                d = duck[c]
                p = pol[c]
            assert d.equals(p, null_equal=True), f"Monthly column {c} mismatch"


# =========================================================================
# Test Class 7: Edge Cases
# =========================================================================


class TestEdgeCases:
    """Test boundary conditions."""

    def test_empty_compustat_files(self, tmp_path: Path) -> None:
        """Pipeline works when comp files have zero rows."""
        _make_crsp_msf(tmp_path, n_permnos=5)
        _make_crsp_dsf(tmp_path, n_permnos=5)
        # Create empty comp parquets with correct schema
        _make_comp_msf(tmp_path, n_gvkeys=0)
        _make_comp_dsf(tmp_path, n_gvkeys=0)
        msf, dsf = _duckdb_combine_crsp_comp_sf(tmp_path)
        assert msf.height > 0
        assert dsf.height > 0
        assert (msf["source_crsp"] == 1).all()
        assert (dsf["source_crsp"] == 1).all()

    def test_single_row_inputs(self, tmp_path: Path) -> None:
        """Pipeline works with 1-row input files."""
        _make_crsp_msf(tmp_path, n_permnos=1)
        _make_comp_msf(tmp_path, n_gvkeys=1)
        _make_crsp_dsf(tmp_path, n_permnos=1)
        _make_comp_dsf(tmp_path, n_gvkeys=1)
        msf, dsf = _duckdb_combine_crsp_comp_sf(tmp_path)
        assert msf.height >= 1
        assert dsf.height >= 1

    def test_leap_year_eom(self, tmp_path: Path) -> None:
        """Feb 29 dates handled correctly by DuckDB last_day()."""
        df = pl.DataFrame(
            {
                "permno": [99999, 99999],
                "permco": [99999, 99999],
                "gvkey": [None, None],
                "iid": [None, None],
                "exch_main": [1, 1],
                "bidask": [0, 0],
                "shrcd": [10, 10],
                "exchcd": [1, 1],
                "date": [date(2020, 2, 15), date(2019, 2, 15)],
                "cfacshr": [1.0, 1.0],
                "shrout": [100.0, 100.0],
                "me": [1000.0, 1000.0],
                "me_company": [1000.0, 1000.0],
                "prc": [50.0, 50.0],
                "prc_high": [51.0, 51.0],
                "prc_low": [49.0, 49.0],
                "dolvol": [5000.0, 5000.0],
                "vol": [100.0, 100.0],
                "ret": [0.01, 0.01],
                "ret_exc": [0.005, 0.005],
                "div_tot": [None, None],
            }
        ).cast(
            {
                "permno": pl.Int64,
                "permco": pl.Int64,
                "exch_main": pl.Int64,
                "bidask": pl.Int64,
                "shrcd": pl.Int64,
                "exchcd": pl.Int64,
                "date": pl.Date,
            }
        )
        df.write_parquet(tmp_path / "crsp_msf.parquet")
        _make_comp_msf(tmp_path, n_gvkeys=0)
        _make_crsp_dsf(tmp_path, n_permnos=1)
        _make_comp_dsf(tmp_path, n_gvkeys=0)

        msf, _ = _duckdb_combine_crsp_comp_sf(tmp_path)
        eoms = msf.sort("date")["eom"].to_list()
        assert eoms[0] == date(2019, 2, 28)  # Non-leap year
        assert eoms[1] == date(2020, 2, 29)  # Leap year

    def test_all_ids_have_gaps(self, tmp_path: Path) -> None:
        """When all ret_lag_dif != 1, all ret_exc_lead1m should be null."""
        # Create comp_msf with all gaps
        n_dates = 12
        dates = pl.date_range(date(2020, 1, 31), date(2020, 12, 31), "1mo", eager=True).to_list()
        df = pl.DataFrame(
            {
                "gvkey": ["000001"] * n_dates,
                "iid": ["01"] * n_dates,
                "excntry": ["USA"] * n_dates,
                "exch_main": [1] * n_dates,
                "primary_sec": [1] * n_dates,
                "tpci": ["0"] * n_dates,
                "prcstd": [1] * n_dates,
                "exchg": [11] * n_dates,
                "curcdd": ["USD"] * n_dates,
                "fx": [1.0] * n_dates,
                "datadate": dates,
                "eom": dates,
                "ajexdi": [1.0] * n_dates,
                "cshoc": [100.0] * n_dates,
                "me": [1000.0] * n_dates,
                "prc": [50.0] * n_dates,
                "prc_local": [50.0] * n_dates,
                "prc_high": [51.0] * n_dates,
                "prc_low": [49.0] * n_dates,
                "dolvol": [5000.0] * n_dates,
                "cshtrm": [100.0] * n_dates,
                "ret": [0.01] * n_dates,
                "ret_local": [0.01] * n_dates,
                "ret_exc": [0.005] * n_dates,
                "ret_lag_dif": [2] * n_dates,  # All gaps!
                "div_tot": [None] * n_dates,
                "div_cash": [None] * n_dates,
                "div_spc": [None] * n_dates,
            }
        ).cast(
            {
                "gvkey": pl.Utf8,
                "iid": pl.Utf8,
                "exch_main": pl.Int64,
                "primary_sec": pl.Int64,
                "prcstd": pl.Int64,
                "exchg": pl.Int64,
                "datadate": pl.Date,
                "eom": pl.Date,
                "ret_lag_dif": pl.Int64,
            }
        )
        df.write_parquet(tmp_path / "comp_msf.parquet")
        _make_crsp_msf(tmp_path, n_permnos=0)
        _make_crsp_dsf(tmp_path, n_permnos=1)
        _make_comp_dsf(tmp_path, n_gvkeys=0)

        msf, _ = _duckdb_combine_crsp_comp_sf(tmp_path)
        assert msf["ret_exc_lead1m"].is_null().all()
