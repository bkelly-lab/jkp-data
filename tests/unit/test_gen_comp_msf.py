"""Tests for ``gen_comp_msf`` (which row survives a duplicate (gvkey, iid, eom)) and for
the monthly panel the Compustat correction screens (``_screen_panel``)."""

from datetime import date

import polars as pl
import pytest

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths

FLOAT_COLS = [
    "prc_local",
    "prc_high",
    "prc_low",
    "ajexdi",
    "cshoc",
    "ri_local",
    "fx",
    "prc",
    "me",
    "cshtrm",
    "dolvol",
    "ri",
    "div_tot",
    "div_cash",
    "div_spc",
]


def _monthly(prcstd: list[int], prc_local: list[float], source: int) -> pl.DataFrame:
    """Rows for gvkey 001234, iid 01, all on datadate 2000-01-31."""
    n = len(prcstd)
    return pl.DataFrame(
        {
            "gvkey": ["001234"] * n,
            "iid": ["01"] * n,
            "datadate": [date(2000, 1, 31)] * n,
            "eom": [date(2000, 1, 31)] * n,
            "tpci": ["0"] * n,
            "exchg": pl.Series([11] * n, dtype=pl.Int32),
            "curcdd": ["USD"] * n,
            **{c: pl.Series(prc_local if c == "prc_local" else [1.0] * n) for c in FLOAT_COLS},
            "prcstd": pl.Series(prcstd, dtype=pl.Int32),
            "source": pl.Series([source] * n, dtype=pl.Int32),
        }
    )


@pytest.mark.unit
def test_same_datadate_keeps_highest_prcstd(
    test_paths: DataPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two SECD rows on the same datadate: the higher prcstd wins, as in prepare_comp_sf."""
    monkeypatch.setattr(af, "gen_secd_data", lambda paths: None)
    monkeypatch.setattr(af, "gen_secm_data", lambda paths: None)
    _monthly([3, 10], [30.0, 100.0], source=1).write_parquet(
        test_paths.interim_dir / "secd_data.parquet"
    )
    _monthly([10], [1.0], source=2).clear().write_parquet(
        test_paths.interim_dir / "secm_data.parquet"
    )
    af.gen_comp_msf(test_paths)
    out = pl.read_parquet(test_paths.interim_dir / "__comp_msf.parquet")
    assert out.select("prcstd", "prc_local").rows() == [(10, 100.0)]


def _month_end(eom: date, prc_local: float, source: int) -> pl.DataFrame:
    """One month-end row for gvkey 001234, iid 01."""
    return _monthly([10], [prc_local], source).with_columns(datadate=pl.lit(eom), eom=pl.lit(eom))


@pytest.mark.unit
def test_screen_panel_prefers_secd_and_takes_qunit_from_secd(
    test_paths: DataPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SECD wins a month both sources hold; QUNIT comes from the SECD month-end row and is
    null in SECM-only months; excntry comes from the exchange."""
    jan, feb, mar = date(2000, 1, 31), date(2000, 2, 29), date(2000, 3, 31)
    secd = pl.concat([_month_end(jan, 1.0, 1), _month_end(feb, 2.0, 1)]).with_columns(
        qunit=pl.lit(100.0)
    )
    monkeypatch.setattr(
        af, "gen_secd_data", lambda paths, dsf_path, out_path: secd.write_parquet(out_path)
    )
    monkeypatch.setattr(
        af, "comp_exchanges", lambda paths: pl.DataFrame({"exchg": [11], "excntry": ["GBR"]})
    )
    pl.concat([_month_end(feb, 20.0, 2), _month_end(mar, 3.0, 2)]).write_parquet(
        test_paths.interim_dir / "secm_data.parquet"
    )
    out = af._screen_panel(test_paths, test_paths.interim_dir / "daily.parquet").collect()
    assert out.sort("eom").select("eom", "prc_local", "qunit", "excntry").rows() == [
        (jan, 1.0, 100.0, "GBR"),
        (feb, 2.0, 100.0, "GBR"),
        (mar, 3.0, None, "GBR"),
    ]
