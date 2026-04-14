"""
Tests for targeted Ibis table builders in aux_functions.py.

This module focuses on schema-level output guarantees for functions that read
parquet inputs from the expected project layout.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from aux_functions import gen_crsp_sf


def _write_lookup_tables(raw_tables: Path) -> None:
    """Write the minimal lookup parquet files required by gen_crsp_sf()."""
    pl.DataFrame(
        {
            "permno": [10001],
            "secinfostartdt": [date(2020, 1, 1)],
            "secinfoenddt": [date(2020, 1, 31)],
            "ticker": ["TEST"],
        }
    ).write_parquet(raw_tables / "crsp_stksecurityinfohist.parquet")

    pl.DataFrame(
        {
            "lpermno": [10001],
            "linkdt": [date(2019, 1, 1)],
            "linkenddt": [date(2021, 12, 31)],
            "linktype": ["LC"],
            "liid": ["01"],
            "gvkey": ["001234"],
        }
    ).write_parquet(raw_tables / "crsp_ccmxpf_lnkhist.parquet")


def _write_sf_fixture(raw_tables: Path, freq: str) -> tuple[date, date]:
    """Write a tiny monthly or daily CRSP SF fixture and return matched/null dates."""
    common_columns = {
        "permno": [10001, 10001],
        "permco": [20001, 20001],
        "shrout": [1000.0, 1000.0],
        "securitytype": ["EQTY", "EQTY"],
        "securitysubtype": ["COM", "COM"],
        "sharetype": ["NS", "NS"],
        "issuertype": ["CORP", "CORP"],
        "primaryexch": ["N", "N"],
        "conditionaltype": ["RW", "RW"],
    }

    if freq == "m":
        matched_date = date(2020, 1, 31)
        unmatched_date = date(2020, 2, 29)
        msf_df = pl.DataFrame(
            {
                **common_columns,
                "mthcaldt": [matched_date, unmatched_date],
                "mthprc": [10.0, 11.0],
                "mthprcflg": ["TR", "TR"],
                "mthret": [0.10, 0.02],
                "mthretx": [0.09, 0.01],
                "mthvol": [1000, 1100],
                "mthcumfacshr": [1.0, 1.0],
                "mthaskhi": [10.5, 11.5],
                "mthbidlo": [9.5, 10.5],
            }
        )
        raw_data_dfs = raw_tables.parent.parent / "code" / "raw_data_dfs"
        raw_data_dfs.mkdir(parents=True, exist_ok=True)
        msf_df.write_parquet(raw_data_dfs / "crsp_msf_v2_aug.parquet")
        return matched_date, unmatched_date

    matched_date = date(2020, 1, 2)
    unmatched_date = date(2020, 2, 3)
    pl.DataFrame(
        {
            **common_columns,
            "dlycaldt": [matched_date, unmatched_date],
            "dlyprc": [20.0, 21.0],
            "dlyprcflg": ["TR", "TR"],
            "dlyret": [0.01, 0.02],
            "dlyretx": [0.009, 0.018],
            "dlyvol": [200, 300],
            "dlycumfacshr": [1.0, 1.0],
            "dlyhigh": [20.5, 21.5],
            "dlylow": [19.5, 20.5],
        }
    ).write_parquet(raw_tables / "crsp_dsf_v2.parquet")
    return matched_date, unmatched_date


@pytest.mark.parametrize("freq", ["m", "d"])
def test_gen_crsp_sf_exposes_ticker_after_senames_join(
    freq: str,
    temp_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gen_crsp_sf() should keep ticker in the final output for monthly and daily data."""
    raw_tables = temp_data_dir / "raw" / "raw_tables"
    code_dir = temp_data_dir / "code"
    code_dir.mkdir()

    _write_lookup_tables(raw_tables)
    matched_date, unmatched_date = _write_sf_fixture(raw_tables, freq)

    monkeypatch.chdir(code_dir)

    result = gen_crsp_sf(freq)
    assert "ticker" in result.columns, f"Expected ticker in schema, got {result.columns}"

    df = result.to_polars().sort("date")

    assert {"permno", "permco", "date", "me", "ticker"}.issubset(df.columns), (
        f"Missing expected columns from output: {df.columns}"
    )

    ticker_by_date = {
        row["date"]: row["ticker"] for row in df.select(["date", "ticker"]).to_dicts()
    }
    assert ticker_by_date[matched_date] == "TEST", (
        f"Expected ticker TEST on {matched_date}, got {ticker_by_date[matched_date]!r}"
    )
    assert ticker_by_date[unmatched_date] is None, (
        f"Expected null ticker on {unmatched_date}, got {ticker_by_date[unmatched_date]!r}"
    )
