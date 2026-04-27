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

from jkp.data.aux_functions import aug_msf_v2, gen_crsp_sf, merge_roll_apply_daily_results


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


def _write_aug_msf_v2_fixtures(raw_tables: Path) -> None:
    """Write minimal raw msf_v2 and dsf_v2 parquet fixtures for aug_msf_v2()."""
    pl.DataFrame(
        {
            "permno": [10001, 10001],
            "yyyymm": [202001, 202002],
            "mthcaldt": [date(2020, 1, 31), date(2020, 2, 29)],
            "mthprcflg": ["TR", "BA"],
        }
    ).write_parquet(raw_tables / "crsp_msf_v2.parquet")

    pl.DataFrame(
        {
            "permno": [10001, 10001, 10001, 10001],
            "dlycaldt": [
                date(2020, 1, 10),
                date(2020, 1, 20),
                date(2020, 2, 10),
                date(2020, 2, 20),
            ],
            "dlyprc": [9.5, 10.5, 11.0, 12.0],
            "dlyprcflg": ["TR", "TR", "TR", "TR"],
        }
    ).write_parquet(raw_tables / "crsp_dsf_v2.parquet")


def test_aug_msf_v2_writes_augmented_file_and_is_idempotent(
    temp_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aug_msf_v2() should produce the augmented parquet and be safe to re-run."""
    raw_tables = temp_data_dir / "raw" / "raw_tables"
    code_dir = temp_data_dir / "code"
    code_dir.mkdir()
    (code_dir / "raw_data_dfs").mkdir()

    _write_aug_msf_v2_fixtures(raw_tables)

    monkeypatch.chdir(code_dir)

    aug_msf_v2()

    output_path = code_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet"
    assert output_path.exists(), f"Expected augmented file at {output_path}"

    schema = pl.scan_parquet(output_path).collect_schema().names()
    assert "mthaskhi" in schema, f"Expected mthaskhi column in {schema}"
    assert "mthbidlo" in schema, f"Expected mthbidlo column in {schema}"

    # Idempotency: a second invocation must not raise.
    aug_msf_v2()


def test_merge_roll_apply_daily_results_writes_once_with_deterministic_order(
    temp_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """merge_roll_apply_daily_results() must produce a single output with deterministic
    column ordering (sorted by source __roll* filename) and be re-run safe."""
    code_dir = temp_data_dir / "code"
    code_dir.mkdir(exist_ok=True)

    pl.DataFrame({"id_int": [1, 2], "id": [10001, 10002]}).write_parquet(
        code_dir / "id_int_key.parquet"
    )

    # aux_date 24300 corresponds to 2025-01; in-range for any run after Jan 2025
    # (date_idx in the function is today.year * 12 + today.month).
    aux_date_val = 24300
    # Write fixtures in non-alphabetical order to exercise sorted() determinism:
    # filesystem-order would be insertion-order on most FSes, so writing __roll_b_*
    # first ensures the test fails without the sorted() fix.
    pl.DataFrame(
        {
            "id_int": [1, 2],
            "aux_date": [aux_date_val, aux_date_val],
            "rmax": [0.5, 0.6],
        }
    ).write_parquet(code_dir / "__roll_b_rmax.parquet")
    pl.DataFrame(
        {
            "id_int": [1, 2],
            "aux_date": [aux_date_val, aux_date_val],
            "rvol": [0.1, 0.2],
        }
    ).write_parquet(code_dir / "__roll_a_rvol.parquet")

    monkeypatch.chdir(code_dir)
    merge_roll_apply_daily_results()

    out = code_dir / "roll_apply_daily.parquet"
    assert out.exists(), f"Expected output at {out}"

    df = pl.read_parquet(out)
    assert {"id", "eom", "rvol", "rmax"}.issubset(df.columns), (
        f"Missing expected columns: {df.columns}"
    )
    # Deterministic order: sorted file_paths puts __roll_a_rvol before __roll_b_rmax,
    # so rvol must precede rmax in the merged schema.
    assert df.columns.index("rvol") < df.columns.index("rmax"), (
        f"Expected rvol before rmax (sorted file order), got {df.columns}"
    )
    # Outer join on shared (id_int, aux_date) keys = 2 rows.
    assert df.height == 2
    assert set(df["id"].to_list()) == {10001, 10002}

    # Re-run must produce identical content (idempotent + single-write safety).
    merge_roll_apply_daily_results()
    df2 = pl.read_parquet(out)
    assert df.equals(df2)
