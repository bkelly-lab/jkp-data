"""Tests for save_daily_ret() output schema.

``save_daily_ret`` reads ``interim/world_dsf_output.parquet``, selects a subset
of return columns, and writes one parquet per country under
``processed/return_data/daily_rets_by_country/{ISO3}.parquet``.

Coverage here:

- Output schema: the per-country files contain exactly the expected columns
  (including ``ret_local``) in the expected order and with correct dtypes.
- Country partitioning: each country's rows end up in the correct file and the
  ``excntry`` partition column is stripped from the row data.
- Null-country handling: rows with ``excntry IS NULL`` are written under
  ``null_country.parquet``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from jkp.data.aux_functions import save_daily_ret
from jkp.data.paths import DataPaths

# The exact output contract of save_daily_ret: column order and dtypes.
# excntry is used only for partitioning and does not appear in the output files.
EXPECTED_COLUMNS = [
    "id",
    "date",
    "me",
    "ret",
    "ret_local",
    "ret_exc",
    "ret_exc_wins",
    "ret_intraday",
    "ret_overnight",
    "ret_intraday_local",
    "ret_overnight_local",
]

EXPECTED_SCHEMA = pl.Schema(
    {
        "id": pl.Int64,
        "date": pl.Date,
        "me": pl.Float64,
        "ret": pl.Float64,
        "ret_local": pl.Float64,
        "ret_exc": pl.Float64,
        "ret_exc_wins": pl.Float64,
        "ret_intraday": pl.Float64,
        "ret_overnight": pl.Float64,
        "ret_intraday_local": pl.Float64,
        "ret_overnight_local": pl.Float64,
    }
)

# Distinct from ret (0.01) and other ~0.00x defaults so a wrong-column
# mapping would fail even when the row does not override ret_local.
DEFAULT_RET_LOCAL = 0.0123


def _make_world_dsf_output(
    rows: list[dict],
) -> pl.DataFrame:
    """Build a minimal ``world_dsf_output`` DataFrame.

    The caller supplies rows with at least ``excntry``, ``id``, and ``date``.
    Missing return columns default to fixture values (see DEFAULT_RET_LOCAL).
    """
    defaults: dict[str, float | None] = {
        "me": 100.0,
        "ret": 0.01,
        "ret_local": DEFAULT_RET_LOCAL,
        "ret_exc": 0.005,
        "ret_exc_wins": 0.005,
        "ret_intraday": 0.004,
        "ret_overnight": 0.006,
        "ret_intraday_local": 0.004,
        "ret_overnight_local": 0.005,
    }
    full_rows = [{**defaults, **r} for r in rows]
    schema = pl.Schema(
        {
            "excntry": pl.Utf8,
            "id": pl.Int64,
            "date": pl.Date,
            "me": pl.Float64,
            "ret": pl.Float64,
            "ret_local": pl.Float64,
            "ret_exc": pl.Float64,
            "ret_exc_wins": pl.Float64,
            "ret_intraday": pl.Float64,
            "ret_overnight": pl.Float64,
            "ret_intraday_local": pl.Float64,
            "ret_overnight_local": pl.Float64,
        }
    )
    return pl.DataFrame(full_rows, schema=schema)


def _write_fixture(paths: DataPaths, df: pl.DataFrame) -> None:
    """Write ``world_dsf_output.parquet`` into the interim directory."""
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(paths.interim_dir / "world_dsf_output.parquet")


def _run_save_daily_ret(paths: DataPaths, df: pl.DataFrame) -> Path:
    """Write fixture, ensure the output directory exists, and run save_daily_ret."""
    _write_fixture(paths, df)
    out_dir = paths.processed_dir / "return_data" / "daily_rets_by_country"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_daily_ret(paths)
    return out_dir


@pytest.mark.unit
def test_output_schema(test_paths: DataPaths) -> None:
    """Per-country files have exactly the expected columns and dtypes."""
    rows = [
        {"excntry": "USA", "id": 1, "date": date(2024, 1, 2)},
        {"excntry": "USA", "id": 2, "date": date(2024, 1, 2)},
        {"excntry": "CAN", "id": 3, "date": date(2024, 1, 2)},
    ]
    out_dir = _run_save_daily_ret(test_paths, _make_world_dsf_output(rows))

    usa = pl.read_parquet(out_dir / "USA.parquet")
    assert usa.columns == EXPECTED_COLUMNS
    for col_name, expected_dtype in EXPECTED_SCHEMA.items():
        assert usa.schema[col_name] == expected_dtype, (
            f"Column {col_name}: expected {expected_dtype}, got {usa.schema[col_name]}"
        )


@pytest.mark.unit
def test_country_partitioning(test_paths: DataPaths) -> None:
    """Each country's rows land in the correct file."""
    rows = [
        {"excntry": "USA", "id": 1, "date": date(2024, 1, 2)},
        {"excntry": "USA", "id": 2, "date": date(2024, 1, 3)},
        {"excntry": "CAN", "id": 3, "date": date(2024, 1, 2)},
    ]
    out_dir = _run_save_daily_ret(test_paths, _make_world_dsf_output(rows))

    usa = pl.read_parquet(out_dir / "USA.parquet")
    can = pl.read_parquet(out_dir / "CAN.parquet")
    assert usa.height == 2
    assert can.height == 1
    assert set(usa["id"].to_list()) == {1, 2}
    assert can["id"].to_list() == [3]


@pytest.mark.unit
def test_null_country(test_paths: DataPaths) -> None:
    """Rows with null excntry are written to null_country.parquet."""
    rows = [
        {"excntry": None, "id": 99, "date": date(2024, 6, 1)},
        {"excntry": "USA", "id": 1, "date": date(2024, 6, 1)},
    ]
    out_dir = _run_save_daily_ret(test_paths, _make_world_dsf_output(rows))

    nc = pl.read_parquet(out_dir / "null_country.parquet")
    assert nc.height == 1
    assert nc["id"].to_list() == [99]
    assert nc.columns == EXPECTED_COLUMNS


@pytest.mark.unit
def test_ret_local_values_preserved(test_paths: DataPaths) -> None:
    """ret_local values (including nulls) pass through correctly from the input."""
    rows = [
        {
            "excntry": "DEU",
            "id": 10,
            "date": date(2024, 3, 15),
            "ret_local": DEFAULT_RET_LOCAL,
        },
        {
            "excntry": "DEU",
            "id": 11,
            "date": date(2024, 3, 15),
            "ret_local": None,
        },
    ]
    out_dir = _run_save_daily_ret(test_paths, _make_world_dsf_output(rows))

    deu = pl.read_parquet(out_dir / "DEU.parquet").sort("id")
    assert deu["ret_local"][0] == pytest.approx(DEFAULT_RET_LOCAL)
    assert deu["ret_local"][1] is None
