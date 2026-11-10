"""Tests for ``header_aux``: which company header wins when a gvkey is in both."""

import polars as pl
import pytest

from jkp.data.aux_functions import header_aux
from jkp.data.paths import DataPaths


def _header(gvkeys: list[str], prirow: list[str | None]) -> pl.DataFrame:
    """Company header rows; priusa and prican are set to the prirow value."""
    return pl.DataFrame(
        {"gvkey": gvkeys, "prirow": prirow, "priusa": prirow, "prican": prirow},
        schema=dict.fromkeys(["gvkey", "prirow", "priusa", "prican"], pl.String),
    )


@pytest.mark.unit
def test_global_header_wins_for_shared_gvkey(test_paths: DataPaths) -> None:
    """A gvkey in both headers takes the Global row, nulls included; others keep their own row."""
    _header(["001", "002"], ["01", "02"]).write_parquet(
        test_paths.raw_tables_dir / "comp_company.parquet"
    )
    _header(["001", "003"], [None, "03W"]).write_parquet(
        test_paths.raw_tables_dir / "comp_g_company.parquet"
    )
    header_aux(test_paths)
    out = pl.read_parquet(test_paths.interim_dir / "raw_data_dfs" / "__header.parquet")
    assert out.sort("gvkey").rows() == [
        ("001", None, None, None),
        ("002", "02", "02", "02"),
        ("003", "03W", "03W", "03W"),
    ]
