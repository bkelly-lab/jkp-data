"""Tests for ``add_helper_vars``: which row survives a duplicate (gvkey, curcd, datadate)."""

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import add_helper_vars
from jkp.data.paths import DataPaths

VALUE_COLS = [
    "ppegt",
    "icapt",
    "txditc",
    "txp",
    "aco",
    "act",
    "ao",
    "ap",
    "at",
    "ceq",
    "che",
    "dlc",
    "dltt",
    "gdwl",
    "gpq",
    "intan",
    "invt",
    "ivao",
    "ivst",
    "lco",
    "lct",
    "lo",
    "lt",
    "mib",
    "ppent",
    "pstk",
    "rect",
    "re",
    "seq",
    "txdb",
    "dltisy",
    "gpy",
    "doq",
    "doy",
    "dltisq",
    "ni_qtr",
    "sale_qtr",
    "ocf_qtr",
    "cogs",
    "xsga",
    "xint",
    "dp",
    "txt",
    "xrd",
    "dv",
    "spi",
    "sale",
    "revt",
    "xopr",
    "oibdp",
    "oiadp",
    "ib",
    "ni",
    "xido",
    "nopi",
    "mii",
    "pi",
    "xi",
    "xidoc",
    "capx",
    "oancf",
    "ibc",
    "dpc",
    "wcapt",
    "prstkc",
    "sstk",
    "purtshr",
    "ds",
    "dltr",
    "ltdch",
    "dlcch",
    "fincf",
    "fiao",
    "txbcof",
    "dvt",
    "me_fiscal",
    "gp",
    "dltis",
    "do",
    "dvc",
    "ebit",
    "ebitda",
    "itcb",
    "pstkl",
    "pstkrv",
    "xad",
    "xlr",
    "emp",
]


def _frame(rows: list[tuple[str, str, int, float]]) -> pl.LazyFrame:
    """One row per (gvkey, source, n, at) on 2000-01-31; every other value column is null."""
    return pl.LazyFrame(
        {
            "gvkey": [r[0] for r in rows],
            "datadate": [date(2000, 1, 31)] * len(rows),
            "n": pl.Series([r[2] for r in rows], dtype=pl.UInt32),
            "curcd": ["USD"] * len(rows),
            "source": [r[1] for r in rows],
            **{
                c: pl.Series([r[3] if c == "at" else None for r in rows], dtype=pl.Float64)
                for c in VALUE_COLS
            },
        }
    )


@pytest.mark.unit
def test_duplicate_date_keeps_global_then_lowest_n(test_paths: DataPaths) -> None:
    """Global beats NA on the same (gvkey, curcd, datadate); within a source the lowest n wins."""
    data = _frame(
        [
            ("001", "NA", 1, 10.0),
            ("001", "GLOBAL", 2, 20.0),
            ("002", "NA", 5, 30.0),
            ("002", "NA", 4, 40.0),
        ]
    )
    out = add_helper_vars(test_paths, data).collect().sort("gvkey")
    assert out.select("gvkey", "source", "n", "at").rows() == [
        ("001", "GLOBAL", 2, 20.0),
        ("002", "NA", 4, 40.0),
    ]
