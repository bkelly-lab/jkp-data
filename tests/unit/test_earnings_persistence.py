"""Tests for ``compute_earnings_persistence`` (ni_ar1, ni_ivol)."""

import os
from datetime import date

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import compute_earnings_persistence
from jkp.data.paths import DataPaths


@pytest.mark.unit
def test_ni_ar1_is_null_when_lagged_ni_at_does_not_vary(test_paths: DataPaths, tmp_path) -> None:
    """A firm with constant NI/AT gets a null ni_ar1 (not NaN); a varying firm gets the OLS slope."""
    eoms = pl.date_range(date(2000, 1, 1), date(2007, 12, 1), "1mo", eager=True).dt.month_end()
    n = len(eoms)
    ni_varying = np.random.default_rng(0).normal(5.0, 2.0, n)
    raw = pl.DataFrame(
        {
            "gvkey": ["A"] * n + ["B"] * n,
            "curcd": "USD",
            "datadate": pl.concat([eoms, eoms]),
            "ni_x": np.concatenate([ni_varying, np.full(n, 5.0)]),
            "at_x": 100.0,
        }
    )
    data_path = tmp_path / "acc.parquet"
    raw.write_parquet(data_path)

    out = compute_earnings_persistence(test_paths, data_path, 5, 5)

    constant = out.filter(pl.col("gvkey") == "B")
    assert constant.height > 0
    assert constant["ni_ar1"].null_count() == constant.height
    assert out["ni_ar1"].is_nan().sum() == 0

    # Last date: December NI/AT for 2003-2007 regressed on its 12-month lag.
    december = ni_varying[11::12] / 100.0
    expected = np.polyfit(december[2:7], december[3:8], 1)[0]
    last = out.filter((pl.col("gvkey") == "A") & (pl.col("datadate") == date(2007, 12, 31)))
    assert last["ni_ar1"].item() == pytest.approx(expected)


@pytest.mark.unit
def test_output_does_not_depend_on_thread_count(
    test_paths: DataPaths, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ni_ar1 and ni_ivol are bit-identical whether DuckDB runs on one thread or eight."""
    eoms = pl.date_range(date(2000, 1, 1), date(2007, 12, 1), "1mo", eager=True).dt.month_end()
    n_firms = 2000
    rng = np.random.default_rng(1)
    raw = pl.DataFrame(
        {
            "gvkey": np.repeat([f"{i:06d}" for i in range(n_firms)], len(eoms)),
            "curcd": "USD",
            "datadate": pl.concat([eoms] * n_firms),
            "ni_x": rng.normal(5.0, 2.0, n_firms * len(eoms)),
            "at_x": rng.uniform(1.0, 1000.0, n_firms * len(eoms)),
        }
    )
    data_path = tmp_path / "acc.parquet"
    raw.write_parquet(data_path)

    outputs = []
    for threads in (1, 8):
        # _duckdb_connect counts the CPUs in the affinity mask (Linux), else os.cpu_count()
        monkeypatch.setattr(os, "cpu_count", lambda threads=threads: threads)
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid, threads=threads: set(range(threads)), raising=False
        )
        outputs.append(compute_earnings_persistence(test_paths, data_path, 5, 5))
    assert outputs[0].height > 0
    assert_frame_equal(outputs[0], outputs[1], check_exact=True)
