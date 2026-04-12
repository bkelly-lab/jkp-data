#!/usr/bin/env python3
"""Benchmark: legacy vs current ``portfolios()`` on synthetic data.

Usage:
    uv run python scripts/bench_portfolio.py

Runs both implementations N times on a medium-sized synthetic dataset and
reports wall-clock times, relative speedup, and peak RSS.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

# Setup import paths.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

import tempfile

import polars as pl

from portfolio import portfolios as current_portfolios
from tests.fixtures.portfolio_legacy import portfolios as legacy_portfolios
from tests.unit.test_portfolio_parity import (
    _make_cutoffs,
    _write_synthetic_country,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_IDS = 500
N_MONTHS = 60
N_CHARS = 50
N_RUNS = 3
SEED = 42


def _build_chars(n: int) -> list[str]:
    return [f"char_{i:03d}" for i in range(n)]


def main() -> None:
    chars = _build_chars(N_CHARS)

    # Patch the module-level constants used by the synthetic helpers.
    import tests.unit.test_portfolio_parity as tm

    orig_ids, orig_months = tm.N_IDS, tm.N_MONTHS
    tm.N_IDS, tm.N_MONTHS = N_IDS, N_MONTHS

    tmp = Path(tempfile.mkdtemp())
    data_root = tmp / "processed"
    data_root.mkdir(parents=True)

    print(f"Generating synthetic data: {N_IDS} ids x {N_MONTHS} months x {N_CHARS} chars ...")
    char_df, _ = _write_synthetic_country(
        data_root=data_root, excntry="USA", chars=chars, seed=SEED
    )
    eoms = char_df["eom"].unique().sort().to_list()
    nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)

    kwargs = {
        "data_path": str(data_root),
        "excntry": "USA",
        "chars": chars,
        "pfs": 3,
        "bps": "non_mc",
        "bp_min_n": 10,
        "nyse_size_cutoffs": nyse_cut,
        "source": ["CRSP", "COMPUSTAT"],
        "wins_ret": True,
        "cmp_key": False,
        "signals": False,
        "signals_standardize": True,
        "signals_w": "vw_cap",
        "daily_pf": True,
        "ind_pf": True,
        "ret_cutoffs": ret_cut,
        "ret_cutoffs_daily": ret_cut_daily,
    }

    # Warm up both implementations.
    print("Warming up ...")
    _ = legacy_portfolios(**kwargs)
    _ = current_portfolios(**kwargs)

    # Timed runs.
    print(f"Running {N_RUNS} timed iterations each ...\n")
    leg_times: list[float] = []
    cur_times: list[float] = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _ = legacy_portfolios(**kwargs)
        leg_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = current_portfolios(**kwargs)
        cur_times.append(time.perf_counter() - t0)

    leg_min = min(leg_times)
    cur_min = min(cur_times)
    leg_avg = sum(leg_times) / len(leg_times)
    cur_avg = sum(cur_times) / len(cur_times)

    # Peak RSS (macOS: bytes, Linux: kilobytes).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_mb = rss / 1024 / 1024
    else:
        rss_mb = rss / 1024

    print("=" * 60)
    print(f"Dataset : {N_IDS} ids x {N_MONTHS} months x {N_CHARS} chars")
    print(f"Runs    : {N_RUNS}")
    print(f"Legacy  : {leg_min:.3f}s min / {leg_avg:.3f}s avg")
    print(f"Current : {cur_min:.3f}s min / {cur_avg:.3f}s avg")
    print(f"Speedup : {leg_min / cur_min:.2f}x (min) / {leg_avg / cur_avg:.2f}x (avg)")
    print(f"Peak RSS: {rss_mb:.0f} MB")
    print("=" * 60)

    # Restore.
    tm.N_IDS, tm.N_MONTHS = orig_ids, orig_months


if __name__ == "__main__":
    main()
