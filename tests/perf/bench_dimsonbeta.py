"""Benchmark for Dimson beta closed-form lazy implementation.

Run:
    uv run python tests/perf/bench_dimsonbeta.py [--preset {small,medium,large,huge}]
                                                 [--n-stocks N] [--n-days D] [--reps R]

Generates synthetic daily data for N stocks over D trading days, splits into
21-day groups (one window per calendar month), runs ``dimsonbeta`` and reports
wall-clock plus peak RSS.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np
import polars as pl

from jkp.data.aux_functions import dimsonbeta

PRESETS = {
    "small": (1000, 252 * 5),
    "medium": (5000, 252 * 15),
    "large": (15000, 252 * 30),
    "huge": (25000, 252 * 40),
}


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss units: bytes on macOS, kibibytes on Linux/BSD.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def build_panel(n_stocks: int, n_days: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_stocks * n_days
    mkt = np.tile(rng.standard_normal(n_days), n_stocks)
    mkt_ld = np.tile(np.roll(mkt[:n_days], -1), n_stocks)
    mkt_lg = np.tile(np.roll(mkt[:n_days], 1), n_stocks)
    mkt_ld[n_days - 1 :: n_days] = 0.0
    mkt_lg[0::n_days] = 0.0
    betas = rng.uniform(0.5, 1.5, n_stocks)
    ret_exc = (
        np.repeat(betas, n_days) * mkt + 0.2 * mkt_ld + 0.3 * mkt_lg + 0.5 * rng.standard_normal(n)
    )
    ids = np.repeat(np.arange(n_stocks, dtype=np.int64), n_days)
    group = np.tile(np.arange(n_days, dtype=np.int64) // 21, n_stocks)
    return pl.DataFrame(
        {
            "id_int": ids,
            "group_number": group,
            "mktrf": mkt,
            "mktrf_ld1": mkt_ld,
            "mktrf_lg1": mkt_lg,
            "ret_exc": ret_exc,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None)
    p.add_argument("--n-stocks", type=int, default=None)
    p.add_argument("--n-days", type=int, default=None)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.preset is not None:
        n_stocks, n_days = PRESETS[args.preset]
        if args.n_stocks is not None:
            n_stocks = args.n_stocks
        if args.n_days is not None:
            n_days = args.n_days
    else:
        n_stocks = args.n_stocks if args.n_stocks is not None else 1000
        n_days = args.n_days if args.n_days is not None else 252 * 5

    df = build_panel(n_stocks, n_days, seed=args.seed)
    n_groups = df.select("group_number").n_unique() * n_stocks
    print(
        f"panel: {n_stocks:,} stocks × {n_days:,} days = {df.height:,} rows, "
        f"{n_groups:,} groups (preset={args.preset})"
    )

    times = []
    out = None
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = dimsonbeta(df.lazy(), "_21d", 15).collect()
        times.append(time.perf_counter() - t0)
    print(f"dimsonbeta: best={min(times):.3f}s mean={np.mean(times):.3f}s reps={args.reps}")
    print(f"output rows: {out.height:,}")
    print(f"peak RSS: {_peak_rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
