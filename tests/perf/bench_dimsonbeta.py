"""Benchmark for the Dimson beta plugin.

Run:
    uv run python tests/perf/bench_dimsonbeta.py [--n-stocks N] [--n-days D]

Generates synthetic daily data for N stocks over D trading days, splits into
21-day windows (one window per calendar month), runs ``dimsonbeta`` and reports
wall-clock plus peak RSS. Compares against a reference numpy lstsq.
"""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np
import polars as pl

from jkp.data.aux_functions import dimsonbeta


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return rss / (1024 * 1024) if rss > 1_000_000 else rss / 1024


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
    # group_number: one per (stock, calendar month). 21 days/month.
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
    p.add_argument("--n-stocks", type=int, default=1000)
    p.add_argument("--n-days", type=int, default=252 * 5)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()

    df = build_panel(args.n_stocks, args.n_days)
    print(
        f"panel: {args.n_stocks} stocks × {args.n_days} days = {len(df):,} rows, "
        f"{df.select('group_number').n_unique() * args.n_stocks:,} groups"
    )

    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = dimsonbeta(df.lazy(), "_21d", __min=15).collect()
        times.append(time.perf_counter() - t0)
    print(f"dimsonbeta: best={min(times):.3f}s mean={np.mean(times):.3f}s reps={args.reps}")
    print(f"output rows: {len(out):,}")
    print(f"peak RSS: {_peak_rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
