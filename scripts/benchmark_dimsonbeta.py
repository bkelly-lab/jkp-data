"""Benchmark Dimson β: polars-ds `pds.lin_reg` vs closed-form sufficient-stats OLS.

Generates synthetic data matching the `dimsonbeta` input schema (~25 obs per
(id_int, group_number) group), sweeps over group counts, and reports median
wall-clock per impl plus max abs diff on β.

Run: `uv run python scripts/benchmark_dimsonbeta.py`
"""

from __future__ import annotations

import functools
import statistics
import time

import numpy as np
import polars as pl
import polars_ds as pds


def make_data(n_groups: int, obs_per_group: int = 25, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_groups * obs_per_group
    mkt = rng.standard_normal(n)
    mkt_ld = rng.standard_normal(n)
    mkt_lg = rng.standard_normal(n)
    eps = 0.1 * rng.standard_normal(n)
    ret_exc = 0.3 * mkt + 0.2 * mkt_ld + 0.1 * mkt_lg + eps  # true β_sum = 0.6
    return pl.DataFrame(
        {
            "id_int": np.repeat(np.arange(n_groups, dtype=np.int64), obs_per_group),
            "group_number": np.zeros(n, dtype=np.int64),
            "mktrf": mkt,
            "mktrf_ld1": mkt_ld,
            "mktrf_lg1": mkt_lg,
            "ret_exc": ret_exc,
        }
    )


# --- impl A: new (polars-ds) -------------------------------------------------
def dimsonbeta_pds(df: pl.DataFrame) -> pl.DataFrame:
    name = "beta"
    beta_expr = pl.col("coeffs").list.head(3).list.sum()
    return (
        df.group_by(["id_int", "group_number"])
        .agg(
            coeffs=pds.lin_reg(
                "mktrf_lg1",
                "mktrf",
                "mktrf_ld1",
                target="ret_exc",
                add_bias=True,
                solver="cholesky",
            )
        )
        .select("id_int", "group_number", beta_expr.alias(name))
        .filter(pl.col(name).is_not_null() & pl.col(name).is_not_nan())
    )


# --- impl B: closed-form (current pre-revert impl, inlined) ------------------
def _solve_beta_sum_sym3(c00, c01, c02, c11, c12, c22, v0, v1, v2):
    col0 = (c00, c01, c02)
    col1 = (c01, c11, c12)
    col2 = (c02, c12, c22)
    v = (v0, v1, v2)

    def det(a, b, c):
        return (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - b[0] * (a[1] * c[2] - a[2] * c[1])
            + c[0] * (a[1] * b[2] - a[2] * b[1])
        )

    det_S = det(col0, col1, col2)
    num = det(v, col1, col2) + det(col0, v, col2) + det(col0, col1, v)
    rcond = det_S.abs() / (c00 * c11 * c22).abs()
    return pl.when(rcond > 1e-12).then(num / det_S).otherwise(None)


@functools.cache
def _dimson_exprs():
    X = ("mktrf_lg1", "mktrf", "mktrf_ld1")
    y = "ret_exc"
    pairs = [(X[i], X[j]) for i in range(3) for j in range(i, 3)] + [(x, y) for x in X]
    agg = tuple(
        (pl.var(a) if a == b else pl.cov(a, b)).alias(f"m{k}") for k, (a, b) in enumerate(pairs)
    )
    beta = _solve_beta_sum_sym3(*(pl.col(f"m{k}") for k in range(9)))
    return agg, beta


def dimsonbeta_closed(df: pl.DataFrame) -> pl.DataFrame:
    name = "beta"
    agg, beta = _dimson_exprs()
    return (
        df.group_by(["id_int", "group_number"])
        .agg(*agg)
        .select("id_int", "group_number", beta.alias(name))
        .filter(pl.col(name).is_not_null())
    )


# --- bench runner ------------------------------------------------------------
def time_runs(fn, df, n_runs: int = 5) -> float:
    fn(df)  # warm-up
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(df)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def max_abs_diff(a: pl.DataFrame, b: pl.DataFrame) -> float:
    joined = a.rename({"beta": "beta_a"}).join(
        b.rename({"beta": "beta_b"}), on=["id_int", "group_number"], how="inner"
    )
    return joined.select((pl.col("beta_a") - pl.col("beta_b")).abs().max()).item()


def main() -> None:
    rows = []
    for n_groups in [1_000, 10_000, 100_000, 1_000_000]:
        df = make_data(n_groups)
        t_closed = time_runs(dimsonbeta_closed, df)
        t_pds = time_runs(dimsonbeta_pds, df)
        diff = max_abs_diff(dimsonbeta_pds(df), dimsonbeta_closed(df))
        rows.append(
            {
                "n_groups": n_groups,
                "obs": df.height,
                "closed_s": t_closed,
                "pds_s": t_pds,
                "pds_vs_closed": t_pds / t_closed,
                "max_abs_diff": diff,
            }
        )
        print(
            f"n_groups={n_groups:>8}  closed={t_closed:.4f}s  pds={t_pds:.4f}s  "
            f"ratio={t_pds / t_closed:.2f}x  max_diff={diff:.2e}"
        )
    print()
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
