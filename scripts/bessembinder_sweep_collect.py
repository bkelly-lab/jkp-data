"""Aggregate Section 8 threshold-sweep results into one metrics table.

Description:
    Reads each sweep sandbox produced by bessembinder_sweep_launch.py and
    computes, per variant: daily accuracy vs CRSP, monthly-horizon accuracy
    (timing errors wash out at month end), deletion cost, and knob
    efficiency vs the 'default' variant (paper thresholds).
Steps:
    1) Per variant: daily stats from interim/bessembinder_<tag>/
       crsp_comparison_after.parquet; monthly stats by compounding within
       (permno, month); obs/securities kept from the corrected parquet;
       removals by reason from the corrections log.
    2) Knob efficiency: extra S8 removals vs default joined to the default
       after-comparison — share with |diff| < 1% is the false-kill proxy;
       tail-50 reduction per extra deleted obs is the marginal trade.
    3) Sort by daily correlation, write parquet + print table.
Output:
    SWEEP_ROOT/sweep_results.parquet and a printed summary table.

Usage (on the cluster, after all sweep jobs finish):
    PYTHONPATH=src python scripts/bessembinder_sweep_collect.py
"""

import json

import polars as pl
from bessembinder_sweep_launch import GRID, SWEEP_ROOT
from polars import col

S8_REASONS = [f"8{c}" for c in "abcdefgh"]


def _interim(tag: str):
    return SWEEP_ROOT / f"sandbox_{tag}" / "data" / "interim"


def _comparison(tag: str) -> pl.LazyFrame:
    return pl.scan_parquet(_interim(tag) / f"bessembinder_{tag}" / "crsp_comparison_after.parquet")


def daily_stats(cmp: pl.LazyFrame) -> dict:
    row = cmp.select(
        pl.corr("ret_comp", "ret_crsp").alias("corr_d"),
        col("ret_diff_abs").mean().alias("mad_d"),
        col("ret_diff").std().alias("std_d"),
        (col("ret_diff_abs") > 0.01).sum().alias("gt1pct"),
        (col("ret_diff_abs") > 0.10).sum().alias("gt10pct"),
        (col("ret_diff_abs") > 0.50).sum().alias("gt50pct"),
        pl.len().alias("matched_obs"),
    ).collect(engine="streaming")
    return row.to_dicts()[0]


def monthly_stats(cmp: pl.LazyFrame) -> dict:
    m = (
        cmp.group_by("permno", col("date").dt.truncate("1mo").alias("month"))
        .agg(
            ((col("ret_comp") + 1.0).product() - 1.0).alias("mret_comp"),
            ((col("ret_crsp") + 1.0).product() - 1.0).alias("mret_crsp"),
        )
        .select(
            pl.corr("mret_comp", "mret_crsp").alias("corr_m"),
            (col("mret_comp") - col("mret_crsp")).abs().mean().alias("mad_m"),
        )
        .collect(engine="streaming")
    )
    return m.to_dicts()[0]


def cost_stats(tag: str) -> dict:
    kept = pl.scan_parquet(_interim(tag) / f"__comp_dsf_bessembinder_{tag}.parquet")
    counts = kept.select(
        pl.len().alias("obs_kept"),
        pl.struct("gvkey", "iid").n_unique().alias("sec_kept"),
    ).collect(engine="streaming")
    log = pl.scan_parquet(
        _interim(tag) / f"bessembinder_corrections_log_bessembinder_{tag}.parquet"
    )
    by_reason = (
        log.filter(col("variable").str.starts_with("8"))
        .group_by("variable")
        .len()
        .collect(engine="streaming")
    )
    out = counts.to_dicts()[0]
    for prefix in S8_REASONS:
        out[f"rm_{prefix}"] = (
            by_reason.filter(col("variable").str.starts_with(prefix))["len"].sum() or 0
        )
    return out


def s8_removal_keys(tag: str) -> pl.LazyFrame:
    log = pl.scan_parquet(
        _interim(tag) / f"bessembinder_corrections_log_bessembinder_{tag}.parquet"
    )
    return log.filter(col("variable").str.starts_with("8")).select("gvkey", "iid", "datadate")


def main() -> None:
    default_cmp = _comparison("default")
    rows = []
    for tag in GRID:
        cmp = _comparison(tag)
        # params as one JSON column: variants override heterogeneous fields
        row = {"variant": tag, "params": json.dumps(GRID[tag])}
        row.update(daily_stats(cmp))
        row.update(monthly_stats(cmp))
        row.update(cost_stats(tag))
        rows.append(row)
        print(f"collected {tag}", flush=True)

    table = pl.DataFrame(rows, infer_schema_length=None)
    base = table.filter(col("variant") == "default").to_dicts()[0]

    # Knob efficiency vs default
    eff = []
    default_rm = s8_removal_keys("default").collect(engine="streaming")
    for tag in GRID:
        if tag == "default":
            eff.append(
                {
                    "variant": tag,
                    "extra_removed": 0,
                    "false_kill_share": None,
                    "tail50_per_1k_removed": None,
                }
            )
            continue
        extra = (
            s8_removal_keys(tag)
            .collect(engine="streaming")
            .join(default_rm, on=["gvkey", "iid", "datadate"], how="anti")
        )
        n_extra = extra.height
        fk = None
        if n_extra:
            graded = (
                extra.lazy()
                .join(
                    default_cmp.select(
                        "gvkey", "iid", col("date").alias("datadate"), "ret_diff_abs"
                    ),
                    on=["gvkey", "iid", "datadate"],
                    how="inner",
                )
                .select(pl.len().alias("n"), (col("ret_diff_abs") < 0.01).sum().alias("fine"))
                .collect(engine="streaming")
            )
        if n_extra and graded["n"][0]:
            fk = graded["fine"][0] / graded["n"][0]
        trow = table.filter(col("variant") == tag).to_dicts()[0]
        tail_fixed = base["gt50pct"] - trow["gt50pct"]
        eff.append(
            {
                "variant": tag,
                "extra_removed": n_extra,
                "false_kill_share": fk,
                "tail50_per_1k_removed": (1000.0 * tail_fixed / n_extra) if n_extra else None,
            }
        )
    table = table.join(pl.DataFrame(eff), on="variant")

    out_path = SWEEP_ROOT / "sweep_results.parquet"
    table.write_parquet(out_path)
    show = table.sort("corr_d", descending=True)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=24):
        print(show)
    print(f"written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
