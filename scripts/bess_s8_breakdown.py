"""Section 8 per-filter drop counts + condition-intersection matrix (US only).

The pre-Section-8 frame (Section-6-corrected, trfd-recovered, USD) is deleted
after every run and the raw tables are gone, so it is reconstructed exactly from
two persisted artifacts:
  * __comp_dsf_uncorrected_trfd.parquet  -- merged, trfd-recovered, pre-correction
  * bessembinder_corrections_log.parquet -- production Section 6 correction_factor
A price decimal error scales prc_local and ri_local by the same factor (both
proportional to prccd); adjcsho corrections scale cshoc. Applying the logged
factors reproduces the exact frame Section 8 ran on. Faithfulness is verified by
matching the slim-kernel sequential counts to the log's 8x filter_reason counts.

Two questions:
  1) per-filter drops -- sequential first-kill attribution (slim kernel reasons).
  2) intersection -- each filter's condition evaluated independently on the SAME
     pre-Section-8 frame, then pairwise overlap.

Usage (PYTHONPATH at a bessembinder_correction checkout):
    PYTHONPATH=src python scripts/bess_s8_breakdown.py
"""

import itertools
import json
from pathlib import Path

import polars as pl

from jkp.data.bessembinder import (
    Section8Params,
    _apply_section8_slim,
    filter_8a_trading_volume,
    filter_8b_ajex_qunit,
    filter_8c_low_price_me,
    filter_8d_data_gaps,
    filter_8e_adjcsho_jumps,
    filter_8f_me_jumps,
    filter_8g_return_filter,
    filter_8h_initial_errors,
)

I = Path("/home/ffr7/bessembinder_validation/data/interim")
US_SORTED = I / "s8_input_us_recon.parquet"
KEYS = ["gvkey", "iid", "datadate"]
GC, SC, CC = ["gvkey", "iid"], "datadate", "excntry"
P = Section8Params()


def reconstruct_pre_s8_us() -> None:
    """Exact US pre-Section-8 frame from uncorrected merge + logged S6 factors."""
    unc = pl.scan_parquet(I / "__comp_dsf_uncorrected_trfd.parquet")
    trfd = pl.scan_parquet(I / "__comp_dsf_bessembinder_trfd.parquet")
    log = pl.scan_parquet(I / "bessembinder_corrections_log.parquet")

    # exchg -> modal excntry from survivors; keep US exchange codes
    exchg2c = (
        trfd.group_by("exchg", "excntry")
        .agg(pl.len().alias("c"))
        .collect()
        .sort("c", descending=True)
        .unique(subset="exchg", keep="first")
    )
    us_exchg = exchg2c.filter(pl.col("excntry") == "USA")["exchg"].to_list()
    print(f"[recon] US exchg codes: {sorted(us_exchg)}", flush=True)

    pf = log.filter(pl.col("variable") == "adjprc").select(
        *KEYS, pl.col("correction_factor").alias("pf")
    )
    cf = log.filter(pl.col("variable") == "adjcsho").select(
        *KEYS, pl.col("correction_factor").alias("cf")
    )

    us = (
        unc.filter(pl.col("exchg").is_in(us_exchg))
        .join(pf, on=KEYS, how="left")
        .join(cf, on=KEYS, how="left")
        # apply Section 6 factors (price scales prc_local & ri_local; shares -> cshoc)
        .with_columns(
            pl.col("prc_local") * pl.col("pf").fill_null(1.0),
            pl.col("ri_local") * pl.col("pf").fill_null(1.0),
            pl.col("cshoc") * pl.col("cf").fill_null(1.0),
        )
        # recompute USD columns exactly as __comp_dsf3 (aux_functions.py:1332)
        .with_columns(
            (pl.col("prc_local") * pl.col("fx")).alias("prc"),
            (pl.col("prc_local") * pl.col("fx") * pl.col("cshoc")).alias("me"),
            (pl.col("cshtrd") * pl.col("prc_local") * pl.col("fx")).alias("dolvol"),
            (pl.col("ri_local") * pl.col("fx")).alias("ri"),
        )
        .with_columns(pl.lit("USA").alias("excntry"))
        .drop("pf", "cf")
    )
    us.sort(KEYS).sink_parquet(US_SORTED)
    print(f"[recon] wrote {US_SORTED}", flush=True)


def main() -> None:
    if not US_SORTED.exists():
        print("=== Step 0: reconstruct US pre-Section-8 frame ===", flush=True)
        reconstruct_pre_s8_us()
    schema = pl.scan_parquet(US_SORTED).collect_schema()
    n = pl.scan_parquet(US_SORTED).select(pl.len()).collect().item()
    print(f"[US] pre-Section-8 N = {n:,}", flush=True)

    # --- Step 1: sequential first-kill counts (production slim kernel) ---
    print("=== Step 1: sequential (first-kill) counts ===", flush=True)
    _, rem = _apply_section8_slim(pl.scan_parquet(US_SORTED), GC, SC, CC, I, US_SORTED, P)
    seq = (
        rem.group_by("filter_reason").agg(pl.len().alias("n")).collect().sort("n", descending=True)
    )
    seq_total = int(seq["n"].sum())
    with pl.Config(tbl_rows=20):
        print(seq, flush=True)
    print(f"total removed (sequential) = {seq_total:,}  survivors = {n - seq_total:,}", flush=True)

    # --- Step 2: independent per-filter condition sets ---
    print("=== Step 2: independent per-filter removal sets ===", flush=True)
    filters = [
        ("8a", lambda d: filter_8a_trading_volume(d, GC)),
        ("8b", lambda d: filter_8b_ajex_qunit(d, GC, SC)),
        ("8c", lambda d: filter_8c_low_price_me(d, GC, SC, CC)),
        ("8d", lambda d: filter_8d_data_gaps(d, GC, SC)),
        ("8e", lambda d: filter_8e_adjcsho_jumps(d, GC, SC, CC, P)),
        ("8f", lambda d: filter_8f_me_jumps(d, GC, SC, P)),
        ("8g", lambda d: filter_8g_return_filter(d, GC, SC, P)),
        ("8h", lambda d: filter_8h_initial_errors(d, GC, SC, P)),
    ]
    sets: dict[str, pl.DataFrame] = {}
    for name, fn in filters:
        _, removed = fn(pl.scan_parquet(US_SORTED))
        if removed is None:
            sets[name] = pl.DataFrame(schema={k: schema[k] for k in KEYS})
        else:
            sets[name] = removed.select(KEYS).unique().collect(streaming=True)
        print(f"[indep] {name}: {sets[name].height:,}", flush=True)

    names = [nm for nm, _ in filters]
    union = pl.concat(list(sets.values())).unique().height

    # --- Step 3: pairwise intersection matrix ---
    print("=== Step 3: pairwise intersection ===", flush=True)
    inter: dict[tuple[str, str], int] = {}
    for a, b in itertools.combinations_with_replacement(names, 2):
        inter[(a, b)] = (
            sets[a].height if a == b else sets[a].join(sets[b], on=KEYS, how="inner").height
        )
    print("      " + " ".join(f"{x:>11}" for x in names), flush=True)
    for a in names:
        row = [f"{a:>5}"]
        for b in names:
            key = (a, b) if (a, b) in inter else (b, a)
            row.append(f"{inter[key]:>11,}")
        print(" ".join(row), flush=True)

    out = {
        "N_pre_s8": n,
        "sequential": {r["filter_reason"]: int(r["n"]) for r in seq.iter_rows(named=True)},
        "sequential_total": seq_total,
        "independent": {x: sets[x].height for x in names},
        "independent_union": int(union),
        "pairwise": {f"{a}&{b}": v for (a, b), v in inter.items()},
    }
    (I / "s8_breakdown_us.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE {I / 's8_breakdown_us.json'}", flush=True)


if __name__ == "__main__":
    main()
