"""
Generate golden fixture parquet files for prc_to_high, turnover, and zero_trades.

Run with:
    uv run python -m tests.fixtures.generate_rolling_golden
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from jkp.data.aux_functions import prc_to_high, turnover, zero_trades

GOLDEN_DIR = Path(__file__).parent / "golden"


# ---------------------------------------------------------------------------
# Input-builder helpers (imported by tests to reproduce identical inputs)
# ---------------------------------------------------------------------------


def build_prc_to_high_input(seed: int = 42) -> pl.DataFrame:
    """Build synthetic prc_to_high input with 500 ids, ~10 groups, 5-40 rows each."""
    rng = np.random.default_rng(seed)

    n_ids = 500
    n_groups = 10
    base_date = date(2024, 1, 1)
    date_range_days = 90  # [2024-01-01, 2024-03-31]

    rows: list[dict] = []
    for id_int in range(1, n_ids + 1):
        group_number = int(rng.integers(1, n_groups + 1))
        n_rows = int(rng.integers(5, 41))
        day_offsets = rng.integers(0, date_range_days, size=n_rows)
        # Allow ties: sometimes duplicate a day offset
        for i in range(len(day_offsets)):
            if rng.random() < 0.05:
                day_offsets[i] = day_offsets[max(0, i - 1)]
        prices = rng.uniform(1.0, 100.0, size=n_rows)
        # Set ~5% prices to 0.0
        zero_mask = rng.random(n_rows) < 0.05
        prices[zero_mask] = 0.0
        for j in range(n_rows):
            rows.append(
                {
                    "id_int": id_int,
                    "group_number": group_number,
                    "date": base_date + timedelta(days=int(day_offsets[j])),
                    "prc_adj": float(prices[j]),
                }
            )

    return pl.DataFrame(rows)


def build_turnover_input(seed: int = 43) -> pl.DataFrame:
    """Build synthetic turnover input with 500 ids × 5 groups, ~30 rows each."""
    rng = np.random.default_rng(seed)

    n_ids = 500
    group_numbers = [10, 20, 30, 40, 50]
    rows_per_id = 30

    rows: list[dict] = []
    for id_int in range(1, n_ids + 1):
        group_number = group_numbers[id_int % len(group_numbers)]
        tvol = rng.uniform(0.0, 1e6, size=rows_per_id)
        # ~10% zeros
        zero_mask = rng.random(rows_per_id) < 0.10
        tvol[zero_mask] = 0.0
        shares = rng.uniform(1.0, 100.0, size=rows_per_id)
        # ~3% zero shares
        zero_shares = rng.random(rows_per_id) < 0.03
        shares[zero_shares] = 0.0
        for j in range(rows_per_id):
            rows.append(
                {
                    "id_int": id_int,
                    "group_number": group_number,
                    "tvol": float(tvol[j]),
                    "shares": float(shares[j]),
                }
            )

    # Add mixed-magnitude stress groups
    for stress_id, group_number in [(9001, 10), (9002, 20)]:
        # scaled so tvol/shares*1e6 stays representable
        for tvol_val, shares_val in [(1e10, 1.0), (1.0, 1.0), (-1e10, 1.0), (1.0, 1.0)]:
            rows.append(
                {
                    "id_int": stress_id,
                    "group_number": group_number,
                    "tvol": tvol_val,
                    "shares": shares_val,
                }
            )
        # pad to satisfy __min=20
        for _ in range(16):
            rows.append(
                {
                    "id_int": stress_id,
                    "group_number": group_number,
                    "tvol": 1.0,
                    "shares": 1.0,
                }
            )

    return pl.DataFrame(rows)


def build_zero_trades_input(seed: int = 44) -> pl.DataFrame:
    """Build synthetic zero_trades input with 500 ids × 5 groups, ~30 rows each."""
    rng = np.random.default_rng(seed)

    n_ids = 500
    group_numbers = [10, 20, 30, 40, 50]
    rows_per_id = 30

    rows: list[dict] = []
    for id_int in range(1, n_ids + 1):
        group_number = group_numbers[id_int % len(group_numbers)]
        tvol = rng.uniform(0.0, 1e4, size=rows_per_id)
        # ~25% exactly zero
        zero_mask = rng.random(rows_per_id) < 0.25
        tvol[zero_mask] = 0.0
        shares = rng.uniform(1.0, 100.0, size=rows_per_id)
        # ~3% zero shares
        zero_shares = rng.random(rows_per_id) < 0.03
        shares[zero_shares] = 0.0
        for j in range(rows_per_id):
            rows.append(
                {
                    "id_int": id_int,
                    "group_number": group_number,
                    "tvol": float(tvol[j]),
                    "shares": float(shares[j]),
                }
            )

    # Add deliberately-tied turnover groups to exercise rank-tie behavior
    # Three ids in group_number=10 that will have identical mean turnover (all tvol=0)
    for tied_id in [8001, 8002, 8003]:
        for _ in range(30):
            rows.append(
                {
                    "id_int": tied_id,
                    "group_number": 10,
                    "tvol": 0.0,
                    "shares": 1.0,
                }
            )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # prc_to_high
    df_prc = build_prc_to_high_input(seed=42)
    result_prc = (
        prc_to_high(df_prc.lazy(), "_21d", __min=10).collect().sort(["id_int", "group_number"])
    )
    out_prc = GOLDEN_DIR / "prc_to_high_21d.parquet"
    result_prc.write_parquet(out_prc)
    print(f"prc_to_high_21d.parquet: {len(result_prc)} rows -> {out_prc}")

    # turnover
    df_tv = build_turnover_input(seed=43)
    result_tv = turnover(df_tv.lazy(), "_126d", __min=20).collect().sort(["id_int", "group_number"])
    out_tv = GOLDEN_DIR / "turnover_126d.parquet"
    result_tv.write_parquet(out_tv)
    print(f"turnover_126d.parquet: {len(result_tv)} rows -> {out_tv}")

    # zero_trades
    df_zt = build_zero_trades_input(seed=44)
    result_zt = (
        zero_trades(df_zt.lazy(), "_126d", __min=20).collect().sort(["id_int", "group_number"])
    )
    out_zt = GOLDEN_DIR / "zero_trades_126d.parquet"
    result_zt.write_parquet(out_zt)
    print(f"zero_trades_126d.parquet: {len(result_zt)} rows -> {out_zt}")


if __name__ == "__main__":
    main()
