"""Equivalence test: the narrow-sort correction and filter steps must reproduce the legacy
full-width-sort output.

correct_decimal_errors and drop_unreliable_observations used to sort every column of their
input to a spill parquet and run the kernels on that file. They now sort only the keys and
kernel columns (plus a row index) and splice the result back onto the unsorted input. The
legacy orchestrators are copied here verbatim and run on the same shuffled panels; the
outputs must be equal row for row after sorting by key (row order is the input's now).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.compustat_correction import (
    _build_filter_inputs,
    _group_starts,
    _run_decimal_correction,
    correct_decimal_errors,
    drop_unreliable_observations,
    run_all_filters,
)
from jkp.data.config import (
    CORRECTION_SPILL_COMPRESSION,
    DECIMAL_CORRECTION_METHODS,
    DECIMAL_DETECTION_WINDOWS,
    LOW_PRICE_COUNTRIES,
)

GROUP = ["gvkey", "iid"]
SORT = "datadate"


# Verbatim copies of the pre-rewrite orchestrators and the helpers they changed.
def _legacy_sort_to_spill(df: pl.LazyFrame, sort_keys: list[str], path: Path) -> None:
    """Streaming-sort the input by sort_keys to a spill parquet."""
    df.sort(sort_keys).sink_parquet(path, compression=CORRECTION_SPILL_COMPRESSION)


def _legacy_load_correction_inputs(
    sorted_path: Path, group_cols: list[str], sort_col: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Collect the correction value columns as f64; return (data, starts)."""
    lf = pl.scan_parquet(sorted_path)
    schema_names = lf.collect_schema().names()
    value_cols = [
        v for v in ("trfd", "qunit", "adrrc", "ajexdi", "prccd", "cshoc") if v in schema_names
    ]
    data = lf.select(
        group_cols + [sort_col] + [pl.col(c).cast(pl.Float64) for c in value_cols]
    ).collect()
    return data, _group_starts(data, group_cols)


def _legacy_reattach_corrected(
    sorted_path: Path, corrected: dict[str, np.ndarray], spill_dir: Path
) -> pl.LazyFrame:
    """Splice corrected columns over the originals (which move to the end)."""
    corr_path = spill_dir / "__corr_corrected_cols.parquet"
    pl.DataFrame(corrected).with_columns(
        [pl.col(name).fill_nan(None) for name in corrected]
    ).write_parquet(corr_path, compression=CORRECTION_SPILL_COMPRESSION)
    return pl.concat(
        [pl.scan_parquet(sorted_path).drop(list(corrected)), pl.scan_parquet(corr_path)],
        how="horizontal",
    )


def _legacy_correct_decimal_errors(
    df: pl.LazyFrame, has_adrrc: bool, correction_method: str, spill_dir: Path
) -> pl.LazyFrame:
    sorted_path = spill_dir / "__corr_sorted.parquet"
    _legacy_sort_to_spill(df, GROUP + [SORT], sorted_path)
    data, starts = _legacy_load_correction_inputs(sorted_path, GROUP, SORT)
    corrected = _run_decimal_correction(
        data, starts, DECIMAL_DETECTION_WINDOWS, has_adrrc, correction_method, 1.3
    )
    return _legacy_reattach_corrected(sorted_path, corrected, spill_dir)


def _legacy_load_filter_inputs(
    sorted_path: Path, group_cols: list[str], sort_col: str, country_col: str
) -> pl.DataFrame:
    """Collect kernel inputs in one pass: f64 values, int32 date, country flags."""
    value_cols = ["ajexdi", "prc", "me", "ri", "cshoc", "dolvol"]
    return (
        pl.scan_parquet(sorted_path)
        .select(
            group_cols
            + [sort_col]
            + [pl.col(c).cast(pl.Float64) for c in value_cols]
            + [
                pl.col(sort_col).cast(pl.Date).to_physical().cast(pl.Int32).alias("dates"),
                pl.col(country_col).is_in(LOW_PRICE_COUNTRIES).fill_null(False).alias("low"),
                (pl.col(country_col) == "CHN").fill_null(False).alias("chn"),
            ]
        )
        .collect()
    )


def _legacy_reattach_reason(sorted_path: Path, reason: np.ndarray, spill_dir: Path) -> pl.LazyFrame:
    """Attach the kernel's keep/remove decision and filter to the kept rows."""
    reason_path = spill_dir / "__corr_filter_reason.parquet"
    pl.DataFrame({"_reason": reason}).write_parquet(
        reason_path, compression=CORRECTION_SPILL_COMPRESSION
    )
    return (
        pl.concat([pl.scan_parquet(sorted_path), pl.scan_parquet(reason_path)], how="horizontal")
        .filter(pl.col("_reason") == 0)
        .drop("_reason")
    )


def _legacy_drop_unreliable_observations(df: pl.LazyFrame, spill_dir: Path) -> pl.LazyFrame:
    sorted_path = spill_dir / "__corr_filter_sorted.parquet"
    _legacy_sort_to_spill(df, GROUP + [SORT], sorted_path)
    data = _legacy_load_filter_inputs(sorted_path, GROUP, SORT, "excntry")
    inp = _build_filter_inputs(data, GROUP)
    reason = np.zeros(data.height, dtype=np.int8)
    run_all_filters(
        inp.starts,
        reason,
        inp.remove_low_volume,
        inp.ajexdi,
        inp.prc,
        inp.me,
        inp.ri,
        inp.cshoc,
        inp.dates,
        inp.low,
        inp.chn,
        inp.gap_days,
    )
    return _legacy_reattach_reason(sorted_path, reason, spill_dir)


def _shuffled_scan(frames: list[pl.DataFrame], path: Path) -> pl.LazyFrame:
    """Concatenate, shuffle the rows and scan from parquet, as the pipeline reads its input.
    NaN marks a missing value in the fixtures and becomes null here."""
    pl.concat(frames).fill_nan(None).sample(fraction=1.0, shuffle=True, seed=11).write_parquet(path)
    return pl.scan_parquet(path)


def _spill_dirs(tmp_path: Path) -> tuple[Path, Path]:
    legacy, new = tmp_path / "legacy", tmp_path / "new"
    legacy.mkdir()
    new.mkdir()
    return legacy, new


def _edited(values: np.ndarray, changes: dict[int, float]) -> np.ndarray:
    out = values.copy()
    for pos, value in changes.items():
        out[pos] = value
    return out


def _correction_security(
    gvkey: str, iid: str, n: int = 60, level: float = 20.0, **edits: dict[int, float]
) -> pl.DataFrame:
    """A random-walk price with flat clean baselines; edits multiply values at positions,
    and a NaN factor makes the value missing."""
    rng = np.random.default_rng(int(gvkey) * 10 + int(iid))
    base = {
        "ajexdi": np.full(n, 2.0),
        "prccd": level * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "cshoc": np.full(n, 8.0),
        "trfd": np.ones(n),
        "qunit": np.ones(n),
        "adrrc": np.ones(n),
    }
    for name, factors in edits.items():
        base[name] = _edited(base[name], {p: base[name][p] * f for p, f in factors.items()})
    return pl.DataFrame(
        {
            "gvkey": [gvkey] * n,
            "iid": [iid] * n,
            "datadate": pl.Series(np.arange(n, dtype=np.int64)).cast(pl.Date),
            **base,
            "curcdd": ["USD"] * n,
        }
    )


def _correction_panel() -> list[pl.DataFrame]:
    """Single-period 10x/100x/1000x price errors, a multi-period run, share/trfd/qunit/adrrc
    errors, NULLs, a sub-$1 price (floor variants) and two iids under one gvkey."""
    return [
        _correction_security("000001", "1", prccd={10: 10.0, 30: 100.0, 45: 1000.0}),
        _correction_security("000001", "2"),
        _correction_security(
            "000002", "1", prccd={15: 10.0, 16: 10.0, 17: 10.0}, cshoc={40: 1000.0}
        ),
        _correction_security("000003", "1", trfd={20: 100.0}, qunit={25: 10.0}),
        _correction_security("000004", "1", adrrc={12: 10.0}, prccd={5: np.nan}, trfd={8: np.nan}),
        _correction_security("000005", "1", level=0.5, prccd={20: 0.1}),
    ]


@pytest.mark.parametrize("has_adrrc", [True, False])
@pytest.mark.parametrize("method", DECIMAL_CORRECTION_METHODS)
def test_correct_decimal_errors_matches_legacy(
    tmp_path: Path, method: str, has_adrrc: bool
) -> None:
    lf = _shuffled_scan(_correction_panel(), tmp_path / "input.parquet")
    legacy_dir, new_dir = _spill_dirs(tmp_path)

    legacy = _legacy_correct_decimal_errors(lf, has_adrrc, method, legacy_dir).collect()
    new = correct_decimal_errors(
        lf, has_adrrc=has_adrrc, correction_method=method, spill_dir=new_dir
    ).collect()

    assert_frame_equal(new.sort(GROUP + [SORT]), legacy.sort(GROUP + [SORT]))
    # the panel exercises the correction, not only the pass-through
    before = lf.sort(GROUP + [SORT]).collect()["prccd"]
    assert (before != new.sort(GROUP + [SORT])["prccd"]).sum() > 0


def _filter_security(g: int, n: int = 40, iid: str = "01", **cols) -> pl.DataFrame:
    """A clean security whose volume rises with g, so the low-volume cut takes g = 0."""
    prc = 10.0 * np.exp(np.cumsum(np.full(n, 0.001)))
    base = {
        "gvkey": [f"{g:06d}"] * n,
        "iid": [iid] * n,
        "datadate": pl.Series(np.arange(n, dtype=np.int64) * 2).cast(pl.Date),
        "ajexdi": np.ones(n),
        "cshoc": np.full(n, 5.0),
        "prc": prc,
        "ri": prc.copy(),
        "dolvol": prc * 1000.0 * (g + 1),
        "excntry": ["USA"] * n,
        "exchg": [11] * n,
    }
    base.update(cols)
    base["me"] = base["prc"] * base["cshoc"]
    return pl.DataFrame(base)


def _filter_panel() -> list[pl.DataFrame]:
    """One defect per filter on high-volume securities, low-price-country and CHN rows,
    and a second iid under the zero-ajexdi gvkey."""
    n = 40
    prc = 10.0 * np.exp(np.cumsum(np.full(n, 0.001)))
    days = np.arange(n, dtype=np.int64) * 2
    sub_cent_from_30 = _edited(prc, dict.fromkeys(range(30, n), 0.005))
    defects = {
        59: {"ajexdi": _edited(np.ones(n), {5: 0.0})},
        58: {"prc": _edited(prc, {0: 0.004})},
        # a sub-cent price from day 30 is a breach in the USA, not in Brazil
        57: {"prc": sub_cent_from_30, "cshoc": np.full(n, 1000.0)},
        56: {"prc": sub_cent_from_30, "cshoc": np.full(n, 1000.0), "excntry": ["BRA"] * n},
        55: {"datadate": pl.Series(np.where(np.arange(n) >= 20, days + 400, days)).cast(pl.Date)},
        54: {"ri": _edited(prc, {25: prc[25] * 2.0})},
        53: {"prc": _edited(prc, {1: prc[1] * 20.0})},
        52: {"cshoc": _edited(np.full(n, 5.0), {3: 50.0})},
        51: {"cshoc": _edited(np.full(n, 5.0), {3: 50.0}), "excntry": ["CHN"] * n},
        50: {"prc": _edited(prc, {10: np.nan})},
    }
    return [_filter_security(g, n, **defects.get(g, {})) for g in range(60)] + [
        _filter_security(59, n, iid="02")
    ]


def test_drop_unreliable_observations_matches_legacy(tmp_path: Path) -> None:
    lf = _shuffled_scan(_filter_panel(), tmp_path / "input.parquet")
    legacy_dir, new_dir = _spill_dirs(tmp_path)

    legacy = _legacy_drop_unreliable_observations(lf, legacy_dir).collect()
    new = drop_unreliable_observations(lf, GROUP, SORT, "excntry", spill_dir=new_dir).collect()

    assert_frame_equal(new.sort(GROUP + [SORT]), legacy.sort(GROUP + [SORT]))
    # the filters removed some rows but not all; zero ajexdi drops only its own iid
    assert 0 < new.height < lf.select(pl.len()).collect().item()
    kept = new.group_by(GROUP).len()
    assert kept.filter(pl.col("gvkey") == "000059")["iid"].to_list() == ["02"]
    assert kept.filter(pl.col("gvkey") == "000000").is_empty()
