"""
Profile the real Compustat inputs and outputs of the accounting stage as rounded statistics.

Runs read-only on Yale (profile_compustat.slurm). The JSON it writes, tests/golden/compustat_profile.json,
is what the synthetic generator samples from and what the synthetic-data tests compare against. Only
aggregates leave Yale: column schemas, counts, rates, quantiles and correlations, never rows or firm
identifiers (the largest gvkey is kept so synthetic gvkeys can start above it).

Usage:
    python -m tests.golden.calibration.profile_compustat DATA_DIR CHARS_PARQUET OUT_JSON --vintage V
"""

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from jkp.data.aux_functions import acc_chars_list
from jkp.data.paths import DataPaths
from tests.golden.compustat_invariants import TABLES, check_invariants, load_tables

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
LIFE_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
MIN_OBS = 100
BATCH = 50
TOP_CURRENCIES = 15
MAX_GAP = 24
SCALE = {"annual": "at", "quarterly": "atq"}
CORR_ITEMS = ["sale", "ib", "lt", "che", "ppent", "ceq", "oancf", "capx"]


def _sig(x: float | None) -> float | None:
    """Round to 4 significant digits; None and NaN become None."""
    return None if x is None or x != x else float(f"{x:.4g}")


def _shares(s: pl.Series, keep: int | None = None) -> dict[str, float | None]:
    """Value shares of s (nulls as 'null'), the `keep` most common values and the rest as 'other'."""
    counts = s.cast(pl.String).fill_null("null").value_counts(sort=True, name="n")
    top = counts if keep is None else counts.head(keep)
    shares = {k: _sig(n / s.len()) for k, n in top.iter_rows()}
    if keep is not None and counts.height > keep:
        shares["other"] = _sig(counts["n"][keep:].sum() / s.len())
    return shares


def column_stats(lf: pl.LazyFrame, scale: str) -> dict[str, dict]:
    """
    Description:
        Per numeric (Decimal) item: null rate, sign shares, and quantiles of item / scale.
    Steps:
        1) In batches of columns: null rate over all rows; negative, zero and positive shares over non-null.
        2) Over rows with scale > 0: quantiles of item / scale when at least MIN_OBS ratios exist.
    Output:
        Dict of item to {null, neg, zero, pos, n_ratio, q}.
    """
    items = [c for c, t in lf.collect_schema().items() if isinstance(t, pl.Decimal)]
    den = pl.col(scale).cast(pl.Float64)
    positive = lf.filter(den > 0)
    out: dict[str, dict] = {}
    for i in range(0, len(items), BATCH):
        batch = items[i : i + BATCH]
        x = {c: pl.col(c).cast(pl.Float64) for c in batch}
        signs = (
            lf.select(
                [x[c].is_null().mean().alias(f"{c}|null") for c in batch]
                + [(x[c] < 0).mean().alias(f"{c}|neg") for c in batch]
                + [(x[c] == 0).mean().alias(f"{c}|zero") for c in batch]
                + [(x[c] > 0).mean().alias(f"{c}|pos") for c in batch]
            )
            .collect()
            .row(0, named=True)
        )
        ratios = (
            positive.select(
                [(x[c] / den).count().alias(f"{c}|n") for c in batch]
                + [(x[c] / den).quantile(q).alias(f"{c}|{q}") for c in batch for q in QUANTILES]
            )
            .collect()
            .row(0, named=True)
        )
        for c in batch:
            n = ratios[f"{c}|n"]
            out[c] = {
                **{k: _sig(signs[f"{c}|{k}"]) for k in ["null", "neg", "zero", "pos"]},
                "n_ratio": n,
                "q": [_sig(ratios[f"{c}|{q}"]) for q in QUANTILES] if n >= MIN_OBS else None,
            }
    return out


def panel_stats(lf: pl.LazyFrame, freq: str) -> dict:
    """
    Description:
        Panel shape of one table: firm counts and lives, reporting gaps, calendars and currencies.
    Steps:
        1) Per firm: first and last datadate, records; life-year and record-count quantiles; entry decade.
        2) Months between a firm's consecutive dates, as shares of 1..MAX_GAP and '>MAX_GAP'.
        3) Shares of fyr, of the currency (top TOP_CURRENCIES) and, for quarterly data, of fqtr;
           null rates and row counts by indfmt.
    Output:
        Dict of panel statistics.
    """
    cur = "curcdq" if freq == "quarterly" else "curcd"
    firms = (
        lf.group_by("gvkey")
        .agg(first=pl.col("datadate").min(), last=pl.col("datadate").max(), n=pl.len())
        .collect()
    )
    life = (firms["last"] - firms["first"]).dt.total_days() / 365.25
    gaps = (
        lf.select("gvkey", "datadate")
        .unique()
        .sort(["gvkey", "datadate"])
        .select(
            gap=(pl.col("datadate").dt.year() * 12 + pl.col("datadate").dt.month())
            .diff()
            .over("gvkey")
        )
        .drop_nulls()
        .collect()["gap"]
    )
    keys = lf.select("fyr", cur, "indfmt", *(["fqtr"] if freq == "quarterly" else [])).collect()
    out = {
        "rows": keys.height,
        "firms": firms.height,
        "first_year": firms["first"].min().year,
        "last_year": firms["last"].max().year,
        "life_years_q": [_sig(life.quantile(q)) for q in LIFE_QUANTILES],
        "records_per_firm_q": [_sig(firms["n"].quantile(q)) for q in LIFE_QUANTILES],
        "entry_decade": _shares(firms["first"].dt.year() // 10 * 10),
        "gap_months": _shares(
            gaps.clip(upper_bound=MAX_GAP + 1)
            .cast(pl.String)
            .replace(str(MAX_GAP + 1), f">{MAX_GAP}")
        ),
        "fyr": _shares(keys["fyr"]),
        "currency": _shares(keys[cur], TOP_CURRENCIES),
        "indfmt": _shares(keys["indfmt"]),
    }
    if freq == "quarterly":
        out["fqtr"] = _shares(keys["fqtr"])
    return out


def null_by_indfmt(lf: pl.LazyFrame) -> dict[str, dict[str, float | None]]:
    """Null rate of every Decimal item within each report format (INDL, FS)."""
    items = [c for c, t in lf.collect_schema().items() if isinstance(t, pl.Decimal)]
    rates = lf.group_by("indfmt").agg(pl.col(items).is_null().mean()).collect()
    return {
        row.pop("indfmt"): {k: _sig(v) for k, v in row.items()}
        for row in rates.iter_rows(named=True)
    }


def correlation_stats(lf: pl.LazyFrame, freq: str) -> dict:
    """
    Description:
        Joint structure for the generator's correlated sampling.
    Steps:
        1) Over records with scale > 0: log(scale) and items / scale (CORR_ITEMS, quarterly 'q' items).
        2) Spearman correlations between every pair, over records where both are present.
        3) Within-firm autocorrelation at one report (annual: 12 months; quarterly: 3 months) and,
           for quarterly data, one year (12 months).
    Output:
        Dict {'vars', 'spearman' (matrix rows), 'autocorr' {lag: {var: corr}}}.
    """
    scale = SCALE[freq]
    suffix = "q" if freq == "quarterly" else ""
    names = lf.collect_schema().names()
    items = [f"{i}{suffix}" for i in CORR_ITEMS if f"{i}{suffix}" in names]
    den = pl.col(scale).cast(pl.Float64)
    variables = [f"log_{scale}"] + [f"{i}/{scale}" for i in items]
    base = (
        lf.filter(den > 0)
        .select(
            "gvkey",
            month=pl.col("datadate").dt.year() * 12 + pl.col("datadate").dt.month(),
            **{f"log_{scale}": den.log()},
            **{f"{i}/{scale}": pl.col(i).cast(pl.Float64) / den for i in items},
        )
        .sort(["gvkey", "month", f"log_{scale}"], nulls_last=True)
        .unique(["gvkey", "month"], keep="first", maintain_order=True)
    )

    def corr(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        both = a.is_not_null() & b.is_not_null()
        return pl.corr(a.filter(both), b.filter(both), method="spearman")

    pairs = [(a, b) for a in variables for b in variables if a < b]
    spearman = base.select(
        [corr(pl.col(a), pl.col(b)).alias(f"{a}~{b}") for a, b in pairs]
    ).collect()
    matrix = [
        [1.0 if a == b else _sig(spearman[f"{min(a, b)}~{max(a, b)}"].item()) for b in variables]
        for a in variables
    ]
    lags = {"report": 12 if freq == "annual" else 3} | ({"year": 12} if freq == "quarterly" else {})
    autocorr = {}
    for lag_name, months in lags.items():
        lagged = base.join(
            base.with_columns(month=pl.col("month") + months), on=["gvkey", "month"], suffix="_lag"
        )
        row = lagged.select(
            [corr(pl.col(v), pl.col(f"{v}_lag")).alias(v) for v in variables]
        ).collect()
        autocorr[lag_name] = {v: _sig(row[v].item()) for v in variables}
    return {"vars": variables, "spearman": matrix, "autocorr": autocorr}


def output_stats(chars: Path) -> dict:
    """
    Description:
        Per accounting characteristic in the stage output: null and NaN rates, sign shares, median.
    Steps:
        1) Rows and firms of the output.
        2) In batches of columns: null rate and NaN rate over all rows; negative, zero and positive
           shares and the median over finite values.
    Output:
        Dict {'rows', 'firms', 'chars': {name: {null, nan, neg, zero, pos, median}}}.
    """
    lf = pl.scan_parquet(chars)
    names = [c for c in acc_chars_list() if c in lf.collect_schema().names()]
    size = lf.select(rows=pl.len(), firms=pl.col("gvkey").n_unique()).collect().row(0, named=True)
    out = {}
    for i in range(0, len(names), BATCH // 2):
        batch = names[i : i + BATCH // 2]
        v = {c: pl.col(c).cast(pl.Float64) for c in batch}
        fin = {c: v[c].fill_nan(None) for c in batch}
        row = (
            lf.select(
                [v[c].is_null().mean().alias(f"{c}|null") for c in batch]
                + [v[c].is_nan().fill_null(False).mean().alias(f"{c}|nan") for c in batch]
                + [(fin[c] < 0).mean().alias(f"{c}|neg") for c in batch]
                + [(fin[c] == 0).mean().alias(f"{c}|zero") for c in batch]
                + [(fin[c] > 0).mean().alias(f"{c}|pos") for c in batch]
                + [fin[c].median().alias(f"{c}|median") for c in batch]
            )
            .collect()
            .row(0, named=True)
        )
        for c in batch:
            out[c] = {
                k: _sig(row[f"{c}|{k}"]) for k in ["null", "nan", "neg", "zero", "pos", "median"]
            }
    return {**size, "chars": out}


def profile(data_dir: Path, chars: Path, vintage: str) -> dict:
    """
    Description:
        Build the full profile of the accounting stage's real inputs and outputs.
    Steps:
        1) Column schemas of the raw tables.
        2) Per fundamentals table (stage read filters): panel shape, item statistics, correlations,
           and for Global tables null rates by indfmt.
        3) Invariant rule results (compustat_invariants) and the largest gvkey.
        4) Output statistics of the accounting characteristics.
    Output:
        JSON-serializable dict.
    """
    paths = DataPaths(data_dir)
    tables = load_tables(paths)
    schemas = {
        t: [
            [c, str(d)]
            for c, d in pl.scan_parquet(paths.raw_tables_dir / f"{t}.parquet")
            .collect_schema()
            .items()
        ]
        for t in [*TABLES, "comp_exrt_dly"]
    }
    per_table = {}
    for t, lf in tables.items():
        freq = TABLES[t][2]
        per_table[t] = {
            "panel": panel_stats(lf, freq),
            "columns": column_stats(lf, SCALE[freq]),
            "correlations": correlation_stats(lf, freq),
        }
        if TABLES[t][0] == "GLOBAL":
            per_table[t]["null_by_indfmt"] = null_by_indfmt(lf)
    invariants = [{**r, "rate": _sig(r["rate"])} for r in check_invariants(paths).to_dicts()]
    max_gvkey = max(
        lf.select(pl.col("gvkey").cast(pl.Int64).max()).collect().item() for lf in tables.values()
    )
    return {
        "meta": {"vintage": vintage, "profiled": date.today().isoformat(), "quantiles": QUANTILES},
        "schemas": schemas,
        "tables": per_table,
        "invariants": invariants,
        "max_gvkey": max_gvkey,
        "outputs": output_stats(chars),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("chars", type=Path, help="acc_chars_world.parquet of a main-code build")
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--vintage", required=True, help="WRDS download date of data_dir, e.g. 2026-10-15"
    )
    args = parser.parse_args()
    result = profile(args.data_dir.expanduser(), args.chars.expanduser(), args.vintage)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
