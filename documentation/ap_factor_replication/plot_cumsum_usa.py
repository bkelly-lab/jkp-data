"""Compare JKP USA monthly factors vs WRDS / published references.

JKP source: data/interim/ap_factors_monthly.parquet (filtered excntry == 'USA').
Refs in factor_comparison/Reference_data/*_monthly.csv.

Exposes:
- MODELS: list[ModelSpec]
- load_jkp_usa(): full USA monthly factor frame
- load_ref(spec): reference frame for a model
- make_figure(spec, jkp=None) -> (fig, stats_df)

CLI: writes one PDF per model to factor_comparison/cumsum_<name>_usa.pdf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

mpl.rcParams.update(
    {
        "font.family": ["IBM Plex Sans", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.titlecolor": "#1a1818",
        "axes.labelsize": 9,
        "axes.labelcolor": "#1a1818",
        "axes.labelweight": "normal",
        "axes.edgecolor": "#1a1818",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "grid.color": "#1a1818",
        "grid.alpha": 0.08,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": "#1a1818",
        "ytick.color": "#1a1818",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.loc": "upper left",
        "legend.labelcolor": "#1a1818",
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    }
)

REF_COLOR = "#1a1818"
JKP_COLOR = "red"

REPO = Path(__file__).resolve().parents[1]
JKP_PATH = REPO / "data" / "interim" / "ap_factors_monthly.parquet"
# Reference series live next to this module (committed), so a fresh clone resolves
# them without the local factor_comparison/ symlink.
REF_DIR = Path(__file__).resolve().parent / "Reference_data"
OUT_DIR = REPO / "factor_comparison"

DateFmt = Literal["yyyymm", "year_month"]


@dataclass
class ModelSpec:
    name: str
    ref_file: str
    ref_skip: int
    ref_in_percent: bool
    date_fmt: DateFmt
    panels: list[tuple[str, str, str]]  # (display, jkp_col, ref_col)
    start: pl.Expr  # start date filter (applied to eom)
    title: str = ""  # figure suptitle
    to_present: bool = False  # last decade label "{ds} to Present"
    daily_ref_file: str = ""
    daily_ref_skip: int = 0
    daily_ref_in_percent: bool = True


NATURAL_START = pl.date(1900, 1, 31)  # no clamp

MODELS: list[ModelSpec] = [
    ModelSpec(
        name="FF3",
        ref_file="FF3_monthly.csv",
        daily_ref_file="FF3_daily.csv",
        daily_ref_skip=4,
        daily_ref_in_percent=True,
        ref_skip=4,
        ref_in_percent=True,
        date_fmt="yyyymm",
        panels=[
            ("SMB", "smb_ff3", "SMB"),
            ("HML", "hml_ff", "HML"),
        ],
        # DFF BE splice extends SMB/HML to 1926-07; compare the full history.
        start=NATURAL_START,
        title="Fama-French 3 factor: SMB and HML",
        to_present=True,
    ),
    ModelSpec(
        name="FF5",
        ref_file="FF5_monthly.csv",
        daily_ref_file="FF5_daily.csv",
        daily_ref_skip=4,
        daily_ref_in_percent=True,
        ref_skip=4,
        ref_in_percent=True,
        date_fmt="yyyymm",
        panels=[
            ("SMB", "smb_ff5", "SMB"),
            ("HML", "hml_ff", "HML"),
            ("RMW", "rmw_ff", "RMW"),
            ("CMA", "cma_ff", "CMA"),
        ],
        # French's FF5 series starts 1963-07; compare the full overlap.
        start=NATURAL_START,
        title="Fama-French 5 factor: SMB, HML, RMW, and CMA",
        to_present=True,
    ),
    ModelSpec(
        name="UMD",
        ref_file="FFmom_monthly.csv",
        daily_ref_file="FFmom_daily.csv",
        daily_ref_skip=13,
        daily_ref_in_percent=True,
        ref_skip=13,
        ref_in_percent=True,
        date_fmt="yyyymm",
        # Longer CRSP panel extends UMD to ~1927-01; compare the full history.
        panels=[("UMD", "umd_ff", "Mom")],
        start=NATURAL_START,
        title="Fama-French Momentum (UMD)",
        to_present=True,
    ),
    ModelSpec(
        name="HXZ",
        ref_file="hxz_monthly.csv",
        daily_ref_file="hxz_daily.csv",
        daily_ref_skip=0,
        daily_ref_in_percent=True,
        ref_skip=0,
        ref_in_percent=True,
        date_fmt="year_month",
        panels=[
            ("ME", "me_hxz", "R_ME"),
            ("IA", "ia_hxz", "R_IA"),
            ("ROE", "roe_hxz", "R_ROE"),
        ],
        start=NATURAL_START,
        title="Hou-Xue-Zhang q-factors: ME, IA, and ROE",
    ),
    ModelSpec(
        name="MP",
        ref_file="mp_factors_monthly.csv",
        daily_ref_file="mp_factors_daily.csv",
        daily_ref_skip=0,
        daily_ref_in_percent=False,
        ref_skip=0,
        ref_in_percent=False,
        date_fmt="yyyymm",
        panels=[
            ("SMB", "smb_mispricing", "SMB"),
            ("MGMT", "mispricing_mgmt", "MGMT"),
            ("PERF", "mispricing_perf", "PERF"),
        ],
        start=NATURAL_START,
        title="Stambaugh-Yuan mispricing factors: SMB, MGMT, and PERF",
    ),
    ModelSpec(
        name="DHS",
        ref_file="dhs_monthly.csv",
        # No daily reference: the original DHS (Lin Sun SAS) code produces only
        # monthly factors, so there is no daily benchmark to compare against — our
        # port additionally produces daily FIN/PEAD, but they are not shown here.
        daily_ref_file="",
        ref_skip=0,
        ref_in_percent=True,
        date_fmt="yyyymm",
        panels=[
            ("FIN", "fin_dhs", "FIN"),
            ("PEAD", "pead_dhs", "PEAD"),
        ],
        # FIN starts 1968-07, PEAD ~1971-10; the inner join with the reference clips it.
        start=NATURAL_START,
        title="Daniel-Hirshleifer-Sun behavioral factors: FIN and PEAD",
        to_present=True,
    ),
]


def load_jkp_usa() -> pl.DataFrame:
    cols: set[str] = {"eom"}
    for m in MODELS:
        for _, jcol, _ in m.panels:
            cols.add(jcol)
    return (
        pl.scan_parquet(JKP_PATH)
        .filter(pl.col("excntry") == "USA")
        .select(sorted(cols))
        .sort("eom")
        .collect()
    )


def load_ref(spec: ModelSpec) -> pl.DataFrame:
    path = REF_DIR / spec.ref_file
    raw = pl.read_csv(
        path,
        skip_rows=spec.ref_skip,
        has_header=True,
        truncate_ragged_lines=True,
        infer_schema_length=0,
    )

    if spec.date_fmt == "yyyymm":
        date_col = raw.columns[0]
        raw = raw.rename({date_col: "ymd"})
        raw = raw.filter(pl.col("ymd").str.strip_chars().str.len_chars() == 6)
        df = raw.with_columns(
            pl.col("ymd")
            .str.strip_chars()
            .str.strptime(pl.Date, "%Y%m")
            .dt.month_end()
            .alias("eom")
        )
    else:  # year_month
        raw = raw.filter(pl.col("year").str.strip_chars().str.len_chars() == 4)
        df = raw.with_columns(
            (
                pl.col("year").str.strip_chars()
                + "-"
                + pl.col("month").str.strip_chars().str.zfill(2)
                + "-01"
            )
            .str.strptime(pl.Date, "%Y-%m-%d")
            .dt.month_end()
            .alias("eom")
        )

    ref_value_cols = [rcol for _, _, rcol in spec.panels]
    cast_exprs = [pl.col(c).str.strip_chars().cast(pl.Float64) for c in ref_value_cols]
    if spec.ref_in_percent:
        cast_exprs = [e / 100.0 for e in cast_exprs]
    df = df.with_columns(cast_exprs)
    return df.select(["eom", *ref_value_cols]).sort("eom")


def _merge(spec: ModelSpec, jkp: pl.DataFrame) -> pl.DataFrame:
    ref = load_ref(spec)
    ref_renamed = ref.rename({rcol: f"__ref_{display}" for display, _, rcol in spec.panels})
    j_sel = jkp.select(["eom", *[jcol for _, jcol, _ in spec.panels]])
    return (
        j_sel.join(ref_renamed, on="eom", how="inner")
        .drop_nulls()
        .filter(pl.col("eom") >= spec.start)
        .sort("eom")
    )


def _period_label_and_mask(years: np.ndarray, decade_start: int) -> tuple[str, np.ndarray]:
    end = decade_start + 9
    label = f"{decade_start}–{end}"
    mask = (years >= decade_start) & (years <= end)
    return label, mask


def period_stats_for_factor(
    merged: pl.DataFrame, jcol: str, rkey: str, to_present: bool = False
) -> pl.DataFrame:
    """Decade-by-decade Mean / Std / Corr for one factor; Full Sample row first."""
    sub = merged.select(["eom", jcol, rkey]).drop_nulls()
    eom = sub["eom"].to_numpy()
    j = sub[jcol].to_numpy()
    r = sub[rkey].to_numpy()
    years = np.array([d.year for d in eom.tolist()])

    rows: list[dict] = []

    def _row(label: str, jj: np.ndarray, rr: np.ndarray) -> dict:
        if len(jj) < 2:
            return {
                "Time Period": label,
                "Mean JKP (%)": None,
                "Mean Benchmark (%)": None,
                "StD JKP (%)": None,
                "StD Benchmark (%)": None,
                "Corr (%)": None,
            }
        return {
            "Time Period": label,
            "Mean JKP (%)": round(float(jj.mean()) * 12 * 100, 2),
            "Mean Benchmark (%)": round(float(rr.mean()) * 12 * 100, 2),
            "StD JKP (%)": round(float(jj.std(ddof=1)) * np.sqrt(12) * 100, 2),
            "StD Benchmark (%)": round(float(rr.std(ddof=1)) * np.sqrt(12) * 100, 2),
            "Corr (%)": round(float(np.corrcoef(jj, rr)[0, 1]) * 100, 2),
        }

    rows.append(_row("Full Sample", j, r))

    if len(years) > 0:
        first_decade = (years.min() // 10) * 10
        last_decade = (years.max() // 10) * 10
        for ds in range(int(first_decade), int(last_decade) + 1, 10):
            label, mask = _period_label_and_mask(years, ds)
            if ds == int(last_decade):
                actual_max = int(years[mask].max()) if mask.any() else ds
                if to_present:
                    label = f"{ds} to Present"
                elif actual_max < ds + 9:
                    label = f"{ds}–{actual_max}"
            rows.append(_row(label, j[mask], r[mask]))

    return pl.DataFrame(rows)


def make_figure(spec: ModelSpec, jkp: pl.DataFrame | None = None):
    """Build cumulative-sum figure + per-factor period-stats tables.

    Returns:
        (fig, list[(display_name, stats_polars_df)])
    """
    if jkp is None:
        jkp = load_jkp_usa()
    merged = _merge(spec, jkp)

    n = len(spec.panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = list(axes[0])

    stats_per_factor: list[tuple[str, pl.DataFrame]] = []
    for ax, (display, jcol, _) in zip(axes, spec.panels):
        rkey = f"__ref_{display}"
        sub = merged.select(["eom", jcol, rkey]).drop_nulls()
        d = sub["eom"].to_numpy()
        j = sub[jcol].to_numpy()
        r = sub[rkey].to_numpy()

        ax.plot(d, np.cumsum(r), color=REF_COLOR, label="Reference", linewidth=1.1)
        ax.plot(d, np.cumsum(j), color=JKP_COLOR, linestyle="--", label="JKP", linewidth=1.1)
        ax.set_title(display)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Sum of Returns")
        ax.legend()

        stats_per_factor.append(
            (display, period_stats_for_factor(merged, jcol, rkey, to_present=spec.to_present))
        )

    fig.tight_layout()
    return fig, stats_per_factor


def _arrays_for_factor(spec: ModelSpec, display: str, jkp: pl.DataFrame):
    merged = _merge(spec, jkp)
    jcol = next(j for d, j, _ in spec.panels if d == display)
    rkey = f"__ref_{display}"
    sub = merged.select(["eom", jcol, rkey]).drop_nulls()
    return (
        sub["eom"].to_numpy(),
        sub[jcol].to_numpy(),
        sub[rkey].to_numpy(),
        merged,
        jcol,
        rkey,
    )


def model_axis_limits(spec: ModelSpec, jkp: pl.DataFrame | None = None):
    """Shared (xlim, ylim) spanning every factor in a model's tabset, so
    switching tabs moves only the plotted lines, not the axes."""
    if jkp is None:
        jkp = load_jkp_usa()
    x_lo = x_hi = y_lo = y_hi = None
    for display_name, _, _ in spec.panels:
        eom, j, r, *_ = _arrays_for_factor(spec, display_name, jkp)
        cj, cr = np.cumsum(j), np.cumsum(r)
        xs, ys = eom.min(), eom.max()
        lo = min(cj.min(), cr.min())
        hi = max(cj.max(), cr.max())
        x_lo = xs if x_lo is None else min(x_lo, xs)
        x_hi = ys if x_hi is None else max(x_hi, ys)
        y_lo = lo if y_lo is None else min(y_lo, lo)
        y_hi = hi if y_hi is None else max(y_hi, hi)
    pad = 0.05 * (y_hi - y_lo)
    return (x_lo, x_hi), (y_lo - pad, y_hi + pad)


def single_panel_figure(
    spec: ModelSpec,
    display: str,
    jkp: pl.DataFrame | None = None,
    xlim=None,
    ylim=None,
):
    """One factor, one figure."""
    if jkp is None:
        jkp = load_jkp_usa()
    eom, j, r, *_ = _arrays_for_factor(spec, display, jkp)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(eom, np.cumsum(r), color=REF_COLOR, label="Reference", linewidth=1.1)
    ax.plot(eom, np.cumsum(j), color=JKP_COLOR, linestyle="--", label="JKP", linewidth=1.1)
    ax.set_title(display, loc="center", fontsize=11, fontweight="semibold", color="#1a1818", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Sum of Returns")
    ax.legend()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Fixed axes rectangle (not tight_layout) so the plot box stays put across tabs.
    fig.subplots_adjust(left=0.135, right=0.97, top=0.9, bottom=0.135)
    return fig


def factor_period_stats(
    spec: ModelSpec, display: str, jkp: pl.DataFrame | None = None
) -> pl.DataFrame:
    if jkp is None:
        jkp = load_jkp_usa()
    _, _, _, merged, jcol, rkey = _arrays_for_factor(spec, display, jkp)
    return period_stats_for_factor(merged, jcol, rkey, to_present=spec.to_present)


def full_sample_summary(spec: ModelSpec, jkp: pl.DataFrame | None = None) -> pl.DataFrame:
    if jkp is None:
        jkp = load_jkp_usa()
    rows = []
    for display, _, _ in spec.panels:
        df = factor_period_stats(spec, display, jkp)
        full = df.row(0, named=True)
        rows.append(
            {
                "Factor": display,
                "Mean JKP (%)": full["Mean JKP (%)"],
                "Mean Benchmark (%)": full["Mean Benchmark (%)"],
                "StD JKP (%)": full["StD JKP (%)"],
                "StD Benchmark (%)": full["StD Benchmark (%)"],
                "Corr (%)": full["Corr (%)"],
            }
        )
    return pl.DataFrame(rows)


JKP_DAILY_PATH = REPO / "data" / "interim" / "ap_factors_daily.parquet"


def load_jkp_usa_daily() -> pl.DataFrame:
    cols: set[str] = {"date"}
    for m in MODELS:
        for _, jcol, _ in m.panels:
            cols.add(jcol)
    return (
        pl.scan_parquet(JKP_DAILY_PATH)
        .filter(pl.col("excntry") == "USA")
        .select(sorted(cols))
        .sort("date")
        .collect()
    )


def load_ref_daily(spec: ModelSpec) -> pl.DataFrame:
    path = REF_DIR / spec.daily_ref_file
    raw = pl.read_csv(
        path,
        skip_rows=spec.daily_ref_skip,
        has_header=True,
        truncate_ragged_lines=True,
        infer_schema_length=0,
    )
    date_col = raw.columns[0]
    raw = raw.rename({date_col: "ymd"})
    daily = raw.filter(pl.col("ymd").str.strip_chars().str.len_chars() == 8)
    df = daily.with_columns(
        pl.col("ymd").str.strip_chars().str.strptime(pl.Date, "%Y%m%d").alias("date")
    )
    ref_value_cols = [rcol for _, _, rcol in spec.panels]
    cast_exprs = [pl.col(c).str.strip_chars().cast(pl.Float64) for c in ref_value_cols]
    if spec.daily_ref_in_percent:
        cast_exprs = [e / 100.0 for e in cast_exprs]
    df = df.with_columns(cast_exprs)
    return df.select(["date", *ref_value_cols]).sort("date")


def _merge_daily(spec: ModelSpec, jkp_d: pl.DataFrame) -> pl.DataFrame:
    ref = load_ref_daily(spec)
    ref_renamed = ref.rename({rcol: f"__ref_{display}" for display, _, rcol in spec.panels})
    j_sel = jkp_d.select(["date", *[jcol for _, jcol, _ in spec.panels]])
    return (
        j_sel.join(ref_renamed, on="date", how="inner")
        .drop_nulls()
        .filter(pl.col("date") >= spec.start)
        .sort("date")
    )


def daily_period_stats_for_factor(
    merged: pl.DataFrame, jcol: str, rkey: str, to_present: bool = False
) -> pl.DataFrame:
    sub = merged.select(["date", jcol, rkey]).drop_nulls()
    dates = sub["date"].to_numpy()
    j = sub[jcol].to_numpy()
    r = sub[rkey].to_numpy()
    years = np.array([d.year for d in dates.tolist()])

    def _row(label: str, jj: np.ndarray, rr: np.ndarray) -> dict:
        if len(jj) < 2:
            return {
                "Time Period": label,
                "Mean JKP (%)": None,
                "Mean Benchmark (%)": None,
                "StD JKP (%)": None,
                "StD Benchmark (%)": None,
                "Corr (%)": None,
            }
        return {
            "Time Period": label,
            "Mean JKP (%)": round(float(jj.mean()) * 252 * 100, 2),
            "Mean Benchmark (%)": round(float(rr.mean()) * 252 * 100, 2),
            "StD JKP (%)": round(float(jj.std(ddof=1)) * np.sqrt(252) * 100, 2),
            "StD Benchmark (%)": round(float(rr.std(ddof=1)) * np.sqrt(252) * 100, 2),
            "Corr (%)": round(float(np.corrcoef(jj, rr)[0, 1]) * 100, 2),
        }

    rows = [_row("Full Sample", j, r)]
    if len(years) > 0:
        first_decade = (years.min() // 10) * 10
        last_decade = (years.max() // 10) * 10
        for ds in range(int(first_decade), int(last_decade) + 1, 10):
            mask = (years >= ds) & (years <= ds + 9)
            label = f"{ds}–{ds + 9}"
            if ds == int(last_decade):
                actual_max = int(years[mask].max()) if mask.any() else ds
                if to_present:
                    label = f"{ds} to Present"
                elif actual_max < ds + 9:
                    label = f"{ds}–{actual_max}"
            rows.append(_row(label, j[mask], r[mask]))
    return pl.DataFrame(rows)


def daily_factor_period_stats(spec: ModelSpec, display: str, jkp_d: pl.DataFrame) -> pl.DataFrame:
    merged = _merge_daily(spec, jkp_d)
    jcol = next(j for d, j, _ in spec.panels if d == display)
    rkey = f"__ref_{display}"
    return daily_period_stats_for_factor(merged, jcol, rkey, to_present=spec.to_present)


def date_window_note(spec: ModelSpec, jkp: pl.DataFrame | None = None) -> str:
    if jkp is None:
        jkp = load_jkp_usa()
    merged = _merge(spec, jkp)
    start = str(merged["eom"].min())
    end = str(merged["eom"].max())
    return f"Comparison window: USA, {start} to {end} (monthly, {len(merged)} obs)."


def main() -> None:
    jkp = load_jkp_usa()
    for spec in MODELS:
        fig, stats = make_figure(spec, jkp)
        out_pdf = OUT_DIR / f"cumsum_{spec.name.lower()}_usa.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_pdf}")
        print(date_window_note(spec, jkp))
        for name, df in stats:
            print(f"\n[{name}]")
            print(df)


if __name__ == "__main__":
    main()
