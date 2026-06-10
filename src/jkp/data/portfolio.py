import time
from pathlib import Path
from typing import Any as _Any

import polars as pl

from .config import (
    PORTFOLIO_CHARS,
    PORTFOLIO_SETTINGS,
)
from .output_writer import (
    configure_output_format,
    convert_outputs_to_csv,
)
from .paths import DataPaths

# add_ecdf / portfolios / regional_data and the private build/write helpers
# are resolved lazily from `aux_functions` via `__getattr__` below so that
# `import jkp.data.portfolio` doesn't pull duckdb/ibis at import time.
__all__ = [  # noqa: F822 — lazy via __getattr__
    "add_ecdf",
    "_build_industry_daily_returns",
    "_build_industry_monthly_returns",
    "_build_hml_lms",
    "portfolios",
    "regional_data",
    "_build_regional_loop",
    "_stack_outputs",
    "_write_filtered",
    "_write_split_by_key",
    "run_portfolio",
]

_LAZY_REEXPORTS = frozenset(
    {
        "add_ecdf",
        "_build_industry_daily_returns",
        "_build_industry_monthly_returns",
        "_build_hml_lms",
        "portfolios",
        "regional_data",
        "_build_regional_loop",
        "_stack_outputs",
        "_write_filtered",
        "_write_split_by_key",
    }
)


def __getattr__(name: str) -> _Any:
    if name in _LAZY_REEXPORTS:
        from . import aux_functions

        value = getattr(aux_functions, name)
        globals()[name] = value  # cache so future lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    standard_dunders = {n for n in globals() if n.startswith("__") and n.endswith("__")}
    return sorted(set(__all__) | standard_dunders)


def _build_oi_factor_returns(
    *,
    paths: DataPaths,
    countries: list[str],
    chars: list[str],
    settings: dict,
    nyse_size_cutoffs: pl.DataFrame,
    ret_cutoffs: pl.DataFrame,
    ret_cutoffs_daily: pl.DataFrame | None,
    char_info: pl.DataFrame,
    cluster_labels: pl.DataFrame,
    market_daily: pl.DataFrame | None,
    regions: pl.DataFrame,
    suffix: str,
    daily_ret_col: str,
) -> dict[str, pl.DataFrame | None]:
    """Build daily-only factor portfolios for a single return component (overnight or intraday).

    Uses ret_exc_lead1m for monthly sorting/filtering (standard universe),
    but only produces daily portfolio returns using the specified daily_ret_col.

    Returns a dict keyed by output name containing DataFrames for daily outputs,
    or None when a frame is empty.
    """
    from .aux_functions import (
        _build_hml_lms,
        _build_regional_loop,
        _stack_outputs,
        portfolios,
    )

    portfolio_data: dict = {}
    for ex in countries:
        print(f"  {suffix} | {ex}: {countries.index(ex) + 1} out of {len(countries)}")
        result = portfolios(
            paths=paths,
            excntry=ex,
            chars=chars,
            pfs=settings["pfs"],
            bps=settings["bps"],
            bp_min_n=settings["bp_min_n"],
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=settings["source"],
            wins_ret=settings["wins_ret"],
            cmp_key=False,
            signals=False,
            daily_pf=True,
            ind_pf=False,
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            monthly_ret_col="ret_exc_lead1m",
            daily_ret_col=daily_ret_col,
        )
        portfolio_data[ex] = result

    pf_daily = _stack_outputs(
        portfolio_data, "pf_daily", ["excntry", "characteristic", "pf", "date"]
    )

    hml_daily, lms_daily = None, None
    if pf_daily is not None and pf_daily.height > 0:
        hml_daily, lms_daily = _build_hml_lms(
            pf_daily, char_info, settings["pfs"], "date", include_signal=False
        )

    cluster_pfs_daily = None
    if lms_daily is not None:
        cluster_pfs_daily = (
            lms_daily.join(cluster_labels, on="characteristic", how="left")
            .group_by(["excntry", "cluster", "date"])
            .agg(
                [
                    pl.len().alias("n_factors"),
                    pl.col("ret_ew").mean().alias("ret_ew"),
                    pl.col("ret_vw").mean().alias("ret_vw"),
                    pl.col("ret_vw_cap").mean().alias("ret_vw_cap"),
                ]
            )
        )

    weighting = settings["regional_pfs"]["country_weights"]
    months_min = settings["regional_pfs"]["months_min"]
    stocks_min = settings["regional_pfs"]["stocks_min"]
    lms_cols_daily = [
        "region",
        "characteristic",
        "direction",
        "date",
        "n_countries",
        "ret_ew",
        "ret_vw",
        "ret_vw_cap",
        "mkt_vw_exc",
    ]
    cluster_cols_daily = [
        "region",
        "cluster",
        "date",
        "n_countries",
        "ret_ew",
        "ret_vw",
        "ret_vw_cap",
        "mkt_vw_exc",
    ]

    regional_pfs_daily = None
    if lms_daily is not None:
        regional_pfs_daily = _build_regional_loop(
            data=lms_daily,
            mkt=market_daily,
            regions=regions,
            date_col="date",
            char_col="characteristic",
            output_cols=lms_cols_daily,
            weighting=weighting,
            periods_min=months_min * 21,
            stocks_min=stocks_min,
        )

    regional_clusters_daily = None
    if cluster_pfs_daily is not None:
        regional_clusters_daily = _build_regional_loop(
            data=cluster_pfs_daily.rename({"n_factors": "n_stocks_min"}).with_columns(
                pl.lit(None).cast(pl.Float64).alias("direction")
            ),
            mkt=market_daily,
            regions=regions,
            date_col="date",
            char_col="cluster",
            output_cols=cluster_cols_daily,
            weighting=weighting,
            periods_min=months_min * 21,
            stocks_min=1,
        )

    return {
        "pf_daily": pf_daily,
        "hml_daily": hml_daily,
        "lms_daily": lms_daily,
        "cluster_pfs_daily": cluster_pfs_daily,
        "regional_pfs_daily": regional_pfs_daily,
        "regional_clusters_daily": regional_clusters_daily,
    }


def run_portfolio(*, output_format: str = "parquet", output_dir: Path) -> None:
    """Run JKP portfolio generation.

    Description:
        Orchestrate portfolio construction: parse arguments, configure output
        format, build factor portfolios for each country, and write results.
    Steps:
        1) Parse CLI arguments and configure output format.
        2) Load country list and characteristic definitions.
        3) Construct portfolios per country (monthly, daily, industry).
        4) Aggregate cross-country results and compute long-minus-short factors.
        5) Write output files and optionally convert to CSV.
    Output:
        Portfolio files written to data/processed/portfolios/.
    """
    # Function-local: LOAD_GLOBAL inside this function bypasses module
    # __getattr__, so helpers must be bound explicitly here. Also keeps
    # importing `jkp.data.portfolio` cheap (no eager duckdb/ibis pull).
    from .aux_functions import (
        _build_hml_lms,
        _build_regional_loop,
        _stack_outputs,
        _write_filtered,
        _write_split_by_key,
        portfolios,
    )

    paths = DataPaths(base_dir=output_dir)
    portfolios_dir = paths.processed_dir / "portfolios"
    other_output_dir = paths.processed_dir / "other_output"
    chars_dir = paths.processed_dir / "characteristics"

    configure_output_format(output_format)

    # Get list of countries from characteristics files
    countries = sorted(p.stem for p in chars_dir.glob("*.parquet") if "world" not in p.stem)

    chars = PORTFOLIO_CHARS
    settings = PORTFOLIO_SETTINGS

    print(
        f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )

    # Extract Necessary Information
    # Read Factor details from bundled Excel file
    from .paths import (
        get_cluster_labels_path,
        get_country_classification_path,
        get_factor_details_path,
    )

    char_info = (
        pl.read_excel(
            get_factor_details_path(),
            sheet_name="details",
        )
        .filter(pl.col("abr_jkp").is_not_null())
        .select([pl.col("abr_jkp").alias("characteristic"), pl.col("direction").cast(pl.Int32)])
    )

    # Read country classification details from bundled Excel file
    country_classification = pl.read_excel(
        get_country_classification_path(),
        sheet_name="countries",
    )

    # Drop rows with NA in 'excntry' and exclude specific countries
    country_classification = country_classification.select(
        ["excntry", "msci_development", "region"]
    ).filter(
        (pl.col("excntry").is_not_null())
        & (~pl.col("excntry").is_in(settings["regional_pfs"]["country_excl"]))
    )

    # Creating the regions DataFrame
    regions = pl.DataFrame(
        {
            "name": ["developed", "emerging", "frontier", "world", "world_ex_us"],
            "country_codes": [
                country_classification.filter(
                    (pl.col("msci_development") == "developed") & (pl.col("excntry") != "USA")
                )["excntry"].to_list(),
                country_classification.filter(pl.col("msci_development") == "emerging")[
                    "excntry"
                ].to_list(),
                country_classification.filter(pl.col("msci_development") == "frontier")[
                    "excntry"
                ].to_list(),
                country_classification["excntry"].to_list(),
                country_classification.filter(pl.col("excntry") != "USA")["excntry"].to_list(),
            ],
            "countries_min": [settings["regional_pfs"]["countries_min"]] * 3 + [1, 3],
        }
    )

    # Read cluster labels from bundled CSV file
    cluster_labels = pl.read_csv(
        get_cluster_labels_path(),
        infer_schema_length=int(1e10),
    )

    # nyse_cutoffs
    nyse_size_cutoffs = pl.read_parquet(other_output_dir / "nyse_cutoffs.parquet")

    # return_cutoffs
    ret_cutoffs = pl.read_parquet(other_output_dir / "return_cutoffs.parquet")
    ret_cutoffs = ret_cutoffs.with_columns(
        (pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1")
    )
    ret_cutoffs_daily = None
    if settings["daily_pf"]:
        ret_cutoffs_daily = pl.read_parquet(other_output_dir / "return_cutoffs_daily.parquet")

    # market_returns
    market = pl.read_parquet(other_output_dir / "market_returns.parquet")

    # daily_market_returns
    market_daily = None
    if settings["daily_pf"]:
        market_daily = pl.read_parquet(other_output_dir / "market_returns_daily.parquet")

    # Creating portfolios by using the portfolios function
    portfolio_data = {}
    for ex in countries:
        print(f"{ex}: {countries.index(ex) + 1} out of {len(countries)}")
        result = portfolios(
            paths=paths,
            excntry=ex,
            chars=chars,
            pfs=settings["pfs"],
            bps=settings["bps"],
            bp_min_n=settings["bp_min_n"],
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=settings["source"],
            wins_ret=settings["wins_ret"],
            cmp_key=settings["cmp"]["us"] if ex.lower() == "usa" else settings["cmp"]["int"],
            signals=settings["signals"]["us"]
            if ex.lower() == "usa"
            else settings["signals"]["int"],
            signals_standardize=settings["signals"]["standardize"],
            signals_w=settings["signals"]["weight"],
            daily_pf=settings["daily_pf"],
            ind_pf=settings["ind_pf"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
        )
        portfolio_data[ex] = result

    # Aggregating portfolio returns across countries
    pf_returns = _stack_outputs(
        portfolio_data,
        "pf_returns",
        sort_cols=["excntry", "characteristic", "pf", "eom"],
        select_cols=[
            "excntry",
            "characteristic",
            "pf",
            "eom",
            "n",
            "signal",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
        ],
    )
    pf_daily = (
        _stack_outputs(portfolio_data, "pf_daily", ["excntry", "characteristic", "pf", "date"])
        if settings["daily_pf"]
        else None
    )

    # Industry classification returns
    if settings["ind_pf"]:
        gics_returns = _stack_outputs(portfolio_data, "gics_returns", ["excntry", "gics", "eom"])
        ff49_returns = _stack_outputs(portfolio_data, "ff49_returns", ["excntry", "ff49", "eom"])
    else:
        gics_returns = None
        ff49_returns = None

    if settings["ind_pf"] and settings["daily_pf"]:
        gics_daily = _stack_outputs(portfolio_data, "gics_daily", ["excntry", "gics", "date"])
        ff49_daily = _stack_outputs(portfolio_data, "ff49_daily", ["excntry", "ff49", "date"])
    else:
        gics_daily = None
        ff49_daily = None

    # Create HML / LMS Returns
    if pf_returns is not None and pf_returns.height > 0:
        hml_returns, lms_returns = _build_hml_lms(
            pf_returns, char_info, settings["pfs"], "eom", include_signal=True
        )
    else:
        hml_returns = None
        lms_returns = None

    if settings["daily_pf"] and pf_daily is not None and pf_daily.height > 0:
        hml_daily, lms_daily = _build_hml_lms(
            pf_daily, char_info, settings["pfs"], "date", include_signal=False
        )
    else:
        hml_daily = None
        lms_daily = None

    # Extract CMP returns
    cmp_list = [d["cmp"] for d in portfolio_data.values() if "cmp" in d]
    cmp_returns = pl.concat(cmp_list) if cmp_list else None
    if cmp_returns is None:
        print("No 'cmp' keys found")

    # Create Clustered Portfolios
    if lms_returns is not None:
        cluster_pfs = (
            lms_returns.join(cluster_labels, on="characteristic", how="left")
            .group_by(["excntry", "cluster", "eom"])
            .agg(
                [
                    pl.len().alias("n_factors"),
                    pl.col("ret_ew").mean().alias("ret_ew"),
                    pl.col("ret_vw").mean().alias("ret_vw"),
                    pl.col("ret_vw_cap").mean().alias("ret_vw_cap"),
                ]
            )
        )
    else:
        cluster_pfs = None

    # Conditional Operation for Daily Clustered Portfolios
    if settings["daily_pf"] and lms_daily is not None:
        cluster_pfs_daily = (
            lms_daily.join(cluster_labels, on="characteristic", how="left")
            .group_by(["excntry", "cluster", "date"])
            .agg(
                [
                    pl.len().alias("n_factors"),
                    pl.col("ret_ew").mean().alias("ret_ew"),
                    pl.col("ret_vw").mean().alias("ret_vw"),
                    pl.col("ret_vw_cap").mean().alias("ret_vw_cap"),
                ]
            )
        )
    else:
        cluster_pfs_daily = None

    weighting = settings["regional_pfs"]["country_weights"]
    months_min = settings["regional_pfs"]["months_min"]
    stocks_min = settings["regional_pfs"]["stocks_min"]
    lms_cols_monthly = [
        "region",
        "characteristic",
        "direction",
        "eom",
        "n_countries",
        "ret_ew",
        "ret_vw",
        "ret_vw_cap",
        "mkt_vw_exc",
    ]
    lms_cols_daily = [c if c != "eom" else "date" for c in lms_cols_monthly]
    cluster_cols_monthly = [
        "region",
        "cluster",
        "eom",
        "n_countries",
        "ret_ew",
        "ret_vw",
        "ret_vw_cap",
        "mkt_vw_exc",
    ]
    cluster_cols_daily = [c if c != "eom" else "date" for c in cluster_cols_monthly]

    # Creating regional portfolios
    if lms_returns is not None:
        regional_pfs = _build_regional_loop(
            data=lms_returns,
            mkt=market,
            regions=regions,
            date_col="eom",
            char_col="characteristic",
            output_cols=lms_cols_monthly,
            weighting=weighting,
            periods_min=months_min,
            stocks_min=stocks_min,
        )
    else:
        regional_pfs = None

    if settings["daily_pf"] and lms_daily is not None:
        regional_pfs_daily = _build_regional_loop(
            data=lms_daily,
            mkt=market_daily,
            regions=regions,
            date_col="date",
            char_col="characteristic",
            output_cols=lms_cols_daily,
            weighting=weighting,
            periods_min=months_min * 21,
            stocks_min=stocks_min,
        )
    else:
        regional_pfs_daily = None

    # Creating regional clusters
    if cluster_pfs is not None:
        regional_clusters = _build_regional_loop(
            data=cluster_pfs.rename({"n_factors": "n_stocks_min"}).with_columns(
                pl.lit(None).cast(pl.Float64).alias("direction")
            ),
            mkt=market,
            regions=regions,
            date_col="eom",
            char_col="cluster",
            output_cols=cluster_cols_monthly,
            weighting=weighting,
            periods_min=months_min,
            stocks_min=1,
        )
    else:
        regional_clusters = None

    if settings["daily_pf"] and cluster_pfs_daily is not None:
        regional_clusters_daily = _build_regional_loop(
            data=cluster_pfs_daily.rename({"n_factors": "n_stocks_min"}).with_columns(
                pl.lit(None).cast(pl.Float64).alias("direction")
            ),
            mkt=market_daily,
            regions=regions,
            date_col="date",
            char_col="cluster",
            output_cols=cluster_cols_daily,
            weighting=weighting,
            periods_min=months_min * 21,
            stocks_min=1,
        )
    else:
        regional_clusters_daily = None

    end_date = settings["end_date"]

    # Single-file outputs (monthly)
    monthly_outputs = [
        (pf_returns, "pfs.parquet"),
        (hml_returns, "hml.parquet"),
        (lms_returns, "lms.parquet"),
        (cluster_pfs, "clusters.parquet"),
    ]
    for df, name in monthly_outputs:
        if df is not None:
            _write_filtered(df, portfolios_dir / name, "eom", end_date)
    if cmp_returns is not None:
        _write_filtered(cmp_returns, portfolios_dir / "cmp.parquet", "eom", end_date)

    # Single-file outputs (daily)
    if settings["daily_pf"]:
        daily_outputs = [
            (pf_daily, "pfs_daily.parquet"),
            (hml_daily, "hml_daily.parquet"),
            (lms_daily, "lms_daily.parquet"),
            (cluster_pfs_daily, "clusters_daily.parquet"),
        ]
        for df, name in daily_outputs:
            if df is not None:
                _write_filtered(df, portfolios_dir / name, "date", end_date)

    # Industry returns
    if settings["ind_pf"]:
        ind_monthly = [
            (gics_returns, "industry_gics.parquet"),
            (ff49_returns, "industry_ff49.parquet"),
        ]
        for df, name in ind_monthly:
            if df is not None:
                _write_filtered(df, portfolios_dir / name, "eom", end_date)

    if settings["ind_pf"] and settings["daily_pf"]:
        ind_daily = [
            (gics_daily, "industry_gics_daily.parquet"),
            (ff49_daily, "industry_ff49_daily.parquet"),
        ]
        for df, name in ind_daily:
            if df is not None:
                _write_filtered(df, portfolios_dir / name, "date", end_date)

    # Partitioned outputs
    if regional_pfs is not None:
        _write_split_by_key(
            regional_pfs, portfolios_dir / "regional_factors", "region", "eom", end_date
        )
    if settings["daily_pf"] and regional_pfs_daily is not None:
        _write_split_by_key(
            regional_pfs_daily,
            portfolios_dir / "regional_factors_daily",
            "region",
            "date",
            end_date,
        )
    if regional_clusters is not None:
        _write_split_by_key(
            regional_clusters,
            portfolios_dir / "regional_clusters",
            "region",
            "eom",
            end_date,
        )
    if settings["daily_pf"] and regional_clusters_daily is not None:
        _write_split_by_key(
            regional_clusters_daily,
            portfolios_dir / "regional_clusters_daily",
            "region",
            "date",
            end_date,
        )
    if lms_returns is not None:
        _write_split_by_key(
            lms_returns,
            portfolios_dir / "country_factors",
            "excntry",
            "eom",
            end_date,
        )
    if settings["daily_pf"] and lms_daily is not None:
        _write_split_by_key(
            lms_daily,
            portfolios_dir / "country_factors_daily",
            "excntry",
            "date",
            end_date,
        )

    # -- Daily overnight and intraday factor returns --
    if settings["daily_pf"]:
        _first_daily_file = (
            paths.processed_dir
            / "return_data"
            / "daily_rets_by_country"
            / f"{countries[0]}.parquet"
        )
        _daily_cols = set(pl.scan_parquet(_first_daily_file).collect_schema().names())
        for suffix, d_col in [
            ("overnight", "ret_overnight"),
            ("intraday", "ret_intraday"),
        ]:
            if d_col not in _daily_cols:
                print(
                    f"\nSkipping daily {suffix} factor returns ({d_col!r} not in daily data)",
                    flush=True,
                )
                continue
            print(f"\nBuilding daily {suffix} factor returns ...", flush=True)
            oi = _build_oi_factor_returns(
                paths=paths,
                countries=countries,
                chars=chars,
                settings=settings,
                nyse_size_cutoffs=nyse_size_cutoffs,
                ret_cutoffs=ret_cutoffs,
                ret_cutoffs_daily=ret_cutoffs_daily,
                char_info=char_info,
                cluster_labels=cluster_labels,
                market_daily=market_daily,
                regions=regions,
                suffix=suffix,
                daily_ret_col=d_col,
            )

            daily_oi = [
                (oi["pf_daily"], f"pfs_daily_{suffix}.parquet"),
                (oi["hml_daily"], f"hml_daily_{suffix}.parquet"),
                (oi["lms_daily"], f"lms_daily_{suffix}.parquet"),
                (oi["cluster_pfs_daily"], f"clusters_daily_{suffix}.parquet"),
            ]
            for df, name in daily_oi:
                if df is not None:
                    _write_filtered(df, portfolios_dir / name, "date", end_date)

            if oi["regional_pfs_daily"] is not None:
                _write_split_by_key(
                    oi["regional_pfs_daily"],
                    portfolios_dir / f"regional_factors_daily_{suffix}",
                    "region",
                    "date",
                    end_date,
                )
            if oi["regional_clusters_daily"] is not None:
                _write_split_by_key(
                    oi["regional_clusters_daily"],
                    portfolios_dir / f"regional_clusters_daily_{suffix}",
                    "region",
                    "date",
                    end_date,
                )
            if oi["lms_daily"] is not None:
                _write_split_by_key(
                    oi["lms_daily"],
                    portfolios_dir / f"country_factors_daily_{suffix}",
                    "excntry",
                    "date",
                    end_date,
                )

    # Convert to CSV if configured
    convert_outputs_to_csv(processed_dir=paths.processed_dir)

    print(
        f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )
