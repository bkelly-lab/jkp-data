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
    "_build_hml",
    "portfolios",
    "regional_data",
    "_build_regional_loop",
    "_stack_outputs",
    "_write_filtered",
    "_write_split_by_key",
    "run_portfolio",
    "run_themes",
]

_LAZY_REEXPORTS = frozenset(
    {
        "add_ecdf",
        "_build_industry_daily_returns",
        "_build_industry_monthly_returns",
        "_build_hml",
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


def _regions(settings: dict) -> pl.DataFrame:
    """Countries in each region of the regional portfolios.

    Description:
        Read the bundled country classification, drop countries without a code
        and the excluded countries, and list each region's countries and the
        minimum number of countries a regional portfolio needs.
    Output:
        DataFrame with ``name``, ``country_codes`` and ``countries_min``.
    """
    from .paths import get_country_classification_path

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
    return pl.DataFrame(
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


def _write_theme_outputs(
    *,
    out_dir: Path,
    suffix: str,
    themes: pl.DataFrame,
    hml: pl.DataFrame,
    date_col: str,
    mkt: pl.DataFrame,
    regions: pl.DataFrame,
    settings: dict,
) -> None:
    """Write one themes version's signed factors and theme portfolios for one return series.

    Description:
        Sign the HML factors of the version's characteristics (LMS), average them
        within subthemes and the subtheme returns within themes, and aggregate
        factors, themes and subthemes across countries into regions. A country's
        factor enters a regional factor only with ``stocks_min`` stocks on each
        side; a country's theme or subtheme enters with one factor.
    Steps:
        1) lms and country_factors/ from the signed factors.
        2) regional_factors/ from lms.
        3) themes and regional_themes/.
        4) subthemes and regional_subthemes/, skipped when every subtheme is its
           own theme, as in v0.
    Output:
        Files in ``out_dir``, each name ending in ``suffix``: "" (monthly),
        "_daily" or an overnight/intraday component such as "_overnight".
    """
    from .aux_functions import (
        _build_regional_loop,
        _sign_factors,
        _theme_returns,
        _write_filtered,
        _write_split_by_key,
    )

    end_date = settings["end_date"]
    regional = settings["regional_pfs"]
    regional_cols = ["n_countries", "ret_ew", "ret_vw", "ret_vw_cap", "mkt_vw_exc"]

    def regional_pfs(
        data: pl.DataFrame, char_col: str, cols: list[str], stocks_min: int
    ) -> pl.DataFrame:
        return _build_regional_loop(
            data=data,
            mkt=mkt,
            regions=regions,
            date_col=date_col,
            char_col=char_col,
            output_cols=["region", *cols, date_col, *regional_cols],
            weighting=regional["country_weights"],
            periods_min=regional["months_min"] * (1 if date_col == "eom" else 21),
            stocks_min=stocks_min,
        )

    # regional_data applies its minimum to n_stocks_min; for themes and
    # subthemes that is the factor count, and the direction is undefined.
    def as_regional_input(df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({"n_factors": "n_stocks_min"}).with_columns(
            pl.lit(None).cast(pl.Float64).alias("direction")
        )

    lms = _sign_factors(hml, themes)
    subthemes, theme_returns = _theme_returns(lms, themes, date_col)

    _write_filtered(lms, out_dir / f"lms{suffix}.parquet", date_col, end_date)
    _write_split_by_key(lms, out_dir / f"country_factors{suffix}", "excntry", date_col, end_date)
    _write_split_by_key(
        regional_pfs(
            lms, "characteristic", ["characteristic", "direction"], regional["stocks_min"]
        ),
        out_dir / f"regional_factors{suffix}",
        "region",
        date_col,
        end_date,
    )
    _write_filtered(theme_returns, out_dir / f"themes{suffix}.parquet", date_col, end_date)
    _write_split_by_key(
        regional_pfs(as_regional_input(theme_returns), "theme", ["theme"], 1),
        out_dir / f"regional_themes{suffix}",
        "region",
        date_col,
        end_date,
    )
    if (themes["subtheme"] != themes["theme"]).any():
        _write_filtered(subthemes, out_dir / f"subthemes{suffix}.parquet", date_col, end_date)
        regional_subthemes = (
            regional_pfs(as_regional_input(subthemes), "subtheme", ["subtheme"], 1)
            .join(
                themes.select("subtheme", "theme").unique(),
                on="subtheme",
                maintain_order="left",
            )
            .select("region", "theme", "subtheme", date_col, *regional_cols)
        )
        _write_split_by_key(
            regional_subthemes,
            out_dir / f"regional_subthemes{suffix}",
            "region",
            date_col,
            end_date,
        )


def _read_saved(path: Path) -> pl.DataFrame:
    """Read a saved output file, as parquet or, after a CSV run, as CSV."""
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        return pl.read_parquet(parquet)
    return pl.read_csv(path.with_suffix(".csv"), try_parse_dates=True)


def run_portfolio(*, output_format: str = "parquet", output_dir: Path) -> None:
    """Run JKP portfolio generation.

    Description:
        Orchestrate portfolio construction: configure output format, build
        factor portfolios for each country, write the unsigned results, and
        build the signed factors and theme portfolios of every released themes
        version.
    Steps:
        1) Configure output format and load the released themes versions.
        2) Load country list and portfolio inputs.
        3) Construct portfolios per country (monthly, daily, industry) for
           ``PORTFOLIO_CHARS`` and every characteristic in a themes version.
        4) Aggregate cross-country results and compute HML factors.
        5) Write output files.
        6) Write each themes version's outputs under ``portfolios/themes/``.
        7) Optionally convert to CSV.
    Output:
        Portfolio files written to data/processed/portfolios/.
    """
    # Function-local: LOAD_GLOBAL inside this function bypasses module
    # __getattr__, so helpers must be bound explicitly here. Also keeps
    # importing `jkp.data.portfolio` cheap (no eager duckdb/ibis pull).
    from .aux_functions import (
        _build_hml,
        _stack_outputs,
        _write_filtered,
        load_themes,
        portfolios,
        theme_versions,
    )
    from .paths import get_themes_dir

    paths = DataPaths(base_dir=output_dir)
    portfolios_dir = paths.processed_dir / "portfolios"
    other_output_dir = paths.processed_dir / "other_output"
    chars_dir = paths.processed_dir / "characteristics"

    configure_output_format(output_format)

    # Get list of countries from characteristics files; null_country holds stocks
    # with a null excntry and is not a country.
    countries = sorted(
        p.stem
        for p in chars_dir.glob("*.parquet")
        if "world" not in p.stem and p.stem != "null_country"
    )

    # Sort every characteristic that a released themes version signs. Loading
    # the versions first stops a run with an invalid themes file before sorting.
    theme_sets = {
        name: load_themes(path) for name, path in theme_versions(get_themes_dir()).items()
    }
    themed = set().union(*(set(t["characteristic"]) for t in theme_sets.values()))
    chars = PORTFOLIO_CHARS + sorted(themed - set(PORTFOLIO_CHARS))
    settings = PORTFOLIO_SETTINGS

    print(
        f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )

    regions = _regions(settings)

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

    # Overnight / intraday daily return components present in the daily files.
    # They are aggregated in the main pass alongside ret_exc.
    oi_components: list[tuple[str, str]] = []
    if settings["daily_pf"]:
        sample_path = (
            paths.processed_dir
            / "return_data"
            / "daily_rets_by_country"
            / f"{countries[0]}.parquet"
        )
        daily_schema = pl.scan_parquet(sample_path).collect_schema()
        for component, ret_col in [
            ("overnight", "ret_overnight"),
            ("intraday", "ret_intraday"),
            ("overnight_local", "ret_overnight_local"),
            ("intraday_local", "ret_intraday_local"),
        ]:
            if ret_col in daily_schema.names():
                oi_components.append((component, ret_col))
            else:
                print(
                    f"Skipping {component}: {ret_col} not found in daily return files",
                    flush=True,
                )

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
            daily_oi_cols=[ret_col for _, ret_col in oi_components],
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

    # Create HML Returns
    if pf_returns is not None and pf_returns.height > 0:
        hml_returns = _build_hml(pf_returns, settings["pfs"], "eom", include_signal=True)
    else:
        hml_returns = None

    if settings["daily_pf"] and pf_daily is not None and pf_daily.height > 0:
        hml_daily = _build_hml(pf_daily, settings["pfs"], "date", include_signal=False)
    else:
        hml_daily = None

    # Extract CMP returns
    cmp_list = [d["cmp"] for d in portfolio_data.values() if "cmp" in d]
    cmp_returns = pl.concat(cmp_list) if cmp_list else None
    if cmp_returns is None:
        print("No 'cmp' keys found")

    end_date = settings["end_date"]

    # Single-file outputs (monthly)
    monthly_outputs = [
        (pf_returns, "pfs.parquet"),
        (hml_returns, "hml.parquet"),
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

    # HML factors by return series; the themes versions sign them below.
    hml_series = [("", hml_returns, "eom"), ("_daily", hml_daily, "date")]

    # Overnight / Intraday daily factor portfolios (LPS 2019 decomposition).
    # The main pass aggregated each component's daily returns with the
    # standard ret_exc_lead1m sort; only post-processing runs per component.
    for component, ret_col in oi_components:
        pf_oi = _stack_outputs(
            portfolio_data, f"pf_daily_{ret_col}", ["excntry", "characteristic", "pf", "date"]
        )
        if pf_oi is None or pf_oi.height == 0:
            print(f"No {component} results produced", flush=True)
            continue

        print(f"Building {component} daily factor portfolios...", flush=True)
        hml_oi = _build_hml(pf_oi, settings["pfs"], "date", include_signal=False)
        _write_filtered(pf_oi, portfolios_dir / f"pfs_{component}.parquet", "date", end_date)
        _write_filtered(hml_oi, portfolios_dir / f"hml_{component}.parquet", "date", end_date)
        hml_series.append((f"_{component}", hml_oi, "date"))

    # Signed factors and theme portfolios of each released themes version
    for suffix, hml, date_col in hml_series:
        if hml is None:
            continue
        for name, themes in theme_sets.items():
            print(f"Building themes {name}{suffix}...", flush=True)
            _write_theme_outputs(
                out_dir=portfolios_dir / "themes" / name,
                suffix=suffix,
                themes=themes,
                hml=hml,
                date_col=date_col,
                mkt=market if date_col == "eom" else market_daily,
                regions=regions,
                settings=settings,
            )

    # Convert to CSV if configured
    convert_outputs_to_csv(processed_dir=paths.processed_dir)

    print(
        f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )


def run_themes(*, output_dir: Path, version: str | None = None, file: Path | None = None) -> None:
    """Build signed factors and theme portfolios from saved HML factors.

    Description:
        Read the HML factors written by ``jkp portfolio`` and build the outputs
        of every released themes version, of one released version
        (``version``), or of an edited themes file in the v1.0 format
        (``file``). Outputs are written in the format of the saved files.
    Steps:
        1) Load the requested themes versions.
        2) For each saved hml file (monthly, daily, overnight/intraday), read it
           and the market returns, and write each version's outputs.
        3) After a CSV run, convert the written folders to CSV.
    Output:
        Files under ``portfolios/themes/<version>/``, or
        ``portfolios/themes/<file name without extension>/`` for ``file``.
    """
    from .aux_functions import load_themes, theme_versions
    from .paths import get_themes_dir

    paths = DataPaths(base_dir=output_dir)
    portfolios_dir = paths.processed_dir / "portfolios"
    other_output_dir = paths.processed_dir / "other_output"
    settings = PORTFOLIO_SETTINGS

    released = theme_versions(get_themes_dir())
    if file is not None:
        if file.stem in released:
            raise ValueError(f"{file.name} has the name of a released themes version; rename it")
        sources = {file.stem: file}
    elif version is not None:
        if version not in released:
            raise ValueError(f"unknown themes version {version!r}; released: {sorted(released)}")
        sources = {version: released[version]}
    else:
        sources = released
    theme_sets = {name: load_themes(path) for name, path in sources.items()}

    hml_names = sorted(
        {p.stem for p in portfolios_dir.glob("hml*") if p.suffix in (".parquet", ".csv")}
    )
    if not hml_names:
        raise FileNotFoundError(f"no hml files in {portfolios_dir}; run jkp portfolio first")
    output_format = "parquet" if (portfolios_dir / "hml.parquet").exists() else "csv"
    configure_output_format(output_format)

    regions = _regions(settings)
    for hml_name in hml_names:
        suffix = hml_name.removeprefix("hml")
        date_col = "eom" if suffix == "" else "date"
        market = "market_returns" if date_col == "eom" else "market_returns_daily"
        hml = _read_saved(portfolios_dir / hml_name)
        mkt = _read_saved(other_output_dir / market)
        for name, themes in theme_sets.items():
            print(f"Building themes {name}{suffix}...", flush=True)
            _write_theme_outputs(
                out_dir=portfolios_dir / "themes" / name,
                suffix=suffix,
                themes=themes,
                hml=hml,
                date_col=date_col,
                mkt=mkt,
                regions=regions,
                settings=settings,
            )

    if output_format == "csv":
        for name in theme_sets:
            convert_outputs_to_csv(processed_dir=portfolios_dir / "themes" / name)
