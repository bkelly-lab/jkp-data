import re
import shutil
import time
from pathlib import Path
from typing import Any as _Any

import polars as pl

from .config import (
    CMP_SIZE_GROUPS,
    PORTFOLIO_CHARS,
    PORTFOLIO_SETTINGS,
    UNSIGNED_CMP_SIZE_GROUPS,
    UNSIGNED_CMP_VERSION,
)
from .output_writer import (
    configure_output_format,
    convert_outputs_to_csv,
)
from .paths import DataPaths

# portfolios / regional_data and the private build/write helpers
# are resolved lazily from `aux_functions` via `__getattr__` below so that
# `import jkp.data.portfolio` doesn't pull duckdb at import time.
__all__ = [  # noqa: F822 — lazy via __getattr__
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


# Overnight / intraday daily return components: (name, daily return column). FX cancels within
# a day, so `ret_intraday_local` equals `ret_intraday` by construction (see aux_functions.py) and
# `intraday_local` is built from `ret_intraday`; `_check_intraday_local` guards the equality.
# At most 3 distinct columns: with `ret_exc` they fill the 16 sums per row of
# `_sum_rows_kernel` (`_SUM_VALUES`), and `_sum_by_portfolio` raises on more.
OI_COMPONENTS = [
    ("overnight", "ret_overnight"),
    ("intraday", "ret_intraday"),
    ("overnight_local", "ret_overnight_local"),
    ("intraday_local", "ret_intraday"),
]


def _check_intraday_local(daily_path: Path) -> None:
    """Stop if ``ret_intraday_local`` isn't the ``ret_intraday`` published under its name.

    Description:
        One country's daily file is enough, as the same builders write every country's
        columns. Files without either column have nothing to check.
    Output:
        None. Raises ``ValueError`` if the columns differ.
    """
    daily = pl.scan_parquet(daily_path)
    if not {"ret_intraday", "ret_intraday_local"} <= set(daily.collect_schema().names()):
        return
    differ = daily.select(pl.col("ret_intraday").ne_missing(pl.col("ret_intraday_local")).any())
    if differ.collect().item():
        raise ValueError(f"ret_intraday_local differs from ret_intraday in {daily_path}")


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


def _write_theme_versions(
    *,
    portfolios_dir: Path,
    theme_sets: dict[str, pl.DataFrame],
    hml_series: list[tuple[str, pl.DataFrame, str]],
    cmp: dict[int, pl.DataFrame],
    market: pl.DataFrame,
    market_daily: pl.DataFrame | None,
    regions: pl.DataFrame,
    settings: dict,
) -> None:
    """Write the signed factors, CMP and theme portfolios of each themes version.

    Description:
        Used by ``jkp portfolio`` and ``jkp themes``. A missing factor
        (a characteristic of any version without an HML factor in any return
        series) stops the build before any folder is touched. Other errors while
        writing can leave a version's folder partly written.
    Steps:
        1) Raise ``ValueError`` listing, per version and series, the
           characteristics without an HML factor.
        2) Delete each version's folder, so no file of an earlier build remains.
        3) Write each version's outputs for each return series.
        4) Write each version's signed ``lms_cmp_{n}`` for the size groupings in ``cmp``,
           which maps the number of size groups to the unsigned CMP.
           ``UNSIGNED_CMP_VERSION`` instead gets ``cmp``, the old file: the unsigned US
           rows of ``UNSIGNED_CMP_SIZE_GROUPS`` size groups for its characteristics. A
           characteristic can lack CMP rows, since CMP skips months where the
           characteristic doesn't vary, so this step prints the characteristics with
           no rows instead of stopping.
    Output:
        Files under ``portfolios_dir/themes/<version>/``.
    """
    from .aux_functions import _sign_factors, _write_filtered

    missing = []
    for suffix, hml, _ in hml_series:
        factors = set(hml["characteristic"].unique())
        for name, themes in theme_sets.items():
            if absent := sorted(set(themes["characteristic"]) - factors):
                missing.append(f"{name} (hml{suffix}): {absent}")
    if missing:
        raise ValueError("no hml factor for: " + "; ".join(missing))

    for name in theme_sets:
        out_dir = portfolios_dir / "themes" / name
        if out_dir.exists():
            shutil.rmtree(out_dir)

    for suffix, hml, date_col in hml_series:
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

    for name, themes in theme_sets.items():
        if name != UNSIGNED_CMP_VERSION:
            version_cmp = {
                f"lms_cmp_{n_groups}": _sign_factors(df, themes) for n_groups, df in cmp.items()
            }
        elif UNSIGNED_CMP_SIZE_GROUPS in cmp:
            version_cmp = {
                "cmp": cmp[UNSIGNED_CMP_SIZE_GROUPS].filter(
                    pl.col("excntry") == "USA",
                    pl.col("characteristic").is_in(themes["characteristic"].implode()),
                )
            }
        else:
            version_cmp = {}
        for file_name, df in version_cmp.items():
            if df.is_empty():
                continue
            cmp_chars = set(df["characteristic"].unique())
            if absent := sorted(set(themes["characteristic"]) - cmp_chars):
                print(f"No {file_name} rows for themes {name}: {absent}", flush=True)
            _write_filtered(
                df,
                portfolios_dir / "themes" / name / f"{file_name}.parquet",
                "eom",
                settings["end_date"],
            )


def _read_saved(path: Path) -> pl.DataFrame:
    """Read a saved output file, as parquet or, after a CSV run, as CSV.

    Description:
        ``path`` is the file path without a suffix. A parquet file wins over a
        CSV file of the same name.
    Steps:
        1) Read ``<path>.parquet`` if it exists.
        2) Otherwise read ``<path>.csv``, parsing dates.
    Output:
        DataFrame with the file's contents.
    """
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        return pl.read_parquet(parquet)
    return pl.read_csv(path.with_suffix(".csv"), try_parse_dates=True)


def run_portfolio(*, output_format: str = "parquet", output_dir: Path) -> None:
    """Run JKP portfolio generation.

    Description:
        Orchestrate portfolio construction: configure output format, build
        factor portfolios for each country, write the unsigned results, and
        build the signed factors, CMP and theme portfolios of the current themes
        version. ``run_themes`` builds any other released version from the saved
        factors.
    Steps:
        1) Configure output format and load the released themes versions.
        2) Load country list and portfolio inputs.
        3) Construct portfolios per country (monthly, daily, industry) for
           ``PORTFOLIO_CHARS`` and every characteristic in a themes version.
        4) Aggregate cross-country results and compute HML factors.
        5) Write output files. When CMP is off, delete ``hml_cmp_{n}`` files left by an
           earlier run, so ``run_themes`` doesn't build from them.
        6) Write the current themes version's outputs under ``portfolios/themes/``.
        7) Optionally convert to CSV.
    Output:
        Portfolio files written to data/processed/portfolios/.
    """
    # Function-local: LOAD_GLOBAL inside this function bypasses module
    # __getattr__, so helpers must be bound explicitly here. Also keeps
    # importing `jkp.data.portfolio` cheap (no eager duckdb pull).
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
    # with a null excntry and is not a country. Sorted by the uppercased stem, which is
    # each country's excntry: the pf stacks below rely on this order.
    countries = sorted(
        (
            p.stem
            for p in chars_dir.glob("*.parquet")
            if "world" not in p.stem and p.stem != "null_country"
        ),
        key=str.upper,
    )

    # Sort every characteristic that a released themes version signs, so jkp themes
    # can build any of them from the saved factors. Loading the versions first stops
    # a run with an invalid themes file before sorting.
    released = {name: load_themes(path) for name, path in theme_versions(get_themes_dir()).items()}
    themed = set().union(*(set(t["characteristic"]) for t in released.values()))
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
        _check_intraday_local(sample_path)
        daily_schema = pl.scan_parquet(sample_path).collect_schema()
        for component, ret_col in OI_COMPONENTS:
            if ret_col in daily_schema.names():
                oi_components.append((component, ret_col))
            else:
                print(
                    f"Skipping {component}: {ret_col} not found in daily return files",
                    flush=True,
                )
    oi_columns = list(dict.fromkeys(ret_col for _, ret_col in oi_components))

    # Polars 2 runs a bare collect() on the streaming engine, whose parallel reductions
    # are not bit-reproducible; keep the in-memory default of Polars 1.x.
    pl.Config.set_engine_affinity("in-memory")

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
            daily_pf=settings["daily_pf"],
            ind_pf=settings["ind_pf"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            daily_oi_cols=oi_columns,
        )
        portfolio_data[ex] = result

    # Aggregating portfolio returns across countries. The pf outputs need no sort: countries
    # run in sorted order and _portfolio_rows sorts each one's rows by characteristic, pf and
    # date, so the stack is in (excntry, characteristic, pf, date) order.
    pf_returns = _stack_outputs(
        portfolio_data,
        "pf_returns",
        sort_cols=None,
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
        _stack_outputs(portfolio_data, "pf_daily", sort_cols=None) if settings["daily_pf"] else None
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
    cmp_returns = {
        n_groups: stacked
        for n_groups in CMP_SIZE_GROUPS
        if (
            stacked := _stack_outputs(
                portfolio_data, f"cmp_{n_groups}", ["excntry", "characteristic", "size_grp", "eom"]
            )
        )
        is not None
    }
    if not cmp_returns:
        print("No 'cmp' keys found")

    end_date = settings["end_date"]

    # Single-file outputs (monthly)
    for df, name in [(pf_returns, "pfs.parquet"), (hml_returns, "hml.parquet")]:
        if df is not None:
            _write_filtered(df, portfolios_dir / name, "eom", end_date)
    for n_groups in CMP_SIZE_GROUPS:
        if n_groups in cmp_returns:
            _write_filtered(
                cmp_returns[n_groups],
                portfolios_dir / f"hml_cmp_{n_groups}.parquet",
                "eom",
                end_date,
            )
        else:
            (portfolios_dir / f"hml_cmp_{n_groups}.parquet").unlink(missing_ok=True)
            (portfolios_dir / f"hml_cmp_{n_groups}.csv").unlink(missing_ok=True)

    # Single-file outputs (daily)
    if settings["daily_pf"]:
        for df, name in [(pf_daily, "pfs_daily.parquet"), (hml_daily, "hml_daily.parquet")]:
            if df is not None:
                _write_filtered(df, portfolios_dir / name, "date", end_date)

    # Industry returns
    if settings["ind_pf"]:
        for df, name in [
            (gics_returns, "industry_gics.parquet"),
            (ff49_returns, "industry_ff49.parquet"),
        ]:
            if df is not None:
                _write_filtered(df, portfolios_dir / name, "eom", end_date)

    if settings["ind_pf"] and settings["daily_pf"]:
        for df, name in [
            (gics_daily, "industry_gics_daily.parquet"),
            (ff49_daily, "industry_ff49_daily.parquet"),
        ]:
            if df is not None:
                _write_filtered(df, portfolios_dir / name, "date", end_date)

    # HML factors by return series; the themes versions sign them below.
    hml_series = [("", hml_returns, "eom"), ("_daily", hml_daily, "date")]

    # Overnight / Intraday daily factor portfolios (LPS 2019 decomposition).
    # The main pass aggregated each component's daily returns with the
    # standard ret_exc_lead1m sort; only post-processing runs per component.
    for ret_col in oi_columns:
        components = [component for component, c in oi_components if c == ret_col]
        names = " and ".join(components)
        # Already in key order, as pf_daily above.
        pf_oi = _stack_outputs(portfolio_data, f"pf_daily_{ret_col}", sort_cols=None)
        if pf_oi is None or pf_oi.height == 0:
            print(f"No {names} results produced", flush=True)
            continue

        print(f"Building {names} daily factor portfolios...", flush=True)
        hml_oi = _build_hml(pf_oi, settings["pfs"], "date", include_signal=False)
        for component in components:
            _write_filtered(pf_oi, portfolios_dir / f"pfs_{component}.parquet", "date", end_date)
            _write_filtered(hml_oi, portfolios_dir / f"hml_{component}.parquet", "date", end_date)
            hml_series.append((f"_{component}", hml_oi, "date"))
        del pf_oi

    # The themes step needs only the HML factors and CMP, so free the rest first.
    del portfolio_data, pf_returns, pf_daily, gics_returns, ff49_returns, gics_daily, ff49_daily

    # Signed factors and theme portfolios of the current themes version.
    # The CSV conversion runs even if the themes step fails.
    current = next(iter(released))
    try:
        _write_theme_versions(
            portfolios_dir=portfolios_dir,
            theme_sets={current: released[current]},
            hml_series=[s for s in hml_series if s[1] is not None],
            cmp=cmp_returns,
            market=market,
            market_daily=market_daily,
            regions=regions,
            settings=settings,
        )
    finally:
        # Convert to CSV if configured
        convert_outputs_to_csv(processed_dir=paths.processed_dir)

    print(
        f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )


def run_themes(*, output_dir: Path, version: str | None = None, file: Path | None = None) -> None:
    """Build signed factors, CMP and theme portfolios from saved HML factors and CMP.

    Description:
        Read the HML factors and CMP written by ``jkp portfolio`` and build the outputs
        of the current themes version, of another released version named by
        ``version``, or of an edited themes file in the v1.0 format (``file``).
        Outputs are written in the format of the saved files.
    Steps:
        1) Load the requested themes version.
        2) Read each saved hml file (monthly, daily, overnight/intraday), the saved
           ``hml_cmp_{n}`` files if any, and the market returns, and write the version's
           outputs, replacing its folder.
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
        # The file's name becomes a folder under themes/ that is replaced.
        if file.suffix != ".csv":
            raise ValueError(f"{file} is not a .csv file")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", file.stem):
            raise ValueError(
                f"{file.name}: the file name must start with a letter or digit and use only "
                "letters, digits, '.', '_' and '-'"
            )
        if file.stem.lower() in {name.lower() for name in released}:
            raise ValueError(f"{file.name} has the name of a released themes version; rename it")
        sources = {file.stem: file}
    elif version is not None:
        if version not in released:
            raise ValueError(f"unknown themes version {version!r}; released: {sorted(released)}")
        sources = {version: released[version]}
    else:
        current = next(iter(released))
        sources = {current: released[current]}
    theme_sets = {name: load_themes(path) for name, path in sources.items()}

    known = {"hml", "hml_daily", *(f"hml_{component}" for component, _ in OI_COMPONENTS)}
    hml_names = sorted(
        {p.stem for p in portfolios_dir.glob("hml*") if p.suffix in (".parquet", ".csv")} & known
    )
    if not hml_names:
        raise FileNotFoundError(f"no hml files in {portfolios_dir}; run jkp portfolio first")
    output_format = "parquet" if (portfolios_dir / "hml.parquet").exists() else "csv"
    configure_output_format(output_format)

    hml_series = []
    for hml_name in hml_names:
        suffix = hml_name.removeprefix("hml")
        date_col = "eom" if suffix == "" else "date"
        hml_series.append((suffix, _read_saved(portfolios_dir / hml_name), date_col))
    daily = any(date_col == "date" for _, _, date_col in hml_series)
    _write_theme_versions(
        portfolios_dir=portfolios_dir,
        theme_sets=theme_sets,
        hml_series=hml_series,
        cmp={
            n_groups: _read_saved(portfolios_dir / f"hml_cmp_{n_groups}")
            for n_groups in CMP_SIZE_GROUPS
            if any(
                (portfolios_dir / f"hml_cmp_{n_groups}.{ext}").exists()
                for ext in ("parquet", "csv")
            )
        },
        market=_read_saved(other_output_dir / "market_returns"),
        market_daily=_read_saved(other_output_dir / "market_returns_daily") if daily else None,
        regions=_regions(settings),
        settings=settings,
    )

    if output_format == "csv":
        for name in theme_sets:
            convert_outputs_to_csv(processed_dir=portfolios_dir / "themes" / name)
