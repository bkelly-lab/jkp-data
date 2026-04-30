import os
import time
import warnings
from pathlib import Path

import polars as pl

from .config import (
    COLLECT_CHUNK_SIZE,
    END_DATE,
    PORTFOLIO_BP_MIN_N,
    PORTFOLIO_PFS,
    REGIONAL_COUNTRIES_MIN,
    REGIONAL_COUNTRY_EXCL,
    REGIONAL_MONTHS_MIN,
    REGIONAL_STOCKS_MIN,
)
from .output_writer import (
    configure_output_format,
    convert_outputs_to_csv,
    write_dataframe,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Sortedness.*by.*provided",
)


# =============================================================================
# Helper Functions
# =============================================================================


def add_ecdf(
    df: pl.DataFrame | pl.LazyFrame,
    group_cols: list[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Attach an empirical-CDF ``cdf`` column per group.

    Description:
        Builds the ECDF of the ``var`` column over rows where ``bp_stock``
        is true (the breakpoint sample), then asof-joins the CDF values back
        onto every row of ``df`` — bp and non-bp alike — within each group
        defined by ``group_cols``.

    Steps:
        1) Count ``var`` occurrences per distinct value within each group on
           the bp-stock sub-frame.
        2) Build ``cdf_val`` as a cumulative share within each group.
        3) Asof-join ``(var)`` within ``group_cols`` onto the full input
           frame; non-bp rows pick up the cdf of the nearest bp value ≤ their
           own ``var``, and rows below any bp value fall back to ``0.0``.

    Output:
        Returns the same container type as the input — LazyFrame in, LazyFrame
        out; DataFrame in, DataFrame out — with a ``cdf`` column appended and
        the original columns preserved.
    """
    if group_cols is None:
        group_cols = ["eom"]
    # 1) counts of reference sample per distinct var within each group
    ref_counts = df.filter(pl.col("bp_stock")).group_by(group_cols + ["var"]).agg(n_ref=pl.len())

    # 2) ECDF steps: cumulative share within each group
    ref_steps = (
        ref_counts.sort(group_cols + ["var"])
        .with_columns(
            # apply the window to the whole fraction to ensure same partition
            cdf_val=(pl.cum_sum("n_ref") / pl.sum("n_ref")).over(group_cols)
        )
        .select(group_cols + ["var", "cdf_val"])
    )

    # 3) MUST pre-sort both sides by group_cols + ["var"] for join_asof with 'by'
    left = df.sort(group_cols + ["var"])
    right = ref_steps.sort(group_cols + ["var"])  # already sorted above

    out = (
        left.join_asof(
            right,
            on="var",
            by=group_cols,
            strategy="backward",
        )
        .with_columns(pl.col("cdf_val").fill_null(0.0).alias("cdf"))
        .drop("cdf_val")
    )
    return out


def _build_industry_daily_returns(
    data: pl.DataFrame,
    daily: pl.DataFrame,
    industry_col: str,
    bp_min_n: int,
    excntry: str,
    industry_transform: pl.Expr | None = None,
) -> pl.DataFrame:
    """Description:
        Build daily industry portfolio returns from monthly formation-month weights.
    Steps:
        1) Filter to rows where industry_col is non-null; select id, eom, industry, me, me_cap.
        2) Optionally apply industry_transform to recode the industry column.
        3) Drop industry-month groups with fewer than bp_min_n stocks.
        4) Compute EW/VW/VW-cap weights within each (eom, industry) group.
        5) Join weights to daily returns for the following month.
        6) Aggregate weighted returns by (industry, date).
    Output:
        DataFrame with columns [industry_col, date, n, ret_ew, ret_vw, ret_vw_cap, excntry].
    """
    weights_data = data.filter(pl.col(industry_col).is_not_null()).select(
        ["id", "eom", industry_col, "me", "me_cap"]
    )
    if industry_transform is not None:
        weights_data = weights_data.with_columns(industry_transform)

    weights_data = (
        weights_data.with_columns(pl.len().over([industry_col, "eom"]).alias("bp_n"))
        .filter(pl.col("bp_n") >= bp_min_n)
        .drop("bp_n")
    )

    weights = (
        weights_data.group_by(["eom", industry_col])
        .agg(
            [
                pl.col("id"),
                (1 / pl.len()).alias("w_ew"),
                (pl.col("me") / pl.col("me").sum()).alias("w_vw"),
                (pl.col("me_cap") / pl.col("me_cap").sum()).alias("w_vw_cap"),
            ]
        )
        .explode("id", "w_vw", "w_vw_cap")
    )

    result = (
        weights.lazy()
        .join(
            daily.lazy(),
            left_on=["id", "eom"],
            right_on=["id", "eom_lag1"],
            how="left",
        )
        .filter(pl.col(industry_col).is_not_null() & pl.col("ret_exc").is_not_null())
        .group_by([industry_col, "date"])
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("w_ew") * pl.col("ret_exc")).sum().alias("ret_ew"),
                (pl.col("w_vw") * pl.col("ret_exc")).sum().alias("ret_vw"),
                (pl.col("w_vw_cap") * pl.col("ret_exc")).sum().alias("ret_vw_cap"),
            ]
        )
        .collect()
    )
    return result.with_columns(pl.lit(excntry).str.to_uppercase().alias("excntry"))


def _build_industry_monthly_returns(
    data: pl.DataFrame,
    industry_col: str,
    bp_min_n: int,
    excntry: str,
    industry_transform: pl.Expr | None = None,
) -> pl.DataFrame:
    """Build monthly industry portfolio returns.

    Description:
        Group `data` by `(industry_col, eom)` and compute EW / VW / VW-cap
        returns of `ret_exc_lead1m`. Optionally recode the industry column
        before grouping (e.g. extracting the first 2 GICS digits).
    Steps:
        1) Filter rows where `industry_col` is non-null; select required cols.
        2) Optionally apply `industry_transform`.
        3) Group by `(industry_col, eom)` and aggregate (n, ret_ew, ret_vw,
           ret_vw_cap).
        4) Attach uppercase `excntry`, advance `eom` by 1mo to month-end,
           and drop industry-month groups with fewer than `bp_min_n` stocks.
    Output:
        DataFrame with columns [industry_col, eom, n, ret_ew, ret_vw,
        ret_vw_cap, excntry].
    """
    ind_data = data.filter(pl.col(industry_col).is_not_null()).select(
        ["eom", industry_col, "ret_exc_lead1m", "me", "me_cap"]
    )
    if industry_transform is not None:
        ind_data = ind_data.with_columns(industry_transform)

    return (
        ind_data.group_by([industry_col, "eom"])
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("ret_exc_lead1m").mean()).alias("ret_ew"),
                ((pl.col("ret_exc_lead1m") * pl.col("me")).sum() / pl.col("me").sum()).alias(
                    "ret_vw"
                ),
                (
                    (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                ).alias("ret_vw_cap"),
            ]
        )
        .with_columns(
            pl.lit(excntry).str.to_uppercase().alias("excntry"),
            (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom"),
        )
        .filter(pl.col("n") >= bp_min_n)
    )


def _build_hml_lms(
    pf_df: pl.DataFrame,
    char_info: pl.DataFrame,
    n_pfs: int,
    date_col: str,
    include_signal: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build HML (top minus bottom pf) and signed LMS factor returns.

    Description:
        Group `pf_df` by ``["excntry", "characteristic", date_col]``, compute
        top-minus-bottom-portfolio diffs for return columns (and optionally
        ``signal``), keep only groups containing both extreme portfolios, then
        join `char_info` and re-sign each factor by ``direction``.
    Output:
        ``(hml, lms)`` tuple of DataFrames.
    """
    diff = lambda c: (  # noqa: E731
        pl.col(c).filter(pl.col("pf") == n_pfs).first()
        - pl.col(c).filter(pl.col("pf") == 1).first()
    )
    agg_exprs = [pl.col("pf").is_in([n_pfs, 1]).sum().alias("pfs")]
    if include_signal:
        agg_exprs.append(diff("signal").alias("signal"))
    agg_exprs.extend(
        [
            (
                pl.col("n").filter(pl.col("pf") == n_pfs).first()
                + pl.col("n").filter(pl.col("pf") == 1).first()
            ).alias("n_stocks"),
            pl.col("n").filter(pl.col("pf").is_in([n_pfs, 1])).min().alias("n_stocks_min"),
            diff("ret_ew").alias("ret_ew"),
            diff("ret_vw").alias("ret_vw"),
            diff("ret_vw_cap").alias("ret_vw_cap"),
        ]
    )

    hml = (
        pf_df.group_by(["excntry", "characteristic", date_col])
        .agg(agg_exprs)
        .filter(pl.col("pfs") == 2)
        .drop("pfs")
        .sort(["excntry", "characteristic", date_col])
    )

    resign_cols = (
        ["signal", "ret_ew", "ret_vw", "ret_vw_cap"]
        if include_signal
        else ["ret_ew", "ret_vw", "ret_vw_cap"]
    )
    lms = char_info.join(hml, on="characteristic", how="left").with_columns(
        [(pl.col(c) * pl.col("direction")).alias(c) for c in resign_cols]
    )
    return hml, lms


# main portfolios function to create the portfolios
def portfolios(
    data_path,
    excntry,
    chars,
    pfs,  # Number of portfolios
    bps,  # What should breakpoints be based on? Non-Microcap stocks ("non_mc") or NYSE stocks "nyse"
    bp_min_n,  # Minimum number of stocks used for breakpoints
    nyse_size_cutoffs,  # Data frame with NYSE size breakpoints
    source=None,  # Use data from "CRSP", "Compustat" or both: ["CRSP", "COMPUSTAT"]. Default: both.
    wins_ret=True,  # Should Compustat returns be winsorized at the 0.1% and 99.9% of CRSP returns?
    cmp_key=False,  # Create characteristics managed size portfolios?
    signals=False,  # Create portfolio signals?
    signals_standardize=False,  # Map chars to [-0.5, +0.5]?,
    signals_w="vw_cap",  # Weighting for signals: in c("ew", "vw", "vw_cap")
    daily_pf=False,  # Should daily return be estimated
    ind_pf=True,  # Should industry portfolio returns be estimated
    ret_cutoffs=None,  # Data frame for monthly winsorization. Neccesary when wins_ret=T
    ret_cutoffs_daily=None,  # Data frame for daily winsorization. Neccesary when wins_ret=T and daily_pf=T
):
    if source is None:
        source = ["CRSP", "COMPUSTAT"]
    # characerteristics data
    file_path = f"{data_path}/characteristics/{excntry}.parquet"

    # Select the required columns
    columns = (
        [
            "id",
            "eom",
            "source_crsp",
            "comp_exchg",
            "crsp_exchcd",
            "size_grp",
            "ret_exc",
            "ret_exc_lead1m",
            "me",
            "gics",
            "ff49",
        ]
        + chars
        + ["excntry"]
    )

    # Build the full preprocessing chain as a single lazy pipeline. This lets
    # polars push predicate/null filters into the parquet reader (skipping row
    # groups on real production files) and fuse the ~15 intermediate steps into
    # one pass instead of allocating ~15 intermediate DataFrames.
    cast_exclude = {"id", "eom", "source_crsp", "size_grp", "excntry"}
    cast_cols = [c for c in columns if c not in cast_exclude]

    if bps == "nyse":
        bp_stock_expr = (
            ((pl.col("crsp_exchcd") == 1) & pl.col("comp_exchg").is_null())
            | ((pl.col("comp_exchg") == 11) & pl.col("crsp_exchcd").is_null())
        ).alias("bp_stock")
    else:  # "non_mc"
        bp_stock_expr = pl.col("size_grp").is_in(["mega", "large", "small"]).alias("bp_stock")

    data_lazy = (
        pl.scan_parquet(file_path)
        .select(columns)
        .join(nyse_size_cutoffs.lazy().select(["eom", "nyse_p80"]), on="eom", how="left")
        .with_columns(pl.min_horizontal(pl.col("me"), pl.col("nyse_p80")).alias("me_cap"))
        .drop("nyse_p80")
        .with_columns([pl.col(c).cast(pl.Float64) for c in cast_cols])
        .filter(
            (pl.col("size_grp").is_not_null())
            & (pl.col("me").is_not_null())
            & (pl.col("ret_exc_lead1m").is_not_null())
        )
        .with_columns(bp_stock_expr)
    )

    # Conditional source screen
    if len(source) == 1:
        if source[0] == "CRSP":
            data_lazy = data_lazy.filter(pl.col("source_crsp") == 1)
        elif source[0] == "COMPUSTAT":
            data_lazy = data_lazy.filter(pl.col("source_crsp") == 0)

    # Daily returns as a lazy chain (scan_parquet + winsorization).
    daily_file_path = f"{data_path}/return_data/daily_rets_by_country/{excntry}.parquet"
    if daily_pf:
        daily_lazy = (
            pl.scan_parquet(daily_file_path)
            .select(["id", "date", "ret_exc"])
            .with_columns((pl.col("date").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1"))
            .with_columns(pl.col("ret_exc").cast(pl.Float64))
        )
    else:
        daily_lazy = None

    # Monthly winsorization: clip Compustat ret_exc_lead1m to CRSP quantiles.
    if wins_ret:
        data_lazy = (
            data_lazy.join(
                ret_cutoffs.lazy()
                .select(["eom_lag1", "ret_exc_0_1", "ret_exc_99_9"])
                .rename({"eom_lag1": "eom", "ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}),
                on="eom",
                how="left",
            )
            .with_columns(
                pl.when((pl.col("source_crsp") == 0) & (pl.col("ret_exc_lead1m") > pl.col("p999")))
                .then(pl.col("p999"))
                .when((pl.col("source_crsp") == 0) & (pl.col("ret_exc_lead1m") < pl.col("p001")))
                .then(pl.col("p001"))
                .otherwise(pl.col("ret_exc_lead1m"))
                .alias("ret_exc_lead1m")
            )
            .drop(["source_crsp", "p001", "p999"])
        )

        # Daily winsorization
        if daily_pf:
            daily_lazy = (
                daily_lazy.with_columns(pl.col("date").dt.month_end().alias("eom"))
                .join(
                    ret_cutoffs_daily.lazy()
                    .select(["eom", "ret_exc_0_1", "ret_exc_99_9"])
                    .rename({"ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}),
                    on="eom",
                    how="left",
                )
                .with_columns(
                    pl.when((pl.col("id") > 99999) & (pl.col("ret_exc") > pl.col("p999")))
                    .then(pl.col("p999"))
                    .when((pl.col("id") > 99999) & (pl.col("ret_exc") < pl.col("p001")))
                    .then(pl.col("p001"))
                    .otherwise(pl.col("ret_exc"))
                    .alias("ret_exc")
                )
                .drop(["p001", "p999", "eom"])
            )

    # Collect the lazy chain once — the single fused read + preprocess pass.
    data = data_lazy.collect()

    # Collect daily lazy frame if present (needed for industry returns and per-char daily).
    daily = daily_lazy.collect() if daily_lazy is not None else None

    # standardizing signals
    if signals_standardize and signals:
        data = (
            data
            # Ranking within groups defined by 'eom'
            .with_columns(
                [
                    (pl.col(char).rank(method="min").over("eom").cast(pl.Int64)).alias(char)
                    for char in chars
                ]
            )
            # normalizing ranks
            .with_columns(
                [
                    (((pl.col(char) / pl.col(char).max()) - pl.lit(0.5)).over("eom")).alias(char)
                    for char in chars
                ]
            )
        )

    if ind_pf:
        ind_gics = _build_industry_monthly_returns(
            data,
            "gics",
            bp_min_n,
            excntry,
            industry_transform=(
                pl.col("gics").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int64).alias("gics")
            ),
        )

        # Estimate industry portfolios by Fama-French portfolios for US data
        if excntry.lower() == "usa":
            ind_ff49 = _build_industry_monthly_returns(data, "ff49", bp_min_n, excntry)

        if daily_pf:
            ind_gics_daily = _build_industry_daily_returns(
                data,
                daily,
                "gics",
                bp_min_n,
                excntry,
                industry_transform=pl.col("gics")
                .cast(pl.Utf8)
                .str.slice(0, 2)
                .cast(pl.Int64)
                .alias("gics"),
            )

            if excntry.lower() == "usa":
                ind_ff49_daily = _build_industry_daily_returns(
                    data,
                    daily,
                    "ff49",
                    bp_min_n,
                    excntry,
                )

    # Creating portfolios for all the characteristics.
    #
    # Each characteristic builds a per-char lazy pipeline (ECDF -> pf
    # assignment -> monthly/daily aggregation).  The pipelines accumulate in
    # ``pf_returns_lazys`` / ``pf_daily_lazys`` and are batch-collected via
    # ``pl.collect_all`` so polars can execute them concurrently on its thread
    # pool.  The signals=True branch is the exception: it needs the
    # intermediate ``sub`` frame alive after each collect to compute
    # pf_signals, so it collects eagerly per-char.
    pf_returns_lazys: list[pl.LazyFrame] = []
    pf_daily_lazys: list[pl.LazyFrame] = []
    char_pfs: list[dict] = []  # populated only when signals=True

    # Single shared LazyFrame node for the preprocessed data. All per-char
    # pipelines branch off this node, enabling polars' common-subplan
    # elimination (CSE) in collect_all to avoid redundant scans.
    data_lazy = data.lazy()

    for x in chars:
        # Alias current char into a 'var' column on the per-char subset.
        # Operate on `sub` only -- `data` is not mutated.
        if not signals:
            sub = (
                data_lazy.with_columns(pl.col(x).cast(pl.Float64).alias("var"))
                .filter(pl.col("var").is_not_null())
                .select(
                    [
                        "id",
                        "eom",
                        "var",
                        "size_grp",
                        "ret_exc_lead1m",
                        "me",
                        "me_cap",
                        "bp_stock",
                    ]
                )
            )
        else:
            sub = data_lazy.with_columns(pl.col(x).cast(pl.Float64).alias("var")).filter(
                pl.col("var").is_not_null()
            )

        sub = sub.with_columns(bp_n=pl.sum("bp_stock").over("eom")).filter(
            pl.col("bp_n") >= bp_min_n
        )

        # Skip chars with no data after the bp_n filter.  For the
        # collect_all path we cannot cheaply check emptiness without an
        # eager collect, so we optimistically build the lazy pipeline and
        # drop empty results after the batch collect.  For signals=True
        # we still need the eager gate because the pf_signals block
        # requires intermediate access to ``sub``.
        if signals and sub.limit(1).collect().height == 0:
            continue

        sub = (
            add_ecdf(sub)
            .with_columns(
                pl.when(pl.col("cdf") == pl.col("cdf").min().over("eom"))
                .then(0.00000001)
                .otherwise(pl.col("cdf"))
                .alias("cdf")
            )
            .with_columns(
                (pl.col("cdf") * pfs).ceil().clip(lower_bound=1, upper_bound=pfs).alias("pf")
            )
        )

        # Monthly pf_returns lazy frame for this char.
        pf_returns_x = (
            sub.group_by(["pf", "eom"])
            .agg(
                [
                    pl.lit(x).alias("characteristic"),
                    pl.len().alias("n"),
                    pl.median("var").alias("signal"),
                    pl.mean("ret_exc_lead1m").alias("ret_ew"),
                    ((pl.col("ret_exc_lead1m") * pl.col("me")).sum() / pl.col("me").sum()).alias(
                        "ret_vw"
                    ),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                    ).alias("ret_vw_cap"),
                ]
            )
            .with_columns(pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom"))
        )

        if signals:
            # Eager path: collect immediately, then compute pf_signals.
            op: dict = {}
            op["pf_returns"] = pf_returns_x.collect()

            if signals_w == "ew":
                sub = sub.with_columns((1 / pl.col("eom").len()).over(["pf", "eom"]).alias("w"))
            elif signals_w == "vw":
                sub = sub.with_columns(
                    (pl.col("me") / pl.col("me").sum()).over(["pf", "eom"]).alias("w")
                )
            elif signals_w == "vw_cap":
                sub = sub.with_columns(
                    (pl.col("me_cap") / pl.col("me_cap").sum()).over(["pf", "eom"]).alias("w")
                )

            sub = sub.with_columns(
                [
                    pl.when(pl.col(var).is_null()).then(pl.lit(0)).otherwise(pl.col(var)).alias(var)
                    for var in chars
                ]
            )
            pf_signals = sub.with_columns(
                [(pl.col("w") * pl.col(var)).sum().over(["pf", "eom"]) for var in chars]
            )
            pf_signals = pf_signals.with_columns(
                [
                    pl.lit(x).alias("characteristic"),
                    pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom"),
                ]
            )
            op["signals"] = pf_signals.collect()

            if daily_pf:
                weights = (
                    sub.group_by(["eom", "pf"])
                    .agg(
                        [
                            pl.col("id"),
                            (1 / pl.len()).alias("w_ew"),
                            (pl.col("me") / pl.col("me").sum()).alias("w_vw"),
                            (pl.col("me_cap") / pl.col("me_cap").sum()).alias("w_vw_cap"),
                        ]
                    )
                    .explode("id", "w_vw", "w_vw_cap")
                )
                daily_sub = weights.join(
                    daily.lazy(),
                    left_on=["id", "eom"],
                    right_on=["id", "eom_lag1"],
                    how="left",
                ).filter((pl.col("pf").is_not_null()) & (pl.col("ret_exc").is_not_null()))
                pf_daily_x = daily_sub.group_by(["pf", "date"]).agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n"),
                        ((pl.col("w_ew") * pl.col("ret_exc")).sum()).alias("ret_ew"),
                        ((pl.col("w_vw") * pl.col("ret_exc")).sum()).alias("ret_vw"),
                        ((pl.col("w_vw_cap") * pl.col("ret_exc")).sum()).alias("ret_vw_cap"),
                    ]
                )
                op["pf_daily"] = pf_daily_x.collect()

            char_pfs.append(op)
        else:
            # Lazy path: accumulate LazyFrames for batch collect_all.
            pf_returns_lazys.append(pf_returns_x)

            if daily_pf:
                weights = (
                    sub.group_by(["eom", "pf"])
                    .agg(
                        [
                            pl.col("id"),
                            (1 / pl.len()).alias("w_ew"),
                            (pl.col("me") / pl.col("me").sum()).alias("w_vw"),
                            (pl.col("me_cap") / pl.col("me_cap").sum()).alias("w_vw_cap"),
                        ]
                    )
                    .explode("id", "w_vw", "w_vw_cap")
                )
                daily_sub = weights.join(
                    daily.lazy(),
                    left_on=["id", "eom"],
                    right_on=["id", "eom_lag1"],
                    how="left",
                ).filter((pl.col("pf").is_not_null()) & (pl.col("ret_exc").is_not_null()))
                pf_daily_x = daily_sub.group_by(["pf", "date"]).agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n"),
                        ((pl.col("w_ew") * pl.col("ret_exc")).sum()).alias("ret_ew"),
                        ((pl.col("w_vw") * pl.col("ret_exc")).sum()).alias("ret_vw"),
                        ((pl.col("w_vw_cap") * pl.col("ret_exc")).sum()).alias("ret_vw_cap"),
                    ]
                )
                pf_daily_lazys.append(pf_daily_x)

    # Batch-collect per-char lazy pipelines in chunks to bound peak memory
    # (see COLLECT_CHUNK_SIZE in config.py).
    pf_returns_df: pl.DataFrame | None = None
    pf_daily_df: pl.DataFrame | None = None

    if signals and char_pfs:
        pf_returns_df = pl.concat([op["pf_returns"] for op in char_pfs])
        if daily_pf:
            pf_daily_df = pl.concat([op["pf_daily"] for op in char_pfs])
    elif pf_returns_lazys:
        ret_dfs: list[pl.DataFrame] = []
        daily_dfs: list[pl.DataFrame] = []
        n_chars = len(pf_returns_lazys)
        n_chunks = (n_chars + COLLECT_CHUNK_SIZE - 1) // COLLECT_CHUNK_SIZE
        for chunk_idx, start in enumerate(range(0, n_chars, COLLECT_CHUNK_SIZE)):
            end = min(start + COLLECT_CHUNK_SIZE, n_chars)
            print(
                f"   Chars {start + 1}-{end} of {n_chars} (chunk {chunk_idx + 1}/{n_chunks})",
                flush=True,
            )
            chunk_ret = pf_returns_lazys[start:end]
            chunk_daily = pf_daily_lazys[start:end] if daily_pf else []
            collected = pl.collect_all(chunk_ret + chunk_daily)
            n_ret_chunk = len(chunk_ret)
            ret_dfs.extend(df for df in collected[:n_ret_chunk] if df.height > 0)
            if daily_pf:
                daily_dfs.extend(df for df in collected[n_ret_chunk:] if df.height > 0)
        pf_returns_df = pl.concat(ret_dfs) if ret_dfs else None
        if daily_pf:
            pf_daily_df = pl.concat(daily_dfs) if daily_dfs else None

    output = {}
    if pf_returns_df is not None:
        output["pf_returns"] = pf_returns_df
    if daily_pf and pf_daily_df is not None:
        output["pf_daily"] = pf_daily_df
    # Handle industry portfolio returns if ind_pf is true
    if ind_pf:
        output["gics_returns"] = ind_gics
        if excntry.lower() == "usa":
            output["ff49_returns"] = ind_ff49
        if daily_pf:
            output["gics_daily"] = ind_gics_daily
            if excntry.lower() == "usa":
                output["ff49_daily"] = ind_ff49_daily

    # Add excntry to pf_returns and pf_daily, and aggregate signals
    if len(output) > 0:
        if "pf_returns" in output and output["pf_returns"].height > 0:
            output["pf_returns"] = output["pf_returns"].with_columns(
                pl.lit(excntry).str.to_uppercase().alias("excntry")
            )
            if daily_pf and "pf_daily" in output:
                output["pf_daily"] = output["pf_daily"].with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )
            if signals and "signals" in output:
                output["signals"] = pl.concat([op["signals"] for op in char_pfs]).with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )

    results = []
    if cmp_key:
        for x in chars:
            print(f"   CMP - {x}: {chars.index(x) + 1} out of {len(chars)}")

            # Create a new column 'var' based on the current 'x'
            data = data.with_columns(pl.col(x).alias("var"))

            # Subsetting and ranking
            sub = data.filter(pl.col("var").is_not_null()).select(
                ["eom", "var", "size_grp", "ret_exc_lead1m"]
            )

            # Calculate ranks, rank deviations, and weights
            sub = (
                sub.with_columns(
                    (
                        (pl.col("var").rank("average").over("size_grp", "eom"))
                        / (pl.len().over("size_grp", "eom") + 1)
                    ).alias("p_rank")
                )
                .with_columns(pl.col("p_rank").mean().over("size_grp", "eom").alias("mean_p_rank"))
                .with_columns((pl.col("p_rank") - pl.col("mean_p_rank")).alias("p_rank_dev"))
                .with_columns(
                    (pl.col("p_rank_dev") / ((pl.col("p_rank_dev").abs().sum()) / 2))
                    .over("size_grp", "eom")
                    .alias("weight")
                )
            )

            # Aggregation
            cmp = (
                sub.group_by(["size_grp", "eom"])
                .agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n_stocks"),
                        ((pl.col("ret_exc_lead1m") * pl.col("weight")).sum()).alias("ret_weighted"),
                        ((pl.col("var") * pl.col("weight")).sum()).alias("signal_weighted"),
                        pl.col("var").std().alias("sd_var"),
                    ]
                )
                .with_columns(pl.lit(excntry).alias("excntry"))
            )

            # Post-processing
            cmp = (
                cmp.filter(pl.col("sd_var") != 0)
                .drop("sd_var")
                .with_columns((pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom"))
            )

            results.append(cmp)

    if len(results) > 0:
        output["cmp"] = pl.concat(results).with_columns(
            pl.col("excntry").str.to_uppercase().alias("excntry")
        )

    return output


# function for regional grouping of portfolios etc
def regional_data(
    data,
    mkt,
    date_col,
    char_col,
    countries,
    weighting,
    countries_min,
    periods_min,
    stocks_min,
):
    # Determine Country Weights
    weights = mkt.select(
        [
            pl.col("excntry"),
            pl.col(date_col).alias(date_col),
            pl.col("mkt_vw_exc"),
            pl.when(weighting == "market_cap")
            .then(pl.col("me_lag1"))
            .when(weighting == "stocks")
            .then(pl.col("stocks").cast(pl.Float64))
            .when(weighting == "ew")
            .then(1)
            .alias("country_weight"),
        ]
    )
    # Portfolio Return
    pf = data.filter(
        (pl.col("excntry").is_in(countries.implode())) & (pl.col("n_stocks_min") >= stocks_min)
    )
    pf = pf.join(weights, on=["excntry", date_col], how="left")
    pf = (
        pf.filter(pl.col("mkt_vw_exc").is_not_null())
        .group_by([char_col, date_col])
        .agg(
            [
                pl.len().alias("n_countries"),
                pl.col("direction").first().alias("direction"),
                (pl.col("ret_ew") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_ew"),
                (pl.col("ret_vw") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_vw"),
                (pl.col("ret_vw_cap") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_vw_cap"),
                (pl.col("mkt_vw_exc") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("mkt_vw_exc"),
            ]
        )
    )

    # Minimum Requirement: Countries
    pf = pf.filter(pl.col("n_countries") >= countries_min)

    # Minimum Requirement: Months
    pf = (
        pf.with_columns(pl.len().over(char_col).alias("periods"))
        .filter(pl.col("periods") >= periods_min)
        .drop("periods")
        .sort([char_col, date_col])
    )

    return pf


def _build_regional_loop(
    data: pl.DataFrame,
    mkt: pl.DataFrame,
    regions: pl.DataFrame,
    date_col: str,
    char_col: str,
    output_cols: list[str],
    weighting: str,
    periods_min: int,
    stocks_min: int,
) -> pl.DataFrame:
    """Aggregate `data` across all regions and concatenate.

    Description:
        Iterate over `regions`, call `regional_data` for each region's country
        set, attach a `region` literal column, and project `output_cols`.
    Output:
        Concatenated DataFrame with one block of rows per region.
    """
    pieces = []
    for i in range(regions.height):
        info = regions[i][0]
        reg_pf = (
            regional_data(
                data=data,
                mkt=mkt,
                countries=info["country_codes"][0],
                date_col=date_col,
                char_col=char_col,
                weighting=weighting,
                countries_min=info["countries_min"][0],
                periods_min=periods_min,
                stocks_min=stocks_min,
            )
            .with_columns(pl.lit(info["name"][0]).alias("region"))
            .select(output_cols)
        )
        pieces.append(reg_pf)
    return pl.concat(pieces)


def _write_filtered(
    df: pl.DataFrame,
    path: str,
    date_col: str,
    end_date,
) -> None:
    """Filter `df` to rows where `date_col <= end_date` and write to `path`."""
    write_dataframe(df.filter(pl.col(date_col) <= end_date), path)


def _write_split_by_key(
    df: pl.DataFrame,
    folder_path: str,
    key_col: str,
    date_col: str,
    end_date,
) -> None:
    """Partition `df` by `key_col` and write one parquet per key into `folder_path`.

    Description:
        Apply the `date_col <= end_date` filter, then for each unique non-null
        truthy value of `key_col`, write the matching rows to
        ``{folder_path}/{key}.parquet``.
    """
    os.makedirs(folder_path, exist_ok=True)
    for key in df[key_col].unique():
        if not key:
            continue
        filtered = df.filter((pl.col(date_col) <= end_date) & (pl.col(key_col) == key))
        write_dataframe(filtered, os.path.join(folder_path, f"{key}.parquet"))


# =============================================================================
# Main Entry Point
# =============================================================================


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
    configure_output_format(output_format)

    # Setting data path and output path
    data_path = str(output_dir / "processed")
    output_path = str(output_dir / "processed" / "portfolios")

    # Get list of countries from characteristics files
    countries = []
    for file in os.listdir(os.path.join(data_path, "characteristics")):
        if file.endswith(".parquet") and "world" not in file:
            countries.append(file.replace(".parquet", ""))
    countries = sorted(countries)

    # Characteristics to process
    chars = [
        "age",
        "aliq_at",
        "aliq_mat",
        "ami_126d",
        "at_be",
        "at_gr1",
        "at_me",
        "at_turnover",
        "be_gr1a",
        "be_me",
        "beta_60m",
        "beta_dimson_21d",
        "betabab_1260d",
        "betadown_252d",
        "bev_mev",
        "bidaskhl_21d",
        "capex_abn",
        "capx_gr1",
        "capx_gr2",
        "capx_gr3",
        "cash_at",
        "chcsho_12m",
        "coa_gr1a",
        "col_gr1a",
        "cop_at",
        "cop_atl1",
        "corr_1260d",
        "coskew_21d",
        "cowc_gr1a",
        "dbnetis_at",
        "debt_gr3",
        "debt_me",
        "dgp_dsale",
        "div12m_me",
        "dolvol_126d",
        "dolvol_var_126d",
        "dsale_dinv",
        "dsale_drec",
        "dsale_dsga",
        "earnings_variability",
        "ebit_bev",
        "ebit_sale",
        "ebitda_mev",
        "emp_gr1",
        "eq_dur",
        "eqnetis_at",
        "eqnpo_12m",
        "eqnpo_me",
        "eqpo_me",
        "f_score",
        "fcf_me",
        "fnl_gr1a",
        "gp_at",
        "gp_atl1",
        "ival_me",
        "inv_gr1",
        "inv_gr1a",
        "iskew_capm_21d",
        "iskew_ff3_21d",
        "iskew_hxz4_21d",
        "ivol_capm_21d",
        "ivol_capm_252d",
        "ivol_ff3_21d",
        "ivol_hxz4_21d",
        "kz_index",
        "lnoa_gr1a",
        "lti_gr1a",
        "market_equity",
        "mispricing_mgmt",
        "mispricing_perf",
        "ncoa_gr1a",
        "ncol_gr1a",
        "netdebt_me",
        "netis_at",
        "nfna_gr1a",
        "ni_ar1",
        "ni_be",
        "ni_inc8q",
        "ni_ivol",
        "ni_me",
        "niq_at",
        "niq_at_chg1",
        "niq_be",
        "niq_be_chg1",
        "niq_su",
        "nncoa_gr1a",
        "noa_at",
        "noa_gr1a",
        "o_score",
        "oaccruals_at",
        "oaccruals_ni",
        "ocf_at",
        "ocf_at_chg1",
        "ocf_me",
        "ocfq_saleq_std",
        "op_at",
        "op_atl1",
        "ope_be",
        "ope_bel1",
        "opex_at",
        "pi_nix",
        "ppeinv_gr1a",
        "prc",
        "prc_highprc_252d",
        "qmj",
        "qmj_growth",
        "qmj_prof",
        "qmj_safety",
        "rd_me",
        "rd_sale",
        "rd5_at",
        "resff3_12_1",
        "resff3_6_1",
        "ret_1_0",
        "ret_12_1",
        "ret_12_7",
        "ret_3_1",
        "ret_6_1",
        "ret_60_12",
        "ret_9_1",
        "rmax1_21d",
        "rmax5_21d",
        "rmax5_rvol_21d",
        "rskew_21d",
        "rvol_21d",
        "sale_bev",
        "sale_emp_gr1",
        "sale_gr1",
        "sale_gr3",
        "sale_me",
        "saleq_gr1",
        "saleq_su",
        "seas_1_1an",
        "seas_1_1na",
        "seas_11_15an",
        "seas_11_15na",
        "seas_16_20an",
        "seas_16_20na",
        "seas_2_5an",
        "seas_2_5na",
        "seas_6_10an",
        "seas_6_10na",
        "sti_gr1a",
        "taccruals_at",
        "taccruals_ni",
        "tangibility",
        "tax_gr1a",
        "turnover_126d",
        "turnover_var_126d",
        "z_score",
        "zero_trades_126d",
        "zero_trades_21d",
        "zero_trades_252d",
    ]

    # Portfolio construction settings
    settings = {
        "end_date": END_DATE,
        "pfs": PORTFOLIO_PFS,
        "source": ["CRSP", "COMPUSTAT"],
        "wins_ret": True,
        "bps": "non_mc",
        "bp_min_n": PORTFOLIO_BP_MIN_N,
        "cmp": {"us": True, "int": False},
        "signals": {"us": False, "int": False, "standardize": True, "weight": "vw_cap"},
        "regional_pfs": {
            "ret_type": "vw_cap",
            "country_excl": list(REGIONAL_COUNTRY_EXCL),
            "country_weights": "market_cap",
            "stocks_min": REGIONAL_STOCKS_MIN,
            "months_min": REGIONAL_MONTHS_MIN,
            "countries_min": REGIONAL_COUNTRIES_MIN,
        },
        "daily_pf": True,
        "ind_pf": True,
    }

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
    nyse_size_cutoffs = pl.read_parquet(f"{data_path}/other_output/nyse_cutoffs.parquet")

    # return_cutoffs
    ret_cutoffs = pl.read_parquet(f"{data_path}/other_output/return_cutoffs.parquet")
    ret_cutoffs = ret_cutoffs.with_columns(
        (pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1")
    )
    if settings["daily_pf"]:
        ret_cutoffs_daily = pl.read_parquet(
            f"{data_path}/other_output/return_cutoffs_daily.parquet"
        )

    # market_returns
    market = pl.read_parquet(f"{data_path}/other_output/market_returns.parquet")

    # daily_market_returns
    if settings["daily_pf"]:
        market_daily = pl.read_parquet(f"{data_path}/other_output/market_returns_daily.parquet")

    # Creating portfolios by using the portfolios function
    portfolio_data = {}
    for ex in countries:
        print(f"{ex}: {countries.index(ex) + 1} out of {len(countries)}")
        result = portfolios(
            data_path=data_path,
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

    # Aggregating portfolio returns
    if any(sub_data and "pf_returns" in sub_data for _, sub_data in portfolio_data.items()):
        pf_returns = pl.concat(
            [
                sub_data["pf_returns"]
                for _, sub_data in portfolio_data.items()
                if sub_data and "pf_returns" in sub_data
            ]
        )
        pf_returns = pf_returns.select(
            [
                "excntry",
                "characteristic",
                "pf",
                "eom",
                "n",
                "signal",
                "ret_ew",
                "ret_vw",
                "ret_vw_cap",
            ]
        )
        pf_returns = pf_returns.sort(["excntry", "characteristic", "pf", "eom"])
    else:
        pf_returns = None

    if settings["daily_pf"] and any(
        sub_data and "pf_daily" in sub_data for _, sub_data in portfolio_data.items()
    ):
        pf_daily = pl.concat(
            [
                sub_data["pf_daily"]
                for _, sub_data in portfolio_data.items()
                if sub_data and "pf_daily" in sub_data
            ]
        )
        pf_daily = pf_daily.sort(["excntry", "characteristic", "pf", "date"])
    else:
        pf_daily = None

    # Aggregating industry classification returns
    # GICS Returns
    if settings["ind_pf"] and any(
        sub_data and "gics_returns" in sub_data for _, sub_data in portfolio_data.items()
    ):
        gics_returns = pl.concat(
            [
                sub_data["gics_returns"]
                for _, sub_data in portfolio_data.items()
                if sub_data and "gics_returns" in sub_data
            ]
        )
        gics_returns = gics_returns.sort(["excntry", "gics", "eom"])
    else:
        gics_returns = None

    # FF49 Returns
    if settings["ind_pf"] and any(
        sub_data and "ff49_returns" in sub_data for _, sub_data in portfolio_data.items()
    ):
        ff49_returns = pl.concat(
            [
                sub_data["ff49_returns"]
                for _, sub_data in portfolio_data.items()
                if sub_data and "ff49_returns" in sub_data
            ]
        )
        ff49_returns = ff49_returns.sort(["excntry", "ff49", "eom"])
    else:
        ff49_returns = None

    # Aggregating daily industry classification returns
    if (
        settings["ind_pf"]
        and settings["daily_pf"]
        and any(sub_data and "gics_daily" in sub_data for _, sub_data in portfolio_data.items())
    ):
        gics_daily = pl.concat(
            [
                sub_data["gics_daily"]
                for _, sub_data in portfolio_data.items()
                if sub_data and "gics_daily" in sub_data
            ]
        )
        gics_daily = gics_daily.sort(["excntry", "gics", "date"])
    else:
        gics_daily = None

    if (
        settings["ind_pf"]
        and settings["daily_pf"]
        and any(sub_data and "ff49_daily" in sub_data for _, sub_data in portfolio_data.items())
    ):
        ff49_daily = pl.concat(
            [
                sub_data["ff49_daily"]
                for _, sub_data in portfolio_data.items()
                if sub_data and "ff49_daily" in sub_data
            ]
        )
        ff49_daily = ff49_daily.sort(["excntry", "ff49", "date"])
    else:
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
    cmp_list = [
        portfolio_data[sub_dict]["cmp"]
        for sub_dict in portfolio_data
        if "cmp" in portfolio_data[sub_dict]
    ]
    if cmp_list:
        cmp_returns = pl.concat(cmp_list)
    else:
        # Handle the empty list case here
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
            _write_filtered(df, f"{output_path}/{name}", "eom", end_date)
    if cmp_list:
        _write_filtered(cmp_returns, f"{output_path}/cmp.parquet", "eom", end_date)

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
                _write_filtered(df, f"{output_path}/{name}", "date", end_date)

    # Industry returns
    if settings["ind_pf"]:
        ind_monthly = [
            (gics_returns, "industry_gics.parquet"),
            (ff49_returns, "industry_ff49.parquet"),
        ]
        for df, name in ind_monthly:
            if df is not None:
                _write_filtered(df, f"{output_path}/{name}", "eom", end_date)

    if settings["ind_pf"] and settings["daily_pf"]:
        ind_daily = [
            (gics_daily, "industry_gics_daily.parquet"),
            (ff49_daily, "industry_ff49_daily.parquet"),
        ]
        for df, name in ind_daily:
            if df is not None:
                _write_filtered(df, f"{output_path}/{name}", "date", end_date)

    # Partitioned outputs
    if regional_pfs is not None:
        _write_split_by_key(
            regional_pfs, os.path.join(output_path, "regional_factors"), "region", "eom", end_date
        )
    if settings["daily_pf"] and regional_pfs_daily is not None:
        _write_split_by_key(
            regional_pfs_daily,
            os.path.join(output_path, "regional_factors_daily"),
            "region",
            "date",
            end_date,
        )
    if regional_clusters is not None:
        _write_split_by_key(
            regional_clusters,
            os.path.join(output_path, "regional_clusters"),
            "region",
            "eom",
            end_date,
        )
    if settings["daily_pf"] and regional_clusters_daily is not None:
        _write_split_by_key(
            regional_clusters_daily,
            os.path.join(output_path, "regional_clusters_daily"),
            "region",
            "date",
            end_date,
        )
    if lms_returns is not None:
        _write_split_by_key(
            lms_returns,
            os.path.join(output_path, "country_factors"),
            "excntry",
            "eom",
            end_date,
        )
    if settings["daily_pf"] and lms_daily is not None:
        _write_split_by_key(
            lms_daily,
            os.path.join(output_path, "country_factors_daily"),
            "excntry",
            "date",
            end_date,
        )

    # Convert to CSV if configured
    convert_outputs_to_csv(processed_dir=data_path)

    print(
        f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )
