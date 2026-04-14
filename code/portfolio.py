import argparse
import os
import time
import warnings

import polars as pl
from config import END_DATE
from output_writer import (
    VALID_OUTPUT_FORMATS,
    configure_output_format,
    convert_outputs_to_csv,
    write_dataframe,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for portfolio generation.

    Description:
        Parse CLI arguments for output format selection.
    Steps:
        1) Create argument parser with output-format option.
        2) Parse and return the arguments.
    Output:
        argparse.Namespace with output_format attribute.
    """
    parser = argparse.ArgumentParser(
        description="Generate JKP portfolio data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python code/portfolio.py                      # Default: parquet output
  uv run python code/portfolio.py --output-format csv  # CSV output with quoted strings
        """,
    )
    parser.add_argument(
        "--output-format",
        choices=VALID_OUTPUT_FORMATS,
        default="parquet",
        help="Output file format (default: parquet)",
    )
    return parser.parse_args()


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Sortedness.*by.*provided",
)

# Larger streaming chunks than polars' default give substantially better
# throughput on the production 450 GB HPC box where memory is abundant. This
# is a module-level tuning knob — harmless on smaller machines (polars still
# bounds memory per chunk) and free for eager queries that don't stream.
pl.Config.set_streaming_chunk_size(1_000_000)

# setting data path and output path
data_path = "data/processed"
output_path = "data/processed/portfolios"

# characteristics
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

# a dictionary which has the parameters for constructing portfolios.
settings = {
    "end_date": END_DATE,
    "pfs": 3,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "bps": "non_mc",
    "bp_min_n": 10,
    "cmp": {"us": True, "int": False},
    "signals": {"us": False, "int": False, "standardize": True, "weight": "vw_cap"},
    "regional_pfs": {
        "ret_type": "vw_cap",
        "country_excl": ["ZWE", "VEN"],
        "country_weights": "market_cap",
        "stocks_min": 5,
        "months_min": 5 * 12,
        "countries_min": 3,
    },
    "daily_pf": True,
    "ind_pf": True,
}


def add_ecdf(
    df: pl.DataFrame | pl.LazyFrame,
    group_cols: list[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Attach an empirical-CDF ``cdf`` column per group.

    Description:
        Builds the ECDF of the ``var`` column over rows where ``bp_stock``
        is true (the breakpoint sample), then asof-joins the CDF values back
        onto every row of ``df`` — bp and non-bp alike — within each group
        defined by ``group_cols``. For the upcoming vectorized path, callers
        pass ``group_cols=["characteristic", "eom"]`` so every
        ``(characteristic, eom)`` partition gets its own ECDF in a single
        pass; the legacy per-char path passes the default ``["eom"]``.

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
    # `data` is used by industry/cmp/signals sections that need eager access.
    data = data_lazy.collect()

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
        # Filter data where 'gics' is not null and select required columns
        ind_data = data.filter(
            pl.col("gics").is_not_null()
        ).select(
            ["eom", "gics", "excntry", "ret_exc_lead1m", "me", "me_cap"]
        )  # original code didn;t select 'excntry' in the above steps in the data table, updating that

        # Process GICS codes (extract first 2 digits and convert to numeric)
        ind_data = ind_data.with_columns(
            (pl.col("gics").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int64)).alias("gics")
        )

        # Calculate industry returns based on GICS
        ind_gics = ind_data.group_by(["gics", "eom"]).agg(
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
        ind_gics = ind_gics.with_columns(pl.lit(excntry).str.to_uppercase().alias("excntry"))
        ind_gics = ind_gics.with_columns(
            (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom")
        )
        ind_gics = ind_gics.filter(pl.col("n") >= bp_min_n)

        # Estimate industry portfolios by Fama-French portfolios for US data
        if excntry.lower() == "usa":
            ind_data = data.filter(pl.col("ff49").is_not_null()).select(
                ["eom", "ff49", "ret_exc_lead1m", "me", "me_cap"]
            )
            ind_ff49 = ind_data.group_by(["ff49", "eom"]).agg(
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
            ind_ff49 = ind_ff49.with_columns(pl.lit(excntry).str.to_uppercase().alias("excntry"))
            ind_ff49 = ind_ff49.with_columns(
                (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom")
            )
            ind_ff49 = ind_ff49.filter(pl.col("n") >= bp_min_n)

        if daily_pf:
            # Daily industry returns: weights are formed from monthly end-of-
            # formation-month `me` and applied to every trading day of the
            # following month (rebalanced daily back to beginning-of-month
            # weights). Mirrors the char factor daily logic below and keeps
            # coverage aligned with the monthly industry output.
            gics_weights_data = (
                data.filter(pl.col("gics").is_not_null())
                .select(["id", "eom", "gics", "me", "me_cap"])
                .with_columns(
                    pl.col("gics").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int64).alias("gics")
                )
            )
            gics_weights_data = (
                gics_weights_data.with_columns(pl.len().over(["gics", "eom"]).alias("bp_n"))
                .filter(pl.col("bp_n") >= bp_min_n)
                .drop("bp_n")
            )

            gics_weights = (
                gics_weights_data.group_by(["eom", "gics"])
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

            ind_gics_daily = (
                gics_weights.lazy()
                .join(
                    daily_lazy,
                    left_on=["id", "eom"],
                    right_on=["id", "eom_lag1"],
                    how="left",
                )
                .filter(pl.col("gics").is_not_null() & pl.col("ret_exc").is_not_null())
                .group_by(["gics", "date"])
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
            ind_gics_daily = ind_gics_daily.with_columns(
                pl.lit(excntry).str.to_uppercase().alias("excntry")
            )

            if excntry.lower() == "usa":
                ff49_weights_data = data.filter(pl.col("ff49").is_not_null()).select(
                    ["id", "eom", "ff49", "me", "me_cap"]
                )
                ff49_weights_data = (
                    ff49_weights_data.with_columns(pl.len().over(["ff49", "eom"]).alias("bp_n"))
                    .filter(pl.col("bp_n") >= bp_min_n)
                    .drop("bp_n")
                )

                ff49_weights = (
                    ff49_weights_data.group_by(["eom", "ff49"])
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

                ind_ff49_daily = (
                    ff49_weights.lazy()
                    .join(
                        daily_lazy,
                        left_on=["id", "eom"],
                        right_on=["id", "eom_lag1"],
                        how="left",
                    )
                    .filter(pl.col("ff49").is_not_null() & pl.col("ret_exc").is_not_null())
                    .group_by(["ff49", "date"])
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
                ind_ff49_daily = ind_ff49_daily.with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
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
                        "crsp_exchcd",
                        "comp_exchg",
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

        sub = add_ecdf(sub)
        sub = sub.with_columns(pl.col("cdf").min().over("eom").alias("min_cdf"))
        sub = sub.with_columns(
            pl.when(pl.col("cdf") == pl.col("min_cdf"))
            .then(0.00000001)
            .otherwise(pl.col("cdf"))
            .alias("cdf")
        )
        sub = sub.with_columns(
            (pl.col("cdf") * pfs).ceil().clip(lower_bound=1, upper_bound=pfs).alias("pf")
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
                    daily_lazy,
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
                    daily_lazy,
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

    # Batch-collect per-char lazy pipelines in chunks to bound peak memory.
    # Each chunk runs its LazyFrames concurrently via collect_all; chunks are
    # processed sequentially so at most COLLECT_CHUNK_SIZE chars' worth of
    # sort buffers / join hash tables live simultaneously.
    COLLECT_CHUNK_SIZE = 20
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
        output["gics_returns"] = ind_gics  # Assuming ind_gics is a DataFrame
        if excntry.lower() == "usa":
            output["ff49_returns"] = ind_ff49.clone()  # Assuming ind_ff49 is a DataFrame
        if daily_pf:
            output["gics_daily"] = ind_gics_daily
            if excntry.lower() == "usa":
                output["ff49_daily"] = ind_ff49_daily.clone()

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
                output["signals"] = pl.concat([op["signals"] for op in char_pfs])
                output["signals"] = output["signals"].with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )

    results = []
    # if (excntry=='usa' and cmp_key['us']) or (excntry!='usa' and cmp_key['int']):
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
            cmp = cmp.filter(pl.col("sd_var") != 0).drop("sd_var")
            cmp = cmp.with_columns((pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom"))

            results.append(cmp)

    if len(results) > 0:
        output_cmp = pl.concat(results)
        output_cmp = output_cmp.with_columns(pl.col("excntry").str.to_uppercase().alias("excntry"))
        output["cmp"] = output_cmp

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


def main() -> None:
    args = parse_args()
    configure_output_format(args.output_format)

    countries = []
    # Iterate through all files in the folder
    for file in os.listdir(os.path.join(data_path, "characteristics")):
        if file.endswith(".parquet") and "world" not in file:
            countries.append(file.replace(".parquet", ""))
    countries = sorted(countries)

    print(
        f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )
    # Extract Neccesary Information
    # Read Factor details from Excel file
    char_info = (
        pl.read_excel(
            "https://github.com/bkelly-lab/ReplicationCrisis/raw/master/GlobalFactors/Factor%20Details.xlsx",
            sheet_name="details",
        )
        .filter(pl.col("abr_jkp").is_not_null())
        .select([pl.col("abr_jkp").alias("characteristic"), pl.col("direction").cast(pl.Int32)])
    )

    # Read country classification details from Excel file
    country_classification = pl.read_excel(
        "https://github.com/bkelly-lab/ReplicationCrisis/raw/master/GlobalFactors/Country%20Classification.xlsx",
        sheet_name="countries",
    )

    # getting relevannt information from country classification file loaded at the start loaded at the start.
    # Select columns
    country_classification = country_classification.select(
        ["excntry", "msci_development", "region"]
    )
    # Filter out rows with NA in 'excntry' and exclude specific countries
    country_classification = country_classification.filter(
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

    # Read cluster lables details from Excel file
    cluster_labels = pl.read_csv(
        "https://raw.githubusercontent.com/bkelly-lab/ReplicationCrisis/refs/heads/master/GlobalFactors/Cluster%20Labels.csv",
        infer_schema_length=int(1e10),
    )

    # nyse_cutoffs
    nyse_size_cutoffs = pl.read_parquet(f"{data_path}/other_output/nyse_cutoffs.parquet")
    # nyse_size_cutoffs = nyse_size_cutoffs.with_columns(pl.col("eom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("eom"))

    # return_cutoffs
    ret_cutoffs = pl.read_parquet(f"{data_path}/other_output/return_cutoffs.parquet")
    # ret_cutoffs = ret_cutoffs.with_columns(pl.col("eom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("eom"))
    ret_cutoffs = ret_cutoffs.with_columns(
        (pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1")
    )
    if settings["daily_pf"]:
        ret_cutoffs_daily = pl.read_parquet(
            f"{data_path}/other_output/return_cutoffs_daily.parquet"
        )

    # market_returns
    market = pl.read_parquet(f"{data_path}/other_output/market_returns.parquet")
    # market = market.with_columns(pl.col("eom").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("eom"))

    # daily_market_returns
    if settings["daily_pf"]:
        market_daily = pl.read_parquet(f"{data_path}/other_output/market_returns_daily.parquet")
        # market_daily = market_daily.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("date"))

    # creating portfolios by using the portfolios function
    portfolio_data = {}
    for ex in countries:
        print(f"{ex}: {countries.index(ex) + 1} out of {len(countries)}")
        result = portfolios(
            data_path=data_path,
            excntry=ex,
            chars=chars,
            pfs=settings["pfs"],  # Number of portfolios
            bps=settings[
                "bps"
            ],  # What should breakpoints be based on? Non-Microcap stocks ("non_mc") or NYSE stocks "nyse"
            bp_min_n=settings["bp_min_n"],  # Minimum number of stocks used for breakpoints
            nyse_size_cutoffs=nyse_size_cutoffs,  # Data frame with NYSE size breakpoints
            source=settings[
                "source"
            ],  # Use data from "CRSP", "Compustat" or both: c("CRSP", "COMPUSTAT")
            wins_ret=settings[
                "wins_ret"
            ],  # Should Compustat returns be winsorized at the 0.1% and 99.9% of CRSP returns?
            cmp_key=settings["cmp"]["us"]
            if ex.lower() == "usa"
            else settings["cmp"]["int"],  # Create characteristics managed size portfolios?
            signals=settings["signals"]["us"]
            if ex.lower() == "usa"
            else settings["signals"]["int"],  # Create portfolio signals?
            signals_standardize=settings["signals"]["standardize"],  # Map chars to [-0.5, +0.5]?,
            signals_w=settings["signals"][
                "weight"
            ],  # Weighting for signals: in c("ew", "vw", "vw_cap")
            daily_pf=settings["daily_pf"],  # Should daily return be estimated
            ind_pf=settings["ind_pf"],  # Should industry portfolio returns be estimated
            ret_cutoffs=ret_cutoffs,  # Data frame for monthly winsorization. Neccesary when wins_ret=T
            ret_cutoffs_daily=ret_cutoffs_daily,  # Data frame for daily winsorization. Neccesary when wins_ret=T and daily_pf=T
        )
        portfolio_data[ex] = result

    # aggregating portfolio returns
    # pf_returns = pl.concat([portfolio_data[sub_dict]['pf_returns']
    #                         for sub_dict in portfolio_data
    #                         if 'pf_returns' in portfolio_data[sub_dict]])

    # pf_returns = pl.concat([portfolio_data[sub_dict]['pf_returns']
    #                         for sub_dict in portfolio_data
    #                          if sub_dict in portfolio_data and 'pf_returns' in portfolio_data[sub_dict]])

    if any(sub_data and "pf_returns" in sub_data for sub_key, sub_data in portfolio_data.items()):
        pf_returns = pl.concat(
            [
                sub_data["pf_returns"]
                for sub_key, sub_data in portfolio_data.items()
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
        sub_data and "pf_daily" in sub_data for sub_key, sub_data in portfolio_data.items()
    ):
        pf_daily = pl.concat(
            [
                sub_data["pf_daily"]
                for sub_key, sub_data in portfolio_data.items()
                if sub_data and "pf_returns" in sub_data
            ]
        )
        # pf_daily = pl.concat([portfolio_data[sub_dict]['pf_daily']
        #                     for sub_dict in portfolio_data
        #                     if 'pf_daily' in portfolio_data[sub_dict]])
        pf_daily = pf_daily.sort(["excntry", "characteristic", "pf", "date"])
    else:
        pf_daily = None

    # aggregating industry classification returns
    # GICS Returns
    if settings["ind_pf"] and any(
        sub_data and "gics_returns" in sub_data for sub_key, sub_data in portfolio_data.items()
    ):
        gics_returns = pl.concat(
            [
                sub_data["gics_returns"]
                for sub_key, sub_data in portfolio_data.items()
                if sub_data and "gics_returns" in sub_data
            ]
        )

        # gics_returns =pl.concat([portfolio_data[sub_dict]['gics_returns']
        #                     for sub_dict in portfolio_data
        #                     if 'gics_returns' in portfolio_data[sub_dict]])
        gics_returns = gics_returns.sort(["excntry", "gics", "eom"])
    else:
        gics_returns = None

    # FF49 Returns
    if settings["ind_pf"] and any(
        sub_data and "ff49_returns" in sub_data for sub_key, sub_data in portfolio_data.items()
    ):
        ff49_returns = pl.concat(
            [
                sub_data["ff49_returns"]
                for sub_key, sub_data in portfolio_data.items()
                if sub_data and "ff49_returns" in sub_data
            ]
        )
        ff49_returns = ff49_returns.sort(["excntry", "ff49", "eom"])
    else:
        ff49_returns = None

    # aggregating daily industry classification returns
    if (
        settings["ind_pf"]
        and settings["daily_pf"]
        and any(
            sub_data and "gics_daily" in sub_data for sub_key, sub_data in portfolio_data.items()
        )
    ):
        gics_daily = pl.concat(
            [
                sub_data["gics_daily"]
                for sub_key, sub_data in portfolio_data.items()
                if sub_data and "gics_daily" in sub_data
            ]
        )
        gics_daily = gics_daily.sort(["excntry", "gics", "date"])
    else:
        gics_daily = None

    if (
        settings["ind_pf"]
        and settings["daily_pf"]
        and any(
            sub_data and "ff49_daily" in sub_data for sub_key, sub_data in portfolio_data.items()
        )
    ):
        ff49_daily = pl.concat(
            [
                sub_data["ff49_daily"]
                for sub_key, sub_data in portfolio_data.items()
                if sub_data and "ff49_daily" in sub_data
            ]
        )
        ff49_daily = ff49_daily.sort(["excntry", "ff49", "date"])
    else:
        ff49_daily = None

    # Create HML Returns
    if pf_returns is not None and pf_returns.height > 0:
        hml_returns = pf_returns.group_by(["excntry", "characteristic", "eom"]).agg(
            [
                pl.col("pf").is_in([settings["pfs"], 1]).sum().alias("pfs"),
                (
                    pl.col("signal").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("signal").filter(pl.col("pf") == 1).first()
                ).alias("signal"),
                (
                    pl.col("n").filter(pl.col("pf") == settings["pfs"]).first()
                    + pl.col("n").filter(pl.col("pf") == 1).first()
                ).alias("n_stocks"),
                (pl.col("n").filter(pl.col("pf").is_in([settings["pfs"], 1])).min()).alias(
                    "n_stocks_min"
                ),
                (
                    pl.col("ret_ew").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_ew").filter(pl.col("pf") == 1).first()
                ).alias("ret_ew"),
                (
                    pl.col("ret_vw").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_vw").filter(pl.col("pf") == 1).first()
                ).alias("ret_vw"),
                (
                    pl.col("ret_vw_cap").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_vw_cap").filter(pl.col("pf") == 1).first()
                ).alias("ret_vw_cap"),
            ]
        )

        hml_returns = hml_returns.filter(pl.col("pfs") == 2).drop("pfs")
        hml_returns = hml_returns.sort(["excntry", "characteristic", "eom"])

        # Create Long-Short Factors [Sign Returns to be consistent with original paper]
        lms_returns = char_info.join(hml_returns, on="characteristic", how="left")

        # Define columns to be modified
        resign_cols = ["signal", "ret_ew", "ret_vw", "ret_vw_cap"]
        lms_returns = lms_returns.with_columns(
            [pl.col(var) * pl.col("direction").alias(var) for var in resign_cols]
        )
    else:
        hml_returns = None
        lms_returns = None

    # daily hml and lms
    if settings["daily_pf"] and pf_daily is not None and pf_daily.height > 0:
        hml_daily = pf_daily.group_by(["excntry", "characteristic", "date"]).agg(
            [
                pl.col("pf").is_in([settings["pfs"], 1]).sum().alias("pfs"),
                (
                    pl.col("n").filter(pl.col("pf") == settings["pfs"]).first()
                    + pl.col("n").filter(pl.col("pf") == 1).first()
                ).alias("n_stocks"),
                (pl.col("n").filter(pl.col("pf").is_in([settings["pfs"], 1])).min()).alias(
                    "n_stocks_min"
                ),
                (
                    pl.col("ret_ew").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_ew").filter(pl.col("pf") == 1).first()
                ).alias("ret_ew"),
                (
                    pl.col("ret_vw").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_vw").filter(pl.col("pf") == 1).first()
                ).alias("ret_vw"),
                (
                    pl.col("ret_vw_cap").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_vw_cap").filter(pl.col("pf") == 1).first()
                ).alias("ret_vw_cap"),
            ]
        )

        hml_daily = hml_daily.filter(pl.col("pfs") == 2).drop("pfs")
        hml_daily = hml_daily.sort(["excntry", "characteristic", "date"])

        lms_daily = char_info.join(hml_daily, on="characteristic", how="left")
        resign_cols = ["ret_ew", "ret_vw", "ret_vw_cap"]

        lms_daily = lms_daily.with_columns(
            [(pl.col(var) * pl.col("direction")).alias(var) for var in resign_cols]
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
        cmp_returns = None

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

    # creating regional portfolios
    if lms_returns is not None:
        regional_pfs = []
        for i in range(regions.height):
            info = regions[i][0]
            reg_pf = regional_data(
                data=lms_returns,
                mkt=market,
                countries=info["country_codes"][0],
                date_col="eom",
                char_col="characteristic",
                weighting=settings["regional_pfs"]["country_weights"],
                countries_min=info["countries_min"][0],
                periods_min=settings["regional_pfs"]["months_min"],
                stocks_min=settings["regional_pfs"]["stocks_min"],
            )
            reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
            reg_pf = reg_pf.select(
                [
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
            )
            regional_pfs.append(reg_pf)

        regional_pfs = pl.concat(regional_pfs)  # .explode('direction')

    else:
        regional_pfs = None

    if settings["daily_pf"] and lms_daily is not None:
        regional_pfs_daily = []
        for i in range(regions.height):
            info = regions[i][0]
            reg_pf = regional_data(
                data=lms_daily,
                mkt=market_daily,
                countries=info["country_codes"][0],
                date_col="date",
                char_col="characteristic",
                weighting=settings["regional_pfs"]["country_weights"],
                countries_min=info["countries_min"][0],
                periods_min=settings["regional_pfs"]["months_min"] * 21,
                stocks_min=settings["regional_pfs"]["stocks_min"],
            )
            reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
            reg_pf = reg_pf.select(
                [
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
            )

            regional_pfs_daily.append(reg_pf)  # .explode('direction')

        regional_pfs_daily = pl.concat(regional_pfs_daily)

    else:
        regional_pfs_daily = None

    # creating regional clusters
    if cluster_pfs is not None:
        regional_clusters = []
        for i in range(regions.height):
            info = regions[i][0]
            reg_pf = cluster_pfs.rename({"n_factors": "n_stocks_min"})
            reg_pf = reg_pf.with_columns(
                pl.lit(None).cast(pl.Float64).alias("direction")
            )  # Adding 'direction' column with NA values
            reg_pf = regional_data(
                data=reg_pf,
                mkt=market,
                countries=info["country_codes"][0],
                date_col="eom",
                char_col="cluster",
                weighting=settings["regional_pfs"]["country_weights"],
                countries_min=info["countries_min"][0],
                periods_min=settings["regional_pfs"]["months_min"],
                stocks_min=1,
            )
            reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
            reg_pf = reg_pf.select(
                [
                    "region",
                    "cluster",
                    "eom",
                    "n_countries",
                    "ret_ew",
                    "ret_vw",
                    "ret_vw_cap",
                    "mkt_vw_exc",
                ]
            )

            regional_clusters.append(reg_pf)

        regional_clusters = pl.concat(regional_clusters)
    else:
        regional_clusters = None

    if settings["daily_pf"] and cluster_pfs_daily is not None:
        regional_clusters_daily = []
        for i in range(regions.height):
            info = regions[i][0]
            reg_pf = cluster_pfs_daily.rename({"n_factors": "n_stocks_min"})
            reg_pf = reg_pf.with_columns(
                pl.lit(None).cast(pl.Float64).alias("direction")
            )  # Adding 'direction' column with NA values
            reg_pf = regional_data(
                data=reg_pf,
                mkt=market_daily,
                countries=info["country_codes"][0],
                date_col="date",
                char_col="cluster",
                weighting=settings["regional_pfs"]["country_weights"],
                countries_min=info["countries_min"][0],
                periods_min=settings["regional_pfs"]["months_min"] * 21,
                stocks_min=1,
            )
            reg_pf = reg_pf.with_columns(pl.lit(info["name"][0]).alias("region"))
            reg_pf = reg_pf.select(
                [
                    "region",
                    "cluster",
                    "date",
                    "n_countries",
                    "ret_ew",
                    "ret_vw",
                    "ret_vw_cap",
                    "mkt_vw_exc",
                ]
            )
            regional_clusters_daily.append(reg_pf)

        regional_clusters_daily = pl.concat(regional_clusters_daily)
    else:
        regional_clusters_daily = None

    # Writing output
    if pf_returns is not None:
        write_dataframe(
            pf_returns.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/pfs.parquet",
        )
    if hml_returns is not None:
        write_dataframe(
            hml_returns.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/hml.parquet",
        )
    if lms_returns is not None:
        write_dataframe(
            lms_returns.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/lms.parquet",
        )
    if cmp_returns is not None:
        write_dataframe(
            cmp_returns.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/cmp.parquet",
        )
    if cluster_pfs is not None:
        write_dataframe(
            cluster_pfs.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/clusters.parquet",
        )

    if settings["daily_pf"]:
        if pf_daily is not None:
            write_dataframe(
                pf_daily.filter(pl.col("date") <= settings["end_date"]),
                f"{output_path}/pfs_daily.parquet",
            )
        if hml_daily is not None:
            write_dataframe(
                hml_daily.filter(pl.col("date") <= settings["end_date"]),
                f"{output_path}/hml_daily.parquet",
            )
        if lms_daily is not None:
            write_dataframe(
                lms_daily.filter(pl.col("date") <= settings["end_date"]),
                f"{output_path}/lms_daily.parquet",
            )
        if cluster_pfs_daily is not None:
            write_dataframe(
                cluster_pfs_daily.filter(pl.col("date") <= settings["end_date"]),
                f"{output_path}/clusters_daily.parquet",
            )

    if settings["ind_pf"]:
        if gics_returns is not None:
            write_dataframe(
                gics_returns.filter(pl.col("eom") <= settings["end_date"]),
                f"{output_path}/industry_gics.parquet",
            )
        if ff49_returns is not None:
            write_dataframe(
                ff49_returns.filter(pl.col("eom") <= settings["end_date"]),
                f"{output_path}/industry_ff49.parquet",
            )

    if settings["ind_pf"] and settings["daily_pf"]:
        if gics_daily is not None:
            write_dataframe(
                gics_daily.filter(pl.col("date") <= settings["end_date"]),
                f"{output_path}/industry_gics_daily.parquet",
            )
        if ff49_daily is not None:
            write_dataframe(
                ff49_daily.filter(pl.col("date") <= settings["end_date"]),
                f"{output_path}/industry_ff49_daily.parquet",
            )

    # Create directory for Regional Factors
    if regional_pfs is not None:
        reg_folder = os.path.join(output_path, "regional_factors")
        if not os.path.exists(reg_folder):
            os.makedirs(reg_folder)

        # Write regional portfolios to parquet files
        for reg in regional_pfs["region"].unique():
            filtered_df = regional_pfs.filter(
                (pl.col("eom") <= settings["end_date"]) & (pl.col("region") == reg)
            )
            file_path = os.path.join(reg_folder, f"{reg}.parquet")
            write_dataframe(filtered_df, file_path)

    # Conditional block for daily regional factors
    if settings["daily_pf"]:
        if regional_pfs_daily is not None:
            # Create directory for Daily Regional Factors
            reg_folder_daily = os.path.join(output_path, "regional_factors_daily")
            if not os.path.exists(reg_folder_daily):
                os.makedirs(reg_folder_daily)

            # Write daily regional portfolios to parquet files
            for reg in regional_pfs_daily["region"].unique():
                filtered_df_daily = regional_pfs_daily.filter(
                    (pl.col("date") <= settings["end_date"]) & (pl.col("region") == reg)
                )
                file_path_daily = os.path.join(reg_folder_daily, f"{reg}.parquet")
                write_dataframe(filtered_df_daily, file_path_daily)

    # Create directory for Regional Clusters
    if regional_clusters is not None:
        reg_folder = os.path.join(output_path, "regional_clusters")
        if not os.path.exists(reg_folder):
            os.makedirs(reg_folder)

        # Write regional clusters to parquet files
        for reg in regional_clusters["region"].unique():
            filtered_df = regional_clusters.filter(
                (pl.col("eom") <= settings["end_date"]) & (pl.col("region") == reg)
            )
            file_path = os.path.join(reg_folder, f"{reg}.parquet")
            write_dataframe(filtered_df, file_path)

    # Conditional block for daily regional clusters
    if settings["daily_pf"]:
        if regional_clusters_daily is not None:
            # Create directory for Daily Regional Clusters
            reg_folder_daily = os.path.join(output_path, "regional_clusters_daily")
            if not os.path.exists(reg_folder_daily):
                os.makedirs(reg_folder_daily)

            # Write daily regional clusters to parquet files
            for reg in regional_clusters_daily["region"].unique():
                filtered_df_daily = regional_clusters_daily.filter(
                    (pl.col("date") <= settings["end_date"]) & (pl.col("region") == reg)
                )
                file_path_daily = os.path.join(reg_folder_daily, f"{reg}.parquet")
                write_dataframe(filtered_df_daily, file_path_daily)

    # Create directory for Country Factors
    if lms_returns is not None:
        cnt_folder = os.path.join(output_path, "country_factors")
        if not os.path.exists(cnt_folder):
            os.makedirs(cnt_folder)

        # Write country factors to parquet files
        for exc in lms_returns["excntry"].unique():
            if exc:
                filtered_df = lms_returns.filter(
                    (pl.col("eom") <= settings["end_date"]) & (pl.col("excntry") == exc)
                )
                file_path = os.path.join(cnt_folder, f"{exc}.parquet")
                write_dataframe(filtered_df, file_path)

    # Conditional block for daily country factors
    if settings["daily_pf"]:
        if lms_daily is not None:
            # Create directory for Daily Country Factors
            cnt_folder_daily = os.path.join(output_path, "country_factors_daily")
            if not os.path.exists(cnt_folder_daily):
                os.makedirs(cnt_folder_daily)

            # Write daily country factors to parquet files
            for exc in lms_daily["excntry"].unique():
                if exc:
                    filtered_df_daily = lms_daily.filter(
                        (pl.col("date") <= settings["end_date"]) & (pl.col("excntry") == exc)
                    )
                    file_path_daily = os.path.join(cnt_folder_daily, f"{exc}.parquet")
                    write_dataframe(filtered_df_daily, file_path_daily)

    convert_outputs_to_csv()

    print(
        f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
