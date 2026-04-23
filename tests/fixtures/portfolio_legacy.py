"""
Frozen copy of the pre-optimization portfolio builder functions.

Imported by ``tests/unit/test_portfolio_parity.py`` to run a legacy baseline
against the current ``code/portfolio.py`` implementation on the same inputs
and assert that outputs match. This file is not used at runtime outside the
test suite.
"""

from __future__ import annotations

import polars as pl
from tqdm import tqdm


def add_ecdf(df: pl.DataFrame, group_cols: list[str] | None = None) -> pl.DataFrame:
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

    # Load the data
    data = pl.read_parquet(file_path, columns=columns)
    data = data

    # capping me at nyse cut-off
    data = data.join(nyse_size_cutoffs.select(["eom", "nyse_p80"]), on="eom", how="left")
    data = data.with_columns(
        pl.min_horizontal(pl.col("me"), pl.col("nyse_p80")).alias("me_cap")
    ).drop("nyse_p80")

    # ensuring numerical columns are float-added later:
    exclude = ["id", "eom", "source_crsp", "size_grp", "excntry"]
    for i in data.columns:
        if i not in exclude:
            data = data.with_columns(pl.col(i).cast(pl.Float64))

    # Screens
    if len(source) == 1:
        if source[0] == "CRSP":
            data = data.filter(pl.col("source_crsp") == 1)
        elif source[0] == "COMPUSTAT":
            data = data.filter(pl.col("source_crsp") == 0)
    data = data.filter(
        (pl.col("size_grp").is_not_null())
        & (pl.col("me").is_not_null())
        & (pl.col("ret_exc_lead1m").is_not_null())
    )

    # Daily Returns
    if daily_pf:
        daily_file_path = f"{data_path}/return_data/daily_rets_by_country/{excntry}.parquet"
        daily = pl.read_parquet(daily_file_path, columns=["id", "date", "ret_exc"])
        # daily = daily.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias("date"))
        daily = daily.with_columns(
            (pl.col("date").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1")
        )
        # ensuring numerical columns are float-added later:
        daily = daily.with_columns(pl.col("ret_exc").cast(pl.Float64))

    if wins_ret:
        data = data.join(
            ret_cutoffs.select(["eom_lag1", "ret_exc_0_1", "ret_exc_99_9"]).rename(
                {"eom_lag1": "eom", "ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}
            ),
            on="eom",
            how="left",
        )
        data = data.with_columns(
            pl.when((pl.col("source_crsp") == 0) & (pl.col("ret_exc_lead1m") > pl.col("p999")))
            .then(pl.col("p999"))
            .when((pl.col("source_crsp") == 0) & (pl.col("ret_exc_lead1m") < pl.col("p001")))
            .then(pl.col("p001"))
            .otherwise(pl.col("ret_exc_lead1m"))
            .alias("ret_exc_lead1m")
        ).drop(["source_crsp", "p001", "p999"])

        if daily_pf:
            # Derive eom from date for joining with daily return cutoffs
            daily = daily.with_columns(pl.col("date").dt.month_end().alias("eom"))

            # Joining with daily return cutoffs
            daily = daily.join(
                ret_cutoffs_daily.select(["eom", "ret_exc_0_1", "ret_exc_99_9"]).rename(
                    {"ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}
                ),
                on="eom",
                how="left",
            )

            # Applying winsorization to daily returns for Compustat data (id > 99999)
            daily = daily.with_columns(
                pl.when((pl.col("id") > 99999) & (pl.col("ret_exc") > pl.col("p999")))
                .then(pl.col("p999"))
                .when((pl.col("id") > 99999) & (pl.col("ret_exc") < pl.col("p001")))
                .then(pl.col("p001"))
                .otherwise(pl.col("ret_exc"))
                .alias("ret_exc")
            ).drop(["p001", "p999", "eom"])

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
                    daily.lazy(),
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
                        daily.lazy(),
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

    # creating portfolios for all the characteristics
    char_pfs = []
    for _i, x in enumerate(
        tqdm(chars, desc="Processing chars", unit="char", ncols=80, disable=True)
    ):
        op = {}

        data = data.with_columns(pl.col(x).cast(pl.Float64).alias("var"))
        if not signals:
            # Select rows where 'var' is not missing and only specific columns
            sub = (
                data.lazy()
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
                    ]
                )
            )
        else:
            # Select rows where 'var' is not missing, retaining all columns
            sub = data.lazy().filter(pl.col("var").is_not_null())

        if bps == "nyse":
            # Create 'bp_stock' column for NYSE criteria
            sub = sub.with_columns(
                (
                    ((pl.col("crsp_exchcd") == 1) & pl.col("comp_exchg").is_null())
                    | ((pl.col("comp_exchg") == 11) & pl.col("crsp_exchcd").is_null())
                ).alias("bp_stock")
            )

        elif bps == "non_mc":
            # Create 'bp_stock' column for non-microcap criteria
            sub = sub.with_columns(
                pl.col("size_grp").is_in(["mega", "large", "small"]).alias("bp_stock")
            )

        sub = sub.with_columns(bp_n=pl.sum("bp_stock").over("eom")).filter(
            pl.col("bp_n") >= bp_min_n
        )

        # Ensure that 'sub' is not empty
        if sub.limit(1).collect().height > 0:
            sub = add_ecdf(sub)

            # Step 1: Find the minimum CDF value within each 'eom' group
            sub = sub.with_columns(pl.col("cdf").min().over("eom").alias("min_cdf"))

            # Step 2: Adjust CDF values for the lowest value in each group
            sub = sub.with_columns(
                pl.when(pl.col("cdf") == pl.col("min_cdf"))
                .then(0.00000001)
                .otherwise(pl.col("cdf"))
                .alias("cdf")
            )

            # Step 3: Calculate portfolio assignments and adjust portfolio numbers (Happens when non-bp stocks extend beyond the bp stock range)
            sub = sub.with_columns(
                (pl.col("cdf") * pfs).ceil().clip(lower_bound=1, upper_bound=pfs).alias("pf")
            )

            pf_returns = sub.group_by(["pf", "eom"]).agg(
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
            pf_returns = pf_returns.with_columns(
                pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom")
            )
            op["pf_returns"] = pf_returns.collect()

            if signals:
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
                        pl.when(pl.col(var).is_null())
                        .then(pl.lit(0))
                        .otherwise(pl.col(var))
                        .alias(var)
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
                signals = pf_signals.clone()  # store in dictionary later
                op["signals"] = signals.collect()

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

                pf_daily = daily_sub.group_by(["pf", "date"]).agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n"),
                        ((pl.col("w_ew") * pl.col("ret_exc")).sum()).alias("ret_ew"),
                        ((pl.col("w_vw") * pl.col("ret_exc")).sum()).alias("ret_vw"),
                        ((pl.col("w_vw_cap") * pl.col("ret_exc")).sum()).alias("ret_vw_cap"),
                    ]
                )
                op["pf_daily"] = pf_daily.collect()

            char_pfs.append(op)

    output = {}

    # Aggregate pf_returns
    if len([op["pf_returns"] for op in char_pfs]) > 0:
        output["pf_returns"] = pl.concat([op["pf_returns"] for op in char_pfs])
    else:
        pass
    # Aggregate pf_daily if daily_pf is true
    if (daily_pf) and len([op["pf_daily"] for op in char_pfs]) > 0:
        output["pf_daily"] = pl.concat([op["pf_daily"] for op in char_pfs])
    else:
        pass
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
