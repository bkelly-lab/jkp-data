#helper functions:
#importing all the required packages:
import polars as pl
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import math
import numpy as np
import statsmodels.api as sm
from typing import Any

def chg_to_lagassets(df, var_gr):
    # Removing '_x' from the column name
    name_gr = var_gr.replace('_x', '')
    
    # Appending '_gr' and '1a' to the name
    name_gr = f"{name_gr}_gr1a"
    
    # Calculating the growth rate
    df = df.with_columns(
        ((pl.col(var_gr) - pl.col(var_gr).shift(12))/pl.col('at_x').shift(12)).alias(name_gr)
    )
    
    # Applying conditions to set certain values to NaN
    df = df.with_columns(
        pl.when((pl.col('at_x').shift(12) <= 0) | 
                (pl.col("count") <= 12))
        .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name_gr)).alias(name_gr)
    )
    
    return df


from typing import List

def chg_to_avgassets(
    df: pl.DataFrame,
    vars_x: List[str],
    at_col: str = "at_x",
    horizon: int = 12
) -> pl.DataFrame:
    """
    For each variable in vars_x (e.g. ['sale_x','revt_x',…]), compute
      (var - lag_horizon(var)) / (at + lag_horizon(at))
    and null it out if count<=horizon or (at+lag_horizon(at))<=0.
    New columns are named <base>_gr{horizon_in_years}a, where base = var.replace('_x','').
    """
    years = horizon / 12
    # int if whole number, else float
    yr_suffix = int(years) if years.is_integer() else years

    exprs = []
    for var in vars_x:
        base = var.replace("_x", "")
        new_name = f"{base}_gr{yr_suffix}a"
        lag_var = pl.col(var).shift(horizon)
        lag_at  = pl.col(at_col).shift(horizon)

        exprs.append(
            pl.when(
                (pl.col("count") <= horizon) |
                ((pl.col(at_col) + lag_at) <= 0)
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(
                (pl.col(var) - lag_var) /
                (pl.col(at_col) + lag_at)
            )
            .alias(new_name)
        )

    return df.with_columns(exprs)




def chg_to_exp(df, var_ce):
    # Removing '_x' from the column name
    name_ce = var_ce.replace('_x', '')
    
    # Appending '_gr' and '1a' to the name
    name_ce = f"{name_ce}_ce"
    
    # Calculating the growth rate
    df = df.with_columns(
        ((pl.col(var_ce)/((pl.col(var_ce).shift(12) + pl.col(var_ce).shift(24))/2)) -1).alias(name_ce)
    )
    
    # Applying conditions to set certain values to NaN
    df = df.with_columns(
        pl.when((pl.col('count') <= 24) | 
                (((pl.col(var_ce).shift(12) + pl.col(var_ce).shift(24))/2) <= 0))
        .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name_ce)).alias(name_ce)
    )
    
    return df



def apply_to_lastq(expr: pl.Expr, qtrs: int, func: str) -> pl.Expr:
    """
    Generate an expression that computes func(current + lag3 + lag6 + ... + lag3*(qtrs-1))
    - expr:        a polars Expr, e.g. pl.col("__chg")
    - qtrs:        number of terms to include (so macro’s _qtrs)
    - func:        one of "mean", "std", or "sum"
    """
    #  build the shifted expressions
    shifts = [expr.shift(i * 3) for i in range(qtrs)]
    # pack into a List column
    lst = pl.concat_list(shifts)
    # dispatch the right list‐method
    if func == "mean":
        return lst.list.mean()
    elif func == "std":
        return lst.list.std()
    elif func == "sum":
        return lst.list.sum()
    else:
        raise ValueError(f"apply_to_lastq: unsupported func={func!r}")


def standardized_unexpected(df, var, qtrs, qtrs_min):
    name = var.replace("_x","") + "_su"

    # 0) sort so that shifts reset per group
    df = (
      df.sort(["gvkey","curcd","datadate"])
        .with_columns(pl.col(var).cast(pl.Float64))  # ensure float
        .with_columns(pl.col("count").cum_count().over(["gvkey","curcd"]).alias("count"))
    )

    # 1) one‐year change
    df = df.with_columns((pl.col(var) - pl.col(var).shift(12)).alias("__chg"))

    # 2) rolling stats over last qtrs (current + lag3 + …)
    df = df.with_columns([
        apply_to_lastq(pl.col("__chg"), qtrs, "mean").alias("__chg_mean"),
        apply_to_lastq(pl.col("__chg"), qtrs, "std").alias("__chg_std"),
        # count non‐nulls in that same window:
        apply_to_lastq(pl.col("__chg").is_not_null().cast(pl.UInt8), qtrs, "sum").alias("__chg_n"),
    ])

    # 3) blank out if too few observations
    df = df.with_columns([
        pl.when(pl.col("__chg_n") <= qtrs_min).then(None).otherwise(pl.col("__chg_mean")).alias("__chg_mean"),
        pl.when(pl.col("__chg_n") <= qtrs_min).then(None).otherwise(pl.col("__chg_std")).alias("__chg_std"),
    ])

    # 4) compute the standardized‐unexpected exactly as in SAS
    df = df.with_columns((
        (
          pl.col(var)
          - (pl.col(var).shift(12) + pl.col("__chg_mean").shift(3))
        )
        / pl.col("__chg_std").shift(3))
    .alias(name)
    ).with_columns(pl.when(pl.col("__chg_std").shift(3)==0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name)).alias(name))

    # 5) enforce the final count‐threshold of SAS: if count<=12+3*qtrs then blank
    df = df.with_columns(
      pl.when(pl.col("count") <= (12 + 3 * qtrs))
        .then(None)
        .otherwise(pl.col(name))
        .alias(name)
    )

    # 6) clean up helpers
    return df.drop(["__chg","__chg_mean","__chg_std","__chg_n"])




def volq(df, name, var, qtrs, qtrs_min):

    #creating helping variables
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (3*qtrs), 3)]).list.eval(pl.element().is_not_null().sum()).alias('__n')).explode('__n')
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (3*qtrs), 3)]).list.eval(pl.element().std()).alias(name)).explode(name)

    
    #dealing with corner cases and deleting helping variables
    df = df.with_columns(pl.when((pl.col('count') <= ((qtrs-1)*3)) | (pl.col('__n') < qtrs_min)).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name)).alias(name))
    df = df.drop('__n')
    
    return df


def vola(df, name, var, yrs, yrs_min):

    #creating helping variables
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (12*yrs), 12)]).list.eval(pl.element().is_not_null().sum()).alias('__n')).explode('__n')
    df = df.with_columns(pl.concat_list([pl.col(var).shift(i) for i in range(0, (12*yrs), 12)]).list.eval(pl.element().std()).alias(name)).explode(name)

    
    #dealing with corner cases and deleting helping variables
    df = df.with_columns(pl.when((pl.col('count') <= ((yrs-1)*12)) | (pl.col('__n') < yrs_min)).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col(name)).alias(name))
    df = df.drop('__n')
    
    return df



def earnings_variability(df: pl.DataFrame, esm_h: int) -> pl.DataFrame:
    # 1) sort for correct group‑wise lag behavior
    df = df.sort(["gvkey", "datadate"])

    # 2) compute the 12‑month lag of at_x
    df = df.with_columns([
        pl.col("at_x").shift(12).over("gvkey").alias("at_lag12")
    ])

    # 3) compute __roa and __croa with their null guards
    df = df.with_columns([
        (pl.col("ni_x") / pl.col("at_lag12")).alias("__roa"),
        (pl.col("ocf_x") / pl.col("at_lag12")).alias("__croa"),
    ]).with_columns([
        pl.when((pl.col("count") <= 12) | (pl.col("at_lag12") <= 0))
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("__roa"))
          .alias("__roa"),
        pl.when((pl.col("count") <= 12) | (pl.col("at_lag12") <= 0))
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("__croa"))
          .alias("__croa"),
    ])

    # 4) build annual lags of __roa and __croa
    shifts = [12 * i for i in range(esm_h)]
    # create all __roa_lag<i> in one go
    df = df.with_columns([
        pl.col("__roa").shift(s).over("gvkey").alias(f"__roa_lag{i}")
        for i, s in enumerate(shifts)
    ] + [
        pl.col("__croa").shift(s).over("gvkey").alias(f"__croa_lag{i}")
        for i, s in enumerate(shifts)
    ])

    # 5) count non‑missing in each window
    df = df.with_columns([
        sum(
            pl.col(f"__roa_lag{i}").is_not_null().cast(pl.Int32)
            for i in range(esm_h)
        ).alias("__roa_n"),
        sum(
            pl.col(f"__croa_lag{i}").is_not_null().cast(pl.Int32)
            for i in range(esm_h)
        ).alias("__croa_n"),
    ])

    # 6) compute std dev in each window
    df = df.with_columns([
        pl.concat_list([pl.col(f"__roa_lag{i}") for i in range(esm_h)])
          .list.std()
          .alias("__roa_std"),
        pl.concat_list([pl.col(f"__croa_lag{i}") for i in range(esm_h)])
          .list.std()
          .alias("__croa_std"),
    ])

    # 7) final ratio
    df = df.with_columns([
        (pl.col("__roa_std") / pl.col("__croa_std"))
          .alias("earnings_variability")
    ])

    # 8) apply the macro’s null logic
    df = df.with_columns([
        pl.when(
            (pl.col("count") <= esm_h * 12)
            | (pl.col("__croa_std") <= 0)
            | (pl.col("__roa_n")   < esm_h)
            | (pl.col("__croa_n")  < esm_h)
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(pl.col("earnings_variability"))
        .alias("earnings_variability")
    ])

    # 9) drop all helpers
    drop_cols = (
        ["at_lag12"]
        + [f"__roa_lag{i}"  for i in range(esm_h)]
        + [f"__croa_lag{i}" for i in range(esm_h)]
        + ["__roa", "__croa", "__roa_n", "__croa_n", "__roa_std", "__croa_std"]
    )
    return df.drop(drop_cols)




def equity_duration_cd(
    df: pl.DataFrame,
    horizon: int,
    r: float,
    roe_mean: float,
    roe_ar1: float,
    g_mean: float,
    g_ar1: float
) -> pl.DataFrame:

    # 0) sort so lag12 is per firm
    df = df.sort(["gvkey", "datadate"])

    # 1) compute 12‑month lags
    df = df.with_columns([
        pl.col("be_x").shift(12).over("gvkey").alias("be_lag12"),
        pl.col("sale_x").shift(12).over("gvkey").alias("sale_lag12"),
    ])

    # 2) initial vars with 0‑denominator guards
    df = df.with_columns([
        pl.when(pl.col("be_lag12") <= 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ni_x") / pl.col("be_lag12"))
          .alias("__roe0"),

        pl.when(pl.col("sale_lag12") <= 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("sale_x") / pl.col("sale_lag12") - 1)
          .alias("__g0"),

        pl.col("be_x").alias("__be0"),
    ]).with_columns([
        pl.when((pl.col("count") <= 12) | (pl.col("be_lag12") <= 1))
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("__roe0"))
          .alias("__roe0"),

        pl.when((pl.col("count") <= 12) | (pl.col("sale_lag12") <= 1))
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("__g0"))
          .alias("__g0"),
    ])

    # 3) forecast loop
    roe_c = roe_mean * (1 - roe_ar1)
    g_c   = g_mean   * (1 - g_ar1)
    for i in range(1, horizon + 1):
        j = i - 1
        df = df.with_columns([
            (pl.lit(roe_c) + pl.lit(roe_ar1) * pl.col(f"__roe{j}")).alias(f"__roe{i}"),
            (pl.lit(g_c)   + pl.lit(g_ar1)   * pl.col(f"__g{j}")).alias(f"__g{i}"),
        ]).with_columns([
            (pl.col(f"__be{j}") * (1 + pl.col(f"__g{i}"))).alias(f"__be{i}"),
            (pl.col(f"__be{j}") * (pl.col(f"__roe{i}") - pl.col(f"__g{i}"))).alias(f"__cd{i}"),
        ])

    # 4) initialize duration helpers
    df = df.with_columns([
        (pl.lit(horizon) + (pl.lit(1) + pl.lit(r)) / pl.lit(r)).alias("ed_constant"),
        pl.lit(0.0).alias("ed_cd_w"),
        pl.lit(0.0).alias("ed_cd"),
        pl.lit(0).cast(pl.Int32).alias("ed_err"),
    ])

    # 5) accumulate with carrying ed_err forward
    for t in range(1, horizon + 1):
        factor = (1 + r) ** t
        df = df.with_columns([
            (pl.col("ed_cd_w") + t * pl.col(f"__cd{t}") / factor).alias("ed_cd_w"),
            (pl.col("ed_cd")   +     pl.col(f"__cd{t}") / factor).alias("ed_cd")]).with_columns(pl.when(pl.col(f"__be{t}") < 0).then(pl.lit(1)).otherwise(pl.col("ed_err")).alias("ed_err"))


    for t in range(1, horizon + 1):
        df = df.with_columns(pl.when((pl.col(f"__be{t}") < 0) | (pl.col(f"__be{t}").is_null())).then(pl.lit(1)).otherwise(pl.col("ed_err")).alias("ed_err"))

    # 6) drop all helpers
    helpers = (
        ["be_lag12", "sale_lag12"]
        + [f"__roe{i}" for i in range(horizon + 1)]
        + [f"__g{i}"   for i in range(horizon + 1)]
        + [f"__be{i}"  for i in range(horizon + 1)]
        + [f"__cd{i}"  for i in range(1, horizon + 1)]
    )
    return df.drop(helpers)

def pitroski_f(df: pl.DataFrame, name: str) -> pl.DataFrame:
    # ---- precompute the ratio expressions we’ll need to shift ----
    ratio_lev  = (pl.col("dltt")  / pl.col("at_x")).alias("__lev_ratio")
    ratio_liq  = (pl.col("ca_x")  / pl.col("cl_x")).alias("__liq_ratio")
    ratio_gm   = (pl.col("gp_x")  / pl.col("sale_x")).alias("__gm_ratio")

    return (
        df
        # ROA
        .with_columns((pl.col("ni_x") / pl.col("at_x").shift(12)).alias("__f_roa"))
        .with_columns(
            pl.when((pl.col("count") <= 12) | (pl.col("at_x").shift(12) <= 0))
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("__f_roa"))
              .alias("__f_roa")
        )

        # CROA
        .with_columns((pl.col("ocf_x") / pl.col("at_x").shift(12)).alias("__f_croa"))
        .with_columns(
            pl.when((pl.col("count") <= 12) | (pl.col("at_x").shift(12) <= 0))
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("__f_croa"))
              .alias("__f_croa")
        )

        # DROA
        .with_columns((pl.col("__f_roa") - pl.col("__f_roa").shift(12)).alias("__f_droa"))
        .with_columns(
            pl.when(pl.col("count") <= 12)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("__f_droa"))
              .alias("__f_droa")
        )

        # ACCRUALS
        .with_columns((pl.col("__f_croa") - pl.col("__f_roa")).alias("__f_acc"))

        # LEVERAGE – correct lag
        .with_columns(ratio_lev)
        .with_columns(
            (ratio_lev - pl.col("__lev_ratio").shift(12))
            .alias("__f_lev")
        )
        .with_columns(
            pl.when(
                (pl.col("count") <= 12)
                | (pl.col("at_x") <= 0)
                | (pl.col("at_x").shift(12) <= 0)
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("__f_lev"))
            .alias("__f_lev")
        )

        # LIQUIDITY – correct lag
        .with_columns(ratio_liq)
        .with_columns(
            (ratio_liq - pl.col("__liq_ratio").shift(12))
            .alias("__f_liq")
        )
        .with_columns(
            pl.when(
                (pl.col("count") <= 12)
                | (pl.col("cl_x") <= 0)
                | (pl.col("cl_x").shift(12) <= 0)
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("__f_liq"))
            .alias("__f_liq")
        )

        # EQIS
        .with_columns(pl.col("eqis_x").alias("__f_eqis"))

        # GROSS MARGIN – correct lag
        .with_columns(ratio_gm)
        .with_columns(
            (ratio_gm - pl.col("__gm_ratio").shift(12))
            .alias("__f_gm")
        )
        .with_columns(
            pl.when(
                (pl.col("count") <= 12)
                | (pl.col("sale_x") <= 0)
                | (pl.col("sale_x").shift(12) <= 0)
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("__f_gm"))
            .alias("__f_gm")
        )

        # ASSET‐TURNOVER
        .with_columns(
            (
                (pl.col("sale_x") / pl.col("at_x").shift(12))
                - (pl.col("sale_x").shift(12) / pl.col("at_x").shift(24))
            )
            .alias("__f_aturn")
        )
        .with_columns(
            pl.when(
                (pl.col("count") <= 24)
                | (pl.col("at_x").shift(12) <= 0)
                | (pl.col("at_x").shift(24) <= 0)
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("__f_aturn"))
            .alias("__f_aturn")
        )

        # Pitroski Score
        .with_columns(
            (
                (pl.col("__f_roa")   > 0).cast(pl.Int32)
              + (pl.col("__f_croa") > 0).cast(pl.Int32)
              + (pl.col("__f_droa") > 0).cast(pl.Int32)
              + (pl.col("__f_acc")  > 0).cast(pl.Int32)
              + (pl.col("__f_lev")  < 0).cast(pl.Int32)
              + (pl.col("__f_liq")  > 0).cast(pl.Int32)
              + (pl.coalesce([pl.col("__f_eqis"), pl.lit(0)]) == 0).cast(pl.Int32)
              + (pl.col("__f_gm")   > 0).cast(pl.Int32)
              + (pl.col("__f_aturn")> 0).cast(pl.Int32)
            )
            .alias(name)
        )
        .with_columns(
            # only allow EQIS to be missing
            pl.when(
                pl.col("__f_roa").is_null()
                | pl.col("__f_croa").is_null()
                | pl.col("__f_droa").is_null()
                | pl.col("__f_acc").is_null()
                | pl.col("__f_lev").is_null()
                | pl.col("__f_liq").is_null()
                | pl.col("__f_gm").is_null()
                | pl.col("__f_aturn").is_null()
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col(name))
            .alias(name)
        )

        # clean up
        .drop([
            "__lev_ratio","__liq_ratio","__gm_ratio",
            "__f_roa","__f_croa","__f_droa","__f_acc",
            "__f_lev","__f_liq","__f_eqis","__f_gm","__f_aturn"
        ])
    )


    df = df.drop(['__f_roa', '__f_croa', '__f_droa', '__f_acc', '__f_lev', '__f_liq', '__f_eqis', '__f_gm', '__f_aturn',])


    return df


def ohlson_o(df, name):
    df = (df
           # Calculate __o_lat
           .with_columns((pl.col('at_x').log()).alias('__o_lat'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_lat')).alias('__o_lat'))

           # Calculate __o_lev
           .with_columns((pl.col('debt_x') / pl.col('at_x')).alias('__o_lev'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_lev')).alias('__o_lev'))

           # Calculate __o_wc
           .with_columns(((pl.col('ca_x') - pl.col('cl_x')) / pl.col('at_x')).alias('__o_wc'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_wc')).alias('__o_wc'))

           # Calculate __o_roe
           .with_columns((pl.col('nix_x') / pl.col('at_x')).alias('__o_roe'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_roe')).alias('__o_roe'))

           # Calculate __o_cacl
           .with_columns((pl.col('cl_x') / pl.col('ca_x')).alias('__o_cacl'))
           .with_columns(pl.when(pl.col('ca_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_cacl')).alias('__o_cacl'))

           # Calculate __o_ffo
           .with_columns(((pl.col('pi_x') + pl.col('dp')) / pl.col('lt')).alias('__o_ffo'))
           .with_columns(pl.when(pl.col('lt') <= 0)
                         .then(None)
                         .otherwise(pl.col('__o_ffo')).alias('__o_ffo'))

           # Calculate __o_neg_eq
           .with_columns((pl.col('lt') > pl.col('at_x')).cast(pl.Int32).alias('__o_neg_eq'))
           .with_columns(pl.when(pl.col('lt').is_null() | pl.col('at_x').is_null())
                         .then(None)
                         .otherwise(pl.col('__o_neg_eq')).alias('__o_neg_eq'))

           # Calculate __o_neg_earn
           .with_columns(((pl.col('nix_x') < 0) & (pl.col('nix_x').shift(12) < 0)).cast(pl.Int32).alias('__o_neg_earn'))
           .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('nix_x').is_null()) | (pl.col('nix_x').shift(12).is_null()))
                         .then(None)
                         .otherwise(pl.col('__o_neg_earn')).alias('__o_neg_earn'))

           # Calculate __o_nich
           .with_columns(((pl.col('nix_x') - pl.col('nix_x').shift(12)) / (pl.col('nix_x').abs() + pl.col('nix_x').shift(12).abs())).alias('__o_nich'))
           .with_columns(pl.when((pl.col('count') <= 12) | ((pl.col('nix_x').abs() + pl.col('nix_x').shift(12).abs()) == 0))
                         .then(None)
                         .otherwise(pl.col('__o_nich')).alias('__o_nich'))

           # Calculate O-score using the variables and their conditions
           .with_columns((-1.32 - 0.407 * pl.col('__o_lat') + 6.03 * pl.col('__o_lev') 
                         - 1.43 * pl.col('__o_wc') + 0.076 * pl.col('__o_cacl') 
                         - 1.72 * pl.col('__o_neg_eq') - 2.37 * pl.col('__o_roe') 
                         - 1.83 * pl.col('__o_ffo') + 0.285 * pl.col('__o_neg_earn') 
                         - 0.52 * pl.col('__o_nich')).alias(name))
         )
    return df


def altman_z(df, name):
    df = (df
            # creating helper variables
           # Calculate __z_wc
           .with_columns(((pl.col('ca_x') - pl.col('cl_x')) / pl.col('at_x')).alias('__z_wc'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_wc')).alias('__z_wc'))

           # Calculate __z_re
           .with_columns((pl.col('re') / pl.col('at_x')).alias('__z_re'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_re')).alias('__z_re'))

           # Calculate __z_eb
           .with_columns((pl.col('ebitda_x') / pl.col('at_x')).alias('__z_eb'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_eb')).alias('__z_eb'))

           # Calculate __z_sa
           .with_columns((pl.col('sale_x') / pl.col('at_x')).alias('__z_sa'))
           .with_columns(pl.when(pl.col('at_x') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_sa')).alias('__z_sa'))

           # Calculate __z_me
           .with_columns((pl.col('me_fiscal') / pl.col('lt')).alias('__z_me'))
           .with_columns(pl.when(pl.col('lt') <= 0)
                         .then(None)
                         .otherwise(pl.col('__z_me')).alias('__z_me'))

           # Calculate temporary z-score
           .with_columns((1.2 * pl.col('__z_wc') + 1.4 * pl.col('__z_re') 
                         + 3.3 * pl.col('__z_eb') + 0.6 * pl.col('__z_me') 
                         + 1.0 * pl.col('__z_sa')).alias(name))
         )
    df =df.drop(['__z_wc', '__z_re', '__z_eb', '__z_sa', '__z_me'])
    
    return df

# __chars5 =  altman_z(df=__chars5, name='z_score')

def intrinsic_value(df, name, r):
    df = (df
            # creating helper variables
           # Calculate __iv_po
           .with_columns((pl.col('div_x')/ pl.col('nix_x')).alias('__iv_po'))
           .with_columns(pl.when(pl.col('nix_x') <= 0)
                         .then((pl.col('div_x')/ (pl.col('at_x') * 0.06)))
                         .otherwise(pl.col('__iv_po')).alias('__iv_po'))

           # Calculate __iv_roe
           .with_columns((pl.col('nix_x') / ((pl.col('be_x') +  pl.col('be_x').shift(12))/2)).alias('__iv_roe'))
           .with_columns(pl.when((pl.col('count') <= 12) | ((pl.col('be_x') +  pl.col('be_x').shift(12)) <= 0))
                         .then(None)
                         .otherwise(pl.col('__iv_roe')).alias('__iv_roe'))

           # Calculate __iv_be1
           .with_columns(((1 + (1 - pl.col('__iv_po')) * pl.col('__iv_roe')) * pl.col('be_x')).alias('__iv_be1'))


           # Calculate intrinsic value
           .with_columns(( pl.col('be_x') + (((pl.col('__iv_roe') - r)/(1+ r)) * pl.col('be_x')) + (((pl.col('__iv_roe') - r)/((1+ r) * r)) * pl.col('__iv_be1'))).alias(name))
           .with_columns(pl.when(pl.col(name) <= 0)
                         .then(None)
                         .otherwise(pl.col(name)).alias(name))
         )
    df =df.drop(['__iv_po', '__iv_roe', '__iv_be1'])
    
    return df

def kz_index(df, name):

# Assume that __chars5 is your initial DataFrame and you have added the appropriate columns.
    df = (df
            # Calculate __kz_cf
            .with_columns(((pl.col('ni_x') + pl.col('dp')) / pl.col('ppent').shift(12)).alias('__kz_cf'))
            .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('ppent').shift(12) <= 0))
                          .then(None)
                          .otherwise(pl.col('__kz_cf')).alias('__kz_cf'))
            
            # Calculate __kz_q
            .with_columns(((pl.col('at_x') + pl.col('me_fiscal') - pl.col('be_x')) / pl.col('at_x')).alias('__kz_q'))
            .with_columns(pl.when(pl.col('at_x') <= 0)
                          .then(None)
                          .otherwise(pl.col('__kz_q')).alias('__kz_q'))
            
            # Calculate __kz_db
            .with_columns((pl.col('debt_x') / (pl.col('debt_x') + pl.col('seq_x'))).alias('__kz_db'))
            .with_columns(pl.when((pl.col('debt_x') + pl.col('seq_x')) == 0)
                          .then(None)
                          .otherwise(pl.col('__kz_db')).alias('__kz_db'))
            
            # Calculate __kz_dv
            .with_columns((pl.col('div_x') / pl.col('ppent').shift(12)).alias('__kz_dv'))
            .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('ppent').shift(12) <= 0))
                          .then(None)
                          .otherwise(pl.col('__kz_dv')).alias('__kz_dv'))
            
            # Calculate __kz_cs
            .with_columns((pl.col('che') / pl.col('ppent').shift(12)).alias('__kz_cs'))
            .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('ppent').shift(12) <= 0))
                          .then(None)
                          .otherwise(pl.col('__kz_cs')).alias('__kz_cs'))
            
            # Calculate the kz_index using the helper variables
            .with_columns((- 1.002 * pl.col('__kz_cf') + 0.283 * pl.col('__kz_q') 
                           + 3.139 * pl.col('__kz_db') - 39.368 * pl.col('__kz_dv') 
                           - 1.315 * pl.col('__kz_cs')).alias(name))
)

    return df

def chg_var1_to_var2(df, name, var1, var2, horizon):

    df = (df
            # Calculate __x
            .with_columns((pl.col(var1) / pl.col(var2)).alias('__x'))
            .with_columns(pl.when((pl.col(var2) <= 0))
                          .then(None)
                          .otherwise(pl.col('__x')).alias('__x')).sort(['gvkey', 'datadate'])
            
            # Calculate main
            .with_columns((pl.col('__x') - pl.col('__x').shift(horizon)).alias(name)))
            # .with_columns(pl.when(pl.col('count') <= horizon)
            #               .then(None)
            #               .otherwise(pl.col(name)).alias(name))
         
    df = df.drop('__x')

    return df




def earnings_persistence(
    df: pl.DataFrame,
    n_years: int,
    min_obs: int
) -> pl.DataFrame:

    # --- 1) PROC SORT data=&data. out=__acc1; BY gvkey curcd datadate; ---
    __acc1 = df.sort(["gvkey", "curcd", "datadate"])

    # --- 2) DATA __acc2: retain count by first.curcd; count=1 else count+1 ---
    __acc2 = __acc1.with_columns([
        # 1-based within-(gvkey,curcd) row number
        (pl.arange(0, pl.count())
           .cum_count()
           .over(["gvkey","curcd"]) + 1
        ).alias("count")
    ])

    # --- 3) DATA __acc3: compute __ni_at, __ni_at_l1 + null‑guards ---
    __acc3 = (__acc2
        # __ni_at = ni_x/at_x; null if at_x<=0
        .with_columns([
            pl.when(pl.col("at_x") > 0)
              .then(pl.col("ni_x") / pl.col("at_x"))
              .otherwise(pl.lit(None).cast(pl.Float64))
              .alias("__ni_at")
        ])
        # lag12(__ni_at) per (gvkey,curcd)
        .with_columns([
            pl.col("__ni_at")
              .shift(12)
              .over(["gvkey","curcd"])
              .alias("__ni_at_l1")
        ])
        # if count<=12 then __ni_at_l1=.
        .with_columns([
            pl.when(pl.col("count") <= 12)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("__ni_at_l1"))
              .alias("__ni_at_l1")
        ])
    )

    # --- 4) PROC SQL __acc4: keep only non‑missing pairs ---
    __acc4 = __acc3.filter(
        pl.col("__ni_at").is_not_null()
        & pl.col("__ni_at_l1").is_not_null()
    ).select(["gvkey","curcd","datadate","__ni_at","__ni_at_l1"])

    # --- 5) PROC SQL month_ends: distinct datadate, ordered ---
    month_ends = (
        __acc4
        .select("datadate")
        .unique()
        .sort("datadate")
    )

    # --- 6) %let __months = n_years*12; build dates_apply by mod(monotonic(),__months) ---
    months = n_years * 12
    dates_apply = month_ends.with_columns([
        (pl.arange(0, pl.count())
           .cum_count() % months
        ).alias("grp")
    ])

    # --- 7) Loop over grp=0…months‑1, for each build calc_dates & calc_data & regress ---
    results = []
    for grp in range(months):
        anchors = dates_apply.filter(pl.col("grp")==grp)["datadate"].to_list()

        for anchor in anchors:
            start_dt = anchor + relativedelta(years=-n_years)
            # move to end of that month
            start_dt = start_dt.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)

            window = __acc4.filter(
                (pl.col("datadate")   > start_dt)
                & (pl.col("datadate") <= anchor)
                & (pl.col("datadate").dt.month() == anchor.month)
            )

            # only those with ≥ min_obs observations per gvkey/curcd
            counts = (
                window
                .group_by(["gvkey","curcd"])
                .agg(pl.count("datadate").alias("obs"))
                .filter(pl.col("obs") >= min_obs)
            )
            calc_data = window.join(counts, on=["gvkey","curcd"], how="inner")
            # add calc_date column
            calc_data = calc_data.with_columns(
                pl.lit(anchor).alias("calc_date")
            )

            # now run the cross‑sectional regressions
            for grp_key, sub in calc_data.group_by(["gvkey","curcd","calc_date"]):
                gv, cc, calc_date = grp_key
                y = sub["__ni_at"].to_numpy()
                X = sub["__ni_at_l1"].to_numpy()
                X = sm.add_constant(X)
                fit = sm.OLS(y, X).fit()
                edf = fit.df_resid
                if edf + 2 >= min_obs:
                    ni_ar1 = float(fit.params[1])
                    ni_ivol = float(np.sqrt(fit.mse_resid * edf/(edf+1)))
                else:
                    ni_ar1 = None
                    ni_ivol = None

                results.append({
                    "gvkey":     gv,
                    "curcd":     cc,
                    "datadate":  calc_date,
                    "ni_ar1":    ni_ar1,
                    "ni_ivol":   ni_ivol
                })

    out = (
        pl.DataFrame(results)
        .sort(["gvkey","curcd","datadate"])
        .unique(subset=["gvkey","curcd","datadate"])
    )
    return out


def scale_me(df, var):
    
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    
    # Appending '_me' to the name
    name = f"{name}_me"
    
    # Scaling
    df = df.with_columns(((pl.col(var) * pl.col('fx'))/pl.col('me_company')).alias(name))
    
    return df

def scale_mev(df, var):
    # Removing '_x' from the column name
    name = var.replace('_x', '')
    
    # Appending '_me' to the name
    name = f"{name}_mev"
    
    # Scaling
    df = df.with_columns(
        ((pl.col(var) * pl.col('fx'))/pl.col('mev')).alias(name)
    )
    
    return df



def combine_ann_qtr_chars(ann_df, qtr_df, char_vars, q_suffix):
    # Create a combined DataFrame by left joining on 'gvkey' and 'public_date'
    combined_df = ann_df.join(qtr_df, left_on=['gvkey', 'public_date'], right_on=['gvkey', 'public_date'], how='left', suffix=q_suffix)
    
    # Define the logic to update annual data with quarterly data if it is more recent
    for char_var in char_vars:
        combined_df = (combined_df.with_columns(
            pl.when((pl.col(char_var).is_null()) | ((pl.col(f"{char_var}{q_suffix}").is_not_null()) & (pl.col(f"datadate{q_suffix}") > pl.col('datadate'))))
            .then(pl.col(f"{char_var}{q_suffix}"))
            .otherwise(pl.col(char_var))
            .alias(char_var)
        ))
        
        # Drop the quarterly variable after the update
        combined_df = combined_df.drop(f"{char_var}{q_suffix}")
    
    # Drop the no longer needed 'datadate' fields
    combined_df = combined_df.drop(['datadate', f'datadate{q_suffix}'])
    
    # Remove duplicates based on 'gvkey' and 'public_date' and sort the DataFrame
    combined_df = combined_df.unique(subset=['gvkey', 'public_date']).sort(['gvkey', 'public_date'])
    
    return combined_df


def var_growth(
    df: pl.DataFrame, 
    var_gr: str, 
    horizon: int  # in months
) -> pl.DataFrame:
    # 1) make the new column name
    base = var_gr.replace("_x", "")
    years = horizon / 12
    # if horizon is a multiple of 12, show as integer; else keep float
    suffix = int(years) if years.is_integer() else years
    new_col = f"{base}_gr{suffix}"

    # 2) compute growth, nulling out per your rules
    return df.with_columns([
        pl.when(
            (pl.col("count") <= horizon) 
            | (pl.col(var_gr).shift(horizon) <= 0)
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(
            pl.col(var_gr) / pl.col(var_gr).shift(horizon) - 1
        )
        .alias(new_col)
    ])


def chg_to_assets(
    df: pl.DataFrame,
    var_gra: str,
    horizon: int  # in months
) -> pl.DataFrame:
    # 1) derive base name and suffix
    base = var_gra.replace("_x", "")
    years = horizon / 12
    suffix = int(years) if years.is_integer() else years
    new_col = f"{base}_gr{suffix}a"

    # 2) compute change-to-assets, nulling per your rules
    return df.with_columns([
        pl.when(
            (pl.col("count") <= horizon)
            | (pl.col("at_x") <= 0)
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(
            (pl.col(var_gra) - pl.col(var_gra).shift(horizon))
            / pl.col("at_x")
        )
        .alias(new_col)
    ])



def expand(data, id_vars, start_date, end_date, freq='day', new_date_name='date'):
    if freq =='day':
        __expanded = data.with_columns((pl.date_ranges(start=start_date, end=end_date, interval='1d')).alias('date_range')).rename({"date_range": new_date_name}).explode(new_date_name).drop([start_date, end_date])
    elif freq == 'month':
         __expanded = data.with_columns((pl.date_ranges(start=start_date, end=end_date, interval='1mo')).alias('date_range')).rename({"date_range": new_date_name}).explode(new_date_name).with_columns(pl.col(new_date_name).dt.month_end()).drop([start_date, end_date])


    __expanded = __expanded.sort(id_vars + [new_date_name]).unique(id_vars + [new_date_name])
    return __expanded

#chars list
# lag_to_public=4
# max_data_lag=18
acc_chars = [
    # Accounting Based Size Measures
    "assets", "sales", "book_equity", "net_income", "enterprise_value",
    
    # 1yr Growth
    "at_gr1", "ca_gr1", "nca_gr1", "lt_gr1", "cl_gr1", "ncl_gr1", "be_gr1", "pstk_gr1", "debt_gr1",
    "sale_gr1", "cogs_gr1", "sga_gr1", "opex_gr1",
    
    # 3yr Growth
    "at_gr3", "ca_gr3", "nca_gr3", "lt_gr3", "cl_gr3", "ncl_gr3", "be_gr3", "pstk_gr3", "debt_gr3",
    "sale_gr3", "cogs_gr3", "sga_gr3", "opex_gr3",
    
    # 1yr Growth Scaled by Assets
    "cash_gr1a", "inv_gr1a", "rec_gr1a", "ppeg_gr1a", "lti_gr1a", "intan_gr1a", "debtst_gr1a", "ap_gr1a",
    "txp_gr1a", "debtlt_gr1a", "txditc_gr1a", "coa_gr1a", "col_gr1a", "cowc_gr1a", "ncoa_gr1a", "ncol_gr1a", "nncoa_gr1a",
    "oa_gr1a", "ol_gr1a", "noa_gr1a", "fna_gr1a", "fnl_gr1a", "nfna_gr1a", "gp_gr1a", "ebitda_gr1a", "ebit_gr1a",
    "ope_gr1a", "ni_gr1a", "nix_gr1a", "dp_gr1a", "ocf_gr1a", "fcf_gr1a", "nwc_gr1a",
    "eqnetis_gr1a", "dltnetis_gr1a", "dstnetis_gr1a", "dbnetis_gr1a", "netis_gr1a", "fincf_gr1a", "eqnpo_gr1a",
    "tax_gr1a", "div_gr1a", "eqbb_gr1a", "eqis_gr1a", "eqpo_gr1a", "capx_gr1a",
    
    # 3yr Growth Scaled by Assets
    "cash_gr3a", "inv_gr3a", "rec_gr3a", "ppeg_gr3a", "lti_gr3a", "intan_gr3a", "debtst_gr3a", "ap_gr3a",
    "txp_gr3a", "debtlt_gr3a", "txditc_gr3a", "coa_gr3a", "col_gr3a", "cowc_gr3a", "ncoa_gr3a", "ncol_gr3a", "nncoa_gr3a",
    "oa_gr3a", "ol_gr3a", "fna_gr3a", "fnl_gr3a", "nfna_gr3a", "gp_gr3a", "ebitda_gr3a", "ebit_gr3a",
    "ope_gr3a", "ni_gr3a", "nix_gr3a", "dp_gr3a", "ocf_gr3a", "fcf_gr3a", "nwc_gr3a",
    "eqnetis_gr3a", "dltnetis_gr3a", "dstnetis_gr3a", "dbnetis_gr3a", "netis_gr3a", "fincf_gr3a", "eqnpo_gr3a",
    "tax_gr3a", "div_gr3a", "eqbb_gr3a", "eqis_gr3a", "eqpo_gr3a", "capx_gr3a",

    # "noa_gr3a"
    # Investment
    "capx_at", "rd_at",
    
    # Profitability
    "gp_sale", "ebitda_sale", "ebit_sale", "pi_sale", "ni_sale", "nix_sale", "ocf_sale", "fcf_sale",
    "gp_at", "ebitda_at", "ebit_at", "fi_at", "cop_at",
    "ope_be", "ni_be", "nix_be", "ocf_be", "fcf_be",
    "gp_bev", "ebitda_bev", "ebit_bev", "fi_bev", "cop_bev",
    "gp_ppen", "ebitda_ppen", "fcf_ppen",
    
    # Issuance
    "fincf_at", "netis_at", "eqnetis_at", "eqis_at", "dbnetis_at", "dltnetis_at", "dstnetis_at",
    
    # Equity Payout
    "eqnpo_at", "eqbb_at", "div_at",
    
    # Accruals
    "oaccruals_at", "oaccruals_ni", "taccruals_at", "taccruals_ni", "noa_at",
    
    # Capitalization/Leverage Ratios
    "be_bev", "debt_bev", "cash_bev", "pstk_bev", "debtlt_bev", "debtst_bev",
    "debt_mev", "pstk_mev", "debtlt_mev", "debtst_mev",
    
    # Financial Soundness Ratios
    "int_debtlt", "int_debt", "cash_lt", "inv_act", "rec_act",
    "ebitda_debt", "debtst_debt", "cl_lt", "debtlt_debt", "profit_cl", "ocf_cl",
    "ocf_debt", "lt_ppen", "debtlt_be", "fcf_ocf",
    "opex_at", "nwc_at",
    
    # Solvency Ratios
    "debt_at", "debt_be", "ebit_int",
    
    # Liquidity Ratios
    "cash_cl", "caliq_cl", "ca_cl",
    "inv_days", "rec_days", "ap_days", "cash_conversion",
    
    # Activity/Efficiency Ratio
    "inv_turnover", "at_turnover", "rec_turnover", "ap_turnover",
    
    # Non-Recurring Items
    "spi_at", "xido_at", "nri_at",
    
    # Miscellaneous
    "adv_sale", "staff_sale", "rd_sale", "div_ni", "sale_bev", "sale_be", "sale_nwc", "tax_pi",
    
    # Balance Sheet Fundamentals to Market Equity
    "be_me", "at_me", "cash_me",
    
    # Income Fundamentals to Market Equity
    "gp_me", "ebitda_me", "ebit_me", "ope_me", "ni_me", "nix_me", "sale_me", "ocf_me", "fcf_me", "cop_me",
    "rd_me",
    
    # Equity Payout/issuance to Market Equity
    "div_me", "eqbb_me", "eqis_me", "eqpo_me", "eqnpo_me", "eqnetis_me",
    
    # Debt Issuance to Market Enterprise Value
    "dltnetis_mev", "dstnetis_mev", "dbnetis_mev",
    
    # Firm Payout/issuance to Market Enterprise Value
    "netis_mev",
    
    # Balance Sheet Fundamentals to Market Enterprise Value
    "at_mev", "be_mev", "bev_mev", "ppen_mev", "cash_mev",
    
    # Income/CF Fundamentals to Market Enterprise Value
    "gp_mev", "ebitda_mev", "ebit_mev", "cop_mev", "sale_mev", "ocf_mev", "fcf_mev", "fincf_mev",
    
    # New Variables from HXZ
    "ni_inc8q", "ppeinv_gr1a", "lnoa_gr1a", "capx_gr1", "capx_gr2", "capx_gr3", "sti_gr1a",
    "niq_at", "niq_at_chg1", "niq_be", "niq_be_chg1", "saleq_gr1", "rd5_at",
    "dsale_dinv", "dsale_drec", "dgp_dsale", "dsale_dsga",
    "saleq_su", "niq_su", "debt_me", "netdebt_me", "capex_abn", "inv_gr1", "be_gr1a",
    "op_at", "pi_nix", "op_atl1", "gp_atl1", "ope_bel1", "cop_atl1",
    "at_be", "ocfq_saleq_std",
    "aliq_at", "aliq_mat", "tangibility",
    "eq_dur", "f_score", "o_score", "z_score", "kz_index", "intrinsic_value", "ival_me",
    "sale_emp_gr1", "emp_gr1", "cash_at",
    "earnings_variability", "ni_ar1", "ni_ivol",

    #New Variables not in HXZ
    "niq_saleq_std", "ni_emp", "sale_emp", "ni_at",
    "ocf_at", "ocf_at_chg1", "roeq_be_std", "roe_be_std",
    "gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5"

]

#function:

def create_acc_chars(data, lag_to_public, max_data_lag, __keep_vars, me_data, suffix):    
    
    __chars3 = data.sort(['gvkey', 'curcd', 'datadate'])
    
    #adding a count column that keeps a count of the number of the obs for a given gvkey (and curcd)
    __chars4=__chars3.sort(["gvkey","curcd","datadate"]).with_columns(
            pl.col("datadate")
              .rank(method="ordinal")
              .over(["gvkey","curcd"])
              .cast(pl.Int32)
              .alias("count")
        )
    
    
    #accounting based size measures
    __chars5 = (__chars4
        .with_columns(pl.col("at_x").alias("assets"))
       .with_columns(pl.col("sale_x").alias("sales"))
       .with_columns(pl.col("be_x").alias("book_equity"))
       .with_columns(pl.col("ni_x").alias("net_income")))
    
    
    #growth characteristics
    growth_vars = [
    "at_x", "ca_x", "nca_x",                 # Assets - Aggregated
    "lt", "cl_x", "ncl_x",                   # Liabilities - Aggregated
    "be_x", "pstk_x", "debt_x",              # Financing Book Values
    "sale_x", "cogs", "xsga", "opex_x",      # Sales and Operating Costs
    "capx", "invt"
    ]
    
    #1-yr growth
    for i in growth_vars:
        __chars5 = var_growth(df=__chars5, var_gr=i, horizon=12)
    
    #3-yr growth
    for i in growth_vars:
        __chars5 = var_growth(df=__chars5, var_gr=i, horizon=36)
    
    
    #Change Scaled by Asset Characteristics 
    
    ch_asset_vars = [
        # Assets - Individual Items
        "che", "invt", "rect", "ppegt", "ivao", "ivst", "intan",
        
        # Liabilities - Individual Items
        "dlc", "ap", "txp", "dltt", "txditc",
        
        # Operating Assets/Liabilities
        "coa_x", "col_x", "cowc_x", "ncoa_x", "ncol_x", "nncoa_x", "oa_x", "ol_x",
        
        # Financial Assets/Liabilities
        "fna_x", "fnl_x", "nfna_x",
        
        # Income Statement
        "gp_x", "ebitda_x", "ebit_x", "ope_x", "ni_x", "nix_x", "dp",
        
        # Aggregated Cash Flow
        "fincf_x", "ocf_x", "fcf_x", "nwc_x",
        
        # Financing Cash Flow
        "eqnetis_x", "dltnetis_x", "dstnetis_x", "dbnetis_x", "netis_x", "eqnpo_x",
        
        # Tax Change
        "txt",
        
        # Financing Cash Flow
        "eqbb_x", "eqis_x", "div_x", "eqpo_x",
        
        # Other
        "capx", "be_x"
    ]
    
    
    
    #1yr Change Scaled by Assets
    for i in ch_asset_vars:
        __chars5 = chg_to_assets(df=__chars5, var_gra=i, horizon=12)
    
    
    #3yr Change Scaled by Assets
    for i in ch_asset_vars:
        __chars5 = chg_to_assets(df=__chars5, var_gra=i, horizon=36)
    
    
    #Investment Measure
    __chars5 = __chars5.with_columns([
        # capx_at: null if at_x == 0, else capx/at_x
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("capx") / pl.col("at_x"))
          .alias("capx_at"),
    
        # rd_at:   null if at_x == 0, else xrd/at_x
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("xrd") / pl.col("at_x"))
          .alias("rd_at"),
    ])
    
    
    #Non-Recurring Items
    __chars5 = __chars5.with_columns([
    pl.when(pl.col("at_x") == 0)
      .then(pl.lit(None).cast(pl.Float64))
      .otherwise(pl.col("spi") / pl.col("at_x"))
      .alias("spi_at"),
    
    # rd_at:   null if at_x == 0, else xrd/at_x
    pl.when(pl.col("at_x") == 0)
      .then(pl.lit(None).cast(pl.Float64))
      .otherwise(pl.col("xido_x") / pl.col("at_x"))
      .alias("xido_at"),
    
    pl.when(pl.col("at_x") == 0)
      .then(pl.lit(None).cast(pl.Float64))
      .otherwise((pl.col("spi")+pl.col("xido_x")) / pl.col("at_x"))
      .alias("nri_at")
    
        
    ])
    
    
    #profitability margins
    __chars5 = __chars5.with_columns([
        # Gross Profit Margin
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("gp_x") / pl.col("sale_x"))
          .alias("gp_sale"),
    
        # Operating Profit Margin before Depreciation
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ebitda_x") / pl.col("sale_x"))
          .alias("ebitda_sale"),
    
        # Operating Profit Margin after Depreciation
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ebit_x") / pl.col("sale_x"))
          .alias("ebit_sale"),
    
        # Pretax Profit Margin
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("pi_x") / pl.col("sale_x"))
          .alias("pi_sale"),
    
        # Net Profit Margin Before XI
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ni_x") / pl.col("sale_x"))
          .alias("ni_sale"),
    
        # Net Profit Margin
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ni") / pl.col("sale_x"))
          .alias("nix_sale"),
    
        # Operating Cash Flow Margin
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ocf_x") / pl.col("sale_x"))
          .alias("ocf_sale"),
    
        # Free Cash Flow Margin
        pl.when(pl.col("sale_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("fcf_x") / pl.col("sale_x"))
          .alias("fcf_sale"),
    ])
    
    
    
    
    #Return on assets:
    __chars5 = __chars5.with_columns([
        # gp_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("gp_x") / pl.col("at_x"))
          .alias("gp_at"),
    
        # ebitda_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ebitda_x") / pl.col("at_x"))
          .alias("ebitda_at"),
    
        # ebit_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ebit_x") / pl.col("at_x"))
          .alias("ebit_at"),
    
        # fi_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("fi_x") / pl.col("at_x"))
          .alias("fi_at"),
    
        # cop_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("cop_x") / pl.col("at_x"))
          .alias("cop_at"),
    
        # ni_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ni_x") / pl.col("at_x"))
          .alias("ni_at"),
    ])
    
    
    
    
    #Return on book equity:
    __chars5 = (__chars5
            .with_columns((pl.col("ope_x")/pl.col('be_x')).alias("ope_be"))
            .with_columns((pl.col("ni_x")/pl.col('be_x')).alias("ni_be"))
            .with_columns((pl.col("nix_x")/pl.col('be_x')).alias("nix_be"))
            .with_columns((pl.col("ocf_x")/pl.col('be_x')).alias("ocf_be"))
            .with_columns((pl.col("fcf_x")/pl.col('be_x')).alias("fcf_be")))
    
    
    #Return on invested book capital:
    __chars5 = (__chars5
            .with_columns((pl.col("gp_x")/pl.col('bev_x')).alias("gp_bev"))
            .with_columns((pl.col("ebitda_x")/pl.col('bev_x')).alias("ebitda_bev"))
            .with_columns((pl.col("ebit_x")/pl.col('bev_x')).alias("ebit_bev"))                            #Pre tax Return on Book Enterprise Value
            .with_columns((pl.col("fi_x")/pl.col('bev_x')).alias("fi_bev"))                                #ROIC
            .with_columns((pl.col("cop_x")/pl.col('bev_x')).alias("cop_bev")))                             #Cash Based Operating Profit to Invested Capital
    
    
    #Return on Physical Capital:
    __chars5 = __chars5.with_columns([
        # gp_ppen
        pl.when(pl.col("ppent") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("gp_x") / pl.col("ppent"))
          .alias("gp_ppen"),
    
        # ebitda_ppen
        pl.when(pl.col("ppent") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ebitda_x") / pl.col("ppent"))
          .alias("ebitda_ppen"),
    
        # fcf_ppen
        pl.when(pl.col("ppent") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("fcf_x") / pl.col("ppent"))
          .alias("fcf_ppen"),
    ])
    
    
    
    
    
    #Issuance Variables:
    __chars5 = __chars5.with_columns([
        # fincf_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("fincf_x") / pl.col("at_x"))
          .alias("fincf_at"),
    
        # netis_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("netis_x") / pl.col("at_x"))
          .alias("netis_at"),
    
        # eqnetis_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("eqnetis_x") / pl.col("at_x"))
          .alias("eqnetis_at"),
    
        # eqis_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("eqis_x") / pl.col("at_x"))
          .alias("eqis_at"),
    
        # dbnetis_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("dbnetis_x") / pl.col("at_x"))
          .alias("dbnetis_at"),
    
        # dltnetis_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("dltnetis_x") / pl.col("at_x"))
          .alias("dltnetis_at"),
    
        # dstnetis_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("dstnetis_x") / pl.col("at_x"))
          .alias("dstnetis_at"),
    ])
    
    
    
    #Equity Payout:
    __chars5 = __chars5.with_columns([
        # eqnpo_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("eqnpo_x") / pl.col("at_x"))
          .alias("eqnpo_at"),
    
        # eqbb_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("eqbb_x") / pl.col("at_x"))
          .alias("eqbb_at"),
    
        # div_at
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("div_x") / pl.col("at_x"))
          .alias("div_at"),
    ])
    
    
    
    #accruals:
    __chars5 = (__chars5
           .with_columns([
        # Operating Accruals to Assets
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("oacc_x") / pl.col("at_x"))
          .alias("oaccruals_at"),
    
        # Operating Accruals to |Net Income|
        pl.when(pl.col("nix_x").abs() == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("oacc_x") / pl.col("nix_x").abs())
          .alias("oaccruals_ni"),
    
        # Total Accruals to Assets
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("tacc_x") / pl.col("at_x"))
          .alias("taccruals_at"),
    
        # Total Accruals to |Net Income|
        pl.when(pl.col("nix_x").abs() == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("tacc_x") / pl.col("nix_x").abs())
          .alias("taccruals_ni"),
    
        # Net Operating Assets to Total Assets (lagged)
        pl.when((pl.col("count") <= 12) | (pl.col("at_x").shift(12) <= 0))
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("noa_x") / pl.col("at_x").shift(12))
          .alias("noa_at"),
    ])
            .with_columns(
                pl.when((pl.col("count") <= 12) | (pl.col("at_x").shift(12) <= 0))
                .then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col("noa_at")).alias("noa_at"))
    )
    
    
    
    #Capitalization/Leverage Ratios Book:
    __chars5 = (__chars5
            .with_columns((pl.col("be_x")/pl.col('bev_x')).alias("be_bev"))                                         # Common Equity as % of Book Enterprise Value
            .with_columns((pl.col("debt_x")/pl.col('bev_x')).alias("debt_bev"))                                     #Total Debt as % of Book Enterprise Value
            .with_columns((pl.col("che")/pl.col('bev_x')).alias("cash_bev"))                                        #Cash and Short-Term Investments to Book Enterprise Value 
            .with_columns((pl.col("pstk_x")/pl.col('bev_x')).alias("pstk_bev"))                                     #Prefered Stock to Book Enterprise Value 
            .with_columns((pl.col("dltt")/pl.col('bev_x')).alias("debtlt_bev"))                                     #Long-term debt as % of Book Enterprise Value
            .with_columns((pl.col("dlc")/pl.col('bev_x')).alias("debtst_bev")))                                     #Short-term debt as % of Book Enterprise Value
    
    
    
    # #Financial Soundness Ratios:
    
    
    __chars5 = __chars5.with_columns([
        # Interest as % of average total debt
        pl.when(pl.col("debt_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("xint") / pl.col("debt_x"))
          .alias("int_debt"),
    
        # Interest as % of average long-term debt
        pl.when(pl.col("dltt") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("xint") / pl.col("dltt"))
          .alias("int_debtlt"),
    
        # Ebitda to total debt
        pl.when(pl.col("debt_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ebitda_x") / pl.col("debt_x"))
          .alias("ebitda_debt"),
    
        # Profit before D&A to current liabilities
        pl.when(pl.col("cl_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ebitda_x") / pl.col("cl_x"))
          .alias("profit_cl"),
    
        # Operating cash flow to current liabilities
        pl.when(pl.col("cl_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ocf_x") / pl.col("cl_x"))
          .alias("ocf_cl"),
    
        # Operating cash flow to total debt
        pl.when(pl.col("debt_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("ocf_x") / pl.col("debt_x"))
          .alias("ocf_debt"),
    
        # Cash balance to Total Liabilities
        pl.when(pl.col("lt") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("che") / pl.col("lt"))
          .alias("cash_lt"),
    
        # inventory as % of current assets
        pl.when(pl.col("act") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("invt") / pl.col("act"))
          .alias("inv_act"),
    
        # receivables as % of current assets
        pl.when(pl.col("act") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("rect") / pl.col("act"))
          .alias("rec_act"),
    
        # short‑term debt as % of total debt
        pl.when(pl.col("debt_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("dlc") / pl.col("debt_x"))
          .alias("debtst_debt"),
    
        # current liabilities as % of total liabilities
        pl.when(pl.col("lt") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("cl_x") / pl.col("lt"))
          .alias("cl_lt"),
    
        # long‑term debt as % of total liabilities
        pl.when(pl.col("debt_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("dltt") / pl.col("debt_x"))
          .alias("debtlt_debt"),
    
        # total liabilities to total tangible assets
        pl.when(pl.col("ppent") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("lt") / pl.col("ppent"))
          .alias("lt_ppen"),
    
        # long‑term debt to book equity
        pl.when(pl.col("be_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("dltt") / pl.col("be_x"))
          .alias("debtlt_be"),
    
        # Operating Leverage ala Novy‑Marx (2011)
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("opex_x") / pl.col("at_x"))
          .alias("opex_at"),
    
        # Net working capital to assets
        pl.when(pl.col("at_x") == 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("nwc_x") / pl.col("at_x"))
          .alias("nwc_at"),
    
        # Free Cash Flow/Operating Cash Flow
        pl.when(pl.col("ocf_x") <= 0)
          .then(pl.lit(None).cast(pl.Float64))
          .otherwise(pl.col("fcf_x") / pl.col("ocf_x"))
          .alias("fcf_ocf"),
    ])
    
    
    
    #Solvency Ratios:
    __chars5 = (__chars5
            .with_columns(pl.when(pl.col('at_x')==0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col("debt_x")/pl.col('at_x')).alias("debt_at"))                                      #Debt-to-assets
            .with_columns((pl.col("debt_x")/pl.col('be_x')).alias("debt_be"))                                       #debt to shareholders' equity ratio
            .with_columns(pl.when(pl.col('xint')==0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col("ebit_x")/pl.col('xint')).alias("ebit_int")))                                     #interest coverage ratio
    
    
    
    
    # #Liquidity Ratios:
    
    __chars5 = (
        __chars5
        # Days Inventory Outstanding 
        .with_columns(
            (
                (pl.concat_list(["invt", pl.col("invt").shift(12)]).list.mean() / pl.col("cogs"))
                * 365
            )
            .alias("inv_days")
        )
        .with_columns(
            pl.when(pl.col("cogs") == 0)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("inv_days"))
              .alias("inv_days")
        )
        .with_columns(
            pl.when(pl.col("count") < 12)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("inv_days"))
              .alias("inv_days")
        )
        # Days Sales Outstanding
        .with_columns(
            (
                (pl.concat_list(["rect", pl.col("rect").shift(12)]).list.mean() / pl.col("sale_x"))
                * 365
            )
            .alias("rec_days")
        )
        .with_columns(
            pl.when(pl.col("sale_x") == 0)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("rec_days"))
              .alias("rec_days")
        )
        .with_columns(
            pl.when(pl.col("count") < 12)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("rec_days"))
              .alias("rec_days")
        )
        # Days Accounts Payable Outstanding
        .with_columns(
            (
                (pl.concat_list(["ap", pl.col("ap").shift(12)]).list.mean() / pl.col("cogs"))
                * 365
            )
            .alias("ap_days")
        )
        .with_columns(
            pl.when(pl.col("cogs") == 0)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("ap_days"))
              .alias("ap_days")
        )
        .with_columns(
            pl.when(pl.col("count") < 12)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("ap_days"))
              .alias("ap_days")
        )
        # Cash Conversion Cycle
        .with_columns(
            (pl.col("inv_days") + pl.col("rec_days") - pl.col("ap_days"))
            .alias("cash_conversion")
        )
        .with_columns(
            pl.when(pl.col("cash_conversion") < 0)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("cash_conversion"))
              .alias("cash_conversion")
        )
        # Cash Ratio
        .with_columns(
            pl.when(pl.col("cl_x") <= 0)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("che") / pl.col("cl_x"))
              .alias("cash_cl")
        )
        # Quick Ratio (acid test)
        .with_columns(
            pl.when(pl.col("cl_x") <= 0)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("caliq_x") / pl.col("cl_x"))
              .alias("caliq_cl")
        )
        # Current Ratio
        .with_columns(
            pl.when(pl.col("cl_x") <= 0)
              .then(pl.lit(None).cast(pl.Float64))
              .otherwise(pl.col("ca_x") / pl.col("cl_x"))
              .alias("ca_cl")
        )
    )
    
    
    
    # Activity/Efficiency Ratios with zero‐denominator guard
    __chars5 = __chars5.with_columns([
        # Inventory Turnover
        pl.when(
            (pl.col("count") <= 12)
            | (pl.concat_list([pl.col("invt"), pl.col("invt").shift(12)]).list.mean() == 0)
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(
            pl.col("cogs")
            / pl.concat_list([pl.col("invt"), pl.col("invt").shift(12)]).list.mean()
        )
        .alias("inv_turnover"),
    
        # Asset Turnover
        pl.when(
            (pl.col("count") <= 12)
            | (pl.concat_list([pl.col("at_x"), pl.col("at_x").shift(12)]).list.mean() == 0)
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(
            pl.col("sale_x")
            / pl.concat_list([pl.col("at_x"), pl.col("at_x").shift(12)]).list.mean()
        )
        .alias("at_turnover"),
    
        # Receivables Turnover
        pl.when(
            (pl.col("count") <= 12)
            | (pl.concat_list([pl.col("rect"), pl.col("rect").shift(12)]).list.mean() == 0)
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(
            pl.col("sale_x")
            / pl.concat_list([pl.col("rect"), pl.col("rect").shift(12)]).list.mean()
        )
        .alias("rec_turnover"),
    
        # Accounts Payables Turnover
        pl.when(
            (pl.col("count") <= 12)
            | (pl.concat_list([pl.col("ap"), pl.col("ap").shift(12)]).list.mean() == 0)
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(
            (pl.col("cogs") + pl.col("invt").diff(12))
            / pl.concat_list([pl.col("ap"), pl.col("ap").shift(12)]).list.mean()
        )
        .alias("ap_turnover"),
    ])
    
    
    if suffix!="":
        __chars5 = (__chars5.with_columns([
            pl.lit(None).cast(pl.Float64).alias('xad'),
             pl.lit(None).cast(pl.Float64).alias('xlr'),
             pl.lit(None).cast(pl.Float64).alias('emp')
        ]))
    
    #Miscellaneous Ratios
    __chars5 = (__chars5
                #advertising as % of sales
               .with_columns(pl.when(pl.col('sale_x')==0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('xad') / pl.col('sale_x')).alias('adv_sale')) 
                #labor expense as % of sales
               .with_columns(pl.when(pl.col('sale_x')==0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('xlr') / pl.col('sale_x')).alias('staff_sale'))
                #sale per $ Book Enterprise Value
               .with_columns((pl.col('sale_x') / pl.col('bev_x')).alias('sale_bev'))
               .with_columns(pl.when(pl.col('sale_x')==0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('xrd') / pl.col('sale_x')).alias('rd_sale'))
                #sales per $ total stockholders' equity
               .with_columns((pl.col('sale_x') / pl.col('be_x')).alias('sale_be'))
               
               # Calculate div_ni and apply condition
               .with_columns((pl.col('div_x') / pl.coalesce(['nix_x','ni_x'])).alias('div_ni'))
               .with_columns(pl.when(pl.coalesce(['nix_x','ni_x']) <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('div_ni')).alias('div_ni'))
               
               # Calculate sale_nwc and apply condition
               .with_columns((pl.col('sale_x') / pl.col('nwc_x')).alias('sale_nwc'))
               .with_columns(pl.when(pl.col('nwc_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('sale_nwc')).alias('sale_nwc'))
               
               # Calculate tax_pi and apply condition
               .with_columns((pl.col('txt') / pl.col('pi_x')).alias('tax_pi'))
               .with_columns(pl.when(pl.col('pi_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('tax_pi')).alias('tax_pi'))
              )
    
    
    #New variables:
    __chars5 = (__chars5
         
               # Calculate cash_at and apply condition
               .with_columns((pl.col('che') / pl.col('at_x')).alias('cash_at'))  #emp not available in quarterly data created by sas code.
               .with_columns(pl.when(pl.col('at_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('cash_at')).alias('cash_at'))
               
               # Calculate ni_emp and apply condition
               .with_columns((pl.col('ni_x') / pl.col('emp')).alias('ni_emp'))
               .with_columns(pl.when(pl.col('emp') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('ni_emp')).alias('ni_emp'))
    
               # Calculate sale_emp and apply condition
               .with_columns((pl.col('sale_x') / pl.col('emp')).alias('sale_emp'))
               .with_columns(pl.when(pl.col('emp') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('sale_emp')).alias('sale_emp'))
    
               # Calculate sale_emp_gr1 and apply condition
               .with_columns(((pl.col('sale_emp') / pl.col('sale_emp').shift(12)) - 1).alias('sale_emp_gr1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('sale_emp').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('sale_emp_gr1')).alias('sale_emp_gr1'))
    
               # Calculate emp_gr1 and apply condition
               .with_columns(((pl.col('emp') - pl.col('emp').shift(12)) / (0.5 * pl.col('emp') + 0.5 * pl.col('emp').shift(12))).alias('emp_gr1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('emp_gr1') == 0) | ((0.5 * pl.col('emp') + 0.5 * pl.col('emp').shift(12)) == 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('emp_gr1')).alias('emp_gr1'))
              )
    
    
    
    # #Number of Consecutive Earnings Increases:

    
    # sort so shifts reset per BY‐group
    __chars5 = __chars5.sort(["gvkey", "curcd", "datadate"])
    
    # compute ni_inc exactly like SAS
    __chars5 = __chars5.with_columns(
        pl.when(
            pl.col("ni_x").is_null() | pl.col("ni_x").shift(12).is_null()
        )
        .then(None)
        .otherwise((pl.col("ni_x") > pl.col("ni_x").shift(12)).cast(pl.UInt8))
        .alias("ni_inc")
    )
    
    # build boolean lags at 0,3,6,…,21 and count consecutive 1’s by prefix‐AND
    lags = [(pl.col("ni_inc").shift(i * 3) == 1) for i in range(8)]
    prefix = []
    for i, expr in enumerate(lags):
        prefix.append(expr if i == 0 else prefix[-1] & expr)
    inc8q_expr = sum(expr.cast(pl.UInt8) for expr in prefix)
    __chars5 = __chars5.with_columns(inc8q_expr.alias("ni_inc8q"))
    
    # compute n_ni_inc = sum(not missing(ni_inc)) over same lags
    __chars5 = __chars5.with_columns(
        apply_to_lastq(
            pl.col("ni_inc").is_not_null().cast(pl.UInt8),
            qtrs=8,
            func="sum"
        ).alias("n_ni_inc")
    )
    
    #blank out ni_inc8q if any SAS condition holds
    __chars5 = __chars5.with_columns(
        pl.when(
            pl.col("ni_inc").is_null() |
            (pl.col("n_ni_inc") != 8) |
            (pl.col("count") < 33)
        )
        .then(None)
        .otherwise(pl.col("ni_inc8q"))
        .alias("ni_inc8q")
    )
    
    # clean up helper cols
    __chars5 = __chars5.drop(["ni_inc", "n_ni_inc"])

    
    
    
    
    #1yr Change Scaled by Lagged Assets
    ch_asset_lag_vars = ['noa_x', 'ppeinv_x']
    for i in ch_asset_lag_vars:
        __chars5 = chg_to_lagassets(df=__chars5, var_gr=i)
    
    
    #1yr Change Scaled by Average Assets
    ch_asset_avg_vars = ['lnoa_x']
    for i in ch_asset_avg_vars:
        __chars5 = chg_to_avgassets(df=__chars5, vars_x=[i])
    
    
    #CAPEX growth over 2 years
    __chars5 = var_growth(df=__chars5, var_gr='capx', horizon=24)
    
    
    #Quarterly profitability measures:
    __chars5 = (__chars5
                
               # Calculate saleq_gr1 and apply condition
               .with_columns(((pl.col('sale_qtr') / pl.col('sale_qtr').shift(12)) - 1).alias('saleq_gr1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('sale_qtr').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('saleq_gr1')).alias('saleq_gr1'))
    
               # Calculate niq_be and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('be_x').shift(3)).alias('niq_be'))
               .with_columns(pl.when((pl.col('count') <= 3) | (pl.col('be_x').shift(3) < 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_be')).alias('niq_be'))
    
               # Calculate niq_at and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('at_x').shift(3)).alias('niq_at'))
               .with_columns(pl.when((pl.col('count') <= 3) | (pl.col('at_x').shift(3) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_at')).alias('niq_at'))
    
               # Calculate niq_be_chg1 and apply condition
               .with_columns((pl.col('niq_be') - pl.col('niq_be').shift(12)).alias('niq_be_chg1'))
               .with_columns(pl.when(pl.col('count') <= 12)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_be_chg1')).alias('niq_be_chg1'))
    
               # Calculate niq_at_chg1 and apply condition
               .with_columns((pl.col('niq_at') - pl.col('niq_at').shift(12)).alias('niq_at_chg1'))
               .with_columns(pl.when(pl.col('count') <= 12)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('niq_at_chg1')).alias('niq_at_chg1'))
              )
    
    
    #R&D capital-to-assets
    __chars5 = (__chars5
    
               # Calculate rd5_at
               .with_columns(
                   ((pl.col('xrd') + pl.col('xrd').shift(12) * 0.8 + pl.col('xrd').shift(24) * 0.6 + pl.col('xrd').shift(36) * 0.4 + pl.col('xrd').shift(48) * 0.2) / pl.col('at_x')).alias('rd5_at')
               )
               # Apply condition to rd5_at
               .with_columns(
                   pl.when((pl.col('count') <= 48) | (pl.col('at_x') <= 0))
                   .then(pl.lit(None).cast(pl.Float64))
                   .otherwise(pl.col('rd5_at')).alias('rd5_at'))
               )
    
    
    
    #Abarbanell and Bushee (1998)
    ch_asset_AandB = ['sale_x', 'invt', 'rect', 'gp_x', 'xsga']
    for i in ch_asset_AandB:
        __chars5 = chg_to_exp(df=__chars5, var_ce=i)
    
    
    __chars5 = (__chars5
    
               # Calculate dsale_dinv
               .with_columns((pl.col('sale_ce') - pl.col('invt_ce')).alias('dsale_dinv'))
    
               # Calculate dsale_drec
               .with_columns((pl.col('sale_ce') - pl.col('rect_ce')).alias('dsale_drec'))
    
               # Calculate dgp_dsale
               .with_columns((pl.col('gp_ce') - pl.col('sale_ce')).alias('dgp_dsale'))
    
               # Calculate dsale_dsga
               .with_columns((pl.col('sale_ce') - pl.col('xsga_ce')).alias('dsale_dsga'))
              ).drop(['sale_ce', 'invt_ce', 'rect_ce', 'gp_ce', 'xsga_ce'])
    
    
    
    #Earnings and Revenue 'Surpise'
    __chars5 = standardized_unexpected(df=__chars5, var='sale_qtr', qtrs=8, qtrs_min=6)
    __chars5 = standardized_unexpected(df=__chars5, var='ni_qtr', qtrs=8, qtrs_min=6)
    
    
    
    #Abnormal Corporate Investment
    __chars5 = (__chars5
               # Calculate __capex_sale and its condition
               .with_columns((pl.col('capx') / pl.col('sale_x')).alias('__capex_sale'))
               .with_columns(pl.when(pl.col('sale_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__capex_sale')).alias('__capex_sale'))
                .with_columns((
                       (pl.col('__capex_sale').shift(12) +
                        pl.col('__capex_sale').shift(24) +
                        pl.col('__capex_sale').shift(36)) / 3).alias('__capex_sale_div'))
    
               # Calculate capex_abn
               .with_columns(
                   ((pl.col('__capex_sale') / pl.col('__capex_sale_div')) - 1).alias('capex_abn'))
               # Apply condition to capex_abn
               .with_columns(
                   pl.when((pl.col('count') <= 36) | (pl.col('__capex_sale_div')==0))
                   .then(pl.lit(None).cast(pl.Float64))
                   .otherwise(pl.col('capex_abn'))
               .alias('capex_abn'))
                
               .drop(['__capex_sale', '__capex_sale_div']))
    
    
    
    #Profit scaled by lagged 
    __chars5 = (__chars5
    
               # Calculate op_atl1 and apply its conditions
               .with_columns((pl.col('op_x') / pl.col('at_x').shift(12)).alias('op_atl1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('op_atl1')).alias('op_atl1'))
    
               # Calculate gp_atl1 and apply its conditions
               .with_columns((pl.col('gp_x') / pl.col('at_x').shift(12)).alias('gp_atl1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('gp_atl1')).alias('gp_atl1'))
    
               # Calculate ope_bel1 and apply its conditions
               .with_columns((pl.col('ope_x') / pl.col('be_x').shift(12)).alias('ope_bel1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('be_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('ope_bel1')).alias('ope_bel1'))
    
               # Calculate cop_atl1 and apply its conditions
               .with_columns((pl.col('cop_x') / pl.col('at_x').shift(12)).alias('cop_atl1'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('cop_atl1')).alias('cop_atl1'))
              )
    
    
    
    #Profitability Measures
    __chars5 = (__chars5
    
               # Calculate pi_nix and apply its conditions
               .with_columns((pl.col('pi_x') / pl.col('nix_x')).alias('pi_nix'))
               .with_columns(pl.when((pl.col('pi_x') <= 0) | (pl.col('nix_x') <= 0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('pi_nix')).alias('pi_nix'))
    
               # Calculate ocf_at and apply its conditions
               .with_columns((pl.col('ocf_x') / pl.col('at_x')).alias('ocf_at'))
               .with_columns(pl.when(pl.col('at_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('ocf_at')).alias('ocf_at'))
    
               # Calculate op_at and apply its conditions
               .with_columns((pl.col('op_x') / pl.col('at_x')).alias('op_at'))
               .with_columns(pl.when(pl.col('at_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('op_at')).alias('op_at'))
    
               # Calculate ocf_at_chg1 and apply its conditions
               .with_columns((pl.col('ocf_at') - pl.col('ocf_at').shift(12)).alias('ocf_at_chg1'))
               .with_columns(pl.when(pl.col('count') <= 12)
                             .then(pl.lit(None).cast(pl.Float64))
                           .otherwise(pl.col('ocf_at_chg1')).alias('ocf_at_chg1'))
              )
    
    
    
    #Book Leverage:
    __chars5 = (__chars5.with_columns((pl.col('at_x') / pl.col('be_x')).alias('at_be')))
    
    
    
    #Volatility Quarterly Items
    __chars5 = (__chars5
    
               # Calculate __ocfq_saleq and apply condition
               .with_columns((pl.col('ocf_qtr') / pl.col('sale_qtr')).alias('__ocfq_saleq'))
               .with_columns(pl.when(pl.col('sale_qtr') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__ocfq_saleq')).alias('__ocfq_saleq'))
    
               # Calculate __niq_saleq and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('sale_qtr')).alias('__niq_saleq'))
               .with_columns(pl.when(pl.col('sale_qtr') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__niq_saleq')).alias('__niq_saleq'))
    
               # Calculate __roeq and apply condition
               .with_columns((pl.col('ni_qtr') / pl.col('be_x')).alias('__roeq'))
               .with_columns(pl.when(pl.col('be_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__roeq')).alias('__roeq'))
              )
    
    __chars5 = volq(df=__chars5, name='ocfq_saleq_std', var='__ocfq_saleq', qtrs=16, qtrs_min=8)
    __chars5 = volq(df=__chars5, name='niq_saleq_std', var='__niq_saleq', qtrs=16, qtrs_min=8)
    __chars5 = volq(df=__chars5, name='roeq_be_std', var='__roeq', qtrs=20, qtrs_min=12)
    __chars5 = __chars5.drop(['__ocfq_saleq', '__niq_saleq', '__roeq'])
    
    
    
    #Volatility Annual Items:
    __chars5 = (__chars5
    
               # Calculate __roe and apply condition
               .with_columns((pl.col('ni_x') / pl.col('be_x')).alias('__roe'))
               .with_columns(pl.when(pl.col('be_x') <= 0)
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('__roe')).alias('__roe'))
               )
    
    __chars5 = vola(df=__chars5, name='roe_be_std', var='__roe', yrs=5, yrs_min=5)
    __chars5 = __chars5.drop('__roe')
    
    
    #Earnings Smoothness
    __chars5 = earnings_variability(df=__chars5, esm_h=5)
    
    
    
    #Asset Liquidity:
    __chars5 = (__chars5
    
               .with_columns((pl.col('aliq_x') / pl.col('at_x').shift(12)).alias('aliq_at'))
               .with_columns(pl.when((pl.col('count') <= 12) | (pl.col('at_x').shift(12) <=0))
                             .then(pl.lit(None).cast(pl.Float64))
                             .otherwise(pl.col('aliq_at')).alias('aliq_at'))
               )
    
    # #Equity Duration
    __chars5 = equity_duration_cd(df=__chars5, horizon=10, r=0.12, roe_mean=0.12, roe_ar1=0.57, g_mean=0.06, g_ar1=0.24)
    
    # #F-score
    __chars5 = pitroski_f(df=__chars5, name='f_score')
    
    # #O-score
    __chars5 =  ohlson_o(df=__chars5, name='o_score')
    
    # #Z-score
    __chars5 =  altman_z(df=__chars5, name='z_score')
    
    # #Intrinsics value
    __chars5 = intrinsic_value(df= __chars5, name ='intrinsic_value', r=0.12)
    
    # #Kz-index
    __chars5 = kz_index(df= __chars5, name ='kz_index')
    
    
    # #5 year ratio change (For quality minus junk variables)
    __chars5 = chg_var1_to_var2(df=__chars5, name='gpoa_ch5', var1='gp_x', var2='at_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='roe_ch5', var1='ni_x', var2='be_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='roa_ch5', var1='ni_x', var2='at_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='cfoa_ch5', var1='ocf_x', var2='at_x', horizon=60);
    __chars5 = chg_var1_to_var2(df=__chars5, name='gmar_ch5', var1='gp_x', var2='sale_x', horizon=60);
    
    

    
    
    __chars5 = (__chars5
    
               # Calculate __roe and apply condition
               .with_columns(((pl.col('che') + 0.715 * pl.col('rect') + 0.547 * pl.col('invt') + 0.535 * pl.col('ppegt'))/(pl.col('at_x'))).alias('tangibility')).
                with_columns(pl.when(pl.col('at_x')==0).then(pl.lit(None).cast(pl.Float64)).otherwise(pl.col('tangibility')).alias('tangibility'))
               )


    #Earning's persistence
    earnings_pers = earnings_persistence(df=__chars5, n_years=5, min_obs=5)
    __chars6 = __chars5.join(earnings_pers, left_on=['gvkey', 'curcd', 'datadate'], right_on=['gvkey', 'curcd', 'datadate'], how='left').select(__chars5.columns + ['ni_ar1', 'ni_ivol'])
    
    #Keep only dates with accounting data
    __chars7  = __chars6.filter(pl.col('data_available')==1).sort(["gvkey", "datadate"])
    
    
    #lagging for public availability of data
    __chars8 =  __chars7.with_columns(pl.col('datadate').dt.offset_by(f'{lag_to_public}mo').dt.month_end().alias('start_date'))
    __chars8 =  __chars8.with_columns(pl.col('start_date').shift(-1).over(['gvkey']).alias('next_start_date'))
    __chars8 =  __chars8.with_columns(pl.min_horizontal((pl.col('next_start_date').dt.offset_by('-1mo').dt.month_end()),(pl.col('datadate').dt.offset_by(f'{max_data_lag}mo').dt.month_end())).alias('end_date'))
    __chars8 = __chars8.drop('next_start_date')
    
    
    __chars9 = expand(data=__chars8, id_vars=['gvkey'], start_date='start_date', end_date='end_date', freq='month', new_date_name='public_date')
    
    
    
    fx=pl.read_parquet('/vast/palmer/scratch/kelly/fa377/JKP/Testing/acc_chars/fx.parquet')
    #Convert All Raw (non-scaled) Variables to USD
    __chars10 = __chars9.join(fx, left_on=['curcd', 'public_date'], right_on=['curcdd', 'date'], how='left').select(__chars9.columns + ['fx'])
    
    var_raw = ['assets', 'sales', 'book_equity', 'net_income']
    __chars11 = __chars10
    for i in var_raw:
        __chars11 = __chars11.with_columns((pl.col(i)*pl.col('fx')).alias(i))
    
    __chars11 = __chars11.drop('curcd')
    
    
    #adding and filtering market return data
    __me_data1 = me_data.filter(
            (pl.col("gvkey").is_not_null()) & 
            (pl.col("primary_sec") == 1) & 
            (pl.col("me_company").is_not_null()) & 
            (pl.col("common") == 1) & 
            (pl.col("obs_main") == 1)
        ).select(
        ['gvkey', 'eom', 'me_company']).group_by(
        ["gvkey", "eom"]).agg(pl.col("me_company").max())
    
    __chars12 = __chars11.join(__me_data1, left_on=['gvkey', 'public_date'], right_on=['gvkey', 'eom'], how='left').select(__chars11.columns + ['me_company'])
    __chars13 = __chars12.sort(['gvkey', 'public_date']).unique(['gvkey', 'public_date'])
    
    #Create Ratios using both Accounting and Market Value
    __chars14 = (__chars13
    
             #calculting market enterprise value    
            .with_columns((pl.col('me_company') + (pl.col('netdebt_x')*pl.col('fx'))).alias('mev'))
            .with_columns(pl.when((pl.col('mev') <= 0))
                          .then(None)
                          .otherwise(pl.col('mev')).alias('mev'))
    
             #calculating market asset value    
                         .with_columns((pl.col('at_x') * pl.col('fx') - pl.col('be_x') * pl.col('fx') + pl.col('me_company')).alias('mat'))
            .with_columns(pl.when((pl.col('mat') <= 0))
                          .then(None)
                          .otherwise(pl.col('mat')).alias('mat'))
    
            #correcting market value in case it is negative (should we do it before calculating the above two?)
            .with_columns(pl.when((pl.col('me_company') <= 0))
                          .then(None)
                          .otherwise(pl.col('me_company')).alias('me_company'))       
         )
    
    
    #Characteristics Scaled by Market Equity
    me_vars = [
        "at_x", "be_x", "debt_x", "netdebt_x", "che", "sale_x", "gp_x", "ebitda_x",
        "ebit_x", "ope_x", "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "div_x",
        "eqbb_x", "eqis_x", "eqpo_x", "eqnpo_x", "eqnetis_x", "xrd"
    ]
    
    for i in me_vars:
        __chars14 = scale_me(df=__chars14, var=i)
    
    #Characteristics Scaled by Market Enterprise Value
    
    mev_vars = [
        "at_x", "bev_x", "ppent", "be_x", "che", "sale_x", "gp_x", "ebitda_x",
        "ebit_x", "ope_x", "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "debt_x",
        "pstk_x", "dltt", "dlc", "dltnetis_x", "dstnetis_x", "dbnetis_x", 
        "netis_x", "fincf_x"
    ]
    
    for i in mev_vars:
        __chars14 = scale_mev(df=__chars14, var=i)
    
    
    __chars14 = (__chars14
        .with_columns(
            ((pl.col('intrinsic_value') * pl.col('fx')) / (pl.col('me_company'))).alias('ival_me')
        )
                )
    
    
    #Characteristics Scaled by Market Assets
    __chars14 = (__chars14
        .with_columns(
            ((pl.col('aliq_x') * pl.col('fx')) / (pl.col('mat').shift(12))).alias('aliq_mat')
        )
        .with_columns(
            pl.when(pl.col('gvkey') != pl.col('gvkey').shift(12))
            .then(None)
            .otherwise(pl.col('aliq_mat')).alias('aliq_mat')
        )
    )
    
    #Size Measure
    __chars14 = (__chars14
        .with_columns(
            (pl.col('mev')).alias('enterprise_value')
        )
    )
    
     
    
    
    #Equity Duration
    __chars14 = (__chars14
        .with_columns(
            ((pl.col('ed_cd_w') * pl.col('fx')) / (pl.col('me_company')) + pl.col('ed_constant') * (pl.col('me_company') - pl.col('ed_cd') * pl.col('fx'))/pl.col('me_company')).alias('eq_dur')
        )
        .with_columns(
            pl.when((pl.col('ed_err') ==1) | (pl.col('eq_dur') <=0))
            .then(None)
            .otherwise(pl.col('eq_dur')).alias('eq_dur')
        )
    )
    
    
    
    #renaming columns:
    __chars15 = __chars14
    rename_dict = {
        "xrd": "rd",
        "xsga": "sga",
        "dlc": "debtst",
        "dltt": "debtlt",
        "oancf": "ocf",
        "ppegt": "ppeg",
        "ppent": "ppen",
        "che": "cash",
        "invt": "inv",
        "rect": "rec",
        "txt": "tax",
        "ivao": "lti",
        "ivst": "sti",
        "sale_qtr": "saleq",
        "ni_qtr": "niq",
        "ocf_qtr": "ocfq"
    }
    
    
    for a, b in rename_dict.items():
        __chars15 = __chars15.rename({col: col.replace(a, b) for col in __chars15.columns})
    
    
    #selecting variable columns of interest
    __chars16 = __chars15.select(['source', 'gvkey', 'public_date', 'datadate'] + __keep_vars)
    
    
    #addinf sufiix if mentioned
    if suffix is None:
        __chars16 = __chars16
    else:
        for i in __keep_vars:
            __chars16 = __chars16.rename({i:i+suffix})
    
    
    output = __chars16.sort(['gvkey', 'public_date']).unique(['gvkey', 'public_date'])

    return output