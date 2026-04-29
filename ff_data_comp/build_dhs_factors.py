"""Build DHS factors (FIN + PEAD) for USA and global.

Standalone test script. Reads pipeline parquet outputs from `data/interim/`,
augments accounting characteristics with `shares_comp` and `rdq`-derived
`date_qtr_qitem` from raw Compustat, downloads IBES `actu_epsint` (cached),
and writes a `dhs_factors.parquet` with columns [excntry, date, FIN, PEAD].

Run BEFORE `save_full_files_and_cleanup` (main.py:160), or set
`clear_interim=False` so the script can read from `data/interim/` and
`data/raw/raw_tables/` (which the cleanup step deletes).

Usage (from data/interim):
    cd data/interim
    python ../../ff_data_comp/build_dhs_factors.py

Caches IBES download + augmented acc chars. Delete those parquets to refresh.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from jkp.data.aux_functions import (
    download_wrds_table,
    gen_wrds_connection_info,
    measure_time,
)
from jkp.data.wrds_credentials import get_wrds_credentials

# ---------------------------------------------------------------------------
# Universe filter — used for non-USA branches and global daily screens.
# Matches build_dhs_factors.py:667/:813 (daily) and the upstream construction
# of size_grp in aux_functions.py.
# ---------------------------------------------------------------------------
UNIVERSE_FILTER = (
    (pl.col("exch_main") == 1) & (pl.col("common") == 1) & (pl.col("primary_sec") == 1)
)


# ---------------------------------------------------------------------------
# IBES download + transform (G2) — split for src/ portability
# download_ibes_actuals: raw WRDS dump (matches download_raw_data_tables shape)
# prepare_ibes_dates:    apply EPS/QTR filter + dedupe → DHS-ready lookup
# ---------------------------------------------------------------------------
@measure_time
def download_ibes_actuals(out_path: Path) -> None:
    """Download whole ibes.actu_epsint to parquet (no row filter)."""
    creds = get_wrds_credentials()
    conninfo = gen_wrds_connection_info(creds.username, creds.password)

    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")
    download_wrds_table(conninfo, con, "ibes.actu_epsint", str(out_path))


@measure_time
def prepare_ibes_dates(raw_path: Path, out_path: Path) -> None:
    """Filter ibes.actu_epsint to quarterly EPS, dedupe (ticker, pends).

    Without measure='EPS' AND pdicity='QTR' the table mixes EPS/BPS/SAL etc.
    across QTR/ANN/LTG, so (ticker, pends) is non-unique. Restatements dedupe
    to the earliest anndats.
    """
    (
        pl.scan_parquet(raw_path)
        .filter((pl.col("measure") == "EPS") & (pl.col("pdicity") == "QTR"))
        .select("ticker", "pends", "anndats")
        .sort("anndats")
        .unique(subset=["ticker", "pends"], keep="first")
        .sink_parquet(out_path)
    )


# ---------------------------------------------------------------------------
# Accounting char augmentation (G1)
# ---------------------------------------------------------------------------
@measure_time
def augment_acc_chars(interim: Path, out_a: Path, out_q: Path) -> None:
    """Add shares_comp / rdq-derived date_qtr_qitem to acc chars.

    Joins missing columns from raw Compustat parquets onto the pipeline's
    {achars,qchars}_world.parquet. USA branch uses comp_funda/comp_fundq;
    global branch uses comp_g_funda (cshoi/ajexi → csho/ajex). Global
    quarterly Compustat lacks rdq/cshoq/ajexq so those rows get nulls
    (filled later by IBES anndats for international names).
    """
    raw = raw_tables_dir(interim)

    # Annual: shares_comp = csho * ajex
    funda_us = (
        pl.scan_parquet(raw / "comp_funda.parquet")
        .select("gvkey", "datadate", "csho", "ajex")
        .with_columns(shares_comp=pl.col("csho") * pl.col("ajex"))
        .select("gvkey", "datadate", "shares_comp")
    )
    funda_glob = (
        pl.scan_parquet(raw / "comp_g_funda.parquet")
        .select("gvkey", "datadate", "cshoi", "ajexi")
        .with_columns(shares_comp=pl.col("cshoi") * pl.col("ajexi"))
        .select("gvkey", "datadate", "shares_comp")
    )
    funda = pl.concat([funda_us, funda_glob], how="vertical_relaxed").unique(
        subset=["gvkey", "datadate"], keep="first"
    )

    achars = pl.scan_parquet(interim / "achars_world.parquet").join(
        funda, on=["gvkey", "datadate"], how="left"
    )
    achars.sink_parquet(out_a)

    # Quarterly: shares_comp_qitem + date_qtr_qitem (USA only — global lacks rdq).
    fundq_us = (
        pl.scan_parquet(raw / "comp_fundq.parquet")
        .select("gvkey", "datadate", "cshoq", "ajexq", "rdq")
        .with_columns(
            shares_comp=pl.col("cshoq") * pl.col("ajexq"),
            date_qtr_qitem=pl.col("rdq"),
        )
        .select("gvkey", "datadate", "shares_comp", "date_qtr_qitem")
    )

    qchars = (
        pl.scan_parquet(interim / "qchars_world.parquet")
        .join(fundq_us, on=["gvkey", "datadate"], how="left")
        .rename({"datadate": "datadate_qitem"})
    )
    qchars.sink_parquet(out_q)


# ---------------------------------------------------------------------------
# Loaders — pipeline parquets, no date casts (parquet columns are typed)
# ---------------------------------------------------------------------------
def load_world_data(interim: Path) -> pl.LazyFrame:
    return pl.scan_parquet(interim / "world_data.parquet")


def load_world_dsf(interim: Path) -> pl.LazyFrame:
    return pl.scan_parquet(interim / "world_dsf.parquet").select(
        "id",
        "excntry",
        "date",
        "eom",
        "me",
        "ret",
        "ret_exc",
        "exch_main",
        "primary_sec",
        "common",
        "source_crsp",
    )


def load_market_returns_daily(interim: Path) -> pl.LazyFrame:
    return pl.scan_parquet(interim / "market_returns_daily.parquet").select(
        "excntry", "date", "mkt_vw"
    )


def raw_tables_dir(interim: Path) -> Path:
    """data/raw/raw_tables/ — sibling of interim under raw/."""
    return interim.parent / "raw" / "raw_tables"


def raw_dfs_dir(interim: Path) -> Path:
    """data/interim/raw_data_dfs/ — inside interim."""
    return interim / "raw_data_dfs"


def load_comp_g_security(interim: Path) -> pl.LazyFrame:
    return pl.scan_parquet(raw_tables_dir(interim) / "comp_g_security.parquet").select(
        "ibtic", "gvkey", "iid"
    )


def load_ibes_dates(ibes_path: Path, comp_g_security: pl.LazyFrame) -> pl.LazyFrame:
    return (
        pl.scan_parquet(ibes_path)
        .select("ticker", "pends", "anndats")
        .join(comp_g_security, left_on="ticker", right_on="ibtic")
    )


def load_nyse_p50(interim: Path) -> pl.LazyFrame:
    return pl.scan_parquet(interim / "nyse_cutoffs.parquet").select("eom", "nyse_p50")


# ---------------------------------------------------------------------------
# CRSP-Compustat link panel (replaces in-script CSV reconstruction)
# Pure Polars over the WRDS-downloaded parquet (lowercase columns).
# ---------------------------------------------------------------------------
@measure_time
def build_link_table(interim: Path) -> pl.LazyFrame:
    today = date.today()
    return (
        pl.scan_parquet(raw_tables_dir(interim) / "crsp_ccmxpf_lnkhist.parquet")
        .filter(pl.col("linktype").is_in(["LC", "LU", "LS"]))
        .filter(pl.col("lpermno").is_not_null())
        .with_columns(linkenddt=pl.col("linkenddt").fill_null(today))
        .with_columns(date=pl.date_ranges("linkdt", "linkenddt"))
        .explode("date")
        .select(
            "date",
            pl.col("gvkey").cast(pl.Utf8),
            pl.col("lpermno").cast(pl.Utf8).alias("id"),
        )
        .unique()
    )


# ---------------------------------------------------------------------------
# qchars IBES augmentation (international `date_qtr_qitem` from anndats)
# ---------------------------------------------------------------------------
@measure_time
def augment_qchars_with_ibes(
    qchars: pl.LazyFrame, ibes_dates: pl.LazyFrame, world_data: pl.LazyFrame
) -> pl.LazyFrame:
    int_companies = (
        world_data.select("excntry", "gvkey")
        .filter(pl.col("excntry") != "USA")
        .unique("gvkey")
        .select("gvkey")
    )

    qchars_int = qchars.join(int_companies, on="gvkey", how="inner")
    qchars_usa = qchars.join(int_companies, on="gvkey", how="anti")

    qchars_int = (
        qchars_int.with_columns(
            anndats_local=pl.col("datadate_qitem").dt.offset_by("4mo").dt.month_end()
        )
        .join(
            ibes_dates.select("gvkey", "pends", "anndats"),
            left_on=["gvkey", "datadate_qitem"],
            right_on=["gvkey", "pends"],
            how="left",
        )
        .with_columns(
            date_qtr_qitem=pl.coalesce(["anndats", "anndats_local"]),
        )
        .drop("anndats", "anndats_local")
    )

    return pl.concat([qchars_usa, qchars_int], how="vertical_relaxed")


# ---------------------------------------------------------------------------
# Combine annual + quarterly chars / merge into world_data / build FM
# ---------------------------------------------------------------------------
ACC_CHARS = [
    "shares_comp"
]  # only column DHS uses from acc chars beyond what world_data already has


def combine_ann_qtr_chars(
    ann_data: pl.LazyFrame, qtr_data: pl.LazyFrame, char_vars: list[str], q_suffix: str
) -> pl.LazyFrame:
    combined = ann_data.join(qtr_data, on=["gvkey", "public_date"], how="left", suffix=q_suffix)
    for ann_var in char_vars:
        qtr_var = ann_var + q_suffix
        combined = combined.with_columns(
            pl.when(
                pl.col(ann_var).is_null()
                | (
                    pl.col(qtr_var).is_not_null()
                    & (pl.col("datadate" + q_suffix) > pl.col("datadate"))
                )
            )
            .then(pl.col(qtr_var))
            .otherwise(pl.col(ann_var))
            .alias(ann_var)
        ).drop(qtr_var)
    return combined.drop("datadate", "datadate" + q_suffix).unique(subset=["gvkey", "public_date"])


@measure_time
def merge_acc_chars_into_world_data(
    world_data: pl.LazyFrame, achars: pl.LazyFrame, qchars: pl.LazyFrame
) -> pl.LazyFrame:
    combined = combine_ann_qtr_chars(achars, qchars, ACC_CHARS, "_qitem")
    ccols = set(combined.collect_schema().names())
    wcols = world_data.collect_schema().names()
    keep = [c for c in wcols if c not in ccols] + ["gvkey"]
    keep = list(dict.fromkeys(keep))  # dedupe preserving order
    return world_data.select(keep).join(
        combined,
        left_on=["gvkey", "eom"],
        right_on=["gvkey", "public_date"],
        how="left",
    )


@measure_time
def build_fm(achars: pl.LazyFrame, qchars: pl.LazyFrame) -> pl.LazyFrame:
    fm_ann = achars.drop("public_date").unique(["gvkey", "datadate"])
    fm_qtr = (
        qchars.drop("public_date")
        .rename({"gvkey": "gvkey_qitem"})
        .unique(["gvkey_qitem", "datadate_qitem"])
    )
    fm = fm_ann.join(
        fm_qtr,
        left_on=["gvkey", "datadate"],
        right_on=["gvkey_qitem", "datadate_qitem"],
        how="full",
        suffix="_qitem",
    )
    for c in fm_ann.collect_schema().names():
        qc = f"{c}_qitem"
        if qc in fm.collect_schema().names():
            fm = fm.with_columns(pl.coalesce([c, qc]).alias(c)).drop(qc)
    return fm.with_columns(
        year=pl.col("datadate").dt.year(),
        month=pl.col("datadate").dt.month(),
    )


def build_date_qtr_data(fm: pl.LazyFrame) -> pl.LazyFrame:
    return fm.select("gvkey", "datadate", "date_qtr_qitem").with_columns(
        date=pl.col("date_qtr_qitem")
    )


# ---------------------------------------------------------------------------
# PEAD CAR factor data (-2 to +1 days around earnings, abnormal vs market_vw)
# ---------------------------------------------------------------------------
@measure_time
def build_pead_factor_data(
    daily: pl.LazyFrame,
    link_table: pl.LazyFrame,
    date_qtr_data: pl.LazyFrame,
    mkt_ret_daily: pl.LazyFrame,
) -> pl.LazyFrame:
    pead_data = (
        daily.filter(pl.col("source_crsp") == 1)
        .select("id", "date", "ret")
        .join(link_table, on=["date", "id"], how="left")
        .join(date_qtr_data, on=["gvkey", "date"], how="left")
    )

    days = (
        pead_data.select("date")
        .unique()
        .sort("date")
        .with_columns(
            lag2=pl.col("date").shift(2),
            lag1=pl.col("date").shift(1),
            lag0=pl.col("date"),
            lead1=pl.col("date").shift(-1),
            diff=pl.col("date").shift(-2),
        )
        .drop_nulls()
        .with_columns(days=pl.concat_list(["lag2", "lag1", "lag0", "lead1"]))
        .drop("lag2", "lag1", "lag0", "lead1")
        .explode("days")
    )

    pead_data_2 = (
        pead_data.drop("ret")
        .filter(pl.col("date_qtr_qitem").is_not_null())
        .join(days, on="date", how="left")
        .join(
            pead_data.select("id", "date", "ret"),
            left_on=["id", "days"],
            right_on=["id", "date"],
            how="left",
        )
        .join(mkt_ret_daily.select("date", "mkt_vw"), left_on="days", right_on="date", how="left")
    )

    factor_data = (
        pead_data_2.with_columns(car_i=pl.col("ret") - pl.col("mkt_vw"))
        .filter(pl.col("car_i").is_not_null())
        .group_by("date_qtr_qitem", "datadate", "id")
        .agg(
            n=pl.len(),
            diff=pl.col("diff").first(),
            car=pl.col("car_i").sum(),
        )
        .sort("id", "date_qtr_qitem")
        .with_columns(
            end_date1=pl.col("date_qtr_qitem").shift(-1).over("id"),
            end_date2=pl.col("datadate").dt.offset_by("6mo"),
        )
        .with_columns(end_date=pl.min_horizontal("end_date1", "end_date2"))
        .drop("end_date1", "end_date2", "datadate")
        .with_columns(date_ranges=pl.date_ranges("date_qtr_qitem", "end_date", closed="left"))
        .explode("date_ranges")
        .filter(
            (pl.col("n") >= 2)
            & (pl.col("diff") <= pl.col("date_qtr_qitem").dt.month_end())
            & (pl.col("date_ranges") == pl.col("date_ranges").dt.month_end())
        )
        .drop("end_date")
        .rename({"date_ranges": "eom"})
    )
    return factor_data


# ---------------------------------------------------------------------------
# NSI (1y net stock issuance, June rebalanced)
# ---------------------------------------------------------------------------
@measure_time
def build_nsi_data(fm: pl.LazyFrame) -> pl.LazyFrame:
    share_data = fm.select("gvkey", "datadate", "year", "shares_comp")
    share_data_lag = (
        fm.select("gvkey", "datadate", "shares_comp")
        .with_columns(datadate=pl.col("datadate").dt.offset_by("12mo").dt.month_end())
        .rename({"shares_comp": "shares_comp_lag"})
    )
    return (
        share_data.join(share_data_lag, on=["gvkey", "datadate"], how="left")
        .with_columns(
            nsi_daniel=pl.when(pl.col("shares_comp_lag") > 0)
            .then((pl.col("shares_comp") / pl.col("shares_comp_lag")).log())
            .otherwise(None)
        )
        .drop("shares_comp_lag")
        .filter(pl.col("nsi_daniel").is_not_null())
        .sort("gvkey", "year", "datadate")
        .group_by("gvkey", "year")
        .agg(nsi_daniel=pl.col("nsi_daniel").last())
        .with_columns(eom=pl.date(pl.col("year") + 1, 6, 30))
        .drop("year")
    )


# ---------------------------------------------------------------------------
# CSI (5y composite stock issuance), refactored with cum_prod pattern
# ---------------------------------------------------------------------------
@measure_time
def build_csi_data(world_data: pl.LazyFrame) -> pl.LazyFrame:
    """Composite stock issuance over 60 months: log(me/me_lag60) - log(ri_60).

    `ri = (1 + coalesce(ret, 0)).cum_prod()` per id, mirroring eqnpo_cols
    (aux_functions.py:6076). Equivalent to (1+ret).log().cum_sum() for non-null
    returns; differs only in null handling.
    """
    base = (
        world_data.select("id", "eom", "me", "ret")
        .sort("id", "eom")
        .with_columns(ri=(1 + pl.coalesce("ret", 0)).cum_prod().over("id"))
    )
    lag = base.with_columns(eom=pl.col("eom").dt.offset_by("60mo").dt.month_end()).rename(
        {"me": "me_lag", "ri": "ri_lag", "ret": "ret_lag"}
    )

    return (
        base.join(lag, on=["id", "eom"], how="left")
        .with_columns(
            csi_return=(pl.col("ri") / pl.col("ri_lag")).log(),
            csi_me=pl.when(pl.col("me_lag") > 0)
            .then((pl.col("me") / pl.col("me_lag")).log())
            .otherwise(None),
        )
        .with_columns(csi_daniel=pl.col("csi_me") - pl.col("csi_return"))
        .filter(pl.col("eom").dt.month() == 6)
        .select("id", "eom", "csi_daniel")
    )


# ---------------------------------------------------------------------------
# Merge signals back onto world_data
# ---------------------------------------------------------------------------
@measure_time
def build_world_data_updated(
    world_data: pl.LazyFrame,
    factor_data: pl.LazyFrame,
    nsi_data: pl.LazyFrame,
    csi_data: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        world_data.with_columns(year=pl.col("eom").dt.year())
        .join(factor_data.select("id", "eom", "car"), on=["id", "eom"], how="left")
        .join(nsi_data, on=["gvkey", "eom"], how="left")
        .join(csi_data, on=["id", "eom"], how="left")
        .sort("id", "eom")
        .with_columns(
            nsi_daniel=pl.col("nsi_daniel").forward_fill(limit=11).over("id"),
            csi_daniel=pl.col("csi_daniel").forward_fill(limit=11).over("id"),
        )
    )


# ---------------------------------------------------------------------------
# FIN factor (defects fixed: nsi_bp_ filter, dead aliases, exchange filter)
# ---------------------------------------------------------------------------
def _classify_size(
    data: pl.LazyFrame, location: str, interim: Path, me_col: str = "me"
) -> pl.LazyFrame:
    if location == "USA":
        nyse = load_nyse_p50(interim)
        return (
            data.join(nyse, on="eom", how="left")
            .with_columns(
                size_grp_dhs=pl.when(pl.col(me_col) <= pl.col("nyse_p50"))
                .then(pl.lit("small"))
                .when(pl.col(me_col) > pl.col("nyse_p50"))
                .then(pl.lit("big"))
                .otherwise(None)
            )
            .drop("nyse_p50")
        )
    median_expr = pl.when(UNIVERSE_FILTER).then(pl.col(me_col)).otherwise(None).median().over("eom")
    return (
        data.with_columns(size_bp_global=median_expr)
        .with_columns(
            size_grp_dhs=pl.when(pl.col(me_col) <= pl.col("size_bp_global"))
            .then(pl.lit("small"))
            .when(pl.col(me_col) > pl.col("size_bp_global"))
            .then(pl.lit("big"))
            .otherwise(None)
        )
        .drop("size_bp_global")
    )


def _location_filter(data: pl.LazyFrame, location: str) -> pl.LazyFrame:
    common_filter = (
        pl.col("me").is_not_null()
        & pl.col("ret_exc_lead1m").is_not_null()
        & (pl.col("book_equity") >= 0)
        & ~pl.col("sic").cast(pl.Utf8).str.starts_with("6")
    )
    if location == "USA":
        return data.filter(
            (pl.col("source_crsp") == 1)
            & pl.col("crsp_exchcd").is_in([1, 2, 3])
            & pl.col("crsp_shrcd").is_in([10, 11])
            & common_filter
        )
    return data.filter(UNIVERSE_FILTER & common_filter)


@measure_time
def FIN(
    data: pl.LazyFrame, daily_data: pl.LazyFrame, location: str, daily: bool, interim: Path
) -> pl.LazyFrame:
    cols = [
        "id",
        "eom",
        "source_crsp",
        "gvkey",
        "me_company",
        "book_equity",
        "crsp_exchcd",
        "crsp_shrcd",
        "sic",
        "ret_exc_lead1m",
        "ret",
        "ret_exc",
        "prc",
        "me",
        "obs_main",
        "csi_daniel",
        "nsi_daniel",
        "exch_main",
        "common",
        "primary_sec",
    ]
    data = _location_filter(data.select(cols), location)

    # CSI breakpoints (NYSE for USA, full eligible universe for global)
    csi_src = data.select(
        "id", "eom", "csi_daniel", "exch_main", "common", "primary_sec", "crsp_exchcd"
    ).filter(pl.col("csi_daniel") < float("inf"))
    if location == "USA":
        csi_src = csi_src.filter(pl.col("crsp_exchcd") == 1)
    else:
        csi_src = csi_src.filter(UNIVERSE_FILTER)
    csi_bp = csi_src.group_by("eom").agg(
        csi_p20=pl.col("csi_daniel").quantile(0.2),
        csi_p80=pl.col("csi_daniel").quantile(0.8),
    )

    # NSI breakpoints — neg side: median of negative finite NSI; pos side: 30/70 of positive finite NSI.
    # G3 fix: aggregate from the correctly filtered frame, not from `data`.
    nsi_neg_src = data.select(
        "id", "eom", "nsi_daniel", "crsp_exchcd", "exch_main", "common", "primary_sec"
    ).filter((pl.col("nsi_daniel") < 0) & (pl.col("nsi_daniel") > float("-inf")))
    nsi_pos_src = data.select(
        "id", "eom", "nsi_daniel", "crsp_exchcd", "exch_main", "common", "primary_sec"
    ).filter((pl.col("nsi_daniel") > 0) & (pl.col("nsi_daniel") < float("inf")))
    if location == "USA":
        nsi_neg_src = nsi_neg_src.filter(pl.col("crsp_exchcd") == 1)
        nsi_pos_src = nsi_pos_src.filter(pl.col("crsp_exchcd") == 1)
    else:
        nsi_neg_src = nsi_neg_src.filter(UNIVERSE_FILTER)
        nsi_pos_src = nsi_pos_src.filter(UNIVERSE_FILTER)
    nsi_neg_bp = nsi_neg_src.group_by("eom").agg(nsi_neg_p50=pl.col("nsi_daniel").median())
    nsi_pos_bp = nsi_pos_src.group_by("eom").agg(
        nsi_pos_p30=pl.col("nsi_daniel").quantile(0.3),
        nsi_pos_p70=pl.col("nsi_daniel").quantile(0.7),
    )

    data = _classify_size(data, location, interim, me_col="me").rename({"size_grp_dhs": "size_grp"})
    data = data.sort("id", "eom").with_columns(
        size_grp=pl.col("size_grp").forward_fill(limit=11).over("id")
    )

    data = (
        data.join(csi_bp, on="eom", how="left")
        .with_columns(
            csi_grp=pl.when(pl.col("csi_daniel") <= pl.col("csi_p20"))
            .then(pl.lit("low"))
            .when(pl.col("csi_daniel") <= pl.col("csi_p80"))
            .then(pl.lit("medium"))
            .when(pl.col("csi_daniel") > pl.col("csi_p80"))
            .then(pl.lit("high"))
            .otherwise(None)
        )
        .drop("csi_p20", "csi_p80")
    )

    data = (
        data.join(nsi_neg_bp, on="eom", how="left")
        .join(nsi_pos_bp, on="eom", how="left")
        .with_columns(
            nsi_grp=pl.when(
                (pl.col("nsi_daniel") < 0) & (pl.col("nsi_daniel") <= pl.col("nsi_neg_p50"))
            )
            .then(pl.lit("low"))
            .when(
                ((pl.col("nsi_daniel") < 0) & (pl.col("nsi_daniel") > pl.col("nsi_neg_p50")))
                | ((pl.col("nsi_daniel") > 0) & (pl.col("nsi_daniel") <= pl.col("nsi_pos_p70")))
            )
            .then(pl.lit("medium"))
            .when((pl.col("nsi_daniel") > 0) & (pl.col("nsi_daniel") > pl.col("nsi_pos_p70")))
            .then(pl.lit("high"))
            .otherwise(None)
        )
        .drop("nsi_neg_p50", "nsi_pos_p30", "nsi_pos_p70")
    )

    data = data.with_columns(
        fin_grp=pl.when(
            ((pl.col("nsi_grp") == "high") & (pl.col("csi_grp") == "high"))
            | ((pl.col("nsi_grp") == "high") & pl.col("csi_grp").is_null())
            | ((pl.col("csi_grp") == "high") & pl.col("nsi_grp").is_null())
        )
        .then(pl.lit("high"))
        .when(
            ((pl.col("nsi_grp") == "low") & (pl.col("csi_grp") == "low"))
            | ((pl.col("nsi_grp") == "low") & pl.col("csi_grp").is_null())
            | ((pl.col("csi_grp") == "low") & pl.col("nsi_grp").is_null())
        )
        .then(pl.lit("low"))
        .otherwise(None)
    ).sort("id", "eom")

    if daily:
        daily_data = (
            daily_data.filter(UNIVERSE_FILTER)
            .with_columns(
                year=pl.col("date").dt.year(),
                month=pl.col("date").dt.month(),
            )
            .select("id", "date", "year", "month", pl.col("ret_exc").alias("daily_ret"))
        )
        data = (
            data.with_columns(eom_temp=pl.col("eom").dt.offset_by("1mo"))
            .with_columns(
                year=pl.col("eom_temp").dt.year(),
                month=pl.col("eom_temp").dt.month(),
            )
            .drop("eom_temp")
        )
        data = (
            daily_data.join(data, on=["id", "year", "month"], how="left")
            .drop("ret_exc_lead1m")
            .rename({"daily_ret": "ret_exc_lead1m"})
            .filter(pl.col("size_grp").is_not_null())
            .filter(pl.col("csi_grp").is_not_null())
            .filter(pl.col("fin_grp").is_not_null())
        )

    pfs = data.with_columns(
        pfs=pl.when((pl.col("size_grp") == "small") & (pl.col("fin_grp") == "low"))
        .then(pl.lit("s/l"))
        .when((pl.col("size_grp") == "small") & (pl.col("fin_grp") == "medium"))
        .then(pl.lit("s/m"))
        .when((pl.col("size_grp") == "small") & (pl.col("fin_grp") == "high"))
        .then(pl.lit("s/h"))
        .when((pl.col("size_grp") == "big") & (pl.col("fin_grp") == "low"))
        .then(pl.lit("b/l"))
        .when((pl.col("size_grp") == "big") & (pl.col("fin_grp") == "medium"))
        .then(pl.lit("b/m"))
        .when((pl.col("size_grp") == "big") & (pl.col("fin_grp") == "high"))
        .then(pl.lit("b/h"))
        .otherwise(None)
    ).filter(pl.col("pfs").is_not_null())

    date_col = "date" if daily else "eom"
    pf_returns = (
        pfs.sort(date_col)
        .group_by(date_col, "pfs")
        .agg(vw_ret=(pl.col("ret_exc_lead1m") * pl.col("me") / pl.col("me").sum()).sum())
    )

    # G3 fix: compute avg_l / avg_h as separate aggregates, then subtract.
    factor = (
        pf_returns.group_by(date_col)
        .agg(
            avg_l=pl.col("vw_ret").filter(pl.col("pfs").is_in(["s/l", "b/l"])).mean(),
            avg_h=pl.col("vw_ret").filter(pl.col("pfs").is_in(["s/h", "b/h"])).mean(),
        )
        .with_columns(FIN=pl.col("avg_l") - pl.col("avg_h"))
        .select(date_col, "FIN")
    )

    if not daily:
        factor = factor.with_columns(date=pl.col("eom").dt.offset_by("1mo").dt.month_end())
    return factor


# ---------------------------------------------------------------------------
# PEAD factor
# ---------------------------------------------------------------------------
@measure_time
def PEAD(
    data: pl.LazyFrame, daily_data: pl.LazyFrame, location: str, daily: bool, interim: Path
) -> pl.LazyFrame:
    cols = [
        "id",
        "eom",
        "source_crsp",
        "ret_exc_lead1m",
        "me",
        "me_company",
        "sic",
        "crsp_exchcd",
        "crsp_shrcd",
        "exch_main",
        "common",
        "primary_sec",
        "car",
    ]
    base = data.select(cols)
    if location == "USA":
        base = base.filter(
            (pl.col("source_crsp") == 1)
            & pl.col("crsp_shrcd").is_in([10, 11])
            & pl.col("crsp_exchcd").is_in([1, 2, 3])
            & pl.col("me_company").is_not_null()
            & pl.col("ret_exc_lead1m").is_not_null()
            & ~pl.col("sic").cast(pl.Utf8).str.starts_with("6")
        )
    else:
        base = base.filter(
            UNIVERSE_FILTER
            & pl.col("me_company").is_not_null()
            & pl.col("ret_exc_lead1m").is_not_null()
            & ~pl.col("sic").cast(pl.Utf8).str.starts_with("6")
        )
    base = base.unique(["id", "eom"])

    base = _classify_size(base, location, interim, me_col="me_company").rename(
        {"size_grp_dhs": "size_grp"}
    )

    pead_src = base.select(
        "id", "eom", "car", "exch_main", "common", "primary_sec", "crsp_exchcd"
    ).filter(pl.col("car") < float("inf"))
    if location == "USA":
        pead_src = pead_src.filter(pl.col("crsp_exchcd") == 1)
    else:
        pead_src = pead_src.filter(UNIVERSE_FILTER)
    pead_bp = pead_src.group_by("eom").agg(
        pead_p20=pl.col("car").quantile(0.2),
        pead_p80=pl.col("car").quantile(0.8),
    )

    base = (
        base.join(pead_bp, on="eom", how="left")
        .with_columns(
            pead_grp=pl.when(pl.col("car") < pl.col("pead_p20"))
            .then(pl.lit("low"))
            .when(pl.col("car") > pl.col("pead_p80"))
            .then(pl.lit("high"))
            .otherwise(pl.lit("medium"))
        )
        .drop("pead_p20", "pead_p80")
    )

    if daily:
        daily_data = (
            daily_data.filter(UNIVERSE_FILTER)
            .with_columns(
                year=pl.col("date").dt.year(),
                month=pl.col("date").dt.month(),
            )
            .select("id", "date", "year", "month", pl.col("ret_exc").alias("daily_ret"))
        )
        base = (
            base.with_columns(eom_temp=pl.col("eom").dt.offset_by("1mo"))
            .with_columns(
                year=pl.col("eom_temp").dt.year(),
                month=pl.col("eom_temp").dt.month(),
            )
            .drop("eom_temp")
        )
        base = (
            daily_data.join(base, on=["id", "year", "month"], how="left")
            .drop("ret_exc_lead1m")
            .rename({"daily_ret": "ret_exc_lead1m"})
            .filter(pl.col("size_grp").is_not_null())
            .filter(pl.col("pead_grp").is_not_null())
        )

    pfs = base.with_columns(
        pfs=pl.when((pl.col("size_grp") == "small") & (pl.col("pead_grp") == "low"))
        .then(pl.lit("s/l"))
        .when((pl.col("size_grp") == "small") & (pl.col("pead_grp") == "medium"))
        .then(pl.lit("s/m"))
        .when((pl.col("size_grp") == "small") & (pl.col("pead_grp") == "high"))
        .then(pl.lit("s/h"))
        .when((pl.col("size_grp") == "big") & (pl.col("pead_grp") == "low"))
        .then(pl.lit("b/l"))
        .when((pl.col("size_grp") == "big") & (pl.col("pead_grp") == "medium"))
        .then(pl.lit("b/m"))
        .when((pl.col("size_grp") == "big") & (pl.col("pead_grp") == "high"))
        .then(pl.lit("b/h"))
        .otherwise(None)
    ).filter(pl.col("pfs").is_not_null())

    date_col = "date" if daily else "eom"
    pf_returns = (
        pfs.sort(date_col)
        .group_by(date_col, "pfs")
        .agg(
            vw_ret=(
                pl.col("ret_exc_lead1m") * pl.col("me_company") / pl.col("me_company").sum()
            ).sum()
        )
    )

    factor = (
        pf_returns.group_by(date_col)
        .agg(
            avg_h=pl.col("vw_ret").filter(pl.col("pfs").is_in(["s/h", "b/h"])).mean(),
            avg_l=pl.col("vw_ret").filter(pl.col("pfs").is_in(["s/l", "b/l"])).mean(),
        )
        .with_columns(PEAD=pl.col("avg_h") - pl.col("avg_l"))
        .select(date_col, "PEAD")
    )

    if not daily:
        factor = factor.with_columns(date=pl.col("eom").dt.offset_by("1mo").dt.month_end())
    return factor


# ---------------------------------------------------------------------------
# Combine FIN + PEAD into DHS factor frame for one location
# ---------------------------------------------------------------------------
@measure_time
def dhs_factors(
    data_main: pl.LazyFrame,
    daily_data: pl.LazyFrame,
    location: str,
    daily: bool,
    interim: Path,
) -> pl.LazyFrame:
    fin = FIN(data_main, daily_data, location, daily, interim)
    pead = PEAD(data_main, daily_data, location, daily, interim)
    date_col = "date" if daily else "eom"
    fin = fin.drop(date_col) if date_col != "date" else fin
    pead = pead.drop(date_col) if date_col != "date" else pead
    return fin.join(pead, on="date", how="inner").select("date", "FIN", "PEAD")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@measure_time
def build_dhs(interim: Path, out_path: Path) -> None:
    print(f"[build_dhs] interim={interim} out={out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = raw_dfs_dir(interim)
    dfs.mkdir(parents=True, exist_ok=True)

    print("[build_dhs] phase 1/6: IBES download + prepare", flush=True)
    ibes_raw_path = raw_tables_dir(interim) / "ibes_actu_epsint.parquet"
    ibes_path = dfs / "ibes_actu_epsint.parquet"
    if not ibes_raw_path.exists():
        ibes_raw_path.parent.mkdir(parents=True, exist_ok=True)
        download_ibes_actuals(ibes_raw_path)
    else:
        print(f"[build_dhs]   cached: {ibes_raw_path}", flush=True)
    if not ibes_path.exists():
        prepare_ibes_dates(ibes_raw_path, ibes_path)
    else:
        print(f"[build_dhs]   cached: {ibes_path}", flush=True)

    print("[build_dhs] phase 2/6: augment acc chars", flush=True)
    achars_path = interim / "achars_world_dhs.parquet"
    qchars_path = interim / "qchars_world_dhs.parquet"
    if not achars_path.exists() or not qchars_path.exists():
        augment_acc_chars(interim, achars_path, qchars_path)
    else:
        print(f"[build_dhs]   cached: {achars_path}, {qchars_path}", flush=True)

    print("[build_dhs] phase 3/6: load lazy frames", flush=True)
    world_data = load_world_data(interim)
    daily = load_world_dsf(interim)
    mkt_ret_daily = load_market_returns_daily(interim)
    achars = pl.scan_parquet(achars_path)
    qchars = pl.scan_parquet(qchars_path)
    comp_g_security = load_comp_g_security(interim)
    ibes_dates = load_ibes_dates(ibes_path, comp_g_security)

    print("[build_dhs] phase 4/6: build fm + link table", flush=True)
    qchars = augment_qchars_with_ibes(qchars, ibes_dates, world_data)
    fm = build_fm(achars, qchars)
    date_qtr_data = build_date_qtr_data(fm)
    link_table = build_link_table(interim)

    print("[build_dhs] phase 5/6: build PEAD/NSI/CSI signals", flush=True)
    factor_data = build_pead_factor_data(daily, link_table, date_qtr_data, mkt_ret_daily)
    nsi_data = build_nsi_data(fm)
    csi_data = build_csi_data(world_data)

    world_data_updated = build_world_data_updated(
        merge_acc_chars_into_world_data(world_data, achars, qchars),
        factor_data,
        nsi_data,
        csi_data,
    )

    print("[build_dhs] phase 6/6: compute DHS factors per location", flush=True)
    parts = []
    for loc in ("USA", "GLOBAL"):
        print(f"[build_dhs]   location={loc}", flush=True)
        out = dhs_factors(world_data_updated, daily, loc, daily=False, interim=interim)
        parts.append(out.with_columns(excntry=pl.lit(loc)))

    print(f"[build_dhs] sink_parquet → {out_path}", flush=True)
    final = pl.concat(parts, how="vertical_relaxed").select("excntry", "date", "FIN", "PEAD")
    final.sink_parquet(out_path)
    nrows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    print(f"[build_dhs] wrote {nrows} rows to {out_path}", flush=True)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    interim = Path.cwd().resolve()
    out = interim / "dhs_factors.parquet"
    print(f"[main] cwd={interim}", flush=True)
    print(f"[main] raw_tables_dir = {raw_tables_dir(interim)}", flush=True)
    print(f"[main] raw_dfs_dir    = {raw_dfs_dir(interim)}", flush=True)
    build_dhs(interim, out)
    print(pl.read_parquet(out).head(), flush=True)
