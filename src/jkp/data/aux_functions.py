from __future__ import annotations

import contextlib
import functools
import operator
import os
import queue
import re
import shutil
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date
from math import exp, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from .paths import DataPaths

import duckdb
import ibis
import polars as pl
import polars_ds as pds
import polars_ols  # noqa: F401 - required for least_squares method on polars expressions
from ibis import _
from polars import col

from .config import (
    COLLECT_CHUNK_SIZE,
    END_DATE,
    FF_CCM_BACKDATE_FROM,
    FF_CCM_BACKDATE_MAX_YEARS,
    FF_CCM_BACKDATE_THROUGH,
    FF_DFF_GATE_COMPUSTAT_PRE1954,
    FF_DFF_MERGE_YEAR,
    FF_DFF_SYNTH_COUNT,
    FF_MIN_STOCKS_BP,
    FF_MIN_STOCKS_PF,
    FF_MISSING_RET_CODE,
    FF_MISSING_RET_TOL,
    FF_UMD_DAILY_LOOKBACK,
    FF_UMD_DAILY_SKIP,
    FF_UMD_MONTHLY_LOOKBACK,
    FF_UMD_MONTHLY_SKIP,
    FF_USE_DFF_BE,
    HXZ_MIN_STOCKS_BP,
    HXZ_MIN_STOCKS_PF,
    MAIN_FILTERS,
    MP_ANOMALY_LIST,
    MP_CCM_LINKTYPES,
    MP_CHS_DECAY,
    MP_DISTRESS_BETAS,
    MP_LAG_M,
    MP_MGMT_PCT_COLS,
    MP_MIN_OBS_PF_WORLD,
    MP_MIN_STKS_BP_WORLD,
    MP_PERF_PCT_COLS,
    MP_POSITIVE_ANOMALIES,
    MP_START_FACTOR_EOM,
    MP_WORLD_JKP_CHAR,
    US_EXCNTRY,
)
from .output_writer import write_dataframe
from .paths import DataPaths, get_dff_be_path


def fl_none():
    return pl.lit(None).cast(pl.Float64)


def bo_false():
    return pl.lit(False).cast(pl.Boolean)


def measure_time(func):
    """
    Description:
        Decorator to time a function and print start/end timestamps and elapsed minutes:seconds.

    Steps:
        1) Record start time and print function name + start.
        2) Execute the wrapped function and capture result.
        3) Record end time; compute and print duration.
        4) Return the original result.

    Output:
        Prints timing info to stdout; returns wrapped function's result.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function       : {func.__name__.upper()}", flush=True)
        print(
            f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}",
            flush=True,
        )
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}",
            flush=True,
        )
        # Calculate total seconds
        total_seconds = end_time - start_time
        # Calculate minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        print(f"Execution time : {minutes} minutes and {seconds:.2f} seconds", flush=True)
        print()
        return result

    return wrapper


@measure_time
def setup_folder_structure(paths: DataPaths) -> None:
    """
    Description:
        Create the pipeline's folder structure under the user-specified output directory.

    Steps:
        1) Create directories: raw_tables, raw_data_dfs, characteristics, return_data, accounting_data, other_output, portfolios.
        2) Copy the data README (license and citation info) into the output directory.

    Output:
        Folders created on disk (no return value).
    """
    from .paths import get_data_readme_path

    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    (paths.interim_dir / "raw_data_dfs").mkdir(exist_ok=True)
    paths.raw_tables_dir.mkdir(parents=True, exist_ok=True)
    (paths.processed_dir / "characteristics").mkdir(parents=True, exist_ok=True)
    (paths.processed_dir / "return_data" / "daily_rets_by_country").mkdir(
        parents=True, exist_ok=True
    )
    (paths.processed_dir / "accounting_data").mkdir(parents=True, exist_ok=True)
    (paths.processed_dir / "other_output").mkdir(parents=True, exist_ok=True)
    (paths.processed_dir / "portfolios").mkdir(parents=True, exist_ok=True)
    shutil.copy2(get_data_readme_path(), paths.base_dir / "README.md")


def collect_and_write(df, filename, collect_streaming=False):
    """
    Description:
        Collect a Polars LazyFrame (optionally streaming) and write to Parquet.
        Used for interim files during pipeline execution in main.py.

        Note: For final output files (in portfolio.py), use write_dataframe() from
        output_writer module instead. write_dataframe() supports shrink_dtype,
        ensures .parquet extension, and integrates with CSV conversion at pipeline end.

    Steps:
        1) df.collect(streaming=collect_streaming).
        2) write_parquet(filename).

    Output:
        Parquet file at `filename`.
    """
    df.collect(streaming=collect_streaming).write_parquet(filename)


def sic_naics_aux(filename):
    """
    Description:
        Load SIC/NAICS from a Compustat security header parquet and standardize column names.

    Steps:
        1) Read parquet lazily.
        2) Select gvkey, datadate, map sich→sic and naicsh→naics.
        3) Drop duplicates.

    Output:
        LazyFrame with unique (gvkey, datadate, sic, naics).
    """
    df = (
        pl.scan_parquet(filename)
        .select(
            [
                "gvkey",
                "datadate",
                col("sich").alias("sic"),
                col("naicsh").alias("naics"),
            ]
        )
        .unique()
    )
    return df


def load_age_aux(filename, filter_monthend=False):
    """
    Description:
        Load gvkey–datadate pairs from a parquet; optionally filter to month-end rows.

    Steps:
        1) Scan parquet lazily.
        2) If filter_monthend, keep rows with monthend == 1.
        3) Select gvkey, datadate.

    Output:
        LazyFrame of (gvkey, datadate).
    """
    df = pl.scan_parquet(filename)
    if filter_monthend:
        df = df.filter(col("monthend") == 1)
    return df.select(["gvkey", "datadate"])


def comp_hgics_aux(filename):
    """
    Description:
        Load Compustat GICS (global sub-industry) histories.

    Steps:
        1) Scan parquet lazily; drop null gvkey.
        2) Select gvkey, indfrom, indthru, gsubind→gics.
        3) Unique rows.

    Output:
        LazyFrame with distinct GICS intervals per gvkey.
    """
    df = (
        pl.scan_parquet(filename)
        .filter(col("gvkey").is_not_null())
        .select(["gvkey", "indfrom", "indthru", col("gsubind").alias("gics")])
        .unique()
    )
    return df


def sec_info_aux(filename):
    """
    Description:
        Load security status info (listing/Delisting) from a parquet.

    Steps:
        1) Scan parquet lazily.
        2) Select gvkey, iid, secstat, dlrsni.

    Output:
        LazyFrame of security status fields.
    """
    df = pl.scan_parquet(filename).select(["gvkey", "iid", "secstat", "dlrsni"])
    return df


def ex_country_aux(filename):
    """
    Description:
        Map exchange codes to country codes.

    Steps:
        1) Scan parquet lazily.
        2) Select exchg, excntry.
        3) Deduplicate.

    Output:
        LazyFrame of unique (exchg, excntry) mappings.
    """
    df = pl.scan_parquet(filename).select(["exchg", "excntry"]).unique()
    return df


def header_aux(comp_path, gcomp_path, output_path):
    """
    Description:
        Build a unified Compustat header table combining NA and Global sources (one row per gvkey).

    Steps:
        1) Read NA and Global header parquets into Ibis.
        2) Select gvkey, prirow, priusa, prican from each.
        3) Union tables and take the first occurrence per gvkey.
        4) Write to `output_path` and close connection.

    Output:
        Parquet header file with unique gvkey rows.
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    comp = con.read_parquet(comp_path).select(["gvkey", "prirow", "priusa", "prican"])
    g_comp = con.read_parquet(gcomp_path).select(["gvkey", "prirow", "priusa", "prican"])
    comp.union(g_comp).distinct(on="gvkey", keep="first").to_parquet(output_path)
    con.disconnect()


def prihist_aux(filename, alias_itemvalue):
    """
    Description:
        Build a unified Compustat header table combining NA and Global sources (one row per gvkey).

    Steps:
        1) Read NA and Global header parquets into Ibis.
        2) Select gvkey, prirow, priusa, prican from each.
        3) Union tables and take the first occurrence per gvkey.
        4) Write to `output_path` and close connection.

    Output:
        Parquet header file with unique gvkey rows.
    """
    df = (
        pl.scan_parquet(filename)
        .filter(col("item") == alias_itemvalue.upper())
        .select(["gvkey", col("itemvalue").alias(alias_itemvalue), "effdate", "thrudate"])
    )
    return df


def gen_firmshares(paths: DataPaths):
    """
    Description:
        Build a unified table of Compustat-reported shares outstanding and split factors
        from quarterly (FUNDQ) and annual (FUNDA) NA files.

    Steps:
        1) Load FUNDQ and FUNDA into DuckDB (rename FUNDA at_→at to avoid conflicts).
        2) Filter to INDL/STD/D/C rows with non-null shares & adj. factors.
        3) SELECT {gvkey, datadate, cshoq→csho_fund, ajexq→ajex_fund} from FUNDQ.
        4) UNION ALL with FUNDA {csho→csho_fund, ajex→ajex_fund}.
        5) Write to raw_data_dfs/__firm_shares1.parquet.

    Output:
        Parquet: raw_data_dfs/__firm_shares1.parquet (gvkey, datadate, csho_fund, ajex_fund).
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    con.create_table("comp_fundq", con.read_parquet(paths.raw_tables_dir / "comp_fundq.parquet"))
    con.create_table(
        "comp_funda",
        con.read_parquet(paths.raw_tables_dir / "comp_funda.parquet").rename({"at_": "at"}),
    )
    con.raw_sql("""
    CREATE TABLE __firm_shares1 AS

    SELECT gvkey, datadate, cshoq AS csho_fund, ajexq AS ajex_fund
    FROM comp_fundq
    WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' AND cshoq IS NOT NULL AND ajexq IS NOT NULL

    UNION ALL

    SELECT gvkey, datadate, csho AS csho_fund, ajex AS ajex_fund
    FROM comp_funda
    WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' AND csho IS NOT NULL AND ajex IS NOT NULL;
    """)
    con.table("__firm_shares1").to_parquet(
        paths.interim_dir / "raw_data_dfs" / "__firm_shares1.parquet"
    )
    con.disconnect()


def gen_prihist_files(paths: DataPaths):
    """
    Description:
        Extract Compustat security history “primary listing” flags (ROW/USA/CAN) with date
        intervals from NA and Global history tables.

    Steps:
        1) Load comp_sec_history and comp_g_sec_history into DuckDB.
        2) Create three tables by item code: PRIHISTROW (global), PRIHISTUSA (NA), PRIHISTCAN (NA).
        3) Keep gvkey, itemvalue→flag, effdate, thrudate.
        4) Write each to raw_data_dfs as separate Parquet files.

    Output:
        Parquets: __prihistrow.parquet, __prihistusa.parquet, __prihistcan.parquet.
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    con.create_table(
        "comp_sec_history",
        con.read_parquet(paths.raw_tables_dir / "comp_sec_history.parquet"),
    )
    con.create_table(
        "comp_g_sec_history",
        con.read_parquet(paths.raw_tables_dir / "comp_g_sec_history.parquet"),
    )
    con.raw_sql("""
    CREATE TABLE __prihistrow AS
    SELECT gvkey,
        itemvalue AS prihistrow,
        effdate,
        thrudate
    FROM comp_g_sec_history
    WHERE item = 'PRIHISTROW';

    CREATE TABLE __prihistusa AS
    SELECT gvkey,
        itemvalue AS prihistusa,
        effdate,
        thrudate
    FROM comp_sec_history
    WHERE item = 'PRIHISTUSA';

    CREATE TABLE __prihistcan AS
    SELECT gvkey,
        itemvalue AS prihistcan,
        effdate,
        thrudate
    FROM comp_sec_history
    WHERE item = 'PRIHISTCAN';
    """)

    con.table("__prihistrow").to_parquet(
        paths.interim_dir / "raw_data_dfs" / "__prihistrow.parquet"
    )
    con.table("__prihistusa").to_parquet(
        paths.interim_dir / "raw_data_dfs" / "__prihistusa.parquet"
    )
    con.table("__prihistcan").to_parquet(
        paths.interim_dir / "raw_data_dfs" / "__prihistcan.parquet"
    )
    con.disconnect()


def gen_fx1(paths: DataPaths):
    """
    Description:
        Build daily FX (to USD) series per currency using Compustat daily exchange rates.

    Steps:
        1) Load comp_exrt_dly into DuckDB.
        2) Join table to itself on (fromcurd, datadate) with fromcurd='GBP' and b.tocurd='USD'.
        3) Compute fx = b.exratd / a.exratd, label a.tocurd as curcdd.
        4) DISTINCT rows and write out.

    Output:
        Parquet: raw_data_dfs/__fx1.parquet (curcdd, datadate, fx to USD).
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    con.create_table(
        "comp_exrt_dly", con.read_parquet(paths.raw_tables_dir / "comp_exrt_dly.parquet")
    )
    con.raw_sql("""
    CREATE TABLE __fx1 AS
    SELECT DISTINCT
    a.tocurd AS curcdd,
    a.datadate,
    b.exratd / a.exratd AS fx
    FROM comp_exrt_dly AS a
    JOIN comp_exrt_dly AS b
    ON a.fromcurd = b.fromcurd
    AND a.datadate = b.datadate
    WHERE a.fromcurd = 'GBP'
    AND b.tocurd  = 'USD';
    """)
    con.table("__fx1").to_parquet(paths.interim_dir / "raw_data_dfs" / "__fx1.parquet")
    con.disconnect()


@measure_time
def gen_raw_data_dfs(paths: DataPaths):
    """
    Description:
        Generate a suite of “raw data” helper Parquet files from Compustat/CRSP sources.

    Steps:
        1) Call gen_firmshares(), gen_prihist_files(), gen_fx1().
        2) Derive SIC/NAICS (NA & Global), GICS (NA & Global), delist files (CRSP m/d),
        security info (NA & Global), T-bill return, FF monthly factors, exchange code map,
        exchange→country map, company headers (NA+Global), and CRSP security files (m & d).
        3) Call build_mcti() to derive the CRSP 30-Year Treasury index table.
        4) Call aug_msf_v2() to augment the monthly CRSP file with daily high/low prices.
        5) Standardize types/columns, sort/deduplicate where needed.
        6) Write all to raw_data_dfs/*.parquet.

    Output:
        Multiple helper Parquets under raw_data_dfs/ used in later pipelines.
    """
    gen_firmshares(paths)
    sic_naics_na = sic_naics_aux(paths.raw_tables_dir / "comp_funda.parquet")
    collect_and_write(sic_naics_na, paths.interim_dir / "raw_data_dfs" / "sic_naics_na.parquet")
    sic_naics_gl = sic_naics_aux(paths.raw_tables_dir / "comp_g_funda.parquet")
    collect_and_write(sic_naics_gl, paths.interim_dir / "raw_data_dfs" / "sic_naics_gl.parquet")
    permno0 = (
        pl.scan_parquet(paths.raw_tables_dir / "crsp_stksecurityinfohist.parquet")
        .select(
            [
                col("permno").cast(pl.Int64),
                col("permco").cast(pl.Int64),
                "secinfostartdt",
                "secinfoenddt",
                col("siccd").cast(pl.Int64).alias("sic"),
                col("naics").cast(pl.Int64),
            ]
        )
        .unique()
        .sort(["permno", "secinfostartdt", "secinfoenddt"])
    )
    collect_and_write(permno0, paths.interim_dir / "raw_data_dfs" / "permno0.parquet")
    comp_hgics_na = comp_hgics_aux(paths.raw_tables_dir / "comp_co_hgic.parquet")
    collect_and_write(comp_hgics_na, paths.interim_dir / "raw_data_dfs" / "comp_hgics_na.parquet")
    comp_hgics_gl = comp_hgics_aux(paths.raw_tables_dir / "comp_g_co_hgic.parquet")
    collect_and_write(comp_hgics_gl, paths.interim_dir / "raw_data_dfs" / "comp_hgics_gl.parquet")
    crsp_dsedelist = pl.scan_parquet(paths.raw_tables_dir / "crsp_stkdelists.parquet").select(
        [
            "delret",
            "delactiontype",
            "delstatustype",
            "delreasontype",
            "delpaymenttype",
            col("permno").cast(pl.Int64),
            "delistingdt",
        ]
    )
    collect_and_write(crsp_dsedelist, paths.interim_dir / "raw_data_dfs" / "crsp_dsedelist.parquet")
    crsp_msedelist = pl.scan_parquet(paths.raw_tables_dir / "crsp_stkdelists.parquet").select(
        [
            "delret",
            "delactiontype",
            "delstatustype",
            "delreasontype",
            "delpaymenttype",
            col("permno").cast(pl.Int64),
            "delistingdt",
        ]
    )
    collect_and_write(crsp_msedelist, paths.interim_dir / "raw_data_dfs" / "crsp_msedelist.parquet")
    __sec_info = pl.concat(
        [
            sec_info_aux(paths.raw_tables_dir / "comp_security.parquet"),
            sec_info_aux(paths.raw_tables_dir / "comp_g_security.parquet"),
        ]
    )
    collect_and_write(__sec_info, paths.interim_dir / "raw_data_dfs" / "__sec_info.parquet")
    build_mcti(paths)
    crsp_mcti_t30ret = pl.scan_parquet(
        paths.interim_dir / "raw_data_dfs" / "crsp_mcti.parquet"
    ).select(["caldt", "t30ret"])
    collect_and_write(
        crsp_mcti_t30ret, paths.interim_dir / "raw_data_dfs" / "crsp_mcti_t30ret.parquet"
    )
    ff_factors_monthly = pl.scan_parquet(
        paths.raw_tables_dir / "ff_factors_monthly.parquet"
    ).select(["date", "rf"])
    collect_and_write(
        ff_factors_monthly, paths.interim_dir / "raw_data_dfs" / "ff_factors_monthly.parquet"
    )
    comp_r_ex_codes = pl.scan_parquet(paths.raw_tables_dir / "comp_r_ex_codes.parquet").select(
        ["exchgdesc", "exchgcd"]
    )
    collect_and_write(
        comp_r_ex_codes, paths.interim_dir / "raw_data_dfs" / "comp_r_ex_codes.parquet"
    )
    __ex_country1 = pl.concat(
        [
            ex_country_aux(paths.raw_tables_dir / "comp_g_security.parquet"),
            ex_country_aux(paths.raw_tables_dir / "comp_security.parquet"),
        ]
    )
    collect_and_write(__ex_country1, paths.interim_dir / "raw_data_dfs" / "__ex_country1.parquet")
    header_aux(
        paths.raw_tables_dir / "comp_company.parquet",
        paths.raw_tables_dir / "comp_g_company.parquet",
        paths.interim_dir / "raw_data_dfs" / "__header.parquet",
    )
    gen_prihist_files(paths)
    gen_fx1(paths)
    aug_msf_v2(paths)
    gen_crsp_sf(paths, "m").to_parquet(paths.interim_dir / "raw_data_dfs" / "__crsp_sf_m.parquet")
    gen_crsp_sf(paths, "d").to_parquet(paths.interim_dir / "raw_data_dfs" / "__crsp_sf_d.parquet")


def gen_crsp_sf(paths: DataPaths, freq):
    """
    Description:
        Build CRSP security file enriched with names and CCM link history (monthly or daily).

    Steps:
        1) Load crsp_{freq}sf, crsp_{freq}senames, and ccmxpf_lnkhist into Ibis.
        2) Join SF to SENAMES on permno with date in [secinfostartdt, secinfoenddt].
        3) Join to CCM link history on permno and link date window, linktype in {LC, LU, LS}.
        4) Compute fields: bidask flag, abs(prc), shrout (thousands), ME, prc_high/low (valid only if prc>0),
        main exchange flag, and carry identifiers.
        5) Select standardized columns and return the Ibis table.

    Output:
        Ibis table (not written) with standardized CRSP {m|d} security fields.
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    if freq == "m":
        sf = con.read_parquet(paths.interim_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")
    elif freq == "d":
        sf = con.read_parquet(paths.raw_tables_dir / "crsp_dsf_v2.parquet")
    else:
        raise ValueError(f"Unknown freq: {freq}")
    senames = con.read_parquet(paths.raw_tables_dir / "crsp_stksecurityinfohist.parquet")
    ccmxpf_lnkhist = con.read_parquet(paths.raw_tables_dir / "crsp_ccmxpf_lnkhist.parquet")

    # CIZ name mapping by frequency
    if freq == "m":
        date_expr = sf.mthcaldt.cast("date")
        prc_expr = sf.mthprc
        prcflg_expr = sf.mthprcflg
        ret_expr = sf.mthret
        retx_expr = sf.mthretx
        vol_expr = sf.mthvol
        prcflg_expr = sf.mthprcflg
        cfacshr_expr = sf.mthcumfacshr
        askhi_expr = sf.mthaskhi
        bidlo_expr = sf.mthbidlo
    else:  # freq == "d", validated above
        date_expr = sf.dlycaldt.cast("date")
        prc_expr = sf.dlyprc
        prcflg_expr = sf.dlyprcflg
        ret_expr = sf.dlyret
        retx_expr = sf.dlyretx
        vol_expr = sf.dlyvol
        cfacshr_expr = sf.dlycumfacshr
        askhi_expr = sf.dlyhigh
        bidlo_expr = sf.dlylow

    sf_senames_join = sf.join(
        senames,
        how="left",
        predicates=[
            (sf.permno == senames.permno),
            (date_expr >= senames.secinfostartdt.cast("date")),
            (date_expr <= senames.secinfoenddt.cast("date")),
        ],
    )

    full_join = sf_senames_join.join(
        ccmxpf_lnkhist,
        how="left",
        predicates=[
            (sf.permno == ccmxpf_lnkhist.lpermno),
            ((date_expr >= ccmxpf_lnkhist.linkdt.cast("date")) | ccmxpf_lnkhist.linkdt.isnull()),
            (
                (date_expr <= ccmxpf_lnkhist.linkenddt.cast("date"))
                | ccmxpf_lnkhist.linkenddt.isnull()
            ),
            ccmxpf_lnkhist.linktype.isin(["LC", "LU", "LS"]),
        ],
    )

    bidask_expr = ibis.cases(
        (prcflg_expr == "BA", 1),
        (prcflg_expr == "TR", 0),
        else_=ibis.null(),
    ).cast("int32")

    securitytype_expr = sf.securitytype
    securitysubtype_expr = sf.securitysubtype
    sharetype_expr = sf.sharetype
    issuertype_expr = sf.issuertype

    is_common_expr = (
        (securitytype_expr == "EQTY")
        & (securitysubtype_expr == "COM")
        & (sharetype_expr == "NS")
        & (issuertype_expr.isin(["ACOR", "CORP"]))
    )

    primaryexch_expr = sf.primaryexch
    conditionaltype_expr = sf.conditionaltype

    exch_main_expr = (primaryexch_expr.isin(["A", "N", "Q"]) & (conditionaltype_expr == "RW")).cast(
        "int32"
    )
    crsp_nyse_expr = ((primaryexch_expr == "N") & (conditionaltype_expr == "RW")).cast("int32")

    result = full_join.mutate(
        date=date_expr,
        bidask=bidask_expr,
        prc=prc_expr,
        shrout=(sf.shrout / 1000),
        me=(prc_expr * (sf.shrout / 1000)),
        prc_high=ibis.cases(((prc_expr > 0) & (askhi_expr > 0), askhi_expr), else_=ibis.null()),
        prc_low=ibis.cases(((prc_expr > 0) & (bidlo_expr > 0), bidlo_expr), else_=ibis.null()),
        iid=ccmxpf_lnkhist.liid,
        ret=ret_expr,
        retx=retx_expr,
        cfacshr=cfacshr_expr,
        vol=vol_expr,
        common=is_common_expr.cast("int32"),
        primaryexch=primaryexch_expr,
        conditionaltype=conditionaltype_expr,
        exch_main=exch_main_expr,
        crsp_nyse=crsp_nyse_expr,
        gvkey=ccmxpf_lnkhist.gvkey,
    ).select(
        [
            "permno",
            "permco",
            "date",
            "bidask",
            "prc",
            "shrout",
            "ret",
            "retx",
            "cfacshr",
            "vol",
            "prc_high",
            "prc_low",
            "common",
            "primaryexch",
            "conditionaltype",
            "crsp_nyse",
            "gvkey",
            "iid",
            "exch_main",
            "me",
            "ticker",
        ]
    )
    return result


def gen_wrds_connection_info(user, password):
    return (
        f"host=wrds-pgdata.wharton.upenn.edu "
        f"port=9737 dbname=wrds "
        f"user={user} password={password} sslmode=require"
    )


def get_columns(conn, conninfo, lib, table):
    cols = conn.execute(f"""
        SELECT *
        FROM postgres_scan('{conninfo}', '{lib}', '{table}')
        LIMIT 0
    """).description
    return [c[0] for c in cols]


def get_columns_attached(conn, db_alias, lib, table):
    """Get column names from an attached PostgreSQL database."""
    cols = conn.execute(f"""
        SELECT *
        FROM {db_alias}.{lib}.{table}
        LIMIT 0
    """).description
    return [c[0] for c in cols]


def _date_where(date_column: str | None, start_date: date | None, end_date: date | None) -> str:
    """Build an inclusive ``WHERE date_column BETWEEN ...`` clause (empty if no date column)."""
    if not date_column:
        return ""
    conds = []
    if start_date is not None:
        conds.append(f"{date_column} >= '{start_date}'")
    if end_date is not None:
        conds.append(f"{date_column} <= '{end_date}'")
    return ("WHERE " + " AND ".join(conds)) if conds else ""


def download_wrds_table_attached(
    duckdb_conn,
    db_alias,
    table_name,
    filename,
    date_column: str | None = None,
    end_date: date | None = None,
    start_date: date | None = None,
):
    """Download a WRDS table (or an inclusive date-range slice) via an attached connection.

    When ``start_date``/``end_date`` are given (and the table has a ``date_column``), only rows
    with ``start_date <= date_column <= end_date`` are downloaded. ``start_date`` enables
    date-range chunking of large tables across parallel workers.
    """
    lib, table = table_name.split(".")
    cols = get_columns_attached(duckdb_conn, db_alias, lib, table)
    projection = build_projection(cols)

    where_clause = _date_where(date_column, start_date, end_date)

    duckdb_conn.execute(f"""
        COPY (
          SELECT {projection}
          FROM {db_alias}.{lib}.{table}
          {where_clause}
        )
        TO '{filename}' (FORMAT PARQUET);
    """)


def build_projection(cols):
    casts = []
    if "permno" in cols:
        casts.append("TRY_CAST(permno AS BIGINT) AS permno")
    if "permco" in cols:
        casts.append("TRY_CAST(permco AS BIGINT) AS permco")
    if "sic" in cols:
        casts.append("TRY_CAST(sic AS BIGINT) AS sic")
    if "sich" in cols:
        casts.append("TRY_CAST(sich AS BIGINT) AS sich")

    if casts:
        return "* REPLACE (" + ", ".join(casts) + ")"
    else:
        return "*"


def download_wrds_table(
    conninfo: str,
    duckdb_conn: duckdb.DuckDBPyConnection,
    table_name: str,
    filename: str,
    date_column: str | None = None,
    end_date: date | None = None,
) -> None:
    lib, table = table_name.split(".")
    cols = get_columns(duckdb_conn, conninfo, lib, table)
    projection = build_projection(cols)

    where_clause = ""
    if date_column and end_date:
        where_clause = f"WHERE {date_column} <= '{end_date}'"

    duckdb_conn.execute(f"""
        COPY (
          SELECT {projection}
          FROM postgres_scan('{conninfo}', '{lib}', '{table}')
          {where_clause}
        )
        TO '{filename}' (FORMAT PARQUET);
    """)


# The raw-data download is the one place in this pipeline that runs an explicit Python thread pool.
# Elsewhere we follow an "engine-level parallelism only" convention: let Polars/DuckDB parallelize
# internally rather than hand-rolling threads in compute code. This is a deliberate, narrow exception
# -- the parallelism here is N concurrent client connections to WRDS, which the query engine cannot
# provide (it parallelizes computation, not independent network connections). It is NOT a precedent
# for introducing thread pools into characteristic/compute code.
#
# WRDS enforces a per-role PostgreSQL CONNECTION LIMIT (currently 7 for our account; see
# `SELECT rolconnlimit FROM pg_roles WHERE rolname = current_user`). Each parallel worker holds
# exactly one client connection (verified: with `SET threads = 1` and these tables being views,
# one ATTACH == one TCP socket to wrds-pgdata), so the pool is capped below the limit to leave a
# slot of headroom for any other WRDS session under the same login.
#
# NOTE for debugging: WRDS fronts PostgreSQL with pgpool, so `pg_stat_activity` reports the
# server-side backend pool (it showed ~21 backends for 6 workers) -- that is NOT the client
# connection count. The number that matters for the role limit is the client connections, which
# equal the worker count (confirmed via the process's own sockets, e.g. psutil/`ss`).
WRDS_MAX_CONNECTIONS = 7

# These daily security tables dominate the download (hundreds of millions of rows each). In
# parallel mode they are split into row-balanced date-range chunks so a single giant table can't
# bottleneck the worker pool; the chunks are concatenated back into one parquet afterward.
SPLIT_TABLES = frozenset({"crsp.dsf_v2", "comp.secd", "comp.g_secd"})


def _effective_download_workers(requested: int, n_tables: int) -> int:
    """Clamp the requested worker count to a safe, useful range.

    Never more workers than there are tables, and never more than WRDS allows concurrent
    connections (minus one for headroom). A value of 1 (or less) means sequential download.
    """
    if requested <= 1:
        return 1
    return max(1, min(requested, n_tables, WRDS_MAX_CONNECTIONS - 1))


@dataclass(frozen=True)
class _DownloadTask:
    """One unit of download work: a whole table, or one date-range chunk of a split table."""

    table: str
    out: str
    date_column: str | None
    start_date: date | None
    end_date: date | None


def _chunk_path(filename: str, i: int) -> str:
    """Per-chunk parquet path, e.g. .../crsp_dsf_v2.parquet -> .../crsp_dsf_v2.part00.parquet."""
    p = Path(filename)
    return str(p.with_name(f"{p.stem}.part{i:02d}{p.suffix}"))


def _attach_wrds(con: duckdb.DuckDBPyConnection, conninfo: str, password: str) -> None:
    """ATTACH the WRDS Postgres database read-only on an existing DuckDB connection.

    DuckDB's postgres extension embeds the full connection string (including the password) in
    ATTACH error text, so on failure suppress the original exception and raise a generic,
    password-free error. Errors that don't contain the password propagate unchanged.
    """
    try:
        con.execute(f"ATTACH '{conninfo}' AS wrds (TYPE postgres, READ_ONLY)")
    except Exception as e:
        if password and password in str(e):
            raise RuntimeError(
                "Failed to attach WRDS connection. Check credentials and MFA approval."
            ) from None
        raise


def _install_postgres_extension() -> None:
    """Install the DuckDB postgres extension once, up front (idempotent).

    Doing it in the main thread means the parallel workers only ``LOAD`` it (a per-connection,
    no-download operation), which avoids a concurrent-INSTALL race across the pool writing the same
    extension file, and surfaces a fetch failure as one clean error here rather than N worker
    tracebacks.
    """
    with duckdb.connect(":memory:") as con:
        con.execute("INSTALL postgres;")


_MapT = TypeVar("_MapT")
_MapR = TypeVar("_MapR")


def _map_interruptible(
    fn: Callable[[_MapT], _MapR], items: Iterable[_MapT], max_workers: int
) -> list[_MapR]:
    """Run ``[fn(item) for item in items]`` concurrently, staying responsive to Ctrl-C.

    Like a ``ThreadPoolExecutor`` map, but a KeyboardInterrupt cancels not-yet-started work and
    propagates promptly instead of blocking in the pool's ``shutdown(wait=True)``. In-flight calls
    still finish (a running query/COPY can't be interrupted mid-call). Task exceptions and results
    propagate in order, exactly like ``list(ex.map(fn, items))``.
    """
    ex = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [ex.submit(fn, item) for item in items]
        pending = set(futures)
        while pending:  # timed waits keep the main thread returning to Python so Ctrl-C can fire
            _, pending = wait(pending, timeout=0.5)
        return [f.result() for f in futures]
    except KeyboardInterrupt:
        print(
            "\nInterrupted: stopping after in-flight work finishes...",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def _year_histogram(
    con: duckdb.DuckDBPyConnection,
    db_alias: str,
    lib: str,
    table: str,
    date_column: str,
    end_date: date | None,
) -> list[tuple[int, int]]:
    """Server-side per-year row counts for a date-filtered table: [(year, n), ...] sorted by year."""
    where = _date_where(date_column, None, end_date)
    rows = con.execute(
        f"SELECT CAST(EXTRACT(YEAR FROM {date_column}) AS INTEGER) AS yr, COUNT(*) AS n "  # noqa: S608
        f"FROM {db_alias}.{lib}.{table} {where} GROUP BY yr ORDER BY yr"
    ).fetchall()
    return [(int(yr), int(n)) for yr, n in rows if yr is not None]


def _balanced_year_chunks(
    histogram: list[tuple[int, int]], n_chunks: int, end_date: date | None
) -> list[tuple[date | None, date | None]]:
    """Split a year histogram into <= n_chunks contiguous inclusive (start, end) date ranges
    holding ~equal row counts.

    The first range's start is None (unbounded below) and the last range's end is the overall
    end_date; ranges are non-overlapping and together cover every counted row. Cuts fall on
    year boundaries, so chunks are only as balanced as the per-year granularity allows.
    """
    hist = [(y, n) for y, n in histogram if n > 0]
    if n_chunks <= 1 or len(hist) <= 1:
        return [(None, end_date)]
    total = sum(n for _, n in hist)
    target = total / n_chunks
    cut_years: list[int] = []
    acc = 0
    for i, (yr, n) in enumerate(hist):
        acc += n
        last_year = i == len(hist) - 1
        if len(cut_years) < n_chunks - 1 and not last_year and acc >= (len(cut_years) + 1) * target:
            cut_years.append(yr)
    starts = [None, *[cy + 1 for cy in cut_years]]
    ends = [date(cy, 12, 31) for cy in cut_years] + [end_date]
    return [
        (date(s, 1, 1) if s is not None else None, e) for s, e in zip(starts, ends, strict=True)
    ]


def _compute_histograms(
    conninfo: str,
    tables: list[str],
    date_columns: dict[str, str],
    end_date: date | None,
    max_conns: int,
    password: str,
) -> dict[str, list[tuple[int, int]]]:
    """Concurrently compute per-year row histograms for ``tables`` (each over its own connection).

    Returns {table: [(year, n), ...]}. Runs up to ``max_conns`` queries at once so the up-front
    boundary scan of the giant tables doesn't serialize and leave the connection pool idle.
    """
    if not tables:
        return {}

    def one(table: str) -> tuple[str, list[tuple[int, int]]]:
        lib, tbl = table.split(".")
        with duckdb.connect(":memory:") as con:
            con.execute("SET threads TO 1")
            con.execute("LOAD postgres;")  # extension installed once up front by the caller
            _attach_wrds(con, conninfo, password)
            try:
                return table, _year_histogram(con, "wrds", lib, tbl, date_columns[table], end_date)
            finally:
                with contextlib.suppress(Exception):
                    con.execute("DETACH wrds")

    return dict(_map_interruptible(one, tables, min(len(tables), max_conns)))


def _build_download_tasks(
    table_names: list[str],
    filenames: dict[str, str],
    date_columns: dict[str, str],
    end_date: date | None,
    split_tables: frozenset[str],
    n_chunks: int,
    histograms: dict[str, list[tuple[int, int]]],
) -> tuple[list[_DownloadTask], dict[str, list[str]]]:
    """Expand the table list into download tasks, splitting ``split_tables`` into date chunks.

    ``histograms`` maps a split table to its per-year row counts (see _compute_histograms).
    Returns (tasks, concat_map), where concat_map maps a final parquet path to its ordered chunk
    files (only for tables actually split into >1 chunk).
    """
    tasks: list[_DownloadTask] = []
    concat_map: dict[str, list[str]] = {}
    for table in table_names:
        date_col = date_columns.get(table)
        hist = histograms.get(table)
        do_split = (
            hist is not None
            and n_chunks > 1
            and table in split_tables
            and date_col
            and end_date is not None
        )
        if not do_split or hist is None:
            tasks.append(_DownloadTask(table, filenames[table], date_col, None, end_date))
            continue
        ranges = _balanced_year_chunks(hist, n_chunks, end_date)
        if len(ranges) <= 1:
            start, end = ranges[0]
            tasks.append(_DownloadTask(table, filenames[table], date_col, start, end))
            continue
        chunk_files = []
        for i, (start, end) in enumerate(ranges):
            cf = _chunk_path(filenames[table], i)
            chunk_files.append(cf)
            tasks.append(_DownloadTask(table, cf, date_col, start, end))
        concat_map[filenames[table]] = chunk_files
    return tasks, concat_map


def _concat_chunks(final_file: str, chunk_files: list[str]) -> None:
    """Concatenate ordered chunk parquets into one final parquet, then remove the chunks (local)."""
    pl.scan_parquet(chunk_files).sink_parquet(final_file)
    for f in chunk_files:
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)


def _remove_chunk_parts(filename: str) -> None:
    """Delete any existing chunk-part files for ``filename`` (see _chunk_path).

    A prior interrupted run may have written chunk parts that were never concatenated — possibly
    with a different worker count, hence different indices — so clear them all before writing fresh
    ones. The final (non-part) file itself is left untouched.
    """
    p = Path(filename)
    # Match exactly what _chunk_path writes (.part00..part99) rather than `part*`, which would also
    # sweep an unrelated file like `<stem>.partial.parquet`.
    for f in p.parent.glob(f"{p.stem}.part[0-9][0-9]{p.suffix}"):
        with contextlib.suppress(FileNotFoundError):
            f.unlink()


def _attach_download_worker(
    task_queue: queue.Queue[_DownloadTask],
    conninfo: str,
    password: str,
    task_errors: list[str],
    startup_errors: list[str],
    errors_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    """One parallel worker: open a private ATTACH connection and drain the task queue.

    Each task is a whole table or one date-range chunk. The worker holds exactly one WRDS
    connection (``threads=1`` keeps DuckDB from opening extra connections for a parallel scan)
    and reuses it. It stops pulling new tasks once ``stop_event`` is set (used to unwind on Ctrl-C).

    Failures are recorded (password-redacted) instead of raised, so one bad task doesn't abandon
    the rest. Worker-startup failures (LOAD/ATTACH) and download (task) failures are kept in separate
    lists: because the task queue is shared, a worker that fails to start is not fatal as long as the
    surviving workers still drain the queue, whereas a failed download leaves a table missing. The
    caller decides what to raise (see _download_tables_parallel).
    """
    # `with duckdb.connect(...)` closes the connection on every exit path (return / exception).
    with duckdb.connect(":memory:") as con:
        try:
            con.execute("SET threads TO 1")
            con.execute("LOAD postgres;")  # extension installed once up front by the caller
            _attach_wrds(con, conninfo, password)
        except Exception as e:  # noqa: BLE001
            # _attach_wrds already raises a password-free error, and LOAD/SET can't leak the
            # password; warn now so the user knows the run is proceeding under-provisioned rather
            # than silently losing a worker.
            with errors_lock:
                startup_errors.append(str(e))
                print(
                    f"Warning: a WRDS download worker failed to start ({e}); continuing with the "
                    "remaining worker(s). Press Ctrl-C to cancel and re-run for full parallelism.",
                    file=sys.stderr,
                    flush=True,
                )
            return
        try:
            while not stop_event.is_set():
                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    download_wrds_table_attached(
                        con,
                        "wrds",
                        task.table,
                        task.out,
                        date_column=task.date_column,
                        end_date=task.end_date,
                        start_date=task.start_date,
                    )
                except Exception as e:  # noqa: BLE001
                    # Redact only the credential, keeping the rest of the error for diagnostics.
                    msg = str(e).replace(password, "***") if password else str(e)
                    with errors_lock:
                        task_errors.append(f"{task.table} ({Path(task.out).name}): {msg}")
        finally:
            with contextlib.suppress(Exception):
                con.execute("DETACH wrds")


def _download_tables_parallel(
    table_names: list[str],
    filenames: dict[str, str],
    conninfo: str,
    date_columns: dict[str, str],
    end_date: date | None,
    password: str,
    workers: int,
    split_tables: frozenset[str] = frozenset(),
    n_chunks: int = 1,
) -> None:
    """Download all tables concurrently over ``workers`` private ATTACH connections.

    Tables in ``split_tables`` are divided into ``n_chunks`` row-balanced date-range chunks
    (boundaries computed up front over one short-lived connection) and concatenated afterward, so
    a single giant table can't bottleneck the pool. Work is pulled from a shared queue (dynamic
    load balancing). Raises a single aggregated ``RuntimeError`` if any task failed.
    """
    # Install the postgres extension once, in this thread, so the parallel workers (and histogram
    # workers) only LOAD it -- no concurrent-INSTALL race, and a fetch failure surfaces here cleanly.
    _install_postgres_extension()

    # Up front: concurrently compute per-year row histograms for the split tables to derive chunk
    # boundaries (the giant tables need a full COUNT scan; running them in parallel hides the cost).
    split_present = [
        t for t in table_names if t in split_tables and date_columns.get(t) and end_date is not None
    ]
    histograms = (
        _compute_histograms(conninfo, split_present, date_columns, end_date, workers, password)
        if split_present and n_chunks > 1
        else {}
    )
    tasks, concat_map = _build_download_tasks(
        table_names, filenames, date_columns, end_date, split_tables, n_chunks, histograms
    )

    # Clear any stale chunk parts left by a prior interrupted run before the workers write fresh
    # ones, so leftover parts (e.g. from a run with a different worker count) can't linger
    # un-concatenated alongside this run's output.
    for table in table_names:
        if table in split_tables:
            _remove_chunk_parts(filenames[table])

    task_queue: queue.Queue[_DownloadTask] = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    task_errors: list[str] = []
    startup_errors: list[str] = []
    errors_lock = threading.Lock()
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_attach_download_worker,
            args=(
                task_queue,
                conninfo,
                password,
                task_errors,
                startup_errors,
                errors_lock,
                stop_event,
            ),
            name=f"wrds-download-{i}",
        )
        for i in range(workers)
    ]
    for t in threads:
        t.start()
    # Join with a timeout so the main thread stays responsive to Ctrl-C (a plain join() would defer
    # the KeyboardInterrupt until the whole download finished). On Ctrl-C, ask the workers to stop
    # pulling new tasks; they finish the download already in flight and exit.
    #
    # We deliberately do NOT call DuckDB's con.interrupt() to abort an in-flight COPY mid-transfer:
    # it would require sharing every worker's connection with this thread plus telling a user-cancel
    # apart from a real failure, and its effect on a COPY blocked in the postgres extension's network
    # read is unverified. Consequently a Ctrl-C takes effect only after the currently-downloading
    # task(s) finish -- bounded by one chunk, not the entire remaining download.
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        print(
            "\nInterrupted: stopping after the in-flight download(s) finish...",
            file=sys.stderr,
            flush=True,
        )
        stop_event.set()
        for t in threads:
            t.join()
        raise

    # A download failure means a table is missing -> always fatal.
    if task_errors:
        raise RuntimeError(
            f"{len(task_errors)} WRDS download task(s) failed:\n  "
            + "\n  ".join(sorted(task_errors))
        )
    # If the queue never drained, every worker failed to start -> nothing was downloaded.
    if not task_queue.empty():
        raise RuntimeError(
            f"all {workers} WRDS download worker(s) failed to start:\n  "
            + "\n  ".join(sorted(startup_errors))
        )
    # Some workers failed to start, but the survivors completed every table -> note it, don't fail.
    if startup_errors:
        print(
            f"Note: {len(startup_errors)} of {workers} WRDS download worker(s) failed to start; "
            "the remaining worker(s) completed all tables.",
            file=sys.stderr,
            flush=True,
        )

    # Concatenate each split table's chunks back into its single parquet. These are local
    # (no WRDS connection) and independent, so run them concurrently. Failures are aggregated
    # like the download phase rather than aborting on the first error; chunks left behind by a
    # failed concat are cleaned up by the stale-part sweep on the next run.
    if concat_map:
        concat_errors: list[str] = []
        concat_lock = threading.Lock()

        def concat_one(item: tuple[str, list[str]]) -> None:
            final_file, chunk_files = item
            try:
                _concat_chunks(final_file, chunk_files)
            except Exception as e:  # noqa: BLE001
                with concat_lock:
                    concat_errors.append(f"{Path(final_file).name}: {e}")

        _map_interruptible(concat_one, list(concat_map.items()), len(concat_map))
        if concat_errors:
            raise RuntimeError(
                f"{len(concat_errors)} chunk concatenation(s) failed:\n  "
                + "\n  ".join(sorted(concat_errors))
            )


@measure_time
def download_raw_data_tables(
    paths: DataPaths,
    username: str,
    password: str,
    end_date: date | None = None,
    persistent_connection: bool = False,
    max_workers: int = 1,
) -> None:
    """
    Description:
        Bulk-download core WRDS tables to raw_tables and a few curated variants with column subsets.

    Steps:
        1) Connect to WRDS; iterate through a fixed list of library.tables.
        2) For each table: download to raw_tables/lib_table.parquet, applying date filtering
           when end_date is provided and the table has a known date column.
        3) If max_workers > 1: download tables concurrently over a pool of persistent ATTACH
           connections (one per worker). Otherwise download sequentially over a single
           connection, using ATTACH when persistent_connection is set or postgres_scan() if not.
        4) Disconnect.

    Args:
        username: WRDS username
        password: WRDS password
        persistent_connection: If True, use a single persistent connection via ATTACH. This
            reduces MFA prompts on systems with NAT IP rotation (e.g., Yale Bouchet). If False
            (default), use postgres_scan() which creates a new connection per query. This flag
            takes precedence over max_workers: when set, the download stays sequential (one
            connection) so the single-MFA-prompt guarantee is preserved.
        max_workers: Number of concurrent download connections. The download is single-threaded
            and transfer-bound, so the tables are otherwise pulled one at a time over a single
            stream; running several in parallel cuts wall time substantially. Clamped to the WRDS
            per-account connection limit (see WRDS_MAX_CONNECTIONS). Default 1 (sequential).
            Ignored when persistent_connection is set. In parallel mode the giant daily tables
            (SPLIT_TABLES) are additionally split into max_workers date-range chunks so one huge
            table can't bottleneck the pool. NOTE: each worker opens its own connection, so on
            NAT-rotated networks (e.g., Bouchet) parallel mode would trigger one MFA prompt per
            worker at startup — which is exactly why persistent_connection forces sequential.

    Output:
        Parquet files under raw_tables/ (Compustat, CRSP, FF, etc.).
    """
    table_names = [
        "comp.exrt_dly",
        "ff.factors_monthly",
        "comp.g_security",
        "comp.security",
        "comp.r_ex_codes",
        "comp.g_sec_history",
        "comp.sec_history",
        "comp.company",
        "comp.g_company",
        "crsp.stksecurityinfohist",
        "crsp.stkissuerinfohist",
        "crsp.ccmxpf_lnkhist",
        "comp.funda",
        "comp.fundq",
        "crsp.stkdelists",
        "comp.secm",
        "crsp.indmthseriesdata_ind",
        "crsp.indseriesinfohdr_ind",
        "crsp.msf_v2",
        "comp.g_co_hgic",
        "crsp.dsf_v2",
        # CRSP daily market index — required by DHS PEAD CAR (vwretd benchmark).
        "crsp.dsi",
        "comp.g_funda",
        "comp.co_hgic",
        "comp.g_fundq",
        "comp.secd",
        "comp.g_secd",
        # Mispricing factors (Stambaugh-Yuan) — CHS DISTRESS inputs.
        "crsp_a_indexes.msp500",
        "crsp_a_indexes.acti",
        # ccmxpf_linktable has usedflag (ccmxpf_lnkhist drops it); needed for SY-parity tiebreak.
        "crsp.ccmxpf_linktable",
        # IBES international EPS actuals — anndats supplies ex-US earnings
        # announcement dates for DHS PEAD (Compustat g_fundq has no rdq).
        "ibes.actu_epsint",
    ]

    # Tables with a known date column are filtered to end_date during download.
    # Reference/metadata tables (not listed here) are downloaded in full.
    date_columns: dict[str, str] = {
        "crsp.msf_v2": "mthcaldt",
        "crsp.dsf_v2": "dlycaldt",
        "crsp.dsi": "date",
        "comp.secd": "datadate",
        "comp.g_secd": "datadate",
        "comp.secm": "datadate",
        "comp.funda": "datadate",
        "comp.fundq": "datadate",
        "comp.g_funda": "datadate",
        "comp.g_fundq": "datadate",
        "crsp_a_indexes.msp500": "caldt",
        "crsp_a_indexes.acti": "caldt",
        "ibes.actu_epsint": "anndats",
    }

    conninfo = gen_wrds_connection_info(username, password)
    filenames = {
        table: str(paths.raw_tables_dir / (table.replace(".", "_") + ".parquet"))
        for table in table_names
    }

    workers = _effective_download_workers(max_workers, len(table_names))
    if persistent_connection and workers > 1:
        # --persistent-connection exists to hold ONE connection so NAT-rotated networks
        # (e.g. Bouchet) get a single MFA prompt. Parallel workers each open their own
        # connection (one MFA prompt apiece), defeating that, so the single-connection request
        # wins and we download sequentially.
        print(
            "Note: --persistent-connection uses a single WRDS connection; "
            f"ignoring max_workers={max_workers} and downloading sequentially.",
            file=sys.stderr,
            flush=True,
        )
        workers = 1
    if workers > 1:
        # Parallel: each worker holds its own persistent ATTACH connection and drains a queue.
        # The giant daily tables (SPLIT_TABLES) are split into `workers` date-range chunks so one
        # huge table doesn't bottleneck the pool, then concatenated back into one parquet.
        _download_tables_parallel(
            table_names,
            filenames,
            conninfo,
            date_columns,
            end_date,
            password,
            workers,
            split_tables=SPLIT_TABLES,
            n_chunks=workers,
        )
        return

    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")

    if persistent_connection:
        # Use ATTACH for a single persistent connection (reduces MFA on NAT-rotated networks).
        _attach_wrds(con, conninfo, password)
        try:
            for table in table_names:
                download_wrds_table_attached(
                    con,
                    "wrds",
                    table,
                    filenames[table],
                    date_column=date_columns.get(table),
                    end_date=end_date,
                )
        finally:
            con.execute("DETACH wrds")
    else:
        # Use postgres_scan() which creates a new connection per query (default)
        for table in table_names:
            download_wrds_table(
                conninfo,
                con,
                table,
                filenames[table],
                date_column=date_columns.get(table),
                end_date=end_date,
            )

    con.close()


@measure_time
def aug_msf_v2(paths: DataPaths):
    """
    Description:
        Add month-level high/low transaction-price fields to the CRSP CIZ monthly file (msf_v2)
        using the CRSP CIZ daily file (dsf_v2). Keep all msf_v2 rows; set the new fields to
        missing for non-TR monthly rows (e.g., BA). Called from gen_raw_data_dfs().

    Steps:
        1) Read msf_v2 (monthly) and dsf_v2 (daily) from raw_tables parquet.
        2) Filter dsf_v2 to dlyprcflg == "TR", construct yyyymm from dlycaldt, and keep (permno, yyyymm, dlyprc).
        3) For each (permno, yyyymm), compute:
           - mthaskhi = max(dlyprc)
           - mthbidlo = min(dlyprc)
        4) Left-join these two fields onto msf_v2 by (permno, yyyymm).
        5) Set mthaskhi/mthbidlo to NULL when mthprcflg != "TR".
        6) Write augmented table to raw_data_dfs/crsp_msf_v2_aug.parquet.

    Output:
        Writes raw_data_dfs/crsp_msf_v2_aug.parquet with all original columns
        plus mthaskhi and mthbidlo.
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    msf = con.read_parquet(paths.raw_tables_dir / "crsp_msf_v2.parquet")
    dsf = con.read_parquet(paths.raw_tables_dir / "crsp_dsf_v2.parquet")

    dt = dsf.dlycaldt.cast("date")

    d = (
        dsf.filter(dsf.dlyprcflg == "TR")
        .mutate(
            yyyymm=(dt.year() * 100 + dt.month()).cast("int32"),
            dlyprc=dsf.dlyprc.cast("double"),
        )
        .select(["permno", "yyyymm", "dlyprc"])
    )

    m = d.group_by(["permno", "yyyymm"]).aggregate(
        mthaskhi=d.dlyprc.max(),
        mthbidlo=d.dlyprc.min(),
    )

    msf_joined = msf.join(
        m,
        how="left",
        predicates=[msf.permno == m.permno, msf.yyyymm == m.yyyymm],
    ).select([msf] + [m.mthaskhi, m.mthbidlo])

    msf_aug = msf_joined.mutate(
        mthaskhi=ibis.cases(
            (msf_joined.mthprcflg == "TR", msf_joined.mthaskhi),
            else_=ibis.null(),
        ),
        mthbidlo=ibis.cases(
            (msf_joined.mthprcflg == "TR", msf_joined.mthbidlo),
            else_=ibis.null(),
        ),
    )

    msf_aug.to_parquet(paths.interim_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")


@measure_time
def build_mcti(paths: DataPaths):
    """
    Description:
        Build monthly t30 return raw data.

    Steps:
        1) Read indmthseriesdata and indseriesinfohdr from parquet.
        2) Inner join indmthseriesdata with indseriesinfohdr on "indno".
        4) Filter for CRSP 30-Year Treasury Returns (indno == 1000708).
        5) Write parquet to raw_data_dfs/crsp_mcti.parquet

    Output:
        Writes raw_data_dfs/crsp_mcti.parquet (no return value).
    """

    a = pl.read_parquet(paths.raw_tables_dir / "crsp_indmthseriesdata_ind.parquet")
    b = pl.read_parquet(paths.raw_tables_dir / "crsp_indseriesinfohdr_ind.parquet")

    ab = a.join(b, on="indno", how="inner")

    out = (
        ab.filter(pl.col("indno") == 1000708)
        .select(["indno", "indnm", "mthcaldt", "mthtotret", "mthtotind"])
        .rename({"mthcaldt": "caldt", "mthtotret": "t30ret"})
    )

    (paths.interim_dir / "raw_data_dfs").mkdir(exist_ok=True)
    out.write_parquet(paths.interim_dir / "raw_data_dfs" / "crsp_mcti.parquet")


@measure_time
def prepare_comp_sf(paths: DataPaths, freq):
    """
    Description:
        Prepare Compustat security-file derivatives (Comp DSF/SSF equivalents) for daily/monthly runs.

    Steps:
        1) Ensure firm-shares table is populated (populate_own), then create Comp DSF (gen_comp_dsf).
        2) Run process_comp_sf1 for requested frequency: 'd', 'm', or 'both'.

    Output:
        Intermediate Comp security files written by downstream helpers (no direct return).
    """
    populate_own(
        paths,
        paths.interim_dir / "raw_data_dfs" / "__firm_shares1.parquet",
        "gvkey",
        "datadate",
        "ddate",
    )
    gen_comp_dsf(paths)
    if freq == "both":
        process_comp_sf1(paths, "d")
        process_comp_sf1(paths, "m")
    else:
        process_comp_sf1(paths, freq)


def populate_own(paths: DataPaths, inset_path, idvar, datevar, datename):
    """
    Description:
        Expand Compustat firm-shares observations to a DAILY panel between each report
        date and the earlier of (next report date) or (report date + 12 months month-end).

    Steps:
        1) Read inset_path and drop duplicates on {idvar, datevar}.
        2) Copy datevar into a new column named `datename` (used as start date 'ddate').
        3) For each {idvar}, compute boundary `n` = min(next datadate, datadate+12m month-end) − 1 day.
        4) Generate pl.date_ranges(ddate, n) per row, explode to daily rows.
        5) Keep daily rows with columns {ddate, gvkey, datadate, csho_fund, ajex_fund}.
        6) Sort and write to __firm_shares2.parquet.

    Output:
        Parquet: __firm_shares2.parquet with daily firm-share data aligned to report windows.
    """
    inset = (
        pl.scan_parquet(inset_path)
        .unique([idvar, datevar])
        .with_columns(col(datevar).alias(datename))
        .sort([idvar, datevar])
        .with_columns(
            n=pl.min_horizontal(
                (col("datadate").shift(-1)).over(idvar),
                col(datevar).dt.offset_by("12mo").dt.month_end(),
            ).dt.offset_by("-1d")
        )
        .with_columns(pl.date_ranges("ddate", "n"))
        .explode("ddate")
        .select(["ddate", "gvkey", "datadate", "csho_fund", "ajex_fund"])
        .sort(["gvkey", "datadate"])
    )
    inset.collect().write_parquet(paths.interim_dir / "__firm_shares2.parquet")


def compustat_fx(paths: DataPaths):
    """
    Description:
        Construct a complete daily FX (currency to USD) time series per currency, including USD=1.
        Fills gaps by expanding each FX observation to all dates up to the next observation.

    Steps:
        1) Create a seed row for USD with fx=1.0 starting 1950-01-01.
        2) Load __fx1 (curcdd, datadate, fx) and vertically concat the USD seed.
        3) For each curcdd, compute next datadate; build a date range [datadate, next) per row.
        4) Explode ranges to daily rows; fallback to single-day list if next is null.
        5) Drop duplicates, sort by {curcdd, datadate}, and collect to a DataFrame.

    Output:
        Polars DataFrame with columns {datadate, curcdd, fx} at daily frequency (to USD).
    """
    aux = (
        pl.DataFrame({"curcdd": "USD", "datadate": "1950-01-01", "fx": 1.0})
        .with_columns(col("datadate").str.to_date("%Y-%m-%d"))
        .lazy()
    )
    __fx1 = pl.scan_parquet(paths.interim_dir / "raw_data_dfs" / "__fx1.parquet")
    __fx1 = (
        pl.concat([aux, __fx1], how="vertical_relaxed")
        .sort(["curcdd", "datadate"])
        .with_columns(aux=col("datadate").shift(-1).over("curcdd"))
        .with_columns(
            datadate=pl.coalesce(
                [
                    pl.date_ranges(start="datadate", end="aux", interval="1d", closed="left"),
                    pl.concat_list([col("datadate")]),
                ]
            )
        )
        .select(["datadate", "curcdd", "fx"])
        .explode("datadate")
        .unique(["curcdd", "datadate"])
        .sort(["curcdd", "datadate"])
    )
    return __fx1.collect()


def adj_trd_vol_NASDAQ(datevar, col_to_adjust, is_nasdaq_expr):
    """
    Description:
        Apply historic NASDAQ trade-volume adjustments (pre-decimalization reporting) to a volume column.

    Steps:
        1) Build date cutoffs: <2001-02-01, ≤2001-12-31, <2003-12-31.
        2) If is_nasdaq_expr is true and within windows, scale col_to_adjust by
        1/2, 1/1.8, or 1/1.6 respectively; otherwise keep original.
        3) Return the adjusted expression aliased as the original column name.

    Output:
        Polars expression that yields adjusted trade volume for NASDAQ histories.
    """
    c1 = is_nasdaq_expr
    c2 = col(datevar) < pl.datetime(2001, 2, 1)
    c3 = col(datevar) <= pl.datetime(2001, 12, 31)
    c4 = col(datevar) < pl.datetime(2003, 12, 31)
    adj_trd_vol = (
        pl.when(c1 & c2)
        .then(col(col_to_adjust) / 2)
        .when(c1 & c3)
        .then(col(col_to_adjust) / 1.8)
        .when(c1 & c4)
        .then(col(col_to_adjust) / 1.6)
        .otherwise(col(col_to_adjust))
    ).alias(col_to_adjust)
    return adj_trd_vol


def gen_comp_dsf(paths: DataPaths):
    """
    Description:
        Build daily Compustat security data (SECD + G_SECD), convert to USD, and compute
        prices/returns/volumes/dividends; store as __comp_dsf.parquet.

    Steps:
        1) Materialize daily FX to fx_data.parquet; register SECD, G_SECD, firm-shares, and FX in DuckDB.
        2) Create __comp_dsf_global from G_SECD: local prices, highs/lows (if prcstd≠5),
        shares traded, shares outstanding, local return index (ri_local), dividend currencies.
        3) Create __comp_dsf_na from SECD with same fields; infer cshoc from firm-shares when missing.
        4) Adjust NASDAQ (exchg=14) cshtrd by historical factors (2001 windows).
        5) FULL OUTER JOIN NA and Global records; LEFT JOIN daily FX for trading and dividend currencies.
        6) Compute USD variables: prc, prc_high, prc_low, market cap (me), USD turnover (dolvol),
        USD return index (ri), dividends (split into total/cash/special); derive month-end eom.
        7) Drop intermediates and write __comp_dsf.parquet.

    Output:
        Parquet: __comp_dsf.parquet (daily Compustat security observations in USD).
    """
    (paths.interim_dir / "aux_comp_dsf.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(str(paths.interim_dir / "aux_comp_dsf.ddb"), threads=os.cpu_count())

    compustat_fx(paths).write_parquet(paths.interim_dir / "fx_data.parquet")
    con.create_table("comp_g_secd", con.read_parquet(paths.raw_tables_dir / "comp_g_secd.parquet"))
    con.create_table(
        "__firm_shares2", con.read_parquet(paths.interim_dir / "__firm_shares2.parquet")
    )
    con.create_table("comp_secd", con.read_parquet(paths.raw_tables_dir / "comp_secd.parquet"))
    con.create_table("fx", con.read_parquet(paths.interim_dir / "fx_data.parquet"))

    con.raw_sql("""
    CREATE TABLE __comp_dsf_global AS
    SELECT
        gvkey, iid, datadate, tpci, exchg, prcstd, curcdd, prccd / qunit AS prc_local, ajexdi, cshoc / 1e6 AS cshoc,
        CASE
            WHEN prcstd != 5 THEN prchd / qunit
            ELSE NULL
        END AS prc_high_lcl,
        CASE
            WHEN prcstd != 5 THEN prcld / qunit
            ELSE NULL
        END AS prc_low_lcl,
        cshtrd, (prccd / qunit) / ajexdi * trfd AS ri_local,
        curcddv, div, divd, divsp
    FROM comp_g_secd;

    CREATE TABLE __comp_dsf_na AS
    SELECT
        a.gvkey, a.iid, a.datadate, a.tpci, a.exchg, a.prcstd, a.curcdd, a.prccd AS prc_local, a.ajexdi,
        CASE
            WHEN a.prcstd != 5 THEN a.prchd
            ELSE NULL
        END AS prc_high_lcl,
        CASE
            WHEN a.prcstd != 5 THEN a.prcld
            ELSE NULL
        END AS prc_low_lcl,
        a.cshtrd, COALESCE(a.cshoc / 1e6, b.csho_fund * b.ajex_fund / a.ajexdi) AS cshoc,
        (a.prccd / a.ajexdi * a.trfd) AS ri_local, a.curcddv, a.div, a.divd, a.divsp
    FROM comp_secd AS a
    LEFT JOIN __firm_shares2 AS b
    ON a.gvkey = b.gvkey AND a.datadate = b.ddate;

    UPDATE __comp_dsf_na
    SET cshtrd =
        CASE
            WHEN datadate <  DATE '2001-02-01' THEN cshtrd / 2
            WHEN datadate <= DATE '2001-12-31' THEN cshtrd / 1.8
            WHEN datadate <  DATE '2003-12-31' THEN cshtrd / 1.6
            ELSE cshtrd
        END
    WHERE exchg = 14;

    CREATE TABLE __comp_dsf1 AS
    SELECT *
    FROM __comp_dsf_na
    FULL OUTER JOIN __comp_dsf_global
    USING (gvkey, iid, datadate, tpci, exchg, prcstd, curcdd, prc_local, ajexdi, prc_high_lcl, prc_low_lcl, cshtrd, cshoc, ri_local, curcddv, div, divd, divsp);

    CREATE TABLE __comp_dsf2 AS
    SELECT a.*, b.fx AS fx, c.fx AS fx_div
    FROM __comp_dsf1 AS a
    LEFT JOIN fx AS b
        ON a.curcdd = b.curcdd AND a.datadate = b.datadate
    LEFT JOIN fx AS c
        ON a.curcddv = c.curcdd AND a.datadate = c.datadate;

    CREATE TABLE __comp_dsf3 AS
    SELECT
        *,
        prc_local    * fx AS prc,
        prc_high_lcl * fx AS prc_high,
        prc_low_lcl  * fx AS prc_low,
        (prc_local   * fx) * cshoc AS me,
        cshtrd       * (prc_local * fx) AS dolvol,
        ri_local     * fx AS ri,
        COALESCE(div, 0)   * fx_div AS div_tot,
        COALESCE(divd, 0)  * fx_div AS div_cash,
        COALESCE(divsp, 0) * fx_div AS div_spc,
        last_day(datadate) AS eom

    FROM __comp_dsf2;

    """)
    t = con.table("__comp_dsf3").drop(
        ["div", "divd", "divsp", "fx_div", "curcddv", "prc_high_lcl", "prc_low_lcl"]
    )
    t.to_parquet(paths.interim_dir / "__comp_dsf.parquet")
    con.disconnect()


def gen_secd_data(paths: DataPaths):
    """
    Description:
        Aggregate daily Compustat (__comp_dsf) to month-end security data (SECD-like monthly),
        resolving multiple daily rows and computing monthly highs/lows/divs/volume.

    Steps:
        1) Read __comp_dsf.parquet into DuckDB.
        2) Define window over {gvkey, iid, eom}.
        3) Compute monthly aggregates at adjusted units:
        - prc_highm/prc_lowm: month max/min using pre-adjusted local price with ajexdi.
        - div_*m: sum of monthly dividends (by type) adjusted.
        - cshtrm: sum of adjusted share turnover; dolvolm: sum of USD dollar volume.
        - source=1 to mark SECD pipeline.
        4) Keep valid rows (local price/currency present, prcstd in {3,4,10});
        pick the last datadate in the month.
        5) Write to secd_data.parquet.

    Output:
        Parquet: secd_data.parquet (monthly aggregates from daily Compustat pipeline).
    """
    (paths.interim_dir / "aux_msf.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(str(paths.interim_dir / "aux_msf.ddb"), threads=os.cpu_count())
    table = con.read_parquet(paths.interim_dir / "__comp_dsf.parquet")
    window = ibis.window(group_by=["gvkey", "iid", "eom"])
    new_table = (
        table.mutate(
            prc_highm=ibis.greatest(
                (_.prc / _.ajexdi).max().over(window),
                (_.prc_high / _.ajexdi).max().over(window),
            )
            * _.ajexdi,
            prc_lowm=ibis.least(
                (_.prc / _.ajexdi).min().over(window),
                (_.prc_low / _.ajexdi).min().over(window),
            )
            * _.ajexdi,
            div_totm=(_.div_tot / _.ajexdi).sum().over(window) * _.ajexdi,
            div_cashm=(_.div_cash / _.ajexdi).sum().over(window) * _.ajexdi,
            div_spcm=(_.div_spc / _.ajexdi).sum().over(window) * _.ajexdi,
            cshtrm=(_.cshtrd / _.ajexdi).sum().over(window) * _.ajexdi,
            dolvolm=_.dolvol.sum().over(window),
            source=1,
        )
        .filter((_.prc_local.notnull()) & (_.curcdd.notnull()) & (_.prcstd.isin([3, 4, 10])))
        .mutate(max_date=_.datadate.max().over(window))
        .filter(_.max_date == _.datadate)
        .drop(
            [
                "cshtrd",
                "div_tot",
                "div_cash",
                "div_spc",
                "dolvol",
                "prc_high",
                "prc_low",
                "max_date",
            ]
        )
        .rename(
            {
                "div_tot": "div_totm",
                "div_cash": "div_cashm",
                "div_spc": "div_spcm",
                "dolvol": "dolvolm",
                "prc_high": "prc_highm",
                "prc_low": "prc_lowm",
            }
        )
        .order_by(["gvkey", "iid", "eom"])
    )
    new_table.to_parquet(paths.interim_dir / "secd_data.parquet")
    con.disconnect()


def gen_secm_data(paths: DataPaths):
    """
    Description:
        Build Compustat SECM-derived monthly records (direct monthly pricing),
        harmonize units, convert to USD, and output month-end rows.

    Steps:
        1) Materialize daily FX as fx_data.parquet; register SECM, firm-shares, and FX in DuckDB.
        2) In CTE:
        - Map SECM fields to local price/hi/low/ajex, compute cshoc fallback from firm-shares,
            adjust NASDAQ cshtrm by historical factors.
        - Join FX for trading and dividend currencies; compute ri_local.
        3) Project to final fields: USD prc, prc_high/low, ME, dolvol, RI, total dividends (cash/special null),
        prcstd=10, source=0, and eom=last_day(datadate).
        4) Write to secm_data.parquet.

    Output:
        Parquet: secm_data.parquet (monthly Compustat SECM-based observations in USD).
    """
    (paths.interim_dir / "aux_comp_secm.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(str(paths.interim_dir / "aux_comp_secm.ddb"), threads=os.cpu_count())

    compustat_fx(paths).rename({"datadate": "date"}).write_parquet(
        paths.interim_dir / "fx_data.parquet"
    )
    con.create_table(
        "comp_secm",
        con.read_parquet(paths.raw_tables_dir / "comp_secm.parquet"),
        overwrite=True,
    )
    con.create_table(
        "__firm_shares2",
        con.read_parquet(paths.interim_dir / "__firm_shares2.parquet"),
        overwrite=True,
    )
    con.create_table("fx", con.read_parquet(paths.interim_dir / "fx_data.parquet"), overwrite=True)

    con.raw_sql("""
        DROP TABLE IF EXISTS __comp_secm2;

        CREATE TABLE __comp_secm2 AS
        WITH base AS (
        SELECT
            a.gvkey, a.iid, a.datadate, last_day(a.datadate) AS eom, a.tpci, a.exchg, a.dvpsxm,
            a.curcdm       AS curcdd,
            a.prccm        AS prc_local,
            a.prchm        AS prc_high_local,
            a.prclm        AS prc_low_local,
            a.ajexm        AS ajexdi,
            coalesce(a.cshom/1e6, a.csfsm/1e3, a.cshoq, b.csho_fund * b.ajex_fund / a.ajexm) AS cshoc,
            CASE
            WHEN a.exchg = 14 AND a.datadate <  DATE '2001-02-01' THEN a.cshtrm/2
            WHEN a.exchg = 14 AND a.datadate <= DATE '2001-12-31' THEN a.cshtrm/1.8
            WHEN a.exchg = 14 AND a.datadate <  DATE '2003-12-31' THEN a.cshtrm/1.6
            ELSE a.cshtrm
            END AS cshtrm,
            CASE WHEN a.curcdm    = 'USD' THEN 1 ELSE c.fx END AS fx,
            CASE WHEN a.curcddvm  = 'USD' THEN 1 ELSE d.fx END AS fx_div,
            a.prccm / a.ajexm * a.trfm AS ri_local
        FROM comp_secm AS a
        LEFT JOIN __firm_shares2 AS b
            ON a.gvkey    = b.gvkey  AND a.datadate = b.ddate
        LEFT JOIN fx AS c
            ON a.curcdm   = c.curcdd AND a.datadate = c.date
        LEFT JOIN fx AS d
            ON a.curcddvm = d.curcdd AND a.datadate = d.date
        )
        SELECT
        gvkey    , iid   , datadate, eom   , tpci    , exchg, curcdd,
        prc_local, ajexdi, cshoc   , cshtrm, ri_local, fx,
        0 AS source,
        10 as prcstd,
        prc_high_local * fx AS prc_high,
        prc_low_local  * fx AS prc_low,
        prc_local      * fx                AS prc,
        prc_local      * fx * cshoc        AS me,
        cshtrm         * fx * prc_local    AS dolvol,
        ri_local       * fx                AS ri,
        dvpsxm         * fx_div            AS div_tot,
        NULL::DOUBLE                       AS div_cash,
        NULL::DOUBLE                       AS div_spc
        FROM base;
    """)
    con.table("__comp_secm2").to_parquet(paths.interim_dir / "secm_data.parquet")


def gen_comp_msf(paths: DataPaths):
    """
    Description:
        Merge SECD-based and SECM-based monthly datasets into a single Compustat MSF-like file.
        Prefer SECD (source=1) when both sources exist for the same {gvkey,iid,eom}.

    Steps:
        1) Ensure secd_data.parquet and secm_data.parquet exist (run gen_secd_data/gen_secm_data).
        2) Load both; select a common column set and cast types consistently.
        3) UNION the two sources; for each {gvkey,iid,eom} window count n rows.
        4) Keep either the single row (n=1) or, if n=2, prefer source=1 (SECD).
        5) Drop helper columns and deduplicate on {gvkey, iid, eom}, keeping the
           row with the latest datadate (closest to month-end).
        6) Write to __comp_msf.parquet.

    Output:
        Parquet: __comp_msf.parquet (monthly Compustat security master in USD).
    """
    gen_secd_data(paths)
    gen_secm_data(paths)
    common_vars = [
        "gvkey",
        "iid",
        "datadate",
        "eom",
        "tpci",
        "exchg",
        "curcdd",
        "prc_local",
        "prc_high",
        "prc_low",
        "ajexdi",
        "cshoc",
        "ri_local",
        "fx",
        "prc",
        "me",
        "cshtrm",
        "dolvol",
        "ri",
        "div_tot",
        "div_cash",
        "div_spc",
        "prcstd",
        "source",
    ]
    (paths.interim_dir / "aux_msf.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(str(paths.interim_dir / "aux_msf.ddb"), threads=os.cpu_count())
    secd = (
        con.read_parquet(paths.interim_dir / "secd_data.parquet")
        .select(common_vars)
        .cast({"ajexdi": "float", "prc_local": "float"})
    )
    secm = (
        con.read_parquet(paths.interim_dir / "secm_data.parquet")
        .select(common_vars)
        .cast({"ajexdi": "float", "prc_local": "float"})
    )
    window = ibis.window(group_by=["gvkey", "iid", "eom"], order_by="datadate")
    # Deterministic dedup: keep latest datadate per {gvkey, iid, eom}.
    # Investigation (issue #69) found 0 duplicates in current data, but the
    # original .distinct() was non-deterministic — different runs/engines could
    # pick different rows if duplicates appear in a future data vintage.
    dedup_window = ibis.window(
        group_by=["gvkey", "iid", "eom"],
        order_by=ibis.desc("datadate"),
    )
    __comp_msf = (
        secd.union(secm)
        .mutate(n=_.gvkey.count().over(window))
        .filter([(_.n == 1) | ((_.n == 2) & (_.source == 1))])
        .drop(["n", "source"])
        .mutate(_rn=ibis.row_number().over(dedup_window))
        .filter(_._rn == 0)
        .drop("_rn")
    )
    __comp_msf.to_parquet(paths.interim_dir / "__comp_msf.parquet")
    con.disconnect()


def comp_exchanges(paths: DataPaths):
    """
    Description:
        Build an exchange→country lookup with “main exchange” flag for Compustat exchanges.

    Steps:
        1) Define a list of special exchange codes that should not be flagged as main.
        2) Using __ex_country1 (exchg→excntry pairs):
        a) SQL: for each exchg, if it maps to multiple countries label 'multi national',
            else use its single country.
        b) Join to comp_r_ex_codes for descriptions/codes; cast exchg to int64.
        3) Compute exch_main = 1 when (excntry ≠ 'multi national') and exchg not in special_exchanges; else 0.
        4) Return the resulting Polars DataFrame.

    Output:
        DataFrame mapping exchg → {excntry, exchgdesc/exchgcd if present, exch_main}.
    """
    special_exchanges = [
        0,
        1,
        2,
        3,
        4,
        15,
        16,
        17,
        18,
        21,
        13,
        19,
        20,
        127,
        150,
        157,
        229,
        263,
        269,
        281,
        283,
        290,
        320,
        326,
        341,
        342,
        347,
        348,
        349,
        352,
    ]
    # 15, 16, 17, 18, 21 US exchanges not in NYSE, Amex and NASDAQ
    # 150 AIAF Mercado De Renta Fija --> Spanish exchange for trading debt securities https://practiceguides.chambers.com/practice-guides/capital-markets-debt-2019/spain/1-debt-marketsexchanges
    # 349 BATS Chi-X Europe --> Trades stocks from various european exchanges. Should we keep it?
    # 352 CHI-X Australia --> Only Trades securities listed on ASX (exchg=106). Should we keep it?
    SQL_query = """
        SELECT DISTINCT exchg,
            CASE
                WHEN COUNT(DISTINCT excntry) > 1 THEN 'multi national'
                ELSE MAX(excntry)
            END AS excntry
        FROM frame
        WHERE excntry IS NOT NULL AND exchg IS NOT NULL
        GROUP BY exchg
        """
    exch_exp = (
        pl.when(
            (col("excntry") != "multi national") & (col("exchg").is_in(special_exchanges).not_())
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("exch_main")
    )
    comp_r_ex_codes = pl.read_parquet(
        paths.interim_dir / "raw_data_dfs" / "comp_r_ex_codes.parquet"
    )
    __ex_country = pl.read_parquet(paths.interim_dir / "raw_data_dfs" / "__ex_country1.parquet")
    __ex_country = (
        pl.SQLContext(frame=__ex_country)
        .execute(SQL_query)
        .collect()
        .sort("exchg")
        .join(comp_r_ex_codes, how="left", left_on="exchg", right_on="exchgcd")
        .with_columns(col("exchg").cast(pl.Int64))
        .with_columns(exch_exp)
    )
    return __ex_country


def add_primary_sec(paths: DataPaths, data_path, datevar, file_name):
    """
    Description:
        Flag primary securities by joining Compustat primary-history tables (ROW/USA/CAN)
        onto a security-level dataset and comparing iid to recorded primary identifiers
        over valid date ranges. Falls back to company header when history is missing.

    Steps:
        1) Open DuckDB; read: data_path, __prihistrow/usa/can, and __header (prirow/priusa/prican).
        2) Range-join each PRI table on {gvkey} where datevar ∈ [effdate, thrudate]; drop join keys.
        3) Left-join header; coalesce prihist* with pri* when historical value is null.
        4) Compute primary_sec = 1 if iid matches any of {prihistrow, prihistusa, prihistcan}; else 0.
        5) Deduplicate on (gvkey, iid, datadate), preferring primary_sec=1 when
           range-join fan-out produces conflicting classifications; sort and write file_name.

    Output:
        Parquet at file_name with a new integer column primary_sec ∈ {0,1}.
    """
    (paths.interim_dir / "aux_prim_sec.ddb").unlink(missing_ok=True)
    con = duckdb.connect(str(paths.interim_dir / "aux_prim_sec.ddb"))
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    data_path_posix = Path(data_path).as_posix()
    prihistrow_path = (raw_data_dfs / "__prihistrow.parquet").as_posix()
    prihistusa_path = (raw_data_dfs / "__prihistusa.parquet").as_posix()
    prihistcan_path = (raw_data_dfs / "__prihistcan.parquet").as_posix()
    header_path = (raw_data_dfs / "__header.parquet").as_posix()
    file_name_posix = Path(file_name).as_posix()
    con.execute(f"""
        CREATE OR REPLACE TABLE __data2 AS
        WITH data AS (
            SELECT * FROM read_parquet('{data_path_posix}')
            ORDER BY gvkey, iid, datadate
        ),
        prihistrow AS (
            SELECT * FROM read_parquet('{prihistrow_path}')
            ORDER BY gvkey, effdate
        ),
        prihistusa AS (
            SELECT * FROM read_parquet('{prihistusa_path}')
            ORDER BY gvkey, effdate
        ),
        prihistcan AS (
            SELECT * FROM read_parquet('{prihistcan_path}')
            ORDER BY gvkey, effdate
        ),
        header AS (
            SELECT * FROM read_parquet('{header_path}')
            ORDER BY gvkey
        ),
        __data1 AS (
            SELECT DISTINCT
                a.*,
                COALESCE(b.prihistrow, e.prirow)  AS prihistrow,
                COALESCE(c.prihistusa, e.priusa)  AS prihistusa,
                COALESCE(d.prihistcan, e.prican)  AS prihistcan
            FROM data AS a
            LEFT JOIN prihistrow AS b
                ON a.gvkey = b.gvkey
                AND a.{datevar} BETWEEN b.effdate AND COALESCE(b.thrudate, DATE '2262-04-11')
            LEFT JOIN prihistusa AS c
                ON a.gvkey = c.gvkey
                AND a.{datevar} BETWEEN c.effdate AND COALESCE(c.thrudate, DATE '2262-04-11')
            LEFT JOIN prihistcan AS d
                ON a.gvkey = d.gvkey
                AND a.{datevar} BETWEEN d.effdate AND COALESCE(d.thrudate, DATE '2262-04-11')
            LEFT JOIN header AS e
                ON a.gvkey = e.gvkey
        )
        SELECT
            * EXCLUDE (prihistrow, prihistusa, prihistcan),
            CASE
                WHEN iid IS NOT NULL
                    AND COALESCE(
                        (iid = prihistrow) OR (iid = prihistusa) OR (iid = prihistcan),
                        FALSE
                        )
                THEN 1
                ELSE 0
            END AS primary_sec
        FROM __data1
        ORDER BY gvkey, iid, datadate;

        -- Deterministic dedup: prefer primary_sec=1 when range-join fan-out
        -- produces conflicting classifications for the same (gvkey, iid, datadate).
        -- Investigation (issue #69) found 40 affected groups in current data,
        -- all for gvkey 327360 (iids 02W, 38W) where overlapping prihist records
        -- yield both primary_sec=0 and primary_sec=1 for the same observation.
        -- Preferring primary_sec=1 avoids wrongly excluding primary securities.
        COPY (
            SELECT DISTINCT ON (gvkey, iid, {datevar}) *
            FROM __data2
            ORDER BY gvkey, iid, {datevar}, primary_sec DESC
        ) TO '{file_name_posix}' (FORMAT parquet);
    """)

    con.close()
    (paths.interim_dir / "aux_prim_sec.ddb").unlink(missing_ok=True)


def load_rf_and_exchange_data(paths: DataPaths):
    """
    Description:
        Load auxiliary monthly T-bill returns, Fama–French RF, and exchange→country mapping.

    Steps:
        1) Read crsp_mcti_t30ret and create merge_aux = MMYY(caldt); drop caldt.
        2) Read ff_factors_monthly and create merge_aux = MMYY(date); drop date.
        3) Build exchange mapping via comp_exchanges().

    Output:
        Tuple of (crsp_mcti LazyFrame, ff_factors_monthly LazyFrame, exchanges DataFrame).
    """
    crsp_mcti = (
        pl.read_parquet(paths.interim_dir / "raw_data_dfs" / "crsp_mcti_t30ret.parquet")
        .with_columns(merge_aux=gen_MMYY_column("caldt"))
        .drop("caldt")
    )
    ff_factors_monthly = (
        pl.read_parquet(paths.interim_dir / "raw_data_dfs" / "ff_factors_monthly.parquet")
        .with_columns(merge_aux=gen_MMYY_column("date"))
        .drop("date")
    )
    __exchanges = comp_exchanges(paths)
    return crsp_mcti, ff_factors_monthly, __exchanges


def gen_returns_df(paths: DataPaths, freq):
    """
    Description:
        Compute returns (USD and local), return-lag gaps, and currency-switch fixes
        from Compustat security files (__comp_?sf.parquet).

    Steps:
        1) Determine ret_lag_dif: monthly via MMYY difference; daily via day difference.
        2) Filter valid rows (ri not null; prcstd in {3,4,10}); deduplicate on
           {gvkey,iid,datadate} keeping highest prcstd (best data quality); sort.
        3) Compute ret and ret_local as pct_change of ri and ri_local over (gvkey,iid).
        4) If iid unchanged but currency changed, set ret_local = ret (reset local base).
        5) Null-out ±∞/NaN returns; select core columns and collect.

    Output:
        Polars DataFrame with {gvkey,iid,datadate,ret,ret_local,ret_lag_dif}.
    """
    ret_lag_dif_exp = (
        (gen_MMYY_column("datadate") - gen_MMYY_column("datadate", 1)).over(["gvkey", "iid"])
        if freq == "m"
        else (
            (col("datadate") - col("datadate").shift(1)).over(["gvkey", "iid"]) / 86_400_000_000
        ).cast(pl.Int64)
    )

    base = pl.scan_parquet(paths.interim_dir / f"__comp_{freq}sf.parquet")
    __returns = (
        # Deterministic dedup: keep highest prcstd (Compustat data-quality ranking:
        # 3=bid/ask avg, 4=official close, 10=last available price).
        # Investigation (issue #69) found 0 duplicates in current monthly and daily
        # data, but the original .unique() was non-deterministic. The upstream
        # FULL OUTER JOIN in gen_comp_dsf() could produce duplicates if NA and
        # Global Compustat have differing computed values for the same security-date.
        base.filter((col("ri").is_not_null()) & (col("prcstd").is_in([3, 4, 10])))
        .sort(["gvkey", "iid", "datadate", "prcstd"])
        .unique(["gvkey", "iid", "datadate"], keep="last")
        .sort(["gvkey", "iid", "datadate"])
        .with_columns(
            ret=col("ri").pct_change().over(["gvkey", "iid"]),
            ret_local=col("ri_local").pct_change().over(["gvkey", "iid"]),
            ret_lag_dif=ret_lag_dif_exp,
            lagged_iid=col("iid").shift(1).over(["gvkey", "iid"]),
            lagged_curcdd=col("curcdd").shift(1).over(["gvkey", "iid"]),
        )
        .with_columns(
            ret_local=pl.when(
                (col("iid") == col("lagged_iid")) & (col("curcdd") != col("lagged_curcdd"))
            )
            .then(col("ret"))
            .otherwise(col("ret_local"))
        )
        .with_columns(
            ret_local=pl.when(col("ret_local").is_infinite() | col("ret_local").is_nan())
            .then(None)
            .otherwise(col("ret_local")),
            ret=pl.when(col("ret").is_infinite() | col("ret").is_nan())
            .then(None)
            .otherwise(col("ret")),
        )
        .select(["gvkey", "iid", "datadate", "ret", "ret_local", "ret_lag_dif"])
    )
    return __returns.collect()


def gen_delist_df(paths: DataPaths, __returns):
    """
    Description:
        Build an gvkey,iid-level delisting date/return table from last valid return and Compustat security info.

    Steps:
        1) From __returns, keep final nonzero/non-null ret_local per (gvkey,iid).
        2) Join __sec_info to get secstat/dlrsni; keep inactive (secstat='I').
        3) Map delisting code {02,03} → dlret = -0.30 else 0.0; rename columns.

    Output:
        DataFrame {gvkey,iid,date_delist,dlret} for use in delisting adjustments.
    """
    __sec_info = pl.scan_parquet(paths.interim_dir / "raw_data_dfs" / "__sec_info.parquet")
    __delist = (
        __returns.lazy()
        .filter((col("ret_local").is_not_null()) & (col("ret_local") != 0.0))
        .select(["gvkey", "iid", "datadate"])
        .sort(["gvkey", "iid", "datadate"])
        .unique(["gvkey", "iid"], keep="last")
        .join(__sec_info, how="left", on=["gvkey", "iid"])
        .rename({"datadate": "date_delist"})
        .filter(col("secstat") == "I")
        .with_columns(
            dlret=pl.when(col("dlrsni").is_in(["02", "03"]))
            .then(pl.lit(-0.3))
            .otherwise(pl.lit(0.0))
        )
        .select(["gvkey", "iid", "date_delist", "dlret"])
    )
    return __delist.collect()


def gen_temporary_sf(paths: DataPaths, freq, __returns, __delist):
    """
    Description:
        Merge base price file with returns and delist info; apply delisting
        returns on the delisting date and truncate after delisting.

    Steps:
        1) Join base (Compustat sf) with __returns and __delist on identifiers.
        2) Keep rows where datadate ≤ date_delist or date_delist is null.
        3) On delist date, compound dlret into both ret and ret_local.
        4) Drop raw RI and delist helper columns.

    Output:
        Polars LazyFrame with returns adjusted for delisting.
    """
    base = pl.read_parquet(paths.interim_dir / f"__comp_{freq}sf.parquet")
    temp_sf = (
        base.join(__returns, how="left", on=["gvkey", "iid", "datadate"])
        .join(__delist, how="left", on=["gvkey", "iid"])
        .filter((col("datadate") <= col("date_delist")) | (col("date_delist").is_null()))
        .with_columns(
            ret=pl.when(col("datadate") == col("date_delist"))
            .then((1 + col("ret")) * (1 + col("dlret")) - 1)
            .otherwise(col("ret")),
            ret_local=pl.when(col("datadate") == col("date_delist"))
            .then((1 + col("ret_local")) * (1 + col("dlret")) - 1)
            .otherwise(col("ret_local")),
        )
        .drop(["ri", "ri_local", "date_delist", "dlret"])
    )
    return temp_sf


def add_rf_and_exchange_data_to_temporary_sf(paths: DataPaths, freq, temp_sf):
    """
    Description:
        Append T-bill / RF and exchange metadata to the temp security file; compute excess returns.

    Steps:
        1) Add merge_aux = MMYY(datadate); left-join T-bill (t30ret) and FF RF on merge_aux.
        2) Compute ret_exc = ret − (t30ret or rf)/scale, with scale=1 (m) or 21 (d).
        3) Cast exchg to int64 and join exchange-country mapping.

    Output:
        Polars LazyFrame temp_sf with ret_exc and exchange info attached.
    """
    crsp_mcti, ff_factors_monthly, __exchanges = load_rf_and_exchange_data(paths)
    scale = 1 if (freq == "m") else 21
    temp_sf = (
        temp_sf.with_columns(merge_aux=gen_MMYY_column("datadate"))
        .join(crsp_mcti, how="left", on="merge_aux")
        .join(ff_factors_monthly, how="left", on="merge_aux")
        .with_columns(ret_exc=col("ret") - pl.coalesce(["t30ret", "rf"]) / scale)
        .drop(["merge_aux", "rf", "t30ret"])
        .with_columns(col("exchg").cast(pl.Int64))
        .join(__exchanges, how="left", on=["exchg"])
    )
    return temp_sf


def process_comp_sf1(paths: DataPaths, freq):
    """
    Description:
        Full pipeline to build Compustat monthly or daily security files with returns,
        excess returns, exchange flags, and primary_sec indicator.

    Steps:
        1) If monthly, run gen_comp_msf() to ensure comp_msf/parquets exist.
        2) Compute __returns → gen_delist_df → gen_temporary_sf.
        3) Add RF/exchange metadata; write __comp_sf2.parquet.
        4) Call add_primary_sec(...) to add primary_sec and write final comp_{freq}sf.parquet.

    Output:
        comp_msf.parquet or comp_dsf.parquet with enriched fields (ret_exc, primary_sec, etc.).
    """
    # Eager mode is faster here
    if freq == "m":
        gen_comp_msf(paths)
    __returns = gen_returns_df(paths, freq)
    __delist = gen_delist_df(paths, __returns)
    __comp_sf2 = gen_temporary_sf(paths, freq, __returns, __delist)
    __comp_sf2 = add_rf_and_exchange_data_to_temporary_sf(paths, freq, __comp_sf2)
    __comp_sf2.write_parquet(paths.interim_dir / "__comp_sf2.parquet")
    del __comp_sf2
    add_primary_sec(
        paths,
        paths.interim_dir / "__comp_sf2.parquet",
        "datadate",
        paths.interim_dir / f"comp_{freq}sf.parquet",
    )


def gen_MMYY_column(var, shift=None):
    """
    Description:
        Create an integer YYYY*12+MM index (optionally using lagged date).

    Steps:
        1) If shift is None: use var; else use var shifted by 1 over its group (as provided by caller).
        2) Compute year*12 + month; cast to Int32.

    Output:
        Polars expression yielding a month index (MMYY integer).
    """
    if shift is None:
        return (col(var).dt.year() * 12 + col(var).dt.month()).cast(pl.Int32)
    else:
        return (col(var).shift(1).dt.year() * 12 + col(var).shift(1).dt.month()).cast(pl.Int32)


def add_MMYY_column_drop_original(df, var):
    """
    Description:
        Convenience helper to add merge_aux = MMYY(var) and drop var.

    Steps:
        1) with_columns(merge_aux = gen_MMYY_column(var)).
        2) drop original date column var.

    Output:
        LazyFrame with merge_aux month index.
    """
    return df.with_columns(merge_aux=gen_MMYY_column(var)).drop(var)


@measure_time
def prepare_crsp_sf(paths: DataPaths, freq):
    """
    Description:
        Clean and finalize the CRSP security-file panel (monthly or daily) produced by gen_crsp_sf.
        This step adds trading-volume diagnostics, dividend totals, delisting-return adjustments,
        excess returns (over T-bill / RF), and company-level market equity, using the CIZ delist
        fields (DelReasonType/DelActionType/DelPaymentType/DelStatusType).

    Steps:
        1) Read raw_data_dfs/__crsp_sf_{freq}.parquet; cast key numeric columns; apply NASDAQ volume adjustment.
        2) Compute dollar volume and infer dividend totals from (ret − retx) scaled by lagged price and split factors.
        3) Join CRSP delists (crsp_{freq}sedelist); impute missing delret = −0.30 for “bad delist” buckets defined by CIZ codes;
           set ret=0 when ret is missing but delret exists; compound ret with delret.
        4) Join risk-free proxies (CRSP T-bill and FF RF) and compute excess return ret_exc; compute company ME by summing ME across permnos within permco-date.
        5) If monthly, rescale vol and dolvol for unit alignment.
        6) Drop helper columns, deduplicate by (permno, date), sort, and write crsp_{freq}sf.parquet.

    Output:
        Writes crsp_msf.parquet (freq="m") or crsp_dsf.parquet (freq="d") with cleaned returns and ret_exc.
    """
    assert freq in ("m", "d")

    merge_vars = ["permno", "merge_aux"] if (freq == "m") else ["permno", "date"]

    __crsp_sf = (
        pl.scan_parquet(paths.interim_dir / "raw_data_dfs" / f"__crsp_sf_{freq}.parquet")
        .with_columns(
            [
                col(var).cast(pl.Float64)
                for var in ["prc", "cfacshr", "ret", "retx", "prc_high", "prc_low"]
            ]
            + [col("vol").cast(pl.Int64)]
        )
        .with_columns(
            adj_trd_vol_NASDAQ(
                "date",
                "vol",
                (pl.col("primaryexch") == "Q") & (pl.col("conditionaltype") == "RW"),
            )
        )
        .sort(["permno", "date"])
        .with_columns(
            dolvol=col("prc").abs() * col("vol"),
            div_tot=pl.when(col("cfacshr").shift(1).over("permno") != 0)
            .then(
                (
                    (col("ret") - col("retx"))
                    * col("prc").shift(1)
                    * (col("cfacshr") / col("cfacshr").shift(1))
                ).over("permno")
            )
            .otherwise(fl_none()),
            permno=col("permno").cast(pl.Int64),
            merge_aux=gen_MMYY_column("date"),
        )
    )

    crsp_sedelist_aux_col = (
        [gen_MMYY_column("delistingdt").alias("merge_aux")]
        if (freq == "m")
        else [col("delistingdt").alias("date")]
    )

    crsp_sedelist = pl.scan_parquet(
        paths.interim_dir / "raw_data_dfs" / f"crsp_{freq}sedelist.parquet"
    ).with_columns(crsp_sedelist_aux_col)

    crsp_mcti = add_MMYY_column_drop_original(
        pl.scan_parquet(paths.interim_dir / "raw_data_dfs" / "crsp_mcti_t30ret.parquet"), "caldt"
    )
    ff_factors_monthly = add_MMYY_column_drop_original(
        pl.scan_parquet(paths.interim_dir / "raw_data_dfs" / "ff_factors_monthly.parquet"), "date"
    )

    c1 = col("delret").is_null()

    # CIZ replacement for legacy "dlstcd == 500"
    c2 = (
        (col("delreasontype") == "UNAV")
        & (col("delactiontype") == "GDR")
        & (col("delpaymenttype") == "PRCF")
        & (col("delstatustype") == "VCL")
    )

    # CIZ replacement for legacy "dlstcd between 520 and 584"
    c3 = (
        (col("delactiontype") == "GDR")
        & (col("delpaymenttype") == "PRCF")
        & (col("delstatustype") == "VCL")
        & col("delreasontype").is_in(
            [
                "MVOT",  # Move to OTC
                "MTMK",  # Market Makers
                "SHLD",  # Shareholders
                "LP",  # Low Price
                "INSC",  # Insufficient Capital
                "INSF",  # Insufficient Float
                "CORQ",  # Company Request
                "DERE",  # Deregistration
                "BKPY",  # Bankruptcy
                "OFFRE",  # Offer Rescinded
                "DELQ",  # Delinquent
                "FARG",  # Failure to Register
                "EQRQ",  # Equity Requirements
                "DEEX",  # Denied Exception
                "FING",  # Financial Guidelines
            ]
        )
    )

    c4 = c1 & (c2 | c3)

    c5 = col("ret").is_null()
    c6 = col("delret").is_not_null()
    c7 = c5 & c6

    me_company_exp = (
        pl.when(pl.count("me").over(["permco", "date"]) != 0)
        .then(pl.coalesce(["me", 0.0]).sum().over(["permco", "date"]))
        .otherwise(fl_none())
    )

    scale = 1 if (freq == "m") else 21
    ret_exc_exp = col("ret") - pl.coalesce(["t30ret", "rf"]) / scale

    __crsp_sf = (
        __crsp_sf.join(crsp_sedelist, how="left", on=merge_vars)
        # impute missing delret to -0.30 for the “bad delist” buckets
        .with_columns(delret=pl.when(c4).then(pl.lit(-0.3)).otherwise(col("delret")))
        # if ret missing but delret exists, set ret=0 so compounding works
        .with_columns(ret=pl.when(c7).then(pl.lit(0.0)).otherwise(col("ret")))
        # compound ret with delret
        .with_columns(ret=((col("ret") + 1) * (pl.coalesce(["delret", 0.0]) + 1) - 1))
        # rf joins
        .join(crsp_mcti, how="left", on="merge_aux")
        .join(ff_factors_monthly, how="left", on="merge_aux")
        .with_columns(ret_exc=ret_exc_exp, me_company=me_company_exp)
    )

    if freq == "m":
        __crsp_sf = __crsp_sf.with_columns(
            [(col(var) * 100).alias(var) for var in ["vol", "dolvol"]]
        )

    __crsp_sf = (
        __crsp_sf.drop(
            [
                "rf",
                "t30ret",
                "merge_aux",
                "delret",
                "delistingdt",
                "delreasontype",
                "delactiontype",
                "delpaymenttype",
                "delstatustype",
            ]
        )
        .unique(["permno", "date"])
        .sort(["permno", "date"])
    )

    __crsp_sf.collect().write_parquet(paths.interim_dir / f"crsp_{freq}sf.parquet")


@measure_time
def combine_crsp_comp_sf(paths: DataPaths) -> None:
    """
    Description:
        Create unified monthly and daily security datasets by combining CRSP and Compustat,
        determining the main observation per id/eom, and writing outputs.

    Steps:
        1) Connect to DuckDB (persistent file for out-of-core processing).
        2) Create monthly world table: normalize CRSP/Comp → UNION ALL → LEAD(ret_exc).
        3) Derive obs_main: prefer CRSP when multiple observations per (gvkey, iid, eom).
        4) Write __msf_world.parquet with deterministic dedup (primary_sec preferred
           on tie via ROW_NUMBER).
        5) Write world_dsf.parquet: normalize daily → UNION ALL → join obs_main → dedup.
        6) Clean up DuckDB file.

    Note on the dedup tie-break (ORDER BY source_crsp DESC, primary_sec DESC):
        CRSP rows use raw permno as id (5-digit ints), while Compustat rows construct
        ids as '1'/'2'/'3' || gvkey || iid[0:2] (9+ digit bigints). These id spaces
        never overlap, so any (id, eom) partition is either entirely CRSP or entirely
        Compustat — a "mixed" partition where source_crsp varies cannot exist.
        That makes `source_crsp DESC` effectively a no-op today; it is kept only to
        document the intended preference hierarchy and to remain correct if a future
        change unifies the id schemes.

        The tie-break that actually does the work is `primary_sec DESC`, and it
        only matters within Compustat partitions. When multiple Compustat rows share
        the same (id, eom) and disagree on primary_sec, preferring primary_sec=1
        gives the principled answer — "if any candidate row says this security is
        primary at this date, treat it as primary" — and makes the output
        deterministic across engines and row orderings.

    Output:
        '__msf_world.parquet' and 'world_dsf.parquet' ready for downstream processing.
    """
    (paths.interim_dir / "aux_combine_sf.ddb").unlink(missing_ok=True)
    con = duckdb.connect(str(paths.interim_dir / "aux_combine_sf.ddb"))
    crsp_msf_path = (paths.interim_dir / "crsp_msf.parquet").as_posix()
    comp_msf_path = (paths.interim_dir / "comp_msf.parquet").as_posix()
    crsp_dsf_path = (paths.interim_dir / "crsp_dsf.parquet").as_posix()
    comp_dsf_path = (paths.interim_dir / "comp_dsf.parquet").as_posix()
    msf_world_out = (paths.interim_dir / "__msf_world.parquet").as_posix()
    world_dsf_out = (paths.interim_dir / "world_dsf.parquet").as_posix()
    try:
        # Monthly: normalize CRSP/Comp, UNION ALL, compute ret_exc_lead1m
        con.execute(f"""
            CREATE TABLE sf_world_m AS
            WITH crsp_msf_norm AS (
                SELECT
                    permno AS id, permno, permco, gvkey, iid,
                    'USA' AS excntry,
                    exch_main::INT AS exch_main,
                    common::INT AS common,
                    1 AS primary_sec,
                    bidask::INT AS bidask,
                    primaryexch,
                    conditionaltype,
                    crsp_nyse::INT AS crsp_nyse,
                    NULL::VARCHAR AS comp_tpci,
                    NULL::BIGINT AS comp_exchg,
                    'USD' AS curcd,
                    1.0 AS fx,
                    date,
                    last_day(date) AS eom,
                    cfacshr AS adjfct,
                    shrout AS shares,
                    me, me_company, prc,
                    prc AS prc_local,
                    prc_high, prc_low, dolvol,
                    vol AS tvol,
                    ret,
                    ret AS ret_local,
                    ret_exc,
                    1::BIGINT AS ret_lag_dif,
                    div_tot,
                    NULL::DOUBLE AS div_cash,
                    NULL::DOUBLE AS div_spc,
                    1 AS source_crsp
                FROM read_parquet('{crsp_msf_path}')
            ),
            comp_msf_norm AS (
                SELECT
                    CAST(
                        CASE
                            WHEN iid LIKE '%W%' THEN '3' || gvkey || SUBSTRING(iid, 1, 2)
                            WHEN iid LIKE '%C%' THEN '2' || gvkey || SUBSTRING(iid, 1, 2)
                            ELSE '1' || gvkey || SUBSTRING(iid, 1, 2)
                        END AS BIGINT
                    ) AS id,
                    NULL::BIGINT AS permno,
                    NULL::BIGINT AS permco,
                    gvkey, iid, excntry,
                    exch_main::INT AS exch_main,
                    CASE WHEN tpci = '0' THEN 1 ELSE 0 END AS common,
                    primary_sec::INT AS primary_sec,
                    CASE WHEN prcstd = 4 THEN 1 ELSE 0 END AS bidask,
                    NULL::VARCHAR AS primaryexch,
                    NULL::VARCHAR AS conditionaltype,
                    0 AS crsp_nyse,
                    tpci AS comp_tpci,
                    exchg::BIGINT AS comp_exchg,
                    curcdd AS curcd,
                    fx,
                    datadate AS date,
                    eom,
                    ajexdi AS adjfct,
                    cshoc AS shares,
                    me,
                    me AS me_company,
                    prc, prc_local, prc_high, prc_low, dolvol,
                    cshtrm AS tvol,
                    ret, ret_local, ret_exc,
                    ret_lag_dif::BIGINT AS ret_lag_dif,
                    div_tot, div_cash, div_spc,
                    0 AS source_crsp
                FROM read_parquet('{comp_msf_path}')
            )
            SELECT *,
                CASE
                    WHEN LEAD(ret_lag_dif, 1) OVER (PARTITION BY id ORDER BY eom) != 1
                    THEN NULL
                    ELSE LEAD(ret_exc, 1) OVER (PARTITION BY id ORDER BY eom)
                END AS ret_exc_lead1m
            FROM (
                SELECT * FROM crsp_msf_norm
                UNION ALL
                SELECT * FROM comp_msf_norm
            ) unioned
        """)

        # Derive obs_main: prefer CRSP when multiple obs per (gvkey, iid, eom).
        # Deduplicate to (id, eom) before the join to avoid transient row inflation.
        con.execute("""
            CREATE TABLE obs_main AS
            SELECT id, eom, MAX(obs_main) AS obs_main
            FROM (
                SELECT id, eom,
                    CAST(CASE
                        WHEN cnt IN (0, 1) THEN 1
                        WHEN cnt > 1 AND source_crsp = 1 THEN 1
                        ELSE 0
                    END AS INT) AS obs_main
                FROM (
                    SELECT id, source_crsp, eom,
                        COUNT(gvkey) OVER (PARTITION BY gvkey, iid, eom) AS cnt
                    FROM sf_world_m
                ) sub
            ) dedup
            GROUP BY id, eom
        """)

        # Write monthly output with deterministic dedup (prefer CRSP)
        con.execute(f"""
            COPY (
                SELECT
                    id, permno, permco, gvkey, iid, excntry, exch_main, common,
                    primary_sec, bidask, primaryexch, conditionaltype, crsp_nyse, comp_tpci, comp_exchg,
                    curcd, fx, date, eom, adjfct, shares, me, me_company, prc, prc_local,
                    prc_high, prc_low, dolvol, tvol, ret, ret_local, ret_exc, ret_lag_dif,
                    div_tot, div_cash, div_spc, source_crsp, ret_exc_lead1m, obs_main
                FROM (
                    -- source_crsp DESC is a no-op today (CRSP/Comp ids don't collide);
                    -- primary_sec DESC is the real tie-break: when Compustat rows
                    -- disagree on primary_sec for the same (id, eom), prefer the
                    -- primary one. See combine_crsp_comp_sf docstring for details.
                    SELECT a.*, b.obs_main,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.id, a.eom
                            ORDER BY a.source_crsp DESC, a.primary_sec DESC
                        ) AS _rn
                    FROM sf_world_m a
                    LEFT JOIN obs_main b ON a.id = b.id AND a.eom = b.eom
                ) ranked
                WHERE _rn = 1
                ORDER BY id, eom
            ) TO '{msf_world_out}' (FORMAT PARQUET)
        """)

        # Free monthly table memory; keep obs_main for daily step
        con.execute("DROP TABLE sf_world_m")

        # Daily: normalize CRSP/Comp, UNION ALL, join obs_main, dedup, write
        con.execute(f"""
            COPY (
                WITH crsp_dsf_norm AS (
                    SELECT
                        permno AS id,
                        'USA' AS excntry,
                        exch_main::INT AS exch_main,
                        common::INT AS common,
                        1 AS primary_sec,
                        bidask::INT AS bidask,
                        'USD' AS curcd,
                        1.0 AS fx,
                        date,
                        last_day(date) AS eom,
                        cfacshr AS adjfct,
                        shrout AS shares,
                        me, dolvol,
                        vol AS tvol,
                        prc, prc_high, prc_low,
                        ret AS ret_local,
                        ret, ret_exc,
                        1::BIGINT AS ret_lag_dif,
                        1 AS source_crsp
                    FROM read_parquet('{crsp_dsf_path}')
                ),
                comp_dsf_norm AS (
                    SELECT
                        CAST(
                            CASE
                                WHEN iid LIKE '%W%' THEN '3' || gvkey || SUBSTRING(iid, 1, 2)
                                WHEN iid LIKE '%C%' THEN '2' || gvkey || SUBSTRING(iid, 1, 2)
                                ELSE '1' || gvkey || SUBSTRING(iid, 1, 2)
                            END AS BIGINT
                        ) AS id,
                        excntry,
                        exch_main::INT AS exch_main,
                        CASE WHEN tpci = '0' THEN 1 ELSE 0 END AS common,
                        primary_sec::INT AS primary_sec,
                        CASE WHEN prcstd = 4 THEN 1 ELSE 0 END AS bidask,
                        curcdd AS curcd,
                        fx,
                        datadate AS date,
                        last_day(datadate) AS eom,
                        ajexdi AS adjfct,
                        cshoc AS shares,
                        me, dolvol,
                        cshtrd AS tvol,
                        prc, prc_high, prc_low,
                        ret_local, ret, ret_exc,
                        ret_lag_dif::BIGINT AS ret_lag_dif,
                        0 AS source_crsp
                    FROM read_parquet('{comp_dsf_path}')
                ),
                sf_world_d AS (
                    SELECT * FROM crsp_dsf_norm
                    UNION ALL
                    SELECT * FROM comp_dsf_norm
                ),
                ranked AS (
                    -- See dedup tie-break note in monthly block above.
                    SELECT a.*, b.obs_main,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.id, a.date
                            ORDER BY a.source_crsp DESC, a.primary_sec DESC
                        ) AS _rn
                    FROM sf_world_d a
                    LEFT JOIN obs_main b ON a.id = b.id AND a.eom = b.eom
                )
                SELECT
                    id, excntry, exch_main, common, primary_sec, bidask, curcd, fx,
                    date, eom, adjfct, shares, me, dolvol, tvol, prc, prc_high, prc_low,
                    ret_local, ret, ret_exc, ret_lag_dif, source_crsp, obs_main
                FROM ranked
                WHERE _rn = 1
                ORDER BY id, date
            ) TO '{world_dsf_out}' (FORMAT PARQUET)
        """)
    finally:
        con.close()
        (paths.interim_dir / "aux_combine_sf.ddb").unlink(missing_ok=True)


@measure_time
def crsp_industry(paths: DataPaths):
    """
    Description:
        Generate a daily panel of CRSP SIC/NAICS codes per permno based on name-date spans.

    Steps:
        1) Read permno0; nullify sic==0; build date ranges from secinfostartdt to secinfoenddt.
        2) Explode to daily rows; keep distinct (permno,date); sort.
        3) Write to crsp_ind.parquet.

    Output:
        Parquet crsp_ind.parquet with {permno,permco,date,sic,naics}.
    """
    permno0 = pl.scan_parquet(paths.interim_dir / "raw_data_dfs" / "permno0.parquet")
    permno0 = (
        permno0.with_columns(
            sic=pl.when(col("sic") == 0).then(pl.lit(None).cast(pl.Int64)).otherwise(col("sic"))
        )
        .with_columns(date=pl.date_ranges("secinfostartdt", "secinfoenddt"))
        .explode("date")
        .select(["permno", "permco", "date", "sic", "naics"])
        .unique(["permno", "date"])
        .sort(["permno", "date"])
    )
    permno0.collect().write_parquet(paths.interim_dir / "crsp_ind.parquet")


def comp_hgics(paths: DataPaths, lib):
    """
    Description:
        Expand Compustat GICS history (national/global) to a daily panel with forward-filled
        terminal dates and write separate outputs.

    Steps:
        1) Load raw file (national/global); replace null gics with -999 sentinel.
        2) Compute row counts and terminal rows; set open-ended indthru to (max(indfrom) or END_DATE).
        3) Create date ranges [indfrom, indthru]; explode; unique per (gvkey,date).
        4) Write to na_hgics.parquet or g_hgics.parquet.

    Output:
        Parquet with {gvkey,date,gics} expanded to daily frequency.
    """
    raw_data_dfs = paths.interim_dir / "raw_data_dfs"
    file_paths = {
        "raw data": {
            "national": raw_data_dfs / "comp_hgics_na.parquet",
            "global": raw_data_dfs / "comp_hgics_gl.parquet",
        },
        "output": {
            "national": paths.interim_dir / "na_hgics.parquet",
            "global": paths.interim_dir / "g_hgics.parquet",
        },
    }
    data = pl.scan_parquet(file_paths["raw data"][lib])  # .sort(['gvkey', 'indfrom'])
    if data.limit(1).collect().is_empty():
        warnings.warn(
            f"comp_hgics: {lib} GICS input is empty "
            f"({file_paths['raw data'][lib]}); "
            "downstream comp_ind.parquet will have null GICS for all firms "
            "unless the other side (national/global) provides coverage.",
            stacklevel=2,
        )
    data = data.with_columns(
        gics=pl.when(col("gics").is_null()).then(-999).otherwise(col("gics")),
        n=pl.len().over("gvkey"),
        n_aux=pl.cum_count("gvkey").over("gvkey"),
    )
    indthru_date = pl.max_horizontal(pl.col("indfrom").max(), pl.lit(END_DATE))
    c1 = col("n") == col("n_aux")
    c2 = col("indthru").is_null()
    data = (
        data.with_columns(indthru=pl.when(c1 & c2).then(indthru_date).otherwise(col("indthru")))
        .select(["gvkey", pl.date_ranges("indfrom", "indthru").alias("date"), "gics"])
        .explode("date")
        .unique(subset=["gvkey", "date"])
        .sort(["gvkey", "date"])
    )
    data.sink_parquet(file_paths["output"][lib])


def hgics_join(paths: DataPaths):
    """
    Description:
        Merge national and global GICS daily panels, preferring local (national) where available.

    Steps:
        1) Ensure comp_hgics('global') and comp_hgics('national') are created.
        2) Full join on (gvkey,date) with coalesce of gics fields.
        3) Deduplicate and sort; write comp_hgics.parquet.

    Output:
        Parquet comp_hgics.parquet with consolidated GICS per (gvkey,date).
    """
    comp_hgics(paths, "global")
    comp_hgics(paths, "national")
    global_data = pl.scan_parquet(paths.interim_dir / "g_hgics.parquet")
    local_data = pl.scan_parquet(paths.interim_dir / "na_hgics.parquet")
    gjoin = local_data.join(global_data, on=["gvkey", "date"], how="full", coalesce=True)
    gjoin = (
        gjoin.select(["gvkey", "date", pl.coalesce(["gics", "gics_right"]).alias("gics")])
        .unique(["gvkey", "date"])
        .sort(["gvkey", "date"])
    )
    gjoin.sink_parquet(paths.interim_dir / "comp_hgics.parquet")


def comp_sic_naics(paths: DataPaths):
    """
    Description:
        Combine and reconcile Compustat SIC/NAICS from US and Global datasets into a
        continuous daily series per gvkey, resolving duplicates and filling daily gaps.

    Steps:
        1) Load NA and GL tables; drop a known problematic row; outer-join on (gvkey,datadate).
        2) Coalesce ids/dates/codes; order to prefer non-null SIC; select distinct per (gvkey,date).
        3) Convert to daily spans by joining to next datadate; expand date ranges [datadate,end_date).
        4) Handle single-date rows; project to {gvkey,date,sic,naics}; deduplicate, sort.
        5) Write comp_other.parquet.

    Output:
        Parquet comp_other.parquet with daily SIC/NAICS per gvkey.
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    con.create_table(
        "sic_naics_na",
        con.read_parquet(paths.interim_dir / "raw_data_dfs" / "sic_naics_na.parquet"),
    )
    con.create_table(
        "sic_naics_gl",
        con.read_parquet(paths.interim_dir / "raw_data_dfs" / "sic_naics_gl.parquet"),
    )
    con.raw_sql("""
                CREATE TABLE comp2 AS
                SELECT *
                FROM sic_naics_na
                WHERE NOT (
                    gvkey   = '175650'
                    AND datadate = DATE '2005-12-31'
                    AND naics IS NULL
                );

                CREATE TABLE comp3 AS
                SELECT *
                FROM sic_naics_gl;

                CREATE TABLE comp4 AS
                SELECT
                    a.gvkey    AS gvkeya,
                    a.datadate AS datea,
                    a.sic      AS sica,
                    a.naics    AS naicsa,
                    b.gvkey    AS gvkeyb,
                    b.datadate AS dateb,
                    b.sic      AS sicb,
                    b.naics    AS naicsb
                FROM comp2 AS a
                FULL OUTER JOIN comp3 AS b
                ON a.gvkey    = b.gvkey
                AND a.datadate = b.datadate;

                CREATE TABLE comp5 AS
                WITH t AS (
                SELECT
                    LPAD(COALESCE(gvkeya, gvkeyb), 6, '0') AS gvkey,
                    COALESCE(datea, dateb)            AS date,
                    COALESCE(sica, sicb)              AS sic,
                    COALESCE(naicsa, naicsb)          AS naics
                FROM comp4
                ORDER BY
                    gvkey,
                    date,
                    sic DESC
                )
                SELECT DISTINCT ON (gvkey, date)
                    t.gvkey,
                    t.date as datadate,
                    t.sic,
                    t.naics
                FROM t
                ORDER BY
                t.gvkey,
                t.date,
                t.sic
                ;
    """)
    comp = (
        con.table("comp5")
        .to_polars()
        .lazy()
        .sort(["gvkey", "datadate"])
        .with_columns(end_date=col("datadate").shift(-1).over("gvkey"))
        .with_columns(
            end_date=pl.when(col("end_date").is_null())
            .then(col("datadate"))
            .otherwise(col("end_date"))
        )
        .with_columns(date=pl.date_ranges("datadate", "end_date", closed="left"))
        .explode("date")
        .with_columns(
            date=pl.when(col("datadate") == col("end_date"))
            .then(col("datadate"))
            .otherwise(col("date"))
        )
        .select(["gvkey", "date", "sic", "naics"])
        .unique(["gvkey", "date"])
        .sort(["gvkey", "date"])
    )
    comp.sink_parquet(paths.interim_dir / "comp_other.parquet")
    con.disconnect()


@measure_time
def comp_industry(paths: DataPaths):
    """
    Description:
        Merge daily GICS and SIC/NAICS into a single daily Compustat industry file with a
        continuous daily date axis; days between observations carry null industry codes.

    Steps:
        1) Run comp_sic_naics() and hgics_join(); scan both panels lazily.
        2) Full-outer-join on (gvkey,date); compute aux_date = next date − 1 day to detect gaps.
        3) Emit interior gap days [date+1, aux_date] with null codes (only the gap-start day
           carries values, matching the historical SQL gap-fill behaviour).
        4) Concatenate joined rows with gap rows; sort; sink comp_ind.parquet.

    Output:
        Parquet comp_ind.parquet with {gvkey,date,gics,sic,naics} daily.
    """
    comp_sic_naics(paths)
    hgics_join(paths)
    comp_gics = pl.scan_parquet(paths.interim_dir / "comp_hgics.parquet")
    comp_other = pl.scan_parquet(paths.interim_dir / "comp_other.parquet")
    joined = (
        comp_gics.join(comp_other, on=["gvkey", "date"], how="full", coalesce=True)
        # Null dates (from GICS records with null indfrom) were silently dropped by the
        # historical SQL: both `WHERE date <> aux_date` and `WHERE date = aux_date`
        # evaluate to NULL for them, excluding the rows from every output branch.
        .filter(pl.col("date").is_not_null())
        .sort(["gvkey", "date"])
        .with_columns(
            aux_date=pl.coalesce(
                pl.col("date").shift(-1).over("gvkey") - pl.duration(days=1),
                pl.col("date"),
            )
        )
    )
    schema = joined.collect_schema()
    # Interior days of each gap get null industry codes: the historical SQL left-joined
    # gap_dates on the exact date, so only the gap-start day (already present in the
    # joined panel) carried values.
    gap_rows = (
        joined.filter(pl.col("date") != pl.col("aux_date"))
        .select(
            "gvkey",
            pl.date_ranges(pl.col("date") + pl.duration(days=1), "aux_date").alias("date"),
            *[pl.lit(None, dtype=schema[c]).alias(c) for c in ["gics", "sic", "naics"]],
        )
        .explode("date")
    )
    out = pl.concat([joined.select(["gvkey", "date", "gics", "sic", "naics"]), gap_rows]).sort(
        ["gvkey", "date"]
    )
    out.sink_parquet(paths.interim_dir / "comp_ind.parquet")


def _parse_siccodes_file(filename: str, label: str) -> pl.DataFrame:
    """Parse a single Fama-French Siccodes text file into a SIC→category DataFrame.

    Each file contains numbered industry categories with SIC code ranges.
    Returns a DataFrame with columns ``sic`` (Int64) and *label* (Int32).
    """
    header_re = re.compile(r"^\s*(\d+)\s+\S+.*$")
    range_re = re.compile(r"^\s*(\d{4})-(\d{4})(?:\b.*)?$")

    result: dict[int, list[int]] = {}
    current_category: int | None = None

    with open(filename, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()

            header_match = header_re.match(line)
            if header_match:
                current_category = int(header_match.group(1))
                result[current_category] = []
                continue

            range_match = range_re.match(line)
            if range_match and current_category is not None:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                result[current_category].extend(range(start, end + 1))

    rows = [{label: ff, "sic": sic_list} for ff, sic_list in result.items()]

    return (
        pl.DataFrame(
            rows,
            schema={"sic": pl.List(pl.Int64), label: pl.Int32},
            orient="row",
        )
        .sort(label)
        .explode("sic")
        .drop_nulls()
    )


@measure_time
def ff_ind_class(paths: DataPaths, data_path: str) -> None:
    """
    Description:
        Assign Fama-French 49 industry classification based on SIC codes.

    Steps:
        1) Parse data/raw/Siccodes49.txt to build a SIC→FF49 mapping.
           Only SIC codes explicitly listed receive non-null values.
        2) Left-join the input data on 'sic' to attach the ff49 column.
        3) Write __msf_world3.parquet.

    Output:
        Parquet __msf_world3.parquet with added ff49 column (Int32).
    """
    # The parser can handle other Fama-French classifications
    # (e.g., Siccodes5.txt through Siccodes48.txt).
    from .paths import get_siccodes_path

    mapping = _parse_siccodes_file(str(get_siccodes_path()), label="ff49").lazy()
    data = pl.scan_parquet(data_path)
    data.join(mapping, on="sic", how="left").collect().write_parquet(
        paths.interim_dir / "__msf_world3.parquet"
    )


@measure_time
def nyse_size_cutoffs(paths: DataPaths, data_path):
    """
    Description:
        Compute NYSE market equity cutoffs (1%,20%,50%,80%) by month.

    Steps:
        1) Load parquet lazily, filter to NYSE common stocks.
        2) Group by eom, count obs.
        3) Apply QUANTILE_DISC for cutoffs.
        4) Collect and save.

    Output:
        'nyse_cutoffs.parquet' with [eom, n, nyse_p1, nyse_p20, nyse_p50, nyse_p80].
    """
    nyse_sf = pl.scan_parquet(data_path).sql("""
            SELECT
                eom,
                COUNT(*)                    AS n,
                QUANTILE_DISC(me, 0.01)     AS nyse_p1,
                QUANTILE_DISC(me, 0.20)     AS nyse_p20,
                QUANTILE_DISC(me, 0.50)     AS nyse_p50,
                QUANTILE_DISC(me, 0.80)     AS nyse_p80
            FROM self
            WHERE  crsp_nyse   = 1
                AND obs_main   = 1
                AND exch_main  = 1
                AND primary_sec= 1
                AND common     = 1
                AND me IS NOT NULL
            GROUP BY eom
            ORDER BY eom
            """)
    nyse_sf.sink_parquet(paths.interim_dir / "nyse_cutoffs.parquet")


@measure_time
def classify_stocks_size_groups(paths: DataPaths):
    """
    Description:
        Join world MSF with NYSE size cutoffs and classify stocks into size buckets.

    Steps:
        1) Read 'nyse_cutoffs.parquet' and '__msf_world3.parquet' lazily.
        2) Left-join on eom; compute size_grp via ME vs NYSE p1/p20/p50/p80 (fallback to 'mega' if cutoffs missing).
        3) Drop cutoff columns; collect and save.

    Output:
        Writes 'world_msf.parquet' with size_grp per row.
    """
    nyse_cutoffs = pl.scan_parquet(paths.interim_dir / "nyse_cutoffs.parquet")
    __msf_world = pl.scan_parquet(paths.interim_dir / "__msf_world3.parquet")
    world_msf = (
        __msf_world.join(nyse_cutoffs, how="left", on="eom")
        .with_columns(
            size_grp=pl.when(col("me").is_null())
            .then(None)
            .when(col("nyse_p80").is_null())
            .then(pl.lit("mega"))  # This is just to match SAS excactly
            .when(col("me") >= col("nyse_p80"))
            .then(pl.lit("mega"))
            .when(col("me") >= col("nyse_p50"))
            .then(pl.lit("large"))
            .when(col("me") >= col("nyse_p20"))
            .then(pl.lit("small"))
            .when(col("me") >= col("nyse_p1"))
            .then(pl.lit("micro"))
            .otherwise(pl.lit("nano"))
        )
        .drop([i for i in nyse_cutoffs.collect_schema().names() if i not in ["eom"]])
    )
    world_msf.collect().write_parquet(paths.interim_dir / "world_msf.parquet")


@measure_time
def return_cutoffs(paths: DataPaths, freq, crsp_only):
    """
    Description:
        Compute return percentile cutoffs by period (monthly or daily), optionally CRSP-only.

    Steps:
        1) Select group vars and output path from freq; scan 'world_{freq}sf.parquet'.
        2) Optional CRSP filter; require common/main/primary, exclude ZWE, non-null ret_exc.
        3) Group by eom and aggregate counts + percentiles for ret, ret_local, ret_exc.
        4) Sort, collect, save.

    Output:
        Writes 'return_cutoffs.parquet' (monthly) or 'return_cutoffs_daily.parquet' (daily).
    """
    group_vars = "eom"
    res_path = (
        paths.interim_dir / "return_cutoffs.parquet"
        if freq == "m"
        else paths.interim_dir / "return_cutoffs_daily.parquet"
    )
    data = pl.scan_parquet(paths.interim_dir / f"world_{freq}sf.parquet").filter(
        (col("common") == 1)
        & (col("obs_main") == 1)
        & (col("exch_main") == 1)
        & (col("primary_sec") == 1)
        & (col("excntry") != "ZWE")
        & (col("ret_exc").is_not_null())
        & ((col("source_crsp") == 1) if crsp_only == 1 else pl.lit(True))
    )
    data = data.sql(f"""
            SELECT
                {group_vars},
                COUNT(ret)                                AS n,

                -- ret percentiles
                QUANTILE_DISC(ret,        0.001)          AS ret_0_1,
                QUANTILE_DISC(ret,        0.01)           AS ret_1,
                QUANTILE_DISC(ret,        0.99)           AS ret_99,
                QUANTILE_DISC(ret,        0.999)          AS ret_99_9,

                -- ret_local percentiles
                QUANTILE_DISC(ret_local,  0.001)          AS ret_local_0_1,
                QUANTILE_DISC(ret_local,  0.01)           AS ret_local_1,
                QUANTILE_DISC(ret_local,  0.99)           AS ret_local_99,
                QUANTILE_DISC(ret_local,  0.999)          AS ret_local_99_9,

                -- ret_exc percentiles
                QUANTILE_DISC(ret_exc,    0.001)          AS ret_exc_0_1,
                QUANTILE_DISC(ret_exc,    0.01)           AS ret_exc_1,
                QUANTILE_DISC(ret_exc,    0.99)           AS ret_exc_99,
                QUANTILE_DISC(ret_exc,    0.999)          AS ret_exc_99_9

            FROM self
            GROUP BY {group_vars}
            ORDER BY {group_vars}
            """)
    if freq == "d":
        data = data.with_columns(
            year=pl.col("eom").dt.year(),
            month=pl.col("eom").dt.month(),
        )
    data.sink_parquet(res_path)


@measure_time
def add_ret_exc_wins(
    paths: DataPaths, freq: str, lower: float = 0.001, upper: float = 0.999
) -> None:
    """
    Description:
        Add a winsorized excess return column (ret_exc_wins) to the world security file.
        Compustat returns (source_crsp == 0) are clipped to the [lower, upper] percentiles
        of ret_exc using precomputed cutoffs from return_cutoffs{,_daily}.parquet.
        CRSP returns are left unchanged.

    Steps:
        1) Validate lower/upper and map them to precomputed cutoff column names.
        2) Read world_{freq}sf.parquet; drop ret_exc_wins if already present (idempotency).
        3) Left-join precomputed cutoffs from return_cutoffs{,_daily}.parquet on eom (monthly)
           or year/month (daily).
        4) Build ret_exc_wins via pl.when: clip Compustat rows to [lower_col, upper_col],
           leave CRSP rows and nulls unchanged.
        5) Collect and overwrite the file.

    Output:
        Overwrites 'world_{freq}sf.parquet' with ret_exc_wins added.
    """
    if not (0 <= lower < upper <= 1):
        raise ValueError(
            f"Percentile bounds must satisfy 0 <= lower < upper <= 1, got {lower=}, {upper=}"
        )

    percentile_to_column = {
        0.001: "ret_exc_0_1",
        0.01: "ret_exc_1",
        0.99: "ret_exc_99",
        0.999: "ret_exc_99_9",
    }
    if lower not in percentile_to_column or upper not in percentile_to_column:
        raise ValueError(
            f"lower/upper must be one of {sorted(percentile_to_column)} "
            f"(matching precomputed cutoffs in return_cutoffs.parquet); "
            f"got {lower=}, {upper=}"
        )
    lower_col = percentile_to_column[lower]
    upper_col = percentile_to_column[upper]

    data_path = paths.interim_dir / f"world_{freq}sf.parquet"
    cutoffs_path = (
        paths.interim_dir / "return_cutoffs.parquet"
        if freq == "m"
        else paths.interim_dir / "return_cutoffs_daily.parquet"
    )
    group_vars = ["eom"] if freq == "m" else ["year", "month"]

    data = pl.scan_parquet(data_path)
    if "ret_exc_wins" in data.collect_schema().names():
        data = data.drop("ret_exc_wins")

    if freq == "d":
        data = data.with_columns(
            year=pl.col("date").dt.year(),
            month=pl.col("date").dt.month(),
        )

    cutoffs = pl.scan_parquet(cutoffs_path).select([*group_vars, lower_col, upper_col])

    drop_cols = [lower_col, upper_col]
    if freq == "d":
        drop_cols += ["year", "month"]

    result = (
        data.join(cutoffs, on=group_vars, how="left")
        .with_columns(
            ret_exc_wins=pl.when((pl.col("source_crsp") == 0) & pl.col("ret_exc").is_not_null())
            .then(pl.col("ret_exc").clip(pl.col(lower_col), pl.col(upper_col)))
            .otherwise(pl.col("ret_exc"))
        )
        .drop(drop_cols)
    )
    result.collect(streaming=(freq == "d")).write_parquet(data_path)


def load_mkt_returns_params(freq):
    """
    Description:
        Provide parameter defaults (column names, lags, groups, columns) by frequency.

    Steps:
        1) Set dt column, max lag, path suffix, group vars based on freq ('d' vs other).
        2) Return common-stocks column list.

    Output:
        Tuple: (dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols).
    """
    dt_col = "date" if freq == "d" else "eom"
    max_date_lag = 14 if freq == "d" else 1
    path_aux = "_daily" if freq == "d" else ""
    group_vars = ["eom"]
    comm_stocks_cols = [
        "source_crsp",
        "id",
        "date",
        "eom",
        "excntry",
        "obs_main",
        "exch_main",
        "primary_sec",
        "common",
        "ret_lag_dif",
        "me",
        "dolvol",
        "ret",
        "ret_local",
        "ret_exc",
    ]
    return dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols


def add_cutoffs_and_winsorize(df, wins_data_path, group_vars, dt_col):
    """
    Description:
        Attach precomputed return cutoffs and winsorize ret, ret_local, ret_exc.

    Steps:
        1) Read winsor cutoff parquet; select group vars + needed thresholds.
        2) Left-join cutoffs on group vars (eom).
        3) Winsorize high (99.9) then low (0.1) for each return series.

    Output:
        Polars LazyFrame/DataFrame with winsorized returns and cutoff columns joined.
    """
    wins_data = pl.scan_parquet(wins_data_path)

    on_clause = " AND ".join([f"a.{k} = b.{k}" for k in group_vars])

    ctx = pl.SQLContext()
    ctx.register("df", df)
    ctx.register("wins_data", wins_data)

    result = ctx.execute(f"""
    SELECT
        a.*
        REPLACE(
            CASE WHEN a.source_crsp = 0 AND a.ret IS NOT NULL
                 THEN GREATEST(b.ret_0_1, LEAST(a.ret, b.ret_99_9))
                 ELSE a.ret
            END AS ret,

            CASE WHEN a.source_crsp = 0 AND a.ret_local IS NOT NULL
                 THEN GREATEST(b.ret_local_0_1, LEAST(a.ret_local, b.ret_local_99_9))
                 ELSE a.ret_local
            END AS ret_local,

            CASE WHEN a.source_crsp = 0 AND a.ret_exc IS NOT NULL
                 THEN GREATEST(b.ret_exc_0_1, LEAST(a.ret_exc, b.ret_exc_99_9))
                 ELSE a.ret_exc
            END AS ret_exc
        )
    FROM df AS a
    LEFT JOIN wins_data AS b
        ON {on_clause}
    """)
    return result


def sas_sum_agg(name):
    """
    Description:
        SAS-like SUM aggregate that returns NULL when no rows; else sum(col).

    Steps:
        1) Check count(col) > 0.
        2) If true, return sum(col); else NULL.

    Output:
        Polars expression yielding SUM(col) or NULL.
    """
    return pl.when(pl.col(name).count() > 0).then(pl.sum(name)).otherwise(None)


def apply_stock_filter_and_compute_indexes(df, dt_col, max_date_lag):
    """
    Description:
        Filter to eligible common stocks and compute value/eq-weighted market returns by country/date.

    Steps:
        1) Filter on main/exchange/primary/common, date-lag ≤ max_date_lag, me_lag1 & ret_local non-null.
        2) Create aux = ret*me_lag1 variants.
        3) Group by [excntry, dt_col] and aggregate counts, sums, VW/EW returns.

    Output:
        LazyFrame/DataFrame aggregated at country-date with VW/EW returns and counts.
    """
    c1 = (
        (col("obs_main") == 1)
        & (col("exch_main") == 1)
        & (col("primary_sec") == 1)
        & (col("common") == 1)
        & (col("ret_lag_dif") <= max_date_lag)
        & (col("me_lag1").is_not_null())
        & (col("ret_local").is_not_null())
    )
    df = (
        df.filter(c1)
        .with_columns(
            aux1=col("ret_local") * col("me_lag1"),
            aux2=col("ret") * col("me_lag1"),
            aux3=col("ret_exc") * col("me_lag1"),
            aux4=col("ret_exc") * col("me_cap_lag1"),
        )
        .group_by(["excntry", dt_col])
        .agg(
            stocks=pl.len(),
            me_lag1=sas_sum_agg("me_lag1"),
            dolvol_lag1=sas_sum_agg("dolvol_lag1"),
            mkt_vw_lcl=sas_sum_agg("aux1") / sas_sum_agg("me_lag1"),
            mkt_ew_lcl=pl.mean("ret_local"),
            mkt_vw=sas_sum_agg("aux2") / sas_sum_agg("me_lag1"),
            mkt_ew=pl.mean("ret"),
            mkt_vw_exc=sas_sum_agg("aux3") / sas_sum_agg("me_lag1"),
            mkt_ew_exc=pl.mean("ret_exc"),
            mkt_vw_cap_exc=sas_sum_agg("aux4") / sas_sum_agg("me_cap_lag1"),
        )
    )
    return df


def drop_non_trading_days(df, n_col, dt_col, over_vars, thresh_fraction):
    """
    Description:
        Remove thin-trading days by country-month (or given window) based on stock coverage.

    Steps:
        1) Derive eom from dt_col; compute max_stocks over over_vars.
        2) Keep rows where n_col / max_stocks ≥ thresh_fraction.
        3) Drop helper columns.

    Output:
        Frame filtered to sufficiently traded dates.
    """
    added_eom = "eom" not in df.collect_schema().names()
    if added_eom:
        df = df.with_columns(eom=pl.col(dt_col).dt.month_end())
    df = (
        df.with_columns(max_stocks=pl.max(n_col).over(over_vars))
        .filter((col(n_col) / col("max_stocks")) >= thresh_fraction)
        .drop(["max_stocks"] + (["eom"] if added_eom else []))
    )
    return df


@measure_time
def market_returns(paths: DataPaths, data_path, freq, wins_comp, wins_data_path, nyse_cutoffs_path):
    """
    Description:
        Build country-level market returns (daily or monthly), optional winsorization, and save to disk.

    Steps:
        1) Load params from freq; scan data; keep common-stock fields; sort by [id, dt].
        2) Add lags me_lag1, dolvol_lag1 per id.
        3) If wins_comp, join cutoffs and winsorize returns.
        4) Join NYSE P80 cutoffs and compute me_cap_lag1.
        5) Apply stock filters & compute VW/EW country returns (including capped VW).
        6) If daily, drop low-coverage trading days.
        7) Sort and write to ``paths.interim_dir / f"market_returns{path_aux}.parquet"``
           (``path_aux`` is ``""`` for monthly and ``"_daily"`` for daily).

    Output:
        Parquet file of country × date market returns.
    """
    dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols = load_mkt_returns_params(freq)
    __common_stocks = (
        pl.scan_parquet(data_path)
        .select(comm_stocks_cols)
        .unique()
        .sort(["id", dt_col])
        .with_columns(
            me_lag1=col("me").shift(1).over("id"),
            dolvol_lag1=col("dolvol").shift(1).over("id"),
        )
    )
    if wins_comp == 1:
        __common_stocks = add_cutoffs_and_winsorize(
            __common_stocks, wins_data_path, group_vars, dt_col
        )
    nyse = pl.scan_parquet(nyse_cutoffs_path).select(["eom", "nyse_p80"])
    __common_stocks = __common_stocks.join(nyse, on="eom", how="left").with_columns(
        me_cap_lag1=pl.min_horizontal(col("me_lag1"), col("nyse_p80"))
    )
    __common_stocks = apply_stock_filter_and_compute_indexes(__common_stocks, dt_col, max_date_lag)
    if freq == "d":
        __common_stocks = drop_non_trading_days(
            __common_stocks, "stocks", dt_col, ["excntry", "eom"], 0.25
        )
    __common_stocks.sort(["excntry", dt_col]).collect().write_parquet(
        paths.interim_dir / f"market_returns{path_aux}.parquet"
    )


def quarterize(df, var_list):
    """
    Description:
        Convert quarterly Compustat levels to quarter-over-quarter flows with guardrails.

    Steps:
        1) Per [gvkey,fyr,fyearq], add running count and diffs var_q = Δ(var).
        2) Build deletion mask for first obs or quarter breaks; null-out invalid diffs.
        3) Return unique, sorted quarterly panel with *_q flows.

    Output:
        Quarterly panel with flow variables var_q for each input in var_list.
    """
    list_aux1 = [col("gvkey").cum_count().over(["gvkey", "fyr", "fyearq"]).alias("count_aux")] + [
        col(var).cast(pl.Float64).diff().alias(var + "_q") for var in var_list
    ]
    c1 = (col("fqtr") != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    c2 = (col("fqtr").diff() != 1).fill_null(pl.lit(True).cast(pl.Boolean))
    list_aux2 = [
        pl.when(col("count_aux") == 1).then(col(var)).otherwise(col(var + "_q")).alias(var + "_q")
        for var in var_list
    ] + [pl.when(col("count_aux") == 1).then(c1).otherwise(c2).alias("del")]
    list_aux3 = [
        pl.when(col("del")).then(fl_none()).otherwise(col(var + "_q")).alias(var + "_q")
        for var in var_list
    ]
    df = (
        df.unique(["gvkey", "fyr", "fyearq", "fqtr"])
        .sort(["gvkey", "fyr", "fyearq", "fqtr"])
        .with_columns(list_aux1)
        .sort(["gvkey", "fyr", "fyearq", "fqtr"])
        .with_columns(list_aux2)
        .with_columns(list_aux3)
        .drop(["del", "count_aux"])
        .unique(["gvkey", "fyr", "fyearq", "fqtr"])
        .sort(["gvkey", "fyr", "fyearq", "fqtr"])
    )
    return df


def ttm(var):
    """
    Description:
        4-quarter trailing total (TTM) as sum of current and previous 3 lags.

    Steps:
        1) Compute var + lag1 + lag2 + lag3.

    Output:
        Polars expression for TTM of var.
    """
    return col(var) + col(var).shift(1) + col(var).shift(2) + col(var).shift(3)


def cumulate_4q(df, var_list):
    """
    Description:
        Create 4-quarter cumulative (TTM) level variables and enforce continuity checks.

    Steps:
        1) For each *_q in var_list, create year-level name by stripping trailing 'q'.
        2) Compute TTM via ttm(*_q) and continuity flags (same gvkey/fyr/currency & ttm(fqtr)==10).
        3) Keep TTM only when continuity holds; backfill at fqtr==4 if missing.
        4) Drop helpers and *_q inputs.

    Output:
        Frame with validated 4Q cumulative variables for each input (e.g., sales, oibdp).
    """
    var_yrl_name_list = [var[:-1] for var in var_list]
    df = (
        df.with_columns(
            [
                ttm(var_yrl).alias(var_yrl_name)
                for var_yrl, var_yrl_name in zip(var_list, var_yrl_name_list, strict=True)
            ]
            + [
                (
                    (col("gvkey") == col("gvkey").shift(3))
                    & (col("fyr") == col("fyr").shift(3))
                    & (col("curcdq") == col("curcdq").shift(3))
                    & (ttm("fqtr") == 10)
                ).alias("not_null_flag")
            ]
        )
        .with_columns(
            [
                pl.when(col("not_null_flag"))
                .then(col(var_yrl_name))
                .otherwise(fl_none())
                .alias(var_yrl_name)
                for var_yrl_name in var_yrl_name_list
            ]
        )
        .with_columns(
            [
                pl.when(col(var_yrl_name).is_null() & (col("fqtr") == 4))
                .then(col(f"{var_yrl_name}y"))
                .otherwise(col(var_yrl_name))
                .alias(var_yrl_name)
                for var_yrl_name in var_yrl_name_list
            ]
        )
        .drop(
            ["not_null_flag"]
            + var_list
            + [f"{var_yrl_name}y" for var_yrl_name in var_yrl_name_list]
        )
    )
    return df


def load_raw_fund_table_and_filter(filename, start_date, source_str, mode):
    """
    Description:
        Load Compustat FUND[A/Q] parquet and filter by format, population, consolidation, and start date.

    Steps:
        1) Choose filters by mode: mode=1 → INDL/FS + HIST_STD + popsrc=I; else INDL + STD + popsrc=D.
        2) Scan parquet, add row index, apply filters (consol='C', datadate ≥ start_date).
        3) Tag rows with source string.

    Output:
        LazyFrame of filtered accounting rows with 'source'.
    """
    c1 = (col("indfmt").is_in(["INDL", "FS"])) if mode == 1 else (col("indfmt") == "INDL")
    datafmt_val = "HIST_STD" if mode == 1 else "STD"
    popsrc_val = "I" if mode == 1 else "D"
    df = (
        pl.scan_parquet(filename)
        .with_row_index("n")
        .filter(
            c1
            & (col("datafmt") == datafmt_val)
            & (col("popsrc") == popsrc_val)
            & (col("consol") == "C")
            & (col("datadate") >= start_date)
        )
        .with_columns(source=pl.lit(source_str))
    )
    return df


def apply_indfmt_filter(df):
    """
    Description:
        Resolve dual-format rows per (gvkey, datadate), preferring INDL when both exist.

    Steps:
        1) Count rows over [gvkey, datadate].
        2) Keep singletons or pairs where indfmt == 'INDL'.
        3) Drop helper columns.

    Output:
        Frame with one format per (gvkey, datadate).
    """
    df = (
        df.with_columns(count_indfmt=pl.len().over(["gvkey", "datadate"]))
        .filter(
            (col("count_indfmt") == 1) | ((col("count_indfmt") == 2) & (col("indfmt") == "INDL"))
        )
        .drop(["indfmt", "count_indfmt"])
    )
    return df


def add_fx_and_convert_vars(df, fx_df, vars, freq):
    """
    Description:
        Join FX rates and convert selected variables to USD; normalize currency code.

    Steps:
        1) Pick currency column by freq: annual→curcd, quarterly→curcdq.
        2) Left-join FX on [datadate, currency]; keep original cols + 'fx'.
        3) Multiply listed vars by fx; set currency column to 'USD'; drop fx.

    Output:
        Frame with specified vars converted to USD and currency code set to USD.
    """
    if freq == "annual":
        fx_var = "curcd"
    else:
        fx_var = "curcdq"
    aux = (
        df.join(
            fx_df,
            left_on=["datadate", fx_var],
            right_on=["datadate", "curcdd"],
            how="left",
        )
        .select(df.collect_schema().names() + ["fx"])
        .with_columns(
            [(col(var) * col("fx")).alias(var) for var in vars] + [pl.lit("USD").alias(fx_var)]
        )
        .drop("fx")
    )
    return aux


def load_mkt_equity_data(filename, alias=True):
    """
    Description:
        Load market equity by gvkey–eom and optionally alias the column name.

    Steps:
        1) Scan parquet; require gvkey, primary/common/main flags, me_company non-null.
        2) Select [gvkey, eom, me_company → (me_fiscal|me_company)].
        3) Group [gvkey, eom] and take max(me).

    Output:
        LazyFrame with one ME value per (gvkey, eom).
    """
    col_name = "me_fiscal" if alias else "me_company"
    df = (
        pl.scan_parquet(filename)
        .filter(
            (col("gvkey").is_not_null())
            & (col("primary_sec") == 1)
            & (col("me_company").is_not_null())
            & (col("common") == 1)
            & (col("obs_main") == 1)
        )
        .select(["gvkey", "eom", col("me_company").alias(col_name)])
        .group_by(["gvkey", "eom"])
        .agg(col(col_name).max())
    )
    return df


@measure_time
def standardized_accounting_data(
    paths: DataPaths, coverage, convert_to_usd, me_data_path, include_helpers_vars, start_date
):
    """
    Description:
        Build standardized annual/quarterly accounting panels (NA, Global, or World), optionally USD-converted; attach ME; write Parquet.

    Steps:
        1) Inspect FUNDQ schemas; define target income/CF/BS/other vars; collect quarterly suffix vars (…q/…y).
        2) Load & filter raw GLOBAL and/or NA (annual/quarterly) via helper; add computed fields (e.g., ni, niq, ppegtq); drop vars as needed; apply INDFMT resolver.
        3) If world: concat NA+GLOBAL and break ties per key by preferring NA.
        4) If convert_to_usd: join FX and convert listed vars (annual & quarterly).
        5) Load ME and join to annual/quarterly panels.
        6) Quarterly: quarterize …y → …y_q, coalesce to …q; create ni_qtr/sale_qtr/ocf_qtr; cumulate 4Q flows with continuity checks; normalize currency codes; de-dupe, prefer later/NA rows.
        7) Annual: add empty quarterly helpers; join ME; sort.
        8) Optionally add helper variables.
        9) Unique by (gvkey, datadate), drop row index, sort, and write 'acc_std_ann.parquet' and 'acc_std_qtr.parquet'.

    Output:
        Two Parquet files: 'acc_std_ann.parquet' (annual) and 'acc_std_qtr.parquet' (quarterly) standardized accounting data.
    """
    g_fundq_cols = (
        pl.scan_parquet(paths.raw_tables_dir / "comp_g_fundq.parquet").collect_schema().names()
    )
    fundq_cols = (
        pl.scan_parquet(paths.raw_tables_dir / "comp_fundq.parquet").collect_schema().names()
    )
    # Compustat Accounting Vars to Extract
    avars_inc = [
        "sale",
        "revt",
        "gp",
        "ebitda",
        "oibdp",
        "ebit",
        "oiadp",
        "pi",
        "ib",
        "ni",
        "mii",
        "cogs",
        "xsga",
        "xopr",
        "xrd",
        "xad",
        "xlr",
        "dp",
        "xi",
        "do",
        "xido",
        "xint",
        "spi",
        "nopi",
        "txt",
        "dvt",
    ]
    avars_cf = [
        "oancf",
        "ibc",
        "dpc",
        "xidoc",
        "capx",
        "wcapt",  # Operating
        "fincf",
        "fiao",
        "txbcof",
        "ltdch",
        "dltis",
        "dltr",
        "dlcch",
        "purtshr",
        "prstkc",
        "sstk",
        "dv",
        "dvc",
    ]  # Financing
    avars_bs = [
        "at",
        "act",
        "aco",
        "che",
        "invt",
        "rect",
        "ivao",
        "ivst",
        "ppent",
        "ppegt",
        "intan",
        "ao",
        "gdwl",
        "re",  # Assets
        "lt",
        "lct",
        "dltt",
        "dlc",
        "txditc",
        "txdb",
        "itcb",
        "txp",
        "ap",
        "lco",
        "lo",
        "seq",
        "ceq",
        "pstkrv",
        "pstkl",
        "pstk",
        "mib",
        "icapt",
    ]  # Liabilities
    # Variables in avars_other are not measured in currency units, and only available in annual data
    avars_other = ["emp"]
    avars = avars_inc + avars_cf + avars_bs
    # finding which variables of interest are available in the quarterly data
    combined_columns = g_fundq_cols + fundq_cols
    qvars_q = list(
        {
            aux_var
            for aux_var in combined_columns
            if aux_var[:-1].lower() in avars and aux_var.endswith("q")
        }
    )  # different from above to get only unique values
    qvars_y = list(
        {
            aux_var
            for aux_var in combined_columns
            if aux_var[:-1].lower() in avars and aux_var.endswith("y")
        }
    )
    qvars = qvars_q + qvars_y
    if coverage in ["global", "world"]:
        # Annual global data:
        vars_not_in_query = ["gp", "pstkrv", "pstkl", "itcb", "xad", "txbcof", "ni"]
        query_vars = [var for var in (avars + avars_other) if var not in vars_not_in_query]
        g_funda = load_raw_fund_table_and_filter(
            paths.raw_tables_dir / "comp_g_funda.parquet", start_date, "GLOBAL", 1
        )
        __gfunda = (
            g_funda.with_columns(
                ni=(col("ib") + pl.coalesce("xi", 0) + pl.coalesce("do", 0)).cast(pl.Float64)
            )
            .select(
                ["gvkey", "datadate", "n", "indfmt", "curcd", "source", "ni"]
                + [fl_none().alias(i) for i in ["gp", "pstkrv", "pstkl", "itcb", "xad", "txbcof"]]
                + query_vars
            )
            .pipe(apply_indfmt_filter)
        )
        # Quarterly global data:
        vars_not_in_query = [
            "icaptq",
            "niy",
            "txditcq",
            "txpq",
            "xidoq",
            "xidoy",
            "xrdq",
            "xrdy",
            "txbcofy",
            "niq",
            "ppegtq",
            "doq",
            "doy",
        ]
        query_vars = [var for var in qvars if var not in vars_not_in_query]
        g_fundq = load_raw_fund_table_and_filter(
            paths.raw_tables_dir / "comp_g_fundq.parquet", start_date, "GLOBAL", 1
        )
        __gfundq = (
            g_fundq.with_columns(
                niq=(col("ibq") + pl.coalesce("xiq", 0.0)).cast(pl.Float64),
                ppegtq=(col("ppentq") + col("dpactq")).cast(pl.Float64),
            )
            .select(
                [
                    "gvkey",
                    "datadate",
                    "n",
                    "indfmt",
                    "fyr",
                    "fyearq",
                    "fqtr",
                    "curcdq",
                    "source",
                    "niq",
                    "ppegtq",
                ]
                + [
                    fl_none().alias(i)
                    for i in [
                        "icaptq",
                        "niy",
                        "txditcq",
                        "txpq",
                        "xidoq",
                        "xidoy",
                        "xrdq",
                        "xrdy",
                        "txbcofy",
                    ]
                ]
                + query_vars
            )
            .pipe(apply_indfmt_filter)
        )
    if coverage in ["na", "world"]:
        # Annual north american data:
        vars_not_in_query = ["wcapt", "ltdch", "purtshr"]
        query_vars = [var for var in (avars + avars_other) if var not in vars_not_in_query]
        funda = load_raw_fund_table_and_filter(
            paths.raw_tables_dir / "comp_funda.parquet", start_date, "NA", 2
        )
        __funda = funda.select(
            ["gvkey", "datadate", "n", "curcd", "source"]
            + [fl_none().alias(i) for i in ["wcapt", "ltdch", "purtshr"]]
            + query_vars
        )
        # Quarterly north american data:
        vars_not_in_query = [
            "dvtq",
            "gpq",
            "dvty",
            "gpy",
            "ltdchy",
            "purtshry",
            "wcapty",
        ]
        query_vars = [var for var in qvars if var not in vars_not_in_query]
        fundq = load_raw_fund_table_and_filter(
            paths.raw_tables_dir / "comp_fundq.parquet", start_date, "NA", 2
        )
        __fundq = fundq.select(
            ["gvkey", "datadate", "n", "fyr", "fyearq", "fqtr", "curcdq", "source"]
            + [
                fl_none().alias(i)
                for i in ["dvtq", "gpq", "dvty", "gpy", "ltdchy", "purtshry", "wcapty"]
            ]
            + query_vars
        )
    if coverage == "world":
        __wfunda = pl.concat([__gfunda, __funda], how="diagonal_relaxed").filter(
            (pl.len().over(["gvkey", "datadate"]) == 1)
            | ((pl.len().over(["gvkey", "datadate"]) == 2) & (col("source") == "GLOBAL"))
        )
        __wfundq = pl.concat([__gfundq, __fundq], how="diagonal_relaxed").filter(
            (pl.len().over(["gvkey", "fyr", "fyearq", "fqtr"]) == 1)
            | (
                (pl.len().over(["gvkey", "fyr", "fyearq", "fqtr"]) == 2)
                & (col("source") == "GLOBAL")
            )
        )
    else:
        pass
    if coverage == "na":
        aname, qname = __funda, __fundq
    elif coverage == "global":
        aname, qname = __gfunda, __gfundq
    else:
        aname, qname = __wfunda, __wfundq

    if convert_to_usd == 1:
        fx = compustat_fx(paths).lazy()
        __compa = add_fx_and_convert_vars(aname, fx, avars, "annual")
        __compq = add_fx_and_convert_vars(qname, fx, qvars, "quarterly")
    else:
        __compa, __compq = aname, qname

    __me_data = load_mkt_equity_data(me_data_path)

    yrl_vars = [
        "cogsq",
        "xsgaq",
        "xintq",
        "dpq",
        "txtq",
        "xrdq",
        "dvq",
        "spiq",
        "saleq",
        "revtq",
        "xoprq",
        "oibdpq",
        "oiadpq",
        "ibq",
        "niq",
        "xidoq",
        "nopiq",
        "miiq",
        "piq",
        "xiq",
        "xidocq",
        "capxq",
        "oancfq",
        "ibcq",
        "dpcq",
        "wcaptq",
        "prstkcq",
        "sstkq",
        "purtshrq",
        "dsq",
        "dltrq",
        "ltdchq",
        "dlcchq",
        "fincfq",
        "fiaoq",
        "txbcofq",
        "dvtq",
    ]
    bs_vars = [
        "seqq",
        "ceqq",
        "pstkq",
        "icaptq",
        "mibq",
        "gdwlq",
        "req",
        "atq",
        "actq",
        "invtq",
        "rectq",
        "ppegtq",
        "ppentq",
        "aoq",
        "acoq",
        "intanq",
        "cheq",
        "ivaoq",
        "ivstq",
        "ltq",
        "lctq",
        "dlttq",
        "dlcq",
        "txpq",
        "apq",
        "lcoq",
        "loq",
        "txditcq",
        "txdbq",
    ]
    __compq = __compq.with_columns(
        [
            col(var).cast(pl.Int64)
            for var in ["fyr", "fyearq", "fqtr"]
            if var in __compq.collect_schema().names()
        ]
    )

    __compq = (
        __compq.pipe(quarterize, var_list=qvars_y)
        .with_columns(
            [
                pl.coalesce([f"{var[:-1]}q", f"{var[:-1]}y_q"]).alias(f"{var[:-1]}q")
                for var in qvars_y
                if f"{var[:-1]}q" in __compq.collect_schema().names()
            ]
            + [
                col(f"{var[:-1]}y_q").alias(f"{var[:-1]}q")
                for var in qvars_y
                if f"{var[:-1]}q" not in __compq.collect_schema().names()
            ]
        )
        .drop([f"{var[:-1]}y_q" for var in qvars_y])
        .with_columns(
            ni_qtr=col("ibq"),
            sale_qtr=col("saleq"),
            ocf_qtr=pl.coalesce(
                ["oancfq", (col("ibq") + col("dpq") - pl.coalesce([col("wcaptq"), 0]))]
            ),
            dsy=fl_none(),
            dsq=fl_none(),
        )
        .sort(["gvkey", "fyr", "fyearq", "fqtr", "n"])
        .pipe(cumulate_4q, var_list=yrl_vars)
        .rename(
            {
                **dict(zip(bs_vars, [aux[:-1] for aux in bs_vars], strict=True)),
                **{"curcdq": "curcd"},
            }
        )
        .unique(["gvkey", "datadate", "fyr"])
        .sort(["gvkey", "datadate", "fyr"])
        .unique(["gvkey", "datadate"], keep="last")
        .drop(["fyr", "fyearq", "fqtr"])
        .join(
            __me_data,
            how="left",
            left_on=["gvkey", "datadate"],
            right_on=["gvkey", "eom"],
        )
        .with_columns(
            [
                fl_none().alias(i)
                for i in [
                    "gp",
                    "dltis",
                    "do",
                    "dvc",
                    "ebit",
                    "ebitda",
                    "itcb",
                    "pstkl",
                    "pstkrv",
                    "xad",
                    "xlr",
                    "emp",
                ]
            ]
        )
        .sort(["gvkey", "curcd", "datadate", "source", "n"])
    )

    __compa = (
        __compa.with_columns(
            ni_qtr=fl_none(),
            sale_qtr=fl_none(),
            ocf_qtr=fl_none(),
            fqtr=pl.lit(None).cast(pl.Int64),
            fyearq=pl.lit(None).cast(pl.Int64),
            fyr=pl.lit(None).cast(pl.Int64),
        )
        .join(
            __me_data,
            how="left",
            left_on=["gvkey", "datadate"],
            right_on=["gvkey", "eom"],
        )
        .sort(["gvkey", "curcd", "datadate", "source", "n"])
    )

    if include_helpers_vars == 1:
        __compq = add_helper_vars(paths, __compq)
        __compa = add_helper_vars(paths, __compa)

    __compa.unique(["gvkey", "datadate"]).drop("n").sort(
        ["gvkey", "datadate"]
    ).collect().write_parquet(paths.interim_dir / "acc_std_ann.parquet")
    __compq.unique(["gvkey", "datadate"]).drop("n").sort(
        ["gvkey", "datadate"]
    ).collect().write_parquet(paths.interim_dir / "acc_std_qtr.parquet")


def expand(data, id_vars, start_date, end_date, freq="day", new_date_name="date"):
    """
    Description:
        Expand each row into a daily/monthly date panel between start_date and end_date.

    Steps:
        1) Build pl.date_ranges with interval 1d or 1mo; explode to one row per date.
        2) If monthly, snap dates to month-end.
        3) Drop source date columns, de-dup on id_vars + date, sort.

    Output:
        Expanded frame with a 'date' (or custom) column at the chosen frequency.
    """
    freq_range = "1d" if (freq == "day") else "1mo"
    expanded_df = (
        data.with_columns(
            pl.date_ranges(start=start_date, end=end_date, interval=freq_range).alias(new_date_name)
        )
        .explode(new_date_name)
        .drop([start_date, end_date])
    )
    if freq == "month":
        expanded_df = expanded_df.with_columns(col(new_date_name).dt.month_end())
    expanded_df = expanded_df.unique(id_vars + [new_date_name]).sort(id_vars + [new_date_name])
    return expanded_df


def sum_sas(col1, col2):
    """
    Description:
        SAS-style sum: if either input exists, sum with 0-fill; else NULL.

    Steps:
        1) Check non-null flags for both inputs.
        2) Return coalesce(col1,0) + coalesce(col2,0) when any non-null; otherwise NULL.

    Output:
        Polars expression with nullable sum.
    """
    c1 = col(col1).is_not_null()
    c2 = col(col2).is_not_null()
    return (
        pl.when(c1 | c2)
        .then(pl.coalesce([col1, 0.0]) + pl.coalesce([col2, 0.0]))
        .otherwise(fl_none())
    )


def sub_sas(col1, col2):
    """
    Description:
        SAS-style subtraction: if either input exists, subtract with 0-fill; else NULL.

    Steps:
        1) Check non-null flags for both inputs.
        2) Return coalesce(col1,0) − coalesce(col2,0) when any non-null; otherwise NULL.

    Output:
        Polars expression with nullable difference.
    """
    c1 = col(col1).is_not_null()
    c2 = col(col2).is_not_null()
    return (
        pl.when(c1 | c2)
        .then(pl.coalesce([col1, 0.0]) - pl.coalesce([col2, 0.0]))
        .otherwise(fl_none())
    )


def add_helper_vars(paths: DataPaths, data):
    """
    Description:
        Generate monthly-complete accounting panels per (gvkey,curcd), join originals, and compute a rich set of helper *_x variables.

    Steps:
        1) Create a temp DuckDB; load data; build monthly grid from min/max datadate; left-join raw rows; deduplicate per date.
        2) Bring grid to Polars; sanitize key vars; construct building blocks (sale_x, debt_x, pstk_x, opex_x, etc.).
        3) Derive financial aggregates (seq_x, nwc_x, be_x/bev_x, ebitda_x/ebit_x/op_x/ope_x, oa_x/ol_x/noa_x, etc.).
        4) Compute flows/changes (cowc_x, nncoa_x, oacc_x, tacc_x, netis_x, dbnetis_x) with 12-month diffs where available; guard with early-count rules.
        5) Finalize financing/cashflow measures (ocf_x, cop_x, fcf_x, fincf_x); drop helpers.

    Output:
        LazyFrame with monthly-complete panel and standardized helper variables (…_x) for downstream ratios/factors.
    """
    (paths.interim_dir / "aux_add_helpers.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(
        str(paths.interim_dir / "aux_add_helpers.ddb"), threads=os.cpu_count()
    )
    con.create_table("data", data.rename({"at": "at_var"}).collect(), overwrite=True)
    con.raw_sql("""
        CREATE OR REPLACE TABLE comp_dates1 AS
        SELECT
            gvkey,
            curcd,
            MIN(datadate) AS start_date,
            MAX(datadate) AS end_date
        FROM data
        GROUP BY gvkey, curcd
        ;

        CREATE OR REPLACE TABLE comp_dates2 AS
        SELECT
        c.gvkey,
        c.curcd,
        (gs.month_start
            + INTERVAL '1 month'
            - INTERVAL '1 day'
        )::DATE AS datadate
        FROM comp_dates1 AS c
        CROSS JOIN LATERAL
        generate_series(
            date_trunc('month', c.start_date),
            date_trunc('month', c.end_date),
            INTERVAL '1 month'
        ) AS gs(month_start)
        ORDER BY c.gvkey, c.curcd, datadate
        ;

        CREATE OR REPLACE TABLE helpers1_base AS
        SELECT
        a.gvkey,
        a.curcd,
        a.datadate,
        b.*,
        (b.gvkey IS NOT NULL)::INTEGER AS data_available
        FROM comp_dates2 AS a
        LEFT JOIN data AS b
        USING (gvkey, curcd, datadate)
        ;

        CREATE OR REPLACE TABLE helpers1 AS
        SELECT
        *
        FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
            PARTITION BY gvkey, curcd, datadate
            ORDER   BY datadate
            ) AS rn
        FROM helpers1_base
        ) t
        WHERE rn = 1
        ORDER BY gvkey, curcd, datadate
        ;

        CREATE OR REPLACE TABLE helpers2 AS
        SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY gvkey, curcd
            ORDER BY datadate
        ) AS count
        FROM helpers1
        ORDER BY gvkey, curcd, datadate
        ;

        ALTER TABLE helpers2 DROP COLUMN gvkey_1;
        ALTER TABLE helpers2 DROP COLUMN curcd_1;
        ALTER TABLE helpers2 DROP COLUMN datadate_1;
        ALTER TABLE helpers2 DROP COLUMN rn;
    """)

    c1 = (
        (col("dltis").is_null())
        & (col("dltr").is_null())
        & (col("ltdch").is_null())
        & (col("count") <= 12)
    )
    c2 = (col("dlcch").is_null()) & (col("count") <= 12)
    sort_vars = ["gvkey", "curcd", "datadate"]

    base = con.table("helpers2").to_polars().rename({"at_var": "at"}).lazy()
    base = base.sort(sort_vars).with_columns(
        [
            pl.when(col(var) >= 0).then(col(var)).otherwise(fl_none()).alias(var)
            for var in ["at", "sale", "revt", "dv", "che"]
        ]
    )
    helpers = (
        base.with_columns(
            sale_x=pl.coalesce(["sale", "revt"]),
            debt_x=sum_sas("dltt", "dlc"),
            pstk_x=pl.coalesce(["pstkrv", "pstkl", "pstk"]),
            opex_x=pl.coalesce(["xopr", col("cogs") + col("xsga")]),
            eqis_x=col("sstk"),
            div_x=pl.coalesce(["dvt", "dv"]),
            eqbb_x=sum_sas("prstkc", "purtshr"),
            xido_x=pl.coalesce(["xido", (col("xi") + pl.coalesce(["do", 0.0]))]),
            ca_x=pl.coalesce(["act", col("rect") + col("invt") + col("che") + col("aco")]),
            cl_x=pl.coalesce([col("lct"), col("ap") + col("dlc") + col("txp") + col("lco")]),
            fna_x=pl.coalesce(["ivst", 0.0]) + pl.coalesce(["ivao", 0.0]),
            ppeinv_x=col("ppegt") + col("invt"),
            lnoa_x=col("ppent") + col("intan") + col("ao") - col("lo") + col("dp"),
            txditc_x=pl.coalesce(["txditc", sum_sas("txdb", "itcb")]),
        )
        .with_columns(
            gp_x=pl.coalesce(["gp", col("sale_x") - col("cogs")]),
            eqnetis_x=sub_sas("eqis_x", "eqbb_x"),
            eqpo_x=col("div_x") + col("eqbb_x"),
            seq_x=pl.coalesce(
                ["seq", col("ceq") + pl.coalesce(["pstk_x", 0]), col("at") - col("lt")]
            ),
            ncl_x=col("lt") - col("cl_x"),
            coa_x=col("ca_x") - col("che"),
            col_x=col("cl_x") - pl.coalesce(["dlc", 0.0]),
            ncol_x=col("lt") - col("cl_x") - col("dltt"),
            fnl_x=col("debt_x") + pl.coalesce(["pstk_x", 0.0]),
            nwc_x=col("ca_x") - col("cl_x"),
            caliq_x=pl.coalesce([col("ca_x") - col("invt"), col("che") + col("rect")]),
            netdebt_x=col("debt_x") - pl.coalesce([col("che"), pl.lit(0.0)]),
        )
        .with_columns(
            ebitda_x=pl.coalesce(
                [
                    "ebitda",
                    "oibdp",
                    col("sale_x") - col("opex_x"),
                    col("gp_x") - col("xsga"),
                ]
            ),
            eqnpo_x=col("div_x") - col("eqnetis_x"),
            at_x=pl.coalesce(
                [
                    "at",
                    col("seq_x")
                    + col("dltt")
                    + pl.coalesce(["lct", 0.0])
                    + pl.coalesce(["lo", 0.0])
                    + pl.coalesce(["txditc", 0.0]),
                ]
            ),
            cowc_x=col("coa_x") - col("col_x"),
            ol_x=col("col_x") + col("ncol_x"),
            nfna_x=col("fna_x") - col("fnl_x"),
            be_x=col("seq_x")
            + pl.coalesce([col("txditc_x"), pl.lit(0)])
            - pl.coalesce([col("pstk_x"), pl.lit(0)]),
            bev_x=pl.coalesce(
                [
                    col("icapt") + pl.coalesce(["dlc", 0.0]) - pl.coalesce(["che", 0.0]),
                    col("netdebt_x") + col("seq_x") + pl.coalesce(["mib", 0.0]),
                ]
            ),
        )
        .with_columns(
            ebit_x=pl.coalesce(["ebit", "oiadp", col("ebitda_x") - col("dp")]),
            op_x=col("ebitda_x") + pl.coalesce(["xrd", 0.0]),
            ope_x=col("ebitda_x") - col("xint"),
            nca_x=col("at_x") - col("ca_x"),
            ncoa_x=col("at_x") - col("ca_x") - pl.coalesce(["ivao", 0]),
            aliq_x=col("che")
            + 0.75 * col("coa_x")
            + 0.5 * (col("at_x") - col("ca_x") - pl.coalesce(["intan", 0.0])),
            be_x=pl.when(col("be_x") > 0).then("be_x").otherwise(fl_none()),
            bev_x=pl.when(col("bev_x") > 0).then("bev_x").otherwise(fl_none()),
        )
        .with_columns(
            pi_x=pl.coalesce(
                [
                    "pi",
                    (
                        col("ebit_x")
                        - col("xint")
                        + pl.coalesce(["spi", 0.0])
                        + pl.coalesce(["nopi", 0.0])
                    ),
                ]
            ),
            oa_x=col("coa_x") + col("ncoa_x"),
            nncoa_x=col("ncoa_x") - col("ncol_x"),
        )
        .with_columns(
            ni_x=pl.coalesce(
                [
                    "ib",
                    (col("ni") - col("xido_x")),
                    (col("pi_x") - col("txt") - pl.coalesce(["mii", 0.0])),
                ]
            ),
            noa_x=col("oa_x") - col("ol_x"),
        )
        .sort(sort_vars)
        .with_columns(
            nix_x=pl.coalesce(
                [
                    "ni",
                    (col("ni_x") + pl.coalesce(["xido_x", 0.0])),
                    col("ni_x") + col("xi") + col("do"),
                ]
            ),
            oacc_x=pl.when(col("count") > 12)
            .then(
                pl.coalesce(
                    [
                        col("ni_x") - col("oancf"),
                        col("cowc_x").diff(n=12) + col("nncoa_x").diff(n=12),
                    ]
                )
            )
            .otherwise(fl_none()),
            dltnetis_x=pl.when(c1)
            .then(fl_none())
            .otherwise(pl.coalesce([sub_sas("dltis", "dltr"), "ltdch", col("dltt").diff(n=12)])),
            dstnetis_x=pl.when(c2)
            .then(fl_none())
            .otherwise(pl.coalesce(["dlcch", col("dlc").diff(n=12)])),
        )
        .sort(sort_vars)
        .with_columns(
            fi_x=col("nix_x") + col("xint"),
            tacc_x=pl.when(col("count") > 12)
            .then(col("oacc_x") + col("nfna_x").diff(n=12))
            .otherwise(fl_none()),
            ocf_x=pl.coalesce(
                [
                    "oancf",
                    col("ni_x") - col("oacc_x"),
                    col("ni_x") + col("dp") - pl.coalesce(["wcapt", 0.0]),
                ]
            ),
            cop_x=col("ebitda_x") + pl.coalesce(["xrd", 0.0]) - col("oacc_x"),
            dbnetis_x=sum_sas("dstnetis_x", "dltnetis_x"),
        )
        .with_columns(
            netis_x=col("eqnetis_x") + col("dbnetis_x"),
            fcf_x=col("ocf_x") - col("capx"),
        )
        .with_columns(
            fincf_x=pl.coalesce(
                [
                    "fincf",
                    (
                        col("netis_x")
                        - col("dv")
                        + pl.coalesce(["fiao", 0.0])
                        + pl.coalesce(["txbcof", 0.0])
                    ),
                ]
            )
        )
        .drop("count")
    )
    return helpers


def var_growth(var_gr, horizon):
    """
    Description:
        Multi-period growth of var_gr over `horizon` months, guarding for coverage.

    Steps:
        1) Build name: strip '_x' → '{base}_gr{horizon/12}'.
        2) Compute (var / var.shift(horizon)) − 1.
        3) Keep only if var.shift(horizon) > 0 and count > horizon.

    Output:
        Polars expression aliasing the growth column.
    """
    name_gr = var_gr.replace("_x", "")
    name_gr = f"{name_gr}_gr{int(horizon / 12)}"
    name_gr_exp = (col(var_gr) / col(var_gr).shift(horizon)) - 1
    c1 = (col(var_gr).shift(horizon) > 0) & (col("count") > horizon)
    name_gr_col = pl.when(c1).then(name_gr_exp).otherwise(fl_none()).alias(name_gr)
    return name_gr_col


def chg_to_assets(var_gr, horizon):
    """
    Description:
        Change in var_gr over `horizon` months scaled by current assets.

    Steps:
        1) Name as '{base}_gr{horizon/12}a'.
        2) Compute var.diff(horizon) / at_x.
        3) Keep only if at_x > 0 and count > horizon.

    Output:
        Polars expression for asset-scaled change.
    """
    name_gr = var_gr.replace("_x", "")
    name_gr = f"{name_gr}_gr{int(horizon / 12)}a"
    # name_gr_exp = ((col(var_gr) - col(var_gr).shift(horizon))/col('at_x'))
    name_gr_exp = col(var_gr).diff(horizon) / col("at_x")
    c1 = (col("at_x") > 0) & (col("count") > horizon)
    name_gr_col = pl.when(c1).then(name_gr_exp).otherwise(fl_none()).alias(name_gr)
    return name_gr_col


def chg_to_lagassets(var_gr):
    """
    Description:
        12-month change in var_gr scaled by lagged assets.

    Steps:
        1) Name as '{base}_gr1a'.
        2) Compute (var − var.shift(12)) / at_x.shift(12).
        3) Keep only if at_x.shift(12) > 0 and count > 12.

    Output:
        Polars expression for change / lagged assets.
    """
    name_gr = var_gr.replace("_x", "")
    name_gr = f"{name_gr}_gr1a"
    name_gr_exp = (col(var_gr) - col(var_gr).shift(12)) / col("at_x").shift(12)
    c1 = (col("at_x").shift(12) > 0) & (col("count") > 12)
    return pl.when(c1).then(name_gr_exp).otherwise(fl_none()).alias(name_gr)


def chg_to_exp(var):
    """
    Description:
        Change relative to 12–24M trailing average of var.

    Steps:
        1) Name as '{base}_ce'.
        2) Denominator = mean(var.shift(12), var.shift(24)).
        3) Return var / denom − 1 when denom > 0 and count > 24.

    Output:
        Polars expression for change-to-expected metric.
    """
    new_name = var.replace("_x", "")
    new_name = f"{new_name}_ce"
    c1 = (col(var).shift(12) + col(var).shift(24)) > 0
    c2 = col("count") > 24
    num = col(var)
    den = (col(var).shift(12) + col(var).shift(24)) / 2
    return pl.when(c1 & c2).then(num / den - 1).otherwise(fl_none()).alias(new_name)


def chg_to_avgassets(var):
    """
    Description:
        12-month change in var scaled by average of current and lagged assets.

    Steps:
        1) Name as '{base}_gr1a'.
        2) Compute (var − var.shift(12)) / (at_x + at_x.shift(12)).
        3) Keep only if denominator > 0 and count > 12.

    Output:
        Polars expression for change / avg assets.
    """
    new_name = var.replace("_x", "")
    new_name = f"{new_name}_gr1a"
    c1 = (col("at_x") + col("at_x").shift(12)) > 0
    c2 = col("count") > 12
    num = col(var) - col(var).shift(12)
    den = col("at_x") + col("at_x").shift(12)
    return pl.when(c1 & c2).then(num / den).otherwise(fl_none()).alias(new_name)


def standardized_unexpected(df, var, qtrs, qtrs_min):
    """
    Description:
        Standardized unexpected change of var vs. rolling 3q×qtrs history.

    Steps:
        1) Compute __chg = var − var.shift(12) per (gvkey,curcd).
        2) Build aux list of past __chg at 3-month steps over 3*qtrs; get mean, std, n.
        3) Keep mean/std only if n > qtrs_min; shift mean/std by 3 months for forecasting.
        4) SU = (Δ12 − mean.shift(3)) / std.shift(3); require count > 12 + 3*qtrs.
        5) Drop helper columns.

    Output:
        DataFrame with '{base}_su' standardized surprise column.
    """

    name = var.replace("_x", "")
    name = f"{name}_su"
    c1 = col("__chg_n") > qtrs_min
    aux_std = (
        pl.when(col("__chg_std").shift(3) != 0).then(col("__chg_std").shift(3)).otherwise(fl_none())
    )
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns((col(var) - col(var).shift(12)).over(["gvkey", "curcd"]).alias("__chg"))
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            aux=pl.concat_list(
                [col("__chg").shift(i).over(["gvkey", "curcd"]) for i in range(0, (3 * qtrs), 3)]
            ).list.drop_nulls()
        )
        .with_columns(
            __chg_mean=col("aux").list.mean(),
            __chg_n=col("aux").list.len(),
            __chg_std=col("aux").list.eval(pl.element().std()),
        )
        .explode("__chg_std")
        .with_columns(
            __chg_mean=pl.when(c1).then(col("__chg_mean")).otherwise(fl_none()),
            __chg_std=pl.when(c1).then(col("__chg_std")).otherwise(fl_none()),
        )
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            ((col(var) - col(var).shift(12) - col("__chg_mean").shift(3)) / aux_std)
            .over(["gvkey", "curcd"])
            .alias(name)
        )
        .with_columns(
            pl.when(col("count") > (12 + qtrs * 3)).then(col(name)).otherwise(fl_none()).alias(name)
        )
        .drop(["__chg", "__chg_mean", "__chg_std", "__chg_n", "aux"])
    )
    return df


def volq(df, name, var, qtrs, qtrs_min):
    """
    Description:
        Quarterly volatility of var over qtrs quarters (3*qtrs months).

    Steps:
        1) Build list of var shifted at 3-month steps for 3*qtrs window.
        2) Compute list std and length; keep if count > (qtrs−1)*3 and n ≥ qtrs_min.
        3) Drop helpers.

    Output:
        DataFrame with a '{name}' volatility column.
    """
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            aux=pl.concat_list(
                [col(var).shift(i).over(["gvkey", "curcd"]) for i in range(0, (3 * qtrs), 3)]
            ).list.drop_nulls()
        )
        .with_columns([col("aux").list.std().alias(name), col("aux").list.len().alias("__n")])
        .with_columns(
            pl.when((col("count") > ((qtrs - 1) * 3)) & (col("__n") >= qtrs_min))
            .then(col(name))
            .otherwise(fl_none())
            .alias(name)
        )
        .drop(["__n", "aux"])
    )
    return df


def vola(df, name, var, yrs, yrs_min):
    """
    Description:
        Annual volatility of var over yrs years (12*yrs months).

    Steps:
        1) Build list of var shifted at 12-month steps for 12*yrs window.
        2) Compute list std and length; keep if count > (yrs−1)*12 and n ≥ yrs_min.
        3) Drop helpers.

    Output:
        DataFrame with a '{name}' annualized volatility column.
    """
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            aux=pl.concat_list(
                [col(var).shift(i).over(["gvkey", "curcd"]) for i in range(0, (12 * yrs), 12)]
            ).list.drop_nulls()
        )
        .with_columns([col("aux").list.std().alias(name), col("aux").list.len().alias("__n")])
        .with_columns(
            pl.when((col("count") > ((yrs - 1) * 12)) & (col("__n") >= yrs_min))
            .then(col(name))
            .otherwise(fl_none())
            .alias(name)
        )
        .drop(["__n", "aux"])
    )
    return df


def earnings_variability(df, esm_h=5):
    """
    Description:
        Ratio of earnings volatility: std(ROA) / std(CROA) over esm_h years.

    Steps:
        1) Compute __roa = ni_x/at_x and __croa = ocf_x/at_x.
        2) Build annual lists over 12*esm_h months; get std and count for each.
        3) earnings_variability = __roa_std / __croa_std where counts ≥ esm_h and __croa_std>0.
        4) Drop helpers.

    Output:
        DataFrame with 'earnings_variability' column.
    """
    c1 = (
        (col("count") > (12 * esm_h))
        & (col("__croa_std") > 0)
        & (col("__roa_n") >= esm_h)
        & (col("__croa_n") >= esm_h)
    )
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            [
                safe_div("ni_x", "at_x", "__roa", 6),
                safe_div("ocf_x", "at_x", "__croa", 6),
            ]
        )
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            aux1=pl.concat_list(
                [col("__roa").shift(i).over(["gvkey", "curcd"]) for i in range(0, (12 * esm_h), 12)]
            ).list.drop_nulls(),
            aux2=pl.concat_list(
                [
                    col("__croa").shift(i).over(["gvkey", "curcd"])
                    for i in range(0, (12 * esm_h), 12)
                ]
            ).list.drop_nulls(),
        )
        .with_columns(
            __roa_std=col("aux1").list.eval(pl.element().std()),
            __roa_n=col("aux1").list.len(),
            __croa_std=col("aux2").list.eval(pl.element().std()),
            __croa_n=col("aux2").list.len(),
        )
        .explode(["__roa_std", "__croa_std"])
        .with_columns(safe_div("__roa_std", "__croa_std", "earnings_variability"))
        .with_columns(
            earnings_variability=pl.when(c1).then(col("earnings_variability")).otherwise(fl_none())
        )
        .drop(
            [
                "__roa",
                "__croa",
                "__roa_n",
                "__croa_n",
                "__roa_std",
                "__croa_std",
                "aux1",
                "aux2",
            ]
        )
    )
    return df


def roe_and_g_exps(i, g_c, g_ar1, roe_c, roe_ar1):
    """
    Description:
        Helpers for AR(1) projections of growth (g) and ROE.

    Steps:
        1) Return expressions: __g{i} = g_c + g_ar1*__g{i-1}; __roe{i} = roe_c + roe_ar1*__roe{i-1}.

    Output:
        List of Polars expressions for next-step __g and __roe.
    """
    return [
        (g_c + g_ar1 * col(f"__g{i - 1}")).alias(f"__g{i}"),
        (roe_c + roe_ar1 * col(f"__roe{i - 1}")).alias(f"__roe{i}"),
    ]


def be_and_cd_exps(i):
    """
    Description:
        Helpers to roll forward book equity and compute cash dividends per period.

    Steps:
        1) __be{i} = __be{i-1} * (1 + __g{i})
        2) __cd{i} = __be{i-1} * (__roe{i} − __g{i})

    Output:
        List of Polars expressions for next-step __be and __cd.
    """

    return [
        (col(f"__be{i - 1}") * (1 + col(f"__g{i}"))).alias(f"__be{i}"),
        (col(f"__be{i - 1}") * (col(f"__roe{i}") - col(f"__g{i}"))).alias(f"__cd{i}"),
    ]


def equity_duration_cd(
    df, horizon=10, r=0.12, roe_mean=0.12, roe_ar1=0.57, g_mean=0.06, g_ar1=0.24
):
    """
    Description:
        Cash-dividend-based equity duration over a finite horizon with AR(1) ROE/g.

    Steps:
        1) Initialize __roe0 = ni_x / be_x.shift(12), __g0 = sale growth, __be0 = be_x (guard with counts>12).
        2) For t=1..horizon: project __g{t}, __roe{t}; update __be{t}, __cd{t}.
        3) Compute ed_cd   = Σ CD_t / (1+r)^t and ed_cd_w = Σ t*CD_t / (1+r)^t.
        4) Flag ed_err if any projected __be_t < 0; set ed_constant = horizon + (1+r)/r.
        5) Drop projection helper columns.

    Output:
        Columns: ed_cd, ed_cd_w, ed_constant, ed_err (and existing data).
    """

    c1 = (col("count") > 12) & (col("be_x").shift(12) > 1)
    c2 = (col("count") > 12) & (col("sale_x").shift(12) > 1)
    roe_c = roe_mean * (1 - roe_ar1)
    g_c = g_mean * (1 - g_ar1)
    roe0_exp = pl.when(c1).then(col("ni_x") / col("be_x").shift(12)).otherwise(fl_none())
    g0_exp = pl.when(c2).then(col("sale_x") / col("sale_x").shift(12) - 1).otherwise(fl_none())
    be0_exp = col("be_x")
    df = df.sort(["gvkey", "curcd", "datadate"]).with_columns(
        __roe0=roe0_exp, __g0=g0_exp, __be0=be0_exp
    )
    for t in range(1, horizon + 1):
        df = df.with_columns(roe_and_g_exps(t, g_c, g_ar1, roe_c, roe_ar1))
    for t in range(1, horizon + 1):
        df = df.with_columns(be_and_cd_exps(t))

    ed_cd_w_exp = sum(t * col(f"__cd{t}") / ((1 + r) ** t) for t in range(1, horizon + 1))
    ed_cd_exp = sum(col(f"__cd{t}") / ((1 + r) ** t) for t in range(1, horizon + 1))
    c_aux = pl.any_horizontal([col(f"__be{t}") < 0 for t in range(1, horizon + 1)])
    ed_err_exp = pl.when(c_aux).then(pl.lit(1.0)).otherwise(pl.lit(0.0))
    # c_aux = pl.any_horizontal(pl.col(r"^__be\d+$") < 0) Much cooler but not safe in case there are any other __be columns from before
    df = df.with_columns(
        ed_constant=(pl.lit(horizon) + ((1 + r) / r)),
        ed_cd_w=ed_cd_w_exp,
        ed_cd=ed_cd_exp,
        ed_err=ed_err_exp,
    )

    cols_to_drop = [
        y
        for x in [[f"__roe{i}", f"__g{i}", f"__be{i}", f"__cd{i}"] for i in range(0, horizon + 1)]
        for y in x
    ]
    cols_to_drop.remove("__cd0")
    df = df.drop(cols_to_drop)
    return df


def pitroski_f(df, name="f_score"):
    """
    Description:
        Compute Pitroski F-score from profitability, leverage/liquidity, and operating efficiency.

    Steps:
        1) Profitability: ROA, ΔROA, CFO>0, Accruals (CFO−ROA).
        2) Leverage/Liquidity: Δleverage (↓), Δcurrent ratio (↑), no equity issuance.
        3) Efficiency: Δgross margin (↑), Δasset turnover (↑).
        4) Score each condition as 1/0 with data-availability guards; sum to name.

    Output:
        DataFrame with '{name}' integer score and helpers removed.
    """

    c1 = col("count") > 12
    c2 = col("at_x").shift(12) > 0
    c3 = col("at_x") > 0
    c4 = col("cl_x") > 0
    c5 = col("cl_x").shift(12) > 0
    c6 = col("sale_x") > 0
    c7 = col("sale_x").shift(12) > 0
    c8 = col("count") > 24
    c9 = col("at_x").shift(24) > 0
    col_exp = (pl.coalesce([col("__f_eqis"), 0]) == 0).cast(pl.Int32) + (col("__f_lev") < 0).cast(
        pl.Int32
    )
    for var_name in [
        "__f_roa",
        "__f_croa",
        "__f_droa",
        "__f_acc",
        "__f_liq",
        "__f_gm",
        "__f_aturn",
    ]:
        col_exp += (col(var_name) > 0).cast(pl.Int32)
    df = (
        df.sort(["gvkey", "curcd"])
        .with_columns(
            __f_roa=pl.when(c1 & c2).then(col("ni_x") / col("at_x").shift(12)).otherwise(fl_none()),
            __f_croa=pl.when(c1 & c2)
            .then(col("ocf_x") / col("at_x").shift(12))
            .otherwise(fl_none()),
        )
        .sort(["gvkey", "curcd"])
        .with_columns(
            __f_droa=pl.when(c1)
            .then(col("__f_roa") - col("__f_roa").shift(12))
            .otherwise(fl_none()),
            __f_acc=col("__f_croa") - col("__f_roa"),
            __f_lev=pl.when(c1 & c2 & c3)
            .then(col("dltt") / col("at_x") - (col("dltt") / col("at_x")).shift(12))
            .otherwise(fl_none()),
            __f_liq=pl.when(c1 & c4 & c5)
            .then(col("ca_x") / col("cl_x") - (col("ca_x") / col("cl_x")).shift(12))
            .otherwise(fl_none()),
            __f_eqis=col("eqis_x"),
            __f_gm=pl.when(c1 & c6 & c7)
            .then(col("gp_x") / col("sale_x") - (col("gp_x") / col("sale_x")).shift(12))
            .otherwise(fl_none()),
            __f_aturn=pl.when(c2 & c8 & c9)
            .then(
                (col("sale_x") / col("at_x").shift(12))
                - (col("sale_x").shift(12) / col("at_x").shift(24))
            )
            .otherwise(fl_none()),
        )
        .with_columns(col_exp.alias(name))
        .drop(
            [
                "__f_roa",
                "__f_croa",
                "__f_droa",
                "__f_acc",
                "__f_lev",
                "__f_liq",
                "__f_eqis",
                "__f_gm",
                "__f_aturn",
            ]
        )
    )
    return df


def ohlson_o(df, name="o_score"):
    """
    Description:
        Ohlson O-score using financial ratios and earnings dynamics.

    Steps:
        1) Build features: lev=debt/at_x, roe=nix_x/at_x, cacl=cl_x/ca_x, lat=log(at_x),
        wc=(ca_x−cl_x)/at_x, ffo=(pi_x+dp)/lt, neg_eq=1[lt>at_x], neg_earn=1[both nix_x<0], nich=(Δnix)/( |nix|+|nix₋₁| ).
        2) Apply linear index: −1.32 −0.407*lat +6.03*lev +1.43*wc +0.076*cacl −1.72*neg_eq −2.37*roe −1.83*ffo +0.285*neg_earn −0.52*nich.

    Output:
        DataFrame with '{name}' continuous score.
    """

    c1 = (col("count") > 12) & (col("nix_x").is_not_null()) & (col("nix_x").shift(12).is_not_null())
    c2 = (col("count") > 12) & (col("nix_x").abs() + col("nix_x").shift(12).abs() != 0)
    exp_aux1 = ((col("nix_x") < 0) & (col("nix_x").shift(12) < 0)).cast(pl.Float64)
    exp_aux2 = (col("nix_x") - col("nix_x").shift(12)) / (
        col("nix_x").abs() + col("nix_x").shift(12).abs()
    )
    col1 = (pl.when(c1).then(exp_aux1).otherwise(fl_none())).alias("__o_neg_earn")
    col2 = (pl.when(c2).then(exp_aux2).otherwise(fl_none())).alias("__o_nich")
    col3 = safe_div("debt_x", "at_x", "__o_lev", 3)
    col4 = safe_div("nix_x", "at_x", "__o_roe", 3)
    col5 = safe_div("cl_x", "ca_x", "__o_cacl", 3)
    col6 = (pl.when(col("at_x") > 0).then(col("at_x").log()).otherwise(fl_none())).alias("__o_lat")
    col7 = (
        pl.when(col("at_x") > 0)
        .then((col("ca_x") - col("cl_x")) / col("at_x"))
        .otherwise(fl_none())
    ).alias("__o_wc")
    col8 = (
        pl.when(col("lt") > 0).then((col("pi_x") + col("dp")) / col("lt")).otherwise(fl_none())
    ).alias("__o_ffo")
    col9 = (
        pl.when((col("lt").is_not_null()) & (col("at_x").is_not_null()))
        .then((col("lt") > col("at_x")).cast(pl.Int32))
        .otherwise(fl_none())
    ).alias("__o_neg_eq")
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns([col1, col2, col3, col4, col5, col6, col7, col8, col9])
        .with_columns(
            (
                -1.32
                - 0.407 * col("__o_lat")
                + 6.03 * col("__o_lev")
                - 1.43 * col("__o_wc")
                + 0.076 * col("__o_cacl")
                - 1.72 * col("__o_neg_eq")
                - 2.37 * col("__o_roe")
                - 1.83 * col("__o_ffo")
                + 0.285 * col("__o_neg_earn")
                - 0.52 * col("__o_nich")
            ).alias(name)
        )
    )
    return df


def altman_z(df, name="z_score"):
    """
    Description:
        Altman Z-score combining five standardized components.

    Steps:
        1) Compute components: WC/TA, RE/TA, EBITDA/TA, ME/LT, Sales/TA.
        2) Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBITDA/TA + 0.6*ME/LT + 1.0*Sales/TA.
        3) Drop helper columns.

    Output:
        DataFrame with '{name}' Z-score.
    """

    df = (
        df.with_columns(
            [
                pl.when(col("at_x") > 0)
                .then((col("ca_x") - col("cl_x")) / col("at_x"))
                .otherwise(fl_none())
                .alias("__z_wc"),
                safe_div("re", "at_x", "__z_re", 3),
                safe_div("ebitda_x", "at_x", "__z_eb", 3),
                safe_div("me_fiscal", "lt", "__z_me", 3),
                safe_div("sale_x", "at_x", "__z_sa", 3),
            ]
        )
        .with_columns(
            (
                1.2 * col("__z_wc")
                + 1.4 * col("__z_re")
                + 3.3 * col("__z_eb")
                + 0.6 * col("__z_me")
                + 1.0 * col("__z_sa")
            ).alias(name)
        )
        .drop(["__z_wc", "__z_re", "__z_eb", "__z_sa", "__z_me"])
    )
    return df


def intrinsic_value(df, name="intrinsic_value", r=0.12):
    """
    Description:
        One-step intrinsic value via payout-adjusted ROE and residual-income terms.

    Steps:
        1) __iv_po = div_x / nix_x if nix_x>0 else div_x / (at_x*0.06) if at_x≠0.
        2) __iv_roe = nix_x / avg(be_x, be_x_lag12) when count>12 and be>0.
        3) __iv_be1 = (1 + (1−__iv_po)*__iv_roe) * be_x.
        4) name = be_x + ((__iv_roe−r)/(1+r))*be_x + ((__iv_roe−r)/((1+r)*r))*__iv_be1; clip to >0.

    Output:
        DataFrame with '{name}' intrinsic value and helpers removed.
    """

    c1 = col("count") > 12
    c2 = col("be_x") + col("be_x").shift(12) > 0
    iv_po_exp = (
        pl.when(col("nix_x") > 0)
        .then(col("div_x") / col("nix_x"))
        .when(col("at_x") != 0)
        .then(col("div_x") / (col("at_x") * 0.06))
        .otherwise(fl_none())
    ).alias("__iv_po")
    iv_roe_exp = (
        pl.when(c1 & c2)
        .then(col("nix_x") / ((col("be_x") + col("be_x").shift(12)) / 2))
        .otherwise(fl_none())
    ).alias("__iv_roe")
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns([iv_roe_exp, iv_po_exp])
        .with_columns(__iv_be1=((1 + (1 - col("__iv_po")) * col("__iv_roe")) * col("be_x")))
        .with_columns(
            (
                col("be_x")
                + (((col("__iv_roe") - r) / (1 + r)) * col("be_x"))
                + (((col("__iv_roe") - r) / ((1 + r) * r)) * col("__iv_be1"))
            ).alias(name)
        )
        .with_columns(pl.when(col(name) > 0).then(col(name)).otherwise(fl_none()).alias(name))
        .drop(["__iv_po", "__iv_roe", "__iv_be1"])
    )
    return df


def kz_index(df, name="kz_index"):
    """
    Description:
        Kaplan–Zingales (KZ) index from five firm variables.

    Steps:
        1) Build: __kz_cf=(ni_x+dp)/ppent_lag; __kz_dv=div_x/ppent_lag; __kz_cs=che/ppent_lag (need count>12, ppent_lag>0).
        2) __kz_q=(at_x+me_fiscal−be_x)/at_x (need at_x>0).
        3) __kz_db=debt_x/(debt_x+seq_x) (den≠0).
        4) KZ = −1.002*cf + 0.283*q + 3.139*db − 39.368*dv − 1.315*cs.

    Output:
        DataFrame with '{name}' continuous KZ score.
    """
    c1 = (col("count") > 12) & (col("ppent").shift(12) > 0)
    c2 = col("at_x") > 0
    c3 = (col("debt_x") + col("seq_x")) != 0
    col1 = (
        pl.when(c1).then((col("ni_x") + col("dp")) / col("ppent").shift(12)).otherwise(fl_none())
    ).alias("__kz_cf")
    col2 = (pl.when(c1).then(col("div_x") / col("ppent").shift(12)).otherwise(fl_none())).alias(
        "__kz_dv"
    )
    col3 = (pl.when(c1).then(col("che") / col("ppent").shift(12)).otherwise(fl_none())).alias(
        "__kz_cs"
    )
    col4 = (
        pl.when(c2)
        .then((col("at_x") + col("me_fiscal") - col("be_x")) / col("at_x"))
        .otherwise(fl_none())
    ).alias("__kz_q")
    col5 = (
        pl.when(c3).then(col("debt_x") / (col("debt_x") + col("seq_x"))).otherwise(fl_none())
    ).alias("__kz_db")
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns([col1, col2, col3, col4, col5])
        .with_columns(
            (
                -1.002 * col("__kz_cf")
                + 0.283 * col("__kz_q")
                + 3.139 * col("__kz_db")
                - 39.368 * col("__kz_dv")
                - 1.315 * col("__kz_cs")
            ).alias(name)
        )
    )
    return df


# Make this be able to take a list of variables and horizons to only sort once for the whole list rather than once per variable.
def chg_var1_to_var2(df, name, var1, var2, horizon):
    """
    Description:
        Horizon change in the ratio var1/var2, guarded for coverage and den≤0.

    Steps:
        1) __x = var1/var2 when var2>0 else NULL.
        2) Sort; compute Δhorizon(__x) when count>horizon.
        3) Drop helper.

    Output:
        DataFrame with '{name}' change over the specified horizon.
    """
    df = (
        df.with_columns(__x=pl.when(col(var2) <= 0).then(None).otherwise(col(var1) / col(var2)))
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            pl.when(col("count") <= horizon)
            .then(None)
            .otherwise(col("__x").diff(horizon))
            .alias(name)
        )
        .drop("__x")
    )
    return df


def compute_earnings_persistence(paths: DataPaths, data_path, __n, __min):
    """
    Description:
        Earnings persistence: AR(1) of NI/AT (annual steps) with rolling cohorts.

    Steps:
        1) Load gvkey,curcd,datadate,ni_x,at_x; create __ni_at=ni_x/at_x (at_x>0).
        2) Keep obs with __ni_at and its 12-month lag; build calendar cohorts by month for N years.
        3) For each (gvkey,curcd,calc_date,grp) with ≥ __min obs, run OLS __ni_at on lag.
        4) Aggregate to slope (ni_ar1) and residual std (ni_ivol); save.

    Output:
        Writes 'ni_ar_res.parquet' with [gvkey, curcd, datadate, ni_ar1, ni_ivol].
    """

    months = 12 * __n
    con = ibis.duckdb.connect(
        str(paths.interim_dir / "aux_earnings_pers.ddb"), threads=os.cpu_count()
    )
    con.create_table(
        "raw_table",
        pl.scan_parquet(data_path).select(["gvkey", "curcd", "datadate", "ni_x", "at_x"]).collect(),
        overwrite=True,
    )

    con.raw_sql(f"""
    CREATE OR REPLACE TABLE base_data AS
    WITH base AS (
      SELECT
        gvkey,
        curcd,
        datadate,
        CASE WHEN at_x > 0 THEN ni_x / at_x END AS __ni_at,
        ROW_NUMBER() OVER (PARTITION BY gvkey, curcd ORDER BY datadate) AS rn
      FROM raw_table
    )
    SELECT
      gvkey,
      curcd,
      datadate,
      __ni_at,
      CASE WHEN rn > 12
           THEN LAG(__ni_at, 12) OVER (PARTITION BY gvkey, curcd ORDER BY datadate)
      END AS __ni_at_l1
    FROM base
    QUALIFY __ni_at IS NOT NULL AND __ni_at_l1 IS NOT NULL
    ORDER BY gvkey, curcd, datadate;

    CREATE OR REPLACE TABLE dates_apply AS
    SELECT
      datadate,
      (ROW_NUMBER() OVER (ORDER BY datadate) % {months}) AS grp
    FROM (SELECT DISTINCT datadate FROM base_data);

    CREATE OR REPLACE TABLE calc_dates AS
    SELECT
      a.datadate,
      b.datadate AS calc_date,
      b.grp      AS calc_grp
    FROM dates_apply a
    JOIN dates_apply b
      ON a.datadate > make_date(EXTRACT(YEAR FROM b.datadate) - {__n}, 12, 31)
     AND a.datadate <= b.datadate
     AND EXTRACT(MONTH FROM a.datadate) = EXTRACT(MONTH FROM b.datadate);

    CREATE OR REPLACE TABLE calc_data AS
    SELECT
      a.*,
      c.calc_date,
      c.calc_grp
    FROM base_data a
    JOIN calc_dates c
      ON a.datadate = c.datadate
    QUALIFY COUNT(*) OVER (
      PARTITION BY a.gvkey, a.curcd, c.calc_date
    ) >= {__min}
    ORDER BY a.gvkey, c.calc_date;

    CREATE OR REPLACE TABLE reg_results AS
    WITH fit AS (
      SELECT
        gvkey,
        curcd,
        calc_grp,
        calc_date,
        __ni_at,
        __ni_at_l1,
        regr_slope(__ni_at, __ni_at_l1) OVER (
          PARTITION BY gvkey, curcd, calc_grp, calc_date
        ) AS slope,
        regr_intercept(__ni_at, __ni_at_l1) OVER (
          PARTITION BY gvkey, curcd, calc_grp, calc_date
        ) AS intercept
      FROM calc_data
      WHERE __ni_at IS NOT NULL AND __ni_at_l1 IS NOT NULL
    ),
    resids AS (
      SELECT
        gvkey,
        curcd,
        calc_grp,
        calc_date,
        slope,
        (__ni_at - (intercept + slope * __ni_at_l1)) AS resid
      FROM fit
    )
    SELECT
      gvkey,
      curcd,
      calc_date as datadate,
      slope as ni_ar1,
      stddev_samp(resid) AS ni_ivol,
    FROM resids
    GROUP BY gvkey, curcd, calc_grp, calc_date, slope
    HAVING COUNT(*) >= {__min}
    ORDER BY gvkey, curcd, calc_date;
    """)
    con.table("reg_results").to_parquet(paths.interim_dir / "ni_ar_res.parquet")
    con.disconnect()
    (paths.interim_dir / "aux_earnings_pers.ddb").unlink(missing_ok=True)


def scale_me(var):
    """
    Description:
        Scale a flow/level by market equity (company), FX-adjusted.

    Steps:
        1) name='{base}_me'; compute (var*fx)/me_company when me_company≠0.

    Output:
        Polars expression '{base}_me'.
    """
    # Removing '_x' from the column name
    name = var.replace("_x", "")
    # Appending '_me' to the name
    name = f"{name}_me"
    col_aux = (col(var) * col("fx")) / col("me_company")
    return pl.when(col("me_company") != 0).then(col_aux).otherwise(fl_none()).alias(name)


def scale_mev(var):
    """
    Description:
        Scale a flow/level by market equity variant 'mev', FX-adjusted.

    Steps:
        1) name='{base}_mev'; compute (var*fx)/mev when mev≠0.

    Output:
        Polars expression '{base}_mev'.
    """

    # Removing '_x' from the column name
    name = var.replace("_x", "")
    # Appending '_me' to the name
    name = f"{name}_mev"
    col_aux = (col(var) * col("fx")) / col("mev")
    return pl.when(col("mev") != 0).then(col_aux).otherwise(fl_none()).alias(name)


def mean_year(var):
    """
    Description:
        Year-mean of a variable using current and 12-month lag, with fallbacks.

    Steps:
        1) If both present: return avg(current, lag12) over (gvkey,curcd).
        2) Else return whichever is present; else NULL.

    Output:
        Polars expression for 1-year mean.
    """
    return (
        pl.when(
            col(var).is_not_null() & (col(var).shift(12).over(["gvkey", "curcd"])).is_not_null()
        )
        .then((col(var) + col(var).shift(12)).over(["gvkey", "curcd"]) / 2)
        .when(col(var).is_not_null())
        .then(col(var))
        .when((col(var).shift(12).over(["gvkey", "curcd"])).is_not_null())
        .then(col(var).shift(12).over(["gvkey", "curcd"]))
        .otherwise(fl_none())
    )


def temp_liq_rat(col_avg, den, alias):
    """
    Description:
        Liquidity ratio using year-mean in numerator: 365*avg(col_avg)/den.

    Steps:
        1) Compute 365*mean_year(col_avg)/den.
        2) Keep when count>12 and den≠0.

    Output:
        Polars expression aliased to provided name.
    """

    col1 = 365 * mean_year(col_avg) / col(den)
    c1 = col("count") > 12
    c2 = col(den) != 0
    return pl.when(c1 & c2).then(col1).otherwise(fl_none()).alias(alias)


def temp_rat_other(num, den, alias):
    """
    Description:
        Generic ratio using year-mean in denominator.

    Steps:
        1) Compute num / mean_year(den).
        2) Keep when count>12 and mean_year(den)≠0.

    Output:
        Polars expression aliased to provided name.
    """

    col_expr = col(num) / mean_year(den)
    c1 = col("count") > 12
    c2 = mean_year(den) != 0
    return pl.when(c1 & c2).then(col_expr).otherwise(fl_none()).alias(alias)


def temp_rat_other_spc():
    """
    Description:
        Accounts payable turnover ratio.

    Steps:
        1) Numerator = cogs + invt − invt_lag12 (per entity).
        2) Denominator = mean_year(ap).
        3) Keep when count>12 and mean_year(ap)≠0.

    Output:
        Polars expression 'ap_turnover'.
    """
    num_expr = col("cogs") + col("invt") - col("invt").shift(12)
    col_expr = num_expr.over(["gvkey", "curcd"]) / mean_year("ap")
    c1 = col("count") > 12
    c2 = mean_year("ap") != 0
    return pl.when(c1 & c2).then(col_expr).otherwise(fl_none()).alias("ap_turnover")


def safe_div(num, den, name, mode=1):
    """
    Description:
        Safe division utility with multiple guard modes.

        `num`/`den` may each be a column-name str, a Polars expression, or a
        numeric literal — they are coerced to expressions internally, so
        composite numerators/denominators (e.g. ``col("a") - col("b")``) can be
        passed directly without materializing an intermediate column.

    Steps:
        Mode 1: num/den if den≠0.
        Mode 2: num/|den| if den≠0.
        Mode 3: num/den if den>0.
        Mode 4: (num/den_lag12) over (gvkey,curcd) if count>12 and den_lag12>0.
        Mode 5: (num*fx)/den if den≠0.
        Mode 6: (num/den_lag12) over (gvkey,curcd) if den_lag12≠0.
        Mode 7: (num/den)_lag12 over (gvkey,curcd) if den_lag12>0.
        Mode 8: num/den if num>0 and den>0.
        Mode 9: num/den_lag3 if count>3 and den_lag3>0.

    Output:
        Polars expression named 'name' with NULLs on guard failures.
    """

    def _expr(x):
        if isinstance(x, str):
            return col(x)
        if isinstance(x, pl.Expr):
            return x
        return pl.lit(x)

    num = _expr(num)
    den = _expr(den)

    if mode == 1:
        return pl.when(den != 0).then(num / den).otherwise(fl_none()).alias(name)
    if mode == 2:
        return pl.when(den != 0).then(num / (den.abs())).otherwise(fl_none()).alias(name)
    if mode == 3:
        return pl.when(den > 0).then(num / den).otherwise(fl_none()).alias(name)
    if mode == 4:
        cond1 = col("count") > 12
        cond2 = (den.shift(12) > 0).over(["gvkey", "curcd"])
        col_exp = (num / den.shift(12)).over(["gvkey", "curcd"])
        return pl.when(cond1 & cond2).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 5:
        return pl.when(den != 0).then(num * col("fx") / den).otherwise(fl_none()).alias(name)
    if mode == 6:
        cond1 = (den.shift(12) != 0).over(["gvkey", "curcd"])
        col_exp = (num / den.shift(12)).over(["gvkey", "curcd"])
        return pl.when(cond1).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 7:
        cond1 = (den.shift(12) > 0).over(["gvkey", "curcd"])
        col_exp = (num / den).shift(12).over(["gvkey", "curcd"])
        return pl.when(cond1).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 8:
        cond1 = num > 0
        cond2 = den > 0
        return pl.when(cond1 & cond2).then(num / den).otherwise(fl_none()).alias(name)
    if mode == 9:
        cond1 = col("count") > 3
        cond2 = den.shift(3) > 0
        col_exp = num / den.shift(3)
        return pl.when(cond1 & cond2).then(col_exp).otherwise(fl_none()).alias(name)


def update_ni_inc_and_decrease(df, lag):
    """
    Description:
        Update running counts for 8 consecutive NI increases with no decreases.

    Steps:
        1) For rows where ni_inc lagged by `lag` is 1 and no_decrease==1:
        - ni_inc8q += 1
        - no_decrease stays 1; else set to 0.
        2) Apply in gvkey–curcd–date order.

    Output:
        DataFrame with refreshed 'ni_inc8q' and 'no_decrease'.
    """
    c1 = (col("ni_inc").shift(lag) == 1) & (col("no_decrease") == 1)
    ni_inc8q_updated_exp = (
        pl.when(c1).then(col("ni_inc8q") + 1).otherwise(col("ni_inc8q")).alias("ni_inc8q")
    )
    no_decrease_updated_exp = (
        pl.when(c1).then(col("no_decrease")).otherwise(pl.lit(0)).alias("no_decrease")
    )
    return df.sort(["gvkey", "curcd", "datadate"]).with_columns(
        [ni_inc8q_updated_exp, no_decrease_updated_exp]
    )


def calculate_consecutive_earnings_increases(df):
    """
    Description:
        Count 8 consecutive quarterly NI increases (y/y, 12m apart) with no decreases.

    Steps:
        1) Build ni_inc (1/0/NULL) and initialize ni_inc8q=0, no_decrease=1.
        2) Iterate 8 lags (every 3 months): update counters via helper.
        3) Track how many non-null ni_inc across lags; set ni_inc8q only when n==8 & count≥33.
        4) Drop helpers.

    Output:
        DataFrame with 'ni_inc8q' (or NULL when not eligible).
    """
    ni_inc_exp = (
        pl.when(col("ni_x") > col("ni_x").shift(12))
        .then(pl.lit(1).cast(pl.Int64))
        .when((col("ni_x").is_null()) | (col("ni_x").shift(12).is_null()))
        .then(pl.lit(None).cast(pl.Int64))
        .otherwise(pl.lit(0).cast(pl.Int64))
        .alias("ni_inc")
    )
    ni_inc8q_exp = pl.lit(0).alias("ni_inc8q")
    no_decrease_exp = pl.lit(1).alias("no_decrease")
    c1 = (col("ni_inc").is_not_null()) & (col("n_ni_inc") == 8) & (col("count") >= 33)
    ni_inc8q_exp_final = pl.when(c1).then(col("ni_inc8q")).otherwise(pl.lit(None))
    n_ni_inc_exp = col("ni_inc").is_not_null()
    df = df.sort(["gvkey", "curcd", "datadate"]).with_columns(
        [ni_inc_exp, ni_inc8q_exp, no_decrease_exp]
    )
    for i in range(8):
        df = update_ni_inc_and_decrease(df, 3 * i)
        if i > 0:
            n_ni_inc_exp += col("ni_inc").shift(3 * i).is_not_null()
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns(n_ni_inc=n_ni_inc_exp)
        .with_columns(ni_inc8q=ni_inc8q_exp_final)
        .drop(["ni_inc", "no_decrease", "n_ni_inc"])
    )
    return df


def compute_capex_abn(df):
    """
    Description:
        Abnormal capex: current CAPX/Sales vs 3-year trailing average.

    Steps:
        1) Compute __capex_sale = capx/sale_x (den>0).
        2) Denominator = avg of __capex_sale at 12, 24, 36-month lags.
        3) capex_abn = __capex_sale / denom − 1 when count>36 and denom≠0.
        4) Drop helper.

    Output:
        DataFrame with 'capex_abn'.
    """
    c1 = (
        col("__capex_sale").shift(12)
        + col("__capex_sale").shift(24)
        + col("__capex_sale").shift(36)
    ) != 0
    c2 = col("count") > 36
    num = col("__capex_sale")
    den = (
        col("__capex_sale").shift(12)
        + col("__capex_sale").shift(24)
        + col("__capex_sale").shift(36)
    ) / 3
    capex_abn_exp = pl.when(c1 & c2).then(num / den - 1).otherwise(fl_none()).alias("capex_abn")
    df = (
        df.with_columns(safe_div("capx", "sale_x", "__capex_sale", 3))
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(capex_abn_exp)
        .drop("__capex_sale")
    )
    return df


def tangibility():
    """
    Description:
        Asset tangibility index.

    Steps:
        1) Compute (che + 0.715*rect + 0.547*invt + 0.535*ppegt) / at_x when at_x≠0.

    Output:
        Polars expression 'tangibility'.
    """
    c1 = pl.col("at_x") != 0
    div_exp = (col("che") + 0.715 * col("rect") + 0.547 * col("invt") + 0.535 * col("ppegt")) / col(
        "at_x"
    )
    return pl.when(c1).then(div_exp).otherwise(fl_none()).alias("tangibility")


def emp_gr(path):
    """
    Description:
        Employee growth over 12 months (annual panel only).

    Steps:
        1) If quarterly file: return NULL.
        2) Else, when count>12 and avg(emp, emp_lag12)≠0:
        emp_gr1 = (emp − emp_lag12) / avg(emp, emp_lag12).

    Output:
        Polars expression 'emp_gr1'.
    """
    if Path(path).name == "acc_std_qtr.parquet":
        col_expr = fl_none().alias("emp_gr1")
    else:
        c1 = col("count") > 12
        c2 = (col("emp") - col("emp").shift(12)) / (
            0.5 * col("emp") + 0.5 * col("emp").shift(12)
        ) != 0
        c3 = (0.5 * col("emp") + 0.5 * col("emp").shift(12)) != 0
        col_expr = (
            pl.when(c1 & c2 & c3)
            .then(
                (col("emp") - col("emp").shift(12))
                / (0.5 * col("emp") + 0.5 * col("emp").shift(12))
            )
            .otherwise(fl_none())
            .alias("emp_gr1")
        )
    return col_expr


def add_accounting_misc_cols_1(df):
    """
    Description:
        Add a broad set of accounting ratios: growth, asset-scaled changes, margins,
        returns (AT/BE/BEV/PPENT), issuance, solvency, capitalization, accruals, NOA.

    Steps:
        1) Build 1y/3y growth for key levels; 1y/3y Δ scaled by assets.
        2) Add investment & non-recurring intensity; profitability margins.
        3) Compute returns on assets/equity/enterprise/PPENT.
        4) Add issuance & solvency/capitalization ratios; accruals (o/t) and NOA.

    Output:
        DataFrame with appended columns for the above metrics.
    """

    # growth characteristics
    growth_vars = [
        "at_x",
        "ca_x",
        "nca_x",  # Assets - Aggregated
        "lt",
        "cl_x",
        "ncl_x",  # Liabilities - Aggregated
        "be_x",
        "pstk_x",
        "debt_x",  # Financing Book Values
        "sale_x",
        "cogs",
        "xsga",
        "opex_x",  # Sales and Operating Costs
        "capx",
        "invt",
    ]
    ch_asset_vars = [
        "che",
        "invt",
        "rect",
        "ppegt",
        "ivao",
        "ivst",
        "intan",  # Assets - Individual Items
        "dlc",
        "ap",
        "txp",
        "dltt",
        "txditc",  # Liabilities - Individual Items
        "coa_x",
        "col_x",
        "cowc_x",
        "ncoa_x",
        "ncol_x",
        "nncoa_x",
        "oa_x",
        "ol_x",  # Operating Assets/Liabilities
        "fna_x",
        "fnl_x",
        "nfna_x",  # Financial Assets/Liabilities
        "gp_x",
        "ebitda_x",
        "ebit_x",
        "ope_x",
        "ni_x",
        "nix_x",
        "dp",  # Income Statement
        "fincf_x",
        "ocf_x",
        "fcf_x",
        "nwc_x",  # Aggregated Cash Flow
        "eqnetis_x",
        "dltnetis_x",
        "dstnetis_x",
        "dbnetis_x",
        "netis_x",
        "eqnpo_x",  # Financing Cash Flow
        "txt",  # Tax Change
        "eqbb_x",
        "eqis_x",
        "div_x",
        "eqpo_x",  # Financing Cash Flow
        "capx",
        "be_x",
    ]  # Other
    # 1-yr growth,  3-yr growth, 1yr Change Scaled by Assets & 3yr Change Scaled by Assets
    grt1 = [var_growth(i, 12) for i in growth_vars]
    grt3 = [var_growth(i, 36) for i in growth_vars]
    chg_at1 = [chg_to_assets(i, 12) for i in ch_asset_vars]
    chg_at3 = [chg_to_assets(i, 36) for i in ch_asset_vars]
    # Investment Measure & Non-Recurring Items & Profitability margins
    c_at_sale = [
        safe_div("capx", "at_x", "capx_at"),
        safe_div("xrd", "at_x", "rd_at"),
        safe_div("spi", "at_x", "spi_at"),
        safe_div("xido_x", "at_x", "xido_at"),
        pl.when(col("at_x") != 0)
        .then((col("spi") + col("xido_x")) / col("at_x"))
        .otherwise(fl_none())
        .alias("nri_at"),
        safe_div("gp_x", "sale_x", "gp_sale"),
        safe_div("ebitda_x", "sale_x", "ebitda_sale"),
        safe_div("ebit_x", "sale_x", "ebit_sale"),
        safe_div("pi_x", "sale_x", "pi_sale"),
        safe_div("ni_x", "sale_x", "ni_sale"),
        safe_div("ni", "sale_x", "nix_sale"),
        safe_div("ocf_x", "sale_x", "ocf_sale"),
        safe_div("fcf_x", "sale_x", "fcf_sale"),
    ]
    # Return on assets:
    c_ret_at = [
        safe_div(f"{i}_x", "at_x", f"{i}_at") for i in ["gp", "ebitda", "ebit", "fi", "cop", "ni"]
    ]
    # Return on book equity:
    c_ret_be = [safe_div(f"{i}_x", "be_x", f"{i}_be") for i in ["ope", "ni", "nix", "ocf", "fcf"]]
    # Return on invested book capital:
    c_ret_bev = [
        safe_div(f"{i}_x", "bev_x", f"{i}_bev") for i in ["gp", "ebitda", "ebit", "fi", "cop"]
    ]
    # Return on Physical Capital:
    c_ret_ppent = [safe_div(f"{i}_x", "ppent", f"{i}_ppen") for i in ["gp", "ebitda", "fcf"]]
    # Issuance Variables & Equity Payout
    aux_iss_eqp = [
        "fincf",
        "netis",
        "eqnetis",
        "eqis",
        "dbnetis",
        "dltnetis",
        "dstnetis",
        "eqnpo",
        "eqbb",
        "div",
    ]
    c_iss_eqp = [safe_div(f"{i}_x", "at_x", f"{i}_at") for i in aux_iss_eqp]
    # Solvency Ratios: Debt-to-assets, debt to shareholders' equity ratio, interest coverage ratio
    c_solv_rat = [
        safe_div("debt_x", "at_x", "debt_at"),
        safe_div("debt_x", "be_x", "debt_be"),
        safe_div("ebit_x", "xint", "ebit_int"),
    ]
    # Capitalization/Leverage Ratios Book:
    c_cap_lev = [safe_div(f"{i}_x", "bev_x", f"{i}_bev") for i in ["be", "debt", "pstk"]] + [
        safe_div("che", "bev_x", "cash_bev"),
        safe_div("dltt", "bev_x", "debtlt_bev"),
        safe_div("dlc", "bev_x", "debtst_bev"),
    ]
    # Accrual ratios
    c_accruals = [
        safe_div("oacc_x", "at_x", "oaccruals_at"),
        safe_div("tacc_x", "at_x", "taccruals_at"),
        safe_div("oacc_x", "nix_x", "oaccruals_ni", 2),
        safe_div("tacc_x", "nix_x", "taccruals_ni", 2),
    ]
    c_noa_at = [safe_div("noa_x", "at_x", "noa_at", 4)]
    acc_columns = (
        grt1
        + grt3
        + chg_at1
        + chg_at3
        + c_at_sale
        + c_ret_at
        + c_ret_be
        + c_ret_bev
        + c_ret_ppent
        + c_iss_eqp
        + c_solv_rat
        + c_cap_lev
        + c_accruals
        + c_noa_at
    )
    return df.sort(["gvkey", "curcd", "datadate"]).with_columns(acc_columns)


def add_accounting_misc_cols_2(df):
    """
    Description:
        Add volatility features and composite quality metrics; multi-year ratio changes.

    Steps:
        1) Compute volatilities: ocfq/sales, niq/sales, ROE quarterly/annual.
        2) Pipe core composites: earnings variability, equity duration, F-score,
        O-score, Z-score, intrinsic value, KZ index.
        3) 5-year changes for gpoa, roe, roa, cfoa, gmar via Δ(var1/var2, 60m).
        4) Drop temporary/driver columns.

    Output:
        DataFrame with stability and quality-change measures.
    """
    # Volatility items
    funcs_vol = [volq, volq, volq, vola]
    names_col = ["ocfq_saleq_std", "niq_saleq_std", "roeq_be_std", "roe_be_std"]
    vars_vol = ["__ocfq_saleq", "__niq_saleq", "__roeq", "__roe"]
    t1_col = [16, 16, 20, 5]
    t2_col = [8, 8, 12, 5]
    for df_function, n_col, var_vol, t1, t2 in zip(
        funcs_vol, names_col, vars_vol, t1_col, t2_col, strict=True
    ):
        df = df_function(df, n_col, var_vol, t1, t2)
    for df_function in [
        earnings_variability,
        equity_duration_cd,
        pitroski_f,
        ohlson_o,
        altman_z,
        intrinsic_value,
        kz_index,
    ]:
        df = df.pipe(df_function)
    # 5 year ratio change (For quality minus junk variables)
    names = ["gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5"]
    vars1 = ["gp_x", "ni_x", "ni_x", "ocf_x", "gp_x"]
    vars2 = ["at_x", "be_x", "at_x", "at_x", "sale_x"]
    for i, j, k in zip(names, vars1, vars2, strict=True):
        df = df.pipe(chg_var1_to_var2, name=i, var1=j, var2=k, horizon=60)
    return df.drop(["count", "__ocfq_saleq", "__niq_saleq", "__roeq", "__roe"])


def add_liq_and_efficiency_ratios(df):
    """
    Description:
        Liquidity & efficiency ratios and cash conversion cycle.

    Steps:
        1) Liquidity days: inv_days, rec_days, ap_days via 365*avg / flow.
        2) Ratios: cash/cl, caliq/cl, ca/cl; cash_conversion = inv+rec−ap (≥0).
        3) Efficiency: inv_turnover, at_turnover, rec_turnover, ap_turnover.

    Output:
        DataFrame with liquidity and activity metrics.
    """
    # Liquidity Ratios:
    # Days Inventory Outstanding, Days Sales Outstanding, Days Accounts Payable Outstanding
    c_days = [
        temp_liq_rat("invt", "cogs", "inv_days"),
        temp_liq_rat("rect", "sale_x", "rec_days"),
        temp_liq_rat("ap", "cogs", "ap_days"),
    ]
    # Cash, quick, and current ratios; cash Conversion Cycle
    c_liq_rat = [
        safe_div("che", "cl_x", "cash_cl", 3),
        safe_div("caliq_x", "cl_x", "caliq_cl", 3),
        safe_div("ca_x", "cl_x", "ca_cl", 3),
        pl.when((col("inv_days") + col("rec_days") - col("ap_days")) >= 0)
        .then(col("inv_days") + col("rec_days") - col("ap_days"))
        .otherwise(fl_none())
        .alias("cash_conversion"),
    ]
    df = (
        df.sort(["gvkey", "curcd", "datadate"])
        .with_columns(c_days)
        .with_columns(c_liq_rat)
        .sort(["gvkey", "curcd", "datadate"])
        # Activity/Efficiency Ratios:
        .with_columns(
            [
                temp_rat_other("cogs", "invt", "inv_turnover"),
                temp_rat_other("sale_x", "at_x", "at_turnover"),
                temp_rat_other("sale_x", "rect", "rec_turnover"),
                temp_rat_other_spc(),
            ]
        )
    )
    return df


def add_profit_scaled_by_lagged_vars(df):
    """
    Description:
        Profitability scaled by prior-period assets/equity (lag-12).

    Steps:
        1) Compute op_atl1, gp_atl1, ope_bel1, cop_atl1 using safe_div mode 4 (lagged den>0).

    Output:
        DataFrame with lag-scaled profitability ratios.
    """
    df = df.sort(["gvkey", "curcd", "datadate"]).with_columns(
        [
            safe_div("op_x", "at_x", "op_atl1", 4),
            safe_div("gp_x", "at_x", "gp_atl1", 4),
            safe_div("ope_x", "be_x", "ope_bel1", 4),
            safe_div("cop_x", "at_x", "cop_atl1", 4),
        ]
    )
    return df


def add_earnings_persistence_and_expand(paths: DataPaths, df, data_path, lag_to_pub, max_lag):
    """
    Description:
        Attach AR(1) earnings persistence (ni_ar1, ni_ivol) and expand to public dates.

    Steps:
        1) Run persistence job over input parquet (N=5 yrs, min=5) → 'ni_ar_res.parquet'.
        2) Join on (gvkey,curcd,datadate); keep rows with data_available=1.
        3) Set start_date = datadate + lag_to_pub months; end_date = min(next_start−1mo, datadate+max_lag).
        4) Expand monthly between start/end to 'public_date'.

    Output:
        Expanded DataFrame keyed by (gvkey, public_date) with persistence fields.
    """
    compute_earnings_persistence(paths, data_path, 5, 5)
    earnings_pers = pl.scan_parquet(paths.interim_dir / "ni_ar_res.parquet")
    df = (
        df.join(earnings_pers, on=["gvkey", "curcd", "datadate"], how="left")
        .filter(col("data_available") == 1)
        .sort(["gvkey", "datadate"])
        .with_columns(start_date=col("datadate").dt.offset_by(f"{lag_to_pub}mo").dt.month_end())
        .sort(["gvkey", "datadate"])
        .with_columns(next_start_date=col("start_date").shift(-1).over(["gvkey"]))
        .with_columns(
            end_date=pl.min_horizontal(
                (col("next_start_date").dt.offset_by("-1mo").dt.month_end()),
                (col("datadate").dt.offset_by(f"{max_lag}mo").dt.month_end()),
            )
        )
        .drop("next_start_date")
    )
    return expand(
        data=df,
        id_vars=["gvkey"],
        start_date="start_date",
        end_date="end_date",
        freq="month",
        new_date_name="public_date",
    )


def add_me_data_and_compute_me_mev_mat_eqdur_vars(df, me_df):
    """
    Description:
        Join market equity; compute MEV/MAT; scale many vars by ME/MEV; equity duration.

    Steps:
        1) Join me_company at (gvkey, public_date); build mev = me + netdebt*fx; mat = at*fx − be*fx + me.
        2) Clean nonpositive me/mev/mat → NULL.
        3) Add {var}_me and {var}_mev via scaling helpers; ival_me = intrinsic_value/ME (fx-adjusted).
        4) Add misc: enterprise_value, aliq_mat, eq_dur (guard with ed_err, eq_dur>0, me>0).

    Output:
        DataFrame with ME/MEV-scaled features and market-based metrics.
    """
    # Characteristics Scaled by Market Equity
    me_vars = [
        "at_x",
        "be_x",
        "debt_x",
        "netdebt_x",
        "che",
        "sale_x",
        "gp_x",
        "ebitda_x",
        "ebit_x",
        "ope_x",
        "ni_x",
        "nix_x",
        "cop_x",
        "ocf_x",
        "fcf_x",
        "div_x",
        "eqbb_x",
        "eqis_x",
        "eqpo_x",
        "eqnpo_x",
        "eqnetis_x",
        "xrd",
    ]
    # Characteristics Scaled by Market Enterprise Value
    mev_vars = [
        "at_x",
        "bev_x",
        "ppent",
        "be_x",
        "che",
        "sale_x",
        "gp_x",
        "ebitda_x",
        "ebit_x",
        "ope_x",
        "ni_x",
        "nix_x",
        "cop_x",
        "ocf_x",
        "fcf_x",
        "debt_x",
        "pstk_x",
        "dltt",
        "dlc",
        "dltnetis_x",
        "dstnetis_x",
        "dbnetis_x",
        "netis_x",
        "fincf_x",
    ]
    c_misc = [
        col("mev").alias("enterprise_value"),
        (
            pl.when((col("gvkey") == col("gvkey").shift(12)) & (col("mat").shift(12) != 0))
            .then(col("aliq_x") * col("fx") / col("mat").shift(12))
            .otherwise(fl_none())
        ).alias("aliq_mat"),
        (
            (col("ed_cd_w") * col("fx") / col("me_company"))
            + col("ed_constant")
            * (col("me_company") - col("ed_cd") * col("fx"))
            / col("me_company")
        ).alias("eq_dur"),
    ]

    df = (
        df.join(
            me_df,
            left_on=["gvkey", "public_date"],
            right_on=["gvkey", "eom"],
            how="left",
        )
        .select(df.collect_schema().names() + ["me_company"])
        .sort(["gvkey", "public_date"])
        .unique(["gvkey", "public_date"], keep="first")
        .with_columns(
            mev=col("me_company") + col("netdebt_x") * col("fx"),
            mat=col("at_x") * col("fx") - col("be_x") * col("fx") + col("me_company"),
            me_company=pl.when(col("me_company") > 0).then(col("me_company")).otherwise(fl_none()),
        )
        .with_columns(
            mev=pl.when(col("mev") > 0).then(col("mev")).otherwise(fl_none()),
            mat=pl.when(col("mat") > 0).then(col("mat")).otherwise(fl_none()),
        )
        .sort(["gvkey", "public_date"])
        .with_columns(
            [scale_me(i) for i in me_vars]
            + [safe_div("intrinsic_value", "me_company", "ival_me", 5)]
            + [scale_mev(i) for i in mev_vars]
            + c_misc
        )
        .with_columns(
            pl.when((col("ed_err") == 1) | (col("eq_dur") <= 0) | (col("me_company") == 0))
            .then(None)
            .otherwise(col("eq_dur"))
            .alias("eq_dur")
        )
    )
    return df


def rename_cols_and_select_keep_vars(df, rename_dict, vars_to_keep, suffix):
    """
    Description:
        Systematically rename columns, then select and optionally suffix keepers.

    Steps:
        1) For every column, apply first-match replacements from rename_dict.
        2) Select ['source','gvkey','public_date','datadate'] + vars_to_keep.
        3) If suffix provided, append to vars_to_keep names.

    Output:
        DataFrame with renamed and reduced columns.
    """
    new_names = {}
    for i in sorted(df.collect_schema().names()):
        col_name = i
        for a, b in rename_dict.items():
            col_name = col_name.replace(a, b, 1)
        new_names[i] = col_name
    df = df.rename(new_names).select(["source", "gvkey", "public_date", "datadate"] + vars_to_keep)
    if suffix is None:
        return df
    else:
        return df.rename({i: (i + suffix) for i in vars_to_keep})


def convert_raw_vars_to_usd(paths: DataPaths, df):
    """
    Description:
        Convert select raw variables to USD using FX at (curcd, public_date).

    Steps:
        1) Join FX table on [curcd, public_date]→fx.
        2) Multiply assets, sales, book_equity, net_income by fx.
        3) Drop currency code.

    Output:
        DataFrame with key fundamentals in USD.
    """
    fx = compustat_fx(paths).rename({"datadate": "date"}).lazy()
    cols_for_new_df = df.collect_schema().names()
    df = (
        df.join(
            fx,
            left_on=["curcd", "public_date"],
            right_on=["curcdd", "date"],
            how="left",
        )
        .select(cols_for_new_df + ["fx"])
        .with_columns(
            [
                (col(i) * col("fx")).alias(i)
                for i in ["assets", "sales", "book_equity", "net_income"]
            ]
        )
        .drop("curcd")
    )
    return df


def financial_soundness_and_misc_ratios_exps():
    """
    Description:
        Return expression list for financial-soundness and miscellaneous ratios.

    Steps:
        1) Financial soundness: interest/ debt, OCF/ debt, EBITDA/ debt, ST/LT splits,
        profitability per CL, liquidity to LT, composition within ACT, opex/AT,
        NWC/AT, LT/PPENT, debtLT/BE, FCF/OCF (guarded).
        2) Misc: advertising/sales, staff/sales, sales/BEV, R&D/sales, sales/BE,
        sales/NWC (guarded), tax/PI (guarded), cash/AT (guarded), NI/emp, Sales/emp,
        dividend/NI (using NI or NIX > 0).

    Output:
        List of Polars expressions ready for with_columns(...).
    """
    # Financial Soundness Ratios:
    c_fin_s_rat = [
        safe_div("xint", "debt_x", "int_debt"),
        safe_div("ocf_x", "debt_x", "ocf_debt"),
        safe_div("ebitda_x", "debt_x", "ebitda_debt"),
        safe_div("dlc", "debt_x", "debtst_debt"),
        safe_div("dltt", "debt_x", "debtlt_debt"),
        safe_div("xint", "dltt", "int_debtlt"),
        safe_div("ebitda_x", "cl_x", "profit_cl"),
        safe_div("ocf_x", "cl_x", "ocf_cl"),
        safe_div("che", "lt", "cash_lt"),
        safe_div("cl_x", "lt", "cl_lt"),
        safe_div("invt", "act", "inv_act"),
        safe_div("rect", "act", "rec_act"),
        safe_div("opex_x", "at_x", "opex_at"),
        safe_div("nwc_x", "at_x", "nwc_at"),
        safe_div("lt", "ppent", "lt_ppen"),
        safe_div("dltt", "be_x", "debtlt_be"),
        safe_div("fcf_x", "ocf_x", "fcf_ocf", 3),
    ]
    c_misc_rat = [
        safe_div("xad", "sale_x", "adv_sale"),
        safe_div("xlr", "sale_x", "staff_sale"),
        safe_div("sale_x", "bev_x", "sale_bev"),
        safe_div("xrd", "sale_x", "rd_sale"),
        safe_div("sale_x", "be_x", "sale_be"),
        safe_div("sale_x", "nwc_x", "sale_nwc", 3),
        safe_div("txt", "pi_x", "tax_pi", 3),
        safe_div("che", "at_x", "cash_at", 3),
        safe_div("ni_x", "emp", "ni_emp", 3),
        safe_div("sale_x", "emp", "sale_emp", 3),
        pl.when((pl.coalesce("nix_x", "ni_x") > 0.0) & (col("nix_x") != 0))
        .then(col("div_x") / col("nix_x"))
        .otherwise(fl_none())
        .alias("div_ni"),
    ]
    return c_fin_s_rat + c_misc_rat


@measure_time
def create_acc_chars(
    paths: DataPaths,
    data_path,
    output_path,
    lag_to_public,
    max_data_lag,
    __keep_vars,
    me_data_path,
    suffix,
):
    """
    Description:
        Build comprehensive accounting characteristics, align to public dates, convert to USD, join ME, and scale/derive market-based metrics.

    Steps:
        1) Load ME (company) and scan input; add counts and core aliases (assets, sales, book_equity, net_income).
        2) Add accounting ratios/features (misc cols 1), financial soundness/misc ratios, and liquidity/efficiency ratios.
        3) Add sales-per-employee growth and employee growth; compute consecutive NI increases.
        4) Add changes/growth (NOA, PPE+Inv, LNOA, CAPEX 2y); quarterly profitability (saleq_gr1, NIQ/BE, NIQ/AT) and their 1y deltas.
        5) RD capital-to-assets (5y decay); Abarbanell–Bushee changes; standardized surprises for sales/NI; abnormal CAPEX; lagged-profit ratios.
        6) Add core ratios (pi_nix, ocf_at, op_at, at_be, ROE/OCF per quarter); tangibility, ALIQ/AT; OCF_AT 1y change.
        7) Append volatility & composite metrics (misc cols 2); compute earnings persistence and expand to monthly public_date.
        8) Convert key raw vars to USD; join ME; compute ME/MEV/MAT scales and equity duration.
        9) Rename selected columns with mapping, select keep-vars (+ ids), optional suffix, dedupe, and write parquet.

    Output:
        Parquet at output_path with standardized, USD/ME/MEV-scaled accounting characteristics keyed by (gvkey, public_date).
    """

    me_data = load_mkt_equity_data(me_data_path, False)

    chars_df = pl.scan_parquet(data_path)

    chars_df = (
        chars_df.sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            count=col("gvkey").cum_count().over(["gvkey", "curcd"]),
            assets=col("at_x"),
            sales=col("sale_x"),
            book_equity=col("be_x"),
            net_income=col("ni_x"),
        )
        .pipe(add_accounting_misc_cols_1)
        .with_columns(financial_soundness_and_misc_ratios_exps())
        .pipe(add_liq_and_efficiency_ratios)
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            sale_emp_gr1=pl.when((col("count") > 12) & (col("sale_emp").shift(12) > 0))
            .then(col("sale_emp") / col("sale_emp").shift(12) - 1)
            .otherwise(fl_none())
        )
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(emp_gr(data_path))
        .pipe(calculate_consecutive_earnings_increases)
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            [
                chg_to_lagassets(i) for i in ["noa_x", "ppeinv_x"]
            ]  # 1yr Change Scaled by Lagged Assets)
            + [chg_to_avgassets(i) for i in ["lnoa_x"]]  # 1yr Change Scaled by Average Assets
            + [var_growth(var_gr="capx", horizon=24)]
        )  # CAPEX growth over 2 years
        .sort(["gvkey", "curcd", "datadate"])
        # Quarterly profitability measures:
        .with_columns(
            [
                pl.when((col("count") > 12) & (col("sale_qtr").shift(12) > 0))
                .then(col("sale_qtr") / col("sale_qtr").shift(12) - 1)
                .otherwise(fl_none())
                .alias("saleq_gr1"),
                safe_div("ni_qtr", "be_x", "niq_be", 9),
                safe_div("ni_qtr", "at_x", "niq_at", 9),
            ]
        )
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            [
                pl.when(col("count") > 12)
                .then(col("niq_be") - col("niq_be").shift(12))
                .otherwise(fl_none())
                .alias("niq_be_chg1"),
                pl.when(col("count") > 12)
                .then(col("niq_at") - col("niq_at").shift(12))
                .otherwise(fl_none())
                .alias("niq_at_chg1"),
            ]
        )
        .sort(["gvkey", "curcd", "datadate"])
        # R&D capital-to-assets
        .with_columns(
            pl.when((col("count") > 48) & (col("at_x") > 0))
            .then(
                (
                    col("xrd")
                    + col("xrd").shift(12) * 0.8
                    + col("xrd").shift(24) * 0.6
                    + col("xrd").shift(36) * 0.4
                    + col("xrd").shift(48) * 0.2
                )
                / col("at_x")
            )
            .otherwise(fl_none())
            .alias("rd5_at")
        )
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            [chg_to_exp(i) for i in ["sale_x", "invt", "rect", "gp_x", "xsga"]]
        )  # Abarbanell and Bushee (1998)
        .with_columns(
            dsale_dinv=col("sale_ce") - col("invt_ce"),
            dsale_drec=col("sale_ce") - col("rect_ce"),
            dgp_dsale=col("gp_ce") - col("sale_ce"),
            dsale_dsga=col("sale_ce") - col("xsga_ce"),
        )
        .drop(["sale_ce", "invt_ce", "rect_ce", "gp_ce", "xsga_ce"])
        .pipe(standardized_unexpected, var="sale_qtr", qtrs=8, qtrs_min=6)
        .pipe(standardized_unexpected, var="ni_qtr", qtrs=8, qtrs_min=6)
        .pipe(compute_capex_abn)
        .pipe(add_profit_scaled_by_lagged_vars)
        .with_columns(
            pi_nix=safe_div("pi_x", "nix_x", "pi_nix", 8),
            ocf_at=safe_div("ocf_x", "at_x", "ocf_at", 3),
            op_at=safe_div("op_x", "at_x", "op_at", 3),
            at_be=safe_div("at_x", "be_x", "at_be"),
            __ocfq_saleq=safe_div("ocf_qtr", "sale_qtr", "__ocfq_saleq", 3),
            __niq_saleq=safe_div("ni_qtr", "sale_qtr", "__niq_saleq", 3),
            __roeq=safe_div("ni_qtr", "be_x", "__roeq", 3),
            __roe=safe_div("ni_x", "be_x", "__roe", 3),
            tangibility=tangibility(),
            aliq_at=safe_div("aliq_x", "at_x", "aliq_at", 4),
        )
        .sort(["gvkey", "curcd", "datadate"])
        .with_columns(
            ocf_at_chg1=pl.when(col("count") > 12)
            .then(col("ocf_at") - col("ocf_at").shift(12))
            .otherwise(fl_none())
        )
        .pipe(add_accounting_misc_cols_2)
        .pipe(
            functools.partial(add_earnings_persistence_and_expand, paths),
            data_path=data_path,
            lag_to_pub=lag_to_public,
            max_lag=max_data_lag,
        )
        .pipe(functools.partial(convert_raw_vars_to_usd, paths))
        .pipe(add_me_data_and_compute_me_mev_mat_eqdur_vars, me_df=me_data)
    )

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
        "ocf_qtr": "ocfq",
    }

    rename_cols_and_select_keep_vars(chars_df, rename_dict, __keep_vars, suffix).sort(
        ["gvkey", "public_date"]
    ).unique(["gvkey", "public_date"], keep="first").sort(
        ["gvkey", "public_date"]
    ).collect().write_parquet(output_path)


@measure_time
def combine_ann_qtr_chars(paths: DataPaths, ann_df_path, qtr_df_path, char_vars, q_suffix):
    """
    Description:
        Combine annual and quarterly characteristic panels, preferring fresher quarterly values at the same public_date.

    Steps:
        1) Load annual and quarterly files into DuckDB with row numbers.
        2) Left-join on (gvkey, public_date); for each char_var choose quarterly value if present and more recent (datadate_qitem > datadate).
        3) Drop redundant join and dated columns; dedupe on (gvkey, public_date).

    Output:
        Writes 'acc_chars_world.parquet' merged panel.
    """
    (paths.interim_dir / "aux_aqtr_chars.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(str(paths.interim_dir / "aux_aqtr_chars.ddb"), threads=os.cpu_count())
    con.create_table(
        "ann",
        con.read_parquet(ann_df_path).mutate(n1=ibis.row_number()),
        overwrite=True,
    )
    con.create_table(
        "qtr",
        con.read_parquet(qtr_df_path)
        .mutate(n2=ibis.row_number())
        .rename({"datadate_qitem": "datadate", "source_qitem": "source"}),
        overwrite=True,
    )
    ann = con.table("ann")
    qtr = con.table("qtr")
    combined = ann.left_join(qtr, [ann.gvkey == qtr.gvkey, ann.public_date == qtr.public_date])
    drop_columns = [
        "datadate",
        f"datadate{q_suffix}",
        "gvkey_right",
        "public_date_right",
        "source_qitem",
    ] + [f"{ann_var}{q_suffix}" for ann_var in char_vars]
    subs = {}
    for ann_var in char_vars:
        qtr_var = f"{ann_var}{q_suffix}"
        subs[ann_var] = ibis.ifelse(
            combined[ann_var].isnull()
            | (combined[qtr_var].notnull() & (combined[f"datadate{q_suffix}"] > combined.datadate)),
            combined[qtr_var],
            combined[ann_var],
        )
    combined = (
        combined.mutate(subs)
        .drop(drop_columns)
        .order_by(["gvkey", "public_date", "n1", "n2"])
        .distinct(on=["gvkey", "public_date"], keep="first")
        .drop(["n1", "n2"])
        .order_by(["gvkey", "public_date"])
    )
    combined.to_parquet(paths.interim_dir / "acc_chars_world.parquet")
    con.disconnect()
    (paths.interim_dir / "aux_aqtr_chars.ddb").unlink(missing_ok=True)


def seasonality(data, ret_x, start_year, end_year):
    """
    Description:
        Seasonality features: average annual-month and non-annual-month returns over a horizon.

    Steps:
        1) Sum shifted returns over all months in [start_year−1, end_year] and over the annual month series.
        2) For rows with sufficient history, compute seas_{start}_{end}an and seas_{start}_{end}na as means.
        3) Keep NULL otherwise.

    Output:
        DataFrame with two seasonal return columns.
    """
    all_r = pl.lit(0.0)
    ann_r = pl.lit(0.0)
    for i in range((start_year - 1) * 12, (end_year * 12)):
        all_r += col(ret_x).shift(i)
    for i in range((start_year * 12 - 1), (end_year * 12), 12):
        ann_r += col(ret_x).shift(i)
    c1 = col("count") >= (end_year * 12)
    seas_an_exp = ann_r / len(range((start_year * 12 - 1), (end_year * 12), 12))
    seas_na_exp = (all_r - ann_r) / (
        len(range((start_year - 1) * 12, (end_year * 12)))
        - len(range((start_year * 12 - 1), (end_year * 12), 12))
    )
    data = data.sort(["id", "eom"]).with_columns(
        [
            pl.when(c1)
            .then(seas_an_exp)
            .otherwise(fl_none())
            .alias(f"seas_{start_year}_{end_year}an"),
            pl.when(c1)
            .then(seas_na_exp)
            .otherwise(fl_none())
            .alias(f"seas_{start_year}_{end_year}na"),
        ]
    )
    return data


def mom_rev_cols(i, j):
    """
    Description:
        Momentum/reversal feature: return between months j and i using RI (return index).

    Steps:
        1) Require ri_x at lag j > 0, count > j, and ret_x at lag i exists.
        2) Compute ri_x.shift(i)/ri_x.shift(j) − 1.

    Output:
        Polars expression named 'ret_{j}_{i}'.
    """

    c1 = col("ri_x").shift(j) != 0
    c2 = col("count") > j
    c3 = (col("ret_x").shift(i)).is_not_null()
    return (
        pl.when(c1 & c2 & c3)
        .then(col("ri_x").shift(i) / col("ri_x").shift(j) - 1)
        .otherwise(fl_none())
    ).alias(f"ret_{j}_{i}")


def chcsho_cols(i):
    """
    Description:
        Share change over i months from an auxiliary column (e.g., shares outstanding).

    Steps:
        1) Require aux_lag_i ≠ 0 and count > i.
        2) Compute aux/aux.shift(i) − 1.

    Output:
        Polars expression 'chcsho_{i}m'.
    """
    c1 = col("aux").shift(i) != 0
    c2 = col("count") > i
    return (pl.when(c1 & c2).then(col("aux") / col("aux").shift(i) - 1).otherwise(fl_none())).alias(
        f"chcsho_{i}m"
    )


def eqnpo_cols(lag):
    """
    Description:
        Equity net payout over lag months: log growth of RI minus log growth of ME.

    Steps:
        1) Require ri and me positive now and at lag; count > lag.
        2) Compute ln(ri/ri_lag) − ln(me/me_lag).

    Output:
        Polars expression 'eqnpo_{lag}m'.
    """
    c1 = (col("ri") > 0) & (col("ri").shift(lag) > 0)
    c2 = (col("me") > 0) & (col("me").shift(lag) > 0)
    c3 = col("count") > lag
    eqnpo_col_exp = (col("ri") / col("ri").shift(lag)).log() - (
        col("me") / col("me").shift(lag)
    ).log()
    return (pl.when(c1 & c2 & c3).then(eqnpo_col_exp).otherwise(fl_none())).alias(f"eqnpo_{lag}m")


def div_cols(i, spc=False):
    """
    Description:
        Rolling dividend-to-market-equity over i months (optionally special series).

    Steps:
        1) Base div var = 'div' or 'divspc'. Use monthly div1m_me, rolling-summed to horizon i.
        2) Return rolling_sum(div1m_me, i) / me when count ≥ i and me ≠ 0.

    Output:
        Polars expression '{div|divspc}{i}m_me'.
    """

    div_var = "div" if (not spc) else "divspc"
    num = (
        col(f"{div_var}1m_me")
        if (i == 1)
        else col(f"{div_var}1m_me").rolling_sum(window_size=i, min_periods=1).over("id")
    )
    return (
        pl.when((col("count") >= i) & (col("me") != 0)).then(num / col("me")).otherwise(fl_none())
    ).alias(f"{div_var}{i}m_me")


@measure_time
def market_chars_monthly(paths: DataPaths, data_path, market_ret_path, local_currency=False):
    """
    Description:
        Build monthly market characteristics per security: dividends, issuance, momentum/reversal, seasonality.

    Steps:
        1) Read stock panel and join country market returns; pick ret_x (local vs USD).
        2) Create complete monthly range per id; compute cumulative indices (ri, ri_x), counts, and missing-return mask.
        3) Zero-out dividend artifacts near 0; derive:
        - Dividend-to-ME (regular/special) over horizons
        - Equity net payout (eqnpo), share change (chcsho)
        - Momentum/reversal ret_{j}_{i}
        - Seasonality windows via helper.
        4) Select/id-sort final feature set.

    Output:
        Writes 'market_chars_m.parquet' with [id, eom, market_equity, div*, eqnpo*, chcsho*, ret_*, seas_*].
    """
    div_range = [1, 3, 6, 12]  # [1,3,6,12,24,36]
    div_spc_range = [1, 12]
    chcsho_lags = [1, 3, 6, 12]
    eqnpo_lags = [1, 3, 6, 12]
    mom_rev_lags = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 3],
        [0, 6],
        [1, 6],
        [0, 9],
        [1, 9],
        [0, 12],
        [1, 12],
        [7, 12],
        [1, 18],
        [1, 24],
        [12, 24],
        [1, 36],
        [12, 36],
        [12, 48],
        [1, 48],
        [1, 60],
        [12, 60],
        [36, 60],
    ]
    ret_var = "ret_local" if local_currency else "ret"
    market_ret = pl.scan_parquet(market_ret_path)
    data = (
        pl.scan_parquet(data_path)
        .join(market_ret, how="left", on=["excntry", "eom"])
        .with_columns(col(ret_var).alias("ret_x"))
        # No need to compute ret_zero because it's not used in the final output
    )
    __stock_coverage = (
        data.group_by("id")
        .agg(start_date=pl.min("eom"), end_date=pl.max("eom"))
        .sort(["id", "start_date"])
    )
    __full_range = expand(__stock_coverage, ["id"], "start_date", "end_date", "month", "eom")
    data = (
        __full_range.join(data, how="left", on=["id", "eom"])
        .sort(["id", "eom"])
        .with_columns(
            ri=((1 + pl.coalesce(["ret", 0])).cum_prod()).over("id"),
            ri_x=((1 + pl.coalesce(["ret_x", 0])).cum_prod()).over("id"),
            count=(col("id").cum_count()).over("id"),
            ret_miss=pl.when((col("ret_x").is_not_null()) & (col("ret_lag_dif") == 1))
            .then(pl.lit(0))
            .otherwise(pl.lit(1)),
        )
        .with_columns(
            [
                pl.when(col("ret_miss") == 1).then(fl_none()).otherwise(i).alias(i)
                for i in ["ret_x", "ret", "ret_local", "ret_exc", "mkt_vw_exc"]
            ]
        )
        .unique(["id", "eom"])
        .with_columns(
            market_equity=col("me"),
            div1m_me=col("div_tot") * col("shares"),
            divspc1m_me=col("div_spc") * col("shares"),
            aux=col("shares") * col("adjfct"),
        )
        .sort(["id", "eom"])
        .with_columns(
            [div_cols(i, spc=False) for i in div_range]
            + [div_cols(i, spc=True) for i in div_spc_range]
            + [eqnpo_cols(i) for i in eqnpo_lags]
            + [chcsho_cols(i) for i in chcsho_lags]
            + [mom_rev_cols(i, j) for i, j in mom_rev_lags]
        )
    )
    for i in [[1, 1], [2, 5], [6, 10], [11, 15], [16, 20]]:
        data = seasonality(data, "ret_x", i[0], i[1])
    data = (
        data.with_columns(
            [
                pl.when(col(var) < 1e-5).then(0.0).otherwise(col(var)).alias(var)
                for var in data.collect_schema().names()
                if var.startswith("div") and var.endswith("me")
            ]
        )
        .with_columns(
            [
                pl.when(col(var).abs() < 1e-5).then(0.0).otherwise(col(var)).alias(var)
                for var in data.collect_schema().names()
                if var.startswith("eqnpo")
            ]
        )
        .select(
            [
                "id",
                "eom",
                "market_equity",
                col("^div.*me$"),
                col("^eqnpo.*$"),
                col("^chcsho.*$"),
                col(r"^ret_\d+_\d+$"),
                col("^seas.*$"),
            ]
        )
        .sort(["id", "eom"])
    )

    data.collect(streaming=True).write_parquet(paths.interim_dir / "market_chars_m.parquet")


@measure_time
def firm_age(paths: DataPaths, data_path):
    """
    Description:
        Compute firm age in months using earliest of CRSP, Compustat accounting, or Compustat returns dates.

    Steps:
        1) Load identifiers/dates from inputs; get earliest dates per gvkey/permco.
        2) Join earliest sources to each (id, eom); also get first observed eom per id.
        3) Age = months between eom and min(first_obs, first_alt). Write result.

    Output:
        'firm_age.parquet' with [id, eom, age].
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    data = con.read_parquet(data_path).select(["gvkey", "permco", "id", "eom"])
    comp_secm = con.read_parquet(paths.raw_tables_dir / "comp_secm.parquet").select(
        ["gvkey", "datadate"]
    )
    comp_gsecm = (
        con.read_parquet(paths.raw_tables_dir / "comp_g_secd.parquet")
        .filter(_.monthend == 1)
        .select(["gvkey", "datadate"])
    )
    comp_ret_age = (
        comp_secm.union(comp_gsecm)
        .group_by("gvkey")
        .agg(comp_ret_first=_.datadate.min())
        .mutate(
            comp_ret_first=(
                (_.comp_ret_first - ibis.interval(years=1)).year().cast("string") + "-12-31"
            ).cast("date")
        )
    )
    comp_funda = con.read_parquet(paths.raw_tables_dir / "comp_funda.parquet").select(
        ["gvkey", "datadate"]
    )
    comp_gfunda = con.read_parquet(paths.raw_tables_dir / "comp_g_funda.parquet").select(
        ["gvkey", "datadate"]
    )
    comp_acc_age = (
        comp_funda.union(comp_gfunda)
        .group_by("gvkey")
        .agg(comp_acc_first=_.datadate.min())
        .mutate(
            comp_acc_first=(
                (_.comp_acc_first - ibis.interval(years=1)).year().cast("string") + "-12-31"
            ).cast("date")
        )
    )
    crsp_age = (
        con.read_parquet(paths.interim_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet")
        .group_by("permco")
        .agg(crsp_first=_.mthcaldt.min())
    )
    con.create_table("data", data.to_polars())
    con.create_table("comp_ret_age", comp_ret_age.to_polars())
    con.create_table("comp_acc_age", comp_acc_age.to_polars())
    con.create_table("crsp_age", crsp_age.to_polars())
    sql_query = """
                    CREATE TABLE age1 AS
                    SELECT
                        a.id,
                        a.eom,
                        LEAST(b.crsp_first, c.comp_acc_first, d.comp_ret_first) AS first_obs
                    FROM data AS a
                    LEFT JOIN crsp_age AS b ON a.permco = b.permco
                    LEFT JOIN comp_acc_age AS c ON a.gvkey = c.gvkey
                    LEFT JOIN comp_ret_age AS d ON a.gvkey = d.gvkey;

                    CREATE TABLE age2 AS
                    SELECT  *, MIN(eom) OVER (PARTITION BY id) AS first_alt
                    FROM age1;

                    CREATE TABLE age3 AS
                    SELECT
                        id, eom,
                        (DATE_PART('year', eom) - DATE_PART('year', LEAST(first_obs, first_alt))) * 12 +
                        (DATE_PART('month', eom) - DATE_PART('month', LEAST(first_obs, first_alt))) AS age
                    FROM age2
                    ORDER BY id, eom;
    """
    con.raw_sql(sql_query)
    con.table("age3").to_parquet(paths.interim_dir / "firm_age.parquet")
    con.disconnect()


def char_pf_rets():
    """
    Description:
        Helper expressions for Fama–French-style factor composites.

    Steps:
        1) lms = average(high) − average(low) across size buckets.
        2) smb = average(smalls) − average(bigs) across char terciles.

    Output:
        List of Polars expressions: [lms, smb].
    """
    lms = (
        (col("small_high") + col("big_high")) / 2 - (col("small_low") + col("big_low")) / 2
    ).alias("lms")
    smb = (
        (col("small_high") + col("small_mid") + col("small_low")) / 3
        - (col("big_high") + col("big_mid") + col("big_low")) / 3
    ).alias("smb")
    return [lms, smb]


def sort_ff_style(char, min_stocks_bp, min_stocks_pf, date_col, data, sf):
    """
    Description:
        FF-style triple-sort by size and a characteristic within country-month, then compute portfolio returns.

    Steps:
        1) Filter eligible stocks (US vs ex-US rules) with available {char}_l.
        2) Compute country breakpoints (30/70) for {char}_l; assign char_pf ∈ {low,mid,high}.
        3) Form size×char portfolios with ME weights; require min stocks.
        4) Join with returns; aggregate to value-weighted ret_exc per (excntry, size_pf, char_pf, date).
        5) Pivot to columns and derive lms/smb composites.

    Output:
        Tidy DataFrame with per-country portfolio returns and composites for the characteristic.
    """
    # print(f"Executing sort_ff_style for {char}", flush=True)
    c1 = (
        ((col("size_grp_l").is_in(["small", "large", "mega"])) & (col("excntry_l") != "USA"))
        | (((col("crsp_nyse_l") == 1) | (col("comp_exchg_l") == 11)) & (col("excntry_l") == "USA"))
    ) & col(f"{char}_l").is_not_null()
    char_pf_exp = (
        pl.when(col(f"{char}_l") >= col("bp_p70"))
        .then(pl.lit("high"))
        .when(col(f"{char}_l") >= col("bp_p30"))
        .then(pl.lit("mid"))
        .otherwise(pl.lit("low"))
    ).alias("char_pf")
    bp_stocks = data.filter(c1).sql(f"""
                SELECT
                    eom,
                    excntry_l,
                    COUNT(*) AS n,
                    QUANTILE_DISC({char}_l, 0.3) AS bp_p30,
                    QUANTILE_DISC({char}_l, 0.7) AS bp_p70
                FROM self
                GROUP BY excntry_l, eom
                ORDER BY excntry_l, eom
            """)
    data = (
        data.join(bp_stocks, how="left", on=["excntry_l", "eom"])
        .filter(
            (col("n") >= min_stocks_bp) & (col(f"{char}_l").is_not_null()) & (col("size_pf") != "")
        )
        .select(
            ["excntry_l", "id", "eom", "size_pf", "me_l", "be_me_l", char_pf_exp]
        )  # This select doesn't impact performance but it helps in debugging
        .group_by(["excntry_l", "size_pf", "char_pf", "eom"])
        .agg(id=col("id"), w=col("me_l") / pl.sum("me_l"), n=pl.len())
        .filter(col("n") >= min_stocks_pf)
        .drop("n")
        .explode(["id", "w"])
    )
    returns = sf.join(
        data,
        how="inner",
        left_on=["id", "eom", "excntry"],
        right_on=["id", "eom", "excntry_l"],
    )
    returns = (
        returns.with_columns(ret_exc=col("ret_exc") * col("w"))
        .group_by(["excntry", "size_pf", "char_pf", date_col])
        .agg(ret_exc=pl.sum("ret_exc"))
        .with_columns(
            characteristic=pl.lit(char),
            combined_pf=(col("size_pf") + "_" + col("char_pf")),
        )
        .collect()
        .pivot(values="ret_exc", index=["excntry", date_col], on="combined_pf")
        .select(["excntry", date_col, *char_pf_rets()])
        .sort(["excntry", date_col])
    )
    return returns


def prep_data_factor_regs(
    paths: DataPaths,
    data_path,
    factors_path,
    lower: float = 0.001,
    upper: float = 0.999,
):
    """
    Description:
        Prepare monthly panel for factor regressions.

    Steps:
        1) Join msf with merged ap_factors_monthly on (excntry, eom) to pick up
           mktrf, hml, and smb_ff3 (renamed locally to smb_ff). The merged file
           carries per-country factors (US + ROW).
        2) Add integer date (aux_date) and cast id_int.
        3) Winsorize ret_exc by eom at configurable percentile bounds.

    Output:
        DuckDB connection containing table '__msf2' (ready for rolling regs).
    """

    (paths.interim_dir / "aux_factor_regs.ddb").unlink(missing_ok=True)
    con = ibis.duckdb.connect(
        str(paths.interim_dir / "aux_factor_regs.ddb"), threads=os.cpu_count()
    )

    data_path_posix = Path(data_path).as_posix()
    factors_path_posix = Path(factors_path).as_posix()
    con.raw_sql(f"""
    CREATE OR REPLACE VIEW data_msf AS
    SELECT *
    FROM read_parquet('{data_path_posix}');

    CREATE OR REPLACE VIEW fcts AS
    SELECT excntry, eom, mktrf, hml_ff AS hml, smb_ff3 AS smb_ff
    FROM read_parquet('{factors_path_posix}');

    CREATE OR REPLACE TABLE __msf2 AS
    WITH __msf1 AS (
    SELECT
        a.id,
        CAST(a.id AS INTEGER) AS id_int,
        a.eom,
        a.ret_exc,
        a.ret_lag_dif,
        b.mktrf,
        b.hml,
        b.smb_ff,
        CAST(
            (EXTRACT(year FROM a.eom) * 12
             + EXTRACT(month FROM a.eom)
            )
          AS INTEGER) AS aux_date

    FROM data_msf AS a
    LEFT JOIN fcts AS b
      ON a.excntry = b.excntry
     AND a.eom     = b.eom
    WHERE
        a.ret_local <> 0
        AND a.ret_exc   IS NOT NULL
        AND a.ret_lag_dif = 1
        AND b.mktrf    IS NOT NULL
    )
    SELECT
        id, id_int, eom, ret_lag_dif, mktrf, hml, smb_ff, aux_date,
        GREATEST(
            QUANTILE_DISC(ret_exc, {lower}) OVER w,
            LEAST(
                ret_exc,
                QUANTILE_DISC(ret_exc, {upper}) OVER w
            )
        ) AS ret_exc
    FROM __msf1
    WINDOW w AS (PARTITION BY eom);
    """)
    return con


@measure_time
def market_beta(paths: DataPaths, output_path, data_path, factors_path, __n, __min):
    """
    Description:
        Estimate rolling CAPM betas and idiosyncratic vol for each stock.

    Steps:
        1) Prep data via prep_data_factor_regs; load '__msf2' lazily.
        2) Generate staggered window specs; run process_window(..., 'capm') per window.
        3) Map back to ids/dates; select beta_{__n}m and ivol_capm_{__n}m; sort.

    Output:
        Parquet at output_path with [id, eom, beta_{__n}m, ivol_capm_{__n}m].
    """
    con = prep_data_factor_regs(paths, data_path, factors_path)
    base_data = con.table("__msf2").to_polars().lazy()
    df = pl.concat(
        [process_window(base_data, w, "capm", __n, __min) for w in gen_aux_windows(__n)]
    ).collect()
    ids = con.table("__msf2").select(["id", "id_int"]).distinct().to_polars()
    dates = (
        con.table("__msf2")
        .select(["aux_date", "eom"])
        .distinct()
        .to_polars()
        .with_columns(col("aux_date").cast(pl.Int32))
    )
    res = (
        df.with_columns(col("aux_date").cast(pl.Int32))
        .join(ids, how="inner", on="id_int")
        .join(dates, how="inner", on="aux_date")
        .select(
            [
                "id",
                "eom",
                col("^beta.*$").alias(f"beta_{__n}m"),
                col("^ivol.*$").alias(f"ivol_capm_{__n}m"),
            ]
        )
        .sort(["id", "eom"])
    )
    res.write_parquet(output_path)
    con.disconnect()


@measure_time
def residual_momentum(
    paths: DataPaths, output_path, data_path, factors_path, __n, __min, incl, skip
):
    """
    Description:
        Compute residual momentum from FF3 regressions with rolling windows and skip/inclusion rules.

    Steps:
        1) Prep '__msf2'; build window specs; run process_window(..., 'res_mom', __n, __min, incl, skip).
        2) Join back ids/dates and keep resff3_{incl}_{skip}; sort.

    Output:
        Parquet at ``paths.interim_dir / f"{output_path}_{incl}_{skip}.parquet"``
        with [id, eom, resff3_{incl}_{skip}]. Note: ``output_path`` is used
        as a *stem*, not a full path — the parameter name is a holdover from
        the pre-refactor signature.
    """
    con = prep_data_factor_regs(paths, data_path, factors_path)
    base_data = con.table("__msf2").to_polars().lazy()
    df = pl.concat(
        [
            process_window(base_data, w, "res_mom", __n, __min, incl, skip)
            for w in gen_aux_windows(__n)
        ]
    ).collect()
    ids = con.table("__msf2").select(["id", "id_int"]).distinct().to_polars()
    dates = (
        con.table("__msf2")
        .select(["aux_date", "eom"])
        .distinct()
        .to_polars()
        .with_columns(col("aux_date").cast(pl.Int32))
    )
    res = (
        df.with_columns(col("aux_date").cast(pl.Int32))
        .join(ids, how="inner", on="id_int")
        .join(dates, how="inner", on="aux_date")
        .select(["id", "eom", f"resff3_{incl}_{skip}"])
        .sort(["id", "eom"])
    )
    res.write_parquet(paths.interim_dir / f"{output_path}_{incl}_{skip}.parquet")
    con.disconnect()


@measure_time
def prepare_daily(paths: DataPaths, data_path, factors_path):
    """
    Description:
        Build daily dataset: align returns with factors, shrink dtypes, and create helpers.

    Steps:
        1) Join daily stock data with merged ap_factors_daily (mktrf, FF legs
           smb_ff3→smb_ff + hml, HXZ legs me_hxz/ia_hxz/roe_hxz) on
           (excntry, date); filter rows with mktrf.
        2) Create zero_obs flags per (id,eom); cap returns to lag ≤14 days; compute prc_adj.
        3) Write dsf1.parquet and id_int_key.parquet.
        4) Build market lead/lag series per day and write mkt_lead_lag.parquet.
        5) Build 3-day rolling sums for stock and market excess returns; filter to non-null sums and zero_obs<10; write corr_data.parquet.

    Output:
        Parquets: dsf1.parquet, id_int_key.parquet, mkt_lead_lag.parquet, corr_data.parquet.
    """
    data = pl.scan_parquet(data_path)
    fcts = pl.scan_parquet(factors_path).select(
        [
            "excntry",
            "date",
            "mktrf",
            col("smb_ff3").alias("smb_ff"),
            col("hml_ff").alias("hml"),
            "me_hxz",
            "ia_hxz",
            "roe_hxz",
        ]
    )
    dsf1 = (
        data.select(
            [
                "excntry",
                "id",
                "date",
                "eom",
                "prc",
                "adjfct",
                "ret",
                "ret_exc",
                "dolvol",
                "shares",
                "tvol",
                "ret_lag_dif",
                "ret_local",
            ]
        )
        .join(fcts, how="left", on=["excntry", "date"])
        .filter(col("mktrf").is_not_null())
        .with_columns(
            zero_obs=pl.when(col("ret_local") == 0).then(1).otherwise(0),
            id_int=pl.col("id").rank(method="min").cast(pl.Int64),
        )
        .with_columns(
            zero_obs=pl.sum("zero_obs").over(["id_int", "eom"]),
            ret_exc=pl.when(col("ret_lag_dif") <= 14).then(col("ret_exc")).otherwise(fl_none()),
            ret=pl.when(col("ret_lag_dif") <= 14).then(col("ret")).otherwise(fl_none()),
            dolvol_d=col("dolvol"),
            prc_adj=safe_div("prc", "adjfct", "prc_adj"),
        )
        .drop(["ret_lag_dif", "ret_local", "adjfct", "prc", "dolvol"])
        .sort(["id_int", "date"])
        .select(
            pl.all().shrink_dtype()
        )  # For computers without enough memory, use this line to apply dtype shrinking
    )
    dsf1.collect().write_parquet(paths.interim_dir / "dsf1.parquet")

    id_int_key = (
        pl.scan_parquet(paths.interim_dir / "dsf1.parquet").select(["id", "id_int"]).unique()
    )
    id_int_key.collect().write_parquet(paths.interim_dir / "id_int_key.parquet")

    mkt_lead_lag = (
        fcts.select(["excntry", "date", "mktrf", col("date").dt.month_end().alias("eom")])
        .sort(["excntry", "date"])
        .with_columns(
            mktrf_ld1=col("mktrf").shift(-1).over(["excntry", "eom"]),
            mktrf_lg1=col("mktrf").shift(1).over(["excntry"]),
        )
        .select(pl.all().shrink_dtype())
        .sort(["excntry", "date"])
    )
    mkt_lead_lag.collect().write_parquet(paths.interim_dir / "mkt_lead_lag.parquet")

    corr_data = (
        pl.scan_parquet(paths.interim_dir / "dsf1.parquet")
        .select(["ret_exc", "id_int", "date", "mktrf", "eom", "zero_obs"])
        .sort(["id_int", "date"])
        .with_columns(
            ret_exc_3l=(col("ret_exc") + col("ret_exc").shift(1) + col("ret_exc").shift(2)).over(
                ["id_int"]
            ),
            mkt_exc_3l=(col("mktrf") + col("mktrf").shift(1) + col("mktrf").shift(2)).over(
                ["id_int"]
            ),
        )
        .filter(
            col("ret_exc_3l").is_not_null()
            & col("mkt_exc_3l").is_not_null()
            & (col("zero_obs") < 10)
        )
        .select(["id_int", "eom", "ret_exc_3l", "mkt_exc_3l"])
        .select(pl.all().shrink_dtype())
        .sort(["id_int", "eom"])
    )
    corr_data.collect().write_parquet(paths.interim_dir / "corr_data.parquet")


def gen_ranks_and_normalize(df, id_vars, geo_vars, time_vars, desc_flag, var, min_stks):
    """
    Description:
        Rank-normalize a variable within geo×time groups and keep percentile ranks.

    Steps:
        1) Build group keys = geo_vars + time_vars.
        2) Count valid var per group; require count ≥ min_stks.
        3) Compute rank / count (descending if desc_flag); keep id_vars + group keys + rank_{var}.

    Output:
        LazyFrame with percentile column f"rank_{var}" per id and group.
    """

    by_vars = geo_vars + time_vars
    var_ranks = (
        df.select([*id_vars, *by_vars, var])
        .with_columns(count=pl.count(var).over(by_vars))
        .filter(col("count") >= min_stks)
        .with_columns(
            (col(var).rank(descending=desc_flag) / col("count")).over(by_vars).alias(f"rank_{var}")
        )
        .drop([*geo_vars, var, "count"])
    )
    return var_ranks


def gen_misp_exp(var_list, min_fcts):
    """
    Description:
        Combine multiple rank-percentiles into a single mispricing score with missingness guard.

    Steps:
        1) Count NULL ranks across var_list; if > min_fcts → set score NULL.
        2) Else take horizontal mean of rank_{var} columns.

    Output:
        Polars expression for the composite mispricing score.
    """
    sum = col("rank_" + var_list[0]).is_null().cast(pl.Int32)
    for i in var_list[1:]:
        sum += col("rank_" + i).is_null().cast(pl.Int32)
    c1 = sum > min_fcts
    return (
        pl.when(c1)
        .then(fl_none())
        .otherwise(pl.mean_horizontal(["rank_" + f"{i}" for i in var_list]))
    )


def _mp_universe() -> pl.Expr:
    """Universe filter — legacy SHRCD={10,11} + EXCHCD={1,2,3} equivalent in
    CIZ. Stricter than `_ciz_universe()` (extra conditionaltype /
    tradingstatusflg clauses); do not collapse the two."""
    return (
        (col("sharetype") == "NS")
        & (col("securitytype") == "EQTY")
        & (col("securitysubtype") == "COM")
        & (col("usincflg") == "Y")
        & col("issuertype").is_in(["ACOR", "CORP"])
        & col("primaryexch").is_in(["N", "A", "Q"])
        & (col("conditionaltype") == "RW")
        & (col("tradingstatusflg") == "A")
    )


def _mp_offset_months(c: str | pl.Expr, n: int | pl.Expr) -> pl.Expr:
    c = col(c) if isinstance(c, str) else c
    if isinstance(n, int):
        return c.dt.offset_by(f"{n}mo").dt.month_end()
    return c.dt.offset_by(pl.format("{}mo", n)).dt.month_end()


def _mp_offset_qtrs_end(c: str | pl.Expr, n: int) -> pl.Expr:
    c = col(c) if isinstance(c, str) else c
    q = (c.dt.month() - 1) // 3
    target = q + n
    yr = c.dt.year() + (target // 4)
    qm = (target % 4 + 4) % 4
    end_month = (qm + 1) * 3
    return pl.date(yr, end_month, 1).dt.month_end()


def _mp_month_diff(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a.dt.year() - b.dt.year()) * 12 + (a.dt.month() - b.dt.month())


def _mp_polars_sql(df: pl.DataFrame, query: str, table: str = "src") -> pl.DataFrame:
    ctx = pl.SQLContext()
    ctx.register(table, df.lazy())
    return ctx.execute(query).collect()


def _mp_winsorize_per_group(
    df: pl.DataFrame, cols_list: list[str], *, lo: int = 5, hi: int = 95, by: str = "eom"
) -> pl.DataFrame:
    select = [by] + [
        f"quantile_disc({c}, {lo / 100}) AS _{c}_lo, quantile_disc({c}, {hi / 100}) AS _{c}_hi"
        for c in cols_list
    ]
    bounds = _mp_polars_sql(df, f"SELECT {', '.join(select)} FROM src GROUP BY {by}")
    return (
        df.join(bounds, on=by, how="left")
        .with_columns(
            [
                pl.when(col(c) > col(f"_{c}_hi"))
                .then(col(f"_{c}_hi"))
                .when((col(c) < col(f"_{c}_lo")) & col(c).is_not_null())
                .then(col(f"_{c}_lo"))
                .otherwise(col(c))
                .alias(c)
                for c in cols_list
            ]
        )
        .drop([f"_{c}_lo" for c in cols_list] + [f"_{c}_hi" for c in cols_list])
    )


def _mp_truncate_thin(
    port_ret: pl.DataFrame, *, by: str = "eom", freq_col: str = "_freq_", min_n: int = 10
) -> pl.DataFrame:
    bad = port_ret.filter(col(freq_col) < min_n)
    if bad.height == 0:
        return port_ret
    return port_ret.filter(col(by) > bad[by].max())


def _mp_filter_full_buckets(
    port_ret: pl.DataFrame, n_buckets: int, *, by: str | list[str] = "eom"
) -> pl.DataFrame:
    return (
        port_ret.with_columns(_cnt=pl.len().over(by)).filter(col("_cnt") == n_buckets).drop("_cnt")
    )


def _mp_add_leg_score(
    df: pl.DataFrame, cols_list: list[str], score_col: str, num_col: str, *, min_count: int = 3
) -> pl.DataFrame:
    return df.with_columns(
        **{num_col: sum((col(c) != 0).cast(pl.Int64) for c in cols_list)}
    ).with_columns(
        **{
            score_col: pl.when(col(num_col) >= min_count)
            .then(sum(col(c) for c in cols_list) / col(num_col))
            .otherwise(None)
        }
    )


def _mp_build_crsp_monthly(raw_dir: Path, interim_dir: Path) -> pl.DataFrame:
    """Writes crsp_monthly + crsp_monthly_full. Source: jkp's crsp_msf_v2.parquet (CIZ
    convenience join with sec info inline). Dedup keep-first to absorb multi-distribution
    duplicate rows that WRDS msf_v2 produces. Numeric cols cast to Float64 (raw is
    Decimal which would silently change arithmetic in downstream anomaly logic)."""
    start_d, end_d = date(1920, 1, 1), END_DATE
    msf_dedup = (
        pl.read_parquet(raw_dir / "crsp_msf_v2.parquet")
        .filter((col("mthcaldt") >= start_d) & (col("mthcaldt") <= end_d))
        .sort(
            ["permno", "mthcaldt", "mthret", "mthcap"],
            descending=[False, False, True, True],
            nulls_last=True,
        )
        .unique(subset=["permno", "mthcaldt"], keep="first", maintain_order=True)
        .with_columns(
            col("mthprc", "mthret", "mthretx", "mthcap", "mthvol").cast(pl.Float64, strict=False)
        )
    )
    universe = (
        msf_dedup.filter(_mp_universe())
        .rename(
            {
                "mthcaldt": "date",
                "mthprc": "prc",
                "mthret": "ret",
                "mthretx": "retx",
                "mthcap": "cap",
                "mthvol": "vol",
            }
        )
        .with_columns(eom=col("date").dt.month_end())
    )

    exchcd = (
        pl.when(col("primaryexch") == "N")
        .then(1)
        .when(col("primaryexch") == "A")
        .then(2)
        .when(col("primaryexch") == "Q")
        .then(3)
        .otherwise(None)
        .cast(pl.Int64)
    )
    # Pure CIZ v2 precedence (two-branch, no SIZ baggage): mthret IS the canonical
    # monthly return — CRSP already daily-compounds the delisting-day DA-row into it.
    # No delret fallback (legacy SIZ field, redundant under CIZ). No Shumway imputation
    # (DA-row carries realized delist return; no gap to patch).
    screened = universe.filter(
        col("permno").is_not_null()
        & col("prc").is_not_null()
        & col("cap").is_not_null()
        & (col("cap") > 0)
        & col("ret").is_not_null()
        & col("date").is_not_null()
    ).with_columns(me=col("cap"), exchcd=exchcd)

    out = screened.select(
        col("eom"),
        col("permno").cast(pl.Int64),
        col("prc").alias("PRC"),
        col("ret").alias("RET"),
        col("me"),
    ).sort("eom", "permno")
    out.write_parquet(interim_dir / "crsp_monthly.parquet")

    full = (
        screened.with_columns(shrcd=pl.lit(10).cast(pl.Int64))
        .select(
            col("eom"),
            col("permno").cast(pl.Int64),
            col("prc").alias("PRC"),
            col("ret").alias("RET"),
            col("me"),
            "shrcd",
            "exchcd",
            "siccd",
        )
        .sort("eom", "permno")
    )
    full.write_parquet(interim_dir / "crsp_monthly_full.parquet")
    return out


def _mp_build_crsp_daily(raw_dir: Path, interim_dir: Path) -> pl.DataFrame:
    """Writes crsp_daily. Source: jkp's crsp_dsf_v2.parquet. Cast numerics to Float64."""
    start_d, end_d = date(1920, 1, 1), END_DATE
    dsf = pl.read_parquet(raw_dir / "crsp_dsf_v2.parquet").with_columns(
        col("dlyprc", "dlyret", "dlyretx", "dlycap", "dlyvol").cast(pl.Float64, strict=False)
    )
    df = (
        dsf.filter(
            (col("dlycaldt") >= start_d)
            & (col("dlycaldt") <= end_d)
            & _mp_universe()
            & col("dlyret").is_not_null()
        )
        .rename({"dlycaldt": "date", "dlyret": "ret"})
        .with_columns(
            col("ret").cast(pl.Float64),
            eom=col("date").dt.month_end(),
        )
        .select("date", col("permno").cast(pl.Int64), "ret", "eom")
        .sort("permno", "date")
    )
    df.write_parquet(interim_dir / "crsp_daily.parquet")
    return df


def _mp_load_ccm(raw_dir: Path) -> pl.DataFrame:
    """Load CCM link table (ccmxpf_linktable preferred — has usedflag for tiebreak;
    falls back to ccmxpf_lnkhist with synthesized usedflag=1 if linktable absent).
    8-link-type filter matches MisprProject SY-parity baseline."""
    lt_path = raw_dir / "crsp_ccmxpf_linktable.parquet"
    src_path = lt_path if lt_path.exists() else raw_dir / "crsp_ccmxpf_lnkhist.parquet"
    df = pl.read_parquet(src_path).rename(str.lower)
    if "usedflag" not in df.columns:
        df = df.with_columns(usedflag=pl.lit(1).cast(pl.Int64))
    return df.filter(
        col("linktype").is_in(MP_CCM_LINKTYPES)
        & (col("linkenddt").is_null() | (col("linkenddt").dt.year() >= 1960))
    )


def _mp_ccm_compustat_join(
    fa: pl.DataFrame, lnk: pl.DataFrame, mp_con: duckdb.DuckDBPyConnection
) -> pl.DataFrame:
    mp_con.register("fa", fa)
    mp_con.register("lnk", lnk)
    sql = f"""
    SELECT CAST(a.lpermno AS BIGINT) AS permno, a.linkprim, a.linktype, a.usedflag,
           a.linkdt, a.linkenddt, b.*
    FROM lnk a
    JOIN fa b
      ON a.gvkey = b.gvkey
     AND a.linktype IN ({",".join(repr(x) for x in MP_CCM_LINKTYPES)})
     AND a.linkprim IN ('P','C')
     AND (a.linkdt IS NULL OR a.linkdt <= b.datadate)
     AND (a.linkenddt IS NULL OR b.datadate <= a.linkenddt)
    WHERE b.indfmt='INDL' AND b.datafmt='STD' AND b.popsrc='D' AND b.consol='C'
    """
    return mp_con.execute(sql).pl()


def _mp_ccm_tiebreak_dedup(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            _lp=pl.when(col("linkprim") == "P").then(0).otherwise(1),
            _used=pl.when(col("usedflag") == 1).then(0).otherwise(1),
            # CCM link-type quality rank: lower = preferred. SAS tiebreak ordering.
            _lt=col("linktype").replace_strict(
                {"LU": 0, "LC": 1, "LD": 2, "LN": 3, "LX": 4, "LF": 5, "LO": 6, "LS": 7},
                default=99,
            ),
            _span=(
                col("linkenddt").fill_null(date(9999, 12, 31))
                - col("linkdt").fill_null(date(1900, 1, 1))
            ).dt.total_days(),
        )
        .sort(
            ["datadate", "permno", "_lp", "_used", "_lt", "_span", "linkdt", "linkenddt", "gvkey"],
            descending=[False, False, False, False, False, True, False, False, False],
            nulls_last=True,
        )
        .unique(subset=["datadate", "permno"], keep="first", maintain_order=True)
        .drop("_lp", "_used", "_lt", "_span", "linktype", "usedflag", "linkdt", "linkenddt")
    )


@functools.cache
def _mp_load_funda(raw_dir: Path) -> pl.DataFrame:
    numeric = [
        "act", "at", "ceq", "che", "cogs", "csho", "dlc", "dltt", "dp", "ib",
        "invt", "lct", "lt", "mib", "oancf", "ppegt", "pstk", "sale", "seq",
        "txditc", "xint", "prcc_f", "adjex_c", "revt", "ni", "pi", "txp",
    ]  # fmt: skip
    df = pl.read_parquet(raw_dir / "comp_funda.parquet").filter(_compustat_na_filter())
    return df.with_columns(
        [col(c).cast(pl.Float64, strict=False) for c in numeric if c in df.columns]
    )


@functools.cache
def _mp_load_fundq(raw_dir: Path) -> pl.DataFrame:
    numeric = [
        "ibq", "saleq", "niq", "cheq", "seqq", "ceqq", "pstkq", "atq",
        "ltq", "pstkrq", "txditcq", "actq", "lctq", "piq", "dlttq", "revtq",
    ]  # fmt: skip
    df = pl.read_parquet(raw_dir / "comp_fundq.parquet").filter(_compustat_na_filter())
    return df.with_columns(
        [col(c).cast(pl.Float64, strict=False) for c in numeric if c in df.columns]
    )


def _mp_build_compustat_annual(
    mp_con: duckdb.DuckDBPyConnection, raw_dir: Path, interim_dir: Path
) -> pl.DataFrame:
    fa_cols = [
        "gvkey",
        "datadate",
        "fyear",
        "fyr",
        "indfmt",
        "datafmt",
        "popsrc",
        "consol",
        "act",
        "at",
        "ceq",
        "che",
        "cogs",
        "csho",
        "dlc",
        "dltt",
        "dp",
        "ib",
        "invt",
        "lct",
        "lt",
        "mib",
        "oancf",
        "ppegt",
        "pstk",
        "sale",
        "seq",
        "txditc",
        "xint",
        "prcc_f",
    ]
    funda = _mp_load_funda(raw_dir)
    fa = funda.select([c for c in fa_cols if c in funda.columns])
    df = (
        _mp_ccm_tiebreak_dedup(_mp_ccm_compustat_join(fa, _mp_load_ccm(raw_dir), mp_con))
        .sort(["permno", "fyear", "datadate"])
        .unique(subset=["permno", "fyear"], keep="last", maintain_order=True)
        .sort(["permno", "datadate"])
        .unique(subset=["permno", "datadate"], keep="first", maintain_order=True)
        .select(
            "permno",
            "gvkey",
            "datadate",
            "fyear",
            "fyr",
            "act",
            "at",
            "ceq",
            "che",
            "cogs",
            "csho",
            "dlc",
            "dltt",
            "dp",
            "ib",
            "invt",
            "lct",
            "lt",
            "mib",
            "oancf",
            "ppegt",
            "pstk",
            "sale",
            "seq",
            "txditc",
            "xint",
            "prcc_f",
        )
    )
    df.write_parquet(interim_dir / "compustat_annual.parquet")
    return df


def _mp_build_compustat_quarterly(
    mp_con: duckdb.DuckDBPyConnection, raw_dir: Path, interim_dir: Path
) -> pl.DataFrame:
    fq_cols = [
        "gvkey",
        "datadate",
        "fyearq",
        "fqtr",
        "fyr",
        "rdq",
        "indfmt",
        "datafmt",
        "popsrc",
        "consol",
        "ibq",
        "saleq",
        "dlttq",
        "revtq",
        "actq",
        "lctq",
        "piq",
        "niq",
        "cheq",
        "seqq",
        "ceqq",
        "pstkq",
        "atq",
        "ltq",
        "pstkrq",
    ]
    fundq = _mp_load_fundq(raw_dir)
    fq = fundq.select([c for c in fq_cols if c in fundq.columns])
    joined = _mp_ccm_compustat_join(fq, _mp_load_ccm(raw_dir), mp_con).filter(
        col("permno").is_not_null() & col("datadate").is_not_null()
    )
    df = (
        _mp_ccm_tiebreak_dedup(joined)
        .sort(["permno", "fyearq", "fqtr", "datadate"])
        .unique(subset=["permno", "fyearq", "fqtr"], keep="last", maintain_order=True)
        # Match SAS v1: after last.fqtr dedup, `proc sort by permno datadate nodupkey`
        # preserves previous (fyearq, fqtr) ascending order → keeps SMALLEST fyearq+fqtr
        # on (permno, datadate) ties (typically only matters on fiscal-year-change rows).
        .sort(["permno", "datadate", "fyearq", "fqtr"])
        .unique(subset=["permno", "datadate"], keep="first", maintain_order=True)
        .with_columns(adj_rdq=pl.coalesce(col("rdq"), _mp_offset_months("datadate", 3)))
        .filter(col("adj_rdq").is_not_null())
        .sort(["permno", "adj_rdq", "datadate"])
        .unique(subset=["permno", "adj_rdq", "datadate"], keep="first", maintain_order=True)
        .with_columns(rdq_crsp=_mp_offset_months("adj_rdq", 1))
        .sort(["permno", "rdq_crsp", "datadate"], descending=[False, True, True])
        .with_columns(
            lead_rdq_crsp=pl.coalesce(
                col("rdq_crsp").shift(1).over("permno"),
                _mp_offset_qtrs_end("rdq_crsp", 1),
            )
        )
        .sort(["permno", "adj_rdq", "datadate"])
        .select(
            "permno",
            "gvkey",
            "datadate",
            "fyearq",
            "fyr",
            "rdq",
            "fqtr",
            "ibq",
            "saleq",
            "niq",
            "cheq",
            "seqq",
            "ceqq",
            "pstkq",
            "atq",
            "ltq",
            "pstkrq",
            "rdq_crsp",
            "lead_rdq_crsp",
        )
    )
    df.write_parquet(interim_dir / "compustat_quarterly.parquet")
    return df


def _mp_expand_to_crsp_window(
    anomaly_df: pl.DataFrame, value_col: str, interim_dir: Path
) -> pl.DataFrame:
    a = (
        anomaly_df.select("permno", "datadate", value_col)
        .with_columns(_off=pl.int_ranges(MP_LAG_M + 1, 13 + MP_LAG_M))
        .explode("_off")
        .with_columns(
            eom=col("datadate").dt.offset_by(pl.format("{}mo", col("_off"))).dt.month_end()
        )
        .select("permno", "eom", "datadate", value_col)
    )
    month_keys = pl.read_parquet(interim_dir / "crsp_monthly.parquet").select("permno", "eom")
    return (
        month_keys.join(a, on=["permno", "eom"], how="left")
        .sort(["permno", "eom", "datadate"], descending=[False, False, True])
        .unique(subset=["permno", "eom"], keep="first", maintain_order=True)
        .select("eom", "permno", value_col)
        .sort("permno", "eom")
    )


def _mp_rolling_calendar_sum(
    df: pl.DataFrame,
    value_col: str,
    *,
    lag_min: int,
    lag_max: int,
    n_required: int,
    id_col: str = "permno",
) -> pl.DataFrame:
    """Calendar-aware rolling sum of `value_col` over the inclusive window
    [eom - lag_max months, eom - lag_min months], per `id_col`.

    Returns [id_col, eom, r] keyed by the same id+eom as `df`. `r` is null
    when fewer than `n_required` non-null source values fall in the window.

    Window is computed on an integer month index (year*12 + month) — using
    polars' date `offset_by` here would silently misalign because it is
    day-preserving (e.g. 2019-02-28 - 2mo = 2018-12-28, which excludes the
    2018-12-31 row that ought to be in the window).

    Returning a keyed frame (not a positional `pl.Series`) lets callers
    join by key, so polars' parallel hash join can permute row order
    freely without misaligning the rolling sum onto the wrong rows.
    """
    indexed = df.with_columns(_midx=col("eom").dt.year() * 12 + col("eom").dt.month()).sort(
        id_col, "_midx"
    )
    rolled = indexed.rolling(
        index_column="_midx",
        group_by=id_col,
        period=f"{lag_max - lag_min}i",
        offset=f"-{lag_max}i",
        closed="both",
    ).agg(
        _sum=col(value_col).sum(),
        _cnt=col(value_col).is_not_null().sum().cast(pl.Int64),
    )
    return (
        indexed.select(id_col, "eom", "_midx")
        .join(rolled, on=[id_col, "_midx"], how="left")
        .select(
            id_col,
            "eom",
            pl.when(col("_cnt") == n_required).then(col("_sum")).otherwise(None).alias("r"),
        )
    )


def _mp_compute_accrual(raw_dir: Path, interim_dir: Path) -> pl.DataFrame:
    ca = (
        pl.read_parquet(interim_dir / "compustat_annual.parquet")
        .filter(
            col("permno").is_not_null()
            & col("fyear").is_not_null()
            & col("fyr").is_not_null()
            & (col("fyr") > 0)
            & (col("fyear") > 0)
            & col("at").is_not_null()
            & (col("at") > 0)
            & col("act").is_not_null()
            & col("che").is_not_null()
            & col("lct").is_not_null()
            & col("dlc").is_not_null()
            & col("dp").is_not_null()
        )
        .select("permno", "gvkey", "datadate", "act", "at", "che", "lct", "dlc", "dp")
        .join(
            _mp_load_funda(raw_dir).select("gvkey", "datadate", "txp"),
            on=["gvkey", "datadate"],
            how="left",
        )
        .filter(col("txp").is_not_null())
        .sort(["permno", "datadate", "gvkey"])
        .unique(subset=["permno", "datadate"], keep="first", maintain_order=True)
        .with_columns(
            col(c).shift(1).over("permno").alias(f"lag_{c}")
            for c in ("act", "che", "lct", "dlc", "txp", "at")
        )
        .with_columns(
            accrual_adj=(
                2
                * (
                    (col("act") - col("lag_act"))
                    - (col("che") - col("lag_che"))
                    - (col("lct") - col("lag_lct"))
                    + (col("dlc") - col("lag_dlc"))
                    + (col("txp") - col("lag_txp"))
                    - col("dp")
                )
                / (col("at") + col("lag_at"))
            )
        )
        .select("permno", "datadate", "accrual_adj")
    )
    out = _mp_expand_to_crsp_window(ca, "accrual_adj", interim_dir)
    out.write_parquet(interim_dir / "accrual_adj.parquet")
    return out


def _mp_compute_gross_profit(raw_dir: Path, interim_dir: Path) -> pl.DataFrame:
    ca = (
        pl.read_parquet(interim_dir / "compustat_annual.parquet")
        .select("permno", "gvkey", "datadate", "fyear", "fyr", "at", "cogs")
        .join(
            _mp_load_funda(raw_dir).select("gvkey", "datadate", "revt"),
            on=["gvkey", "datadate"],
            how="left",
        )
        .filter(
            col("permno").is_not_null()
            & col("fyear").is_not_null()
            & col("fyr").is_not_null()
            & (col("fyr") > 0)
            & (col("fyear") > 0)
            & col("at").is_not_null()
            & (col("at") > 0)
            & col("revt").is_not_null()
            & col("cogs").is_not_null()
        )
        .with_columns(gp_adj=(col("revt") - col("cogs")) / col("at"))
        .filter(col("gp_adj").is_not_null())
        .select("permno", "datadate", "gp_adj")
        .sort(["permno", "datadate", "gp_adj"], nulls_last=True)
        .unique(subset=["permno", "datadate"], keep="first", maintain_order=True)
    )
    out = _mp_expand_to_crsp_window(ca, "gp_adj", interim_dir)
    out.write_parquet(interim_dir / "gp_adj.parquet")
    return out


def _mp_compute_oscore(raw_dir: Path, interim_dir: Path) -> pl.DataFrame:
    base = (
        pl.read_parquet(interim_dir / "compustat_annual.parquet")
        .select("permno", "gvkey", "datadate", "fyear", "fyr", "at", "lt", "act", "lct")
        .join(
            _mp_load_funda(raw_dir).select("gvkey", "datadate", "ni", "pi"),
            on=["gvkey", "datadate"],
            how="left",
        )
        .filter(
            col("permno").is_not_null()
            & col("fyear").is_not_null()
            & col("fyr").is_not_null()
            & (col("fyr") > 0)
            & (col("fyear") > 0)
            & col("at").is_not_null()
            & (col("at") > 0)
            & col("lt").is_not_null()
            & (col("lt") != 0)
            & col("act").is_not_null()
            & (col("act") != 0)
        )
        .with_columns(
            cyear=pl.when((col("fyr") > 0) & (col("fyr") <= 5))
            .then(col("fyear") + 1)
            .otherwise(col("fyear")),
            oeneg=pl.when(col("lt") > col("at")).then(1).otherwise(0),
            tlta=safe_div("lt", "at", "tlta"),
            wcta=(col("act") - col("lct")) / col("at"),
            clca=safe_div("lct", "act", "clca"),
            nita=safe_div("ni", "at", "nita"),
            futl=safe_div("pi", "lt", "futl"),
        )
    )

    self_lag = base.select("permno", "cyear", col("ni").alias("lagni")).with_columns(
        cyear=col("cyear") + 1
    )
    base = base.join(self_lag, on=["permno", "cyear"], how="inner").with_columns(
        chin=(col("ni") - col("lagni")) / (col("ni").abs() + col("lagni").abs()),
        intwo=pl.when((col("ni") < 0) & (col("lagni") < 0)).then(1).otherwise(0),
    )

    out = (
        base.join(
            pl.read_parquet(raw_dir / "crsp_a_indexes_acti.parquet")
            .with_columns(col("cpiind").cast(pl.Float64, strict=False))
            .filter(col("caldt").dt.month() == 12)
            .with_columns(cyear=col("caldt").dt.year() + 1)
            .select("cyear", "cpiind"),
            on="cyear",
            how="inner",
        )
        .with_columns(size=(100 * col("at") / col("cpiind")).log())
        .with_columns(
            oscore=-1.32
            - 0.407 * col("size")
            + 6.03 * col("tlta")
            - 1.43 * col("wcta")
            + 0.076 * col("clca")
            - 1.72 * col("oeneg")
            - 2.37 * col("nita")
            - 1.83 * col("futl")
            + 0.285 * col("intwo")
            - 0.521 * col("chin")
        )
        .filter(col("oscore").is_not_null())
        .select("permno", "datadate", "cyear", "oscore")
        .sort(["permno", "cyear", "datadate"], descending=[False, False, True])
    )
    out = _mp_expand_to_crsp_window(out, "oscore", interim_dir)
    out.write_parquet(interim_dir / "oscore.parquet")
    return out


def _mp_compute_nsi_ag_inv_noa(raw_dir: Path, interim_dir: Path) -> None:
    base = (
        pl.read_parquet(interim_dir / "compustat_annual.parquet")
        .filter(
            col("permno").is_not_null()
            & col("fyear").is_not_null()
            & col("fyr").is_not_null()
            & (col("fyr") > 0)
            & (col("fyear") > 0)
        )
        .select(
            "permno",
            "gvkey",
            "datadate",
            "csho",
            "ppegt",
            "invt",
            "at",
            "che",
            "dlc",
            "dltt",
            "mib",
            "pstk",
            "ceq",
        )
        .join(
            _mp_load_funda(raw_dir).select("gvkey", "datadate", "adjex_c"),
            on=["gvkey", "datadate"],
            how="left",
        )
        .sort(["permno", "datadate", "gvkey"])
        .unique(subset=["permno", "datadate"], keep="first", maintain_order=True)
        .with_columns(
            lag_csho=col("csho").shift(1).over("permno"),
            lag_adjexc=col("adjex_c").shift(1).over("permno"),
            lag_ppegt=col("ppegt").shift(1).over("permno"),
            lag_invt=col("invt").shift(1).over("permno"),
            lag_at=col("at").shift(1).over("permno"),
        )
        .with_columns(col("mib").fill_null(0.0), col("pstk").fill_null(0.0))
        .with_columns(
            nsi=pl.when(
                col("csho").is_null()
                | (col("csho") == 0)
                | col("adjex_c").is_null()
                | (col("adjex_c") == 0)
                | (col("lag_csho") == 0)
                | (col("lag_adjexc") == 0)
            )
            .then(0.0)
            .when(col("lag_csho").is_null() | col("lag_adjexc").is_null())
            .then(None)
            .otherwise(
                (col("csho") * col("adjex_c") / (col("lag_csho") * col("lag_adjexc"))).log()
            ),
            inv=pl.when(
                col("lag_ppegt").is_null()
                | col("lag_invt").is_null()
                | col("lag_at").is_null()
                | (col("lag_at") == 0)
            )
            .then(None)
            .otherwise(
                safe_div(
                    col("ppegt") - col("lag_ppegt") + col("invt") - col("lag_invt"),
                    col("lag_at"),
                    "inv",
                )
            ),
            ag=pl.when(col("lag_at") > 0)
            .then(safe_div(col("at") - col("lag_at"), col("lag_at"), "ag"))
            .otherwise(None),
            noa=pl.when(col("lag_at") > 0)
            .then(
                safe_div(
                    (col("at") - col("che"))
                    - (
                        col("at") - col("dlc") - col("dltt") - col("mib") - col("pstk") - col("ceq")
                    ),
                    col("lag_at"),
                    "noa",
                )
            )
            .otherwise(None),
        )
    )

    for src_col, name in [
        ("nsi", "stock_issue"),
        ("ag", "asset_growth"),
        ("inv", "invasset"),
        ("noa", "noa"),
    ]:
        slice_df = base.select("permno", "datadate", col(src_col).alias(name))
        _mp_expand_to_crsp_window(slice_df, name, interim_dir).write_parquet(
            interim_dir / f"{name}.parquet"
        )


def _mp_compute_momentum(interim_dir: Path) -> pl.DataFrame:
    log_rets = (
        pl.read_parquet(interim_dir / "crsp_monthly.parquet")
        .select("permno", "eom", "RET")
        .with_columns(log_1p_ret=(1 + col("RET")).log())
    )
    roll = _mp_rolling_calendar_sum(log_rets, "log_1p_ret", lag_min=2, lag_max=12, n_required=11)
    out = (
        log_rets.select("permno", "eom")
        .join(roll, on=["permno", "eom"], how="left")
        .with_columns(momentum=col("r").exp() - 1)
        .select("eom", "permno", "momentum")
        .sort("permno", "eom")
    )
    out.write_parquet(interim_dir / "momentum.parquet")
    return out


def _mp_compute_composite_issue(interim_dir: Path) -> pl.DataFrame:
    log_rets = (
        pl.read_parquet(interim_dir / "crsp_monthly.parquet")
        .select("permno", "eom", "me", "RET")
        .with_columns(log_1p_ret=(1 + col("RET")).log())
    )
    cumret = _mp_rolling_calendar_sum(log_rets, "log_1p_ret", lag_min=0, lag_max=11, n_required=12)
    me_lag12 = log_rets.select("permno", "eom", col("me").alias("me_lag12")).with_columns(
        eom=_mp_offset_months("eom", 12)
    )
    composite_t = (
        log_rets.select("permno", "eom", "me")
        .join(me_lag12, on=["permno", "eom"], how="left")
        .join(cumret, on=["permno", "eom"], how="left")
        .select(
            "permno",
            col("eom").alias("eom_src"),
            composite_issue=(col("me") / col("me_lag12")).log() - col("r"),
        )
    )
    out = (
        log_rets.select("permno", "eom")
        .with_columns(eom_src=_mp_offset_months("eom", -5))
        .join(composite_t, on=["permno", "eom_src"], how="left")
        .select("eom", "permno", "composite_issue")
        .sort("permno", "eom")
    )
    out.write_parquet(interim_dir / "composite_issue.parquet")
    return out


def _mp_compute_roa(mp_con: duckdb.DuckDBPyConnection, interim_dir: Path) -> pl.DataFrame:
    cq = (
        pl.read_parquet(interim_dir / "compustat_quarterly.parquet")
        .select("permno", "datadate", "rdq", "rdq_crsp", "lead_rdq_crsp", "ibq", "atq")
        .sort("permno", "datadate")
        .with_columns(
            lag_dt=col("datadate").shift(1).over("permno"),
            lag_atq=col("atq").shift(1).over("permno"),
        )
        .with_columns(lag_dt=col("lag_dt").fill_null(col("datadate")))
        .with_columns(
            gap_mo=_mp_month_diff(col("datadate"), col("lag_dt")),
            roa=pl.when(col("lag_atq").is_null() | (col("lag_atq") == 0))
            .then(None)
            .otherwise(safe_div("ibq", "lag_atq", "roa")),
        )
        .with_columns(roa=pl.when(col("gap_mo") >= 6).then(None).otherwise(col("roa")))
        .filter(col("roa").is_not_null())
        .select("permno", "datadate", "rdq", "rdq_crsp", "lead_rdq_crsp", "roa")
    )

    month_keys = pl.read_parquet(interim_dir / "crsp_monthly.parquet").select("permno", "eom")
    mp_con.register("cq", cq)
    mp_con.register("month_keys", month_keys)
    df = mp_con.execute("""
        SELECT a.permno, a.eom, b.roa, b.datadate, b.rdq
        FROM month_keys a
        LEFT JOIN cq b
          ON a.permno = b.permno AND a.eom >= b.rdq_crsp AND a.eom < b.lead_rdq_crsp
    """).pl()

    out = (
        df.sort(["permno", "eom", "datadate", "rdq"], descending=[False, False, True, True])
        .unique(subset=["permno", "eom"], keep="first", maintain_order=True)
        .select("eom", "permno", "roa")
        .sort("permno", "eom")
    )
    out.write_parquet(interim_dir / "roa.parquet")
    return out


def _mp_distress_market_inputs(crsp_full: pl.DataFrame, sp500_idx: pl.DataFrame) -> pl.DataFrame:
    return (
        crsp_full.with_columns(prc=col("PRC").abs().clip(upper_bound=15.0))
        .join(sp500_idx, on="eom", how="left")
        .with_columns(
            PRICE=col("prc").log(),
            EXRET=(1 + col("RET")).log() - (1 + col("sprtrn")).log(),
            RSIZE=(col("me") / col("totval")).log(),
        )
        .sort("permno", "eom")
        .with_columns(
            form_eom=col("eom").shift(1).over("permno"),
            lag_ME=col("me").shift(1).over("permno"),
            lag_PRICE=col("PRICE").shift(1).over("permno"),
            lag_EXRET=col("EXRET").shift(1).over("permno"),
            lag_RSIZE=col("RSIZE").shift(1).over("permno"),
        )
        .with_columns(gap=_mp_month_diff(col("eom"), col("form_eom")))
        .with_columns(
            [
                pl.when(col("gap") > 1).then(None).otherwise(col(c)).alias(c)
                for c in ["lag_ME", "lag_PRICE", "lag_EXRET", "lag_RSIZE"]
            ]
        )
        .select(
            "eom",
            "permno",
            col("lag_ME").alias("ME"),
            col("lag_PRICE").alias("PRICE"),
            col("lag_EXRET").alias("EXRET"),
            col("lag_RSIZE").alias("RSIZE"),
        )
    )


def _mp_distress_book_equity_mb(cq_base: pl.DataFrame, crsp_full: pl.DataFrame) -> pl.DataFrame:
    return (
        cq_base.with_columns(pre_stock=pl.coalesce(col("pstkrq"), col("pstkq"), pl.lit(0.0)))
        .with_columns(extra=-col("pre_stock"))
        .with_columns(
            BEQ=pl.when(col("seqq").is_not_null())
            .then(col("seqq") + col("extra"))
            .when(col("ceqq").is_not_null() & col("pstkq").is_not_null())
            .then(col("ceqq") + col("pstkq") + col("extra"))
            .when(col("atq").is_not_null() & col("ltq").is_not_null())
            .then((col("atq") - col("ltq")) + col("extra"))
            .otherwise(None)
        )
        .with_columns(eom_match=col("datadate").dt.month_end())
        .join(
            crsp_full.select("permno", "eom", "me").rename({"me": "ME"}),
            left_on=["permno", "eom_match"],
            right_on=["permno", "eom"],
            how="inner",
        )
        .rename({"eom_match": "eom"})
        .with_columns(BE=col("BEQ") * 1000)
        .with_columns(adj_BE=col("BE") + 0.1 * (col("ME") - col("BE")))
        .with_columns(
            adj_BE=pl.when((col("adj_BE") < 0) & col("adj_BE").is_not_null())
            .then(1.0)
            .otherwise(col("adj_BE"))
        )
        .with_columns(MB=col("ME") / col("adj_BE"))
    )


def _mp_distress_quarterly_inputs(
    market_inputs: pl.DataFrame,
    be: pl.DataFrame,
    cq_base: pl.DataFrame,
    compustat_quarterly: pl.DataFrame,
    mp_con: duckdb.DuckDBPyConnection,
) -> pl.DataFrame:
    mp_con.register("market_inputs", market_inputs)
    mp_con.register(
        "be",
        be.select(
            "permno",
            "datadate",
            "rdq",
            "rdq_crsp",
            "lead_rdq_crsp",
            "MB",
            col("ME").alias("ME_lag"),
        ),
    )
    quarterly_joined = mp_con.execute("""
        SELECT a.*, b.MB, b.ME_lag, b.datadate AS datadate_q, b.rdq AS rdq_be
        FROM market_inputs a LEFT JOIN be b
          ON a.permno = b.permno AND a.eom >= b.rdq_crsp AND a.eom < b.lead_rdq_crsp
    """).pl()
    quarterly_joined = quarterly_joined.sort(
        ["permno", "eom", "datadate_q", "rdq_be"], descending=[False, False, True, True]
    ).unique(subset=["permno", "eom"], keep="first", maintain_order=True)

    cq_extra = (
        cq_base.select("permno", "datadate", "rdq", "rdq_crsp", "lead_rdq_crsp")
        .join(
            compustat_quarterly.select("permno", "datadate", "niq", "cheq"),
            on=["permno", "datadate"],
            how="left",
        )
        .join(cq_base.select("permno", "datadate", "ltq"), on=["permno", "datadate"], how="left")
    )
    mp_con.register("quarterly_joined", quarterly_joined)
    mp_con.register("cq", cq_extra)
    panel = mp_con.execute("""
        SELECT a.*, b.niq AS NIQ, b.ltq AS LTQ, b.cheq AS CHEQ,
               b.datadate AS datadate_q2, b.rdq AS rdq_q2
        FROM quarterly_joined a LEFT JOIN cq b
          ON a.permno = b.permno AND a.eom >= b.rdq_crsp AND a.eom < b.lead_rdq_crsp
    """).pl()
    return (
        panel.sort(
            ["permno", "eom", "datadate_q2", "rdq_q2"], descending=[False, False, True, True]
        )
        .unique(subset=["permno", "eom"], keep="first", maintain_order=True)
        .with_columns(
            NIMTA=col("NIQ") / (col("LTQ") + col("ME_lag") / 1000),
            TLMTA=col("LTQ") / (col("LTQ") + col("ME_lag") / 1000),
            CASHMTA=col("CHEQ") / (col("LTQ") + col("ME_lag") / 1000),
        )
    )


def _mp_distress_nimtaavg(panel: pl.DataFrame) -> tuple[pl.DataFrame, date]:
    nimta_mean = (
        panel.filter(col("NIMTA").is_not_null())
        .group_by("eom")
        .agg(mean_NIMTA=col("NIMTA").mean(), _f=pl.len())
        .filter(col("_f") >= 10)
    )
    if nimta_mean.height == 0:
        raise RuntimeError("distress: no eom with >=10 NIMTA obs")
    min_eom = nimta_mean["eom"].min()
    floor_eom = pl.select(_mp_offset_months(pl.lit(min_eom), -12)).item()

    panel = (
        panel.join(nimta_mean.select("eom", "mean_NIMTA"), on="eom", how="left")
        .with_columns(adj_NIMTA=pl.coalesce(col("NIMTA"), col("mean_NIMTA")))
        .drop("mean_NIMTA")
        .sort("permno", "eom")
    )

    scale_n = (1 - MP_CHS_DECAY**3) / (1 - MP_CHS_DECAY**12)
    panel = (
        panel.with_columns(
            nimta_lag0=col("adj_NIMTA"),
            nimta_lag3=col("adj_NIMTA").shift(3).over("permno"),
            nimta_lag6=col("adj_NIMTA").shift(6).over("permno"),
            nimta_lag9=col("adj_NIMTA").shift(9).over("permno"),
            eom_lag3=col("eom").shift(3).over("permno"),
            eom_lag6=col("eom").shift(6).over("permno"),
            eom_lag9=col("eom").shift(9).over("permno"),
        )
        .with_columns(
            nimta_ok=(
                (col("eom_lag3") == _mp_offset_months("eom", -3))
                & (col("eom_lag6") == _mp_offset_months("eom", -6))
                & (col("eom_lag9") == _mp_offset_months("eom", -9))
                & col("nimta_lag0").is_not_null()
                & col("nimta_lag3").is_not_null()
                & col("nimta_lag6").is_not_null()
                & col("nimta_lag9").is_not_null()
            )
        )
        .with_columns(
            NIMTAAVG=pl.when(col("nimta_ok"))
            .then(
                scale_n
                * (
                    col("nimta_lag0")
                    + col("nimta_lag3") * MP_CHS_DECAY
                    + col("nimta_lag6") * MP_CHS_DECAY**2
                    + col("nimta_lag9") * MP_CHS_DECAY**3
                )
            )
            .otherwise(None)
        )
    )
    return panel, floor_eom


def _mp_distress_exretavg(panel: pl.DataFrame, floor_eom: date) -> pl.DataFrame:
    exret_mean = (
        panel.filter(col("EXRET").is_not_null()).group_by("eom").agg(mean_EXRET=col("EXRET").mean())
    )
    panel = (
        panel.join(exret_mean, on="eom", how="left")
        .with_columns(adj_EXRET=pl.coalesce(col("EXRET"), col("mean_EXRET")))
        .drop("mean_EXRET")
        .filter(col("eom") >= floor_eom)
        .sort("permno", "eom")
    )

    scale_e = (1 - MP_CHS_DECAY) / (1 - MP_CHS_DECAY**12)
    panel = panel.with_columns(
        [col("adj_EXRET").shift(i).over("permno").alias(f"el{i}") for i in range(12)]
        + [col("eom").shift(i).over("permno").alias(f"eoml{i}") for i in range(12)]
    )
    consec = pl.lit(True)
    for i in range(12):
        consec = consec & (col(f"eoml{i}") == _mp_offset_months("eom", -i))
        consec = consec & col(f"el{i}").is_not_null()
    weighted = sum(col(f"el{i}") * (MP_CHS_DECAY**i) for i in range(12))
    return panel.with_columns(EXRETAVG=pl.when(consec).then(scale_e * weighted).otherwise(None))


def _mp_distress_sigma(crsp_daily: pl.DataFrame) -> pl.DataFrame:
    crsp_daily = crsp_daily.with_columns(eom=col("date").dt.month_end())
    monthly = (
        crsp_daily.with_columns(ret2=col("ret") ** 2)
        .group_by("permno", "eom")
        .agg(sum_ret2=col("ret2").sum(), obs_m=col("ret").is_not_null().sum().cast(pl.Int64))
        .sort("permno", "eom")
    )
    expanded = (
        monthly.select("permno", "eom")
        .with_columns(_off=pl.lit([1, 2, 3]))
        .explode("_off")
        .with_columns(
            eom_src=col("eom").dt.offset_by(pl.format("-{}mo", col("_off"))).dt.month_end()
        )
    )
    sigma_agg = (
        expanded.join(
            monthly.select("permno", col("eom").alias("eom_src"), "sum_ret2", "obs_m"),
            on=["permno", "eom_src"],
            how="left",
        )
        .group_by("permno", "eom")
        .agg(sum_total=col("sum_ret2").sum(), obs_total=col("obs_m").sum().cast(pl.Int64))
    )
    sigma = (
        monthly.join(sigma_agg, on=["permno", "eom"], how="left")
        .with_columns(
            SIGMA=pl.when(col("obs_total") > 0)
            .then(((252.0 / col("obs_total")) * col("sum_total")).sqrt())
            .otherwise(None)
        )
        .select("permno", "eom", "SIGMA")
    )
    sigma_mean = (
        sigma.filter(col("SIGMA").is_not_null()).group_by("eom").agg(avg_SIGMA=col("SIGMA").mean())
    )
    return (
        sigma.join(sigma_mean, on="eom", how="left")
        .with_columns(SIGMA=pl.coalesce(col("SIGMA"), col("avg_SIGMA")))
        .drop("avg_SIGMA")
    )


def _mp_distress_final_score(chs_inputs: pl.DataFrame, sigma: pl.DataFrame) -> pl.DataFrame:
    scored = (
        chs_inputs.join(sigma, on=["permno", "eom"], how="left")
        .sort("permno", "eom")
        .with_columns(
            eom_prev=col("eom").shift(1).over("permno"),
            lag_MB=col("MB").shift(1).over("permno"),
            lag_TLMTA=col("TLMTA").shift(1).over("permno"),
            lag_CASHMTA=col("CASHMTA").shift(1).over("permno"),
        )
        .with_columns(gap_m=_mp_month_diff(col("eom"), col("eom_prev")))
        .with_columns(
            [
                pl.when(col("gap_m") >= 6).then(None).otherwise(col(c)).alias(c)
                for c in ["lag_MB", "lag_TLMTA", "lag_CASHMTA"]
            ]
        )
    )

    scored = _mp_winsorize_per_group(
        scored,
        ["NIMTAAVG", "lag_TLMTA", "EXRETAVG", "RSIZE", "lag_CASHMTA", "lag_MB", "SIGMA"],
        lo=5,
        hi=95,
        by="eom",
    )

    scored = scored.filter(
        col("NIMTAAVG").is_not_null()
        & col("lag_TLMTA").is_not_null()
        & col("EXRETAVG").is_not_null()
        & col("RSIZE").is_not_null()
        & col("lag_CASHMTA").is_not_null()
        & col("lag_MB").is_not_null()
        & col("PRICE").is_not_null()
        & col("SIGMA").is_not_null()
    )

    return scored.with_columns(
        distress=MP_DISTRESS_BETAS["intercept"]
        + MP_DISTRESS_BETAS["NIMTAAVG"] * col("NIMTAAVG")
        + MP_DISTRESS_BETAS["TLMTA"] * col("lag_TLMTA")
        + MP_DISTRESS_BETAS["EXRETAVG"] * col("EXRETAVG")
        + MP_DISTRESS_BETAS["SIGMA"] * col("SIGMA")
        + MP_DISTRESS_BETAS["RSIZE"] * col("RSIZE")
        + MP_DISTRESS_BETAS["CASHMTA"] * col("lag_CASHMTA")
        + MP_DISTRESS_BETAS["MB"] * col("lag_MB")
        + MP_DISTRESS_BETAS["PRICE"] * col("PRICE")
    ).select("eom", "permno", "distress")


def _mp_compute_distress(
    mp_con: duckdb.DuckDBPyConnection, raw_dir: Path, interim_dir: Path
) -> pl.DataFrame:
    crsp_monthly_panel = pl.read_parquet(interim_dir / "crsp_monthly.parquet").select(
        "permno", "eom", "me", "RET", "PRC"
    )
    sp500_idx = (
        pl.read_parquet(raw_dir / "crsp_a_indexes_msp500.parquet")
        .with_columns(
            col("totval").cast(pl.Float64, strict=False),
            col("sprtrn").cast(pl.Float64, strict=False),
        )
        .with_columns(eom=col("caldt").dt.month_end())
        .select("eom", "totval", "sprtrn")
    )
    compustat_quarterly = pl.read_parquet(interim_dir / "compustat_quarterly.parquet")
    cq_base = compustat_quarterly.select(
        "permno",
        "gvkey",
        "datadate",
        "fyearq",
        "fqtr",
        "rdq",
        "rdq_crsp",
        "lead_rdq_crsp",
        "atq",
        "ltq",
        "seqq",
        "ceqq",
        "pstkq",
        "pstkrq",
    )

    market_inputs = _mp_distress_market_inputs(crsp_monthly_panel, sp500_idx)
    be = _mp_distress_book_equity_mb(cq_base, crsp_monthly_panel)
    panel = _mp_distress_quarterly_inputs(market_inputs, be, cq_base, compustat_quarterly, mp_con)
    panel, floor_eom = _mp_distress_nimtaavg(panel)
    panel = _mp_distress_exretavg(panel, floor_eom)

    chs_inputs = panel.select(
        "eom", "permno", "NIMTAAVG", "TLMTA", "EXRETAVG", "RSIZE", "CASHMTA", "MB", "PRICE"
    )
    sigma = _mp_distress_sigma(pl.read_parquet(interim_dir / "crsp_daily.parquet"))
    distress = _mp_distress_final_score(chs_inputs, sigma)

    out = (
        crsp_monthly_panel.select("permno", "eom")
        .join(distress, on=["permno", "eom"], how="left")
        .unique(subset=["permno", "eom"], keep="first", maintain_order=True)
        .sort("permno", "eom")
        .select("eom", "permno", "distress")
    )
    out.write_parquet(interim_dir / "distress.parquet")
    return out


def _mp_monthly_lagged_me_panel(interim_dir: Path) -> pl.DataFrame:
    # Pure CIZ: ME = mthcap directly (CRSP end-of-month market cap). Sort by
    # (permno, eom) ascending so .shift(1).over("permno") yields the prior eom obs.
    return (
        pl.read_parquet(interim_dir / "crsp_monthly_full.parquet")
        .select("eom", "permno", "exchcd", "PRC", "RET", "me")
        .rename({"PRC": "prc", "RET": "ret", "me": "ME"})
        .with_columns(prc=col("prc").abs())
        .sort("permno", "eom")
        .with_columns(
            mktcap=col("ME").shift(1).over("permno"),
            lag_prc=col("prc").shift(1).over("permno"),
        )
        .filter(
            col("ret").is_not_null() & col("mktcap").is_not_null() & col("lag_prc").is_not_null()
        )
        .select("eom", "permno", "exchcd", "prc", "ret", "ME", "mktcap", "lag_prc")
    )


def _mp_percentile_rank(
    m: pl.DataFrame,
    descending: bool,
    mp_con: duckdb.DuckDBPyConnection,
    *,
    id_select: list[str],
    group_keys: list[str],
) -> pl.DataFrame:
    """Percentile rank (1..100) via 99 quantile_disc breakpoints per group.
    Returns id/group cols, rank_var, cnt (group size), eff (non-null count),
    pct_raw. Sign-flips rank_var into _v for descending anomalies."""
    m = m.with_columns(_v=(-col("rank_var")) if descending else col("rank_var"))
    mp_con.register("m", m)
    breaks_sql = ", ".join(f"quantile_disc(_v, {k / 100.0}) AS q{k}" for k in range(1, 100))
    bucket_sql = " + ".join(f"(CASE WHEN a._v >= b.q{k} THEN 1 ELSE 0 END)" for k in range(1, 100))
    ids = ", ".join(f"a.{c}" for c in id_select)
    gk = ", ".join(group_keys)
    gk_a = ", ".join(f"a.{c}" for c in group_keys)
    join_on = " AND ".join(f"a.{c} = b.{c}" for c in group_keys)
    return mp_con.execute(f"""
        WITH breaks AS (
            SELECT {gk}, {breaks_sql}
            FROM m WHERE _v IS NOT NULL GROUP BY {gk}
        )
        SELECT {ids}, {gk_a}, a.rank_var,
               COUNT(*) OVER (PARTITION BY {gk_a}) AS cnt,
               COUNT(a.rank_var) OVER (PARTITION BY {gk_a}) AS eff,
               ({bucket_sql}) + 1 AS pct_raw
        FROM m a LEFT JOIN breaks b ON {join_on}
    """).pl()


def _mp_anomaly_pct_panel(
    me_panel: pl.DataFrame,
    min_stks: int,
    mp_con: duckdb.DuckDBPyConnection,
    *,
    panel_cols: list[str],
    sort_keys: list[str],
    id_cols: list[str],
    group_keys: list[str],
    anom_source: Callable[[str], pl.DataFrame | None],
) -> pl.DataFrame:
    """Per-group percentile-rank each anomaly and left-join (value, pct) onto
    the base return panel. Base universe: lag_prc > 5, mktcap > 0, non-null ret.
    anom_source(anom) returns a (id, eom, rank_var) frame or None to skip.
    pct nulled when effective group count < min_stks.
    Returns panel_cols plus one (anom, pct_anom) column pair per available anomaly."""
    base = me_panel.filter((col("lag_prc") > 5) & (col("mktcap") > 0) & col("ret").is_not_null())
    panel = base.select(*panel_cols).sort(*sort_keys)
    src_join = [*id_cols, "eom"]  # anomaly frames are keyed by (id, eom) only
    panel_join = [*id_cols, *group_keys]

    for anom in MP_ANOMALY_LIST:
        anom_df = anom_source(anom)
        if anom_df is None:
            continue
        m = (
            base.select(*panel_join, "mktcap")
            .join(anom_df, on=src_join, how="inner")
            .pipe(
                _mp_percentile_rank,
                descending=anom in MP_POSITIVE_ANOMALIES,
                mp_con=mp_con,
                id_select=id_cols,
                group_keys=group_keys,
            )
            .filter(col("cnt") >= min_stks)
            .with_columns(
                pct=pl.when((col("eff") < min_stks) | col("rank_var").is_null())
                .then(None)
                .otherwise(col("pct_raw").cast(pl.Float64))
            )
        )
        panel = panel.join(
            m.select(*panel_join, col("rank_var").alias(anom), col("pct").alias(f"pct_{anom}")),
            on=panel_join,
            how="left",
        )
    return panel


def _mp_percentile_rank_anomalies(
    me_panel: pl.DataFrame,
    min_stks: int,
    mp_con: duckdb.DuckDBPyConnection,
    interim_dir: Path,
) -> pl.DataFrame:
    def anom_source(anom: str) -> pl.DataFrame:
        fcol = anom.lower()
        return pl.read_parquet(interim_dir / f"{fcol}.parquet").select(
            "permno", "eom", col(fcol).alias("rank_var")
        )

    return _mp_anomaly_pct_panel(
        me_panel,
        min_stks,
        mp_con,
        panel_cols=["permno", "eom", "ret", "lag_prc", "mktcap", "exchcd"],
        sort_keys=["permno", "eom"],
        id_cols=["permno"],
        group_keys=["eom"],
        anom_source=anom_source,
    )


def _mp_build_mispricing_panel(
    min_stks: int, mp_con: duckdb.DuckDBPyConnection, interim_dir: Path
) -> pl.DataFrame:
    panel = _mp_percentile_rank_anomalies(
        _mp_monthly_lagged_me_panel(interim_dir), min_stks, mp_con, interim_dir
    )
    pct_cols = [f"pct_{a}" for a in MP_ANOMALY_LIST]
    return (
        panel.with_columns(anomal_num=sum(col(c).is_not_null().cast(pl.Int64) for c in pct_cols))
        .filter(col("anomal_num") > 0)
        .with_columns([col(c).fill_null(0.0) for c in pct_cols])
        .sort("eom", "permno")
    )


def _mp_stock_mispricing_scores(
    mispricing_panel: pl.DataFrame,
    min_fcts: int,
    *,
    sort_keys: list[str],
    select_cols: list,
) -> pl.DataFrame:
    """Mgmt/perf leg scores per stock-month; keeps rows with either score."""
    return (
        _mp_add_leg_score(
            mispricing_panel, MP_MGMT_PCT_COLS, "mispricing_mgmt", "_n_mgmt", min_count=min_fcts
        )
        .pipe(_mp_add_leg_score, MP_PERF_PCT_COLS, "mispricing_perf", "_n_perf", min_count=min_fcts)
        .filter(col("mispricing_mgmt").is_not_null() | col("mispricing_perf").is_not_null())
        .sort(*sort_keys)
        .select(*select_cols, "mispricing_mgmt", "mispricing_perf")
    )


def _mp_filter_active(
    mispricing_panel: pl.DataFrame,
    cols_list: list[str],
    score_col: str,
    num_col: str,
    min_fcts: int,
) -> pl.DataFrame:
    return _mp_add_leg_score(
        mispricing_panel, cols_list, score_col, num_col, min_count=min_fcts
    ).filter(col(num_col) >= min_fcts)


def _mp_double_sort_buckets(
    df: pl.DataFrame,
    score_col: str,
    *,
    group_keys: list[str],
    size_break_sql: str,
) -> pl.DataFrame:
    """2×3 double-sort: size median → port_size 1/2, score 20/80 terciles
    → port_var 1/2/3. Breakpoints computed per group_keys. Returns input
    frame plus port_size and port_var columns."""
    gk = ", ".join(group_keys)
    breaks = _mp_polars_sql(
        df,
        f"""
        SELECT {gk},
               quantile_disc({size_break_sql}, 0.50) AS size50,
               quantile_disc({score_col}, 0.20) AS v20,
               quantile_disc({score_col}, 0.80) AS v80
        FROM src GROUP BY {gk}
    """,
    )
    return (
        df.join(breaks, on=group_keys, how="left")
        .with_columns(
            port_size=pl.when((col("mktcap") > 0) & (col("mktcap") < col("size50")))
            .then(1)
            .when(col("size50") <= col("mktcap"))
            .then(2)
            .otherwise(None),
            port_var=pl.when(col(score_col) < col("v20"))
            .then(1)
            .when((col("v20") <= col(score_col)) & (col(score_col) < col("v80")))
            .then(2)
            .when(col("v80") <= col(score_col))
            .then(3)
            .otherwise(None),
        )
        .drop("size50", "v20", "v80")
    )


def _mp_vw_monthly(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    return df.group_by(group_cols).agg(
        vwret=(col("ret") * col("mktcap")).sum() / col("mktcap").sum(), _freq_=pl.len()
    )


def _mp_daily_vw_returns(
    panel: pl.DataFrame,
    group_cols: list[str],
    mp_con: duckdb.DuckDBPyConnection,
    *,
    daily_path: str,
    id_col: str,
    table: str,
    qualify_group: bool,
) -> pl.DataFrame:
    """Daily VW portfolio returns. Joins the monthly bucket panel onto the
    daily return file per (id, eom) via DuckDB read_parquet.
    Returns one row per (date, *group_cols) with vwret and _freq_."""
    mp_con.register(table, panel)
    group_sql = ", ".join(group_cols)
    # Qualify with a.<col> when group cols (e.g. excntry) exist on both join
    # sides, else DuckDB raises a BinderError.
    sel_sql = ", ".join(f"a.{c}" for c in group_cols) if qualify_group else group_sql
    return mp_con.execute(f"""
        WITH joined AS (
            SELECT b.date AS date, a.mktcap, b.ret AS ret_d, {sel_sql}
            FROM {table} a
            JOIN read_parquet('{daily_path}') b
              ON a.{id_col} = b.{id_col} AND a.eom = b.eom
            WHERE b.ret IS NOT NULL AND a.mktcap IS NOT NULL
        )
        SELECT date, {group_sql},
               SUM(ret_d * mktcap) / SUM(mktcap) AS vwret,
               COUNT(*) AS _freq_
        FROM joined GROUP BY date, {group_sql}
    """).pl()


def _mp_diff_legs(
    df: pl.DataFrame,
    by: str | list[str],
    key_col: str,
    key_a: int,
    key_b: int,
    out_name: str,
) -> pl.DataFrame:
    by_cols = [by] if isinstance(by, str) else list(by)
    a = df.filter(col(key_col) == key_a).select(*by_cols, col("vwret").alias("a"))
    b = df.filter(col(key_col) == key_b).select(*by_cols, col("vwret").alias("b"))
    return a.join(b, on=by_cols, how="inner").with_columns((col("a") - col("b")).alias(out_name))


def _mp_spread_per_size_then_diff(port_ret: pl.DataFrame, by: str | list[str]) -> pl.DataFrame:
    by_cols = [by] if isinstance(by, str) else list(by)
    misp = port_ret.group_by(*by_cols, "port_var").agg(misp=col("vwret").mean())
    return _mp_diff_legs(misp.rename({"misp": "vwret"}), by_cols, "port_var", 1, 3, "umo")


def _mp_mispricing_legs(
    mispricing_panel: pl.DataFrame,
    min_fcts: int,
    *,
    group_keys: list[str],
    size_break_sql: str,
) -> dict[str, pl.DataFrame]:
    """Mgmt/perf legs: filter to active stocks, then 2x3 double-sort buckets."""
    spec = {
        "mgmt": (MP_MGMT_PCT_COLS, "score_mgmt", "n_mgmt"),
        "perf": (MP_PERF_PCT_COLS, "score_perf", "n_perf"),
    }
    return {
        name: _mp_double_sort_buckets(
            _mp_filter_active(mispricing_panel, cols_list, score_col, num_col, min_fcts),
            score_col,
            group_keys=group_keys,
            size_break_sql=size_break_sql,
        )
        for name, (cols_list, score_col, num_col) in spec.items()
    }


def _mp_smb_panel_from_legs(
    legs: dict[str, pl.DataFrame], key_cols: tuple[str, ...]
) -> pl.DataFrame:
    """SMB neutral panel (port_var == 2 stocks). key_cols identifies the
    stock: ("permno",) for US, ("id", "excntry") for world."""
    a = (
        legs["mgmt"]
        .filter(col("port_var") == 2)
        .select(*key_cols, "eom", "ret", "mktcap", "port_size")
    )
    b = legs["perf"].filter(col("port_var") == 2).select(*key_cols, "eom", "port_size")
    return a.join(b, on=[*key_cols, "eom", "port_size"], how="inner")


def _mp_thin_buckets(
    port_ret: pl.DataFrame, *, truncate: bool, by: str | list[str], min_n: int
) -> pl.DataFrame:
    """Thin-bucket handling: US truncates cross-sectionally, world hard-filters."""
    if truncate:
        return _mp_truncate_thin(port_ret, by=by, freq_col="_freq_", min_n=min_n)
    return port_ret.filter(col("_freq_") >= min_n)


def _mp_monthly_factor_returns(
    legs: dict[str, pl.DataFrame],
    smb_panel: pl.DataFrame,
    min_obs: int,
    *,
    by: str | list[str],
    truncate_thin: bool,
    out_cols: list[str],
    out_sort: list[str],
) -> pl.DataFrame:
    """Monthly UMO (mgmt/perf) and SMB factor returns from bucketed legs.
    Per leg: VW per (by, port_size, port_var), thin-bucket handling
    (truncate or hard-filter per truncate_thin), 6-bucket completeness
    gate, 1-3 spread per size → UMO. SMB: 1-2 diff per port_size.
    Returns one row per `by` group with smb/mgmt/perf factor returns."""
    by_cols = [by] if isinstance(by, str) else list(by)
    leg_umo = {}
    for name, panel in legs.items():
        port_returns = (
            _mp_vw_monthly(
                panel.filter(col("port_size").is_not_null() & col("port_var").is_not_null()),
                [*by_cols, "port_size", "port_var"],
            )
            .pipe(_mp_thin_buckets, truncate=truncate_thin, by=by, min_n=min_obs)
            .pipe(_mp_filter_full_buckets, 6, by=by)
            .sort(*by_cols, "port_size", "port_var")
        )
        leg_umo[name] = _mp_spread_per_size_then_diff(port_returns, by).select(
            *by_cols, col("umo").alias(f"mispricing_{name}")
        )

    smb_port = (
        _mp_vw_monthly(smb_panel, [*by_cols, "port_size"])
        .pipe(_mp_thin_buckets, truncate=truncate_thin, by=by, min_n=min_obs)
        .pipe(_mp_filter_full_buckets, 2, by=by)
        .sort(*by_cols, "port_size")
    )
    smb_df = _mp_diff_legs(smb_port, by, "port_size", 1, 2, "smb_mispricing").select(
        *by_cols, "smb_mispricing"
    )

    return (
        leg_umo["mgmt"]
        .join(leg_umo["perf"], on=by_cols, how="inner")
        .join(smb_df, on=by_cols, how="inner")
        .filter(col("eom") >= MP_START_FACTOR_EOM)
        .select(*out_cols, "smb_mispricing", "mispricing_mgmt", "mispricing_perf")
        .sort(*out_sort)
    )


def _mp_us_daily_factor_returns(
    legs: dict[str, pl.DataFrame],
    smb_panel: pl.DataFrame,
    min_obs: int,
    mp_con: duckdb.DuckDBPyConnection,
    interim_dir: Path,
) -> pl.DataFrame:
    """US daily UMO (mgmt/perf) and SMB factor returns per date, VW against
    crsp_daily on permno. Per leg: min_obs filter, 6-bucket completeness
    gate, 1-3 spread per size -> UMO; SMB: 1-2 diff per port_size."""
    daily_path = (interim_dir / "crsp_daily.parquet").as_posix()
    leg_umo = {}
    for name, panel in legs.items():
        bucketed = panel.filter(
            col("port_size").is_not_null() & col("port_var").is_not_null()
        ).select("permno", "eom", "mktcap", "port_size", "port_var")
        port_returns = (
            _mp_daily_vw_returns(
                bucketed,
                ["port_size", "port_var"],
                mp_con,
                daily_path=daily_path,
                id_col="permno",
                table="panel",
                qualify_group=False,
            )
            .filter(col("_freq_") >= min_obs)
            .pipe(_mp_filter_full_buckets, 6, by="date")
            .sort("date", "port_size", "port_var")
        )
        leg_umo[name] = _mp_spread_per_size_then_diff(port_returns, "date").select(
            "date", col("umo").alias(f"mispricing_{name}")
        )

    smb_daily = smb_panel.select("permno", "eom", "mktcap", "port_size")
    smb_port = (
        _mp_daily_vw_returns(
            smb_daily,
            ["port_size"],
            mp_con,
            daily_path=daily_path,
            id_col="permno",
            table="panel",
            qualify_group=False,
        )
        .filter(col("_freq_") >= min_obs)
        .pipe(_mp_filter_full_buckets, 2, by="date")
        .sort("date", "port_size")
    )
    smb_df = _mp_diff_legs(smb_port, "date", "port_size", 1, 2, "smb_mispricing").select(
        "date", "smb_mispricing"
    )

    daily_start = date(MP_START_FACTOR_EOM.year, MP_START_FACTOR_EOM.month, 1)
    return (
        leg_umo["mgmt"]
        .join(leg_umo["perf"], on=["date"], how="inner")
        .join(smb_df, on=["date"], how="inner")
        .filter(col("date") >= daily_start)
        .select("date", "smb_mispricing", "mispricing_mgmt", "mispricing_perf")
        .sort("date")
    )


def _mp_world_daily_factor_returns(
    legs: dict[str, pl.DataFrame],
    smb_panel: pl.DataFrame,
    min_obs: int,
    mp_con: duckdb.DuckDBPyConnection,
    interim_dir: Path,
) -> pl.DataFrame:
    """World daily UMO (mgmt/perf) and SMB factor returns per (excntry, date),
    VW against world_dsf on id. Same construction as the US version with
    per-country grouping."""
    daily_path = (interim_dir / "world_dsf.parquet").as_posix()
    leg_umo = {}
    for name, panel in legs.items():
        bucketed = panel.filter(
            col("port_size").is_not_null() & col("port_var").is_not_null()
        ).select("id", "excntry", "eom", "mktcap", "port_size", "port_var")
        port_returns = (
            _mp_daily_vw_returns(
                bucketed,
                ["excntry", "port_size", "port_var"],
                mp_con,
                daily_path=daily_path,
                id_col="id",
                table="panel_w",
                qualify_group=True,
            )
            .filter(col("_freq_") >= min_obs)
            .pipe(_mp_filter_full_buckets, 6, by=["excntry", "date"])
            .sort("excntry", "date", "port_size", "port_var")
        )
        leg_umo[name] = _mp_spread_per_size_then_diff(port_returns, ["excntry", "date"]).select(
            "excntry", "date", col("umo").alias(f"mispricing_{name}")
        )

    smb_daily = smb_panel.select("id", "excntry", "eom", "mktcap", "port_size")
    smb_port = (
        _mp_daily_vw_returns(
            smb_daily,
            ["excntry", "port_size"],
            mp_con,
            daily_path=daily_path,
            id_col="id",
            table="panel_w",
            qualify_group=True,
        )
        .filter(col("_freq_") >= min_obs)
        .pipe(_mp_filter_full_buckets, 2, by=["excntry", "date"])
        .sort("excntry", "date", "port_size")
    )
    smb_df = _mp_diff_legs(
        smb_port, ["excntry", "date"], "port_size", 1, 2, "smb_mispricing"
    ).select("excntry", "date", "smb_mispricing")

    daily_start = date(MP_START_FACTOR_EOM.year, MP_START_FACTOR_EOM.month, 1)
    return (
        leg_umo["mgmt"]
        .join(leg_umo["perf"], on=["excntry", "date"], how="inner")
        .join(smb_df, on=["excntry", "date"], how="inner")
        .filter(col("date") >= daily_start)
        .select("date", "excntry", "smb_mispricing", "mispricing_mgmt", "mispricing_perf")
        .sort("excntry", "date")
    )


def _mp_world_load_world_data(interim_dir: Path, start_dt: date = date(1963, 1, 1)) -> pl.DataFrame:
    """Load jkp's world_data_prelim.parquet (per-(id, eom) panel with jkp-derived chars).
    Filter to MAIN_FILTERS, exclude USA (US flows through US branch). Returns the
    columns needed by the world pipeline. Cast numerics to Float64.

    Reads the prelim file (not final world_data.parquet) because gen_mispricing_data
    runs before merge_qmj_to_world_data in run_pipeline; the 17 columns selected here
    all live in the prelim and are not modified by the downstream merges."""
    p = interim_dir / "world_data_prelim.parquet"
    cols_keep = [
        "id",
        "permno",
        "gvkey",
        "excntry",
        "eom",
        "me",
        "prc",
        "ret",
        "adjfct",
        "shares",
        "size_grp",
        "common",
        "primary_sec",
        "obs_main",
        "exch_main",
    ] + list(MP_WORLD_JKP_CHAR.values())
    df = (
        pl.scan_parquet(p)
        .filter(
            (col("excntry") != "USA")
            & (col("common") == 1)
            & (col("primary_sec") == 1)
            & (col("obs_main") == 1)
            & (col("exch_main") == 1)
            & (col("eom") >= start_dt)
        )
        .select(cols_keep)
        .collect()
    )
    floats = ["me", "prc", "ret", "adjfct", "shares"] + list(MP_WORLD_JKP_CHAR.values())
    return df.with_columns(
        [col(c).cast(pl.Float64, strict=False) for c in floats if c in df.columns]
    )


def _mp_world_load_market(interim_dir: Path, daily: bool = False) -> pl.DataFrame:
    """Per-country VW market return + total market cap. Cols: excntry, eom (or date),
    mkt_vw, me_lag1. USD."""
    name = "market_returns_daily" if daily else "market_returns"
    p = interim_dir / f"{name}.parquet"
    date_col = "date" if daily else "eom"
    return (
        pl.read_parquet(p)
        .select("excntry", date_col, "mkt_vw", "me_lag1")
        .with_columns([col(c).cast(pl.Float64, strict=False) for c in ["mkt_vw", "me_lag1"]])
    )


def _mp_world_load_world_dsf(interim_dir: Path, start_dt: date = date(1963, 1, 1)) -> pl.DataFrame:
    """Daily world stock panel filtered to common/exch_main. For DISTRESS SIGMA."""
    p = interim_dir / "world_dsf.parquet"
    return (
        pl.scan_parquet(p)
        .filter(
            (col("excntry") != "USA")
            & (col("common") == 1)
            & (col("primary_sec") == 1)
            & (col("obs_main") == 1)
            & (col("exch_main") == 1)
            & (col("date") >= start_dt)
            & col("ret").is_not_null()
        )
        .select("id", "excntry", "date", "eom", col("ret").cast(pl.Float64, strict=False))
        .collect()
    )


def _mp_world_load_fundq_global(raw_dir: Path) -> pl.DataFrame:
    """Load global + NA Compustat quarterly with USD conversion, return union.
    Used by DISTRESS to build NIMTA/TLMTA/CASHMTA/MB per excntry."""
    # NA (US-only quarterly) — already used by US distress; reuse.
    fq_us = _mp_load_fundq(raw_dir).with_columns(excntry_src=pl.lit("USA"))
    # Global — comp.g_fundq, pulled by jkp.
    g_path = raw_dir / "comp_g_fundq.parquet"
    fq_g = (
        pl.read_parquet(g_path)
        .with_columns(
            [
                col(c).cast(pl.Float64, strict=False)
                for c in [
                    "ibq",
                    "saleq",
                    "niq",
                    "cheq",
                    "seqq",
                    "ceqq",
                    "pstkq",
                    "atq",
                    "ltq",
                    "pstkrq",
                    "txditcq",
                    "actq",
                    "lctq",
                    "piq",
                    "dlttq",
                    "revtq",
                ]  # fmt: skip
                if c in pl.read_parquet(g_path, n_rows=1).columns
            ]
        )
        .with_columns(excntry_src=pl.lit("ROW"))
    )
    return pl.concat([fq_us, fq_g], how="diagonal_relaxed")


def _mp_world_compute_jkp_anomalies(world_data: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """For 7 SY anomalies covered by jkp pre-computed chars, return per-anomaly
    (id, eom, value) frames. World only — US still computes its own."""
    out = {}
    for sy_name, jkp_col in MP_WORLD_JKP_CHAR.items():
        out[sy_name] = world_data.select(
            "id", "eom", "excntry", col(jkp_col).alias(sy_name.lower())
        ).filter(col(sy_name.lower()).is_not_null())
    return out


def _mp_world_compute_momentum(world_monthly: pl.DataFrame) -> pl.DataFrame:
    """11-month log-return momentum (months t-2..t-12), per id. world_monthly has id, eom, ret."""
    log_rets = world_monthly.select("id", "excntry", "eom", "ret").with_columns(
        log_1p_ret=(1 + col("ret")).log()
    )
    roll = _mp_rolling_calendar_sum(
        log_rets, "log_1p_ret", lag_min=2, lag_max=12, n_required=11, id_col="id"
    )
    return (
        log_rets.select("id", "excntry", "eom")
        .join(roll, on=["id", "eom"], how="left")
        .with_columns(momentum=col("r").exp() - 1)
        .select("id", "excntry", "eom", "momentum")
        .sort("id", "eom")
    )


def _mp_world_compute_composite_issue(world_monthly: pl.DataFrame) -> pl.DataFrame:
    """log(me_t / me_{t-12}) - cum_log_ret_12, lagged 5 months. USD-consistent
    (both me and ret from world_msf, both end-of-period USD)."""
    log_rets = world_monthly.select("id", "excntry", "eom", "me", "ret").with_columns(
        log_1p_ret=(1 + col("ret")).log()
    )
    cumret = _mp_rolling_calendar_sum(
        log_rets, "log_1p_ret", lag_min=0, lag_max=11, n_required=12, id_col="id"
    )
    me_lag12 = log_rets.select("id", "eom", col("me").alias("me_lag12")).with_columns(
        eom=_mp_offset_months("eom", 12)
    )
    composite_t = (
        log_rets.select("id", "excntry", "eom", "me")
        .join(me_lag12, on=["id", "eom"], how="left")
        .join(cumret, on=["id", "eom"], how="left")
        .select(
            "id",
            col("eom").alias("eom_src"),
            composite_issue=(col("me") / col("me_lag12")).log() - col("r"),
        )
    )
    return (
        log_rets.select("id", "excntry", "eom")
        .with_columns(eom_src=_mp_offset_months("eom", -5))
        .join(composite_t, on=["id", "eom_src"], how="left")
        .select("eom", "id", "excntry", "composite_issue")
        .sort("id", "eom")
    )


def _mp_world_compute_stock_issue(world_monthly: pl.DataFrame) -> pl.DataFrame:
    """log(shares_t * adjfct_t / shares_{t-12} * adjfct_{t-12}) — share count, FX-agnostic."""
    log_rets = (
        world_monthly.select("id", "excntry", "eom", "shares", "adjfct")
        .with_columns(adj_shares=col("shares") * col("adjfct"))
        .sort("id", "eom")
    )
    # Lag 12 months
    src = log_rets.select(
        "id", col("eom").alias("eom_src"), col("adj_shares").alias("adj_shares_lag12")
    )
    tgt = log_rets.select("id", "excntry", "eom", "adj_shares").with_columns(
        eom_src=_mp_offset_months("eom", -12)
    )
    return (
        tgt.join(src, on=["id", "eom_src"], how="left")
        .with_columns(
            stock_issue=pl.when(
                col("adj_shares").is_null()
                | (col("adj_shares") == 0)
                | col("adj_shares_lag12").is_null()
                | (col("adj_shares_lag12") == 0)
            )
            .then(0.0)
            .otherwise((col("adj_shares") / col("adj_shares_lag12")).log())
        )
        .select("eom", "id", "excntry", "stock_issue")
        .sort("id", "eom")
    )


def _mp_world_distress_market_inputs(
    world_monthly: pl.DataFrame, market_m: pl.DataFrame
) -> pl.DataFrame:
    """Market inputs per (id, eom): formation-lagged ME, PRICE, EXRET, RSIZE.
    World analog of _mp_distress_market_inputs (market_returns vs msp500)."""
    market_idx = market_m.rename({"mkt_vw": "sprtrn", "me_lag1": "totval"})
    monthly_panel = world_monthly.select(
        "id", "excntry", "eom", col("prc").alias("PRC"), col("ret").alias("RET"), col("me")
    )
    return (
        monthly_panel.with_columns(prc=col("PRC").abs().clip(upper_bound=15.0))
        .join(market_idx, on=["excntry", "eom"], how="left")
        .with_columns(
            PRICE=col("prc").log(),
            EXRET=(1 + col("RET")).log() - (1 + col("sprtrn")).log(),
            RSIZE=(col("me") / col("totval")).log(),
        )
        .sort("id", "eom")
        .with_columns(
            form_eom=col("eom").shift(1).over("id"),
            lag_ME=col("me").shift(1).over("id"),
            lag_PRICE=col("PRICE").shift(1).over("id"),
            lag_EXRET=col("EXRET").shift(1).over("id"),
            lag_RSIZE=col("RSIZE").shift(1).over("id"),
        )
        .with_columns(gap=_mp_month_diff(col("eom"), col("form_eom")))
        .with_columns(
            [
                pl.when(col("gap") > 1).then(None).otherwise(col(c)).alias(c)
                for c in ["lag_ME", "lag_PRICE", "lag_EXRET", "lag_RSIZE"]
            ]
        )
        .select(
            "id",
            "excntry",
            "eom",
            col("lag_ME").alias("ME"),
            col("lag_PRICE").alias("PRICE"),
            col("lag_EXRET").alias("EXRET"),
            col("lag_RSIZE").alias("RSIZE"),
        )
    )


def _mp_world_distress_fundq(raw_dir: Path) -> pl.DataFrame:
    """Quarterly Compustat (NA + Global union) with rdq_crsp / lead_rdq_crsp
    availability windows per (gvkey, datadate)."""
    fundq = (
        _mp_world_load_fundq_global(raw_dir)
        .filter(
            (col("indfmt") == "INDL")
            & (col("datafmt") == "STD")
            & (col("popsrc").is_in(["D", "I"]))
            & (col("consol") == "C")
        )
        .select(
            "gvkey",
            "datadate",
            "rdq",
            "ibq",
            "atq",
            "ltq",
            "seqq",
            "ceqq",
            "pstkq",
            "pstkrq",
            "niq",
            "cheq",
            "excntry_src",
        )
    )
    # NA+Global can carry the same (gvkey, datadate) with different values; prefer NA
    # (HXZ/FF convention), then a fully-valued ladder so the parallel sort is stable.
    return (
        fundq.with_columns(_src=(col("excntry_src") != "USA").cast(pl.Int8))
        .sort(
            ["gvkey", "datadate", "_src", "rdq", "ibq", "atq", "ltq", "niq", "cheq"],
            descending=[False, False, False, False, True, True, True, True, True],
            nulls_last=True,
        )
        .with_columns(adj_rdq=pl.coalesce(col("rdq"), _mp_offset_months("datadate", 3)))
        .filter(col("adj_rdq").is_not_null())
        .unique(subset=["gvkey", "datadate"], keep="first", maintain_order=True)
        .drop("_src", "excntry_src")
        .with_columns(rdq_crsp=_mp_offset_months("adj_rdq", 1))
        .sort(["gvkey", "rdq_crsp", "datadate"], descending=[False, True, True])
        .with_columns(
            lead_rdq_crsp=pl.coalesce(
                col("rdq_crsp").shift(1).over("gvkey"),
                _mp_offset_qtrs_end("rdq_crsp", 1),
            )
        )
        .sort("gvkey", "datadate")
    )


def _mp_world_distress_book_equity_mb(
    fundq: pl.DataFrame, world_monthly: pl.DataFrame
) -> pl.DataFrame:
    """BE chain: SEQQ -> CEQQ+PSTKQ -> ATQ-LTQ, plus TXDITCQ,
    minus preferred (PSTKRQ -> PSTKQ -> 0; missing pstkrq handled for global).
    Then adj_BE = BE + 0.1*(ME - BE) floored at 1 and MB = ME / adj_BE."""
    return (
        fundq.with_columns(pre_stock=pl.coalesce(col("pstkrq"), col("pstkq"), pl.lit(0.0)))
        .with_columns(extra=-col("pre_stock"))
        .with_columns(
            BEQ=pl.when(col("seqq").is_not_null())
            .then(col("seqq") + col("extra"))
            .when(col("ceqq").is_not_null() & col("pstkq").is_not_null())
            .then(col("ceqq") + col("pstkq") + col("extra"))
            .when(col("atq").is_not_null() & col("ltq").is_not_null())
            .then((col("atq") - col("ltq")) + col("extra"))
            .otherwise(None)
        )
        .with_columns(eom_match=col("datadate").dt.month_end())
        .join(
            world_monthly.select("id", "gvkey", "eom", col("me").alias("ME")),
            left_on=["gvkey", "eom_match"],
            right_on=["gvkey", "eom"],
            how="inner",
        )
        .rename({"eom_match": "eom"})
        .with_columns(BE=col("BEQ") * 1000)
        .with_columns(adj_BE=col("BE") + 0.1 * (col("ME") - col("BE")))
        .with_columns(
            adj_BE=pl.when((col("adj_BE") < 0) & col("adj_BE").is_not_null())
            .then(1.0)
            .otherwise(col("adj_BE"))
        )
        .with_columns(MB=col("ME") / col("adj_BE"))
    )


def _mp_world_distress_quarterly_join(
    market_inputs: pl.DataFrame,
    id_gvkey: pl.DataFrame,
    be: pl.DataFrame,
    fundq: pl.DataFrame,
    mp_con: duckdb.DuckDBPyConnection,
) -> pl.DataFrame:
    """Window-join MB/ME then NIQ/LTQ/CHEQ onto (id, eom) via the gvkey rdq
    windows; derive NIMTA/TLMTA/CASHMTA."""
    # Join market_inputs to quarterly via id+eom window. Window uses gvkey-defined rdq_crsp.
    # Need (id, eom) → applicable (gvkey, rdq window). Use id_gvkey to translate.
    mp_con.register("market_inputs", market_inputs.join(id_gvkey, on=["id", "eom"], how="left"))
    mp_con.register(
        "be",
        be.select(
            "gvkey", "datadate", "rdq", "rdq_crsp", "lead_rdq_crsp", "MB", col("ME").alias("ME_lag")
        ),
    )
    quarterly_joined = mp_con.execute("""
        SELECT a.*, b.MB, b.ME_lag, b.datadate AS datadate_q, b.rdq AS rdq_be
        FROM market_inputs a LEFT JOIN be b
          ON a.gvkey = b.gvkey AND a.eom >= b.rdq_crsp AND a.eom < b.lead_rdq_crsp
    """).pl()
    # `be` has one row per (gvkey, datadate, share-class id), so a panel row can match
    # several b rows tied on (datadate_q, rdq_be); MB/ME_lag DESC makes the pick
    # deterministic (remaining ties have identical payloads).
    quarterly_joined = quarterly_joined.sort(
        ["id", "eom", "datadate_q", "rdq_be", "MB", "ME_lag"],
        descending=[False, False, True, True, True, True],
        nulls_last=True,
    ).unique(subset=["id", "eom"], keep="first", maintain_order=True)

    cq_extra = fundq.select(
        "gvkey", "datadate", "rdq", "rdq_crsp", "lead_rdq_crsp", "niq", "cheq", "ltq"
    )
    mp_con.register("quarterly_joined", quarterly_joined)
    mp_con.register("cq", cq_extra)
    panel = mp_con.execute("""
        SELECT a.*, b.niq AS NIQ, b.ltq AS LTQ, b.cheq AS CHEQ,
               b.datadate AS datadate_q2, b.rdq AS rdq_q2
        FROM quarterly_joined a LEFT JOIN cq b
          ON a.gvkey = b.gvkey AND a.eom >= b.rdq_crsp AND a.eom < b.lead_rdq_crsp
    """).pl()
    return (
        panel.sort(["id", "eom", "datadate_q2", "rdq_q2"], descending=[False, False, True, True])
        .unique(subset=["id", "eom"], keep="first", maintain_order=True)
        .with_columns(
            NIMTA=safe_div(col("NIQ"), col("LTQ") + col("ME_lag") / 1000, "NIMTA"),
            TLMTA=safe_div(col("LTQ"), col("LTQ") + col("ME_lag") / 1000, "TLMTA"),
            CASHMTA=safe_div(col("CHEQ"), col("LTQ") + col("ME_lag") / 1000, "CASHMTA"),
        )
    )


def _mp_world_distress_nimtaavg(panel: pl.DataFrame) -> pl.DataFrame:
    """NIMTAAVG: per-(excntry, eom) mean fill (>= 10 obs), then 4-quarter
    geometrically weighted average over consecutive months."""
    nimta_mean = (
        panel.filter(col("NIMTA").is_not_null())
        .group_by("excntry", "eom")
        .agg(mean_NIMTA=col("NIMTA").mean(), _f=pl.len())
        .filter(col("_f") >= 10)
    )
    scale_n = (1 - MP_CHS_DECAY**3) / (1 - MP_CHS_DECAY**12)
    return (
        panel.join(
            nimta_mean.select("excntry", "eom", "mean_NIMTA"), on=["excntry", "eom"], how="left"
        )
        .with_columns(adj_NIMTA=pl.coalesce(col("NIMTA"), col("mean_NIMTA")))
        .drop("mean_NIMTA")
        .sort("id", "eom")
        .with_columns(
            nimta_lag0=col("adj_NIMTA"),
            nimta_lag3=col("adj_NIMTA").shift(3).over("id"),
            nimta_lag6=col("adj_NIMTA").shift(6).over("id"),
            nimta_lag9=col("adj_NIMTA").shift(9).over("id"),
            eom_lag3=col("eom").shift(3).over("id"),
            eom_lag6=col("eom").shift(6).over("id"),
            eom_lag9=col("eom").shift(9).over("id"),
        )
        .with_columns(
            nimta_ok=(
                (col("eom_lag3") == _mp_offset_months("eom", -3))
                & (col("eom_lag6") == _mp_offset_months("eom", -6))
                & (col("eom_lag9") == _mp_offset_months("eom", -9))
                & col("nimta_lag0").is_not_null()
                & col("nimta_lag3").is_not_null()
                & col("nimta_lag6").is_not_null()
                & col("nimta_lag9").is_not_null()
            )
        )
        .with_columns(
            NIMTAAVG=pl.when(col("nimta_ok"))
            .then(
                scale_n
                * (
                    col("nimta_lag0")
                    + col("nimta_lag3") * MP_CHS_DECAY
                    + col("nimta_lag6") * MP_CHS_DECAY**2
                    + col("nimta_lag9") * MP_CHS_DECAY**3
                )
            )
            .otherwise(None)
        )
    )


def _mp_world_distress_exretavg(panel: pl.DataFrame) -> pl.DataFrame:
    """EXRETAVG: per-(excntry, eom) mean fill, then 12-month geometrically
    weighted average over consecutive months."""
    exret_mean = (
        panel.filter(col("EXRET").is_not_null())
        .group_by("excntry", "eom")
        .agg(mean_EXRET=col("EXRET").mean())
    )
    scale_e = (1 - MP_CHS_DECAY) / (1 - MP_CHS_DECAY**12)
    consec = pl.lit(True)
    for i in range(12):
        consec = consec & (col(f"eoml{i}") == _mp_offset_months("eom", -i))
        consec = consec & col(f"el{i}").is_not_null()
    weighted = sum(col(f"el{i}") * (MP_CHS_DECAY**i) for i in range(12))
    return (
        panel.join(exret_mean, on=["excntry", "eom"], how="left")
        .with_columns(adj_EXRET=pl.coalesce(col("EXRET"), col("mean_EXRET")))
        .drop("mean_EXRET")
        .sort("id", "eom")
        .with_columns(
            [col("adj_EXRET").shift(i).over("id").alias(f"el{i}") for i in range(12)]
            + [col("eom").shift(i).over("id").alias(f"eoml{i}") for i in range(12)]
        )
        .with_columns(EXRETAVG=pl.when(consec).then(scale_e * weighted).otherwise(None))
    )


def _mp_world_distress_sigma(
    world_daily: pl.DataFrame, world_monthly: pl.DataFrame
) -> pl.DataFrame:
    """SIGMA: annualized 3-month rolling daily volatility from world_dsf,
    cross-sectionally mean-filled per (excntry, eom)."""
    daily = world_daily.with_columns(eom=col("date").dt.month_end())
    monthly = (
        daily.with_columns(ret2=col("ret") ** 2)
        .group_by("id", "eom")
        .agg(sum_ret2=col("ret2").sum(), obs_m=col("ret").is_not_null().sum().cast(pl.Int64))
        .sort("id", "eom")
    )
    expanded = (
        monthly.select("id", "eom")
        .with_columns(_off=pl.lit([1, 2, 3]))
        .explode("_off")
        .with_columns(
            eom_src=col("eom").dt.offset_by(pl.format("-{}mo", col("_off"))).dt.month_end()
        )
    )
    sigma_agg = (
        expanded.join(
            monthly.select("id", col("eom").alias("eom_src"), "sum_ret2", "obs_m"),
            on=["id", "eom_src"],
            how="left",
        )
        .group_by("id", "eom")
        .agg(sum_total=col("sum_ret2").sum(), obs_total=col("obs_m").sum().cast(pl.Int64))
    )
    sigma = (
        monthly.join(sigma_agg, on=["id", "eom"], how="left")
        .with_columns(
            SIGMA=pl.when(col("obs_total") > 0)
            .then((safe_div(252.0, col("obs_total"), "_s") * col("sum_total")).sqrt())
            .otherwise(None)
        )
        .select("id", "eom", "SIGMA")
    )

    # Cross-sectional fill per (excntry, eom). Attach excntry first.
    sigma_with_x = sigma.join(
        world_monthly.select("id", "eom", "excntry")
        .sort(["id", "eom", "excntry"])
        .unique(["id", "eom"], keep="first", maintain_order=True),
        on=["id", "eom"],
        how="left",
    )
    sigma_mean = (
        sigma_with_x.filter(col("SIGMA").is_not_null())
        .group_by("excntry", "eom")
        .agg(avg_SIGMA=col("SIGMA").mean())
    )
    return (
        sigma_with_x.join(sigma_mean, on=["excntry", "eom"], how="left")
        .with_columns(SIGMA=pl.coalesce(col("SIGMA"), col("avg_SIGMA")))
        .select("id", "eom", "SIGMA")
    )


def _mp_world_distress_final_score(panel: pl.DataFrame, sigma: pl.DataFrame) -> pl.DataFrame:
    """Lag MB/TLMTA/CASHMTA (gap-gated), winsorize per (excntry, eom) at
    5/95, require all inputs, apply CHS coefficients."""
    chs_inputs = panel.select(
        "id", "excntry", "eom", "NIMTAAVG", "TLMTA", "EXRETAVG", "RSIZE", "CASHMTA", "MB", "PRICE"
    )
    scored = (
        chs_inputs.join(sigma, on=["id", "eom"], how="left")
        .sort("id", "eom")
        .with_columns(
            eom_prev=col("eom").shift(1).over("id"),
            lag_MB=col("MB").shift(1).over("id"),
            lag_TLMTA=col("TLMTA").shift(1).over("id"),
            lag_CASHMTA=col("CASHMTA").shift(1).over("id"),
        )
        .with_columns(gap_m=_mp_month_diff(col("eom"), col("eom_prev")))
        .with_columns(
            [
                pl.when(col("gap_m") >= 6).then(None).otherwise(col(c)).alias(c)
                for c in ["lag_MB", "lag_TLMTA", "lag_CASHMTA"]
            ]
        )
    )

    # Winsorize per (excntry, eom)
    wins_cols = ["NIMTAAVG", "lag_TLMTA", "EXRETAVG", "RSIZE", "lag_CASHMTA", "lag_MB", "SIGMA"]
    bounds = _mp_polars_sql(
        scored,
        f"""SELECT excntry, eom,
                  {", ".join(f"quantile_disc({c}, 0.05) AS _{c}_lo, quantile_disc({c}, 0.95) AS _{c}_hi" for c in wins_cols)}
            FROM src GROUP BY excntry, eom""",
    )
    return (
        scored.join(bounds, on=["excntry", "eom"], how="left")
        .with_columns(
            [
                pl.when(col(c) > col(f"_{c}_hi"))
                .then(col(f"_{c}_hi"))
                .when((col(c) < col(f"_{c}_lo")) & col(c).is_not_null())
                .then(col(f"_{c}_lo"))
                .otherwise(col(c))
                .alias(c)
                for c in wins_cols
            ]
        )
        .drop([f"_{c}_lo" for c in wins_cols] + [f"_{c}_hi" for c in wins_cols])
        .filter(
            col("NIMTAAVG").is_not_null()
            & col("lag_TLMTA").is_not_null()
            & col("EXRETAVG").is_not_null()
            & col("RSIZE").is_not_null()
            & col("lag_CASHMTA").is_not_null()
            & col("lag_MB").is_not_null()
            & col("PRICE").is_not_null()
            & col("SIGMA").is_not_null()
        )
        .with_columns(
            distress=MP_DISTRESS_BETAS["intercept"]
            + MP_DISTRESS_BETAS["NIMTAAVG"] * col("NIMTAAVG")
            + MP_DISTRESS_BETAS["TLMTA"] * col("lag_TLMTA")
            + MP_DISTRESS_BETAS["EXRETAVG"] * col("EXRETAVG")
            + MP_DISTRESS_BETAS["SIGMA"] * col("SIGMA")
            + MP_DISTRESS_BETAS["RSIZE"] * col("RSIZE")
            + MP_DISTRESS_BETAS["CASHMTA"] * col("lag_CASHMTA")
            + MP_DISTRESS_BETAS["MB"] * col("lag_MB")
            + MP_DISTRESS_BETAS["PRICE"] * col("PRICE")
        )
        .select("eom", "id", "excntry", "distress")
    )


def _mp_world_compute_distress(
    world_monthly: pl.DataFrame,
    world_daily: pl.DataFrame,
    market_m: pl.DataFrame,
    mp_con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
) -> pl.DataFrame:
    """World CHS DISTRESS (same MP_DISTRESS_BETAS as US). Substitutes
    per-excntry market returns for msp500, world_dsf for crsp_daily, and
    comp_g_fundq + comp_fundq (unioned, gvkey→id via world_msf) for CRSP
    quarterly. Returns (eom, id, excntry, distress)."""
    market_inputs = _mp_world_distress_market_inputs(world_monthly, market_m)

    if "gvkey" not in world_monthly.columns:
        raise RuntimeError("world wm must carry gvkey for distress linkage")
    id_gvkey = (
        world_monthly.select("id", "gvkey", "eom")
        .sort(["id", "eom", "gvkey"])
        .unique(subset=["id", "eom"], keep="first", maintain_order=True)
    )

    fundq = _mp_world_distress_fundq(raw_dir)
    be = _mp_world_distress_book_equity_mb(fundq, world_monthly)
    return (
        _mp_world_distress_quarterly_join(market_inputs, id_gvkey, be, fundq, mp_con)
        .pipe(_mp_world_distress_nimtaavg)
        .pipe(_mp_world_distress_exretavg)
        .pipe(
            _mp_world_distress_final_score,
            _mp_world_distress_sigma(world_daily, world_monthly),
        )
    )


def _mp_world_build_anomalies_panel(
    world_monthly: pl.DataFrame,
    world_data: pl.DataFrame,
    market_m: pl.DataFrame,
    world_daily: pl.DataFrame,
    mp_con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
) -> dict[str, pl.DataFrame]:
    """Assemble per-anomaly (id, eom, excntry, value) frames into a unified panel."""
    anomalies = _mp_world_compute_jkp_anomalies(world_data)
    anomalies["MOMENTUM"] = _mp_world_compute_momentum(world_monthly)
    anomalies["COMPOSITE_ISSUE"] = _mp_world_compute_composite_issue(world_monthly)
    anomalies["STOCK_ISSUE"] = _mp_world_compute_stock_issue(world_monthly)
    anomalies["DISTRESS"] = _mp_world_compute_distress(
        world_monthly, world_daily, market_m, mp_con, raw_dir
    )
    return anomalies


def _mp_world_monthly_lagged_me_panel(world_monthly: pl.DataFrame) -> pl.DataFrame:
    """Lag-ME panel for world: id, eom, excntry, prc, ret, ME, mktcap, lag_prc."""
    return (
        world_monthly.select("id", "eom", "excntry", "prc", "ret", "me")
        .rename({"prc": "prc", "ret": "ret", "me": "ME"})
        .with_columns(prc=col("prc").abs())
        .sort("id", "eom")
        .with_columns(
            mktcap=col("ME").shift(1).over("id"),
            lag_prc=col("prc").shift(1).over("id"),
        )
        .filter(
            col("ret").is_not_null() & col("mktcap").is_not_null() & col("lag_prc").is_not_null()
        )
        .select("eom", "id", "excntry", "prc", "ret", "ME", "mktcap", "lag_prc")
    )


def _mp_world_percentile_rank_anomalies(
    me_panel: pl.DataFrame,
    anomalies: dict[str, pl.DataFrame],
    min_stks: int,
    mp_con: duckdb.DuckDBPyConnection,
) -> pl.DataFrame:
    """Per-(excntry, eom) percentile rank for each world anomaly value."""

    def anom_source(anom: str) -> pl.DataFrame | None:
        if anom not in anomalies:
            return None
        a = anomalies[anom]
        val_col = [c for c in a.columns if c not in ("id", "excntry", "eom")][0]
        return a.select("id", "eom", col(val_col).alias("rank_var"))

    return _mp_anomaly_pct_panel(
        me_panel,
        min_stks,
        mp_con,
        panel_cols=["id", "excntry", "eom", "ret", "lag_prc", "mktcap"],
        sort_keys=["id", "eom"],
        id_cols=["id"],
        group_keys=["excntry", "eom"],
        anom_source=anom_source,
    )


def _mp_world_build_mispricing_panel(
    world_monthly: pl.DataFrame,
    world_data: pl.DataFrame,
    market_m: pl.DataFrame,
    world_daily: pl.DataFrame,
    min_stks: int,
    mp_con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
) -> pl.DataFrame:
    anomalies = _mp_world_build_anomalies_panel(
        world_monthly, world_data, market_m, world_daily, mp_con, raw_dir
    )
    panel = _mp_world_percentile_rank_anomalies(
        _mp_world_monthly_lagged_me_panel(world_monthly), anomalies, min_stks, mp_con
    )
    pct_cols = [f"pct_{a}" for a in MP_ANOMALY_LIST]
    return (
        panel.with_columns(
            anomal_num=sum(
                col(c).is_not_null().cast(pl.Int64) for c in pct_cols if c in panel.columns
            )
        )
        .filter(col("anomal_num") > 0)
        .with_columns([col(c).fill_null(0.0) for c in pct_cols if c in panel.columns])
        .sort("eom", "excntry", "id")
    )


def _mp_concat_write(
    us: pl.DataFrame, world: pl.DataFrame, cols: list[str], sort_keys: list[str], path: Path
) -> None:
    """Concat the US and world frames on a shared column set, sort, write."""
    pl.concat([us.select(*cols), world.select(*cols)], how="vertical_relaxed").sort(
        *sort_keys
    ).write_parquet(path)


@measure_time
def gen_mispricing_data(
    paths: DataPaths,
    monthly_factors_path: Path,
    daily_factors_path: Path,
    chars_path: Path,
    freqs: tuple[str, ...] = ("monthly", "daily"),
    min_stks: int = 30,
    min_fcts: int = 3,
    min_obs: int = 10,
    min_stks_world: int = MP_MIN_STKS_BP_WORLD,
    min_obs_world: int = MP_MIN_OBS_PF_WORLD,
) -> None:
    """Stambaugh-Yuan 11-anomaly mispricing factor pipeline (incl. CHS DISTRESS), US + WORLD.

    Reference: Stambaugh and Yuan (2017) 'Mispricing factors', Review of
    Financial Studies 30(4), pp. 1270-1315.

    Ported from the MisprProject standalone. Always runs both branches;
    outputs concatenated with `excntry` column. Reads jkp's CIZ raws from
    `paths.raw_tables_dir` ({crsp_msf_v2, crsp_dsf_v2, ccmxpf_lnkhist, funda,
    fundq, crsp_a_indexes_msp500, crsp_a_indexes_acti}.parquet) and world
    inputs from `paths.interim_dir`; raw_dir/interim_dir are threaded
    explicitly through every _mp_* helper that does IO. The 3 output paths
    are passed in explicitly:
      <monthly_factors_path>  — eom, excntry, smb_mispricing, mispricing_mgmt, mispricing_perf
      <daily_factors_path>    — date, excntry, smb_mispricing, mispricing_mgmt, mispricing_perf (US + world)
      <chars_path>            — id, eom, mispricing_mgmt, mispricing_perf
    """
    # _mp_load_funda/_mp_load_fundq are @functools.cache'd on raw_dir; clear so repeated
    # calls with different paths in one process don't reuse stale frames.
    _mp_load_funda.cache_clear()
    _mp_load_fundq.cache_clear()
    raw_dir = paths.raw_tables_dir
    interim_dir = paths.interim_dir
    mp_con = duckdb.connect()
    mp_con_world = duckdb.connect()
    try:
        mp_con.execute("PRAGMA disable_progress_bar")
        mp_con_world.execute("PRAGMA disable_progress_bar")

        # 1. Return/ME panels: US to interim parquets, world in memory
        _mp_build_crsp_monthly(raw_dir, interim_dir)
        _mp_build_crsp_daily(raw_dir, interim_dir)
        world_data = _mp_world_load_world_data(interim_dir)
        world_daily = _mp_world_load_world_dsf(interim_dir)
        market_monthly = _mp_world_load_market(interim_dir, daily=False)

        # 2. US fundamentals (CCM-linked) to interim parquets
        _mp_build_compustat_annual(mp_con, raw_dir, interim_dir)
        _mp_build_compustat_quarterly(mp_con, raw_dir, interim_dir)

        # 3. US anomaly characteristics (11 anomalies + CHS DISTRESS),
        # one interim parquet each; world chars are computed in-memory inside
        # the world panel build below
        _mp_compute_accrual(raw_dir, interim_dir)
        _mp_compute_gross_profit(raw_dir, interim_dir)
        _mp_compute_oscore(raw_dir, interim_dir)
        _mp_compute_nsi_ag_inv_noa(raw_dir, interim_dir)
        _mp_compute_momentum(interim_dir)
        _mp_compute_composite_issue(interim_dir)
        _mp_compute_roa(mp_con, interim_dir)
        _mp_compute_distress(mp_con, raw_dir, interim_dir)

        # 4. Per-stock panels, 2x3 double-sort legs, SMB panels
        us_panel = _mp_build_mispricing_panel(min_stks, mp_con, interim_dir)
        # US size median is NYSE-only (exchcd == 1), per-eom breaks.
        us_legs = _mp_mispricing_legs(
            us_panel,
            min_fcts,
            group_keys=["eom"],
            size_break_sql="CASE WHEN exchcd = 1 THEN mktcap END",
        )
        us_smb = _mp_smb_panel_from_legs(us_legs, key_cols=("permno",))
        world_monthly = world_data.select(
            "id", "permno", "gvkey", "excntry", "eom", "me", "prc", "ret", "adjfct", "shares"
        )
        world_panel = _mp_world_build_mispricing_panel(
            world_monthly,
            world_data,
            market_monthly,
            world_daily,
            min_stks_world,
            mp_con_world,
            raw_dir,
        )
        world_legs = _mp_mispricing_legs(
            world_panel, min_fcts, group_keys=["excntry", "eom"], size_break_sql="mktcap"
        )
        world_smb = _mp_smb_panel_from_legs(world_legs, key_cols=("id", "excntry"))

        # 5. Factor returns and per-stock scores
        # 6. Concat US + world; write outputs (per requested frequency)
        factor_cols = ["smb_mispricing", "mispricing_mgmt", "mispricing_perf"]
        if "monthly" in freqs:
            monthly_us = _mp_monthly_factor_returns(
                us_legs,
                us_smb,
                min_obs,
                by="eom",
                truncate_thin=True,
                out_cols=["eom"],
                out_sort=["eom"],
            ).with_columns(excntry=pl.lit("USA"))
            monthly_world = _mp_monthly_factor_returns(
                world_legs,
                world_smb,
                min_obs_world,
                by=["excntry", "eom"],
                truncate_thin=False,
                out_cols=["eom", "excntry"],
                out_sort=["excntry", "eom"],
            )
            _mp_concat_write(
                monthly_us,
                monthly_world,
                ["eom", "excntry", *factor_cols],
                ["excntry", "eom"],
                monthly_factors_path,
            )
            # Keep `id` as Int64 to match jkp convention (world_msf.id = BIGINT for both
            # US permnos and Compustat-derived ids per `combine_crsp_comp_sf`). Casting to
            # String would break outer-joins in `ap_factor_model_data` against ff/hxz chars.
            chars_us = _mp_stock_mispricing_scores(
                us_panel,
                min_fcts,
                sort_keys=["permno", "eom"],
                select_cols=[col("permno").alias("id"), "eom"],
            ).with_columns(excntry=pl.lit("USA"), id=col("id").cast(pl.Int64))
            chars_world = _mp_stock_mispricing_scores(
                world_panel,
                min_fcts,
                sort_keys=["id", "eom"],
                select_cols=["id", "eom", "excntry"],
            ).with_columns(id=col("id").cast(pl.Int64))
            _mp_concat_write(
                chars_us,
                chars_world,
                ["id", "eom", "mispricing_mgmt", "mispricing_perf"],
                ["id", "eom"],
                chars_path,
            )
        if "daily" in freqs:
            daily_us = _mp_us_daily_factor_returns(
                us_legs, us_smb, min_obs, mp_con, interim_dir
            ).with_columns(excntry=pl.lit("USA"))
            daily_world = _mp_world_daily_factor_returns(
                world_legs, world_smb, min_obs_world, mp_con_world, interim_dir
            )
            _mp_concat_write(
                daily_us,
                daily_world,
                ["date", "excntry", *factor_cols],
                ["excntry", "date"],
                daily_factors_path,
            )
    finally:
        mp_con.close()
        mp_con_world.close()


def impute_high_low(df):
    """
    Description:
        Clean/impute daily high/low prices using forward-filled anchors and logical bounds.

    Steps:
        1) Null out impossible HL (bid-ask flag, zero volume, nonpositive or equal HL).
        2) Initialize rolling anchors prc_low_r/prc_high_r; update when valid HL arrives.
        3) Forward-fill anchors; replace HL when price falls within anchor band or violates it; record hlreset reason.
        4) Null extreme spreads where high/low ratio > 8.

    Output:
        LazyFrame with corrected prc_low/prc_high (+ diagnostics).
    """

    sc1 = (
        (col("bidask") == 1)
        | (col("tvol") == 0)
        | (col("prc_low") <= 0)
        | (col("prc_high") <= 0)
        | (col("prc_low") == col("prc_high"))
    )
    sc2 = (col("prc_low") > 0) & (col("prc_high") > col("prc_low"))
    sc3 = (col("flag_replace")) & (
        (col("prc") >= col("prc_low_r")) & (col("prc") <= col("prc_high_r"))
        | (col("prc").is_null() & col("prc_low_r").is_null() & col("prc_high_r").is_null())
    )
    sc4 = (col("flag_replace")) & (col("prc") < col("prc_low_r"))
    sc5 = (col("flag_replace")) & (col("prc") > col("prc_high_r")).fill_null(True)
    sc6 = (col("prc_low") != 0) & (col("prc_high") / col("prc_low") > 8)
    df = (
        df.sort(["id", "date"])
        .with_columns(
            prc_low_in=col("prc_low"),
            prc_high_in=col("prc_high"),
            prc_high=pl.when(sc1).then(None).otherwise(col("prc_high")),
            prc_low=pl.when(sc1).then(None).otherwise(col("prc_low")),
            count=pl.cum_count("id").over("id"),
            hlreset=pl.lit(0),
        )
        .with_columns(
            prc_low_r=(pl.when(col("count") == 1).then(None).otherwise(col("prc_low"))),
            prc_high_r=(pl.when(col("count") == 1).then(None).otherwise(col("prc_high"))),
        )
        .with_columns(
            prc_low_r=(pl.when(sc2).then(col("prc_low")).otherwise(col("prc_low_r"))),
            prc_high_r=(pl.when(sc2).then(col("prc_high")).otherwise(col("prc_high_r"))),
        )
        .sort(["id", "date"])
        .with_columns(
            flag_replace=col("prc_low_r").is_null(),
            prc_low_r=col("prc_low_r").forward_fill().over("id"),
            prc_high_r=col("prc_high_r").forward_fill().over("id"),
        )
        .with_columns(
            prc_low=(
                pl.when(sc3)
                .then(col("prc_low_r"))
                .when(sc4)
                .then(col("prc"))
                .when(sc5)
                .then(col("prc_low_r") + (col("prc") - col("prc_high_r")))
                .otherwise(col("prc_low"))
            ),
            prc_high=(
                pl.when(sc3)
                .then(col("prc_high_r"))
                .when(sc4)
                .then(col("prc_high_r") - (col("prc_low_r") - col("prc")))
                .when(sc5)
                .then(col("prc"))
                .otherwise(col("prc_high"))
            ),
            hlreset=(
                pl.when(sc3)
                .then(pl.lit(1))
                .when(sc4)
                .then(pl.lit(2))
                .when(sc5)
                .then(pl.lit(3))
                .otherwise(col("hlreset"))
            ),
        )
        .with_columns(
            prc_low=pl.when(sc6).then(None).otherwise(col("prc_low")),
            prc_high=pl.when(sc6).then(None).otherwise(col("prc_high")),
        )
    )

    return df


def adjust_overnight_returns(df):
    """
    Description:
        Adjust today's HL using prior close if it lies outside today’s reported range.

    Steps:
        1) Add lagged HL and prior close (prc_l1).
        2) If prc_l1 < low: raise low to prc_l1 and shrink high by the gap; vice versa if prc_l1 > high.
        3) Build 2-day envelope: prc_high_2d = max(today_t, lag1), prc_low_2d = min(today_t, lag1).

    Output:
        LazyFrame with prc_low_t/prc_high_t and 2-day bounds (prc_low_2d/prc_high_2d).
    """
    sc1 = (col("prc_l1") < col("prc_low")) & (col("prc_l1") > 0)
    sc2 = (col("prc_l1") > col("prc_high")) & (col("prc_l1") > 0)
    df = (
        df.sort(["id", "date"])
        .with_columns(
            prc_low_t=col("prc_low"),
            prc_high_t=col("prc_high"),
            prc_low_l1=col("prc_low").shift(1).over("id"),
            prc_high_l1=col("prc_high").shift(1).over("id"),
            prc_l1=col("prc").shift(1).over("id"),
        )
        .with_columns(
            prc_high_t=pl.when(sc1)
            .then(col("prc_high") - (col("prc_low") - col("prc_l1")))
            .otherwise(col("prc_high_t")),
            prc_low_t=pl.when(sc1).then(col("prc_l1")).otherwise(col("prc_low_t")),
        )
        .with_columns(
            prc_high_t=pl.when(sc2).then(col("prc_l1")).otherwise(col("prc_high_t")),
            prc_low_t=pl.when(sc2)
            .then(col("prc_low") + (col("prc_l1") - col("prc_high")))
            .otherwise(col("prc_low_t")),
        )
        .with_columns(
            prc_high_2d=pl.max_horizontal("prc_high_t", "prc_high_l1"),
            prc_low_2d=pl.min_horizontal("prc_low_t", "prc_low_l1"),
        )
    )
    return df


def compute_bidask_spread(df, __min_obs):
    """
    Description:
        Compute daily bid-ask spread and HL-based volatility, then monthly-average.

    Steps:
        1) Derive beta, gamma from log HL ranges (today and 2-day); compute alpha.
        2) Map to spread and sigma; clamp negatives to 0.
        3) Group by (id, eom); take 21-day means; require ≥ __min_obs days.

    Output:
        LazyFrame with [id, eom, bidaskhl_21d, rvolhl_21d].
    """

    pi = 3.141592653589793
    k2 = sqrt(8 / pi)
    const = 3 - 2 * sqrt(2)
    df = (
        df.with_columns(
            beta=(
                pl.when((col("prc_low_t") > 0) & (col("prc_low_l1") > 0))
                .then(
                    ((col("prc_high_t") / col("prc_low_t")).log() ** 2)
                    + ((col("prc_high_l1") / col("prc_low_l1")).log() ** 2)
                )
                .otherwise(fl_none())
            ),
            gamma=pl.when(col("prc_low_2d") > 0)
            .then((col("prc_high_2d") / col("prc_low_2d")).log() ** 2)
            .otherwise(fl_none()),
        )
        .with_columns(
            alpha=((sqrt(2) - 1) * col("beta").sqrt()) / const - (col("gamma") / const).sqrt()
        )
        .with_columns(
            spread=2
            * (pl.lit(exp(1)).pow(col("alpha")) - 1)
            / (1 + pl.lit(exp(1)).pow(col("alpha"))),
            sigma=(((col("beta") / 2).sqrt() - col("beta").sqrt()) / (k2 * const))
            + (col("gamma") / (k2 * k2 * const)).sqrt(),
        )
        .with_columns(
            spread_0=pl.when(col("spread") < 0).then(pl.lit(0.0)).otherwise(col("spread")),
            sigma_0=pl.when(col("sigma") < 0).then(pl.lit(0.0)).otherwise(col("sigma")),
        )
        .group_by(["id", "eom"])
        .agg(
            bidaskhl_21d=pl.mean("spread_0"),
            rvolhl_21d=pl.mean("sigma_0"),
            count=pl.count("spread_0"),
        )
        .filter(col("count") > __min_obs)
        .drop("count")
        .sort(["id", "eom"])
    )
    return df


@measure_time
def bidask_hl(paths: DataPaths, output_path, data_path, market_returns_daily_path, __min_obs):
    """
    Description:
        End-to-end HL-based bid-ask and volatility factors from daily prices and market returns.

    Steps:
        1) Read daily stock data; join daily market returns; deflate prices by adjfct.
        2) Impute HL, adjust using prior close, compute spread/vol with monthly aggregation.
        3) Write parquet.

    Output:
        '{output_path}' with monthly [id, eom, bidaskhl_21d, rvolhl_21d].
    """
    # `paths` is retained for the path-explicit convention but not used here:
    # all inputs and the output are passed as fully-formed absolute paths.
    market_returns_daily = pl.scan_parquet(market_returns_daily_path)
    __dsf = (
        pl.scan_parquet(data_path)
        .join(market_returns_daily, how="left", on=["excntry", "date"])
        .filter(col("mkt_vw_exc").is_not_null())
        .with_columns([safe_div(var, "adjfct", var) for var in ["prc", "prc_high", "prc_low"]])
    )
    bdhl = (
        __dsf.pipe(impute_high_low)
        .pipe(adjust_overnight_returns)
        .pipe(compute_bidask_spread, __min_obs=__min_obs)
    )
    bdhl.collect().write_parquet(output_path)


@measure_time
def create_world_data_prelim(
    paths: DataPaths, msf_path, market_chars_monthly_path, acc_chars_world_path, output_path
):
    """
    Description:
        Build preliminary world dataset by merging returns, market characteristics, and accounting data.

    Steps:
        1) Load msf (stock returns), monthly market chars, and accounting chars parquet files.
        2) Left-join msf with market chars on (id, eom).
        3) Left-join with accounting chars on (gvkey, eom vs. public_date).
        4) Drop dividend-related fields and source flag.

    Output:
        '{output_path}' parquet with merged stock, market, and accounting data.
    """
    # `paths` is retained for the path-explicit convention but not used here:
    # all inputs and the output are passed as fully-formed absolute paths.
    a = pl.scan_parquet(msf_path)
    b = pl.scan_parquet(market_chars_monthly_path)
    c = pl.scan_parquet(acc_chars_world_path)
    world_data_prelim = (
        a.join(b, how="left", on=["id", "eom"])
        .join(c, how="left", left_on=["gvkey", "eom"], right_on=["gvkey", "public_date"])
        .drop(["div_tot", "div_cash", "div_spc", "source"])
    )
    world_data_prelim.collect().write_parquet(output_path)
    # Streaming can be used here if needed
    # world_data_prelim.collect(streaming = True).write_parquet('world_data_prelim.parquet')


def acc_chars_list():
    acc_chars = [
        # Accounting Based Size Measures
        "assets",
        "sales",
        "book_equity",
        "net_income",
        "enterprise_value",
        # 1yr Growth
        "at_gr1",
        "ca_gr1",
        "nca_gr1",
        "lt_gr1",
        "cl_gr1",
        "ncl_gr1",
        "be_gr1",
        "pstk_gr1",
        "debt_gr1",
        "sale_gr1",
        "cogs_gr1",
        "sga_gr1",
        "opex_gr1",
        # 3yr Growth
        "at_gr3",
        "ca_gr3",
        "nca_gr3",
        "lt_gr3",
        "cl_gr3",
        "ncl_gr3",
        "be_gr3",
        "pstk_gr3",
        "debt_gr3",
        "sale_gr3",
        "cogs_gr3",
        "sga_gr3",
        "opex_gr3",
        # 1yr Growth Scaled by Assets
        "cash_gr1a",
        "inv_gr1a",
        "rec_gr1a",
        "ppeg_gr1a",
        "lti_gr1a",
        "intan_gr1a",
        "debtst_gr1a",
        "ap_gr1a",
        "txp_gr1a",
        "debtlt_gr1a",
        "txditc_gr1a",
        "coa_gr1a",
        "col_gr1a",
        "cowc_gr1a",
        "ncoa_gr1a",
        "ncol_gr1a",
        "nncoa_gr1a",
        "oa_gr1a",
        "ol_gr1a",
        "noa_gr1a",
        "fna_gr1a",
        "fnl_gr1a",
        "nfna_gr1a",
        "gp_gr1a",
        "ebitda_gr1a",
        "ebit_gr1a",
        "ope_gr1a",
        "ni_gr1a",
        "nix_gr1a",
        "dp_gr1a",
        "ocf_gr1a",
        "fcf_gr1a",
        "nwc_gr1a",
        "eqnetis_gr1a",
        "dltnetis_gr1a",
        "dstnetis_gr1a",
        "dbnetis_gr1a",
        "netis_gr1a",
        "fincf_gr1a",
        "eqnpo_gr1a",
        "tax_gr1a",
        "div_gr1a",
        "eqbb_gr1a",
        "eqis_gr1a",
        "eqpo_gr1a",
        "capx_gr1a",
        # 3yr Growth Scaled by Assets
        "cash_gr3a",
        "inv_gr3a",
        "rec_gr3a",
        "ppeg_gr3a",
        "lti_gr3a",
        "intan_gr3a",
        "debtst_gr3a",
        "ap_gr3a",
        "txp_gr3a",
        "debtlt_gr3a",
        "txditc_gr3a",
        "coa_gr3a",
        "col_gr3a",
        "cowc_gr3a",
        "ncoa_gr3a",
        "ncol_gr3a",
        "nncoa_gr3a",
        "oa_gr3a",
        "ol_gr3a",
        "fna_gr3a",
        "fnl_gr3a",
        "nfna_gr3a",
        "gp_gr3a",
        "ebitda_gr3a",
        "ebit_gr3a",
        "ope_gr3a",
        "ni_gr3a",
        "nix_gr3a",
        "dp_gr3a",
        "ocf_gr3a",
        "fcf_gr3a",
        "nwc_gr3a",
        "eqnetis_gr3a",
        "dltnetis_gr3a",
        "dstnetis_gr3a",
        "dbnetis_gr3a",
        "netis_gr3a",
        "fincf_gr3a",
        "eqnpo_gr3a",
        "tax_gr3a",
        "div_gr3a",
        "eqbb_gr3a",
        "eqis_gr3a",
        "eqpo_gr3a",
        "capx_gr3a",
        # Investment
        "capx_at",
        "rd_at",
        # Profitability
        "gp_sale",
        "ebitda_sale",
        "ebit_sale",
        "pi_sale",
        "ni_sale",
        "nix_sale",
        "ocf_sale",
        "fcf_sale",
        # Return on Assets
        "gp_at",
        "ebitda_at",
        "ebit_at",
        "fi_at",
        "cop_at",
        # Return on Book Equity
        "ope_be",
        "ni_be",
        "nix_be",
        "ocf_be",
        "fcf_be",
        # Return on Invested Capital
        "gp_bev",
        "ebitda_bev",
        "ebit_bev",
        "fi_bev",
        "cop_bev",
        # Return on Physical Capital
        "gp_ppen",
        "ebitda_ppen",
        "fcf_ppen",
        # Issuance
        "fincf_at",
        "netis_at",
        "eqnetis_at",
        "eqis_at",
        "dbnetis_at",
        "dltnetis_at",
        "dstnetis_at",
        # Equity Payout
        "eqnpo_at",
        "eqbb_at",
        "div_at",
        # Accruals
        "oaccruals_at",
        "oaccruals_ni",
        "taccruals_at",
        "taccruals_ni",
        "noa_at",
        # Capitalization/Leverage Ratios
        "be_bev",
        "debt_bev",
        "cash_bev",
        "pstk_bev",
        "debtlt_bev",
        "debtst_bev",
        "debt_mev",
        "pstk_mev",
        "debtlt_mev",
        "debtst_mev",
        # Financial Soundness Ratios
        "int_debtlt",
        "int_debt",
        "cash_lt",
        "inv_act",
        "rec_act",
        "ebitda_debt",
        "debtst_debt",
        "cl_lt",
        "debtlt_debt",
        "profit_cl",
        "ocf_cl",
        "ocf_debt",
        "lt_ppen",
        "debtlt_be",
        "fcf_ocf",
        "opex_at",
        "nwc_at",
        # Solvency Ratios
        "debt_at",
        "debt_be",
        "ebit_int",
        # Liquidity Ratios
        "cash_cl",
        "caliq_cl",
        "ca_cl",
        "inv_days",
        "rec_days",
        "ap_days",
        "cash_conversion",
        # Activity/Efficiency Ratio
        "inv_turnover",
        "at_turnover",
        "rec_turnover",
        "ap_turnover",
        # Non-Recurring Items
        "spi_at",
        "xido_at",
        "nri_at",
        # Miscellaneous
        "adv_sale",
        "staff_sale",
        "rd_sale",
        "div_ni",
        "sale_bev",
        "sale_be",
        "sale_nwc",
        "tax_pi",
        # Balance Sheet Fundamentals to Market Equity
        "be_me",
        "at_me",
        "cash_me",
        # Income Fundamentals to Market Equity
        "gp_me",
        "ebitda_me",
        "ebit_me",
        "ope_me",
        "ni_me",
        "nix_me",
        "sale_me",
        "ocf_me",
        "fcf_me",
        "cop_me",
        "rd_me",
        # Equity Payout/issuance to Market Equity
        "div_me",
        "eqbb_me",
        "eqis_me",
        "eqpo_me",
        "eqnpo_me",
        "eqnetis_me",
        # Debt Issuance to Market Enterprise Value
        "dltnetis_mev",
        "dstnetis_mev",
        "dbnetis_mev",
        # Firm Payout/issuance to Market Enterprise Value
        "netis_mev",
        # Balance Sheet Fundamentals to Market Enterprise Value
        "at_mev",
        "be_mev",
        "bev_mev",
        "ppen_mev",
        "cash_mev",
        # Income/CF Fundamentals to Market Enterprise Value
        "gp_mev",
        "ebitda_mev",
        "ebit_mev",
        "cop_mev",
        "sale_mev",
        "ocf_mev",
        "fcf_mev",
        "fincf_mev",
        # New Variables from HXZ
        "ni_inc8q",
        "ppeinv_gr1a",
        "lnoa_gr1a",
        "capx_gr1",
        "capx_gr2",
        "capx_gr3",
        "sti_gr1a",
        "niq_at",
        "niq_at_chg1",
        "niq_be",
        "niq_be_chg1",
        "saleq_gr1",
        "rd5_at",
        "dsale_dinv",
        "dsale_drec",
        "dgp_dsale",
        "dsale_dsga",
        "saleq_su",
        "niq_su",
        "debt_me",
        "netdebt_me",
        "capex_abn",
        "inv_gr1",
        "be_gr1a",
        "op_at",
        "pi_nix",
        "op_atl1",
        "gp_atl1",
        "ope_bel1",
        "cop_atl1",
        "at_be",
        "ocfq_saleq_std",
        "aliq_at",
        "aliq_mat",
        "tangibility",
        "eq_dur",
        "f_score",
        "o_score",
        "z_score",
        "kz_index",
        "intrinsic_value",
        "ival_me",
        "sale_emp_gr1",
        "emp_gr1",
        "cash_at",
        "earnings_variability",
        "ni_ar1",
        "ni_ivol",
        # New Variables not in HXZ
        "niq_saleq_std",
        "ni_emp",
        "sale_emp",
        "ni_at",
        "ocf_at",
        "ocf_at_chg1",
        "roeq_be_std",
        "roe_be_std",
        "gpoa_ch5",
        "roe_ch5",
        "roa_ch5",
        "cfoa_ch5",
        "gmar_ch5",
    ]
    return acc_chars


@measure_time
def finish_daily_chars(paths: DataPaths, output_path):
    """
    Description:
        Combine bid-ask spread and roll-based daily metrics into a final daily chars file.

    Steps:
        1) Load Corwin-Schultz and roll_apply_daily parquet files.
        2) Outer join on (id, eom).
        3) Add betabab (beta * rvol / mktvol) and rmax5_rvol ratio.
        4) Drop helper columns.

    Output:
        '{output_path}' parquet with final daily characteristics.
    """
    bidask = pl.scan_parquet(paths.interim_dir / "corwin_schultz.parquet")
    r1 = pl.scan_parquet(paths.interim_dir / "roll_apply_daily.parquet").with_columns(
        col("id").cast(pl.Int64)
    )
    daily_chars = bidask.join(r1, how="outer_coalesce", on=["id", "eom"])
    daily_chars = daily_chars.with_columns(
        betabab_1260d=col("corr_1260d") * col("rvol_252d") / col("__mktvol_252d"),
        rmax5_rvol_21d=col("rmax5_21d") / col("rvol_252d"),
    ).drop("__mktvol_252d")
    daily_chars.collect().write_parquet(output_path)


def z_ranks(data, var, __min, sort):
    order = sort != "ascending"
    exp_z_var = (col("rank") - pl.mean("rank").over(["excntry", "eom"])) / pl.std("rank").over(
        ["excntry", "eom"]
    )
    z_df = (
        data.filter(col(var).is_not_nan())
        .filter(pl.count(var).over(["excntry", "eom"]) >= __min)
        .with_columns(
            rank=col(var).rank(method="average", descending=order).over(["excntry", "eom"])
        )
        .select(["excntry", "id", "eom", exp_z_var.alias(f"z_{var}")])
        .with_columns(
            pl.when(col(f"z_{var}").is_nan())
            .then(fl_none())
            .otherwise(col(f"z_{var}"))
            .alias(f"z_{var}")
        )
        .filter(col(f"z_{var}").is_not_null())
    )
    return z_df


@measure_time
def quality_minus_junk(paths: DataPaths, data_path, min_stks):
    """
    Description:
        Compute standardized within-country z-scores for a variable.

    Steps:
        1) Rank variable by eom within each country (ascending/descending).
        2) Keep months with at least __min stocks.
        3) Standardize rank by mean/std within (excntry, eom).
        4) Clean NaN values.

    Output:
        LazyFrame with ['excntry','id','eom', z_{var}].
    """
    z_vars = [
        "gp_at",
        "ni_be",
        "ni_at",
        "ocf_at",
        "gp_sale",
        "oaccruals_at",
        "gpoa_ch5",
        "roe_ch5",
        "roa_ch5",
        "cfoa_ch5",
        "gmar_ch5",
        "betabab_1260d",
        "debt_at",
        "o_score",
        "z_score",
        "__evol",
    ]
    direction = [
        "ascending",
        "ascending",
        "ascending",
        "ascending",
        "ascending",
        "descending",
        "ascending",
        "ascending",
        "ascending",
        "ascending",
        "ascending",
        "descending",
        "descending",
        "descending",
        "ascending",
        "descending",
    ]
    cols = [
        "id",
        "eom",
        "excntry",
        "gp_at",
        "ni_be",
        "ni_at",
        "ocf_at",
        "gp_sale",
        "oaccruals_at",
        "gpoa_ch5",
        "roe_ch5",
        "roa_ch5",
        "cfoa_ch5",
        "gmar_ch5",
        "betabab_1260d",
        "debt_at",
        "o_score",
        "z_score",
        "roeq_be_std",
        "roe_be_std",
        pl.coalesce(2 * col("roeq_be_std"), "roe_be_std").alias("__evol"),
    ]
    c1 = (
        (col("common") == 1)
        & (col("primary_sec") == 1)
        & (col("obs_main") == 1)
        & (col("exch_main") == 1)
        & (col("ret_exc").is_not_null())
        & (col("me").is_not_null())
    )
    # NOTE: input must be unique on (excntry, eom, id) — guaranteed upstream by
    # construction of world_data_-1. Duplicate keys would fan out multiplicatively
    # across the 16+3 full joins below and panic at the Polars frame-length limit.
    qmj = pl.scan_parquet(data_path).filter(c1).select(cols).sort(["excntry", "eom"]).collect()
    for var_z, dir in zip(z_vars, direction, strict=True):
        __z = z_ranks(qmj, var_z, min_stks, dir)
        qmj = qmj.join(__z, how="left", on=["excntry", "eom", "id"])

    qmj = qmj.with_columns(
        __prof=pl.mean_horizontal(
            "z_gp_at", "z_ni_be", "z_ni_at", "z_ocf_at", "z_gp_sale", "z_oaccruals_at"
        ),
        __growth=pl.mean_horizontal(
            "z_gpoa_ch5", "z_roe_ch5", "z_roa_ch5", "z_cfoa_ch5", "z_gmar_ch5"
        ),
        __safety=pl.mean_horizontal(
            "z_betabab_1260d", "z_debt_at", "z_o_score", "z_z_score", "z___evol"
        ),
    ).select(["excntry", "id", "eom", "__prof", "__growth", "__safety"])

    ranks = {
        i: z_ranks(qmj, f"__{i}", min_stks, "ascending").rename({f"z___{i}": f"qmj_{i}"})
        for i in ["prof", "growth", "safety"]
    }
    qmj = (
        qmj.select(["excntry", "id", "eom"])
        .join(ranks["prof"], how="left", on=["excntry", "id", "eom"])
        .join(ranks["growth"], how="left", on=["excntry", "id", "eom"])
        .join(ranks["safety"], how="left", on=["excntry", "id", "eom"])
        .with_columns(__qmj=(col("qmj_prof") + col("qmj_growth") + col("qmj_safety")) / 3)
    )
    __qmj = z_ranks(qmj, "__qmj", min_stks, "ascending").rename({"z___qmj": "qmj"})
    qmj = qmj.join(__qmj, how="left", on=["excntry", "id", "eom"]).drop("__qmj")
    qmj.write_parquet(paths.interim_dir / "qmj.parquet")


def _main_filter_expr() -> pl.Expr:
    """Build a Polars filter expression from the MAIN_FILTERS config dict."""
    return functools.reduce(operator.and_, (pl.col(k) == v for k, v in MAIN_FILTERS.items()))


@measure_time
def filter_dsf(paths: DataPaths):
    """
    Description:
        Filter world_dsf to main securities.

    Steps:
        1) Load world_dsf.parquet.
        2) Filter to MAIN_FILTERS (primary_sec, common, obs_main, exch_main).
        3) Write world_dsf_output.parquet.

    Output:
        'world_dsf_output.parquet'.
    """
    pl.scan_parquet(paths.interim_dir / "world_dsf.parquet").filter(
        _main_filter_expr()
    ).sink_parquet(paths.interim_dir / "world_dsf_output.parquet")


@measure_time
def filter_msf(paths: DataPaths):
    """
    Description:
        Filter world_msf to main securities.

    Steps:
        1) Load world_msf.parquet.
        2) Filter to MAIN_FILTERS (primary_sec, common, obs_main, exch_main).
        3) Write world_msf_output.parquet.

    Output:
        'world_msf_output.parquet'.
    """
    pl.scan_parquet(paths.interim_dir / "world_msf.parquet").filter(
        _main_filter_expr()
    ).sink_parquet(paths.interim_dir / "world_msf_output.parquet")


@measure_time
def filter_world(paths: DataPaths):
    """
    Description:
        Filter world_data to main securities.

    Steps:
        1) Load world_data.parquet.
        2) Filter to MAIN_FILTERS (primary_sec, common, obs_main, exch_main).
        3) Write world_data_output.parquet.

    Output:
        'world_data_output.parquet'.
    """
    pl.scan_parquet(paths.interim_dir / "world_data.parquet").filter(
        _main_filter_expr()
    ).sink_parquet(paths.interim_dir / "world_data_output.parquet")


@measure_time
def save_main_data(paths: DataPaths) -> None:
    """
    Description:
        Compute lagged market equity and export country-level files.

    Steps:
        1) Load world_data_output.parquet (pre-filtered) and compute lagged market equity.
        2) Save dataset and split into country parquet files.

    Output:
        'world_data_output.parquet' and 'characteristics/{country}.parquet'.
    """
    world_data_output = paths.interim_dir / "world_data_output.parquet"
    world_data_output_temp = paths.interim_dir / "world_data_output_temp.parquet"
    months_exp = (col("eom").dt.year() * 12 + col("eom").dt.month()).cast(pl.Int64)
    data = (
        pl.scan_parquet(world_data_output)
        .with_columns(dif_aux=months_exp)
        .sort(["id", "eom"])
        .with_columns(
            me_lag1=col("me").shift(1).over("id"),
            dif_aux=(col("dif_aux") - col("dif_aux").shift(1)).over("id"),
        )
        .with_columns(
            me_lag1=pl.when(col("dif_aux") == 1).then(col("me_lag1")).otherwise(fl_none())
        )
        .drop("dif_aux")
    )
    data.select(pl.all().shrink_dtype()).sink_parquet(world_data_output_temp)
    os.replace(world_data_output_temp, world_data_output)

    out_dir = paths.processed_dir / "characteristics"
    con = duckdb.connect()
    con.execute(f"""
    COPY (SELECT * FROM read_parquet('{world_data_output.as_posix()}'))
    TO '{out_dir.as_posix()}'
    ( FORMAT PARQUET,
      COMPRESSION ZSTD,
      PARTITION_BY (excntry),
      WRITE_PARTITION_COLUMNS TRUE,
      OVERWRITE
      );
    """)
    con.close()
    for d in out_dir.glob("excntry=*"):
        if d.is_dir():
            country = d.name[len("excntry=") :]
            part_files = sorted(d.glob("*.parquet"))
            if part_files:
                part_files[0].rename(out_dir / f"{country}.parquet")
            shutil.rmtree(d)


@measure_time
def save_output_files(paths: DataPaths):
    """
    Description:
        Copy main market returns and cutoff files to Output folder.

    Steps:
        1) Copy parquet outputs from interim/ to other_output/.
        2) Includes market returns (monthly/daily) and cutoff files.
        3) Interim files are preserved for downstream steps.

    Output:
        Files copied into 'other_output/' directory.
    """
    other_output = paths.processed_dir / "other_output"
    for name in (
        "market_returns.parquet",
        "market_returns_daily.parquet",
        "nyse_cutoffs.parquet",
        "return_cutoffs.parquet",
        "return_cutoffs_daily.parquet",
        "ap_factors_monthly.parquet",
        "ap_factors_daily.parquet",
        "ap_factors_characteristics.parquet",
    ):
        shutil.copy2(paths.interim_dir / name, other_output / name)


@measure_time
def save_daily_ret(paths: DataPaths):
    """
    Description:
        Export daily returns split by country.

    Steps:
        1) Load world_dsf_output.parquet with daily returns.
        2) Identify unique countries.
        3) For each country, filter and save parquet file (compressed).

    Output:
        'return_data/daily_rets_by_country/{country}.parquet' files for all countries.
    """
    data = (
        pl.scan_parquet(paths.interim_dir / "world_dsf_output.parquet")
        .select(["excntry", "id", "date", "me", "ret", "ret_exc", "ret_exc_wins"])
        .with_columns(
            excntry=pl.when(col("excntry").is_null())
            .then(pl.lit("null_country"))
            .otherwise(col("excntry"))
        )
    )
    daily_returns_temp = paths.interim_dir / "daily_returns_temp.parquet"
    data.collect(engine="streaming").write_parquet(daily_returns_temp)

    out_dir = paths.processed_dir / "return_data" / "daily_rets_by_country"
    con = duckdb.connect()
    con.execute(f"""
    COPY (SELECT * FROM read_parquet('{daily_returns_temp.as_posix()}'))
    TO '{out_dir.as_posix()}'
    ( FORMAT PARQUET,
      COMPRESSION ZSTD,
      PARTITION_BY (excntry),
      OVERWRITE
      );
    """)
    con.close()
    for d in out_dir.glob("excntry=*"):
        if d.is_dir():
            country = d.name[len("excntry=") :]
            part_files = sorted(d.glob("*.parquet"))
            if part_files:
                part_files[0].rename(out_dir / f"{country}.parquet")
            shutil.rmtree(d)


@measure_time
def save_accounting_data(paths: DataPaths):
    """
    Description:
        Export quarterly and annual accounting datasets.

    Steps:
        1) Load acc_std_qtr and acc_std_ann parquet files.
        2) Filter rows with non-null source.
        3) Write results to accounting_data folder.

    Output:
        'accounting_data/quarterly.parquet' and 'accounting_data/annual.parquet'.
    """
    pl.scan_parquet(paths.interim_dir / "acc_std_qtr.parquet").filter(
        col("source").is_not_null()
    ).collect().write_parquet(paths.processed_dir / "accounting_data" / "quarterly.parquet")
    pl.scan_parquet(paths.interim_dir / "acc_std_ann.parquet").filter(
        col("source").is_not_null()
    ).collect().write_parquet(paths.processed_dir / "accounting_data" / "annual.parquet")


@measure_time
def save_full_files_and_cleanup(paths: DataPaths, clear_interim=True):
    """
    Description:
        Save full datasets and remove temporary files.

    Steps:
        1) Write compressed versions of world_dsf and world_data.
        2) Remove raw parquet files and raw_tables/raw_data_dfs folders.

    Output:
        Compressed parquet files in return_data/ and characteristics/, cleanup of temp files.
    """
    pl.scan_parquet(paths.interim_dir / "world_dsf_output.parquet").select(
        pl.all().shrink_dtype()
    ).sink_parquet(paths.processed_dir / "return_data" / "world_dsf.parquet")
    pl.scan_parquet(paths.interim_dir / "world_data_output.parquet").select(
        pl.all().shrink_dtype()
    ).sink_parquet(paths.processed_dir / "characteristics" / "world_data.parquet")
    if clear_interim:
        for child in paths.interim_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in paths.raw_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


@measure_time
def save_monthly_ret(paths: DataPaths):
    """
    Description:
        Save monthly returns for world securities.

    Steps:
        1) Load world_msf_output.parquet and select relevant columns.
        2) Shrink dtypes and collect results.
        3) Write to return_data/world_ret_monthly.parquet.

    Output:
        Parquet file with monthly returns by country/security.
    """
    data = pl.scan_parquet(paths.interim_dir / "world_msf_output.parquet").select(
        ["excntry", "id", "source_crsp", "eom", "me", "ret_exc", "ret", "ret_local", "ret_exc_wins"]
    )
    data.select(pl.all().shrink_dtype()).collect().write_parquet(
        paths.processed_dir / "return_data" / "world_ret_monthly.parquet"
    )


@measure_time
def merge_roll_apply_daily_results(paths: DataPaths):
    """
    Description:
        Merge rolling regression daily results into one dataset.

    Steps:
        1) Build date index from earliest to END_DATE month.
        2) Load id_int mapping and all '__roll*' parquet files (sorted for
           deterministic join order).
        3) Outer join them on (id_int, aux_date).
        4) Map aux_date to calendar eom and join id keys.
        5) Save consolidated roll_apply_daily.parquet.

    Output:
        'roll_apply_daily.parquet' with merged roll regression results.
    """
    date_idx = END_DATE.month + END_DATE.year * 12
    df_dates = pl.DataFrame({"aux_date": range(23113, date_idx + 2)}).with_columns(
        col("aux_date").cast(pl.Int64),
        eom=pl.date((col("aux_date") - 1) // 12, (col("aux_date") - 1) % 12 + 1, 1).dt.month_end(),
    )
    df_id = pl.scan_parquet(paths.interim_dir / "id_int_key.parquet")
    file_paths = sorted(
        p for p in paths.interim_dir.iterdir() if p.is_file() and p.name.startswith("__roll")
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No '__roll*' parquet files found in {paths.interim_dir}; "
            "run roll_apply_daily(...) first."
        )
    joint_file = pl.scan_parquet(file_paths[0])
    for i in file_paths[1:]:
        df_aux = pl.scan_parquet(i)
        joint_file = joint_file.join(df_aux, how="outer_coalesce", on=["id_int", "aux_date"])

    joint_file.with_columns(col("aux_date").cast(pl.Int64)).join(
        df_dates.lazy(),
        how="left",
        on="aux_date",
    ).join(df_id, how="left", on="id_int").drop(["aux_date", "id_int"]).collect().write_parquet(
        paths.interim_dir / "roll_apply_daily.parquet"
    )


@measure_time
def merge_world_data_prelim(paths: DataPaths):
    """
    Description:
        Combine preliminary world data with factor/regression outputs.

    Steps:
        1) Load world_data_prelim and factor files (beta, resmom, mispricing, etc.).
        2) Join all on (id, eom).
        3) Include firm age variable.

    Output:
        'world_data_-1.parquet' with enriched world dataset.
    """
    interim = paths.interim_dir
    a = pl.scan_parquet(interim / "world_data_prelim.parquet")
    b = pl.scan_parquet(interim / "beta_60m.parquet")
    c = pl.scan_parquet(interim / "resmom_ff3_12_1.parquet")
    d = pl.scan_parquet(interim / "resmom_ff3_6_1.parquet")
    e = pl.scan_parquet(interim / "ap_factors_characteristics.parquet")
    f = pl.scan_parquet(interim / "market_chars_d.parquet")
    g = pl.scan_parquet(interim / "firm_age.parquet").select(["id", "eom", "age"])
    world_data = (
        a.join(b, how="left", on=["id", "eom"])
        .join(c, how="left", on=["id", "eom"])
        .join(d, how="left", on=["id", "eom"])
        .join(e, how="left", on=["id", "eom"])
        .join(f, how="left", on=["id", "eom"])
        .join(g, how="left", on=["id", "eom"])
    )
    world_data.collect().write_parquet(paths.interim_dir / "world_data_-1.parquet")


@measure_time
def merge_qmj_to_world_data(paths: DataPaths):
    """
    Description:
        Append QMJ factor to world_data.

    Steps:
        1) Load world_data_-1 and qmj.parquet.
        2) Join on (excntry, id, eom).
        3) Deduplicate and sort results.

    Output:
        'world_data.parquet' with QMJ added.
    """
    a = pl.scan_parquet(paths.interim_dir / "world_data_-1.parquet")
    b = pl.scan_parquet(paths.interim_dir / "qmj.parquet")
    result = (
        a.join(b, how="left", on=["excntry", "id", "eom"]).unique(["id", "eom"]).sort(["id", "eom"])
    )
    result.collect().write_parquet(paths.interim_dir / "world_data.parquet")


@measure_time
def merge_industry_to_world_msf(paths: DataPaths):
    """
    Description:
        Merge industry codes into world MSF dataset.

    Steps:
        1) Load __msf_world, comp_ind, and crsp_ind datasets.
        2) Join compustat and CRSP industry codes on matching keys.
        3) Coalesce SIC/NAICS from both sources.
        4) Drop redundant columns.

    Output:
        '__msf_world2.parquet' with industry codes appended.
    """
    __msf_world = pl.scan_parquet(paths.interim_dir / "__msf_world.parquet")
    comp_ind = pl.scan_parquet(paths.interim_dir / "comp_ind.parquet")
    crsp_ind = pl.scan_parquet(paths.interim_dir / "crsp_ind.parquet").rename(
        {"sic": "sic_crsp", "naics": "naics_crsp"}
    )
    __msf_world = (
        __msf_world.join(comp_ind, how="left", left_on=["gvkey", "eom"], right_on=["gvkey", "date"])
        .join(
            crsp_ind,
            how="left",
            left_on=["permco", "permno", "eom"],
            right_on=["permco", "permno", "date"],
        )
        .with_columns(
            sic=pl.coalesce(["sic", "sic_crsp"]),
            naics=pl.coalesce(["naics", "naics_crsp"]),
        )
        .drop(["sic_crsp", "naics_crsp"])
    )
    __msf_world.collect().write_parquet(paths.interim_dir / "__msf_world2.parquet")


@measure_time
def roll_apply_daily(paths: DataPaths, stats, sfx, __min):
    """
    Description:
        Run rolling daily-stat calculations over grouped date windows and save results.

    Steps:
        1) Generate staggered window specs from sfx (e.g., _21d → k=1, _252d → k=12).
        2) Prepare base daily data per stat.
        3) Apply process_window for each window spec and concat results.
        4) Write to '__roll{sfx}_{stats}.parquet'.

    Output:
        Parquet with per-(id_int, aux_date) rolling metrics for `stats`.
    """
    print(f"Processing {stats} - {sfx.replace('_', '')} - {__min}", flush=True)
    base_data = prepare_base_data(paths, stat=stats)
    results = pl.concat(
        [process_window(base_data, w, stats, sfx, __min) for w in gen_aux_windows(sfx)]
    )
    results.collect(engine="streaming").write_parquet(
        paths.interim_dir / f"__roll{sfx}_{stats}.parquet"
    )


def gen_aux_windows(sfx: str | int) -> list[tuple[int, int, int]]:
    """
    Description:
        Build k staggered window specs from suffix window length.

    Steps:
        1) Map suffix to k: {'_21d':1,'_126d':6,'_252d':12,'_1260d':60} or int(sfx).
        2) For each offset in [0..k-1], compute the window start month and the
           end of the last full k-month window that fits before the END_DATE
           month index. Within an offset, month m belongs to
           group_number = (m - start) // k, and the group's window ends at
           start + (group_number + 1) * k - 1.

    Output:
        List of k tuples (start, k, last_end), one per offset.
    """
    parameter_mapping = {"_21d": 1, "_126d": 6, "_252d": 12, "_1260d": 60}
    k = parameter_mapping[sfx] if sfx in parameter_mapping else int(sfx)
    date_aux = END_DATE.month + END_DATE.year * 12
    windows = []
    for offset in range(k):
        start = 23113 - k + offset
        n_groups = (date_aux - start + 1) // k
        windows.append((start, k, start + n_groups * k - 1))
    return windows


def base_data_filter_exp(stat):
    """
    Description:
        Filter predicate for base daily data by statistic type.

    Steps:
        1) Choose required non-null columns by stat.
        2) For return-based stats, also require zero_obs < 10.

    Output:
        Polars expression usable in .filter().
    """
    if stat in ("zero_trades", "turnover"):
        return col("tvol").is_not_null()
    elif stat == "dolvol":
        return col("dolvol_d").is_not_null()
    elif stat == "mktcorr":
        # corr_data.parquet pre-filtered upstream in prepare_daily.
        return pl.lit(True)
    else:
        return (col("ret_exc").is_not_null()) & (col("zero_obs") < 10)


def prepare_base_data(paths: DataPaths, stat):
    """
    Description:
        Load and minimally prepare base daily dataset for a given stat.

    Steps:
        1) Read 'corr_data.parquet' (mktcorr) or 'dsf1.parquet' (others).
        2) Add integer aux_date via gen_MMYY_column('eom').
        3) Filter using base_data_filter_exp(stat).
        4) For 'dimsonbeta', join lead/lag market returns.

    Output:
        LazyFrame base_data ready for grouping.
    """
    base_data_path = (
        paths.interim_dir / "corr_data.parquet"
        if stat == "mktcorr"
        else paths.interim_dir / "dsf1.parquet"
    )
    base_data = (
        pl.scan_parquet(base_data_path)
        .with_columns(aux_date=gen_MMYY_column("eom"))
        .filter(base_data_filter_exp(stat))
    )

    if stat == "dimsonbeta":
        lead_lag = pl.scan_parquet(paths.interim_dir / "mkt_lead_lag.parquet").drop(
            ["eom", "mktrf"]
        )
        base_data = base_data.join(lead_lag, how="inner", on=["excntry", "date"])

    return base_data


def apply_group_filter(df, stat, min_obs):
    """
    Description:
        Apply per-stat observation-count filters within groups.

    Steps:
        1) For 'dimsonbeta': require counts by (id_int,eom) and (id_int,group_number) and non-null lags.
        2) For zero_trades/dolvol/others: count needed column within (id_int,group_number), require ≥ min_obs.
        3) Pass-through for 'turnover' and 'mktcorr' (later filters inside function).

    Output:
        Filtered LazyFrame for subsequent aggregation/regression.
    """
    if stat in ("turnover", "mktcorr"):
        pass
    elif stat == "dimsonbeta":
        df = df.with_columns(
            n1=pl.len().over(["id_int", "eom"]),
            n2=pl.count("ret_exc").over(["id_int", "group_number"]),
        ).filter(
            (col("n1") >= min_obs - 1)
            & (col("n2") >= min_obs)
            & (col("mktrf_lg1").is_not_null())
            & (col("mktrf_ld1").is_not_null())
        )
    else:
        filter_var = {"zero_trades": "tvol", "dolvol": "dolvol_d"}.get(stat, "ret_exc")
        df = df.with_columns(n=pl.count(filter_var).over(["id_int", "group_number"])).filter(
            col("n") >= min_obs
        )
    return df


def process_window(
    base_data, window: tuple[int, int, int], stats, sfx, __min, incl=None, skip=None
):
    """
    Description:
        Execute a rolling computation for one staggered window offset:
        assign groups arithmetically, filter, compute stat, stamp end date.

    Steps:
        1) Filter base_data to [start, last_end] and assign
           group_number = (aux_date - start) // k.
        2) Apply apply_group_filter(stat, __min).
        3) Run the appropriate function from `funcs` dict (res_mom with incl/skip).
        4) Replace group_number by the window end month:
           aux_date = start + (group_number + 1) * k - 1.

    Output:
        LazyFrame of per-(id_int, aux_date) results.
    """
    funcs = {
        "rvol": rvol,
        "rmax": rmax,
        "skew": skew,
        "prc_to_high": prc_to_high,
        "capm": capm,
        "ami": ami,
        "downbeta": downbeta,
        "mktrf_vol": mktrf_vol,
        "capm_ext": capm_ext,
        "ff3": ff3,
        "hxz4": hxz4,
        "zero_trades": zero_trades,
        "dolvol": dolvol,
        "turnover": turnover,
        "mktcorr": mktcorr,
        "mktvol": mktrf_vol,
        "dimsonbeta": dimsonbeta,
        "res_mom": res_mom,
    }

    start, k, last_end = window
    df = (
        base_data.filter(pl.col("aux_date").is_between(start, last_end))
        .with_columns(group_number=(pl.col("aux_date") - start) // k)
        .pipe(apply_group_filter, stat=stats, min_obs=__min)
    )

    if stats == "res_mom":
        df = df.pipe(funcs[stats], sfx=sfx, __min=__min, incl=incl, skip=skip)
    else:
        df = df.pipe(funcs[stats], sfx=sfx, __min=__min)

    return df.with_columns(
        aux_date=(start + (pl.col("group_number") + 1) * k - 1).cast(pl.Int64)
    ).drop("group_number")


def res_mom(df, sfx, __min, incl, skip):
    """
    Description:
        Residual momentum: standardize mean residuals from FF3 over a lookback, excluding most recent skip.

    Steps:
        1) OLS: ret_exc ~ mktrf + hml + smb_ff (with intercept) → residuals per (id_int,group_number).
        2) Filter to windows inside (max_date - incl, max_date - skip].
        3) Require at least __min obs; compute mean(res)/std(res).

    Output:
        LazyFrame with column f'resff3_{incl}_{skip}' per (id_int, group_number).
    """

    res_exp = (
        pl.col("ret_exc")
        .least_squares.ols("mktrf", "hml", "smb_ff", add_intercept=True, mode="residuals")
        .over(["id_int", "group_number"])
    )
    df = (
        # Fix within-group row order before the OLS: base_data comes from a
        # multithreaded DuckDB scan whose row order is nondeterministic, and the
        # least-squares solve is float-order-sensitive, so an unsorted input
        # yields byte-different residuals across runs.
        df.sort(["id_int", "group_number", "aux_date"])
        .filter(col("hml").is_not_null() & col("smb_ff").is_not_null())
        .with_columns(
            res=res_exp.alias("res"),
            max_date_gn=pl.max("aux_date").over("group_number"),
            n=pl.len().over(["id", "group_number"]),
        )
        .filter(
            (col("aux_date") <= col("max_date_gn") - skip)
            & (col("aux_date") > col("max_date_gn") - incl)
            & (col("n") >= __min)
        )
        .group_by(["id_int", "group_number"])
        .agg((col("res").mean() / col("res").std()).fill_nan(None).alias(f"resff3_{incl}_{skip}"))
    )
    return df


def rvol(df, sfx, __min):
    """
    Description:
        Rolling return volatility (std of ret_exc) within each date group.

    Steps:
        1) Group by (id_int, group_number).
        2) Compute std(ret_exc).

    Output:
        LazyFrame with f'rvol{sfx}'.
    """
    df = df.group_by(["id_int", "group_number"]).agg(
        col("ret_exc").cast(pl.Float64).std().alias(f"rvol{sfx}")
    )
    return df


def rmax(df, sfx, __min):
    """
    Description:
        Rolling extreme return measures.

    Steps:
        1) Group by (id_int,group_number).
        2) Compute mean of top 5 returns and max return.

    Output:
        LazyFrame with f'rmax5{sfx}' and f'rmax1{sfx}'.
    """
    df = df.group_by(["id_int", "group_number"]).agg(
        [
            col("ret").top_k(5).mean().alias(f"rmax5{sfx}"),
            col("ret").max().alias(f"rmax1{sfx}"),
        ]
    )
    return df


def skew(df, sfx, __min):
    """
    Description:
        Rolling skewness of excess returns.

    Steps:
        1) Group by (id_int,group_number).
        2) Compute unbiased skew(ret_exc).

    Output:
        LazyFrame with f'rskew{sfx}'.
    """
    df = df.group_by(["id_int", "group_number"]).agg(
        col("ret_exc").skew(bias=False).alias(f"rskew{sfx}")
    )
    return df


def prc_to_high(df, sfx, __min):
    """
    Description:
        Price-to-high: last price over group max price, with min obs filter.

    Steps:
        1) For each (id_int,group_number), compute last(prc_adj sorted by date)/max(prc_adj)
           and count.
        2) Keep groups with n ≥ __min.

    Output:
        LazyFrame with f'prc_highprc{sfx}'.

    Note:
        Last-by-date is well-defined only when dsf1.parquet is unique on (id_int, date);
        invariant locked by test_dsf1_unique_id_int_date.
    """

    df = (
        df.group_by(["id_int", "group_number"])
        .agg(
            [
                (col("prc_adj").sort_by("date").last() / col("prc_adj").max()).alias(
                    f"prc_highprc{sfx}"
                ),
                pl.count("prc_adj").alias("n"),
            ]
        )
        .filter(col("n") >= __min)
        .drop("n")
    )
    return df


def capm(df, sfx, __min):
    """
    Description:
        CAPM beta and idiosyncratic volatility in rolling windows.

    Steps:
        1) For each (id_int,group_number), compute beta = cov(ret_exc,mktrf)/var(mktrf).
        2) Compute residuals and their std as ivol_capm.

    Output:
        LazyFrame with f'beta{sfx}' and f'ivol_capm{sfx}'.
    """
    # Fix within-group row order before the reductions: base_data comes from a
    # multithreaded DuckDB scan whose row order is nondeterministic, and cov/var/
    # std are float-order-sensitive, so an unsorted input yields byte-different
    # betas across runs.
    df = (
        df.sort(["id_int", "group_number", "aux_date"])
        .group_by(["id_int", "group_number"])
        .agg(
            [
                (pl.cov("ret_exc", "mktrf") / pl.var("mktrf")).alias(f"beta{sfx}"),
                (col("ret_exc") - col("mktrf") * (pl.cov("ret_exc", "mktrf") / pl.var("mktrf")))
                .std()
                .alias(f"ivol_capm{sfx}"),
            ]
        )
    )
    return df


def ami(df, sfx, __min):
    """
    Description:
        Amihud illiquidity proxy using daily abs returns over dollar volume.

    Steps:
        1) Define dolvol guard (None if zero).
        2) Group by (id_int,group_number); compute mean(|ret|/dolvol * 1e6) and count.
        3) Keep groups with n ≥ __min.

    Output:
        LazyFrame with f'ami{sfx}'.
    """
    aux_1 = pl.when(col("dolvol_d") == 0).then(fl_none()).otherwise(col("dolvol_d"))
    df = (
        df.group_by(["id_int", "group_number"])
        .agg(
            [
                (col("ret").abs() / aux_1 * 1e6).mean().alias(f"ami{sfx}"),
                pl.count("dolvol_d").alias("n"),
            ]
        )
        .filter(col("n") >= __min)
        .drop("n")
    )
    return df


def downbeta(df, sfx, __min):
    """
    Description:
        Downside beta using days with negative market returns.

    Steps:
        1) Filter mktrf < 0.
        2) Group by (id_int,group_number); compute beta as cov/var; require n ≥ __min/2.

    Output:
        LazyFrame with f'betadown{sfx}'.
    """
    df = (
        df.filter(col("mktrf") < 0)
        .group_by(["id_int", "group_number"])
        .agg(
            [
                (pl.cov("ret_exc", "mktrf") / pl.var("mktrf")).alias(f"betadown{sfx}"),
                pl.count("ret_exc").alias("n"),
            ]
        )
        .filter(col("n") >= __min / 2)
        .drop("n")
    )
    return df


def mktrf_vol(df, sfx, __min):
    """
    Description:
        Market factor volatility within window.

    Steps:
        1) Group by (id_int,group_number) or simply group_number context.
        2) Compute std(mktrf).

    Output:
        LazyFrame with f'__mktvol{sfx}'.
    """
    df = df.group_by(["id_int", "group_number"]).agg(
        col("mktrf").cast(pl.Float64).std().alias(f"__mktvol{sfx}")
    )
    return df


def capm_ext(df, sfx, __min):
    """
    Description:
        Extended CAPM diagnostics: beta, idio vol, idio skew, and coskewness.

    Steps:
        1) Compute beta and alpha; residuals = ret_exc − (alpha + beta*mktrf).
        2) Aggregate per (id_int,group_number): std(res), skew(res), coskew = E[res*(mktrf−E mktrf)^2]/(sqrt(E[res^2])*sqrt(E[(mktrf−E)^2])).

    Output:
        LazyFrame with [f'beta_{sfx}', f'ivol_capm{sfx}', f'iskew_capm{sfx}', f'coskew{sfx}'].
    """
    beta_col = pl.cov("ret_exc", "mktrf") / pl.var("mktrf")
    alpha_col = pl.mean("ret_exc") - beta_col * pl.mean("mktrf")
    residual_col = col("ret_exc") - (alpha_col + col("mktrf") * beta_col)
    exp_mkt = col("mktrf") - col("mktrf").mean()
    exp_coskew1 = (residual_col * (exp_mkt**2)).mean()
    exp_coskew2 = (residual_col**2).mean() ** 0.5 * (exp_mkt**2).mean()

    df = df.group_by(["id_int", "group_number"]).agg(
        [
            beta_col.cast(pl.Float64).alias(f"beta{sfx}"),
            residual_col.std().alias(f"ivol_capm{sfx}"),
            residual_col.skew(bias=False).alias(f"iskew_capm{sfx}"),
            (exp_coskew1 / exp_coskew2).alias(f"coskew{sfx}"),
        ]
    )
    return df


def ff3(df, sfx, __min):
    """
    Description:
        FF3 residual volatility and skewness in rolling windows.

    Steps:
        1) OLS: ret_exc ~ mktrf + smb_ff + hml; require factors present.
        2) Aggregate per (id_int,group_number): std(residuals, ddof=3), skew(residuals).

    Output:
        LazyFrame with [f'ivol_ff3{sfx}', f'iskew_ff3{sfx}'].
    """
    res_exp = pl.col("ret_exc").least_squares.ols(
        "mktrf", "smb_ff", "hml", add_intercept=True, mode="residuals"
    )
    df = (
        df.filter(col("smb_ff").is_not_null() & col("hml").is_not_null())
        .group_by(["id_int", "group_number"])
        .agg(
            res_exp.std(ddof=3).alias(f"ivol_ff3{sfx}"),
            res_exp.skew(bias=False).alias(f"iskew_ff3{sfx}"),
        )
    )
    return df


def hxz4(df, sfx, __min):
    """
    Description:
        HXZ 4-factor residual volatility and skewness.

    Steps:
        1) OLS: ret_exc ~ mktrf + me_hxz + roe_hxz + ia_hxz; require all factors.
        2) Aggregate per (id_int,group_number): std(residuals, ddof=4), skew(residuals).

    Output:
        LazyFrame with [f'ivol_hxz4{sfx}', f'iskew_hxz4{sfx}'].
    """
    res_exp = pl.col("ret_exc").least_squares.ols(
        "mktrf", "me_hxz", "roe_hxz", "ia_hxz", add_intercept=True, mode="residuals"
    )
    df = (
        df.filter(
            col("me_hxz").is_not_null() & col("roe_hxz").is_not_null() & col("ia_hxz").is_not_null()
        )
        .group_by(["id_int", "group_number"])
        .agg(
            res_exp.std(ddof=4).alias(f"ivol_hxz4{sfx}"),
            res_exp.skew(bias=False).alias(f"iskew_hxz4{sfx}"),
        )
    )
    return df


def _turnover_d_expr():
    """tvol / (shares*1e6) when shares!=0, else null."""
    return (
        pl.when(col("shares") != 0).then(col("tvol") / (col("shares") * 1e6)).otherwise(fl_none())
    )


def zero_trades(df, sfx, __min):
    """
    Description:
        Zero-trade days and turnover-based illiquidity composite.

    Steps:
        1) For each (id_int,group_number): zero_trades = mean(tvol==0)*21;
           turnover = mean(tvol/(shares*1e6)) for rows with shares>0.
        2) Drop groups where zero_trades is null.
        3) Rank turnover descending within group_number (method="average");
           composite = rank/count/100 + zero_trades.

    Output:
        LazyFrame with f'zero_trades{sfx}'.
    """
    df = (
        df.group_by(["id_int", "group_number"])
        .agg(
            ((col("tvol") == 0).mean() * 21).alias("zero_trades"),
            _turnover_d_expr().mean().alias("turnover"),
        )
        .filter(col("zero_trades").is_not_null())
        .with_columns(
            (
                (
                    col("turnover").rank(descending=True, method="average") / pl.count("turnover")
                ).over("group_number")
                / 100
                + col("zero_trades")
            ).alias(f"zero_trades{sfx}")
        )
        .select(["id_int", "group_number", f"zero_trades{sfx}"])
    )
    return df


def dolvol(df, sfx, __min):
    """
    Description:
        Dollar volume level and variability within window.

    Steps:
        1) Group by (id_int,group_number); compute mean(dolvol_d).
        2) Compute std/mean as variability (guard when mean==0).

    Output:
        LazyFrame with [f'dolvol{sfx}', f'dolvol_var{sfx}'].
    """
    df = df.group_by(["id_int", "group_number"]).agg(
        [
            col("dolvol_d").mean().alias(f"dolvol{sfx}"),
            pl.when(col("dolvol_d").mean() != 0)
            .then(col("dolvol_d").std() / col("dolvol_d").mean())
            .otherwise(fl_none())
            .alias(f"dolvol_var{sfx}"),
        ]
    )
    return df


def turnover(df, sfx, __min):
    """
    Description:
        Turnover level and variability within window.

    Steps:
        1) Compute scalar mean and std of turnover_d = tvol/(shares*1e6) (guard shares>0)
           and total row count n per (id_int,group_number) in one .agg call.
        2) Derive variability ratio std/mean; require n ≥ __min.

    Output:
        LazyFrame with [f'turnover{sfx}', f'turnover_var{sfx}'].
    """
    turnover_d = _turnover_d_expr()
    df = (
        df.group_by(["id_int", "group_number"])
        .agg(
            turnover_d.mean().alias(f"turnover{sfx}"),
            turnover_d.std().alias("turnover_std"),
            pl.len().alias("n"),
        )
        .with_columns(
            pl.when(col(f"turnover{sfx}") != 0)
            .then(col("turnover_std") / col(f"turnover{sfx}"))
            .otherwise(fl_none())
            .alias(f"turnover_var{sfx}"),
        )
        .filter(col("n") >= __min)
        .drop(["turnover_std", "n"])
    )
    return df


def mktcorr(df, sfx, __min):
    """
    Description:
        Rolling correlation between 3-day summed stock and market excess returns.

    Steps:
        1) Group by (id_int, group_number); count rows.
        2) Require count >= __min; compute Pearson corr.

    Output:
        LazyFrame with f'corr{sfx}'.
    """
    return (
        df.group_by(["id_int", "group_number"])
        .agg(
            pl.len().alias("n"),
            pl.corr("ret_exc_3l", "mkt_exc_3l").alias(f"corr{sfx}"),
        )
        .filter(col("n") >= __min)
        .drop("n")
    )


def dimsonbeta(
    df: pl.DataFrame | pl.LazyFrame, sfx: str, __min: int
) -> pl.DataFrame | pl.LazyFrame:
    """
    Description:
        Dimson β = sum of slopes from OLS ret_exc ~ mktrf_{-1,0,+1} per
        (id_int, group_number) via polars-ds `pds.lin_reg`.
    Output:
        LazyFrame with f'beta_dimson{sfx}'.
    """
    name = f"beta_dimson{sfx}"
    beta_expr = pl.col("coeffs").list.head(3).list.sum()
    return (
        df.group_by(["id_int", "group_number"])
        .agg(
            coeffs=pds.lin_reg(
                "mktrf_lg1",
                "mktrf",
                "mktrf_ld1",
                target="ret_exc",
                add_bias=True,
                solver="cholesky",
            )
        )
        .select("id_int", "group_number", beta_expr.alias(name))
        .filter(pl.col(name).is_not_null() & pl.col(name).is_not_nan())
    )


def _ff_port_year() -> pl.Expr:
    """Formation year y for any date t in Jul(y)..Jun(y+1) — the July-to-June
    portfolio-assignment convention shared by the FF and HXZ broadcasts."""
    return (
        pl.when(pl.col("date").dt.month() >= 7)
        .then(pl.col("date").dt.year())
        .otherwise(pl.col("date").dt.year() - 1)
        .alias("port_year")
    )


def _ff_count_gate() -> pl.Expr:
    """Two-years-on-Compustat gate for the B/M and OP sorts: not applied to
    the US (French's modern portfolio descriptions never state it); ROW
    keeps the JKP convention (count >= 2)."""
    return (pl.col("excntry") == US_EXCNTRY) | (pl.col("count") >= 2)


def _ff_bm_eligible() -> pl.Expr:
    """B/M sort eligibility: positive book-to-market and me, count gate."""
    return (pl.col("beme") > 0) & (pl.col("me") > 0) & _ff_count_gate()


def _ff_op_eligible() -> pl.Expr:
    """OP sort eligibility: positive me and BE, count gate, op present."""
    return (pl.col("me") > 0) & (pl.col("be") > 0) & _ff_count_gate() & pl.col("op").is_not_null()


def _ff_inv_eligible() -> pl.Expr:
    """INV sort eligibility: positive me, inv present (no count gate)."""
    return (pl.col("me") > 0) & pl.col("inv").is_not_null()


def _ff_mom_eligible() -> pl.Expr:
    """Momentum sort eligibility: 2-12 signal present, positive lagged me."""
    return pl.col("mom_2_12").is_not_null() & (pl.col("me_lag1") > 0)


def _ciz_universe() -> pl.Expr:
    """CRSP CIZ universe filter shared by FF (ff_load_crsp_panel) and HXZ
    (hxz_load_crsp). Mispricing has its own `_mp_universe()` with extra
    conditionaltype / tradingstatusflg clauses; do not collapse those."""
    return (
        (pl.col("sharetype") == "NS")
        & (pl.col("securitytype") == "EQTY")
        & (pl.col("securitysubtype") == "COM")
        & (pl.col("usincflg") == "Y")
        & pl.col("issuertype").is_in(["ACOR", "CORP"])
        & pl.col("primaryexch").is_in(["N", "A", "Q"])
    )


def _collapse_to_latest_fy(lf: pl.LazyFrame, *, resort: bool = True) -> pl.LazyFrame:
    """One accounting row per (gvkey, fiscal year), keeping the latest datadate.

    A fiscal-year-end change can yield two datadates in one calendar year;
    without this the at.shift(1) lag spans the sub-annual stub (fy_gap=0)
    and nulls inv on the surviving row. Requires a `year` column.
    """
    out = lf.sort(["gvkey", "datadate"]).unique(
        subset=["gvkey", "year"], keep="last", maintain_order=True
    )
    return out.sort(["gvkey", "datadate"]) if resort else out


def _ff_us_accounting(raw_dir: Path, ff5: bool) -> pl.LazyFrame:
    """Scan comp_funda.parquet and compute FF-strict BE/OP/INV.

    BE = SHE + txditc_FASB109 − coalesce(pstkrv, pstkl, pstk, 0),
    SHE = coalesce(seq, ceq + pstk, at − lt) per French's ladder; null when BE < 0.
    ff5: OP = (revt − cogs − xsga − xint) / (BE + mib),
    INV = Δat / at_lag with strict 1-year FY gap.

    Output: Lazy [gvkey, datadate, year, be, (op, inv,) count].
    """
    base = ["pstkrv", "pstkl", "pstk", "seq", "ceq", "lt", "at", "txditc"]
    extra = ["revt", "cogs", "xsga", "xint", "mib"]
    floats = base + (extra if ff5 else [])
    # Stockholders' equity ladder per French (data library / DFF 2000):
    # seq, else ceq + pstk, else at - lt. ceq + pstk nulls when pstk is
    # missing, falling to at - lt (SAS sum semantics).
    she = pl.coalesce(
        pl.col("seq"),
        pl.col("ceq") + pl.col("pstk"),
        pl.col("at") - pl.col("lt"),
    )
    be_raw = she + pl.col("txditc") - pl.col("ps")

    lf = (
        pl.scan_parquet(raw_dir / "comp_funda.parquet")
        .with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col(*floats).cast(pl.Float64),
        )
        .filter(_compustat_na_filter())
        .with_columns(
            ps=pl.coalesce("pstkrv", "pstkl", "pstk", pl.lit(0.0)),
            txditc=pl.when(pl.col("datadate").dt.year() < 1993)
            .then(pl.col("txditc").fill_null(0.0))
            .otherwise(pl.lit(0.0)),
            year=pl.col("datadate").dt.year(),
        )
        .with_columns(be=pl.when(be_raw < 0).then(None).otherwise(be_raw))
        # BEFORE the inv lag below; mirrors the ROW loaders (_ff_load_funda_na/_ff_load_funda_global).
        .pipe(_collapse_to_latest_fy)
    )

    if ff5:
        op_elig = pl.col("revt").is_not_null() & (
            pl.col("cogs").is_not_null()
            | pl.col("xsga").is_not_null()
            | pl.col("xint").is_not_null()
        )
        op_num = (
            pl.col("revt")
            - pl.coalesce("cogs", pl.lit(0.0))
            - pl.coalesce("xsga", pl.lit(0.0))
            - pl.coalesce("xint", pl.lit(0.0))
        )
        fy_gap = pl.col("datadate").dt.year() - pl.col("datadate_lag").dt.year()
        # OP denominator per French's variable definitions: book equity plus
        # minority interest; the positive-BE screen stays on BE alone.
        lf = lf.with_columns(
            op=pl.when(op_elig & (pl.col("be") > 0))
            .then(safe_div(op_num, pl.col("be") + pl.coalesce("mib", pl.lit(0.0)), "op"))
            .otherwise(None),
            at_lag=pl.col("at").shift(1).over("gvkey"),
            datadate_lag=pl.col("datadate").shift(1).over("gvkey"),
        ).with_columns(
            inv=pl.when(
                pl.col("at").is_not_null()
                & pl.col("at_lag").is_not_null()
                & (pl.col("at_lag") > 0)
                & (fy_gap == 1)
            )
            .then(safe_div(pl.col("at") - pl.col("at_lag"), pl.col("at_lag"), "inv"))
            .otherwise(None),
        )

    keep = ["op", "inv"] if ff5 else []
    return lf.with_columns(count=pl.int_range(pl.len()).over("gvkey") + 1).select(
        ["gvkey", "datadate", "year", "be", *keep, "count"]
    )


def _ff_ccm_link(comp: pl.LazyFrame, raw_dir: Path) -> pl.DataFrame:
    """Link Compustat gvkey→permno via CCM; dedup to one row per (permno, year).

    Dedup ladder: in-window > backdated (FF_CCM_BACKDATE_*), then P > C on
    (datadate, permno), then last entry per (permno, year).

    Output: Eager [gvkey, permno, year, datadate, be, (op, inv,) count, ...].
    """
    lnk = (
        pl.scan_parquet(raw_dir / "crsp_ccmxpf_lnkhist.parquet")
        .filter(pl.col("linktype").str.starts_with("L") & pl.col("linkprim").is_in(["P", "C"]))
        .select("gvkey", "lpermno", "linkprim", "linkdt", "linkenddt")
        .with_columns(
            pl.col("lpermno").cast(pl.Int64),
            pl.col("linkdt", "linkenddt").cast(pl.Date),
        )
        .with_columns(first_linkdt=pl.col("linkdt").min().over("lpermno"))
    )
    # Backfill rescue (FF_CCM_BACKDATE_* in config): a June formation may use fiscal
    # rows predating the link window when the link is the permno's first and starts
    # within MAX_YEARS of the formation. In-window links win any collision.
    in_window = (pl.col("linkdt") <= pl.col("jun_end")) & (
        pl.col("linkenddt").is_null() | (pl.col("linkenddt") >= pl.col("jun_end"))
    )
    backdated = (
        pl.col("jun_end").is_between(
            pl.date(FF_CCM_BACKDATE_FROM, 6, 30), pl.date(FF_CCM_BACKDATE_THROUGH, 6, 30)
        )
        & (pl.col("linkdt") == pl.col("first_linkdt"))
        & (pl.col("linkdt") <= pl.col("jun_end").dt.offset_by(f"{FF_CCM_BACKDATE_MAX_YEARS}y"))
    )
    return (
        comp.with_columns(jun_end=pl.date(pl.col("datadate").dt.year() + 1, 6, 30))
        .join(lnk, on="gvkey", how="inner")
        .with_columns(in_window=in_window)
        .filter(pl.col("in_window") | backdated)
        .rename({"lpermno": "permno"})
        # gvkey last: duplicate same-prim links on (datadate, permno) exist;
        # without a total order the kept row is engine-order-dependent.
        .sort(
            ["datadate", "permno", "in_window", "linkprim", "gvkey"],
            descending=[False, False, True, True, False],
        )
        .unique(subset=["datadate", "permno"], keep="first")
        .sort(["permno", "year", "datadate"])
        .unique(subset=["permno", "year"], keep="last")
        .drop("in_window", "first_linkdt")
        .collect()
    )


def _ff_merge_dff_be(
    comp_ccm: pl.DataFrame,
    ff5: bool,
    use_dff: bool,
    dff_path: Path | None,
) -> pl.DataFrame:
    """Optionally union DFF hand-collected Moody's BE to extend coverage back to 1926.

    DFF be(t) is aligned to year t−1 so the June join pairs it with Dec(t−1) ME.
    Value-level merge: be = coalesce(dff.be, comp.be); Compustat row keeps its
    op/inv/count. Anti-join appends pure-DFF rows.

    Output: Eager [gvkey, permno, year, datadate, be, (op, inv,) count, ...].
    """
    if not use_dff:
        return comp_ccm

    # DFF hand-collected BE union (see ff_load_compustat_us docstring step 6)
    keep = ["op", "inv"] if ff5 else []
    select_cols = ["gvkey", "permno", "year", "datadate", "be", *keep, "count"]
    dff = (
        load_dff_be(dff_path)
        # Match the US Compustat path: BE < 0 → null → dropped. BE == 0 flows
        # through and is later excluded by the beme > 0 eligibility gate.
        .filter(pl.col("be").is_not_null() & (pl.col("be") >= 0))
        .with_columns(
            # datadate from DFF year t BEFORE the year offset below: a synthetic
            # formation-aligned stamp (June 30 of t), used only as a dedup
            # tie-break downstream — never for BE timing itself.
            datadate=pl.date(pl.col("year"), 6, 30),
            # ALIGNMENT OFFSET: DFF be(t) → unioned year t−1, so the June-frame
            # join (cyp1 = year + 1 = t = june_year) pairs it with Dec(t−1) ME.
            year=pl.col("year") - 1,
            gvkey=pl.lit(None, dtype=pl.Utf8),
            count=pl.lit(FF_DFF_SYNTH_COUNT, dtype=pl.Int64),
            **{c: pl.lit(None, dtype=pl.Float64) for c in keep},
        )
        .select(select_cols)
    )
    comp_pool = (
        comp_ccm.filter(pl.col("year") >= FF_DFF_MERGE_YEAR - 1)
        if FF_DFF_GATE_COMPUSTAT_PRE1954
        else comp_ccm
    )
    n_collisions = (
        comp_pool.select("permno", "year")
        .join(dff.select("permno", "year"), on=["permno", "year"], how="inner")
        .height
    )
    if n_collisions:
        print(
            f"DFF/Compustat BE overlap: {n_collisions} (permno, year) collisions; "
            "be coalesced (DFF first)",
            flush=True,
        )
    # be = coalesce(dff.be, comp.be) on (permno, year); sources agree where both are
    # non-null (median |log ratio| ~ 0). Typical collision: the CCM-linked Compustat
    # row has BE NULL, so DFF fills BE while that row keeps op/inv/count for FF5.
    keys = ["permno", "year"]
    comp_pool = (
        comp_pool.select(select_cols)
        .join(dff.select([*keys, "be"]).rename({"be": "_be_dff"}), on=keys, how="left")
        .with_columns(
            be=pl.coalesce("_be_dff", "be"),
            # A DFF-covered cell is eligible regardless of Compustat history:
            # hand-collected BE needs no backfill-bias seasoning, so it clears the
            # count >= 2 BM gate like a standalone DFF row.
            count=pl.when(pl.col("_be_dff").is_not_null())
            .then(pl.max_horizontal("count", pl.lit(FF_DFF_SYNTH_COUNT, dtype=pl.Int64)))
            .otherwise(pl.col("count")),
        )
        .drop("_be_dff")
    )
    dff_only = dff.join(comp_pool.select(keys), on=keys, how="anti")
    return pl.concat([comp_pool, dff_only], how="vertical_relaxed")


def ff_load_compustat_us(
    raw_dir: Path,
    ff5: bool,
    *,
    use_dff: bool = False,
    dff_path: Path | None = None,
) -> pl.DataFrame:
    """FF-strict Compustat NA funda linked gvkey→permno via CCM. FF3: BE only; FF5 adds OP/INV.

    use_dff: unions DFF hand-collected Moody's BE (permno-keyed) to extend coverage to
    June 1926. DFF be(t) pairs with Dec(t−1) ME (year = t − 1 in the output); BE in
    $M, identical to Compustat — the 1000× beme scale applies unchanged. Modern
    Compustat backfill creates collisions resolved by be = coalesce(dff.be, comp.be).

    Output: Eager [gvkey, permno, year, datadate, be, (op, inv,) count, ...].
    """
    comp = _ff_us_accounting(raw_dir, ff5)
    comp_ccm = _ff_ccm_link(comp, raw_dir)
    return _ff_merge_dff_be(comp_ccm, ff5, use_dff, dff_path)


def ff_load_crsp_panel(raw_dir: Path, freq: Literal["monthly", "daily"]) -> pl.LazyFrame:
    """Load CRSP msf_v2/dsf_v2 with CIZ universe filters; permco-level ME aggregation.

    ME = sum of permno meq assigned to the largest-meq permno per (date, permco).

    Output: Lazy [permno, permco, date, retadj, retx, prc, shrout, me, exchcd, shrcd].
    """
    pq, p = ("crsp_msf_v2.parquet", "mth") if freq == "monthly" else ("crsp_dsf_v2.parquet", "dly")
    date_c, ret_c, retx_c, prc_c = f"{p}caldt", f"{p}ret", f"{p}retx", f"{p}prc"
    lf = (
        pl.scan_parquet(raw_dir / pq)
        .with_columns(
            pl.col(date_c).cast(pl.Date),
            pl.col("permno", "permco").cast(pl.Int64),
            pl.col(ret_c, retx_c, prc_c, "shrout").cast(pl.Float64),
        )
        .filter(_ciz_universe())
        .with_columns(
            meq=pl.col(prc_c).abs() * pl.col("shrout"),
            exchcd=pl.col("primaryexch").replace_strict(
                {"N": 1, "A": 2, "Q": 3}, default=None, return_dtype=pl.Int64
            ),
            shrcd=pl.lit(10, dtype=pl.Int64),
        )
        .rename({date_c: "date", ret_c: "retadj", retx_c: "retx", prc_c: "prc"})
    )
    date_expr = pl.col("date").dt.month_end() if freq == "monthly" else pl.col("date")
    return (
        lf.with_columns(date=date_expr)
        .select(
            "permno",
            "permco",
            "date",
            "retadj",
            "retx",
            "prc",
            "shrout",
            "meq",
            "exchcd",
            "shrcd",
        )
        .sort(["date", "permco", "meq"])
        .with_columns(me=pl.col("meq").sum().over(["date", "permco"]))
        .unique(subset=["date", "permco"], keep="last")
        .drop("meq")
    )


def _ff_us_rets_weights_lazy(
    raw_panel: pl.LazyFrame, freq: Literal["monthly", "daily"]
) -> pl.LazyFrame:
    """Lazy US returns/weights panel: FF intra-year ME-compounding weight
    (`weight_port` = me_base x cumretx at the prior bar within the July-June
    portfolio year). Fuses with the loader scan; collected once per leg.
    Output: [permno, date, retadj, weight_port, me, exchcd, shrcd]."""
    period = (
        (pl.col("date").dt.month() == 7).cast(pl.Int32).cum_sum().over("permno")
        if freq == "monthly"
        else pl.when(pl.col("date").dt.month() >= 7)
        .then(pl.col("date").dt.year())
        .otherwise(pl.col("date").dt.year() - 1)
    )
    return (
        raw_panel.sort(["permno", "date"])
        .with_columns(
            period=period,
            retx_safe=pl.col("retx").fill_null(0.0),
            lag_me=pl.col("me").shift(1).over("permno"),
        )
        .with_columns(
            me_base=pl.when(
                (pl.col("period") != pl.col("period").shift(1).over("permno")).fill_null(True)
            )
            .then(pl.col("lag_me"))
            .otherwise(None)
            .forward_fill()
            .over(["permno", "period"]),
            cumretx=(1.0 + pl.col("retx_safe")).cum_prod().over(["permno", "period"]),
        )
        .with_columns(
            weight_port=pl.when(pl.col("lag_me").is_not_null() & (pl.col("lag_me") > 0))
            .then(
                pl.col("me_base")
                * pl.col("cumretx").shift(1).over(["permno", "period"]).fill_null(1.0)
            )
            .otherwise(None),
        )
        .select("permno", "date", "retadj", "weight_port", "me", "exchcd", "shrcd")
    )


def _ff_row_rets_weights_lazy(raw_panel: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy ROW returns/weights panel: w = me.shift(1) over (excntry, id).
    world_dsf carries no clean retx, so cumretx intra-year compounding is skipped."""
    return raw_panel.sort(["excntry", "id", "date"]).with_columns(
        w=pl.col("me").shift(1).over("excntry", "id")
    )


def _ff_us_finish_prepare(
    data_rets_weights: pl.DataFrame, comp: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Eager US tail of the prepare step: June formation frame from the
    collected returns/weights panel, projected to the common US/ROW schema.

    comp = ccm2a (gvkey->permno CCM-linked Compustat); joins on permno.
    beme uses the 1000x multiplier to reconcile CRSP $K vs Compustat $M.
    Output:
      data_rets_weights: [excntry, id, date, ret, w, me, exchcd_us, size_grp]
      data_chars:        [excntry, id, date, me, dec_me, be, op, inv, count,
                          beme, exchcd_us, size_grp]
    (exchcd_us US-only; size_grp null here so vertical_relaxed concat with
    the ROW frames works directly.)
    """

    # Per-month filter-then-dedup (cheap for daily; no-op dedup for monthly)
    def _last_per_year(month: int) -> pl.DataFrame:
        return (
            data_rets_weights.filter(pl.col("date").dt.month() == month)
            .with_columns(_y=pl.col("date").dt.year())
            .sort(["permno", "_y", "date"])
            .unique(subset=["permno", "_y"], keep="last")
        )

    june_only = (
        _last_per_year(6)
        .with_columns(date=pl.date(pl.col("_y"), 6, 30), june_year=pl.col("_y"))
        .drop("_y")
    )
    dec_me = (
        _last_per_year(12)
        .filter(pl.col("me") > 0)
        .select("permno", june_year=pl.col("_y") + 1, dec_me=pl.col("me"))
    )
    # Left joins keep June stocks lacking accounting or Dec(t-1) ME (null beme) so the
    # US size-median pool spans all NYSE stocks (DFF 2000); sort gates still drop them
    # from B/M, but US stocks missing only Dec ME stay sortable on OP/INV (French's
    # OP/INV need June ME, not Dec ME).
    data_chars = (
        june_only.join(dec_me, on=["permno", "june_year"], how="left")
        .join(
            comp.select("permno", "be", "op", "inv", "count", "datadate", cyp1=pl.col("year") + 1),
            left_on=["permno", "june_year"],
            right_on=["permno", "cyp1"],
            how="left",
        )
        .drop("june_year")
        .with_columns(beme=1000.0 * safe_div(pl.col("be"), pl.col("dec_me"), "beme"))
    )

    # Shared output schema with _ff_row_finish_prepare.
    data_rets_weights = data_rets_weights.select(
        excntry=pl.lit(US_EXCNTRY),
        id=pl.col("permno"),
        date=pl.col("date"),
        ret=pl.col("retadj"),
        w=pl.col("weight_port"),
        me=pl.col("me"),
        exchcd_us=pl.col("exchcd"),
        size_grp=pl.lit(None, dtype=pl.Utf8),
    )
    data_chars = data_chars.select(
        excntry=pl.lit(US_EXCNTRY),
        id=pl.col("permno"),
        date=pl.col("date"),
        me=pl.col("me"),
        dec_me=pl.col("dec_me"),
        be=pl.col("be"),
        op=pl.col("op"),
        inv=pl.col("inv"),
        count=pl.col("count"),
        beme=pl.col("beme"),
        exchcd_us=pl.col("exchcd"),
        size_grp=pl.lit(None, dtype=pl.Utf8),
    )
    return data_rets_weights, data_chars


def _ff_row_finish_prepare(
    data_rets_weights: pl.DataFrame, comp: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Eager ROW tail of the prepare step: June formation frame from the
    collected returns/weights panel, projected to the common US/ROW schema.

    comp = ff_load_world_compustat output; joins on gvkey. Daily input is
    downsampled to month-end for formation only — the returned panel keeps
    the original frequency. beme = be / dec_me (both already in $M); the
    dec_me join is inner (JKP convention).
    Output schema: see _ff_us_finish_prepare.
    """

    # Per-month filter-then-dedup (cheap for daily; no-op dedup for monthly)
    def _last_per_year(month: int) -> pl.DataFrame:
        return (
            data_rets_weights.filter(pl.col("date").dt.month() == month)
            .with_columns(_y=pl.col("date").dt.year())
            .sort(["excntry", "id", "_y", "date"])
            .unique(subset=["excntry", "id", "_y"], keep="last")
        )

    june_only = (
        _last_per_year(6)
        .with_columns(date=pl.date(pl.col("_y"), 6, 30), june_year=pl.col("_y"))
        .drop("_y")
    )
    dec_me = (
        _last_per_year(12)
        .filter(pl.col("me") > 0)
        .select("excntry", "id", june_year=pl.col("_y") + 1, dec_me=pl.col("me"))
    )
    data_chars = (
        june_only.join(dec_me, on=["excntry", "id", "june_year"], how="inner")
        .join(
            comp.select("gvkey", "be", "op", "inv", "count", "datadate", cyp1=pl.col("year") + 1),
            left_on=["gvkey", "june_year"],
            right_on=["gvkey", "cyp1"],
            how="left",
        )
        .drop("june_year")
        .with_columns(beme=safe_div(pl.col("be"), pl.col("dec_me"), "beme"))
    )

    # Shared output schema with _ff_us_finish_prepare.
    data_rets_weights = data_rets_weights.select(
        "excntry",
        "id",
        "date",
        ret=pl.col("ret"),
        w=pl.col("w"),
        me=pl.col("me"),
        exchcd_us=pl.lit(None, dtype=pl.Int64),
        size_grp=pl.col("size_grp"),
    )
    data_chars = data_chars.select(
        "excntry",
        "id",
        "date",
        "me",
        "dec_me",
        "be",
        "op",
        "inv",
        "count",
        "beme",
        exchcd_us=pl.lit(None, dtype=pl.Int64),
        size_grp=pl.col("size_grp"),
    )
    return data_rets_weights, data_chars


def _ff_bucket(val: str, breaks: list[str], labels: list[str]) -> pl.Expr:
    """Ladder label: `val <= breaks[i]` -> `labels[i]`, else `labels[-1]`.
    Requires `len(labels) == len(breaks) + 1`."""
    if len(labels) != len(breaks) + 1:
        raise ValueError("labels must have exactly one more entry than breaks")
    out = pl.when(pl.col(val) <= pl.col(breaks[0])).then(pl.lit(labels[0]))
    for br, lab in zip(breaks[1:], labels[1:-1], strict=True):
        out = out.when(pl.col(val) <= pl.col(br)).then(pl.lit(lab))
    return out.otherwise(pl.lit(labels[-1]))


def _ff_na_be_op_inv(lf: pl.LazyFrame) -> pl.LazyFrame:
    """NA-funda BE/OP/INV per JKP (covers Canada and other NA-coverage non-US gvkeys).

    BE: seq → ceq+pstk → at-lt, plus txditc → txdb+itcb,
    minus preferred pstkrv → pstkl → pstk; no FASB-109 gate, no BE<0 cut.
    OP = ope_x / BE_local; ope_x = ebitda → oibdp → sale_x − opex_x, minus xint.
    INV = 1-yr asset growth with strict 1-FY gap.
    """
    ps_x = pl.coalesce("pstkrv", "pstkl", "pstk")
    txditc = pl.coalesce("txditc", pl.col("txdb") + pl.col("itcb"))
    return (
        lf.with_columns(
            be_local=pl.coalesce(
                "seq", pl.col("ceq") + pl.coalesce(ps_x, 0.0), pl.col("at") - pl.col("lt")
            )
            + pl.coalesce(txditc, 0.0)
            - pl.coalesce(ps_x, 0.0),
            ope_x=pl.coalesce(
                "ebitda",
                "oibdp",
                pl.coalesce("sale", "revt") - pl.coalesce("xopr", pl.col("cogs") + pl.col("xsga")),
            )
            - pl.col("xint"),
            year=pl.col("datadate").dt.year(),
        )
        .sort(["gvkey", "datadate"])
        .with_columns(
            op=pl.when(pl.col("ope_x").is_not_null() & (pl.col("be_local") > 0))
            .then(safe_div(pl.col("ope_x"), pl.col("be_local"), "op"))
            .otherwise(None),
            at_lag=pl.col("at").shift(1).over("gvkey"),
            datadate_lag=pl.col("datadate").shift(1).over("gvkey"),
        )
        .with_columns(
            inv=pl.when(
                pl.col("at").is_not_null()
                & pl.col("at_lag").is_not_null()
                & (pl.col("at_lag") > 0)
                & (pl.col("datadate").dt.year() - pl.col("datadate_lag").dt.year() == 1)
            )
            .then(safe_div(pl.col("at") - pl.col("at_lag"), pl.col("at_lag"), "inv"))
            .otherwise(None),
        )
    )


def _ff_global_be_op_inv(lf: pl.LazyFrame) -> pl.LazyFrame:
    """g_funda BE/OP/INV per JKP. comp.g_funda lacks pstkrv/pstkl/itcb: preferred-stock
    term collapses to pstk; txditc fallback collapses to txdb (+null preserves null
    semantics when txdb is null)."""
    ps_x = pl.col("pstk")
    txditc = pl.coalesce("txditc", pl.col("txdb") + pl.lit(None, dtype=pl.Float64))
    return (
        lf.with_columns(
            be_local=pl.coalesce(
                "seq", pl.col("ceq") + pl.coalesce(ps_x, 0.0), pl.col("at") - pl.col("lt")
            )
            + pl.coalesce(txditc, 0.0)
            - pl.coalesce(ps_x, 0.0),
            ope_x=pl.coalesce(
                "ebitda",
                "oibdp",
                pl.coalesce("sale", "revt") - pl.coalesce("xopr", pl.col("cogs") + pl.col("xsga")),
            )
            - pl.col("xint"),
            year=pl.col("datadate").dt.year(),
        )
        .sort(["gvkey", "datadate"])
        .with_columns(
            op=pl.when(pl.col("ope_x").is_not_null() & (pl.col("be_local") > 0))
            .then(safe_div(pl.col("ope_x"), pl.col("be_local"), "op"))
            .otherwise(None),
            at_lag=pl.col("at").shift(1).over("gvkey"),
            datadate_lag=pl.col("datadate").shift(1).over("gvkey"),
        )
        .with_columns(
            inv=pl.when(
                pl.col("at").is_not_null()
                & pl.col("at_lag").is_not_null()
                & (pl.col("at_lag") > 0)
                & (pl.col("datadate").dt.year() - pl.col("datadate_lag").dt.year() == 1)
            )
            .then(safe_div(pl.col("at") - pl.col("at_lag"), pl.col("at_lag"), "inv"))
            .otherwise(None),
        )
    )


def _ff_load_funda_na(raw_dir: Path) -> pl.LazyFrame:
    """Load comp_funda (NA) with the standard INDL/STD/D/C screen; floats
    include the NA-only pstkrv/pstkl/itcb. Collapses to one row per
    (gvkey, year) BEFORE be/op/inv are computed (_collapse_to_latest_fy)."""
    floats = [
        "pstk",
        "seq",
        "ceq",
        "lt",
        "txditc",
        "txdb",
        "revt",
        "sale",
        "cogs",
        "xsga",
        "xopr",
        "xint",
        "ebitda",
        "oibdp",
        "at",
        "pstkrv",
        "pstkl",
        "itcb",
    ]
    return (
        pl.scan_parquet(raw_dir / "comp_funda.parquet")
        .with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col(*floats).cast(pl.Float64),
        )
        .filter(_compustat_na_filter())
        .with_columns(year=pl.col("datadate").dt.year())
        .pipe(_collapse_to_latest_fy, resort=False)
    )


def _ff_load_funda_global(raw_dir: Path) -> pl.LazyFrame:
    """Load comp_g_funda (Global): INDL-or-FS / HIST_STD / I / C screen with
    INDL-over-FS preference when a (gvkey, datadate) carries both formats.
    Collapses to one row per (gvkey, year) BEFORE be/op/inv are computed."""
    floats = [
        "pstk",
        "seq",
        "ceq",
        "lt",
        "txditc",
        "txdb",
        "revt",
        "sale",
        "cogs",
        "xsga",
        "xopr",
        "xint",
        "ebitda",
        "oibdp",
        "at",
    ]
    return (
        pl.scan_parquet(raw_dir / "comp_g_funda.parquet")
        .with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col(*floats).cast(pl.Float64),
        )
        .filter(
            pl.col("indfmt").is_in(["INDL", "FS"])
            & (pl.col("datafmt") == "HIST_STD")
            & (pl.col("popsrc") == "I")
            & (pl.col("consol") == "C")
        )
        .with_columns(_n=pl.len().over(["gvkey", "datadate"]))
        .filter((pl.col("_n") == 1) | ((pl.col("_n") == 2) & (pl.col("indfmt") == "INDL")))
        .drop("_n")
        .with_columns(year=pl.col("datadate").dt.year())
        .pipe(_collapse_to_latest_fy, resort=False)
    )


def ff_load_world_compustat(raw_dir: Path, interim_dir: Path) -> pl.DataFrame:
    """
    Description:
        Worldwide Compustat BE/OP/INV for the ROW FF path: unions comp.funda (NA)
        and comp.g_funda (Global). JKP BE formula (no FASB-109 gate, no BE<0 cut);
        FF OP/INV formulas. BE converted local→USD via comp_exrt_dly.
    Steps:
        1) Load NA (INDL/STD/D/C) and Global (INDL-or-FS/HIST_STD/I/C,
           INDL-over-FS dedup); compute BE/OP/INV per each schema.
        2) Dedup by (gvkey, datadate) preferring NA, then by (gvkey, year)
           preferring NA then latest datadate.
        3) As-of join FX (local→USD) at datadate per curcd; multiply BE.
    Output:
        Eager [gvkey, datadate, year, be, op, inv, count] (BE in USD, OP/INV unitless).
    """
    na = _ff_na_be_op_inv(_ff_load_funda_na(raw_dir)).select(
        "gvkey", "datadate", "year", "be_local", "op", "inv", "curcd", _src=pl.lit(0)
    )
    gl = _ff_global_be_op_inv(_ff_load_funda_global(raw_dir)).select(
        "gvkey", "datadate", "year", "be_local", "op", "inv", "curcd", _src=pl.lit(1)
    )
    combined = (
        pl.concat([na, gl], how="vertical_relaxed")
        .sort(["gvkey", "datadate", "_src"])
        .unique(subset=["gvkey", "datadate"], keep="first")
        # A gvkey in both NA and Global yields two (gvkey, year) rows; collapse to
        # one (prefer NA, then latest datadate) so the June join can't fan out.
        .sort(["gvkey", "year", "_src", "datadate"], descending=[False, False, False, True])
        .unique(subset=["gvkey", "year"], keep="first")
        .drop("_src")
        .collect()
    )
    fx = (
        pl.scan_parquet(interim_dir / "raw_data_dfs" / "__fx1.parquet")
        .select(curcd=pl.col("curcdd"), datadate=pl.col("datadate"), fx=pl.col("fx"))
        .sort(["curcd", "datadate"])
        .collect()
        .set_sorted("datadate")
    )
    return (
        combined.sort(["curcd", "datadate"])
        .join_asof(fx, by="curcd", on="datadate", strategy="backward", check_sortedness=False)
        .with_columns(
            be=pl.when(pl.col("curcd") == "USD")
            .then(pl.col("be_local"))
            .otherwise(pl.col("be_local") * pl.col("fx"))
        )
        .filter(
            pl.col("be").is_not_null() | pl.col("op").is_not_null() | pl.col("inv").is_not_null()
        )
        .sort(["gvkey", "datadate"])
        .with_columns(count=pl.int_range(pl.len()).over("gvkey") + 1)
        .select("gvkey", "datadate", "year", "be", "op", "inv", "count")
    )


def ff_load_world_panel(interim_dir: Path, freq: str) -> pl.LazyFrame:
    """Scan world_{m|d}sf.parquet for ROW FF computation; drop USA + apply
    universe (common/obs_main/primary_sec/exch_main/me/gvkey-not-null).

    Daily world_dsf lacks gvkey + size_grp (added only at monthly upstream
    steps), so join those from world_msf on (id, eom). Returns a LazyFrame —
    fused with the downstream prepare chain and collected once per leg."""
    base_filter = (
        (pl.col("excntry") != US_EXCNTRY)
        & (pl.col("common") == 1)
        & (pl.col("obs_main") == 1)
        & (pl.col("primary_sec") == 1)
        & (pl.col("exch_main") == 1)
        & pl.col("me").is_not_null()
    )
    if freq == "monthly":
        return (
            pl.scan_parquet(interim_dir / "world_msf.parquet")
            .filter(base_filter & pl.col("gvkey").is_not_null())
            .select(
                "excntry",
                "id",
                "gvkey",
                pl.col("eom").alias("date"),
                "ret",
                "me",
                "size_grp",
            )
        )
    msf_keys = pl.scan_parquet(interim_dir / "world_msf.parquet").select(
        "id", "eom", "gvkey", "size_grp"
    )
    return (
        pl.scan_parquet(interim_dir / "world_dsf.parquet")
        .filter(base_filter)
        .select("excntry", "id", "eom", pl.col("date"), "ret", "me")
        .join(msf_keys, on=["id", "eom"], how="inner")
        .filter(pl.col("gvkey").is_not_null())
        .select("excntry", "id", "gvkey", "date", "ret", "me", "size_grp")
    )


def ff_country_breakpoints(
    june: pl.DataFrame,
    specs: list[tuple[str, float, str]],
    eligible: pl.Expr,
    min_stocks_bp: int = FF_MIN_STOCKS_BP,
) -> pl.DataFrame:
    """Per-(excntry, date) DuckDB QUANTILE_DISC over the country breakpoint pool.

    Breakpoint pool by country:
      - USA:  rows with exchcd_us == 1 (NYSE-only).
      - ROW:  rows with size_grp ∈ {small, large, mega} (ME ≥ US-NYSE p20).
              Country main-exchange (exch_main==1) gate applied upstream in
              ff_load_world_panel.

    ROW additionally drops (excntry, date) where pool count < FF_MIN_STOCKS_BP;
    USA is exempt (always emits breakpoints).
    """
    # US = NYSE-only (exchcd_us==1); ROW = size_grp pool (ME >= US-NYSE p20)
    pool = eligible & (
        ((pl.col("excntry") == US_EXCNTRY) & (pl.col("exchcd_us") == 1))
        | ((pl.col("excntry") != US_EXCNTRY) & pl.col("size_grp").is_in(["small", "large", "mega"]))
    )
    return country_breakpoints(
        june,
        specs,
        group_keys=["excntry", "date"],
        pool=pool,
        min_stocks=min_stocks_bp,
        usa_exempt=True,
    )


def _ff_size_breaks(
    june: pl.DataFrame, value_eligible: pl.Expr, min_stocks_bp: int = FF_MIN_STOCKS_BP
) -> pl.DataFrame:
    """Per-country size median (sizemedn). US pools ALL NYSE me>0 stocks per
    DFF 2000 (no BE requirement), while value breakpoints stay on the
    sort-eligible pool. ROW keeps the JKP convention (sort-eligible pool)."""
    size_elig = (
        pl.when(pl.col("excntry") == US_EXCNTRY).then(pl.col("me") > 0).otherwise(value_eligible)
    )
    return ff_country_breakpoints(june, [("me", 0.50, "sizemedn")], size_elig, min_stocks_bp)


def _ff_bm_breaks(june: pl.DataFrame, min_stocks_bp: int = FF_MIN_STOCKS_BP) -> pl.DataFrame:
    """B/M NYSE 30/70 breakpoints + all-NYSE size median per (excntry, date)."""
    bps = ff_country_breakpoints(
        june, [("beme", 0.30, "beme30"), ("beme", 0.70, "beme70")], _ff_bm_eligible(), min_stocks_bp
    )
    return _ff_size_breaks(june, _ff_bm_eligible(), min_stocks_bp).join(
        bps, on=["excntry", "date"], how="inner"
    )


def _ff_op_breaks(june: pl.DataFrame, min_stocks_bp: int = FF_MIN_STOCKS_BP) -> pl.DataFrame:
    """OP NYSE 30/70 breakpoints per (excntry, date)."""
    return ff_country_breakpoints(
        june, [("op", 0.30, "op30"), ("op", 0.70, "op70")], _ff_op_eligible(), min_stocks_bp
    )


def _ff_inv_breaks(june: pl.DataFrame, min_stocks_bp: int = FF_MIN_STOCKS_BP) -> pl.DataFrame:
    """INV NYSE 30/70 breakpoints per (excntry, date)."""
    return ff_country_breakpoints(
        june, [("inv", 0.30, "inv30"), ("inv", 0.70, "inv70")], _ff_inv_eligible(), min_stocks_bp
    )


def _ff_mom_breaks(mom_signal: pl.DataFrame, min_stocks_bp: int = FF_MIN_STOCKS_BP) -> pl.DataFrame:
    """Momentum NYSE 30/70 breakpoints + all-NYSE size median per
    (excntry, date); size dimension is ME at end of t-1 (caller swaps
    me <- me_lag1 before calling)."""
    bps = ff_country_breakpoints(
        mom_signal,
        [("mom_2_12", 0.30, "mom30"), ("mom_2_12", 0.70, "mom70")],
        _ff_mom_eligible(),
        min_stocks_bp,
    )
    return _ff_size_breaks(mom_signal, _ff_mom_eligible(), min_stocks_bp).join(
        bps, on=["excntry", "date"], how="inner"
    )


def ff_assign_portfolios(
    june: pl.DataFrame,
    bm_breaks: pl.DataFrame,
    op_breaks: pl.DataFrame,
    inv_breaks: pl.DataFrame,
) -> pl.DataFrame:
    """June Size + B/M + OP + INV portfolio assignment per (excntry, id, date).

    bm_breaks (with sizemedn) is inner-joined; op/inv breakpoints are
    left-joined. Each sort requires firm eligibility AND a non-null
    breakpoint; the size split requires only me > 0.
    """
    df = (
        june.join(bm_breaks, on=["excntry", "date"], how="inner")
        .join(op_breaks, on=["excntry", "date"], how="left")
        .join(inv_breaks, on=["excntry", "date"], how="left")
    )
    bm_elig = _ff_bm_eligible() & pl.col("beme30").is_not_null()
    op_elig = _ff_op_eligible() & pl.col("op30").is_not_null()
    inv_elig = _ff_inv_eligible() & pl.col("inv30").is_not_null()
    return df.with_columns(
        sizeport=pl.when(pl.col("me") > 0)
        .then(_ff_bucket("me", ["sizemedn"], ["S", "B"]))
        .otherwise(None),
        positivebeme=bm_elig.cast(pl.Int64),
        btmport=pl.when(bm_elig)
        .then(_ff_bucket("beme", ["beme30", "beme70"], ["L", "M", "H"]))
        .otherwise(None),
        nonmiss_op=op_elig.cast(pl.Int64),
        opport=pl.when(op_elig)
        .then(_ff_bucket("op", ["op30", "op70"], ["W", "N", "R"]))
        .otherwise(None),
        nonmiss_inv=inv_elig.cast(pl.Int64),
        invport=pl.when(inv_elig)
        .then(_ff_bucket("inv", ["inv30", "inv70"], ["C", "N", "A"]))
        .otherwise(None),
    ).select(
        "excntry",
        "id",
        "date",
        "sizeport",
        "btmport",
        "positivebeme",
        "opport",
        "nonmiss_op",
        "invport",
        "nonmiss_inv",
    )


def ff_assign_mom_portfolios(mom_signal: pl.DataFrame, mom_breaks: pl.DataFrame) -> pl.DataFrame:
    """Size + momentum portfolio assignment per (excntry, id, date).

    Size split is gated on momentum-leg eligibility (eligible_mom and
    positive lagged me), matching French's UMD construction.
    """
    df = mom_signal.join(mom_breaks, on=["excntry", "date"], how="inner")
    mom_elig = _ff_mom_eligible() & pl.col("mom30").is_not_null()
    return df.with_columns(
        sizeport=pl.when(pl.col("eligible_mom") & (pl.col("me_lag1") > 0))
        .then(_ff_bucket("me", ["sizemedn"], ["S", "B"]))
        .otherwise(None),
        nonmiss_mom=mom_elig.cast(pl.Int64),
        momport=pl.when(mom_elig)
        .then(_ff_bucket("mom_2_12", ["mom30", "mom70"], ["L", "N", "H"]))
        .otherwise(None),
    ).select("excntry", "id", "date", "sizeport", "momport", "nonmiss_mom")


def ff_assign_to_panel(panel: pl.DataFrame, june: pl.DataFrame) -> pl.DataFrame:
    """Broadcast (excntry, id, June(y)) assignments to the daily/monthly panel
    via port_year (Jul(y)..Jun(y+1) → formation-year y)."""
    return (
        panel.with_columns(_ff_port_year())
        .join(
            june.select(
                "excntry",
                "id",
                "sizeport",
                "btmport",
                "positivebeme",
                "opport",
                "nonmiss_op",
                "invport",
                "nonmiss_inv",
                june_year=pl.col("date").dt.year(),
            ),
            left_on=["excntry", "id", "port_year"],
            right_on=["excntry", "id", "june_year"],
            how="inner",
        )
        .drop("port_year")
    )


def vw_bucket_returns(
    frame: pl.DataFrame,
    *,
    buckets: list[str],
    weight_col: str,
    bucket_col: str = "bucket",
    ret_col: str = "ret",
    date_col: str = "date",
    min_stocks_pf: int,
    ordered: bool = False,
) -> pl.DataFrame:
    """Value-weight `ret_col` by `weight_col` per (excntry, date_col, bucket_col),
    apply the ROW min-stocks gate (USA bypassed), and pivot wide to `buckets`
    (absent buckets filled as null Float64). Shared by the FF/HXZ/DHS size×char
    sorts; the caller supplies the pre-built `bucket_col` and has already applied any
    weight/return validity filter. `ordered`=True keeps input row order within each
    group (maintain_order + a group-key presort) for float-sum determinism at a
    discrete breakpoint; FF/HXZ leave it False. `safe_div` returns null on a zero
    weight sum. Output: [excntry, date_col, *buckets]."""
    if ordered:
        frame = frame.sort(["excntry", date_col, bucket_col], maintain_order=True)
    wide = (
        frame.group_by(["excntry", date_col, bucket_col], maintain_order=ordered)
        .agg(
            vwret=safe_div(
                (pl.col(ret_col) * pl.col(weight_col)).sum(), pl.col(weight_col).sum(), "vwret"
            ),
            _n=pl.len(),
        )
        .with_columns(
            vwret=pl.when((pl.col("excntry") == US_EXCNTRY) | (pl.col("_n") >= min_stocks_pf))
            .then(pl.col("vwret"))
            .otherwise(None)
        )
        .pivot(values="vwret", index=["excntry", date_col], on=bucket_col)
    )
    missing = [c for c in buckets if c not in wide.columns]
    if missing:
        wide = wide.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
    return wide.select("excntry", date_col, *buckets)


def _ff_vw_leg(
    panel: pl.DataFrame,
    *,
    port: str,
    flag: str,
    buckets: list[str],
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
) -> pl.DataFrame:
    """FF size×`port` VW leg → [excntry, date, *buckets] via vw_bucket_returns. The
    caller has already applied the shared w>0 / ret-not-null / sizeport-not-null
    filter."""
    return vw_bucket_returns(
        panel.filter((pl.col(flag) == 1) & pl.col(port).is_not_null()).with_columns(
            bucket=pl.col("sizeport") + pl.col(port)
        ),
        buckets=buckets,
        weight_col="w",
        min_stocks_pf=min_stocks_pf,
    )


def ff_compute_factors(ccm4: pl.DataFrame, min_stocks_pf: int = FF_MIN_STOCKS_PF) -> pl.DataFrame:
    """3 independent 2x3 sorts → per (excntry, date) factors.

    For each sort (BM/OP/INV): VW per (excntry, date, sizeport+sortport)
    bucket via `_ff_vw_leg`. Legs joined on (excntry, date).

    smb_ff3 = BM size leg only; smb_ff5 = average of BM/OP/INV size legs.
    """
    # One shared-predicate pass over the full panel instead of three;
    # filter(a).filter(b) ≡ filter(a & b) row-for-row, so each leg's group
    # float-sum sequence is unchanged.
    base = ccm4.filter(
        (pl.col("w") > 0) & pl.col("ret").is_not_null() & pl.col("sizeport").is_not_null()
    )
    bm_leg = _ff_vw_leg(
        base,
        port="btmport",
        flag="positivebeme",
        buckets=["SL", "SM", "SH", "BL", "BM", "BH"],
        min_stocks_pf=min_stocks_pf,
    )
    op_leg = _ff_vw_leg(
        base,
        port="opport",
        flag="nonmiss_op",
        buckets=["SW", "SN", "SR", "BW", "BN", "BR"],
        min_stocks_pf=min_stocks_pf,
    ).rename({"SN": "SN_op", "BN": "BN_op"})
    inv_leg = _ff_vw_leg(
        base,
        port="invport",
        flag="nonmiss_inv",
        buckets=["SC", "SN", "SA", "BC", "BN", "BA"],
        min_stocks_pf=min_stocks_pf,
    ).rename({"SN": "SN_inv", "BN": "BN_inv"})
    wide = (
        bm_leg.join(op_leg, on=["excntry", "date"], how="full", coalesce=True)
        .join(inv_leg, on=["excntry", "date"], how="full", coalesce=True)
        .sort("excntry", "date")
    )

    smb_bm = pl.mean_horizontal("SL", "SM", "SH", ignore_nulls=False) - pl.mean_horizontal(
        "BL", "BM", "BH", ignore_nulls=False
    )
    smb_op = pl.mean_horizontal("SW", "SN_op", "SR", ignore_nulls=False) - pl.mean_horizontal(
        "BW", "BN_op", "BR", ignore_nulls=False
    )
    smb_inv = pl.mean_horizontal("SC", "SN_inv", "SA", ignore_nulls=False) - pl.mean_horizontal(
        "BC", "BN_inv", "BA", ignore_nulls=False
    )
    return wide.select(
        pl.col("excntry"),
        pl.col("date"),
        smb_ff3=smb_bm,
        smb_ff5=(smb_bm + smb_op + smb_inv) / 3,
        hml_ff=pl.mean_horizontal("SH", "BH", ignore_nulls=False)
        - pl.mean_horizontal("SL", "BL", ignore_nulls=False),
        rmw_ff=pl.mean_horizontal("SR", "BR", ignore_nulls=False)
        - pl.mean_horizontal("SW", "BW", ignore_nulls=False),
        cma_ff=pl.mean_horizontal("SC", "BC", ignore_nulls=False)
        - pl.mean_horizontal("SA", "BA", ignore_nulls=False),
    )


def _ff_is_neg99(col: str) -> pl.Expr:
    return pl.col(col).is_not_null() & (
        (pl.col(col) - FF_MISSING_RET_CODE).abs() < FF_MISSING_RET_TOL
    )


def _ff_neg99_to_zero(expr: pl.Expr) -> pl.Expr:
    """Replace -99.0 sentinel with 0; pass through null untouched."""
    return (
        pl.when((expr - FF_MISSING_RET_CODE).abs() < FF_MISSING_RET_TOL)
        .then(pl.lit(0.0))
        .otherwise(expr)
    )


def _ff_mom_signal_monthly(panel: pl.LazyFrame) -> pl.LazyFrame:
    L = FF_UMD_MONTHLY_LOOKBACK
    skip = FF_UMD_MONTHLY_SKIP
    by = ["excntry", "id"]
    idx = pl.col("date").dt.year() * 12 + pl.col("date").dt.month()

    df = (
        panel.with_columns(mth_idx=idx.cast(pl.Int64))
        .sort([*by, "mth_idx"])
        .with_columns(
            pl.col("me").shift(1).over(by).alias("me_lag1"),
            pl.col("me").shift(L + 1).over(by).alias(f"me_lag{L + 1}"),
            *[pl.col("ret").shift(k).over(by).alias(f"ret_lag{k}") for k in range(1, L + 1)],
            *[
                (pl.col("mth_idx") - pl.col("mth_idx").shift(k).over(by)).alias(f"gap{k}")
                for k in (*range(1, L + 1), L + 1)
            ],
        )
        .with_columns(
            pl.when(pl.col("gap1") == 1).then(pl.col("me_lag1")).otherwise(None).alias("me_lag1"),
            pl.when(pl.col(f"gap{L + 1}") == L + 1)
            .then(pl.col(f"me_lag{L + 1}"))
            .otherwise(None)
            .alias(f"me_lag{L + 1}"),
            *[
                pl.when(pl.col(f"gap{k}") == k)
                .then(pl.col(f"ret_lag{k}"))
                .otherwise(None)
                .alias(f"ret_lag{k}")
                for k in range(1, L + 1)
            ],
        )
    )

    elig = (
        pl.col("me_lag1").is_not_null()
        & (pl.col("me_lag1") > 0)
        & pl.col(f"me_lag{L + 1}").is_not_null()
        & pl.col(f"ret_lag{skip + 1}").is_not_null()
        & ~_ff_is_neg99(f"ret_lag{skip + 1}")
    )
    # Interior months (t-12..t-3): French requires the row on file but tolerates a
    # missing price (-99.0 on CRSP; null under CIZ), so the US gate needs only gap==k
    # and the compounding below zero-fills. ROW keeps the strict non-null gate (JKP).
    is_us = pl.col("excntry") == US_EXCNTRY
    for k in range(skip + 2, L + 1):
        elig = elig & pl.when(is_us).then(pl.col(f"gap{k}") == k).otherwise(
            pl.col(f"ret_lag{k}").is_not_null()
        )

    terms = [
        1.0 + _ff_neg99_to_zero(pl.col(f"ret_lag{k}")).fill_null(0.0)
        for k in range(skip + 1, L + 1)
    ]
    mom_expr = functools.reduce(operator.mul, terms) - 1.0

    return df.with_columns(
        eligible_mom=elig,
        mom_2_12=pl.when(elig).then(mom_expr).otherwise(None),
    ).select(
        "excntry",
        "id",
        "date",
        "ret",
        "w",
        "me",
        "exchcd_us",
        "size_grp",
        "me_lag1",
        "mom_2_12",
        "eligible_mom",
    )


def _ff_mom_signal_daily(panel: pl.LazyFrame) -> pl.LazyFrame:
    skip = FF_UMD_DAILY_SKIP
    look = FF_UMD_DAILY_LOOKBACK
    by = ["excntry", "id"]

    trading_idx = (
        panel.select("excntry", "date")
        .unique()
        .sort(["excntry", "date"])
        .with_columns(tidx=pl.int_range(0, pl.len()).cast(pl.Int64).over("excntry"))
    )
    # US window (French daily detail): returns t-250..t-21 (t-21 the last day, mirrors
    # monthly t-2), price anchor t-251, interior missing days tolerated (-99.0/CIZ-null,
    # log_eff zero-fills). ROW keeps the legacy t-251..t-22 window + zero-missing gate.
    is_us = pl.col("excntry") == US_EXCNTRY
    skip_eff = pl.when(is_us).then(skip).otherwise(skip + 1)

    def _us_row_lag(col: str, us_n: int) -> pl.Expr:
        """Lag by us_n (US) / us_n + 1 (ROW) — the window anchors above."""
        return (
            pl.when(is_us)
            .then(pl.col(col).shift(us_n).over(by))
            .otherwise(pl.col(col).shift(us_n + 1).over(by))
        )

    df = (
        panel.join(trading_idx, on=["excntry", "date"], how="left")
        .sort([*by, "tidx"])
        .with_columns(
            is_null_ret=pl.col("ret").is_null(),
            log_eff=(1.0 + _ff_neg99_to_zero(pl.col("ret")).fill_null(0.0)).log(),
        )
        .with_columns(
            cum_log=pl.col("log_eff").cum_sum().over(by),
            cum_null=pl.col("is_null_ret").cast(pl.Int64).cum_sum().over(by),
        )
        .with_columns(
            gap_one=pl.col("tidx") - pl.col("tidx").shift(1).over(by),
            gap_skip=pl.col("tidx") - _us_row_lag("tidx", skip),
            gap_look=pl.col("tidx") - pl.col("tidx").shift(look).over(by),
            cum_log_lag_skip=_us_row_lag("cum_log", skip),
            cum_log_lag_look=_us_row_lag("cum_log", look),
            nulls_in_window=(
                pl.col("cum_null").shift(skip + 1).over(by)
                - pl.col("cum_null").shift(look + 1).over(by)
            ),
            ret_lag_skip=pl.col("ret").shift(skip).over(by),
            me_lag1=pl.col("me").shift(1).over(by),
            me_lag_look=pl.col("me").shift(look).over(by),
        )
    )
    elig = (
        (pl.col("gap_one") == 1)
        & (pl.col("gap_skip") == skip_eff)
        & (pl.col("gap_look") == look)
        & pl.col("me_lag1").is_not_null()
        & (pl.col("me_lag1") > 0)
        & pl.col("ret_lag_skip").is_not_null()
        & ~_ff_is_neg99("ret_lag_skip")
        & pl.col("me_lag_look").is_not_null()
        & (is_us | (pl.col("nulls_in_window") == 0))
    )
    mom_raw = (pl.col("cum_log_lag_skip") - pl.col("cum_log_lag_look")).exp() - 1.0
    return df.with_columns(
        eligible_mom=elig,
        mom_2_12=pl.when(elig).then(mom_raw).otherwise(None),
    ).select(
        "excntry",
        "id",
        "date",
        "ret",
        "w",
        "me",
        "exchcd_us",
        "size_grp",
        "me_lag1",
        "mom_2_12",
        "eligible_mom",
    )


def ff_build_mom_signal(panel: pl.DataFrame, freq: Literal["monthly", "daily"]) -> pl.DataFrame:
    """Per-(excntry, id) prior (2-12) return signal + eligibility.

    Dispatches to monthly (calendar-month-indexed lags, sentinel-aware
    compound product) or daily (per-excntry trading-day rank, cum_log
    diff; US window t-250..t-21 with interior missing-price tolerance
    per French's detail pages, ROW t-251..t-22 zero-missing per the
    JKP convention). Builds lazily and collects once at the boundary —
    Polars fuses the chained with_columns into one streaming pass.

    Output: panel + [me_lag1, mom_2_12, eligible_mom].
    """
    lf = panel.lazy()
    out = _ff_mom_signal_monthly(lf) if freq == "monthly" else _ff_mom_signal_daily(lf)
    # Streaming engine: ~35% faster on the daily panel (312M rows) and
    # verified bit-identical to the in-memory engine on full data.
    return out.collect(engine="streaming")


def ff_compute_umd_factor(
    panel: pl.DataFrame, min_stocks_pf: int = FF_MIN_STOCKS_PF
) -> pl.DataFrame:
    """2x3 size x mom_2_12 → per (excntry, date) umd_ff.

    Single-sort momentum leg collapsed to the UMD spread. Same per-bucket
    VW + ROW min-stocks gate as the other FF sorts.
    """
    base = panel.filter(
        (pl.col("w") > 0) & pl.col("ret").is_not_null() & pl.col("sizeport").is_not_null()
    )
    return (
        _ff_vw_leg(
            base,
            port="momport",
            flag="nonmiss_mom",
            buckets=["SL", "SN", "SH", "BL", "BN", "BH"],
            min_stocks_pf=min_stocks_pf,
        )
        .select(
            "excntry",
            "date",
            umd_ff=0.5 * (pl.col("SH") + pl.col("BH")) - 0.5 * (pl.col("SL") + pl.col("BL")),
        )
        .sort("excntry", "date")
    )


def ff_build_characteristics(
    panel: pl.DataFrame,
    june: pl.DataFrame,
    freq: Literal["monthly", "daily"],
    mom_signal: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-(excntry, id, eom) characteristics on monthly grid. Daily input
    downsampled to last trading day per (excntry, id, month). The returned
    frame carries `excntry`; the writer in `gen_ff_data` drops it (`id` is
    globally unique).

    If `mom_signal` is provided (output of `ff_build_mom_signal`), the
    per-stock `mom_2_12` is joined on (excntry, id, date) and emitted as
    `umd_ff`. Caller should pass the MONTHLY signal so the characteristic
    aligns to monthly eom rows.
    """
    p = panel.with_columns(_ff_port_year()).join(
        june.select("excntry", "id", "beme", "op", "inv", fy=pl.col("date").dt.year()),
        left_on=["excntry", "id", "port_year"],
        right_on=["excntry", "id", "fy"],
    )
    if freq == "daily":
        p = (
            p.with_columns(_month=pl.col("date").dt.truncate("1mo"))
            .sort(["excntry", "id", "date"])
            .unique(subset=["excntry", "id", "_month"], keep="last")
            .drop("_month")
        )
    if mom_signal is not None:
        p = p.join(
            mom_signal.select("excntry", "id", "date", "mom_2_12"),
            on=["excntry", "id", "date"],
            how="left",
        )
    cols = {
        "me_ff": pl.col("me"),
        "beme_ff": pl.col("beme"),
        "op_ff": pl.col("op"),
        "inv_ff": pl.col("inv"),
    }
    if mom_signal is not None:
        cols["umd_ff"] = pl.col("mom_2_12")
    return p.select("date", "excntry", "id", **cols).sort("date", "excntry", "id")


def _ff_build_freq(
    freq: Literal["monthly", "daily"],
    ccm2a: pl.DataFrame,
    comp_world: pl.DataFrame,
    raw_dir: Path,
    interim_dir: Path,
    min_stocks_bp: int = FF_MIN_STOCKS_BP,
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Build FF3/FF5/UMD factors for one frequency; also returns monthly characteristics.

    Output: (factors, chars) — chars is the characteristics frame for monthly, None for daily.
    """
    # Both legs as fused lazy plans (loader scan + weights chain), executed
    # together by polars' own scheduler — single collect_all, no Python
    # threads. The chars tails then run eagerly off the collected panels.
    us_lf = ff_load_crsp_panel(raw_dir, freq).pipe(_ff_us_rets_weights_lazy, freq)
    row_lf = ff_load_world_panel(interim_dir, freq).pipe(_ff_row_rets_weights_lazy)
    us_rw, row_rw = pl.collect_all([us_lf, row_lf])
    us_panel, us_chars = _ff_us_finish_prepare(us_rw, ccm2a)
    row_panel, row_chars = _ff_row_finish_prepare(row_rw, comp_world)
    panel = pl.concat([us_panel, row_panel], how="vertical_relaxed")
    data_chars = pl.concat([us_chars, row_chars], how="vertical_relaxed")

    ports = ff_assign_portfolios(
        data_chars,
        _ff_bm_breaks(data_chars, min_stocks_bp),
        _ff_op_breaks(data_chars, min_stocks_bp),
        _ff_inv_breaks(data_chars, min_stocks_bp),
    )
    factors = ff_compute_factors(ff_assign_to_panel(panel, ports), min_stocks_pf)

    # UMD: same per-country breakpoints + ROW gates as the other FF sorts, but size is
    # ME at t-1, so swap me ← me_lag1 first. mom_signal feeds ff_build_characteristics'
    # umd_ff column.
    mom_signal = ff_build_mom_signal(panel, freq).with_columns(me=pl.col("me_lag1"))
    ports_mom = ff_assign_mom_portfolios(mom_signal, _ff_mom_breaks(mom_signal, min_stocks_bp))
    umd = ff_compute_umd_factor(
        mom_signal.join(
            ports_mom.select("excntry", "id", "date", "sizeport", "momport", "nonmiss_mom"),
            on=["excntry", "id", "date"],
            how="left",
        ).with_columns(w=pl.col("me_lag1")),
        min_stocks_pf,
    )
    factors = factors.join(umd, on=["excntry", "date"], how="left")

    if freq != "monthly":
        return factors, None

    # count is non-null iff the June stock had a Compustat/DFF row; with the dec_me
    # condition this reproduces the pre-left-join chars row set (extra rows only widen
    # the US size-median / OP-INV pool).
    chars = ff_build_characteristics(
        panel,
        data_chars.filter(pl.col("count").is_not_null() & pl.col("dec_me").is_not_null()),
        freq,
        mom_signal=mom_signal,
    )
    return factors, chars


@measure_time
def gen_ff_data(
    paths: DataPaths,
    monthly_factors_path: Path,
    daily_factors_path: Path,
    chars_path: Path,
    freqs: tuple[str, ...] = ("monthly", "daily"),
    min_stocks_bp: int = FF_MIN_STOCKS_BP,
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
) -> None:
    """
    Description:
        Reference: Fama and French (1993) 'Common risk factors in the returns
        on stocks and bonds', Journal of Financial Economics 33(1), pp. 3-56;
        Fama and French (2015) 'A five-factor asset pricing model', Journal of
        Financial Economics 116(1), pp. 1-22. WRDS "Fama-French Factors
        (CIZ Format)":
        https://wrds-www.wharton.upenn.edu/login/?next=/pages/wrds-research/applications/risk-factors-and-industry-benchmarks/fama-french-factors-ciz-format/
        Historical book equity from Ken French's data library:
        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#BookEquity

        Build FF3/FF5/UMD factor and characteristic files for all countries.
        US follows strict-FF methodology on the CIZ universe: raw CRSP CIZ
        tables + Compustat funda, CIZ universe filters, permco-level ME
        aggregation, NYSE QUANTILE_DISC breakpoints, July-to-June portfolio
        assignment; numerics track the FF data-library reference within
        1 ULP. Reads raw CRSP/Compustat from `paths.raw_tables_dir` and
        world_msf/world_dsf from `paths.interim_dir`; output paths are passed
        in explicitly by the caller (composed against `paths.interim_dir`).

    Steps:
        1) Build US panel/June via raw CRSP path (FF-strict). Tag excntry,
           rename retadj→ret, weight_port→w, exchcd→exchcd_us; size_grp=null.
        2) Build ROW panel/June via world_msf path (JKP-spec). Compute
           me_lag1→w; rename ret_exc→ret; exchcd_us=null.
        3) Concat US + ROW panels and Junes (vertical_relaxed).
        4) Compute breakpoints (per (excntry, date), branched pool filter +
           ROW min-stocks gate), assign portfolios, broadcast to panel, VW
           per (excntry, date, bucket) with ROW min-stocks gate, factors.

    Output:
        - <monthly_factors_path>, <daily_factors_path>
              [excntry, eom/date, smb_ff3, smb_ff5, hml_ff, rmw_ff, cma_ff, umd_ff]
        - <chars_path>
              [eom, id, me_ff, beme_ff, op_ff, inv_ff, umd_ff]
              (excntry dropped: `id` is globally unique)
    """
    raw_dir = paths.raw_tables_dir
    interim_dir = paths.interim_dir
    out_paths = {
        "monthly": monthly_factors_path,
        "daily": daily_factors_path,
    }
    chars_out = chars_path

    # US Compustat (FF-strict)
    ccm2a = ff_load_compustat_us(raw_dir, ff5=True, use_dff=FF_USE_DFF_BE)

    # ROW Compustat (JKP)
    comp_world = ff_load_world_compustat(raw_dir, interim_dir)

    for freq in freqs:
        factors, chars = _ff_build_freq(
            freq, ccm2a, comp_world, raw_dir, interim_dir, min_stocks_bp, min_stocks_pf
        )
        dt = "eom" if freq == "monthly" else "date"
        factors.rename({"date": dt}).write_parquet(out_paths[freq])
        if chars is not None:
            chars.rename({"date": "eom"}).drop("excntry").write_parquet(chars_out)


def hxz_attach_siccd(lf: pl.LazyFrame, raw_dir: Path, date_col: str) -> pl.LazyFrame:
    """Event-time join with stksecurityinfohist on permno where
    secinfostartdt ≤ `date_col` ≤ secinfoenddt; adds siccd + is_fin."""
    sec = (
        pl.scan_parquet(raw_dir / "crsp_stksecurityinfohist.parquet")
        .select("permno", "secinfostartdt", "secinfoenddt", "siccd")
        .unique()
    )
    # Tie-break for overlapping CRSP security-history windows: newest secinfostartdt
    # (latest re-classification = current industry), then siccd ASC for determinism.
    return (
        lf.join(sec, on="permno", how="left")
        .filter(
            pl.col("secinfostartdt").is_null()
            | (pl.col("secinfostartdt") <= pl.col(date_col))
            & (pl.col("secinfoenddt") >= pl.col(date_col))
        )
        .sort(
            ["permno", date_col, "secinfostartdt", "siccd"],
            descending=[False, False, True, False],
            nulls_last=True,
        )
        .unique(subset=["permno", date_col], keep="first", maintain_order=True)
        .with_columns(is_fin=pl.col("siccd").is_between(6000, 6999))
        .drop("secinfostartdt", "secinfoenddt")
    )


def hxz_load_crsp(
    raw_dir: Path, freq: Literal["monthly", "daily"], date_start: date | None = None
) -> pl.DataFrame:
    """Load CRSP CIZ (msf_v2 or dsf_v2); permno-level (no permco aggregation).

    Output: Eager [permno, permco, date, ret, retx, prc, shrout, me, exchcd, siccd, is_fin].
    """
    pq, p = ("crsp_msf_v2.parquet", "mth") if freq == "monthly" else ("crsp_dsf_v2.parquet", "dly")
    date_c, ret_c, retx_c, prc_c = f"{p}caldt", f"{p}ret", f"{p}retx", f"{p}prc"
    date_floor = pl.lit(True) if date_start is None else pl.col(date_c) >= date_start
    return (
        pl.scan_parquet(raw_dir / pq)
        .filter(date_floor & _ciz_universe())
        .with_columns(pl.col(ret_c, retx_c, prc_c).cast(pl.Float64))
        .pipe(hxz_attach_siccd, raw_dir, date_c)
        .select(
            "permno",
            "permco",
            pl.col(date_c).alias("date"),
            pl.col(ret_c).alias("ret"),
            pl.col(retx_c).alias("retx"),
            pl.col(prc_c).alias("prc"),
            "shrout",
            (pl.col(prc_c).abs() * pl.col("shrout")).alias("me"),
            pl.col("primaryexch")
            .replace_strict({"N": 1, "A": 2, "Q": 3}, default=None, return_dtype=pl.Int64)
            .alias("exchcd"),
            "siccd",
            "is_fin",
        )
        .collect()
    )


def _compustat_na_filter() -> pl.Expr:
    """Standard Compustat NA screen (INDL/STD/D/C), shared by the FF, HXZ and
    mispricing funda/fundq loaders."""
    return (
        (pl.col("indfmt") == "INDL")
        & (pl.col("datafmt") == "STD")
        & (pl.col("popsrc") == "D")
        & (pl.col("consol") == "C")
    )


def hxz_load_funda(raw_dir: Path) -> pl.DataFrame:
    """Annual BE and I/A per (gvkey, fiscal year).

    BE: SEQ → CEQ+PSTK → AT-LT, plus TXDITC unconditionally,
    minus preferred (PSTKRV → PSTKL → PSTK).

    Output: Eager [gvkey, datadate, be_a, inv, csho, ajex].
    """
    at_lag = pl.col("at").shift(1).over("gvkey")
    fy_gap = pl.col("datadate").dt.year() - pl.col("datadate").shift(1).over("gvkey").dt.year()
    floats = ["pstkrv", "pstkl", "pstk", "seq", "ceq", "txditc", "lt", "at", "csho", "ajex"]
    return (
        pl.scan_parquet(raw_dir / "comp_funda.parquet")
        .filter(_compustat_na_filter())
        .with_columns(pl.col(*floats).cast(pl.Float64), pl.col("datadate").cast(pl.Date))
        .with_columns(
            ps_a=pl.coalesce("pstkrv", "pstkl", "pstk", pl.lit(0.0)),
            sh_a=pl.coalesce("seq", pl.col("ceq") + pl.col("pstk"), pl.col("at") - pl.col("lt")),
        )
        .with_columns(be_a=pl.col("sh_a") + pl.col("txditc").fill_null(0.0) - pl.col("ps_a"))
        # BEFORE the inv lag; mirrors ff_load_compustat_us. (year is transient;
        # dropped by the select.)
        .with_columns(year=pl.col("datadate").dt.year())
        .pipe(_collapse_to_latest_fy)
        .with_columns(
            inv=pl.when(at_lag.is_not_null() & (at_lag > 0) & (fy_gap == 1))
            .then((pl.col("at") - at_lag) / at_lag)
            .otherwise(None)
        )
        .select("gvkey", "datadate", "be_a", "inv", "csho", "ajex")
        .collect()
    )


def hxz_load_fundq(raw_dir: Path) -> pl.DataFrame:
    """Compustat quarterly BEQ: SEQQ → CEQQ+PSTKQ → ATQ−LTQ, plus TXDITCQ,
    minus PS_q (PSTKRQ → PSTKQ → 0). One row per (gvkey, fyearq, fqtr).

    Output: Eager [gvkey, datadate, fyearq, fqtr, rdq, ibq, dvpsxq, cshoq, ajexq, beq].
    """
    floats = [
        "ibq", "dvpsxq", "cshoq", "ajexq",
        "seqq", "ceqq", "pstkq", "atq", "ltq", "txditcq", "pstkrq",
    ]  # fmt: skip
    return (
        pl.scan_parquet(raw_dir / "comp_fundq.parquet")
        .filter(_compustat_na_filter())
        .with_columns(
            pl.col(*floats).cast(pl.Float64),
            pl.col("datadate").cast(pl.Date),
            pl.col("rdq").cast(pl.Date),
        )
        .with_columns(
            ps_q=pl.coalesce("pstkrq", "pstkq", pl.lit(0.0)),
            sh_q=pl.coalesce(
                "seqq", pl.col("ceqq") + pl.col("pstkq"), pl.col("atq") - pl.col("ltq")
            ),
        )
        .with_columns(beq=pl.col("sh_q") + pl.col("txditcq").fill_null(0.0) - pl.col("ps_q"))
        # One row per (gvkey, fiscal quarter): FYE changes can file the same
        # (fyearq, fqtr) under two datadate, and dups fan the dense qidx axis in
        # hxz_impute_be_clean_surplus -> nondeterministic roe_hxz. Keep latest
        # datadate, then rdq, then the more valued filing.
        .sort(
            ["gvkey", "fyearq", "fqtr", "datadate", "rdq", "ibq", "beq"],
            descending=[False, False, False, True, True, True, True],
            nulls_last=True,
        )
        .unique(subset=["gvkey", "fyearq", "fqtr"], keep="first", maintain_order=True)
        .sort(["gvkey", "fyearq", "fqtr"])
        .select(
            "gvkey",
            "datadate",
            "fyearq",
            "fqtr",
            "rdq",
            "ibq",
            "dvpsxq",
            "cshoq",
            "ajexq",
            "beq",
        )  # fmt: skip
        .collect()
    )


def hxz_supplement_q4_shares(
    fundq: pl.DataFrame, funda: pl.DataFrame, raw_dir: Path
) -> pl.DataFrame:
    """HXZ footnote-3 share supplementation: cshoq/ajexq → annual csho/ajex on Q4 rows
    → CRSP shrout (thousands) + mthcumfacshr at fiscal-quarter month-end.

    Output: fundq schema + cshoq_eff, ajexq_eff.
    """
    df = (
        fundq.lazy()
        .join(
            funda.lazy().select("gvkey", "datadate", "csho", "ajex"),
            on=["gvkey", "datadate"],
            how="left",
        )
        .with_columns(
            cshoq_eff=pl.coalesce(
                "cshoq", pl.when(pl.col("fqtr") == 4).then(pl.col("csho")).otherwise(None)
            ),
            ajexq_eff=pl.coalesce(
                "ajexq", pl.when(pl.col("fqtr") == 4).then(pl.col("ajex")).otherwise(None)
            ),
        )
        .drop("csho", "ajex")
    )
    needs = (
        df.filter(pl.col("cshoq_eff").is_null() | pl.col("ajexq_eff").is_null())
        .select("gvkey", "datadate")
        .unique()
    )
    lnk = (
        pl.scan_parquet(raw_dir / "crsp_ccmxpf_lnkhist.parquet")
        .filter(pl.col("linktype").str.starts_with("L") & pl.col("linkprim").is_in(["P", "C"]))
        .select("gvkey", "lpermno", "linkprim", "linktype", "linkdt", "linkenddt")
        .with_columns(
            pl.col("lpermno").cast(pl.Int64),
            pl.col("linkdt", "linkenddt").cast(pl.Date),
        )
    )
    nl = (
        needs.join(lnk, on="gvkey", how="inner")
        .filter(
            (pl.col("linkdt") <= pl.col("datadate"))
            & (pl.col("linkenddt").is_null() | (pl.col("linkenddt") >= pl.col("datadate")))
        )
        .with_columns(
            _link_days=(
                pl.coalesce(pl.col("linkenddt"), pl.lit(date(9999, 12, 31))) - pl.col("linkdt")
            ).dt.total_days()
        )
        .pipe(_hxz_best_link, subset=["gvkey", "datadate"], tail_key="lpermno")
        .drop("_link_days")
    )
    cfac = (
        pl.scan_parquet(raw_dir / "crsp_msf_v2.parquet")
        .select(
            "permno",
            pl.col("mthcaldt").dt.month_end().alias("eom"),
            pl.col("shrout").cast(pl.Float64),
            pl.col("mthcumfacshr").cast(pl.Float64).alias("cfacshr"),
        )
        # CIZ msf_v2 has one row per (permno, mthcaldt); after month_end
        # bucketing the (permno, eom) pair is unique. Sort enforces stable
        # order pre-unique so any future schema quirk stays deterministic.
        .sort(["permno", "eom", "cfacshr", "shrout"], nulls_last=True)
        .unique(subset=["permno", "eom"], keep="last", maintain_order=True)
    )
    nl = (
        nl.with_columns(eom=pl.col("datadate").dt.month_end())
        .rename({"lpermno": "permno"})
        .join(cfac, on=["permno", "eom"], how="left")
        .select("gvkey", "datadate", "shrout", "cfacshr")
    )
    return (
        df.join(nl, on=["gvkey", "datadate"], how="left")
        .with_columns(
            cshoq_eff=pl.coalesce("cshoq_eff", safe_div(pl.col("shrout"), 1000.0, "cshoq_eff")),
            ajexq_eff=pl.coalesce("ajexq_eff", "cfacshr"),
        )
        .drop("shrout", "cfacshr")
        .collect()
    )


def hxz_merge_be_chain(fundq: pl.DataFrame, funda: pl.DataFrame) -> pl.DataFrame:
    """Q4 BEQ ← annual BE_a where quarterly is missing."""
    return (
        fundq.join(funda.select("gvkey", "datadate", "be_a"), on=["gvkey", "datadate"], how="left")
        .with_columns(
            beq=pl.when((pl.col("fqtr") == 4) & pl.col("beq").is_null())
            .then(pl.col("be_a"))
            .otherwise(pl.col("beq"))
        )
        .drop("be_a")
    )


def hxz_impute_be_clean_surplus(fq: pl.DataFrame) -> pl.DataFrame:
    """
    Description:
        Backward + forward (1≤j≤4) clean-surplus impute of BEQ on a dense
        calendar-quarter axis (HXZ Appendix A.4).
    Steps:
        1) Pad missing quarters per gvkey (qidx = fyearq*4 + fqtr − 1).
        2) DVQ = dvpsxq × lag(cshoq_eff) × (lag ajexq_eff / ajexq_eff).
        3) Backward: BEQ_{t−1} = BEQ_t − IBQ_t + DVQ_t (when t−1 null).
        4) Forward: BEQ_t = BEQ_{t−j} + ΣIBQ − ΣDVQ over j-window, 1≤j≤4,
           smallest j wins via coalesce.
        5) Compute calendar-q-lagged BEQ (beq_lag1) for Roe.
    Output:
        DataFrame with imputed beq + beq_lag1, padded rows dropped.
    """
    df = fq.with_columns(qidx=pl.col("fyearq") * 4 + pl.col("fqtr") - 1)
    ranges = df.group_by("gvkey").agg(qmin=pl.col("qidx").min(), qmax=pl.col("qidx").max())
    dense = (
        ranges.with_columns(qidx=pl.int_ranges("qmin", pl.col("qmax") + 1))
        .explode("qidx")
        .select("gvkey", "qidx")
    )
    df = dense.join(df, on=["gvkey", "qidx"], how="left").sort(["gvkey", "qidx"]).lazy()

    cshoq_lag = pl.col("cshoq_eff").shift(1).over("gvkey")
    ajexq_lag = pl.col("ajexq_eff").shift(1).over("gvkey")
    df = df.with_columns(
        dvq=pl.when(pl.col("dvpsxq").fill_null(0.0) > 0)
        .then(
            pl.col("dvpsxq")
            * cshoq_lag
            * pl.when(pl.col("ajexq_eff") > 0)
            .then(safe_div(ajexq_lag, pl.col("ajexq_eff"), "_ajexq_ratio"))
            .otherwise(1.0)
        )
        .otherwise(0.0)
    ).with_columns(
        beq=pl.coalesce(
            "beq",
            pl.col("beq").shift(-1).over("gvkey")
            - pl.col("ibq").shift(-1).over("gvkey")
            + pl.col("dvq").shift(-1).over("gvkey"),
        )
    )
    fw_cols = [f"beq_fw{j}" for j in range(1, 5)]
    df = df.with_columns(
        **{
            fw: pl.col("beq").shift(j).over("gvkey")
            + pl.col("ibq").rolling_sum(window_size=j, min_samples=j).over("gvkey")
            - pl.col("dvq").rolling_sum(window_size=j, min_samples=j).over("gvkey")
            for j, fw in enumerate(fw_cols, start=1)
        }
    ).with_columns(beq=pl.coalesce("beq", *fw_cols))
    return (
        df.drop(*fw_cols, "qidx")
        .with_columns(beq_lag1=pl.col("beq").shift(1).over("gvkey"))
        .filter(pl.col("datadate").is_not_null())
        .collect()
    )


def _hxz_best_link(
    df: pl.DataFrame | pl.LazyFrame, *, subset: list[str], tail_key: str
) -> pl.DataFrame | pl.LazyFrame:
    """linkprim DESC, linktype ASC, longest-lived link DESC, linkdt ASC, tail_key ASC."""
    return df.sort(
        [*subset, "linkprim", "linktype", "_link_days", "linkdt", tail_key],
        descending=[False] * len(subset) + [True, False, True, False, False],
    ).unique(subset=subset, keep="first", maintain_order=True)


def hxz_link_compustat(comp: pl.DataFrame, raw_dir: Path) -> pl.DataFrame:
    """gvkey → permno via CCM; per-row datadate gating (NOT jun_end as FF
    uses — HXZ links at fiscal-q-end). Primary first; one (datadate, permno)."""
    lnk = (
        pl.scan_parquet(raw_dir / "crsp_ccmxpf_lnkhist.parquet")
        .filter(pl.col("linktype").str.starts_with("L") & pl.col("linkprim").is_in(["P", "C"]))
        .select("gvkey", "lpermno", "linkprim", "linktype", "linkdt", "linkenddt")
        .with_columns(
            pl.col("lpermno").cast(pl.Int64),
            pl.col("linkdt", "linkenddt").cast(pl.Date),
        )
    )
    return (
        comp.lazy()
        .join(lnk, on="gvkey", how="inner")
        .filter(
            (pl.col("linkdt") <= pl.col("datadate"))
            & (pl.col("linkenddt").is_null() | (pl.col("linkenddt") >= pl.col("datadate")))
        )
        .rename({"lpermno": "permno"})
        .with_columns(
            _link_days=(
                pl.coalesce(pl.col("linkenddt"), pl.lit(date(9999, 12, 31))) - pl.col("linkdt")
            ).dt.total_days()
        )
        .pipe(_hxz_best_link, subset=["datadate", "permno"], tail_key="gvkey")
        .drop("linktype", "_link_days")
        .collect()
    )


def country_breakpoints(
    df: pl.DataFrame,
    specs: list[tuple[str, float, str]],
    *,
    group_keys: list[str],
    pool: pl.Expr,
    min_stocks: int | None = None,
    usa_exempt: bool = False,
) -> pl.DataFrame:
    """Per-`group_keys` QUANTILE_DISC breakpoints over `pool`; one row per group.
    specs = [(col, quantile, alias)]. `min_stocks` drops groups below that count;
    `usa_exempt` keeps USA regardless. Shared by the FF/HXZ/DHS sorts — FF and DHS
    exempt their US NYSE pool via usa_exempt, HXZ instead splits US/ROW into two
    calls (US min_stocks=None, ROW gated), so the two idioms both route here."""
    breaks_sql = ", ".join(f"QUANTILE_DISC({c}, {q}) AS {a}" for c, q, a in specs)
    cols = list(dict.fromkeys(c for c, _, _ in specs))
    gk = ", ".join(group_keys)
    having = ""
    if min_stocks is not None:
        gate = f"COUNT(*) >= {min_stocks}"
        having = f" HAVING excntry = '{US_EXCNTRY}' OR {gate}" if usa_exempt else f" HAVING {gate}"
    return (
        df.filter(pool)
        .select(*group_keys, *cols)
        .sql(f"SELECT {gk}, {breaks_sql} FROM self GROUP BY {gk}{having}")
    )


def _hxz_na_be_inv(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute be_a, inv, ib on a Compustat NA funda LazyFrame.
    Full pstkrv → pstkl → pstk and txditc → txdb+itcb fallback chains."""
    ps = pl.coalesce("pstkrv", "pstkl", "pstk")
    txditc = pl.coalesce("txditc", pl.col("txdb") + pl.col("itcb"))
    at_lag = pl.col("at").shift(1).over("gvkey")
    return (
        lf.with_columns(
            be_a=pl.coalesce(
                "seq", pl.col("ceq") + pl.coalesce(ps, 0.0), pl.col("at") - pl.col("lt")
            )
            + pl.coalesce(txditc, 0.0)
            - pl.coalesce(ps, 0.0),
            year=pl.col("datadate").dt.year(),
        )
        # BEFORE the inv lag; mirrors the FF loaders.
        .pipe(_collapse_to_latest_fy)
        .with_columns(
            inv=pl.when(
                at_lag.is_not_null()
                & (at_lag > 0)
                & (
                    pl.col("datadate").dt.year()
                    - pl.col("datadate").shift(1).over("gvkey").dt.year()
                    == 1
                )
            )
            .then((pl.col("at") - at_lag) / at_lag)
            .otherwise(None),
        )
        .select("gvkey", "datadate", "year", "be_a", "inv", "ib")
    )


def _hxz_global_be_inv(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute be_a, inv, ib on a Compustat Global funda LazyFrame.
    comp.g_funda lacks pstkrv/pstkl/itcb: preferred-stock term collapses to pstk;
    txditc fallback collapses to txdb (+null preserves null semantics when txdb is null)."""
    ps = pl.col("pstk")
    txditc = pl.coalesce("txditc", pl.col("txdb") + pl.lit(None, dtype=pl.Float64))
    at_lag = pl.col("at").shift(1).over("gvkey")
    return (
        lf.with_columns(
            be_a=pl.coalesce(
                "seq", pl.col("ceq") + pl.coalesce(ps, 0.0), pl.col("at") - pl.col("lt")
            )
            + pl.coalesce(txditc, 0.0)
            - pl.coalesce(ps, 0.0),
            year=pl.col("datadate").dt.year(),
        )
        # BEFORE the inv lag; mirrors the FF loaders.
        .pipe(_collapse_to_latest_fy)
        .with_columns(
            inv=pl.when(
                at_lag.is_not_null()
                & (at_lag > 0)
                & (
                    pl.col("datadate").dt.year()
                    - pl.col("datadate").shift(1).over("gvkey").dt.year()
                    == 1
                )
            )
            .then((pl.col("at") - at_lag) / at_lag)
            .otherwise(None),
        )
        .select("gvkey", "datadate", "year", "be_a", "inv", "ib")
    )


def hxz_load_compustat_row(raw_dir: Path) -> pl.DataFrame:
    """Annual Compustat for ROW HXZ: unions comp_funda (NA) and comp_g_funda (Global).

    Dedup by (gvkey, datadate) preferring NA. JKP BE chain; I/A = Δat / at_lag (1y FY gap).
    Annual ROE: roe_a = ib / be_a_lag1 (ratio is unitless, no FX needed).

    Output: Eager [gvkey, datadate, year, be_a, inv, roe_a]. be_a in local currency
    (used only for eligibility gate be_a > 0).
    """
    base_floats = ["pstk", "seq", "ceq", "lt", "txditc", "txdb", "at", "ib"]
    na_floats = base_floats + ["pstkrv", "pstkl", "itcb"]
    na_lf = (
        pl.scan_parquet(raw_dir / "comp_funda.parquet")
        .filter(
            (pl.col("indfmt") == "INDL")
            & (pl.col("datafmt") == "STD")
            & (pl.col("popsrc") == "D")
            & (pl.col("consol") == "C")
        )
        .with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col(*na_floats).cast(pl.Float64),
        )
    )
    gl_lf = (
        pl.scan_parquet(raw_dir / "comp_g_funda.parquet")
        .filter(
            pl.col("indfmt").is_in(["INDL", "FS"])
            & (pl.col("datafmt") == "HIST_STD")
            & (pl.col("popsrc") == "I")
            & (pl.col("consol") == "C")
        )
        .with_columns(
            pl.col("datadate").cast(pl.Date),
            pl.col(*base_floats).cast(pl.Float64),
        )
    )
    gl_lf = (
        gl_lf.with_columns(_n=pl.len().over(["gvkey", "datadate"]))
        .filter((pl.col("_n") == 1) | ((pl.col("_n") == 2) & (pl.col("indfmt") == "INDL")))
        .drop("_n")
    )

    na = _hxz_na_be_inv(na_lf).with_columns(_src=pl.lit(0))
    gl = _hxz_global_be_inv(gl_lf).with_columns(_src=pl.lit(1))
    # Fully numeric tie-break (after _src=NA<GL) so polars' parallel sort can't surface
    # different tied rows: within (gvkey, datadate, _src) prefer non-null then larger
    # be_a, then larger `at` (via inv); ib/year ASC are final deterministic breakers.
    combined = (
        pl.concat([na, gl], how="vertical_relaxed")
        .with_columns(_be_null=pl.col("be_a").is_null())
        .sort(
            ["gvkey", "datadate", "_src", "_be_null", "be_a", "inv", "ib", "year"],
            descending=[False, False, False, False, True, True, True, False],
            nulls_last=True,
        )
        .unique(subset=["gvkey", "datadate"], keep="first", maintain_order=True)
        # A gvkey in both NA and Global yields two (gvkey, year) rows; collapse before
        # the roe_a lag (prefer NA, then latest datadate). Mirrors ff_load_world_compustat.
        .sort(["gvkey", "year", "_src", "datadate"], descending=[False, False, False, True])
        .unique(subset=["gvkey", "year"], keep="first", maintain_order=True)
        .drop("_src", "_be_null")
        .sort(["gvkey", "datadate"])
    )
    be_a_lag1 = pl.col("be_a").shift(1).over("gvkey")
    return (
        combined.with_columns(
            roe_a=pl.when(pl.col("ib").is_not_null() & (be_a_lag1 > 0))
            .then(pl.col("ib") / be_a_lag1)
            .otherwise(None),
        )
        .drop("ib")
        .collect()
    )


def hxz_load_panel_row(interim_dir: Path, freq: Literal["monthly", "daily"]) -> pl.DataFrame:
    """ROW panel from ff_load_world_panel; attaches (naics, sic) from world_msf to derive is_fin.

    is_fin: NAICS sector 52 or 53 when present; SIC 6000-6999 fallback when naics is null.

    Output: Eager [excntry, id, gvkey, date, ret, me, size_grp, naics, sic, is_fin].
    """
    panel = ff_load_world_panel(interim_dir, freq).collect()
    # Dedup (id, eom): world_msf can carry duplicate share-class records per
    # security-month; without unique() the join multiplies rows non-deterministically
    # under the parallel scan. Tie-break: non-null naics, then larger (more specific) sic.
    ind = (
        pl.scan_parquet(interim_dir / "world_msf.parquet")
        .select("id", "eom", "naics", "sic")
        .with_columns(_naics_null=pl.col("naics").is_null())
        .sort(
            ["id", "eom", "_naics_null", "naics", "sic"],
            descending=[False, False, False, True, True],
            nulls_last=True,
        )
        .unique(subset=["id", "eom"], keep="first", maintain_order=True)
        .drop("_naics_null")
        .collect()
    )
    eom_expr = pl.col("date").dt.month_end() if freq == "daily" else pl.col("date")
    is_fin = (
        pl.col("naics").cast(pl.Utf8).str.slice(0, 2).is_in(["52", "53"])
        | (pl.col("naics").is_null() & pl.col("sic").is_between(6000, 6999))
    ).fill_null(False)
    return (
        panel.with_columns(_eom=eom_expr)
        .join(ind, left_on=["id", "_eom"], right_on=["id", "eom"], how="left")
        .with_columns(is_fin=is_fin)
        .drop("_eom")
    )


def hxz_compute_factors(
    panel: pl.DataFrame, min_stocks_pf: int = HXZ_MIN_STOCKS_PF
) -> pl.DataFrame:
    """18-bucket VW returns per (excntry, date) → HXZ formulas 1-3
    (me_hxz, ia_hxz, roe_hxz). US bypasses the ROW min_stocks_pf gate."""
    buckets = [f"S{i}I{j}R{k}" for i in (1, 2) for j in (1, 2, 3) for k in (1, 2, 3)]
    vw = vw_bucket_returns(
        panel.filter(
            (pl.col("me_lag1") > 0)
            & pl.col("me_lag1").is_finite()
            & pl.col("ret").is_not_null()
            & pl.col("ret").is_finite()
            & pl.col("sizeport").is_not_null()
            & pl.col("invport").is_not_null()
            & pl.col("roeport").is_not_null()
        ).with_columns(
            bucket=pl.format(
                "S{}I{}R{}",
                pl.col("sizeport").cast(pl.Utf8),
                pl.col("invport").cast(pl.Utf8),
                pl.col("roeport").cast(pl.Utf8),
            )
        ),
        buckets=buckets,
        weight_col="me_lag1",
        min_stocks_pf=min_stocks_pf,
    )

    # mean_horizontal skips nulls so a missing bucket doesn't void the leg
    # — needed for thin ROW countries where 18 buckets often have empties.
    def avg(cols: list[str]) -> pl.Expr:
        return pl.mean_horizontal(*[pl.col(c) for c in cols])

    small = [c for c in buckets if c.startswith("S1")]
    big = [c for c in buckets if c.startswith("S2")]
    lo_i = [c for c in buckets if "I1" in c]
    hi_i = [c for c in buckets if "I3" in c]
    hi_r = [c for c in buckets if c.endswith("R3")]
    lo_r = [c for c in buckets if c.endswith("R1")]
    return vw.select(
        "excntry",
        "date",
        me_hxz=avg(small) - avg(big),
        ia_hxz=avg(lo_i) - avg(hi_i),
        roe_hxz=avg(hi_r) - avg(lo_r),
    ).sort(["excntry", "date"])


def hxz_build_characteristics(panel_m: pl.DataFrame) -> pl.DataFrame:
    """(excntry, id, eom) stock-level chars used to form HXZ sorts. The
    returned frame carries `excntry`; the writer in `gen_hxz_data` drops it
    (`id` is globally unique)."""
    return panel_m.select(
        eom=pl.col("date").dt.month_end(),
        excntry=pl.col("excntry"),
        id=pl.col("id"),
        me_hxz=pl.col("me"),
        ia_hxz=pl.col("inv"),
        roe_hxz=pl.col("roe"),
    ).sort(["eom", "excntry", "id"])


def _hxz_add_me_lag1_ym(
    df: pl.DataFrame, key: str, freq: Literal["monthly", "daily"]
) -> pl.DataFrame:
    """Attach me_lag1 to a returns frame keyed by `key`.

    Monthly: YM-int (year*100 + month) join — gap-robust + correct when `date`
    is not calendar month-end (CIZ `mthcaldt` is last trading day).
    Daily: shift(1).over(key).
    """
    if freq == "monthly":
        ym = (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("ym")
        prev_ym_date = pl.col("date").dt.offset_by("-1mo")
        # Sort by (key, date) so `last()` per (key, ym) group is deterministic
        # (max-date within month) regardless of polars' hash-grouped ordering.
        me_by_ym = (
            df.with_columns(ym)
            .sort([key, "date"])
            .group_by([key, "ym"], maintain_order=True)
            .agg(me_lag1=pl.col("me").last())
        )
        return (
            df.with_columns(prev_ym=prev_ym_date.dt.year() * 100 + prev_ym_date.dt.month())
            .join(me_by_ym, left_on=[key, "prev_ym"], right_on=[key, "ym"], how="left")
            .drop("prev_ym")
        )
    return df.with_columns(me_lag1=pl.col("me").shift(1).over(key))


def hxz_load_returns(
    raw_dir: Path,
    interim_dir: Path,
    freq: Literal["monthly", "daily"],
    date_start: date | None = None,
) -> pl.DataFrame:
    """Unified US + ROW returns/me panel with me_lag1 attached.

    US keyed by excntry=USA, id=permno; ROW by excntry/id/gvkey.
    Country-specific cols null on the opposite branch.

    Output: Eager [excntry, id, date, ret, me, me_lag1, is_fin, ...].
    """
    us = (
        hxz_load_crsp(raw_dir, freq, date_start=date_start)
        .sort(["permno", "date"])
        .with_columns(excntry=pl.lit(US_EXCNTRY), id=pl.col("permno"))
    )
    us = _hxz_add_me_lag1_ym(us, key="id", freq=freq)

    row = hxz_load_panel_row(interim_dir, freq).sort(["id", "date"])
    if date_start is not None:
        row = row.filter(pl.col("date") >= date_start)
    row = _hxz_add_me_lag1_ym(row, key="id", freq=freq)

    return pl.concat([us, row], how="diagonal_relaxed")


def _hxz_us_roe_monthly(
    fundq: pl.DataFrame, raw_dir: Path, formation_dates: pl.DataFrame
) -> pl.DataFrame:
    """Compute Roe = IBQ / calendar-q-lagged BEQ and assign to each US
    formation date per HXZ (2015) rules:
      - formation_date ≥ rdq.month_end + 1mo (earnings publicly announced)
      - 6mo staleness cap: formation_date ≤ datadate + 6mo (month-end)
    We follow the HXZ (2015) methodology over the full sample and do not
    use the Hou et al. (2019) 1967-1972 extension (4mo-lag proxy in place
    of RDQ): quarterly Compustat coverage in those years is thin and the
    resulting sorts noisy. The US sample therefore starts January 1972,
    as in the original paper.
    """
    fq = fundq.with_columns(safe_div("ibq", "beq_lag1", "roe", mode=3)).select(
        "gvkey", "datadate", "rdq", "roe"
    )
    fq_lnk = (
        hxz_link_compustat(fq, raw_dir)
        .filter(pl.col("rdq").is_null() | (pl.col("rdq") > pl.col("datadate")))
        .with_columns(
            formation_min=pl.when(pl.col("rdq").is_not_null())
            .then(pl.col("rdq").dt.month_end().dt.offset_by("1mo"))
            .otherwise(None),
            formation_max=pl.col("datadate").dt.offset_by("6mo").dt.month_end(),
        )
    )
    return (
        fq_lnk.join(formation_dates.select("date"), how="cross")
        .filter(
            pl.col("roe").is_not_null()
            & pl.col("formation_min").is_not_null()
            & (pl.col("date") >= pl.col("formation_min"))
            & (pl.col("date") <= pl.col("formation_max"))
        )
        # Sort tie-break: roe DESC then rdq nulls-last so identical
        # (permno, date, datadate) tuples — rare but possible if linker keeps
        # multiple gvkeys — resolve the same way under polars threading.
        .sort(
            ["permno", "date", "datadate", "roe", "rdq"],
            descending=[False, False, False, True, False],
            nulls_last=True,
        )
        .unique(subset=["permno", "date"], keep="last", maintain_order=True)
        .select("permno", "date", "roe")
    )


def _hxz_us_chars(
    us_panel: pl.DataFrame,
    funda_us: pl.DataFrame,
    fundq_us: pl.DataFrame,
    raw_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """US size+inv (June, CCM-linked) and monthly ROE (RDQ-gated) frames."""
    # US: June size + inv (annual, link via CCM at fiscal-y-end)
    us_june = us_panel.filter(pl.col("date").dt.month() == 6)
    ia_us_lnk = (
        hxz_link_compustat(
            funda_us.with_columns(year=pl.col("datadate").dt.year()).select(
                "gvkey", "datadate", "year", "inv", "be_a"
            ),
            raw_dir,
        )
        .sort(["permno", "year", "datadate"])
        # `unique(keep="last")` after sort: deterministic latest-datadate-per-
        # (permno, year). `group_by().agg(last())` was non-deterministic because
        # polars group_by doesn't preserve outer sort order within groups.
        .unique(subset=["permno", "year"], keep="last", maintain_order=True)
        .with_columns(comp_year_plus_1=pl.col("year") + 1)
        .select("permno", "comp_year_plus_1", "inv", "be_a")
    )
    us_size_ia = (
        us_june.with_columns(crsp_year=pl.col("date").dt.year())
        .join(
            ia_us_lnk,
            left_on=["permno", "crsp_year"],
            right_on=["permno", "comp_year_plus_1"],
            how="left",
        )
        .with_columns(elig=(~pl.col("is_fin")) & (pl.col("be_a") > 0) & (pl.col("me") > 0))
        .select(
            "excntry",
            "id",
            "permno",
            "date",
            "me",
            "inv",
            "be_a",
            "is_fin",
            "exchcd",
            "elig",
        )  # fmt: skip
    )

    # US: monthly ROE (RDQ-gated; pre-1972 extension deliberately not used)
    us_roe_raw = _hxz_us_roe_monthly(fundq_us, raw_dir, us_panel.select("date").unique())
    us_roe = us_panel.select("excntry", "id", "permno", "date", "is_fin", "exchcd").join(
        us_roe_raw, on=["permno", "date"], how="left"
    )

    return us_size_ia, us_roe


def _hxz_row_chars(
    row_panel: pl.DataFrame,
    funda_row: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """ROW size+inv (June, gvkey-linked) and monthly ROE (annual broadcast) frames."""
    # ROW: June size + inv (annual, link via gvkey directly)
    row_june = row_panel.filter(pl.col("date").dt.month() == 6)
    ia_row_lnk = (
        funda_row.sort(["gvkey", "year", "datadate"])
        .unique(subset=["gvkey", "year"], keep="last", maintain_order=True)
        .with_columns(comp_year_plus_1=pl.col("year") + 1)
        .select("gvkey", "comp_year_plus_1", "inv", "be_a")
    )
    row_size_ia = (
        row_june.with_columns(crsp_year=pl.col("date").dt.year())
        .join(
            ia_row_lnk,
            left_on=["gvkey", "crsp_year"],
            right_on=["gvkey", "comp_year_plus_1"],
            how="left",
        )
        .with_columns(elig=(~pl.col("is_fin")) & (pl.col("be_a") > 0) & (pl.col("me") > 0))
        .select(
            "excntry",
            "id",
            "gvkey",
            "date",
            "me",
            "inv",
            "be_a",
            "is_fin",
            "size_grp",
            "elig",
        )  # fmt: skip
    )

    # ROW: monthly ROE broadcast (datadate+6mo..datadate+18mo)
    fmtimes = row_panel.select("date").unique()
    row_roe_raw = (
        funda_row.select("gvkey", "datadate", "roe_a")
        .filter(pl.col("roe_a").is_not_null())
        .with_columns(
            formation_min=pl.col("datadate").dt.offset_by("6mo").dt.month_end(),
            formation_max=pl.col("datadate").dt.offset_by("18mo").dt.month_end(),
        )
        .join(fmtimes, how="cross")
        .filter(
            (pl.col("date") >= pl.col("formation_min"))
            & (pl.col("date") <= pl.col("formation_max"))
        )
        # datadate ASC + keep="last" takes the most recent fiscal date; the trailing
        # roe_a DESC only breaks rare (gvkey, date, datadate) ties deterministically
        # (which row wins is immaterial).
        .sort(
            ["gvkey", "date", "datadate", "roe_a"],
            descending=[False, False, False, True],
            nulls_last=True,
        )
        .unique(subset=["gvkey", "date"], keep="last", maintain_order=True)
        .select("gvkey", "date", roe=pl.col("roe_a"))
    )
    row_roe = row_panel.select("excntry", "id", "gvkey", "date", "is_fin", "size_grp").join(
        row_roe_raw, on=["gvkey", "date"], how="left"
    )

    return row_size_ia, row_roe


def hxz_compute_chars(
    panel_m: pl.DataFrame,
    funda_us: pl.DataFrame,
    fundq_us: pl.DataFrame,
    funda_row: pl.DataFrame,
    raw_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute formation-date characteristics for all countries (stage 3).

    Returns:
      - size_ia_form: June annual frame [me, inv, be_a, elig, permno/exchcd (US), gvkey/size_grp (ROW)]
      - roe_m: monthly frame [roe] — US: quarterly RDQ-gated; ROW: annual roe_a broadcast 6mo..18mo
    """
    us_panel = panel_m.filter(pl.col("excntry") == US_EXCNTRY)
    row_panel = panel_m.filter(pl.col("excntry") != US_EXCNTRY)

    us_size_ia, us_roe = _hxz_us_chars(us_panel, funda_us, fundq_us, raw_dir)
    row_size_ia, row_roe = _hxz_row_chars(row_panel, funda_row)

    size_ia_form = pl.concat([us_size_ia, row_size_ia], how="diagonal_relaxed")
    roe_m = pl.concat([us_roe, row_roe], how="diagonal_relaxed")
    return size_ia_form, roe_m


def _hxz_tercile_port(value_col: str, b30: str, b70: str, gate: pl.Expr) -> pl.Expr:
    """1/2/3 tercile bucket of `value_col` vs break columns, null when gated off."""
    return (
        pl.when(gate)
        .then(
            pl.when(pl.col(value_col) <= pl.col(b30))
            .then(1)
            .when(pl.col(value_col) <= pl.col(b70))
            .then(2)
            .otherwise(3)
        )
        .otherwise(None)
        .cast(pl.Int64)
    )


def _hxz_classify_size_ia(
    size_ia_form: pl.DataFrame, min_stocks_bp: int = HXZ_MIN_STOCKS_BP
) -> pl.DataFrame:
    """Apply country-aware breakpoints + bucket cuts to June size+inv frame.
    US: NYSE-only pool, per-date breaks. ROW: size_grp pool + min_stocks_bp
    gate, per-(excntry, date) breaks. Returns June-level classified frame
    with `sizeport`, `invport` (and `inv` preserved for downstream)."""
    specs = [("me", 0.50, "sizemedn"), ("inv", 0.30, "inv30"), ("inv", 0.70, "inv70")]

    # US: NYSE-only pool, per-date breaks (sizemedn always present)
    us = size_ia_form.filter(pl.col("excntry") == US_EXCNTRY)
    us_breaks = country_breakpoints(
        us, specs, group_keys=["date"], pool=(pl.col("exchcd") == 1) & pl.col("elig")
    )
    us_cls = (
        us.join(us_breaks, on=["date"], how="left")
        .with_columns(
            sizeport=pl.when(pl.col("elig"))
            .then(pl.when(pl.col("me") <= pl.col("sizemedn")).then(1).otherwise(2))
            .otherwise(None)
            .cast(pl.Int64),
            # Tercile inv bucket; gated on elig + non-null inv and inv30 break.
            invport=_hxz_tercile_port(
                "inv",
                "inv30",
                "inv70",
                pl.col("elig") & pl.col("inv").is_not_null() & pl.col("inv30").is_not_null(),
            ),
        )
        .select("excntry", "id", "permno", "date", "sizeport", "invport", "inv")
    )

    # ROW: size_grp pool + HXZ_MIN_STOCKS_BP gate, per-(excntry, date)
    # ROW joins breaks per (excntry, date) with a HAVING gate, so sizemedn can be
    # null after the left join; the size gate therefore also requires it non-null.
    row = size_ia_form.filter(pl.col("excntry") != US_EXCNTRY)
    row_breaks = country_breakpoints(
        row,
        specs,
        group_keys=["excntry", "date"],
        pool=pl.col("elig") & pl.col("size_grp").is_in(["small", "large", "mega"]),
        min_stocks=min_stocks_bp,
    )
    row_cls = (
        row.join(row_breaks, on=["excntry", "date"], how="left")
        .with_columns(
            sizeport=pl.when(pl.col("elig") & pl.col("sizemedn").is_not_null())
            .then(pl.when(pl.col("me") <= pl.col("sizemedn")).then(1).otherwise(2))
            .otherwise(None)
            .cast(pl.Int64),
            # Tercile inv bucket; gated on elig + non-null inv and inv30 break.
            invport=_hxz_tercile_port(
                "inv",
                "inv30",
                "inv70",
                pl.col("elig") & pl.col("inv").is_not_null() & pl.col("inv30").is_not_null(),
            ),
        )
        .select("excntry", "id", "gvkey", "date", "sizeport", "invport", "inv")
    )
    return pl.concat([us_cls, row_cls], how="diagonal_relaxed")


def _hxz_classify_roe(roe_m: pl.DataFrame, min_stocks_bp: int = HXZ_MIN_STOCKS_BP) -> pl.DataFrame:
    """Apply country-aware breakpoints + bucket cuts to monthly ROE frame.
    US: NYSE-only pool, per-date breaks. ROW: size_grp pool, per-(excntry,
    date) breaks with min_stocks_bp gate."""
    specs = [("roe", 0.30, "roe30"), ("roe", 0.70, "roe70")]
    elig = (~pl.col("is_fin")) & pl.col("roe").is_not_null()

    # US: NYSE-only pool, per-date breaks
    us = roe_m.filter(pl.col("excntry") == US_EXCNTRY).with_columns(elig=elig)
    us_breaks = country_breakpoints(
        us, specs, group_keys=["date"], pool=(pl.col("exchcd") == 1) & pl.col("elig")
    )
    us_cls = (
        us.join(us_breaks, on=["date"], how="left")
        .with_columns(
            roeport=_hxz_tercile_port(
                "roe", "roe30", "roe70", pl.col("elig") & pl.col("roe30").is_not_null()
            )
        )
        .select("excntry", "id", "date", "roeport", "roe")
    )

    # ROW: size_grp pool, per-(excntry, date) breaks with HXZ_MIN_STOCKS_BP gate
    row = roe_m.filter(pl.col("excntry") != US_EXCNTRY).with_columns(elig=elig)
    row_breaks = country_breakpoints(
        row,
        specs,
        group_keys=["excntry", "date"],
        pool=pl.col("elig") & pl.col("size_grp").is_in(["small", "large", "mega"]),
        min_stocks=min_stocks_bp,
    )
    row_cls = (
        row.join(row_breaks, on=["excntry", "date"], how="left")
        .with_columns(
            roeport=_hxz_tercile_port(
                "roe", "roe30", "roe70", pl.col("elig") & pl.col("roe30").is_not_null()
            )
        )
        .select("excntry", "id", "date", "roeport", "roe")
    )
    return pl.concat([us_cls, row_cls], how="vertical_relaxed")


def hxz_classify_portfolios(
    panel_m: pl.DataFrame,
    size_ia_form: pl.DataFrame,
    roe_m: pl.DataFrame,
    min_stocks_bp: int = HXZ_MIN_STOCKS_BP,
) -> pl.DataFrame:
    """Classify stocks into HXZ 2×3×3 portfolios (stage 4).

    Country-aware breakpoints: US NYSE-only per-date; ROW size_grp pool
    per-(excntry, date) with min_stocks_bp gate. June (size, I/A)
    classification broadcast to Jul(y)..Jun(y+1) via _ff_port_year; monthly
    ROE classification attached on (excntry, id, date).

    Output: Eager panel [excntry, id, date, ret, me, me_lag1,
    sizeport, invport, roeport, inv, roe, ...].
    """
    size_ia_classified = _hxz_classify_size_ia(size_ia_form, min_stocks_bp)
    roe_classified = _hxz_classify_roe(roe_m, min_stocks_bp)

    # US broadcast: join June classification on permno + port_year
    us_panel = panel_m.filter(pl.col("excntry") == US_EXCNTRY)
    us_size_ia_keyed = (
        size_ia_classified.filter(pl.col("excntry") == US_EXCNTRY)
        .with_columns(june_year=pl.col("date").dt.year())
        .select("excntry", "permno", "june_year", "sizeport", "invport", "inv")
    )
    us_attached = (
        us_panel.with_columns(_ff_port_year())
        .join(
            us_size_ia_keyed,
            left_on=["excntry", "permno", "port_year"],
            right_on=["excntry", "permno", "june_year"],
            how="left",
        )
        .drop("port_year")
    )

    # ROW broadcast: join June classification on (id, port_year). Key on id, NOT gvkey
    # — a multi-share-class firm has >1 id, and a gvkey join would fan each panel row
    # per share class. Matches the ROE broadcast, the US permno branch, and the FF analog.
    row_panel = panel_m.filter(pl.col("excntry") != US_EXCNTRY)
    row_size_ia_keyed = (
        size_ia_classified.filter(pl.col("excntry") != US_EXCNTRY)
        .with_columns(june_year=pl.col("date").dt.year())
        .select("excntry", "id", "june_year", "sizeport", "invport", "inv")
    )
    row_attached = (
        row_panel.with_columns(_ff_port_year())
        .join(
            row_size_ia_keyed,
            left_on=["excntry", "id", "port_year"],
            right_on=["excntry", "id", "june_year"],
            how="left",
        )
        .drop("port_year")
    )

    panel_with_size_ia = pl.concat([us_attached, row_attached], how="diagonal_relaxed")

    return panel_with_size_ia.join(
        roe_classified.select("excntry", "id", "date", "roeport", "roe"),
        on=["excntry", "id", "date"],
        how="left",
    )


@measure_time
def gen_hxz_data(
    paths: DataPaths,
    monthly_factors_path: Path,
    daily_factors_path: Path,
    chars_path: Path,
    freqs: tuple[str, ...] = ("monthly", "daily"),
    min_stocks_bp: int = HXZ_MIN_STOCKS_BP,
    min_stocks_pf: int = HXZ_MIN_STOCKS_PF,
) -> None:
    """
    Description:
        Reference: Hou, Xue, and Zhang (2015) 'Digesting anomalies: An
        investment approach', Review of Financial Studies 28(3), pp. 650-705.

        HXZ q-factor replication: triple 2×3×3 sort on size × I/A × Roe,
        building R_ME, R_IA, R_ROE at monthly + daily on CRSP CIZ
        msf_v2/dsf_v2 + Compustat funda/fundq. Permno-level (no permco
        aggregation). CIZ ret already daily-compounds delret per CRSP CIZ docs.
        Reads raw CRSP/Compustat from `paths.raw_tables_dir` and world inputs
        from `paths.interim_dir`; output paths are passed in explicitly.
    Steps:
        1) Load + filter return/me data (US + ROW unified)
        2) Load + filter fundamentals (US quarterly chain + ROW annual)
        3) Compute characteristics at portfolio formation (size, inv, roe)
        4) Classify stocks into portfolios (2×3×3, country-aware)
        5) Portfolio sorts → factor returns
        6) Output
    Output:
        - <monthly_factors_path>  [excntry, eom, me_hxz, ia_hxz, roe_hxz]
        - <daily_factors_path>    [excntry, date, me_hxz, ia_hxz, roe_hxz]
        - <chars_path>            [eom, id, me_hxz, ia_hxz, roe_hxz]
          (excntry dropped: `id` is globally unique)
    """
    raw_dir = paths.raw_tables_dir
    interim_dir = paths.interim_dir
    output_start = date(1967, 1, 1)
    # US starts at the HXZ (2015) sample start — limited by RDQ availability;
    # the RDQ gate in _hxz_us_roe_monthly leaves only stub months before this.
    us_output_start = date(1972, 1, 1)
    start_filter = pl.col("date") >= pl.when(pl.col("excntry") == US_EXCNTRY).then(
        pl.lit(us_output_start)
    ).otherwise(pl.lit(output_start))

    # 1. Load + filter returns/me (monthly)
    panel_m_raw = hxz_load_returns(raw_dir, interim_dir, freq="monthly")

    # 2. Load + filter fundamentals
    funda_us = hxz_load_funda(raw_dir)
    fundq_us = hxz_impute_be_clean_surplus(
        hxz_merge_be_chain(
            hxz_supplement_q4_shares(hxz_load_fundq(raw_dir), funda_us, raw_dir),
            funda_us,
        )
    )
    funda_row = hxz_load_compustat_row(raw_dir)

    # 3. Compute characteristics at formation date
    size_ia_form, roe_m = hxz_compute_chars(panel_m_raw, funda_us, fundq_us, funda_row, raw_dir)

    # 4. Classify into portfolios + broadcast to monthly grid
    panel_m = hxz_classify_portfolios(panel_m_raw, size_ia_form, roe_m, min_stocks_bp).filter(
        start_filter
    )

    port_assigns = panel_m.select(
        "excntry",
        "id",
        "sizeport",
        "invport",
        "roeport",
        eom=pl.col("date").dt.month_end(),
    )

    # 5+6. Monthly sorts + output
    if "monthly" in freqs:
        (
            hxz_compute_factors(panel_m, min_stocks_pf)
            .with_columns(eom=pl.col("date").dt.month_end())
            .drop("date")
            .write_parquet(monthly_factors_path)
        )
        hxz_build_characteristics(panel_m).drop("excntry").write_parquet(chars_path)

    # free the monthly frames before the 1925+ daily scan (peak-memory control)
    del panel_m_raw, panel_m, funda_us, fundq_us, funda_row, size_ia_form, roe_m

    # 5+6. Daily sorts + output
    if "daily" in freqs:
        # Floor scan at 6mo before output_start so me_lag1 has prior-day data
        # on the first output day without holding the full 1925+ daily file.
        daily_start = date(output_start.year - 1, 7, 1)
        panel_d_raw = hxz_load_returns(raw_dir, interim_dir, freq="daily", date_start=daily_start)
        panel_d = (
            panel_d_raw.filter(start_filter)
            .with_columns(eom=pl.col("date").dt.month_end())
            .join(port_assigns, on=["excntry", "id", "eom"], how="left")
            .drop("eom")
        )
        hxz_compute_factors(panel_d, min_stocks_pf).write_parquet(daily_factors_path)


def _ap_outer_join_all(frames: list[pl.LazyFrame], key: list[str]) -> pl.LazyFrame:
    """Left-to-right full outer join of `frames` on `key` (coalesced keys)."""
    if not frames:
        raise ValueError("ap_factor_model_data: empty input list")
    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.join(frame, on=key, how="full", coalesce=True)
    return joined


def _ap_check_no_collisions(frames: list[pl.LazyFrame], key: list[str]) -> None:
    """Raise if a non-key column appears in more than one input (a full join
    would silently suffix it with `_right` instead of failing)."""
    seen: dict[str, int] = {}
    for i, frame in enumerate(frames):
        for c in frame.collect_schema().names():
            if c in key:
                continue
            if c in seen:
                raise ValueError(
                    f"ap_factor_model_data: column {c!r} appears in inputs {seen[c]} and {i}"
                )
            seen[c] = i


def _ap_join_factor_inputs(
    mkt_path: Path, factor_inputs: list[Path], date_key: str
) -> pl.LazyFrame:
    """Market mktrf (mkt_vw_exc) outer-joined with each factor parquet on
    (excntry, `date_key`)."""
    mkt = pl.scan_parquet(mkt_path).select(
        ["excntry", date_key, pl.col("mkt_vw_exc").alias("mktrf")]
    )
    frames = [mkt, *(pl.scan_parquet(p) for p in factor_inputs)]
    key = ["excntry", date_key]
    _ap_check_no_collisions(frames, key)
    return _ap_outer_join_all(frames, key)


@measure_time
def ap_factor_model_data(
    monthly_factor_inputs: list[Path],
    daily_factor_inputs: list[Path],
    chars_inputs: list[Path],
    mkt_monthly_path: Path,
    mkt_daily_path: Path,
    out_monthly: Path,
    out_daily: Path,
    out_chars: Path,
) -> None:
    """
    Description:
        Unify factor-model outputs into 3 parquets:
          - monthly factors keyed by (excntry, eom)
          - daily   factors keyed by (excntry, date)
          - per-stock chars keyed by (id, eom); `id` is globally unique so
            `excntry` is dropped by the per-model chars writers.
        mktrf is sourced from market_returns_*.parquet (mkt_vw_exc).
        Unlike the gen_*_data stages this is a pure combinator: it takes explicit
        Path lists rather than `paths: DataPaths` on purpose, so the set of models
        to merge stays defined by the caller (the generators mapping in main.py),
        not hard-coded here. Extensible: append more factor-model parquets to the
        input lists.

    Steps:
        1) Monthly: scan mkt_monthly_path → (excntry, eom, mktrf); outer-join
           each monthly factor parquet on (excntry, eom); sink.
        2) Daily:   same with date key, mkt_daily_path.
        3) Chars:   outer-join each chars parquet on (id, eom); sink.

    Output:
        - <out_monthly>: [excntry, eom, mktrf] + union of columns from each
                          factor parquet in monthly_factor_inputs.
        - <out_daily>:   [excntry, date, mktrf] + union of daily_factor_inputs.
        - <out_chars>:   [id, eom]              + union of chars_inputs.
    """

    _ap_join_factor_inputs(mkt_monthly_path, monthly_factor_inputs, "eom").sink_parquet(out_monthly)
    _ap_join_factor_inputs(mkt_daily_path, daily_factor_inputs, "date").sink_parquet(out_daily)

    chars_frames = [pl.scan_parquet(p) for p in chars_inputs]
    _ap_check_no_collisions(chars_frames, ["id", "eom"])
    _ap_outer_join_all(chars_frames, ["id", "eom"]).sink_parquet(out_chars)


def add_ecdf(
    df: pl.DataFrame | pl.LazyFrame,
    group_cols: list[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Attach an empirical-CDF ``cdf`` column per group.

    Build the ECDF of ``var`` over the rows where ``bp_stock`` is true (the
    breakpoint sample), then asof-join the CDF values back onto every row of
    ``df`` within each group defined by ``group_cols``. Non-breakpoint rows
    receive the CDF of the nearest breakpoint ``var`` ≤ their own; rows
    below any breakpoint value receive ``0.0``.

    Args:
        df: Frame containing ``var`` and boolean ``bp_stock`` columns.
        group_cols: Columns defining the ECDF groups. Defaults to ``["eom"]``.

    Returns:
        Same container type as ``df`` (eager → eager, lazy → lazy) with a
        ``cdf`` column appended; original columns are preserved and rows
        are sorted by ``group_cols + ["var"]``.
    """
    group_cols = ["eom"] if group_cols is None else list(group_cols)
    sort_cols = group_cols + ["var"]

    # ECDF on the breakpoint sample: cumulative share of distinct var values
    # within each group, sorted by var so cum_sum runs in ascending order.
    bp_ecdf = (
        df.filter(pl.col("bp_stock"))
        .group_by(sort_cols)
        .agg(pl.len().alias("n_ref"))
        .sort(sort_cols)
        .select(
            *group_cols,
            "var",
            (pl.col("n_ref").cum_sum() / pl.col("n_ref").sum()).over(group_cols).alias("cdf"),
        )
    )

    # asof-join the ECDF onto every row; rows below any bp get null -> 0.0 (downstream
    # convention). Skip the sortedness check: with `by`, Polars can't verify it and
    # would warn at collect, outside any catch_warnings context.
    return (
        df.sort(sort_cols)
        .join_asof(bp_ecdf, on="var", by=group_cols, strategy="backward", check_sortedness=False)
        .with_columns(pl.col("cdf").fill_null(0.0))
    )


def _build_industry_daily_returns(
    data: pl.DataFrame,
    daily: pl.DataFrame,
    industry_col: str,
    bp_min_n: int,
    excntry: str,
    industry_transform: pl.Expr | None = None,
) -> pl.DataFrame:
    """Build daily industry portfolio returns from monthly formation-month weights.

    Description:
        Form (industry, eom) weights on the monthly frame, then apply them to
        the next month's daily returns. Weights are EW, VW (by ``me``), and
        VW-cap (by ``me_cap``). Groups with fewer than ``bp_min_n`` stocks
        are dropped.
    Output:
        DataFrame with columns [industry_col, date, n, ret_ew, ret_vw,
        ret_vw_cap, excntry].
    """
    weights_data = (
        data.lazy()
        .filter(pl.col(industry_col).is_not_null())
        .select("id", "eom", industry_col, "me", "me_cap")
    )
    if industry_transform is not None:
        weights_data = weights_data.with_columns(industry_transform)

    grp = [industry_col, "eom"]
    weights = (
        weights_data.filter(pl.len().over(grp) >= bp_min_n)
        .with_columns(
            (1 / pl.len().over(grp)).alias("w_ew"),
            (pl.col("me") / pl.col("me").sum().over(grp)).alias("w_vw"),
            (pl.col("me_cap") / pl.col("me_cap").sum().over(grp)).alias("w_vw_cap"),
        )
        .join(daily.lazy(), left_on=["id", "eom"], right_on=["id", "eom_lag1"], how="left")
        .filter(pl.col("ret_exc").is_not_null())
        .group_by(industry_col, "date")
        .agg(
            pl.len().alias("n"),
            (pl.col("w_ew") * pl.col("ret_exc")).sum().alias("ret_ew"),
            (pl.col("w_vw") * pl.col("ret_exc")).sum().alias("ret_vw"),
            (pl.col("w_vw_cap") * pl.col("ret_exc")).sum().alias("ret_vw_cap"),
        )
        .with_columns(pl.lit(excntry).str.to_uppercase().alias("excntry"))
    )
    return weights.collect()


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
    paths: DataPaths,
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
    file_path = paths.processed_dir / "characteristics" / f"{excntry}.parquet"

    # Select the required columns
    columns = (
        [
            "id",
            "eom",
            "source_crsp",
            "comp_exchg",
            "crsp_nyse",
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

    cast_exclude = {
        "id",
        "eom",
        "source_crsp",
        "size_grp",
        "excntry",
        "crsp_nyse",
    }
    cast_cols = [c for c in columns if c not in cast_exclude]

    if bps == "nyse":
        bp_stock_expr = (
            ((pl.col("crsp_nyse") == 1) & pl.col("comp_exchg").is_null())
            | ((pl.col("comp_exchg") == 11) & (pl.col("crsp_nyse") != 1))
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
    daily_file_path = (
        paths.processed_dir / "return_data" / "daily_rets_by_country" / f"{excntry}.parquet"
    )
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

    # Per-char lazy pipelines accumulate here and are batch-collected via
    # collect_all. signals=True is the exception: it collects eagerly per-char
    # because pf_signals needs the intermediate `sub` frame alive after collect.
    pf_returns_lazys: list[pl.LazyFrame] = []
    pf_daily_lazys: list[pl.LazyFrame] = []
    char_pfs: list[dict] = []  # populated only when signals=True

    # Shared node so per-char branches share one scan (polars CSE in collect_all).
    data_lazy = data.lazy()

    for x in chars:
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

        # Lazy path drops empty results after batch collect; signals=True needs
        # this eager emptiness gate because pf_signals reads `sub` intermediately.
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
                w_expr = (1 / pl.col("eom").len()).over(["pf", "eom"])
            elif signals_w == "vw":
                w_expr = (pl.col("me") / pl.col("me").sum()).over(["pf", "eom"])
            elif signals_w == "vw_cap":
                w_expr = (pl.col("me_cap") / pl.col("me_cap").sum()).over(["pf", "eom"])

            sub = sub.with_columns(w_expr.alias("w"))

            # Aggregate to one row per (pf, eom): weighted sum of the current
            # characteristic ``x`` (treating null var values as 0).
            pf_signals = (
                sub.group_by(["pf", "eom"])
                .agg(
                    (pl.col("w") * pl.col("var").fill_null(0)).sum().alias("signal"),
                )
                .with_columns(
                    pl.lit(x).alias("characteristic"),
                    pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom"),
                )
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
            if signals and char_pfs:
                output["signals"] = pl.concat([op["signals"] for op in char_pfs]).with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )

    if cmp_key:
        # Unpivot to long form so ranks/weights/aggregates run over all chars at once.
        grp = ["characteristic", "size_grp", "eom"]
        n_grp = pl.len().over(grp)
        p_rank = pl.col("var").rank("average").over(grp) / (n_grp + 1)
        p_rank_dev = p_rank - p_rank.mean().over(grp)
        weight = p_rank_dev / (p_rank_dev.abs().sum().over(grp) / 2)

        output["cmp"] = (
            data.lazy()
            .unpivot(
                on=chars,
                index=["eom", "size_grp", "ret_exc_lead1m"],
                variable_name="characteristic",
                value_name="var",
            )
            .filter(pl.col("var").is_not_null())
            .with_columns(weight.alias("weight"))
            .group_by(grp)
            .agg(
                pl.len().alias("n_stocks"),
                (pl.col("ret_exc_lead1m") * pl.col("weight")).sum().alias("ret_weighted"),
                (pl.col("var") * pl.col("weight")).sum().alias("signal_weighted"),
                pl.col("var").std().alias("sd_var"),
            )
            .filter(pl.col("sd_var") != 0)
            .drop("sd_var")
            .with_columns(
                pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom"),
                pl.lit(excntry.upper()).alias("excntry"),
            )
            .collect()
        )

    return output


def regional_data(
    data: pl.DataFrame,
    mkt: pl.DataFrame,
    date_col: str,
    char_col: str,
    countries: pl.Series,
    weighting: str,
    countries_min: int,
    periods_min: int,
    stocks_min: int,
) -> pl.DataFrame:
    """Aggregate per-country factor returns up to a region.

    Description:
        Combine factor returns from each country in ``countries`` into a
        single regional series per (characteristic, date). Country weights
        come from ``weighting`` ∈ ``{"market_cap", "stocks", "ew"}``. Sparse
        rows are dropped at two stages: cohorts with fewer than
        ``countries_min`` countries, then characteristics with fewer than
        ``periods_min`` total dates.
    Steps:
        1) Build per-(country, date) weights from ``mkt``.
        2) Restrict ``data`` to in-scope countries with enough stocks.
        3) Join weights; drop rows where the country has no market return.
        4) Aggregate to one row per (char, date).
        5) Apply sparsity filters.
        6) Sort.
    Output:
        DataFrame with columns ``[char_col, date_col, n_countries, direction,
        ret_ew, ret_vw, ret_vw_cap, mkt_vw_exc]``, sorted by
        ``(char_col, date_col)``.
    """
    # 1) Per-country weights — one row per (excntry, date).
    country_weight_expr = (
        pl.when(weighting == "market_cap")
        .then(pl.col("me_lag1"))
        .when(weighting == "stocks")
        .then(pl.col("stocks").cast(pl.Float64))
        .when(weighting == "ew")
        .then(pl.lit(1.0))
        .alias("country_weight")
    )
    country_weights = mkt.select("excntry", date_col, "mkt_vw_exc", country_weight_expr)

    # 2) Restrict input to in-scope countries with enough stocks.
    in_scope = data.filter(
        pl.col("excntry").is_in(countries) & (pl.col("n_stocks_min") >= stocks_min)
    )

    # 3) Attach country weights; drop rows missing market data for that date.
    joined = in_scope.join(country_weights, on=["excntry", date_col], how="left").filter(
        pl.col("mkt_vw_exc").is_not_null()
    )

    # 4) Aggregate to one row per (char, date) — country-weighted means
    #    plus the count of contributing countries and the (constant) direction.
    aggregated = joined.group_by(char_col, date_col).agg(
        pl.len().alias("n_countries"),
        pl.col("direction").first(),
        *(
            ((pl.col(c) * pl.col("country_weight")).sum() / pl.col("country_weight").sum()).alias(c)
            for c in ("ret_ew", "ret_vw", "ret_vw_cap", "mkt_vw_exc")
        ),
    )

    # 5) Sparsity filters: drop cohorts with too few countries, then chars
    #    with too few surviving dates.
    dense = aggregated.filter(pl.col("n_countries") >= countries_min).filter(
        pl.len().over(char_col) >= periods_min
    )

    # 6) Deterministic ordering.
    return dense.sort(char_col, date_col)


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
    return pl.concat(
        regional_data(
            data=data,
            mkt=mkt,
            countries=pl.Series(region["country_codes"]),
            date_col=date_col,
            char_col=char_col,
            weighting=weighting,
            countries_min=region["countries_min"],
            periods_min=periods_min,
            stocks_min=stocks_min,
        )
        .with_columns(pl.lit(region["name"]).alias("region"))
        .select(output_cols)
        for region in regions.iter_rows(named=True)
    )


def _stack_outputs(
    portfolio_data: dict,
    key: str,
    sort_cols: list[str],
    select_cols: list[str] | None = None,
) -> pl.DataFrame | None:
    """Concatenate per-country sub-results under `key`, sort, optionally project.

    Returns None when no country produced this key.
    """
    pieces = [d[key] for d in portfolio_data.values() if d and key in d]
    if not pieces:
        return None
    out = pl.concat(pieces)
    if select_cols is not None:
        out = out.select(select_cols)
    return out.sort(sort_cols)


def _write_filtered(
    df: pl.DataFrame,
    path: Path,
    date_col: str,
    end_date: date,
) -> None:
    """Filter `df` to rows where `date_col <= end_date` and write to `path`."""
    write_dataframe(df.filter(pl.col(date_col) <= end_date), path)


def _write_split_by_key(
    df: pl.DataFrame,
    folder_path: Path,
    key_col: str,
    date_col: str,
    end_date: date,
) -> None:
    """Partition `df` by `key_col` and write one parquet per key into `folder_path`.

    Description:
        Apply the `date_col <= end_date` filter, then for each unique key value
        that is neither None nor the empty string, write the matching rows to
        ``{folder_path}/{key}.parquet``. Numeric/boolean keys (including 0 and
        False) are kept; only None and "" are skipped.
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    for key in df[key_col].unique():
        if key is None or key == "":
            continue
        filtered = df.filter((pl.col(date_col) <= end_date) & (pl.col(key_col) == key))
        write_dataframe(filtered, folder_path / f"{key}.parquet")


def load_dff_be(path: Path | None = None) -> pl.DataFrame:
    """
    Description:
        Parse the Davis-Fama-French hand-collected Moody's book equity file
        into a long Polars DataFrame. Missing values (-99.990) become null.
        Distributed via Ken French's data library:
        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#BookEquity
        The BE year range is derived from the file itself: the first BE
        column corresponds to min(First_Moody_Year), and the number of BE
        columns is the record field count minus the three id fields.
    Steps:
        1) Read each line as one string; collapse whitespace; split to a list.
        2) Derive year range from field count and min(First_Moody_Year).
        3) Name fields; unpivot BE columns to long; null -99.990; cast dtypes.
    Output:
        DataFrame: permno (i64), year (i32), be (f64 nullable, $ millions).
        BE for year t is publicly available by June 30 of year t.
    """
    if path is None:
        path = get_dff_be_path()
    raw = pl.read_csv(path, has_header=False, separator="\x01", new_columns=["raw"]).with_columns(
        pl.col("raw").str.strip_chars().str.replace_all(r"\s+", " ").str.split(" ")
    )

    n_fields = raw.select(pl.col("raw").list.len().unique()).to_series()
    if len(n_fields) != 1:
        msg = f"ragged records: field counts {sorted(n_fields)}"
        raise ValueError(msg)
    start_year = raw.select(pl.col("raw").list.get(1).cast(pl.Int32).min()).item()
    years = [start_year + i for i in range(n_fields[0] - 3)]

    fields = ["permno", "first_moody_year", "last_moody_year", *[f"be_{y}" for y in years]]
    return (
        raw.with_columns(pl.col("raw").list.to_struct(fields=fields))
        .unnest("raw")
        .unpivot(
            index="permno",
            on=[f"be_{y}" for y in years],
            variable_name="year",
            value_name="be",
        )
        .with_columns(
            pl.col("permno").cast(pl.Int64),
            pl.col("year").str.strip_prefix("be_").cast(pl.Int32),
            pl.when(pl.col("be").cast(pl.Float64) == -99.99)
            .then(None)
            .otherwise(pl.col("be"))
            .cast(pl.Float64)
            .alias("be"),
        )
        .sort(["permno", "year"])
    )


def _dhs_wd_idx(col: pl.Expr) -> pl.Expr:
    """Weekday (business-day) index for a date column, so a difference of two
    indices counts the trading days between them for the ABR event window.

    Anchor 1960-01-04 (a Monday) has index 0; each weekday steps by 1 and Sat/Sun
    collapse onto the prior Friday (both share that Friday's index). For calendar
    offset n days: full weeks contribute 5 each, and the within-week remainder is
    capped at 4 (Fri) so weekend days don't advance the count.
    """
    n = (col - pl.date(1960, 1, 4)).dt.total_days()
    weeks = n // 7
    return weeks * 5 + pl.min_horizontal(n - weeks * 7, pl.lit(4))


def _dhs_non_financial(col: str = "siccd") -> pl.Expr:
    """Keep non-financials: drop SIC 6000-6999, but KEEP a missing SIC."""
    return pl.col(col).is_null() | ~((pl.col(col) >= 6000) & (pl.col(col) <= 6999))


def _dhs_size_group(me: str, p50: str = "P50") -> pl.Expr:
    """NYSE-median size split: B if ME>P50; S if ME<=P50 or ME missing; null
    (dropped) if the P50 breakpoint is missing."""
    return (
        pl.when(pl.col(me).is_not_null() & pl.col(p50).is_not_null() & (pl.col(me) > pl.col(p50)))
        .then(pl.lit("B"))
        .when(pl.col(me).is_null() | (pl.col(p50).is_not_null() & (pl.col(me) <= pl.col(p50))))
        .then(pl.lit("S"))
        .otherwise(None)
    )


def _dhs_group_quantiles(
    df: pl.DataFrame,
    by: str,
    var: str,
    pcts: list[int],
    prefix: str,
    *,
    pool: pl.Expr,
    min_stocks: int = 0,
) -> pl.DataFrame:
    """QUANTILE_DISC percentiles of `var` per (excntry, `by`) over the breakpoint
    `pool` (US = primaryexch=='N'). Null `var` rows are dropped so the min-stocks
    COUNT sees only rankable stocks. Thin wrapper over country_breakpoints."""
    specs = [(var, p / 100, f"{prefix}{p}") for p in pcts]
    return country_breakpoints(
        df,
        specs,
        group_keys=["excntry", by],
        pool=pool & pl.col(var).is_not_null(),
        min_stocks=min_stocks,
        usa_exempt=True,
    )


def _dhs_form_factor(
    base: pl.DataFrame,
    *,
    long: tuple[str, str],
    short: tuple[str, str],
    name: str,
    date_col: str = "eom",
    ret_col: str = "ret",
    w_col: str = "lagCRSPSIZE",
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
) -> pl.DataFrame:
    """VW the `portfolio` cells (via vw_bucket_returns, ordered for float
    determinism) and form ((long1+long2)-(short1+short2))/2. Shared by the monthly
    sorts (defaults) and the daily leg (date_col="date", w_col="w"); the ROW-only
    min_stocks_pf gate is applied in vw_bucket_returns (USA bypassed). Returns
    [excntry, date_col, name] in the (decimal) return unit."""
    wide = vw_bucket_returns(
        base.filter(
            pl.col("portfolio").is_not_null()
            & pl.col(ret_col).is_not_null()
            & pl.col(w_col).is_not_null()
        ),
        buckets=[*long, *short],
        weight_col=w_col,
        bucket_col="portfolio",
        ret_col=ret_col,
        date_col=date_col,
        min_stocks_pf=min_stocks_pf,
        ordered=True,
    )
    long_leg = (pl.col(long[0]) + pl.col(long[1])) / 2
    short_leg = (pl.col(short[0]) + pl.col(short[1])) / 2
    return wide.select("excntry", date_col, (long_leg - short_leg).alias(name))


def _dhs_universe() -> pl.Expr:
    """US common-stock tradable universe — the CIZ legacy-SHRCD/EXCHCD replica
    (== _mp_universe): CIZ screen + conditionaltype=='RW' + tradingstatusflg=='A'
    (drops halted/suspended). The NYSE size-median pool uses the plain _ciz_universe
    (via ff_load_crsp_panel), not this screen."""
    return _mp_universe()


def _dhs_load_msf(raw_dir: Path, beg: int, end: int) -> pl.LazyFrame:
    """Scan crsp_msf_v2 for [beg, end] (no universe screen: the monthly panel
    screens itself and the IR build needs full history for its 60-month window).
    (permno, mthcaldt) is unique in CIZ v2, so no de-duplication is needed. Decimal
    ret/cap are cast to Float64 — Decimal division truncates to the operands' scale
    (12.34/10.00 -> 1.23), corrupting the IR ratio. `cap` is mthcap ($1000s)."""
    return (
        pl.scan_parquet(raw_dir / "crsp_msf_v2.parquet")
        .rename({"mthcaldt": "date", "mthret": "ret", "mthcap": "cap"})
        .with_columns(pl.col("ret", "cap").cast(pl.Float64))
        .select(
            "permno",
            "date",
            "ret",
            "cap",
            "siccd",
            "securitytype",
            "securitysubtype",
            "sharetype",
            "issuertype",
            "usincflg",
            "conditionaltype",
            "tradingstatusflg",
            "primaryexch",
        )
        .filter(pl.col("date").dt.year().is_between(beg, end))
    )


def _dhs_monthly_panel(msf: pl.DataFrame, beg: int, end: int) -> pl.DataFrame:
    """Master monthly return panel: _dhs_universe screen, price clean, SIZE =
    mthcap/1000 ($MM), lagCRSPSIZE = prior-month SIZE nulled across a calendar gap.
    cyear maps July(t)..June(t+1) to t within [beg, end]. The NYSE flag is
    primaryexch ("N"), not a numeric exchcd; siccd is dense per-row. Output:
    [excntry, id, CRSPDATE, eom, year, month, cyear, ret, CRSPSIZE, lagCRSPSIZE,
    primaryexch, siccd, size_grp]."""
    cbase = pl.when(pl.col("month") >= 7).then(pl.col("year")).otherwise(pl.col("year") - 1)
    # reportbreak gates the lagged SIZE (a >1 month gap nulls the lag). Per-firm
    # ops partition on the globally-unique `id` (== permno for US), so ROW ids
    # never share a partition with a US permno.
    reportbreak = 12 * (pl.col("year") - pl.col("year").shift(1).over("id")) + (
        pl.col("month") - pl.col("month").shift(1).over("id")
    )
    return (
        # passing rows are already RW + N/A/Q; NYSE flag carried as primaryexch (no exchcd screen)
        msf.filter(_dhs_universe())
        .filter((pl.col("cap") > 0) & pl.col("ret").is_not_null())  # cap-native validity
        .with_columns(
            SIZE=pl.col("cap") / 1000,  # mthcap ($1000s) -> $MM
            eom=pl.col("date").dt.month_end(),  # canonical monthly key (real Date)
            year=pl.col("date").dt.year(),
            month=pl.col("date").dt.month(),
            excntry=pl.lit("USA"),  # country dimension
            id=pl.col("permno"),  # country-generic stock key (== permno for US)
            size_grp=pl.lit(
                None, dtype=pl.String
            ),  # ROW-only breakpoint pool tag; null for US (NYSE pool)
        )
        # sort by (excntry, id, date) so each id's rows are date-ordered for the
        # per-id lags; id == permno for US, so the ordering is unchanged.
        .sort(["excntry", "id", "date"])
        .with_columns(
            cyear=pl.when((cbase >= beg) & (cbase <= end)).then(cbase).otherwise(None),
            lagCRSPSIZE=pl.when(reportbreak > 1)
            .then(None)
            .otherwise(pl.col("SIZE").shift(1).over("id")),
        )
        .rename({"date": "CRSPDATE", "SIZE": "CRSPSIZE"})
        .select(
            "excntry",
            "id",
            "CRSPDATE",
            "eom",
            "year",
            "month",
            "cyear",
            "ret",
            "CRSPSIZE",
            "lagCRSPSIZE",
            "primaryexch",
            "siccd",
            "size_grp",
        )
    )


def _dhs_row_universe() -> pl.Expr:
    """ROW (ex-US) jkp main-sample screen: common stock, primary security, main
    observation, main exchange, ME present; excntry != USA (US flows the CRSP path)."""
    return (
        (pl.col("excntry") != US_EXCNTRY)
        & (pl.col("common") == 1)
        & (pl.col("primary_sec") == 1)
        & (pl.col("obs_main") == 1)
        & (pl.col("exch_main") == 1)
        & pl.col("me").is_not_null()
    )


def _dhs_load_row_panel(interim_dir: Path, beg: int, end: int) -> pl.DataFrame:
    """ROW (ex-US) monthly panel from jkp's world_data, in the SAME schema as the
    US _dhs_monthly_panel so the two concat. me -> CRSPSIZE; lagCRSPSIZE = prior-
    month me nulled across a calendar gap (world_msf has no me_lag1, so it is
    computed like the US panel); sic (from world_msf) -> siccd; primaryexch null
    (ROW has no NYSE; the FIN breakpoint pool uses size_grp). cyear maps
    July(t)..June(t+1) to t in [beg, end]. Dtypes match the US panel's so
    pl.concat(vertical) leaves US bytes intact."""
    sic = pl.scan_parquet(interim_dir / "world_msf.parquet").select(
        "id", "eom", pl.col("sic").cast(pl.Int64).alias("siccd")
    )
    cbase = pl.when(pl.col("month") >= 7).then(pl.col("year")).otherwise(pl.col("year") - 1)
    reportbreak = 12 * (pl.col("year") - pl.col("year").shift(1).over("id")) + (
        pl.col("month") - pl.col("month").shift(1).over("id")
    )
    return (
        pl.scan_parquet(interim_dir / "world_data_prelim.parquet")
        .filter(_dhs_row_universe() & pl.col("eom").dt.year().is_between(beg, end))
        .join(sic, on=["id", "eom"], how="left")
        .with_columns(year=pl.col("eom").dt.year(), month=pl.col("eom").dt.month())
        .sort(["id", "eom"])  # per-id date order for the gap-nulled prior-month lag
        .with_columns(
            lagCRSPSIZE=pl.when(reportbreak > 1)
            .then(None)
            .otherwise(pl.col("me").shift(1).over("id"))
            .cast(pl.Float64),
        )
        .select(
            "excntry",
            pl.col("id").cast(pl.Int64),
            pl.col("eom").alias("CRSPDATE"),
            "eom",
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int8),
            pl.when((cbase >= beg) & (cbase <= end))
            .then(cbase)
            .otherwise(None)
            .cast(pl.Int32)
            .alias("cyear"),
            pl.col("ret").cast(pl.Float64),
            pl.col("me").cast(pl.Float64).alias("CRSPSIZE"),
            "lagCRSPSIZE",
            pl.lit(None, dtype=pl.String).alias("primaryexch"),
            "siccd",
            pl.col("size_grp").cast(pl.String),
        )
        .collect()
    )


def _dhs_row_chars(interim_dir: Path, beg: int, end: int) -> pl.DataFrame:
    """ROW (ex-US) annual FIN characteristics from jkp's world_data, in the US
    annual_chars schema [id, datadate, YEAR, lagBE, NS, IR] (excntry flows in later
    via the id join to the ROW panel). Computed at each June(t) per id:
        NS = log(1 + chcsho_12m) = log(adj_shares_June(t) / adj_shares_June(t-1))
             (reuses jkp's chcsho_12m 12-month share-change characteristic)
        IR = log(me_June(t) / me_June(t-5)) - log(60-month gross cumret)  (DHS 5-yr)
        lagBE = be_me_June(t-1)  (lagged positive-book-equity guard; be_me>0 <=> BE>0;
                                  fin_factor keeps >=0)
    IR's 60-month sum needs the full monthly series, so it is computed before the
    June filter; the 5-yr / 1-yr / prior-June lags are positional shifts over the
    June rows (after the filter). YEAR-relabel keeps the validated fin_factor
    offsets untouched: IR belongs to the June(t) rebalance (cyear=t, YEAR==cyear)
    so YEAR=t; NS uses YEAR+1==cyear so it is emitted at YEAR=t-1. [beg, end] clip
    the eom year, matching the US msf window (60-mo warm-up applies as for the US)."""
    log1p = pl.when((1 + pl.col("ret")) > 0).then((1 + pl.col("ret")).log()).otherwise(None)
    n_valid = (
        log1p.is_not_null().cast(pl.Int32).rolling_sum(window_size=60, min_samples=60).over("id")
    )

    # world_data has ~1% per-id month gaps, so positional lags are CALENDAR-guarded: a
    # char is kept only if the prior row sits at the exact offset (12mo NS / 60mo IR),
    # else nulled. _mgap = calendar months between eom and its `n`-rows-prior eom.
    def _mgap(n: int) -> pl.Expr:
        e = pl.col("eom").shift(n).over("id")
        return (pl.col("eom").dt.year() - e.dt.year()) * 12 + (
            pl.col("eom").dt.month() - e.dt.month()
        )

    me_lag5 = pl.col("me").shift(5).over("id")  # 5 June rows back (post-June-filter)
    me_ratio = (
        pl.when(_mgap(5) == 60).then(safe_div(pl.col("me"), me_lag5, "me_ratio")).otherwise(None)
    )
    # NS reuses jkp's chcsho_12m characteristic (trailing-12m share change, arithmetic
    # ratio-1 of the same shares*adjfct) instead of recomputing the June-over-June ratio:
    # NS = log(adj_shares(t)/adj_shares(t-12)) = log(1 + chcsho_12m), sampled at the June
    # formation. Value-exact vs the prior direct computation on ROW (both use the same
    # Compustat-Global shares*adjfct); chcsho_12m carries its own 12-month/count guard.
    ns_expr = pl.when((1 + pl.col("chcsho_12m")) > 0).then((1 + pl.col("chcsho_12m")).log())
    june = (
        pl.scan_parquet(interim_dir / "world_data_prelim.parquet")
        .filter((pl.col("excntry") != US_EXCNTRY) & pl.col("eom").dt.year().is_between(beg, end))
        .select("id", "eom", "me", "ret", "be_me", "chcsho_12m")
        .sort(["id", "eom"])
        .with_columns(
            # 60-month cumret: 60 valid rows AND the window spans exactly 60
            # contiguous calendar months (window-start eom is 59 months back)
            log_cumret=pl.when((n_valid == 60) & (_mgap(59) == 59))
            .then(log1p.rolling_sum(window_size=60, min_samples=60).over("id"))
            .otherwise(None),
        )
        .filter(pl.col("eom").dt.month() == 6)
        .with_columns(
            NS=ns_expr.otherwise(None),
            IR=pl.when(me_ratio > 0).then(me_ratio.log()).otherwise(None) - pl.col("log_cumret"),
            lagBE=pl.col("be_me").shift(1).over("id"),  # prior-June be_me (book/market)
            YEAR=pl.col("eom").dt.year().cast(pl.Int32),
            datadate=pl.col("eom"),
        )
        .select("id", "datadate", "YEAR", "lagBE", "NS", "IR")
        .collect()
    )
    # split so fin_factor's offsets land both on the June(t) rebalance:
    #   IR-row at YEAR=t   (IR filter YEAR==cyear),  NS-row at YEAR=t-1 (NS filter YEAR+1==cyear)
    ir_rows = june.select("id", "datadate", "YEAR", "lagBE", "IR").with_columns(
        NS=pl.lit(None, pl.Float64)
    )
    ns_rows = june.select(
        "id", "datadate", (pl.col("YEAR") - 1).alias("YEAR"), "lagBE", "NS"
    ).with_columns(IR=pl.lit(None, pl.Float64))
    return pl.concat([ir_rows, ns_rows], how="diagonal_relaxed")


def _dhs_ccm_link(comp: pl.LazyFrame, endfyr_col: str, raw_dir: Path) -> pl.LazyFrame:
    """Link Compustat rows to CRSP permnos via CCM: LINKTYPE in (LU,LC,LD,LN,LS,LX),
    usedflag == 1, LINKDT <= endfyr <= LINKENDDT (NULL bound = open). lpermno is
    cast off Decimal (join key vs Int64 permno)."""
    link = (
        pl.scan_parquet(raw_dir / "crsp_ccmxpf_linktable.parquet")
        .with_columns(pl.col("lpermno").cast(pl.Int64))
        .filter(
            pl.col("linktype").is_in(["LU", "LC", "LD", "LN", "LS", "LX"])
            & (pl.col("usedflag") == 1)  # usedflag is Float64; == 1 serves
            & pl.col("lpermno").is_not_null()
        )
    )
    return comp.join(link, on="gvkey", how="inner").filter(
        (pl.col("linkdt").is_null() | (pl.col("linkdt") <= pl.col(endfyr_col)))
        & (pl.col("linkenddt").is_null() | (pl.col(endfyr_col) <= pl.col("linkenddt")))
    )


def _dhs_issuance_chars(msf: pl.DataFrame, raw_dir: Path, beg: int, end: int) -> pl.DataFrame:
    """Annual Compustat characteristics (NS, lagBE, IR per fiscal year).
    `msf` is the RAW unfiltered monthly file (IR is built
    off it). Financials are screened later on the CRSP siccd, never on funda's
    sich.

    Steps:
        1) Scan comp_funda (industrial screen + fyear window), cast numerics to
           Float64, link to CRSP via CCM; keep LATEST datadate per (LPERMNO, YEAR).
        2) NS = log(adjusted-shares ratio) with divide/nonpositive guards; BE via
           the equity waterfall; lagged BE across a calendar gap.
        3) IR = log(ME_June/ME_June_5yr_ago) - log(60-month gross cumret), on the
           raw msf's June rows.

    Output: one row per (LPERMNO, YEAR) with NS, lagBE, IR.
    """
    funda_num = [
        "at",
        "csho",
        "ajex",
        "pstkrv",
        "pstkl",
        "pstk",
        "seq",
        "ceq",
        "lt",
        "txditc",
        "txdb",
        "itcb",
    ]
    funda = (
        pl.scan_parquet(raw_dir / "comp_funda.parquet")
        .filter(_compustat_na_filter())
        .filter(pl.col("fyear").is_between(beg, end))  # datadate already Date
        .with_columns(
            *[pl.col(c).cast(pl.Float64) for c in funda_num],
            endfyr=pl.col("datadate"),
        )
    )
    # collect: linked pool feeds the eager per-(LPERMNO, YEAR) dedupe + char build
    linked = _dhs_ccm_link(funda, "endfyr", raw_dir).rename({"lpermno": "LPERMNO"}).collect()

    reportbreak = pl.col("YEAR") - pl.col("YEAR").shift(1).over("LPERMNO")

    def _lag(val: pl.Expr) -> pl.Expr:
        # gap-nulled 1-period lag over LPERMNO; missing break fails >1 (lag taken), first row null via shift()
        return pl.when(reportbreak > 1).then(None).otherwise(val.shift(1).over("LPERMNO"))

    # NS = log(adjusted-shares ratio); div-by-zero or nonpositive ratio -> missing
    adj_csho = pl.col("csho") * pl.col("ajex")
    lag_adj = _lag(adj_csho)
    ratio = safe_div(adj_csho, lag_adj, "ratio")

    # Book-equity waterfall; each component falls back through its alternatives.
    # preferred: redemption value, else liquidating value, else par
    preferred = pl.coalesce("pstkrv", "pstkl", "pstk")
    # shareholders' equity: SEQ, else CEQ + preferred par, else AT - LT
    shareholders_equity = (
        pl.when(pl.col("seq").is_not_null())
        .then(pl.col("seq"))
        .when(pl.col("ceq").is_not_null())
        .then(pl.col("ceq") + pl.col("pstk"))
        .otherwise(pl.col("at") - pl.col("lt"))
    )
    # deferred taxes: TXDITC, else TXDB+ITCB (missing addend -> the other alone; both missing -> missing)
    deferred_tax = (
        pl.when(pl.col("txditc").is_not_null())
        .then(pl.col("txditc"))
        .when(pl.col("txdb").is_null())
        .then(pl.col("itcb"))
        .when(pl.col("itcb").is_null())
        .then(pl.col("txdb"))
        .otherwise(pl.col("txdb") + pl.col("itcb"))
    )
    # BE = equity + deferred_tax - preferred; missing deferred_tax/preferred -> 0, missing equity -> missing
    be = shareholders_equity + deferred_tax.fill_null(0) - preferred.fill_null(0)

    chars = (
        linked.sort(["gvkey", "datadate"])  # gvkey order is the dedupe tie-break
        .with_columns(YEAR=pl.col("datadate").dt.year())  # calendar year of FY-end, NOT fyear
        # dedupe: keep LATEST datadate per (LPERMNO, YEAR)
        .sort(["LPERMNO", "YEAR", "datadate"])
        .unique(subset=["LPERMNO", "YEAR"], keep="last", maintain_order=True)
        .sort(["LPERMNO", "YEAR"])
        .with_columns(
            NS=pl.when(ratio > 0).then(ratio.log()).otherwise(None),
            lagBE=_lag(be),
        )
    )

    # IR = log_ME - log_cumret in ONE (permno,date)-sorted pass over raw msf:
    #   log_ME     = log(ME_June / ME_June_5yr_ago), 5-yr positional June lag; ME/0 -> null
    #   log_cumret = trailing-60-month sum of log1p(ret); nonpositive gross or any missing
    #                window month -> invalidated. One sort serves window + shift(5); no self-join.
    me_june = pl.when(pl.col("cap") > 0).then(pl.col("cap") / 1000).otherwise(None)  # $MM
    log1p = pl.when((1 + pl.col("ret")) > 0).then((1 + pl.col("ret")).log()).otherwise(None)
    n_valid = (
        log1p.is_not_null()
        .cast(pl.Int32)
        .rolling_sum(window_size=60, min_samples=60)
        .over("permno")
    )
    me_lag5 = pl.col("ME_June").shift(5).over("permno")
    me_ratio = safe_div(pl.col("ME_June"), me_lag5, "me_ratio")
    ir = (
        msf.select("permno", "date", "cap", "ret")
        .sort(["permno", "date"])
        # ME_June + rolling sum need the full series, so compute before the June filter
        .with_columns(
            ME_June=me_june,
            log_cumret=pl.when(n_valid == 60)
            .then(log1p.rolling_sum(window_size=60, min_samples=60).over("permno"))
            .otherwise(None),
        )
        .filter(pl.col("date").dt.month() == 6)
        # the 5-year June lag is positional over the June rows, so it comes after the filter
        .with_columns(log_ME=pl.when(me_ratio > 0).then(me_ratio.log()).otherwise(None))
        .select(
            pl.col("permno").alias("LPERMNO"),
            pl.col("date").dt.year().alias("YEAR"),
            (pl.col("log_ME") - pl.col("log_cumret")).alias("IR"),
        )
    )
    # expose the firm key as the globally-unique `id` (== LPERMNO for this US
    # builder) so the concat-bound FIN path keys uniformly on id.
    return chars.join(ir, on=["LPERMNO", "YEAR"], how="left").rename({"LPERMNO": "id"})


def _dhs_market_daily(interim_dir: Path) -> pl.LazyFrame:
    """Daily market return -> [excntry, date, mkt] DECIMAL: the interim
    market_returns_daily mkt_vw, per country, for the announcement (ret - mkt)
    window. Under US-only stock data only the USA rows are ever joined."""
    return (
        pl.scan_parquet(interim_dir / "market_returns_daily.parquet")
        .select(
            pl.col("excntry"),
            pl.col("date").cast(pl.Date),
            pl.col("mkt_vw").cast(pl.Float64).alias("mkt"),
        )
        .sort(["excntry", "date"])
    )


def _dhs_abr_window(ann: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Abr per (id, RDQ) over the weekday window [RDQ-2, RDQ+1]. `ann` = [id,
    datadate, rdq]; `daily` = [id, date, ret, mkt] (ret, mkt decimal). An inner
    window join drops a missing RDQ, the window
    sum and non-missing day count are each scaled by the announcement rows sharing
    (id, RDQ), counts are kept in [2,4], and RDQ must sit >=1 weekday before its
    calendar month end. The window sum is ORDERED (sort + maintain_order) so the
    float add is deterministic at a discrete quantile breakpoint. Country-generic:
    the US path renames LPERMNO->id, the ROW path is already id-keyed. Returns
    [id, DATADATE, RDQ, Abr, eom]."""
    ann = ann.filter(pl.col("rdq").is_not_null()).select("id", "datadate", "rdq")
    ann_groups = ann.group_by(["id", "rdq"]).agg(n_arows=pl.len())
    rdq_wd = _dhs_wd_idx(pl.col("rdq"))
    win = ann_groups.with_columns(
        wd=pl.concat_list(rdq_wd - 2, rdq_wd - 1, rdq_wd, rdq_wd + 1)
    ).explode("wd")
    daily_idx = daily.with_columns(wd=_dhs_wd_idx(pl.col("date")))
    joined = win.join(daily_idx, on=["id", "wd"], how="inner")
    stats = (
        joined.sort(["id", "rdq", "wd"], maintain_order=True)
        .group_by(["id", "rdq"], maintain_order=True)
        .agg(
            abr_sum=(pl.col("ret") - pl.col("mkt")).sum(),  # sum skips null terms
            cnt=pl.col("ret").is_not_null().sum(),  # count of non-missing ret
        )
    )
    rme = pl.col("rdq").dt.month_end()
    return (
        ann.join(ann_groups, on=["id", "rdq"], how="left")
        .join(stats, on=["id", "rdq"], how="inner")
        .with_columns(
            Abr=pl.col("abr_sum") * pl.col("n_arows"),
            count=pl.col("cnt") * pl.col("n_arows"),
        )
        .filter((pl.col("count") >= 2) & (pl.col("count") <= 4))
        .with_columns(n_weekdays=_dhs_wd_idx(rme) - _dhs_wd_idx(pl.col("rdq")))
        .filter(pl.col("n_weekdays") >= 1)
        .select(
            "id",
            pl.col("datadate").alias("DATADATE"),
            pl.col("rdq").alias("RDQ"),
            "Abr",
            rme.alias("eom"),  # announcement month-end
        )
        .sort(["id", "eom", "RDQ", "DATADATE"])
    )


def _dhs_align_abr(panel_rows: pl.DataFrame, abr: pl.DataFrame) -> pl.DataFrame:
    """Align per-(id, RDQ) Abr to the monthly panel. Two announcements in one
    id-month: the later RDQ is ordered FIRST, so its
    lagAbr is null across the month gap and it drops; the earlier-RDQ row survives
    carrying the later Abr as lagAbr, and the forward-fill feeds the earlier quarter's
    DATADATE into the 6-month staleness test. `panel_rows` must carry [excntry, eom,
    id, primaryexch, lagCRSPSIZE, siccd, size_grp, year, month, CRSPDATE, ret]; `abr`
    is [id, DATADATE, RDQ, Abr, eom]. Per-firm ops partition on the globally-unique id."""
    filled_datadate = pl.col("DATADATE").forward_fill().over("id")
    mreportbreak = 12 * (pl.col("year") - pl.col("year").shift(1).over("id")) + (
        pl.col("month") - pl.col("month").shift(1).over("id")
    )
    return (
        panel_rows.join(abr, on=["id", "eom"], how="left")
        .sort(["id", "CRSPDATE", "RDQ", "DATADATE"], descending=[False, False, True, True])
        # fill DATADATE/RDQ, then the fiscal-quarter->CRSP-month gap off the FILLED DATADATE
        .with_columns(
            DATADATE=filled_datadate,
            RDQ=pl.col("RDQ").forward_fill().over("id"),
            gap_crsp_fqtr=(pl.col("CRSPDATE").dt.year() - filled_datadate.dt.year()) * 12
            + (pl.col("CRSPDATE").dt.month() - filled_datadate.dt.month()),
        )
        # keep rows within 6 months; rows with a missing gap are also kept
        .filter(pl.col("gap_crsp_fqtr").is_null() | (pl.col("gap_crsp_fqtr") <= 6))
        # Abr fill runs AFTER the gap filter (on the surviving rows)
        .with_columns(Abr=pl.col("Abr").forward_fill().over("id"))
        .sort(["id", "eom", "RDQ", "DATADATE"], descending=[False, False, True, True])
        .with_columns(
            lagAbr=pl.when(mreportbreak > 1).then(None).otherwise(pl.col("Abr").shift(1).over("id"))
        )
        # drop financials (SIC 6000-6999, missing siccd KEPT) AND require non-null lagAbr
        .filter(_dhs_non_financial() & pl.col("lagAbr").is_not_null())
    )


def _dhs_ibes_announcements(raw_dir: Path, interim_dir: Path, beg: int, end: int) -> pl.DataFrame:
    """IBES international quarterly EPS announcement dates bridged to the world `id`.
    Compustat g_fundq has no rdq ex-US, so ibes.actu_epsint (usfirm=0, measure EPS,
    QTR periodicity) supplies anndats (announcement date, the ex-US RDQ) and pends
    (fiscal period end). The IBES ticker is linked to gvkey via the Compustat-global
    security master's ibtic (the vendor-maintained IBES link), then to the world id
    of the gvkey's primary security. Bridges are deduped one-to-one (deterministic
    sort + keep-first). One (id, period-end) row keeps the EARLIEST anndats. Returns
    [id, datadate (=pends), rdq (=anndats)]."""
    # ibtic (IBES ticker) -> gvkey, one gvkey per ticker
    ibtic_gv = (
        pl.scan_parquet(raw_dir / "comp_g_security.parquet")
        .filter(pl.col("ibtic").is_not_null() & pl.col("gvkey").is_not_null())
        .select(pl.col("ibtic").alias("ticker"), pl.col("gvkey").cast(pl.Utf8))
        .unique()
        .sort(["ticker", "gvkey"])
        .unique(subset=["ticker"], keep="first", maintain_order=True)
    )
    # gvkey -> world id (primary security in the ROW universe), one id per gvkey
    gv_id = (
        pl.scan_parquet(interim_dir / "world_data_prelim.parquet")
        .filter(_dhs_row_universe())
        .select(pl.col("gvkey").cast(pl.Utf8), pl.col("id").cast(pl.Int64))
        .unique()
        .sort(["gvkey", "id"])
        .unique(subset=["gvkey"], keep="first", maintain_order=True)
    )
    return (
        pl.scan_parquet(raw_dir / "ibes_actu_epsint.parquet")
        .filter(
            (pl.col("usfirm") == 0)
            & (pl.col("measure") == "EPS")
            & (pl.col("pdicity") == "QTR")
            & pl.col("ticker").is_not_null()
            & pl.col("anndats").is_not_null()
            & pl.col("pends").is_not_null()
        )
        .select(
            "ticker",
            pl.col("anndats").cast(pl.Date).alias("rdq"),
            pl.col("pends").cast(pl.Date).alias("datadate"),
        )
        .filter(pl.col("rdq").dt.year().is_between(beg, end))
        .join(ibtic_gv, on="ticker", how="inner")
        .join(gv_id, on="gvkey", how="inner")
        .select("id", "datadate", "rdq")
        # earliest announcement per (id, period-end)
        .sort(["id", "datadate", "rdq"])
        .unique(subset=["id", "datadate"], keep="first", maintain_order=True)
        .collect()
    )


def _dhs_row_announcement_returns(
    monthly_panel: pl.DataFrame,
    raw_dir: Path,
    interim_dir: Path,
    beg: int,
    end: int,
) -> pl.DataFrame:
    """ROW (ex-US) announcement returns (Abr) for international PEAD, in the SAME
    schema as the US _dhs_announcement_returns abr_panel so the two concat. The
    announcement date is IBES anndats (bridged to the world id via ibtic->gvkey);
    Abr is the [anndats-2, anndats+1] weekday CAR of world_dsf daily returns net of
    the per-country market_returns_daily benchmark, aligned to the ROW monthly panel
    exactly like the US path."""
    row_panel = monthly_panel.filter(pl.col("excntry") != US_EXCNTRY)
    # ROW daily returns net of the per-country market benchmark: [id, date, ret, mkt]
    daily = (
        _dhs_load_row_daily(interim_dir)
        .join(_dhs_market_daily(interim_dir).collect(), on=["excntry", "date"], how="left")
        .filter(pl.col("date").dt.year().is_between(beg, end))
        .select("id", "date", "ret", "mkt")
    )
    ann = _dhs_ibes_announcements(raw_dir, interim_dir, beg, end)
    abr = _dhs_abr_window(ann, daily)
    return _dhs_align_abr(
        row_panel.select(
            "excntry",
            "eom",
            "id",
            "primaryexch",
            "lagCRSPSIZE",
            "siccd",
            "size_grp",
            "year",
            "month",
            "CRSPDATE",
            "ret",
        ),
        abr,
    )


def _dhs_announcement_returns(
    monthly_panel: pl.DataFrame,
    raw_dir: Path,
    interim_dir: Path,
    beg: int,
    end: int,
    beg_q: int,
    end_q: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cumulative abnormal return around earnings announcements (Abr), aligned to
    the monthly CRSP panel. Daily stock data
    is crsp_dsf_v2 under the per-row universe screen; the market return is the
    interim US panel.

    Steps:
        1) Clean daily CRSP to (permno, date, ret, mkt) DECIMAL; the one decimal
           ret feeds both the Abr window and the daily factor build.
        2) Link fundq to CRSP, dedupe to one row per (LPERMNO, datadate).
        3) Abr over the weekday window [RDQ-2, RDQ+1], scaling window sum and day
           count by the announcement rows sharing (LPERMNO, RDQ); keep counts in
           [2,4] and RDQ >=1 weekday before month end.
        4) Align to the monthly panel: later-RDQ fan-out row ordered first,
           forward-fill DATADATE/RDQ/Abr, 6-month staleness gate (missing gap
           kept), financials screen (missing siccd kept), require non-null lagAbr.

    Output: (abr_panel with Abr/lagAbr + returns/size; daily_stock [permno, date,
        ret] DECIMAL for the daily factor build).
    """
    # daily CRSP cleaned to (permno, date, ret decimal); (permno, dlycaldt) is unique
    # in CIZ v2, so no de-duplication is needed; universe screen at scan time.
    # cap-native validity: dlycap>0 & ret present (dlycap == |dlyprc|*shrout, $1000s).
    daily_clean = (
        pl.scan_parquet(raw_dir / "crsp_dsf_v2.parquet")
        .filter(_dhs_universe())
        .select(
            pl.col("permno"),
            pl.col("dlycaldt").alias("date"),
            pl.col("dlyret").cast(pl.Float64).alias("ret"),
            pl.col("dlycap").cast(pl.Float64).alias("cap"),
        )
        .filter(pl.col("date").dt.year().is_between(beg, end))
        .filter((pl.col("cap") > 0) & pl.col("ret").is_not_null())
        .with_columns(excntry=pl.lit("USA"))  # country dimension (US-only stock feed)
    )
    market = _dhs_market_daily(interim_dir)
    # collect once: daily reused by the Abr window, daily_stock, and coverage guard.
    # join on (excntry, date): with US-only stock rows only USA market rows match,
    # identical to first filtering the benchmark to USA and joining on date.
    daily = (
        daily_clean.join(market, on=["excntry", "date"], how="left")
        .select("excntry", "permno", "date", "ret", "mkt")
        .sort(["excntry", "permno", "date"])
        .collect()
    )
    daily_stock = daily.select("excntry", pl.col("permno").alias("id"), "date", "ret")
    # benchmark must cover the daily sample, else (ret - mkt) loses its mkt term and Abr -> ~0 on the uncovered tail
    daily_max = daily["date"].max()
    # coverage bound of the US benchmark specifically (daily is US-only here)
    mkt_max = (
        market.filter(pl.col("excntry") == "USA").select(pl.col("date").max()).collect().item()
    )
    if daily_max > mkt_max:
        raise ValueError(
            f"daily CRSP sample extends to {daily_max} but the market benchmark "
            f"(market_returns_daily.parquet) only covers through {mkt_max}; "
            f"refresh the stale market_returns_daily.parquet input"
        )

    fundq = (
        pl.scan_parquet(raw_dir / "comp_fundq.parquet")
        .filter(_compustat_na_filter())
        .filter(pl.col("fyearq").is_between(beg_q, end_q))  # datadate/rdq already Date
        .with_columns(endfyr=pl.col("datadate"))
    )
    # collect: linked quarterly pool feeds the eager dedupe + Abr build
    qtr_linked = _dhs_ccm_link(fundq, "endfyr", raw_dir).rename({"lpermno": "LPERMNO"}).collect()
    qtr_linked_dedup = (
        # deterministic tie-break (gvkey, linkdt) among rows sharing (LPERMNO, datadate)
        qtr_linked.sort(["LPERMNO", "datadate", "gvkey", "linkdt"]).unique(
            subset=["LPERMNO", "datadate"], keep="first", maintain_order=True
        )
    )

    # Abr per (id, RDQ) over the weekday window, then aligned to the monthly panel;
    # both steps are country-generic (US keys on LPERMNO == id). size_grp (null for
    # US) rides along for the ROW PEAD breakpoint pool; it does not affect US values.
    ann = qtr_linked_dedup.filter(pl.col("rdq").is_not_null()).select(
        pl.col("LPERMNO").alias("id"), "datadate", "rdq"
    )
    abr = _dhs_abr_window(ann, daily.select(pl.col("permno").alias("id"), "date", "ret", "mkt"))
    abr_panel = _dhs_align_abr(
        monthly_panel.select(
            "excntry",
            "eom",
            "id",
            "primaryexch",
            "lagCRSPSIZE",
            "siccd",
            "size_grp",
            "year",
            "month",
            "CRSPDATE",
            "ret",
        ),
        abr,
    )
    return abr_panel, daily_stock


def _dhs_nyse_size_median(
    raw_dir: Path, interim_dir: Path, min_stocks_bp: int = FF_MIN_STOCKS_BP
) -> pl.DataFrame:
    """Per-(excntry) size median P50, via jkp's ff_country_breakpoints (US branch =
    NYSE exchcd_us==1; ROW branch = size_grp ∈ {small,large,mega}; me>0; per-country
    min-stocks gate, USA exempt). US pool = ff_load_crsp_panel (CIZ, permco ME,
    DFF-2000); ROW pool = world_data (when present). US me is /1000 at INPUT so its
    P50 is byte-identical to before AND in the same $MM unit as US CRSPSIZE; ROW me
    is left in world units, matching ROW CRSPSIZE. Output: [excntry, eom, P50].
    """
    us = (
        ff_load_crsp_panel(raw_dir, "monthly")
        .select("date", (pl.col("me") / 1000).alias("me"), pl.col("exchcd").alias("exchcd_us"))
        .with_columns(excntry=pl.lit("USA"), size_grp=pl.lit(None, dtype=pl.String))
        .collect()  # ff_country_breakpoints takes an eager frame
    )
    panel = us
    if (interim_dir / "world_data_prelim.parquet").exists():
        row = (
            pl.scan_parquet(interim_dir / "world_data_prelim.parquet")
            .filter(_dhs_row_universe())
            .select(
                pl.col("eom").alias("date"),
                pl.col("me").cast(pl.Float64),
                "excntry",
                pl.lit(None, dtype=pl.Int64).alias("exchcd_us"),
                pl.col("size_grp").cast(pl.String),
            )
            .collect()
        )
        panel = pl.concat([us, row], how="diagonal_relaxed")
    return (
        ff_country_breakpoints(panel, [("me", 0.50, "P50")], pl.col("me") > 0, min_stocks_bp)
        .select("excntry", pl.col("date").alias("eom"), "P50")
        .sort(["excntry", "eom"])
    )


def _dhs_fin_factor(
    annual_chars: pl.DataFrame,
    monthly_panel: pl.DataFrame,
    nyse_p50: pl.DataFrame,
    min_stocks_bp: int = FF_MIN_STOCKS_BP,
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """FIN factor (NS×IR overlap, 2 size groups, annual June rebalance).
    Returns (factor [excntry, eom, fin_dhs], chars [id, eom, ns, ir],
    members [excntry, id, eom, portfolio, w]).

    Steps:
        1) Clean NS (lagBE guard, NS != 0) and IR into per-variable monthly panels
           merged to the monthly returns.
        2) Rank NS into L/2/3/4/H (negatives and positives separately, NYSE
           breakpoints), IR into NYSE quintiles; take the H/L overlap.
        3) Cross with the NYSE-median size split on June-of-CYEAR ME.
        4) Value-weight the six SL..BH legs; FIN = low-issuance minus high.
    """
    # firm key is the globally-unique `id` (annual_chars and monthly_panel both
    # expose it; == permno for US), so id-alone joins/dedupes are country-safe.
    keep = ["id", "datadate", "YEAR", "lagBE"]
    # 'lagBE < 0' also drops rows with a missing lagBE
    lagbe_ok = pl.col("lagBE").is_not_null() & (pl.col("lagBE") >= 0)
    ns_chars = annual_chars.select(keep + ["NS"]).filter(
        lagbe_ok & pl.col("NS").is_not_null() & (pl.col("NS") != 0)
    )
    ir_chars = annual_chars.select(keep + ["IR"]).filter(lagbe_ok & pl.col("IR").is_not_null())

    non_fin = _dhs_non_financial()  # drop financials (SIC 6000-6999), missing siccd KEPT
    crsp_cols = monthly_panel.select(
        "excntry", "id", "eom", "cyear", "primaryexch", "siccd", "size_grp"
    )
    ns_panel = (
        ns_chars.join(crsp_cols, on="id", how="inner")
        .filter((pl.col("YEAR") + 1 == pl.col("cyear")) & non_fin)  # NS: YEAR+1 = CYEAR
        .sort(["id", "datadate", "eom"])
    )
    ir_panel = (
        ir_chars.join(crsp_cols, on="id", how="inner")
        .filter((pl.col("YEAR") == pl.col("cyear")) & non_fin)  # IR: YEAR = CYEAR (no shift)
        .sort(["id", "datadate", "eom"])
    )

    # rank NS L/2/3/4/H, negatives and positives separately. One obs per (id,
    # YEAR): first in (id, datadate, eom) order = the FY's first month (its NYSE flag feeds the pool).
    ns_rank = (
        ns_panel.sort(["id", "datadate", "eom"])
        .unique(subset=["id", "YEAR"], keep="first", maintain_order=True)
        .select("excntry", "id", "YEAR", "NS", "primaryexch", "size_grp")
    )
    is_neg = pl.col("NS").is_not_null() & (pl.col("NS") < 0)
    # country breakpoint pool: US = NYSE (primaryexch=="N"); ROW = size_grp eligible.
    # US rows have size_grp null and ROW rows have primaryexch null, so exactly one
    # branch fires per row -> US pool is unchanged (byte-identical).
    bp_pool = (pl.col("primaryexch") == "N") | pl.col("size_grp").is_in(["small", "large", "mega"])
    bp_neg = _dhs_group_quantiles(
        ns_rank, "YEAR", "NS", [50], "P", pool=is_neg & bp_pool, min_stocks=min_stocks_bp
    )
    # a missing breakpoint = the (excntry, YEAR) pool was below min_stocks (dropped by
    # HAVING), so those stocks are EXCLUDED (group1 null), not defaulted — mirrors jkp
    # ff_assign_portfolios' inner join. USA is HAVING-exempt, so this never fires for US.
    ns_neg = (
        ns_rank.filter(is_neg)
        .join(bp_neg, on=["excntry", "YEAR"], how="left")
        .with_columns(
            group1=pl.when(pl.col("P50").is_null())
            .then(None)
            .when(pl.col("NS") > pl.col("P50"))
            .then(pl.lit("2"))
            .otherwise(pl.lit("L"))
        )
    )
    bp_pos = _dhs_group_quantiles(
        ns_rank,
        "YEAR",
        "NS",
        [30, 70],
        "P",
        pool=(pl.col("NS") > 0) & bp_pool,
        min_stocks=min_stocks_bp,
    )
    ns_pos = (
        ns_rank.filter(pl.col("NS") > 0)
        .join(bp_pos, on=["excntry", "YEAR"], how="left")
        .with_columns(
            group1=pl.when(pl.col("P30").is_null())
            .then(None)
            .when(pl.col("NS") > pl.col("P70"))
            .then(pl.lit("H"))
            .when(pl.col("NS") > pl.col("P30"))
            .then(pl.lit("4"))
            .otherwise(pl.lit("3"))
        )
    )
    umo_ns = pl.concat(
        [ns_neg.select("id", "YEAR", "group1"), ns_pos.select("id", "YEAR", "group1")]
    )

    # rank IR into NYSE quintiles L/2/3/4/H
    ir_rank = (
        ir_panel.sort(["id", "datadate", "eom"])
        .unique(subset=["id", "YEAR"], keep="first", maintain_order=True)
        .select("excntry", "id", "YEAR", "IR", "primaryexch", "size_grp")
    )
    bp_ir = _dhs_group_quantiles(
        ir_rank, "YEAR", "IR", [20, 40, 60, 80], "P", pool=bp_pool, min_stocks=min_stocks_bp
    )
    umo_ir = (
        ir_rank.join(bp_ir, on=["excntry", "YEAR"], how="left")
        .with_columns(
            group1=pl.when(pl.col("IR").is_null() | pl.col("P20").is_null())
            .then(None)  # thin pool dropped -> exclude
            .when(pl.col("IR") > pl.col("P80"))
            .then(pl.lit("H"))
            .when(pl.col("IR") > pl.col("P60"))
            .then(pl.lit("4"))
            .when(pl.col("IR") > pl.col("P40"))
            .then(pl.lit("3"))
            .when(pl.col("IR") > pl.col("P20"))
            .then(pl.lit("2"))
            .otherwise(pl.lit("L"))
        )
        .select("id", "YEAR", "group1")
    )

    ns_monthly = ns_panel.join(umo_ns, on=["id", "YEAR"], how="left").select(
        "id", "eom", pl.col("group1").alias("group_NS")
    )
    ir_monthly = ir_panel.join(umo_ir, on=["id", "YEAR"], how="left").select(
        "id", "eom", pl.col("group1").alias("group_IR")
    )
    # overlap: H needs H+H or H+missing; L symmetric; drop both-missing; else M
    g = (
        pl.when(
            ((pl.col("group_NS") == "H") & (pl.col("group_IR") == "H"))
            | ((pl.col("group_NS") == "H") & pl.col("group_IR").is_null())
            | (pl.col("group_NS").is_null() & (pl.col("group_IR") == "H"))
        )
        .then(pl.lit("H"))
        .when(
            ((pl.col("group_NS") == "L") & (pl.col("group_IR") == "L"))
            | (pl.col("group_NS").is_null() & (pl.col("group_IR") == "L"))
            | ((pl.col("group_NS") == "L") & pl.col("group_IR").is_null())
        )
        .then(pl.lit("L"))
        .otherwise(None)
    )
    me_june = monthly_panel.select(
        "excntry", "id", pl.col("eom").alias("rankdate"), pl.col("CRSPSIZE").alias("MEJune")
    )
    bp = nyse_p50.filter(pl.col("eom").dt.month() == 6)  # June NYSE P50
    group2 = _dhs_size_group("MEJune")  # size leg vs the June NYSE ME median
    base = (
        monthly_panel.select("excntry", "id", "eom", "lagCRSPSIZE", "cyear", "ret")
        .join(ns_monthly, on=["id", "eom"], how="left")
        .join(ir_monthly, on=["id", "eom"], how="left")
        .filter(~(pl.col("group_NS").is_null() & pl.col("group_IR").is_null()))
        # group1 = overlap (M fills the interior); rankdate = June-of-CYEAR
        .with_columns(
            group1=g.fill_null("M"),
            rankdate=pl.date(pl.col("cyear"), 6, 1).dt.month_end(),
        )
        .join(me_june, on=["excntry", "id", "rankdate"], how="left")
        .join(bp.rename({"eom": "rankdate"}), on=["excntry", "rankdate"], how="left")
        .with_columns(
            portfolio=pl.when(group2.is_null() | pl.col("group1").is_null())
            .then(None)
            .otherwise(group2 + pl.col("group1"))
        )
    )

    # FIN = low-issuance (L) long minus high (H) short
    factor = _dhs_form_factor(
        base, long=("SL", "BL"), short=("SH", "BH"), name="fin_dhs", min_stocks_pf=min_stocks_pf
    )
    # per-stock raw sort values, one row per (id, eom) where either char exists
    chars = ns_panel.select("id", "eom", pl.col("NS").alias("ns")).join(
        ir_panel.select("id", "eom", pl.col("IR").alias("ir")),
        on=["id", "eom"],
        how="full",
        coalesce=True,
    )
    members = base.select("excntry", "id", "eom", "portfolio", pl.col("lagCRSPSIZE").alias("w"))
    return factor, chars, members


def _dhs_pead_factor(
    abr_panel: pl.DataFrame,
    nyse_p50: pl.DataFrame,
    min_stocks_bp: int = FF_MIN_STOCKS_BP,
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """PEAD factor (monthly 2×5 sort on lagged Abr).
    Returns (factor [excntry, eom, pead_dhs], chars [id, eom, abr], members
    [excntry, id, eom, portfolio, w]).

    Steps:
        1) One obs per (id, eom); lagAbr is the sort char.
        2) Size split: month-t formation uses month t-1's NYSE P50, attached via a
           +1-month date offset then joined on the formation month.
        3) NYSE quintiles of lagAbr (2×5; only 20/80 bind).
        4) Value-weight the six SL..BH legs; PEAD = high-Abr minus low-Abr.
    """
    # one row per (id, eom), keeping the first in the current fan-out order
    # (abr_panel is sorted (id, eom) with the fan-out preserved)
    pead = abr_panel.unique(subset=["id", "eom"], keep="first", maintain_order=True).with_columns(
        SIZE=pl.col("lagCRSPSIZE")
    )

    # size leg: month-t formation uses month t-1's NYSE P50, shifted to the next
    # month-end via a date offset (gap/year-safe), then joined on the formation eom.
    bp = nyse_p50.with_columns(eom_next=pl.col("eom").dt.offset_by("1mo").dt.month_end())
    sized = (
        pead.join(
            bp.select("excntry", pl.col("eom_next").alias("eom"), "P50"),
            on=["excntry", "eom"],
            how="left",
        )
        .filter(pl.col("SIZE").is_not_null() & pl.col("lagAbr").is_not_null())
        .with_columns(group1=_dhs_size_group("SIZE"))  # size leg vs the prior-month NYSE ME median
    )

    # lagAbr breakpoints per (excntry, formation month) (P20..P80; only 20/80 bind).
    # Pool: US = NYSE (primaryexch=="N"), ROW = size_grp eligible; US rows have size_grp
    # null and ROW rows primaryexch null, so exactly one branch fires (US byte-identical).
    bp_pool = (pl.col("primaryexch") == "N") | pl.col("size_grp").is_in(["small", "large", "mega"])
    bp_char = _dhs_group_quantiles(
        sized,
        "eom",
        "lagAbr",
        [20, 40, 60, 80],
        "NYSEcharP",
        pool=bp_pool,
        min_stocks=min_stocks_bp,
    )
    # a missing breakpoint means the (excntry, eom) pool was below min_stocks and was
    # dropped by the HAVING gate; those stocks are EXCLUDED (group2 null -> portfolio
    # null), not forced into L. USA is HAVING-exempt, so this never fires for US.
    group2 = (
        pl.when(pl.col("NYSEcharP20").is_null())
        .then(None)
        .when(pl.col("lagAbr") > pl.col("NYSEcharP80"))
        .then(pl.lit("H"))
        .when(pl.col("lagAbr") > pl.col("NYSEcharP20"))
        .then(pl.lit("M"))
        .otherwise(pl.lit("L"))
    )
    ranked = (
        sized.join(bp_char, on=["excntry", "eom"], how="left")
        .with_columns(
            portfolio=pl.when(pl.col("group1").is_null() | group2.is_null())
            .then(None)
            .otherwise(pl.col("group1") + group2)
        )
        .select("id", "eom", "portfolio")
    )
    merged = pead.join(ranked, on=["id", "eom"], how="left")

    # PEAD = high-Abr (H) long minus low-Abr (L) short
    factor = _dhs_form_factor(
        merged, long=("SH", "BH"), short=("SL", "BL"), name="pead_dhs", min_stocks_pf=min_stocks_pf
    )
    chars = pead.select("id", "eom", pl.col("lagAbr").alias("abr"))
    members = merged.select("excntry", "id", "eom", "portfolio", pl.col("lagCRSPSIZE").alias("w"))
    return factor, chars, members


def _dhs_load_row_daily(interim_dir: Path) -> pl.DataFrame:
    """ROW (ex-US) daily stock returns from jkp's world_dsf, matching the US
    daily_stock schema [excntry, id, date, ret] so ROW monthly FIN membership can
    be value-weighted on ROW daily returns. world_dsf has no ME, so the universe
    screen omits the me>0 clause used for the monthly ROW panel."""
    return (
        pl.scan_parquet(interim_dir / "world_dsf.parquet")
        .filter(
            (pl.col("excntry") != US_EXCNTRY)
            & (pl.col("common") == 1)
            & (pl.col("primary_sec") == 1)
            & (pl.col("obs_main") == 1)
            & (pl.col("exch_main") == 1)
            & pl.col("ret").is_not_null()
        )
        .select("excntry", pl.col("id").cast(pl.Int64), "date", pl.col("ret").cast(pl.Float64))
        .collect()
    )


def _dhs_daily_factors(
    fin_members: pl.DataFrame,
    pead_members: pl.DataFrame,
    daily_stock: pl.DataFrame,
    row_daily: pl.DataFrame | None = None,
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
) -> pl.DataFrame:
    """Daily FIN/PEAD factor returns from the monthly portfolio membership.

    Each factor's monthly membership (excntry, id, eom, portfolio, w=lagCRSPSIZE)
    is applied to the SAME month's daily returns: the row at eom = m holds the
    portfolio during month m weighted by ME(m-1), so daily returns join on
    eom = month_end(date). _dhs_form_factor value-weights per (excntry, date, portfolio)
    with the decimal `ret`, forming the same long-minus-short spreads as the monthly
    path. `row_daily` (ex-US world_dsf returns) extends both FIN and PEAD daily to
    ROW when the memberships carry ROW rows (ROW PEAD members exist only when the
    IBES announcement inputs were present in the monthly build; else ROW drops out
    of the PEAD inner join and only US daily PEAD is produced)."""
    if row_daily is not None:
        daily_stock = pl.concat([daily_stock, row_daily], how="vertical_relaxed")
    daily = daily_stock.with_columns(eom=pl.col("date").dt.month_end())
    fin = _dhs_form_factor(
        daily.join(fin_members, on=["excntry", "id", "eom"], how="inner"),
        long=("SL", "BL"),
        short=("SH", "BH"),
        name="fin_dhs",
        date_col="date",
        w_col="w",
        min_stocks_pf=min_stocks_pf,
    )
    pead = _dhs_form_factor(
        daily.join(pead_members, on=["excntry", "id", "eom"], how="inner"),
        long=("SH", "BH"),
        short=("SL", "BL"),
        name="pead_dhs",
        date_col="date",
        w_col="w",
        min_stocks_pf=min_stocks_pf,
    )
    return fin.join(pead, on=["excntry", "date"], how="full", coalesce=True).sort(
        ["excntry", "date"]
    )


def _dhs_write_factors(frame: pl.DataFrame, date_col: str, path: Path) -> None:
    """Write [excntry, <date_col>, fin_dhs, pead_dhs] (decimal); excntry is carried
    from _dhs_form_factor (== "USA" under the US-only feed)."""
    frame.select("excntry", date_col, "fin_dhs", "pead_dhs").write_parquet(path)


@measure_time
def gen_dhs_data(
    paths: DataPaths,
    monthly_factors_path: Path,
    daily_factors_path: Path,
    chars_path: Path,
    freqs: tuple[str, ...] = ("monthly", "daily"),
    min_stocks_bp: int = FF_MIN_STOCKS_BP,
    min_stocks_pf: int = FF_MIN_STOCKS_PF,
    beg: int = 1966,
    end: int = 2025,
    beg_q: int = 1971,
    end_q: int = 2025,
) -> None:
    """Build the DHS FIN + PEAD factors (US + international) from the jkp data
    directory (no downloads).

    Reference: Daniel, Hirshleifer, and Sun (2020) 'Short- and long-horizon
    behavioral factors', Review of Financial Studies 33(4), pp. 1673-1736.

    Reads
    raw CRSP/Compustat from `paths.raw_tables_dir` and market_returns_daily from
    `paths.interim_dir`. [beg, end] bound the monthly/annual sample; [beg_q, end_q]
    the quarterly announcement sample (coverage starts ~1971).

    Steps:
        1) Monthly return panel from crsp_msf_v2 (read once).
        2) Annual issuance chars (NS, IR) + the Abr panel (which also emits the
           cleaned daily stock panel).
        3) NYSE size median (June P50 for FIN, prior-month P50 for PEAD).
        4) FIN and PEAD monthly factors + per-stock chars, keeping monthly
           membership for the daily build.
        5) If "daily" in freqs, value-weight the memberships on daily returns.

    Output (monthly factors + chars always written, daily only when requested):
        - <monthly_factors_path>: [excntry, eom, fin_dhs, pead_dhs]  (decimal)
        - <daily_factors_path>:   [excntry, date, fin_dhs, pead_dhs]  (decimal)
        - <chars_path>: [id, eom, ns_dhs, ir_dhs, abr_dhs]; id = permno (Int64),
              ns/ir the log measures, abr_dhs the decimal abnormal return.
    """
    raw_dir, interim_dir = paths.raw_tables_dir, paths.interim_dir

    # collect once: msf is reused by the monthly panel + the IR build (has a dedupe)
    msf = _dhs_load_msf(raw_dir, beg, end).collect()
    monthly_panel = _dhs_monthly_panel(msf, beg, end)
    annual_chars = _dhs_issuance_chars(msf, raw_dir, beg, end)
    # FIN goes international: append the ROW panel + chars when the ROW inputs are present.
    # Read world_data_prelim (not the final world_data.parquet): gen_dhs_data runs inside
    # generate_factor_models, before merge_qmj_to_world_data writes world_data.parquet, so
    # the final file does not yet exist here. The prelim carries every column the ROW path
    # needs and the downstream QMJ merge never touches them (same pattern as
    # gen_mispricing_data / _mp_world_load_world_data). US byte-identity still holds —
    # per-country breakpoints/size/VW keep the US self-contained.
    row = (interim_dir / "world_data_prelim.parquet").exists()
    if row:
        # require every ROW input up front so a run can't write partial outputs
        # (monthly with ROW, then crash before the daily leg)
        needed = ["world_data_prelim.parquet", "world_msf.parquet"] + (
            ["world_dsf.parquet"] if "daily" in freqs else []
        )
        missing = [f for f in needed if not (interim_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"ROW inputs missing from {interim_dir}: {missing}")
        row_panel = _dhs_load_row_panel(interim_dir, beg, end)
        # ROW ids and US permnos share the id column; they are disjoint by construction
        # (US CRSP permno is 5-digit; jkp ROW id is a 9+ digit BIGINT). Assert it so a
        # future id-scheme change can't silently cross-contaminate the id-keyed joins.
        if row_panel.height and row_panel["id"].min() <= monthly_panel["id"].max():
            raise ValueError(
                "ROW id overlaps the US permno range; id-keyed joins would cross countries"
            )
        # vertical_relaxed: siccd is Int32 (US CRSP) vs Int64 (ROW) on real data;
        # it only feeds the financial screen (never the output), so the supertype
        # cast is byte-safe. All output-feeding columns already share dtypes.
        monthly_panel = pl.concat([monthly_panel, row_panel], how="vertical_relaxed")
        annual_chars = pl.concat(
            [
                annual_chars.select("id", "datadate", "YEAR", "lagBE", "NS", "IR"),
                _dhs_row_chars(interim_dir, beg, end),
            ],
            how="diagonal_relaxed",
        )
    abr_panel, daily_stock = _dhs_announcement_returns(
        monthly_panel, raw_dir, interim_dir, beg, end, beg_q, end_q
    )
    # International PEAD: append the ROW announcement panel (IBES anndats bridged via
    # ibtic->gvkey->id, Abr on world_dsf) when the ex-US daily + IBES inputs are
    # present; without them ROW pead_dhs stays null (FIN-international is unaffected).
    if (
        row
        and (interim_dir / "world_dsf.parquet").exists()
        and (raw_dir / "ibes_actu_epsint.parquet").exists()
    ):
        abr_panel = pl.concat(
            [
                abr_panel,
                _dhs_row_announcement_returns(monthly_panel, raw_dir, interim_dir, beg, end),
            ],
            how="vertical_relaxed",
        )
    nyse_p50 = _dhs_nyse_size_median(raw_dir, interim_dir, min_stocks_bp)
    fin, fin_chars, fin_members = _dhs_fin_factor(
        annual_chars, monthly_panel, nyse_p50, min_stocks_bp, min_stocks_pf
    )
    pead, pead_chars, pead_members = _dhs_pead_factor(
        abr_panel, nyse_p50, min_stocks_bp, min_stocks_pf
    )

    # monthly factors + per-stock chars always written; daily only when requested
    _dhs_write_factors(
        fin.join(pead, on=["excntry", "eom"], how="full", coalesce=True).sort(["excntry", "eom"]),
        "eom",
        monthly_factors_path,
    )
    (
        fin_chars.join(pead_chars, on=["id", "eom"], how="full", coalesce=True)
        .select(
            "id",
            "eom",
            pl.col("ns").alias("ns_dhs"),
            pl.col("ir").alias("ir_dhs"),
            pl.col("abr").alias("abr_dhs"),
        )
        .sort(["id", "eom"])
        .write_parquet(chars_path)
    )
    if "daily" in freqs:
        row_daily = _dhs_load_row_daily(interim_dir) if row else None
        _dhs_write_factors(
            _dhs_daily_factors(fin_members, pead_members, daily_stock, row_daily, min_stocks_pf),
            "date",
            daily_factors_path,
        )
