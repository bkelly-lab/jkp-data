import datetime
import os
import time
from datetime import date
from math import exp, sqrt

import duckdb
import ibis
import polars as pl
import polars_ds as pds
import polars_ols  # noqa: F401 - required for least_squares method on polars expressions
from ibis import _
from polars import col


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
def setup_folder_structure():
    """
    Description:
        Create the project’s folder structure if missing.

    Steps:
        1) Make directories: raw_tables, raw_data_dfs, characteristics, return_data, accounting_data, other_output.

    Output:
        Folders created on disk (no return value).
    """
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "data/interim"))
    os.system(
        "mkdir -p raw_data_dfs ../raw/raw_tables ../processed/characteristics ../processed/return_data ../processed/accounting_data ../processed/other_output"
    )
    os.system("mkdir -p ../processed/return_data/daily_rets_by_country")


def collect_and_write(df, filename, collect_streaming=False):
    """
    Description:
        Collect a Polars LazyFrame (optionally streaming) and write to Parquet.

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


def gen_firmshares():
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
    con.create_table("comp_fundq", con.read_parquet("../raw/raw_tables/comp_fundq.parquet"))
    con.create_table(
        "comp_funda",
        con.read_parquet("../raw/raw_tables/comp_funda.parquet").rename({"at_": "at"}),
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
    con.table("__firm_shares1").to_parquet("raw_data_dfs/__firm_shares1.parquet")
    con.disconnect()


def gen_prihist_files():
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
        con.read_parquet("../raw/raw_tables/comp_sec_history.parquet"),
    )
    con.create_table(
        "comp_g_sec_history",
        con.read_parquet("../raw/raw_tables/comp_g_sec_history.parquet"),
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

    con.table("__prihistrow").to_parquet("raw_data_dfs/__prihistrow.parquet")
    con.table("__prihistusa").to_parquet("raw_data_dfs/__prihistusa.parquet")
    con.table("__prihistcan").to_parquet("raw_data_dfs/__prihistcan.parquet")
    con.disconnect()


def gen_fx1():
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
    con.create_table("comp_exrt_dly", con.read_parquet("../raw/raw_tables/comp_exrt_dly.parquet"))
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
    con.table("__fx1").to_parquet("raw_data_dfs/__fx1.parquet")
    con.disconnect()


@measure_time
def gen_raw_data_dfs():
    """
    Description:
        Generate a suite of “raw data” helper Parquet files from Compustat/CRSP sources.

    Steps:
        1) Call gen_firmshares(), gen_prihist_files(), gen_fx1().
        2) Derive SIC/NAICS (NA & Global), GICS (NA & Global), delist files (CRSP m/d),
        security info (NA & Global), T-bill return, FF monthly factors, exchange code map,
        exchange→country map, company headers (NA+Global), and CRSP security files (m & d).
        3) Standardize types/columns, sort/deduplicate where needed.
        4) Write all to raw_data_dfs/*.parquet.

    Output:
        Multiple helper Parquets under raw_data_dfs/ used in later pipelines.
    """
    gen_firmshares()
    sic_naics_na = sic_naics_aux("../raw/raw_tables/comp_funda.parquet")
    collect_and_write(sic_naics_na, "raw_data_dfs/sic_naics_na.parquet")
    sic_naics_gl = sic_naics_aux("../raw/raw_tables/comp_g_funda.parquet")
    collect_and_write(sic_naics_gl, "raw_data_dfs/sic_naics_gl.parquet")
    permno0 = (
        pl.scan_parquet("../raw/raw_tables/crsp_stksecurityinfohist.parquet")
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
    collect_and_write(permno0, "raw_data_dfs/permno0.parquet")
    comp_hgics_na = comp_hgics_aux("../raw/raw_tables/comp_co_hgic.parquet")
    collect_and_write(comp_hgics_na, "raw_data_dfs/comp_hgics_na.parquet")
    comp_hgics_gl = comp_hgics_aux("../raw/raw_tables/comp_g_co_hgic.parquet")
    collect_and_write(comp_hgics_gl, "raw_data_dfs/comp_hgics_gl.parquet")
    crsp_dsedelist = pl.scan_parquet("../raw/raw_tables/crsp_stkdelists.parquet").select(
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
    collect_and_write(crsp_dsedelist, "raw_data_dfs/crsp_dsedelist.parquet")
    crsp_msedelist = pl.scan_parquet("../raw/raw_tables/crsp_stkdelists.parquet").select(
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
    collect_and_write(crsp_msedelist, "raw_data_dfs/crsp_msedelist.parquet")
    __sec_info = pl.concat(
        [
            sec_info_aux("../raw/raw_tables/comp_security.parquet"),
            sec_info_aux("../raw/raw_tables/comp_g_security.parquet"),
        ]
    )
    collect_and_write(__sec_info, "raw_data_dfs/__sec_info.parquet")
    crsp_mcti_t30ret = pl.scan_parquet("../raw/raw_tables/crsp_mcti.parquet").select(
        ["caldt", "t30ret"]
    )
    collect_and_write(crsp_mcti_t30ret, "raw_data_dfs/crsp_mcti_t30ret.parquet")
    ff_factors_monthly = pl.scan_parquet("../raw/raw_tables/ff_factors_monthly.parquet").select(
        ["date", "rf"]
    )
    collect_and_write(ff_factors_monthly, "raw_data_dfs/ff_factors_monthly.parquet")
    comp_r_ex_codes = pl.scan_parquet("../raw/raw_tables/comp_r_ex_codes.parquet").select(
        ["exchgdesc", "exchgcd"]
    )
    collect_and_write(comp_r_ex_codes, "raw_data_dfs/comp_r_ex_codes.parquet")
    __ex_country1 = pl.concat(
        [
            ex_country_aux("../raw/raw_tables/comp_g_security.parquet"),
            ex_country_aux("../raw/raw_tables/comp_security.parquet"),
        ]
    )
    collect_and_write(__ex_country1, "raw_data_dfs/__ex_country1.parquet")
    header_aux(
        "../raw/raw_tables/comp_company.parquet",
        "../raw/raw_tables/comp_g_company.parquet",
        "raw_data_dfs/__header.parquet",
    )
    gen_prihist_files()
    gen_fx1()
    gen_crsp_sf("m").to_parquet("raw_data_dfs/__crsp_sf_m.parquet")
    gen_crsp_sf("d").to_parquet("raw_data_dfs/__crsp_sf_d.parquet")


def gen_crsp_sf(freq):
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
    sf = con.read_parquet(f"../raw/raw_tables/crsp_{freq}sf_v2.parquet")
    senames = con.read_parquet("../raw/raw_tables/crsp_stksecurityinfohist.parquet")
    ccmxpf_lnkhist = con.read_parquet("../raw/raw_tables/crsp_ccmxpf_lnkhist.parquet")

    # ---------- CIZ name mapping by frequency ----------
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
    else:
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
    usincflg_expr = sf.usincflg
    issuertype_expr = sf.issuertype

    is_common_expr = (
        (securitytype_expr == "EQTY")
        & (securitysubtype_expr == "COM")
        & (sharetype_expr == "NS")
        & (issuertype_expr.isin(["ACOR", "CORP"]))
        & (usincflg_expr == "Y")
    )

    shrcd_expr = ibis.cases(
        (is_common_expr, 10),
        else_=ibis.null(),
    ).cast("int32")

    primaryexch_expr = sf.primaryexch
    conditionaltype_expr = sf.conditionaltype

    exch_main_expr = (primaryexch_expr.isin(["A", "N", "Q"]) & (conditionaltype_expr == "RW")).cast(
        "int32"
    )

    exchcd_expr = ibis.cases(
        ((primaryexch_expr == "N") & (conditionaltype_expr == "RW"), 1),
        ((primaryexch_expr == "A") & (conditionaltype_expr == "RW"), 2),
        ((primaryexch_expr == "Q") & (conditionaltype_expr == "RW"), 3),
        else_=ibis.null(),
    ).cast("int32")

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
        exchcd=exchcd_expr,
        exch_main=exch_main_expr,
        shrcd=shrcd_expr,
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
            "exchcd",
            "gvkey",
            "iid",
            "exch_main",
            "shrcd",
            "me",
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


def download_wrds_table(conninfo, duckdb_conn, table_name, filename):
    lib, table = table_name.split(".")
    cols = get_columns(duckdb_conn, conninfo, lib, table)
    projection = build_projection(cols)

    duckdb_conn.execute(f"""
        COPY (
          SELECT {projection}
          FROM postgres_scan('{conninfo}', '{lib}', '{table}')
        )
        TO '{filename}' (FORMAT PARQUET);
    """)


@measure_time
def download_raw_data_tables(username, password):
    """
    Description:
        Bulk-download core WRDS tables to raw_tables and a few curated variants with column subsets.

    Steps:
        1) Connect to WRDS; iterate through a fixed list of library.tables.
        2) For each table: download to raw_tables/lib_table.parquet; periodically reset connection.
        3) Additionally fetch comp.secd and comp.g_secd with curated columns.
        4) Disconnect.

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
        "comp.g_funda",
        "comp.co_hgic",
        "comp.g_fundq",
        "comp.secd",
        "comp.g_secd",
    ]

    wrds_session_data = gen_wrds_connection_info(username, password)
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")

    for table in table_names:
        # print(f"Downloading WRDS table: {table}", flush=True)
        download_wrds_table(
            wrds_session_data,
            con,
            table,
            "../raw/raw_tables/" + table.replace(".", "_") + ".parquet",
        )

    con.close()


def aug_msf_v2():
    """
    Description:
        Add month-level high/low transaction-price fields to the CRSP CIZ monthly file (msf_v2)
        using the CRSP CIZ daily file (dsf_v2). Keep all msf_v2 rows; set the new fields to
        missing for non-TR monthly rows (e.g., BA).

    Steps:
        1) Read msf_v2 (monthly) and dsf_v2 (daily) from parquet.
        2) Filter dsf_v2 to dlyprcflg == "TR", construct yyyymm from dlycaldt, and keep (permno, yyyymm, dlyprc).
        3) For each (permno, yyyymm), compute:
           - mthaskhi = max(dlyprc)
           - mthbidlo = min(dlyprc)
        4) Left-join these two fields onto msf_v2 by (permno, yyyymm).
        5) Set mthaskhi/mthbidlo to NULL when mthprcflg != "TR".
        6) Overwrite crsp_msf_v2.parquet (via a temp file).

    Output:
        Overwrites ../raw/raw_tables/crsp_msf_v2.parquet with two new columns:
        mthaskhi and mthbidlo.
    """
    con = ibis.duckdb.connect(threads=os.cpu_count())
    msf = con.read_parquet("../raw/raw_tables/crsp_msf_v2.parquet")
    dsf = con.read_parquet("../raw/raw_tables/crsp_dsf_v2.parquet")

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

    msf_aug.to_parquet("../raw/raw_tables/crsp_msf_v2.parquet.tmp")
    os.replace("../raw/raw_tables/crsp_msf_v2.parquet.tmp", "../raw/raw_tables/crsp_msf_v2.parquet")


def build_mcti():
    """
    Description:
        Build monthly t30 return raw data.

    Steps:
        1) Read indmthseriesdata and indseriesinfohdr from parquet.
        2) Inner join indmthseriesdata with indseriesinfohdr on "indno".
        4) Filter for CRSP 30-Year Treasury Returns (indno == 1000708).
        5) Write parquet to ../raw/raw_tables/crsp_mcti.parquet

    Output:
        Polars DataFrame (also written to parquet).
    """

    a = pl.read_parquet("../raw/raw_tables/crsp_indmthseriesdata_ind.parquet")
    b = pl.read_parquet("../raw/raw_tables/crsp_indseriesinfohdr_ind.parquet")

    ab = a.join(b, on="indno", how="inner")

    out = (
        ab.filter(pl.col("indno") == 1000708)
        .select(["indno", "indnm", "mthcaldt", "mthtotret", "mthtotind"])
        .rename({"mthcaldt": "caldt", "mthtotret": "t30ret"})
    )

    os.makedirs("../raw/raw_tables", exist_ok=True)
    out.write_parquet("../raw/raw_tables/crsp_mcti.parquet")


@measure_time
def prepare_comp_sf(freq):
    """
    Description:
        Prepare Compustat security-file derivatives (Comp DSF/SSF equivalents) for daily/monthly runs.

    Steps:
        1) Ensure firm-shares table is populated (populate_own), then create Comp DSF (gen_comp_dsf).
        2) Run process_comp_sf1 for requested frequency: 'd', 'm', or 'both'.

    Output:
        Intermediate Comp security files written by downstream helpers (no direct return).
    """
    populate_own("raw_data_dfs/__firm_shares1.parquet", "gvkey", "datadate", "ddate")
    gen_comp_dsf()
    if freq == "both":
        process_comp_sf1("d")
        process_comp_sf1("m")
    else:
        process_comp_sf1(freq)


def populate_own(inset_path, idvar, datevar, datename):
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
    inset.collect().write_parquet("__firm_shares2.parquet")


def compustat_fx():
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
    __fx1 = pl.scan_parquet("raw_data_dfs/__fx1.parquet")
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


def adj_trd_vol_NASDAQ(datevar, col_to_adjust, exchg_var, exchg_val):
    """
    Description:
        Apply historic NASDAQ trade-volume adjustments (pre-decimalization reporting) to a volume column.

    Steps:
        1) Build date cutoffs: <2001-02-01, ≤2001-12-31, <2003-12-31.
        2) If exchg_var == exchg_val (NASDAQ) and within windows, scale col_to_adjust by
        1/2, 1/1.8, or 1/1.6 respectively; otherwise keep original.
        3) Return the adjusted expression aliased as the original column name.

    Output:
        Polars expression that yields adjusted trade volume for NASDAQ histories.
    """
    c1 = col(exchg_var) == exchg_val
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


def gen_comp_dsf():
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
    os.system("rm -f aux_comp_dsf.ddb")
    con = ibis.duckdb.connect("aux_comp_dsf.ddb", threads=os.cpu_count())

    compustat_fx().write_parquet("fx_data.parquet")
    con.create_table("comp_g_secd", con.read_parquet("../raw/raw_tables/comp_g_secd.parquet"))
    con.create_table("__firm_shares2", con.read_parquet("__firm_shares2.parquet"))
    con.create_table("comp_secd", con.read_parquet("../raw/raw_tables/comp_secd.parquet"))
    con.create_table("fx", con.read_parquet("fx_data.parquet"))

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
    t.to_parquet("__comp_dsf.parquet")
    con.disconnect()


def gen_secd_data():
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
    os.system("rm -f aux_msf.ddb")
    con = ibis.duckdb.connect("aux_msf.ddb", threads=os.cpu_count())
    table = con.read_parquet("__comp_dsf.parquet")
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
    new_table.to_parquet("secd_data.parquet")
    con.disconnect()


def gen_secm_data():
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
    os.system("rm -f aux_comp_secm.ddb")
    con = ibis.duckdb.connect("aux_comp_secm.ddb", threads=os.cpu_count())

    compustat_fx().rename({"datadate": "date"}).write_parquet("fx_data.parquet")
    con.create_table(
        "comp_secm",
        con.read_parquet("../raw/raw_tables/comp_secm.parquet"),
        overwrite=True,
    )
    con.create_table("__firm_shares2", con.read_parquet("__firm_shares2.parquet"), overwrite=True)
    con.create_table("fx", con.read_parquet("fx_data.parquet"), overwrite=True)

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
    con.table("__comp_secm2").to_parquet("secm_data.parquet")


def gen_comp_msf():
    """
    Description:
        Merge SECD-based and SECM-based monthly datasets into a single Compustat MSF-like file.
        Prefer SECD (source=1) when both sources exist for the same {gvkey,iid,eom}.

    Steps:
        1) Ensure secd_data.parquet and secm_data.parquet exist (run gen_secd_data/gen_secm_data).
        2) Load both; select a common column set and cast types consistently.
        3) UNION the two sources; for each {gvkey,iid,eom} window count n rows.
        4) Keep either the single row (n=1) or, if n=2, prefer source=1 (SECD).
        5) Drop helper columns and deduplicate on {gvkey, iid, eom}.
        6) Write to __comp_msf.parquet.

    Output:
        Parquet: __comp_msf.parquet (monthly Compustat security master in USD).
    """
    gen_secd_data()
    gen_secm_data()
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
    os.system("rm -f aux_msf.ddb")
    con = ibis.duckdb.connect("aux_msf.ddb", threads=os.cpu_count())
    secd = (
        con.read_parquet("secd_data.parquet")
        .select(common_vars)
        .cast({"ajexdi": "float", "prc_local": "float"})
    )
    secm = (
        con.read_parquet("secm_data.parquet")
        .select(common_vars)
        .cast({"ajexdi": "float", "prc_local": "float"})
    )
    window = ibis.window(group_by=["gvkey", "iid", "eom"], order_by="datadate")
    __comp_msf = (
        secd.union(secm)
        .mutate(n=_.gvkey.count().over(window))
        .filter([(_.n == 1) | ((_.n == 2) & (_.source == 1))])
        .drop(["n", "source"])
        .distinct(on=["gvkey", "iid", "eom"])
    )
    __comp_msf.to_parquet("__comp_msf.parquet")
    con.disconnect()


def comp_exchanges():
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
    comp_r_ex_codes = pl.read_parquet("raw_data_dfs/comp_r_ex_codes.parquet")
    __ex_country = pl.read_parquet("raw_data_dfs/__ex_country1.parquet")
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


def add_primary_sec(data_path, datevar, file_name):
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
        5) Clean columns, cast to int, deduplicate on (gvkey,iid,datadate), sort, and write file_name.

    Output:
        Parquet at file_name with a new integer column primary_sec ∈ {0,1}.
    """
    os.system("rm -f aux_prim_sec.ddb")
    con = duckdb.connect("aux_prim_sec.ddb")
    con.execute(f"""
        CREATE OR REPLACE TABLE __data2 AS
        WITH data AS (
            SELECT * FROM read_parquet('{data_path}')
            ORDER BY gvkey, iid, datadate
        ),
        prihistrow AS (
            SELECT * FROM read_parquet('raw_data_dfs/__prihistrow.parquet')
            ORDER BY gvkey, effdate
        ),
        prihistusa AS (
            SELECT * FROM read_parquet('raw_data_dfs/__prihistusa.parquet')
            ORDER BY gvkey, effdate
        ),
        prihistcan AS (
            SELECT * FROM read_parquet('raw_data_dfs/__prihistcan.parquet')
            ORDER BY gvkey, effdate
        ),
        header AS (
            SELECT * FROM read_parquet('raw_data_dfs/__header.parquet')
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

        COPY
            (SELECT * FROM __data2)
        TO '{file_name}' (FORMAT parquet);
    """)

    con.close()
    os.system("rm -f aux_prim_sec.ddb")


def load_rf_and_exchange_data():
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
        pl.read_parquet("raw_data_dfs/crsp_mcti_t30ret.parquet")
        .with_columns(merge_aux=gen_MMYY_column("caldt"))
        .drop("caldt")
    )
    ff_factors_monthly = (
        pl.read_parquet("raw_data_dfs/ff_factors_monthly.parquet")
        .with_columns(merge_aux=gen_MMYY_column("date"))
        .drop("date")
    )
    __exchanges = comp_exchanges()
    return crsp_mcti, ff_factors_monthly, __exchanges


def gen_returns_df(freq):
    """
    Description:
        Compute returns (USD and local), return-lag gaps, and currency-switch fixes
        from Compustat security files (__comp_?sf.parquet).

    Steps:
        1) Determine ret_lag_dif: monthly via MMYY difference; daily via day difference.
        2) Filter valid rows (ri not null; prcstd in {3,4,10}); unique and sort.
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

    base = pl.scan_parquet(f"__comp_{freq}sf.parquet")
    __returns = (
        base.filter((col("ri").is_not_null()) & (col("prcstd").is_in([3, 4, 10])))
        .unique(["gvkey", "iid", "datadate"])
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


def gen_delist_df(__returns):
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
    __sec_info = pl.scan_parquet("raw_data_dfs/__sec_info.parquet")
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


def gen_temporary_sf(freq, __returns, __delist):
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
    base = pl.read_parquet(f"__comp_{freq}sf.parquet")
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


def add_rf_and_exchange_data_to_temporary_sf(freq, temp_sf):
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
    crsp_mcti, ff_factors_monthly, __exchanges = load_rf_and_exchange_data()
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


def process_comp_sf1(freq):
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
        gen_comp_msf()
    __returns = gen_returns_df(freq)
    __delist = gen_delist_df(__returns)
    __comp_sf2 = gen_temporary_sf(freq, __returns, __delist)
    __comp_sf2 = add_rf_and_exchange_data_to_temporary_sf(freq, __comp_sf2)
    __comp_sf2.write_parquet("__comp_sf2.parquet")
    del __comp_sf2
    add_primary_sec("__comp_sf2.parquet", "datadate", f"comp_{freq}sf.parquet")


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
def prepare_crsp_sf(freq):
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
        pl.scan_parquet(f"raw_data_dfs/__crsp_sf_{freq}.parquet")
        .with_columns(
            [
                col(var).cast(pl.Float64)
                for var in ["prc", "cfacshr", "ret", "retx", "prc_high", "prc_low"]
            ]
            + [col("vol").cast(pl.Int64)]
        )
        .with_columns(adj_trd_vol_NASDAQ("date", "vol", "exchcd", 3))
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

    crsp_sedelist = pl.scan_parquet(f"raw_data_dfs/crsp_{freq}sedelist.parquet").with_columns(
        crsp_sedelist_aux_col
    )

    crsp_mcti = add_MMYY_column_drop_original(
        pl.scan_parquet("raw_data_dfs/crsp_mcti_t30ret.parquet"), "caldt"
    )
    ff_factors_monthly = add_MMYY_column_drop_original(
        pl.scan_parquet("raw_data_dfs/ff_factors_monthly.parquet"), "date"
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

    __crsp_sf.collect().write_parquet(f"crsp_{freq}sf.parquet")


def prepare_crsp_sfs_for_merging():
    """
    Description:
        Normalize CRSP monthly/daily frames for concatenation with Compustat, adding
        shared schema fields and flags.

    Steps:
        1) For crsp_msf: set USA defaults (USD, fx=1), common/primary flags, map columns
        to shared names, compute eom, ret_lag_dif=1, source_crsp=1.
        2) For crsp_dsf: similar normalization (eom, flags, USD), rename to shared schema.

    Output:
        Tuple (crsp_msf LazyFrame, crsp_dsf LazyFrame) ready for merging.
    """
    crsp_msf = (
        pl.scan_parquet("crsp_msf.parquet")
        .with_columns(
            exch_main=col("exch_main").cast(pl.Int32),
            bidask=col("bidask").cast(pl.Int32),
            id=col("permno"),
            excntry=pl.lit("USA"),
            common=(col("shrcd").is_in([10, 11, 12]).fill_null(bo_false())).cast(pl.Int32),
            primary_sec=pl.lit(1),
            comp_tpci=pl.lit(None).cast(pl.Utf8),
            comp_exchg=pl.lit(None).cast(pl.Int64),
            curcd=pl.lit("USD"),
            fx=pl.lit(1.0),
            eom=col("date").dt.month_end(),
            prc_local=col("prc"),
            tvol=col("vol"),
            ret_local=col("ret"),
            ret_lag_dif=pl.lit(1).cast(pl.Int64),
            div_cash=fl_none(),
            div_spc=fl_none(),
            source_crsp=pl.lit(1),
        )
        .rename(
            {
                "shrcd": "crsp_shrcd",
                "exchcd": "crsp_exchcd",
                "cfacshr": "adjfct",
                "shrout": "shares",
            }
        )
    )

    crsp_dsf = (
        pl.scan_parquet("crsp_dsf.parquet")
        .with_columns(
            id=col("permno"),
            excntry=pl.lit("USA"),
            common=(col("shrcd").is_in([10, 11, 12]).fill_null(bo_false())).cast(pl.Int32),
            primary_sec=pl.lit(1),
            curcd=pl.lit("USD"),
            fx=pl.lit(1.0),
            eom=col("date").dt.month_end(),
            ret_local=col("ret"),
            ret_lag_dif=pl.lit(1).cast(pl.Int64),
            exch_main=col("exch_main").cast(pl.Int32),
            bidask=col("bidask").cast(pl.Int32),
            source_crsp=pl.lit(1),
        )
        .rename({"cfacshr": "adjfct", "shrout": "shares", "vol": "tvol"})
    )
    return crsp_msf, crsp_dsf


def prepare_comp_sfs_for_merging():
    """
    Description:
        Normalize Compustat monthly/daily frames to match CRSP schema (ids, flags, names).

    Steps:
        1) Build integer id from gvkey+iid with instrument-type prefix (1 common, 2 ADR, 3 when-Issued).
        2) For comp_msf: map/rename fields; set flags (common/bidask), me_company=me, source_crsp=0.
        3) For comp_dsf: map/rename; compute eom from datadate; set source_crsp=0.

    Output:
        Tuple (comp_msf LazyFrame, comp_dsf LazyFrame) ready for concatenation.
    """
    id_exp = (
        pl.when(col("iid").str.contains("W"))
        .then(pl.lit("3") + col("gvkey") + col("iid").str.slice(0, 2))
        .when(col("iid").str.contains("C"))
        .then(pl.lit("2") + col("gvkey") + col("iid").str.slice(0, 2))
        .otherwise(pl.lit("1") + col("gvkey") + col("iid").str.slice(0, 2))
    ).cast(pl.Int64)

    comp_msf = (
        pl.scan_parquet("comp_msf.parquet")
        .with_columns(
            id=id_exp,
            permno=pl.lit(None).cast(pl.Int64),
            permco=pl.lit(None).cast(pl.Int64),
            common=pl.when(col("tpci") == "0").then(pl.lit(1)).otherwise(pl.lit(0)),
            bidask=pl.when(col("prcstd") == 4).then(pl.lit(1)).otherwise(pl.lit(0)),
            crsp_shrcd=fl_none(),
            crsp_exchcd=fl_none(),
            me_company=col("me"),
            source_crsp=pl.lit(0),
            ret_lag_dif=col("ret_lag_dif").cast(pl.Int64),
        )
        .rename(
            {
                "tpci": "comp_tpci",
                "exchg": "comp_exchg",
                "curcdd": "curcd",
                "datadate": "date",
                "ajexdi": "adjfct",
                "cshoc": "shares",
                "cshtrm": "tvol",
            }
        )
    )

    comp_dsf = (
        pl.scan_parquet("comp_dsf.parquet")
        .with_columns(
            id=id_exp,
            common=pl.when(col("tpci") == "0").then(pl.lit(1)).otherwise(pl.lit(0)),
            bidask=pl.when(col("prcstd") == 4).then(pl.lit(1)).otherwise(pl.lit(0)),
            eom=col("datadate").dt.month_end(),
            source_crsp=pl.lit(0),
        )
        .rename(
            {
                "curcdd": "curcd",
                "datadate": "date",
                "ajexdi": "adjfct",
                "cshoc": "shares",
                "cshtrd": "tvol",
            }
        )
    )
    return comp_msf, comp_dsf


def gen_temp_sf(freq, crsp_df, comp_df):
    """
    Description:
        Vertically merge CRSP and Compustat security files to a unified world file
        with a harmonized column set; optionally compute lead ret_exc for monthly.

    Steps:
        1) Define cols_to_keep per frequency; concat CRSP/Compustat frames vertically.
        2) If monthly: compute ret_exc_lead1m where next observation is contiguous.

    Output:
        LazyFrame unified security file (__msf_world or world_dsf before filtering).
    """
    cols_to_keep = (
        [
            "id",
            "permno",
            "permco",
            "gvkey",
            "iid",
            "excntry",
            "exch_main",
            "common",
            "primary_sec",
            "bidask",
            "crsp_shrcd",
            "crsp_exchcd",
            "comp_tpci",
            "comp_exchg",
            "curcd",
            "fx",
            "date",
            "eom",
            "adjfct",
            "shares",
            "me",
            "me_company",
            "prc",
            "prc_local",
            "prc_high",
            "prc_low",
            "dolvol",
            "tvol",
            "ret",
            "ret_local",
            "ret_exc",
            "ret_lag_dif",
            "div_tot",
            "div_cash",
            "div_spc",
            "source_crsp",
        ]
        if freq == "m"
        else [
            "id",
            "excntry",
            "exch_main",
            "common",
            "primary_sec",
            "bidask",
            "curcd",
            "fx",
            "date",
            "eom",
            "adjfct",
            "shares",
            "me",
            "dolvol",
            "tvol",
            "prc",
            "prc_high",
            "prc_low",
            "ret_local",
            "ret",
            "ret_exc",
            "ret_lag_dif",
            "source_crsp",
        ]
    )
    sf_world = pl.concat(
        [crsp_df.select(cols_to_keep), comp_df.select(cols_to_keep)],
        how="vertical_relaxed",
    )
    if freq == "m":
        sf_world = sf_world.sort(["id", "eom"]).with_columns(
            ret_exc_lead1m=pl.when(col("ret_lag_dif").shift(-1).over("id") != 1)
            .then(None)
            .otherwise(col("ret_exc").shift(-1).over("id"))
        )
    return sf_world


def add_obs_main_to_sf_and_write_file(freq, sf_df, obs_main):
    """
    Description:
        Add an observation selection flag (obs_main) and write the final world file.

    Steps:
        1) Left-join obs_main; unique and sort by {id,eom} or {id,date}.
        2) Collect (streaming) and save to '__msf_world.parquet' (monthly) or 'world_dsf.parquet' (daily).

    Output:
        Parquet file with obs_main indicating preferred monthly observation per id-date.
    """
    file_path = "__msf_world.parquet" if freq == "m" else "world_dsf.parquet"
    sort_vars = ["id", "eom"] if freq == "m" else ["id", "date"]
    (
        sf_df.join(obs_main, on=["id", "eom"], how="left")
        .unique(sort_vars)
        .sort(sort_vars)
        .collect(streaming=True)
        .write_parquet(file_path)
    )


@measure_time
def combine_crsp_comp_sf():
    """
    Description:
        Create unified monthly and daily security datasets by combining CRSP and Compustat,
        determining the main observation per id/eom, and writing outputs.

    Steps:
        1) Prepare normalized CRSP and Compustat frames (monthly & daily).
        2) Build __msf_world and __dsf_world via gen_temp_sf.
        3) From __msf_world derive obs_main: if only one obs or multiple but source_crsp==1 → 1 else 0.
        4) Write final monthly and daily world files, injecting obs_main where applicable.

    Output:
        '__msf_world.parquet' and 'world_dsf.parquet' ready for downstream processing.
    """
    crsp_msf, crsp_dsf = prepare_crsp_sfs_for_merging()
    comp_msf, comp_dsf = prepare_comp_sfs_for_merging()
    __msf_world = gen_temp_sf("m", crsp_msf, comp_msf)
    __dsf_world = gen_temp_sf("d", crsp_dsf, comp_dsf)
    obs_main = (
        __msf_world.select(["id", "source_crsp", "gvkey", "iid", "eom"])
        .with_columns(count=pl.count("gvkey").over(["gvkey", "iid", "eom"]))
        .with_columns(
            obs_main=pl.when(
                (col("count").is_in([0, 1])) | ((col("count") > 1) & (col("source_crsp") == 1))
            )
            .then(1)
            .otherwise(0)
        )
        .drop(["count", "iid", "gvkey", "source_crsp"])
    )
    add_obs_main_to_sf_and_write_file("m", __msf_world, obs_main)
    add_obs_main_to_sf_and_write_file("d", __dsf_world, obs_main)


@measure_time
def crsp_industry():
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
    permno0 = pl.scan_parquet("raw_data_dfs/permno0.parquet")
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
    permno0.collect().write_parquet("crsp_ind.parquet")


def comp_hgics(lib):
    """
    Description:
        Expand Compustat GICS history (national/global) to a daily panel with forward-filled
        terminal dates and write separate outputs.

    Steps:
        1) Load raw file (national/global); replace null gics with -999 sentinel.
        2) Compute row counts and terminal rows; set open-ended indthru to (max(indfrom), or today).
        3) Create date ranges [indfrom, indthru]; explode; unique per (gvkey,date).
        4) Write to na_hgics.parquet or g_hgics.parquet.

    Output:
        Parquet with {gvkey,date,gics} expanded to daily frequency.
    """
    paths = {
        "raw data": {
            "national": "raw_data_dfs/comp_hgics_na.parquet",
            "global": "raw_data_dfs/comp_hgics_gl.parquet",
        },
        "output": {"national": "na_hgics.parquet", "global": "g_hgics.parquet"},
    }
    data = pl.read_parquet(paths["raw data"][lib])  # .sort(['gvkey', 'indfrom'])
    data = data.with_columns(
        gics=pl.when(col("gics").is_null()).then(-999).otherwise(col("gics")),
        n=pl.len().over("gvkey"),
        n_aux=pl.cum_count("gvkey").over("gvkey"),
    )
    indthru_date = (
        pl.lit(data[["indfrom"]].max()[0, 0])
        if data[["indfrom"]].max()[0, 0] > date.today()
        else pl.lit(date.today())
    )
    c1 = col("n") == col("n_aux")
    c2 = col("indthru").is_null()
    data = (
        data.with_columns(indthru=pl.when(c1 & c2).then(indthru_date).otherwise(col("indthru")))
        .select(["gvkey", pl.date_ranges("indfrom", "indthru").alias("date"), "gics"])
        .explode("date")
        .unique(subset=["gvkey", "date"])
        .sort(["gvkey", "date"])
    )
    data.write_parquet(paths["output"][lib])


def hgics_join():
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
    comp_hgics("global")
    comp_hgics("national")
    global_data = pl.scan_parquet("g_hgics.parquet")
    local_data = pl.scan_parquet("na_hgics.parquet")
    gjoin = local_data.join(global_data, on=["gvkey", "date"], how="full", coalesce=True)
    gjoin = (
        gjoin.select(["gvkey", "date", pl.coalesce(["gics", "gics_right"]).alias("gics")])
        .unique(["gvkey", "date"])
        .sort(["gvkey", "date"])
    )
    gjoin.collect().write_parquet("comp_hgics.parquet")


def comp_sic_naics():
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
    con.create_table("sic_naics_na", con.read_parquet("raw_data_dfs/sic_naics_na.parquet"))
    con.create_table("sic_naics_gl", con.read_parquet("raw_data_dfs/sic_naics_gl.parquet"))
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
    comp.collect().write_parquet("comp_other.parquet")
    con.disconnect()


@measure_time
def comp_industry():
    """
    Description:
        Merge daily GICS and SIC/NAICS into a single daily Compustat industry file,
        filling gaps day-by-day to ensure continuity.

    Steps:
        1) Run comp_sic_naics() and hgics_join(); load into DuckDB.
        2) Full-outer-join on (gvkey,date); compute aux_date = next date − 1 day to detect gaps.
        3) Build gap ranges via generate_series and fill from gap_dates; union with continuous rows.
        4) Select distinct first by (gvkey,date); write comp_ind.parquet.

    Output:
        Parquet comp_ind.parquet with {gvkey,date,gics,sic,naics} daily.
    """
    comp_sic_naics()
    hgics_join()
    os.system("rm -f aux_comp_ind.ddb")
    con = ibis.duckdb.connect("aux_comp_ind.ddb", threads=os.cpu_count())
    con.create_table("comp_other", con.read_parquet("comp_other.parquet"))
    con.create_table("comp_gics", con.read_parquet("comp_hgics.parquet"))
    con.raw_sql("""
                DROP TABLE IF EXISTS join_table;
                CREATE TABLE join_table AS
                SELECT          *,
                                COALESCE( LEAD(date) OVER (PARTITION BY gvkey ORDER BY date) - INTERVAL '1 day', date )::DATE AS aux_date
                FROM            comp_gics
                FULL OUTER JOIN comp_other
                USING           (gvkey, date);

                DROP TABLE IF EXISTS gap_dates;
                CREATE TABLE gap_dates AS
                SELECT *
                FROM join_table
                WHERE date <> aux_date;

                DROP TABLE IF EXISTS gaps;
                CREATE TABLE gaps AS
                WITH full_span AS (
                SELECT
                    j.gvkey, gs.gap_date::DATE AS date,
                    FROM gap_dates as j
                    CROSS JOIN LATERAL
                    generate_series(j.date, j.aux_date, INTERVAL '1 day') AS gs(gap_date)
                    ORDER BY gvkey, date
                )
                SELECT
                fs.gvkey, fs.date, gd.gics, gd.sic, gd.naics
                FROM full_span fs
                LEFT JOIN gap_dates gd
                ON gd.gvkey = fs.gvkey
                AND gd.date  = fs.date
                ORDER BY fs.gvkey, fs.date;

                DROP TABLE IF EXISTS continuous;
                CREATE TABLE continuous AS
                SELECT *
                FROM join_table
                WHERE date = aux_date;

                DROP TABLE IF EXISTS merged_data;
                CREATE TABLE merged_data AS
                SELECT gvkey, date, gics, sic, naics FROM continuous
                UNION
                SELECT gvkey, date, gics, sic, naics FROM gaps;

                DROP TABLE IF EXISTS comp_industry;
                CREATE TABLE comp_industry AS
                SELECT DISTINCT ON (gvkey, date)
                    *
                FROM merged_data
                ORDER BY (gvkey, date);
    """)
    con.table("comp_industry").to_parquet("comp_ind.parquet")
    con.disconnect()


@measure_time
def ff_ind_class(data_path, ff_grps):
    """
    Description:
        Assign Fama–French industry classifications (38-group or 49-group) based on SIC codes.

    Steps:
        1) If ff_grps==38: define lower/upper SIC bounds; iterate to map to ff38 groups (2..N),
        with special rule for (100–999) → 1; set null when no match.
        2) Else: build ff49 via explicit SIC enumerations/ranges per FF (Ken French) taxonomy.
        3) Attach the classification column to input data and write __msf_world3.parquet.

    Output:
        Parquet __msf_world3.parquet with an added ff38 or ff49 integer column.
    """
    data = pl.scan_parquet(data_path)
    if ff_grps == 38:
        lower_bounds = [
            1000,
            1300,
            1400,
            1500,
            2000,
            2100,
            2200,
            2300,
            2400,
            2500,
            2600,
            2700,
            2800,
            2900,
            3000,
            3100,
            3200,
            3300,
            3400,
            3500,
            3600,
            3700,
            3800,
            3900,
            4000,
            4800,
            4830,
            4900,
            4950,
            4960,
            4970,
            5000,
            5200,
            6000,
            7000,
            9000,
        ]
        upper_bounds = [
            1299,
            1399,
            1499,
            1799,
            2099,
            2199,
            2299,
            2399,
            2499,
            2599,
            2661,
            2799,
            2899,
            2999,
            3099,
            3199,
            3299,
            3399,
            3499,
            3599,
            3699,
            3799,
            3879,
            3999,
            4799,
            4829,
            4899,
            4949,
            4959,
            4969,
            4979,
            5199,
            5999,
            6999,
            8999,
            9999,
        ]
        classification = [i + 2 for i in range(len(lower_bounds))]
        ff38_exp = pl.when(col("sic").is_between(100, 999)).then(1)
        for i, j, k in zip(lower_bounds, upper_bounds, classification, strict=True):
            ff38_exp = ff38_exp.when(col("sic").is_between(i, j)).then(k)
        ff38_exp = (ff38_exp.otherwise(pl.lit(None))).alias("ff38")
        data = data.with_columns(ff38_exp)
    else:
        data = data.with_columns(
            pl.when(
                col("sic").is_in(
                    [
                        2048,
                        *range(100, 299 + 1),
                        *range(700, 799 + 1),
                        *range(910, 919 + 1),
                    ]
                )
            )
            .then(1)
            .when(
                col("sic").is_in(
                    [
                        2095,
                        2098,
                        2099,
                        *range(2000, 2046 + 1),
                        *range(2050, 2063 + 1),
                        *range(2070, 2079 + 1),
                        *range(2090, 2092 + 1),
                    ]
                )
            )
            .then(2)
            .when(col("sic").is_in([2086, 2087, 2096, 2097, *range(2064, 2068 + 1)]))
            .then(3)
            .when(col("sic").is_in([2080, *range(2082, 2085 + 1)]))
            .then(4)
            .when(col("sic").is_in([*range(2100, 2199 + 1)]))
            .then(5)
            .when(
                col("sic").is_in(
                    [
                        3732,
                        3930,
                        3931,
                        *range(920, 999 + 1),
                        *range(3650, 3652 + 1),
                        *range(3940, 3949 + 1),
                    ]
                )
            )
            .then(6)
            .when(
                col("sic").is_in(
                    [
                        7840,
                        7841,
                        7900,
                        7910,
                        7911,
                        7980,
                        *range(7800, 7833 + 1),
                        *range(7920, 7933 + 1),
                        *range(7940, 7949 + 1),
                        *range(7990, 7999 + 1),
                    ]
                )
            )
            .then(7)
            .when(col("sic").is_in([2770, 2771, *range(2700, 2749 + 1), *range(2780, 2799 + 1)]))
            .then(8)
            .when(
                col("sic").is_in(
                    [
                        2047,
                        2391,
                        2392,
                        3160,
                        3161,
                        3229,
                        3260,
                        3262,
                        3263,
                        3269,
                        3230,
                        3231,
                        3750,
                        3751,
                        3800,
                        3860,
                        3861,
                        3910,
                        3911,
                        3914,
                        3915,
                        3991,
                        3995,
                        *range(2510, 2519 + 1),
                        *range(2590, 2599 + 1),
                        *range(2840, 2844 + 1),
                        *range(3170, 3172 + 1),
                        *range(3190, 3199 + 1),
                        *range(3630, 3639 + 1),
                        *range(3870, 3873 + 1),
                        *range(3960, 3962 + 1),
                    ]
                )
            )
            .then(9)
            .when(
                col("sic").is_in(
                    [
                        3020,
                        3021,
                        3130,
                        3131,
                        3150,
                        3151,
                        *range(2300, 2390 + 1),
                        *range(3100, 3111 + 1),
                        *range(3140, 3149 + 1),
                        *range(3963, 3965 + 1),
                    ]
                )
            )
            .then(10)
            .when(col("sic").is_in([*range(8000, 8099 + 1)]))
            .then(11)
            .when(col("sic").is_in([3693, 3850, 3851, *range(3840, 3849 + 1)]))
            .then(12)
            .when(col("sic").is_in([2830, 2831, *range(2833, 2836 + 1)]))
            .then(13)
            .when(
                col("sic").is_in(
                    [
                        *range(2800, 2829 + 1),
                        *range(2850, 2879 + 1),
                        *range(2890, 2899 + 1),
                    ]
                )
            )
            .then(14)
            .when(col("sic").is_in([3031, 3041, *range(3050, 3053 + 1), *range(3060, 3099 + 1)]))
            .then(15)
            .when(
                col("sic").is_in(
                    [
                        *range(2200, 2284 + 1),
                        *range(2290, 2295 + 1),
                        *range(2297, 2299 + 1),
                        *range(2393, 2395 + 1),
                        *range(2397, 2399 + 1),
                    ]
                )
            )
            .then(16)
            .when(
                col("sic").is_in(
                    [
                        2660,
                        2661,
                        3200,
                        3210,
                        3211,
                        3240,
                        3241,
                        3261,
                        3264,
                        3280,
                        3281,
                        3446,
                        3996,
                        *range(800, 899 + 1),
                        *range(2400, 2439 + 1),
                        *range(2450, 2459 + 1),
                        *range(2490, 2499 + 1),
                        *range(2950, 2952 + 1),
                        *range(3250, 3259 + 1),
                        *range(3270, 3275 + 1),
                        *range(3290, 3293 + 1),
                        *range(3295, 3299 + 1),
                        *range(3420, 3429 + 1),
                        *range(3430, 3433 + 1),
                        *range(3440, 3442 + 1),
                        *range(3448, 3452 + 1),
                        *range(3490, 3499 + 1),
                    ]
                )
            )
            .then(17)
            .when(
                col("sic").is_in(
                    [
                        *range(1500, 1511 + 1),
                        *range(1520, 1549 + 1),
                        *range(1600, 1799 + 1),
                    ]
                )
            )
            .then(18)
            .when(
                col("sic").is_in(
                    [
                        3300,
                        *range(3310, 3317 + 1),
                        *range(3320, 3325 + 1),
                        *range(3330, 3341 + 1),
                        *range(3350, 3357 + 1),
                        *range(3360, 3379 + 1),
                        *range(3390, 3399 + 1),
                    ]
                )
            )
            .then(19)
            .when(col("sic").is_in([3400, 3443, 3444, *range(3460, 3479 + 1)]))
            .then(20)
            .when(
                col("sic").is_in(
                    [
                        3538,
                        3585,
                        3586,
                        *range(3510, 3536 + 1),
                        *range(3540, 3569 + 1),
                        *range(3580, 3582 + 1),
                        *range(3589, 3599 + 1),
                    ]
                )
            )
            .then(21)
            .when(
                col("sic").is_in(
                    [
                        3600,
                        3620,
                        3621,
                        3648,
                        3649,
                        3660,
                        3699,
                        *range(3610, 3613 + 1),
                        *range(3623, 3629 + 1),
                        *range(3640, 3646 + 1),
                        *range(3690, 3692 + 1),
                    ]
                )
            )
            .then(22)
            .when(
                col("sic").is_in(
                    [
                        2296,
                        2396,
                        3010,
                        3011,
                        3537,
                        3647,
                        3694,
                        3700,
                        3710,
                        3711,
                        3799,
                        *range(3713, 3716 + 1),
                        *range(3790, 3792 + 1),
                    ]
                )
            )
            .then(23)
            .when(col("sic").is_in([3720, 3721, 3728, 3729, *range(3723, 3725 + 1)]))
            .then(24)
            .when(col("sic").is_in([3730, 3731, *range(3740, 3743 + 1)]))
            .then(25)
            .when(col("sic").is_in([3795, *range(3760, 3769 + 1), *range(3480, 3489 + 1)]))
            .then(26)
            .when(col("sic").is_in([*range(1040, 1049 + 1)]))
            .then(27)
            .when(
                col("sic").is_in(
                    [
                        *range(1000, 1039 + 1),
                        *range(1050, 1119 + 1),
                        *range(1400, 1499 + 1),
                    ]
                )
            )
            .then(28)
            .when(col("sic").is_in([*range(1200, 1299 + 1)]))
            .then(29)
            .when(
                col("sic").is_in(
                    [
                        1300,
                        1389,
                        *range(1310, 1339 + 1),
                        *range(1370, 1382 + 1),
                        *range(2900, 2912 + 1),
                        *range(2990, 2999 + 1),
                    ]
                )
            )
            .then(30)
            .when(
                col("sic").is_in(
                    [
                        4900,
                        4910,
                        4911,
                        4939,
                        *range(4920, 4925 + 1),
                        *range(4930, 4932 + 1),
                        *range(4940, 4942 + 1),
                    ]
                )
            )
            .then(31)
            .when(
                col("sic").is_in(
                    [
                        4800,
                        4899,
                        *range(4810, 4813 + 1),
                        *range(4820, 4822 + 1),
                        *range(4830, 4841 + 1),
                        *range(4880, 4892 + 1),
                    ]
                )
            )
            .then(32)
            .when(
                col("sic").is_in(
                    [
                        7020,
                        7021,
                        7200,
                        7230,
                        7231,
                        7240,
                        7241,
                        7250,
                        7251,
                        7395,
                        7500,
                        7600,
                        7620,
                        7622,
                        7623,
                        7640,
                        7641,
                        *range(7030, 7033 + 1),
                        *range(7210, 7212 + 1),
                        *range(7214, 7217 + 1),
                        *range(7219, 7221 + 1),
                        *range(7260, 7299 + 1),
                        *range(7520, 7549 + 1),
                        *range(7629, 7631 + 1),
                        *range(7690, 7699 + 1),
                        *range(8100, 8499 + 1),
                        *range(8600, 8699 + 1),
                        *range(8800, 8899 + 1),
                        *range(7510, 7515 + 1),
                    ]
                )
            )
            .then(33)
            .when(
                col("sic").is_in(
                    [
                        3993,
                        7218,
                        7300,
                        7374,
                        7396,
                        7397,
                        7399,
                        7519,
                        8700,
                        8720,
                        8721,
                        *range(2750, 2759 + 1),
                        *range(7310, 7342 + 1),
                        *range(7349, 7353 + 1),
                        *range(7359, 7369 + 1),
                        *range(7376, 7385 + 1),
                        *range(7389, 7394 + 1),
                        *range(8710, 8713 + 1),
                        *range(8730, 8734 + 1),
                        *range(8740, 8748 + 1),
                        *range(8900, 8911 + 1),
                        *range(8920, 8999 + 1),
                        *range(4220, 4229 + 1),
                    ]
                )
            )
            .then(34)
            .when(col("sic").is_in([3695, *range(3570, 3579 + 1), *range(3680, 3689 + 1)]))
            .then(35)
            .when(col("sic").is_in([7375, *range(7370, 7373 + 1)]))
            .then(36)
            .when(
                col("sic").is_in([3622, 3810, 3812, *range(3661, 3666 + 1), *range(3669, 3679 + 1)])
            )
            .then(37)
            .when(col("sic").is_in([3811, *range(3820, 3827 + 1), *range(3829, 3839 + 1)]))
            .then(38)
            .when(
                col("sic").is_in(
                    [
                        2760,
                        2761,
                        *range(2520, 2549 + 1),
                        *range(2600, 2639 + 1),
                        *range(2670, 2699 + 1),
                        *range(3950, 3955 + 1),
                    ]
                )
            )
            .then(39)
            .when(
                col("sic").is_in(
                    [
                        3220,
                        3221,
                        *range(2440, 2449 + 1),
                        *range(2640, 2659 + 1),
                        *range(3410, 3412 + 1),
                    ]
                )
            )
            .then(40)
            .when(
                col("sic").is_in(
                    [
                        4100,
                        4130,
                        4131,
                        4150,
                        4151,
                        4230,
                        4231,
                        4780,
                        4789,
                        *range(4000, 4013 + 1),
                        *range(4040, 4049 + 1),
                        *range(4110, 4121 + 1),
                        *range(4140, 4142 + 1),
                        *range(4170, 4173 + 1),
                        *range(4190, 4200 + 1),
                        *range(4210, 4219 + 1),
                        *range(4240, 4249 + 1),
                        *range(4400, 4700 + 1),
                        *range(4710, 4712 + 1),
                        *range(4720, 4749 + 1),
                        *range(4782, 4785 + 1),
                    ]
                )
            )
            .then(41)
            .when(
                col("sic").is_in(
                    [
                        5000,
                        5099,
                        5100,
                        *range(5010, 5015 + 1),
                        *range(5020, 5023 + 1),
                        *range(5030, 5060 + 1),
                        *range(5063, 5065 + 1),
                        *range(5070, 5078 + 1),
                        *range(5080, 5088 + 1),
                        *range(5090, 5094 + 1),
                        *range(5110, 5113 + 1),
                        *range(5120, 5122 + 1),
                        *range(5130, 5172 + 1),
                        *range(5180, 5182 + 1),
                        *range(5190, 5199 + 1),
                    ]
                )
            )
            .then(42)
            .when(
                col("sic").is_in(
                    [
                        5200,
                        5250,
                        5251,
                        5260,
                        5261,
                        5270,
                        5271,
                        5300,
                        5310,
                        5311,
                        5320,
                        5330,
                        5331,
                        5334,
                        5900,
                        5999,
                        *range(5210, 5231 + 1),
                        *range(5340, 5349 + 1),
                        *range(5390, 5400 + 1),
                        *range(5410, 5412 + 1),
                        *range(5420, 5469 + 1),
                        *range(5490, 5500 + 1),
                        *range(5510, 5579 + 1),
                        *range(5590, 5700 + 1),
                        *range(5710, 5722 + 1),
                        *range(5730, 5736 + 1),
                        *range(5750, 5799 + 1),
                        *range(5910, 5912 + 1),
                        *range(5920, 5932 + 1),
                        *range(5940, 5990 + 1),
                        *range(5992, 5995 + 1),
                    ]
                )
            )
            .then(43)
            .when(
                col("sic").is_in(
                    [
                        7000,
                        7213,
                        *range(5800, 5829 + 1),
                        *range(5890, 5899 + 1),
                        *range(7010, 7019 + 1),
                        *range(7040, 7049 + 1),
                    ]
                )
            )
            .then(44)
            .when(
                col("sic").is_in(
                    [
                        6000,
                        *range(6010, 6036 + 1),
                        *range(6040, 6062 + 1),
                        *range(6080, 6082 + 1),
                        *range(6090, 6100 + 1),
                        *range(6110, 6113 + 1),
                        *range(6120, 6179 + 1),
                        *range(6190, 6199 + 1),
                    ]
                )
            )
            .then(45)
            .when(
                col("sic").is_in(
                    [
                        6300,
                        6350,
                        6351,
                        6360,
                        6361,
                        *range(6310, 6331 + 1),
                        *range(6370, 6379 + 1),
                        *range(6390, 6411 + 1),
                    ]
                )
            )
            .then(46)
            .when(
                col("sic").is_in(
                    [
                        6500,
                        6510,
                        6540,
                        6541,
                        6610,
                        6611,
                        *range(6512, 6515 + 1),
                        *range(6517, 6532 + 1),
                        *range(6550, 6553 + 1),
                        *range(6590, 6599 + 1),
                    ]
                )
            )
            .then(47)
            .when(
                col("sic").is_in(
                    [
                        6700,
                        6798,
                        6799,
                        *range(6200, 6299 + 1),
                        *range(6710, 6726 + 1),
                        *range(6730, 6733 + 1),
                        *range(6740, 6779 + 1),
                        *range(6790, 6795 + 1),
                    ]
                )
            )
            .then(48)
            .when(col("sic").is_in([4970, 4971, 4990, 4991, *range(4950, 4961 + 1)]))
            .then(49)
            .otherwise(pl.lit(None))
            .alias("ff49")
        )

    data.collect().write_parquet("__msf_world3.parquet")


@measure_time
def nyse_size_cutoffs(data_path):
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
            WHERE  crsp_exchcd = 1
                AND obs_main   = 1
                AND exch_main  = 1
                AND primary_sec= 1
                AND common     = 1
                AND me IS NOT NULL
            GROUP BY eom
            ORDER BY eom
            """)
    nyse_sf.sink_parquet("nyse_cutoffs.parquet")


@measure_time
def classify_stocks_size_groups():
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
    nyse_cutoffs = pl.scan_parquet("nyse_cutoffs.parquet")
    __msf_world = pl.scan_parquet("__msf_world3.parquet")
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
    world_msf.collect().write_parquet("world_msf.parquet")


@measure_time
def return_cutoffs(freq, crsp_only):
    """
    Description:
        Compute return percentile cutoffs by period (monthly or daily), optionally CRSP-only.

    Steps:
        1) Select group vars and output path from freq; scan 'world_{freq}sf.parquet'.
        2) Optional CRSP filter; require common/main/primary, exclude ZWE, non-null ret_exc.
        3) Add year/month; group and aggregate counts + percentiles for ret, ret_local, ret_exc.
        4) Sort, collect, save.

    Output:
        Writes 'return_cutoffs.parquet' (monthly) or 'return_cutoffs_daily.parquet' (daily).
    """
    group_vars = "eom" if freq == "m" else "year, month"
    res_path = "return_cutoffs.parquet" if freq == "m" else "return_cutoffs_daily.parquet"
    data = (
        pl.scan_parquet(f"world_{freq}sf.parquet")
        .filter(
            (col("common") == 1)
            & (col("obs_main") == 1)
            & (col("exch_main") == 1)
            & (col("primary_sec") == 1)
            & (col("excntry") != "ZWE")
            & (col("ret_exc").is_not_null())
            & ((col("source_crsp") == 1) if crsp_only == 1 else pl.lit(True))
        )
        .with_columns(year=col("date").dt.year(), month=col("date").dt.month())
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
    data.sink_parquet(res_path)


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
    group_vars = ["year", "month"] if freq == "d" else ["eom"]
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
        2) Add year/month from dt_col; left-join cutoffs on group vars.
        3) Winsorize high (99.9) then low (0.1) for each return series.

    Output:
        Polars LazyFrame/DataFrame with winsorized returns and cutoff columns joined.
    """
    df = df.with_columns(year=col(dt_col).dt.year(), month=col(dt_col).dt.month())
    wins_data = pl.scan_parquet(wins_data_path)

    on_clause = " AND ".join([f"a.{k} = b.{k}" for k in group_vars])

    ctx = pl.SQLContext()
    ctx.register("df", df)
    ctx.register("wins_data", wins_data)

    result = ctx.execute(f"""
    SELECT
        a.*
        REPLACE(
            CASE
                WHEN a.source_crsp = 0
                AND a.ret IS NOT NULL
                AND a.ret > b.ret_99_9 THEN b.ret_99_9
                WHEN a.source_crsp = 0
                AND a.ret IS NOT NULL
                AND a.ret < b.ret_0_1 THEN b.ret_0_1
                ELSE a.ret
            END AS ret,

            CASE
                WHEN a.source_crsp = 0
                AND a.ret_local IS NOT NULL
                AND a.ret_local > b.ret_local_99_9 THEN b.ret_local_99_9
                WHEN a.source_crsp = 0
                AND a.ret_local IS NOT NULL
                AND a.ret_local < b.ret_local_0_1 THEN b.ret_local_0_1
                ELSE a.ret_local
            END AS ret_local,

            CASE
                WHEN a.source_crsp = 0
                AND a.ret_exc IS NOT NULL
                AND a.ret_exc > b.ret_exc_99_9 THEN b.ret_exc_99_9
                WHEN a.source_crsp = 0
                AND a.ret_exc IS NOT NULL
                AND a.ret_exc < b.ret_exc_0_1 THEN b.ret_exc_0_1
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
        )
    )
    return df


def drop_non_trading_days(df, n_col, dt_col, over_vars, thresh_fraction):
    """
    Description:
        Remove thin-trading days by country-month (or given window) based on stock coverage.

    Steps:
        1) Add year/month from dt_col; compute max_stocks over over_vars.
        2) Keep rows where n_col / max_stocks ≥ thresh_fraction.
        3) Drop helper columns.

    Output:
        Frame filtered to sufficiently traded dates.
    """
    df = (
        df.with_columns(year=col(dt_col).dt.year(), month=col(dt_col).dt.month())
        .with_columns(max_stocks=pl.max(n_col).over(over_vars))
        .filter((col(n_col) / col("max_stocks")) >= thresh_fraction)
        .drop(["year", "month", "max_stocks"])
    )
    return df


@measure_time
def market_returns(data_path, freq, wins_comp, wins_data_path):
    """
    Description:
        Build country-level market returns (daily or monthly), optional winsorization, and save to disk.

    Steps:
        1) Load params from freq; scan data; keep common-stock fields; sort by [id, dt].
        2) Add lags me_lag1, dolvol_lag1 per id.
        3) If wins_comp, join cutoffs and winsorize returns.
        4) Apply stock filters & compute VW/EW country returns.
        5) If daily, drop low-coverage trading days.
        6) Sort and write 'market_returns{_daily}.parquet'.

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
    __common_stocks = apply_stock_filter_and_compute_indexes(__common_stocks, dt_col, max_date_lag)
    if freq == "d":
        __common_stocks = drop_non_trading_days(
            __common_stocks, "stocks", dt_col, ["excntry", "year", "month"], 0.25
        )
    __common_stocks.sort(["excntry", dt_col]).collect().write_parquet(
        f"market_returns{path_aux}.parquet"
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
    coverage, convert_to_usd, me_data_path, include_helpers_vars, start_date
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
        pl.scan_parquet("../raw/raw_tables/comp_g_fundq.parquet").collect_schema().names()
    )
    fundq_cols = pl.scan_parquet("../raw/raw_tables/comp_fundq.parquet").collect_schema().names()
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
            "../raw/raw_tables/comp_g_funda.parquet", start_date, "GLOBAL", 1
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
            "../raw/raw_tables/comp_g_fundq.parquet", start_date, "GLOBAL", 1
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
            "../raw/raw_tables/comp_funda.parquet", start_date, "NA", 2
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
            "../raw/raw_tables/comp_fundq.parquet", start_date, "NA", 2
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
        fx = compustat_fx().lazy()
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
        __compq = add_helper_vars(__compq)
        __compa = add_helper_vars(__compa)

    __compa.unique(["gvkey", "datadate"]).drop("n").sort(
        ["gvkey", "datadate"]
    ).collect().write_parquet("acc_std_ann.parquet")
    __compq.unique(["gvkey", "datadate"]).drop("n").sort(
        ["gvkey", "datadate"]
    ).collect().write_parquet("acc_std_qtr.parquet")


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


def add_helper_vars(data):
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
    os.system("rm -f aux_add_helpers.ddb")
    con = ibis.duckdb.connect("aux_add_helpers.ddb", threads=os.cpu_count())
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


def compute_earnings_persistence(data_path, __n, __min):
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
    con = ibis.duckdb.connect("aux_earnings_pers.ddb", threads=os.cpu_count())
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
    con.table("reg_results").to_parquet("ni_ar_res.parquet")
    con.disconnect()
    os.system("rm -f aux_earnings_pers.ddb")


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

    if mode == 1:
        return pl.when(col(den) != 0).then(col(num) / col(den)).otherwise(fl_none()).alias(name)
    if mode == 2:
        return (
            pl.when(col(den) != 0)
            .then(col(num) / (col(den).abs()))
            .otherwise(fl_none())
            .alias(name)
        )
    if mode == 3:
        return pl.when(col(den) > 0).then(col(num) / col(den)).otherwise(fl_none()).alias(name)
    if mode == 4:
        cond1 = col("count") > 12
        cond2 = (col(den).shift(12) > 0).over(["gvkey", "curcd"])
        col_exp = (col(num) / col(den).shift(12)).over(["gvkey", "curcd"])
        return pl.when(cond1 & cond2).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 5:
        return (
            pl.when(col(den) != 0)
            .then(col(num) * col("fx") / col(den))
            .otherwise(fl_none())
            .alias(name)
        )
    if mode == 6:
        cond1 = (col(den).shift(12) != 0).over(["gvkey", "curcd"])
        col_exp = (col(num) / col(den).shift(12)).over(["gvkey", "curcd"])
        return pl.when(cond1).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 7:
        cond1 = (col(den).shift(12) > 0).over(["gvkey", "curcd"])
        col_exp = (col(num) / col(den)).shift(12).over(["gvkey", "curcd"])
        return pl.when(cond1).then(col_exp).otherwise(fl_none()).alias(name)
    if mode == 8:
        cond1 = col(num) > 0
        cond2 = col(den) > 0
        return pl.when(cond1 & cond2).then(col(num) / col(den)).otherwise(fl_none()).alias(name)
    if mode == 9:
        cond1 = col("count") > 3
        cond2 = col(den).shift(3) > 0
        col_exp = col(num) / col(den).shift(3)
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
    if path == "acc_std_qtr.parquet":
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


def add_earnings_persistence_and_expand(df, data_path, lag_to_pub, max_lag):
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
    compute_earnings_persistence(data_path, 5, 5)
    earnings_pers = pl.scan_parquet("ni_ar_res.parquet")
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


def convert_raw_vars_to_usd(df):
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
    fx = compustat_fx().rename({"datadate": "date"}).lazy()
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
            add_earnings_persistence_and_expand,
            data_path=data_path,
            lag_to_pub=lag_to_public,
            max_lag=max_data_lag,
        )
        .pipe(convert_raw_vars_to_usd)
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
def combine_ann_qtr_chars(ann_df_path, qtr_df_path, char_vars, q_suffix):
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
    os.system("rm -f aux_aqtr_chars.ddb")
    con = ibis.duckdb.connect("aux_aqtr_chars.ddb", threads=os.cpu_count())
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
    combined.to_parquet("acc_chars_world.parquet")
    con.disconnect()
    os.system("rm -f aux_aqtr_chars.ddb")


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
def market_chars_monthly(data_path, market_ret_path, local_currency=False):
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

    data.collect(streaming=True).write_parquet("market_chars_m.parquet")


@measure_time
def firm_age(data_path):
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
    comp_secm = con.read_parquet("../raw/raw_tables/comp_secm.parquet").select(
        ["gvkey", "datadate"]
    )
    comp_gsecm = (
        con.read_parquet("../raw/raw_tables/comp_g_secd.parquet")
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
    comp_funda = con.read_parquet("../raw/raw_tables/comp_funda.parquet").select(
        ["gvkey", "datadate"]
    )
    comp_gfunda = con.read_parquet("../raw/raw_tables/comp_g_funda.parquet").select(
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
        con.read_parquet("../raw/raw_tables/crsp_msf_v2.parquet")
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
    con.table("age3").to_parquet("firm_age.parquet")
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
        | (
            ((col("crsp_exchcd_l") == 1) | (col("comp_exchg_l") == 11))
            & (col("excntry_l") == "USA")
        )
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


@measure_time
def ap_factors(output_path, freq, sf_path, mchars_path, mkt_path, min_stocks_bp, min_stocks_pf):
    """
    Description:
        Build AP-style factor panels (FF HML/SMB and HXZ INV/ROE/SMB) by country and month (or day).

    Steps:
        1) Load security returns; winsorize ret_exc by date.
        2) Load market characteristics; lag key vars 1 period with continuity guard; filter to eligible stocks.
        3) Size-bucket each stock; run FF-style sorts for BE/ME, asset growth, and ROE.
        4) Compose factors: mktrf from market file; HML/SMB (FF); INV/ROE/SMB (HXZ).
        5) Join and write factors to output_path.

    Output:
        Parquet factor file with columns: [excntry, date/eom, mktrf, hml, smb_ff, inv, roe, smb_hxz].
    """
    date_col = "eom" if freq == "m" else "date"
    sf_cond = (col("ret_lag_dif") == 1) if freq == "m" else (col("ret_lag_dif") <= 5)
    lag_vars = [
        "comp_exchg",
        "crsp_exchcd",
        "exch_main",
        "obs_main",
        "common",
        "primary_sec",
        "excntry",
        "size_grp",
        "me",
        "be_me",
        "at_gr1",
        "niq_be",
    ]

    # print(f"Executing AP factors with frequency {freq}", flush=True)

    world_sf1 = (
        pl.scan_parquet(sf_path)
        .filter(sf_cond & col("ret_exc").is_not_null())
        .select(["excntry", "id", "eom", "date", "ret_exc"])
    )
    world_sf2 = world_sf1.sql(f"""
                                WITH bounds AS (
                                SELECT
                                    eom,
                                    QUANTILE_DISC(ret_exc, {0.1 / 100}) AS low,
                                    QUANTILE_DISC(ret_exc, {99.9 / 100}) AS high
                                FROM self
                                GROUP BY eom
                                )
                                SELECT
                                    excntry,
                                    id,
                                    eom,
                                    date,
                                    CASE
                                        WHEN ret_exc < low  THEN low
                                        WHEN ret_exc > high THEN high
                                        ELSE ret_exc
                                    END AS ret_exc
                                FROM self
                                LEFT JOIN bounds
                                USING (eom)
                            """)

    base = (
        pl.scan_parquet(mchars_path)
        .sort(["id", "eom"])
        .with_columns(
            [col(i).shift(1).over(["id", "source_crsp"]).alias(i + "_l") for i in lag_vars]
        )
        .sort(["id", "eom"])
        .with_columns(
            [
                pl.when(
                    (
                        12 * (col("eom").dt.year() - col("eom").shift(1).dt.year())
                        + (col("eom").dt.month() - col("eom").shift(1).dt.month()).cast(pl.Int32)
                    ).over("id")
                    != 1
                )
                .then(pl.lit(None))
                .otherwise(i + "_l")
                .alias(i + "_l")
                for i in lag_vars
            ]
        )
        .filter(
            (col("obs_main_l") == 1)
            & (col("exch_main_l") == 1)
            & (col("common_l") == 1)
            & (col("primary_sec_l") == 1)
            & (col("ret_lag_dif") == 1)
            & col("me_l").is_not_null()
        )
        .with_columns(
            size_pf=(
                pl.when(col("size_grp_l").is_null())
                .then(pl.lit(""))
                .when(col("size_grp_l").is_in(["large", "mega"]))
                .then(pl.lit("big"))
                .otherwise(pl.lit("small"))
            )
        )
    )

    ff = sort_ff_style("be_me", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2).rename(
        {"lms": "hml", "smb": "smb_ff"}
    )
    asset_growth = sort_ff_style(
        "at_gr1", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2
    ).rename({"lms": "at_gr1_lms", "smb": "at_gr1_smb"})
    roeq = sort_ff_style("niq_be", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2).rename(
        {"lms": "niq_be_lms", "smb": "niq_be_smb"}
    )
    hxz = asset_growth.join(roeq, how="left", on=["excntry", date_col]).select(
        [
            "excntry",
            date_col,
            (-1 * col("at_gr1_lms")).alias("inv"),
            col("niq_be_lms").alias("roe"),
            ((col("at_gr1_smb") + col("niq_be_smb")) / 2).alias("smb_hxz"),
        ]
    )

    output = (
        pl.scan_parquet(mkt_path)
        .select(["excntry", date_col, col("mkt_vw_exc").alias("mktrf")])
        .collect()
        .join(ff, how="left", on=["excntry", date_col])
        .join(hxz, how="left", on=["excntry", date_col])
    )
    output.write_parquet(output_path)


def prep_data_factor_regs(data_path, fcts_path):
    """
    Description:
        Prepare monthly panel for factor regressions (join data with factors, filter, winsorize).

    Steps:
        1) Create __msf1: join msf with factors on (excntry, eom); keep ret_exc, mktrf, hml, smb_ff and valid monthly obs.
        2) Add integer date (aux_date) and cast id_int.
        3) Winsorize ret_exc by eom into __msf2.

    Output:
        DuckDB connection containing tables '__msf2' (ready for rolling regs).
    """

    os.system("rm -f aux_factor_regs.ddb")
    con = ibis.duckdb.connect("aux_factor_regs.ddb", threads=os.cpu_count())

    con.raw_sql(f"""
    CREATE OR REPLACE VIEW data_msf AS
    SELECT *
    FROM read_parquet('{data_path}');

    CREATE OR REPLACE VIEW fcts AS
    SELECT *
    FROM read_parquet('{fcts_path}');

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
    ),
    percs AS (
        SELECT
          eom,
          QUANTILE_DISC(ret_exc, {0.1 / 100}) AS low,
          QUANTILE_DISC(ret_exc, {99.9 / 100}) AS high
        FROM __msf1
        GROUP BY eom
    )
    SELECT
       d.* EXCLUDE (ret_exc),
      CASE
        WHEN d.ret_exc < p.low  THEN p.low
        WHEN d.ret_exc > p.high THEN p.high
        ELSE d.ret_exc
      END AS ret_exc
    FROM __msf1 AS d
    LEFT JOIN percs AS p
      USING (eom);
    """)
    return con


@measure_time
def market_beta(output_path, data_path, fcts_path, __n, __min):
    """
    Description:
        Estimate rolling CAPM betas and idiosyncratic vol for each stock.

    Steps:
        1) Prep data via prep_data_factor_regs; load '__msf2' lazily.
        2) Generate rolling-window mappings; run process_map_chunks(..., 'capm') per mapping.
        3) Map back to ids/dates; select beta_{__n}m and ivol_capm_{__n}m; sort.

    Output:
        Parquet at output_path with [id, eom, beta_{__n}m, ivol_capm_{__n}m].
    """
    con = prep_data_factor_regs(data_path, fcts_path)
    base_data = con.table("__msf2").to_polars().lazy()
    aux_maps = gen_aux_maps(__n)
    df = pl.concat(
        [process_map_chunks(base_data, mapping, "capm", __n, __min) for mapping in aux_maps]
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
def residual_momentum(output_path, data_path, fcts_path, __n, __min, incl, skip):
    """
    Description:
        Compute residual momentum from FF3 regressions with rolling windows and skip/inclusion rules.

    Steps:
        1) Prep '__msf2'; build window mappings; run process_map_chunks(..., 'res_mom', __n, __min, incl, skip).
        2) Join back ids/dates and keep resff3_{incl}_{skip}; sort.

    Output:
        Parquet '{output_path}_{incl}_{skip}.parquet' with [id, eom, resff3_{incl}_{skip}].
    """
    con = prep_data_factor_regs(data_path, fcts_path)
    base_data = con.table("__msf2").to_polars().lazy()
    aux_maps = gen_aux_maps(__n)
    df = pl.concat(
        [
            process_map_chunks(base_data, mapping, "res_mom", __n, __min, incl, skip)
            for mapping in aux_maps
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
    res.write_parquet(output_path + f"_{incl}_{skip}.parquet")
    con.disconnect()


@measure_time
def prepare_daily(data_path, fcts_path):
    """
    Description:
        Build daily dataset: align returns with factors, shrink dtypes, and create helpers.

    Steps:
        1) Join daily stock data with daily factors; filter rows with mktrf.
        2) Create zero_obs flags per (id,eom); cap returns to lag ≤14 days; compute prc_adj.
        3) Write dsf1.parquet and id_int_key.parquet.
        4) Build market lead/lag series per day and write mkt_lead_lag.parquet.
        5) Build 3-day rolling sums for stock and market excess returns for correlations; write corr_data.parquet.

    Output:
        Parquets: dsf1.parquet, id_int_key.parquet, mkt_lead_lag.parquet, corr_data.parquet.
    """
    data = pl.scan_parquet(data_path)
    fcts = pl.scan_parquet(fcts_path)
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
    dsf1.collect().write_parquet("dsf1.parquet")

    id_int_key = pl.scan_parquet("dsf1.parquet").select(["id", "id_int"]).unique()
    id_int_key.collect().write_parquet("id_int_key.parquet")

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
    mkt_lead_lag.collect().write_parquet("mkt_lead_lag.parquet")

    corr_data = (
        pl.scan_parquet("dsf1.parquet")
        .select(["ret_exc", "id", "id_int", "date", "mktrf", "eom", "zero_obs"])
        .sort(["id_int", "date"])
        .with_columns(
            ret_exc_3l=(col("ret_exc") + col("ret_exc").shift(1) + col("ret_exc").shift(2)).over(
                ["id_int"]
            ),
            mkt_exc_3l=(col("mktrf") + col("mktrf").shift(1) + col("mktrf").shift(2)).over(
                ["id_int"]
            ),
        )
        .select(["id_int", "eom", "zero_obs", "ret_exc_3l", "mkt_exc_3l"])
        .select(pl.all().shrink_dtype())
        .sort(["id_int", "eom"])
    )
    corr_data.collect().write_parquet("corr_data.parquet")


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


@measure_time
def mispricing_factors(data_path, min_stks, min_fcts=3, output_path="mp_factors.parquet"):
    """
    Description:
        Compute two mispricing composites (management & performance) from ranked inputs.

    Steps:
        1) Load/Filter monthly stock panel; keep id/eom/excntry + factor inputs.
        2) Iteratively join rank-normalized columns per variable with proper direction.
        3) Build mispricing_mgmt from vars_mgmt and mispricing_perf from vars_perf using gen_misp_exp.
        4) Keep id/eom + both composites and write parquet.

    Output:
        '{output_path}' with [id, eom, mispricing_perf, mispricing_mgmt].
    """
    vars_mgmt = [
        "chcsho_12m",
        "eqnpo_12m",
        "oaccruals_at",
        "noa_at",
        "at_gr1",
        "ppeinv_gr1a",
    ]
    vars_perf = ["o_score", "ret_12_1", "gp_at", "niq_at"]
    direction = [True, False, True, True, True, True, True, False, False, False]
    index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    aux_df = (
        pl.scan_parquet(data_path)
        .filter(
            (col("common") == 1)
            & (col("primary_sec") == 1)
            & (col("obs_main") == 1)
            & (col("exch_main") == 1)
            & (col("ret_exc").is_not_null())
            & (col("me").is_not_null())
        )
        .select(["id", "eom", "excntry", *(vars_mgmt + vars_perf)])
        .sort(["excntry", "eom"])
    )
    chars = {"1": aux_df}
    for __d, __v, i in zip(direction, vars_mgmt + vars_perf, index, strict=True):
        subset = gen_ranks_and_normalize(aux_df, ["id"], ["excntry"], ["eom"], __d, __v, min_stks)
        chars[f"{(i + 1) % 2}"] = chars[f"{i % 2}"].join(subset, on=["id", "eom"], how="left")
    chars["1"] = (
        chars["1"]
        .with_columns(
            mispricing_mgmt=gen_misp_exp(vars_mgmt, min_fcts),
            mispricing_perf=gen_misp_exp(vars_perf, min_fcts),
        )
        .select(["id", "eom", "mispricing_perf", "mispricing_mgmt"])
    )
    chars["1"].collect().write_parquet(output_path)


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
def bidask_hl(output_path, data_path, market_returns_daily_path, __min_obs):
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
    msf_path, market_chars_monthly_path, acc_chars_world_path, output_path
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
def finish_daily_chars(output_path):
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
    bidask = pl.scan_parquet("corwin_schultz.parquet")
    r1 = pl.scan_parquet("roll_apply_daily.parquet").with_columns(col("id").cast(pl.Int64))
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
def quality_minus_junk(data_path, min_stks):
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
    qmj = pl.scan_parquet(data_path).filter(c1).select(cols).sort(["excntry", "eom"]).collect()
    for var_z, dir in zip(z_vars, direction, strict=True):
        __z = z_ranks(qmj, var_z, min_stks, dir)
        qmj = qmj.join(__z, how="full", coalesce=True, on=["excntry", "eom", "id"])

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
        .join(ranks["prof"], how="full", coalesce=True, on=["excntry", "id", "eom"])
        .join(ranks["growth"], how="full", coalesce=True, on=["excntry", "id", "eom"])
        .join(ranks["safety"], how="full", coalesce=True, on=["excntry", "id", "eom"])
        .with_columns(__qmj=(col("qmj_prof") + col("qmj_growth") + col("qmj_safety")) / 3)
    )
    __qmj = z_ranks(qmj, "__qmj", min_stks, "ascending").rename({"z___qmj": "qmj"})
    qmj = qmj.join(__qmj, how="left", on=["excntry", "id", "eom"]).drop("__qmj")
    qmj.write_parquet("qmj.parquet")


@measure_time
def save_main_data(end_date):
    """
    Description:
        Filter world_data to main securities and export country-level files.

    Steps:
        1) Load world_data.parquet and compute lagged market equity.
        2) Filter to valid securities up to end_date.
        3) Save filtered dataset and split into country parquet files.

    Output:
        'world_data_filtered.parquet' and 'characteristics/{country}.parquet'.
    """
    months_exp = (col("eom").dt.year() * 12 + col("eom").dt.month()).cast(pl.Int64)
    data = (
        pl.scan_parquet("world_data.parquet")
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
        .filter(
            (col("primary_sec") == 1)
            & (col("common") == 1)
            & (col("obs_main") == 1)
            & (col("exch_main") == 1)
            & (col("eom") <= end_date)
        )
    )
    data.select(pl.all().shrink_dtype()).collect(streaming=True).write_parquet(
        "world_data_filtered.parquet"
    )

    os.chdir(os.path.join(os.path.dirname(__file__), "..", "data/processed"))

    OUT_DIR = "characteristics"
    con = duckdb.connect()
    con.execute(f"""
    COPY (SELECT * FROM read_parquet('../interim/world_data_filtered.parquet'))
    TO '{OUT_DIR}'
    ( FORMAT PARQUET,
      COMPRESSION ZSTD,
      PARTITION_BY (excntry),
      WRITE_PARTITION_COLUMNS TRUE,
      OVERWRITE
      );
    """)
    con.close()
    os.system(f"""
    for d in {OUT_DIR}/excntry=*; do
        if [ -d "$d" ]; then
            country="${{d#*=}}"   # strip "excntry="
            partfile=$(find "$d" -type f -name "*.parquet" | head -n1)
            if [ -n "$partfile" ]; then
                mv "$partfile" "{OUT_DIR}/${{country}}.parquet"
            fi
            rm -rf "$d"
        fi
    done
    """)


@measure_time
def save_output_files():
    """
    Description:
        Move main market returns and cutoff files to Output folder.

    Steps:
        1) Use system mv to move parquet outputs.
        2) Includes market returns (monthly/daily) and cutoff files.

    Output:
        Files relocated into 'Output/' directory.
    """
    os.system("mv ../interim/market_returns.parquet other_output/")
    os.system("mv ../interim/market_returns_daily.parquet other_output/")
    os.system("mv ../interim/nyse_cutoffs.parquet other_output/")
    os.system("mv ../interim/return_cutoffs.parquet other_output/")
    os.system("mv ../interim/return_cutoffs_daily.parquet other_output/")
    os.system("mv ../interim/ap_factors_monthly.parquet other_output/")
    os.system("mv ../interim/ap_factors_daily.parquet other_output/")


@measure_time
def save_daily_ret():
    """
    Description:
        Export daily returns split by country.

    Steps:
        1) Load world_dsf.parquet with daily returns.
        2) Identify unique countries.
        3) For each country, filter and save parquet file (compressed).

    Output:
        'return_data/daily_rets_by_country/{country}.parquet' files for all countries.
    """
    data = (
        pl.scan_parquet("../interim/world_dsf.parquet")
        .select(["excntry", "id", "date", "me", "ret", "ret_exc"])
        .with_columns(
            excntry=pl.when(col("excntry").is_null())
            .then(pl.lit("null_country"))
            .otherwise(col("excntry"))
        )
    )
    data.collect(engine="streaming").write_parquet("../interim/daily_returns_temp.parquet")

    OUT_DIR = "return_data/daily_rets_by_country"
    con = duckdb.connect()
    con.execute(f"""
    COPY (SELECT * FROM read_parquet('../interim/daily_returns_temp.parquet'))
    TO '{OUT_DIR}'
    ( FORMAT PARQUET,
      COMPRESSION ZSTD,
      PARTITION_BY (excntry),
      OVERWRITE
      );
    """)
    con.close()
    os.system(f"""
    for d in {OUT_DIR}/excntry=*; do
        if [ -d "$d" ]; then
            country="${{d#*=}}"   # strip "excntry="
            partfile=$(find "$d" -type f -name "*.parquet" | head -n1)
            if [ -n "$partfile" ]; then
                mv "$partfile" "{OUT_DIR}/${{country}}.parquet"
            fi
            rm -rf "$d"
        fi
    done
    """)


@measure_time
def save_accounting_data():
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
    pl.scan_parquet("../interim/acc_std_qtr.parquet").filter(
        col("source").is_not_null()
    ).collect().write_parquet("accounting_data/quarterly.parquet")
    pl.scan_parquet("../interim/acc_std_ann.parquet").filter(
        col("source").is_not_null()
    ).collect().write_parquet("accounting_data/annual.parquet")


@measure_time
def save_full_files_and_cleanup(clear_interim=True):
    """
    Description:
        Save full datasets and remove temporary files.

    Steps:
        1) Write compressed versions of world_dsf, world_data, and filtered world_data.
        2) Remove raw parquet files and raw_tables/raw_data_dfs folders.

    Output:
        Compressed parquet files in return_data/ and characteristics/, cleanup of temp files.
    """
    pl.scan_parquet("../interim/world_dsf.parquet").select(pl.all().shrink_dtype()).collect(
        streaming=True
    ).write_parquet("return_data/world_dsf.parquet")
    pl.scan_parquet("../interim/world_data.parquet").select(pl.all().shrink_dtype()).collect(
        streaming=True
    ).write_parquet("characteristics/world_data_unfiltered.parquet")
    pl.scan_parquet("../interim/world_data_filtered.parquet").select(
        pl.all().shrink_dtype()
    ).collect(streaming=True).write_parquet("characteristics/world_data_filtered.parquet")
    if clear_interim:
        os.system("rm -rf ../interim/* ../raw/*")


@measure_time
def save_monthly_ret():
    """
    Description:
        Save monthly returns for world securities.

    Steps:
        1) Load world_msf.parquet and select relevant columns.
        2) Shrink dtypes and collect results.
        3) Write to return_data/world_ret_monthly.parquet.

    Output:
        Parquet file with monthly returns by country/security.
    """
    data = pl.scan_parquet("../interim/world_msf.parquet").select(
        ["excntry", "id", "source_crsp", "eom", "me", "ret_exc", "ret", "ret_local"]
    )
    data.select(pl.all().shrink_dtype()).collect().write_parquet(
        "return_data/world_ret_monthly.parquet"
    )


@measure_time
def merge_roll_apply_daily_results():
    """
    Description:
        Merge rolling regression daily results into one dataset.

    Steps:
        1) Build date index from earliest to current month.
        2) Load id_int mapping and all '__roll*' parquet files.
        3) Outer join them on (id_int, aux_date).
        4) Map aux_date to calendar eom and join id keys.
        5) Save consolidated roll_apply_daily.parquet.

    Output:
        'roll_apply_daily.parquet' with merged roll regression results.
    """
    date_idx = datetime.datetime.today().month + datetime.datetime.today().year * 12
    df_dates = pl.DataFrame(
        {
            "aux_date": [i + 1 for i in range(23112, date_idx + 1)],
            "eom": [f"{i // 12}-{i % 12 + 1}-1" for i in range(23112, date_idx + 1)],
        }
    )
    df_dates = df_dates.with_columns(
        col("eom").str.strptime(pl.Date, "%Y-%m-%d").dt.month_end().alias("eom"),
        col("aux_date").cast(pl.Int64),
    )
    df_id = pl.scan_parquet("id_int_key.parquet")
    file_paths = [i for i in os.listdir() if i.startswith("__roll")]
    if len(file_paths) != 1:
        joint_file = pl.scan_parquet(file_paths[0])
        for i in file_paths[1:]:
            df_aux = pl.scan_parquet(i)
            joint_file = joint_file.join(df_aux, how="outer_coalesce", on=["id_int", "aux_date"])
        joint_file.with_columns(col("aux_date").cast(pl.Int64)).join(
            df_dates.lazy(), how="left", on="aux_date"
        ).join(df_id, how="left", on="id_int").drop(["aux_date", "id_int"]).collect().write_parquet(
            "roll_apply_daily.parquet"
        )
    else:
        joint_file = pl.scan_parquet(file_paths[0])

    joint_file.with_columns(col("aux_date").cast(pl.Int64)).join(
        df_dates.lazy().with_columns(col("aux_date").cast(pl.Int64)),
        how="left",
        on="aux_date",
    ).join(df_id, how="left", on="id_int").drop(["aux_date", "id_int"]).collect().write_parquet(
        "roll_apply_daily.parquet"
    )


@measure_time
def merge_world_data_prelim():
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
    a = pl.scan_parquet("world_data_prelim.parquet")
    b = pl.scan_parquet("beta_60m.parquet")
    c = pl.scan_parquet("resmom_ff3_12_1.parquet")
    d = pl.scan_parquet("resmom_ff3_6_1.parquet")
    e = pl.scan_parquet("mp_factors.parquet")
    f = pl.scan_parquet("market_chars_d.parquet")
    g = pl.scan_parquet("firm_age.parquet").select(["id", "eom", "age"])
    world_data = (
        a.join(b, how="left", on=["id", "eom"])
        .join(c, how="left", on=["id", "eom"])
        .join(d, how="left", on=["id", "eom"])
        .join(e, how="left", on=["id", "eom"])
        .join(f, how="left", on=["id", "eom"])
        .join(g, how="left", on=["id", "eom"])
    )
    world_data.collect().write_parquet("world_data_-1.parquet")


@measure_time
def merge_qmj_to_world_data():
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
    a = pl.scan_parquet("world_data_-1.parquet")
    b = pl.scan_parquet("qmj.parquet")
    result = (
        a.join(b, how="left", on=["excntry", "id", "eom"]).unique(["id", "eom"]).sort(["id", "eom"])
    )
    result.collect().write_parquet("world_data.parquet")


@measure_time
def merge_industry_to_world_msf():
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
    __msf_world = pl.scan_parquet("__msf_world.parquet")
    comp_ind = pl.scan_parquet("comp_ind.parquet")
    crsp_ind = pl.scan_parquet("crsp_ind.parquet").rename(
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
    __msf_world.collect().write_parquet("__msf_world2.parquet")


@measure_time
def roll_apply_daily(stats, sfx, __min):
    """
    Description:
        Run rolling daily-stat calculations over grouped date windows and save results.

    Steps:
        1) Generate date-group mappings from sfx (e.g., _21d → k=1, _252d → k=12).
        2) Prepare base daily data per stat.
        3) Apply process_map_chunks for each mapping and concat results.
        4) Write to '__roll{sfx}_{stats}.parquet'.

    Output:
        Parquet with per-(id_int, group_number) rolling metrics for `stats`.
    """
    print(f"Processing {stats} - {sfx.replace('_', '')} - {__min}", flush=True)
    aux_maps = gen_aux_maps(sfx)
    base_data = prepare_base_data(stat=stats)
    results = pl.concat(
        [process_map_chunks(base_data, mapping, stats, sfx, __min) for mapping in aux_maps]
    )
    results.collect(engine="streaming").write_parquet(f"__roll{sfx}_{stats}.parquet")


def gen_consecutive_lists(input_list, k):
    """
    Description:
        Split a list into consecutive, non-overlapping sublists of length k.

    Steps:
        1) Slice input_list in steps of k.
        2) Keep only full-length chunks.

    Output:
        List of k-length sublists.
    """
    return [
        input_list[i : i + k]
        for i in range(0, len(input_list), k)
        if len(input_list[i : i + k]) == k
    ]


def build_groups(input_list, k):
    """
    Description:
        Build k staggered groupings (offset windows) over a list.

    Steps:
        1) For each offset in [0..k-1], take consecutive k-sublists from input_list[offset:].
        2) Aggregate into a list of group lists.

    Output:
        List of k lists, each containing k-length sublists.
    """
    return [gen_consecutive_lists(input_list[offset:], k) for offset in range(k)]


def group_mapping_dfs(input_list, k):
    """
    Description:
        Create mapping DataFrames linking aux_date to group_number, and group_number to new (max) aux_date.

    Steps:
        1) Build groups via build_groups(input_list, k).
        2) For each group, create a DataFrame with aux_date arrays and group_number.
        3) Return:
        - group_map: exploded (aux_date, group_number)
        - date_map : (group_number, aux_date=max group date)

    Output:
        List of dicts: {'group_map': LazyFrame, 'date_map': LazyFrame}.
    """
    groups = build_groups(input_list, k)
    dfs = [
        pl.DataFrame({"aux_date": group}).with_columns(
            group_number=pl.cum_count("aux_date"), new_date=col("aux_date").list.max()
        )
        for group in groups
    ]
    return [
        {
            "group_map": df.explode("aux_date")
            .select([col("aux_date").cast(pl.Int32), "group_number"])
            .lazy(),
            "date_map": df.select(["group_number", col("new_date").alias("aux_date")])
            .unique()
            .sort(["group_number"])
            .lazy(),
        }
        for df in dfs
    ]


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
    if stat == "zero_trades":
        return col("tvol").is_not_null()
    elif stat == "dolvol":
        return col("dolvol_d").is_not_null()
    elif stat == "turnover":
        return col("tvol").is_not_null()
    elif stat == "mktcorr":
        return (col("ret_exc_3l").is_not_null()) & (col("zero_obs") < 10)
    else:
        return (col("ret_exc").is_not_null()) & (col("zero_obs") < 10)


def prepare_base_data(stat):
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
    base_data_path = "corr_data.parquet" if stat == "mktcorr" else "dsf1.parquet"
    base_data = (
        pl.scan_parquet(base_data_path)
        .with_columns(aux_date=gen_MMYY_column("eom"))
        .filter(base_data_filter_exp(stat))
    )

    if stat == "dimsonbeta":
        lead_lag = pl.scan_parquet("mkt_lead_lag.parquet").drop(["eom", "mktrf"])
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
    if stat == "turnover" or stat == "mktcorr":
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
        if stat == "zero_trades":
            filter_var = "tvol"
        elif stat == "dolvol":
            filter_var = "dolvol_d"
        else:
            filter_var = "ret_exc"
        df = df.with_columns(n=pl.count(filter_var).over(["id_int", "group_number"])).filter(
            col("n") >= min_obs
        )
    return df


def process_map_chunks(base_data, mapping, stats, sfx, __min, incl=None, skip=None):
    """
    Description:
        Execute a rolling computation for a mapping: join groups, filter, compute stat, remap to end date.

    Steps:
        1) Join base_data with mapping['group_map'] on aux_date.
        2) Apply apply_group_filter(stat, __min).
        3) Run the appropriate function from `funcs` dict (res_mom with incl/skip).
        4) Join mapping['date_map'] to replace group_number by new aux_date.

    Output:
        LazyFrame of per-(id_int, group_number) results with remapped aux_date.
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

    df = base_data.join(mapping["group_map"], how="inner", on="aux_date").pipe(
        apply_group_filter, stat=stats, min_obs=__min
    )

    if stats == "res_mom":
        df = df.pipe(funcs[stats], sfx=sfx, __min=__min, incl=incl, skip=skip)
    else:
        df = df.pipe(funcs[stats], sfx=sfx, __min=__min)

    df = df.join(mapping["date_map"], how="left", on="group_number").drop("group_number")

    return df


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
        df.filter(col("hml").is_not_null() & col("smb_ff").is_not_null())
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


def gen_aux_maps(sfx):
    """
    Description:
        Build date-group maps from suffix window length.

    Steps:
        1) Map suffix to k: {'_21d':1,'_126d':6,'_252d':12,'_1260d':60} or int(sfx).
        2) Build aux_date range from start index to current month index.
        3) Create grouped mappings via group_mapping_dfs(date_idx, k).

    Output:
        List of {'group_map','date_map'} mappings.
    """
    parameter_mapping = {"_21d": 1, "_126d": 6, "_252d": 12, "_1260d": 60}
    date_aux = datetime.datetime.today().month + datetime.datetime.today().year * 12
    if sfx in parameter_mapping:
        date_idx = list(range(23113 - parameter_mapping[sfx], date_aux + 1))
        aux_maps = group_mapping_dfs(date_idx, parameter_mapping[sfx])
    else:
        date_idx = list(range(23113 - int(sfx), date_aux + 1))
        aux_maps = group_mapping_dfs(date_idx, int(sfx))
    return aux_maps


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
        1) Sort by (id_int,date).
        2) For each (id_int,group_number), compute last(prc_adj)/max(prc_adj) and count.
        3) Keep groups with n ≥ __min.

    Output:
        LazyFrame with f'prc_highprc{sfx}'.
    """

    df = (
        df.sort(["id_int", "date"])
        .group_by(["id_int", "group_number"])
        .agg(
            [
                (col("prc_adj").last() / col("prc_adj").max()).alias(f"prc_highprc{sfx}"),
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
    df = df.group_by(["id_int", "group_number"]).agg(
        [
            (pl.cov("ret_exc", "mktrf") / pl.var("mktrf")).alias(f"beta{sfx}"),
            (col("ret_exc") - col("mktrf") * (pl.cov("ret_exc", "mktrf") / pl.var("mktrf")))
            .std()
            .alias(f"ivol_capm{sfx}"),
        ]
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
        1) OLS: ret_exc ~ mktrf + smb_hxz + roe + inv; require all factors.
        2) Aggregate per (id_int,group_number): std(residuals, ddof=4), skew(residuals).

    Output:
        LazyFrame with [f'ivol_hxz4{sfx}', f'iskew_hxz4{sfx}'].
    """
    res_exp = pl.col("ret_exc").least_squares.ols(
        "mktrf", "smb_hxz", "roe", "inv", add_intercept=True, mode="residuals"
    )
    df = (
        df.filter(
            col("smb_hxz").is_not_null() & col("roe").is_not_null() & col("inv").is_not_null()
        )
        .group_by(["id_int", "group_number"])
        .agg(
            res_exp.std(ddof=4).alias(f"ivol_hxz4{sfx}"),
            res_exp.skew(bias=False).alias(f"iskew_hxz4{sfx}"),
        )
    )
    return df


def zero_trades(df, sfx, __min):
    """
    Description:
        Zero-trade days and turnover-based illiquidity composite.

    Steps:
        1) zero_trades = mean(tvol==0) * 21.
        2) turnover = tvol/(shares*1e6) when shares>0; take group mean.
        3) Rank turnover within group_number; composite = rank/100 + zero_trades.

    Output:
        LazyFrame with f'zero_trades{sfx}'.
    """

    aux_1 = (pl.col("tvol") == 0).mean() * 21
    aux_2 = (
        pl.when(pl.col("shares") != 0)
        .then(pl.col("tvol") / (pl.col("shares") * 1e6))
        .otherwise(fl_none())
    )
    aux_3 = (
        pl.col("turnover").rank(descending=True, method="average") / pl.count("turnover")
    ).over("group_number")
    aux_4 = (aux_3 / 100) + pl.col("zero_trades")
    df = (
        df.group_by(["id_int", "group_number"])
        .agg([aux_1.alias("zero_trades"), aux_2.alias("turnover")])
        .filter(pl.col("zero_trades").is_not_null() & pl.col("turnover").is_not_null())
        .with_columns(pl.col("turnover").list.mean())
        .with_columns(aux_4.alias(f"zero_trades{sfx}"))
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
        1) Build list turnover_d = tvol/(shares*1e6) (guard shares>0).
        2) Compute mean(turnover_d), std/mean, and n; require n ≥ __min.

    Output:
        LazyFrame with [f'turnover{sfx}', f'turnover_var{sfx}'].
    """
    aux_1 = (
        pl.when(col("turnover_d").list.mean() != 0)
        .then(col("turnover_d").list.std() / col("turnover_d").list.mean())
        .otherwise(fl_none())
    )
    df = (
        df.group_by(["id_int", "group_number"])
        .agg(
            [
                pl.when(col("shares") != 0)
                .then(col("tvol") / (col("shares") * 1e6))
                .otherwise(fl_none())
                .alias("turnover_d")
            ]
        )
        .with_columns(
            [
                col("turnover_d").list.mean().alias(f"turnover{sfx}"),
                aux_1.alias(f"turnover_var{sfx}"),
                (col("turnover_d").list.len()).alias("n"),
            ]
        )
        .filter(col("n") >= __min)
        .drop(["n", "turnover_d"])
    )
    return df


def mktcorr(df, sfx, __min):
    """
    Description:
        Rolling correlation between 3-day summed stock and market excess returns.

    Steps:
        1) Group by (id_int,group_number); count obs for ret_exc_3l and mkt_exc_3l.
        2) Require both counts ≥ __min; compute Pearson corr.

    Output:
        LazyFrame with f'corr{sfx}'.
    """
    df = (
        df.group_by(["id_int", "group_number"])
        .agg(
            [
                pl.min_horizontal([pl.count("ret_exc_3l"), pl.count("mkt_exc_3l")]).alias("n"),
                pl.corr("ret_exc_3l", "mkt_exc_3l").alias(f"corr{sfx}"),
            ]
        )
        .filter(col("n") >= __min)
        .drop("n")
    )
    return df


def dimsonbeta(df, sfx, __min):
    """
    Description:
        Dimson beta (lead-lag adjusted market beta) per window.

    Steps:
        1) OLS: ret_exc ~ mktrf + mktrf_ld1 + mktrf_lg1; get coefficients per group.
        2) Sum coefficients on market terms.

    Output:
        LazyFrame with f'beta_dimson{sfx}'.
    """
    # pds.lin_reg returns [b_mktrf, b_mktrf_ld1, b_mktrf_lg1, intercept]
    # Dimson beta should sum only the three market-loadings.
    beta_exp = col("coeffs").list.head(3).list.sum()
    df = (
        df.group_by(["id_int", "group_number"])
        .agg(
            coeffs=pds.lin_reg(
                "mktrf",
                "mktrf_ld1",
                "mktrf_lg1",
                target="ret_exc",
                add_bias=True,  # intercept will be the last element
            )
        )
        .select(["id_int", "group_number", beta_exp.alias(f"beta_dimson{sfx}")])
    )
    return df
