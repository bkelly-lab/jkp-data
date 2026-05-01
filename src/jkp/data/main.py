from pathlib import Path

from .aux_functions import (
    acc_chars_list,
    add_ret_exc_wins,
    ap_factors,
    bidask_hl,
    classify_stocks_size_groups,
    combine_ann_qtr_chars,
    combine_crsp_comp_sf,
    comp_industry,
    create_acc_chars,
    create_world_data_prelim,
    crsp_industry,
    download_raw_data_tables,
    ff_ind_class,
    filter_dsf,
    filter_msf,
    filter_world,
    finish_daily_chars,
    firm_age,
    gen_raw_data_dfs,
    market_beta,
    market_chars_monthly,
    market_returns,
    merge_industry_to_world_msf,
    merge_qmj_to_world_data,
    merge_roll_apply_daily_results,
    merge_world_data_prelim,
    mispricing_factors,
    nyse_size_cutoffs,
    prepare_comp_sf,
    prepare_crsp_sf,
    prepare_daily,
    quality_minus_junk,
    residual_momentum,
    return_cutoffs,
    roll_apply_daily,
    save_accounting_data,
    save_daily_ret,
    save_full_files_and_cleanup,
    save_main_data,
    save_monthly_ret,
    save_output_files,
    setup_folder_structure,
    standardized_accounting_data,
)
from .config import ACCOUNTING_START_DATE, END_DATE, ROLLING_DAILY_SPECS
from .paths import DataPaths
from .wrds_credentials import get_wrds_credentials


def run_pipeline(*, persistent_connection: bool = False, output_dir: Path) -> None:
    """Run the full JKP data generation pipeline."""
    paths = DataPaths(base_dir=output_dir.resolve())
    creds = get_wrds_credentials()

    interim = paths.interim_dir

    setup_folder_structure(paths)
    download_raw_data_tables(
        paths,
        username=creds.username,
        password=creds.password,
        end_date=END_DATE,
        persistent_connection=persistent_connection,
    )
    gen_raw_data_dfs(paths)
    prepare_comp_sf(paths, "both")
    prepare_crsp_sf(paths, "m")
    prepare_crsp_sf(paths, "d")
    combine_crsp_comp_sf(paths)
    crsp_industry(paths)
    comp_industry(paths)
    merge_industry_to_world_msf(paths)
    ff_ind_class(paths, interim / "__msf_world2.parquet")
    nyse_size_cutoffs(paths, interim / "__msf_world3.parquet")
    classify_stocks_size_groups(paths)
    return_cutoffs(paths, "m", 0)
    return_cutoffs(paths, "d", 0)
    add_ret_exc_wins(paths, "m")
    add_ret_exc_wins(paths, "d")
    market_returns(
        paths,
        interim / "world_dsf.parquet",
        "d",
        1,
        interim / "return_cutoffs_daily.parquet",
        interim / "nyse_cutoffs.parquet",
    )
    market_returns(
        paths,
        interim / "world_msf.parquet",
        "m",
        1,
        interim / "return_cutoffs.parquet",
        interim / "nyse_cutoffs.parquet",
    )
    standardized_accounting_data(
        paths, "world", 1, interim / "world_msf.parquet", 1, ACCOUNTING_START_DATE
    )
    create_acc_chars(
        paths,
        interim / "acc_std_ann.parquet",
        interim / "achars_world.parquet",
        4,
        18,
        acc_chars_list(),
        interim / "world_msf.parquet",
        "",
    )
    create_acc_chars(
        paths,
        interim / "acc_std_qtr.parquet",
        interim / "qchars_world.parquet",
        4,
        18,
        acc_chars_list(),
        interim / "world_msf.parquet",
        "_qitem",
    )
    combine_ann_qtr_chars(
        paths,
        interim / "achars_world.parquet",
        interim / "qchars_world.parquet",
        acc_chars_list(),
        "_qitem",
    )
    market_chars_monthly(paths, interim / "world_msf.parquet", interim / "market_returns.parquet")
    create_world_data_prelim(
        paths,
        interim / "world_msf.parquet",
        interim / "market_chars_m.parquet",
        interim / "acc_chars_world.parquet",
        interim / "world_data_prelim.parquet",
    )
    ap_factors(
        paths,
        interim / "ap_factors_daily.parquet",
        "d",
        interim / "world_dsf.parquet",
        interim / "world_data_prelim.parquet",
        interim / "market_returns_daily.parquet",
        10,
        3,
    )
    ap_factors(
        paths,
        interim / "ap_factors_monthly.parquet",
        "m",
        interim / "world_msf.parquet",
        interim / "world_data_prelim.parquet",
        interim / "market_returns.parquet",
        10,
        3,
    )
    firm_age(paths, interim / "world_msf.parquet")
    mispricing_factors(paths, interim / "world_data_prelim.parquet", 10, min_fcts=3)
    market_beta(
        paths,
        interim / "beta_60m.parquet",
        interim / "world_msf.parquet",
        interim / "ap_factors_monthly.parquet",
        60,
        36,
    )
    residual_momentum(
        paths,
        "resmom_ff3",
        interim / "world_msf.parquet",
        interim / "ap_factors_monthly.parquet",
        36,
        24,
        12,
        1,
    )
    residual_momentum(
        paths,
        "resmom_ff3",
        interim / "world_msf.parquet",
        interim / "ap_factors_monthly.parquet",
        36,
        24,
        6,
        1,
    )
    bidask_hl(
        paths,
        interim / "corwin_schultz.parquet",
        interim / "world_dsf.parquet",
        interim / "market_returns_daily.parquet",
        10,
    )
    prepare_daily(paths, interim / "world_dsf.parquet", interim / "ap_factors_daily.parquet")
    for sfx, min_obs, vars_ in ROLLING_DAILY_SPECS:
        for var in vars_:
            roll_apply_daily(paths, var, sfx, min_obs)
    merge_roll_apply_daily_results(paths)
    finish_daily_chars(paths, interim / "market_chars_d.parquet")
    merge_world_data_prelim(paths)
    quality_minus_junk(paths, interim / "world_data_-1.parquet", 10)
    merge_qmj_to_world_data(paths)
    filter_dsf(paths)
    filter_msf(paths)
    filter_world(paths)
    save_main_data(paths)
    save_daily_ret(paths)
    save_monthly_ret(paths)
    save_accounting_data(paths)
    save_output_files(paths)
    save_full_files_and_cleanup(paths, clear_interim=True)
