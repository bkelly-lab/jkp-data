"""Economic themes for the JKP characteristics: the source of truth for the themes files.

Description:
    Groups characteristics by economic meaning into subthemes, and subthemes into broad
    themes. Each characteristic belongs to exactly one subtheme and each subtheme to
    exactly one theme. Three signs, one per level: subtheme_sign (per characteristic)
    points the characteristic toward its subtheme's name, theme_sign (per subtheme)
    points the subtheme toward its theme's name, and factor_sign (per theme) points the
    theme toward higher returns, the long side of its factor. Each theme and subtheme also
    has a factor name, the usual name of its long-short factor. See README.md in this folder for the versioning rules and
    CHANGELOG_THEMES.md for what changed in each version.
Steps:
    1) Check the grouping: no characteristic in two subthemes, every subtheme in exactly
       one theme, and a description for every theme and subtheme.
    2) Write the mapping and the descriptions, both tagged with VERSION.
Output:
    src/jkp/data/resources/themes/themes.csv (characteristic, theme, subtheme, subtheme_sign,
    theme_sign, factor_sign, version) and
    src/jkp/data/resources/themes/theme_descriptions.csv (level, name, factor_name, theme,
    description, version), plus a console summary.

Run with: uv run python scripts/themes/economic_themes.py
"""

from pathlib import Path

import polars as pl

VERSION = "v1.0"  # text, so v1.10 never reads as v1.1
OUT_DIR = Path(__file__).resolve().parents[2] / "src" / "jkp" / "data" / "resources" / "themes"

# Themes: each subtheme belongs to exactly one -------------------------------------------
THEMES: dict[str, list[str]] = {
    "Value": ["Value", "Earnings yield", "Cash flow yield", "Dividend yield", "Equity duration"],
    "Profitability": ["Profitability", "Profit margins", "Cash flow profitability"],
    "Fundamental momentum": ["Profit growth", "Earnings surprise"],
    "Earnings quality": ["Accruals", "Special items", "Earnings persistence", "Tax burden"],
    "Investment": [
        "Asset growth",
        "Capex growth",
        "Working capital growth",
        "Operating payables growth",
        "Book equity growth",
        "Long-term investment growth",
        "Financial asset growth",
        "Operating growth",
    ],
    "Intangibles": ["R&D intensity", "Advertising intensity"],
    "Operating efficiency": [
        "Asset turnover",
        "Working capital efficiency",
        "Capital intensity",
        "Labor intensity",
        "Labor productivity growth",
    ],
    "Financing": [
        "Equity issuance",
        "Equity payout",
        "Debt issuance",
        "Liability growth",
        "External financing",
    ],
    "Leverage": [
        "Book leverage",
        "Market leverage",
        "Debt maturity",
        "Cost of debt",
        "Preferred stock",
        "Balance sheet liquidity",
        "Distress",
        "Financial constraints",
    ],
    "Risk": ["Volatility", "Lottery", "Beta", "Skewness", "Earnings volatility"],
    "Size": ["Size", "Firm age", "Share price", "Liquidity", "Share turnover"],
    "Short-term returns": ["Short-term returns"],
    "Medium-term returns": ["Returns 2-6", "Returns 7-12", "Medium-term returns"],
    "Long-term returns": ["Long-term returns"],
    "Seasonal returns": ["Seasonal returns"],
    "Composites": ["Composites"],
}

# Subthemes: characteristics in each subtheme, ordered by theme --------------------
SUBTHEMES: dict[str, list[str]] = {
    # Value
    "Value": ["be_me", "at_me", "sale_me", "be_mev", "bev_mev", "at_mev", "sale_mev", "aliq_mat"],
    "Earnings yield": [
        "ni_me",
        "nix_me",
        "ope_me",
        "ebit_me",
        "ebitda_me",
        "ebit_mev",
        "ebitda_mev",
        "gp_me",
        "gp_mev",
        "ival_me",
    ],
    "Cash flow yield": ["ocf_me", "ocf_mev", "fcf_me", "fcf_mev", "cop_me", "cop_mev"],
    "Dividend yield": [
        "div1m_me",
        "div3m_me",
        "div6m_me",
        "div12m_me",
        "div_me",
        "div_at",
        "div_ni",
    ],
    "Equity duration": ["eq_dur"],
    # Profitability
    "Profitability": [
        "gp_at",
        "gp_atl1",
        "gp_bev",
        "op_at",
        "op_atl1",
        "ope_be",
        "ope_bel1",
        "cop_at",
        "cop_atl1",
        "cop_bev",
        "ebit_at",
        "ebit_bev",
        "ebitda_at",
        "ebitda_bev",
        "ni_at",
        "ni_be",
        "nix_be",
        "niq_at",
        "niq_be",
        "qmj_prof",
        "fi_at",
        "fi_bev",
        "ebit_int",
        "ebitda_debt",
        "ebitda_ppen",
    ],
    "Profit margins": [
        "gp_sale",
        "ebitda_sale",
        "ebit_sale",
        "pi_sale",
        "ni_sale",
        "nix_sale",
        "profit_cl",
        "ni_emp",
    ],
    "Cash flow profitability": [
        "ocf_at",
        "ocf_be",
        "fcf_be",
        "fcf_ocf",
        "fcf_ppen",
        "ocf_cl",
        "fcf_sale",
        "ocf_debt",
        "ocf_sale",
    ],
    # Fundamental momentum
    "Profit growth": [
        "gp_gr1a",
        "ebitda_gr1a",
        "ebit_gr1a",
        "ope_gr1a",
        "ni_gr1a",
        "nix_gr1a",
        "gp_gr3a",
        "ebitda_gr3a",
        "ebit_gr3a",
        "ope_gr3a",
        "ni_gr3a",
        "nix_gr3a",
        "ocf_gr1a",
        "fcf_gr1a",
        "ocf_gr3a",
        "fcf_gr3a",
        "ocf_at_chg1",
        "ni_inc8q",
        "tax_gr1a",
        "tax_gr3a",
        "roe_ch5",
        "roa_ch5",
        "gpoa_ch5",
        "cfoa_ch5",
        "gmar_ch5",
        "qmj_growth",
        "dgp_dsale",
        "dsale_dsga",
    ],
    "Earnings surprise": ["niq_su", "saleq_su", "niq_at_chg1", "niq_be_chg1"],
    # Earnings quality
    "Accruals": ["oaccruals_at", "oaccruals_ni", "taccruals_at", "taccruals_ni"],
    "Special items": ["spi_at", "xido_at", "nri_at"],
    "Earnings persistence": ["ni_ar1"],
    "Tax burden": ["tax_pi", "pi_nix"],
    # Investment
    "Asset growth": [
        "at_gr1",
        "at_gr3",
        "noa_gr1a",
        "lnoa_gr1a",
        "ppeinv_gr1a",
        "nca_gr1",
        "nca_gr3",
        "ncoa_gr1a",
        "nncoa_gr1a",
        "oa_gr1a",
        "ncoa_gr3a",
        "nncoa_gr3a",
        "oa_gr3a",
        "noa_at",
        "ppeg_gr1a",
        "ppeg_gr3a",
        "intan_gr1a",
        "intan_gr3a",
        "dp_gr1a",
        "dp_gr3a",
    ],
    "Capex growth": ["capx_gr1", "capx_gr2", "capx_gr3", "capx_gr1a", "capx_gr3a", "capex_abn"],
    "Working capital growth": [
        "ca_gr1",
        "ca_gr3",
        "coa_gr1a",
        "coa_gr3a",
        "cowc_gr1a",
        "cowc_gr3a",
        "nwc_gr1a",
        "nwc_gr3a",
        "inv_gr1",
        "inv_gr1a",
        "inv_gr3a",
        "rec_gr1a",
        "rec_gr3a",
        "dsale_dinv",
        "dsale_drec",
    ],
    "Operating payables growth": [
        "col_gr1a",
        "ncol_gr1a",
        "ol_gr1a",
        "col_gr3a",
        "ncol_gr3a",
        "ol_gr3a",
        "ap_gr1a",
        "ap_gr3a",
        "txp_gr1a",
        "txp_gr3a",
        "txditc_gr1a",
        "txditc_gr3a",
    ],
    "Book equity growth": ["be_gr1", "be_gr3", "be_gr1a"],
    "Long-term investment growth": ["lti_gr1a", "lti_gr3a"],
    "Financial asset growth": ["sti_gr1a", "fna_gr1a", "fna_gr3a"],
    "Operating growth": [
        "sale_gr1",
        "sale_gr3",
        "saleq_gr1",
        "cogs_gr1",
        "sga_gr1",
        "opex_gr1",
        "cogs_gr3",
        "sga_gr3",
        "opex_gr3",
        "emp_gr1",
    ],
    # Intangibles
    "R&D intensity": ["rd_at", "rd_sale", "rd_me", "rd5_at"],
    "Advertising intensity": ["adv_sale"],
    # Operating efficiency
    "Asset turnover": ["at_turnover", "sale_bev", "opex_at", "sale_be"],
    "Working capital efficiency": [
        "inv_days",
        "rec_days",
        "ap_days",
        "cash_conversion",
        "inv_turnover",
        "rec_turnover",
        "ap_turnover",
        "inv_act",
    ],
    "Capital intensity": ["capx_at", "ppen_mev", "gp_ppen", "lt_ppen"],
    "Labor intensity": ["staff_sale", "sale_emp"],
    "Labor productivity growth": ["sale_emp_gr1"],
    # Financing
    "Equity issuance": [
        "chcsho_1m",
        "chcsho_3m",
        "chcsho_6m",
        "chcsho_12m",
        "eqnpo_1m",
        "eqnpo_3m",
        "eqnpo_6m",
        "eqnpo_12m",
        "eqis_at",
        "eqis_me",
        "eqnetis_at",
        "eqnetis_me",
        "eqnpo_at",
        "eqnpo_me",
        "eqis_gr1a",
        "eqis_gr3a",
        "eqnetis_gr1a",
        "eqnetis_gr3a",
        "eqnpo_gr1a",
        "eqnpo_gr3a",
    ],
    "Equity payout": [
        "eqbb_at",
        "eqbb_me",
        "eqpo_me",
        "eqbb_gr1a",
        "eqbb_gr3a",
        "eqpo_gr1a",
        "eqpo_gr3a",
        "div_gr1a",
        "div_gr3a",
    ],
    "Debt issuance": [
        "dbnetis_at",
        "dltnetis_at",
        "dstnetis_at",
        "debt_gr1",
        "debt_gr3",
        "debtst_gr1a",
        "debtst_gr3a",
        "debtlt_gr1a",
        "debtlt_gr3a",
        "dbnetis_mev",
        "dltnetis_mev",
        "dstnetis_mev",
        "dbnetis_gr1a",
        "dbnetis_gr3a",
        "dltnetis_gr1a",
        "dltnetis_gr3a",
        "dstnetis_gr1a",
        "dstnetis_gr3a",
        "fnl_gr1a",
        "fnl_gr3a",
        "nfna_gr1a",
        "nfna_gr3a",
    ],
    "Liability growth": ["lt_gr1", "lt_gr3", "ncl_gr1", "ncl_gr3", "cl_gr1", "cl_gr3"],
    "External financing": [
        "netis_at",
        "netis_mev",
        "fincf_at",
        "fincf_mev",
        "netis_gr1a",
        "netis_gr3a",
        "fincf_gr1a",
        "fincf_gr3a",
    ],
    # Leverage
    "Book leverage": [
        "debt_at",
        "debt_be",
        "debt_bev",
        "debtlt_be",
        "debtlt_bev",
        "at_be",
        "be_bev",
    ],
    "Market leverage": ["debt_me", "debt_mev", "debtlt_mev", "netdebt_me"],
    "Debt maturity": ["debtst_debt", "debtlt_debt", "debtst_bev", "debtst_mev", "cl_lt"],
    "Cost of debt": ["int_debt", "int_debtlt"],
    "Preferred stock": ["pstk_bev", "pstk_mev", "pstk_gr1", "pstk_gr3"],
    "Balance sheet liquidity": [
        "cash_at",
        "cash_bev",
        "cash_lt",
        "cash_cl",
        "caliq_cl",
        "ca_cl",
        "cash_me",
        "cash_mev",
        "aliq_at",
        "nwc_at",
        "sale_nwc",
        "cash_gr1a",
        "cash_gr3a",
        "rec_act",
        "tangibility",
    ],
    "Distress": ["o_score", "z_score"],
    "Financial constraints": ["kz_index"],
    # Risk
    "Volatility": [
        "rvol_21d",
        "rvol_252d",
        "rvolhl_21d",
        "ivol_capm_21d",
        "ivol_capm_252d",
        "ivol_capm_60m",
        "ivol_ff3_21d",
        "ivol_hxz4_21d",
    ],
    "Lottery": ["rmax1_21d", "rmax5_21d", "rmax5_rvol_21d"],
    "Beta": [
        "beta_60m",
        "beta_252d",
        "beta_21d",
        "beta_dimson_21d",
        "betadown_252d",
        "betabab_1260d",
        "corr_1260d",
    ],
    "Skewness": ["rskew_21d", "iskew_capm_21d", "iskew_ff3_21d", "iskew_hxz4_21d", "coskew_21d"],
    "Earnings volatility": [
        "earnings_variability",
        "ni_ivol",
        "roe_be_std",
        "roeq_be_std",
        "niq_saleq_std",
        "ocfq_saleq_std",
    ],
    # Size
    "Size": [
        "market_equity",
        "enterprise_value",
        "book_equity",
        "assets",
        "sales",
        "net_income",
        "intrinsic_value",
    ],
    "Firm age": ["age"],
    "Share price": ["prc"],
    "Liquidity": [
        "ami_126d",
        "dolvol_126d",
        "dolvol_var_126d",
        "turnover_var_126d",
        "bidaskhl_21d",
    ],
    "Share turnover": ["turnover_126d", "zero_trades_21d", "zero_trades_126d", "zero_trades_252d"],
    # Medium-term returns
    "Returns 2-6": ["ret_2_0", "ret_3_0", "ret_3_1", "ret_6_1", "ret_6_0", "resff3_6_1"],
    "Returns 7-12": ["ret_12_7"],
    "Medium-term returns": [
        "ret_12_1",
        "ret_9_1",
        "resff3_12_1",
        "ret_9_0",
        "ret_12_0",
        "ret_18_1",
        "ret_24_1",
        "prc_highprc_252d",
        "seas_1_1na",
    ],
    # Short-term returns
    "Short-term returns": ["ret_1_0"],
    # Long-term returns
    "Long-term returns": [
        "ret_60_12",
        "ret_60_36",
        "ret_48_12",
        "ret_36_12",
        "ret_24_12",
        "ret_36_1",
        "ret_48_1",
        "ret_60_1",
        "seas_2_5na",
        "seas_6_10na",
        "seas_11_15na",
        "seas_16_20na",
    ],
    # Seasonal returns
    "Seasonal returns": [
        "seas_1_1an",
        "seas_2_5an",
        "seas_6_10an",
        "seas_11_15an",
        "seas_16_20an",
    ],
    # Composites
    "Composites": ["qmj", "qmj_safety", "f_score", "mispricing_mgmt", "mispricing_perf"],
}

# Signs ------------------------------------------------------------------------------------
# Characteristics where a high value means less of what their subtheme's name says.
NEGATIVE_IN_SUBTHEME: set[str] = {
    # Balance sheet liquidity
    "sale_nwc",
    "rec_act",
    # Capital intensity
    "gp_ppen",
    "lt_ppen",
    # Debt issuance
    "nfna_gr1a",
    "nfna_gr3a",
    # Debt maturity (high = longer maturity)
    "debtst_debt",
    "debtst_bev",
    "debtst_mev",
    "cl_lt",
    # Distress
    "z_score",
    # Equity issuance (net payout is the opposite of net issuance)
    "eqnpo_1m",
    "eqnpo_3m",
    "eqnpo_6m",
    "eqnpo_12m",
    "eqnpo_at",
    "eqnpo_me",
    "eqnpo_gr1a",
    "eqnpo_gr3a",
    # Labor intensity
    "sale_emp",
    # Book leverage
    "be_bev",
    # Liquidity
    "ami_126d",
    "dolvol_var_126d",
    "turnover_var_126d",
    "bidaskhl_21d",
    # Share turnover
    "zero_trades_21d",
    "zero_trades_126d",
    "zero_trades_252d",
    # Working capital efficiency
    "inv_days",
    "rec_days",
    "ap_days",
    "cash_conversion",
    "inv_act",
    # Working capital growth (sales growth minus inventory or receivables growth)
    "dsale_dinv",
    "dsale_drec",
}

# Subthemes where a high value means less of what their theme's name says.
NEGATIVE_IN_THEME: set[str] = {
    "Equity duration",  # Value: long duration is the opposite of a high yield
    "Equity payout",  # Financing: payout flows to investors, issuance from them
    "Balance sheet liquidity",  # Leverage
    "Debt maturity",  # Leverage: longer maturity, less refinancing risk
    "Accruals",  # Earnings quality
    "Special items",  # Earnings quality
    "Capital intensity",  # Operating efficiency: more input per unit of output
    "Labor intensity",  # Operating efficiency
}

# Factor signs: +1 when a high value of the theme earns higher returns in the literature ----
# Mostly the majority direction of the theme's factors in Jensen, Kelly and Pedersen (2023).
FACTOR_SIGNS: dict[str, int] = {
    "Value": 1,
    "Profitability": 1,
    "Fundamental momentum": 1,
    "Earnings quality": 1,
    "Investment": -1,
    "Intangibles": 1,
    "Operating efficiency": 1,
    "Financing": -1,
    "Leverage": -1,
    "Risk": -1,
    "Size": -1,
    "Short-term returns": -1,
    "Medium-term returns": 1,
    "Long-term returns": -1,
    "Seasonal returns": 1,
    "Composites": 1,
}

# Descriptions -------------------------------------------------------------------------
THEME_DESCRIPTIONS: dict[str, str] = {
    "Value": "Price relative to fundamentals.",
    "Profitability": "How profitable the firm is.",
    "Fundamental momentum": "Recent improvement in fundamentals.",
    "Earnings quality": "How reliable reported earnings are.",
    "Investment": "Growth of the balance sheet and operations.",
    "Intangibles": "Spending on intangible capital, such as R&D and advertising.",
    "Operating efficiency": "How the firm uses its assets and labor.",
    "Financing": "Net flows to and from investors.",
    "Leverage": "Capital structure and financial strength.",
    "Risk": "Return and earnings risk.",
    "Size": "Firm size, age and trading activity.",
    "Short-term returns": "Last month's return.",
    "Medium-term returns": "Returns over the past two months to two years.",
    "Long-term returns": "Returns over multi-year windows, up to 20 years back.",
    "Seasonal returns": "Returns in the same calendar month in past years.",
    "Composites": "Multi-signal scores that span several themes.",
}

SUBTHEME_DESCRIPTIONS: dict[str, str] = {
    "Value": "Book equity, assets and sales relative to market value.",
    "Earnings yield": "Earnings, gross profit and intrinsic value relative to price.",
    "Cash flow yield": "Operating and free cash flow relative to market value.",
    "Dividend yield": "Dividends relative to price, assets and earnings.",
    "Equity duration": "How far in the future expected cash flows arrive.",
    "Profitability": "Profits relative to assets, book equity or PP&E.",
    "Profit margins": "Profits relative to sales or employees.",
    "Cash flow profitability": "Cash flow relative to assets, equity, sales or debt.",
    "Profit growth": "Changes in profits and profitability over one to five years.",
    "Earnings surprise": "Unexpected quarterly earnings and sales.",
    "Accruals": "Gap between accounting earnings and cash flow.",
    "Special items": "Non-recurring and extraordinary items.",
    "Earnings persistence": "How strongly earnings carry over from year to year.",
    "Tax burden": "Taxes relative to income.",
    "Asset growth": "Growth in total, operating and long-lived assets.",
    "Capex growth": "Growth in capital expenditure.",
    "Working capital growth": "Growth in inventory, receivables and other working capital.",
    "Operating payables growth": "Growth in payables to suppliers, accrued expenses and taxes: liabilities that arise from operations.",
    "Book equity growth": "Growth in book equity.",
    "Long-term investment growth": "Growth in long-term investments.",
    "Financial asset growth": "Growth in short-term investments and financial assets.",
    "Operating growth": "Growth in sales, costs and employees.",
    "R&D intensity": "R&D spending relative to assets, sales or price.",
    "Advertising intensity": "Advertising spending relative to sales.",
    "Asset turnover": "Sales and operating costs relative to assets.",
    "Working capital efficiency": "Inventory, receivable and payable days.",
    "Capital intensity": "PP&E and capex relative to assets, price or profit.",
    "Labor intensity": "Staff costs and sales per employee.",
    "Labor productivity growth": "Growth in sales per employee.",
    "Equity issuance": "Net share issuance, from market and accounting data.",
    "Equity payout": "Buybacks and total payout to shareholders.",
    "Debt issuance": "Net issuance and growth of debt.",
    "Liability growth": "Growth in total, current and non-current liabilities, including debt.",
    "External financing": "Total net financing from debt and equity.",
    "Book leverage": "Debt relative to book assets, book equity or book enterprise value.",
    "Market leverage": "Debt relative to market equity or market enterprise value.",
    "Debt maturity": "Short-term versus long-term debt.",
    "Cost of debt": "Interest expense relative to debt.",
    "Preferred stock": "Preferred stock outstanding and its growth.",
    "Balance sheet liquidity": "Cash and current assets relative to liabilities or assets.",
    "Distress": "Bankruptcy-risk scores.",
    "Financial constraints": "Kaplan–Zingales index.",
    "Volatility": "Total and idiosyncratic return volatility.",
    "Lottery": "Largest daily returns over the past month.",
    "Beta": "Sensitivity to and correlation with the market.",
    "Skewness": "Asymmetry of daily returns.",
    "Earnings volatility": "Volatility of earnings and cash flow.",
    "Size": "Market value, book value, assets and sales.",
    "Firm age": "Months since the firm first appears in the data.",
    "Share price": "Share price level.",
    "Liquidity": "Trading volume, illiquidity and bid-ask spreads.",
    "Share turnover": "Share turnover and days without trading.",
    "Returns 2-6": "Returns over windows of up to six months, mostly skipping the last month.",
    "Returns 7-12": "Returns over months 7–12.",
    "Medium-term returns": "Returns over windows of 9 months or longer, and the 52-week high.",
    "Short-term returns": "Last month's return.",
    "Long-term returns": "Returns over multi-year windows, up to 20 years back.",
    "Seasonal returns": "Past returns in the same calendar month.",
    "Composites": "Scores that combine signals from several themes: QMJ, QMJ safety, F-score and the two Stambaugh–Yuan mispricing factors.",
}

# Factor names. By default a factor is named after its theme or subtheme, with "Low" in
# front when it goes long low values (factor_sign, times theme_sign for a subtheme, is -1).
# Exceptions: established names for past returns, and plain antonyms where they read better.
FACTOR_NAMES: dict[str, str] = {
    "Size": "Small",
    "Firm age": "Young",
    "Liquidity": "Illiquidity",
    "Equity duration": "Short duration",
    "Earnings volatility": "Stable earnings",
    "Short-term returns": "Short-term reversal",
    "Medium-term returns": "Momentum",
    "Returns 2-6": "Momentum 2-6",
    "Returns 7-12": "Momentum 7-12",
    "Long-term returns": "Long-term reversal",
    "Seasonal returns": "Seasonality",
}


def validate() -> None:
    """Check that the grouping is a two-level partition with a description for every group."""
    listed = [s for subs in THEMES.values() for s in subs]
    assert sorted(listed) == sorted(SUBTHEMES), "each subtheme must be under exactly one theme"
    members = [c for chars in SUBTHEMES.values() for c in chars]
    dup = sorted({c for c in members if members.count(c) > 1})
    assert not dup, f"characteristics in more than one subtheme: {dup}"
    assert set(members) >= NEGATIVE_IN_SUBTHEME, "unknown characteristics in NEGATIVE_IN_SUBTHEME"
    assert SUBTHEMES.keys() >= NEGATIVE_IN_THEME, "unknown subthemes in NEGATIVE_IN_THEME"
    assert FACTOR_SIGNS.keys() == THEMES.keys(), "factor signs do not match themes"
    assert (THEMES.keys() | SUBTHEMES.keys()) >= FACTOR_NAMES.keys(), (
        "unknown names in FACTOR_NAMES"
    )
    assert THEME_DESCRIPTIONS.keys() == THEMES.keys(), "theme descriptions do not match"
    assert SUBTHEME_DESCRIPTIONS.keys() == SUBTHEMES.keys(), "subtheme descriptions do not match"


def themes_frame() -> pl.DataFrame:
    """One row per characteristic: characteristic, theme, subtheme, the three signs, version."""
    rows = []
    for theme, subs in THEMES.items():
        for sub in subs:
            for c in SUBTHEMES[sub]:
                sub_sign = -1 if c in NEGATIVE_IN_SUBTHEME else 1
                rows.append(
                    {
                        "characteristic": c,
                        "theme": theme,
                        "subtheme": sub,
                        "subtheme_sign": sub_sign,
                        "theme_sign": -1 if sub in NEGATIVE_IN_THEME else 1,
                        "factor_sign": FACTOR_SIGNS[theme],
                        "version": VERSION,
                    }
                )
    return pl.DataFrame(rows)


def factor_name(level: str, name: str, theme: str) -> str:
    """Name of the long-short factor of a theme or subtheme (see FACTOR_NAMES)."""
    if name in FACTOR_NAMES:
        return FACTOR_NAMES[name]
    sign = FACTOR_SIGNS[theme] * (-1 if level == "subtheme" and name in NEGATIVE_IN_THEME else 1)
    return name if sign > 0 else "Low " + name[0].lower() + name[1:]


def descriptions_frame() -> pl.DataFrame:
    """One row per theme, then its subthemes: level, name, factor_name, theme, description, version."""
    rows = []
    for theme, subs in THEMES.items():
        rows.append(("theme", theme, theme, THEME_DESCRIPTIONS[theme]))
        rows += [("subtheme", s, theme, SUBTHEME_DESCRIPTIONS[s]) for s in subs]
    return pl.DataFrame(
        [(lvl, n, factor_name(lvl, n, t), t, d) for lvl, n, t, d in rows],
        schema=["level", "name", "factor_name", "theme", "description"],
        orient="row",
    ).with_columns(version=pl.lit(VERSION))


def main() -> None:
    validate()
    themes = themes_frame()
    themes.write_csv(OUT_DIR / "themes.csv")
    descriptions_frame().write_csv(OUT_DIR / "theme_descriptions.csv")
    print(
        f"Themes {VERSION}: {themes.height} characteristics in {len(SUBTHEMES)} subthemes, "
        f"{len(THEMES)} themes. Written to {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
