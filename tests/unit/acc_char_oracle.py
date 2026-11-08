"""Plain-Python reference for the accounting characteristics (issue #303, step 5).

The stage's characteristics computed again, one firm at a time, from the SAS that defines them:
GlobalFactors/accounting_chars.sas and char_macros.sas in bkelly-lab/ReplicationCrisis (commit
1b7a9be). Each section cites the SAS lines it follows; nothing here calls aux_functions.py.
test_acc_chars_oracle.py runs it on the synthetic stage inputs and compares it with the stage's
output, characteristic by characteristic.

How the SAS data steps map to Python:
    - A firm is the monthly rows of one (gvkey, curcd) in acc_std_ann or acc_std_qtr, sorted by
      datadate, held as columns: a dict of column name to list. Row i has count i + 1
      (accounting_chars.sas:628-638).
    - Missing is None. Arithmetic with a missing operand is missing, and so is division by zero
      (SAS sets it missing with a note). sas_sum, sas_mean, sas_std, sas_min and sas_max skip
      missing operands, as SAS's functions do.
    - In SAS comparisons missing is below every number, so `x <= 0` is true for a missing x.
      lt, le, gt, ge and eq follow that rule; use them instead of Python's operators.
    - lag(f, name, i, k) is the value k rows before row i in the same firm, missing before the
      first row. SAS's lag queue runs over the whole data set, so a lagk the count guards miss
      reaches the previous firm's rows; the oracle never crosses firms.
    - A lag inside a guard such as `if count <= 12 or lag12(x) <= 0` is a plain lag. If SAS
      stops evaluating the `or` once the count test is true, that lag's queue is fed only on
      later rows and returns another row's value; the oracle, like the stage, uses the rule the
      guard states.
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
from collections.abc import Callable
from itertools import groupby

import polars as pl

Value = float | None
Firm = dict[str, list]
Row = dict[str, object]


def add(*xs: Value) -> Value:
    """a + b + ...: missing if any operand is missing."""
    return None if any(x is None for x in xs) else math.fsum(xs)  # type: ignore[arg-type]


def sub(a: Value, b: Value) -> Value:
    return None if a is None or b is None else a - b


def mul(*xs: Value) -> Value:
    """a * b * ...: missing if any operand is missing."""
    if any(x is None for x in xs):
        return None
    return math.prod(xs)  # type: ignore[arg-type]


def div(a: Value, b: Value) -> Value:
    """a / b: missing if either is missing or b is 0."""
    return None if a is None or b is None or b == 0 else a / b


def _present(xs: tuple[Value, ...]) -> list[float]:
    return [x for x in xs if x is not None]


def sas_sum(*xs: Value) -> Value:
    """SAS sum(): the sum of the non-missing operands, missing if all are missing."""
    p = _present(xs)
    return math.fsum(p) if p else None


def sas_mean(*xs: Value) -> Value:
    """SAS mean(): the mean of the non-missing operands, missing if all are missing."""
    p = _present(xs)
    return math.fsum(p) / len(p) if p else None


def sas_std(*xs: Value) -> Value:
    """SAS std(): sample standard deviation (n - 1) of the non-missing operands; needs two."""
    p = _present(xs)
    if len(p) < 2:
        return None
    m = math.fsum(p) / len(p)
    return math.sqrt(math.fsum((x - m) ** 2 for x in p) / (len(p) - 1))


def sas_min(*xs: Value) -> Value:
    p = _present(xs)
    return min(p) if p else None


def sas_max(*xs: Value) -> Value:
    p = _present(xs)
    return max(p) if p else None


def sas_n(*xs: Value) -> int:
    """SAS n(): the number of non-missing operands."""
    return len(_present(xs))


def sas_log(x: Value) -> Value:
    """SAS log(): missing for a missing or non-positive argument."""
    return None if x is None or x <= 0 else math.log(x)


def sas_abs(x: Value) -> Value:
    return None if x is None else abs(x)


def _key(x: Value) -> float:
    return -math.inf if x is None else x


def lt(a: Value, b: Value) -> bool:
    """SAS a < b: missing is below every number, and missing equals missing."""
    return _key(a) < _key(b) if a is not None or b is not None else False


def le(a: Value, b: Value) -> bool:
    return lt(a, b) or eq(a, b)


def gt(a: Value, b: Value) -> bool:
    return lt(b, a)


def ge(a: Value, b: Value) -> bool:
    return le(b, a)


def eq(a: Value, b: Value) -> bool:
    """SAS a = b, where missing equals missing."""
    return (a is None and b is None) or (a is not None and b is not None and a == b)


def flag(x: bool) -> float:
    """A SAS comparison used as a number: 1 if true, 0 if false."""
    return 1.0 if x else 0.0


def rows(f: Firm) -> range:
    return range(len(f["datadate"]))


def count(i: int) -> int:
    """SAS's count for row i: 1 on the firm's first row (accounting_chars.sas:631-638)."""
    return i + 1


def lag(f: Firm, name: str, i: int, k: int) -> Value:
    """Column name k rows before row i in the same firm; missing before the firm's first row."""
    return f[name][i - k] if i >= k else None


def column(f: Firm, value: Callable[[int], Value]) -> list[Value]:
    """One new column: value(i) for each row i, in row order."""
    return [value(i) for i in rows(f)]


def size_measures(f: Firm) -> None:
    """accounting_chars.sas:645-649. Converted to USD after the expansion (lines 1092-1099)."""
    f["assets"] = list(f["at_x"])
    f["sales"] = list(f["sale_x"])
    f["book_equity"] = list(f["be_x"])
    f["net_income"] = list(f["ni_x"])


GROWTH_VARS = [
    "at_x",
    "ca_x",
    "nca_x",
    "lt",
    "cl_x",
    "ncl_x",
    "be_x",
    "pstk_x",
    "debt_x",
    "sale_x",
    "cogs",
    "xsga",
    "opex_x",
    "capx",
    "invt",
]
CH_ASSET_VARS = [
    "che",
    "invt",
    "rect",
    "ppegt",
    "ivao",
    "ivst",
    "intan",
    "dlc",
    "ap",
    "txp",
    "dltt",
    "txditc",
    "coa_x",
    "col_x",
    "cowc_x",
    "ncoa_x",
    "ncol_x",
    "nncoa_x",
    "oa_x",
    "ol_x",
    "fna_x",
    "fnl_x",
    "nfna_x",
    "gp_x",
    "ebitda_x",
    "ebit_x",
    "ope_x",
    "ni_x",
    "nix_x",
    "dp",
    "fincf_x",
    "ocf_x",
    "fcf_x",
    "nwc_x",
    "eqnetis_x",
    "dltnetis_x",
    "dstnetis_x",
    "dbnetis_x",
    "netis_x",
    "eqnpo_x",
    "txt",
    "eqbb_x",
    "eqis_x",
    "div_x",
    "eqpo_x",
    "capx",
    "be_x",
]


def growth(f: Firm) -> None:
    """
    Description:
        1- and 3-year growth (var_growth, char_macros.sas:330-336; accounting_chars.sas:651-669).
    Steps:
        1) var / lag(var) - 1 over 12 and 36 months, with '_x' dropped from the name.
        2) Missing if count <= horizon or the lag is <= 0.
    Output:
        <var>_gr1 and <var>_gr3 columns.
    """
    for h in (12, 36):
        for var in GROWTH_VARS:
            x = f[var]
            f[f"{var.replace('_x', '')}_gr{h // 12}"] = [
                None
                if count(i) <= h or le(lag(f, var, i, h), 0)
                else sub(div(x[i], lag(f, var, i, h)), 1)
                for i in rows(f)
            ]


def change_to_assets(f: Firm) -> None:
    """
    Description:
        1- and 3-year change scaled by assets (chg_to_assets, char_macros.sas:339-346;
        accounting_chars.sas:671-696).
    Steps:
        1) (var - lag(var)) / at_x over 12 and 36 months, '_x' dropped from the name plus 'a'.
        2) Missing if count <= horizon or at_x <= 0.
    Output:
        <var>_gr1a and <var>_gr3a columns.
    """
    for h in (12, 36):
        for var in CH_ASSET_VARS:
            x, at = f[var], f["at_x"]
            f[f"{var.replace('_x', '')}_gr{h // 12}a"] = [
                None if count(i) <= h or le(at[i], 0) else div(sub(x[i], lag(f, var, i, h)), at[i])
                for i in rows(f)
            ]


RATIOS = [
    ("capx_at", "capx", "at_x"),
    ("rd_at", "xrd", "at_x"),
    ("spi_at", "spi", "at_x"),
    ("xido_at", "xido_x", "at_x"),
]
MARGINS_RETURNS = [
    ("gp_sale", "gp_x", "sale_x"),
    ("ebitda_sale", "ebitda_x", "sale_x"),
    ("ebit_sale", "ebit_x", "sale_x"),
    ("pi_sale", "pi_x", "sale_x"),
    ("ni_sale", "ni_x", "sale_x"),
    ("nix_sale", "ni", "sale_x"),
    ("ocf_sale", "ocf_x", "sale_x"),
    ("fcf_sale", "fcf_x", "sale_x"),
    ("gp_at", "gp_x", "at_x"),
    ("ebitda_at", "ebitda_x", "at_x"),
    ("ebit_at", "ebit_x", "at_x"),
    ("fi_at", "fi_x", "at_x"),
    ("cop_at", "cop_x", "at_x"),
    ("ni_at", "ni_x", "at_x"),
    ("ope_be", "ope_x", "be_x"),
    ("ni_be", "ni_x", "be_x"),
    ("nix_be", "nix_x", "be_x"),
    ("ocf_be", "ocf_x", "be_x"),
    ("fcf_be", "fcf_x", "be_x"),
    ("gp_bev", "gp_x", "bev_x"),
    ("ebitda_bev", "ebitda_x", "bev_x"),
    ("ebit_bev", "ebit_x", "bev_x"),
    ("fi_bev", "fi_x", "bev_x"),
    ("cop_bev", "cop_x", "bev_x"),
    ("gp_ppen", "gp_x", "ppent"),
    ("ebitda_ppen", "ebitda_x", "ppent"),
    ("fcf_ppen", "fcf_x", "ppent"),
    ("fincf_at", "fincf_x", "at_x"),
    ("netis_at", "netis_x", "at_x"),
    ("eqnetis_at", "eqnetis_x", "at_x"),
    ("eqis_at", "eqis_x", "at_x"),
    ("dbnetis_at", "dbnetis_x", "at_x"),
    ("dltnetis_at", "dltnetis_x", "at_x"),
    ("dstnetis_at", "dstnetis_x", "at_x"),
    ("eqnpo_at", "eqnpo_x", "at_x"),
    ("eqbb_at", "eqbb_x", "at_x"),
    ("div_at", "div_x", "at_x"),
    ("oaccruals_at", "oacc_x", "at_x"),
    ("taccruals_at", "tacc_x", "at_x"),
]


def _ratios(f: Firm, specs: list[tuple[str, str, str]]) -> None:
    for name, num, den in specs:
        f[name] = [div(f[num][i], f[den][i]) for i in rows(f)]


def investment_and_nonrecurring(f: Firm) -> None:
    """
    Description:
        Investment and non-recurring items to assets (accounting_chars.sas:698-706).
    Steps:
        1) capx, xrd, spi and xido_x over at_x; then (spi + xido_x) / at_x.
    Output:
        capx_at, rd_at, spi_at, xido_at, nri_at columns.
    """
    _ratios(f, RATIOS)
    f["nri_at"] = column(f, lambda i: div(add(f["spi"][i], f["xido_x"][i]), f["at_x"][i]))


def profitability(f: Firm) -> None:
    """
    Description:
        Margins, returns on assets, equity, invested capital and PP&E, issuance and payout
        ratios, and accruals (accounting_chars.sas:708-768).
    Steps:
        1) Each ratio is numerator / denominator, missing if either is missing or the
           denominator is 0.
        2) The accrual-to-earnings ratios divide by abs(nix_x).
        3) noa_at = noa_x / lag12(at_x), missing if count <= 12 or the lag is <= 0.
    Output:
        The ratio columns, oaccruals_ni, taccruals_ni and noa_at.
    """
    _ratios(f, MARGINS_RETURNS)
    f["oaccruals_ni"] = column(f, lambda i: div(f["oacc_x"][i], sas_abs(f["nix_x"][i])))
    f["taccruals_ni"] = column(f, lambda i: div(f["tacc_x"][i], sas_abs(f["nix_x"][i])))
    f["noa_at"] = column(f, lambda i: div(f["noa_x"][i], lag(f, "at_x", i, 12)))
    f["noa_at"] = [
        None if count(i) <= 12 or le(lag(f, "at_x", i, 12), 0) else v
        for i, v in enumerate(f["noa_at"])
    ]


LEVERAGE_SOUNDNESS = [
    ("be_bev", "be_x", "bev_x"),
    ("debt_bev", "debt_x", "bev_x"),
    ("cash_bev", "che", "bev_x"),
    ("pstk_bev", "pstk_x", "bev_x"),
    ("debtlt_bev", "dltt", "bev_x"),
    ("debtst_bev", "dlc", "bev_x"),
    ("int_debt", "xint", "debt_x"),
    ("int_debtlt", "xint", "dltt"),
    ("ebitda_debt", "ebitda_x", "debt_x"),
    ("profit_cl", "ebitda_x", "cl_x"),
    ("ocf_cl", "ocf_x", "cl_x"),
    ("ocf_debt", "ocf_x", "debt_x"),
    ("cash_lt", "che", "lt"),
    ("inv_act", "invt", "act"),
    ("rec_act", "rect", "act"),
    ("debtst_debt", "dlc", "debt_x"),
    ("cl_lt", "cl_x", "lt"),
    ("debtlt_debt", "dltt", "debt_x"),
    ("lt_ppen", "lt", "ppent"),
    ("debtlt_be", "dltt", "be_x"),
    ("opex_at", "opex_x", "at_x"),
    ("nwc_at", "nwc_x", "at_x"),
]
SOLVENCY = [
    ("debt_at", "debt_x", "at_x"),
    ("debt_be", "debt_x", "be_x"),
    ("ebit_int", "ebit_x", "xint"),
]


def leverage_and_soundness(f: Firm) -> None:
    """
    Description:
        Book leverage, financial soundness and solvency ratios (accounting_chars.sas:770-801).
    Steps:
        1) Plain ratios, in SAS order, with fcf_ocf (fcf_x / ocf_x) computed only where
           ocf_x > 0 (missing otherwise) between nwc_at and debt_at.
    Output:
        The ratio columns, including fcf_ocf.
    """
    _ratios(f, LEVERAGE_SOUNDNESS)
    f["fcf_ocf"] = column(
        f, lambda i: div(f["fcf_x"][i], f["ocf_x"][i]) if gt(f["ocf_x"][i], 0) else None
    )
    _ratios(f, SOLVENCY)


def mean_year(f: Firm, var: str, i: int) -> Value:
    """mean(var, lag12(var)): the average of the two values that are present."""
    return sas_mean(f[var][i], lag(f, var, i, 12))


def liquidity(f: Firm) -> None:
    """
    Description:
        Days ratios, cash conversion and balance-sheet liquidity (accounting_chars.sas:803-820).
    Steps:
        1) inv_days, rec_days, ap_days = mean(var, lag12) / cogs or sale_x * 365, missing for
           count <= 12.
        2) cash_conversion = inv_days + rec_days - ap_days, missing if negative (or missing).
        3) Where cl_x > 0: che, caliq_x and ca_x over cl_x.
    Output:
        inv_days, rec_days, ap_days, cash_conversion, cash_cl, caliq_cl, ca_cl.
    """
    days = [("inv_days", "invt", "cogs"), ("rec_days", "rect", "sale_x"), ("ap_days", "ap", "cogs")]
    for name, var, den in days:
        f[name] = [
            None if count(i) <= 12 else mul(div(mean_year(f, var, i), f[den][i]), 365)
            for i in rows(f)
        ]
    f["cash_conversion"] = column(
        f, lambda i: sub(add(f["inv_days"][i], f["rec_days"][i]), f["ap_days"][i])
    )
    f["cash_conversion"] = [None if lt(v, 0) else v for v in f["cash_conversion"]]
    for name, num in [("cash_cl", "che"), ("caliq_cl", "caliq_x"), ("ca_cl", "ca_x")]:
        f[name] = [div(f[num][i], f["cl_x"][i]) if gt(f["cl_x"][i], 0) else None for i in rows(f)]


def activity(f: Firm) -> None:
    """
    Description:
        Turnover ratios (accounting_chars.sas:822-832).
    Steps:
        1) cogs / mean(invt), sale_x / mean(at_x), sale_x / mean(rect) and
           (cogs + dif12(invt)) / mean(ap), where mean(var) averages var and its 12-month lag.
        2) Missing for count <= 12.
    Output:
        inv_turnover, at_turnover, rec_turnover, ap_turnover.
    """
    specs = {
        "inv_turnover": lambda i: div(f["cogs"][i], mean_year(f, "invt", i)),
        "at_turnover": lambda i: div(f["sale_x"][i], mean_year(f, "at_x", i)),
        "rec_turnover": lambda i: div(f["sale_x"][i], mean_year(f, "rect", i)),
        "ap_turnover": lambda i: div(
            add(f["cogs"][i], sub(f["invt"][i], lag(f, "invt", i, 12))), mean_year(f, "ap", i)
        ),
    }
    for name, value in specs.items():
        f[name] = [None if count(i) <= 12 else value(i) for i in rows(f)]


def misc_ratios(f: Firm) -> None:
    """
    Description:
        Miscellaneous ratios (accounting_chars.sas:834-847 region end of section).
    Steps:
        1) xad, xlr, xrd, over sale_x; sale_x over bev_x and be_x.
        2) div_ni = div_x / nix_x only where coalesce(nix_x, ni_x) > 0.
        3) sale_nwc = sale_x / nwc_x where nwc_x > 0; tax_pi = txt / pi_x where pi_x > 0.
    Output:
        adv_sale, staff_sale, sale_bev, rd_sale, sale_be, div_ni, sale_nwc, tax_pi.
    """
    _ratios(
        f,
        [
            ("adv_sale", "xad", "sale_x"),
            ("staff_sale", "xlr", "sale_x"),
            ("sale_bev", "sale_x", "bev_x"),
            ("rd_sale", "xrd", "sale_x"),
            ("sale_be", "sale_x", "be_x"),
        ],
    )
    f["div_ni"] = column(
        f,
        lambda i: (
            div(f["div_x"][i], f["nix_x"][i])
            if gt(f["nix_x"][i] if f["nix_x"][i] is not None else f["ni_x"][i], 0)
            else None
        ),
    )
    f["sale_nwc"] = column(
        f, lambda i: div(f["sale_x"][i], f["nwc_x"][i]) if gt(f["nwc_x"][i], 0) else None
    )
    f["tax_pi"] = column(
        f, lambda i: div(f["txt"][i], f["pi_x"][i]) if gt(f["pi_x"][i], 0) else None
    )


def _lastq(f: Firm, name: str, i: int, qtrs: int) -> tuple[Value, ...]:
    """Description:
        The arguments of apply_to_lastq (accounting_chars.sas:609-616).
    Steps:
        1) Row i's value of name and its lags 3, 6, ..., 3 * (qtrs - 1); missing before the start.
    Output:
        Tuple of qtrs values.
    """
    return tuple(lag(f, name, i, 3 * k) for k in range(qtrs))


def _lasty(f: Firm, name: str, i: int, yrs: int) -> tuple[Value, ...]:
    """Description:
        The arguments of apply_to_lasty (accounting_chars.sas:618-625).
    Steps:
        1) Row i's value of name and its lags 12, 24, ..., 12 * (yrs - 1); missing before the start.
    Output:
        Tuple of yrs values.
    """
    return tuple(lag(f, name, i, 12 * k) for k in range(yrs))


def _chg_to_exp(f: Firm, var: str, out: str) -> None:
    """Description:
        Change to expectations (char_macros.sas:360-368).
    Steps:
        1) expect = mean of the values 12 and 24 rows back; out = var / expect - 1.
        2) Missing if count <= 24 or expect <= 0.
    Output:
        Adds column out.
    """
    expect = column(f, lambda i: div(add(lag(f, var, i, 12), lag(f, var, i, 24)), 2.0))
    ratio = column(f, lambda i: sub(div(f[var][i], expect[i]), 1.0))
    f[out] = column(f, lambda i: None if count(i) <= 24 or le(expect[i], 0) else ratio[i])


def _standardized_unexpected(f: Firm, var: str, qtrs: int, qtrs_min: int) -> None:
    """Description:
        Standardized unexpected realization (char_macros.sas:372-387).
    Steps:
        1) chg = var - lag12(var); mean and std of chg over the last qtrs quarters (lags 0, 3, ...).
        2) If the non-missing count of chg is <= qtrs_min, the mean and std are missing.
        3) <var>_su = (var - (lag12(var) + lag3(mean))) / lag3(std); missing if count <=
           12 + 3 * qtrs.
    Output:
        Adds column <var>_su (plus helper columns __chg, __chg_mean, __chg_std).
    """
    f["__chg"] = column(f, lambda i: sub(f[var][i], lag(f, var, i, 12)))
    mean = column(f, lambda i: sas_mean(*_lastq(f, "__chg", i, qtrs)))
    std = column(f, lambda i: sas_std(*_lastq(f, "__chg", i, qtrs)))
    n = column(f, lambda i: sas_n(*_lastq(f, "__chg", i, qtrs)))
    f["__chg_mean"] = [None if n[i] <= qtrs_min else mean[i] for i in range(len(n))]
    f["__chg_std"] = [None if n[i] <= qtrs_min else std[i] for i in range(len(n))]
    num = column(f, lambda i: sub(f[var][i], add(lag(f, var, i, 12), lag(f, "__chg_mean", i, 3))))
    su = column(f, lambda i: div(num[i], lag(f, "__chg_std", i, 3)))
    f[f"{var}_su"] = column(f, lambda i: None if count(i) <= 12 + qtrs * 3 else su[i])


def _volq(f: Firm, name: str, var: str, qtrs: int, qtrs_min: int) -> None:
    """Description:
        Volatility of quarterly data (char_macros.sas:390-396).
    Steps:
        1) std of var over the last qtrs quarters (lags 0, 3, ...), n = their non-missing count.
        2) Missing if count <= (qtrs - 1) * 3 or n < qtrs_min.
    Output:
        Adds column name.
    """
    n = column(f, lambda i: sas_n(*_lastq(f, var, i, qtrs)))
    std = column(f, lambda i: sas_std(*_lastq(f, var, i, qtrs)))
    f[name] = column(f, lambda i: None if count(i) <= (qtrs - 1) * 3 or n[i] < qtrs_min else std[i])


def _vola(f: Firm, name: str, var: str, yrs: int, yrs_min: int) -> None:
    """Description:
        Volatility of annual data (char_macros.sas:399-405).
    Steps:
        1) std of var over the last yrs years (lags 0, 12, ...), n = their non-missing count.
        2) Missing if count <= (yrs - 1) * 12 or n < yrs_min.
    Output:
        Adds column name.
    """
    n = column(f, lambda i: sas_n(*_lasty(f, var, i, yrs)))
    std = column(f, lambda i: sas_std(*_lasty(f, var, i, yrs)))
    f[name] = column(f, lambda i: None if count(i) <= (yrs - 1) * 12 or n[i] < yrs_min else std[i])


def cash_and_employees(f: Firm) -> None:
    """Description:
        Cash to assets and the employee variables (accounting_chars.sas:849-861).
    Steps:
        1) cash_at = che / at_x, missing if at_x <= 0.
        2) ni_emp and sale_emp, missing if emp <= 0.
        3) sale_emp_gr1 = sale_emp / lag12(sale_emp) - 1; missing if count <= 12 or the lag <= 0.
        4) emp_gr1 = (emp - lag12(emp)) / (0.5 emp + 0.5 lag12(emp)); missing if count <= 12, the
           growth is 0 or the denominator is 0.
    Output:
        Adds cash_at, ni_emp, sale_emp, sale_emp_gr1, emp_gr1.
    """
    f["cash_at"] = column(
        f, lambda i: None if le(f["at_x"][i], 0) else div(f["che"][i], f["at_x"][i])
    )
    f["ni_emp"] = column(
        f, lambda i: None if le(f["emp"][i], 0) else div(f["ni_x"][i], f["emp"][i])
    )
    f["sale_emp"] = column(
        f, lambda i: None if le(f["emp"][i], 0) else div(f["sale_x"][i], f["emp"][i])
    )
    f["sale_emp_gr1"] = column(
        f,
        lambda i: (
            None
            if count(i) <= 12 or le(lag(f, "sale_emp", i, 12), 0)
            else sub(div(f["sale_emp"][i], lag(f, "sale_emp", i, 12)), 1.0)
        ),
    )

    def emp_den(i: int) -> Value:
        return add(mul(0.5, f["emp"][i]), mul(0.5, lag(f, "emp", i, 12)))

    emp_gr1 = column(f, lambda i: div(sub(f["emp"][i], lag(f, "emp", i, 12)), emp_den(i)))
    f["emp_gr1"] = column(
        f,
        lambda i: None if count(i) <= 12 or eq(emp_gr1[i], 0) or eq(emp_den(i), 0) else emp_gr1[i],
    )


def consecutive_earnings_increases(f: Firm) -> None:
    """Description:
        Number of consecutive earnings increases (accounting_chars.sas:863-879).
    Steps:
        1) ni_inc = 1 if ni_x > lag12(ni_x) else 0; missing if either is missing.
        2) Walk lags 0, 3, ..., 21 of ni_inc: add 1 while each is 1, stop at the first that is not.
        3) n_ni_inc = number of non-missing ni_inc over those lags.
        4) ni_inc8q is missing if ni_inc is missing, n_ni_inc != 8 or count < 33.
    Output:
        Adds ni_inc and ni_inc8q.
    """
    f["ni_inc"] = column(
        f,
        lambda i: (
            None
            if f["ni_x"][i] is None or lag(f, "ni_x", i, 12) is None
            else flag(gt(f["ni_x"][i], lag(f, "ni_x", i, 12)))
        ),
    )

    def run(i: int) -> float:
        total, no_decrease = 0.0, True
        for q in range(8):
            if eq(lag(f, "ni_inc", i, 3 * q), 1) and no_decrease:
                total += 1
            else:
                no_decrease = False
        return total

    f["ni_inc8q"] = column(
        f,
        lambda i: (
            None
            if f["ni_inc"][i] is None or sas_n(*_lastq(f, "ni_inc", i, 8)) != 8 or count(i) < 33
            else run(i)
        ),
    )


def asset_scaled_changes(f: Firm) -> None:
    """Description:
        One-year changes scaled by lagged or average assets and two-year capex growth
        (accounting_chars.sas:881-906, var_growth in char_macros.sas:330-336).
    Steps:
        1) noa_gr1a and ppeinv_gr1a = (v - lag12(v)) / lag12(at_x); missing if count <= 12 or
           lag12(at_x) <= 0.
        2) lnoa_gr1a = (v - lag12(v)) / (at_x + lag12(at_x)); missing if count <= 12 or the
           denominator <= 0.
        3) capx_gr2 = capx / lag24(capx) - 1; missing if count <= 24 or the lag <= 0.
    Output:
        Adds noa_gr1a, ppeinv_gr1a, lnoa_gr1a, capx_gr2.
    """
    for var, name in (("noa_x", "noa_gr1a"), ("ppeinv_x", "ppeinv_gr1a")):
        f[name] = column(
            f,
            lambda i, var=var: (
                None
                if count(i) <= 12 or le(lag(f, "at_x", i, 12), 0)
                else div(sub(f[var][i], lag(f, var, i, 12)), lag(f, "at_x", i, 12))
            ),
        )

    def avg_den(i: int) -> Value:
        return add(f["at_x"][i], lag(f, "at_x", i, 12))

    f["lnoa_gr1a"] = column(
        f,
        lambda i: (
            None
            if count(i) <= 12 or le(avg_den(i), 0)
            else div(sub(f["lnoa_x"][i], lag(f, "lnoa_x", i, 12)), avg_den(i))
        ),
    )
    f["capx_gr2"] = column(
        f,
        lambda i: (
            None
            if count(i) <= 24 or le(lag(f, "capx", i, 24), 0)
            else sub(div(f["capx"][i], lag(f, "capx", i, 24)), 1.0)
        ),
    )


def quarterly_profitability(f: Firm) -> None:
    """Description:
        Quarterly profitability and its changes, and R&D capital to assets
        (accounting_chars.sas:908-928).
    Steps:
        1) saleq_gr1 = sale_qtr / lag12(sale_qtr) - 1; missing if count <= 12 or the lag < 0.
        2) niq_be = ni_qtr / lag3(be_x) and niq_at = ni_qtr / lag3(at_x); missing if count <= 3 or
           the lag < 0.
        3) niq_be_chg1 and niq_at_chg1 = value - lag12(value); missing if count <= 12.
        4) rd5_at = (xrd + 0.8 lag12 + 0.6 lag24 + 0.4 lag36 + 0.2 lag48 of xrd) / at_x; missing if
           count <= 48 or at_x <= 0.
    Output:
        Adds saleq_gr1, niq_be, niq_at, niq_be_chg1, niq_at_chg1, rd5_at.
    """
    f["saleq_gr1"] = column(
        f,
        lambda i: (
            None
            if count(i) <= 12 or lt(lag(f, "sale_qtr", i, 12), 0)
            else sub(div(f["sale_qtr"][i], lag(f, "sale_qtr", i, 12)), 1.0)
        ),
    )
    f["niq_be"] = column(
        f,
        lambda i: (
            None
            if count(i) <= 3 or lt(lag(f, "be_x", i, 3), 0)
            else div(f["ni_qtr"][i], lag(f, "be_x", i, 3))
        ),
    )
    f["niq_at"] = column(
        f,
        lambda i: (
            None
            if count(i) <= 3 or lt(lag(f, "at_x", i, 3), 0)
            else div(f["ni_qtr"][i], lag(f, "at_x", i, 3))
        ),
    )
    f["niq_be_chg1"] = column(
        f,
        lambda i: None if count(i) <= 12 else sub(f["niq_be"][i], lag(f, "niq_be", i, 12)),
    )
    f["niq_at_chg1"] = column(
        f,
        lambda i: None if count(i) <= 12 else sub(f["niq_at"][i], lag(f, "niq_at", i, 12)),
    )

    def rd_capital(i: int) -> Value:
        weights = (1.0, 0.8, 0.6, 0.4, 0.2)
        terms = [mul(lag(f, "xrd", i, 12 * k), w) for k, w in enumerate(weights)]
        return div(add(*terms), f["at_x"][i])

    f["rd5_at"] = column(
        f, lambda i: None if count(i) <= 48 or le(f["at_x"][i], 0) else rd_capital(i)
    )


def abarbanell_bushee(f: Firm) -> None:
    """Description:
        Abarbanell and Bushee (1998) changes (accounting_chars.sas:930-941).
    Steps:
        1) Change to expectations of sale_x, invt, rect, gp_x and xsga.
        2) dsale_dinv = sale_ce - invt_ce, dsale_drec = sale_ce - rect_ce,
           dgp_dsale = gp_ce - sale_ce, dsale_dsga = sale_ce - xsga_ce.
    Output:
        Adds dsale_dinv, dsale_drec, dgp_dsale, dsale_dsga (and the _ce helper columns).
    """
    for var, out in (
        ("sale_x", "sale_ce"),
        ("invt", "invt_ce"),
        ("rect", "rect_ce"),
        ("gp_x", "gp_ce"),
        ("xsga", "xsga_ce"),
    ):
        _chg_to_exp(f, var, out)
    f["dsale_dinv"] = column(f, lambda i: sub(f["sale_ce"][i], f["invt_ce"][i]))
    f["dsale_drec"] = column(f, lambda i: sub(f["sale_ce"][i], f["rect_ce"][i]))
    f["dgp_dsale"] = column(f, lambda i: sub(f["gp_ce"][i], f["sale_ce"][i]))
    f["dsale_dsga"] = column(f, lambda i: sub(f["sale_ce"][i], f["xsga_ce"][i]))


def standardized_surprises(f: Firm) -> None:
    """Description:
        Standardized unexpected sales and earnings (accounting_chars.sas:943-945).
    Steps:
        1) Standardized unexpected realization of sale_qtr and ni_qtr over 8 quarters, at least 6
           (more than 6 observed changes) required.
    Output:
        Adds sale_qtr_su and ni_qtr_su.
    """
    _standardized_unexpected(f, "sale_qtr", 8, 6)
    _standardized_unexpected(f, "ni_qtr", 8, 6)


def abnormal_investment(f: Firm) -> None:
    """Description:
        Abnormal corporate investment (accounting_chars.sas:947-954).
    Steps:
        1) __capex_sale = capx / sale_x. The guard for sale_x <= 0 sets __capx_sale, a different
           name, so it never blanks anything.
        2) capex_abn = __capex_sale / (mean of lags 12, 24, 36) - 1; missing if count <= 36.
    Output:
        Adds capex_abn.
    """
    f["__capex_sale"] = column(f, lambda i: div(f["capx"][i], f["sale_x"][i]))

    def abn(i: int) -> Value:
        past = div(add(*(lag(f, "__capex_sale", i, k) for k in (12, 24, 36))), 3.0)
        return sub(div(f["__capex_sale"][i], past), 1.0)

    f["capex_abn"] = column(f, lambda i: None if count(i) <= 36 else abn(i))


def profit_scaled_by_lagged(f: Firm) -> None:
    """Description:
        Profit scaled by lagged assets or equity (accounting_chars.sas:956-968).
    Steps:
        1) op_atl1, gp_atl1, cop_atl1 = op_x, gp_x, cop_x / lag12(at_x); missing if count <= 12 or
           lag12(at_x) <= 0.
        2) ope_bel1 = ope_x / lag12(be_x); missing if count <= 12 or lag12(be_x) <= 0.
    Output:
        Adds op_atl1, gp_atl1, ope_bel1, cop_atl1.
    """
    for num, name, den in (
        ("op_x", "op_atl1", "at_x"),
        ("gp_x", "gp_atl1", "at_x"),
        ("ope_x", "ope_bel1", "be_x"),
        ("cop_x", "cop_atl1", "at_x"),
    ):
        f[name] = column(
            f,
            lambda i, num=num, den=den: (
                None
                if count(i) <= 12 or le(lag(f, den, i, 12), 0)
                else div(f[num][i], lag(f, den, i, 12))
            ),
        )


def profitability_and_leverage(f: Firm) -> None:
    """Description:
        Profitability measures and book leverage (accounting_chars.sas:970-985).
    Steps:
        1) pi_nix = pi_x / nix_x; missing if pi_x <= 0 or nix_x <= 0.
        2) ocf_at = ocf_x / at_x and op_at = op_x / at_x; missing if at_x <= 0.
        3) ocf_at_chg1 = ocf_at - lag12(ocf_at); missing if count <= 12.
        4) at_be = at_x / be_x.
    Output:
        Adds pi_nix, ocf_at, op_at, ocf_at_chg1, at_be.
    """
    f["pi_nix"] = column(
        f,
        lambda i: (
            None
            if le(f["pi_x"][i], 0) or le(f["nix_x"][i], 0)
            else div(f["pi_x"][i], f["nix_x"][i])
        ),
    )
    f["ocf_at"] = column(
        f, lambda i: None if le(f["at_x"][i], 0) else div(f["ocf_x"][i], f["at_x"][i])
    )
    f["op_at"] = column(
        f, lambda i: None if le(f["at_x"][i], 0) else div(f["op_x"][i], f["at_x"][i])
    )
    f["ocf_at_chg1"] = column(
        f, lambda i: None if count(i) <= 12 else sub(f["ocf_at"][i], lag(f, "ocf_at", i, 12))
    )
    f["at_be"] = column(f, lambda i: div(f["at_x"][i], f["be_x"][i]))


def volatility(f: Firm) -> None:
    """Description:
        Quarterly and annual volatility items (accounting_chars.sas:987-1007).
    Steps:
        1) __ocfq_saleq and __niq_saleq = ocf_qtr, ni_qtr / sale_qtr; missing if sale_qtr <= 0.
        2) __roeq = ni_qtr / be_x, missing if be_x <= 0.
        3) volq: ocfq_saleq_std and niq_saleq_std (16 quarters, min 8), roeq_be_std (20, min 12).
        4) __roe = ni_x / be_x, missing if be_x <= 0; roe_be_std = vola over 5 years, min 5.
    Output:
        Adds ocfq_saleq_std, niq_saleq_std, roeq_be_std, roe_be_std.
    """
    bad_sale = lambda i: le(f["sale_qtr"][i], 0)  # noqa: E731
    f["__ocfq_saleq"] = column(
        f, lambda i: None if bad_sale(i) else div(f["ocf_qtr"][i], f["sale_qtr"][i])
    )
    f["__niq_saleq"] = column(
        f, lambda i: None if bad_sale(i) else div(f["ni_qtr"][i], f["sale_qtr"][i])
    )
    f["__roeq"] = column(
        f, lambda i: None if le(f["be_x"][i], 0) else div(f["ni_qtr"][i], f["be_x"][i])
    )
    _volq(f, "ocfq_saleq_std", "__ocfq_saleq", 16, 8)
    _volq(f, "niq_saleq_std", "__niq_saleq", 16, 8)
    _volq(f, "roeq_be_std", "__roeq", 20, 12)
    f["__roe"] = column(
        f, lambda i: None if le(f["be_x"][i], 0) else div(f["ni_x"][i], f["be_x"][i])
    )
    _vola(f, "roe_be_std", "__roe", 5, 5)


HORIZON = 10
RATE = 0.12
ROE_C = 0.12 * (1 - 0.57)
G_C = 0.06 * (1 - 0.24)
ROE_AR1 = 0.57
G_AR1 = 0.24


def tangibility(f: Firm) -> None:
    """
    Description:
        Asset tangibility (accounting_chars.sas:1009-1010).
    Steps:
        1) (che + 0.715 rect + 0.547 invt + 0.535 ppegt) / at_x; plain sums, so a missing
           component gives a missing result.
    Output:
        Adds column tangibility.
    """
    f["tangibility"] = column(
        f,
        lambda i: div(
            add(
                f["che"][i],
                mul(0.715, f["rect"][i]),
                mul(0.547, f["invt"][i]),
                mul(0.535, f["ppegt"][i]),
            ),
            f["at_x"][i],
        ),
    )


def earnings_variability(f: Firm) -> None:
    """
    Description:
        Earnings smoothness over five years (accounting_chars.sas:1012-1013, char_macros.sas:408-419).
    Steps:
        1) roa = ni_x / at_x twelve rows back and croa = ocf_x / at_x twelve rows back.
        2) Over this row and the four annual lags (rows 12, 24, 36, 48 back): count the non-missing
           values and take the sample std (missing values skipped).
        3) std(roa) / std(croa); missing if count <= 60, std(croa) <= 0 (missing counts as
           <= 0) or either count is below 5.
    Output:
        Adds column earnings_variability.
    """
    roa = column(f, lambda i: div(f["ni_x"][i], lag(f, "at_x", i, 12)))
    croa = column(f, lambda i: div(f["ocf_x"][i], lag(f, "at_x", i, 12)))

    def window(x: list[Value], i: int) -> list[Value]:
        return [x[i - k] if i >= k else None for k in range(0, 60, 12)]

    def value(i: int) -> Value:
        roa_n = sum(v is not None for v in window(roa, i))
        croa_n = sum(v is not None for v in window(croa, i))
        croa_std = sas_std(*window(croa, i))
        if count(i) <= 60 or le(croa_std, 0) or roa_n < 5 or croa_n < 5:
            return None
        return div(sas_std(*window(roa, i)), croa_std)

    f["earnings_variability"] = column(f, value)


def aliq_at(f: Firm) -> None:
    """
    Description:
        Asset liquidity scaled by lagged assets (accounting_chars.sas:1016-1017).
    Steps:
        1) aliq_x / at_x twelve rows back.
        2) Missing if count <= 12 or lagged at_x <= 0 (missing counts as <= 0).
    Output:
        Adds column aliq_at.
    """

    def value(i: int) -> Value:
        back = lag(f, "at_x", i, 12)
        if count(i) <= 12 or le(back, 0):
            return None
        return div(f["aliq_x"][i], back)

    f["aliq_at"] = column(f, value)


def equity_duration_cd(f: Firm) -> None:
    """
    Description:
        Helper variables of the equity duration, forecasting ten years of cash distributions
        (accounting_chars.sas:1019-1020, char_macros.sas:422-450).
    Steps:
        1) Starting roe = ni_x / be_x twelve rows back and growth = sale_x / sale_x twelve rows
           back - 1, each missing if count <= 12 or its lagged denominator <= 1.
        2) Years 1-10: roe and growth mean-revert (constants 0.12*(1-0.57) and 0.06*(1-0.24),
           slopes 0.57 and 0.24); book equity grows by growth; distribution = prior book equity
           * (roe - growth). Missing propagates.
        3) ed_constant = 10 + 1.12/0.12; ed_cd_w = sum t * cd_t / 1.12^t; ed_cd = sum cd_t /
           1.12^t; ed_err = 1 if any forecast book equity is below 0 (a missing one counts as
           below 0), else 0.
    Output:
        Adds columns ed_constant, ed_cd_w, ed_cd, ed_err.
    """
    cols: dict[str, list] = {"ed_constant": [], "ed_cd_w": [], "ed_cd": [], "ed_err": []}
    for i in rows(f):
        be_back, sale_back = lag(f, "be_x", i, 12), lag(f, "sale_x", i, 12)
        roe = div(f["ni_x"][i], be_back)
        g = sub(div(f["sale_x"][i], sale_back), 1)
        if count(i) <= 12 or le(be_back, 1):
            roe = None
        if count(i) <= 12 or le(sale_back, 1):
            g = None
        be = f["be_x"][i]
        cd_w: Value = 0.0
        cd: Value = 0.0
        err = 0.0
        for t in range(1, HORIZON + 1):
            roe = add(ROE_C, mul(ROE_AR1, roe))
            g = add(G_C, mul(G_AR1, g))
            cd_t = mul(be, sub(roe, g))
            be = mul(be, add(1, g))
            cd_w = add(cd_w, div(mul(t, cd_t), (1 + RATE) ** t))
            cd = add(cd, div(cd_t, (1 + RATE) ** t))
            if lt(be, 0):
                err = 1.0
        cols["ed_constant"].append(HORIZON + (1 + RATE) / RATE)
        cols["ed_cd_w"].append(cd_w)
        cols["ed_cd"].append(cd)
        cols["ed_err"].append(err)
    f.update(cols)


def _prev(xs: list[Value], i: int, k: int) -> Value:
    """Value k rows before row i of a local column; missing before the firm's first row."""
    return xs[i - k] if i >= k else None


def pitroski_f(f: Firm) -> None:
    """
    Description:
        Piotroski F-score, nine yes/no signals (char_macros.sas:453-477).
    Steps:
        1) Year-over-year changes in roa, leverage, liquidity, gross margin and asset turnover,
           each blanked when the row lacks the lags or a positive denominator (lines 454-469).
        2) Sum nine indicators; a missing equity issuance counts as no issuance (lines 470-472).
        3) Missing if any signal except equity issuance is missing (lines 473-475).
    Output:
        f["f_score"]. Lags stay within the firm.
    """
    at, ni, ocf = f["at_x"], f["ni_x"], f["ocf_x"]
    n = len(at)

    def roa_like(num: list[Value]) -> list[Value]:
        return column(
            f,
            lambda i: (
                None
                if count(i) <= 12 or le(lag(f, "at_x", i, 12), 0)
                else div(num[i], lag(f, "at_x", i, 12))
            ),
        )

    roa = roa_like(ni)
    croa = roa_like(ocf)
    droa = column(f, lambda i: None if count(i) <= 12 else sub(roa[i], _prev(roa, i, 12)))
    acc = column(f, lambda i: sub(croa[i], roa[i]))

    dltt_at = [div(f["dltt"][i], at[i]) for i in range(n)]
    lev = column(
        f,
        lambda i: (
            None
            if count(i) <= 12 or le(at[i], 0) or le(lag(f, "at_x", i, 12), 0)
            else sub(dltt_at[i], _prev(dltt_at, i, 12))
        ),
    )
    ca_cl = [div(f["ca_x"][i], f["cl_x"][i]) for i in range(n)]
    liq = column(
        f,
        lambda i: (
            None
            if count(i) <= 12 or le(f["cl_x"][i], 0) or le(lag(f, "cl_x", i, 12), 0)
            else sub(ca_cl[i], _prev(ca_cl, i, 12))
        ),
    )
    eqis = list(f["eqis_x"])
    gp_sale = [div(f["gp_x"][i], f["sale_x"][i]) for i in range(n)]
    gm = column(
        f,
        lambda i: (
            None
            if count(i) <= 12 or le(f["sale_x"][i], 0) or le(lag(f, "sale_x", i, 12), 0)
            else sub(gp_sale[i], _prev(gp_sale, i, 12))
        ),
    )
    aturn = column(
        f,
        lambda i: (
            None
            if count(i) <= 24 or le(lag(f, "at_x", i, 12), 0) or le(lag(f, "at_x", i, 24), 0)
            else sub(
                div(f["sale_x"][i], lag(f, "at_x", i, 12)),
                div(lag(f, "sale_x", i, 12), lag(f, "at_x", i, 24)),
            )
        ),
    )

    def score(i: int) -> Value:
        if any(x[i] is None for x in (roa, croa, droa, acc, lev, liq, gm, aturn)):
            return None
        return sum(
            (
                flag(gt(roa[i], 0)),
                flag(gt(croa[i], 0)),
                flag(gt(droa[i], 0)),
                flag(gt(acc[i], 0)),
                flag(lt(lev[i], 0)),
                flag(gt(liq[i], 0)),
                flag(eq(0 if eqis[i] is None else eqis[i], 0)),
                flag(gt(gm[i], 0)),
                flag(gt(aturn[i], 0)),
            )
        )

    f["f_score"] = column(f, score)


def ohlson_o(f: Firm) -> None:
    """
    Description:
        Ohlson (1980) O-score (char_macros.sas:480-506).
    Steps:
        1) Size, leverage, working capital and roe, blanked when assets are not positive (479-488).
        2) Current ratio (blank if current assets not positive), funds from operations over
           liabilities (blank if liabilities not positive) (489-493).
        3) Liabilities-exceed-assets flag, two-year-loss flag, earnings change; the last two
           need a year of history and non-missing earnings (494-499).
        4) Linear combination; missing if any term is missing (501-503).
    Output:
        f["o_score"]. A missing earnings pair leaves the loss flag missing, not 0.
    """
    at, lt_ = f["at_x"], f["lt"]
    nix = f["nix_x"]

    def at_guard(value: Callable[[int], Value]) -> Callable[[int], Value]:
        return lambda i: None if le(at[i], 0) else value(i)

    lat = column(f, at_guard(lambda i: sas_log(at[i])))
    lev = column(f, at_guard(lambda i: div(f["debt_x"][i], at[i])))
    wc = column(f, at_guard(lambda i: div(sub(f["ca_x"][i], f["cl_x"][i]), at[i])))
    roe = column(f, at_guard(lambda i: div(nix[i], at[i])))
    cacl = column(f, lambda i: None if le(f["ca_x"][i], 0) else div(f["cl_x"][i], f["ca_x"][i]))
    ffo = column(
        f,
        lambda i: None if le(lt_[i], 0) else div(add(f["pi_x"][i], f["dp"][i]), lt_[i]),
    )
    neg_eq = column(
        f,
        lambda i: None if lt_[i] is None or at[i] is None else flag(gt(lt_[i], at[i])),
    )

    def neg_earn_at(i: int) -> Value:
        lag_nix = lag(f, "nix_x", i, 12)
        if count(i) <= 12 or nix[i] is None or lag_nix is None:
            return None
        return flag(lt(nix[i], 0) and lt(lag_nix, 0))

    neg_earn = column(f, neg_earn_at)

    def nich_at(i: int) -> Value:
        lag_nix = lag(f, "nix_x", i, 12)
        denom = add(sas_abs(nix[i]), sas_abs(lag_nix))
        if count(i) <= 12 or eq(denom, 0):
            return None
        return div(sub(nix[i], lag_nix), denom)

    nich = column(f, nich_at)

    f["o_score"] = column(
        f,
        lambda i: add(
            -1.32,
            mul(-0.407, lat[i]),
            mul(6.03, lev[i]),
            mul(-1.43, wc[i]),
            mul(0.076, cacl[i]),
            mul(-1.72, neg_eq[i]),
            mul(-2.37, roe[i]),
            mul(-1.83, ffo[i]),
            mul(0.285, neg_earn[i]),
            mul(-0.52, nich[i]),
        ),
    )


def altman_z(f: Firm) -> None:
    """
    Description:
        Altman (1968) Z-score (char_macros.sas:509-526).
    Steps:
        1) Working capital, retained earnings, ebitda and sales over assets, blank when assets are
           not positive; fiscal-year market equity over liabilities, blank when liabilities are
           not positive (lines 511-522).
        2) Weighted sum; missing if any term is missing (line 524).
    Output:
        f["z_score"].
    """
    at = f["at_x"]
    lt_ = f["lt"]

    def at_guard(value: Callable[[int], Value]) -> Callable[[int], Value]:
        return lambda i: None if le(at[i], 0) else value(i)

    wc = column(f, at_guard(lambda i: div(sub(f["ca_x"][i], f["cl_x"][i]), at[i])))
    re = column(f, at_guard(lambda i: div(f["re"][i], at[i])))
    eb = column(f, at_guard(lambda i: div(f["ebitda_x"][i], at[i])))
    sa = column(f, at_guard(lambda i: div(f["sale_x"][i], at[i])))
    me = column(f, lambda i: None if le(lt_[i], 0) else div(f["me_fiscal"][i], lt_[i]))
    f["z_score"] = column(
        f,
        lambda i: add(
            mul(1.2, wc[i]),
            mul(1.4, re[i]),
            mul(3.3, eb[i]),
            mul(0.6, me[i]),
            mul(1.0, sa[i]),
        ),
    )


def intrinsic_value(f: Firm) -> None:
    """
    Description:
        Frankel and Lee (1998) roe-based intrinsic value with r = 0.12 (char_macros.sas:529-541).
    Steps:
        1) Payout: dividends over earnings, or over 6% of assets when earnings are not positive
           (a missing earnings value counts as not positive) (lines 530-532).
        2) roe on average book equity, blank without a year of history or when the sum of the two
           book equities is not positive (lines 533-534).
        3) Next-year book equity and the two-term value; non-positive becomes missing
           (lines 535-539).
    Output:
        f["intrinsic_value"].
    """
    r = 0.12
    be, nix = f["be_x"], f["nix_x"]

    def po_at(i: int) -> Value:
        if le(nix[i], 0):
            return div(f["div_x"][i], mul(f["at_x"][i], 0.06))
        return div(f["div_x"][i], nix[i])

    po = column(f, po_at)

    def roe_at(i: int) -> Value:
        be_sum = add(be[i], lag(f, "be_x", i, 12))
        if count(i) <= 12 or le(be_sum, 0):
            return None
        return div(nix[i], div(be_sum, 2))

    roe = column(f, roe_at)
    be1 = column(f, lambda i: mul(add(1, mul(sub(1, po[i]), roe[i])), be[i]))

    def value_at(i: int) -> Value:
        v = add(
            be[i],
            mul(div(sub(roe[i], r), 1 + r), be[i]),
            mul(div(sub(roe[i], r), (1 + r) * r), be1[i]),
        )
        return None if le(v, 0) else v

    f["intrinsic_value"] = column(f, value_at)


def kz_index(f: Firm) -> None:
    """
    Description:
        Kaplan-Zingales index (char_macros.sas:544-559).
    Steps:
        1) Cash flow, dividends and cash over lagged net PP&E, blank without a year of history or
           when lagged PP&E is not positive (lines 546-547, 552-555).
        2) Q: (assets + fiscal-year market equity - book equity) over assets, blank when assets
           are not positive (lines 548-549).
        3) Debt over debt plus equity, blank when that sum is exactly 0 (lines 550-551).
        4) Weighted sum; missing if any term is missing (lines 557-558).
    Output:
        f["kz_index"].
    """
    at = f["at_x"]

    def ppe_guard(value: Callable[[int], Value]) -> Callable[[int], Value]:
        return lambda i: None if count(i) <= 12 or le(lag(f, "ppent", i, 12), 0) else value(i)

    cf = column(f, ppe_guard(lambda i: div(add(f["ni_x"][i], f["dp"][i]), lag(f, "ppent", i, 12))))
    q = column(
        f,
        lambda i: (
            None if le(at[i], 0) else div(sub(add(at[i], f["me_fiscal"][i]), f["be_x"][i]), at[i])
        ),
    )

    def db_at(i: int) -> Value:
        total = add(f["debt_x"][i], f["seq_x"][i])
        return None if eq(total, 0) else div(f["debt_x"][i], total)

    db = column(f, db_at)
    dv = column(f, ppe_guard(lambda i: div(f["div_x"][i], lag(f, "ppent", i, 12))))
    cs = column(f, ppe_guard(lambda i: div(f["che"][i], lag(f, "ppent", i, 12))))
    f["kz_index"] = column(
        f,
        lambda i: add(
            mul(-1.002, cf[i]),
            mul(0.283, q[i]),
            mul(3.139, db[i]),
            mul(-39.368, dv[i]),
            mul(-1.315, cs[i]),
        ),
    )


def _ratio_change(f: Firm, name: str, var1: str, var2: str) -> None:
    ratio = column(f, lambda i: None if le(f[var2][i], 0) else div(f[var1][i], f[var2][i]))
    f[name] = column(
        f, lambda i: None if count(i) <= 60 else sub(ratio[i], lag({"r": ratio}, "r", i, 60))
    )


def ratio_changes(f: Firm) -> None:
    """
    Description:
        Five-year change of five ratios (accounting_chars.sas:1038-1043,
        char_macros.sas:349-357).
    Steps:
        1) ratio = var1 / var2, missing if var2 <= 0 (missing counts as <= 0).
        2) ratio - ratio sixty rows back; missing if count <= 60.
    Output:
        Adds gpoa_ch5 (gp_x/at_x), roe_ch5 (ni_x/be_x), roa_ch5 (ni_x/at_x), cfoa_ch5
        (ocf_x/at_x), gmar_ch5 (gp_x/sale_x).
    """
    _ratio_change(f, "gpoa_ch5", "gp_x", "at_x")
    _ratio_change(f, "roe_ch5", "ni_x", "be_x")
    _ratio_change(f, "roa_ch5", "ni_x", "at_x")
    _ratio_change(f, "cfoa_ch5", "ocf_x", "at_x")
    _ratio_change(f, "gmar_ch5", "gp_x", "sale_x")


def earnings_persistence(f: Firm) -> None:
    """
    Description:
        AR(1) slope and residual volatility of income scaled by assets over the last five years
        (accounting_chars.sas:1048-1056, char_macros.sas:563-651).
    Steps:
        1) ni_at = ni_x / at_x (missing if at_x <= 0) and its value twelve rows back (missing if
           count <= 12). Keep rows where both are present.
        2) For a row dated d, take the kept rows with the same calendar month in the years
           year(d)-4 .. year(d) (datadates after Dec 31 of year(d)-5, up to d). SAS evaluates
           every date with a kept row in any firm, but the window holds at most five dates and
           needs five, so only the firm's own kept dates can give a regression.
        3) Need at least 5 rows. Regress ni_at on its lag with an intercept: ni_ar1 is the slope,
           ni_ivol = sqrt(rmse^2 * edf / (edf + 1)) with edf = n - 2. When the regressor has no
           variance both are missing (judgment: SAS's proc reg has no slope then).
    Output:
        Adds columns ni_ar1 and ni_ivol (missing where no regression was run).
    """
    ni_at = column(f, lambda i: None if le(f["at_x"][i], 0) else div(f["ni_x"][i], f["at_x"][i]))
    pairs = {
        f["datadate"][i]: (ni_at[i - 12], ni_at[i])
        for i in rows(f)
        if count(i) > 12 and ni_at[i] is not None and ni_at[i - 12] is not None
    }
    ar1: list[Value] = []
    ivol: list[Value] = []
    for i in rows(f):
        d = f["datadate"][i]
        slope = sd = None
        if d in pairs:
            win = [pairs[e] for e in pairs if e.month == d.month and d.year - 4 <= e.year <= d.year]
            if len(win) >= 5:
                n = len(win)
                mx = math.fsum(x for x, _ in win) / n
                my = math.fsum(y for _, y in win) / n
                sxx = math.fsum((x - mx) ** 2 for x, _ in win)
                if sxx > 0:
                    slope = math.fsum((x - mx) * (y - my) for x, y in win) / sxx
                    icpt = my - slope * mx
                    sse = math.fsum((y - icpt - slope * x) ** 2 for x, y in win)
                    edf = n - 2
                    sd = math.sqrt(sse / edf * edf / (edf + 1))
        ar1.append(slope)
        ivol.append(sd)
    f["ni_ar1"] = ar1
    f["ni_ivol"] = ivol


# The sections of the characteristics data step (accounting_chars.sas:641-1046), then the earnings
# persistence that is merged onto it (lines 1048-1056), in SAS order.
SECTIONS: list[Callable[[Firm], None]] = [
    size_measures,
    growth,
    change_to_assets,
    investment_and_nonrecurring,
    profitability,
    leverage_and_soundness,
    liquidity,
    activity,
    misc_ratios,
    cash_and_employees,
    consecutive_earnings_increases,
    asset_scaled_changes,
    quarterly_profitability,
    abarbanell_bushee,
    standardized_surprises,
    abnormal_investment,
    profit_scaled_by_lagged,
    profitability_and_leverage,
    volatility,
    tangibility,
    earnings_variability,
    aliq_at,
    equity_duration_cd,
    pitroski_f,
    ohlson_o,
    altman_z,
    intrinsic_value,
    kz_index,
    ratio_changes,
    earnings_persistence,
]


def month_end(d: dt.date, months: int = 0) -> dt.date:
    """SAS intnx('month', d, months, 'e'): the last day of the month `months` after d's."""
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return dt.date(y, m + 1, calendar.monthrange(y, m + 1)[1])


def months_between(a: dt.date, b: dt.date) -> int:
    """SAS intck('month', a, b): month boundaries crossed from a to b."""
    return (b.year - a.year) * 12 + b.month - a.month


def firms(std: pl.DataFrame) -> list[Firm]:
    """
    Description:
        Split a standardized table into firms (accounting_chars.sas:628).
    Steps:
        1) Sort by (gvkey, curcd, datadate) and split by (gvkey, curcd).
        2) Hold each as columns; emp (Decimal in the annual table) as float, as SAS stores it.
    Output:
        List of firms in (gvkey, curcd) order.
    """
    std = std.with_columns(pl.col(pl.Decimal).cast(pl.Float64)).sort("gvkey", "curcd", "datadate")
    return [
        part.to_dict(as_series=False)
        for part in std.partition_by(["gvkey", "curcd"], maintain_order=True)
    ]


def firm_rows(f: Firm) -> list[Row]:
    names = list(f)
    return [dict(zip(names, values, strict=True)) for values in zip(*f.values(), strict=True)]


def expand_to_public_dates(reports: list[Row], lag_to_public: int, max_data_lag: int) -> list[Row]:
    """
    Description:
        Each report's months of public availability (accounting_chars.sas:1058-1080 and
        %expand, project_macros.sas).
    Steps:
        1) Keep rows with data_available = 1.
        2) Per gvkey, from the latest datadate back (ties keep curcd order, as SAS's stable sort):
           start = month-end lag_to_public months after datadate; end = the earlier of the month
           before the next later report's start and max_data_lag months after datadate.
        3) One row per month-end from start to end (none if end is before start); for a
           repeated (gvkey, public_date) the first row in that order wins (%expand's nodupkey).
    Output:
        Rows sorted by (gvkey, public_date), each with public_date added.
    """
    out: dict[tuple[str, dt.date], Row] = {}
    available = [r for r in reports if r["data_available"] == 1]
    available.sort(key=lambda r: r["gvkey"])  # stable: keeps (curcd, datadate) order
    for _, group in groupby(available, key=lambda r: r["gvkey"]):
        later_start = None
        for r in sorted(group, key=lambda r: r["datadate"], reverse=True):
            start = month_end(r["datadate"], lag_to_public)
            ends = [month_end(r["datadate"], max_data_lag)]
            if later_start is not None:
                ends.append(month_end(later_start, -1))
            later_start = start
            for k in range(months_between(start, min(ends)) + 1):
                public_date = month_end(start, k)
                out.setdefault((r["gvkey"], public_date), {**r, "public_date": public_date})
    return [out[k] for k in sorted(out)]


def me_company_by_month(msf: pl.DataFrame) -> dict[tuple[str, dt.date], float]:
    """
    Description:
        Company market equity per (gvkey, month-end) (accounting_chars.sas:1103-1110).
    Steps:
        1) Keep primary, common, obs_main securities with a gvkey and an me_company.
        2) Take the largest me_company per (gvkey, eom).
    Output:
        Dict (gvkey, eom) -> me_company.
    """
    me: dict[tuple[str, dt.date], float] = {}
    for r in msf.iter_rows(named=True):
        ok = r["primary_sec"] == 1 and r["common"] == 1 and r["obs_main"] == 1
        if ok and r["gvkey"] is not None and r["me_company"] is not None:
            k = (r["gvkey"], r["eom"])
            me[k] = max(me.get(k, -math.inf), r["me_company"])
    return me


ME_VARS = [
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
MEV_VARS = [
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


def market_characteristics(public: list[Row], me: dict[tuple[str, dt.date], float]) -> None:
    """
    Description:
        Join company market equity and add the characteristics that use it
        (accounting_chars.sas:1112-1160). Rows must already carry fx.
    Steps:
        1) me_company at (gvkey, public_date).
        2) mev = me_company + netdebt_x*fx and mat = at_x*fx - be_x*fx + me_company, then
           mev, me_company and mat set missing where <= 0 (after mev and mat use me_company).
        3) <var>_me = var*fx / me_company and <var>_mev = var*fx / mev, with '_x' removed from
           the name; ival_me from intrinsic_value.
        4) aliq_mat = aliq_x*fx / mat twelve rows earlier for the same gvkey.
        5) enterprise_value = mev; eq_dur from the equity-duration helpers, missing if ed_err = 1
           or eq_dur <= 0.
    Output:
        None; adds the columns to each row in place.
    """
    for i, r in enumerate(public):
        fx, me_company = r["fx"], me.get((r["gvkey"], r["public_date"]))
        r["mev"] = add(me_company, mul(r["netdebt_x"], fx))
        r["mat"] = add(sub(mul(r["at_x"], fx), mul(r["be_x"], fx)), me_company)
        if le(r["mev"], 0):
            r["mev"] = None
        if le(me_company, 0):
            me_company = None
        if le(r["mat"], 0):
            r["mat"] = None
        r["me_company"] = me_company
        for var in ME_VARS:
            r[var.replace("_x", "") + "_me"] = div(mul(r[var], fx), me_company)
        r["ival_me"] = div(mul(r.get("intrinsic_value"), fx), me_company)
        for var in MEV_VARS:
            r[var.replace("_x", "") + "_mev"] = div(mul(r[var], fx), r["mev"])
        back = public[i - 12] if i >= 12 else None
        same = back is not None and back["gvkey"] == r["gvkey"]
        r["aliq_mat"] = div(mul(r["aliq_x"], fx), back["mat"]) if same else None  # type: ignore[index]
        r["enterprise_value"] = r["mev"]
        ed = [r.get(c) for c in ("ed_cd_w", "ed_constant", "ed_cd", "ed_err")]
        eq_dur = add(
            div(mul(ed[0], fx), me_company),
            div(mul(ed[1], sub(me_company, mul(ed[2], fx))), me_company),
        )
        r["eq_dur"] = None if eq(ed[3], 1) or le(eq_dur, 0) else eq_dur


# accounting_chars.sas:1173-1188: each pattern's first occurrence in every name, in this order.
RENAMES = [
    ("xrd", "rd"),
    ("xsga", "sga"),
    ("dlc", "debtst"),
    ("dltt", "debtlt"),
    ("oancf", "ocf"),
    ("ppegt", "ppeg"),
    ("ppent", "ppen"),
    ("che", "cash"),
    ("invt", "inv"),
    ("rect", "rec"),
    ("txt", "tax"),
    ("ivao", "lti"),
    ("ivst", "sti"),
    ("sale_qtr", "saleq"),
    ("ni_qtr", "niq"),
    ("ocf_qtr", "ocfq"),
]


def sas_name(name: str) -> str:
    for old, new in RENAMES:
        name = name.replace(old, new, 1)
    return name


def stage_characteristics(
    std: pl.DataFrame,
    fx: dict[tuple[str, dt.date], float],
    me: dict[tuple[str, dt.date], float],
    keep: list[str],
    lag_to_public: int = 4,
    max_data_lag: int = 18,
    sections: list[Callable[[Firm], None]] | None = None,
) -> dict[tuple[str, dt.date], Row]:
    """
    Description:
        One create_acc_chars run (accounting_chars.sas:605-1217) on a standardized table.
    Steps:
        1) Per firm, run the data-step sections and the earnings persistence.
        2) Expand to public dates, convert assets, sales, book_equity and net_income to USD with
           fx at (curcd, public_date) (lines 1082-1099), add the market characteristics.
        3) Rename (lines 1164-1191) and keep source, datadate and the keep list; a keep-list
           name that no section produces raises KeyError.
    Output:
        Dict (gvkey, public_date) -> row.
    """
    reports = []
    for f in firms(std):
        for section in sections if sections is not None else SECTIONS:
            section(f)
        reports.extend(firm_rows(f))
    public = expand_to_public_dates(reports, lag_to_public, max_data_lag)
    for r in public:
        r["fx"] = fx.get((r["curcd"], r["public_date"]))
        for var in ("assets", "sales", "book_equity", "net_income"):
            r[var] = mul(r[var], r["fx"])
    market_characteristics(public, me)
    out = {}
    for r in public:
        renamed = {sas_name(k): v for k, v in r.items()}
        if absent := [c for c in keep if c not in renamed]:
            raise KeyError(f"no oracle section produces {absent}")
        out[(r["gvkey"], r["public_date"])] = {
            "source": r["source"],
            "datadate": r["datadate"],
            **{c: renamed[c] for c in keep},
        }
    return out


def combine(
    annual: dict[tuple[str, dt.date], Row],
    quarterly: dict[tuple[str, dt.date], Row],
    keep: list[str],
) -> dict[tuple[str, dt.date], Row]:
    """
    Description:
        Merge the annual and quarterly runs (accounting_chars.sas:1220-1247).
    Steps:
        1) Full join on (gvkey, public_date); source from the annual row, else the quarterly.
        2) Each characteristic takes the quarterly value if the annual one is missing, or if
           the quarterly one is present and its datadate is later.
    Output:
        Dict (gvkey, public_date) -> row with source and the keep list.
    """
    out = {}
    for k in sorted(annual.keys() | quarterly.keys()):
        a, q = annual.get(k, {}), quarterly.get(k, {})
        qd, ad = q.get("datadate"), a.get("datadate")
        newer = qd is not None and (ad is None or qd > ad)  # type: ignore[operator]
        row: Row = {"source": a.get("source", q.get("source"))}
        for c in keep:
            av, qv = a.get(c), q.get(c)
            row[c] = qv if av is None or (qv is not None and newer) else av
        out[k] = row
    return out
