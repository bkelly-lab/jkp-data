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


# The sections of the characteristics data step (accounting_chars.sas:641-1046), then the earnings
# persistence that is merged onto it (lines 1048-1056), in SAS order.
SECTIONS: list[Callable[[Firm], None]] = [size_measures]


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
        3) Rename (lines 1164-1191) and keep source, datadate and the keep list.
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
        out[(r["gvkey"], r["public_date"])] = {
            "source": r["source"],
            "datadate": r["datadate"],
            **{c: renamed.get(c) for c in keep},
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
