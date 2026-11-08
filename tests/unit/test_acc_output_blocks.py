"""Tests for the output stages of create_acc_chars.

Covers add_earnings_persistence_and_expand (public-date expansion; the AR(1) numbers are tested in
test_earnings_persistence.py), convert_raw_vars_to_usd, add_me_data_and_compute_me_mev_mat_eqdur_vars and
rename_cols_and_select_keep_vars. SAS reference: accounting_chars.sas lines 1060-1175.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import (
    add_earnings_persistence_and_expand,
    add_me_data_and_compute_me_mev_mat_eqdur_vars,
    convert_raw_vars_to_usd,
    rename_cols_and_select_keep_vars,
)
from jkp.data.paths import DataPaths

# ---------------------------------------------------------------------------
# add_earnings_persistence_and_expand
# ---------------------------------------------------------------------------


def _annual_frame(rows: list[tuple[str, date, int]]) -> pl.DataFrame:
    """Annual rows (gvkey, datadate, data_available) with the columns the persistence job reads."""
    return pl.DataFrame(
        [(g, "USD", d, 5.0, 100.0, a) for g, d, a in rows],
        schema={
            "gvkey": pl.String,
            "curcd": pl.String,
            "datadate": pl.Date,
            "ni_x": pl.Float64,
            "at_x": pl.Float64,
            "data_available": pl.Int64,
        },
        orient="row",
    )


def _expand(test_paths: DataPaths, df: pl.DataFrame, lag_to_pub: int, max_lag: int) -> pl.DataFrame:
    data_path = test_paths.interim_dir / "acc_in.parquet"
    df.write_parquet(data_path)
    out = add_earnings_persistence_and_expand(test_paths, df.lazy(), data_path, lag_to_pub, max_lag)
    return out.collect().sort("gvkey", "datadate", "public_date")


def _span(out: pl.DataFrame, gvkey: str, datadate: date) -> tuple[date, date, int]:
    sub = out.filter((pl.col("gvkey") == gvkey) & (pl.col("datadate") == datadate))
    return sub["public_date"].min(), sub["public_date"].max(), sub.height


@pytest.mark.unit
def test_expansion_window_runs_to_next_report_or_max_lag(test_paths: DataPaths) -> None:
    """With lag_to_pub=4 and max_lag=18 (months), a report is public from datadate+4 to the earlier of
    (next report's start - 1 month) and datadate+18, snapped to month-ends.

    A: 2000-12-31 -> start 2001-04-30; next start 2002-04-30, so end = 2002-03-31 (12 months).
    A: 2001-12-31 -> start 2002-04-30; no next report, end = 2001-12-31+18mo = 2003-06-30 (15 months).
    B: 2000-12-31 -> alone, end = 2002-06-30: A's reports must not limit B (per-firm .over).
    C: 2000-12-31 then 2005-12-31 -> the first runs to datadate+18 = 2002-06-30, not to the next start.
    """
    df = _annual_frame(
        [
            ("A", date(2000, 12, 31), 1),
            ("A", date(2001, 12, 31), 1),
            ("B", date(2000, 12, 31), 1),
            ("C", date(2000, 12, 31), 1),
            ("C", date(2005, 12, 31), 1),
        ]
    )
    out = _expand(test_paths, df, 4, 18)

    assert _span(out, "A", date(2000, 12, 31)) == (date(2001, 4, 30), date(2002, 3, 31), 12)
    assert _span(out, "A", date(2001, 12, 31)) == (date(2002, 4, 30), date(2003, 6, 30), 15)
    assert _span(out, "B", date(2000, 12, 31)) == (date(2001, 4, 30), date(2002, 6, 30), 15)
    assert _span(out, "C", date(2000, 12, 31)) == (date(2001, 4, 30), date(2002, 6, 30), 15)
    assert _span(out, "C", date(2005, 12, 31)) == (date(2006, 4, 30), date(2007, 6, 30), 15)
    # months step by one month-end, no gaps
    months = out.filter(pl.col("gvkey") == "B")["public_date"]
    assert months.dt.month_end().to_list() == months.to_list()
    assert months.diff().drop_nulls().dt.total_days().min() >= 28


@pytest.mark.unit
def test_rows_without_data_available_are_dropped_before_windows(test_paths: DataPaths) -> None:
    """A report with data_available=0 is removed first, so its neighbour's window is not cut short by it.

    A: 2000-12-31 (available), 2001-12-31 (not available). The only window is 2001-04-30..2002-06-30.
    """
    df = _annual_frame([("A", date(2000, 12, 31), 1), ("A", date(2001, 12, 31), 0)])
    out = _expand(test_paths, df, 4, 18)
    assert out["datadate"].unique().to_list() == [date(2000, 12, 31)]
    assert _span(out, "A", date(2000, 12, 31)) == (date(2001, 4, 30), date(2002, 6, 30), 15)


@pytest.mark.unit
def test_expansion_drops_helper_columns_and_keeps_keys_unique(test_paths: DataPaths) -> None:
    """Output carries public_date and the persistence columns, not start/end dates, unique on (gvkey, public_date)."""
    df = _annual_frame([("A", date(2000, 12, 31), 1), ("A", date(2001, 12, 31), 1)])
    out = _expand(test_paths, df, 4, 18)
    assert {"public_date", "ni_ar1", "ni_ivol"} <= set(out.columns)
    assert not {"start_date", "end_date", "next_start_date"} & set(out.columns)
    assert out.select("gvkey", "public_date").is_unique().all()


# ---------------------------------------------------------------------------
# convert_raw_vars_to_usd
# ---------------------------------------------------------------------------


def _write_fx1(paths: DataPaths) -> None:
    """Quotes as Compustat has them (a last quote closes each range): EUR 2.0 from 2000-01-01, 3.0 from
    2000-03-01, both USD and EUR end on 2000-04-30; no CAD rows."""
    pl.DataFrame(
        {
            "curcdd": ["EUR", "EUR", "EUR", "USD", "USD"],
            "datadate": [
                date(2000, 1, 1),
                date(2000, 3, 1),
                date(2000, 4, 30),
                date(2000, 1, 1),
                date(2000, 4, 30),
            ],
            "fx": [2.0, 3.0, 3.0, 1.0, 1.0],
        },
        schema={"curcdd": pl.String, "datadate": pl.Date, "fx": pl.Float64},
    ).write_parquet(paths.interim_dir / "raw_data_dfs" / "__fx1.parquet")


@pytest.mark.unit
def test_convert_raw_vars_to_usd(test_paths: DataPaths) -> None:
    """The four raw variables are multiplied by the rate in force on public_date; other columns are untouched.

    - USD row: fx = 1, values unchanged.
    - EUR on 2000-02-29: rate 2.0 (the 2000-01-01 quote carried forward): assets 10 -> 20, sales 5 -> 10,
      book_equity 4 -> 8, net_income 1 -> 2.
    - EUR on 2000-03-31: rate 3.0: assets 10 -> 30.
    - EUR on 1999-12-31: before the first quote, no rate: null values.
    - Unknown currency: no rate: null values. Null input stays null.
    The 'other' column is not scaled and curcd is dropped.
    """
    _write_fx1(test_paths)
    df = pl.LazyFrame(
        {
            "gvkey": ["U", "E", "E", "E", "X", "N"],
            "curcd": ["USD", "EUR", "EUR", "EUR", "XXX", "EUR"],
            "public_date": [
                date(2000, 2, 29),
                date(2000, 2, 29),
                date(2000, 3, 31),
                date(1999, 12, 31),
                date(2000, 2, 29),
                date(2000, 2, 29),
            ],
            "assets": [10.0, 10.0, 10.0, 10.0, 10.0, None],
            "sales": [5.0] * 6,
            "book_equity": [4.0] * 6,
            "net_income": [1.0] * 6,
            "other": [7.0] * 6,
        }
    )
    out = convert_raw_vars_to_usd(test_paths, df).collect().sort("gvkey", "public_date")

    assert "curcd" not in out.columns
    expected = pl.DataFrame(
        {
            "gvkey": ["E", "E", "E", "N", "U", "X"],
            "public_date": [
                date(1999, 12, 31),
                date(2000, 2, 29),
                date(2000, 3, 31),
                date(2000, 2, 29),
                date(2000, 2, 29),
                date(2000, 2, 29),
            ],
            "assets": [None, 20.0, 30.0, None, 10.0, None],
            "sales": [None, 10.0, 15.0, 10.0, 5.0, None],
            "book_equity": [None, 8.0, 12.0, 8.0, 4.0, None],
            "net_income": [None, 2.0, 3.0, 2.0, 1.0, None],
            "other": [7.0] * 6,
            "fx": [None, 2.0, 3.0, 2.0, 1.0, None],
        }
    )
    assert_frame_equal(out.select(expected.columns), expected)


# ---------------------------------------------------------------------------
# add_me_data_and_compute_me_mev_mat_eqdur_vars
# ---------------------------------------------------------------------------

_ME_VARS = [
    "at_x", "be_x", "debt_x", "netdebt_x", "che", "sale_x", "gp_x", "ebitda_x", "ebit_x", "ope_x",
    "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "div_x", "eqbb_x", "eqis_x", "eqpo_x", "eqnpo_x",
    "eqnetis_x", "xrd",
]  # fmt: skip
_MEV_VARS = [
    "at_x", "bev_x", "ppent", "be_x", "che", "sale_x", "gp_x", "ebitda_x", "ebit_x", "ope_x",
    "ni_x", "nix_x", "cop_x", "ocf_x", "fcf_x", "debt_x", "pstk_x", "dltt", "dlc", "dltnetis_x",
    "dstnetis_x", "dbnetis_x", "netis_x", "fincf_x",
]  # fmt: skip
_OTHER_INPUTS = ["intrinsic_value", "aliq_x", "ed_cd_w", "ed_constant", "ed_cd"]


def _me_input(rows: list[dict]) -> pl.LazyFrame:
    """Rows with every input column present. Defaults: fx 1, all numeric inputs 1, ed_err 0, then overrides."""
    numeric = sorted(set(_ME_VARS + _MEV_VARS + _OTHER_INPUTS + ["fx"]))
    full = []
    for r in rows:
        base: dict = dict.fromkeys(numeric, 1.0)
        base.update({"ed_err": 0, "public_date": date(2000, 1, 31)})
        base.update(r)
        full.append(base)
    schema = dict.fromkeys(numeric, pl.Float64) | {
        "gvkey": pl.String,
        "public_date": pl.Date,
        "ed_err": pl.Int64,
    }
    return pl.DataFrame(full, schema=schema).lazy()


def _me_table(rows: list[tuple[str, date, float | None]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema={"gvkey": pl.String, "eom": pl.Date, "me_company": pl.Float64},
        orient="row",
    )


def _run_me(df: pl.LazyFrame, me: pl.LazyFrame) -> pl.DataFrame:
    return (
        add_me_data_and_compute_me_mev_mat_eqdur_vars(df, me).collect().sort("gvkey", "public_date")
    )


@pytest.mark.unit
def test_me_mev_mat_and_scaled_ratios_by_hand() -> None:
    """One firm-month: me 100, netdebt 50, fx 2, at 300, be 100, intrinsic 40, aliq 8.

    mev = me + netdebt*fx = 100 + 100 = 200
    mat = at*fx - be*fx + me = 600 - 200 + 100 = 500
    at_me = at*fx/me = 600/100 = 6;  at_mev = 600/200 = 3
    ival_me = intrinsic*fx/me = 80/100 = 0.8;  enterprise_value = mev = 200
    eq_dur = ed_cd_w*fx/me + ed_constant*(me - ed_cd*fx)/me with ed_cd_w 10, ed_constant 3, ed_cd 20
           = 20/100 + 3*(100 - 40)/100 = 0.2 + 1.8 = 2.0
    """
    df = _me_input(
        [
            {
                "gvkey": "A",
                "fx": 2.0,
                "netdebt_x": 50.0,
                "at_x": 300.0,
                "be_x": 100.0,
                "intrinsic_value": 40.0,
                "ed_cd_w": 10.0,
                "ed_constant": 3.0,
                "ed_cd": 20.0,
            }
        ]
    )
    out = _run_me(df, _me_table([("A", date(2000, 1, 31), 100.0)]))
    row = out.row(0, named=True)
    assert row["me_company"] == pytest.approx(100.0)
    assert row["at_me"] == pytest.approx(6.0)
    assert row["at_mev"] == pytest.approx(3.0)
    assert row["ival_me"] == pytest.approx(0.8)
    assert row["enterprise_value"] == pytest.approx(200.0)
    assert row["eq_dur"] == pytest.approx(2.0)
    # '_x' is stripped from the scaled names and suffixes added
    assert {"be_me", "netdebt_me", "bev_mev", "ppent_mev", "xrd_me", "fincf_mev"} <= set(
        out.columns
    )
    # netdebt_x: 50*2/100 = 1.0 ; bev_x (default 1): 1*2/200 = 0.01
    assert row["netdebt_me"] == pytest.approx(1.0)
    assert row["bev_mev"] == pytest.approx(0.01)


@pytest.mark.unit
def test_nonpositive_me_is_nulled_after_mev_is_built() -> None:
    """me = -10, netdebt 50, fx 2, at 300, be 100 (SAS builds mev and mat from the raw me, then nulls me <= 0).

    mev = -10 + 100 = 90 (positive, kept); mat = 600 - 200 - 10 = 390 (kept)
    me_company -> null, so every *_me and ival_me and eq_dur are null.
    at_mev = 600/90 = 6.6667 stays; enterprise_value = 90.
    """
    df = _me_input([{"gvkey": "A", "fx": 2.0, "netdebt_x": 50.0, "at_x": 300.0, "be_x": 100.0}])
    row = _run_me(df, _me_table([("A", date(2000, 1, 31), -10.0)])).row(0, named=True)
    assert row["me_company"] is None
    assert row["at_me"] is None and row["ival_me"] is None and row["eq_dur"] is None
    assert row["enterprise_value"] == pytest.approx(90.0)
    assert row["at_mev"] == pytest.approx(600.0 / 90.0)


@pytest.mark.unit
def test_zero_me_gives_null_me_ratios() -> None:
    """me = 0 is not positive: me_company null, no division by zero in the _me ratios or eq_dur."""
    df = _me_input([{"gvkey": "A"}])
    row = _run_me(df, _me_table([("A", date(2000, 1, 31), 0.0)])).row(0, named=True)
    assert row["me_company"] is None
    assert row["at_me"] is None and row["eq_dur"] is None
    # mev = 0 + 1*1 = 1 > 0 is still valid
    assert row["at_mev"] == pytest.approx(1.0)


@pytest.mark.unit
def test_nonpositive_mev_and_mat_are_null() -> None:
    """me 10, netdebt -20, fx 1, at 5, be 10: mev = 10 - 20 = -10 -> null; mat = 5 - 10 + 10 = 5 stays.
    A second firm with me 10, at 5, be 30 has mat = 5 - 30 + 10 = -15 -> null (aliq_mat uses lagged mat).
    """
    df = _me_input(
        [
            {"gvkey": "A", "netdebt_x": -20.0, "at_x": 5.0, "be_x": 10.0},
            {"gvkey": "B", "netdebt_x": 0.0, "at_x": 5.0, "be_x": 30.0},
        ]
    )
    me = _me_table([("A", date(2000, 1, 31), 10.0), ("B", date(2000, 1, 31), 10.0)])
    out = _run_me(df, me)
    a, b = out.row(0, named=True), out.row(1, named=True)
    assert a["enterprise_value"] is None and a["at_mev"] is None
    assert a["at_me"] == pytest.approx(0.5)
    assert b["enterprise_value"] == pytest.approx(10.0)
    assert b["at_mev"] == pytest.approx(0.5)


@pytest.mark.unit
def test_missing_market_equity_gives_null_market_variables() -> None:
    """A firm-month absent from the ME table (left join) gets null me, mev, mat-based and eq_dur values."""
    df = _me_input([{"gvkey": "A"}, {"gvkey": "B"}])
    out = _run_me(df, _me_table([("A", date(2000, 1, 31), 100.0)]))
    b = out.filter(pl.col("gvkey") == "B").row(0, named=True)
    assert b["me_company"] is None and b["enterprise_value"] is None
    assert b["at_me"] is None and b["at_mev"] is None and b["eq_dur"] is None
    assert out.filter(pl.col("gvkey") == "A")["at_me"].item() == pytest.approx(1.0 / 100.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # me 100, fx 1: 10/100 + 3*(100-20)/100 = 0.1 + 2.4 = 2.5
        ({"ed_cd_w": 10.0, "ed_constant": 3.0, "ed_cd": 20.0}, 2.5),
        # ed_err = 1 forces null
        ({"ed_cd_w": 10.0, "ed_constant": 3.0, "ed_cd": 20.0, "ed_err": 1}, None),
        # 0 + 3*(100-200)/100 = -3 <= 0 -> null
        ({"ed_cd_w": 0.0, "ed_constant": 3.0, "ed_cd": 200.0}, None),
        # exactly 0 -> null (SAS: eq_dur <= 0)
        ({"ed_cd_w": 0.0, "ed_constant": 0.0, "ed_cd": 0.0}, None),
    ],
)
def test_eq_dur_guards(overrides: dict, expected: float | None) -> None:
    df = _me_input([{"gvkey": "A", **overrides}])
    value = _run_me(df, _me_table([("A", date(2000, 1, 31), 100.0)]))["eq_dur"].item()
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


@pytest.mark.unit
def test_aliq_mat_uses_mat_twelve_months_earlier_within_firm() -> None:
    """aliq_mat = aliq_x*fx / mat lagged 12 firm-months, null if the lag belongs to another firm.

    Firms A and B each have 13 monthly rows. at 10, be 4, me 6 -> mat = 10 - 4 + 6 = 12 for A in the first
    month; A's month 0 has at 20 instead -> mat = 20 - 4 + 6 = 22. aliq_x = 11, fx = 1.
    Row 13 of A: 11/22 = 0.5. Row 13 of B: lag is B's month 0 with mat 12 -> 11/12.
    Rows 0..11 of either firm have no lag: null. B's first row must not pick up A's rows (gvkey guard).
    """
    months = pl.date_range(date(2000, 1, 1), date(2001, 1, 1), "1mo", eager=True).dt.month_end()
    rows, me_rows = [], []
    for g in ("A", "B"):
        for i, m in enumerate(months):
            at = 20.0 if (g == "A" and i == 0) else 10.0
            rows.append({"gvkey": g, "public_date": m, "at_x": at, "be_x": 4.0, "aliq_x": 11.0})
            me_rows.append((g, m, 6.0))
    out = _run_me(_me_input(rows), _me_table(me_rows))
    a = out.filter(pl.col("gvkey") == "A")["aliq_mat"].to_list()
    b = out.filter(pl.col("gvkey") == "B")["aliq_mat"].to_list()
    assert a[:12] == [None] * 12 and b[:12] == [None] * 12
    assert a[12] == pytest.approx(0.5)
    assert b[12] == pytest.approx(11.0 / 12.0)


@pytest.mark.unit
def test_duplicate_gvkey_public_date_rows_collapse_to_one() -> None:
    """Two input rows for the same (gvkey, public_date) leave one row (SAS: nodupkey on gvkey public_date)."""
    df = _me_input([{"gvkey": "A"}, {"gvkey": "A"}])
    out = _run_me(df, _me_table([("A", date(2000, 1, 31), 100.0)]))
    assert out.height == 1


# ---------------------------------------------------------------------------
# rename_cols_and_select_keep_vars
# ---------------------------------------------------------------------------


def _rename_input() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "source": ["NA"],
            "gvkey": ["001"],
            "public_date": [date(2000, 1, 31)],
            "datadate": [date(1999, 12, 31)],
            "xrd_at": [1.0],
            "sale_qtr_gr": [2.0],
            "xrd_xrd": [3.0],
            "ppent": [4.0],
            "dropped": [5.0],
        }
    )


@pytest.mark.unit
def test_rename_applies_first_match_and_selects_keep_vars_in_order() -> None:
    """Each rename key replaces only its first occurrence in a name; key columns come first, then keepers
    in vars_to_keep order; every other column is dropped.

    xrd_at -> rd_at;  sale_qtr_gr -> saleq_gr;  xrd_xrd -> rd_xrd (second 'xrd' untouched);  ppent -> ppen.
    """
    rename = {"xrd": "rd", "sale_qtr": "saleq", "ppent": "ppen"}
    out = rename_cols_and_select_keep_vars(
        _rename_input(), rename, ["ppen", "rd_xrd", "rd_at", "saleq_gr"], None
    ).collect()
    expected = pl.DataFrame(
        {
            "source": ["NA"],
            "gvkey": ["001"],
            "public_date": [date(2000, 1, 31)],
            "datadate": [date(1999, 12, 31)],
            "ppen": [4.0],
            "rd_xrd": [3.0],
            "rd_at": [1.0],
            "saleq_gr": [2.0],
        }
    )
    assert_frame_equal(out, expected)


@pytest.mark.unit
def test_suffix_is_added_to_kept_vars_only() -> None:
    """With a suffix, kept variables are renamed but source/gvkey/public_date/datadate are not."""
    out = rename_cols_and_select_keep_vars(
        _rename_input(), {"xrd": "rd"}, ["rd_at", "ppent"], "_qitem"
    ).collect()
    assert out.columns == [
        "source",
        "gvkey",
        "public_date",
        "datadate",
        "rd_at_qitem",
        "ppent_qitem",
    ]
    assert out["rd_at_qitem"].item() == 1.0 and out["ppent_qitem"].item() == 4.0


@pytest.mark.unit
def test_rename_replacements_chain_in_dict_order() -> None:
    """A later key can match the output of an earlier one: with {'zz': 'yy', 'yy': 'ww'}, 'zz_x' becomes 'ww_x'."""
    df = pl.LazyFrame(
        {
            "source": ["NA"],
            "gvkey": ["001"],
            "public_date": [date(2000, 1, 31)],
            "datadate": [date(1999, 12, 31)],
            "zz_x": [1.0],
        }
    )
    out = rename_cols_and_select_keep_vars(df, {"zz": "yy", "yy": "ww"}, ["ww_x"], None).collect()
    assert out["ww_x"].item() == 1.0
