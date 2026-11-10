"""Property-based invariants for accounting-stage helpers: quarterize, safe_div and combine_ann_qtr_chars.

Each test compares the Polars/DuckDB implementation with a plain-Python oracle written from the SAS
definition (accounting_chars.sas). Values are whole numbers stored as floats, so every arithmetic
operation is exact and the oracle can be compared with ==.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths

SETTINGS = settings(max_examples=30, deadline=None)

# quarterize


@st.composite
def ytd_panels(draw) -> tuple[pl.DataFrame, dict[tuple[str, int, int], float | None]]:
    """Fiscal years of year-to-date values built as the running sum of quarterly flows.

    Returns the panel and, per (gvkey, fyearq, fqtr) of each present row, the true flow of that quarter.
    Quarters are randomly missing, but the YTD of a later quarter still includes the missing flow, as in
    Compustat. Each firm keeps one fiscal year-end (fyr) so the fiscal year is the grouping key.
    """
    rows = []
    flows: dict[tuple[str, int, int], float | None] = {}
    for g in range(draw(st.integers(1, 2))):
        for fy in range(draw(st.integers(1, 3))):
            ytd = 0.0
            for q in range(1, 5):
                flow = float(draw(st.integers(-50, 100)))
                ytd += flow
                if draw(st.booleans()) or q == 1 and draw(st.booleans()):
                    month = 3 * q
                    rows.append(
                        {
                            "gvkey": f"{g:03d}",
                            "fyr": 12,
                            "fyearq": 2000 + fy,
                            "fqtr": q,
                            "datadate": date(2000 + fy, month, 28),
                            "ytd": ytd,
                        }
                    )
                    flows[(f"{g:03d}", 2000 + fy, q)] = flow
    schema = {
        "gvkey": pl.String,
        "fyr": pl.Int64,
        "fyearq": pl.Int64,
        "fqtr": pl.Int64,
        "datadate": pl.Date,
        "ytd": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema), flows


class TestQuarterizeInvariants:
    @given(panel=ytd_panels())
    @SETTINGS
    def test_inverts_year_to_date_sum(self, panel) -> None:
        """quarterize returns the quarter's own flow when the previous quarter of the same fiscal year is
        present (or the quarter is Q1), and null when a quarter before it is missing (the YTD difference
        would span several quarters)."""
        df, flows = panel
        out = af.quarterize(df.lazy(), ["ytd"]).collect()
        assert out.height == df.height
        for r in out.iter_rows(named=True):
            key = (r["gvkey"], r["fyearq"], r["fqtr"])
            has_prev = (r["gvkey"], r["fyearq"], r["fqtr"] - 1) in flows
            if r["fqtr"] == 1 or has_prev:
                assert r["ytd_q"] == flows[key]
            else:
                assert r["ytd_q"] is None

    @given(panel=ytd_panels())
    @SETTINGS
    def test_complete_years_sum_back_to_year_end(self, panel) -> None:
        """Where all four quarters of a fiscal year are present, the four flows add up to the Q4 YTD value."""
        df, _ = panel
        out = af.quarterize(df.lazy(), ["ytd"]).collect()
        for (_, _), grp in out.group_by(["gvkey", "fyearq"]):
            if grp.height == 4:
                assert grp["ytd_q"].sum() == grp.sort("fqtr")["ytd"][-1]


# safe_div

NUM = st.one_of(st.none(), st.integers(-6, 6).map(float))
DEN = st.one_of(st.none(), st.integers(-3, 3).map(float))


def _positive(x: float | None) -> bool:
    return x is not None and x > 0


def _nonzero(x: float | None) -> bool:
    return x is not None and x != 0


def _div(n: float | None, d: float | None) -> float | None:
    return None if n is None or d is None else n / d


@st.composite
def division_rows(draw) -> list[tuple[float | None, float | None, float | None]]:
    return draw(st.lists(st.tuples(NUM, DEN, st.integers(1, 3).map(float)), min_size=1, max_size=8))


class TestSafeDivRowModes:
    @given(rows=division_rows())
    @SETTINGS
    def test_row_wise_modes(self, rows) -> None:
        """Modes 1, 2, 3, 5 and 8 against their guard definitions.

        1: num/den if den != 0.  2: num/|den| if den != 0.  3: num/den if den > 0.
        5: num*fx/den if den != 0.  8: num/den if num > 0 and den > 0.  A failed guard or a null input gives null.
        """
        df = pl.DataFrame(
            rows,
            schema={"num": pl.Float64, "den": pl.Float64, "fx": pl.Float64},
            orient="row",
        )
        out = df.select(
            af.safe_div("num", "den", "m1", 1),
            af.safe_div("num", "den", "m2", 2),
            af.safe_div("num", "den", "m3", 3),
            af.safe_div("num", "den", "m5", 5),
            af.safe_div("num", "den", "m8", 8),
        )
        exp: dict[str, list[float | None]] = {k: [] for k in ("m1", "m2", "m3", "m5", "m8")}
        for n, d, fx in rows:
            exp["m1"].append(_div(n, d) if _nonzero(d) else None)
            exp["m2"].append(_div(n, abs(d)) if _nonzero(d) and n is not None else None)
            exp["m3"].append(_div(n, d) if _positive(d) else None)
            exp["m5"].append(_div(n * fx, d) if _nonzero(d) and n is not None else None)
            exp["m8"].append(_div(n, d) if _positive(n) and _positive(d) else None)
        for name, values in exp.items():
            assert out[name].to_list() == values, name


@st.composite
def lagged_panels(draw) -> pl.DataFrame:
    """Two firms, up to 16 consecutive months each, with a running count column."""
    rows = []
    for g in ("A", "B"):
        n = draw(st.integers(1, 16))
        for i in range(n):
            rows.append(
                {"gvkey": g, "curcd": "USD", "count": i + 1, "num": draw(NUM), "den": draw(DEN)}
            )
    schema = {
        "gvkey": pl.String,
        "curcd": pl.String,
        "count": pl.Int64,
        "num": pl.Float64,
        "den": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


class TestSafeDivLagModes:
    @given(df=lagged_panels())
    @SETTINGS
    def test_lag_modes_stay_within_firm(self, df: pl.DataFrame) -> None:
        """Modes 4, 6, 7 (12-month lag) and 9 (3-month lag) against the definition, per firm.

        4: num/den[-12] if count > 12 and den[-12] > 0.   6: num/den[-12] if den[-12] != 0.
        7: (num/den)[-12] if den[-12] > 0.                9: num/den[-3] if count > 3 and den[-3] > 0.
        Lags never reach into the other firm: a firm's first rows have no lag value, so they are null.
        """
        out = df.with_columns(
            af.safe_div("num", "den", "m4", 4),
            af.safe_div("num", "den", "m6", 6),
            af.safe_div("num", "den", "m7", 7),
            af.safe_div("num", "den", "m9", 9),
        )
        exp: dict[str, list[float | None]] = {k: [] for k in ("m4", "m6", "m7", "m9")}
        for g in ("A", "B"):
            sub = df.filter(pl.col("gvkey") == g).to_dicts()

            def lag(i: int, k: int, sub: list[dict] = sub) -> dict | None:
                return sub[i - k] if i >= k else None

            for i, r in enumerate(sub):
                l12, l3 = lag(i, 12), lag(i, 3)
                d12 = l12["den"] if l12 else None
                d3 = l3["den"] if l3 else None
                exp["m4"].append(
                    _div(r["num"], d12) if r["count"] > 12 and _positive(d12) else None
                )
                exp["m6"].append(_div(r["num"], d12) if _nonzero(d12) else None)
                exp["m7"].append(_div(l12["num"], d12) if l12 and _positive(d12) else None)
                exp["m9"].append(_div(r["num"], d3) if r["count"] > 3 and _positive(d3) else None)
        ordered = out.sort("gvkey", "count")
        for name, values in exp.items():
            assert ordered[name].to_list() == values, name


# combine_ann_qtr_chars

GVKEYS = ["001", "002", "003"]
MONTHS = [date(2016, m, 28) for m in (1, 2, 3)]
DATADATES = st.one_of(
    st.none(), st.sampled_from([date(2015, 6, 30), date(2015, 12, 31), date(2016, 3, 31)])
)
VALUE = st.one_of(st.none(), st.integers(0, 9).map(float))


@st.composite
def ann_qtr_inputs(draw) -> tuple[list[tuple], list[tuple]]:
    """Annual and quarterly rows, each unique on (gvkey, public_date), with random datadates and nulls."""
    keys = [(g, m) for g in GVKEYS for m in MONTHS]
    ann_keys = draw(st.lists(st.sampled_from(keys), unique=True, max_size=len(keys)))
    qtr_keys = draw(st.lists(st.sampled_from(keys), unique=True, max_size=len(keys)))
    ann = [("NA", g, m, draw(DATADATES), draw(VALUE)) for g, m in ann_keys]
    qtr = [("GLOBAL", g, m, draw(DATADATES), draw(VALUE)) for g, m in qtr_keys]
    return ann, qtr


class TestCombineInvariants:
    @given(inputs=ann_qtr_inputs())
    @SETTINGS
    def test_precedence_matches_sas_rule(self, inputs) -> None:
        """Output has one row per key in either input, sorted. The value is the quarterly one when the
        annual one is null, or when the quarterly one is present and its datadate is strictly later than the
        annual datadate (a null datadate never compares as later). Source is the annual one when it exists."""
        ann_rows, qtr_rows = inputs
        ann = pl.DataFrame(
            ann_rows,
            schema={
                "source": pl.String,
                "gvkey": pl.String,
                "public_date": pl.Date,
                "datadate": pl.Date,
                "assets": pl.Float64,
            },
            orient="row",
        )
        qtr = pl.DataFrame(
            qtr_rows,
            schema={
                "source": pl.String,
                "gvkey": pl.String,
                "public_date": pl.Date,
                "datadate": pl.Date,
                "assets_qitem": pl.Float64,
            },
            orient="row",
        )
        with tempfile.TemporaryDirectory() as td:
            paths = DataPaths(base_dir=Path(td))
            af.setup_folder_structure(paths)
            ann_path = paths.interim_dir / "a.parquet"
            qtr_path = paths.interim_dir / "q.parquet"
            ann.write_parquet(ann_path)
            qtr.write_parquet(qtr_path)
            af.combine_ann_qtr_chars(paths, ann_path, qtr_path, ["assets"], "_qitem")
            out = pl.read_parquet(paths.interim_dir / "acc_chars_world.parquet")

        a = {(g, m): (s, d, v) for s, g, m, d, v in ann_rows}
        q = {(g, m): (s, d, v) for s, g, m, d, v in qtr_rows}
        expected = []
        for key in sorted(set(a) | set(q)):
            a_src, a_dd, a_val = a.get(key, (None, None, None))
            q_src, q_dd, q_val = q.get(key, (None, None, None))
            # A missing annual datadate keeps the annual value: Python's rule (SAS would take the
            # quarterly one); the stage never builds such a row.
            q_newer = q_dd is not None and a_dd is not None and q_dd > a_dd
            take_q = key in q and (a_val is None or (q_val is not None and q_newer))
            expected.append((a_src or q_src, key[0], key[1], q_val if take_q else a_val))
        assert out.rows() == expected
