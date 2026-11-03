"""Property-based invariant for the four-quarter (TTM) sums built by cumulate_4q."""

from __future__ import annotations

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from jkp.data import aux_functions as af

SETTINGS = settings(max_examples=50, deadline=None)

SCHEMA = {
    "gvkey": pl.String,
    "fyr": pl.Int64,
    "fyearq": pl.Int64,
    "fqtr": pl.Int64,
    "curcdq": pl.String,
    "saleq": pl.Float64,
    "saley": pl.Float64,
}


@st.composite
def quarterly_panels(draw) -> pl.DataFrame:
    """Firms whose fiscal quarters step by 1, 2, 5 or 81 quarters (steps of 4k + 1 keep fqtr cycling
    1..4 across a gap), with one fiscal year-end change, at most one currency change, and 10% nulls.
    q = 4*fyearq + fqtr numbers the fiscal quarters. Rows are unique on (gvkey, fyr, fyearq, fqtr) and
    sorted by it, as cumulate_4q assumes. Values are whole numbers, so sums are exact in any order."""
    value = st.tuples(st.integers(0, 9), st.integers(-100, 1000)).map(
        lambda t: None if t[0] == 0 else float(t[1])
    )
    rows = []
    for g in range(draw(st.integers(1, 2))):
        steps = draw(st.lists(st.sampled_from([1, 1, 2, 5, 81]), max_size=11))
        quarters = [draw(st.integers(8000, 8003))]
        for step in steps:
            quarters.append(quarters[-1] + step)
        fyr_before, fyr_after = draw(st.permutations([12, 6]))
        fyr_switch = draw(st.integers(0, len(quarters)))
        cur_switch = draw(st.integers(0, len(quarters)))
        for i, q in enumerate(quarters):
            rows.append(
                {
                    "gvkey": f"{g:03d}",
                    "fyr": fyr_before if i < fyr_switch else fyr_after,
                    "fyearq": (q - 1) // 4,
                    "fqtr": (q - 1) % 4 + 1,
                    "curcdq": "USD" if i < cur_switch else "EUR",
                    "saleq": draw(value),
                    "saley": draw(value),
                }
            )
    return pl.DataFrame(rows, schema=SCHEMA).sort("gvkey", "fyr", "fyearq", "fqtr")


def _oracle(df: pl.DataFrame, var: str) -> list:
    """The TTM rule in plain Python: the sum of quarters q-3..q when all four exist for the same firm,
    fiscal year-end and currency and are non-null; otherwise the year-to-date value at Q4, else null."""
    rows = df.to_dicts()
    by_quarter = {(r["gvkey"], r["fyr"], 4 * r["fyearq"] + r["fqtr"]): r for r in rows}
    out = []
    for r in rows:
        q = 4 * r["fyearq"] + r["fqtr"]
        window = [by_quarter.get((r["gvkey"], r["fyr"], q - k)) for k in range(4)]
        if all(w and w["curcdq"] == r["curcdq"] and w[var] is not None for w in window):
            out.append(sum(w[var] for w in window))
        else:
            out.append(r[f"{var[:-1]}y"] if r["fqtr"] == 4 else None)
    return out


class TestCumulate4qInvariants:
    @given(df=quarterly_panels())
    @SETTINGS
    def test_matches_consecutive_quarter_rule(self, df: pl.DataFrame) -> None:
        out = af.cumulate_4q(df.lazy(), ["saleq"]).collect()
        assert out["sale"].to_list() == _oracle(df, "saleq")
