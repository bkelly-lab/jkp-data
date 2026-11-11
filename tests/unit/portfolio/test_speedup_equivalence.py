"""Equivalence of the fast portfolio paths with the code they replaced.

The ``_old_*`` functions are frozen copies of the slower implementations. The panel
is built to break a rewrite: ties across breakpoints, a third of breakpoint stocks
tied at the minimum, signed zeros, infinities, nulls in one return column only, NaN
daily returns, zero-``me`` months, thin months, single-stock groups and a month with
more than 255 breakpoint stocks.

Keys, counts and null, NaN and infinity positions must match exactly. Finite floats
may differ by ``TOLERANCE * max(1, |old|)``. Every comparison runs twice: with the
compiled Numba kernels and with the kernels run as plain Python, as under
``NUMBA_DISABLE_JIT=1``. NumPy keeps ``uint8`` arithmetic in ``uint8``, where it
wraps past 255; Numba widens it to ``int64``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numba.core.dispatcher import Dispatcher

from jkp.data import aux_functions
from jkp.data.aux_functions import (
    _build_hml,
    _cmp_returns,
    _month_totals,
    _pf_daily_returns,
    _pf_returns,
    _rank_by_month,
    _stock_days,
)
from jkp.data.config import CMP_SIZE_GROUPS
from jkp.data.portfolio import run_portfolio
from tests.fixtures.portfolio_oi_legacy import add_ecdf
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    build_synthetic_data,
    make_multi_region_classification,
    make_themes_dir,
    month_ends,
    weekdays_in_month_after,
)

TOLERANCE = 1e-10
PFS = 3
BP_MIN_N = 5
RET_COLS = ["ret_exc", "ret_overnight"]
CHARS = [
    "c_cont",
    "c_ties",
    "c_min_tied",
    "c_signed_zero",
    "c_inf",
    "c_nan",
    "c_all_null",
    "c_constant_month",
    "c_inf_month",
]
EOMS = month_ends(6)
WIDE_MONTH = EOMS[4]


def _old_pf_returns_lazy(sub: pl.LazyFrame, x: str) -> pl.LazyFrame:
    """Frozen: one group-by per characteristic."""
    return (
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


def _old_pf_daily_lazy(
    sub: pl.LazyFrame, daily: pl.DataFrame, x: str, ret_cols: list[str]
) -> pl.LazyFrame:
    """Frozen: weights exploded per characteristic and joined to all daily rows."""
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
    return (
        weights.join(
            daily.lazy(),
            left_on=["id", "eom"],
            right_on=["id", "eom_lag1"],
            how="left",
        )
        .filter(pl.col("pf").is_not_null() & pl.any_horizontal(pl.col(ret_cols).is_not_null()))
        .group_by(["pf", "date"])
        .agg(
            [pl.lit(x).alias("characteristic")]
            + [
                expr
                for c in ret_cols
                for expr in (
                    pl.col(c).count().alias(f"n_{c}"),
                    (pl.col("w_ew") * pl.col(c)).sum().alias(f"ret_ew_{c}"),
                    (pl.col("w_vw") * pl.col(c)).sum().alias(f"ret_vw_{c}"),
                    (pl.col("w_vw_cap") * pl.col(c)).sum().alias(f"ret_vw_cap_{c}"),
                )
            ]
        )
    )


def _old_split_daily_leg(wide: pl.DataFrame, c: str) -> pl.DataFrame:
    """Frozen: one return column's rows of the wide daily result."""
    return wide.filter(pl.col(f"n_{c}") > 0).select(
        "pf",
        "date",
        "characteristic",
        pl.col(f"n_{c}").alias("n"),
        pl.col(f"ret_ew_{c}").alias("ret_ew"),
        pl.col(f"ret_vw_{c}").alias("ret_vw"),
        pl.col(f"ret_vw_cap_{c}").alias("ret_vw_cap"),
    )


def _old_cmp_size_group(n_groups: int) -> pl.Expr:
    """Frozen: size group of a stock with ``n_groups`` size groups."""
    if n_groups == 1:
        return pl.lit("all")
    if n_groups == 2:
        return (
            pl.when(pl.col("size_grp").is_in(["mega", "large"]))
            .then(pl.lit("big"))
            .otherwise(pl.lit("small"))
        )
    return pl.col("size_grp")


def _old_cmp_returns(
    data: pl.DataFrame, chars: list[str], n_groups: int, excntry: str
) -> pl.DataFrame:
    """Frozen: one unpivoted long table with window functions on three keys."""
    grp = ["characteristic", "size_grp", "eom"]
    n_grp = pl.len().over(grp)
    p_rank = pl.col("var").rank("average").over(grp) / (n_grp + 1)
    p_rank_dev = p_rank - p_rank.mean().over(grp)
    weight = p_rank_dev / (p_rank_dev.abs().sum().over(grp) / 2)
    return (
        data.lazy()
        .with_columns(_old_cmp_size_group(n_groups).alias("size_grp"))
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


def _old_build_hml(
    pf_df: pl.DataFrame, n_pfs: int, date_col: str, include_signal: bool
) -> pl.DataFrame:
    """Frozen: filter-then-first aggregations per (country, characteristic, date)."""
    diff = lambda c: (  # noqa: E731
        pl.col(c).filter(pl.col("pf") == n_pfs).first()
        - pl.col(c).filter(pl.col("pf") == 1).first()
    )
    is_extreme_pf = (pl.col("pf") == n_pfs) | (pl.col("pf") == 1)
    agg_exprs = [is_extreme_pf.sum().alias("pfs")]
    if include_signal:
        agg_exprs.append(diff("signal").alias("signal"))
    agg_exprs.extend(
        [
            (
                pl.col("n").filter(pl.col("pf") == n_pfs).first()
                + pl.col("n").filter(pl.col("pf") == 1).first()
            ).alias("n_stocks"),
            pl.col("n").filter(is_extreme_pf).min().alias("n_stocks_min"),
            diff("ret_ew").alias("ret_ew"),
            diff("ret_vw").alias("ret_vw"),
            diff("ret_vw_cap").alias("ret_vw_cap"),
        ]
    )
    return (
        pf_df.group_by(["excntry", "characteristic", date_col])
        .agg(agg_exprs)
        .filter(pl.col("pfs") == 2)
        .drop("pfs")
        .sort(["excntry", "characteristic", date_col])
    )


def _assert_within_tolerance(old: pl.DataFrame, new: pl.DataFrame, keys: list[str]) -> None:
    """Exact on schema, keys, counts and special values; tolerance on finite floats."""
    assert old.schema == new.schema
    assert old.height > 0
    a, b = old.sort(keys), new.sort(keys)
    assert a.select(keys).equals(b.select(keys))
    for c in a.columns:
        if c in keys:
            continue
        if a.schema[c] != pl.Float64:
            assert a[c].equals(b[c]), c
            continue
        assert a[c].is_null().equals(b[c].is_null()), f"{c}: null positions"
        x, y = a[c].to_numpy(), b[c].to_numpy()
        assert np.array_equal(np.isnan(x), np.isnan(y)), f"{c}: NaN positions"
        assert np.array_equal(np.isposinf(x), np.isposinf(y)), f"{c}: +inf positions"
        assert np.array_equal(np.isneginf(x), np.isneginf(y)), f"{c}: -inf positions"
        finite = np.isfinite(x)
        limit = TOLERANCE * np.maximum(1.0, np.abs(x[finite]))
        assert (np.abs(x[finite] - y[finite]) <= limit).all(), c


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    """Stock-months with the awkward cases named in the module docstring."""
    rng = np.random.default_rng(7)
    frame = (
        pl.DataFrame({"id": np.arange(1, 121)})
        .join(pl.DataFrame({"eom": pl.Series(EOMS, dtype=pl.Date)}), how="cross")
        .filter(pl.Series(rng.random(720) > 0.1))
    )
    # More breakpoint stocks than a uint8 counts.
    wide = pl.DataFrame({"id": np.arange(121, 521)}).with_columns(
        eom=pl.lit(WIDE_MONTH, dtype=pl.Date)
    )
    frame = pl.concat([frame, wide])
    n = frame.height
    groups = ["mega", "large", "small", "micro"]
    zero_me_month = pl.col("eom") == EOMS[2]
    thin_month = pl.col("eom") == EOMS[5]
    return (
        frame.with_columns(
            pl.Series("ret_exc_lead1m", rng.normal(0, 0.1, n)),
            pl.Series("me", rng.lognormal(3, 2, n)),
            pl.Series("size_grp", rng.choice(groups, n)),
            pl.Series("c_cont", rng.normal(0, 1, n)),
            pl.Series("c_ties", rng.integers(0, 3, n), dtype=pl.Float64),
            pl.Series("c_min_tied", np.where(rng.random(n) < 0.5, 0.0, rng.random(n) + 1)),
            pl.Series("c_signed_zero", rng.choice([-0.0, 0.0, 1.0, -1.0], n)),
            pl.Series(
                "c_inf",
                np.where(
                    rng.random(n) < 0.05,
                    rng.choice([np.inf, -np.inf], n),
                    rng.normal(0, 1, n),
                ),
            ),
            pl.Series("c_nan", np.where(rng.random(n) < 0.4, np.nan, rng.normal(0, 1, n))),
            pl.Series("c_constant_month", rng.normal(0, 1, n)),
            pl.Series("c_inf_month", rng.normal(0, 1, n)),
        )
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("c_all_null"),
            pl.col("c_nan").fill_nan(None),
            pl.when(pl.col("eom") == EOMS[1])
            .then(2.5)
            .otherwise(pl.col("c_constant_month"))
            .alias("c_constant_month"),
            # Every stock infinite: the CMP standard deviation is NaN, so the group stays.
            pl.when(pl.col("eom") == EOMS[3])
            .then(np.inf)
            .otherwise(pl.col("c_inf_month"))
            .alias("c_inf_month"),
            pl.when(zero_me_month).then(0.0).otherwise(pl.col("me")).alias("me"),
            # One nano stock a month: a single-stock group in the five-group CMP.
            pl.when(pl.col("id") == pl.col("id").min().over("eom"))
            .then(pl.lit("nano"))
            .otherwise(pl.col("size_grp"))
            .alias("size_grp"),
        )
        .with_columns(
            pl.min_horizontal("me", pl.lit(50.0)).alias("me_cap"),
            # Too few breakpoint stocks in the thin month.
            (
                pl.col("size_grp").is_in(["mega", "large", "small"])
                & (~thin_month | (pl.col("id") < 6))
            ).alias("bp_stock"),
        )
        .sort("eom", "id")
        .with_row_index("sm")
    )


@pytest.fixture(scope="module")
def daily(panel: pl.DataFrame) -> pl.DataFrame:
    """Five days after each month: some without a stock-month, some with one return."""
    rng = np.random.default_rng(11)
    dates = [d for eom in EOMS for d in weekdays_in_month_after(eom, 5)]
    frame = (
        pl.DataFrame({"id": [*range(1, 521), 999]})
        .join(pl.DataFrame({"date": pl.Series(dates, dtype=pl.Date)}), how="cross")
        .with_columns(pl.col("date").dt.month_start().dt.offset_by("-1d").alias("eom_lag1"))
    )
    # Stock-months with no daily rows.
    dropped = panel.select(id="id", eom_lag1="eom").sample(fraction=0.05, seed=3)
    frame = frame.join(dropped, on=["id", "eom_lag1"], how="anti")
    n = frame.height
    ret_exc = rng.normal(0, 0.02, n)
    ret_exc[rng.random(n) < 0.01] = np.nan
    only_overnight = pl.col("date") == dates[3]
    only_ret_exc = pl.col("date") == dates[7]
    return frame.with_columns(
        pl.Series("ret_exc", ret_exc),
        pl.Series("ret_overnight", rng.normal(0, 0.01, n)),
        pl.Series("drop_exc", rng.random(n) < 0.05),
        pl.Series("drop_on", rng.random(n) < 0.1),
    ).with_columns(
        pl.when(pl.col("drop_exc") | only_overnight)
        .then(None)
        .otherwise("ret_exc")
        .alias("ret_exc"),
        pl.when(pl.col("drop_on") | only_ret_exc)
        .then(None)
        .otherwise("ret_overnight")
        .alias("ret_overnight"),
    )


def _assigned(panel: pl.DataFrame, x: str) -> pl.LazyFrame:
    """Portfolio assignment as in ``portfolios()``."""
    sub = (
        panel.lazy()
        .with_columns(pl.col(x).alias("var"))
        .filter(pl.col("var").is_not_null())
        .with_columns(bp_n=pl.sum("bp_stock").over("eom"))
        .filter(pl.col("bp_n") >= BP_MIN_N)
    )
    return (
        add_ecdf(sub)
        .with_columns(
            pl.when(pl.col("cdf") == pl.col("cdf").min().over("eom"))
            .then(0.00000001)
            .otherwise(pl.col("cdf"))
            .alias("cdf")
        )
        .with_columns((pl.col("cdf") * PFS).ceil().clip(lower_bound=1, upper_bound=PFS).alias("pf"))
    )


@contextmanager
def _kernels(mode: str) -> Iterator[None]:
    """Run the Numba kernels compiled, or as plain Python as under ``NUMBA_DISABLE_JIT=1``.

    Plain Python ignores float division by zero and invalid operations, as Numba's
    ``error_model="numpy"`` does. Integer overflow still warns.
    """
    if mode == "compiled":
        yield
        return
    with pytest.MonkeyPatch.context() as mp, np.errstate(divide="ignore", invalid="ignore"):
        for name, obj in list(vars(aux_functions).items()):
            if isinstance(obj, Dispatcher):
                mp.setattr(aux_functions, name, obj.py_func)
        yield


@pytest.fixture(scope="module", params=["compiled", "plain_python"])
def kernels(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def ranked(panel: pl.DataFrame, kernels: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with _kernels(kernels):
        return _rank_by_month(panel, CHARS, PFS, BP_MIN_N)


@pytest.fixture(scope="module")
def pf_by_row(ranked: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return ranked[0]


@pytest.fixture(scope="module")
def month_sums(
    panel: pl.DataFrame, pf_by_row: np.ndarray, kernels: str
) -> tuple[pl.Series, np.ndarray]:
    with _kernels(kernels):
        return _month_totals(panel, pf_by_row, PFS)


@pytest.fixture(scope="module")
def new_monthly(
    ranked: tuple[np.ndarray, np.ndarray, np.ndarray],
    month_sums: tuple[pl.Series, np.ndarray],
    kernels: str,
) -> pl.DataFrame:
    months, totals = month_sums
    with _kernels(kernels):
        return _pf_returns(months, totals, ranked[1], CHARS)


@pytest.fixture(scope="module")
def new_daily(
    panel: pl.DataFrame,
    daily: pl.DataFrame,
    pf_by_row: np.ndarray,
    month_sums: tuple[pl.Series, np.ndarray],
    kernels: str,
) -> dict[str, pl.DataFrame]:
    months, totals = month_sums
    stock_days = _stock_days(panel, daily, RET_COLS)
    with _kernels(kernels):
        return _pf_daily_returns(stock_days, pf_by_row, months, totals, CHARS, RET_COLS, PFS)


@pytest.mark.unit
class TestAssignmentEquivalence:
    @pytest.mark.parametrize("x", CHARS)
    def test_portfolios_match_old_exactly(
        self, panel: pl.DataFrame, pf_by_row: np.ndarray, x: str
    ) -> None:
        assigned = _assigned(panel, x).collect()
        old = np.zeros(panel.height, dtype=np.uint8)
        old[assigned["sm"].to_numpy()] = assigned["pf"].to_numpy()
        assert np.array_equal(old, pf_by_row[:, CHARS.index(x)])


@pytest.mark.unit
class TestMonthlyEquivalence:
    @pytest.mark.parametrize("x", [c for c in CHARS if c != "c_all_null"])
    def test_monthly_matches_old(
        self, panel: pl.DataFrame, new_monthly: pl.DataFrame, x: str
    ) -> None:
        old = _old_pf_returns_lazy(_assigned(panel, x), x).collect()
        new = new_monthly.filter(pl.col("characteristic") == x)
        _assert_within_tolerance(old, new, ["pf", "eom"])

    def test_all_null_characteristic_gives_no_rows(self, new_monthly: pl.DataFrame) -> None:
        assert "c_all_null" not in new_monthly["characteristic"].to_list()


@pytest.mark.unit
class TestDailyEquivalence:
    @pytest.mark.parametrize("x", [c for c in CHARS if c != "c_all_null"])
    @pytest.mark.parametrize("ret_col", RET_COLS)
    def test_daily_leg_matches_old(
        self,
        panel: pl.DataFrame,
        daily: pl.DataFrame,
        new_daily: dict[str, pl.DataFrame],
        x: str,
        ret_col: str,
    ) -> None:
        wide = _old_pf_daily_lazy(_assigned(panel, x), daily, x, RET_COLS).collect()
        new = new_daily[ret_col].filter(pl.col("characteristic") == x)
        _assert_within_tolerance(_old_split_daily_leg(wide, ret_col), new, ["pf", "date"])

    @pytest.mark.parametrize("ret_col", RET_COLS)
    def test_all_null_characteristic_gives_no_rows(
        self, new_daily: dict[str, pl.DataFrame], ret_col: str
    ) -> None:
        assert "c_all_null" not in new_daily[ret_col]["characteristic"].to_list()

    def test_panel_reaches_the_awkward_cases(
        self, panel: pl.DataFrame, daily: pl.DataFrame
    ) -> None:
        """Guard the fixture: each edge case must actually occur."""
        min_tied = _assigned(panel, "c_min_tied").collect()
        assert min_tied.filter(pl.col("var") == 0.0)["cdf"].unique().to_list() == [0.00000001]
        assert min_tied.filter(pl.col("var") == 0.0)["pf"].unique().to_list() == [1.0]
        leg = _old_split_daily_leg(
            _old_pf_daily_lazy(_assigned(panel, "c_cont"), daily, "c_cont", RET_COLS).collect(),
            "ret_exc",
        )
        assert leg["ret_vw"].is_nan().sum() > 0
        monthly = _old_pf_returns_lazy(_assigned(panel, "c_cont"), "c_cont").collect()
        assert monthly["ret_vw"].is_nan().sum() > 0
        assert EOMS[5] not in _assigned(panel, "c_cont").collect()["eom"].to_list()
        assert panel.filter(pl.col("eom") == WIDE_MONTH)["bp_stock"].sum() > 255


@pytest.mark.unit
class TestCmpEquivalence:
    @pytest.mark.parametrize("n_groups", CMP_SIZE_GROUPS)
    def test_cmp_matches_old(
        self,
        panel: pl.DataFrame,
        ranked: tuple[np.ndarray, np.ndarray, np.ndarray],
        month_sums: tuple[pl.Series, np.ndarray],
        n_groups: int,
    ) -> None:
        old = _old_cmp_returns(panel, CHARS, n_groups, "usa")
        new = _cmp_returns(month_sums[0], ranked[2], CHARS, n_groups, "usa")
        _assert_within_tolerance(old, new, ["characteristic", "size_grp", "eom"])

    def test_constant_and_single_stock_groups_are_dropped(
        self,
        ranked: tuple[np.ndarray, np.ndarray, np.ndarray],
        month_sums: tuple[pl.Series, np.ndarray],
    ) -> None:
        new = _cmp_returns(month_sums[0], ranked[2], CHARS, 5, "usa")
        assert "nano" not in new["size_grp"].to_list()
        assert "c_all_null" not in new["characteristic"].to_list()
        constant_return_month = pl.Series([EOMS[1]]).dt.offset_by("1mo").dt.month_end()[0]
        constant = new.filter(pl.col("characteristic") == "c_constant_month")
        assert constant_return_month not in constant["eom"].to_list()

    def test_group_of_infinite_values_stays_with_nan(
        self,
        ranked: tuple[np.ndarray, np.ndarray, np.ndarray],
        month_sums: tuple[pl.Series, np.ndarray],
    ) -> None:
        new = _cmp_returns(month_sums[0], ranked[2], CHARS, 1, "usa")
        return_month = pl.Series([EOMS[3]]).dt.offset_by("1mo").dt.month_end()[0]
        row = new.filter(
            (pl.col("characteristic") == "c_inf_month") & (pl.col("eom") == return_month)
        )
        assert row.height == 1
        assert row["ret_weighted"].is_nan().all()


@pytest.mark.unit
class TestHmlEquivalence:
    @pytest.fixture
    def pf_df(self) -> pl.DataFrame:
        """Full dates, plus dates without a top, a bottom, or either extreme portfolio."""
        rng = np.random.default_rng(5)
        rows = [
            {"excntry": e, "characteristic": c, "eom": d, "pf": float(pf)}
            for e in ("GBR", "USA")
            for c in ("a", "b")
            for d in EOMS
            for pf in (1, 2, 3)
            if not (d == EOMS[1] and pf == 3)
            and not (d == EOMS[2] and pf == 1)
            and not (d == EOMS[3] and pf != 2)
        ]
        n = len(rows)
        ret_ew = rng.normal(0, 0.02, n)
        ret_ew[0] = np.nan
        return (
            pl.DataFrame(rows)
            .with_columns(
                pl.Series("n", rng.integers(1, 50, n), dtype=pl.UInt32),
                pl.Series("signal", rng.normal(0, 1, n)),
                pl.Series("ret_ew", ret_ew),
                pl.Series("ret_vw", rng.normal(0, 0.02, n)),
                pl.Series("ret_vw_cap", rng.normal(0, 0.02, n)),
            )
            .with_columns(
                pl.when(pl.col("n") < 5).then(None).otherwise("ret_vw").alias("ret_vw"),
            )
        )

    @pytest.mark.parametrize("include_signal", [True, False])
    def test_hml_is_identical(self, pf_df: pl.DataFrame, include_signal: bool) -> None:
        old = _old_build_hml(pf_df, PFS, "eom", include_signal)
        new = _build_hml(pf_df, PFS, "eom", include_signal)
        assert old.schema == new.schema
        assert old.equals(new)
        assert EOMS[1] not in new["eom"].to_list()
        assert EOMS[3] not in new["eom"].to_list()


@pytest.mark.unit
def test_intraday_local_files_reuse_intraday(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_output_writer: None
) -> None:
    """The local intraday outputs come from one aggregation of ``ret_intraday``."""
    import jkp.data.portfolio as pf_mod

    chars = SYNTHETIC_CHARS[:3]
    build_synthetic_data(tmp_path, ["USA", "GBR"], chars, n_ids=40, n_months=6)
    daily_dir = tmp_path / "processed" / "return_data" / "daily_rets_by_country"
    for path in daily_dir.glob("*.parquet"):
        rng = np.random.default_rng(1)
        df = pl.read_parquet(path)
        df.with_columns(
            pl.Series("ret_overnight", rng.normal(0, 0.01, df.height)),
            pl.Series("ret_intraday", rng.normal(0, 0.01, df.height)),
            pl.Series("ret_overnight_local", rng.normal(0, 0.01, df.height)),
        ).with_columns(pl.col("ret_intraday").alias("ret_intraday_local")).write_parquet(path)

    resources = tmp_path / "resources"
    resources.mkdir()
    themes_dir = make_themes_dir(resources, dict.fromkeys(chars, "cluster_alpha"))
    classification = make_multi_region_classification(resources)
    monkeypatch.setattr("jkp.data.paths.get_themes_dir", lambda: themes_dir)
    monkeypatch.setattr("jkp.data.paths.get_country_classification_path", lambda: classification)
    monkeypatch.setattr(pf_mod, "PORTFOLIO_CHARS", chars)
    monkeypatch.setattr(
        pf_mod,
        "PORTFOLIO_SETTINGS",
        {
            "end_date": date(2030, 12, 31),
            "pfs": PFS,
            "source": ["CRSP", "COMPUSTAT"],
            "wins_ret": True,
            "bps": "non_mc",
            "bp_min_n": 3,
            "cmp": {"us": False, "int": False},
            "regional_pfs": {
                "country_excl": [],
                "country_weights": "market_cap",
                "stocks_min": 1,
                "months_min": 1,
                "countries_min": 1,
            },
            "daily_pf": True,
            "ind_pf": False,
        },
    )

    run_portfolio(output_dir=tmp_path)

    portfolios = tmp_path / "processed" / "portfolios"
    for stem in ("pfs", "hml"):
        intraday = pl.read_parquet(portfolios / f"{stem}_intraday.parquet")
        assert intraday.height > 0
        assert intraday.equals(pl.read_parquet(portfolios / f"{stem}_intraday_local.parquet"))
        assert (portfolios / f"{stem}_overnight_local.parquet").exists()
