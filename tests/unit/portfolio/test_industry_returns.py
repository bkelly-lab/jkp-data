"""Unit tests for industry-portfolio builders.

Covers ``_build_industry_monthly_returns``, ``_build_industry_daily_returns``,
plus the ``portfolios()``-level industry behaviour (GICS extraction, FF49
USA-only guard, ``bp_min_n`` filter, case-insensitive ``excntry`` gating).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from jkp.data.portfolio import (
    _build_industry_daily_returns,
    _build_industry_monthly_returns,
    portfolios,
)
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    make_country_characteristics,
    make_cutoffs,
    make_daily_returns,
)

_TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def _write_synthetic_country(
    data_root: Path,
    excntry: str,
    chars: list[str],
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Write synthetic characteristics + daily returns for ``excntry``."""
    char_dir = data_root / "characteristics"
    daily_dir = data_root / "return_data" / "daily_rets_by_country"
    char_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    char_df = make_country_characteristics(excntry=excntry, chars=chars, seed=seed)
    char_df.write_parquet(char_dir / f"{excntry}.parquet")
    daily_df = make_daily_returns(char_df, seed=seed + 1)
    daily_df.write_parquet(daily_dir / f"{excntry}.parquet")
    return char_df, daily_df


def _assert_frames_parity(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    key_cols: list[str],
    numeric_cols: dict[str, dict[str, float]],
    label: str,
) -> None:
    """Compare two frames: key cols exact, numeric cols within tolerance."""
    import numpy as np

    assert actual.height == expected.height, (
        f"[{label}] height mismatch: {actual.height} vs {expected.height}"
    )
    assert actual.height > 0, f"[{label}] empty frame"
    a = actual.sort(key_cols)
    e = expected.sort(key_cols)
    for col in key_cols:
        assert a[col].to_list() == e[col].to_list(), f"[{label}] key {col!r} mismatch"
    for col, tol in numeric_cols.items():
        a_np = a[col].to_numpy().astype(np.float64)
        e_np = e[col].to_numpy().astype(np.float64)
        a_nan = np.isnan(a_np)
        e_nan = np.isnan(e_np)
        assert np.array_equal(a_nan, e_nan), f"[{label}] NaN positions differ in {col!r}"
        mask = ~a_nan
        if mask.any():
            np.testing.assert_allclose(
                a_np[mask],
                e_np[mask],
                err_msg=f"[{label}] {col!r} beyond tolerance",
                **tol,
            )


# =============================================================================
# Direct unit tests for _build_industry_monthly_returns
# =============================================================================


def _make_monthly_input(
    n_per_group: int = 3,
    eoms: list[date] | None = None,
    industries: list[int] | None = None,
) -> pl.DataFrame:
    """Build a small monthly input frame with ``id, eom, gics, ff49, me, me_cap, ret_exc_lead1m``."""
    if eoms is None:
        eoms = [date(2020, 1, 31), date(2020, 2, 29)]
    if industries is None:
        industries = [10, 20]

    ids: list[int] = []
    eom_col: list[date] = []
    gics_col: list[str] = []
    ff49_col: list[int] = []
    me_col: list[float] = []
    me_cap_col: list[float] = []
    ret_col: list[float] = []
    next_id = 1
    for eom in eoms:
        for ind in industries:
            for k in range(n_per_group):
                ids.append(next_id)
                next_id += 1
                eom_col.append(eom)
                gics_col.append(f"{ind:02d}101010")
                ff49_col.append(ind)
                me_col.append(100.0 + k * 10.0)
                me_cap_col.append(100.0 + k * 10.0)
                ret_col.append(0.01 + 0.001 * k)
    return pl.DataFrame(
        {
            "id": pl.Series("id", ids, dtype=pl.Int64),
            "eom": pl.Series("eom", eom_col, dtype=pl.Date),
            "gics": pl.Series("gics", gics_col, dtype=pl.Utf8),
            "ff49": pl.Series("ff49", ff49_col, dtype=pl.Int64),
            "me": pl.Series("me", me_col, dtype=pl.Float64),
            "me_cap": pl.Series("me_cap", me_cap_col, dtype=pl.Float64),
            "ret_exc_lead1m": pl.Series("ret_exc_lead1m", ret_col, dtype=pl.Float64),
        }
    )


class TestBuildIndustryMonthlyReturns:
    """Direct tests for ``_build_industry_monthly_returns``."""

    def test_groups_by_industry_and_eom(self):
        df = _make_monthly_input(n_per_group=3)
        out = _build_industry_monthly_returns(
            data=df, industry_col="ff49", bp_min_n=1, excntry="USA"
        )
        # 2 industries x 2 eoms = 4 rows
        assert out.height == 4
        assert set(out.columns) == {
            "ff49",
            "eom",
            "n",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
            "excntry",
        }
        # n = 3 in every group
        assert out["n"].to_list() == [3, 3, 3, 3]

    def test_advances_eom_by_one_month(self):
        df = _make_monthly_input(n_per_group=2, eoms=[date(2020, 1, 31)])
        out = _build_industry_monthly_returns(
            data=df, industry_col="ff49", bp_min_n=1, excntry="USA"
        )
        eoms = out["eom"].unique().to_list()
        assert eoms == [date(2020, 2, 29)]

    def test_uppercase_excntry(self):
        df = _make_monthly_input(n_per_group=2)
        out = _build_industry_monthly_returns(
            data=df, industry_col="ff49", bp_min_n=1, excntry="usa"
        )
        assert out["excntry"].unique().to_list() == ["USA"]

    def test_industry_transform_applied(self):
        df = _make_monthly_input(
            n_per_group=2,
            eoms=[date(2020, 1, 31)],
            industries=[10, 20],
        )
        out = _build_industry_monthly_returns(
            data=df,
            industry_col="gics",
            bp_min_n=1,
            excntry="USA",
            industry_transform=pl.col("gics").str.slice(0, 2).cast(pl.Int64),
        )
        sectors = sorted(out["gics"].unique().to_list())
        assert sectors == [10, 20]
        assert out["gics"].dtype == pl.Int64

    def test_bp_min_n_filters_small_groups(self):
        # 3 stocks per group: with bp_min_n=10 nothing should remain.
        df = _make_monthly_input(n_per_group=3)
        out = _build_industry_monthly_returns(
            data=df, industry_col="ff49", bp_min_n=10, excntry="USA"
        )
        assert out.height == 0

        # With bp_min_n=3 every group is kept.
        out = _build_industry_monthly_returns(
            data=df, industry_col="ff49", bp_min_n=3, excntry="USA"
        )
        assert out.height == 4
        assert (out["n"] >= 3).all()

    def test_returns_schema(self):
        df = _make_monthly_input(n_per_group=2)
        out = _build_industry_monthly_returns(
            data=df, industry_col="ff49", bp_min_n=1, excntry="USA"
        )
        assert out.columns == [
            "ff49",
            "eom",
            "n",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
            "excntry",
        ]


# =============================================================================
# Direct unit tests for _build_industry_daily_returns
# =============================================================================


def _make_daily_input(monthly: pl.DataFrame, dates_per_eom: int = 3) -> pl.DataFrame:
    """Build a daily frame keyed on ``(id, eom_lag1, date, ret_exc)``.

    ``eom_lag1`` for each daily row equals the formation-month ``eom`` from
    ``monthly``; daily ``date`` falls in the month after that ``eom``.
    """
    ids: list[int] = []
    dates: list[date] = []
    eom_lag1_col: list[date] = []
    rets: list[float] = []
    for row in monthly.select(["id", "eom"]).unique().sort(["eom", "id"]).iter_rows(named=True):
        eom = row["eom"]
        # naive: pick first ``dates_per_eom`` weekdays after eom
        d = date(
            eom.year + (1 if eom.month == 12 else 0), 1 if eom.month == 12 else eom.month + 1, 1
        )
        added = 0
        cur = d
        while added < dates_per_eom:
            if cur.weekday() < 5:
                ids.append(row["id"])
                eom_lag1_col.append(eom)
                dates.append(cur)
                rets.append(0.001 + 0.0001 * added)
                added += 1
            from datetime import timedelta

            cur = cur + timedelta(days=1)
    return pl.DataFrame(
        {
            "id": pl.Series("id", ids, dtype=pl.Int64),
            "eom_lag1": pl.Series("eom_lag1", eom_lag1_col, dtype=pl.Date),
            "date": pl.Series("date", dates, dtype=pl.Date),
            "ret_exc": pl.Series("ret_exc", rets, dtype=pl.Float64),
        }
    )


class TestBuildIndustryDailyReturns:
    """Direct tests for ``_build_industry_daily_returns``."""

    def test_joins_monthly_weights_with_daily_returns(self):
        monthly = _make_monthly_input(n_per_group=3, eoms=[date(2020, 1, 31)])
        daily = _make_daily_input(monthly, dates_per_eom=2)
        out = _build_industry_daily_returns(
            data=monthly,
            daily=daily,
            industry_col="ff49",
            bp_min_n=1,
            excntry="USA",
        )
        # 2 industries x 2 dates = 4 (industry, date) cohorts
        assert out.height == 4
        # n equals the number of stocks per industry (3) per date.
        assert (out["n"] == 3).all()

    def test_groups_by_industry_and_date(self):
        monthly = _make_monthly_input(n_per_group=2, eoms=[date(2020, 1, 31)])
        daily = _make_daily_input(monthly, dates_per_eom=3)
        out = _build_industry_daily_returns(
            data=monthly,
            daily=daily,
            industry_col="ff49",
            bp_min_n=1,
            excntry="USA",
        )
        # 2 industries x 3 dates = 6 unique (industry, date) rows.
        keys = out.select(["ff49", "date"]).unique()
        assert keys.height == 6
        assert out.height == 6

    def test_bp_min_n_filters_small_industry_cohorts(self):
        monthly = _make_monthly_input(n_per_group=3, eoms=[date(2020, 1, 31)])
        daily = _make_daily_input(monthly, dates_per_eom=2)
        out = _build_industry_daily_returns(
            data=monthly,
            daily=daily,
            industry_col="ff49",
            bp_min_n=10,
            excntry="USA",
        )
        assert out.height == 0

    def test_returns_schema(self):
        monthly = _make_monthly_input(n_per_group=2, eoms=[date(2020, 1, 31)])
        daily = _make_daily_input(monthly, dates_per_eom=2)
        out = _build_industry_daily_returns(
            data=monthly,
            daily=daily,
            industry_col="ff49",
            bp_min_n=1,
            excntry="USA",
        )
        assert set(out.columns) == {
            "ff49",
            "date",
            "n",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
            "excntry",
        }


# =============================================================================
# TestPortfoliosIndustry (migrated)
# =============================================================================


class TestPortfoliosIndustry:
    """Tests for GICS and FF49 industry portfolio outputs."""

    @pytest.fixture()
    def usa_result(self, tmp_path: Path, seed: int):
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="USA", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)
        return portfolios(
            data_path=str(data_root),
            excntry="USA",
            chars=SYNTHETIC_CHARS,
            pfs=3,
            bps="non_mc",
            bp_min_n=10,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=True,
            cmp_key=False,
            signals=False,
            daily_pf=True,
            ind_pf=True,
            ret_cutoffs=ret_cut,
            ret_cutoffs_daily=ret_cut_daily,
        )

    def test_gics_first_2_digits_extracted_as_int(self, usa_result):
        gics = usa_result["gics_returns"]
        assert gics["gics"].dtype == pl.Float64 or gics["gics"].dtype == pl.Int64
        # All values should be 2-digit sector codes (10-60 range)
        vals = gics["gics"].unique().sort().to_list()
        for v in vals:
            assert 10 <= v <= 60

    def test_ff49_returns_present_for_usa(self, usa_result):
        assert "ff49_returns" in usa_result
        assert usa_result["ff49_returns"].height > 0

    def test_ff49_daily_present_for_usa(self, usa_result):
        assert "ff49_daily" in usa_result
        assert usa_result["ff49_daily"].height > 0

    def test_ff49_returns_only_for_usa(self, tmp_path: Path, seed: int):
        """Non-USA inputs should NOT produce ff49_returns."""
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="GBR", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)
        result = portfolios(
            data_path=str(data_root),
            excntry="GBR",
            chars=SYNTHETIC_CHARS,
            pfs=3,
            bps="non_mc",
            bp_min_n=10,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=True,
            cmp_key=False,
            signals=False,
            daily_pf=True,
            ind_pf=True,
            ret_cutoffs=ret_cut,
            ret_cutoffs_daily=ret_cut_daily,
        )
        assert "ff49_returns" not in result
        assert "ff49_daily" not in result

    def test_ind_returns_filter_by_bp_min_n(self, tmp_path: Path, seed: int):
        """Industries with n < bp_min_n should be filtered out."""
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="SYN", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)
        result = portfolios(
            data_path=str(data_root),
            excntry="SYN",
            chars=SYNTHETIC_CHARS,
            pfs=3,
            bps="non_mc",
            bp_min_n=10,
            nyse_size_cutoffs=nyse_cut,
            source=["CRSP", "COMPUSTAT"],
            wins_ret=True,
            cmp_key=False,
            signals=False,
            daily_pf=False,
            ind_pf=True,
            ret_cutoffs=ret_cut,
            ret_cutoffs_daily=ret_cut_daily,
        )
        gics = result["gics_returns"]
        # All groups should have n >= bp_min_n
        assert gics.filter(pl.col("n") < 10).height == 0


# =============================================================================
# TestExcntryGating (migrated)
# =============================================================================


class TestExcntryGating:
    """Tests that excntry comparison is case-insensitive."""

    def test_usa_gating_case_insensitive(self, tmp_path, seed):
        """portfolios() with excntry='USA' and excntry='usa' produce identical
        ff49_returns and ff49_daily."""
        # macOS filesystem is case-insensitive, so we use separate directories
        # for the two runs to avoid SameFileError.
        upper_root = tmp_path / "upper" / "processed"
        lower_root = tmp_path / "lower" / "processed"

        char_df, _ = _write_synthetic_country(
            data_root=upper_root, excntry="USA", chars=SYNTHETIC_CHARS, seed=seed
        )
        _write_synthetic_country(
            data_root=lower_root, excntry="usa", chars=SYNTHETIC_CHARS, seed=seed
        )

        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)

        shared = {
            "chars": SYNTHETIC_CHARS,
            "pfs": 3,
            "bps": "non_mc",
            "bp_min_n": 10,
            "nyse_size_cutoffs": nyse_cut,
            "source": ["CRSP", "COMPUSTAT"],
            "wins_ret": True,
            "cmp_key": False,
            "signals": False,
            "signals_standardize": True,
            "signals_w": "vw_cap",
            "daily_pf": True,
            "ind_pf": True,
            "ret_cutoffs": ret_cut,
            "ret_cutoffs_daily": ret_cut_daily,
        }

        upper = portfolios(data_path=str(upper_root), excntry="USA", **shared)
        lower = portfolios(data_path=str(lower_root), excntry="usa", **shared)

        for key in ("ff49_returns", "ff49_daily"):
            assert key in upper, f"{key!r} missing from excntry='USA' output"
            assert key in lower, f"{key!r} missing from excntry='usa' output"

        ind_key_cols = ["ff49", "excntry"]
        ind_numeric = {
            "n": _TIGHT,
            "ret_ew": _TIGHT,
            "ret_vw": _TIGHT,
            "ret_vw_cap": _TIGHT,
        }

        _assert_frames_parity(
            upper["ff49_returns"],
            lower["ff49_returns"],
            key_cols=["ff49", "eom", "excntry"],
            numeric_cols=ind_numeric,
            label="ff49_returns case parity",
        )
        _assert_frames_parity(
            upper["ff49_daily"],
            lower["ff49_daily"],
            key_cols=[*ind_key_cols, "date"],
            numeric_cols=ind_numeric,
            label="ff49_daily case parity",
        )
