"""Unit tests for the ``cmp_key=True`` branch of ``portfolios()``."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from jkp.data.config import CMP_SIZE_GROUPS
from jkp.data.paths import DataPaths
from jkp.data.portfolio import portfolios
from tests.unit.portfolio.conftest import (
    make_cmp_dataset,
    make_country_characteristics,
    make_cutoffs,
)

COMMON_KWARGS = {
    "pfs": 3,
    "bp_min_n": 5,
    "signals": False,
    "signals_standardize": True,
    "signals_w": "vw_cap",
    "daily_pf": False,
    "ind_pf": False,
    "wins_ret": True,
}


def _write_chars(base_dir: Path, excntry: str, df: pl.DataFrame) -> None:
    char_dir = base_dir / "processed" / "characteristics"
    char_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(char_dir / f"{excntry}.parquet")


class TestCmpBranch:
    """Tests for the cmp (characteristics-managed-portfolio) branch."""

    def test_cmp_key_true_emits_cmp_in_output(self, tmp_path: Path) -> None:
        excntry = "USA"
        chars = ["char_a", "char_b"]
        df = make_country_characteristics(
            excntry=excntry, chars=chars, n_ids=60, n_months=4, seed=3
        )
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        out = portfolios(
            paths=DataPaths(base_dir=tmp_path),
            excntry=excntry,
            chars=chars,
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            cmp_key=True,
            **COMMON_KWARGS,
        )
        for n_groups in CMP_SIZE_GROUPS:
            assert isinstance(out[f"cmp_{n_groups}"], pl.DataFrame)
            assert out[f"cmp_{n_groups}"].height > 0

    def test_cmp_schema(self, tmp_path: Path) -> None:
        excntry = "USA"
        chars = ["char_a"]
        df = make_country_characteristics(
            excntry=excntry, chars=chars, n_ids=60, n_months=4, seed=4
        )
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        out = portfolios(
            paths=DataPaths(base_dir=tmp_path),
            excntry=excntry,
            chars=chars,
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            cmp_key=True,
            **COMMON_KWARGS,
        )
        # Schema produced by portfolio.py:715-734: group keys (size_grp, eom)
        # plus characteristic, n_stocks, ret_weighted, signal_weighted, excntry
        # (sd_var is dropped after filtering).
        expected = {
            "size_grp",
            "eom",
            "characteristic",
            "n_stocks",
            "ret_weighted",
            "signal_weighted",
            "excntry",
        }
        for n_groups in CMP_SIZE_GROUPS:
            assert set(out[f"cmp_{n_groups}"].columns) == expected

    def test_cmp_key_false_omits_cmp(self, tmp_path: Path) -> None:
        excntry = "USA"
        chars = ["char_a"]
        df = make_country_characteristics(
            excntry=excntry, chars=chars, n_ids=60, n_months=4, seed=5
        )
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        out = portfolios(
            paths=DataPaths(base_dir=tmp_path),
            excntry=excntry,
            chars=chars,
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            cmp_key=False,
            **COMMON_KWARGS,
        )
        assert not [key for key in out if key.startswith("cmp")]

    def test_cmp_drops_zero_sd_var_cohorts(self, tmp_path: Path) -> None:
        excntry = "USA"
        df = make_cmp_dataset(seed=42)
        # The cmp branch operates on char_a (will be passed in chars).
        # make_cmp_dataset forces (size_grp="large", eom=eoms[0]) to constant char_a.
        eoms = df["eom"].unique().sort().to_list()
        zero_sd_eom = eoms[0]
        _write_chars(tmp_path, excntry, df)
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        out = portfolios(
            paths=DataPaths(base_dir=tmp_path),
            excntry=excntry,
            chars=["char_a"],
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            cmp_key=True,
            **COMMON_KWARGS,
        )
        cmp = out["cmp_5"]
        # The cohort's eom is advanced one month after aggregation, so the
        # filtered-out cohort would appear at next-month-end if not dropped.
        advanced_eom = (
            pl.DataFrame({"eom": [zero_sd_eom]})
            .with_columns(pl.col("eom").dt.offset_by("1mo").dt.month_end())["eom"]
            .to_list()[0]
        )
        present = cmp.filter(
            (pl.col("size_grp") == "large")
            & (pl.col("eom") == advanced_eom)
            & (pl.col("characteristic") == "char_a")
        )
        assert present.height == 0, (
            f"sd_var=0 cohort (large, {advanced_eom}) should have been dropped"
        )

    def test_cmp_eom_advanced_one_month_to_month_end(self, tmp_path: Path) -> None:
        excntry = "USA"
        chars = ["char_a"]
        # Single month: 2020-01-31 → expect output eom 2020-02-29.
        df = make_country_characteristics(
            excntry=excntry,
            chars=chars,
            n_ids=60,
            n_months=1,
            seed=9,
            start_year=2020,
            start_month=1,
        )
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        assert eoms == [date(2020, 1, 31)]
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        out = portfolios(
            paths=DataPaths(base_dir=tmp_path),
            excntry=excntry,
            chars=chars,
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            cmp_key=True,
            **COMMON_KWARGS,
        )
        cmp = out["cmp_5"]
        assert cmp.height > 0
        out_eoms = sorted(set(cmp["eom"].to_list()))
        assert out_eoms == [date(2020, 2, 29)]

    def test_size_groupings_by_hand(self, tmp_path: Path) -> None:
        excntry = "USA"
        df = make_country_characteristics(
            excntry=excntry, chars=["char_a", "char_b"], n_ids=60, n_months=1, seed=11
        )
        # Only five stocks have char_b, one in each size group, so the ranks are by hand.
        size_grp = ["mega", "large", "small", "micro", "nano"]
        char_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        ret = [0.01, 0.02, 0.04, 0.08, 0.16]
        n_rest = df.height - 5
        df = df.with_columns(
            pl.Series("size_grp", size_grp + df["size_grp"].to_list()[5:]),
            pl.Series("char_b", char_b + [None] * n_rest),
            pl.Series("ret_exc_lead1m", ret + df["ret_exc_lead1m"].to_list()[5:]),
            pl.Series("me", [100.0] * 5 + df["me"].to_list()[5:]),
            pl.Series("source_crsp", [1] * 5 + df["source_crsp"].to_list()[5:]),
        )
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        out = portfolios(
            paths=DataPaths(base_dir=tmp_path),
            excntry=excntry,
            chars=["char_a", "char_b"],
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            cmp_key=True,
            **COMMON_KWARGS,
        )

        def rows(n_groups: int) -> dict[str, tuple[int, float, float]]:
            return {
                row["size_grp"]: (row["n_stocks"], row["ret_weighted"], row["signal_weighted"])
                for row in out[f"cmp_{n_groups}"]
                .filter(pl.col("characteristic") == "char_b")
                .iter_rows(named=True)
            }

        # One group: ranks 1 to 5, so the weights are (-2, -1, 0, 1, 2) / 3.
        assert rows(1) == {"all": (5, pytest.approx(0.12), pytest.approx(10 / 3))}
        # big = mega, large with weights (-1, 1); small = small, micro, nano with (-1, 0, 1).
        assert rows(2) == {
            "big": (2, pytest.approx(0.01), pytest.approx(1.0)),
            "small": (3, pytest.approx(0.12), pytest.approx(2.0)),
        }
        # Five groups: each group has one stock, so no spread, so no row.
        assert rows(5) == {}

        # Every stock is in the single group; size groups have the promised names.
        n_char_a = df.filter(
            pl.col("char_a").fill_nan(None).is_not_null()
            & pl.col("me").is_not_null()
            & pl.col("size_grp").is_not_null()
            & pl.col("ret_exc_lead1m").is_not_null()
        ).height
        char_a = out["cmp_1"].filter(pl.col("characteristic") == "char_a")
        assert char_a["n_stocks"].to_list() == [n_char_a]
        assert set(out["cmp_1"]["size_grp"]) == {"all"}
        assert set(out["cmp_2"]["size_grp"]) == {"big", "small"}
        assert set(out["cmp_5"]["size_grp"]) <= set(size_grp)
