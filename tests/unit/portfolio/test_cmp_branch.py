"""Unit tests for the ``cmp_key=True`` branch of ``portfolios()``."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

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


def _write_chars(data_path: Path, excntry: str, df: pl.DataFrame) -> None:
    char_dir = data_path / "characteristics"
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
            data_path=str(tmp_path),
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
        assert "cmp" in out
        assert isinstance(out["cmp"], pl.DataFrame)
        assert out["cmp"].height > 0

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
            data_path=str(tmp_path),
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
        assert set(out["cmp"].columns) == expected

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
            data_path=str(tmp_path),
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
        assert "cmp" not in out or out.get("cmp") is None

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
            data_path=str(tmp_path),
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
        cmp = out["cmp"]
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
            data_path=str(tmp_path),
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
        cmp = out["cmp"]
        assert cmp.height > 0
        out_eoms = sorted(set(cmp["eom"].to_list()))
        assert out_eoms == [date(2020, 2, 29)]
