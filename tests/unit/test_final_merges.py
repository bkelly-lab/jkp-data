"""Unit tests for the final merge steps of the world dataset.

Covers ``create_world_data_prelim``, ``merge_world_data_prelim`` and
``merge_industry_to_world_msf``. Each test writes small hand-built frames to
``test_paths.interim_dir``, runs the real function, and reads back the output
parquet. There is no float math, so frames are compared exactly.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import (
    acc_chars_list,
    create_world_data_prelim,
    merge_industry_to_world_msf,
    merge_world_data_prelim,
)
from jkp.data.config import ROLLING_DAILY_SPECS
from jkp.data.paths import DataPaths
from tests.conftest import assert_unique_keys

JAN = date(2020, 1, 31)
FEB = date(2020, 2, 29)
MAR = date(2020, 3, 31)


class TestCreateWorldDataPrelim:
    """Left joins of msf with monthly market chars and accounting chars."""

    @staticmethod
    def _run(
        paths: DataPaths, msf: pl.DataFrame, mkt: pl.DataFrame, acc: pl.DataFrame
    ) -> pl.DataFrame:
        msf.write_parquet(paths.interim_dir / "msf.parquet")
        mkt.write_parquet(paths.interim_dir / "mkt.parquet")
        acc.write_parquet(paths.interim_dir / "acc.parquet")
        out = paths.interim_dir / "prelim.parquet"
        create_world_data_prelim(
            paths,
            paths.interim_dir / "msf.parquet",
            paths.interim_dir / "mkt.parquet",
            paths.interim_dir / "acc.parquet",
            out,
        )
        return pl.read_parquet(out)

    @staticmethod
    def _msf() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id": [1, 1, 2],
                "eom": [JAN, FEB, JAN],
                "gvkey": ["A", "A", None],
                "ret": [10, 11, 12],
                "div_tot": [1, 1, 1],
                "div_cash": [2, 2, 2],
                "div_spc": [3, 3, 3],
                "source": [1, 1, 0],
            }
        )

    def test_keeps_all_msf_rows_and_nulls_unmatched(self, test_paths: DataPaths) -> None:
        """Every msf row survives; rows without a match get null right-side columns."""
        mkt = pl.DataFrame({"id": [1], "eom": [JAN], "mkt_x": [100]})
        acc = pl.DataFrame({"gvkey": ["A"], "public_date": [JAN], "assets": [7]})
        out = self._run(test_paths, self._msf(), mkt, acc)
        expected = pl.DataFrame(
            {
                "id": [1, 1, 2],
                "eom": [JAN, FEB, JAN],
                "gvkey": ["A", "A", None],
                "ret": [10, 11, 12],
                "mkt_x": [100, None, None],
                "assets": [7, None, None],
            }
        )
        assert_frame_equal(out, expected, check_exact=True)

    def test_accounting_matches_eom_to_public_date_not_datadate(
        self, test_paths: DataPaths
    ) -> None:
        """A row where only datadate equals eom stays null on accounting columns."""
        mkt = pl.DataFrame({"id": [1], "eom": [JAN], "mkt_x": [100]})
        acc = pl.DataFrame(
            {
                "gvkey": ["A", "A"],
                "datadate": [FEB, JAN],
                "public_date": [MAR, FEB],
                "assets": [1, 2],
            }
        )
        out = self._run(test_paths, self._msf(), mkt, acc).sort("id", "eom")
        # JAN has datadate match only -> null; FEB has public_date match -> 2.
        assert out["assets"].to_list() == [None, 2, None]
        assert "datadate" in out.columns

    def test_drops_dividend_and_source_columns(self, test_paths: DataPaths) -> None:
        """div_tot, div_cash, div_spc and source are removed from the output."""
        mkt = pl.DataFrame({"id": [1], "eom": [JAN], "mkt_x": [100]})
        acc = pl.DataFrame({"gvkey": ["A"], "public_date": [JAN], "assets": [7]})
        out = self._run(test_paths, self._msf(), mkt, acc)
        for c in ["div_tot", "div_cash", "div_spc", "source"]:
            assert c not in out.columns

    def test_no_public_date_column_and_unique_keys(self, test_paths: DataPaths) -> None:
        """public_date is not left behind and (id, eom) stays unique."""
        mkt = pl.DataFrame({"id": [1, 2], "eom": [JAN, JAN], "mkt_x": [100, 200]})
        acc = pl.DataFrame({"gvkey": ["A"], "public_date": [JAN], "assets": [7]})
        out = self._run(test_paths, self._msf(), mkt, acc)
        assert "public_date" not in out.columns
        assert out.height == 3
        assert_unique_keys(out, ["id", "eom"])


def _daily_chars_columns() -> set[str]:
    """Non-key column names of market_chars_d, derived from the producers.

    Derived from ``ROLLING_DAILY_SPECS`` and the output names in ``roll_apply_daily``
    helpers, ``bidask_hl`` and ``finish_daily_chars``.
    """
    per_stat = {
        "rvol": ["rvol"],
        "rmax": ["rmax5", "rmax1"],
        "skew": ["rskew"],
        "capm_ext": ["beta", "ivol_capm", "iskew_capm", "coskew"],
        "ff3": ["ivol_ff3", "iskew_ff3"],
        "hxz4": ["ivol_hxz4", "iskew_hxz4"],
        "dimsonbeta": ["beta_dimson"],
        "zero_trades": ["zero_trades"],
        "turnover": ["turnover"],
        "dolvol": ["dolvol"],
        "ami": ["ami"],
        "capm": ["beta", "ivol_capm"],
        "downbeta": ["betadown"],
        "prc_to_high": ["prc_highprc"],
        "mktcorr": ["corr"],
    }
    cols = {"bidaskhl_21d", "rvolhl_21d", "betabab_1260d", "rmax5_rvol_21d"}
    for sfx, _, stats in ROLLING_DAILY_SPECS:
        for stat in stats:
            if stat == "mktvol":  # dropped by finish_daily_chars
                continue
            cols |= {f"{name}{sfx}" for name in per_stat[stat]}
    return cols


def _prelim_columns() -> set[str]:
    """Representative world_data_prelim non-key columns (msf + market chars + accounting)."""
    msf = {
        "excntry",
        "gvkey",
        "permno",
        "permco",
        "iid",
        "ret",
        "ret_exc",
        "ret_lag_dif",
        "ret_local",
        "me",
        "market_equity",
        "prc",
        "shares",
        "dolvol",
        "tvol",
        "bidask",
        "common",
        "comp_exchg",
        "crsp_exchcd",
        "exch_main",
        "primary_sec",
        "obs_main",
        "curcd",
        "fx",
        "gics",
        "sic",
        "naics",
        "ff49",
        "size_grp",
    }
    market_m = {
        "div12m_me",
        "divspc12m_me",
        "chcsho_12m",
        "eqnpo_12m",
        "ret_1_0",
        "ret_12_1",
        "ret_60_12",
        "seas_1_1an",
        "seas_2_5na",
    }
    return msf | market_m | set(acc_chars_list())


def _right_side_columns() -> dict[str, set[str]]:
    """Non-key columns of the six merge_world_data_prelim right-side inputs."""
    return {
        "beta_60m": {"beta_60m", "ivol_capm_60m"},
        "resmom_ff3_12_1": {"resff3_12_1"},
        "resmom_ff3_6_1": {"resff3_6_1"},
        "mp_factors": {"mispricing_perf", "mispricing_mgmt"},
        "market_chars_d": _daily_chars_columns(),
        "firm_age": {"age"},
    }


class TestMergeWorldDataPrelim:
    """Six left joins onto world_data_prelim on (id, eom)."""

    @staticmethod
    def _write(paths: DataPaths, prelim: pl.DataFrame, **extra: pl.DataFrame) -> pl.DataFrame:
        keys = prelim.select("id", "eom")
        inputs = {
            "world_data_prelim": prelim,
            "beta_60m": keys.with_columns(beta_60m=pl.lit(1), ivol_capm_60m=pl.lit(2)),
            "resmom_ff3_12_1": keys.with_columns(resff3_12_1=pl.lit(3)),
            "resmom_ff3_6_1": keys.with_columns(resff3_6_1=pl.lit(4)),
            "mp_factors": keys.with_columns(mispricing_perf=pl.lit(5), mispricing_mgmt=pl.lit(6)),
            "market_chars_d": keys.with_columns(bidaskhl_21d=pl.lit(7)),
            "firm_age": keys.with_columns(age=pl.lit(8), age_extra=pl.lit(9)),
        }
        inputs.update(extra)
        for name, df in inputs.items():
            df.write_parquet(paths.interim_dir / f"{name}.parquet")
        merge_world_data_prelim(paths)
        return pl.read_parquet(paths.interim_dir / "world_data_-1.parquet")

    @staticmethod
    def _prelim() -> pl.DataFrame:
        return pl.DataFrame({"id": [1, 1, 2], "eom": [JAN, FEB, JAN], "x": [10, 11, 12]})

    def test_all_inputs_arrive_and_rows_preserved(self, test_paths: DataPaths) -> None:
        """Row count equals world_data_prelim and each input's columns are present."""
        prelim = self._prelim()
        out = self._write(test_paths, prelim)
        assert out.height == prelim.height
        assert_unique_keys(out, ["id", "eom"])
        expected = prelim.with_columns(
            beta_60m=pl.lit(1),
            ivol_capm_60m=pl.lit(2),
            resff3_12_1=pl.lit(3),
            resff3_6_1=pl.lit(4),
            mispricing_perf=pl.lit(5),
            mispricing_mgmt=pl.lit(6),
            bidaskhl_21d=pl.lit(7),
            age=pl.lit(8),
        )
        assert_frame_equal(out, expected, check_exact=True)

    def test_only_age_comes_from_firm_age(self, test_paths: DataPaths) -> None:
        """Other firm_age columns are not added to the output."""
        out = self._write(test_paths, self._prelim())
        assert "age" in out.columns
        assert "age_extra" not in out.columns

    def test_unmatched_rows_get_nulls(self, test_paths: DataPaths) -> None:
        """Prelim rows absent from a right-side input get nulls for its columns."""
        prelim = self._prelim()
        partial = prelim.head(1).with_columns(beta_60m=pl.lit(1), ivol_capm_60m=pl.lit(2))
        out = self._write(test_paths, prelim, beta_60m=partial.drop("x")).sort("id", "eom")
        assert out["beta_60m"].to_list() == [1, None, None]
        assert out["ivol_capm_60m"].to_list() == [2, None, None]
        assert out["resff3_12_1"].to_list() == [3, 3, 3]

    def test_no_right_suffix_with_real_column_sets(self, test_paths: DataPaths) -> None:
        """With the real producer column sets, no output column collides into ``_right``."""
        right = _right_side_columns()
        prelim_cols = _prelim_columns()
        keys = pl.DataFrame({"id": [1], "eom": [JAN]})
        prelim = keys.with_columns(pl.lit(0).alias(c) for c in sorted(prelim_cols))
        extra = {
            name: keys.with_columns(pl.lit(0).alias(c) for c in sorted(cols))
            for name, cols in right.items()
        }
        # Real firm_age.parquet carries only (id, eom, age) in the golden fixture.
        out = self._write(test_paths, prelim, **extra)
        assert not [c for c in out.columns if c.endswith("_right")]
        expected_cols = {"id", "eom"} | prelim_cols | set().union(*right.values())
        assert set(out.columns) == expected_cols

    def test_right_side_columns_are_disjoint(self) -> None:
        """The six right-side inputs share no non-key columns with each other or with prelim."""
        right = _right_side_columns()
        names = list(right)
        for i, a in enumerate(names):
            assert not right[a] & _prelim_columns(), a
            for b in names[i + 1 :]:
                assert not right[a] & right[b], (a, b)


class TestMergeIndustryToWorldMsf:
    """Compustat-first, CRSP-fallback industry codes."""

    @staticmethod
    def _run(
        paths: DataPaths, msf: pl.DataFrame, comp: pl.DataFrame, crsp: pl.DataFrame
    ) -> pl.DataFrame:
        msf.write_parquet(paths.interim_dir / "__msf_world.parquet")
        comp.write_parquet(paths.interim_dir / "comp_ind.parquet")
        crsp.write_parquet(paths.interim_dir / "crsp_ind.parquet")
        merge_industry_to_world_msf(paths)
        return pl.read_parquet(paths.interim_dir / "__msf_world2.parquet")

    @staticmethod
    def _comp(gvkey: list, date_: list, sic: list, naics: list) -> pl.DataFrame:
        return pl.DataFrame(
            {"gvkey": gvkey, "date": date_, "sic": sic, "naics": naics},
            schema={"gvkey": pl.String, "date": pl.Date, "sic": pl.Int64, "naics": pl.Int64},
        )

    @staticmethod
    def _crsp(permco: list, permno: list, date_: list, sic: list, naics: list) -> pl.DataFrame:
        return pl.DataFrame(
            {"permco": permco, "permno": permno, "date": date_, "sic": sic, "naics": naics},
            schema={
                "permco": pl.Int64,
                "permno": pl.Int64,
                "date": pl.Date,
                "sic": pl.Int64,
                "naics": pl.Int64,
            },
        )

    @staticmethod
    def _msf(rows: list[tuple]) -> pl.DataFrame:
        return pl.DataFrame(
            rows,
            schema={
                "id": pl.Int64,
                "gvkey": pl.String,
                "permco": pl.Int64,
                "permno": pl.Int64,
                "eom": pl.Date,
            },
            orient="row",
        )

    @pytest.mark.methodology
    def test_compustat_wins_and_crsp_fills_nulls(self, test_paths: DataPaths) -> None:
        """Compustat sic/naics win when present; CRSP fills in when Compustat is null.

        documentation/documentation.tex (industry section): industry codes are
        "based on Compustat data if available and otherwise CRSP".
        """
        msf = self._msf(
            [
                (1, "A", 10, 100, JAN),  # both present -> Compustat
                (2, "B", 20, 200, JAN),  # Compustat null -> CRSP
                (3, "C", 30, 300, JAN),  # Compustat sic only -> naics from CRSP
                (4, "D", 40, 400, JAN),  # neither -> null
            ]
        )
        comp = self._comp(
            ["A", "B", "C", "D"], [JAN] * 4, [1111, None, 3333, None], [11, None, None, None]
        )
        crsp = self._crsp(
            [10, 20, 30, 40],
            [100, 200, 300, 400],
            [JAN] * 4,
            [9999, 2222, 8888, None],
            [99, 22, 33, None],
        )
        out = self._run(test_paths, msf, comp, crsp).sort("id")
        assert out["sic"].to_list() == [1111, 2222, 3333, None]
        assert out["naics"].to_list() == [11, 22, 33, None]

    def test_drops_crsp_helper_columns_and_keeps_rows(self, test_paths: DataPaths) -> None:
        """sic_crsp/naics_crsp are dropped and the row count is preserved."""
        msf = self._msf([(1, "A", 10, 100, JAN), (2, "B", 20, 200, FEB), (3, "Z", 90, 900, JAN)])
        comp = self._comp(["A"], [JAN], [1111], [11])
        crsp = self._crsp([20], [200], [FEB], [2222], [22])
        out = self._run(test_paths, msf, comp, crsp)
        assert "sic_crsp" not in out.columns
        assert "naics_crsp" not in out.columns
        assert out.height == 3
        assert_unique_keys(out, ["id", "eom"])
        assert out.sort("id")["sic"].to_list() == [1111, 2222, None]

    def test_null_keys_do_not_match(self, test_paths: DataPaths) -> None:
        """Null gvkey or null permno on the msf side never matches null keys on the other side."""
        msf = self._msf(
            [
                (1, None, 10, 100, JAN),  # null gvkey vs comp row with null gvkey
                (2, "B", 20, None, JAN),  # null permno vs crsp row with null permno
            ]
        )
        comp = self._comp([None], [JAN], [1111], [11])
        crsp = self._crsp([20], [None], [JAN], [2222], [22])
        out = self._run(test_paths, msf, comp, crsp).sort("id")
        assert out["sic"].to_list() == [None, None]
        assert out["naics"].to_list() == [None, None]
        assert out.height == 2
