"""
Tests for targeted Ibis table builders in aux_functions.py.

This module focuses on schema-level output guarantees for functions that read
parquet inputs from the expected project layout.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.aux_functions import (
    apply_group_filter,
    aug_msf_v2,
    gen_aux_windows,
    gen_crsp_sf,
    merge_roll_apply_daily_results,
    prepare_daily,
    process_window,
    rvol,
    turnover,
    zero_trades,
)
from jkp.data.config import END_DATE


def _write_lookup_tables(raw_tables: Path) -> None:
    """Write the minimal lookup parquet files required by gen_crsp_sf()."""
    pl.DataFrame(
        {
            "permno": [10001],
            "secinfostartdt": [date(2020, 1, 1)],
            "secinfoenddt": [date(2020, 1, 31)],
            "ticker": ["TEST"],
        }
    ).write_parquet(raw_tables / "crsp_stksecurityinfohist.parquet")

    pl.DataFrame(
        {
            "lpermno": [10001],
            "linkdt": [date(2019, 1, 1)],
            "linkenddt": [date(2021, 12, 31)],
            "linktype": ["LC"],
            "liid": ["01"],
            "gvkey": ["001234"],
        }
    ).write_parquet(raw_tables / "crsp_ccmxpf_lnkhist.parquet")


def _write_sf_fixture(raw_tables: Path, freq: str) -> tuple[date, date]:
    """Write a tiny monthly or daily CRSP SF fixture and return matched/null dates."""
    common_columns = {
        "permno": [10001, 10001],
        "permco": [20001, 20001],
        "shrout": [1000.0, 1000.0],
        "securitytype": ["EQTY", "EQTY"],
        "securitysubtype": ["COM", "COM"],
        "sharetype": ["NS", "NS"],
        "issuertype": ["CORP", "CORP"],
        "primaryexch": ["N", "N"],
        "conditionaltype": ["RW", "RW"],
    }

    if freq == "m":
        matched_date = date(2020, 1, 31)
        unmatched_date = date(2020, 2, 29)
        msf_df = pl.DataFrame(
            {
                **common_columns,
                "mthcaldt": [matched_date, unmatched_date],
                "mthprc": [10.0, 11.0],
                "mthprcflg": ["TR", "TR"],
                "mthret": [0.10, 0.02],
                "mthretx": [0.09, 0.01],
                "mthvol": [1000, 1100],
                "mthcumfacshr": [1.0, 1.0],
                "mthaskhi": [10.5, 11.5],
                "mthbidlo": [9.5, 10.5],
            }
        )
        # raw_tables is paths.raw_tables_dir, so interim is its grandparent + interim.
        raw_data_dfs = raw_tables.parent.parent / "interim" / "raw_data_dfs"
        raw_data_dfs.mkdir(parents=True, exist_ok=True)
        msf_df.write_parquet(raw_data_dfs / "crsp_msf_v2_aug.parquet")
        return matched_date, unmatched_date

    matched_date = date(2020, 1, 2)
    unmatched_date = date(2020, 2, 3)
    pl.DataFrame(
        {
            **common_columns,
            "dlycaldt": [matched_date, unmatched_date],
            "dlyprc": [20.0, 21.0],
            "dlyprcflg": ["TR", "TR"],
            "dlyret": [0.01, 0.02],
            "dlyretx": [0.009, 0.018],
            "dlyvol": [200, 300],
            "dlycumfacshr": [1.0, 1.0],
            "dlyhigh": [20.5, 21.5],
            "dlylow": [19.5, 20.5],
        }
    ).write_parquet(raw_tables / "crsp_dsf_v2.parquet")
    return matched_date, unmatched_date


@pytest.mark.parametrize("freq", ["m", "d"])
def test_gen_crsp_sf_exposes_ticker_after_senames_join(freq: str, test_paths) -> None:
    """gen_crsp_sf() should keep ticker in the final output for monthly and daily data."""
    raw_tables = test_paths.raw_tables_dir

    _write_lookup_tables(raw_tables)
    matched_date, unmatched_date = _write_sf_fixture(raw_tables, freq)

    result = gen_crsp_sf(test_paths, freq)
    assert "ticker" in result.columns, f"Expected ticker in schema, got {result.columns}"

    df = result.to_polars().sort("date")

    assert {"permno", "permco", "date", "me", "ticker"}.issubset(df.columns), (
        f"Missing expected columns from output: {df.columns}"
    )

    ticker_by_date = {
        row["date"]: row["ticker"] for row in df.select(["date", "ticker"]).to_dicts()
    }
    assert ticker_by_date[matched_date] == "TEST", (
        f"Expected ticker TEST on {matched_date}, got {ticker_by_date[matched_date]!r}"
    )
    assert ticker_by_date[unmatched_date] is None, (
        f"Expected null ticker on {unmatched_date}, got {ticker_by_date[unmatched_date]!r}"
    )


def _write_aug_msf_v2_fixtures(raw_tables: Path) -> None:
    """Write minimal raw msf_v2 and dsf_v2 parquet fixtures for aug_msf_v2()."""
    pl.DataFrame(
        {
            "permno": [10001, 10001],
            "yyyymm": [202001, 202002],
            "mthcaldt": [date(2020, 1, 31), date(2020, 2, 29)],
            "mthprcflg": ["TR", "BA"],
        }
    ).write_parquet(raw_tables / "crsp_msf_v2.parquet")

    pl.DataFrame(
        {
            "permno": [10001, 10001, 10001, 10001],
            "dlycaldt": [
                date(2020, 1, 10),
                date(2020, 1, 20),
                date(2020, 2, 10),
                date(2020, 2, 20),
            ],
            "dlyprc": [9.5, 10.5, 11.0, 12.0],
            "dlyprcflg": ["TR", "TR", "TR", "TR"],
        }
    ).write_parquet(raw_tables / "crsp_dsf_v2.parquet")


def test_aug_msf_v2_writes_augmented_file_and_is_idempotent(test_paths) -> None:
    """aug_msf_v2() should produce the augmented parquet and be safe to re-run."""
    _write_aug_msf_v2_fixtures(test_paths.raw_tables_dir)

    aug_msf_v2(test_paths)

    output_path = test_paths.interim_dir / "raw_data_dfs" / "crsp_msf_v2_aug.parquet"
    assert output_path.exists(), f"Expected augmented file at {output_path}"

    schema = pl.scan_parquet(output_path).collect_schema().names()
    assert "mthaskhi" in schema, f"Expected mthaskhi column in {schema}"
    assert "mthbidlo" in schema, f"Expected mthbidlo column in {schema}"

    # Idempotency: a second invocation must not raise.
    aug_msf_v2(test_paths)


def test_merge_roll_apply_daily_results_writes_once_with_deterministic_order(test_paths) -> None:
    """merge_roll_apply_daily_results() must produce a single output with deterministic
    column ordering (sorted by source __roll* filename) and be re-run safe."""
    interim = test_paths.interim_dir

    pl.DataFrame({"id_int": [1, 2], "id": [10001, 10002]}).write_parquet(
        interim / "id_int_key.parquet"
    )

    # Use the function's hardcoded start index (23113) so this test stays
    # valid regardless of system date. The function generates aux_date in
    # [23113, today.year*12 + today.month + 1].
    aux_date_val = 23113
    # Write fixtures in non-alphabetical order to exercise sorted() determinism:
    # filesystem-order would be insertion-order on most FSes, so writing __roll_b_*
    # first ensures the test fails without the sorted() fix.
    pl.DataFrame(
        {
            "id_int": [1, 2],
            "aux_date": [aux_date_val, aux_date_val],
            "rmax": [0.5, 0.6],
        }
    ).write_parquet(interim / "__roll_b_rmax.parquet")
    pl.DataFrame(
        {
            "id_int": [1, 2],
            "aux_date": [aux_date_val, aux_date_val],
            "rvol": [0.1, 0.2],
        }
    ).write_parquet(interim / "__roll_a_rvol.parquet")

    merge_roll_apply_daily_results(test_paths)

    out = interim / "roll_apply_daily.parquet"
    assert out.exists(), f"Expected output at {out}"

    df = pl.read_parquet(out)
    assert {"id", "eom", "rvol", "rmax"}.issubset(df.columns), (
        f"Missing expected columns: {df.columns}"
    )
    # Deterministic order: sorted file_paths puts __roll_a_rvol before __roll_b_rmax,
    # so rvol must precede rmax in the merged schema.
    assert df.columns.index("rvol") < df.columns.index("rmax"), (
        f"Expected rvol before rmax (sorted file order), got {df.columns}"
    )
    # Outer join on shared (id_int, aux_date) keys = 2 rows.
    assert df.height == 2
    assert set(df["id"].to_list()) == {10001, 10002}

    # Re-run must produce identical content (idempotent + single-write safety).
    merge_roll_apply_daily_results(test_paths)
    df2 = pl.read_parquet(out)
    assert df.equals(df2)


def test_dsf1_unique_id_int_date(test_paths) -> None:
    """dsf1.parquet (post-prepare_daily) must be unique on (id_int, date).

    prc_to_high's within-group sort_by('date').last() is well-defined only under this
    invariant. Upstream guarantee is combine_crsp_comp_sf's ROW_NUMBER dedup over
    (id, date) (locked by test_no_duplicates_daily); this test locks the property
    after prepare_daily.
    """
    # Synthetic world_dsf with two ids across three dates each.
    rows = []
    for id_val in [101, 202]:
        for d in [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]:
            rows.append(
                {
                    "excntry": "USA",
                    "id": id_val,
                    "date": d,
                    "eom": date(2020, 1, 31),
                    "prc": 100.0,
                    "adjfct": 1.0,
                    "ret": 0.01,
                    "ret_exc": 0.005,
                    "dolvol": 1000.0,
                    "shares": 10.0,
                    "tvol": 100.0,
                    "ret_lag_dif": 1,
                    "ret_local": 0.01,
                }
            )
    world_dsf_path = test_paths.interim_dir / "world_dsf.parquet"
    pl.DataFrame(rows).write_parquet(world_dsf_path)

    # Synthetic ap_factors_daily — unique on (excntry, date).
    ap_factors_path = test_paths.interim_dir / "ap_factors_daily.parquet"
    pl.DataFrame(
        {
            "excntry": ["USA"] * 3,
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
            "mktrf": [0.001, 0.002, 0.003],
            "hml": [0.0, 0.0, 0.0],
            "smb_ff": [0.0, 0.0, 0.0],
            "inv": [0.0, 0.0, 0.0],
            "roe": [0.0, 0.0, 0.0],
            "smb_hxz": [0.0, 0.0, 0.0],
        }
    ).write_parquet(ap_factors_path)

    prepare_daily(test_paths, world_dsf_path, ap_factors_path)

    dsf1 = pl.read_parquet(test_paths.interim_dir / "dsf1.parquet")
    dup_count = dsf1.select(["id_int", "date"]).is_duplicated().sum()
    assert dup_count == 0, f"dsf1 has {dup_count} duplicate (id_int, date) rows"


class TestRollWindowEquivalence:
    """The arithmetic window machinery must reproduce the legacy map-based output.

    The legacy implementation (gen_consecutive_lists / build_groups /
    group_mapping_dfs / gen_aux_maps and the join-based process_map_chunks)
    was replaced by gen_aux_windows + process_window. Verbatim copies of the
    legacy functions live here as the reference implementation.
    """

    @staticmethod
    def _legacy_gen_consecutive_lists(input_list: list[int], k: int) -> list[list[int]]:
        return [
            input_list[i : i + k]
            for i in range(0, len(input_list), k)
            if len(input_list[i : i + k]) == k
        ]

    @classmethod
    def _legacy_build_groups(cls, input_list: list[int], k: int) -> list[list[list[int]]]:
        return [cls._legacy_gen_consecutive_lists(input_list[offset:], k) for offset in range(k)]

    @classmethod
    def _legacy_group_mapping_dfs(cls, input_list: list[int], k: int) -> list[dict]:
        groups = cls._legacy_build_groups(input_list, k)
        dfs = [
            pl.DataFrame({"aux_date": group}).with_columns(
                group_number=pl.cum_count("aux_date"),
                new_date=pl.col("aux_date").list.max(),
            )
            for group in groups
        ]
        return [
            {
                "group_map": df.explode("aux_date")
                .select([pl.col("aux_date").cast(pl.Int32), "group_number"])
                .lazy(),
                "date_map": df.select(["group_number", pl.col("new_date").alias("aux_date")])
                .unique()
                .sort(["group_number"])
                .lazy(),
            }
            for df in dfs
        ]

    @classmethod
    def _legacy_gen_aux_maps(cls, sfx: str | int) -> list[dict]:
        parameter_mapping = {"_21d": 1, "_126d": 6, "_252d": 12, "_1260d": 60}
        date_aux = END_DATE.month + END_DATE.year * 12
        k = parameter_mapping[sfx] if sfx in parameter_mapping else int(sfx)
        date_idx = list(range(23113 - k, date_aux + 1))
        return cls._legacy_group_mapping_dfs(date_idx, k)

    @staticmethod
    def _legacy_process_map_chunks(base_data, mapping, stats, sfx, __min):
        funcs = {"rvol": rvol, "zero_trades": zero_trades, "turnover": turnover}
        df = base_data.join(mapping["group_map"], how="inner", on="aux_date").pipe(
            apply_group_filter, stat=stats, min_obs=__min
        )
        df = df.pipe(funcs[stats], sfx=sfx, __min=__min)
        return df.join(mapping["date_map"], how="left", on="group_number").drop("group_number")

    @staticmethod
    def _synthetic_base_data() -> pl.LazyFrame:
        """Deterministic daily panel: months 23113-23152, ids with dense and
        sparse observations (id 3 trips min_obs for k=1 but not k=3)."""
        rows: dict[str, list] = {
            "id_int": [],
            "aux_date": [],
            "ret_exc": [],
            "tvol": [],
            "shares": [],
        }
        for id_int, n_days in [(1, 8), (2, 8), (3, 2)]:
            for month in range(23113, 23153):
                for day in range(n_days):
                    rows["id_int"].append(id_int)
                    rows["aux_date"].append(month)
                    rows["ret_exc"].append(((id_int * 7 + month * 3 + day * 5) % 11 - 5) / 100)
                    rows["tvol"].append(float((id_int + month + day) % 4))
                    rows["shares"].append(100.0 + id_int)
        return pl.DataFrame(rows).with_columns(pl.col("aux_date").cast(pl.Int32)).lazy()

    @pytest.mark.parametrize("sfx", ["_21d", "_126d", "_252d", "_1260d", 36, 60])
    def test_gen_aux_windows_matches_legacy_maps(self, sfx) -> None:
        """Arithmetic (aux_date -> window end) relation must equal the legacy maps."""
        legacy = self._legacy_gen_aux_maps(sfx)
        windows = gen_aux_windows(sfx)
        assert len(legacy) == len(windows)
        for mapping, (start, k, last_end) in zip(legacy, windows, strict=True):
            rel_legacy = (
                mapping["group_map"]
                .join(mapping["date_map"].rename({"aux_date": "end_date"}), on="group_number")
                .select(["aux_date", "end_date"])
                .sort("aux_date")
                .collect()
            )
            rel_new = pl.DataFrame({"aux_date": range(start, last_end + 1)}).with_columns(
                end_date=start + ((pl.col("aux_date") - start) // k + 1) * k - 1
            )
            assert rel_legacy["aux_date"].to_list() == rel_new["aux_date"].to_list()
            assert rel_legacy["end_date"].to_list() == rel_new["end_date"].to_list()

    @pytest.mark.parametrize("stat", ["rvol", "zero_trades", "turnover"])
    @pytest.mark.parametrize("sfx", ["_21d", 3])
    def test_process_window_matches_legacy(self, stat, sfx) -> None:
        """process_window output must equal the legacy join-based path."""
        base = self._synthetic_base_data()
        min_obs = 5
        legacy = pl.concat(
            [
                self._legacy_process_map_chunks(base, mapping, stat, sfx, min_obs)
                for mapping in self._legacy_gen_aux_maps(sfx)
            ]
        ).collect()
        new = pl.concat(
            [process_window(base, w, stat, sfx, min_obs) for w in gen_aux_windows(sfx)]
        ).collect()
        assert legacy.height > 0
        sort_cols = ["id_int", "aux_date"]
        assert_frame_equal(legacy.sort(sort_cols), new.select(legacy.columns).sort(sort_cols))

    def test_process_window_aux_date_dtype(self) -> None:
        """Output aux_date must stay Int64, matching the legacy __roll* schema."""
        base = self._synthetic_base_data()
        out = process_window(base, gen_aux_windows("_21d")[0], "rvol", "_21d", 5).collect()
        assert out.schema["aux_date"] == pl.Int64
