"""Property-based invariants for the final merge and save functions."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.golden.final_stage_inputs import RET_COLS

SETTINGS = settings(max_examples=20, deadline=None)

EOMS = [date(2020, m, 28) for m in range(1, 7)]
GVKEYS = [f"G{i}" for i in range(6)]


def _paths(td: str) -> DataPaths:
    paths = DataPaths(base_dir=Path(td))
    af.setup_folder_structure(paths)
    return paths


@st.composite
def msf_and_right_sides(draw):
    """A unique-key msf plus right sides keyed (id, eom) / (gvkey, public_date) with unique keys."""
    keys = draw(
        st.lists(
            st.tuples(st.integers(1, 8), st.sampled_from(EOMS)),
            unique=True,
            min_size=1,
            max_size=20,
        )
    )
    gv = {i: draw(st.sampled_from(GVKEYS)) for i in {k[0] for k in keys}}
    msf = [
        {
            "id": i,
            "gvkey": gv[i],
            "eom": e,
            "me": float(n),
            "source": "x",
            "div_tot": 0.0,
            "div_cash": 0.0,
            "div_spc": 0.0,
        }
        for n, (i, e) in enumerate(keys)
    ]
    ids = st.tuples(st.integers(1, 12), st.sampled_from(EOMS))
    mc = draw(st.lists(ids, unique=True, max_size=20))
    ac = draw(
        st.lists(
            st.tuples(st.sampled_from(GVKEYS + ["Gx"]), st.sampled_from(EOMS)),
            unique=True,
            max_size=20,
        )
    )
    return msf, mc, ac


def _frame(rows: list[dict], schema: dict) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema)


class TestMergeSaveInvariants:
    @given(data=msf_and_right_sides())
    @SETTINGS
    def test_prelim_left_joins_keep_rows_and_keys(self, data) -> None:
        msf, mc, ac = data
        with tempfile.TemporaryDirectory() as td:
            paths = _paths(td)
            i = paths.interim_dir
            pl.DataFrame(msf).with_columns(pl.col("eom").cast(pl.Date)).write_parquet(
                i / "msf.parquet"
            )
            _frame(
                [{"id": a, "eom": e, "mc": 1.0} for a, e in mc],
                {"id": pl.Int64, "eom": pl.Date, "mc": pl.Float64},
            ).write_parquet(i / "mc.parquet")
            _frame(
                [{"gvkey": g, "public_date": e, "ac": 2.0} for g, e in ac],
                {"gvkey": pl.Utf8, "public_date": pl.Date, "ac": pl.Float64},
            ).write_parquet(i / "ac.parquet")
            af.create_world_data_prelim(
                paths, i / "msf.parquet", i / "mc.parquet", i / "ac.parquet", i / "out.parquet"
            )
            out = pl.read_parquet(i / "out.parquet")
        want = {(r["id"], r["eom"]) for r in msf}
        assert out.height == len(msf)
        assert set(out.select("id", "eom").rows()) == want

    @given(data=msf_and_right_sides())
    @SETTINGS
    def test_merge_world_data_prelim_keeps_rows_and_keys(self, data) -> None:
        msf, mc, _ = data
        with tempfile.TemporaryDirectory() as td:
            paths = _paths(td)
            i = paths.interim_dir
            base = pl.DataFrame(msf).with_columns(pl.col("eom").cast(pl.Date))
            base.write_parquet(i / "world_data_prelim.parquet")
            schema = {"id": pl.Int64, "eom": pl.Date}
            for n, (fname, col) in enumerate(
                [
                    ("beta_60m", "beta_60m"),
                    ("resmom_ff3_12_1", "r12"),
                    ("resmom_ff3_6_1", "r6"),
                    ("mp_factors", "mp"),
                    ("market_chars_d", "mcd"),
                    ("firm_age", "age"),
                ]
            ):
                sub = mc[n::2] if n % 2 else mc
                _frame(
                    [{"id": a, "eom": e, col: float(n)} for a, e in sub],
                    {**schema, col: pl.Float64},
                ).write_parquet(i / f"{fname}.parquet")
            af.merge_world_data_prelim(paths)
            out = pl.read_parquet(i / "world_data_-1.parquet")
        assert out.height == len(msf)
        assert set(out.select("id", "eom").rows()) == {(r["id"], r["eom"]) for r in msf}

    @given(
        rows=st.lists(
            st.tuples(
                st.booleans(),
                st.one_of(st.none(), st.integers(1, 9999)),
                st.booleans(),
                st.one_of(st.none(), st.integers(1, 9999)),
                st.booleans(),
                st.one_of(st.none(), st.integers(1, 9999)),
                st.booleans(),
                st.one_of(st.none(), st.integers(1, 9999)),
            ),
            min_size=1,
            max_size=15,
        )
    )
    @SETTINGS
    def test_industry_sic_naics_comp_else_crsp(self, rows) -> None:
        msf, comp, crsp = [], [], []
        for n, (hc, cs, hr, rs, hc2, cn, hr2, rn) in enumerate(rows):
            e = EOMS[n % len(EOMS)]
            msf.append({"id": n, "gvkey": f"G{n}", "permco": n, "permno": 10 * n, "eom": e})
            if hc:
                comp.append({"gvkey": f"G{n}", "date": e, "sic": cs, "naics": cn if hc2 else None})
            if hr:
                crsp.append(
                    {
                        "permco": n,
                        "permno": 10 * n,
                        "date": e,
                        "sic": rs,
                        "naics": rn if hr2 else None,
                    }
                )
        with tempfile.TemporaryDirectory() as td:
            paths = _paths(td)
            i = paths.interim_dir
            pl.DataFrame(
                msf,
                schema={
                    "id": pl.Int64,
                    "gvkey": pl.Utf8,
                    "permco": pl.Int64,
                    "permno": pl.Int64,
                    "eom": pl.Date,
                },
            ).write_parquet(i / "__msf_world.parquet")
            pl.DataFrame(
                comp, schema={"gvkey": pl.Utf8, "date": pl.Date, "sic": pl.Int64, "naics": pl.Int64}
            ).write_parquet(i / "comp_ind.parquet")
            pl.DataFrame(
                crsp,
                schema={
                    "permco": pl.Int64,
                    "permno": pl.Int64,
                    "date": pl.Date,
                    "sic": pl.Int64,
                    "naics": pl.Int64,
                },
            ).write_parquet(i / "crsp_ind.parquet")
            af.merge_industry_to_world_msf(paths)
            out = pl.read_parquet(i / "__msf_world2.parquet").sort("id")
        assert out.height == len(rows)
        for n, row in enumerate(out.iter_rows(named=True)):
            hc, cs, hr, rs, hc2, cn, hr2, rn = rows[n]
            c_sic, c_nai = (cs if hc else None), (cn if hc and hc2 else None)
            r_sic, r_nai = (rs if hr else None), (rn if hr and hr2 else None)
            assert row["sic"] == (c_sic if c_sic is not None else r_sic)
            assert row["naics"] == (c_nai if c_nai is not None else r_nai)

    @given(
        rows=st.lists(
            st.tuples(
                st.sampled_from(["USA", "GBR", "DEU", None]), st.integers(1, 5), st.integers(0, 9)
            ),
            min_size=1,
            max_size=40,
            unique=True,
        )
    )
    @SETTINGS
    def test_save_daily_ret_conserves_rows_one_country_per_file(self, rows) -> None:
        daily = pl.DataFrame(
            [
                {
                    "excntry": c,
                    "id": a,
                    "date": date(2020, 3, 1 + d),
                    "me": 1.0,
                    **dict.fromkeys(
                        (
                            "ret",
                            "ret_exc",
                            "ret_exc_wins",
                            "ret_intraday",
                            "ret_overnight",
                            "ret_intraday_local",
                            "ret_overnight_local",
                        ),
                        0.01,
                    ),
                }
                for c, a, d in rows
            ],
            schema_overrides={"excntry": pl.Utf8},
        )
        with tempfile.TemporaryDirectory() as td:
            paths = _paths(td)
            daily.write_parquet(paths.interim_dir / "world_dsf_output.parquet")
            af.save_daily_ret(paths)
            d = paths.processed_dir / "return_data" / "daily_rets_by_country"
            files = sorted(d.glob("*.parquet"))
            assert sorted(f.stem for f in files) == sorted(
                {c or "null_country" for c, _, _ in rows}
            )
            assert not [x for x in d.iterdir() if x.is_dir()]
            total = 0
            for f in files:
                got = pl.read_parquet(f)
                expect = [
                    (a, date(2020, 3, 1 + k)) for c, a, k in rows if (c or "null_country") == f.stem
                ]
                assert sorted(got.select("id", "date").rows()) == sorted(expect)
                total += got.height
            assert total == len(rows)

    @given(
        rows=st.lists(
            st.tuples(
                st.integers(1, 10**9),
                st.integers(0, 1),
                st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False, width=64)),
            ),
            min_size=1,
            max_size=15,
        )
    )
    @SETTINGS
    def test_save_monthly_ret_preserves_values_and_dtypes(self, rows) -> None:
        frame = pl.DataFrame(
            {
                "excntry": ["USA"] * len(rows),
                "id": [r[0] for r in rows],
                "source_crsp": [r[1] for r in rows],
                "eom": [date(2020, 1, 31)] * len(rows),
                "me": [r[2] for r in rows],
                **{c: [r[2] for r in rows] for c in RET_COLS},
            },
            schema_overrides={"me": pl.Float64, **dict.fromkeys(RET_COLS, pl.Float64)},
        )
        with tempfile.TemporaryDirectory() as td:
            paths = _paths(td)
            frame.write_parquet(paths.interim_dir / "world_msf_output.parquet")
            af.save_monthly_ret(paths)
            out = pl.read_parquet(paths.processed_dir / "return_data" / "world_ret_monthly.parquet")
        assert_frame_equal(out, frame, check_exact=True)
