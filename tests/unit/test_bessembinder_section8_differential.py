"""
Differential tests: Section 8 numba kernel path vs the polars filter chain.

The polars chain (filter_8a..filter_8h applied sequentially) is the reference
implementation; the spill_dir kernel path must produce an identical kept
frame and removal log. The crux is sequential-survivor semantics: each filter
derives shifts / obs_num / totals over the rows still alive after the
previous filters, so the suite includes interaction cases where an earlier
filter's removal changes a later filter's neighbor.
"""

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from polars.testing import assert_frame_equal

from jkp.data.bessembinder import apply_bessembinder_section8

GROUP_COLS = ["gvkey", "iid"]
SORT_COL = "datadate"
SORT_KEYS = GROUP_COLS + [SORT_COL]


def _panel(seed: int = 11, n_groups: int = 120) -> pl.LazyFrame:
    """
    Description:
        Synthetic post-merge panel (pre-filter schema) with every Section 8
        filter triggered, including country specials and sequential
        interactions.
    Steps:
        Per group (cycling 14 scenarios): clean walk; 8a low volume; 8b zero
        ajexdi; 8c initial breach; 8c mid-history breach; 8d calendar gap;
        8e early adjCSHO jump (plus a CHN variant where standard thresholds
        fire but China's do not); 8f early ME jump; 8g return/ME mismatch;
        8h initial-obs ratio error; 8d-gap row that is also an 8e neighbor
        (sequential interaction); BRA low-price-threshold security; zero
        cshoc before an 8e jump (inf ratio); zero ri mid-series (8g inf
        return).
    Output:
        LazyFrame with the __comp_dsf_pre_filter schema + excntry.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        n = int(rng.integers(20, 90))
        prc = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        cshoc = np.full(n, 5.0)
        ri = prc.copy()
        ajexdi = np.ones(n)
        dolvol = prc * float(rng.uniform(50, 5000))
        dates = np.arange(n, dtype=np.int64) * 2  # every other day
        excntry = "USA"
        me_override: tuple[int, float] | None = None
        kind = g % 14
        if kind == 1:
            dolvol = prc * 0.001  # 8a: tiny average volume
        elif kind == 2 and n > 6:
            ajexdi[5] = 0.0  # 8b
        elif kind == 3:
            prc[0] = 0.004  # 8c initial breach
        elif kind == 4 and n > 12:
            prc[8] = 0.004  # 8c mid-history breach
        elif kind == 5 and n > 14:
            dates[10:] += 400  # 8d gap
        elif kind == 6 and n > 10:
            cshoc[4:] *= 8.0  # 8e early jump (standard thresholds)
            if g % 24 >= 12:
                excntry = "CHN"  # same jump, China thresholds: no trigger
        elif kind == 7 and n > 10:
            ri[6:] *= 1.001  # ME jumps without return: bump me via cshoc
            cshoc[6:] *= 15.0
        elif kind == 8 and n > 8:
            ri[5:] *= 2.5  # 8g: 150% return with flat ME
        elif kind == 9 and n > 5:
            prc[1] = prc[0] * 20  # 8h: initial-obs price ratio error
            prc[2:] = prc[0]
        elif kind == 10 and n > 16:
            # Sequential interaction: a gap row removed by 8d sits right
            # before an adjCSHO jump, changing 8e's prev-alive neighbor
            dates[8:] += 400
            cshoc[9:] *= 8.0
        elif kind == 11:
            excntry = "BRA"  # low price threshold 0.001
            prc[:] = 0.005  # breaches USA threshold but not BRA's
            prc[0] = 0.0005  # ...except the first obs: initial breach
        elif kind == 12 and n > 10:
            # zero adjCSHO before an early jump: 8e ratio = x/0 = inf, must
            # match polars float semantics (me decoupled so 8c doesn't bite)
            cshoc[3] = 0.0
            cshoc[4:] = 40.0
            me_override = (3, prc[3] * 5.0)
        elif kind == 13 and n > 8:
            ri[5] = 0.0  # 8g: inf return at i=6 and -100% at i=5
        me = prc * cshoc
        if me_override is not None:
            me[me_override[0]] = me_override[1]
        frames.append(
            pl.DataFrame(
                {
                    "gvkey": [f"{g:06d}"] * n,
                    "iid": ["01"] * n,
                    "datadate": pl.Series(dates).cast(pl.Date),
                    "curcdd": ["USD"] * n,
                    "prc_local": prc,
                    "ajexdi": ajexdi,
                    "cshtrd": np.full(n, 1000.0),
                    "cshoc": cshoc,
                    "fx": np.ones(n),
                    "prc": prc,
                    "me": me,
                    "dolvol": dolvol,
                    "ri": ri,
                    "excntry": [excntry] * n,
                }
            )
        )
    return pl.concat(frames).lazy()


def _assert_equivalent(df: pl.LazyFrame, tmp_path, presorted: bool = False) -> None:
    """Polars chain vs kernel path: kept frame and removal log identical."""
    ref_df, ref_log = apply_bessembinder_section8(df, GROUP_COLS, SORT_COL, "excntry")
    presorted_path = None
    if presorted:
        presorted_path = tmp_path / "__bess_s8_sorted.parquet"
        df.sort(SORT_KEYS).collect().write_parquet(presorted_path)
    new_df, new_log = apply_bessembinder_section8(
        df, GROUP_COLS, SORT_COL, "excntry", spill_dir=tmp_path, presorted_path=presorted_path
    )
    assert_frame_equal(
        new_df.sort(SORT_KEYS).collect(),
        ref_df.sort(SORT_KEYS).collect(),
        check_column_order=False,
    )
    ref_rows = ref_log.collect() if ref_log is not None else pl.DataFrame()
    new_rows = new_log.collect() if new_log is not None else pl.DataFrame()
    assert new_rows.height == ref_rows.height
    if ref_rows.height:
        assert_frame_equal(
            new_rows.sort(SORT_KEYS + ["filter_reason"]),
            ref_rows.sort(SORT_KEYS + ["filter_reason"]),
            check_column_order=False,
        )


class TestSection8Differential:
    def test_full_panel(self, tmp_path):
        _assert_equivalent(_panel(), tmp_path)

    def test_alternate_seed(self, tmp_path):
        _assert_equivalent(_panel(seed=99, n_groups=60), tmp_path)

    def test_clean_panel_no_removals(self, tmp_path):
        # All-clean panel except 8a's bottom-percentile, which always bites
        df = _panel(seed=5, n_groups=28)
        clean = df.filter(pl.col("gvkey").cast(pl.Int32) % 14 == 0)
        _assert_equivalent(clean, tmp_path)


class TestSection8Presorted:
    def test_duckdb_sort_matches_polars_sort(self, tmp_path):
        # gen_comp_dsf presorts the spill file via DuckDB ORDER BY; the slim
        # path's group index assumes polars sort order. Prove they agree on
        # the key types involved (VARCHAR gvkey/iid, DATE datadate).
        import duckdb

        df = _panel(seed=7, n_groups=40).collect()
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=0)
        src = tmp_path / "src.parquet"
        out = tmp_path / "duck_sorted.parquet"
        shuffled.write_parquet(src)
        duckdb.connect().execute(f"""
            COPY (
                SELECT * FROM read_parquet('{src.as_posix()}')
                ORDER BY gvkey NULLS FIRST, iid NULLS FIRST, datadate NULLS FIRST
            ) TO '{out.as_posix()}' (FORMAT PARQUET);
        """)
        assert_frame_equal(pl.read_parquet(out), df.sort(SORT_KEYS))

    def test_presorted_path_equivalent(self, tmp_path):
        # A presorted file must produce the same output as the sorting path
        _assert_equivalent(_panel(seed=13, n_groups=48), tmp_path, presorted=True)

    def test_presorted_assertion_fires(self, tmp_path):
        # An UNSORTED file passed as presorted must raise, not silently
        # corrupt the panel
        shuffled = _panel(seed=13, n_groups=24).collect().sample(fraction=1.0, shuffle=True, seed=1)
        bad = tmp_path / "__bess_s8_sorted.parquet"
        shuffled.write_parquet(bad)
        with pytest.raises(ValueError, match="not contiguous|not sorted"):
            apply_bessembinder_section8(
                shuffled.lazy(),
                GROUP_COLS,
                SORT_COL,
                "excntry",
                spill_dir=tmp_path,
                presorted_path=bad,
            )


@st.composite
def _security_strategy(draw):
    """One short random security exercising row-level filters and nulls."""
    n = draw(st.integers(min_value=3, max_value=30))
    prc = [draw(st.floats(min_value=0.005, max_value=50.0, allow_nan=False)) for _ in range(n)]
    cshoc = [draw(st.sampled_from([0.0, 1.0, 5.0, 40.0, None])) for _ in range(n)]
    ri = [draw(st.floats(min_value=1.0, max_value=30.0, allow_nan=False)) for _ in range(n)]
    gap = draw(st.booleans())
    return n, prc, cshoc, ri, gap


class TestSection8Fuzz:
    @given(spec=_security_strategy())
    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_equivalence(self, spec, tmp_path_factory):
        n, prc, cshoc, ri, gap = spec
        dates = np.arange(n, dtype=np.int64) * 3
        if gap and n > 6:
            dates[n // 2 :] += 500
        cshoc_s = pl.Series(cshoc, dtype=pl.Float64)
        prc_s = pl.Series(prc, dtype=pl.Float64)
        df = pl.LazyFrame(
            {
                "gvkey": ["000001"] * n,
                "iid": ["01"] * n,
                "datadate": pl.Series(dates).cast(pl.Date),
                "curcdd": ["USD"] * n,
                "prc_local": prc_s,
                "ajexdi": np.ones(n),
                "cshtrd": np.full(n, 1000.0),
                "cshoc": cshoc_s,
                "fx": np.ones(n),
                "prc": prc_s,
                "me": prc_s * cshoc_s,
                "dolvol": prc_s * 1000,
                "ri": pl.Series(ri, dtype=pl.Float64),
                "excntry": ["USA"] * n,
            }
        )
        _assert_equivalent(df, tmp_path_factory.mktemp("s8fuzz"))
