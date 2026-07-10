"""
Unit tests for the Bessembinder et al. (2023) Section 6 decimal corrections and
Section 8 filters (production slim/numba path).
"""

import numpy as np
import polars as pl
import pytest

from jkp.data.bessembinder import (
    apply_bessembinder_section6,
    apply_bessembinder_section8,
)

GROUP = ["gvkey", "iid"]
SORT = "datadate"


def _section6_frame(prccd: list[float], **overrides) -> pl.LazyFrame:
    """One security with a flat clean baseline for the non-price variables."""
    n = len(prccd)
    cols = {
        "gvkey": ["000001"] * n,
        "iid": ["01"] * n,
        "datadate": pl.Series(np.arange(n, dtype=np.int64)).cast(pl.Date),
        "ajexdi": [1.0] * n,
        "prccd": prccd,
        "cshoc": [8.0] * n,
        "trfd": [1.0] * n,
        "qunit": [1.0] * n,
    }
    cols.update(overrides)
    return pl.DataFrame(cols).lazy()


def _corrected_prccd(lf: pl.LazyFrame, tmp_path, method="bessembinder", **kwargs) -> list[float]:
    out = apply_bessembinder_section6(lf, spill_dir=tmp_path, correction_method=method, **kwargs)
    return out.sort(GROUP + [SORT]).collect()["prccd"].to_list()


class TestSection6:
    def test_single_period_high_10x_corrected(self, tmp_path):
        prc = _corrected_prccd(_section6_frame([20.0, 20.0, 200.0, 20.0, 20.0]), tmp_path)
        assert prc[2] == pytest.approx(20.0)

    def test_single_period_high_100x_corrected(self, tmp_path):
        prc = _corrected_prccd(_section6_frame([20.0, 20.0, 2000.0, 20.0, 20.0]), tmp_path)
        assert prc[2] == pytest.approx(20.0)

    def test_single_period_low_10x_corrected(self, tmp_path):
        prc = _corrected_prccd(_section6_frame([20.0, 20.0, 2.0, 20.0, 20.0]), tmp_path)
        assert prc[2] == pytest.approx(20.0)

    def test_multi_period_span_corrected(self, tmp_path):
        raw = [20.0, 20.0, 200.0, 200.0, 200.0, 20.0, 20.0]
        prc = _corrected_prccd(_section6_frame(raw), tmp_path)
        assert prc[2:5] == pytest.approx([20.0, 20.0, 20.0])

    def test_clean_series_unchanged(self, tmp_path):
        raw = [20.0, 20.5, 19.8, 20.1, 20.3]
        prc = _corrected_prccd(_section6_frame(raw), tmp_path)
        assert prc == pytest.approx(raw)

    def test_interpolation_uses_endpoint_geomean(self, tmp_path):
        # neighbors 16 and 25 -> geometric mean 20, not the fixed-multiplier 20.
        raw = [16.0, 160.0, 25.0]
        prc = _corrected_prccd(_section6_frame(raw), tmp_path, method="interpolation")
        assert prc[1] == pytest.approx(np.sqrt(16.0 * 25.0))

    def test_floor_skips_multiply_direction(self, tmp_path):
        # A low error (divide-direction correction is a multiply) is left alone
        # by the floor method, which only corrects divide-direction on >=$1.
        raw = [20.0, 20.0, 2.0, 20.0, 20.0]
        prc = _corrected_prccd(_section6_frame(raw), tmp_path, method="floor")
        assert prc[2] == pytest.approx(2.0)

    def test_trfd_corrected(self, tmp_path):
        lf = _section6_frame([20.0] * 5, trfd=[1.0, 1.0, 100.0, 1.0, 1.0])
        out = apply_bessembinder_section6(lf, spill_dir=tmp_path)
        trfd = out.sort(GROUP + [SORT]).collect()["trfd"].to_list()
        assert trfd[2] == pytest.approx(1.0)

    def test_requires_spill_dir(self):
        with pytest.raises(ValueError, match="requires spill_dir"):
            apply_bessembinder_section6(_section6_frame([20.0, 20.0, 20.0]))

    def test_rejects_unknown_method(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown correction_method"):
            apply_bessembinder_section6(
                _section6_frame([20.0]), spill_dir=tmp_path, correction_method="bogus"
            )

    def test_determinism_rerun(self, tmp_path):
        lf = _section6_frame([20.0, 20.0, 200.0, 200.0, 20.0, 2.0, 20.0])
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a = _corrected_prccd(lf, tmp_path / "a")
        b = _corrected_prccd(lf, tmp_path / "b")
        assert a == b


def _section8_group(g, n, **cols) -> pl.DataFrame:
    # Volume scales with g so filter 8a (bottom 2% by volume) only trims the
    # lowest-g securities; defects are injected into high-g securities to keep
    # 8a from confounding the filter under test.
    prc = 10.0 * np.exp(np.cumsum(np.full(n, 0.001)))
    base = {
        "gvkey": [f"{g:06d}"] * n,
        "iid": ["01"] * n,
        "datadate": pl.Series(np.arange(n, dtype=np.int64) * 2).cast(pl.Date),
        "ajexdi": np.ones(n),
        "cshoc": np.full(n, 5.0),
        "prc": prc,
        "ri": prc.copy(),
        "dolvol": prc * 1000.0 * (g + 1),
        "excntry": ["USA"] * n,
    }
    base.update(cols)
    base["me"] = base["prc"] * base["cshoc"]
    return pl.DataFrame(base)


def _kept_dates(lf, tmp_path, gvkey) -> set:
    out = apply_bessembinder_section8(lf, GROUP, SORT, "excntry", spill_dir=tmp_path)
    kept = out.filter(pl.col("gvkey") == gvkey).collect()
    return set(kept["datadate"].to_list())


def _clean_panel(n=40, n_groups=60):
    return pl.concat([_section8_group(g, n) for g in range(n_groups)]).lazy()


class TestSection8:
    def test_clean_high_volume_security_fully_kept(self, tmp_path):
        # A high-volume, well-behaved security survives every filter intact.
        assert len(_kept_dates(_clean_panel(), tmp_path, "000059")) == 40

    def test_8a_trims_lowest_volume_security(self, tmp_path):
        # Filter 8a always removes the bottom 2% by average positive volume.
        assert _kept_dates(_clean_panel(), tmp_path, "000000") == set()

    def test_8b_zero_ajexdi_drops_security(self, tmp_path):
        n = 40
        ajex = np.ones(n)
        ajex[5] = 0.0
        panel = pl.concat(
            [_section8_group(g, n) for g in range(59)] + [_section8_group(59, n, ajexdi=ajex)]
        ).lazy()
        assert _kept_dates(panel, tmp_path, "000059") == set()

    def test_8c_initial_low_price_drops_security(self, tmp_path):
        n = 40
        prc = 10.0 * np.exp(np.cumsum(np.full(n, 0.001)))
        prc[0] = 0.004
        panel = pl.concat(
            [_section8_group(g, n) for g in range(59)] + [_section8_group(59, n, prc=prc)]
        ).lazy()
        assert _kept_dates(panel, tmp_path, "000059") == set()

    def test_requires_spill_dir(self):
        with pytest.raises(ValueError, match="requires spill_dir"):
            apply_bessembinder_section8(_section8_group(0, 5).lazy())

    def test_determinism_rerun(self, tmp_path):
        panel = _clean_panel()
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a = apply_bessembinder_section8(panel, GROUP, SORT, "excntry", spill_dir=tmp_path / "a")
        b = apply_bessembinder_section8(panel, GROUP, SORT, "excntry", spill_dir=tmp_path / "b")
        assert a.sort(GROUP + [SORT]).collect().equals(b.sort(GROUP + [SORT]).collect())
