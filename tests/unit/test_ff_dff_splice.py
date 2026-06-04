"""
Unit tests for the DFF (Davis-Fama-French) hand-collected BE splice in
ff_load_compustat_us.

DFF (2000) collected Moody's BE for NYSE firms absent from Compustat; the
splice unions the permno-keyed DFF rows with the CCM-linked Compustat rows,
Compustat-first on any (permno, year) collision. DFF be(t) is publicly
available by June 30 of year t and pairs with Dec(t-1) ME, so the unioned
row carries year = t - 1 (the June-frame join uses cyp1 = year + 1).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data import config
from jkp.data.aux_functions import ff_load_compustat_us

# Reuse the synthetic comp_funda/lnkhist builders from the dedup tests.
from tests.unit.test_ff_compustat_dedup import _funda_row, _write_funda, _write_lnkhist


def _write_dff(path, rows):
    """rows: list of (permno, first_year, last_year, {year: be}).

    The loader derives the first BE column's year from min(first_moody_year)
    (the real file's invariant), so columns here start at min(fy) across rows.
    """
    start = min(fy for _, fy, _, _ in rows)
    years = range(start, 2002)
    lines = []
    for permno, fy, ly, be_by_year in rows:
        vals = [f"{be_by_year.get(y, -99.990):.3f}" for y in years]
        lines.append(f"{permno} {fy} {ly} " + " ".join(vals))
    path.write_text("\n".join(lines) + "\n")
    return path


def _compustat_inputs(tmp_path, *, gvkey="001", permno=10001, years=(2018, 2019)):
    rows = [_funda_row(gvkey, date(y, 12, 31), 100.0 + i, indl=True) for i, y in enumerate(years)]
    _write_funda(tmp_path / "comp_funda.parquet", rows)
    _write_lnkhist(tmp_path / "crsp_ccmxpf_lnkhist.parquet", gvkey, permno)


class TestDffSplice:
    def test_use_dff_false_is_noop(self, tmp_path):
        _compustat_inputs(tmp_path)
        base = ff_load_compustat_us(tmp_path, ff5=True).sort(["permno", "year"])
        off = ff_load_compustat_us(tmp_path, ff5=True, use_dff=False).sort(["permno", "year"])
        assert off.equals(base)

    def test_dff_fills_missing_permno_year(self, tmp_path):
        _compustat_inputs(tmp_path, permno=10001)
        dff = _write_dff(tmp_path / "dff.txt", [(20002, 1930, 1932, {1930: 5.0, 1931: 6.0})])
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        row = out.filter((pl.col("permno") == 20002) & (pl.col("year") == 1929))
        assert row.height == 1
        assert row["be"][0] == 5.0
        assert row["gvkey"][0] is None
        assert row["op"][0] is None
        assert row["inv"][0] is None
        assert row["count"][0] == config.FF_DFF_SYNTH_COUNT

    def test_year_offset_alignment(self, tmp_path):
        # DFF be(t=1930) must land as unioned year=1929 so cyp1 = 1930 = june_year,
        # pairing with Dec(1929) ME downstream.
        _compustat_inputs(tmp_path)
        dff = _write_dff(tmp_path / "dff.txt", [(20002, 1930, 1930, {1930: 5.0})])
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        dff_rows = out.filter(pl.col("permno") == 20002)
        assert dff_rows["year"].to_list() == [1929]
        # Synthetic datadate stamps the formation June (of DFF year t), not t-1.
        assert dff_rows["datadate"].to_list() == [date(1930, 6, 30)]

    def test_compustat_wins_on_overlap(self, tmp_path):
        # Compustat fiscal 2019 row (year=2019) vs DFF be(2020) (-> year=2019),
        # same permno: Compustat must win, exactly one row.
        _compustat_inputs(tmp_path, permno=10001, years=(2018, 2019))
        dff = _write_dff(tmp_path / "dff.txt", [(10001, 2020, 2020, {2020: 999.0})])
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        rows = out.filter((pl.col("permno") == 10001) & (pl.col("year") == 2019))
        assert rows.height == 1
        assert rows["be"][0] != 999.0
        assert rows["gvkey"][0] == "001"

    def test_disjoint_union_row_count(self, tmp_path):
        _compustat_inputs(tmp_path, permno=10001, years=(2018, 2019))
        dff = _write_dff(
            tmp_path / "dff.txt",
            [(20002, 1930, 1932, {1930: 5.0, 1931: 6.0, 1932: 7.0})],
        )
        base = ff_load_compustat_us(tmp_path, ff5=True)
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        assert out.height == base.height + 3
        keyed = out.select("permno", "year")
        assert keyed.height == keyed.unique().height

    def test_negative_dff_be_dropped(self, tmp_path):
        _compustat_inputs(tmp_path)
        dff = _write_dff(tmp_path / "dff.txt", [(20002, 1930, 1931, {1930: -5.0, 1931: 6.0})])
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        assert out.filter(pl.col("permno") == 20002)["year"].to_list() == [1930]

    def test_units_no_double_scaling(self, tmp_path):
        # be flows through unchanged ($ millions, like Compustat); the 1000x
        # beme scale is applied downstream, not in the loader.
        _compustat_inputs(tmp_path)
        dff = _write_dff(tmp_path / "dff.txt", [(20002, 1930, 1930, {1930: 67.743})])
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        assert out.filter(pl.col("permno") == 20002)["be"][0] == pytest.approx(67.743)

    def test_gate_pre1954_constant(self, tmp_path, monkeypatch):
        monkeypatch.setattr("jkp.data.aux_functions.FF_DFF_GATE_COMPUSTAT_PRE1954", True)
        # Compustat fiscal years 1950/1951 -> formations June 1951/1952 < 1954: dropped.
        _compustat_inputs(tmp_path, permno=10001, years=(1950, 1951))
        dff = _write_dff(tmp_path / "dff.txt", [(20002, 1930, 1930, {1930: 5.0})])
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        assert out.filter(pl.col("permno") == 10001).height == 0
        assert out.filter(pl.col("permno") == 20002).height == 1

    def test_no_fanout_on_overlapping_inputs(self, tmp_path):
        _compustat_inputs(tmp_path, permno=10001, years=(2018, 2019))
        dff = _write_dff(
            tmp_path / "dff.txt",
            [
                (10001, 2019, 2020, {2019: 1.0, 2020: 2.0}),  # both collide
                (20002, 1930, 1930, {1930: 5.0}),
            ],
        )
        out = ff_load_compustat_us(tmp_path, ff5=True, use_dff=True, dff_path=dff)

        keyed = out.select("permno", "year")
        assert keyed.height == keyed.unique().height
        # Both colliding DFF rows lost to Compustat; the disjoint one survives.
        assert out.filter(pl.col("gvkey").is_null()).height == 1
