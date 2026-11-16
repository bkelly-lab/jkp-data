"""
Unit tests for the DFF (Davis-Fama-French) hand-collected BE splice in
link_accounting_to_securities.

DFF (2000) collected Moody's BE for NYSE firms absent from Compustat; the
splice unions the permno-keyed DFF rows with the linked Compustat rows,
DFF-first on any (id, year) collision (be = coalesce(dff, comp); the
sources agree where both exist). DFF be(t) is publicly
available by June 30 of year t and pairs with Dec(t-1) ME, so the unioned
row carries year = t - 1 (the June-frame join uses cyp1 = year + 1).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import link_accounting_to_securities

# Reuse the synthetic fiscal-record and world_msf builders from the dedup tests.
from tests.unit.test_ff_compustat_dedup import _comp, _write_us_msf


def _write_dff(path, rows):
    """rows: list of (permno, first_year, last_year, {year: be}).

    The loader derives the first BE column's year from min(first_moody_year)
    (the real file's invariant), so columns here start at min(fy) across rows.
    """
    start = min(fy for _, fy, _, _ in rows)
    end = max(max(be_by_year, default=start) for _, _, _, be_by_year in rows)
    years = range(start, max(end, start) + 1)
    lines = []
    for permno, fy, ly, be_by_year in rows:
        vals = [f"{be_by_year.get(y, -99.990):.3f}" for y in years]
        lines.append(f"{permno} {fy} {ly} " + " ".join(vals))
    path.write_text("\n".join(lines) + "\n")
    return path


def _compustat_inputs(
    test_paths, *, gvkey="001", permno=10001, years=(2018, 2019), be=None, inv=None
):
    """Fiscal records of one gvkey (be 50 + i unless given) and the June rows
    that link them to the permno; returns the records."""
    be, inv = be or {}, inv or {}
    _write_us_msf(test_paths, [(permno, date(y + 1, 6, 30), gvkey) for y in years])
    return _comp([(gvkey, y, be.get(y, 50.0 + i), 0.1, inv.get(y)) for i, y in enumerate(years)])


class TestDffSplice:
    def test_dff_fills_missing_permno_year(self, test_paths):
        comp = _compustat_inputs(test_paths, permno=10001)
        dff = _write_dff(
            test_paths.raw_tables_dir / "dff.txt", [(20002, 1930, 1932, {1930: 5.0, 1931: 6.0})]
        )
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        row = out.filter((pl.col("id") == 20002) & (pl.col("year") == 1929))
        assert row.height == 1
        assert row["be"][0] == 5.0
        assert row["gvkey"][0] is None
        assert row["op"][0] is None
        assert row["inv"][0] is None

    def test_year_offset_alignment(self, test_paths):
        # DFF be(t=1930) must land as unioned year=1929 so cyp1 = 1930 = june_year,
        # pairing with Dec(1929) ME downstream.
        comp = _compustat_inputs(test_paths)
        dff = _write_dff(test_paths.raw_tables_dir / "dff.txt", [(20002, 1930, 1930, {1930: 5.0})])
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        dff_rows = out.filter(pl.col("id") == 20002)
        assert dff_rows["year"].to_list() == [1929]
        # Synthetic datadate stamps the formation June (of DFF year t), not t-1.
        assert dff_rows["datadate"].to_list() == [date(1930, 6, 30)]

    def test_collision_coalesces_dff_be_keeps_compustat_fields(self, test_paths):
        # Compustat fiscal 2019 row (year=2019) vs DFF be(2020) (-> year=2019),
        # same permno: ONE merged row — DFF be, Compustat identity/accounting.
        comp = _compustat_inputs(test_paths, permno=10001, years=(2018, 2019), inv={2019: 0.1})
        dff = _write_dff(
            test_paths.raw_tables_dir / "dff.txt", [(10001, 2020, 2020, {2020: 999.0})]
        )
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        rows = out.filter((pl.col("id") == 10001) & (pl.col("year") == 2019))
        assert rows.height == 1
        assert rows["be"][0] == 999.0  # coalesce(dff, comp) -> DFF value
        assert rows["gvkey"][0] == "001"  # Compustat row identity kept
        assert rows["inv"][0] == 0.1  # Compustat accounting fields survive

    def test_collision_fills_null_compustat_be(self, test_paths):
        # A linked Compustat row exists but its BE is null (seq missing, the
        # dominant 1950s collision case): coalesce fills BE from DFF.
        comp = _compustat_inputs(test_paths, years=(2018, 2019), be={2019: None})
        dff = _write_dff(test_paths.raw_tables_dir / "dff.txt", [(10001, 2020, 2020, {2020: 55.0})])
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        row = out.filter((pl.col("id") == 10001) & (pl.col("year") == 2019))
        assert row.height == 1
        assert row["be"][0] == 55.0
        assert row["gvkey"][0] == "001"

    def test_disjoint_union_row_count(self, test_paths):
        comp = _compustat_inputs(test_paths, permno=10001, years=(2018, 2019))
        dff = _write_dff(
            test_paths.raw_tables_dir / "dff.txt",
            [(20002, 1930, 1932, {1930: 5.0, 1931: 6.0, 1932: 7.0})],
        )
        base = link_accounting_to_securities(test_paths, comp)
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        assert out.height == base.height + 3
        keyed = out.select("id", "year")
        assert keyed.height == keyed.unique().height

    def test_negative_dff_be_dropped(self, test_paths):
        comp = _compustat_inputs(test_paths)
        dff = _write_dff(
            test_paths.raw_tables_dir / "dff.txt", [(20002, 1930, 1931, {1930: -5.0, 1931: 6.0})]
        )
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        assert out.filter(pl.col("id") == 20002)["year"].to_list() == [1930]

    def test_units_no_double_scaling(self, test_paths):
        # be flows through unchanged ($ millions, like Compustat); the 1000x
        # beme scale is applied downstream, not in the loader.
        comp = _compustat_inputs(test_paths)
        dff = _write_dff(
            test_paths.raw_tables_dir / "dff.txt", [(20002, 1930, 1930, {1930: 67.743})]
        )
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        assert out.filter(pl.col("id") == 20002)["be"][0] == pytest.approx(67.743)

    def test_no_fanout_on_overlapping_inputs(self, test_paths):
        comp = _compustat_inputs(test_paths, permno=10001, years=(2018, 2019))
        dff = _write_dff(
            test_paths.raw_tables_dir / "dff.txt",
            [
                (10001, 2019, 2020, {2019: 1.0, 2020: 2.0}),  # both collide
                (20002, 1930, 1930, {1930: 5.0}),
            ],
        )
        out = link_accounting_to_securities(test_paths, comp, dff_path=dff)

        keyed = out.select("id", "year")
        assert keyed.height == keyed.unique().height
        # Colliding cells merge into the Compustat rows (DFF be via coalesce);
        # only the disjoint DFF row carries a null gvkey.
        assert out.filter(pl.col("gvkey").is_null()).height == 1
        # And the colliding rows carry the DFF be values.
        assert out.filter((pl.col("id") == 10001) & (pl.col("year") == 2018))["be"][0] == 1.0
        assert out.filter((pl.col("id") == 10001) & (pl.col("year") == 2019))["be"][0] == 2.0
