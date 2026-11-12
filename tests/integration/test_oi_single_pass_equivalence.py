"""Equivalence test for the single-pass overnight/intraday portfolio build (#284).

The legacy ``run_portfolio`` reran ``portfolios()`` once per overnight/intraday
component. The current one aggregates every daily return column in the main
pass. Both run end-to-end on the same synthetic inputs and every file under
``processed/portfolios/`` must match: same files, schemas and rows, exact
integer columns, and float columns within tight tolerance. Since #296 the
signed files are under ``themes/v0/`` and the cluster files are theme files,
so the legacy files are compared through ``_new_path``.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jkp.data.portfolio import run_portfolio, run_themes
from tests.fixtures import portfolio_oi_legacy
from tests.fixtures.portfolio_build import (
    OI_CHARS,
    OI_SETTINGS,
    allow_output_format_reset,
    build_oi_inputs,
    output_files,
    patch_oi_build,
)

pytestmark = pytest.mark.integration

SETTINGS = {
    **OI_SETTINGS,
    # Read only by the frozen legacy code.
    "signals": {"us": False, "int": False, "standardize": True, "weight": "vw_cap"},
}


def _patch(monkeypatch: pytest.MonkeyPatch, resources: Path, settings: dict) -> None:
    themes_dir = patch_oi_build(monkeypatch, resources, settings)
    legacy = "tests.fixtures.portfolio_oi_legacy"
    monkeypatch.setattr(f"{legacy}.V0_DIR", themes_dir / "superseded_themes" / "v0")
    monkeypatch.setattr(f"{legacy}.PORTFOLIO_CHARS", OI_CHARS)
    monkeypatch.setattr(f"{legacy}.PORTFOLIO_SETTINGS", settings)
    monkeypatch.setattr(f"{legacy}.configure_output_format", allow_output_format_reset)


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by key columns: every non-float column plus the float-typed keys."""
    keys = [c for c, t in df.schema.items() if not t.is_float() or c in {"pf", "ff49", "gics"}]
    assert df.select(keys).n_unique() == df.height
    return df.sort(keys, nulls_last=True)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Run the legacy and the single-pass builds on identical inputs."""
    mp = pytest.MonkeyPatch()
    legacy_root = tmp_path_factory.mktemp("legacy")
    new_root = tmp_path_factory.mktemp("new")
    build_oi_inputs(legacy_root)
    build_oi_inputs(new_root)
    _patch(mp, tmp_path_factory.mktemp("resources"), SETTINGS)
    try:
        portfolio_oi_legacy.run_portfolio(output_format="parquet", output_dir=legacy_root)
        run_portfolio(output_format="parquet", output_dir=new_root)
        run_themes(output_dir=new_root, version="v0")
    finally:
        mp.undo()
    return legacy_root / "processed" / "portfolios", new_root / "processed" / "portfolios"


_MOVED = [
    (r"^cmp\.parquet$", "hml_cmp_5.parquet"),
    (r"^lms(.*)\.parquet$", r"themes/v0/lms\1.parquet"),
    (r"^clusters(.*)\.parquet$", r"themes/v0/themes\1.parquet"),
    (r"^(country_factors|regional_factors)(.*)/", r"themes/v0/\1\2/"),
    (r"^regional_clusters(.*)/", r"themes/v0/regional_themes\1/"),
]


def _new_path(legacy_rel: str) -> str:
    """Where the current build writes a legacy output file."""
    for pattern, repl in _MOVED:
        legacy_rel = re.sub(pattern, repl, legacy_rel)
    return legacy_rel


class TestOiSinglePassEquivalence:
    def test_same_files(self, outputs: tuple[Path, Path]) -> None:
        legacy, new = outputs
        # The legacy code has only the top-level cmp, with five size groups, not the
        # other groupings or the copy in themes/v0/.
        not_in_legacy = {
            "hml_cmp_1.parquet",
            "hml_cmp_2.parquet",
            "themes/v0/cmp.parquet",
        }
        v0_and_unsigned = {
            f
            for f in output_files(new)
            if not f.startswith("themes/v1.0/") and f not in not_in_legacy
        }
        assert v0_and_unsigned == {_new_path(f) for f in output_files(legacy)}

    def test_oi_outputs_written(self, outputs: tuple[Path, Path]) -> None:
        _, new = outputs
        files = output_files(new)
        for component in ("overnight", "intraday", "overnight_local", "intraday_local"):
            for name in ("pfs", "hml", "themes/v0/lms", "themes/v0/themes"):
                assert f"{name}_{component}.parquet" in files
            assert any(f.startswith(f"themes/v0/regional_factors_{component}/") for f in files)

    def test_all_outputs_match(self, outputs: tuple[Path, Path], tolerance) -> None:
        legacy, new = outputs
        for rel in sorted(output_files(legacy)):
            a = pl.read_parquet(legacy / rel).rename({"cluster": "theme"}, strict=False)
            b = pl.read_parquet(new / _new_path(rel)).drop("n_subthemes", strict=False)
            assert b.schema == a.schema, rel
            assert b.height == a.height, rel
            # Keys and integer columns must match exactly. Float sums may
            # differ in the last bits because the shared join feeds rows to
            # each sum in a different order than the per-component passes.
            try:
                assert_frame_equal(
                    _sorted(b), _sorted(a), check_exact=False, **tolerance.polars(tolerance.TIGHT)
                )
            except AssertionError as e:
                raise AssertionError(f"{rel}: {e}") from e

    def test_edge_cases_present(self, outputs: tuple[Path, Path]) -> None:
        """Guard that the fixture exercises the per-leg null handling."""
        _, new = outputs
        std = pl.read_parquet(new / "pfs_daily.parquet")
        ov = pl.read_parquet(new / "pfs_overnight.parquet")
        ov_local = pl.read_parquet(new / "pfs_overnight_local.parquet")
        assert ov.filter(pl.col("excntry") == "DEU").height == 0
        assert ov.height < std.height
        assert ov_local["n"].sum() < ov["n"].sum()
        assert (ov["ret_ew"] != 0).all()
