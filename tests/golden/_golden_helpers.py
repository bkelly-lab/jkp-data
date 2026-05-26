"""Shared helpers for factor-builder golden tests.

Three responsibilities:
  1. Stage column-projected WRDS slices into a DataPaths layout so the
     cwd-based builders (gen_ff_data, gen_hxz_data, gen_mispricing_data)
     see the inputs they expect.
  2. Run a builder with cwd pinned to ``paths.interim_dir`` (the builders
     read ``Path("../raw/raw_tables")`` and ``Path(".")`` relative to cwd).
  3. Compare actual-vs-golden factor parquets with NaN-aware tolerance
     against ``ToleranceSpec``.

Fixture slice layout (committed under ``tests/golden/fixtures/synthetic_wrds/``):
  crsp_msf_v2.parquet              -> raw_tables_dir
  crsp_dsf_v2.parquet              -> raw_tables_dir
  comp_funda.parquet               -> raw_tables_dir
  comp_fundq.parquet               -> raw_tables_dir
  comp_g_funda.parquet             -> raw_tables_dir
  comp_g_fundq.parquet             -> raw_tables_dir
  crsp_ccmxpf_lnkhist.parquet      -> raw_tables_dir
  crsp_stksecurityinfohist.parquet -> raw_tables_dir
  crsp_a_indexes_msp500.parquet    -> raw_tables_dir
  crsp_a_indexes_acti.parquet      -> raw_tables_dir
  world_msf.parquet                -> interim_dir
  world_dsf.parquet                -> interim_dir
  world_data.parquet               -> interim_dir
  market_returns.parquet           -> interim_dir
  market_returns_daily.parquet     -> interim_dir
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from jkp.data.paths import DataPaths


SYNTHETIC_SLICES_DIR = Path(__file__).parent / "fixtures" / "synthetic_wrds"

RAW_FILES: tuple[str, ...] = (
    "crsp_msf_v2.parquet",
    "crsp_dsf_v2.parquet",
    "comp_funda.parquet",
    "comp_fundq.parquet",
    "comp_g_funda.parquet",
    "comp_g_fundq.parquet",
    "crsp_ccmxpf_lnkhist.parquet",
    "crsp_stksecurityinfohist.parquet",
    "crsp_a_indexes_msp500.parquet",
    "crsp_a_indexes_acti.parquet",
)
INTERIM_FILES: tuple[str, ...] = (
    "world_msf.parquet",
    "world_dsf.parquet",
    "world_data.parquet",
    "market_returns.parquet",
    "market_returns_daily.parquet",
)
INTERIM_SUBDIR_FILES: tuple[tuple[str, str], ...] = (
    # (relative subdir under interim_dir, filename)
    ("raw_data_dfs", "__fx1.parquet"),
)


def stage_synthetic_slices(paths: DataPaths, src_dir: Path = SYNTHETIC_SLICES_DIR) -> None:
    """Symlink committed synthetic slice parquets into the DataPaths layout.

    Missing source files are silently skipped so a builder that doesn't need
    a particular slice doesn't fail.
    """
    for name in RAW_FILES:
        _link(src_dir / name, paths.raw_tables_dir / name)
    for name in INTERIM_FILES:
        _link(src_dir / name, paths.interim_dir / name)
    for subdir, name in INTERIM_SUBDIR_FILES:
        _link(src_dir / name, paths.interim_dir / subdir / name)


def _link(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        # Fallback for filesystems without symlink support.
        import shutil

        shutil.copy2(src, dst)


@contextmanager
def cwd(path: Path) -> Iterator[None]:
    """Temporarily chdir into ``path``. Restores prior cwd on exit."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_builder(paths: DataPaths, fn: Callable[..., None], **kwargs: object) -> None:
    """Run a cwd-based builder (gen_ff_data / gen_hxz_data / gen_mispricing_data)
    with cwd pinned to ``paths.interim_dir``.

    The mispricing builder takes no path kwargs and writes ``mp_*.parquet``
    into cwd; the FF / HXZ builders take three explicit path kwargs.
    """
    with cwd(paths.interim_dir):
        fn(**kwargs)


def compare_factor_parquet(
    actual: Path,
    golden: Path,
    key_cols: Iterable[str],
    rtol: float = 1e-6,
    atol: float = 1e-10,
) -> list[str]:
    """Compare two factor parquets. Returns failure messages (empty = pass).

    - Schema equality (column set + dtypes).
    - Sorted by ``key_cols``; key columns must be bit-equal.
    - All other numeric columns compared with ``rtol`` / ``atol`` and
      NaN positions must match.
    """
    failures: list[str] = []
    if not golden.exists():
        return [f"{golden.name}: golden missing — run the generator script"]
    if not actual.exists():
        return [f"{actual.name}: actual missing — builder did not write output"]

    a = pl.read_parquet(actual)
    g = pl.read_parquet(golden)

    if a.height != g.height:
        return [f"{actual.name}: height mismatch actual={a.height} golden={g.height}"]
    if set(a.columns) != set(g.columns):
        only_a = sorted(set(a.columns) - set(g.columns))
        only_g = sorted(set(g.columns) - set(a.columns))
        return [
            f"{actual.name}: column mismatch (only in actual: {only_a}, only in golden: {only_g})"
        ]
    if a.height == 0:
        return []

    keys = [k for k in key_cols if k in a.columns]
    if keys:
        a = a.sort(keys)
        g = g.sort(keys)

    for k in keys:
        if not a[k].equals(g[k]):
            return [f"{actual.name}: key column {k!r} differs"]

    for col in a.columns:
        if col in keys:
            continue
        a_s = a[col]
        g_s = g[col]
        if a_s.dtype != g_s.dtype:
            failures.append(f"{actual.name}: {col!r} dtype actual={a_s.dtype} golden={g_s.dtype}")
            continue
        if not a_s.dtype.is_numeric():
            if not a_s.equals(g_s):
                failures.append(f"{actual.name}: {col!r} (non-numeric) differs")
            continue
        a_np = a_s.to_numpy().astype(np.float64)
        g_np = g_s.to_numpy().astype(np.float64)
        a_nan = np.isnan(a_np)
        g_nan = np.isnan(g_np)
        if not np.array_equal(a_nan, g_nan):
            n = int(np.sum(a_nan != g_nan))
            failures.append(
                f"{actual.name}: {col!r} NaN positions differ at {n} rows"
                f" (actual={int(a_nan.sum())} golden={int(g_nan.sum())})"
            )
            continue
        mask = ~a_nan
        if mask.any():
            try:
                np.testing.assert_allclose(a_np[mask], g_np[mask], rtol=rtol, atol=atol)
            except AssertionError as exc:
                failures.append(f"{actual.name}: {col!r}: {exc!s}")
    return failures


def regen_or_compare(
    actual: Path,
    golden: Path,
    key_cols: Iterable[str],
    *,
    regen: bool,
    rtol: float = 1e-6,
    atol: float = 1e-10,
) -> list[str]:
    """Copy actual -> golden when ``regen`` is True; else compare.

    Returns failure messages (empty = pass). Use under ``--regen-golden``
    flag for golden tests.
    """
    if regen:
        golden.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(actual, golden)
        return []
    return compare_factor_parquet(actual, golden, key_cols, rtol=rtol, atol=atol)
