"""Tests for scripts/themes/theme_diagnostics.py: correlations, fits and the misfit rule."""

import importlib.util
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "themes" / "theme_diagnostics.py"


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("theme_diagnostics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def toy():
    """Standardized simulated data for six characteristics in three subthemes, and their themes."""
    rng = np.random.default_rng(0)
    n = 5_000
    fa, fb, fc = rng.normal(size=(3, n))
    cols = {
        "a1": fa + rng.normal(size=n),
        "a2": fa + rng.normal(size=n),
        "a3": -fa + rng.normal(size=n),  # the opposite of A, so subtheme_sign -1
        "b1": fb + rng.normal(size=n),
        "b2": 2 * fa + 0.2 * fb + rng.normal(size=n),  # planted misfit: mostly A
        "c1": fc + rng.normal(size=n),
    }
    z = np.column_stack([(v - v.mean()) / v.std() for v in cols.values()])
    themes = pl.DataFrame(
        {
            "characteristic": list(cols),
            "theme": ["T1", "T1", "T1", "T2", "T2", "T2"],
            "subtheme": ["A", "A", "A", "B", "B", "C"],
            "subtheme_sign": [1, 1, -1, 1, 1, 1],
            "theme_sign": [1, 1, 1, 1, 1, 1],
        }
    )
    return z, np.corrcoef(z, rowvar=False), themes


def own_fits(diag, corr: np.ndarray, themes: pl.DataFrame) -> dict:
    return dict(diag.member_fits(corr, themes).select("feature", "own").iter_rows())


@pytest.mark.unit
def test_monthly_rank_corr_matches_pairwise_complete_corrcoef(diag):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4))
    x[:, 1] += x[:, 0]
    x[rng.random(x.shape) < 0.2] = np.nan
    x[diag.MIN_OBS - 5 :, 3] = np.nan  # column 3 has too few rows to correlate
    got = diag.monthly_rank_corr(x)
    for i in range(3):
        for j in range(3):
            ok = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
            assert got[i, j] == pytest.approx(np.corrcoef(x[ok, i], x[ok, j])[0, 1])
    assert np.isnan(got[3]).all()


@pytest.mark.unit
def test_compute_corr_treats_nan_as_missing(diag, tmp_path):
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    b = a + rng.normal(size=200)
    b[:20] = np.nan  # Polars ranks NaN above every number
    data = pl.DataFrame({"eom": [date(2000, 1, 31)] * 200, "a": a, "b": b})
    data.write_parquet(tmp_path / "nan.parquet")
    data.with_columns(pl.col("b").fill_nan(None)).write_parquet(tmp_path / "null.parquet")
    with_nan = diag.compute_corr(tmp_path / "nan.parquet", ["a", "b"])
    with_null = diag.compute_corr(tmp_path / "null.parquet", ["a", "b"])
    assert with_nan.equals(with_null)


@pytest.mark.unit
def test_corr_with_composite_matches_explicit_composite(diag, toy):
    z, corr, _ = toy
    idx, w = np.array([0, 1, 2]), np.array([1.0, 1.0, -1.0])
    explicit = np.corrcoef(z[:, 3], z[:, idx] @ w)[0, 1]
    assert diag.corr_with_composite(corr, 3, idx, w) == pytest.approx(explicit)


@pytest.mark.unit
def test_member_fits_leave_one_out_and_signs(diag, toy):
    z, corr, themes = toy
    own = own_fits(diag, corr, themes)
    # a1 against the signed composite of the other members of A: a2 - a3
    assert own["a1"] == pytest.approx(np.corrcoef(z[:, 0], z[:, 1] - z[:, 2])[0, 1])
    # a3 moves against A; its -1 sign turns that into a positive fit
    assert own["a3"] == pytest.approx(-np.corrcoef(z[:, 2], z[:, 0] + z[:, 1])[0, 1])
    assert own["a3"] > 0.3
    assert own["c1"] is None  # the only member of C
    flipped = themes.with_columns(
        subtheme_sign=pl.when(pl.col("characteristic") == "a3").then(1).otherwise("subtheme_sign")
    )
    assert own_fits(diag, corr, flipped)["a3"] == pytest.approx(-own["a3"])


@pytest.mark.unit
def test_planted_misfit_is_flagged(diag, toy):
    _, corr, themes = toy
    flagged = diag.misfits(diag.member_fits(corr, themes))
    assert flagged.select("feature", "best_other").rows() == [("b2", "A")]


@pytest.mark.unit
def test_load_corr_uses_cache_unless_input_is_newer(diag, tmp_path, monkeypatch):
    feats = ["a", "b"]
    cache = tmp_path / "corr.parquet"
    pl.DataFrame({"feature": feats, "a": [1.0, 0.5], "b": [0.5, 1.0]}).write_parquet(cache)
    usa = tmp_path / "USA.parquet"
    recomputed = pl.DataFrame({"feature": feats, "a": [1.0, 0.9], "b": [0.9, 1.0]})
    calls = []

    def fake_compute_corr(path: Path, features: list[str]) -> pl.DataFrame:
        calls.append(path)
        return recomputed

    monkeypatch.setattr(diag, "OUT_DIR", tmp_path)
    monkeypatch.setattr(diag, "INPUT", usa)
    monkeypatch.setattr(diag, "compute_corr", fake_compute_corr)

    assert diag.load_corr(feats)[0, 1] == 0.5  # no input file: the cache is used
    usa.touch()
    os.utime(usa, (0, 0))
    assert diag.load_corr(feats)[0, 1] == 0.5  # input older than the cache
    newer = cache.stat().st_mtime + 10
    os.utime(usa, (newer, newer))
    assert diag.load_corr(feats)[0, 1] == 0.9  # input newer: recomputed
    assert calls == [usa]


@pytest.mark.unit
def test_misfit_rule_thresholds(diag):
    members = pl.DataFrame(
        {
            "feature": ["clear", "weak_everywhere", "small_gap"],
            "own": [0.10, 0.10, 0.50],
            "best_other_r": [0.60, 0.25, 0.60],
        }
    )
    assert diag.misfits(members)["feature"].to_list() == ["clear"]
