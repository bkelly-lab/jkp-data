"""Golden-fixture parity test for run_portfolio.

For each profile, run ``run_portfolio`` and build themes v0 with ``run_themes``
on deterministic synthetic data (seed=42) and compare every emitted parquet against a committed fixture
tree under ``tests/golden/fixtures/<profile>/``.

Run ``pytest tests/golden/test_run_portfolio_golden.py --regen-golden -v``
to regenerate fixtures.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tests.unit.portfolio.conftest import (
    build_synthetic_data,
    compare_parquets,
    make_multi_region_classification,
    make_themes_dir,
)

# ---------------------------------------------------------------------------
# Profile matrix
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "default_monthly": {
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "cmp_us": False,
        "daily_pf": False,
        "ind_pf": True,
    },
    "nyse_breakpoints": {
        "bps": "nyse",
        "source": ["CRSP", "COMPUSTAT"],
        "cmp_us": False,
        "daily_pf": False,
        "ind_pf": True,
    },
    "crsp_only": {
        "bps": "non_mc",
        "source": ["CRSP"],
        "cmp_us": False,
        "daily_pf": False,
        "ind_pf": False,
    },
    "cmp_on": {
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "cmp_us": True,
        "daily_pf": False,
        "ind_pf": True,
    },
    "full_daily": {
        "bps": "non_mc",
        "source": ["CRSP", "COMPUSTAT"],
        "cmp_us": False,
        "daily_pf": True,
        "ind_pf": True,
    },
}

FIXTURES_DIR = Path(__file__).parent / "fixtures"
COUNTRIES = ["USA", "FRA"]
CHARS = ["char_a", "char_b"]


def _build_settings(cfg: dict) -> dict:
    """Build a PORTFOLIO_SETTINGS dict from a profile cfg."""
    return {
        "end_date": date(2030, 12, 31),
        "pfs": 3,
        "source": cfg["source"],
        "wins_ret": True,
        "bps": cfg["bps"],
        "bp_min_n": 5,
        "cmp": {"us": cfg["cmp_us"], "int": cfg["cmp_us"]},
        "regional_pfs": {
            "country_excl": ["ZWE", "VEN"],
            "country_weights": "market_cap",
            "stocks_min": 1,
            "months_min": 1,
            "countries_min": 1,
        },
        "daily_pf": cfg["daily_pf"],
        "ind_pf": cfg["ind_pf"],
    }


@pytest.mark.parametrize("profile", list(PROFILES.keys()))
def test_run_portfolio_golden(
    profile: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    reset_output_writer: None,
) -> None:
    cfg = PROFILES[profile]
    chars = CHARS

    build_synthetic_data(tmp_path, COUNTRIES, chars, seed=42)

    cc = make_multi_region_classification(tmp_path)
    themes_dir = make_themes_dir(tmp_path, dict.fromkeys(chars, "cluster_x"))

    # Paths are imported locally inside run_portfolio via `from .paths import ...`,
    # so patch the source module rather than jkp.data.portfolio.
    monkeypatch.setattr("jkp.data.paths.get_country_classification_path", lambda: cc)
    monkeypatch.setattr("jkp.data.paths.get_themes_dir", lambda: themes_dir)
    monkeypatch.setattr("jkp.data.portfolio.PORTFOLIO_CHARS", chars)
    monkeypatch.setattr("jkp.data.portfolio.PORTFOLIO_SETTINGS", _build_settings(cfg))

    from jkp.data.portfolio import run_portfolio, run_themes

    run_portfolio(output_dir=tmp_path)
    monkeypatch.setattr("jkp.data.output_writer._configured", False)  # a new CLI process
    run_themes(output_dir=tmp_path, version="v0")

    output_root = tmp_path / "processed" / "portfolios"
    fixture_root = FIXTURES_DIR / profile

    if request.config.getoption("--regen-golden"):
        if fixture_root.exists():
            shutil.rmtree(fixture_root)
        shutil.copytree(output_root, fixture_root)
        pytest.skip(f"Regenerated fixture {profile}")

    assert fixture_root.exists(), f"missing fixture tree: {fixture_root}"

    failures: list[str] = []
    fixture_files = list(fixture_root.rglob("*.parquet"))
    assert fixture_files, f"no parquet fixtures under {fixture_root}"
    for fixture_file in fixture_files:
        rel = fixture_file.relative_to(fixture_root)
        actual = output_root / rel
        failures.extend(compare_parquets(fixture_file, actual, str(rel)))

    # Check for unexpected extra files in actual output.
    actual_rels = {p.relative_to(output_root) for p in output_root.rglob("*.parquet")}
    fixture_rels = {p.relative_to(fixture_root) for p in fixture_files}
    extra = sorted(str(r) for r in actual_rels - fixture_rels)
    if extra:
        failures.append(f"unexpected extra outputs: {extra}")

    # The pf stacks are not sorted after the country loop, so check they come in key order.
    # compare_parquets sorts both sides and would not see an order error.
    for path in sorted(output_root.glob("pfs*.parquet")):
        pfs = pl.read_parquet(path)
        keys = ["excntry", "characteristic", "pf", "eom" if "eom" in pfs.columns else "date"]
        if not pfs.equals(pfs.sort(keys)):
            failures.append(f"{path.name} is not sorted by {keys}")

    assert not failures, "\n".join(failures)
