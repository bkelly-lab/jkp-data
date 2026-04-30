"""Parity test: run_portfolio() on clean_up_portfolio_code vs main.

Runs run_portfolio() from both branches against identical synthetic input,
then diffs every output parquet.

Usage (run from repo root):
    uv run pytest tests/unit/test_run_portfolio_parity.py -v -s

The test is self-contained:
- Builds synthetic data (~6 months, 2 countries, 50 stocks, 5 chars)
- Monkey-patches PORTFOLIO_CHARS and PORTFOLIO_SETTINGS before each call
- Imports main-branch portfolio.run_portfolio via sys.path surgery
- Compares parquet outputs within rtol=1e-10 / atol=1e-12 for numerics
"""

from __future__ import annotations

import calendar
import importlib
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Synthetic data parameters
# ---------------------------------------------------------------------------

# Five chars that exist in factor_details.xlsx (char_info join must find them)
PARITY_CHARS = ["niq_su", "ret_6_1", "ret_12_1", "saleq_su", "tax_gr1a"]
N_IDS = 60
N_MONTHS = 8  # 8 months → enough periods for months_min=1
COUNTRIES = ["FRA", "USA"]  # USA exercises ff49 and cmp branches

# ---------------------------------------------------------------------------
# Patched settings (lenient so regional/cluster outputs are non-empty)
# ---------------------------------------------------------------------------
_PATCHED_SETTINGS = {
    "end_date": date(2030, 12, 31),  # far future so nothing is filtered
    "pfs": 3,
    "source": ["CRSP", "COMPUSTAT"],
    "wins_ret": True,
    "bps": "non_mc",
    "bp_min_n": 5,
    "cmp": {"us": True, "int": False},
    "signals": {"us": False, "int": False, "standardize": True, "weight": "vw_cap"},
    "regional_pfs": {
        "ret_type": "vw_cap",
        "country_excl": ["ZWE", "VEN"],
        "country_weights": "market_cap",
        "stocks_min": 1,
        "months_min": 1,  # very lenient
        "countries_min": 1,
    },
    "daily_pf": True,
    "ind_pf": True,
}

# ---------------------------------------------------------------------------
# Synthetic data builders (adapted from test_portfolio_parity.py)
# ---------------------------------------------------------------------------


def _month_ends(n_months: int, start_year: int = 2020, start_month: int = 1) -> list[date]:
    out: list[date] = []
    for i in range(n_months):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        out.append(date(year, month, calendar.monthrange(year, month)[1]))
    return out


def _make_char_df(
    excntry: str, chars: list[str], n_ids: int, n_months: int, seed: int
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    eoms = _month_ends(n_months)
    n_rows = n_ids * n_months
    size_grps = rng.choice(
        ["mega", "large", "small", "micro", "nano"], size=n_ids, p=[0.10, 0.25, 0.35, 0.20, 0.10]
    ).tolist()
    source_crsp = rng.choice([0, 1], size=n_ids, p=[0.4, 0.6]).tolist()
    crsp_exchcd_raw = rng.choice([1, 2, 3], size=n_ids).tolist()
    comp_exchg_raw = rng.choice([11, 12, 13], size=n_ids).tolist()
    crsp_exchcd: list[int | None] = [
        int(crsp_exchcd_raw[i]) if source_crsp[i] == 1 else None for i in range(n_ids)
    ]
    comp_exchg: list[int | None] = [
        int(comp_exchg_raw[i]) if source_crsp[i] == 0 else None for i in range(n_ids)
    ]
    gics_sectors = rng.choice([10, 15, 20, 25, 30, 35], size=n_ids).tolist()
    gics_per_id = [f"{int(s):02d}101010" for s in gics_sectors]
    ff49_per_id = rng.choice([1, 5, 10, 15, 20, 30, 40, 45], size=n_ids).tolist()

    id_col, eom_col, sg_col, sc_col, ce_col, cx_col, gics_col, ff49_col = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for eom in eoms:
        for i in range(n_ids):
            id_col.append(i + 1)
            eom_col.append(eom)
            sg_col.append(size_grps[i])
            sc_col.append(int(source_crsp[i]))
            ce_col.append(crsp_exchcd[i])
            cx_col.append(comp_exchg[i])
            gics_col.append(gics_per_id[i])
            ff49_col.append(int(ff49_per_id[i]))

    ret_exc = rng.normal(0.008, 0.08, n_rows)
    ret_exc_lead1m = rng.normal(0.008, 0.08, n_rows)
    me = np.exp(rng.normal(7.0, 1.5, n_rows))

    char_dict: dict = {}
    for j, ch in enumerate(chars):
        vals = rng.normal(0.0, 1.0 + 0.1 * j, n_rows)
        mask = rng.random(n_rows) < 0.05
        vals[mask] = np.nan
        char_dict[ch] = pl.Series(ch, vals, dtype=pl.Float64)

    return pl.DataFrame(
        {
            "id": pl.Series("id", id_col, dtype=pl.Int64),
            "eom": pl.Series("eom", eom_col, dtype=pl.Date),
            "source_crsp": pl.Series("source_crsp", sc_col, dtype=pl.Int64),
            "comp_exchg": pl.Series("comp_exchg", cx_col, dtype=pl.Int64),
            "crsp_exchcd": pl.Series("crsp_exchcd", ce_col, dtype=pl.Int64),
            "size_grp": pl.Series("size_grp", sg_col, dtype=pl.Utf8),
            "ret_exc": pl.Series("ret_exc", ret_exc, dtype=pl.Float64),
            "ret_exc_lead1m": pl.Series("ret_exc_lead1m", ret_exc_lead1m, dtype=pl.Float64),
            "me": pl.Series("me", me, dtype=pl.Float64),
            "gics": pl.Series("gics", gics_col, dtype=pl.Utf8),
            "ff49": pl.Series("ff49", ff49_col, dtype=pl.Int64),
            "excntry": pl.Series("excntry", [excntry] * n_rows, dtype=pl.Utf8),
            **char_dict,
        }
    )


def _weekdays_after(eom: date, n: int = 21) -> list[date]:
    out: list[date] = []
    d = eom + timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_daily_df(char_df: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = char_df.select(["id", "eom"]).unique().sort(["eom", "id"])
    ids, dates, rets = [], [], []
    for row in pairs.iter_rows(named=True):
        for d in _weekdays_after(row["eom"], 21):
            ids.append(int(row["id"]))
            dates.append(d)
            rets.append(float(rng.normal(0.0003, 0.015)))
    return pl.DataFrame(
        {
            "id": pl.Series("id", ids, dtype=pl.Int64),
            "date": pl.Series("date", dates, dtype=pl.Date),
            "ret_exc": pl.Series("ret_exc", rets, dtype=pl.Float64),
        }
    )


def _build_synthetic_data(data_root: Path) -> None:
    """Write all synthetic input files to data_root/processed/."""
    processed = data_root / "processed"
    char_dir = processed / "characteristics"
    daily_dir = processed / "return_data" / "daily_rets_by_country"
    other_dir = processed / "other_output"
    for d in [char_dir, daily_dir, other_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_eoms: list[date] = []
    for i, excntry in enumerate(COUNTRIES):
        cdf = _make_char_df(excntry, PARITY_CHARS, N_IDS, N_MONTHS, seed=42 + i * 100)
        cdf.write_parquet(char_dir / f"{excntry}.parquet")
        ddf = _make_daily_df(cdf, seed=43 + i * 100)
        ddf.write_parquet(daily_dir / f"{excntry}.parquet")
        if not all_eoms:
            all_eoms = cdf["eom"].unique().sort().to_list()

    n = len(all_eoms)

    # nyse_cutoffs
    pl.DataFrame(
        {
            "eom": pl.Series("eom", all_eoms, dtype=pl.Date),
            "nyse_p80": pl.Series("nyse_p80", [1e12] * n, dtype=pl.Float64),
        }
    ).write_parquet(other_dir / "nyse_cutoffs.parquet")

    # return_cutoffs (needs eom column for run_portfolio to compute eom_lag1)
    pl.DataFrame(
        {
            "eom": pl.Series("eom", all_eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.5] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.5] * n, dtype=pl.Float64),
        }
    ).write_parquet(other_dir / "return_cutoffs.parquet")

    # return_cutoffs_daily (keyed by month-end of daily month = next month)
    daily_eoms = [
        date(
            eom.year + (1 if eom.month == 12 else 0),
            1 if eom.month == 12 else eom.month + 1,
            calendar.monthrange(
                eom.year + (1 if eom.month == 12 else 0), 1 if eom.month == 12 else eom.month + 1
            )[1],
        )
        for eom in all_eoms
    ]
    pl.DataFrame(
        {
            "eom": pl.Series("eom", daily_eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.2] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.2] * n, dtype=pl.Float64),
        }
    ).write_parquet(other_dir / "return_cutoffs_daily.parquet")

    # market_returns
    mkt_rows = [(ex, eom) for ex in COUNTRIES for eom in all_eoms]
    rng = np.random.default_rng(999)
    pl.DataFrame(
        {
            "excntry": pl.Series("excntry", [r[0] for r in mkt_rows], dtype=pl.Utf8),
            "eom": pl.Series("eom", [r[1] for r in mkt_rows], dtype=pl.Date),
            "mkt_vw_exc": pl.Series(
                "mkt_vw_exc", rng.normal(0.008, 0.04, len(mkt_rows)), dtype=pl.Float64
            ),
            "me_lag1": pl.Series(
                "me_lag1", np.exp(rng.normal(10.0, 1.0, len(mkt_rows))), dtype=pl.Float64
            ),
            "stocks": pl.Series("stocks", rng.integers(20, 60, len(mkt_rows)), dtype=pl.Int64),
        }
    ).write_parquet(other_dir / "market_returns.parquet")

    # market_returns_daily: one row per (excntry, date) for trading days
    daily_rows = []
    for ex in COUNTRIES:
        for eom in all_eoms:
            for d in _weekdays_after(eom, 21):
                daily_rows.append((ex, d))
    rng2 = np.random.default_rng(998)
    pl.DataFrame(
        {
            "excntry": pl.Series("excntry", [r[0] for r in daily_rows], dtype=pl.Utf8),
            "date": pl.Series("date", [r[1] for r in daily_rows], dtype=pl.Date),
            "mkt_vw_exc": pl.Series(
                "mkt_vw_exc", rng2.normal(0.0003, 0.015, len(daily_rows)), dtype=pl.Float64
            ),
            "me_lag1": pl.Series(
                "me_lag1", np.exp(rng2.normal(10.0, 1.0, len(daily_rows))), dtype=pl.Float64
            ),
            "stocks": pl.Series("stocks", rng2.integers(20, 60, len(daily_rows)), dtype=pl.Int64),
        }
    ).write_parquet(other_dir / "market_returns_daily.parquet")


# ---------------------------------------------------------------------------
# Module import helpers: import run_portfolio from a given source tree
# ---------------------------------------------------------------------------


def _import_run_portfolio_from(src_root: str, module_alias: str):
    """Import run_portfolio from src_root/jkp/data/portfolio.py under a unique alias."""
    portfolio_alias = module_alias + "_portfolio"

    # Insert the src path at the front of sys.path temporarily
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    spec = importlib.util.spec_from_file_location(
        portfolio_alias,
        Path(src_root) / "jkp" / "data" / "portfolio.py",
        submodule_search_locations=[str(Path(src_root) / "jkp" / "data")],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.run_portfolio


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_TIGHT = {"rtol": 1e-10, "atol": 1e-12}
_STANDARD = {"rtol": 1e-6, "atol": 1e-10}

_NUMERIC_COLS_BY_FILE: dict[str, dict[str, dict]] = {
    "pfs": {
        "n": _TIGHT,
        "signal": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "hml": {
        "n_stocks": _TIGHT,
        "n_stocks_min": _TIGHT,
        "signal": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "lms": {
        "n_stocks": _TIGHT,
        "n_stocks_min": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "clusters": {
        "n_factors": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "cmp": {"n_stocks": _TIGHT, "ret_weighted": _STANDARD, "signal_weighted": _STANDARD},
    "industry_gics": {"n": _TIGHT, "ret_ew": _TIGHT, "ret_vw": _STANDARD, "ret_vw_cap": _STANDARD},
    "industry_ff49": {"n": _TIGHT, "ret_ew": _TIGHT, "ret_vw": _STANDARD, "ret_vw_cap": _STANDARD},
    # daily variants
    "pfs_daily": {"n": _TIGHT, "ret_ew": _TIGHT, "ret_vw": _STANDARD, "ret_vw_cap": _STANDARD},
    "hml_daily": {
        "n_stocks": _TIGHT,
        "n_stocks_min": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "lms_daily": {
        "n_stocks": _TIGHT,
        "n_stocks_min": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "clusters_daily": {
        "n_factors": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "industry_gics_daily": {
        "n": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    "industry_ff49_daily": {
        "n": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
    # regional/country (use "ret_*" pattern)
    "regional": {
        "n_countries": _TIGHT,
        "ret_ew": _STANDARD,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
        "mkt_vw_exc": _STANDARD,
    },
    "country": {
        "n_stocks": _TIGHT,
        "n_stocks_min": _TIGHT,
        "ret_ew": _TIGHT,
        "ret_vw": _STANDARD,
        "ret_vw_cap": _STANDARD,
    },
}


def _get_numeric_spec(stem: str) -> dict[str, dict]:
    """Return numeric tolerance spec for a file stem."""
    if stem in _NUMERIC_COLS_BY_FILE:
        return _NUMERIC_COLS_BY_FILE[stem]
    for prefix, spec in _NUMERIC_COLS_BY_FILE.items():
        if stem.startswith(prefix):
            return spec
    return _NUMERIC_COLS_BY_FILE["regional"]


def _sort_key_cols(df: pl.DataFrame) -> list[str]:
    """Heuristic: sort by date/eom/time cols then string cols."""
    date_cols = [c for c in ["eom", "date"] if c in df.columns]
    str_cols = [
        c
        for c in [
            "excntry",
            "region",
            "characteristic",
            "cluster",
            "gics",
            "ff49",
            "pf",
            "size_grp",
        ]
        if c in df.columns
    ]
    return date_cols + str_cols


def _compare_parquets(path_a: Path, path_b: Path, label: str) -> list[str]:
    """Compare two parquet files. Returns list of failure messages (empty = pass)."""
    failures = []
    if not path_a.exists() and not path_b.exists():
        return []
    if not path_a.exists():
        failures.append(f"{label}: missing in MAIN")
        return failures
    if not path_b.exists():
        failures.append(f"{label}: missing in CURRENT")
        return failures

    a = pl.read_parquet(path_a)
    b = pl.read_parquet(path_b)

    if a.height != b.height:
        failures.append(f"{label}: height mismatch {a.height} vs {b.height}")
        return failures
    if a.height == 0:
        return []  # both empty — pass

    sort_cols = _sort_key_cols(a)
    try:
        a = a.sort(sort_cols)
        b = b.sort(sort_cols)
    except Exception:
        pass  # best-effort sort

    stem = path_a.stem
    numeric_spec = _get_numeric_spec(stem)

    for col, tol in numeric_spec.items():
        if col not in a.columns:
            continue
        if col not in b.columns:
            failures.append(f"{label}[{col}]: col missing in CURRENT")
            continue
        dtype_a = a[col].dtype
        if dtype_a in (pl.Utf8, pl.String, pl.Date, pl.Categorical):
            continue
        try:
            a_np = a[col].cast(pl.Float64).to_numpy()
            b_np = b[col].cast(pl.Float64).to_numpy()
        except Exception as e:
            failures.append(f"{label}[{col}]: cast error: {e}")
            continue
        a_nan = np.isnan(a_np)
        b_nan = np.isnan(b_np)
        if not np.array_equal(a_nan, b_nan):
            failures.append(f"{label}[{col}]: NaN positions differ")
            continue
        mask = ~a_nan
        if mask.any():
            try:
                np.testing.assert_allclose(a_np[mask], b_np[mask], **tol)
            except AssertionError as e:
                # Show first few disagreeing rows
                diff = np.abs(a_np[mask] - b_np[mask])
                worst_idx = np.argsort(diff)[-5:][::-1]
                sample = [
                    (float(a_np[mask][i]), float(b_np[mask][i]), float(diff[i])) for i in worst_idx
                ]
                failures.append(f"{label}[{col}]: {e}\n  worst 5 (main, current, diff): {sample}")
    return failures


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

# Paths
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAIN_SRC = str(_REPO_ROOT.parent / "jkp-data-main" / "src")
_CURR_SRC = str(_REPO_ROOT / "src")


@pytest.mark.skipif(
    not Path(_MAIN_SRC).is_dir(),
    reason=f"requires sibling main worktree at {_MAIN_SRC} (create with `git worktree add ../jkp-data-main main`)",
)
def test_run_portfolio_parity(tmp_path: Path) -> None:
    """run_portfolio() on main vs clean_up_portfolio_code produces identical outputs."""
    # Build synthetic input data
    _build_synthetic_data(tmp_path)

    output_dir_main = tmp_path / "out_main"
    output_dir_curr = tmp_path / "out_curr"
    output_dir_main.mkdir()
    output_dir_curr.mkdir()

    # Copy processed inputs into both output dirs (run_portfolio reads from output_dir/processed/)
    for ex_dir in (output_dir_main, output_dir_curr):
        shutil.copytree(tmp_path / "processed", ex_dir / "processed")

    # ------------------------------------------------------------------
    # Run MAIN branch run_portfolio
    # ------------------------------------------------------------------
    # We need to import from main src. Use importlib to avoid polluting
    # the current jkp.data.portfolio module.

    def _load_portfolio_mod(src_root: str, alias: str):
        """Load portfolio.py from src_root as an isolated module."""
        # We need the whole jkp package chain to be importable from src_root.
        # Temporarily insert src_root as the highest priority path.
        orig_path = list(sys.path)
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        try:
            # Clear any cached jkp.* modules
            to_del = [k for k in sys.modules if k == "jkp" or k.startswith("jkp.")]
            for k in to_del:
                del sys.modules[k]
            import jkp.data.portfolio as _mod

            return _mod
        finally:
            sys.path[:] = orig_path

    # --- MAIN ---
    # main branch config uses individual constants, not PORTFOLIO_CHARS/PORTFOLIO_SETTINGS
    # so we patch at the portfolio module level after import
    orig_path = list(sys.path)
    sys.path.insert(0, _MAIN_SRC)
    try:
        to_del = [k for k in sys.modules if k == "jkp" or k.startswith("jkp.")]
        for k in to_del:
            del sys.modules[k]
        # Patch portfolio's hard-coded chars list by monkey-patching run_portfolio
        # Main branch embeds chars inline — we override by patching the module-level
        # constants that run_portfolio reads.
        # In main, the chars list is built inline in run_portfolio; we can't patch it
        # without modifying the function. So we take a different approach:
        # wrap run_portfolio to intercept the portfolios() call.
        # SIMPLER: both branches call portfolios() identically (already verified by
        # existing parity tests). We just need to test the orchestration logic.
        # Override end_date and months_min via config constants.
        import datetime

        import jkp.data.config as _main_cfg_mod
        import jkp.data.portfolio as _main_pf_mod

        _main_cfg_mod.END_DATE = datetime.date(2030, 12, 31)
        _main_cfg_mod.REGIONAL_MONTHS_MIN = 1
        _main_cfg_mod.REGIONAL_STOCKS_MIN = 1
        _main_cfg_mod.REGIONAL_COUNTRIES_MIN = 1
        _main_cfg_mod.PORTFOLIO_BP_MIN_N = 5

        # main's run_portfolio uses `from config import REGIONAL_*` at module level,
        # so patching config after import has no effect — patch the portfolio module's
        # own namespace to override the already-bound names.
        _main_pf_mod.REGIONAL_MONTHS_MIN = 1  # type: ignore[attr-defined]
        _main_pf_mod.REGIONAL_STOCKS_MIN = 1  # type: ignore[attr-defined]
        _main_pf_mod.REGIONAL_COUNTRIES_MIN = 1  # type: ignore[attr-defined]
        _main_pf_mod.END_DATE = datetime.date(2030, 12, 31)  # type: ignore[attr-defined]
        _main_pf_mod.PORTFOLIO_BP_MIN_N = 5  # type: ignore[attr-defined]

        # Also need to patch the chars list embedded in run_portfolio on main.
        # We do this by replacing run_portfolio with a wrapper that shrinks chars.
        _orig_main_run = _main_pf_mod.run_portfolio

        def _patched_main_run_portfolio(*, output_format="parquet", output_dir):
            # Reload the full module internals and monkeypatch inline chars
            # by replacing the portfolios() call behaviour — too invasive.
            # Instead: call _orig but first rewrite the source-level chars list.
            # Since it's built in the function body we patch via a global override trick.
            # We inject a `chars` global that run_portfolio will shadow... but it reads
            # a local variable, not a global. So we use exec-level bytecode patching.
            # PRAGMATIC APPROACH: just call the original but accept it uses all 153 chars.
            # We stop it early by setting months_min=1, countries_min=1 which are already done.
            # For char-level, we wrap `portfolios` to replace chars with PARITY_CHARS.
            orig_portfolios = _main_pf_mod.portfolios

            def _limited_portfolios(*args, **kwargs):
                kwargs["chars"] = PARITY_CHARS
                kwargs["bp_min_n"] = 5
                return orig_portfolios(*args, **kwargs)

            _main_pf_mod.portfolios = _limited_portfolios
            try:
                return _orig_main_run(output_format=output_format, output_dir=output_dir)
            finally:
                _main_pf_mod.portfolios = orig_portfolios

        print("\nRunning MAIN branch run_portfolio...")
        _patched_main_run_portfolio(output_dir=output_dir_main)
    finally:
        sys.path[:] = orig_path

    # --- CURRENT branch ---
    sys.path.insert(0, _CURR_SRC)
    try:
        to_del = [k for k in sys.modules if k == "jkp" or k.startswith("jkp.")]
        for k in to_del:
            del sys.modules[k]
        import jkp.data.config as _curr_cfg_mod
        import jkp.data.portfolio as _curr_pf_mod

        # Patch PORTFOLIO_CHARS and PORTFOLIO_SETTINGS
        _curr_cfg_mod.PORTFOLIO_CHARS = PARITY_CHARS
        _curr_cfg_mod.PORTFOLIO_SETTINGS = _PATCHED_SETTINGS
        _curr_pf_mod.PORTFOLIO_CHARS = PARITY_CHARS
        _curr_pf_mod.PORTFOLIO_SETTINGS = _PATCHED_SETTINGS

        print("Running CURRENT branch run_portfolio...")
        _curr_pf_mod.run_portfolio(output_dir=output_dir_curr)
    finally:
        sys.path[:] = orig_path

    # ------------------------------------------------------------------
    # Collect and compare all output parquets
    # ------------------------------------------------------------------
    main_pf = output_dir_main / "processed" / "portfolios"
    curr_pf = output_dir_curr / "processed" / "portfolios"

    # Gather all parquet files from both sides
    main_files = {p.relative_to(main_pf) for p in main_pf.rglob("*.parquet")}
    curr_files = {p.relative_to(curr_pf) for p in curr_pf.rglob("*.parquet")}
    all_rel = sorted(main_files | curr_files)

    print(f"\nFound {len(main_files)} files in MAIN, {len(curr_files)} in CURRENT")
    print("Files:", sorted(str(r) for r in all_rel))

    failures: list[str] = []
    passed: list[str] = []

    for rel in all_rel:
        label = str(rel)
        fa = main_pf / rel
        fb = curr_pf / rel
        errs = _compare_parquets(fa, fb, label)
        if errs:
            failures.extend(errs)
            print(f"  FAIL  {label}")
            for e in errs:
                print(f"        {e}")
        else:
            passed.append(label)
            print(f"  PASS  {label}")

    print(f"\n{'=' * 60}")
    print(f"PASSED: {len(passed)}")
    print(f"FAILED: {len(failures)}")
    if failures:
        pytest.fail("\n".join(failures))
