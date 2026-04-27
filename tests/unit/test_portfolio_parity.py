"""Parity tests: legacy vs current ``portfolios()`` on synthetic data.

These tests drive both the pre-optimization baseline (frozen at
``tests/fixtures/portfolio_legacy.py``) and the current ``code/portfolio.py``
implementation against the same synthetic country/daily inputs and assert that
all output DataFrames match. The harness is intentionally run BEFORE any
optimization lands: at that point legacy and current produce identical output,
so all tests pass and the harness itself is validated. Subsequent optimization
PRs re-run these tests and must not break parity.

Tolerance conventions (documented here; enforced by ``_assert_frames_parity``):
- Key columns (``characteristic``, ``pf``, ``eom``/``date``, ``gics``, ``ff49``,
  ``excntry``) are compared via exact equality.
- Stock counts (``n``) and ``pl.median``-derived ``signal`` are exact.
- ``ret_ew`` (plain arithmetic mean) is compared at TIGHT (rtol=1e-10).
- Weighted returns ``ret_vw`` and ``ret_vw_cap`` are compared at STANDARD
  (rtol=1e-6). Once the monthly/daily paths are vectorized, the sum reduction
  order for weighted averages changes and TIGHT is unrealistic; the legacy
  implementation is used here so TIGHT still holds, but the helper supports
  per-column tolerance overrides for future phases.
"""

from __future__ import annotations

import calendar
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Ensure the legacy fixture is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from jkp.data.portfolio import portfolios as current_portfolios  # noqa: E402
from tests.fixtures.portfolio_legacy import (  # noqa: E402
    portfolios as legacy_portfolios,
)

# =============================================================================
# Synthetic data helpers
# =============================================================================

# Characteristics are small and deterministic so parity checks are cheap and
# ECDF partitions are well-populated. Keep this list in sync with the chars
# column names written to the synthetic parquet.
SYNTHETIC_CHARS: list[str] = [
    "char_a",
    "char_b",
    "char_c",
    "char_d",
    "char_e",
    "char_f",
    "char_g",
    "char_h",
    "char_i",
    "char_j",
]

N_IDS = 120
N_MONTHS = 24


def _month_ends(n_months: int, start_year: int = 2018, start_month: int = 1) -> list[date]:
    """Return `n_months` consecutive month-end `datetime.date` values."""
    out: list[date] = []
    for i in range(n_months):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        last_day = calendar.monthrange(year, month)[1]
        out.append(date(year, month, last_day))
    return out


def _make_country_characteristics(
    excntry: str,
    chars: list[str],
    n_ids: int,
    n_months: int,
    seed: int,
) -> pl.DataFrame:
    """Build a synthetic monthly characteristics DataFrame for one country.

    Shape: n_ids * n_months rows, one row per (id, eom). Columns match what
    ``portfolios()`` selects from a real ``characteristics/{excntry}.parquet``.
    """
    rng = np.random.default_rng(seed)
    eoms = _month_ends(n_months)
    n_rows = n_ids * n_months

    # Static per-id attributes assigned once and broadcast across months as
    # plain Python lists. Nullable int columns (crsp_exchcd / comp_exchg) use
    # None for missing values — polars builds proper nullable Int64 columns
    # from that directly.
    size_grps_per_id = rng.choice(
        ["mega", "large", "small", "micro", "nano"],
        size=n_ids,
        p=[0.10, 0.25, 0.35, 0.20, 0.10],
    ).tolist()
    source_crsp_per_id: list[int] = rng.choice([0, 1], size=n_ids, p=[0.4, 0.6]).tolist()
    crsp_exchcd_choices = rng.choice([1, 2, 3], size=n_ids).tolist()
    comp_exchg_choices = rng.choice([11, 12, 13], size=n_ids).tolist()
    crsp_exchcd_per_id: list[int | None] = [
        int(crsp_exchcd_choices[i]) if source_crsp_per_id[i] == 1 else None for i in range(n_ids)
    ]
    comp_exchg_per_id: list[int | None] = [
        int(comp_exchg_choices[i]) if source_crsp_per_id[i] == 0 else None for i in range(n_ids)
    ]
    # Keep gics and ff49 cardinality low so every industry has enough stocks
    # to satisfy bp_min_n=10 in the synthetic n_ids=120 regime.
    gics_sectors = rng.choice([10, 15, 20, 25, 30, 35], size=n_ids).tolist()
    gics_per_id = [f"{int(s):02d}101010" for s in gics_sectors]
    ff49_per_id = rng.choice([1, 5, 10, 15, 20, 30, 40, 45], size=n_ids).tolist()

    # Broadcast across months as Python lists (one (id, eom) per row).
    id_col: list[int] = []
    eom_col: list[date] = []
    size_grp_col: list[str] = []
    source_crsp_col: list[int] = []
    crsp_exchcd_col: list[int | None] = []
    comp_exchg_col: list[int | None] = []
    gics_col: list[str] = []
    ff49_col: list[int] = []
    for eom in eoms:
        for i in range(n_ids):
            id_col.append(i + 1)
            eom_col.append(eom)
            size_grp_col.append(size_grps_per_id[i])
            source_crsp_col.append(int(source_crsp_per_id[i]))
            crsp_exchcd_col.append(crsp_exchcd_per_id[i])
            comp_exchg_col.append(comp_exchg_per_id[i])
            gics_col.append(gics_per_id[i])
            ff49_col.append(int(ff49_per_id[i]))

    # Monthly returns and market equity
    ret_exc = rng.normal(loc=0.008, scale=0.08, size=n_rows)
    ret_exc_lead1m = rng.normal(loc=0.008, scale=0.08, size=n_rows)
    me = np.exp(rng.normal(loc=7.0, scale=1.5, size=n_rows))

    # Characteristic values: each char has its own distribution and partial
    # null coverage. Keep nulls light so bp_min_n=10 is easy to satisfy.
    char_arrays: dict[str, np.ndarray] = {}
    for j, char in enumerate(chars):
        vals = rng.normal(loc=0.0, scale=1.0 + 0.1 * j, size=n_rows)
        null_rate = 0.05 + 0.01 * j
        mask = rng.random(n_rows) < null_rate
        vals[mask] = np.nan
        char_arrays[char] = vals

    df = pl.DataFrame(
        {
            "id": pl.Series("id", id_col, dtype=pl.Int64),
            "eom": pl.Series("eom", eom_col, dtype=pl.Date),
            "source_crsp": pl.Series("source_crsp", source_crsp_col, dtype=pl.Int64),
            "comp_exchg": pl.Series("comp_exchg", comp_exchg_col, dtype=pl.Int64),
            "crsp_exchcd": pl.Series("crsp_exchcd", crsp_exchcd_col, dtype=pl.Int64),
            "size_grp": pl.Series("size_grp", size_grp_col, dtype=pl.Utf8),
            "ret_exc": pl.Series("ret_exc", ret_exc, dtype=pl.Float64),
            "ret_exc_lead1m": pl.Series("ret_exc_lead1m", ret_exc_lead1m, dtype=pl.Float64),
            "me": pl.Series("me", me, dtype=pl.Float64),
            "gics": pl.Series("gics", gics_col, dtype=pl.Utf8),
            "ff49": pl.Series("ff49", ff49_col, dtype=pl.Int64),
            "excntry": pl.Series("excntry", [excntry] * n_rows, dtype=pl.Utf8),
            **{k: pl.Series(k, v, dtype=pl.Float64) for k, v in char_arrays.items()},
        }
    )
    return df


def _weekdays_in_month_after(eom: date, n: int = 21) -> list[date]:
    """Return up to `n` consecutive weekdays (Mon–Fri) starting the day after `eom`."""
    out: list[date] = []
    d = eom + timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:  # Mon=0 .. Fri=4
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_daily_returns(char_df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Build a synthetic daily returns DataFrame consistent with `char_df`.

    For every (id, eom) row in char_df, emit 21 weekdays in the month after
    eom, with ``ret_exc`` drawn from a Normal. ``date`` is a pl.Date.
    """
    rng = np.random.default_rng(seed)
    pairs = char_df.select(["id", "eom"]).unique().sort(["eom", "id"])
    id_list: list[int] = []
    date_list: list[date] = []
    ret_list: list[float] = []
    for row in pairs.iter_rows(named=True):
        id_ = int(row["id"])
        eom = row["eom"]
        for d in _weekdays_in_month_after(eom, n=21):
            id_list.append(id_)
            date_list.append(d)
            ret_list.append(float(rng.normal(loc=0.0003, scale=0.015)))

    return pl.DataFrame(
        {
            "id": pl.Series("id", id_list, dtype=pl.Int64),
            "date": pl.Series("date", date_list, dtype=pl.Date),
            "ret_exc": pl.Series("ret_exc", ret_list, dtype=pl.Float64),
        }
    )


def _next_month_end(d: date) -> date:
    """Return the month-end `date` of the month after `d`."""
    year = d.year + (1 if d.month == 12 else 0)
    month = 1 if d.month == 12 else d.month + 1
    return date(year, month, calendar.monthrange(year, month)[1])


def _make_cutoffs(eoms: list[date]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily matching the
    schemas used by ``portfolios()``.
    """
    n = len(eoms)

    nyse_size_cutoffs = pl.DataFrame(
        {
            "eom": pl.Series("eom", eoms, dtype=pl.Date),
            # A large nyse_p80 so me_cap == me in practice (tests still exercise
            # the join).
            "nyse_p80": pl.Series("nyse_p80", [1e12] * n, dtype=pl.Float64),
        }
    )

    # eom_lag1 is `eom.month_start - 1d`, matching main()'s derivation.
    eom_lag1 = [date(eom.year, eom.month, 1) - timedelta(days=1) for eom in eoms]
    ret_cutoffs = pl.DataFrame(
        {
            "eom": pl.Series("eom", eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.5] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.5] * n, dtype=pl.Float64),
            "eom_lag1": pl.Series("eom_lag1", eom_lag1, dtype=pl.Date),
        }
    )

    # Daily cutoffs are keyed by the daily date's month-end (the month AFTER
    # each char eom, since daily returns live in that following month).
    daily_eoms = [_next_month_end(eom) for eom in eoms]
    ret_cutoffs_daily = pl.DataFrame(
        {
            "eom": pl.Series("eom", daily_eoms, dtype=pl.Date),
            "ret_exc_0_1": pl.Series("ret_exc_0_1", [-0.2] * n, dtype=pl.Float64),
            "ret_exc_99_9": pl.Series("ret_exc_99_9", [0.2] * n, dtype=pl.Float64),
        }
    )

    return nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily


def _write_synthetic_country(
    data_root: Path,
    excntry: str,
    chars: list[str],
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Write a synthetic country's characteristics and daily returns parquets.

    Returns the monthly characteristics frame and the daily returns frame so
    tests can also inspect them without re-reading the parquets.
    """
    char_dir = data_root / "characteristics"
    daily_dir = data_root / "return_data" / "daily_rets_by_country"
    char_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    char_df = _make_country_characteristics(
        excntry=excntry, chars=chars, n_ids=N_IDS, n_months=N_MONTHS, seed=seed
    )
    char_df.write_parquet(char_dir / f"{excntry}.parquet")

    daily_df = _make_daily_returns(char_df, seed=seed + 1)
    daily_df.write_parquet(daily_dir / f"{excntry}.parquet")

    return char_df, daily_df


# =============================================================================
# Assertion helper
# =============================================================================


def _assert_frames_parity(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    key_cols: list[str],
    numeric_cols: dict[str, dict[str, float]],
    label: str,
) -> None:
    """Compare two Polars frames for numerical parity.

    Sorts both frames by `key_cols`, asserts equal height, then:
    - `key_cols` compared via exact equality (list of values).
    - `numeric_cols` compared with `np.testing.assert_allclose(**tol)`, NaN-aware.
    """
    assert actual.height == expected.height, (
        f"[{label}] height mismatch: {actual.height} vs {expected.height}"
    )
    assert actual.height > 0, f"[{label}] empty frame — nothing to compare"

    a = actual.sort(key_cols)
    e = expected.sort(key_cols)

    for col in key_cols:
        a_vals = a[col].to_list()
        e_vals = e[col].to_list()
        assert a_vals == e_vals, f"[{label}] key column {col!r} mismatch"

    for col, tol in numeric_cols.items():
        a_np = a[col].to_numpy().astype(np.float64)
        e_np = e[col].to_numpy().astype(np.float64)
        a_nan = np.isnan(a_np)
        e_nan = np.isnan(e_np)
        assert np.array_equal(a_nan, e_nan), f"[{label}] NaN positions differ in column {col!r}"
        mask = ~a_nan
        if mask.any():
            np.testing.assert_allclose(
                a_np[mask],
                e_np[mask],
                err_msg=f"[{label}] column {col!r} beyond tolerance",
                **tol,
            )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=["SYN", "USA"])
def synthetic_setup(tmp_path: Path, seed: int, request: pytest.FixtureRequest):
    """Per-test synthetic country setup.

    Parametrized on excntry so each test runs twice: once against a non-USA
    country (skips the FF49 branch) and once against USA (exercises FF49).
    Returns a dict the test can ``**kwargs`` into ``portfolios()``.
    """
    excntry = request.param
    data_root = tmp_path / "processed"
    data_root.mkdir(parents=True, exist_ok=True)

    char_df, _ = _write_synthetic_country(
        data_root=data_root, excntry=excntry, chars=SYNTHETIC_CHARS, seed=seed
    )
    eoms = char_df["eom"].unique().sort().to_list()
    nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)

    return {
        "data_path": str(data_root),
        "excntry": excntry,
        "chars": SYNTHETIC_CHARS,
        "pfs": 3,
        "bps": "non_mc",
        "bp_min_n": 10,
        "nyse_size_cutoffs": nyse_cut,
        "source": ["CRSP", "COMPUSTAT"],
        "wins_ret": True,
        "cmp_key": False,
        "signals": False,
        "signals_standardize": True,
        "signals_w": "vw_cap",
        "daily_pf": True,
        "ind_pf": True,
        "ret_cutoffs": ret_cut,
        "ret_cutoffs_daily": ret_cut_daily,
    }


# =============================================================================
# Parity tests
# =============================================================================

# Tolerance specs. See module docstring for the rationale.
_TIGHT = {"rtol": 1e-10, "atol": 1e-12}
_STANDARD = {"rtol": 1e-6, "atol": 1e-10}

_PF_RETURNS_NUMERIC = {
    "n": _TIGHT,
    "signal": _TIGHT,
    "ret_ew": _TIGHT,
    "ret_vw": _STANDARD,
    "ret_vw_cap": _STANDARD,
}
_PF_DAILY_NUMERIC = {
    "n": _TIGHT,
    "ret_ew": _TIGHT,
    "ret_vw": _STANDARD,
    "ret_vw_cap": _STANDARD,
}
_IND_RETURNS_NUMERIC = {
    "n": _TIGHT,
    "ret_ew": _TIGHT,
    "ret_vw": _STANDARD,
    "ret_vw_cap": _STANDARD,
}


class TestPortfoliosParity:
    """Compare frozen-legacy and current `portfolios()` output, key by key."""

    def test_pf_returns_parity(self, synthetic_setup):
        legacy = legacy_portfolios(**synthetic_setup)
        current = current_portfolios(**synthetic_setup)
        _assert_frames_parity(
            actual=current["pf_returns"],
            expected=legacy["pf_returns"],
            key_cols=["characteristic", "pf", "eom"],
            numeric_cols=_PF_RETURNS_NUMERIC,
            label="pf_returns",
        )

    def test_pf_daily_parity(self, synthetic_setup):
        legacy = legacy_portfolios(**synthetic_setup)
        current = current_portfolios(**synthetic_setup)
        _assert_frames_parity(
            actual=current["pf_daily"],
            expected=legacy["pf_daily"],
            key_cols=["characteristic", "pf", "date"],
            numeric_cols=_PF_DAILY_NUMERIC,
            label="pf_daily",
        )

    def test_gics_returns_parity(self, synthetic_setup):
        legacy = legacy_portfolios(**synthetic_setup)
        current = current_portfolios(**synthetic_setup)
        _assert_frames_parity(
            actual=current["gics_returns"],
            expected=legacy["gics_returns"],
            key_cols=["gics", "eom"],
            numeric_cols=_IND_RETURNS_NUMERIC,
            label="gics_returns",
        )

    def test_gics_daily_parity(self, synthetic_setup):
        legacy = legacy_portfolios(**synthetic_setup)
        current = current_portfolios(**synthetic_setup)
        _assert_frames_parity(
            actual=current["gics_daily"],
            expected=legacy["gics_daily"],
            key_cols=["gics", "date"],
            numeric_cols=_IND_RETURNS_NUMERIC,
            label="gics_daily",
        )

    @pytest.mark.parametrize("synthetic_setup", ["USA"], indirect=True)
    def test_ff49_returns_parity_usa_only(self, synthetic_setup):
        legacy = legacy_portfolios(**synthetic_setup)
        current = current_portfolios(**synthetic_setup)
        _assert_frames_parity(
            actual=current["ff49_returns"],
            expected=legacy["ff49_returns"],
            key_cols=["ff49", "eom"],
            numeric_cols=_IND_RETURNS_NUMERIC,
            label="ff49_returns",
        )

    @pytest.mark.parametrize("synthetic_setup", ["USA"], indirect=True)
    def test_ff49_daily_parity_usa_only(self, synthetic_setup):
        legacy = legacy_portfolios(**synthetic_setup)
        current = current_portfolios(**synthetic_setup)
        _assert_frames_parity(
            actual=current["ff49_daily"],
            expected=legacy["ff49_daily"],
            key_cols=["ff49", "date"],
            numeric_cols=_IND_RETURNS_NUMERIC,
            label="ff49_daily",
        )

    @pytest.mark.parametrize("synthetic_setup", ["SYN"], indirect=True)
    def test_ff49_not_in_output_for_non_usa(self, synthetic_setup):
        legacy = legacy_portfolios(**synthetic_setup)
        current = current_portfolios(**synthetic_setup)
        assert "ff49_returns" not in legacy
        assert "ff49_returns" not in current
        assert "ff49_daily" not in legacy
        assert "ff49_daily" not in current

    def test_cmp_key_true_parity_usa(self, tmp_path: Path, seed: int):
        """Parity for the cmp_key=True branch (runs for USA in production)."""
        data_root = tmp_path / "processed"
        data_root.mkdir(parents=True, exist_ok=True)
        char_df, _ = _write_synthetic_country(
            data_root=data_root, excntry="USA", chars=SYNTHETIC_CHARS, seed=seed
        )
        eoms = char_df["eom"].unique().sort().to_list()
        nyse_cut, ret_cut, ret_cut_daily = _make_cutoffs(eoms)
        kwargs = {
            "data_path": str(data_root),
            "excntry": "USA",
            "chars": SYNTHETIC_CHARS,
            "pfs": 3,
            "bps": "non_mc",
            "bp_min_n": 10,
            "nyse_size_cutoffs": nyse_cut,
            "source": ["CRSP", "COMPUSTAT"],
            "wins_ret": True,
            "cmp_key": True,
            "signals": False,
            "signals_standardize": True,
            "signals_w": "vw_cap",
            "daily_pf": True,
            "ind_pf": True,
            "ret_cutoffs": ret_cut,
            "ret_cutoffs_daily": ret_cut_daily,
        }
        legacy = legacy_portfolios(**kwargs)
        current = current_portfolios(**kwargs)
        assert "cmp" in legacy
        assert "cmp" in current
        _assert_frames_parity(
            actual=current["cmp"],
            expected=legacy["cmp"],
            key_cols=["characteristic", "size_grp", "eom"],
            numeric_cols={
                "n_stocks": _TIGHT,
                "ret_weighted": _STANDARD,
                "signal_weighted": _STANDARD,
            },
            label="cmp",
        )
