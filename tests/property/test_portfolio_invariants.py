"""Property-based invariant tests for ``portfolios()``.

Each test generates a small synthetic single-country dataset via
Hypothesis, runs ``portfolios()`` with a fixed flag profile, and asserts
a structural or numerical invariant on the resulting ``pf_returns``
frame.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jkp.data.portfolio import portfolios
from tests.unit.portfolio.conftest import (
    SYNTHETIC_CHARS,
    make_country_characteristics,
    make_cutoffs,
)

INPUT_CONFIG = st.builds(
    lambda seed, n_chars, n_eoms, n_ids: {
        "seed": seed,
        "n_chars": n_chars,
        "n_eoms": n_eoms,
        "n_ids": n_ids,
    },
    seed=st.integers(min_value=0, max_value=1000),
    n_chars=st.integers(min_value=2, max_value=4),
    n_eoms=st.integers(min_value=4, max_value=8),
    n_ids=st.integers(min_value=40, max_value=100),
)

PFS = 5
EXCNTRY = "SYN"


def _build_inputs(
    tmp_path: Path, config: dict
) -> tuple[str, list[str], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Materialize a synthetic one-country input tree under ``tmp_path``."""
    chars = SYNTHETIC_CHARS[: config["n_chars"]]
    data_root = tmp_path / "processed"
    char_dir = data_root / "characteristics"
    char_dir.mkdir(parents=True, exist_ok=True)

    char_df = make_country_characteristics(
        excntry=EXCNTRY,
        chars=chars,
        n_ids=config["n_ids"],
        n_months=config["n_eoms"],
        seed=config["seed"],
    )
    char_df.write_parquet(char_dir / f"{EXCNTRY}.parquet")
    eoms = char_df["eom"].unique().sort().to_list()
    nyse_cut, ret_cut, ret_cut_daily = make_cutoffs(eoms)
    return str(data_root), chars, nyse_cut, ret_cut, char_df


def _run_portfolios(
    data_path: str,
    chars: list[str],
    nyse_cut: pl.DataFrame,
    ret_cut: pl.DataFrame,
) -> dict:
    """Invoke ``portfolios()`` with the fixed flag profile used across tests."""
    return portfolios(
        data_path=data_path,
        excntry=EXCNTRY,
        chars=chars,
        pfs=PFS,
        bps="non_mc",
        bp_min_n=1,
        nyse_size_cutoffs=nyse_cut,
        source=["CRSP", "COMPUSTAT"],
        wins_ret=True,
        cmp_key=False,
        signals=False,
        daily_pf=False,
        ind_pf=False,
        ret_cutoffs=ret_cut,
        ret_cutoffs_daily=None,
    )


class TestPortfolioInvariants:
    """Property-based invariants on ``portfolios()`` ``pf_returns`` output."""

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_pf_values_in_one_to_pfs(self, config: dict, tmp_path: Path) -> None:
        """Every ``pf`` label must lie in ``[1, PFS]``."""
        data_path, chars, nyse_cut, ret_cut, _ = _build_inputs(tmp_path, config)
        out = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        pfs_col = out["pf_returns"]["pf"]
        assert pfs_col.min() >= 1
        assert pfs_col.max() <= PFS

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_pf_returns_schema_has_required_columns(self, config: dict, tmp_path: Path) -> None:
        """``pf_returns`` must expose the expected schema columns."""
        data_path, chars, nyse_cut, ret_cut, _ = _build_inputs(tmp_path, config)
        out = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        required = {
            "excntry",
            "characteristic",
            "pf",
            "eom",
            "n",
            "signal",
            "ret_ew",
            "ret_vw",
            "ret_vw_cap",
        }
        assert required.issubset(set(out["pf_returns"].columns))

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_n_per_pf_eom_sums_to_total_observations(self, config: dict, tmp_path: Path) -> None:
        """Sum of ``n`` per (characteristic, eom) <= non-null obs in raw data.

        The eom in pf_returns is shifted forward one month relative to the
        characteristic data, so we lag it back before aggregation.
        """
        data_path, chars, nyse_cut, ret_cut, char_df = _build_inputs(tmp_path, config)
        out = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        pf = out["pf_returns"].with_columns(
            pl.col("eom").dt.offset_by("-1mo").dt.month_end().alias("eom_orig")
        )
        sums = pf.group_by(["characteristic", "eom_orig"]).agg(pl.col("n").sum())

        # Build per-(char, eom) non-null counts on the raw frame after the
        # same baseline filters portfolios() applies.
        baseline = char_df.filter(
            pl.col("size_grp").is_not_null()
            & pl.col("me").is_not_null()
            & pl.col("ret_exc_lead1m").is_not_null()
        )
        for row in sums.iter_rows(named=True):
            ch, eom, n_sum = row["characteristic"], row["eom_orig"], row["n"]
            non_null = baseline.filter((pl.col("eom") == eom) & pl.col(ch).is_not_null()).height
            assert n_sum <= non_null, f"sum(n)={n_sum} > non_null obs={non_null} for {ch} @ {eom}"

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_pf_monotone_signal_medians(self, config: dict, tmp_path: Path) -> None:
        """Within (characteristic, eom), median signal is non-decreasing in pf."""
        data_path, chars, nyse_cut, ret_cut, _ = _build_inputs(tmp_path, config)
        out = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        pf = out["pf_returns"].sort(["characteristic", "eom", "pf"])
        for (_ch, _eom), sub in pf.group_by(["characteristic", "eom"], maintain_order=True):
            # Drop NaN-signal rows: pf buckets where every observation had a
            # NaN ``var`` (e.g. the input characteristic encodes nulls as NaN).
            signals = [s for s in sub["signal"].to_list() if s == s]
            for a, b in zip(signals, signals[1:], strict=False):
                assert a <= b + 1e-12, f"non-monotone signal in pf order: {signals}"

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_ret_ew_within_input_bounds(self, config: dict, tmp_path: Path) -> None:
        """Approximate EW invariant: ``ret_ew`` within winsorized return bounds.

        Exact verification is impractical without re-deriving the pf
        assignments, so we assert the equal-weighted return falls inside
        the global min/max of ``ret_exc_lead1m`` (post-winsorization, the
        cutoffs are -0.5 and +0.5).
        """
        data_path, chars, nyse_cut, ret_cut, char_df = _build_inputs(tmp_path, config)
        out = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        # Post-winsorization, Compustat returns are clipped to [-0.5, 0.5];
        # CRSP rows use raw values. Take a generous bound from raw range.
        rets = char_df["ret_exc_lead1m"].drop_nans().drop_nulls()
        lo = min(float(rets.min()), -0.5)
        hi = max(float(rets.max()), 0.5)
        ret_ew = out["pf_returns"]["ret_ew"]
        assert ret_ew.min() >= lo - 1e-9
        assert ret_ew.max() <= hi + 1e-9

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_no_nan_in_returns(self, config: dict, tmp_path: Path) -> None:
        """``ret_ew``, ``ret_vw``, ``ret_vw_cap`` columns must have no NaN/null."""
        data_path, chars, nyse_cut, ret_cut, _ = _build_inputs(tmp_path, config)
        out = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        pf = out["pf_returns"]
        for col in ("ret_ew", "ret_vw", "ret_vw_cap"):
            s = pf[col]
            assert s.null_count() == 0, f"{col} has nulls"
            assert int(s.is_nan().sum()) == 0, f"{col} has NaN"

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_pf_returns_height_positive(self, config: dict, tmp_path: Path) -> None:
        """Output ``pf_returns`` frame must be non-empty."""
        data_path, chars, nyse_cut, ret_cut, _ = _build_inputs(tmp_path, config)
        out = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        assert out["pf_returns"].height > 0

    @given(config=INPUT_CONFIG)
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_idempotent_under_seed_reuse(self, config: dict, tmp_path: Path) -> None:
        """Calling ``portfolios()`` twice with the same inputs is idempotent."""
        data_path, chars, nyse_cut, ret_cut, _ = _build_inputs(tmp_path, config)
        out1 = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        out2 = _run_portfolios(data_path, chars, nyse_cut, ret_cut)
        sort_keys = ["characteristic", "eom", "pf"]
        a = out1["pf_returns"].sort(sort_keys)
        b = out2["pf_returns"].sort(sort_keys)
        assert a.equals(b)
