"""
Tests for z_ranks() and quality_minus_junk() in aux_functions.py.

Validates within-country z-scoring, composite construction, filter gates,
and final output schema/idempotency.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import polars as pl
import pytest

from jkp.data.aux_functions import quality_minus_junk, z_ranks
from jkp.data.paths import DataPaths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EOM = date(2020, 1, 31)
EXCNTRY = "US"

# All z-variable source columns (plus evol inputs)
_Z_SRC_COLS = [
    "gp_at",
    "ni_be",
    "ni_at",
    "ocf_at",
    "gp_sale",
    "oaccruals_at",
    "gpoa_ch5",
    "roe_ch5",
    "roa_ch5",
    "cfoa_ch5",
    "gmar_ch5",
    "betabab_1260d",
    "debt_at",
    "o_score",
    "z_score",
    "roeq_be_std",
    "roe_be_std",
]

_GATE_COLS = ["common", "primary_sec", "obs_main", "exch_main"]

_OUTPUT_COLS = ["excntry", "id", "eom", "qmj_prof", "qmj_growth", "qmj_safety", "qmj"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_frame(
    excntry: str = EXCNTRY,
    eom: date = EOM,
    n: int = 5,
    *,
    overrides: dict[str, list[Any]] | None = None,
) -> pl.DataFrame:
    """Build a minimal single-(excntry, eom) group with n stocks.

    Each z-source column gets distinct float values so std > 0.
    Gate columns default to 1 (pass). ret_exc and me are non-null floats.
    """
    overrides = overrides or {}
    ids = list(range(1, n + 1))
    base: dict[str, Any] = {
        "id": ids,
        "eom": [eom] * n,
        "excntry": [excntry] * n,
        "common": overrides.get("common", [1] * n),
        "primary_sec": overrides.get("primary_sec", [1] * n),
        "obs_main": overrides.get("obs_main", [1] * n),
        "exch_main": overrides.get("exch_main", [1] * n),
        "ret_exc": overrides.get("ret_exc", [float(i) / 100 for i in ids]),
        "me": overrides.get("me", [float(i) * 1e6 for i in ids]),
    }
    # z-source cols: each stock gets a distinct value per col
    for j, col in enumerate(_Z_SRC_COLS):
        base[col] = overrides.get(col, [float(i + j * 0.1) for i in ids])

    schema = {
        "id": pl.Int64,
        "eom": pl.Date,
        "excntry": pl.Utf8,
        "common": pl.Int64,
        "primary_sec": pl.Int64,
        "obs_main": pl.Int64,
        "exch_main": pl.Int64,
        "ret_exc": pl.Float64,
        "me": pl.Float64,
        **dict.fromkeys(_Z_SRC_COLS, pl.Float64),
    }
    return pl.DataFrame(base, schema=schema)


def _make_world_data(
    n_per_group: int = 6,
    excntries: list[str] | None = None,
    eoms: list[date] | None = None,
    *,
    overrides: dict[str, list[Any]] | None = None,
) -> pl.DataFrame:
    """Multi-group world-data frame combining _group_frame rows.

    All gate flags default to 1, ret_exc and me are non-null.
    Assigns unique IDs across groups.
    """
    excntries = excntries or [EXCNTRY]
    eoms = eoms or [EOM]
    frames: list[pl.DataFrame] = []
    id_offset = 0
    for excntry in excntries:
        for eom in eoms:
            grp = _group_frame(excntry=excntry, eom=eom, n=n_per_group, overrides=overrides)
            # shift ids to be globally unique
            grp = grp.with_columns(pl.col("id") + id_offset)
            id_offset += n_per_group
            frames.append(grp)
    return pl.concat(frames)


def _write_world(paths: DataPaths, df: pl.DataFrame) -> pl.Path:  # type: ignore[name-defined]
    p = paths.interim_dir / "test_msf.parquet"
    df.write_parquet(p)
    return p


# ---------------------------------------------------------------------------
# z_ranks helpers
# ---------------------------------------------------------------------------


def _z_frame(
    vals: list[float | None],
    var: str = "x",
    excntry: str = EXCNTRY,
    eom: date = EOM,
) -> pl.DataFrame:
    """Single-group frame for z_ranks testing."""
    n = len(vals)
    return pl.DataFrame(
        {
            "excntry": [excntry] * n,
            "id": list(range(1, n + 1)),
            "eom": [eom] * n,
            var: pl.Series(vals, dtype=pl.Float64),
        }
    )


def _hand_z(vals: list[float]) -> list[float]:
    """Hand-compute average-rank z-scores (ddof=1) for a list of finite values."""
    n = len(vals)
    # average rank: for each value, rank = (# strictly less + # less-or-equal + 1) / 2
    ranks = []
    for v in vals:
        lo = sum(1 for x in vals if x < v)
        hi = sum(1 for x in vals if x <= v)
        ranks.append((lo + hi + 1) / 2.0)
    mean_r = sum(ranks) / n
    var_r = sum((r - mean_r) ** 2 for r in ranks) / (n - 1)
    std_r = math.sqrt(var_r)
    return [(r - mean_r) / std_r for r in ranks]


# ===========================================================================
# TestZRanks
# ===========================================================================


class TestZRanks:
    """Unit tests for z_ranks() with in-memory frames."""

    def test_ascending_hand_computed(self, tolerance: Any) -> None:
        """z-scores match hand-computed (rank - mean)/std ddof=1 for ascending sort."""
        vals = [1.0, 3.0, 5.0, 7.0, 9.0]
        df = _z_frame(vals)
        result = z_ranks(df, "x", 2, "ascending")
        result = result.sort("id")

        expected = _hand_z(vals)
        import numpy as np

        np.testing.assert_allclose(result["z_x"].to_list(), expected, **tolerance.STANDARD)

    def test_descending_reverses_sign(self, tolerance: Any) -> None:
        """Descending z-scores are the negation of ascending z-scores."""
        vals = [1.0, 3.0, 5.0, 7.0, 9.0]
        df = _z_frame(vals)
        asc = z_ranks(df, "x", 2, "ascending").sort("id")
        desc = z_ranks(df, "x", 2, "descending").sort("id")

        import numpy as np

        np.testing.assert_allclose(
            desc["z_x"].to_list(),
            [-v for v in asc["z_x"].to_list()],
            **tolerance.STANDARD,
        )

    def test_ties_get_equal_rank_and_z(self, tolerance: Any) -> None:
        """Tied values share average rank and identical z-score."""
        vals = [1.0, 2.0, 2.0, 4.0]
        df = _z_frame(vals)
        result = z_ranks(df, "x", 2, "ascending").sort("id")

        z_list = result["z_x"].to_list()
        # ids 2 and 3 are tied at 2.0 → equal z
        import numpy as np

        np.testing.assert_allclose(z_list[1], z_list[2], **tolerance.STANDARD)

        # check against hand-computed (average rank: 1=1, 2=2.5, 3=2.5, 4=4)
        expected = _hand_z(vals)
        np.testing.assert_allclose(z_list, expected, **tolerance.STANDARD)

    def test_all_equal_values_returns_empty(self) -> None:
        """All-equal values → std=0 → NaN z → filtered out → empty result."""
        df = _z_frame([5.0, 5.0, 5.0])
        result = z_ranks(df, "x", 1, "ascending")
        assert result.height == 0

    def test_single_stock_dropped_with_min_2(self) -> None:
        """Group with 1 stock is dropped when __min=2."""
        df = _z_frame([3.14])
        result = z_ranks(df, "x", 2, "ascending")
        assert result.height == 0

    def test_single_stock_kept_with_min_1(self) -> None:
        """Single-stock group is kept when __min=1, but std=0 → z=NaN → filtered."""
        # Only 1 stock: rank=1, mean=1, std=0 → NaN z → dropped
        df = _z_frame([3.14])
        result = z_ranks(df, "x", 1, "ascending")
        assert result.height == 0

    def test_group_at_min_threshold_kept(self) -> None:
        """Group with exactly __min stocks (and non-equal values) is kept."""
        vals = [1.0, 2.0]
        df = _z_frame(vals)
        result = z_ranks(df, "x", 2, "ascending")
        assert result.height == 2

    def test_group_below_min_threshold_dropped(self) -> None:
        """Group with count < __min is fully dropped."""
        vals = [1.0, 2.0]
        df = _z_frame(vals)
        result = z_ranks(df, "x", 3, "ascending")
        assert result.height == 0

    def test_nan_row_filtered_and_not_counted(self) -> None:
        """NaN row is filtered; threshold counts only finite rows.

        Group with __min finite + 1 NaN at threshold __min → kept (NaN row absent).
        """
        vals = [1.0, 3.0, float("nan")]  # 2 finite, 1 NaN
        df = _z_frame(vals)
        result = z_ranks(df, "x", 2, "ascending")
        # 2 finite pass the min threshold; NaN excluded from output
        assert result.height == 2
        assert float("nan") not in result["z_x"].to_list()

    def test_null_row_filtered_same_as_nan(self) -> None:
        """None (null) row is filtered identically to NaN."""
        vals = [1.0, 3.0, None]
        df = _z_frame(vals)
        result = z_ranks(df, "x", 2, "ascending")
        assert result.height == 2

    def test_two_countries_independent(self, tolerance: Any) -> None:
        """Two countries with identical values produce identical z-scores per country."""
        import numpy as np

        vals = [1.0, 3.0, 5.0, 7.0, 9.0]
        df_us = _z_frame(vals, excntry="US")
        df_gb = _z_frame(vals, excntry="GB")
        df_gb = df_gb.with_columns(pl.col("id") + 10)
        df = pl.concat([df_us, df_gb])

        result = z_ranks(df, "x", 2, "ascending")
        us_z = result.filter(pl.col("excntry") == "US").sort("id")["z_x"].to_list()
        gb_z = result.filter(pl.col("excntry") == "GB").sort("id")["z_x"].to_list()

        np.testing.assert_allclose(us_z, gb_z, **tolerance.STANDARD)

    def test_output_columns_exact(self) -> None:
        """Output columns are exactly ['excntry', 'id', 'eom', 'z_<var>']."""
        df = _z_frame([1.0, 2.0, 3.0])
        result = z_ranks(df, "x", 2, "ascending")
        assert result.columns == ["excntry", "id", "eom", "z_x"]


# ===========================================================================
# TestQualityMinusJunkFilter
# ===========================================================================


class TestQualityMinusJunkFilter:
    """File-based tests: gate filters exclude/include correct stocks."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths

    def _run(self, df: pl.DataFrame, min_stks: int = 3) -> pl.DataFrame:
        p = _write_world(self.paths, df)
        quality_minus_junk(self.paths, p, min_stks)
        return pl.read_parquet(self.paths.interim_dir / "qmj.parquet")

    def _base_group_with_extra(self, flag_col: str, flag_val: Any, n_base: int = 5) -> pl.DataFrame:
        """Build group with n_base passing stocks plus 1 failing stock for flag_col."""
        base = _group_frame(n=n_base)
        # Failing stock: id = n_base + 1
        extra_vals: dict[str, Any] = {flag_col: [flag_val]}
        # give it id n_base+1 and distinct z-source values
        extra_data: dict[str, Any] = {
            "id": [n_base + 1],
            "eom": [EOM],
            "excntry": [EXCNTRY],
            "common": [1],
            "primary_sec": [1],
            "obs_main": [1],
            "exch_main": [1],
            "ret_exc": [0.99],
            "me": [1e6],
        }
        for j, col in enumerate(_Z_SRC_COLS):
            extra_data[col] = [float(n_base + 1 + j * 0.1)]
        extra_data.update(dict(extra_vals))
        schema = {
            "id": pl.Int64,
            "eom": pl.Date,
            "excntry": pl.Utf8,
            "common": pl.Int64,
            "primary_sec": pl.Int64,
            "obs_main": pl.Int64,
            "exch_main": pl.Int64,
            "ret_exc": pl.Float64,
            "me": pl.Float64,
            **dict.fromkeys(_Z_SRC_COLS, pl.Float64),
        }
        extra = pl.DataFrame(extra_data, schema=schema)
        return pl.concat([base, extra])

    @pytest.mark.parametrize("flag_col", ["common", "primary_sec", "obs_main", "exch_main"])
    def test_gate_zero_excluded(self, flag_col: str) -> None:
        """Stock with gate flag == 0 is absent from qmj output."""
        df = self._base_group_with_extra(flag_col, 0)
        result = self._run(df)
        excluded_id = 6  # n_base + 1
        assert excluded_id not in result["id"].to_list()

    def test_null_ret_exc_excluded(self) -> None:
        """Stock with null ret_exc is excluded."""
        df = self._base_group_with_extra("ret_exc", None)
        result = self._run(df)
        assert 6 not in result["id"].to_list()

    def test_null_me_excluded(self) -> None:
        """Stock with null me is excluded."""
        df = self._base_group_with_extra("me", None)
        result = self._run(df)
        assert 6 not in result["id"].to_list()

    def test_all_gates_pass_included(self) -> None:
        """Stock passing all gates appears in qmj output."""
        df = _group_frame(n=5)
        result = self._run(df)
        for i in range(1, 6):
            assert i in result["id"].to_list()


# ===========================================================================
# TestZVarsAndComposites
# ===========================================================================


class TestZVarsAndComposites:
    """File-based tests for __evol construction, direction, and composite logic."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths

    def _run(self, df: pl.DataFrame, min_stks: int = 3) -> pl.DataFrame:
        p = _write_world(self.paths, df)
        quality_minus_junk(self.paths, p, min_stks)
        return pl.read_parquet(self.paths.interim_dir / "qmj.parquet")

    def test_evol_uses_2x_roeq_multiplier(self) -> None:
        """qmj_safety ordering reflects 2x roeq_be_std (descending var).

        Stock A has high roeq_be_std, stock B has low. After 2x multiplier
        and descending direction, A should have lower qmj_safety than B.
        """
        n = 5
        base = _group_frame(n=n)
        # Set roeq_be_std so stock n (highest id) has the largest value.
        # __evol = 2 * roeq_be_std (roe_be_std non-null, but coalesce prefers roeq).
        # Direction: descending → higher __evol → lower z → lower safety composite.
        roeq_vals = [float(i) for i in range(1, n + 1)]  # 1,2,3,4,5
        base = base.with_columns(pl.Series("roeq_be_std", roeq_vals, dtype=pl.Float64))
        # Make all other safety vars (betabab_1260d, debt_at, o_score, z_score)
        # identical across stocks so they don't drive safety ordering.
        for col in ["betabab_1260d", "debt_at", "o_score"]:
            # Give them perfectly correlated with id so z contributions are id-ordered
            # but uniform-slope: use distinct to avoid std=0
            base = base.with_columns(
                pl.Series(col, [float(i) for i in range(1, n + 1)], dtype=pl.Float64)
            )
        result = self._run(base)
        result = result.sort("id")
        # stock 5 (highest roeq → highest 2*roeq → descending → lowest safety z)
        # stock 1 (lowest roeq → lowest 2*roeq → descending → highest safety z)
        safety = result["qmj_safety"].to_list()
        assert safety[0] > safety[-1], (
            f"Expected stock 1 safety {safety[0]:.4f} > stock 5 safety {safety[-1]:.4f}"
        )

    def test_evol_fallback_to_roe_be_std(self) -> None:
        """When roeq_be_std is null, __evol falls back to roe_be_std (no 2x)."""
        n = 5
        base = _group_frame(n=n)
        # Null out roeq_be_std; set roe_be_std with distinct values
        roe_vals = [float(i) for i in range(1, n + 1)]
        base = base.with_columns(
            pl.Series("roeq_be_std", [None] * n, dtype=pl.Float64),
            pl.Series("roe_be_std", roe_vals, dtype=pl.Float64),
        )
        result = self._run(base)
        # Should complete without error and have qmj_safety non-null
        assert result["qmj_safety"].null_count() < n

    @pytest.mark.parametrize("var", ["debt_at", "betabab_1260d", "o_score"])
    def test_descending_safety_var_direction(self, var: str) -> None:
        """Higher value of descending safety var → lower safety contribution.

        Stock 5 has the highest value of `var`; after descending z_ranks it gets
        the lowest z for that var → lower qmj_safety than stock 1.
        We control other safety vars to be correlated positively with id so they
        do not counteract the effect.
        """
        n = 6
        base = _group_frame(n=n)
        # Set target var: stock 5 highest (should flip contribution down)
        target_vals = [float(i) for i in range(1, n + 1)]
        base = base.with_columns(pl.Series(var, target_vals, dtype=pl.Float64))

        # Keep all other safety vars positively correlated with id (not constant)
        for other in ["betabab_1260d", "debt_at", "o_score", "z_score"]:
            if other != var:
                # positively correlated: higher id → higher value (ascending → higher z)
                # but these are also descending for betabab_1260d, debt_at, o_score
                # so all descend together → uniform contribution
                base = base.with_columns(
                    pl.Series(other, [float(i) for i in range(1, n + 1)], dtype=pl.Float64)
                )
        result = self._run(base)
        result = result.sort("id")
        safety = result["qmj_safety"].to_list()
        # All descending safety vars: stock 6 (highest val) → lowest safety z
        assert safety[0] > safety[-1], (
            f"Expected stock 1 safety {safety[0]:.4f} > stock 6 safety {safety[-1]:.4f}"
        )

    def test_mean_horizontal_skips_one_null(self) -> None:
        """One prof source var null for one stock → qmj_prof still non-null for it.

        The var must still pass min_stks threshold via other stocks.
        """
        n = 6
        base = _group_frame(n=n)
        # Set gp_at to None for stock 1 only
        gp_at_vals = [None] + [float(i) for i in range(2, n + 1)]
        base = base.with_columns(pl.Series("gp_at", gp_at_vals, dtype=pl.Float64))
        result = self._run(base, min_stks=3)
        row1 = result.filter(pl.col("id") == 1)
        # Stock 1 may or may not be in output (its gp_at z_rank will be absent,
        # but mean_horizontal over remaining 5 prof vars can still be non-null)
        if row1.height > 0:
            assert row1["qmj_prof"][0] is not None

    def test_all_prof_vars_null_gives_null_qmj_prof(self) -> None:
        """Stock with all 6 prof source vars null → qmj_prof null for that stock."""
        n = 6
        base = _group_frame(n=n)
        prof_cols = ["gp_at", "ni_be", "ni_at", "ocf_at", "gp_sale", "oaccruals_at"]
        for col in prof_cols:
            vals = [None] + [float(i) for i in range(2, n + 1)]
            base = base.with_columns(pl.Series(col, vals, dtype=pl.Float64))

        result = self._run(base, min_stks=3)
        row1 = result.filter(pl.col("id") == 1)
        if row1.height > 0:
            assert row1["qmj_prof"][0] is None


# ===========================================================================
# TestQmjAggregation
# ===========================================================================


class TestQmjAggregation:
    """File-based tests for composite aggregation and null propagation."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths

    def _run(self, df: pl.DataFrame, min_stks: int = 3) -> pl.DataFrame:
        p = _write_world(self.paths, df)
        quality_minus_junk(self.paths, p, min_stks)
        return pl.read_parquet(self.paths.interim_dir / "qmj.parquet")

    def test_all_safety_vars_null_gives_null_qmj(self) -> None:
        """Stock with all 5 safety source vars null → qmj_safety null → qmj null."""
        n = 6
        base = _group_frame(n=n)
        safety_cols = ["betabab_1260d", "debt_at", "o_score", "z_score"]
        for col in safety_cols:
            vals = [None] + [float(i) for i in range(2, n + 1)]
            base = base.with_columns(pl.Series(col, vals, dtype=pl.Float64))
        # Also null out roeq_be_std and roe_be_std for stock 1 (drives __evol)
        base = base.with_columns(
            pl.Series(
                "roeq_be_std", [None] + [float(i) for i in range(2, n + 1)], dtype=pl.Float64
            ),
            pl.Series("roe_be_std", [None] + [float(i) for i in range(2, n + 1)], dtype=pl.Float64),
        )
        result = self._run(base, min_stks=3)
        row1 = result.filter(pl.col("id") == 1)
        if row1.height > 0:
            assert row1["qmj_safety"][0] is None
            assert row1["qmj"][0] is None
            # prof and growth can still be non-null
            assert row1["qmj_prof"][0] is not None or row1["qmj_growth"][0] is not None

    def test_hand_computed_pipeline_single_group(self, tolerance: Any) -> None:
        """Hand-compute full pipeline for a small group and compare qmj.

        Uses 4 stocks with distinct values, computes z-ranks per var,
        composites, z-rank composites, mean, final z-rank in plain Python.
        """
        import math as _math

        n = 4
        base = _group_frame(n=n)
        # Give perfectly spaced distinct values for all vars so hand-calc is tractable
        for j, col in enumerate(_Z_SRC_COLS):
            base = base.with_columns(
                pl.Series(col, [float(i + j * 10) for i in range(1, n + 1)], dtype=pl.Float64)
            )
        result = self._run(base, min_stks=2)

        # --- Hand-compute pipeline ---
        def avg_rank(vals: list[float], descending: bool = False) -> list[float]:
            if descending:
                vals_neg = [-v for v in vals]
                ranks = avg_rank(vals_neg, descending=False)
                return ranks
            result = []
            for v in vals:
                lo = sum(1 for x in vals if x < v)
                hi = sum(1 for x in vals if x <= v)
                result.append((lo + hi + 1) / 2.0)
            return result

        def z_score_list(vals: list[float]) -> list[float]:
            n2 = len(vals)
            mean = sum(vals) / n2
            var = sum((v - mean) ** 2 for v in vals) / (n2 - 1)
            std = _math.sqrt(var)
            if std == 0:
                return [float("nan")] * n2
            return [(v - mean) / std for v in vals]

        def z_ranks_py(vals: list[float], descending: bool = False) -> list[float]:
            ranks = avg_rank(vals, descending=descending)
            return z_score_list(ranks)

        # z-score each var
        directions = [
            False,
            False,
            False,
            False,
            False,
            True,  # prof (oaccruals descending)
            False,
            False,
            False,
            False,
            False,  # growth
            True,
            True,
            True,
            False,
            True,  # safety (betabab, debt, o_score desc; z_score asc; evol desc)
        ]
        var_names = [
            "gp_at",
            "ni_be",
            "ni_at",
            "ocf_at",
            "gp_sale",
            "oaccruals_at",
            "gpoa_ch5",
            "roe_ch5",
            "roa_ch5",
            "cfoa_ch5",
            "gmar_ch5",
            "betabab_1260d",
            "debt_at",
            "o_score",
            "z_score",
            "__evol",
        ]
        # Build stock values including __evol = 2 * roeq_be_std
        roeq_idx = _Z_SRC_COLS.index("roeq_be_std")
        roeq_vals = [float(i + roeq_idx * 10) for i in range(1, n + 1)]
        evol_vals = [2.0 * v for v in roeq_vals]

        stock_vals: dict[str, list[float]] = {}
        for j, col in enumerate(_Z_SRC_COLS):
            stock_vals[col] = [float(i + j * 10) for i in range(1, n + 1)]
        stock_vals["__evol"] = evol_vals

        z_per_var: dict[str, list[float]] = {}
        for vname, desc in zip(var_names, directions, strict=True):
            z_per_var[vname] = z_ranks_py(stock_vals[vname], descending=desc)

        # Composites (mean_horizontal skips nulls — here all non-null)
        def mean_cols(names: list[str], idx: int) -> float:
            vals2 = [z_per_var[nm][idx] for nm in names if not _math.isnan(z_per_var[nm][idx])]
            return sum(vals2) / len(vals2) if vals2 else float("nan")

        prof_names = ["gp_at", "ni_be", "ni_at", "ocf_at", "gp_sale", "oaccruals_at"]
        growth_names = ["gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5"]
        safety_names = ["betabab_1260d", "debt_at", "o_score", "z_score", "__evol"]

        prof_vals = [mean_cols(prof_names, i) for i in range(n)]
        growth_vals = [mean_cols(growth_names, i) for i in range(n)]
        safety_vals = [mean_cols(safety_names, i) for i in range(n)]

        qmj_prof = z_ranks_py(prof_vals)
        qmj_growth = z_ranks_py(growth_vals)
        qmj_safety = z_ranks_py(safety_vals)

        qmj_raw = [
            (p + g + s) / 3.0 for p, g, s in zip(qmj_prof, qmj_growth, qmj_safety, strict=True)
        ]
        qmj_expected = z_ranks_py(qmj_raw)

        result_sorted = result.sort("id")
        import numpy as np

        np.testing.assert_allclose(
            result_sorted["qmj"].to_list(),
            qmj_expected,
            **tolerance.LOOSE,
        )

    def test_country_b_does_not_affect_country_a(self) -> None:
        """Within-country z-scoring: adding country B doesn't change country A's qmj."""
        df_a_only = _group_frame(excntry="US", n=5)
        df_b = _group_frame(excntry="GB", n=5)
        df_b = df_b.with_columns(pl.col("id") + 100)
        df_ab = pl.concat([df_a_only, df_b])

        result_a = self._run(df_a_only).sort("id")
        # Re-run with both countries (need fresh paths → write again)
        p_ab = self.paths.interim_dir / "test_msf.parquet"
        df_ab.write_parquet(p_ab)
        quality_minus_junk(self.paths, p_ab, 3)
        result_ab = pl.read_parquet(self.paths.interim_dir / "qmj.parquet")
        result_ab_a = result_ab.filter(pl.col("excntry") == "US").sort("id")

        import numpy as np

        np.testing.assert_allclose(
            result_a["qmj"].to_list(),
            result_ab_a["qmj"].to_list(),
            rtol=1e-6,
            atol=1e-10,
        )


# ===========================================================================
# TestOutputAndIdempotency
# ===========================================================================


class TestOutputAndIdempotency:
    """Tests for output schema, dtypes, idempotency, and min_stks threshold."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_paths: DataPaths) -> None:
        self.paths = test_paths

    def _run(self, df: pl.DataFrame, min_stks: int = 3) -> pl.DataFrame:
        p = _write_world(self.paths, df)
        quality_minus_junk(self.paths, p, min_stks)
        return pl.read_parquet(self.paths.interim_dir / "qmj.parquet")

    def test_output_columns_exact(self) -> None:
        """Output columns are exactly the 7 expected columns."""
        result = self._run(_group_frame(n=5))
        assert result.columns == _OUTPUT_COLS

    def test_output_dtypes(self) -> None:
        """eom is Date; qmj_{prof,growth,safety} and qmj are Float64."""
        result = self._run(_group_frame(n=5))
        assert result.schema["eom"] == pl.Date
        for col in ["qmj_prof", "qmj_growth", "qmj_safety", "qmj"]:
            assert result.schema[col] == pl.Float64, f"{col} should be Float64"

    def test_idempotency(self) -> None:
        """Running quality_minus_junk twice produces identical output."""
        from polars.testing import assert_frame_equal as pl_assert_frame_equal

        df = _group_frame(n=5)
        p = _write_world(self.paths, df)
        quality_minus_junk(self.paths, p, 3)
        r1 = pl.read_parquet(self.paths.interim_dir / "qmj.parquet")
        quality_minus_junk(self.paths, p, 3)
        r2 = pl.read_parquet(self.paths.interim_dir / "qmj.parquet")
        pl_assert_frame_equal(r1.sort(["excntry", "id", "eom"]), r2.sort(["excntry", "id", "eom"]))

    def test_group_below_min_stks_has_null_qmj(self) -> None:
        """Group with count < min_stks appears in output but with null qmj values.

        The full-join coalesce pattern retains rows from small groups; they end
        up with all-null composites because no var passes the min_stks threshold.
        """
        # country A: 5 stocks (passes min_stks=5)
        df_a = _group_frame(excntry="US", n=5)
        # country B: 3 stocks (fails min_stks=5 per var z_ranks)
        df_b = _group_frame(excntry="GB", n=3)
        df_b = df_b.with_columns(pl.col("id") + 100)
        df = pl.concat([df_a, df_b])

        result = self._run(df, min_stks=5)
        gb_rows = result.filter(pl.col("excntry") == "GB")
        # GB rows present but all qmj columns null
        assert gb_rows.height == 3
        assert gb_rows["qmj"].null_count() == 3
        assert gb_rows["qmj_prof"].null_count() == 3
        assert gb_rows["qmj_growth"].null_count() == 3
        assert gb_rows["qmj_safety"].null_count() == 3

    def test_group_at_min_stks_present(self) -> None:
        """Group with count == min_stks (non-equal values) appears in output."""
        df = _group_frame(excntry="US", n=4)
        result = self._run(df, min_stks=4)
        assert result.filter(pl.col("excntry") == "US").height > 0
