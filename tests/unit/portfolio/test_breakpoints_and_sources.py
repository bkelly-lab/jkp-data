"""Unit tests for ``portfolios()`` bps modes, source filter, and me_cap capping."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jkp.data.portfolio import portfolios
from tests.unit.portfolio.conftest import (
    make_breakpoint_divergent_dataset,
    make_country_characteristics,
    make_cutoffs,
)

# Common kwargs used by every test in this file.
COMMON_KWARGS = {
    "pfs": 3,
    "bp_min_n": 5,
    "cmp_key": False,
    "signals": False,
    "signals_standardize": False,
    "signals_w": "vw_cap",
    "daily_pf": False,
    "ind_pf": False,
    "wins_ret": True,
}


def _write_chars(data_path: Path, excntry: str, df: pl.DataFrame) -> None:
    char_dir = data_path / "characteristics"
    char_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(char_dir / f"{excntry}.parquet")


# =============================================================================
# TestBpsModes
# =============================================================================


class TestBpsModes:
    """Tests for the ``bps`` argument (``"non_mc"`` vs ``"nyse"``)."""

    def test_bps_non_mc_excludes_microcap_from_breakpoint_sample(self, tmp_path: Path) -> None:
        """With ``bps="non_mc"``, only non-microcap rows act as breakpoint stocks.

        ``bp_stock`` is an internal mask (not exposed in ``pf_returns``), so
        we verify the behavior in two ways:
          1) Replicate the ``bp_stock`` expression on the input and confirm
             it equals ``size_grp.is_in(["mega", "large", "small"])``
             (i.e., excludes micro and nano).
          2) Confirm ``portfolios()`` runs and produces non-empty output
             when this mode is used. With ``bp_min_n=5`` and 40 large rows
             per eom, the bp_n filter passes.
        """
        excntry = "USA"
        df = make_breakpoint_divergent_dataset(seed=0)
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        # Replicate the non_mc bp_stock expression from portfolio.py:293.
        bp_stock = df.select(
            pl.col("size_grp").is_in(["mega", "large", "small"]).alias("bp_stock"),
            pl.col("size_grp"),
        )
        # Micro rows must NOT be bp_stocks; large rows MUST be bp_stocks.
        assert bp_stock.filter(pl.col("size_grp") == "micro")["bp_stock"].sum() == 0
        n_large = bp_stock.filter(pl.col("size_grp") == "large").height
        assert bp_stock.filter(pl.col("size_grp") == "large")["bp_stock"].sum() == n_large

        out = portfolios(
            data_path=str(tmp_path),
            excntry=excntry,
            chars=["char_a"],
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            **COMMON_KWARGS,
        )
        assert out["pf_returns"].height > 0

    def test_bps_nyse_uses_nyse_flagged_rows(self, tmp_path: Path) -> None:
        """With ``bps="nyse"``, only NYSE-flagged rows act as breakpoint stocks.

        ``bp_stock`` is internal, so we replicate the NYSE bp_stock
        expression and confirm it picks out the rows where either
        ``crsp_exchcd == 1`` (and ``comp_exchg`` null) or
        ``comp_exchg == 11`` (and ``crsp_exchcd`` null). In the divergent
        dataset, only the micros (with ``crsp_exchcd=1``, ``comp_exchg``
        null) match.
        """
        excntry = "USA"
        df = make_breakpoint_divergent_dataset(seed=0)
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        # Replicate the nyse bp_stock expression from portfolio.py:288-291.
        bp_stock = df.select(
            (
                ((pl.col("crsp_exchcd") == 1) & pl.col("comp_exchg").is_null())
                | ((pl.col("comp_exchg") == 11) & pl.col("crsp_exchcd").is_null())
            ).alias("bp_stock"),
            pl.col("size_grp"),
        )
        # Only micros (with crsp_exchcd=1, comp_exchg null) are NYSE flagged.
        n_micro = bp_stock.filter(pl.col("size_grp") == "micro").height
        assert bp_stock.filter(pl.col("size_grp") == "micro")["bp_stock"].sum() == n_micro
        # Large rows have comp_exchg=12 (not NYSE) → not flagged.
        assert bp_stock.filter(pl.col("size_grp") == "large")["bp_stock"].sum() == 0

        out = portfolios(
            data_path=str(tmp_path),
            excntry=excntry,
            chars=["char_a"],
            bps="nyse",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            **COMMON_KWARGS,
        )
        assert out["pf_returns"].height > 0

    def test_bps_modes_produce_different_pf_assignments(self, tmp_path: Path) -> None:
        """Same input, different ``bps`` mode → at least one (pf, eom) row differs.

        The ECDF is computed only over ``bp_stock=True`` rows, so changing
        which rows are bp_stocks changes the cdf cutoffs and thus the pf
        assignments for non-bp-stock rows. Compare the per-(eom) distribution
        of n across pfs between the two outputs.
        """
        excntry = "USA"
        df = make_breakpoint_divergent_dataset(seed=0)
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        common = dict(
            data_path=str(tmp_path),
            excntry=excntry,
            chars=["char_a"],
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            **COMMON_KWARGS,
        )
        out_nonmc = portfolios(bps="non_mc", **common)["pf_returns"]
        out_nyse = portfolios(bps="nyse", **common)["pf_returns"]

        a = out_nonmc.sort(["eom", "pf"]).select(["eom", "pf", "n"])
        b = out_nyse.sort(["eom", "pf"]).select(["eom", "pf", "n"])
        # The two frames may have different shapes (different (pf, eom) keys
        # populated). If the shapes differ we already have divergence.
        if a.height != b.height:
            return
        # Same shape → at least one n value must differ.
        diffs = (a["n"].to_numpy() != b["n"].to_numpy()).any() or (
            a["pf"].to_list() != b["pf"].to_list()
        )
        assert diffs, "bps modes produced identical pf assignments"


# =============================================================================
# TestSourceFilter
# =============================================================================


class TestSourceFilter:
    """Tests for the ``source`` filter."""

    @pytest.mark.parametrize(
        "source",
        [["CRSP"], ["COMPUSTAT"], ["CRSP", "COMPUSTAT"]],
        ids=["crsp_only", "compustat_only", "both"],
    )
    def test_source_filter_retains_only_specified_source(
        self, tmp_path: Path, source: list[str]
    ) -> None:
        """Output ``n`` totals match the number of rows with the right source_crsp.

        For each source setting we compute the expected count of rows that
        survive the (size_grp, me, ret_exc_lead1m, char non-null, source)
        filters from the raw input, then compare to the sum of ``n`` in
        ``pf_returns`` (for one characteristic, summed across pfs).
        """
        excntry = "USA"
        chars = ["char_a"]
        df = make_country_characteristics(
            excntry=excntry, chars=chars, n_ids=80, n_months=4, seed=7
        )
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        nyse_size_cutoffs, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)

        out = portfolios(
            data_path=str(tmp_path),
            excntry=excntry,
            chars=chars,
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=source,
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            **COMMON_KWARGS,
        )
        pf_returns = out["pf_returns"]
        assert pf_returns.height > 0

        # Compute expected n: rows surviving size_grp/me/ret_exc_lead1m
        # non-null filters AND the source filter AND char_a non-null AND
        # bp_n>=bp_min_n per eom. The bp_n filter applies to every row whose
        # eom has enough bp stocks; with ~80 rows per eom and the size_grp
        # distribution, bp_n is comfortably >= 5.
        expected = df.filter(
            pl.col("size_grp").is_not_null()
            & pl.col("me").is_not_null()
            & pl.col("ret_exc_lead1m").is_not_null()
            & pl.col("char_a").is_not_null()
        )
        if source == ["CRSP"]:
            expected = expected.filter(pl.col("source_crsp") == 1)
        elif source == ["COMPUSTAT"]:
            expected = expected.filter(pl.col("source_crsp") == 0)
        # For both, no source filter applied.
        expected_n = expected.height
        actual_n = int(pf_returns["n"].sum())
        assert actual_n == expected_n, (
            f"source={source}: expected {expected_n} rows in pf_returns, got {actual_n}"
        )


# =============================================================================
# TestMeCap
# =============================================================================


class TestMeCap:
    """Tests for the ``me_cap = min(me, nyse_p80)`` capping logic."""

    def test_me_cap_equals_min_me_nyse_p80(self, tmp_path: Path) -> None:
        """With a tiny ``nyse_p80``, capping bites and ``ret_vw_cap`` ≠ ``ret_vw``.

        ``me_cap`` is computed inside the lazy chain and is not surfaced in
        the output, so we verify the capping behavior indirectly: when
        ``nyse_p80`` is small relative to ``me``, ``me_cap`` collapses toward
        a constant for many rows, so ``ret_vw_cap`` (which uses ``me_cap``)
        diverges from ``ret_vw`` (which uses raw ``me``).
        """
        excntry = "USA"
        chars = ["char_a"]
        df = make_country_characteristics(
            excntry=excntry, chars=chars, n_ids=80, n_months=4, seed=11
        )
        _write_chars(tmp_path, excntry, df)
        eoms = df["eom"].unique().sort().to_list()
        _, ret_cutoffs, ret_cutoffs_daily = make_cutoffs(eoms)
        # Override nyse_size_cutoffs with a tiny p80 so capping bites.
        small_p80 = 1.0  # well below typical me values (≈ exp(7) ≈ 1100)
        nyse_size_cutoffs = pl.DataFrame(
            {
                "eom": pl.Series("eom", eoms, dtype=pl.Date),
                "nyse_p80": pl.Series("nyse_p80", [small_p80] * len(eoms), dtype=pl.Float64),
            }
        )

        out = portfolios(
            data_path=str(tmp_path),
            excntry=excntry,
            chars=chars,
            bps="non_mc",
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=["CRSP", "COMPUSTAT"],
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=ret_cutoffs_daily,
            **COMMON_KWARGS,
        )
        pf_returns = out["pf_returns"]
        assert pf_returns.height > 0
        # When me_cap = min(me, 1.0) and me is typically >> 1.0, me_cap is
        # ~constant across rows so ret_vw_cap approaches ret_ew. ret_vw
        # uses raw me and is value-weighted — these should differ.
        ret_vw = pf_returns["ret_vw"].to_numpy()
        ret_vw_cap = pf_returns["ret_vw_cap"].to_numpy()
        # At least one row must differ; in practice essentially all do.
        assert (ret_vw != ret_vw_cap).any(), (
            "ret_vw_cap and ret_vw are identical even though nyse_p80 should bite"
        )
