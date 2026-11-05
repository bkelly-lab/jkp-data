from datetime import date

import polars as pl
import pytest

from jkp.data.aux_functions import _sign_factors
from jkp.data.portfolio import _build_hml


def _make_pf_df() -> pl.DataFrame:
    """Build a small pf_df: 1 country, 2 chars, 3 eoms, 3 pfs each."""
    rows = []
    eoms = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
    chars = ["char_a", "char_b"]
    for ch in chars:
        for i, eom in enumerate(eoms):
            for pf in (1, 2, 3):
                rows.append(
                    {
                        "excntry": "USA",
                        "characteristic": ch,
                        "eom": eom,
                        "pf": pf,
                        "signal": float(pf) + 0.1 * i + (0.0 if ch == "char_a" else 5.0),
                        "ret_ew": 0.01 * pf + 0.001 * i + (0.0 if ch == "char_a" else 0.1),
                        "ret_vw": 0.02 * pf + 0.001 * i + (0.0 if ch == "char_a" else 0.1),
                        "ret_vw_cap": 0.03 * pf + 0.001 * i + (0.0 if ch == "char_a" else 0.1),
                        "n": 10 * pf + i + (0 if ch == "char_a" else 100),
                    }
                )
    return pl.DataFrame(rows)


def _themes(directions: list[int], chars: tuple[str, ...] = ("char_a", "char_b")) -> pl.DataFrame:
    """Themes table as ``load_themes`` returns it."""
    return pl.DataFrame(
        {"characteristic": chars, "theme": "t", "subtheme": "t", "direction": directions},
        schema_overrides={"direction": pl.Int32},
    )


class TestBuildHmlAndSignFactors:
    def test_direction_plus_one_lms_equals_hml(self):
        pf_df = _make_pf_df()
        hml = _build_hml(pf_df, n_pfs=3, date_col="eom", include_signal=True)
        lms = _sign_factors(hml, _themes([1, 1]))
        key = ["excntry", "characteristic", "eom"]
        merged = hml.join(lms, on=key, how="inner", suffix="_lms").sort(key)
        for c in ("ret_ew", "ret_vw", "ret_vw_cap"):
            assert merged[c].to_list() == merged[f"{c}_lms"].to_list()

    def test_direction_minus_one_flips_signs(self):
        pf_df = _make_pf_df()
        hml = _build_hml(pf_df, n_pfs=3, date_col="eom", include_signal=True)
        lms = _sign_factors(hml, _themes([-1, -1]))
        key = ["excntry", "characteristic", "eom"]
        merged = hml.join(lms, on=key, how="inner", suffix="_lms").sort(key)
        for c in ("ret_ew", "ret_vw", "ret_vw_cap", "signal"):
            for hv, lv in zip(merged[c].to_list(), merged[f"{c}_lms"].to_list(), strict=True):
                assert lv == pytest.approx(-hv)

    def test_pfs_filter_drops_singleton_cohorts(self):
        pf_df = _make_pf_df()
        # Drop the pf=3 row for (USA, char_a, 2020-01-31): cohort has only pf=1 (and pf=2).
        eom_drop = date(2020, 1, 31)
        pf_df = pf_df.filter(
            ~(
                (pl.col("characteristic") == "char_a")
                & (pl.col("eom") == eom_drop)
                & (pl.col("pf") == 3)
            )
        )
        hml = _build_hml(pf_df, n_pfs=3, date_col="eom", include_signal=True)
        dropped = hml.filter((pl.col("characteristic") == "char_a") & (pl.col("eom") == eom_drop))
        assert dropped.height == 0
        # Other cohorts retained.
        assert hml.height == 2 * 3 - 1

    def test_n_stocks_is_sum_of_extremes(self):
        pf_df = _make_pf_df()
        hml = _build_hml(pf_df, n_pfs=3, date_col="eom", include_signal=True)
        for row in hml.iter_rows(named=True):
            expected = (
                pf_df.filter(
                    (pl.col("excntry") == row["excntry"])
                    & (pl.col("characteristic") == row["characteristic"])
                    & (pl.col("eom") == row["eom"])
                    & (pl.col("pf") == 3)
                )["n"].item()
                + pf_df.filter(
                    (pl.col("excntry") == row["excntry"])
                    & (pl.col("characteristic") == row["characteristic"])
                    & (pl.col("eom") == row["eom"])
                    & (pl.col("pf") == 1)
                )["n"].item()
            )
            assert row["n_stocks"] == expected

    def test_include_signal_false_omits_signal_column(self):
        pf_df = _make_pf_df()
        hml = _build_hml(pf_df, n_pfs=3, date_col="eom", include_signal=False)
        lms = _sign_factors(hml, _themes([1, 1]))
        assert "signal" not in hml.columns
        assert "signal" not in lms.columns

    def test_sign_factors_flips_cmp_columns(self):
        cmp = pl.DataFrame(
            {
                "characteristic": ["char_a", "char_b"],
                "size_grp": "large",
                "eom": date(2020, 1, 31),
                "n_stocks": [10, 20],
                "ret_weighted": [0.1, 0.2],
                "signal_weighted": [1.0, 2.0],
                "excntry": "USA",
            }
        )
        signed = _sign_factors(cmp, _themes([1, -1]))
        assert signed.columns == [
            "characteristic",
            "direction",
            "size_grp",
            "eom",
            "n_stocks",
            "ret_weighted",
            "signal_weighted",
            "excntry",
        ]
        assert signed["ret_weighted"].to_list() == [0.1, -0.2]
        assert signed["signal_weighted"].to_list() == [1.0, -2.0]
        assert signed["n_stocks"].to_list() == [10, 20]

    def test_sign_factors_flips_any_return_or_signal_column(self):
        factors = pl.DataFrame(
            {"characteristic": ["char_a"], "n_stocks": [10], "ret_new": [0.1], "signal_new": [1.0]}
        )
        signed = _sign_factors(factors, _themes([-1], chars=("char_a",)))
        assert signed["ret_new"].to_list() == [-0.1]
        assert signed["signal_new"].to_list() == [-1.0]
        assert signed["n_stocks"].to_list() == [10]

    def test_lms_keeps_only_themed_characteristics_in_hml_order(self):
        hml = _build_hml(_make_pf_df(), n_pfs=3, date_col="eom", include_signal=True)
        lms = _sign_factors(hml, _themes([-1], chars=("char_b",)))
        expected = hml.filter(pl.col("characteristic") == "char_b")
        assert lms.columns == [
            "characteristic",
            "direction",
            *expected.drop("characteristic").columns,
        ]
        assert lms["eom"].equals(expected["eom"])
        assert lms["n_stocks"].equals(expected["n_stocks"])
