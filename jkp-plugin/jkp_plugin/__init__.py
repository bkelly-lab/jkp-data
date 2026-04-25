"""Polars expression plugins for jkp-data."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

_PLUGIN_PATH = Path(__file__).parent

__all__ = ["dimson_beta"]


def dimson_beta(
    target: pl.Expr | str,
    *mkt_cols: pl.Expr | str,
    min_obs: int = 15,
) -> pl.Expr:
    """Sum of market-return coefficients from a Dimson regression.

    Regresses ``target`` on ``mkt_cols`` plus an intercept and returns the
    sum of the market-return coefficients (intercept excluded).

    Returns null when:
      - group has fewer than ``min_obs`` non-null rows, or
      - the X'X matrix is singular (Cholesky fails).
    """
    args = [target, *mkt_cols]
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="dimson_beta",
        args=args,
        kwargs={"min_obs": min_obs},
        is_elementwise=False,
        returns_scalar=True,
    )
