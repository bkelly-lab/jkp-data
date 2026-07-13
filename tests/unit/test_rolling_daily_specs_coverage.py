"""Coverage tests: every char in ROLLING_DAILY_SPECS dispatches in roll_apply_daily."""

from __future__ import annotations

import ast
import inspect
import textwrap

import polars as pl
import pytest

from jkp.data import aux_functions
from jkp.data.aux_functions import base_data_filter_exp, process_window
from jkp.data.config import ROLLING_DAILY_SPECS


def _process_window_funcs_keys() -> set[str]:
    """Parse the `funcs` dict literal inside process_window via AST."""
    source = inspect.getsource(process_window)
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "funcs"
            and isinstance(node.value, ast.Dict)
        ):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("funcs dict not found in process_window source")


SPEC_PAIRS = [(sfx, var) for sfx, _, vars_ in ROLLING_DAILY_SPECS for var in vars_]


def test_rolling_daily_specs_non_empty() -> None:
    assert ROLLING_DAILY_SPECS, "ROLLING_DAILY_SPECS must not be empty"
    for sfx, min_obs, vars_ in ROLLING_DAILY_SPECS:
        assert sfx.startswith("_") and sfx.endswith("d"), sfx
        assert isinstance(min_obs, int) and min_obs > 0
        assert vars_, f"empty var list for {sfx}"


@pytest.mark.parametrize(("sfx", "var"), SPEC_PAIRS)
def test_var_dispatches_in_process_window(sfx: str, var: str) -> None:
    """Every var in ROLLING_DAILY_SPECS must be a key in process_window `funcs`."""
    keys = _process_window_funcs_keys()
    assert var in keys, f"ROLLING_DAILY_SPECS[{sfx}] var '{var}' has no handler in process_window"


@pytest.mark.parametrize(("sfx", "var"), SPEC_PAIRS)
def test_var_handler_is_callable(sfx: str, var: str) -> None:
    """The dispatched handler resolves to a callable on aux_functions."""
    keys = _process_window_funcs_keys()
    assert var in keys
    handler_name = {"mktvol": "mktrf_vol"}.get(var, var)
    handler = getattr(aux_functions, handler_name, None)
    assert callable(handler), f"{var} → {handler_name} not callable on aux_functions"


@pytest.mark.parametrize("var", sorted({v for _, _, vs in ROLLING_DAILY_SPECS for v in vs}))
def test_base_data_filter_exp_accepts_var(var: str) -> None:
    """base_data_filter_exp must return a Polars expression for every spec var."""
    expr = base_data_filter_exp(var)
    assert isinstance(expr, pl.Expr)
