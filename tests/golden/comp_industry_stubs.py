"""Shared upstream stubs for ``comp_industry`` tests and golden generators.

``comp_industry`` calls ``comp_sic_naics(paths)`` and ``hgics_join(paths)``
before running its own DuckDB SQL.  When testing ``comp_industry`` in
isolation we stub both calls to no-ops so only the SQL is exercised.

This module is the single source of truth for those stubs, used by:

- ``tests.unit.test_comp_industry`` (via ``patch_comp_industry_upstream_stubs``)
- ``tests.golden.generate_comp_industry_golden`` (via ``comp_industry_upstream_stubs``)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Generator

    import pytest

    from jkp.data.paths import DataPaths

_PATCH_TARGET = "jkp.data.aux_functions"


@dataclass
class CompIndustryUpstreamStubs:
    """Sentinel tracker: records whether the stubbed functions were invoked."""

    comp_sic_naics_called: bool = field(default=False, init=False)
    hgics_join_called: bool = field(default=False, init=False)

    def _comp_sic_naics(self, _paths: DataPaths) -> None:
        self.comp_sic_naics_called = True

    def _hgics_join(self, _paths: DataPaths) -> None:
        self.hgics_join_called = True

    def assert_called(self) -> None:
        """Assert both upstream stubs were actually invoked by ``comp_industry``."""
        assert self.comp_sic_naics_called, (
            "comp_sic_naics stub was never called — the patch may be stale"
        )
        assert self.hgics_join_called, "hgics_join stub was never called — the patch may be stale"


def patch_comp_industry_upstream_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> CompIndustryUpstreamStubs:
    """Apply upstream stubs via *monkeypatch* (for pytest fixtures).

    Returns the ``CompIndustryUpstreamStubs`` instance so the caller can
    later call ``stubs.assert_called()`` to verify the stubs were hit.
    """
    import jkp.data.aux_functions as aux_functions

    stubs = CompIndustryUpstreamStubs()
    monkeypatch.setattr(aux_functions, "comp_sic_naics", stubs._comp_sic_naics)
    monkeypatch.setattr(aux_functions, "hgics_join", stubs._hgics_join)
    return stubs


@contextmanager
def comp_industry_upstream_stubs() -> Generator[CompIndustryUpstreamStubs]:
    """Context-manager variant for non-pytest callers (golden generators)."""
    import jkp.data.aux_functions as aux_functions

    stubs = CompIndustryUpstreamStubs()
    with (
        patch.object(aux_functions, "comp_sic_naics", stubs._comp_sic_naics),
        patch.object(aux_functions, "hgics_join", stubs._hgics_join),
    ):
        yield stubs
