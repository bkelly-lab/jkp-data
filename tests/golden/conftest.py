"""Pytest config for golden-fixture tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.golden.generate_synthetic_wrds import write_synthetic_wrds


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help="Regenerate golden fixtures",
    )


@pytest.fixture(scope="session")
def synthetic_wrds_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic WRDS-shaped inputs, generated once per session (nothing is committed)."""
    out = tmp_path_factory.mktemp("synthetic_wrds")
    write_synthetic_wrds(out)
    return out
