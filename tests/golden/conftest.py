"""Pytest config for golden-fixture tests."""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help="Regenerate golden fixtures",
    )
