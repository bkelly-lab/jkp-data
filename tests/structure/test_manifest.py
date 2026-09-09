"""
Description:
    Assert the repository still contains what pipeline_structure.txt says it does.

    Updating the manifest alongside a real change is intentional — the same
    contract as EXPECTED_PUBLIC_API in tests/unit/test_public_api.py. The
    manifest's own diff is the review artifact: a deletion shows up as a red line
    in the PR rather than as silence. See issue #254.
Steps:
    1) Compare the live manifest against the checked-in one.
    2) Assert no package module is orphaned.
    3) Assert the three characteristic sources still agree.
Output:
    Test results.
"""

from __future__ import annotations

import difflib

import polars as pl
import pytest

from jkp.data.config import PORTFOLIO_CHARS
from jkp.data.paths import get_cluster_labels_path, get_factor_details_path
from tests.structure.build_manifest import (
    MANIFEST_PATH,
    build_manifest,
    import_edges,
    reachable_modules,
    repo_root,
    tracked_files,
)

pytestmark = pytest.mark.unit

REGEN_HINT = "Run: PYTHONPATH=src uv run python -m tests.structure.build_manifest --regen"


def test_manifest_matches_repository() -> None:
    """The checked-in manifest should describe the repository as it is."""
    root = repo_root()
    actual = build_manifest(root)
    expected = (root / MANIFEST_PATH).read_text(encoding="utf-8").splitlines()

    if actual != expected:
        diff = list(
            difflib.unified_diff(expected, actual, "checked-in", "actual", lineterm="", n=0)
        )
        body = "\n".join(diff[:40])
        extra = f"\n… {len(diff) - 40} more diff lines" if len(diff) > 40 else ""
        pytest.fail(
            f"Inventory manifest is out of date.\n\n{body}{extra}\n\n"
            f"If this change is intentional, regenerate and commit the manifest so the\n"
            f"deletion or addition is visible in review.\n{REGEN_HINT}"
        )


def test_no_orphaned_modules() -> None:
    """
    Every module under src/jkp/data/ should be reachable from an entry point.

    Asserted against the live AST rather than the manifest text, so that
    regenerating the manifest cannot paper over an un-wired module. This is the
    PR #247 failure: compustat_correction kept passing its own unit tests while
    nothing in the pipeline imported it.
    """
    root = repo_root()
    paths = tracked_files(root)
    present = {
        p.rsplit("/", 1)[-1].removesuffix(".py")
        for p in paths
        if p.startswith("src/jkp/data/") and p.endswith(".py")
    }
    edges = import_edges(root, paths)
    orphans = sorted(present - reachable_modules(edges, present))

    assert not orphans, (
        f"Module(s) no longer imported by the pipeline: {orphans}. "
        f"The code and its tests may still be present and passing, but nothing calls it. "
        f"Either re-wire it into the pipeline or delete it deliberately."
    )


def test_characteristic_sources_agree() -> None:
    """
    PORTFOLIO_CHARS, cluster_labels.csv and factor_details.xlsx should list the
    same characteristics.

    Two of the three are opaque to code review, so a row silently vanishing from
    either would otherwise shrink the published output unnoticed.
    """
    config_chars = set(PORTFOLIO_CHARS)
    cluster_chars = set(pl.read_csv(get_cluster_labels_path())["characteristic"].to_list())
    details = pl.read_excel(get_factor_details_path(), sheet_name="details")
    detail_chars = set(details["abr_jkp"].drop_nulls().to_list())

    assert config_chars == cluster_chars, (
        "PORTFOLIO_CHARS and cluster_labels.csv disagree. "
        f"Only in config: {sorted(config_chars - cluster_chars)}. "
        f"Only in cluster_labels.csv: {sorted(cluster_chars - config_chars)}."
    )
    assert config_chars == detail_chars, (
        "PORTFOLIO_CHARS and factor_details.xlsx[abr_jkp] disagree. "
        f"Only in config: {sorted(config_chars - detail_chars)}. "
        f"Only in factor_details.xlsx: {sorted(detail_chars - config_chars)}."
    )
