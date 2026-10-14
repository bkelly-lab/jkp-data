"""The bundled themes files are a valid grouping and match scripts/themes/economic_themes.py."""

import importlib.util
from pathlib import Path

import polars as pl
import pytest

THEMES_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "themes" / "economic_themes.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("economic_themes", THEMES_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_grouping_is_valid(script):
    script.validate()


@pytest.mark.unit
def test_bundled_files_match_script(script):
    """Fails if the script was edited without re-running it (or the CSVs edited by hand)."""
    for name, frame in [
        ("themes.csv", script.themes_frame()),
        ("theme_descriptions.csv", script.descriptions_frame()),
    ]:
        bundled = pl.read_csv(script.OUT_DIR / name, infer_schema=False)
        assert bundled.equals(frame.cast(pl.String)), (
            f"{name} is out of date: run {THEMES_SCRIPT.name}"
        )
