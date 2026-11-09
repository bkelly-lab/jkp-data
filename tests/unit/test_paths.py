"""Tests for the paths module."""

from pathlib import Path

from jkp.data.paths import (
    DataPaths,
    _resource_path,
    get_country_classification_path,
    get_data_readme_path,
    get_fx_cad_backfill_path,
    get_siccodes_path,
    get_themes_dir,
)


class TestDataPaths:
    """Tests for DataPaths dataclass."""

    def test_raw_dir(self, tmp_path):
        paths = DataPaths(base_dir=tmp_path)
        assert paths.raw_dir == tmp_path / "raw"

    def test_raw_tables_dir(self, tmp_path):
        paths = DataPaths(base_dir=tmp_path)
        assert paths.raw_tables_dir == tmp_path / "raw" / "raw_tables"

    def test_interim_dir(self, tmp_path):
        paths = DataPaths(base_dir=tmp_path)
        assert paths.interim_dir == tmp_path / "interim"

    def test_processed_dir(self, tmp_path):
        paths = DataPaths(base_dir=tmp_path)
        assert paths.processed_dir == tmp_path / "processed"

    def test_frozen(self, tmp_path):
        """DataPaths should be immutable."""
        import pytest

        paths = DataPaths(base_dir=tmp_path)
        with pytest.raises(AttributeError):
            paths.base_dir = tmp_path / "other"


class TestResourcePaths:
    """Tests that bundled resource files are accessible."""

    def test_siccodes_exists(self):
        path = get_siccodes_path()
        assert isinstance(path, Path)
        assert path.exists(), f"Siccodes49.txt not found at {path}"

    def test_themes_dir_exists(self):
        path = get_themes_dir()
        assert isinstance(path, Path)
        assert (path / "themes.csv").exists(), f"themes.csv not found in {path}"
        assert (path / "superseded_themes" / "v0" / "cluster_labels.csv").exists()

    def test_country_classification_exists(self):
        path = get_country_classification_path()
        assert isinstance(path, Path)
        assert path.exists(), f"country_classification.xlsx not found at {path}"

    def test_factor_details_exists(self):
        path = _resource_path("factor_details.xlsx")
        assert isinstance(path, Path)
        assert path.exists(), f"factor_details.xlsx not found at {path}"

    def test_fx_cad_backfill_exists(self):
        path = get_fx_cad_backfill_path()
        assert isinstance(path, Path)
        assert path.exists(), f"fx_cad_1960_1982.csv not found at {path}"

    def test_data_readme_exists(self):
        path = get_data_readme_path()
        assert isinstance(path, Path)
        assert path.exists(), f"README.md not found at {path}"

    def test_siccodes_is_readable(self):
        """Siccodes49.txt should be readable as text."""
        path = get_siccodes_path()
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0, "Siccodes49.txt is empty"
        assert "Agric" in content, "Siccodes49.txt should contain industry labels"
