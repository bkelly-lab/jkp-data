"""Centralized path management for the JKP data pipeline."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Holds the user-specified output directory and exposes subdirectory paths.

    The directory layout under base_dir mirrors the pipeline's expected structure:
        base_dir/
            raw/raw_tables/     # downloaded WRDS data
            interim/            # intermediate pipeline files
            interim/raw_data_dfs/
            processed/          # final outputs
    """

    base_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "raw"

    @property
    def raw_tables_dir(self) -> Path:
        return self.base_dir / "raw" / "raw_tables"

    @property
    def interim_dir(self) -> Path:
        return self.base_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "processed"


def _resource_path(filename: str) -> Path:
    """Return the path to a bundled resource file."""
    return files("jkp.data") / "resources" / filename


def get_siccodes_path() -> Path:
    """Return the path to the bundled Siccodes49.txt file."""
    return _resource_path("Siccodes49.txt")


def get_cluster_labels_path() -> Path:
    """Return the path to the bundled cluster_labels.csv file."""
    return _resource_path("cluster_labels.csv")


def get_country_classification_path() -> Path:
    """Return the path to the bundled country_classification.xlsx file."""
    return _resource_path("country_classification.xlsx")


def get_factor_details_path() -> Path:
    """Return the path to the bundled factor_details.xlsx file."""
    return _resource_path("factor_details.xlsx")
