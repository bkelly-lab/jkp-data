"""Tests for portfolio output cleanup on reruns."""

from __future__ import annotations

from pathlib import Path

import pytest

from jkp.data.portfolio import _remove_stale_hml


@pytest.mark.unit
def test_remove_stale_hml_deletes_unwritten_outputs(tmp_path: Path) -> None:
    """A rerun that produces fewer series should not leave earlier hml files."""
    for name in ("hml", "hml_daily", "hml_overnight"):
        (tmp_path / f"{name}.parquet").write_bytes(b"old")

    _remove_stale_hml(tmp_path, {"hml", "hml_daily"})

    assert (tmp_path / "hml.parquet").exists()
    assert (tmp_path / "hml_daily.parquet").exists()
    assert not (tmp_path / "hml_overnight.parquet").exists()


@pytest.mark.unit
def test_remove_stale_hml_removes_csv_counterparts(tmp_path: Path) -> None:
    """CSV outputs from earlier runs should be removed along with parquet."""
    (tmp_path / "hml_intraday.parquet").write_bytes(b"old")
    (tmp_path / "hml_intraday.csv").write_text("old", encoding="utf-8")

    _remove_stale_hml(tmp_path, {"hml"})

    assert not (tmp_path / "hml_intraday.parquet").exists()
    assert not (tmp_path / "hml_intraday.csv").exists()


@pytest.mark.unit
def test_remove_stale_hml_keeps_written_files(tmp_path: Path) -> None:
    """Files produced by the current run must not be deleted."""
    for name in ("hml", "hml_daily", "hml_overnight"):
        (tmp_path / f"{name}.parquet").write_bytes(b"new")

    _remove_stale_hml(tmp_path, {"hml", "hml_daily", "hml_overnight"})

    for name in ("hml", "hml_daily", "hml_overnight"):
        assert (tmp_path / f"{name}.parquet").exists()
