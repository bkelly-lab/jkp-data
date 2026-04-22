"""Tests for the portfolio module."""

from unittest.mock import patch

import pytest


class TestOutputFormatIntegration:
    """Tests that run_portfolio() forwards output_format to configure_output_format."""

    def test_default_format_is_parquet(self, tmp_path):
        """run_portfolio() defaults to parquet format."""
        from jkp.data.portfolio import run_portfolio

        with patch(
            "jkp.data.portfolio.configure_output_format",
            side_effect=SystemExit("bail"),
        ) as mock_configure:
            with pytest.raises(SystemExit):
                run_portfolio(output_dir=tmp_path)
            mock_configure.assert_called_once_with("parquet")

    def test_csv_format_passed_through(self, tmp_path):
        """run_portfolio(output_format='csv') forwards 'csv' to configure_output_format."""
        from jkp.data.portfolio import run_portfolio

        with patch(
            "jkp.data.portfolio.configure_output_format",
            side_effect=SystemExit("bail"),
        ) as mock_configure:
            with pytest.raises(SystemExit):
                run_portfolio(output_format="csv", output_dir=tmp_path)
            mock_configure.assert_called_once_with("csv")
