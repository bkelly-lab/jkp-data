"""Tests for the portfolio module."""

import pytest

from jkp_data.output_writer import configure_output_format


class TestOutputFormatIntegration:
    """Tests that run_portfolio() forwards output_format to configure_output_format."""

    def test_default_format_is_parquet(self):
        """run_portfolio() defaults to parquet format."""
        from unittest.mock import patch

        from jkp_data.portfolio import run_portfolio

        # configure_output_format raises to bail out early — proves it was called
        with patch(
            "jkp_data.portfolio.configure_output_format",
            side_effect=SystemExit("bail"),
        ) as mock_configure:
            with pytest.raises(SystemExit):
                run_portfolio()
            mock_configure.assert_called_once_with("parquet")

    def test_csv_format_passed_through(self):
        """run_portfolio(output_format='csv') forwards 'csv' to configure_output_format."""
        from unittest.mock import patch

        from jkp_data.portfolio import run_portfolio

        with patch(
            "jkp_data.portfolio.configure_output_format",
            side_effect=SystemExit("bail"),
        ) as mock_configure:
            with pytest.raises(SystemExit):
                run_portfolio(output_format="csv")
            mock_configure.assert_called_once_with("csv")
