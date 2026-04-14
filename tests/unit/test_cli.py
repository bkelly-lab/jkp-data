"""Tests for the JKP CLI entry point."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from jkp_data.cli import app

runner = CliRunner()


@pytest.mark.unit
class TestCliHelp:
    """Test that --help output works for all commands."""

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "JKP Factor Data generation pipeline" in result.output

    def test_build_help(self):
        result = runner.invoke(app, ["build", "--help"])
        assert result.exit_code == 0
        assert "--persistent-connection" in result.output

    def test_portfolio_help(self):
        result = runner.invoke(app, ["portfolio", "--help"])
        assert result.exit_code == 0
        assert "factor portfolios" in result.output.lower()

    def test_connect_help(self):
        result = runner.invoke(app, ["connect", "--help"])
        assert result.exit_code == 0
        assert "--reset" in result.output


@pytest.mark.unit
class TestBuildCommand:
    """Test the build command routes to run_pipeline correctly."""

    @patch("jkp_data.main.run_pipeline")
    def test_build_calls_run_pipeline(self, mock_run_pipeline):
        result = runner.invoke(app, ["build"])
        assert result.exit_code == 0
        mock_run_pipeline.assert_called_once_with(persistent_connection=False)

    @patch("jkp_data.main.run_pipeline")
    def test_build_persistent_connection(self, mock_run_pipeline):
        result = runner.invoke(app, ["build", "--persistent-connection"])
        assert result.exit_code == 0
        mock_run_pipeline.assert_called_once_with(persistent_connection=True)

    @patch("jkp_data.main.run_pipeline")
    def test_build_persistent_connection_short(self, mock_run_pipeline):
        result = runner.invoke(app, ["build", "-p"])
        assert result.exit_code == 0
        mock_run_pipeline.assert_called_once_with(persistent_connection=True)


@pytest.mark.unit
class TestPortfolioCommand:
    """Test the portfolio command routes to run_portfolio correctly."""

    @patch("jkp_data.portfolio.run_portfolio")
    def test_portfolio_calls_run_portfolio(self, mock_run_portfolio):
        result = runner.invoke(app, ["portfolio"])
        assert result.exit_code == 0
        mock_run_portfolio.assert_called_once_with(output_format="parquet")

    @patch("jkp_data.portfolio.run_portfolio")
    def test_portfolio_csv_format(self, mock_run_portfolio):
        result = runner.invoke(app, ["portfolio", "--output-format", "csv"])
        assert result.exit_code == 0
        mock_run_portfolio.assert_called_once_with(output_format="csv")


@pytest.mark.unit
class TestConnectCommand:
    """Test the connect command routes to wrds_credentials correctly."""

    @patch("jkp_data.wrds_credentials.get_wrds_credentials")
    def test_connect_shows_username(self, mock_get_creds):
        mock_get_creds.return_value = MagicMock(username="testuser")
        result = runner.invoke(app, ["connect"])
        assert result.exit_code == 0
        assert "testuser" in result.output
        mock_get_creds.assert_called_once()

    @patch("jkp_data.wrds_credentials.reset_credentials")
    def test_connect_reset(self, mock_reset):
        result = runner.invoke(app, ["connect", "--reset"])
        assert result.exit_code == 0
        mock_reset.assert_called_once_with(full_reset=True)
        assert "reset" in result.output.lower()

    @patch("jkp_data.wrds_credentials.reset_credentials")
    def test_connect_reset_short(self, mock_reset):
        result = runner.invoke(app, ["connect", "-r"])
        assert result.exit_code == 0
        mock_reset.assert_called_once_with(full_reset=True)
