"""Tests for the portfolio module."""

import pytest


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_format_is_parquet(self, monkeypatch):
        """Default output format should be parquet when no args provided."""
        monkeypatch.setattr("sys.argv", ["portfolio.py"])
        from portfolio import parse_args

        args = parse_args()
        assert args.output_format == "parquet"

    def test_can_specify_csv_format(self, monkeypatch):
        """Can specify CSV output format via command line."""
        monkeypatch.setattr("sys.argv", ["portfolio.py", "--output-format", "csv"])
        from portfolio import parse_args

        args = parse_args()
        assert args.output_format == "csv"

    def test_can_specify_parquet_format(self, monkeypatch):
        """Can explicitly specify parquet output format."""
        monkeypatch.setattr("sys.argv", ["portfolio.py", "--output-format", "parquet"])
        from portfolio import parse_args

        args = parse_args()
        assert args.output_format == "parquet"

    def test_invalid_format_raises_error(self, monkeypatch):
        """Invalid output format raises SystemExit."""
        monkeypatch.setattr("sys.argv", ["portfolio.py", "--output-format", "json"])
        from portfolio import parse_args

        with pytest.raises(SystemExit):
            parse_args()
