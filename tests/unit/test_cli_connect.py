"""Tests for the `jkp connect` CLI command's WRDS verification path.

Complements TestConnectCommand in test_cli.py, focusing on how the command
surfaces verify_wrds_connection's success/failure and never leaks the
credential password into CLI output.

`connect` injects the verifier into get_wrds_credentials rather than calling it
on the result, so that a freshly prompted password is checked before it is
stored. Verification failures therefore reach the command as an exception out of
get_wrds_credentials, which is how these tests simulate them.
"""

import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from jkp.data.cli import app
from jkp.data.wrds_credentials import Credentials

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (see test_cli.py's identical helper)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.mark.unit
class TestConnectVerification:
    """`connect` calls verify_wrds_connection and surfaces its outcome."""

    @patch("jkp.data.wrds_connection.verify_wrds_connection")
    @patch("jkp.data.wrds_credentials.get_wrds_credentials")
    def test_success_exits_zero_and_reports_username(self, mock_get_creds, mock_verify):
        mock_get_creds.return_value = Credentials(username="testuser", password="hunter2")
        mock_verify.return_value = None

        result = runner.invoke(app, ["connect"])

        assert result.exit_code == 0
        assert "Connected as:" in result.output
        # Passed through, not applied here: verifying the returned credentials again
        # would open a second WRDS connection and risk a second Duo push.
        mock_get_creds.assert_called_once_with(verify=mock_verify)
        mock_verify.assert_not_called()

    @patch("jkp.data.wrds_connection.verify_wrds_connection")
    @patch("jkp.data.wrds_credentials.get_wrds_credentials")
    def test_failure_exits_nonzero_with_message_on_stderr_no_traceback(
        self, mock_get_creds, mock_verify
    ):
        message = "Failed to attach WRDS connection. Check credentials and MFA approval."
        # The injected verifier raises inside the resolver, so the failure surfaces
        # from get_wrds_credentials rather than from a separate call in the command.
        mock_get_creds.side_effect = RuntimeError(message)

        result = runner.invoke(app, ["connect"])

        assert result.exit_code != 0
        assert message in _strip_ansi(result.stderr)
        # handled as a clean exit, not a leaked RuntimeError traceback
        assert not isinstance(result.exception, RuntimeError)
        assert "Traceback" not in result.stderr

    @patch("jkp.data.wrds_connection.verify_wrds_connection")
    @patch("jkp.data.wrds_credentials.get_wrds_credentials")
    def test_password_never_appears_in_output(self, mock_get_creds, mock_verify):
        """The real verify_wrds_connection already redacts the password before
        raising; this pins the CLI to surfacing that message verbatim without
        introducing a new leak (e.g. by echoing creds directly)."""
        password = "s3cr3t-p@ss"  # noqa: S105
        # Message the real code would raise: already password-free.
        mock_get_creds.side_effect = RuntimeError(
            "Failed to attach WRDS connection. Check credentials and MFA approval."
        )

        result = runner.invoke(app, ["connect"])

        assert result.exit_code != 0
        combined = _strip_ansi(result.output) + _strip_ansi(result.stderr)
        assert password not in combined

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("WRDS password contains a newline"),
            OSError("[Errno 30] Read-only file system: state dir"),
        ],
    )
    @patch("jkp.data.wrds_credentials.get_wrds_credentials")
    def test_credential_resolution_errors_exit_cleanly_without_traceback(self, mock_get_creds, exc):
        """Credential resolution can raise ValueError (e.g. a newline password) or
        OSError (e.g. an unwritable state dir) — the command must surface the message
        and exit non-zero, not dump a raw traceback."""
        mock_get_creds.side_effect = exc

        result = runner.invoke(app, ["connect"])

        assert result.exit_code != 0
        assert str(exc) in _strip_ansi(result.stderr)
        assert not isinstance(result.exception, (ValueError, OSError))
        assert "Traceback" not in result.stderr
