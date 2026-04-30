"""Tests for WRDS credential resolution.

Covers the documented precedence order:
    1. WRDS_USERNAME / WRDS_PASSWORD environment variables.
    2. The system keyring.
    3. Plaintext keyring, only if JKP_ALLOW_PLAINTEXT_KEYRING=1.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_import_does_not_mutate_environ():
    """Importing the module must not change PYTHON_KEYRING_BACKEND or any other
    environment variable. The previous implementation mutated the environment at
    import time, silently switching the user's keyring backend to plaintext."""
    import os

    before = dict(os.environ)
    import jkp.data.wrds_credentials as mod  # noqa: F401

    # Force a re-import to also exercise the import path.
    importlib.reload(mod)
    after = dict(os.environ)
    assert before == after, "import must not mutate os.environ"


@pytest.mark.unit
def test_env_vars_take_precedence(monkeypatch):
    """When WRDS_USERNAME and WRDS_PASSWORD are set, return those without
    touching the keyring."""
    monkeypatch.setenv("WRDS_USERNAME", "ci-user")
    monkeypatch.setenv("WRDS_PASSWORD", "ci-secret")

    # If keyring is touched, fail loudly.
    import jkp.data.wrds_credentials as mod

    def _boom(*a, **kw):
        raise AssertionError("keyring must not be queried when env vars are set")

    monkeypatch.setattr(mod.keyring, "get_password", _boom)

    creds = mod.get_wrds_credentials()
    assert creds.username == "ci-user"
    assert creds.password == "ci-secret"


@pytest.mark.unit
def test_env_partial_falls_through_to_keyring(monkeypatch, tmp_path):
    """If only one of the env vars is set, fall through to keyring resolution."""
    monkeypatch.setenv("WRDS_USERNAME", "ci-user")
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)

    import jkp.data.wrds_credentials as mod

    monkeypatch.setattr(mod, "LAST_USER_FILE", tmp_path / ".wrds_user")
    (tmp_path / ".wrds_user").write_text("kept-user")

    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: "kept-pw")
    creds = mod.get_wrds_credentials()
    assert creds.username == "kept-user", "env-var partial set must not be used"
    assert creds.password == "kept-pw"


@pytest.mark.unit
def test_plaintext_opt_in_emits_warning(monkeypatch):
    """Setting JKP_ALLOW_PLAINTEXT_KEYRING=1 should emit a warning the first
    time credential resolution runs."""
    monkeypatch.setenv("WRDS_USERNAME", "u")
    monkeypatch.setenv("WRDS_PASSWORD", "p")  # short-circuits keyring code path
    monkeypatch.setenv("JKP_ALLOW_PLAINTEXT_KEYRING", "1")

    import jkp.data.wrds_credentials as mod

    # Reset_credentials triggers _maybe_use_file_keyring even with env vars set.
    monkeypatch.setattr(mod, "LAST_USER_FILE", __import__("pathlib").Path("/tmp/__no__"))

    with pytest.warns(UserWarning, match="keyrings.alt.file.PlaintextKeyring"):
        mod._maybe_use_file_keyring()


@pytest.mark.unit
def test_plaintext_no_optin_no_warning(monkeypatch, recwarn):
    """Without the opt-in env var, no warning is emitted and the keyring backend
    is not mutated."""
    monkeypatch.delenv("JKP_ALLOW_PLAINTEXT_KEYRING", raising=False)

    import jkp.data.wrds_credentials as mod

    set_keyring_calls = []
    monkeypatch.setattr(mod.keyring, "set_keyring", lambda kr: set_keyring_calls.append(kr))
    mod._maybe_use_file_keyring()
    assert not set_keyring_calls, "keyring backend should not be swapped without opt-in"
    assert len(recwarn.list) == 0, "no warning should fire without opt-in"
