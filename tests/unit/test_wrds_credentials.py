"""Tests for WRDS credential resolution.

Covers the documented precedence order:
    1. WRDS_USERNAME / WRDS_PASSWORD environment variables.
    2. The system keyring.
    3. Plaintext keyring, only if JKP_ALLOW_PLAINTEXT_KEYRING=1.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _clear_keyring_backend_cache():
    """The backend swap is cached per-process, so reset it around every test to
    keep cases independent (otherwise the first opt-in leaks into later tests)."""
    import jkp.data.wrds_credentials as mod

    mod._ensure_keyring_backend.cache_clear()
    yield
    mod._ensure_keyring_backend.cache_clear()


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
def test_plaintext_opt_in_swaps_backend_and_warns(monkeypatch):
    """Setting JKP_ALLOW_PLAINTEXT_KEYRING=1 should swap the keyring backend to
    PlaintextKeyring and emit a warning when credential resolution runs."""
    monkeypatch.setenv("JKP_ALLOW_PLAINTEXT_KEYRING", "1")

    from keyrings.alt.file import PlaintextKeyring

    import jkp.data.wrds_credentials as mod

    # Capture the backend swap instead of mutating the process-global keyring.
    captured = []
    monkeypatch.setattr(mod.keyring, "set_keyring", captured.append)

    with pytest.warns(UserWarning, match="keyrings.alt.file.PlaintextKeyring"):
        mod._ensure_keyring_backend()

    assert len(captured) == 1, "the backend should be swapped exactly once"
    assert isinstance(captured[0], PlaintextKeyring)


@pytest.mark.unit
def test_plaintext_no_optin_no_warning(monkeypatch, recwarn):
    """Without the opt-in env var, no warning is emitted and the keyring backend
    is not mutated."""
    monkeypatch.delenv("JKP_ALLOW_PLAINTEXT_KEYRING", raising=False)

    import jkp.data.wrds_credentials as mod

    set_keyring_calls = []
    monkeypatch.setattr(mod.keyring, "set_keyring", lambda kr: set_keyring_calls.append(kr))
    mod._ensure_keyring_backend()
    assert not set_keyring_calls, "keyring backend should not be swapped without opt-in"
    assert len(recwarn.list) == 0, "no warning should fire without opt-in"


@pytest.mark.unit
@pytest.mark.parametrize("value", ["true", "True", "yes", "on", "0", "2", ""])
def test_plaintext_opt_in_requires_exact_1(monkeypatch, recwarn, value):
    """Only the exact string "1" opts in. Anything else — including
    truthy-looking values like "true"/"yes" and falsy ones like "0" — is treated
    as not set: no backend swap, no warning. Guards the strict comparison against
    being loosened into a fuzzy truthiness check later."""
    monkeypatch.setenv("JKP_ALLOW_PLAINTEXT_KEYRING", value)

    import jkp.data.wrds_credentials as mod

    set_keyring_calls = []
    monkeypatch.setattr(mod.keyring, "set_keyring", lambda kr: set_keyring_calls.append(kr))
    mod._ensure_keyring_backend()
    assert not set_keyring_calls, f"value {value!r} must not swap the keyring backend"
    assert len(recwarn.list) == 0, f"value {value!r} must not emit a warning"


@pytest.mark.unit
def test_backend_swap_is_cached(monkeypatch, recwarn):
    """The opt-in swap happens at most once per process: repeated keyring access
    must not re-run the swap or re-emit the warning."""
    monkeypatch.setenv("JKP_ALLOW_PLAINTEXT_KEYRING", "1")

    import jkp.data.wrds_credentials as mod

    captured = []
    monkeypatch.setattr(mod.keyring, "set_keyring", captured.append)

    mod._ensure_keyring_backend()
    mod._ensure_keyring_backend()
    with mod._keyring_backend():
        pass

    assert len(captured) == 1, "backend swap should happen once despite repeated calls"
    warnings_emitted = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert len(warnings_emitted) == 1, "the swap warning should fire once, not per call"


@pytest.mark.unit
def test_get_wrds_credentials_missing_backend_raises_guidance(monkeypatch, tmp_path):
    """On a headless system with no keyring backend and no opt-in, the generic
    NoKeyringError is re-raised with guidance pointing at the env-var and
    plaintext-opt-in escape hatches."""
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    monkeypatch.delenv("JKP_ALLOW_PLAINTEXT_KEYRING", raising=False)

    import jkp.data.wrds_credentials as mod

    monkeypatch.setattr(mod, "LAST_USER_FILE", tmp_path / ".wrds_user")
    (tmp_path / ".wrds_user").write_text("someuser")

    def _no_backend(*a, **kw):
        raise mod.keyring.errors.NoKeyringError("No recommended backend was available")

    monkeypatch.setattr(mod.keyring, "get_password", _no_backend)

    with pytest.raises(mod.keyring.errors.NoKeyringError) as excinfo:
        mod.get_wrds_credentials()

    msg = str(excinfo.value)
    assert "WRDS_USERNAME" in msg
    assert "WRDS_PASSWORD" in msg
    assert "JKP_ALLOW_PLAINTEXT_KEYRING" in msg


@pytest.mark.unit
def test_reset_credentials_missing_backend_raises_guidance(monkeypatch, tmp_path):
    """`jkp connect --reset` on a headless system must surface the same guidance
    rather than the bare NoKeyringError from keyring.delete_password."""
    monkeypatch.delenv("JKP_ALLOW_PLAINTEXT_KEYRING", raising=False)

    import jkp.data.wrds_credentials as mod

    monkeypatch.setattr(mod, "LAST_USER_FILE", tmp_path / ".wrds_user")
    (tmp_path / ".wrds_user").write_text("someuser")

    def _no_backend(*a, **kw):
        raise mod.keyring.errors.NoKeyringError("No recommended backend was available")

    monkeypatch.setattr(mod.keyring, "delete_password", _no_backend)

    with pytest.raises(mod.keyring.errors.NoKeyringError) as excinfo:
        mod.reset_credentials(full_reset=True)

    msg = str(excinfo.value)
    assert "JKP_ALLOW_PLAINTEXT_KEYRING" in msg


@pytest.mark.unit
def test_reset_credentials_deletes_existing_entry(monkeypatch, tmp_path, capsys):
    """A successful delete reports the deletion and removes the username file."""
    monkeypatch.delenv("JKP_ALLOW_PLAINTEXT_KEYRING", raising=False)

    import jkp.data.wrds_credentials as mod

    user_file = tmp_path / ".wrds_user"
    monkeypatch.setattr(mod, "LAST_USER_FILE", user_file)
    user_file.write_text("someuser")

    deleted = []
    monkeypatch.setattr(mod.keyring, "delete_password", lambda *a, **kw: deleted.append(a))

    mod.reset_credentials(full_reset=True)

    assert deleted, "delete_password should have been called"
    out = capsys.readouterr().out
    assert "Deleted password for 'someuser'" in out
    assert not user_file.exists(), "username file should be removed"


@pytest.mark.unit
def test_reset_credentials_missing_entry_is_handled(monkeypatch, tmp_path, capsys):
    """A genuine 'no entry to delete' (PasswordDeleteError) is still handled
    gracefully and must not be swallowed by the NoKeyringError translation."""
    monkeypatch.delenv("JKP_ALLOW_PLAINTEXT_KEYRING", raising=False)

    import jkp.data.wrds_credentials as mod

    monkeypatch.setattr(mod, "LAST_USER_FILE", tmp_path / ".wrds_user")
    (tmp_path / ".wrds_user").write_text("someuser")

    def _no_entry(*a, **kw):
        raise mod.keyring.errors.PasswordDeleteError("not found")

    monkeypatch.setattr(mod.keyring, "delete_password", _no_entry)

    mod.reset_credentials(full_reset=True)  # must not raise

    out = capsys.readouterr().out
    assert "No keyring entry found" in out
