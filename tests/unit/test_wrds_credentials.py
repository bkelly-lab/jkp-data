"""Tests for WRDS credential resolution.

See ``jkp connect --help`` for the canonical credential precedence order;
these tests exercise it: WRDS_USERNAME/WRDS_PASSWORD env, then the system
keyring, then a libpq password file (``$PGPASSFILE`` / ``~/.pgpass``), plus the
one-time migration off the legacy plaintext keyring.
"""

from __future__ import annotations

import base64

import pytest


@pytest.fixture(autouse=True)
def _isolate_credential_state(monkeypatch, tmp_path):
    """Point every credential path at a temp dir and clear the WRDS env vars, so
    no test can read or write the developer's real keyring / ~/.pgpass / state."""
    import jkp.data.wrds_credentials as mod

    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)

    pgpass = tmp_path / "pgpass"
    monkeypatch.setenv("PGPASSFILE", str(pgpass))
    monkeypatch.setattr(mod, "LAST_USER_FILE", tmp_path / "state" / "wrds_user")
    monkeypatch.setattr(mod, "_LEGACY_USER_FILE", tmp_path / ".wrds_user")
    monkeypatch.setattr(mod, "_LEGACY_KEYRING_FILE", tmp_path / "keyring_pass.cfg")

    # Default: no keyring backend and non-interactive, unless a test overrides.
    # Stubbing the keyring here (not just per-test) is what actually keeps tests
    # off the developer's real keyring, as this fixture's docstring promises.
    def _no_keyring(*_a, **_kw):
        raise mod.keyring.errors.NoKeyringError("no keyring backend in tests")

    monkeypatch.setattr(mod.keyring, "get_password", _no_keyring)
    monkeypatch.setattr(mod.keyring, "set_password", _no_keyring)
    monkeypatch.setattr(mod.keyring, "delete_password", _no_keyring)
    monkeypatch.setattr(mod, "_interactive", lambda: False)
    return mod


# --------------------------------------------------------------------------- #
# Environment variables
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_env_vars_take_precedence(monkeypatch, _isolate_credential_state):
    """Both env vars set → use them without touching the keyring."""
    mod = _isolate_credential_state
    monkeypatch.setenv("WRDS_USERNAME", "ci-user")
    monkeypatch.setenv("WRDS_PASSWORD", "ci-secret")

    def _boom(*a, **kw):
        raise AssertionError("keyring must not be queried when env vars are set")

    monkeypatch.setattr(mod.keyring, "get_password", _boom)

    creds = mod.get_wrds_credentials()
    assert creds.username == "ci-user"
    assert creds.password == "ci-secret"


@pytest.mark.unit
def test_env_username_alone_used_with_keyring_password(monkeypatch, _isolate_credential_state):
    """WRDS_USERNAME alone is now honored as the username source (decoupled from
    the password), with the password coming from the keyring."""
    mod = _isolate_credential_state
    monkeypatch.setenv("WRDS_USERNAME", "env-user")
    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: "kr-pw")

    creds = mod.get_wrds_credentials()
    assert creds.username == "env-user"
    assert creds.password == "kr-pw"


# --------------------------------------------------------------------------- #
# Username resolution
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_username_from_state_file(monkeypatch, _isolate_credential_state):
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("saved-user")
    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: "pw")

    creds = mod.get_wrds_credentials()
    assert creds.username == "saved-user"


@pytest.mark.unit
def test_legacy_username_file_migrates_to_state_dir(monkeypatch, _isolate_credential_state):
    """A legacy ~/.wrds_user is moved into the new per-OS state file."""
    mod = _isolate_credential_state
    mod._LEGACY_USER_FILE.write_text("legacy-user")
    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: "pw")

    creds = mod.get_wrds_credentials()

    assert creds.username == "legacy-user"
    assert mod.LAST_USER_FILE.read_text().strip() == "legacy-user"
    assert not mod._LEGACY_USER_FILE.exists(), "legacy username file should be removed"


# --------------------------------------------------------------------------- #
# Password sources
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_password_from_keyring(monkeypatch, _isolate_credential_state):
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("u")
    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: "kr-secret")

    creds = mod.get_wrds_credentials()
    assert creds.password == "kr-secret"


@pytest.mark.unit
def test_locked_keyring_warns_and_falls_through(monkeypatch, _isolate_credential_state, capsys):
    """A present-but-locked keyring (KeyringError, not NoKeyringError) must warn
    and fall through to ~/.pgpass rather than crash resolution."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.KeyringLocked))
    _write_pgpass(mod, "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:secret\n")

    creds = mod.get_wrds_credentials()

    assert creds.password is None  # served from ~/.pgpass
    assert "keyring is present but unusable" in capsys.readouterr().err


@pytest.mark.unit
def test_prompt_stores_to_pgpass_when_keyring_locked(monkeypatch, _isolate_credential_state):
    """The *store* side of a locked keyring: if set_password raises a KeyringError
    (not NoKeyringError), _prompt_and_store must fall through to ~/.pgpass rather
    than crash — the write-side mirror of the locked-keyring read fallback."""
    mod = _isolate_credential_state
    monkeypatch.setattr(mod.keyring, "set_password", _raises(mod.keyring.errors.KeyringLocked))
    monkeypatch.setattr("getpass.getpass", lambda *a, **kw: "typed-secret")

    creds = mod._prompt_and_store("testuser")

    assert creds.username == "testuser"
    assert creds.password is None  # written to ~/.pgpass, not returned
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:" in mod._pgpass_path().read_text()


@pytest.mark.unit
def test_password_from_pgpass_returns_none_password(monkeypatch, _isolate_credential_state):
    """A matching ~/.pgpass entry → Credentials.password is None (libpq reads it)."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    # no keyring backend
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))
    _write_pgpass(mod, "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:secret\n")

    creds = mod.get_wrds_credentials()
    assert creds.username == "testuser"
    assert creds.password is None


@pytest.mark.unit
def test_pgpass_ignored_when_permissions_too_open(monkeypatch, _isolate_credential_state):
    """libpq ignores a group/world-readable .pgpass; so must our detection."""
    import sys

    if sys.platform.startswith("win"):
        pytest.skip("permission check is Unix-only")
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))
    path = _write_pgpass(mod, "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:secret\n")
    path.chmod(0o644)  # too open

    assert mod._pgpass_has_entry("testuser") is False
    with pytest.raises(RuntimeError, match="No WRDS password found"):
        mod.get_wrds_credentials()


@pytest.mark.unit
def test_no_credentials_non_interactive_raises_guidance(monkeypatch, _isolate_credential_state):
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("u")
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    with pytest.raises(RuntimeError) as excinfo:
        mod.get_wrds_credentials()
    msg = str(excinfo.value)
    assert "WRDS_USERNAME" in msg
    assert ".pgpass" in msg


# --------------------------------------------------------------------------- #
# .pgpass writing (append/replace, never clobber other entries)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_write_pgpass_preserves_other_entries(_isolate_credential_state):
    mod = _isolate_credential_state
    _write_pgpass(mod, "otherhost:5432:otherdb:someone:otherpw\n")

    mod._write_pgpass("testuser", "s3cret")

    contents = mod._pgpass_path().read_text()
    assert "otherhost:5432:otherdb:someone:otherpw" in contents
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:s3cret" in contents


@pytest.mark.unit
def test_write_pgpass_replaces_our_line_in_place(_isolate_credential_state):
    mod = _isolate_credential_state
    mod._write_pgpass("testuser", "old")
    mod._write_pgpass("testuser", "new")

    lines = [
        ln
        for ln in mod._pgpass_path().read_text().splitlines()
        if "wrds-pgdata" in ln and ":testuser:" in ln
    ]
    assert lines == ["wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:new"]


@pytest.mark.unit
def test_write_pgpass_is_mode_600(_isolate_credential_state):
    import sys

    if sys.platform.startswith("win"):
        pytest.skip("permission check is Unix-only")
    mod = _isolate_credential_state
    path = mod._write_pgpass("testuser", "s3cret")
    assert (path.stat().st_mode & 0o777) == 0o600


@pytest.mark.unit
def test_pgpass_escapes_special_characters(_isolate_credential_state):
    """A password containing ':' round-trips: it is escaped on write and libpq's
    matcher still finds the entry (the has-entry check unescapes)."""
    mod = _isolate_credential_state
    mod._write_pgpass("testuser", "pa:ss\\word")
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:pa\\:ss\\\\word" in (
        mod._pgpass_path().read_text()
    )
    assert mod._pgpass_has_entry("testuser") is True


# --------------------------------------------------------------------------- #
# Legacy plaintext-keyring migration
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_legacy_keyring_migrates_to_pgpass_and_removes_only_our_section(
    monkeypatch, _isolate_credential_state
):
    """The migration seeds ~/.pgpass from the plaintext keyring's [WRDS] entry
    (plain base64, no keyrings-alt) and deletes ONLY that section — a second
    tool's section must survive."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")

    b64 = base64.encodebytes(b"legacy-secret").decode()
    other = base64.encodebytes(b"other-tool").decode()
    mod._LEGACY_KEYRING_FILE.write_text(
        f"[WRDS]\ntestuser = {b64}\n[some-other-service]\nbob = {other}\n"
    )
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    creds = mod.get_wrds_credentials()

    assert creds.username == "testuser"
    assert creds.password is None  # now served from ~/.pgpass
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:legacy-secret" in (
        mod._pgpass_path().read_text()
    )

    import configparser

    cp = configparser.RawConfigParser()
    cp.read(mod._LEGACY_KEYRING_FILE, encoding="utf-8")
    assert not cp.has_section("WRDS"), "our section must be removed"
    assert cp.has_section("some-other-service"), "other tools' section must survive"


@pytest.mark.unit
def test_legacy_keyring_file_removed_when_only_our_section(monkeypatch, _isolate_credential_state):
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    b64 = base64.encodebytes(b"legacy-secret").decode()
    mod._LEGACY_KEYRING_FILE.write_text(f"[WRDS]\ntestuser = {b64}\n")
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    mod.get_wrds_credentials()

    assert not mod._LEGACY_KEYRING_FILE.exists(), "empty keyring file should be removed"


# --------------------------------------------------------------------------- #
# reset_credentials
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_reset_removes_username_keyring_and_pgpass(monkeypatch, _isolate_credential_state, capsys):
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    mod._write_pgpass("testuser", "s3cret")
    _write_pgpass_append(mod, "otherhost:5432:db:bob:pw\n")

    deleted = []
    monkeypatch.setattr(mod.keyring, "delete_password", lambda *a, **kw: deleted.append(a))

    mod.reset_credentials(full_reset=True)

    out = capsys.readouterr().out
    assert "Removed stored username 'testuser'" in out
    assert not mod.LAST_USER_FILE.exists()
    assert deleted, "keyring delete should have been attempted"
    # our pgpass line gone, the other tool's line preserved
    contents = mod._pgpass_path().read_text()
    assert ":testuser:" not in contents
    assert "otherhost:5432:db:bob:pw" in contents


@pytest.mark.unit
def test_reset_no_username_is_handled(_isolate_credential_state, capsys):
    mod = _isolate_credential_state
    mod.reset_credentials(full_reset=True)  # must not raise
    assert "nothing to reset" in capsys.readouterr().out


@pytest.mark.unit
def test_reset_clears_legacy_keyring_section(monkeypatch, _isolate_credential_state):
    """full_reset must remove a lingering legacy [WRDS] section, so the next
    resolution cannot migrate the just-revoked password back into ~/.pgpass —
    while preserving any other tool's section."""
    import base64
    import configparser

    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    cp = configparser.RawConfigParser()
    cp.add_section("WRDS")
    cp.set("WRDS", "testuser", "\n" + base64.encodebytes(b"revoked").decode())
    cp.add_section("other")
    cp.set("other", "bob", "\n" + base64.encodebytes(b"x").decode())
    with mod._LEGACY_KEYRING_FILE.open("w") as fh:
        cp.write(fh)
    monkeypatch.setattr(mod.keyring, "delete_password", lambda *a, **kw: None)

    mod.reset_credentials(full_reset=True)

    check = configparser.RawConfigParser()
    check.read(mod._LEGACY_KEYRING_FILE)
    assert not check.has_section("WRDS"), "revoked legacy section must be removed"
    assert check.has_section("other"), "other tools' section must survive"


@pytest.mark.unit
def test_reset_warns_when_keyring_unreachable(monkeypatch, _isolate_credential_state, capsys):
    """A locked/unreachable keyring during --reset must warn that an entry may
    survive, rather than silently reporting a clean success."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    monkeypatch.setattr(mod.keyring, "delete_password", _raises(mod.keyring.errors.KeyringLocked))

    mod.reset_credentials(full_reset=True)

    captured = capsys.readouterr()
    assert "a stored entry may still exist" in captured.err
    # ...and the run must not also claim nothing was found (that would contradict
    # the warning): the keyring answer here is "unknown", not "definitively empty".
    assert "No stored password found" not in captured.out + captured.err


# --------------------------------------------------------------------------- #
# Migration — real on-disk format, escaped keys, fallback, I/O isolation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_migration_handles_real_keyring_on_disk_format(monkeypatch, _isolate_credential_state):
    """keyrings.alt writes the value on a tab-indented continuation line with a
    leading newline; the migration must decode that real format, not just a
    single-line convenience form."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    _write_real_keyring(mod, {"testuser": "real-secret"})
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    creds = mod.get_wrds_credentials()

    assert creds.password is None
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:real-secret" in (
        mod._pgpass_path().read_text()
    )


@pytest.mark.unit
def test_migration_matches_escaped_username_key(monkeypatch, _isolate_credential_state):
    """A username with a dot is stored by keyrings.alt under an escaped key
    (``test.user`` -> ``test_2euser``). With two stored users the single-entry
    fallback can't rescue a bad match, so the escaped-key lookup must work."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("test.user")
    # Keys hard-coded (not via the module's own escaper) so the test is an
    # independent check of the escaping, not a tautology.
    _write_real_keyring(mod, {"test_2euser": "dotted-secret", "otheruser": "other-pw"})
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    creds = mod.get_wrds_credentials()

    assert creds.password is None
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:test.user:dotted-secret" in (
        mod._pgpass_path().read_text()
    )


@pytest.mark.unit
def test_migration_single_entry_fallback(monkeypatch, _isolate_credential_state):
    """When the resolved username doesn't match the stored key but exactly one
    WRDS entry exists, migrate it under the resolved username."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("resolved-user")
    _write_real_keyring(mod, {"testuser": "the-secret"})  # key != resolved username
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    creds = mod.get_wrds_credentials()

    assert creds.password is None
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:resolved-user:the-secret" in (
        mod._pgpass_path().read_text()
    )


@pytest.mark.unit
def test_migration_io_failure_does_not_abort_resolution(
    monkeypatch, _isolate_credential_state, capsys
):
    """A migration I/O error must warn and fall through, not abort resolution
    when a usable ~/.pgpass fallback exists."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    _write_pgpass(mod, "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:fallback\n")
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    def _boom(_username):
        raise OSError("disk full")

    monkeypatch.setattr(mod, "_migrate_legacy_keyring", _boom)

    creds = mod.get_wrds_credentials()  # must not raise

    assert creds.password is None  # served from the fallback pgpass
    assert "could not migrate" in capsys.readouterr().err


@pytest.mark.unit
def test_migration_corrupt_non_utf8_keyring_falls_through(monkeypatch, _isolate_credential_state):
    """A non-UTF8/corrupt legacy keyring_pass.cfg must be skipped (not abort
    resolution) so a usable ~/.pgpass fallback still serves credentials."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    mod._LEGACY_KEYRING_FILE.write_bytes(b"[WRDS]\ntestuser = \xff\xfe not utf8\n")
    _write_pgpass(mod, "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:fallback\n")
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    creds = mod.get_wrds_credentials()  # must not raise

    assert creds.password is None
    assert "testuser:fallback" in mod._pgpass_path().read_text()


@pytest.mark.unit
def test_migration_preserves_existing_pgpass_entry(monkeypatch, _isolate_credential_state):
    """If ~/.pgpass already has a WRDS entry (the user's current credential), the
    migration must NOT overwrite it with the stale legacy-keyring value — it only
    drops the superseded legacy [WRDS] section."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    _write_pgpass(mod, "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:CURRENT_PW\n")
    _write_real_keyring(mod, {"testuser": "STALE_PW"})
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    creds = mod.get_wrds_credentials()

    assert creds.password is None
    contents = mod._pgpass_path().read_text()
    assert "testuser:CURRENT_PW" in contents, "current pgpass password must be preserved"
    assert "STALE_PW" not in contents, "stale legacy password must not overwrite it"

    import configparser

    cp = configparser.RawConfigParser()
    cp.read(mod._LEGACY_KEYRING_FILE)
    assert not cp.has_section("WRDS"), "superseded legacy section should still be cleaned up"


@pytest.mark.unit
def test_write_pgpass_rejects_newline_password(_isolate_credential_state):
    """A newline in the password would inject a spurious .pgpass line; reject it."""
    mod = _isolate_credential_state
    with pytest.raises(ValueError, match="newline"):
        mod._write_pgpass("testuser", "line1\nwildcard:injected")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("username", "expected"),
    [
        ("testuser", "testuser"),
        ("test.user", "test_2euser"),
        ("a-b", "a_2db"),
        ("a\tb", "a_09b"),  # control byte -> zero-padded hex
        ("josé", "jos_c3_a9"),  # non-ASCII -> UTF-8 bytes
    ],
)
def test_escape_keyring_option_format(_isolate_credential_state, username, expected):
    """Exact reproduction of keyrings.alt's option-key escaping (_%02x per byte)."""
    assert _isolate_credential_state._escape_keyring_option(username) == expected


# --------------------------------------------------------------------------- #
# .pgpass wildcard matching + preservation of foreign lines/comments/blanks
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "*:*:*:testuser:secret",
        "wrds-pgdata.wharton.upenn.edu:9737:wrds:*:secret",
        "*:*:*:*:secret",
    ],
)
def test_pgpass_wildcard_entry_matches(_isolate_credential_state, line):
    mod = _isolate_credential_state
    _write_pgpass(mod, line + "\n")
    assert mod._pgpass_has_entry("testuser") is True


@pytest.mark.unit
def test_write_pgpass_preserves_comments_blanks_and_wildcards(_isolate_credential_state):
    mod = _isolate_credential_state
    _write_pgpass(
        mod,
        "# my pgpass\n\notherhost:5432:db:bob:pw\n*:*:*:wild:wpw\n",
    )

    mod._write_pgpass("testuser", "s3cret")

    lines = mod._pgpass_path().read_text().splitlines()
    assert "# my pgpass" in lines
    assert "" in lines, "blank line should be preserved"
    assert "otherhost:5432:db:bob:pw" in lines
    assert "*:*:*:wild:wpw" in lines
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:testuser:s3cret" in lines


@pytest.mark.unit
def test_remove_pgpass_entry_preserves_comments_and_others(_isolate_credential_state):
    mod = _isolate_credential_state
    _write_pgpass(mod, "# keep me\notherhost:5432:db:bob:pw\n")
    mod._write_pgpass("testuser", "s3cret")

    assert mod._remove_pgpass_entry("testuser") is True

    contents = mod._pgpass_path().read_text()
    assert "# keep me" in contents
    assert "otherhost:5432:db:bob:pw" in contents
    assert ":testuser:" not in contents


@pytest.mark.unit
def test_pgpass_escapes_special_username(_isolate_credential_state):
    """A username containing ':' / '\\' is escaped on write and still matched."""
    mod = _isolate_credential_state
    mod._write_pgpass("we:ird\\name", "pw")
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:we\\:ird\\\\name:pw" in (
        mod._pgpass_path().read_text()
    )
    assert mod._pgpass_has_entry("we:ird\\name") is True


# --------------------------------------------------------------------------- #
# Interactive prompt-and-store (keyring vs .pgpass fallback)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_prompt_and_store_uses_keyring_when_available(monkeypatch, _isolate_credential_state):
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("u")
    monkeypatch.setattr(mod, "_interactive", lambda: True)
    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: None)
    stored = {}
    monkeypatch.setattr(mod.keyring, "set_password", lambda s, u, p: stored.update({u: p}))
    monkeypatch.setattr(mod.getpass, "getpass", lambda *a, **kw: "typed-pw")

    creds = mod.get_wrds_credentials()

    assert creds.password == "typed-pw"
    assert stored == {"u": "typed-pw"}
    assert not mod._pgpass_path().exists(), "keyring path must not write ~/.pgpass"


@pytest.mark.unit
def test_prompt_and_store_falls_back_to_pgpass_without_keyring(
    monkeypatch, _isolate_credential_state
):
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("u")
    monkeypatch.setattr(mod, "_interactive", lambda: True)
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))
    monkeypatch.setattr(mod.keyring, "set_password", _raises(mod.keyring.errors.NoKeyringError))
    monkeypatch.setattr(mod.getpass, "getpass", lambda *a, **kw: "typed-pw")

    creds = mod.get_wrds_credentials()

    assert creds.password is None  # libpq reads it from ~/.pgpass
    assert "wrds-pgdata.wharton.upenn.edu:9737:wrds:u:typed-pw" in (mod._pgpass_path().read_text())


@pytest.mark.unit
def test_empty_env_vars_treated_as_unset(monkeypatch, _isolate_credential_state):
    """Empty-string WRDS_USERNAME/WRDS_PASSWORD must be treated as unset, not as
    a blank username/password."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("fileuser")
    monkeypatch.setenv("WRDS_USERNAME", "")
    monkeypatch.setenv("WRDS_PASSWORD", "")
    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: "kr-pw")

    creds = mod.get_wrds_credentials()
    assert creds.username == "fileuser"
    assert creds.password == "kr-pw"


@pytest.mark.unit
def test_password_from_env_with_username_from_file(monkeypatch, _isolate_credential_state):
    """WRDS_PASSWORD alone (username from the state file) is honored without
    touching the keyring."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("fileuser")
    monkeypatch.setenv("WRDS_PASSWORD", "envpw")

    def _boom(*a, **kw):
        raise AssertionError("keyring must not be queried when WRDS_PASSWORD is set")

    monkeypatch.setattr(mod.keyring, "get_password", _boom)

    creds = mod.get_wrds_credentials()
    assert creds.username == "fileuser"
    assert creds.password == "envpw"


@pytest.mark.unit
def test_keyring_password_cleans_up_lingering_legacy_section(
    monkeypatch, _isolate_credential_state
):
    """When the system keyring supplies the password, a lingering legacy [WRDS]
    section is cleaned up — but other tools' sections are preserved."""
    import base64
    import configparser

    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    cp = configparser.RawConfigParser()
    cp.add_section("WRDS")
    cp.set("WRDS", "testuser", "\n" + base64.encodebytes(b"old-plaintext").decode())
    cp.add_section("other")
    cp.set("other", "bob", "\n" + base64.encodebytes(b"x").decode())
    with mod._LEGACY_KEYRING_FILE.open("w") as fh:
        cp.write(fh)
    monkeypatch.setattr(mod.keyring, "get_password", lambda *a, **kw: "kr-pw")

    creds = mod.get_wrds_credentials()

    assert creds.password == "kr-pw"  # keyring wins
    check = configparser.RawConfigParser()
    check.read(mod._LEGACY_KEYRING_FILE)
    assert not check.has_section("WRDS"), "superseded legacy section should be cleaned up"
    assert check.has_section("other"), "other tools' section must survive"


@pytest.mark.unit
def test_legacy_keyring_path_uses_keyring_convention():
    """The legacy file was written by keyrings.alt, which uses keyring's
    data_root() (not platformdirs) — matters on macOS/Windows where the two
    diverge. Guards against a silent revert to a platformdirs-derived path."""
    from keyring.util.platform_ import data_root

    import jkp.data.wrds_credentials as mod

    assert mod._keyring_data_root is data_root


@pytest.mark.unit
def test_migration_newline_legacy_password_discarded_and_idempotent(
    monkeypatch, _isolate_credential_state
):
    """A legacy password that ~/.pgpass cannot store (contains a newline) is
    discarded and its dead section dropped — so resolution doesn't loop forever
    decoding-then-failing, and a second run is a clean no-op."""
    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    _write_real_keyring(mod, {"testuser": "line1\nline2"})
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    with pytest.raises(RuntimeError, match="No WRDS password found"):
        mod.get_wrds_credentials()

    assert not mod._LEGACY_KEYRING_FILE.exists(), "dead section should be dropped"
    # idempotent: nothing left to re-process
    with pytest.raises(RuntimeError, match="No WRDS password found"):
        mod.get_wrds_credentials()


@pytest.mark.unit
def test_migration_defers_to_wildcard_pgpass_entry(monkeypatch, _isolate_credential_state):
    """A catch-all ~/.pgpass entry already covers WRDS (libpq wildcard semantics),
    so migration preserves the user's file and drops the superseded legacy section
    rather than overwriting. Documents this intentional trade-off."""
    import configparser

    mod = _isolate_credential_state
    mod.LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LAST_USER_FILE.write_text("testuser")
    _write_pgpass(mod, "*:*:*:*:existing-pw\n")
    _write_real_keyring(mod, {"testuser": "legacy-pw"})
    monkeypatch.setattr(mod.keyring, "get_password", _raises(mod.keyring.errors.NoKeyringError))

    creds = mod.get_wrds_credentials()

    assert creds.password is None  # libpq serves it from the wildcard entry
    assert mod._pgpass_path().read_text().strip() == "*:*:*:*:existing-pw"
    assert "legacy-pw" not in mod._pgpass_path().read_text()
    cp = configparser.RawConfigParser()
    cp.read(mod._LEGACY_KEYRING_FILE)
    assert not cp.has_section("WRDS")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_real_keyring(mod, entries):
    """Write a keyring_pass.cfg the way keyrings.alt does — values on tab-indented
    continuation lines with a leading newline, via configparser — so the parsing
    test reflects what actually lands on users' disks. ``entries`` maps the
    (already keyrings.alt-escaped) option key to the plaintext password."""
    import configparser

    cp = configparser.RawConfigParser()
    cp.add_section(mod.SERVICE_NAME)
    for key, password in entries.items():
        cp.set(mod.SERVICE_NAME, key, "\n" + base64.encodebytes(password.encode()).decode())
    with mod._LEGACY_KEYRING_FILE.open("w") as fh:
        cp.write(fh)


def _raises(exc):
    def _fn(*a, **kw):
        raise exc("no backend")

    return _fn


def _write_pgpass(mod, content):
    path = mod._pgpass_path()
    path.write_text(content)
    if not __import__("sys").platform.startswith("win"):
        path.chmod(0o600)
    return path


def _write_pgpass_append(mod, content):
    path = mod._pgpass_path()
    with path.open("a") as fh:
        fh.write(content)
