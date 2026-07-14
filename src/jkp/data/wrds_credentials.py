"""WRDS credential resolution.

Username and password are resolved independently:

* **Username** — ``WRDS_USERNAME`` → saved state file → interactive prompt.
* **Password** — ``WRDS_PASSWORD`` → system keyring → ``$PGPASSFILE`` / ``~/.pgpass``.

When the password comes from a libpq password file, jkp never handles it: the
connection string omits ``password=`` and libpq reads the file itself
(:attr:`Credentials.password` is ``None``). The resolved source is printed to
stderr on every run so it is always clear which one was used.

On a headless node with no system keyring daemon, provision credentials by
running ``jkp connect`` on a login node (it writes ``~/.pgpass``), by exporting
``WRDS_USERNAME``/``WRDS_PASSWORD``, or by creating ``~/.pgpass`` directly — it
is the standard Postgres/WRDS mechanism and any tool can populate it.

Migration (one-time, automatic, non-interactive): a legacy plaintext-keyring
entry (``~/.local/share/python_keyring/keyring_pass.cfg``, written by the removed
``JKP_ALLOW_PLAINTEXT_KEYRING`` opt-in) is moved to ``~/.pgpass`` — reading only
jkp's own ``[WRDS]`` entry and deleting only that entry, never the shared file.
The username cache moves from ``~/.wrds_user`` to a per-OS state directory.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import configparser
import contextlib
import getpass
import io
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import keyring
import keyring.errors
import platformdirs
from keyring.util.platform_ import data_root as _keyring_data_root

# WRDS Postgres endpoint. Shared with aux_functions.gen_wrds_connection_info so
# the conninfo host/port/db and the ~/.pgpass match key can never drift apart.
WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = "9737"
WRDS_DB = "wrds"

SERVICE_NAME = "WRDS"  # keyring service name and legacy plaintext-keyring section

ENV_USERNAME = "WRDS_USERNAME"
ENV_PASSWORD = "WRDS_PASSWORD"
ENV_PGPASSFILE = "PGPASSFILE"

# Username cache: a per-OS state file (auto-migrated from the old ~/.wrds_user).
LAST_USER_FILE = Path(platformdirs.user_state_dir("jkp")) / "wrds_user"
_LEGACY_USER_FILE = Path.home() / ".wrds_user"

# Legacy plaintext-keyring store. Its location is decided by *keyring's* own path
# logic (keyrings.alt writes to ``keyring.util.platform_.data_root()``), which
# differs from platformdirs on macOS/Windows — so we must use keyring's function,
# not platformdirs, to find the file keyrings.alt actually wrote.
_LEGACY_KEYRING_FILE = Path(_keyring_data_root()) / "keyring_pass.cfg"


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str | None  # None => authenticate via ~/.pgpass (omit password=)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _interactive() -> bool:
    return sys.stdin.isatty()


def _log_source(desc: str) -> None:
    """Announce which credential source authenticated (never the secret itself)."""
    print(f"Using WRDS credentials from {desc}.", file=sys.stderr, flush=True)


def _persist_username(username: str) -> None:
    LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_USER_FILE.write_text(username)


def _atomic_write(path: Path, text: str, mode: int = 0o600) -> None:
    """Write ``text`` to ``path`` atomically with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- #
# System keyring (encrypted OS store only; no file-backed fallback)
# --------------------------------------------------------------------------- #
def _keyring_get(username: str) -> str | None:
    try:
        return keyring.get_password(SERVICE_NAME, username)
    except keyring.errors.NoKeyringError:
        return None  # no backend at all — the normal headless case
    except keyring.errors.KeyringError as exc:
        # Backend present but unusable (a locked SecretService/KWallet on headless
        # SSH, or an init failure). Warn so a genuinely-stored-but-locked password
        # isn't silently ignored, then fall through to the other sources.
        print(
            f"Warning: the system keyring is present but unusable ({exc}); "
            "falling back to other credential sources.",
            file=sys.stderr,
            flush=True,
        )
        return None


def _keyring_set(username: str, password: str) -> bool:
    """Store the password in the system keyring; return False if it is unusable."""
    try:
        keyring.set_password(SERVICE_NAME, username, password)
        return True
    except keyring.errors.KeyringError:
        return False


def _keyring_delete(username: str) -> bool | None:
    """Delete the keyring entry.

    Returns True if one was removed, False when there is definitively nothing to
    delete (no entry, or no keyring backend at all — in both cases nothing
    survives), and None when the backend is present but unreachable (locked): the
    entry may still exist, so we warn and let the caller avoid claiming a clean
    success it can't guarantee.
    """
    try:
        keyring.delete_password(SERVICE_NAME, username)
        return True
    except (keyring.errors.NoKeyringError, keyring.errors.PasswordDeleteError):
        return False  # no keyring, or nothing to delete
    except keyring.errors.KeyringError as exc:
        print(
            f"Warning: could not reach the system keyring ({exc}); a stored entry may still exist.",
            file=sys.stderr,
            flush=True,
        )
        return None


# --------------------------------------------------------------------------- #
# libpq password file (~/.pgpass / $PGPASSFILE)
# --------------------------------------------------------------------------- #
def _pgpass_path() -> Path:
    """The file libpq reads: $PGPASSFILE if set, else the per-OS default."""
    env = os.environ.get(ENV_PGPASSFILE)
    if env:
        return Path(env)
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "postgresql" / "pgpass.conf"
    return Path.home() / ".pgpass"


def _perms_too_open(path: Path) -> bool:
    """True if libpq would ignore the file for lax permissions (Unix only)."""
    if sys.platform.startswith("win"):
        return False
    return bool(path.stat().st_mode & 0o077)


def _split_pgpass(line: str) -> list[str]:
    """Split a .pgpass line into its fields, honoring backslash escapes."""
    fields: list[str] = []
    cur: list[str] = []
    escaped = False
    for ch in line:
        if escaped:
            cur.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ":":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return fields


def _esc_pgpass(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:")


def _pgpass_line(username: str, password: str) -> str:
    return ":".join(_esc_pgpass(v) for v in (WRDS_HOST, WRDS_PORT, WRDS_DB, username, password))


def _is_wrds_line(fields: list[str], username: str) -> bool:
    """True if a parsed line is jkp's own exact WRDS entry for ``username``."""
    return (
        len(fields) == 5
        and fields[0] == WRDS_HOST
        and fields[1] == WRDS_PORT
        and fields[2] == WRDS_DB
        and fields[3] == username
    )


def _pgpass_has_entry(username: str) -> bool:
    """True if libpq would find a usable WRDS password for ``username``.

    Returns False (rather than raising) if the file cannot be read — an
    unreadable ~/.pgpass is, for libpq's purposes, no usable entry.
    """
    path = _pgpass_path()
    try:
        if not path.exists():
            return False
        if _perms_too_open(path):
            print(
                f"Warning: {path} has permissions looser than 0600; libpq ignores it. "
                f"Run: chmod 600 {path}",
                file=sys.stderr,
                flush=True,
            )
            return False
        lines = path.read_text().splitlines()
    except OSError:
        return False
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = _split_pgpass(stripped)
        if len(fields) != 5:
            continue
        host, port, db, user, _pw = fields
        if (
            host in ("*", WRDS_HOST)
            and port in ("*", WRDS_PORT)
            and db in ("*", WRDS_DB)
            and user in ("*", username)
        ):
            return True
    return False


def _write_pgpass(username: str, password: str) -> Path:
    """Append or replace jkp's WRDS line, leaving every other entry untouched."""
    if "\n" in password or "\r" in password:
        # A .pgpass entry is one line; a newline in the password would inject a
        # spurious second line. libpq's format cannot represent it, so reject it.
        raise ValueError("WRDS password contains a newline, which ~/.pgpass cannot store.")
    path = _pgpass_path()
    new_line = _pgpass_line(username, password)
    out: list[str] = []
    replaced = False
    if path.exists():
        for raw in path.read_text().splitlines():
            if _is_wrds_line(_split_pgpass(raw.strip()), username):
                if not replaced:
                    out.append(new_line)
                    replaced = True
                # drop any duplicate WRDS lines for this user
            else:
                out.append(raw)
    if not replaced:
        out.append(new_line)
    _atomic_write(path, "\n".join(out) + "\n")
    return path


def _remove_pgpass_entry(username: str) -> bool:
    """Remove jkp's WRDS line; return True if one was removed. Preserves others."""
    path = _pgpass_path()
    if not path.exists():
        return False
    out: list[str] = []
    removed = False
    for raw in path.read_text().splitlines():
        if _is_wrds_line(_split_pgpass(raw.strip()), username):
            removed = True
        else:
            out.append(raw)
    if not removed:
        return False
    if any(line.strip() for line in out):
        _atomic_write(path, "\n".join(out) + "\n")
    else:
        with contextlib.suppress(OSError):
            path.unlink()
    return True


# --------------------------------------------------------------------------- #
# One-time migration off the legacy plaintext keyring
# --------------------------------------------------------------------------- #
def _escape_keyring_option(username: str) -> str:
    """Reproduce keyrings.alt's option-key escaping for a username.

    keyrings.alt stores each option name as ``escape_for_ini(username)`` — every
    non-alphanumeric character becomes ``_`` followed by the zero-padded 2-digit
    hex of each of its UTF-8 bytes (``_%02X``; e.g. ``a.b`` → ``a_2Eb``,
    tab → ``_09``) — and configparser then lowercases it. We escape here (using
    lowercase hex, which configparser's ``optionxform`` also produces) so
    migration finds entries for usernames containing dots, hyphens, uppercase, or
    non-ASCII characters, not just plain lowercase-alnum ones.
    """
    return re.sub(
        r"[^A-Za-z0-9]",
        lambda m: "".join(f"_{b:02x}" for b in m.group(0).encode("utf-8")),
        username,
    )


def _drop_legacy_keyring_section(cp: configparser.RawConfigParser, path: Path) -> None:
    """Remove only jkp's ``[WRDS]`` section, preserving any other tools' sections;
    delete the file if ours was the last section."""
    cp.remove_section(SERVICE_NAME)
    if cp.sections():
        buf = io.StringIO()
        cp.write(buf)
        _atomic_write(path, buf.getvalue())
    else:
        with contextlib.suppress(OSError):
            path.unlink()


def _cleanup_legacy_keyring_section() -> None:
    """Best-effort removal of a lingering ``[WRDS]`` section from the legacy
    plaintext keyring — used when the password now comes from another source (env
    or system keyring) so the obsolete plaintext copy does not linger on disk.
    Never raises."""
    path = _LEGACY_KEYRING_FILE
    if not path.exists():
        return
    try:
        cp = configparser.RawConfigParser()
        cp.read(path, encoding="utf-8")
        if cp.has_section(SERVICE_NAME):
            _drop_legacy_keyring_section(cp, path)
    except (configparser.Error, UnicodeDecodeError, OSError):
        pass


def _migrate_legacy_keyring(username: str) -> None:
    """Move jkp's legacy plaintext-keyring entry to ~/.pgpass, non-interactively.

    ``keyring_pass.cfg`` is the shared keyrings.alt store; its values are plain
    ``base64(password)`` (PlaintextKeyring does no encryption). We read only the
    ``[WRDS]`` section, write the equivalent ~/.pgpass line, and delete only that
    section — never the file, which may hold other tools' credentials.

    An existing ~/.pgpass entry for the user is their current credential and is
    never overwritten; in that case we only drop the superseded legacy section.
    """
    path = _LEGACY_KEYRING_FILE
    if not path.exists():
        return
    cp = configparser.RawConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError):
        # A corrupt or non-UTF8 legacy file is not migratable; skip it and let
        # normal resolution (env / ~/.pgpass / prompt) proceed. (An OSError on
        # read propagates to the best-effort wrapper in get_wrds_credentials.)
        return
    if not cp.has_section(SERVICE_NAME):
        return

    # If ~/.pgpass already resolves for this user, that is their current
    # credential — never clobber it with the (possibly stale) legacy value.
    # Just drop the now-superseded legacy [WRDS] section.
    if _pgpass_has_entry(username):
        _drop_legacy_keyring_section(cp, path)
        print(
            "Removed a superseded WRDS entry from the legacy plaintext keyring; "
            "kept your existing ~/.pgpass entry.",
            file=sys.stderr,
            flush=True,
        )
        return

    value = None
    option = _escape_keyring_option(username)
    if cp.has_option(SERVICE_NAME, option):
        value = cp.get(SERVICE_NAME, option)
    else:
        items = cp.items(SERVICE_NAME)
        if len(items) == 1:
            # Exactly one stored user: take it even if the key doesn't match the
            # resolved username. jkp only ever stored the current user's entry,
            # so this rescues a renamed/re-escaped username. Risk if the file
            # somehow holds one *other* user's entry: its password migrates under
            # the resolved username and auth fails (recoverable via `jkp connect`).
            value = items[0][1]
    if not value:
        return
    try:
        password = base64.decodebytes(value.encode()).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return

    try:
        pgpass = _write_pgpass(username, password)
    except ValueError as exc:
        # The legacy password can't be stored in ~/.pgpass (e.g. it contains a
        # newline) — it's unusable, and libpq could never authenticate with it.
        # Drop the dead section so we don't decode-and-fail on every future run,
        # and let resolution surface a clean "no credentials" path instead.
        _drop_legacy_keyring_section(cp, path)
        print(
            f"Discarded an unusable legacy WRDS credential ({exc}); "
            "run `jkp connect` to re-provision.",
            file=sys.stderr,
            flush=True,
        )
        return
    _drop_legacy_keyring_section(cp, path)
    print(
        f"Migrated WRDS credential from the legacy plaintext keyring to {pgpass}.",
        file=sys.stderr,
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Username / password resolution
# --------------------------------------------------------------------------- #
def _resolve_username(env_user: str | None = None) -> str:
    if env_user:
        return env_user.strip()
    if LAST_USER_FILE.exists():
        return LAST_USER_FILE.read_text().strip()
    if _LEGACY_USER_FILE.exists():  # migrate ~/.wrds_user -> state dir
        username = _LEGACY_USER_FILE.read_text().strip()
        _persist_username(username)
        with contextlib.suppress(OSError):
            _LEGACY_USER_FILE.unlink()
        return username
    if _interactive():
        username = input(f"Username for {SERVICE_NAME}: ").strip()
        _persist_username(username)
        return username
    raise RuntimeError(
        f"No WRDS username available and no terminal to prompt at. Set "
        f"{ENV_USERNAME}, or run `jkp connect` on a login node to store one."
    )


def _prompt_and_store(username: str) -> Credentials:
    """Interactively obtain a password and persist it to the best available store."""
    password = getpass.getpass(f"Password or token for {username} at {SERVICE_NAME}: ")
    if _keyring_set(username, password):
        print(
            f"Stored WRDS password for '{username}' in the system keyring.",
            file=sys.stderr,
            flush=True,
        )
        return Credentials(username, password)
    pgpass = _write_pgpass(username, password)
    print(
        f"No system keyring available; wrote WRDS credentials to {pgpass} (mode 600).",
        file=sys.stderr,
        flush=True,
    )
    return Credentials(username, None)


def _no_credentials_error(username: str) -> RuntimeError:
    return RuntimeError(
        f"No WRDS password found for '{username}' and no terminal to prompt at. "
        f"Provide credentials one of these ways:\n"
        f"  - set {ENV_USERNAME} and {ENV_PASSWORD} in the environment, or\n"
        f"  - run `jkp connect` on a login node (writes ~/.pgpass on headless "
        f"machines), or\n"
        f"  - create a ~/.pgpass / $PGPASSFILE entry for {WRDS_HOST}."
    )


def get_wrds_credentials() -> Credentials:
    """Resolve WRDS credentials following the documented precedence order.

    Steps:
      1. If ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` are both set, use those.
      2. Resolve the username (env, saved state file, or prompt).
      3. Password: ``WRDS_PASSWORD`` → system keyring → ``~/.pgpass``/``$PGPASSFILE``.
      4. If nothing is found and a terminal is available, prompt and store;
         otherwise raise with actionable guidance.
    """
    env_user = os.environ.get(ENV_USERNAME)
    env_pw = os.environ.get(ENV_PASSWORD)
    if env_user and env_pw:
        # Literal env-var names in the log message (not the ENV_* constants) so no
        # "password"-named identifier flows into a logging sink — the value is
        # never logged, only the source label.
        _log_source("the WRDS_USERNAME/WRDS_PASSWORD environment variables")
        return Credentials(env_user.strip(), env_pw)

    username = _resolve_username(env_user)

    if env_pw:
        _log_source("the WRDS_PASSWORD environment variable")
        return Credentials(username, env_pw)

    keyring_pw = _keyring_get(username)
    if keyring_pw:
        # The keyring supersedes any legacy plaintext copy; clean it up.
        _cleanup_legacy_keyring_section()
        _log_source("the system keyring")
        return Credentials(username, keyring_pw)

    # One-time migration off the legacy plaintext keyring, reached only when
    # neither an env password nor a system-keyring entry supplied the password.
    # It no-ops unless a legacy keyring_pass.cfg with a [WRDS] entry exists. A
    # migration I/O failure must not abort resolution: env / existing ~/.pgpass /
    # an interactive prompt may still provide credentials.
    try:
        _migrate_legacy_keyring(username)
    except (OSError, ValueError) as exc:
        print(
            f"Warning: could not migrate the legacy WRDS credential ({exc}); "
            "continuing without it.",
            file=sys.stderr,
            flush=True,
        )

    if _pgpass_has_entry(username):
        _log_source(f"{_pgpass_path()} (libpq password file)")
        return Credentials(username, None)

    if _interactive():
        return _prompt_and_store(username)

    raise _no_credentials_error(username)


def reset_credentials(full_reset: bool = False) -> None:
    """Clear the stored username and (optionally) the stored password.

    With ``full_reset``, removes the password from both the system keyring and
    jkp's ``~/.pgpass`` line (leaving any other ``.pgpass`` entries intact).
    """
    username: str | None = None
    for user_file in (LAST_USER_FILE, _LEGACY_USER_FILE):
        if user_file.exists():
            if username is None:
                username = user_file.read_text().strip()
            user_file.unlink()

    if username is None:
        print("No stored username found — nothing to reset.")
        return
    print(f"Removed stored username '{username}'")

    if full_reset:
        removed_keyring = _keyring_delete(username)
        removed_pgpass = _remove_pgpass_entry(username)
        # Also drop any legacy plaintext-keyring [WRDS] section: otherwise the
        # next resolution would migrate the just-revoked password back into
        # ~/.pgpass, undoing the reset.
        _cleanup_legacy_keyring_section()
        if removed_keyring:
            print(f"Deleted password for '{username}' from the system keyring")
        if removed_pgpass:
            print(f"Removed WRDS entry for '{username}' from {_pgpass_path()}")
        # Only claim nothing was found when the keyring answer was definitive
        # (False, not None): an unreachable backend already warned that an entry
        # may survive, so a "nothing found" line here would contradict it.
        if removed_keyring is False and not removed_pgpass:
            print(f"No stored password found for '{username}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage stored WRDS credentials.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove stored username and password (keyring and ~/.pgpass entry).",
    )
    args = parser.parse_args()

    if args.reset:
        reset_credentials(full_reset=True)
    else:
        creds = get_wrds_credentials()
        print(f"Using credentials for '{creds.username}'")
