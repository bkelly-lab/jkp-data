"""WRDS credential resolution.

Username and password are resolved independently:

* **Username** — ``WRDS_USERNAME`` → saved state file → interactive prompt.
* **Password** — ``WRDS_PASSWORD`` → system keyring → ``$PGPASSFILE`` / ``~/.pgpass``.

When the password comes from a libpq password file, jkp never handles it: the
connection string omits ``password=`` and libpq reads the file itself
(:attr:`Credentials.password` is ``None``). The resolved source is printed to
stderr on every run so it is always clear which one was used.

On a headless node with no system keyring daemon, provision credentials by
running ``jkp connect`` on a login node (it stores both the username and the
``~/.pgpass`` line) or by exporting ``WRDS_USERNAME``/``WRDS_PASSWORD``. A
hand-created ``~/.pgpass`` supplies only the password, so it must be paired with
``WRDS_USERNAME`` — resolution never reads the username out of the file.

Migration (one-time, automatic, non-interactive): a legacy plaintext-keyring
entry (``~/.local/share/python_keyring/keyring_pass.cfg``, written by the removed
``JKP_ALLOW_PLAINTEXT_KEYRING`` opt-in) is moved to ``~/.pgpass`` — reading only
jkp's own ``[WRDS]`` entry and deleting only that entry, never the shared file.
The username cache moves from ``~/.wrds_user`` to a per-OS state directory.
"""

from __future__ import annotations

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

    def __repr__(self) -> str:
        # Mask the password so tools that render frame locals via repr() (pretty
        # tracebacks, pytest --showlocals, debuggers) cannot leak the secret. The
        # None vs non-None distinction is preserved because password=None marks
        # ~/.pgpass authentication, which is useful to see when debugging. The
        # username is intentionally not masked (the CLI already prints it).
        pw = "***" if self.password is not None else None
        return f"Credentials(username={self.username!r}, password={pw!r})"


def _interactive() -> bool:
    return sys.stdin.isatty()


def _log_source(desc: str) -> None:
    """Announce which credential source authenticated (never the secret itself)."""
    print(f"Using WRDS credentials from {desc}.", file=sys.stderr, flush=True)


def _persist_username(username: str) -> None:
    LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_USER_FILE.write_text(username)


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, with 0600 permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        if hasattr(os, "fchmod"):
            # Enforce 0600 explicitly rather than relying on mkstemp's default:
            # this file may hold a plaintext password, so the mode must be
            # guaranteed even if that default ever changes.
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if path.is_symlink():
            # os.replace swaps the symlink itself for the new regular file, so its
            # original target is left untouched — and, for ~/.pgpass, may still
            # hold the old WRDS entry (e.g. in a dotfiles copy). We don't write
            # through the link (that would push a secret into a tracked tree), but
            # we warn so the detachment isn't silent.
            print(
                f"Note: {path} is a symlink; replacing it with a regular file. Its "
                "original target is now detached and may still contain the old entry.",
                file=sys.stderr,
                flush=True,
            )
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


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


def _pgpass_scan(username: str, *, check_perms: bool = True) -> tuple[bool, bool]:
    """Scan the pgpass file for WRDS entries, returning ``(matched, has_exact)``.

    ``matched`` is whether libpq would find a usable WRDS password (via an exact
    line or a wildcard); ``has_exact`` is whether the *first* matching line — the
    one libpq actually uses — is jkp's own exact entry (so `jkp connect` can
    manage it). A wildcard sitting above an exact line therefore yields
    ``(True, False)``. With ``check_perms`` (the default), a too-open file counts
    as no entry (libpq ignores it); pass ``check_perms=False`` to ask only whether
    an entry exists. Returns ``(False, False)`` if the file is missing, unreadable,
    or not valid UTF-8 (jkp reads it as UTF-8 and warns in that last case).
    """
    path = _pgpass_path()
    try:
        if not path.exists():
            return False, False
        if check_perms and _perms_too_open(path):
            print(
                f"Warning: {path} has permissions looser than 0600; libpq ignores it. "
                f"Run: chmod 600 {path}",
                file=sys.stderr,
                flush=True,
            )
            return False, False
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        # jkp reads the file as UTF-8 — a deliberate choice, since libpq itself is
        # byte-oriented and mandates no encoding. A file another tool wrote in a
        # different encoding isn't "invalid"; jkp just can't read it here, so
        # ignore it (with a warning) rather than crash the whole resolution.
        # Use the literal env-var name here, not the ENV_PASSWORD constant: CodeQL's
        # clear-text-logging check is name-based, so a "password"-named identifier
        # flowing into this stderr sink is a false positive (cf. get_wrds_credentials).
        print(
            f"Warning: jkp reads {path} as UTF-8, but it is not valid UTF-8, so jkp "
            "cannot use it. Set WRDS_PASSWORD or use the system keyring.",
            file=sys.stderr,
            flush=True,
        )
        return False, False
    except OSError:
        return False, False
    matched = has_exact = False
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
            # libpq uses the FIRST matching line, so only that one decides whether
            # the effective credential is jkp's exact entry or a wildcard. A
            # wildcard sitting above jkp's exact line wins, and jkp cannot manage
            # it — exactly the case the wildcard warning needs to catch.
            matched = True
            has_exact = _is_wrds_line(fields, username)
            break
    return matched, has_exact


def _pgpass_has_entry(username: str, *, check_perms: bool = True) -> bool:
    """True if libpq would find a usable WRDS password for ``username`` (exact or
    wildcard). Pass ``check_perms=False`` to ask only whether an entry exists,
    ignoring whether the file's permissions would make libpq skip it."""
    return _pgpass_scan(username, check_perms=check_perms)[0]


def _read_pgpass_lines(path: Path) -> list[str]:
    """Read an existing pgpass file's lines for a read-modify-write, raising an
    actionable ``RuntimeError`` rather than a raw traceback when it can't be read.

    jkp reads the file as UTF-8 — a deliberate choice, since libpq itself is
    byte-oriented and mandates no ``.pgpass`` encoding. A file another tool wrote
    in a different encoding isn't "invalid"; jkp simply can't edit it in place, so
    say so and point at the alternatives instead of clobbering it.
    """
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"jkp reads {path} as UTF-8, but it is not valid UTF-8 ({exc}); "
            "re-save it as UTF-8, or use WRDS_PASSWORD or the system keyring."  # literal, not ENV_PASSWORD
        ) from exc
    except OSError as exc:
        # e.g. a root-owned ~/.pgpass from an old sudo — surface an actionable
        # message rather than a raw error after the user typed their password.
        raise RuntimeError(
            f"Cannot read existing {path} ({exc}); fix its permissions and retry."
        ) from exc


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
        existing = _read_pgpass_lines(path)
        for raw in existing:
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
    lines = _read_pgpass_lines(path)
    out: list[str] = []
    removed = False
    for raw in lines:
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


def _drop_legacy_keyring_option(
    cp: configparser.RawConfigParser, path: Path, username: str
) -> bool:
    """Remove only ``username``'s entry from jkp's ``[WRDS]`` section, preserving
    any other usernames' entries in it. Drop the ``[WRDS]`` section — and delete
    the file if that leaves no other tools' sections — only once that was its last
    entry. Returns True if an entry was actually removed.

    (The whole ``[WRDS]`` section can hold one option per WRDS username, so removing
    the section wholesale would destroy other users' never-migrated passwords.)
    """
    option = _escape_keyring_option(username)
    if not cp.has_option(SERVICE_NAME, option):
        return False
    cp.remove_option(SERVICE_NAME, option)
    if not cp.options(SERVICE_NAME):
        cp.remove_section(SERVICE_NAME)
    if cp.sections():
        buf = io.StringIO()
        cp.write(buf)
        _atomic_write(path, buf.getvalue())
    else:
        with contextlib.suppress(OSError):
            path.unlink()
    return True


def _load_legacy_cfg() -> configparser.RawConfigParser | None:
    """Load the legacy plaintext-keyring file if it exists and has a ``[WRDS]``
    section; return None otherwise (missing, unreadable, unparseable, or no
    section). Never raises."""
    path = _LEGACY_KEYRING_FILE
    if not path.exists():
        return None
    cp = configparser.RawConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError, OSError):
        return None
    return cp if cp.has_section(SERVICE_NAME) else None


def _cleanup_legacy_keyring_section(username: str) -> None:
    """Best-effort removal of ``username``'s lingering entry from the legacy
    plaintext keyring's ``[WRDS]`` section — used when the password now comes from
    another source (env or system keyring) so the obsolete plaintext copy does not
    linger. Other usernames' entries are preserved. Never raises."""
    cp = _load_legacy_cfg()
    if cp is None:
        return
    with contextlib.suppress(OSError):
        _drop_legacy_keyring_option(cp, _LEGACY_KEYRING_FILE, username)


def _migrate_legacy_keyring(username: str) -> None:
    """Move jkp's legacy plaintext-keyring entry to ~/.pgpass, non-interactively.

    ``keyring_pass.cfg`` is the shared keyrings.alt store; its values are plain
    ``base64(password)`` (PlaintextKeyring does no encryption). We read only this
    user's option in the ``[WRDS]`` section, write the equivalent ~/.pgpass line,
    and delete only that option — never other usernames' entries or the file
    itself, which may hold other tools' credentials.

    An existing ~/.pgpass entry for the user is their current credential and is
    never overwritten; in that case we only drop this user's superseded legacy
    entry.
    """
    cp = _load_legacy_cfg()
    if cp is None:
        # No file, no [WRDS] section, or unparseable/unreadable: nothing to
        # migrate — let normal resolution (env / ~/.pgpass / prompt) proceed.
        return
    path = _LEGACY_KEYRING_FILE

    # If ~/.pgpass already resolves for this user, that is their current
    # credential — never clobber it with the (possibly stale) legacy value.
    # Just drop this user's now-superseded legacy entry (only announce it if there
    # actually was one — another user's entry may be all that remains).
    if _pgpass_has_entry(username, check_perms=False):
        if _drop_legacy_keyring_option(cp, path, username):
            print(
                "Removed a superseded WRDS entry from the legacy plaintext keyring; "
                "kept your existing ~/.pgpass entry.",
                file=sys.stderr,
                flush=True,
            )
        return

    # Only migrate jkp's own exact-key entry. Since jkp always stored the entry
    # via keyring.set_password (the same escaping _escape_keyring_option
    # reproduces), the exact-key lookup covers everything jkp ever wrote; a
    # fallback for a mismatched key would only serve a hypothetical rename and
    # risks migrating another user's password under the resolved username.
    option = _escape_keyring_option(username)
    if not cp.has_option(SERVICE_NAME, option):
        return
    value = cp.get(SERVICE_NAME, option)
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
        # Drop this user's dead entry so we don't decode-and-fail on every future
        # run, and let resolution surface a clean "no credentials" path instead.
        _drop_legacy_keyring_option(cp, path, username)
        print(
            f"Discarded an unusable legacy WRDS credential ({exc}); "
            "run `jkp connect` to re-provision.",
            file=sys.stderr,
            flush=True,
        )
        return
    _drop_legacy_keyring_option(cp, path, username)
    print(
        f"Migrated WRDS credential from the legacy plaintext keyring to {pgpass}.",
        file=sys.stderr,
        flush=True,
    )


def _resolve_username(env_user: str | None = None) -> str:
    if env_user and env_user.strip():  # ignore a whitespace-only WRDS_USERNAME
        return env_user.strip()
    if LAST_USER_FILE.exists():
        cached = LAST_USER_FILE.read_text().strip()
        if cached:  # treat an empty cache as missing rather than user ''
            return cached
    if _LEGACY_USER_FILE.exists():  # migrate ~/.wrds_user -> state dir
        username = _LEGACY_USER_FILE.read_text().strip()
        if username:  # an empty legacy file is treated as missing, not cached as ''
            _persist_username(username)
            with contextlib.suppress(OSError):
                _LEGACY_USER_FILE.unlink()
            return username
    if _interactive():
        username = input(f"Username for {SERVICE_NAME}: ").strip()
        if not username:  # don't cache an empty username entered by mistake
            raise RuntimeError(f"Empty username entered for {SERVICE_NAME}.")
        _persist_username(username)
        return username
    raise RuntimeError(
        f"No WRDS username available and no terminal to prompt at. Set "
        f"{ENV_USERNAME}, or run `jkp connect` on a login node to store one."
    )


def _prompt_and_store(username: str) -> Credentials:
    """Interactively obtain a password and persist it to the best available store."""
    password = getpass.getpass(f"Password or token for {username} at {SERVICE_NAME}: ")
    if not password:
        # An empty password would be stored as a junk ``…:username:`` pgpass line
        # that later "resolves" while auth silently fails — reject it, mirroring
        # the empty-username guard.
        raise RuntimeError(f"Empty password entered for {username}; nothing stored.")
    # Cache the username alongside the stored password, so a credential provisioned
    # for an env-provided WRDS_USERNAME (which _resolve_username does not persist)
    # is still revocable via `jkp connect --reset` rather than orphaned.
    _persist_username(username)
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
    # Strip before the truthiness check so WRDS_USERNAME="   " is treated as
    # unset rather than accepted as username="".
    env_user = (os.environ.get(ENV_USERNAME) or "").strip() or None
    env_pw = os.environ.get(ENV_PASSWORD)
    if env_user and env_pw:
        # Literal env-var names in the log message (not the ENV_* constants) so no
        # "password"-named identifier flows into a logging sink — the value is
        # never logged, only the source label.
        _log_source("the WRDS_USERNAME/WRDS_PASSWORD environment variables")
        return Credentials(env_user, env_pw)  # env_user already stripped/non-empty

    username = _resolve_username(env_user)

    if env_pw:
        _log_source("the WRDS_PASSWORD environment variable")
        return Credentials(username, env_pw)

    keyring_pw = _keyring_get(username)
    if keyring_pw:
        # The keyring supersedes any legacy plaintext copy; clean it up.
        _cleanup_legacy_keyring_section(username)
        _log_source("the system keyring")
        return Credentials(username, keyring_pw)

    # One-time migration off the legacy plaintext keyring, reached only when
    # neither an env password nor a system-keyring entry supplied the password.
    # It no-ops unless a legacy keyring_pass.cfg with a [WRDS] entry exists. A
    # migration I/O failure must not abort resolution: env / existing ~/.pgpass /
    # an interactive prompt may still provide credentials.
    #
    # Note: if the keyring was merely *locked* (not absent), we reached here
    # without consulting it, so migrating the legacy plaintext copy could activate
    # a password the user has since rotated in the keyring. That's inherent to
    # falling through a locked keyring; `jkp connect --reset` clears the legacy
    # copy, and the ~/.pgpass entry (if any) is preferred over the legacy value.
    try:
        _migrate_legacy_keyring(username)
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            f"Warning: could not migrate the legacy WRDS credential ({exc}); "
            "continuing without it.",
            file=sys.stderr,
            flush=True,
        )

    matched, has_exact = _pgpass_scan(username)
    if matched:
        _log_source(f"{_pgpass_path()} (libpq password file)")
        if not has_exact:
            print(
                f"Warning: WRDS is served by a catch-all entry in {_pgpass_path()}, "
                "not a WRDS-specific line, so `jkp connect` cannot set or change it.",
                file=sys.stderr,
                flush=True,
            )
        return Credentials(username, None)

    if _interactive():
        return _prompt_and_store(username)

    raise _no_credentials_error(username)


def reset_credentials(full_reset: bool = False) -> None:
    """Clear the stored username and (optionally) the stored password.

    With ``full_reset``, removes the password from both the system keyring and
    jkp's ``~/.pgpass`` line (leaving any other ``.pgpass`` entries intact).
    """
    # Read the username but do NOT delete the cache files yet (see the unlink at
    # the end). Treat an empty/whitespace cache as missing, so we don't reset
    # user '' and don't consume the legacy ~/.wrds_user unread.
    username: str | None = None
    for user_file in (LAST_USER_FILE, _LEGACY_USER_FILE):
        if user_file.exists() and username is None:
            username = user_file.read_text().strip() or None

    if username is None:
        print("No stored username found — nothing to reset.")
        return

    if full_reset:
        removed_keyring = _keyring_delete(username)
        # Drop any legacy plaintext-keyring [WRDS] section before removing the
        # ~/.pgpass line (otherwise the next resolution would migrate the
        # just-revoked password back into ~/.pgpass). Cleanup never raises, so
        # doing it first means a failure removing an unreadable ~/.pgpass can't
        # skip it and leave a plaintext copy behind on a security-motivated reset.
        _cleanup_legacy_keyring_section(username)
        if removed_keyring:
            print(f"Deleted password for '{username}' from the system keyring")
        removed_pgpass = _remove_pgpass_entry(username)
        if removed_pgpass:
            print(f"Removed WRDS entry for '{username}' from {_pgpass_path()}")
        # Only claim nothing was found when the keyring answer was definitive
        # (False, not None): an unreachable backend already warned that an entry
        # may survive, so a "nothing found" line here would contradict it.
        if removed_keyring is False and not removed_pgpass:
            print(f"No stored password found for '{username}'")

    # Remove the username cache last: if a password removal above raised (e.g. an
    # unreadable ~/.pgpass), the cache survives so a retry can still find the
    # username and finish the reset instead of orphaning the stored password.
    for user_file in (LAST_USER_FILE, _LEGACY_USER_FILE):
        with contextlib.suppress(FileNotFoundError):
            user_file.unlink()
    print(f"Removed stored username '{username}'")
