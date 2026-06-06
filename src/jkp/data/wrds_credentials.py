"""WRDS credential resolution.

Credentials are resolved in a defined precedence order: environment variables,
then the system keyring, then an opt-in file-backed keyring. Run
``jkp connect --help`` for the full order and the ``JKP_ALLOW_PLAINTEXT_KEYRING``
opt-in details.

The file-backed keyring is never selected automatically: the user opts in with
``JKP_ALLOW_PLAINTEXT_KEYRING=1``, and the swap happens only inside this module's
credential-resolution functions — importing ``jkp.data`` never changes the
process-wide keyring backend.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import getpass
import os
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import keyring
import keyring.errors

SERVICE_NAME = "WRDS"
LAST_USER_FILE = Path.home() / ".wrds_user"  # remembers last username

# Environment variable names.
ENV_USERNAME = "WRDS_USERNAME"
ENV_PASSWORD = "WRDS_PASSWORD"
ENV_ALLOW_PLAINTEXT = "JKP_ALLOW_PLAINTEXT_KEYRING"

# Shown when no keyring backend is available, in place of keyring's generic
# NoKeyringError ("No recommended backend was available ..."), which gives the
# user no way forward. The env-var names are interpolated (not their values) so
# the guidance stays in sync if a name ever changes.
_NO_BACKEND_GUIDANCE = (
    "No usable keyring backend is available on this system. On a headless "
    "machine (HPC compute node, minimal container) choose one of:\n"
    f"  - set {ENV_USERNAME} and {ENV_PASSWORD} in the environment, or\n"
    f"  - set {ENV_ALLOW_PLAINTEXT}=1 to use a mode-600 file-backed keyring "
    "under ~/.local/share/python_keyring/."
)


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


@functools.cache
def _ensure_keyring_backend() -> None:
    """Select the keyring backend for this process, applying the opt-in
    file-backed swap if (and only if) the user asked for it.

    Opt-in: set the ``JKP_ALLOW_PLAINTEXT_KEYRING`` environment variable to
    exactly the string ``"1"``. Any other value (including ``"true"``,
    ``"yes"``, ``"True"``, or ``"0"``) is treated as not set and the system
    keyring is used. We use a strict comparison rather than truthy parsing so
    that the opt-in is unambiguous and matches the documented form.

    Cached so the swap runs at most once per process: every keyring operation
    routes through this, but the env var is read and the warning emitted only on
    the first call. Called at credential-resolution time, never at import. The
    backend stores the password in a mode-600 file under
    ``~/.local/share/python_keyring/``; the security posture is the same as any
    other user-private dotfile.
    """
    if os.environ.get(ENV_ALLOW_PLAINTEXT) != "1":
        return
    warnings.warn(
        f"{ENV_ALLOW_PLAINTEXT}=1 is set: switching the keyring backend to "
        "keyrings.alt.file.PlaintextKeyring for this process. The WRDS "
        "password will be read from / written to a mode-600 file under "
        "~/.local/share/python_keyring/.",
        stacklevel=2,
    )
    from keyrings.alt.file import PlaintextKeyring  # noqa: PLC0415

    keyring.set_keyring(PlaintextKeyring())


@dataclass(frozen=True)
class _WrdsKeyring:
    """WRDS-scoped wrapper over the keyring library.

    The one place that speaks keyring's vocabulary for credential storage:
    callers get/set/delete the WRDS password without naming ``SERVICE_NAME`` or
    handling keyring's exception types. Handed out only via ``_keyring_backend``.
    """

    def get_password(self, username: str) -> str | None:
        return keyring.get_password(SERVICE_NAME, username)

    def set_password(self, username: str, password: str) -> None:
        keyring.set_password(SERVICE_NAME, username, password)

    def delete_password(self, username: str) -> bool:
        """Delete the stored password; return ``False`` if there was nothing to
        delete (or the keyring declined).

        ``NoKeyringError`` is intentionally not caught here, so the
        missing-backend case still propagates to ``_keyring_backend`` for
        translation rather than being reported as an absent entry.
        """
        try:
            keyring.delete_password(SERVICE_NAME, username)
        except keyring.errors.PasswordDeleteError:
            return False
        return True


@contextlib.contextmanager
def _keyring_backend() -> Iterator[_WrdsKeyring]:
    """Single entry point for any keyring access.

    Ensures the opt-in backend is selected (once), then yields a WRDS-scoped
    handle for credential storage. Translates keyring's generic
    ``NoKeyringError`` into actionable guidance pointing at the env-var and
    plaintext-opt-in escape hatches. All keyring access goes through the yielded
    handle, so callers never touch the keyring library directly.
    """
    _ensure_keyring_backend()
    try:
        yield _WrdsKeyring()
    except keyring.errors.NoKeyringError as exc:
        raise keyring.errors.NoKeyringError(_NO_BACKEND_GUIDANCE) from exc


def _credentials_from_env() -> Credentials | None:
    """Return env-var credentials if both ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` are set."""
    user = os.environ.get(ENV_USERNAME)
    password = os.environ.get(ENV_PASSWORD)
    if user and password:
        return Credentials(user, password)
    return None


def get_wrds_credentials() -> Credentials:
    """Resolve WRDS credentials following the documented precedence order.

    Steps:
      1. If ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` are both set, return those.
      2. Otherwise, look up the saved username in ``~/.wrds_user`` (prompt
         interactively on first run).
      3. Fetch the password from the (possibly plaintext-opt-in) keyring;
         prompt and store on first run.
    """
    env_creds = _credentials_from_env()
    if env_creds is not None:
        return env_creds

    if LAST_USER_FILE.exists():
        username = LAST_USER_FILE.read_text().strip()
    else:
        username = input(f"Username for {SERVICE_NAME}: ").strip()
        LAST_USER_FILE.write_text(username)

    with _keyring_backend() as kr:
        password = kr.get_password(username)

        if not password:
            password = getpass.getpass(f"Password or token for {username} at {SERVICE_NAME}: ")
            kr.set_password(username, password)
            print(f"Stored credentials for '{username}' in keyring under '{SERVICE_NAME}'")

    return Credentials(username, password)


def reset_credentials(full_reset: bool = False) -> None:
    """Clear the stored username and (optionally) remove the password from the keyring."""
    if LAST_USER_FILE.exists():
        username = LAST_USER_FILE.read_text().strip()
        LAST_USER_FILE.unlink()
        print(f"Removed stored username '{username}'")

        if full_reset:
            with _keyring_backend() as kr:
                if kr.delete_password(username):
                    print(f"Deleted password for '{username}' from keyring under '{SERVICE_NAME}'")
                else:
                    print(f"No keyring entry found for '{username}'")

    else:
        print("No stored username found — nothing to reset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage stored wrds credentials.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove both stored username and password from keyring.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_credentials(full_reset=args.reset)
    else:
        creds = get_wrds_credentials()
        print(f"Using credentials for '{creds.username}'")
