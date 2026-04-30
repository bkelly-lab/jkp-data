"""WRDS credential resolution.

Credential precedence (highest first):

1. ``WRDS_USERNAME`` / ``WRDS_PASSWORD`` environment variables. Useful for
   container deployments and shared service accounts where the source of
   truth shouldn't be a per-user filesystem location.
2. The system keyring (Keychain on macOS, Secret Service on Linux desktop,
   Credential Vault on Windows). The default for interactive sessions.
3. The file-backed keyring (``keyrings.alt.file.PlaintextKeyring``), which
   stores the password in a mode-600 file under ``~/.local/share/python_keyring/``.
   This is appropriate for headless environments (HPC compute nodes,
   minimal Docker images) where no system keyring daemon is available — the
   stored secret has the same security posture as any other 600-mode dotfile
   in the user's home directory.

   This backend is not selected automatically. The previous implementation
   set ``PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring`` at
   module-import time, which silently changed the keyring backend for every
   process that imported anything from ``jkp.data`` — surprising, undocumented,
   and unnecessary on machines where a system keyring is available. Now the
   user must opt in with ``JKP_ALLOW_PLAINTEXT_KEYRING=1``, and the swap
   happens only inside this module's credential-resolution functions, never
   at import time.
"""

from __future__ import annotations

import argparse
import getpass
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import keyring

SERVICE_NAME = "WRDS"
LAST_USER_FILE = Path.home() / ".wrds_user"  # remembers last username

# Environment variable names.
ENV_USERNAME = "WRDS_USERNAME"
ENV_PASSWORD = "WRDS_PASSWORD"
ENV_ALLOW_PLAINTEXT = "JKP_ALLOW_PLAINTEXT_KEYRING"


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


def _maybe_use_file_keyring() -> None:
    """If the user explicitly opts in, switch the keyring backend to the
    file-backed ``keyrings.alt.file.PlaintextKeyring``.

    Called at credential-resolution time, never at import. The backend stores
    the password in a mode-600 file under ``~/.local/share/python_keyring/``;
    the security posture is the same as any other user-private dotfile.
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

    _maybe_use_file_keyring()

    if LAST_USER_FILE.exists():
        username = LAST_USER_FILE.read_text().strip()
    else:
        username = input(f"Username for {SERVICE_NAME}: ").strip()
        LAST_USER_FILE.write_text(username)

    password = keyring.get_password(SERVICE_NAME, username)

    if not password:
        password = getpass.getpass(f"Password or token for {username} at {SERVICE_NAME}: ")
        keyring.set_password(SERVICE_NAME, username, password)
        print(f"Stored credentials for '{username}' in keyring under '{SERVICE_NAME}'")

    return Credentials(username, password)


def reset_credentials(full_reset: bool = False) -> None:
    """Clear the stored username and (optionally) remove the password from the keyring."""
    _maybe_use_file_keyring()
    if LAST_USER_FILE.exists():
        username = LAST_USER_FILE.read_text().strip()
        LAST_USER_FILE.unlink()
        print(f"Removed stored username '{username}'")

        if full_reset:
            try:
                keyring.delete_password(SERVICE_NAME, username)
                print(f"Deleted password for '{username}' from keyring under '{SERVICE_NAME}'")
            except keyring.errors.PasswordDeleteError:
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
