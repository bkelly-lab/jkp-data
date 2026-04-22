import os

os.environ["PYTHON_KEYRING_BACKEND"] = "keyrings.alt.file.PlaintextKeyring"
import argparse
import getpass
from dataclasses import dataclass
from pathlib import Path

import keyring

SERVICE_NAME = "WRDS"
LAST_USER_FILE = Path.home() / ".wrds_user"  # remembers last username


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


def get_wrds_credentials() -> Credentials:
    """
    Automatically retrieves credentials for wrds.
    - On first run: asks for username and password/token, stores them.
    - On later runs: loads both silently from the system keyring.
    """
    # Try to remember the last-used username
    if LAST_USER_FILE.exists():
        username = LAST_USER_FILE.read_text().strip()
    else:
        username = input(f"Username for {SERVICE_NAME}: ").strip()
        LAST_USER_FILE.write_text(username)

    # Try to retrieve the stored password for this username
    password = keyring.get_password(SERVICE_NAME, username)

    # If not found, prompt once and store it securely
    if not password:
        password = getpass.getpass(f"Password or token for {username} at {SERVICE_NAME}: ")
        keyring.set_password(SERVICE_NAME, username, password)
        print(f"Stored credentials for '{username}' in keyring under '{SERVICE_NAME}'")

    return Credentials(username, password)


def reset_credentials(full_reset: bool = False):
    """
    Clears stored username and optionally removes password from keyring.
    """
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
