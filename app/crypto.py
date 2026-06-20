"""Application-level secret encryption (Fernet / MultiFernet).

Secrets are encrypted in the app before they reach Postgres and decrypted only at
point of use, so the encryption key never touches the database. The master key(s)
come from APP_ENCRYPTION_KEYS (comma-separated, newest first) so rotation is just:
prepend a new key, re-encrypt rows lazily, drop the old key.
"""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.config import get_settings

# Bump when a new key is prepended to APP_ENCRYPTION_KEYS. Stored per-row so we can
# tell which key version encrypted a value; MultiFernet still decrypts across all.
KEY_VERSION = 1


class SecretConfigError(RuntimeError):
    pass


def _multifernet() -> MultiFernet:
    keys = get_settings().encryption_keys
    if not keys:
        raise SecretConfigError(
            "APP_ENCRYPTION_KEYS is not set. Generate one with:\n"
            '  python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"'
        )
    return MultiFernet([Fernet(k.encode()) for k in keys])


def encrypt(plaintext: str) -> bytes:
    return _multifernet().encrypt(plaintext.encode())


def decrypt(token: bytes) -> str:
    try:
        return _multifernet().decrypt(bytes(token)).decode()
    except InvalidToken as exc:  # wrong / rotated-out key
        raise SecretConfigError(
            "Failed to decrypt secret - the encrypting key may have been removed "
            "from APP_ENCRYPTION_KEYS."
        ) from exc


def last4(secret: str) -> str:
    """Non-sensitive display suffix, e.g. for showing 'sk-...a3f9' in the UI."""
    return secret[-4:] if len(secret) >= 4 else "*" * len(secret)


class Secret:
    """Wrapper that refuses to reveal itself in logs, reprs, or tracebacks.

    Call .reveal() explicitly at the exact point the plaintext is needed.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__
