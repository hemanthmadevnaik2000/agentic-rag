import os

from cryptography.fernet import Fernet

# Ensure an encryption key exists before any app module imports settings.
os.environ.setdefault("APP_ENCRYPTION_KEYS", Fernet.generate_key().decode())
