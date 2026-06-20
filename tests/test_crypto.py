from app.crypto import Secret, decrypt, encrypt, last4


def test_encrypt_decrypt_roundtrip():
    secret = "sk-ant-secret-12345"
    token = encrypt(secret)
    assert bytes(token) != secret.encode()
    assert decrypt(token) == secret


def test_last4():
    assert last4("abcdef") == "cdef"
    assert last4("ab") == "**"


def test_secret_does_not_leak_in_repr_or_str():
    sec = Secret("topsecret")
    assert "topsecret" not in repr(sec)
    assert "topsecret" not in str(sec)
    assert sec.reveal() == "topsecret"
