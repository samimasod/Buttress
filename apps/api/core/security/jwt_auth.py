"""
Local JWT Token & Password Hashing Security Module.
Provides 100% offline authentication without external cloud dependencies.
"""
import base64
import hashlib
import hmac
import json
import time
from typing import Optional, Dict, Any

from apps.api.config import settings
from apps.api.core.security.firebase_auth import FirebaseUser


def hash_password(password: str) -> str:
    """Generates a PBKDF2 SHA256 password hash."""
    salt = b"skeleton_local_auth_salt_2026"
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies password against stored PBKDF2 hash."""
    computed_hash = hash_password(password)
    return hmac.compare_digest(computed_hash, hashed_password)


def b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def b64_decode(data: str) -> bytes:
    padding = "=" * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(payload: Dict[str, Any], expires_in_seconds: int = 30 * 86400) -> str:
    """Creates a signed JWT access token for local user authentication."""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    token_payload = {
        **payload,
        "iat": now,
        "exp": now + expires_in_seconds,
    }

    header_bytes = b64_encode(json.dumps(header).encode("utf-8"))
    payload_bytes = b64_encode(json.dumps(token_payload).encode("utf-8"))
    signing_input = f"{header_bytes}.{payload_bytes}"

    signature = hmac.new(
        settings.jwt_secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256
    ).digest()

    signature_bytes = b64_encode(signature)
    return f"{signing_input}.{signature_bytes}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies signature and decodes local JWT access token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"

        expected_sig = hmac.new(
            settings.jwt_secret.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256
        ).digest()

        if not hmac.compare_digest(b64_encode(expected_sig), signature_b64):
            return None

        payload = json.loads(b64_decode(payload_b64).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None

        return payload
    except Exception:
        return None


def verify_local_token(token: str) -> Optional[FirebaseUser]:
    """Verifies local JWT token or explicitly development-only mock tokens."""
    # Mock tokens are test/dev conveniences and must never authenticate in staging/production.
    if settings.is_development_environment and (
        token.startswith("dev-user-") or token.startswith("mock_")
    ):
        return FirebaseUser(
            uid="dev-user-123",
            email="dev@example.com",
            email_verified=True,
            name="Dev User",
        )

    payload = decode_access_token(token)
    if not payload:
        return None

    return FirebaseUser(
        uid=payload.get("uid", ""),
        email=payload.get("email"),
        email_verified=payload.get("email_verified", True),
        name=payload.get("name", "Local User"),
        picture=payload.get("picture"),
    )
