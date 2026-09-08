"""Security and Firebase authentication middleware dependency for SuperAdmin Monitoring API."""

import hmac
from typing import Optional
from fastapi import Depends, Header, HTTPException, status
from apps.api_admin.config import admin_settings

# Attempt to import firebase_admin for ID token verification
try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


async def verify_admin_auth(
    x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-Api-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Verify SuperAdmin authentication using Firebase ID Token or Secret API Key header.

    When ADMIN_AUTH_ENABLED is true:
    1. Checks 'X-Admin-Api-Key' or 'Authorization: Bearer <token>'.
    2. Accepts matching SUPER_ADMIN_API_KEY when configured.
    3. Accepts mock dev tokens only in explicit local/test environments.
    4. Verifies Firebase ID tokens and requires the email to be allowlisted.
    """
    if not admin_settings.admin_auth_enabled:
        # Config validation prevents this outside explicit development environments.
        return True

    # Normalize token candidate from header
    token_candidate = (x_admin_api_key or "").strip()
    if not token_candidate and authorization and authorization.startswith("Bearer "):
        token_candidate = authorization.split("Bearer ", 1)[1].strip()

    if not token_candidate:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid SuperAdmin authentication. Provide a valid Firebase Bearer token or X-Admin-Api-Key header.",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )

    # 1. Check configured SUPER_ADMIN_API_KEY using constant-time comparison.
    if (
        admin_settings.super_admin_api_key
        and hmac.compare_digest(token_candidate, admin_settings.super_admin_api_key)
    ):
        return True

    # 2. Mock admin tokens are only valid in explicit local/test environments.
    if (
        admin_settings.is_development_environment
        and token_candidate.startswith("mock_firebase_admin_token_")
    ):
        return True

    # 3. Check real Firebase Bearer Token if available.
    if FIREBASE_AVAILABLE:
        try:
            decoded = firebase_auth.verify_id_token(token_candidate)
            email = decoded.get("email", "").lower()

            # Firebase-based SuperAdmin auth is deny-by-default unless an allowlist exists.
            if not admin_settings.super_admin_emails:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="SUPER_ADMIN_EMAILS must be configured for Firebase SuperAdmin authentication.",
                )
            if email not in admin_settings.super_admin_emails:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User email '{email}' is not configured in SUPER_ADMIN_EMAILS.",
                )
            return True
        except HTTPException:
            raise
        except Exception as e:
            print(f"Firebase token verification failed: {e}")

    # If verification failed, including when Firebase is unavailable, fail closed.
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid SuperAdmin authentication credentials.",
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )
