"""
Firebase Authentication service.
Handles token verification and user management.
"""
import json
from typing import Optional
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Firebase Admin SDK - conditionally imported
try:
    import firebase_admin
    from firebase_admin import auth, credentials
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


security = HTTPBearer(auto_error=False)


@dataclass
class FirebaseUser:
    """Represents an authenticated Firebase user."""
    uid: str
    email: Optional[str] = None
    email_verified: bool = False
    name: Optional[str] = None
    picture: Optional[str] = None
    
    @classmethod
    def from_token(cls, decoded_token: dict) -> "FirebaseUser":
        """Create a FirebaseUser from a decoded token."""
        return cls(
            uid=decoded_token.get("uid", ""),
            email=decoded_token.get("email"),
            email_verified=decoded_token.get("email_verified", False),
            name=decoded_token.get("name"),
            picture=decoded_token.get("picture"),
        )


class FirebaseAuthService:
    """Service for Firebase authentication operations."""
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        credentials_json: Optional[str] = None,
    ):
        """Initialize Firebase Admin SDK."""
        self.initialized = False
        
        if not FIREBASE_AVAILABLE:
            print("Warning: firebase-admin not installed. Firebase auth is unavailable.")
            return
        
        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            self.initialized = True
        except ValueError:
            # Initialize Firebase with credentials
            if credentials_json:
                try:
                    cred = credentials.Certificate(json.loads(credentials_json))
                    firebase_admin.initialize_app(cred)
                    self.initialized = True
                except Exception as e:
                    print(f"Warning: Could not initialize Firebase from JSON: {e}")
            elif credentials_path:
                try:
                    cred = credentials.Certificate(credentials_path)
                    firebase_admin.initialize_app(cred)
                    self.initialized = True
                except Exception as e:
                    print(f"Warning: Could not initialize Firebase: {e}")
            else:
                # Try to initialize with default credentials (for GCP environments)
                try:
                    firebase_admin.initialize_app()
                    self.initialized = True
                except Exception as e:
                    print(f"Warning: Could not initialize Firebase with defaults: {e}")
    
    def verify_token(self, token: str) -> Optional[FirebaseUser]:
        """Verify a JWT or Firebase ID token and return the user."""
        from apps.api.core.security.jwt_auth import verify_local_token
        local_user = verify_local_token(token)
        if local_user:
            return local_user

        if not self.initialized or not FIREBASE_AVAILABLE:
            # Local JWT and explicitly dev-only mock tokens are handled above.
            # Never turn an arbitrary bearer token into an authenticated user.
            return None
        
        try:
            decoded_token = auth.verify_id_token(token)
            return FirebaseUser.from_token(decoded_token)
        except Exception as e:
            print(f"Token verification failed: {e}")
            return None
    
    def get_user_by_uid(self, uid: str) -> Optional[dict]:
        """Get user info by UID."""
        if not self.initialized or not FIREBASE_AVAILABLE:
            return None
        
        try:
            user = auth.get_user(uid)
            return {
                "uid": user.uid,
                "email": user.email,
                "email_verified": user.email_verified,
                "display_name": user.display_name,
                "photo_url": user.photo_url,
            }
        except Exception:
            return None


# Global auth service instance (initialized lazily)
_auth_service: Optional[FirebaseAuthService] = None


def get_auth_service() -> FirebaseAuthService:
    """Get the Firebase auth service instance."""
    global _auth_service
    if _auth_service is None:
        from apps.api.config import settings
        _auth_service = FirebaseAuthService(
            credentials_path=settings.firebase_admin_credentials_path,
            credentials_json=settings.firebase_admin_credentials_json,
        )
    return _auth_service


def is_super_admin_email(email: Optional[str]) -> bool:
    if not email:
        return False
    from apps.api.config import settings

    return email.lower() in settings.super_admin_emails


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> FirebaseUser:
    """
    Dependency to get the current authenticated user.
    
    Usage:
        @router.get("/profile")
        async def get_profile(user: FirebaseUser = Depends(get_current_user)):
            return {"uid": user.uid, "email": user.email}
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    auth_service = get_auth_service()
    user = auth_service.verify_token(credentials.credentials)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[FirebaseUser]:
    """
    Dependency to optionally get the current user.
    Returns None if no valid token is provided.
    """
    if credentials is None:
        return None
    
    auth_service = get_auth_service()
    return auth_service.verify_token(credentials.credentials)


async def get_current_super_admin(
    user: FirebaseUser = Depends(get_current_user),
) -> FirebaseUser:
    """Dependency to require a configured super admin."""
    if not is_super_admin_email(user.email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required",
        )
    return user
