import pytest
from fastapi import HTTPException

from apps.api.config import settings as api_settings
import apps.api.core.security.firebase_auth as firebase_auth_module
from apps.api.core.security.firebase_auth import FirebaseAuthService
from apps.api.core.security.jwt_auth import verify_local_token
from apps.api_admin.config import admin_settings
import apps.api_admin.security as admin_security_module


def test_mock_user_tokens_are_rejected_outside_dev(monkeypatch):
    monkeypatch.setattr(api_settings, "environment", "production")

    assert verify_local_token("mock_anything") is None
    assert verify_local_token("dev-user-anything") is None


def test_mock_user_tokens_still_work_in_dev(monkeypatch):
    monkeypatch.setattr(api_settings, "environment", "local")

    user = verify_local_token("mock_unit_test")

    assert user is not None
    assert user.uid == "dev-user-123"


def test_firebase_auth_fails_closed_when_sdk_is_unavailable(monkeypatch):
    monkeypatch.setattr(api_settings, "environment", "production")
    monkeypatch.setattr(firebase_auth_module, "FIREBASE_AVAILABLE", False)

    service = FirebaseAuthService()
    service.initialized = False

    assert service.verify_token("arbitrary-bearer-token") is None
    assert service.get_user_by_uid("someone") is None


@pytest.mark.asyncio
async def test_superadmin_auth_fails_closed_without_firebase(monkeypatch):
    monkeypatch.setattr(admin_settings, "environment", "production")
    monkeypatch.setattr(admin_settings, "admin_auth_enabled", True)
    monkeypatch.setattr(admin_settings, "super_admin_api_key", None)
    monkeypatch.setattr(admin_security_module, "FIREBASE_AVAILABLE", False)

    with pytest.raises(HTTPException) as exc:
        await admin_security_module.verify_admin_auth(
            x_admin_api_key="arbitrary-token",
            authorization=None,
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_superadmin_mock_token_rejected_in_production(monkeypatch):
    monkeypatch.setattr(admin_settings, "environment", "production")
    monkeypatch.setattr(admin_settings, "admin_auth_enabled", True)
    monkeypatch.setattr(admin_settings, "super_admin_api_key", None)
    monkeypatch.setattr(admin_security_module, "FIREBASE_AVAILABLE", False)

    with pytest.raises(HTTPException) as exc:
        await admin_security_module.verify_admin_auth(
            x_admin_api_key="mock_firebase_admin_token_should_not_work",
            authorization=None,
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_superadmin_mock_token_allowed_in_dev(monkeypatch):
    monkeypatch.setattr(admin_settings, "environment", "local")
    monkeypatch.setattr(admin_settings, "admin_auth_enabled", True)
    monkeypatch.setattr(admin_settings, "super_admin_api_key", None)
    monkeypatch.setattr(admin_security_module, "FIREBASE_AVAILABLE", False)

    result = await admin_security_module.verify_admin_auth(
        x_admin_api_key="mock_firebase_admin_token_unit_test",
        authorization=None,
    )

    assert result is True
