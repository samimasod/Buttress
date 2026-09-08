import pytest
from fastapi import HTTPException

from apps.api.config import settings as api_settings
import apps.api.core.security.firebase_auth as firebase_auth_module
from apps.api.core.security.firebase_auth import FirebaseAuthService
from apps.api.core.security.jwt_auth import verify_local_token
from apps.api.modules.agents.repository import AgentToolRepository
from apps.api.modules.agents.schemas import AgentToolCreate
from apps.api.modules.agents.service import AgentService
import apps.api.modules.agents.service as agent_service_module
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


def test_in_process_tool_execution_is_disabled_outside_dev(monkeypatch):
    monkeypatch.setattr(agent_service_module.settings, "environment", "production")

    with pytest.raises(RuntimeError, match="disabled outside local/test"):
        agent_service_module.run_python_sandbox(
            "def run(): return 'unsafe'",
            {},
        )


@pytest.mark.asyncio
async def test_tool_creation_persists_approval_policy():
    class FakeSession:
        def add(self, obj):
            self.added = obj

        async def flush(self):
            return None

    repo = AgentToolRepository(FakeSession())
    tool = await repo.create_tool(
        AgentToolCreate(
            name="delete_record",
            description="Deletes a record",
            parameter_schema={"type": "object"},
            code="def run(): return 'deleted'",
            is_active=True,
            ui_mode="both",
            display_label_running="Deleting...",
            display_label_completed="Deleted",
            require_approval=True,
            approval_required_for_roles=["member", "viewer"],
        )
    )

    assert tool.require_approval is True
    assert tool.approval_required_for_roles == ["member", "viewer"]
    assert tool.ui_mode == "both"
    assert tool.display_label_running == "Deleting..."
    assert tool.display_label_completed == "Deleted"


@pytest.mark.asyncio
async def test_execute_tool_rejects_unattached_global_tool():
    class FakeToolRepo:
        async def get_tool_for_agent(self, agent_id, tool_name):
            return None

        async def get_tool(self, tool_name):
            raise AssertionError("Global lookup must not be used for agent execution")

    service = AgentService(
        agent_repo=None,
        tool_repo=FakeToolRepo(),
        session_repo=None,
        usage_repo=None,
    )

    with pytest.raises(ValueError, match="not available to this agent"):
        await service.execute_tool(
            tool_name="global_admin_tool",
            arguments={},
            triggered_by="user-1",
            organization_id=1,
            agent_id=10,
        )
