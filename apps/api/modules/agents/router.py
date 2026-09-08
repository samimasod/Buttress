"""
REST endpoints for Agent CRUD and Admin Tools management.
"""
import math
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.database import get_db
from apps.api.core.security.firebase_auth import FirebaseUser, get_current_user, get_current_super_admin
from apps.api.core.security.permissions import Permission, check_permission
from apps.api.core.cache import get_cache_service
from apps.api.modules.organizations.repository import OrganizationRepository
from apps.api.modules.organizations.service import OrganizationService
from apps.api.modules.agents.repository import AgentRepository, AgentToolRepository, AgentSessionRepository
from apps.api.modules.agents.service import AgentService, run_python_sandbox
from apps.api.modules.agents.schemas import (
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentListResponse,
    AgentToolCreate,
    AgentToolUpdate,
    AgentToolResponse,
    AgentToolDetailResponse,
    AgentToolListResponse,
    AgentToolRunResponse,
    AgentToolRunListResponse,
    AdminToolTestRequest,
    AdminToolTestResponse,
    AgentSessionResponse,
    AgentSessionDetailResponse,
    AgentSessionListResponse,
    AgentMessageResponse,
    UsageOverviewResponse,
    UsageTimeSeriesDataPoint,
    UsageModelBreakdownItem,
    UsageAgentBreakdownItem,
    UsageToolBreakdownItem,
    QuotaUpdateInput,
    OrganizationQuotaResponse,
)

# Main router for Agent workspace actions (user-level)
router = APIRouter()

# Separate router for admin-level tool controls
admin_tools_router = APIRouter()


# --- DEPENDENCIES ---
async def get_org_service(session: AsyncSession = Depends(get_db)) -> OrganizationService:
    repository = OrganizationRepository(session)
    return OrganizationService(repository, cache=get_cache_service())


from apps.api.modules.agents.repository import (
    AgentToolRepository,
    AgentRepository,
    AgentSessionRepository,
    AgentUsageRepository,
)

async def get_agent_service(session: AsyncSession = Depends(get_db)) -> AgentService:
    agent_repo = AgentRepository(session)
    tool_repo = AgentToolRepository(session)
    session_repo = AgentSessionRepository(session)
    usage_repo = AgentUsageRepository(session)
    return AgentService(agent_repo, tool_repo, session_repo, usage_repo)


async def get_repos(session: AsyncSession = Depends(get_db)):
    return {
        "agent": AgentRepository(session),
        "tool": AgentToolRepository(session),
        "session": AgentSessionRepository(session),
        "usage": AgentUsageRepository(session),
    }


# --- USER AGENT ENDPOINTS ---
@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    data: AgentCreate,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Create a new conversational agent."""
    role = await org_service.get_user_role(data.organization_id, user.uid)
    check_permission(role, Permission.PROJECT_CREATE)

    agent = await repos["agent"].create_agent(data, user.uid)
    return AgentResponse.model_validate(agent)


from apps.api.core.pagination import PaginationParams, build_paginated_response

@router.get("", response_model=AgentListResponse)
async def list_agents(
    organization_id: int,
    pagination: PaginationParams = Depends(),
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """List agents inside an organization with pagination."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)

    agents, total = await repos["agent"].list_paginated_agents(
        organization_id, offset=pagination.offset, limit=pagination.limit
    )

    return AgentListResponse(
        agents=[AgentResponse.model_validate(a) for a in agents],
        **build_paginated_response(total, pagination),
    )


@router.get("/tools/available", response_model=List[AgentToolResponse])
async def list_available_tools(
    user: FirebaseUser = Depends(get_current_user),
    repos=Depends(get_repos),
):
    """List all active tools users can attach to agents."""
    tools = await repos["tool"].list_tools(active_only=True)
    return [AgentToolResponse.model_validate(t) for t in tools]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Get detail of an agent."""
    agent = await repos["agent"].get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    role = await org_service.get_user_role(agent.organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)

    return AgentResponse.model_validate(agent)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    data: AgentUpdate,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Update agent configuration."""
    agent = await repos["agent"].get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    role = await org_service.get_user_role(agent.organization_id, user.uid)
    check_permission(role, Permission.PROJECT_UPDATE)

    updated = await repos["agent"].update_agent(agent_id, data)
    return updated


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: int,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Delete agent from workspace."""
    agent = await repos["agent"].get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    role = await org_service.get_user_role(agent.organization_id, user.uid)
    check_permission(role, Permission.PROJECT_DELETE)

    await repos["agent"].delete_agent(agent_id)


# --- SESSIONS AND HISTORY ---
@router.get("/{agent_id}/sessions", response_model=AgentSessionListResponse)
async def list_agent_sessions(
    agent_id: int,
    pagination: PaginationParams = Depends(),
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """List past chat threads user created with this agent with pagination."""
    agent = await repos["agent"].get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    role = await org_service.get_user_role(agent.organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)

    sessions, total = await repos["session"].list_paginated_sessions(
        agent_id, user.uid, offset=pagination.offset, limit=pagination.limit
    )

    return AgentSessionListResponse(
        sessions=[
            AgentSessionResponse(
                id=s.id,
                agent_id=s.agent_id,
                user_uid=s.user_uid,
                created_at=s.created_at,
                agent_name=agent.name,
            )
            for s in sessions
        ],
        **build_paginated_response(total, pagination),
    )


@router.get("/sessions/{session_id}/messages", response_model=AgentSessionDetailResponse)
async def get_session_messages(
    session_id: str,
    pagination: PaginationParams = Depends(),
    user: FirebaseUser = Depends(get_current_user),
    repos=Depends(get_repos),
):
    """Fetch paginated list of messages in a thread session."""
    session_thread = await repos["session"].get_session(session_id)
    if not session_thread:
        raise HTTPException(status_code=404, detail="Chat thread not found")

    # Access security check (ensure user owns session)
    if session_thread.user_uid != user.uid:
        raise HTTPException(status_code=403, detail="Unauthorized access to chat thread")

    messages, total = await repos["session"].get_paginated_messages(
        session_id, offset=pagination.offset, limit=pagination.limit
    )

    return AgentSessionDetailResponse(
        id=session_thread.id,
        agent_id=session_thread.agent_id,
        user_uid=session_thread.user_uid,
        created_at=session_thread.created_at,
        agent_name=session_thread.agent.name if session_thread.agent else None,
        messages=[AgentMessageResponse.model_validate(m) for m in messages],
        **build_paginated_response(total, pagination),
    )


# --- USAGE TELEMETRY & QUOTA ENDPOINTS ---
@router.get("/usage/overview", response_model=UsageOverviewResponse)
async def get_usage_overview(
    organization_id: int,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Fetch aggregated usage overview and monthly quota metrics for an organization."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)
    data = await repos["usage"].get_usage_overview(organization_id)
    return UsageOverviewResponse(**data)


@router.get("/usage/time-series", response_model=List[UsageTimeSeriesDataPoint])
async def get_usage_time_series(
    organization_id: int,
    days: int = Query(default=14, ge=1, le=90),
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Fetch daily token and cost time-series for an organization."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)
    return await repos["usage"].get_time_series_usage(organization_id, days=days)


@router.get("/usage/breakdown/models", response_model=List[UsageModelBreakdownItem])
async def get_model_usage_breakdown(
    organization_id: int,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Fetch token and cost breakdown by LLM model for an organization."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)
    return await repos["usage"].get_model_breakdown(organization_id)


@router.get("/usage/breakdown/agents", response_model=List[UsageAgentBreakdownItem])
async def get_agent_usage_breakdown(
    organization_id: int,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Fetch token and cost breakdown by Agent Profile for an organization."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)
    return await repos["usage"].get_agent_breakdown(organization_id)


@router.get("/usage/breakdown/tools", response_model=List[UsageToolBreakdownItem])
async def get_tool_usage_breakdown(
    organization_id: int,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Fetch tool run execution reliability, approval counts, and duration statistics."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)
    return await repos["usage"].get_tool_breakdown(organization_id)


@router.get("/usage/quota", response_model=OrganizationQuotaResponse)
async def get_organization_quota(
    organization_id: int,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Fetch token quota and budget configuration for an organization."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.PROJECT_READ)
    quota = await repos["usage"].get_or_create_quota(organization_id)
    quota_pct = round((quota.tokens_used_this_month / quota.monthly_token_quota * 100.0), 1) if quota.monthly_token_quota > 0 else 0.0
    return OrganizationQuotaResponse(
        organization_id=quota.organization_id,
        monthly_token_quota=quota.monthly_token_quota,
        monthly_budget_usd=quota.monthly_budget_usd,
        tokens_used_this_month=quota.tokens_used_this_month,
        reserved_tokens_in_flight=quota.reserved_tokens_in_flight,
        cost_usd_this_month=quota.cost_usd_this_month,
        hard_limit_enabled=quota.hard_limit_enabled,
        alert_threshold_percentage=quota.alert_threshold_percentage,
        quota_used_percentage=quota_pct,
    )


@router.patch("/usage/quota", response_model=OrganizationQuotaResponse)
async def update_organization_quota(
    organization_id: int,
    data: QuotaUpdateInput,
    user: FirebaseUser = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    repos=Depends(get_repos),
):
    """Update organization monthly token quota and hard-limit policies."""
    role = await org_service.get_user_role(organization_id, user.uid)
    check_permission(role, Permission.ORG_UPDATE)
    quota = await repos["usage"].update_quota_config(organization_id, data.model_dump(exclude_unset=True))
    quota_pct = round((quota.tokens_used_this_month / quota.monthly_token_quota * 100.0), 1) if quota.monthly_token_quota > 0 else 0.0
    return OrganizationQuotaResponse(
        organization_id=quota.organization_id,
        monthly_token_quota=quota.monthly_token_quota,
        monthly_budget_usd=quota.monthly_budget_usd,
        tokens_used_this_month=quota.tokens_used_this_month,
        reserved_tokens_in_flight=quota.reserved_tokens_in_flight,
        cost_usd_this_month=quota.cost_usd_this_month,
        hard_limit_enabled=quota.hard_limit_enabled,
        alert_threshold_percentage=quota.alert_threshold_percentage,
        quota_used_percentage=quota_pct,
    )


# --- SUPER ADMIN TOOLS MANAGEMENT ---
@admin_tools_router.post("", response_model=AgentToolResponse, status_code=status.HTTP_201_CREATED)
async def create_tool(
    data: AgentToolCreate,
    _: FirebaseUser = Depends(get_current_super_admin),
    repos=Depends(get_repos),
):
    """Register a new global script tool."""
    existing = await repos["tool"].get_tool(data.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Tool with name '{data.name}' already exists")

    tool = await repos["tool"].create_tool(data)
    return AgentToolResponse.model_validate(tool)


@admin_tools_router.get("", response_model=AgentToolListResponse)
async def list_tools(
    pagination: PaginationParams = Depends(),
    _: FirebaseUser = Depends(get_current_super_admin),
    repos=Depends(get_repos),
):
    """List all global tools with pagination."""
    tools, total = await repos["tool"].list_paginated_tools(
        active_only=False, offset=pagination.offset, limit=pagination.limit
    )

    return AgentToolListResponse(
        tools=[AgentToolDetailResponse.model_validate(t) for t in tools],
        **build_paginated_response(total, pagination),
    )


@admin_tools_router.patch("/{name}", response_model=AgentToolResponse)
async def update_tool(
    name: str,
    data: AgentToolUpdate,
    _: FirebaseUser = Depends(get_current_super_admin),
    repos=Depends(get_repos),
):
    """Update global tool metadata or code script."""
    updated = await repos["tool"].update_tool(name, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Tool not found")
    return AgentToolResponse.model_validate(updated)


@admin_tools_router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool(
    name: str,
    _: FirebaseUser = Depends(get_current_super_admin),
    repos=Depends(get_repos),
):
    """Delete a tool from registry."""
    deleted = await repos["tool"].delete_tool(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tool not found")


@admin_tools_router.get("/runs", response_model=AgentToolRunListResponse)
async def list_tool_runs(
    tool_name: Optional[str] = None,
    limit: int = 100,
    _: FirebaseUser = Depends(get_current_super_admin),
    repos=Depends(get_repos),
):
    """Monitor recent tool run executions."""
    runs = await repos["tool"].list_tool_runs(tool_name=tool_name, limit=limit)
    return AgentToolRunListResponse(
        runs=[AgentToolRunResponse.model_validate(r) for r in runs],
        total=len(runs),
    )


@admin_tools_router.post("/{name}/test", response_model=AdminToolTestResponse)
async def test_run_tool(
    name: str,
    data: AdminToolTestRequest,
    user: FirebaseUser = Depends(get_current_super_admin),
    service: AgentService = Depends(get_agent_service),
):
    """Run a dynamic synchronous tool script test run, generating execution metrics."""
    res = await service.execute_tool(
        tool_name=name,
        arguments=data.arguments,
        triggered_by=user.uid,
    )
    return AdminToolTestResponse(
        success=res["success"],
        output=res["output"],
        error=res["error"],
        duration_ms=res["duration_ms"],
    )