"""
Database repository for the Agent module.
"""
from typing import List, Optional, Tuple
from sqlalchemy import select, delete, desc, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apps.api.modules.agents.models import (
    Agent,
    AgentTool,
    AgentToolRun,
    AgentSession,
    AgentMessage,
    AgentUsageLog,
    OrganizationUsageQuota,
    agent_tool_association,
)
from apps.api.modules.agents.schemas import AgentCreate, AgentUpdate, AgentToolCreate, AgentToolUpdate


class AgentToolRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_tool(self, data: AgentToolCreate) -> AgentTool:
        tool = AgentTool(
            name=data.name,
            description=data.description,
            parameter_schema=data.parameter_schema,
            code=data.code,
            is_active=data.is_active,
            ui_mode=data.ui_mode,
            display_label_running=data.display_label_running,
            display_label_completed=data.display_label_completed,
            require_approval=data.require_approval,
            approval_required_for_roles=data.approval_required_for_roles,
        )
        self.session.add(tool)
        await self.session.flush()
        return tool

    async def get_tool(self, name: str) -> Optional[AgentTool]:
        result = await self.session.execute(
            select(AgentTool).where(AgentTool.name == name)
        )
        return result.scalar_one_or_none()

    async def get_tool_for_agent(self, agent_id: int, name: str) -> Optional[AgentTool]:
        """Return a tool only when it is explicitly attached to the agent."""
        result = await self.session.execute(
            select(AgentTool)
            .join(
                agent_tool_association,
                AgentTool.name == agent_tool_association.c.tool_name,
            )
            .where(
                agent_tool_association.c.agent_id == agent_id,
                AgentTool.name == name,
            )
        )
        return result.scalar_one_or_none()

    async def list_tools(self, active_only: bool = False) -> List[AgentTool]:
        stmt = select(AgentTool)
        if active_only:
            stmt = stmt.where(AgentTool.is_active == True)
        result = await self.session.execute(stmt.order_by(AgentTool.name))
        return list(result.scalars().all())

    async def list_paginated_tools(
        self, active_only: bool = False, offset: int = 0, limit: int = 20
    ) -> Tuple[List[AgentTool], int]:
        stmt_count = select(func.count(AgentTool.name))
        if active_only:
            stmt_count = stmt_count.where(AgentTool.is_active == True)
        count_res = await self.session.execute(stmt_count)
        total = count_res.scalar() or 0

        stmt = select(AgentTool)
        if active_only:
            stmt = stmt.where(AgentTool.is_active == True)
        result = await self.session.execute(stmt.order_by(AgentTool.name).offset(offset).limit(limit))
        return list(result.scalars().all()), total

    async def update_tool(self, name: str, data: AgentToolUpdate) -> Optional[AgentTool]:
        tool = await self.get_tool(name)
        if not tool:
            return None
        
        update_dict = data.model_dump(exclude_unset=True)
        for key, val in update_dict.items():
            setattr(tool, key, val)
        
        await self.session.flush()
        return tool

    async def delete_tool(self, name: str) -> bool:
        tool = await self.get_tool(name)
        if not tool:
            return False
        await self.session.delete(tool)
        await self.session.flush()
        return True

    async def log_tool_run(
        self,
        tool_name: str,
        triggered_by: str,
        arguments: dict,
        output: Optional[str],
        error: Optional[str],
        duration_ms: int,
        approval_status: Optional[str] = None,
        approved_by: Optional[str] = None,
        organization_id: Optional[int] = None,
        agent_id: Optional[int] = None,
    ) -> AgentToolRun:
        run = AgentToolRun(
            tool_name=tool_name,
            triggered_by=triggered_by,
            organization_id=organization_id,
            agent_id=agent_id,
            arguments=arguments,
            output=output,
            error=error,
            duration_ms=duration_ms,
            approval_status=approval_status,
            approved_by=approved_by,
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def list_tool_runs(self, tool_name: Optional[str] = None, limit: int = 100) -> List[AgentToolRun]:
        stmt = select(AgentToolRun)
        if tool_name:
            stmt = stmt.where(AgentToolRun.tool_name == tool_name)
        result = await self.session.execute(
            stmt.order_by(desc(AgentToolRun.id)).limit(limit)
        )
        return list(result.scalars().all())


class AgentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_agent(self, data: AgentCreate, created_by_uid: str) -> Agent:
        agent = Agent(
            organization_id=data.organization_id,
            name=data.name,
            description=data.description,
            system_prompt=data.system_prompt,
            model_id=data.model_id,
            summary_model_id=data.summary_model_id,
            tool_pruning_turns=data.tool_pruning_turns,
            summarization_threshold=data.summarization_threshold,
            temperature=data.temperature,
            created_by_uid=created_by_uid,
            is_active=data.is_active,
        )
        
        if data.tool_names:
            tools_result = await self.session.execute(
                select(AgentTool).where(AgentTool.name.in_(data.tool_names))
            )
            agent.tools = list(tools_result.scalars().all())
            
        self.session.add(agent)
        await self.session.flush()
        # Refresh to populate relationships
        await self.session.refresh(agent, ["tools"])
        return agent

    async def get_agent(self, agent_id: int) -> Optional[Agent]:
        result = await self.session.execute(
            select(Agent)
            .options(selectinload(Agent.tools))
            .where(Agent.id == agent_id)
        )
        return result.scalar_one_or_none()

    async def list_agents(self, organization_id: int, active_only: bool = False) -> List[Agent]:
        stmt = select(Agent).options(selectinload(Agent.tools)).where(Agent.organization_id == organization_id)
        if active_only:
            stmt = stmt.where(Agent.is_active == True)
        result = await self.session.execute(stmt.order_by(Agent.name))
        return list(result.scalars().all())

    async def list_paginated_agents(
        self, organization_id: int, active_only: bool = False, offset: int = 0, limit: int = 20
    ) -> Tuple[List[Agent], int]:
        stmt_count = select(func.count(Agent.id)).where(Agent.organization_id == organization_id)
        if active_only:
            stmt_count = stmt_count.where(Agent.is_active == True)
        count_res = await self.session.execute(stmt_count)
        total = count_res.scalar() or 0

        stmt = select(Agent).options(selectinload(Agent.tools)).where(Agent.organization_id == organization_id)
        if active_only:
            stmt = stmt.where(Agent.is_active == True)
        result = await self.session.execute(stmt.order_by(Agent.name).offset(offset).limit(limit))
        return list(result.scalars().all()), total

    async def update_agent(self, agent_id: int, data: AgentUpdate) -> Optional[Agent]:
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        update_dict = data.model_dump(exclude_unset=True, exclude={"tool_names"})
        for key, val in update_dict.items():
            setattr(agent, key, val)
            
        if data.tool_names is not None:
            tools_result = await self.session.execute(
                select(AgentTool).where(AgentTool.name.in_(data.tool_names))
            )
            agent.tools = list(tools_result.scalars().all())

        await self.session.flush()
        await self.session.refresh(agent)
        return agent

    async def delete_agent(self, agent_id: int) -> bool:
        agent = await self.get_agent(agent_id)
        if not agent:
            return False
        await self.session.delete(agent)
        await self.session.flush()
        return True


class AgentSessionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_session(self, session_id: str, agent_id: int, user_uid: str) -> AgentSession:
        agent_session = AgentSession(
            id=session_id,
            agent_id=agent_id,
            user_uid=user_uid,
        )
        self.session.add(agent_session)
        await self.session.flush()
        return agent_session

    async def get_session(self, session_id: str) -> Optional[AgentSession]:
        result = await self.session.execute(
            select(AgentSession)
            .options(
                selectinload(AgentSession.messages),
                selectinload(AgentSession.agent),
            )
            .where(AgentSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def list_sessions(self, agent_id: int, user_uid: str) -> List[AgentSession]:
        result = await self.session.execute(
            select(AgentSession)
            .where(AgentSession.agent_id == agent_id, AgentSession.user_uid == user_uid)
            .order_by(desc(AgentSession.created_at))
        )
        return list(result.scalars().all())

    async def list_paginated_sessions(
        self, agent_id: int, user_uid: str, offset: int = 0, limit: int = 20
    ) -> Tuple[List[AgentSession], int]:
        count_res = await self.session.execute(
            select(func.count(AgentSession.id)).where(
                AgentSession.agent_id == agent_id, AgentSession.user_uid == user_uid
            )
        )
        total = count_res.scalar() or 0

        result = await self.session.execute(
            select(AgentSession)
            .where(AgentSession.agent_id == agent_id, AgentSession.user_uid == user_uid)
            .order_by(desc(AgentSession.created_at))
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def get_paginated_messages(
        self, session_id: str, offset: int = 0, limit: int = 20
    ) -> Tuple[List[AgentMessage], int]:
        count_res = await self.session.execute(
            select(func.count(AgentMessage.id)).where(AgentMessage.session_id == session_id)
        )
        total = count_res.scalar() or 0

        result = await self.session.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.id.asc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def get_recent_messages(self, session_id: str, limit: int = 10) -> List[AgentMessage]:
        result = await self.session.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .order_by(desc(AgentMessage.id))
            .limit(limit)
        )
        messages = list(result.scalars().all())
        messages.reverse()
        return messages

    async def update_session_summary(self, session_id: str, summary: str) -> None:
        db_session = await self.get_session(session_id)
        if db_session:
            db_session.summary = summary
            await self.session.flush()

    async def create_message(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[dict]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> AgentMessage:
        message = AgentMessage(
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
        )
        self.session.add(message)
        await self.session.flush()
        return message


class AgentUsageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_quota(self, organization_id: int) -> OrganizationUsageQuota:
        result = await self.session.execute(
            select(OrganizationUsageQuota).where(OrganizationUsageQuota.organization_id == organization_id)
        )
        quota = result.scalar_one_or_none()
        if not quota:
            quota = OrganizationUsageQuota(
                organization_id=organization_id,
                monthly_token_quota=1_000_000,
                monthly_budget_usd=50.0,
                tokens_used_this_month=0,
                reserved_tokens_in_flight=0,
                cost_usd_this_month=0.0,
                hard_limit_enabled=True,
                alert_threshold_percentage=80,
            )
            self.session.add(quota)
            await self.session.flush()
        return quota

    async def reserve_quota(
        self, organization_id: int, estimated_tokens: int = 4000
    ) -> Tuple[bool, OrganizationUsageQuota]:
        quota = await self.get_or_create_quota(organization_id)
        if quota.hard_limit_enabled:
            current_total = quota.tokens_used_this_month + quota.reserved_tokens_in_flight
            if (current_total + estimated_tokens) > quota.monthly_token_quota:
                return False, quota

        quota.reserved_tokens_in_flight += estimated_tokens
        await self.session.flush()
        return True, quota

    async def settle_quota(
        self,
        organization_id: int,
        actual_tokens: int,
        actual_cost_usd: float,
        reserved_estimate: int = 4000,
    ) -> OrganizationUsageQuota:
        quota = await self.get_or_create_quota(organization_id)
        quota.reserved_tokens_in_flight = max(0, quota.reserved_tokens_in_flight - reserved_estimate)
        quota.tokens_used_this_month += actual_tokens
        quota.cost_usd_this_month += actual_cost_usd
        await self.session.flush()
        return quota

    async def release_reservation(self, organization_id: int, reserved_estimate: int = 4000) -> None:
        quota = await self.get_or_create_quota(organization_id)
        quota.reserved_tokens_in_flight = max(0, quota.reserved_tokens_in_flight - reserved_estimate)
        await self.session.flush()

    async def update_quota_config(
        self, organization_id: int, update_data: dict
    ) -> OrganizationUsageQuota:
        quota = await self.get_or_create_quota(organization_id)
        for key, val in update_data.items():
            if val is not None and hasattr(quota, key):
                setattr(quota, key, val)
        await self.session.flush()
        return quota

    async def log_turn_usage(
        self,
        organization_id: int,
        agent_id: int,
        session_id: str,
        user_uid: str,
        provider: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        toon_tokens_saved: int,
        estimated_cost_usd: float,
        latency_ms: int,
        tool_calls_count: int = 0,
    ) -> AgentUsageLog:
        log_entry = AgentUsageLog(
            organization_id=organization_id,
            agent_id=agent_id,
            session_id=session_id,
            user_uid=user_uid,
            provider=provider,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            toon_tokens_saved=toon_tokens_saved,
            estimated_cost_usd=estimated_cost_usd,
            latency_ms=latency_ms,
            tool_calls_count=tool_calls_count,
        )
        self.session.add(log_entry)
        await self.session.flush()
        return log_entry

    async def get_usage_overview(self, organization_id: int) -> dict:
        quota = await self.get_or_create_quota(organization_id)
        
        stmt = select(
            func.coalesce(func.sum(AgentUsageLog.total_tokens), 0),
            func.coalesce(func.sum(AgentUsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(AgentUsageLog.completion_tokens), 0),
            func.coalesce(func.sum(AgentUsageLog.toon_tokens_saved), 0),
            func.coalesce(func.sum(AgentUsageLog.estimated_cost_usd), 0.0),
            func.coalesce(func.count(AgentUsageLog.id), 0),
            func.coalesce(func.avg(AgentUsageLog.latency_ms), 0.0),
        ).where(AgentUsageLog.organization_id == organization_id)
        
        res = await self.session.execute(stmt)
        tot_tok, p_tok, c_tok, toon_sav, tot_cost, tot_turns, avg_lat = res.one()

        unoptimized_total = tot_tok + toon_sav
        toon_pct = round((toon_sav / unoptimized_total * 100.0), 1) if unoptimized_total > 0 else 0.0
        quota_pct = round((quota.tokens_used_this_month / quota.monthly_token_quota * 100.0), 1) if quota.monthly_token_quota > 0 else 0.0

        return {
            "organization_id": organization_id,
            "total_tokens": tot_tok,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "toon_tokens_saved": toon_sav,
            "toon_savings_percentage": toon_pct,
            "total_cost_usd": round(tot_cost, 4),
            "total_turns": tot_turns,
            "avg_latency_ms": round(avg_lat, 1),
            "monthly_token_quota": quota.monthly_token_quota,
            "tokens_used_this_month": quota.tokens_used_this_month,
            "reserved_tokens_in_flight": quota.reserved_tokens_in_flight,
            "quota_used_percentage": quota_pct,
            "hard_limit_enabled": quota.hard_limit_enabled,
        }

    async def get_time_series_usage(self, organization_id: int, days: int = 14) -> List[dict]:
        stmt = (
            select(
                func.date(AgentUsageLog.created_at).label("date_str"),
                func.coalesce(func.sum(AgentUsageLog.total_tokens), 0),
                func.coalesce(func.sum(AgentUsageLog.prompt_tokens), 0),
                func.coalesce(func.sum(AgentUsageLog.completion_tokens), 0),
                func.coalesce(func.sum(AgentUsageLog.estimated_cost_usd), 0.0),
                func.coalesce(func.count(AgentUsageLog.id), 0),
            )
            .where(AgentUsageLog.organization_id == organization_id)
            .group_by(func.date(AgentUsageLog.created_at))
            .order_by(func.date(AgentUsageLog.created_at).asc())
        )
        res = await self.session.execute(stmt)
        data_points = []
        for d_str, tot_tok, p_tok, c_tok, cost, turns in res.all():
            data_points.append({
                "date": str(d_str),
                "total_tokens": tot_tok,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "cost_usd": round(cost, 4),
                "turns_count": turns,
            })
        return data_points

    async def get_model_breakdown(self, organization_id: int) -> List[dict]:
        stmt = (
            select(
                AgentUsageLog.model_id,
                AgentUsageLog.provider,
                func.coalesce(func.sum(AgentUsageLog.total_tokens), 0),
                func.coalesce(func.sum(AgentUsageLog.estimated_cost_usd), 0.0),
                func.coalesce(func.count(AgentUsageLog.id), 0),
                func.coalesce(func.avg(AgentUsageLog.latency_ms), 0.0),
            )
            .where(AgentUsageLog.organization_id == organization_id)
            .group_by(AgentUsageLog.model_id, AgentUsageLog.provider)
            .order_by(func.sum(AgentUsageLog.total_tokens).desc())
        )
        res = await self.session.execute(stmt)
        return [
            {
                "model_id": m_id,
                "provider": prov,
                "total_tokens": tot_tok,
                "cost_usd": round(cost, 4),
                "turns_count": turns,
                "avg_latency_ms": round(lat, 1),
            }
            for m_id, prov, tot_tok, cost, turns, lat in res.all()
        ]

    async def get_agent_breakdown(self, organization_id: int) -> List[dict]:
        stmt = (
            select(
                AgentUsageLog.agent_id,
                Agent.name,
                func.coalesce(func.sum(AgentUsageLog.total_tokens), 0),
                func.coalesce(func.sum(AgentUsageLog.estimated_cost_usd), 0.0),
                func.coalesce(func.count(AgentUsageLog.id), 0),
            )
            .join(Agent, Agent.id == AgentUsageLog.agent_id)
            .where(AgentUsageLog.organization_id == organization_id)
            .group_by(AgentUsageLog.agent_id, Agent.name)
            .order_by(func.sum(AgentUsageLog.total_tokens).desc())
        )
        res = await self.session.execute(stmt)
        return [
            {
                "agent_id": a_id,
                "agent_name": a_name,
                "total_tokens": tot_tok,
                "cost_usd": round(cost, 4),
                "turns_count": turns,
            }
            for a_id, a_name, tot_tok, cost, turns in res.all()
        ]

    async def get_tool_breakdown(self, organization_id: int) -> List[dict]:
        stmt = (
            select(
                AgentToolRun.tool_name,
                func.count(AgentToolRun.id).label("total_runs"),
                func.sum(case((AgentToolRun.error == None, 1), else_=0)).label("successful_runs"),
                func.sum(case((AgentToolRun.error != None, 1), else_=0)).label("failed_runs"),
                func.sum(case((AgentToolRun.approval_status == "approved", 1), else_=0)).label("approved_runs"),
                func.sum(case((AgentToolRun.approval_status == "rejected", 1), else_=0)).label("rejected_runs"),
                func.coalesce(func.avg(AgentToolRun.duration_ms), 0.0).label("avg_duration_ms"),
            )
            .where(AgentToolRun.organization_id == organization_id)
            .group_by(AgentToolRun.tool_name)
            .order_by(func.count(AgentToolRun.id).desc())
        )
        res = await self.session.execute(stmt)
        return [
            {
                "tool_name": t_name,
                "total_runs": tot,
                "successful_runs": succ or 0,
                "failed_runs": fail or 0,
                "approved_runs": appr or 0,
                "rejected_runs": rej or 0,
                "avg_duration_ms": round(dur, 1),
            }
            for t_name, tot, succ, fail, appr, rej, dur in res.all()
        ]
