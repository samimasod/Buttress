"""
WebSocket handler for real-time bi-directional agent streaming.

Architecture: Two concurrent async tasks share a single WebSocket connection.
  - Receiver task: reads inbound messages (user chat, tool_approval signals).
  - Sender task: iterates the run_agent_loop generator and streams events to the client.

Approval Gate:
  When a tool with require_approval=True is requested by the LLM, run_agent_loop
  emits a tool_approval_requested event and awaits an asyncio.Future keyed by
  tool_call_id. The receiver task resolves that Future when it receives a
  tool_approval message from the client — no extra LLM roundtrip is needed.
"""
import uuid
import json
import asyncio
import logging
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select

from apps.api.core.database.session import AsyncSessionLocal
from apps.api.core.security.firebase_auth import get_auth_service
from apps.api.modules.organizations.models import OrganizationMember
from apps.api.modules.agents.models import Agent
from apps.api.modules.agents.repository import AgentRepository, AgentToolRepository, AgentSessionRepository
from apps.api.modules.agents.service import AgentService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/{agent_id}/chat/ws")
async def agent_chat_websocket(websocket: WebSocket, agent_id: int):
    """
    WebSocket endpoint for streaming agent chat interactions.
    Requires Firebase token auth and validates org membership.

    Client → Server message types:
      { "type": "message",       "content": "..." }          — user chat message
      { "type": "tool_approval", "tool_call_id": "...",
        "approved": true/false,  "reason": "..." }           — approval gate response

    Server → Client event types:
      session_created | text_delta | tool_started | tool_approval_requested |
      tool_completed  | tool_denied | message_completed | error
    """
    await websocket.accept()

    # 1. Fetch connection parameters
    token = websocket.query_params.get("token")
    session_id = websocket.query_params.get("session_id")

    if not token:
        logger.warning("WebSocket rejected: Missing authentication token")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Verify Firebase token
    auth_service = get_auth_service()
    user = auth_service.verify_token(token)
    if not user:
        logger.warning("WebSocket rejected: Invalid authentication token")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Create separate DB session scoped to the WebSocket connection lifetime
    async with AsyncSessionLocal() as session:
        agent_repo = AgentRepository(session)
        tool_repo = AgentToolRepository(session)
        session_repo = AgentSessionRepository(session)
        service = AgentService(agent_repo, tool_repo, session_repo)

        # 3. Retrieve agent
        agent = await agent_repo.get_agent(agent_id)
        if not agent or not agent.is_active:
            logger.warning(f"WebSocket rejected: Agent {agent_id} not found or inactive")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # 4. Check org membership and resolve user role
        member_check = await session.execute(
            select(OrganizationMember).where(
                OrganizationMember.organization_id == agent.organization_id,
                OrganizationMember.firebase_uid == user.uid,
            )
        )
        member = member_check.scalar_one_or_none()
        if not member:
            logger.warning(f"WebSocket rejected: User {user.uid} lacks org membership for agent {agent_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # member.role is the user's Role enum value — passed into run_agent_loop
        user_role: str = member.role.value if member.role else "member"

        # 5. Load or create conversational session
        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex[:16]}"
            await session_repo.create_session(
                session_id=session_id,
                agent_id=agent_id,
                user_uid=user.uid,
            )
            await session.commit()
            history_data = []
        else:
            db_session = await session_repo.get_session(session_id)
            if not db_session:
                db_session = await session_repo.create_session(
                    session_id=session_id,
                    agent_id=agent_id,
                    user_uid=user.uid,
                )
                await session.commit()
                history_data = []
            elif db_session.user_uid != user.uid or db_session.agent_id != agent_id:
                logger.warning(
                    "WebSocket rejected: session ownership mismatch "
                    f"session={session_id} user={user.uid} agent={agent_id}"
                )
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            else:
                history_data = [
                    {
                        "id": m.id,
                        "session_id": m.session_id,
                        "role": m.role,
                        "content": m.content,
                        "tool_calls": m.tool_calls,
                        "tool_call_id": m.tool_call_id,
                        "name": m.name,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in db_session.messages
                ]

        # 6. Notify connection success and seed history
        await websocket.send_json({
            "type": "session_created",
            "session_id": session_id,
            "history": history_data,
        })

        # --- Approval Gate State ---
        # Maps tool_call_id → asyncio.Future[bool]. The sender task puts Futures
        # here when approval is needed; the receiver task resolves them.
        pending_approvals: Dict[str, asyncio.Future] = {}

        async def approval_gate(tool_call_id: str) -> bool:
            """Suspends the agent loop until the client approves or rejects the tool call.
            Auto-rejects after a 5-minute timeout so the loop never hangs forever.
            """
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            pending_approvals[tool_call_id] = future
            try:
                return await asyncio.wait_for(asyncio.shield(future), timeout=300.0)
            except asyncio.TimeoutError:
                logger.info(f"Approval gate timed out for tool_call_id={tool_call_id}; auto-rejecting.")
                return False
            finally:
                pending_approvals.pop(tool_call_id, None)

        # --- Two concurrent tasks ---

        # Queue for user messages arriving while the agent loop is running
        message_queue: asyncio.Queue = asyncio.Queue()

        async def receiver():
            """Reads all inbound WebSocket messages. Routes:
              - type == "message"       → queues user input for the agent loop
              - type == "tool_approval" → resolves pending approval Future
            """
            try:
                while True:
                    raw = await websocket.receive_text()
                    payload = json.loads(raw)

                    if payload.get("type") == "message":
                        content = payload.get("content", "").strip()
                        if content:
                            await message_queue.put(content)

                    elif payload.get("type") == "tool_approval":
                        tool_call_id = payload.get("tool_call_id", "")
                        approved = bool(payload.get("approved", False))
                        future = pending_approvals.get(tool_call_id)
                        if future and not future.done():
                            future.set_result(approved)
                            logger.info(
                                f"Tool approval received: tool_call_id={tool_call_id} approved={approved} "
                                f"user={user.uid}"
                            )
                        else:
                            logger.warning(f"Received tool_approval for unknown/expired tool_call_id={tool_call_id}")

            except WebSocketDisconnect:
                logger.info(f"WebSocket session {session_id} disconnected (receiver).")
            except Exception as e:
                logger.error(f"Receiver task error: {e}", exc_info=True)

        async def sender():
            """Waits for user messages from message_queue, runs the agent loop,
            and streams each event back to the client.
            """
            try:
                while True:
                    user_input = await message_queue.get()

                    async for event in service.run_agent_loop(
                        agent_id=agent_id,
                        session_id=session_id,
                        user_uid=user.uid,
                        new_message_content=user_input,
                        user_role=user_role,
                        approval_gate=approval_gate,
                    ):
                        await session.commit()
                        await websocket.send_json(event)

            except WebSocketDisconnect:
                logger.info(f"WebSocket session {session_id} disconnected (sender).")
            except Exception as e:
                logger.error(f"Sender task error in session {session_id}: {e}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Internal conversation loop failure",
                    })
                except Exception:
                    pass

        # 7. Run receiver and sender concurrently; cancel both when either exits
        receiver_task = asyncio.create_task(receiver())
        sender_task = asyncio.create_task(sender())

        done, pending = await asyncio.wait(
            [receiver_task, sender_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
