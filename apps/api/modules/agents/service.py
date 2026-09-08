"""
Business services for compiling tools, executing sandbox runs, and LLM loops.
"""
import time
import traceback
import json
import logging
import asyncio
from typing import AsyncGenerator, Awaitable, Callable, Dict, Any, List, Optional

from apps.api.config import settings
from apps.api.modules.llm.client import create_async_client
from apps.api.modules.agents.models import Agent, AgentTool
from apps.api.modules.agents.repository import (
    AgentToolRepository,
    AgentRepository,
    AgentSessionRepository,
    AgentUsageRepository,
)
from apps.api.modules.agents.pricing import TokenExtractor, calculate_turn_cost
from apps.api.modules.agents.sample_tools import (
    SAMPLE_TOOLS_METADATA,
    MOCK_WELCOME_MESSAGE,
    match_sample_tool_trigger,
)
from apps.api.modules.agents.toon_utils import compress_tool_output_for_llm
from apps.api.modules.agents.memory import prune_tool_messages, generate_and_update_session_summary

logger = logging.getLogger(__name__)


def run_python_sandbox(code: str, arguments: Dict[str, Any]) -> str:
    """
    Compiles and executes python code in a clean namespace.
    The code MUST define a 'run' function.

    This is not OS-level sandboxing. Until tools execute in an isolated worker,
    dynamic Python execution is restricted to explicit local/test environments.
    """
    if not settings.is_development_environment:
        raise RuntimeError(
            "In-process Python tool execution is disabled outside local/test environments"
        )

    local_env: Dict[str, Any] = {}
    global_env = {
        "__builtins__": __builtins__,
        "import": __import__,
    }
    
    # Compile and execute the script block to populate env with definitions
    try:
        compiled = compile(code, "<agent_tool>", "exec")
        exec(compiled, global_env, local_env)
    except Exception as e:
        error_msg = f"Compilation/Setup failed:\n{traceback.format_exc()}"
        raise RuntimeError(error_msg) from e

    # Find the 'run' callable
    run_func = local_env.get("run") or global_env.get("run")
    if not run_func:
        raise ValueError("The tool script must define a function named 'run(**kwargs)'")
    if not callable(run_func):
        raise TypeError("'run' must be a callable function")

    # Run the function
    try:
        result = run_func(**arguments)
        if result is None:
            return ""
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)
    except Exception as e:
        error_msg = f"Execution failed:\n{traceback.format_exc()}"
        raise RuntimeError(error_msg) from e


class AgentService:
    def __init__(
        self,
        agent_repo: AgentRepository,
        tool_repo: AgentToolRepository,
        session_repo: AgentSessionRepository,
        usage_repo: Optional[AgentUsageRepository] = None,
    ):
        self.agent_repo = agent_repo
        self.tool_repo = tool_repo
        self.session_repo = session_repo
        self.usage_repo = usage_repo or (AgentUsageRepository(agent_repo.session) if hasattr(agent_repo, "session") and agent_repo.session else None)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        triggered_by: str,
        approval_status: Optional[str] = None,
        approved_by: Optional[str] = None,
        organization_id: Optional[int] = None,
        agent_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Loads and runs an authorized tool script, logging results."""
        if agent_id is not None:
            tool = await self.tool_repo.get_tool_for_agent(agent_id, tool_name)
        else:
            # Global lookup is reserved for explicit SuperAdmin test execution.
            tool = await self.tool_repo.get_tool(tool_name)

        if not tool or not tool.is_active:
            raise ValueError(f"Tool '{tool_name}' is not available to this agent")

        start_time = time.perf_counter()
        output = None
        error = None
        success = False

        try:
            output = run_python_sandbox(tool.code, arguments)
            success = True
        except Exception as e:
            error = str(e)
            logger.error(f"Error running tool {tool_name}: {error}")
        finally:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            await self.tool_repo.log_tool_run(
                tool_name=tool_name,
                triggered_by=triggered_by,
                arguments=arguments,
                output=output,
                error=error,
                duration_ms=duration_ms,
                approval_status=approval_status,
                approved_by=approved_by,
                organization_id=organization_id,
                agent_id=agent_id,
            )

        return {
            "success": success,
            "output": output,
            "error": error,
            "duration_ms": duration_ms,
        }

    def _requires_approval(self, tool: Any, user_role: Optional[str]) -> bool:
        """
        Determines whether this tool call should be gated behind user approval.

        Logic:
          - If tool.require_approval is False → no gate
          - If approval_required_for_roles is None/empty → ALL roles are gated
          - Otherwise only roles listed in approval_required_for_roles are gated
        """
        if not getattr(tool, "require_approval", False):
            return False
        required_for_roles = getattr(tool, "approval_required_for_roles", None)
        if not required_for_roles:
            # Empty list or None means everyone is gated
            return True
        return (user_role or "member") in required_for_roles

    async def run_agent_loop(
        self,
        agent_id: int,
        session_id: str,
        user_uid: str,
        new_message_content: str,
        user_role: Optional[str] = None,
        approval_gate: Optional[Callable[[str], Awaitable[bool]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the conversational loop for an agent.
        1. Appends the user's message to the session history.
        2. Pulls complete historical messages context.
        3. Calls LLM with tool descriptors.
        4. Streams content and handles recursive tool calls.

        When a tool has require_approval=True and the user's role is in the
        gated roles list, emits tool_approval_requested and suspends via
        approval_gate until the user approves or rejects from the UI.
        No extra LLM call is needed — the result (or rejection) is fed back
        directly into the tool message context.
        """
        agent = await self.agent_repo.get_agent(agent_id)
        if not agent:
            yield {"type": "error", "message": "Agent not found"}
            return

        org_id = getattr(agent, "organization_id", 1) or 1

        # Pre-flight token quota reservation check (< 1ms)
        if self.usage_repo:
            allowed, quota = await self.usage_repo.reserve_quota(org_id, estimated_tokens=4000)
            if not allowed:
                yield {
                    "type": "error",
                    "message": f"Monthly token quota exceeded ({quota.tokens_used_this_month:,} / {quota.monthly_token_quota:,} tokens used). Contact your administrator to upgrade quota."
                }
                return

        # 1. Log the new user message
        await self.session_repo.create_message(
            session_id=session_id,
            role="user",
            content=new_message_content,
        )

        loop_start_time = time.perf_counter()
        accumulated_assistant_chunks = []
        total_tool_calls_count = 0
        toon_tokens_saved = 0
        turn_success = False

        try:
            if agent.model_id == "test/mock-model":
                user_msg = new_message_content.strip()
                
                # Match explicit instructions: "run <tool_name> <arguments_json>"
                import re
                match = re.match(r"^run\s+([a-zA-Z0-9_-]+)(?:\s+(.+))?$", user_msg, re.IGNORECASE)
                
                tool_to_run = None
                tool_args = {}
                
                if match:
                    target_tool_name = match.group(1)
                    args_str = match.group(2) or "{}"
                    for t in agent.tools:
                        if t.name == target_tool_name:
                            tool_to_run = t
                            break
                    if tool_to_run:
                        try:
                            tool_args = json.loads(args_str)
                        except Exception:
                            tool_args = {}
                
                if tool_to_run:
                    thinking_text = f"*(thinking: compiling and executing tool script '{tool_to_run.name}' in sandbox...)*\n\n"
                    yield {"type": "text_delta", "text": thinking_text}
                    accumulated_assistant_chunks.append(thinking_text)
                    await asyncio.sleep(0.4)

                    tool_call_id = f"call_mock_{tool_to_run.name}_{int(time.time())}"
                    total_tool_calls_count += 1
                    yield {
                        "type": "tool_started",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_to_run.name,
                        "arguments": tool_args
                    }

                    # --- Approval gate (mock model path) ---
                    if self._requires_approval(tool_to_run, user_role) and approval_gate:
                        yield {
                            "type": "tool_approval_requested",
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_to_run.name,
                            "arguments": tool_args,
                            "ui_mode": getattr(tool_to_run, "ui_mode", "inline"),
                            "display_label_running": getattr(tool_to_run, "display_label_running", None),
                            "display_label_completed": getattr(tool_to_run, "display_label_completed", None),
                        }
                        approved = await approval_gate(tool_call_id)
                        if not approved:
                            rejection = "Action was rejected by the user."
                            yield {"type": "tool_denied", "tool_call_id": tool_call_id, "tool_name": tool_to_run.name, "reason": rejection}
                            await self.session_repo.create_message(
                                session_id=session_id, role="assistant", content=None,
                                tool_calls=[{"id": tool_call_id, "type": "function", "function": {"name": tool_to_run.name, "arguments": json.dumps(tool_args)}}]
                            )
                            await self.session_repo.create_message(
                                session_id=session_id, role="tool", content=rejection,
                                tool_call_id=tool_call_id, name=tool_to_run.name
                            )
                            yield {"type": "message_completed"}
                            turn_success = True
                            return

                    try:
                        tool_output = run_python_sandbox(tool_to_run.code, tool_args)
                        is_error = False
                    except Exception as e:
                        tool_output = f"Execution Error: {str(e)}"
                        is_error = True

                    await asyncio.sleep(0.6)

                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="assistant",
                        content=None,
                        tool_calls=[{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_to_run.name, "arguments": json.dumps(tool_args)}
                        }]
                    )
                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="tool",
                        content=tool_output,
                        tool_call_id=tool_call_id,
                        name=tool_to_run.name
                    )

                    yield {
                        "type": "tool_completed",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_to_run.name,
                        "output": tool_output,
                        "error": tool_output if is_error else None
                    }
                    await asyncio.sleep(0.4)

                    final_text = f"Sandbox Result: {tool_output}"
                    yield {"type": "text_delta", "text": final_text}
                    accumulated_assistant_chunks.append(final_text)
                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="assistant",
                        content=thinking_text + final_text
                    )
                    yield {"type": "message_completed"}
                    turn_success = True
                    return

                sample_tool_name = match_sample_tool_trigger(user_msg)
                if sample_tool_name and sample_tool_name in SAMPLE_TOOLS_METADATA:
                    cfg = SAMPLE_TOOLS_METADATA[sample_tool_name]
                    matched_tool = next((t for t in agent.tools if sample_tool_name in t.name.lower()), None)
                    tool_args = cfg["default_args"]
                    tool_name = matched_tool.name if matched_tool else sample_tool_name
                    ui_mode = getattr(matched_tool, "ui_mode", cfg["ui_mode"]) if matched_tool else cfg["ui_mode"]
                    label_running = getattr(matched_tool, "display_label_running", cfg["display_label_running"]) if matched_tool else cfg["display_label_running"]
                    label_completed = getattr(matched_tool, "display_label_completed", cfg["display_label_completed"]) if matched_tool else cfg["display_label_completed"]

                    thinking_text = cfg["thinking_text"]
                    yield {"type": "text_delta", "text": thinking_text}
                    accumulated_assistant_chunks.append(thinking_text)
                    await asyncio.sleep(0.4)

                    tool_call_id = f"call_mock_{sample_tool_name}_{int(time.time())}"
                    total_tool_calls_count += 1
                    yield {
                        "type": "tool_started",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "ui_mode": ui_mode,
                        "display_label_running": label_running,
                        "display_label_completed": label_completed,
                    }

                    # --- Approval gate (sample tools path) ---
                    sample_tool_obj = matched_tool
                    if sample_tool_obj and self._requires_approval(sample_tool_obj, user_role) and approval_gate:
                        yield {
                            "type": "tool_approval_requested",
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "ui_mode": ui_mode,
                            "display_label_running": label_running,
                            "display_label_completed": label_completed,
                        }
                        approved = await approval_gate(tool_call_id)
                        if not approved:
                            rejection = "Action was rejected by the user."
                            yield {"type": "tool_denied", "tool_call_id": tool_call_id, "tool_name": tool_name, "reason": rejection}
                            await self.session_repo.create_message(
                                session_id=session_id, role="assistant", content=None,
                                tool_calls=[{"id": tool_call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]
                            )
                            await self.session_repo.create_message(
                                session_id=session_id, role="tool", content=rejection,
                                tool_call_id=tool_call_id, name=tool_name
                            )
                            yield {"type": "message_completed"}
                            turn_success = True
                            return

                    if matched_tool:
                        try:
                            run_res = await self.execute_tool(
                                tool_name=matched_tool.name,
                                arguments=tool_args,
                                triggered_by=user_uid,
                                approval_status="approved" if self._requires_approval(matched_tool, user_role) else None,
                                approved_by=user_uid if self._requires_approval(matched_tool, user_role) else None,
                                organization_id=org_id,
                                agent_id=agent.id,
                            )
                            tool_output = run_res["output"] if run_res["success"] else f"Error: {run_res['error']}"
                            is_error = not run_res["success"]
                        except Exception as e:
                            tool_output = f"Execution Error: {str(e)}"
                            is_error = True
                    else:
                        tool_output = cfg["mock_output"]
                        is_error = False
                        await self.tool_repo.log_tool_run(
                            tool_name=tool_name,
                            triggered_by=user_uid,
                            arguments=tool_args,
                            output=tool_output,
                            error=None,
                            duration_ms=45,
                            approval_status=None,
                            approved_by=None,
                            organization_id=org_id,
                            agent_id=agent.id,
                        )

                    await asyncio.sleep(0.8)

                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="assistant",
                        content=None,
                        tool_calls=[{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)}
                        }]
                    )
                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="tool",
                        content=tool_output,
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )

                    yield {
                        "type": "tool_completed",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "output": tool_output,
                        "error": tool_output if is_error else None,
                        "ui_mode": ui_mode,
                        "display_label_running": label_running,
                        "display_label_completed": label_completed,
                    }
                    await asyncio.sleep(0.4)

                    final_text = cfg["final_text"]
                    for word in final_text.split(" "):
                        yield {"type": "text_delta", "text": word + " "}
                        accumulated_assistant_chunks.append(word + " ")
                        await asyncio.sleep(0.06)

                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="assistant",
                        content=thinking_text + final_text
                    )
                    yield {"type": "message_completed"}
                    turn_success = True
                    return

                else:
                    assistant_text = MOCK_WELCOME_MESSAGE
                    for word in assistant_text.split(" "):
                        yield {"type": "text_delta", "text": word + " "}
                        accumulated_assistant_chunks.append(word + " ")
                        await asyncio.sleep(0.06)

                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="assistant",
                        content=assistant_text
                    )
                    yield {"type": "message_completed"}
                    turn_success = True
                    return

            # 2. Re-fetch session context using DB query pagination (LIMIT to agent.summarization_threshold)
            db_session = await self.session_repo.get_session(session_id)
            limit = getattr(agent, "summarization_threshold", 10) or 10
            recent_msgs = await self.session_repo.get_recent_messages(session_id, limit=limit)

            session_summary = getattr(db_session, "summary", None)
            if len(recent_msgs) >= limit and not session_summary:
                summary_model = getattr(agent, "summary_model_id", "openai/gpt-4o-mini") or "openai/gpt-4o-mini"
                messages_to_summarize = [
                    {"role": m.role, "content": m.content}
                    for m in recent_msgs[:max(1, len(recent_msgs) - 5)]
                ]
                session_summary = await generate_and_update_session_summary(
                    session_repo=self.session_repo,
                    session_id=session_id,
                    messages_to_summarize=messages_to_summarize,
                    summary_model_id=summary_model,
                    existing_summary=session_summary,
                )

            messages_context = []
            messages_context.append({"role": "system", "content": agent.system_prompt})
            
            if session_summary:
                messages_context.append({
                    "role": "system",
                    "content": f"Summary of earlier conversation history:\n{session_summary}"
                })

            for msg in recent_msgs:
                msg_dict = {"role": msg.role}
                if msg.content is not None:
                    msg_dict["content"] = msg.content
                if msg.tool_calls is not None:
                    msg_dict["tool_calls"] = msg.tool_calls
                if msg.tool_call_id is not None:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if msg.name is not None:
                    msg_dict["name"] = msg.name
                messages_context.append(msg_dict)

            pruning_turns = getattr(agent, "tool_pruning_turns", 3) or 3
            messages_context = prune_tool_messages(messages_context, max_recent_tool_turns=pruning_turns)

            tools_specs = []
            for t in agent.tools:
                if t.is_active:
                    tools_specs.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameter_schema,
                        }
                    })

            client = create_async_client()

            max_turns = 5
            for turn in range(max_turns):
                logger.info(f"Agent Loop turn {turn} for agent {agent_id}")
                
                call_params = {
                    "model": agent.model_id,
                    "messages": messages_context,
                    "temperature": agent.temperature,
                    "stream": True,
                }
                if tools_specs:
                    call_params["tools"] = tools_specs

                response_stream = await client.chat.completions.create(**call_params)

                full_text = []
                tool_calls_map = {}

                async for chunk in response_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        full_text.append(delta.content)
                        accumulated_assistant_chunks.append(delta.content)
                        yield {"type": "text_delta", "text": delta.content}

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": tc.id,
                                    "name": tc.function.name if tc.function else "",
                                    "arguments_chunks": [tc.function.arguments if (tc.function and tc.function.arguments) else ""],
                                }
                            else:
                                if tc.id:
                                    tool_calls_map[idx]["id"] = tc.id
                                if tc.function and tc.function.name:
                                    tool_calls_map[idx]["name"] = tc.function.name
                                if tc.function and tc.function.arguments:
                                    tool_calls_map[idx]["arguments_chunks"].append(tc.function.arguments)

                tool_calls = []
                for idx, tc in tool_calls_map.items():
                    args_str = "".join(tc["arguments_chunks"])
                    args = {}
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except Exception:
                        logger.warning(f"Failed to parse arguments JSON: {args_str}")
                    
                    tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": args_str,
                        },
                        "parsed_arguments": args
                    })

                if not tool_calls:
                    assistant_response_content = "".join(full_text)
                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="assistant",
                        content=assistant_response_content,
                    )
                    yield {"type": "message_completed"}
                    turn_success = True
                    return

                openai_tool_calls_format = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    }
                    for tc in tool_calls
                ]
                
                await self.session_repo.create_message(
                    session_id=session_id,
                    role="assistant",
                    content=None,
                    tool_calls=openai_tool_calls_format,
                )
                messages_context.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": openai_tool_calls_format,
                })

                for tc in tool_calls:
                    total_tool_calls_count += 1
                    tool_name = tc["function"]["name"]
                    tool_id = tc["id"]
                    tool_args = tc["parsed_arguments"]

                    matched_tool = next((t for t in agent.tools if t.name == tool_name and t.is_active), None)
                    if matched_tool is None:
                        rejection = f"Tool '{tool_name}' is not available to this agent."
                        logger.warning(
                            "Rejected unattached or inactive tool call: "
                            f"agent_id={agent.id} tool={tool_name} user={user_uid}"
                        )
                        yield {
                            "type": "tool_denied",
                            "tool_call_id": tool_id,
                            "tool_name": tool_name,
                            "reason": rejection,
                        }
                        await self.session_repo.create_message(
                            session_id=session_id,
                            role="tool",
                            content=rejection,
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                        messages_context.append({
                            "role": "tool",
                            "content": rejection,
                            "tool_call_id": tool_id,
                            "name": tool_name,
                        })
                        continue

                    ui_mode = getattr(matched_tool, "ui_mode", "inline")
                    label_running = getattr(matched_tool, "display_label_running", None)
                    label_completed = getattr(matched_tool, "display_label_completed", None)

                    yield {
                        "type": "tool_started",
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "arguments": tool_args,
                        "ui_mode": ui_mode,
                        "display_label_running": label_running,
                        "display_label_completed": label_completed,
                    }

                    needs_approval = self._requires_approval(matched_tool, user_role)
                    if needs_approval and approval_gate:
                        yield {
                            "type": "tool_approval_requested",
                            "tool_call_id": tool_id,
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "ui_mode": ui_mode,
                            "display_label_running": label_running,
                            "display_label_completed": label_completed,
                        }
                        approved = await approval_gate(tool_id)
                        if not approved:
                            rejection = "Action was rejected by the user."
                            yield {"type": "tool_denied", "tool_call_id": tool_id, "tool_name": tool_name, "reason": rejection}
                            await self.session_repo.create_message(
                                session_id=session_id, role="tool", content=rejection,
                                tool_call_id=tool_id, name=tool_name,
                            )
                            messages_context.append({
                                "role": "tool",
                                "content": rejection,
                                "tool_call_id": tool_id,
                                "name": tool_name,
                            })
                            continue

                    run_res = await self.execute_tool(
                        tool_name=tool_name,
                        arguments=tool_args,
                        triggered_by=user_uid,
                        approval_status="approved" if needs_approval else None,
                        approved_by=user_uid if needs_approval else None,
                        organization_id=org_id,
                        agent_id=agent.id,
                    )

                    tool_output_str = run_res["output"] if run_res["success"] else f"Error: {run_res['error']}"

                    yield {
                        "type": "tool_completed",
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "output": tool_output_str,
                        "error": run_res["error"],
                        "ui_mode": ui_mode,
                        "display_label_running": label_running,
                        "display_label_completed": label_completed,
                    }

                    compressed_output = compress_tool_output_for_llm(tool_output_str)
                    toon_tokens_saved += max(0, (len(tool_output_str) - len(compressed_output)) // 4)

                    await self.session_repo.create_message(
                        session_id=session_id,
                        role="tool",
                        content=tool_output_str,
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                    messages_context.append({
                        "role": "tool",
                        "content": compressed_output,
                        "tool_call_id": tool_id,
                        "name": tool_name,
                    })
            else:
                yield {"type": "error", "message": "Max turns reached in tool calling loops"}
                turn_success = True
        finally:
            latency_ms = int((time.perf_counter() - loop_start_time) * 1000)
            if self.usage_repo:
                if turn_success:
                    full_assistant_reply = "".join(accumulated_assistant_chunks)
                    p_tok, c_tok, tot_tok = TokenExtractor.extract(
                        provider=settings.llm_provider,
                        usage_payload=None,
                        input_text=new_message_content,
                        output_text=full_assistant_reply,
                    )
                    cost_usd = calculate_turn_cost(agent.model_id, p_tok, c_tok)
                    await self.usage_repo.settle_quota(
                        organization_id=org_id,
                        actual_tokens=tot_tok,
                        actual_cost_usd=cost_usd,
                        reserved_estimate=4000,
                    )
                    await self.usage_repo.log_turn_usage(
                        organization_id=org_id,
                        agent_id=agent.id,
                        session_id=session_id,
                        user_uid=user_uid,
                        provider=settings.llm_provider,
                        model_id=agent.model_id,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=tot_tok,
                        toon_tokens_saved=toon_tokens_saved,
                        estimated_cost_usd=cost_usd,
                        latency_ms=latency_ms,
                        tool_calls_count=total_tool_calls_count,
                    )
                else:
                    await self.usage_repo.release_reservation(org_id, reserved_estimate=4000)
