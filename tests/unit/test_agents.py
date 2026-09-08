import pytest
import json
from types import SimpleNamespace
from apps.api.modules.agents.service import run_python_sandbox, AgentService
from apps.api.modules.agents.schemas import AgentResponse

def test_run_python_sandbox_success():
    code = """
def run(x, y):
    return x * y
"""
    result = run_python_sandbox(code, {"x": 3, "y": 4})
    assert result == "12"

def test_run_python_sandbox_list_dict_output():
    code = """
def run(items):
    return {"length": len(items), "processed": [i * 2 for i in items]}
"""
    result_str = run_python_sandbox(code, {"items": [1, 2, 3]})
    result = json.loads(result_str)
    assert result["length"] == 3
    assert result["processed"] == [2, 4, 6]

def test_run_python_sandbox_missing_run():
    code = """
def another_name(x):
    return x
"""
    with pytest.raises(ValueError) as exc:
        run_python_sandbox(code, {"x": 5})
    assert "must define a function named 'run" in str(exc.value)

def test_run_python_sandbox_compile_error():
    code = """
def run(x):
    invalid syntax here!!
"""
    with pytest.raises(RuntimeError) as exc:
        run_python_sandbox(code, {"x": 5})
    assert "Compilation/Setup failed" in str(exc.value)

def test_run_python_sandbox_runtime_error():
    code = """
def run(x):
    return x / 0
"""
    with pytest.raises(RuntimeError) as exc:
        run_python_sandbox(code, {"x": 5})
    assert "Execution failed" in str(exc.value)

def test_agent_schema_validation():
    data = {
        "id": 1,
        "name": "Test Agent",
        "description": "A test agent description",
        "system_prompt": "You are a test agent.",
        "model_id": "google/gemini-3.1-flash",
        "temperature": 0.5,
        "organization_id": 1,
        "created_by_uid": "user-abc",
        "created_at": "2026-08-08T00:00:00Z",
        "updated_at": "2026-08-08T00:00:00Z",
        "tools": []
    }
    response = AgentResponse.model_validate(data)
    assert response.name == "Test Agent"
    assert response.temperature == 0.5

class FakeAgentRepository:
    def __init__(self, agent):
        self.agent = agent
    async def get_agent(self, agent_id):
        return self.agent

class FakeAgentToolRepository:
    def __init__(self, tools=None):
        self.tools = tools or []
    async def get_tool(self, tool_name):
        return next((t for t in self.tools if getattr(t, "name", None) == tool_name), None)
    async def get_tool_for_agent(self, agent_id, tool_name):
        # Tests treat tools supplied to the fake repository as explicitly attached.
        return await self.get_tool(tool_name)
    async def log_tool_run(self, *args, **kwargs):
        pass

class FakeAgentSessionRepository:
    def __init__(self):
        self.messages = []
    async def create_message(self, session_id, role, content, name=None, tool_call_id=None, tool_calls=None):
        msg = SimpleNamespace(
            id=len(self.messages) + 1,
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls
        )
        self.messages.append(msg)
        return msg
    async def get_session(self, session_id):
        return SimpleNamespace(id=session_id, messages=self.messages)

@pytest.mark.asyncio
async def test_agent_service_mock_weather_loop():
    weather_tool_code = """
def run(city="London"):
    return f"Weather for {city}: 15C"
"""
    weather_tool = SimpleNamespace(
        id=1,
        name="get_weather",
        description="Gets weather details",
        code=weather_tool_code,
        parameter_schema={},
        is_active=True
    )
    agent = SimpleNamespace(
        id=42,
        name="Test Mock Agent",
        description="Demo",
        system_prompt="Be a test",
        model_id="test/mock-model",
        temperature=0.5,
        tools=[weather_tool]
    )
    
    agent_repo = FakeAgentRepository(agent)
    tool_repo = FakeAgentToolRepository([weather_tool])
    session_repo = FakeAgentSessionRepository()
    service = AgentService(agent_repo, tool_repo, session_repo)
    
    generator = service.run_agent_loop(
        agent_id=42,
        session_id="session-xyz",
        user_uid="user-abc",
        new_message_content="test weather"
    )
    
    events = []
    async for event in generator:
        events.append(event)
        
    assert len(events) > 0
    assert any(e["type"] == "tool_started" and e["tool_name"] == "get_weather" for e in events)
    assert any(e["type"] == "tool_completed" and "15C" in e["output"] for e in events)
    assert any(e["type"] == "text_delta" for e in events)
    assert any(e["type"] == "message_completed" for e in events)


def test_requires_approval_logic():
    service = AgentService(None, None, None)

    # Tool with require_approval=False -> False for everyone
    t1 = SimpleNamespace(require_approval=False, approval_required_for_roles=None)
    assert service._requires_approval(t1, "member") is False
    assert service._requires_approval(t1, "admin") is False

    # Tool with require_approval=True and empty roles list -> True for everyone
    t2 = SimpleNamespace(require_approval=True, approval_required_for_roles=None)
    assert service._requires_approval(t2, "member") is True
    assert service._requires_approval(t2, "owner") is True

    # Tool with require_approval=True and specific roles list -> True only for listed roles
    t3 = SimpleNamespace(require_approval=True, approval_required_for_roles=["member", "viewer"])
    assert service._requires_approval(t3, "member") is True
    assert service._requires_approval(t3, "viewer") is True
    assert service._requires_approval(t3, "admin") is False
    assert service._requires_approval(t3, "owner") is False


@pytest.mark.asyncio
async def test_agent_service_approval_gate_rejected():
    tool_code = "def run(): return 'done'"
    tool = SimpleNamespace(
        id=1,
        name="delete_database",
        description="Destructive tool",
        code=tool_code,
        parameter_schema={},
        is_active=True,
        require_approval=True,
        approval_required_for_roles=["member"]
    )
    agent = SimpleNamespace(
        id=42,
        name="Destructive Agent",
        description="Demo",
        system_prompt="Be a test",
        model_id="test/mock-model",
        temperature=0.5,
        tools=[tool]
    )

    agent_repo = FakeAgentRepository(agent)
    tool_repo = FakeAgentToolRepository()
    session_repo = FakeAgentSessionRepository()
    service = AgentService(agent_repo, tool_repo, session_repo)

    # Gate callback that returns False (reject)
    async def mock_reject_gate(tcid):
        return False

    events = []
    async for event in service.run_agent_loop(
        agent_id=42,
        session_id="session-xyz",
        user_uid="user-abc",
        new_message_content="run delete_database {}",
        user_role="member",
        approval_gate=mock_reject_gate,
    ):
        events.append(event)

    assert any(e["type"] == "tool_approval_requested" for e in events)
    assert any(e["type"] == "tool_denied" for e in events)
    assert not any(e["type"] == "tool_completed" for e in events)

