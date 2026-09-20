import pytest

import ai_sdk.llm.factory as llm_factory_module
from ai_sdk.agents import (
    AgentModelResponse,
    AgentTask,
    AgentTaskStatus,
    AgentTextBlock,
    MultiAgentCoordinator,
    create_provider_worker,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)

pytestmark = pytest.mark.integration


class ScriptedWorkerClient(BaseToolLLMClient):
    def __init__(self, responses):
        self.responses = list(responses)

    def ask(self, messages):
        return "unused"

    def stream(self, messages):
        yield "unused"

    def complete_tool_turn(self, messages, schemas, events):
        return self.responses.pop(0)


def test_named_provider_workers_produce_isolated_ordered_results(
    monkeypatch,
):
    registry = ToolRegistry()
    registry.register(
        ToolSchema(
            name="lookup",
            description="Look up a topic.",
            parameters=[
                ToolParameter(
                    "topic",
                    ToolParameterType.STRING,
                    "Topic to look up.",
                ),
            ],
        ),
        lambda topic: {"topic": topic, "fact": "verified"},
    )
    clients = {
        "gemini": ScriptedWorkerClient(
            [
                AgentModelResponse(
                    [
                        ToolCall(
                            "call_lookup",
                            "lookup",
                            {"topic": "Python"},
                        ),
                    ]
                ),
                AgentModelResponse(
                    [
                        AgentTextBlock("Research complete"),
                    ]
                ),
            ]
        ),
        "openai": ScriptedWorkerClient(
            [
                AgentModelResponse(
                    [
                        AgentTextBlock("Draft complete"),
                    ]
                ),
            ]
        ),
    }
    created_for = []

    def create_client(provider):
        created_for.append(provider)
        return clients[provider]

    monkeypatch.setattr(
        llm_factory_module,
        "create_llm_client",
        create_client,
    )
    researcher = create_provider_worker(
        "researcher",
        "Collect verified facts",
        "gemini",
        ToolExecutor(registry),
    )
    writer = create_provider_worker(
        "writer",
        "Write concise text",
        "openai",
    )
    coordinator = MultiAgentCoordinator(
        [
            researcher,
            writer,
        ]
    )

    result = coordinator.run(
        [
            AgentTask(
                "task_research",
                "researcher",
                "Research Python",
            ),
            AgentTask(
                "task_write",
                "writer",
                "Write a short draft",
            ),
        ]
    )

    assert [item.status for item in result.results] == [
        AgentTaskStatus.COMPLETED,
        AgentTaskStatus.COMPLETED,
    ]
    assert [item.output for item in result.results] == [
        "Research complete",
        "Draft complete",
    ]
    assert result.results[0].state.tool_rounds == 1
    assert result.results[1].state.tool_rounds == 0
    assert created_for == ["gemini", "openai"]
    assert researcher.provider == "gemini"
    assert writer.provider == "openai"
    assert result.results[0].state is not result.results[1].state
