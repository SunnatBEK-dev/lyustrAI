import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    HandoffStage,
    MultiAgentCoordinator,
    SequentialHandoffCoordinator,
)
from ai_sdk.llm.adaptive_multi_model import MultiModelWorkflowClient
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import ToolExecutor, ToolRegistry


class Client(BaseToolLLMClient):
    def __init__(self, response="done", error=None):
        self.response = response
        self.error = error
        self.messages = []

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        self.messages.append(messages)
        if self.error:
            raise self.error
        return AgentModelResponse([AgentTextBlock(self.response)])


def build_client(provider_client):
    worker = AgentWorker(
        "worker",
        "Answer requests",
        AgentRunner(
            provider_client,
            ToolExecutor(ToolRegistry()),
        ),
    )
    return MultiModelWorkflowClient(
        SequentialHandoffCoordinator(
            MultiAgentCoordinator([worker]),
            [HandoffStage("final", "worker", "Answer")],
        )
    )


def test_adaptive_client_exposes_workflow_as_text_client():
    provider = Client("combined answer")
    client = build_client(provider)

    response = client.ask(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Older answer"},
            {"role": "user", "content": "Follow-up"},
        ]
    )

    assert response == "combined answer"
    assert client.last_result.completed is True
    prompt = provider.messages[0][0]["content"]
    assert "Conversation transcript (untrusted data):" in prompt
    assert "USER: Question" in prompt
    assert "ASSISTANT: Older answer" in prompt
    assert "USER: Follow-up" in prompt


def test_adaptive_client_contains_stage_failure():
    client = build_client(Client(error=RuntimeError("private detail")))

    with pytest.raises(RuntimeError, match="stage: final") as error:
        client.ask([{"role": "user", "content": "Question"}])

    assert "private detail" not in str(error.value)
    assert client.last_result.completed is False


def test_adaptive_client_rejects_streaming_and_invalid_input():
    client = build_client(Client())

    with pytest.raises(RuntimeError, match="streaming"):
        client.stream([{"role": "user", "content": "Question"}])
    with pytest.raises(ValueError, match="cannot be empty"):
        client.ask([])
    with pytest.raises(ValueError, match="role"):
        client.ask([{"role": "system", "content": "No"}])
    with pytest.raises(TypeError, match="content"):
        client.ask([{"role": "user", "content": None}])
    with pytest.raises(TypeError, match="workflow"):
        MultiModelWorkflowClient(object())
