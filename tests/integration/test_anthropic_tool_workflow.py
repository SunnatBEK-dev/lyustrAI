from types import SimpleNamespace

import pytest

from ai_sdk.application.conversation_manager import ConversationManager
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.llm.anthropic import AnthropicClient
from ai_sdk.storage.json import JSONConversationRepository
from ai_sdk.tools import (
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)

pytestmark = pytest.mark.integration


class ScriptedMessagesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ScriptedAnthropicClient:
    def __init__(self, responses):
        self.messages = ScriptedMessagesAPI(responses)


def test_anthropic_tool_result_reaches_final_persisted_answer(tmp_path):
    provider = ScriptedAnthropicClient(
        [
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="tool_weather_1",
                        name="get_weather",
                        input={"city": "Samarqand"},
                    ),
                ],
                stop_reason="tool_use",
            ),
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text="Samarqandda havo 24°C.",
                    ),
                ],
                stop_reason="end_turn",
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        ToolSchema(
            name="get_weather",
            description="Get current weather for a city.",
            parameters=[
                ToolParameter(
                    "city",
                    ToolParameterType.STRING,
                    "City name.",
                ),
            ],
        ),
        lambda city: {"city": city, "temperature_c": 24},
    )
    repository = JSONConversationRepository(tmp_path / "chat.json")
    conversation = repository.load()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=AnthropicClient(
            client=provider,
            model="claude-test",
        ),
        repository=repository,
        tool_executor=ToolExecutor(registry),
    )

    answer = manager.send_message("Samarqanddagi ob-havoni tekshir.")
    restored = repository.load()

    assert answer == "Samarqandda havo 24°C."
    assert [message.content for message in restored.history()] == [
        "Samarqanddagi ob-havoni tekshir.",
        "Samarqandda havo 24°C.",
    ]
    assert len(provider.messages.calls) == 2
    tool_result = provider.messages.calls[1]["messages"][-1]
    assert tool_result == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool_weather_1",
                "content": ('{"city": "Samarqand", "temperature_c": 24}'),
                "is_error": False,
            },
        ],
    }
