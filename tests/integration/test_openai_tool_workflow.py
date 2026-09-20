from types import SimpleNamespace

import pytest

from ai_sdk.application.conversation_manager import ConversationManager
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.llm.openai import OpenAIClient
from ai_sdk.storage.json import JSONConversationRepository
from ai_sdk.tools import (
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)

pytestmark = pytest.mark.integration


class ScriptedResponsesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ScriptedOpenAI:
    def __init__(self, responses):
        self.responses = ScriptedResponsesAPI(responses)


def message(text):
    return SimpleNamespace(
        type="message",
        content=[
            SimpleNamespace(
                type="output_text",
                text=text,
            )
        ],
    )


def test_openai_tool_result_reaches_final_persisted_answer(tmp_path):
    provider = ScriptedOpenAI(
        [
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        call_id="call_weather_1",
                        name="get_weather",
                        arguments='{"city": "Samarqand"}',
                    )
                ]
            ),
            SimpleNamespace(output=[message("Samarqandda havo 24°C.")]),
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
                )
            ],
        ),
        lambda city: {"city": city, "temperature_c": 24},
    )
    repository = JSONConversationRepository(tmp_path / "chat.json")
    conversation = repository.load()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=OpenAIClient(
            client=provider,
            model="gpt-test",
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
    assert len(provider.responses.calls) == 2
    tool_result = provider.responses.calls[1]["input"][-1]
    assert tool_result == {
        "type": "function_call_output",
        "call_id": "call_weather_1",
        "output": ('{"city": "Samarqand", "temperature_c": 24}'),
    }
