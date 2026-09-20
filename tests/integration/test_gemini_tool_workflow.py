from types import SimpleNamespace

import pytest
from google.genai import interactions

from ai_sdk.application.conversation_manager import ConversationManager
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.llm.gemini import GeminiClient
from ai_sdk.storage.json import JSONConversationRepository
from ai_sdk.tools import (
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)

pytestmark = pytest.mark.integration


class ScriptedInteractionsAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ScriptedGemini:
    def __init__(self, responses):
        self.interactions = ScriptedInteractionsAPI(responses)


def model_output(text):
    return interactions.ModelOutputStep(content=[interactions.TextContent(text=text)])


def test_gemini_tool_result_reaches_final_persisted_answer(tmp_path):
    provider = ScriptedGemini(
        [
            SimpleNamespace(
                steps=[
                    interactions.ThoughtStep(signature="opaque"),
                    interactions.FunctionCallStep(
                        id="call_weather_1",
                        name="get_weather",
                        arguments={"city": "Samarqand"},
                    ),
                ]
            ),
            SimpleNamespace(steps=[model_output("Samarqandda havo 24°C.")]),
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
        client=GeminiClient(
            client=provider,
            model="gemini-test",
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
    assert len(provider.interactions.calls) == 2
    second_input = provider.interactions.calls[1]["input"]
    assert second_input[1] == {
        "signature": "opaque",
        "type": "thought",
    }
    assert second_input[-1] == {
        "type": "function_result",
        "name": "get_weather",
        "call_id": "call_weather_1",
        "result": [
            {
                "type": "text",
                "text": ('{"city": "Samarqand", "temperature_c": 24}'),
            }
        ],
    }
