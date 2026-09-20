from copy import deepcopy
from types import SimpleNamespace

import pytest

import ai_sdk.llm.gemini as gemini_module
from ai_sdk.agents.model import (
    AgentEvent,
    AgentModelResponse,
    AgentTextBlock,
)
from ai_sdk.llm.gemini import GeminiClient
from ai_sdk.tools import (
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)
from ai_sdk.tools.model import ToolCall, ToolResult


class FakeStep(SimpleNamespace):
    def model_dump(self):
        return deepcopy(self.dump)


class QueuedInteractionsAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.responses.pop(0)


class FakeGemini:
    def __init__(self, responses):
        self.interactions = QueuedInteractionsAPI(responses)


def step(step_type, **values):
    dump = {
        "type": step_type,
        **{key: _dump_value(value) for key, value in values.items()},
    }
    return FakeStep(type=step_type, dump=dump, **values)


def _dump_value(value):
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return {
            key: _dump_value(item) for key, item in vars(value).items() if key != "dump"
        }
    return deepcopy(value)


def interaction(*steps, output_text=""):
    return SimpleNamespace(
        steps=list(steps),
        output_text=output_text,
    )


def text_content(text):
    return SimpleNamespace(type="text", text=text)


def model_output(text):
    return step(
        "model_output",
        content=[text_content(text)],
    )


def function_call(call_id, name, arguments):
    return step(
        "function_call",
        id=call_id,
        name=name,
        arguments=arguments,
    )


def build_add_executor(handler=lambda left, right: left + right):
    registry = ToolRegistry()
    registry.register(
        ToolSchema(
            name="add",
            description="Add two integers.",
            parameters=[
                ToolParameter(
                    "left",
                    ToolParameterType.INTEGER,
                    "First integer.",
                ),
                ToolParameter(
                    "right",
                    ToolParameterType.INTEGER,
                    "Second integer.",
                ),
            ],
        ),
        handler,
    )
    return ToolExecutor(registry)


def test_ask_uses_stateless_interactions_api():
    provider = FakeGemini(
        [
            interaction(output_text="Salom"),
        ]
    )
    client = GeminiClient(
        client=provider,
        model="gemini-test",
        max_output_tokens=77,
        timeout=12.0,
    )
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Continue"},
    ]

    result = client.ask(messages)

    assert result == "Salom"
    assert provider.interactions.create_calls == [
        {
            "model": "gemini-test",
            "input": [
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": "Hi"}],
                },
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Hello"}],
                },
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": "Continue"}],
                },
            ],
            "generation_config": {"max_output_tokens": 77},
            "store": False,
            "timeout": 12.0,
            "system_instruction": "Be concise.",
        }
    ]
    assert messages[1]["content"] == "Hi"


def test_multiple_system_messages_are_combined():
    provider = FakeGemini([interaction(output_text="OK")])
    client = GeminiClient(client=provider, model="gemini-test")

    client.ask(
        [
            {"role": "system", "content": "First"},
            {"role": "system", "content": "Second"},
            {"role": "user", "content": "Go"},
        ]
    )

    request = provider.interactions.create_calls[0]
    assert request["system_instruction"] == "First\n\nSecond"


def test_stream_yields_only_text_step_deltas():
    stream = [
        SimpleNamespace(event_type="interaction.created"),
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="thought_summary"),
        ),
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="text", text="one"),
        ),
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="text", text="two"),
        ),
    ]
    provider = FakeGemini([stream])
    client = GeminiClient(client=provider, model="gemini-test")

    chunks = list(client.stream([{"role": "user", "content": "Hi"}]))

    assert chunks == ["one", "two"]
    assert provider.interactions.create_calls[0]["stream"] is True


def test_ask_with_tools_preserves_all_gemini_steps():
    thought_dump = {
        "type": "thought",
        "signature": "opaque-signature",
    }
    provider = FakeGemini(
        [
            interaction(
                FakeStep(type="thought", dump=thought_dump),
                model_output("I will calculate it."),
                function_call(
                    "call_1",
                    "add",
                    {"left": 2, "right": 3},
                ),
            ),
            interaction(model_output("The answer is 5.")),
        ]
    )
    client = GeminiClient(client=provider, model="gemini-test")
    messages = [{"role": "user", "content": "What is 2 + 3?"}]

    result = client.ask_with_tools(
        messages,
        build_add_executor(),
    )

    assert result == "The answer is 5."
    assert messages == [{"role": "user", "content": "What is 2 + 3?"}]
    first_call, second_call = provider.interactions.create_calls
    assert first_call["tools"] == [
        {
            "type": "function",
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "left": {
                        "type": "integer",
                        "description": "First integer.",
                    },
                    "right": {
                        "type": "integer",
                        "description": "Second integer.",
                    },
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
        }
    ]
    assert second_call["input"] == [
        {
            "type": "user_input",
            "content": [
                {
                    "type": "text",
                    "text": "What is 2 + 3?",
                }
            ],
        },
        thought_dump,
        {
            "type": "model_output",
            "content": [
                {
                    "type": "text",
                    "text": "I will calculate it.",
                }
            ],
        },
        {
            "type": "function_call",
            "id": "call_1",
            "name": "add",
            "arguments": {"left": 2, "right": 3},
        },
        {
            "type": "function_result",
            "name": "add",
            "call_id": "call_1",
            "result": [{"type": "text", "text": "5"}],
        },
    ]


def test_tool_errors_are_marked_in_function_result():
    provider = FakeGemini(
        [
            interaction(function_call("missing_1", "missing", {})),
            interaction(model_output("Tool unavailable.")),
        ]
    )
    client = GeminiClient(client=provider, model="gemini-test")

    result = client.ask_with_tools(
        [{"role": "user", "content": "Use missing."}],
        build_add_executor(),
    )

    assert result == "Tool unavailable."
    tool_result = provider.interactions.create_calls[1]["input"][-1]
    assert tool_result == {
        "type": "function_result",
        "name": "missing",
        "call_id": "missing_1",
        "result": [
            {
                "type": "text",
                "text": "Unknown tool: missing",
            }
        ],
        "is_error": True,
    }


def test_complete_turn_without_tools_omits_tools_field():
    provider = FakeGemini(
        [
            interaction(model_output("Done.")),
        ]
    )
    client = GeminiClient(client=provider, model="gemini-test")

    response = client.complete_tool_turn(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        [],
        (),
    )

    assert response.text == "Done."
    request = provider.interactions.create_calls[0]
    assert request["system_instruction"] == "Be concise."
    assert "tools" not in request


def test_agent_input_can_reconstruct_neutral_events():
    response = AgentModelResponse(
        [
            AgentTextBlock("Checking."),
            ToolCall("call_1", "add", {"left": 1, "right": 2}),
        ]
    )
    event = AgentEvent(
        1,
        response,
        [ToolResult("call_1", "add", "3")],
    )

    _, provider_input = GeminiClient._agent_input(
        [{"role": "user", "content": "Add."}],
        (event,),
    )

    assert [item["type"] for item in provider_input] == [
        "user_input",
        "model_output",
        "function_call",
        "function_result",
    ]


def test_parser_ignores_non_text_content_and_unknown_steps():
    parsed = GeminiClient._parse_agent_response(
        [
            step("thought", signature="opaque"),
            step(
                "model_output",
                content=[
                    SimpleNamespace(type="image"),
                    text_content("Answer"),
                ],
            ),
        ]
    )

    assert parsed.text == "Answer"
    assert parsed.tool_calls == ()


class InvalidDumpStep(SimpleNamespace):
    def model_dump(self):
        return "invalid"


@pytest.mark.parametrize(
    "steps",
    [
        None,
        [SimpleNamespace(type="thought")],
        [InvalidDumpStep(type="thought")],
        [step("model_output", content="invalid")],
        [
            step(
                "model_output",
                content=[SimpleNamespace(type="text", text=None)],
            )
        ],
        [function_call("", "add", {})],
        [function_call("call_1", "add", [])],
    ],
)
def test_parser_rejects_malformed_provider_steps(steps):
    with pytest.raises(RuntimeError, match="Gemini"):
        GeminiClient._parse_agent_response(steps)


def test_ask_rejects_invalid_output_text():
    provider = FakeGemini(
        [
            SimpleNamespace(output_text=None),
        ]
    )
    client = GeminiClient(client=provider, model="gemini-test")

    with pytest.raises(RuntimeError, match="output text"):
        client.ask([{"role": "user", "content": "Hi"}])


def test_stream_rejects_invalid_text_delta():
    provider = FakeGemini(
        [
            [
                SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text=None),
                )
            ]
        ]
    )
    client = GeminiClient(client=provider, model="gemini-test")

    with pytest.raises(RuntimeError, match="delta"):
        list(client.stream([{"role": "user", "content": "Hi"}]))


def test_unsupported_message_role_is_rejected():
    client = GeminiClient(client=FakeGemini([]), model="gemini-test")

    with pytest.raises(RuntimeError, match="role"):
        client.ask([{"role": "tool", "content": "No"}])


def test_missing_configuration_fails_before_request(monkeypatch):
    monkeypatch.setattr(gemini_module, "GEMINI_MODEL", None)
    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        GeminiClient(client=FakeGemini([]))

    monkeypatch.setattr(gemini_module, "GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(gemini_module, "GEMINI_API_KEY", None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_real_client_constructor_receives_api_key(monkeypatch):
    received = {}

    def build_client(**kwargs):
        received.update(kwargs)
        return FakeGemini([])

    monkeypatch.setattr(gemini_module.genai, "Client", build_client)

    client = GeminiClient(
        api_key="test-value",
        model="gemini-test",
    )

    assert isinstance(client.client, FakeGemini)
    assert received == {
        "api_key": "test-value",
        "http_options": {
            "retry_options": {"attempts": 1},
        },
    }
