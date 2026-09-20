import json
from types import SimpleNamespace

import pytest

import ai_sdk.llm.openai as openai_module
from ai_sdk.llm.openai import OpenAIClient
from ai_sdk.tools import (
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)


class QueuedResponsesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.responses.pop(0)


class FakeOpenAI:
    def __init__(self, responses):
        self.responses = QueuedResponsesAPI(responses)


def response(*items, output_text=""):
    return SimpleNamespace(
        output=list(items),
        output_text=output_text,
    )


def message(*blocks):
    return SimpleNamespace(
        type="message",
        content=list(blocks),
    )


def text_block(text):
    return SimpleNamespace(
        type="output_text",
        text=text,
    )


def refusal_block(text):
    return SimpleNamespace(
        type="refusal",
        refusal=text,
    )


def function_call(call_id, name, arguments):
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
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


def test_ask_uses_responses_api_with_provider_configuration():
    provider = FakeOpenAI(
        [
            response(output_text="Hello world"),
        ]
    )
    client = OpenAIClient(
        client=provider,
        model="gpt-test",
        max_output_tokens=77,
    )
    messages = [{"role": "user", "content": "Hi"}]

    result = client.ask(messages)

    assert result == "Hello world"
    assert provider.responses.create_calls == [
        {
            "model": "gpt-test",
            "max_output_tokens": 77,
            "input": [{"role": "user", "content": "Hi"}],
            "store": False,
        }
    ]
    assert messages == [{"role": "user", "content": "Hi"}]


def test_stream_yields_only_text_and_refusal_deltas():
    stream = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.delta",
            delta="one",
        ),
        SimpleNamespace(
            type="response.refusal.delta",
            delta="two",
        ),
        SimpleNamespace(type="response.completed"),
    ]
    provider = FakeOpenAI([stream])
    client = OpenAIClient(client=provider, model="gpt-test")

    chunks = list(client.stream([{"role": "user", "content": "Hi"}]))

    assert chunks == ["one", "two"]
    assert provider.responses.create_calls[0]["stream"] is True


def test_ask_with_tools_completes_openai_tool_loop():
    provider = FakeOpenAI(
        [
            response(
                message(text_block("I will calculate it.")),
                function_call(
                    "call_1",
                    "add",
                    '{"left": 2, "right": 3}',
                ),
            ),
            response(
                message(text_block("The answer is 5.")),
            ),
        ]
    )
    executor = build_add_executor()
    client = OpenAIClient(client=provider, model="gpt-test")
    messages = [{"role": "user", "content": "What is 2 + 3?"}]

    result = client.ask_with_tools(messages, executor)

    assert result == "The answer is 5."
    assert messages == [{"role": "user", "content": "What is 2 + 3?"}]
    first_call, second_call = provider.responses.create_calls
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
            "strict": True,
        }
    ]
    assert second_call["input"] == [
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "content": "I will calculate it."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "add",
            "arguments": '{"left": 2, "right": 3}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "5",
        },
    ]


def test_tool_errors_are_marked_in_openai_function_output():
    provider = FakeOpenAI(
        [
            response(function_call("missing_1", "missing", "{}")),
            response(message(text_block("Tool unavailable."))),
        ]
    )
    client = OpenAIClient(client=provider, model="gpt-test")

    result = client.ask_with_tools(
        [{"role": "user", "content": "Use missing."}],
        build_add_executor(),
    )

    assert result == "Tool unavailable."
    tool_output = provider.responses.create_calls[1]["input"][-1]
    assert tool_output["type"] == "function_call_output"
    assert json.loads(tool_output["output"]) == {
        "is_error": True,
        "content": "Unknown tool: missing",
    }


def test_optional_tool_parameters_disable_openai_strict_mode():
    schema = ToolSchema(
        name="search",
        description="Search documents.",
        parameters=[
            ToolParameter(
                "query",
                ToolParameterType.STRING,
                "Search query.",
                required=False,
            )
        ],
    )
    provider = FakeOpenAI(
        [
            response(message(text_block("No search needed."))),
        ]
    )
    client = OpenAIClient(client=provider, model="gpt-test")

    result = client.complete_tool_turn(
        [{"role": "user", "content": "Hello"}],
        [schema],
        (),
    )

    assert result.text == "No search needed."
    assert provider.responses.create_calls[0]["tools"][0]["strict"] is False


def test_agent_response_preserves_refusal_and_ignores_unknown_items():
    parsed = OpenAIClient._parse_agent_response(
        [
            SimpleNamespace(type="reasoning"),
            message(
                SimpleNamespace(type="annotation"),
                refusal_block("Cannot help."),
            ),
        ]
    )

    assert parsed.text == "Cannot help."
    assert parsed.tool_calls == ()


@pytest.mark.parametrize(
    "output",
    [
        None,
        [SimpleNamespace(type="message", content="invalid")],
        [message(SimpleNamespace(type="output_text", text=None))],
        [function_call("call_1", "add", "not-json")],
        [function_call("call_1", "add", "[]")],
    ],
)
def test_agent_response_rejects_malformed_provider_output(output):
    with pytest.raises(RuntimeError, match="OpenAI"):
        OpenAIClient._parse_agent_response(output)


def test_ask_rejects_invalid_output_text():
    provider = FakeOpenAI(
        [
            SimpleNamespace(output_text=None),
        ]
    )
    client = OpenAIClient(client=provider, model="gpt-test")

    with pytest.raises(RuntimeError, match="output text"):
        client.ask([{"role": "user", "content": "Hi"}])


def test_stream_rejects_invalid_text_delta():
    provider = FakeOpenAI(
        [
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta=None,
                )
            ]
        ]
    )
    client = OpenAIClient(client=provider, model="gpt-test")

    with pytest.raises(RuntimeError, match="delta"):
        list(client.stream([{"role": "user", "content": "Hi"}]))


def test_missing_configuration_fails_before_request(monkeypatch):
    monkeypatch.setattr(openai_module, "OPENAI_MODEL", None)
    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        OpenAIClient(client=FakeOpenAI([]))

    monkeypatch.setattr(openai_module, "OPENAI_MODEL", "gpt-test")
    monkeypatch.setattr(openai_module, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_real_client_constructor_receives_explicit_safe_configuration(
    monkeypatch,
):
    received = {}

    def build_client(**kwargs):
        received.update(kwargs)
        return FakeOpenAI([])

    monkeypatch.setattr(openai_module, "OpenAI", build_client)

    client = OpenAIClient(
        api_key="test-value",
        model="gpt-test",
        timeout=12.0,
    )

    assert isinstance(client.client, FakeOpenAI)
    assert received == {
        "api_key": "test-value",
        "timeout": 12.0,
        "max_retries": 0,
    }
