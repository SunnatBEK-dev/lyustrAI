from types import SimpleNamespace

import pytest

import ai_sdk.llm.anthropic as anthropic_module
from ai_sdk.llm.anthropic import AnthropicClient
from ai_sdk.tools import (
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)


class FakeStream:
    def __init__(self, chunks):
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeMessagesAPI:
    def __init__(self):
        self.create_call = None
        self.stream_call = None

    def create(self, **kwargs):
        self.create_call = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Hello "),
                SimpleNamespace(type="tool_use", name="ignored"),
                SimpleNamespace(type="text", text="world"),
            ]
        )

    def stream(self, **kwargs):
        self.stream_call = kwargs
        return FakeStream(["one", "two"])


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessagesAPI()


class QueuedMessagesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.responses.pop(0)


class QueuedAnthropicClient:
    def __init__(self, responses):
        self.messages = QueuedMessagesAPI(responses)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
    )


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(call_id, name, arguments):
    return SimpleNamespace(
        type="tool_use",
        id=call_id,
        name=name,
        input=arguments,
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


def test_ask_uses_injected_client_and_provider_configuration():
    fake = FakeAnthropicClient()
    client = AnthropicClient(
        client=fake,
        model="claude-test",
        max_tokens=77,
    )
    messages = [{"role": "user", "content": "Hi"}]

    response = client.ask(messages)

    assert response == "Hello world"
    assert fake.messages.create_call == {
        "model": "claude-test",
        "max_tokens": 77,
        "messages": messages,
    }


def test_stream_uses_injected_client_without_network_access():
    fake = FakeAnthropicClient()
    client = AnthropicClient(
        client=fake,
        model="claude-test",
        max_tokens=55,
    )
    messages = [{"role": "user", "content": "Hi"}]

    assert list(client.stream(messages)) == ["one", "two"]
    assert fake.messages.stream_call == {
        "model": "claude-test",
        "max_tokens": 55,
        "messages": messages,
    }


def test_missing_model_fails_before_request(monkeypatch):
    monkeypatch.setattr(anthropic_module, "ANTHROPIC_MODEL", None)

    with pytest.raises(RuntimeError, match="MODEL"):
        AnthropicClient(client=FakeAnthropicClient())


def test_missing_api_key_fails_before_real_client_is_created(monkeypatch):
    monkeypatch.setattr(anthropic_module, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(anthropic_module, "ANTHROPIC_MODEL", "claude-test")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_real_client_disables_hidden_provider_retries(monkeypatch):
    received = {}

    def build_client(**kwargs):
        received.update(kwargs)
        return FakeAnthropicClient()

    monkeypatch.setattr(anthropic_module, "Anthropic", build_client)

    client = AnthropicClient(
        api_key="test-value",
        model="claude-test",
        timeout=12.0,
    )

    assert isinstance(client.client, FakeAnthropicClient)
    assert received == {
        "api_key": "test-value",
        "timeout": 12.0,
        "max_retries": 0,
    }


def test_ask_with_tools_completes_claude_tool_loop():
    fake = QueuedAnthropicClient(
        [
            response(
                text_block("I will calculate it."),
                tool_block(
                    "tool_1",
                    "add",
                    {"left": 2, "right": 3},
                ),
                stop_reason="tool_use",
            ),
            response(text_block("The answer is 5.")),
        ]
    )
    executor = build_add_executor()
    client = AnthropicClient(client=fake, model="claude-test")
    messages = [{"role": "user", "content": "What is 2 + 3?"}]

    result = client.ask_with_tools(messages, executor)

    assert result == "The answer is 5."
    assert messages == [
        {"role": "user", "content": "What is 2 + 3?"},
    ]
    assert len(fake.messages.create_calls) == 2
    first_call, second_call = fake.messages.create_calls
    assert first_call["tools"] == (executor.registry.provider_schemas())
    assert second_call["messages"] == [
        {"role": "user", "content": "What is 2 + 3?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I will calculate it.",
                },
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "add",
                    "input": {"left": 2, "right": 3},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": "5",
                    "is_error": False,
                },
            ],
        },
    ]


def test_ask_with_tools_returns_execution_errors_to_claude():
    fake = QueuedAnthropicClient(
        [
            response(
                tool_block("tool_missing", "missing", {}),
                stop_reason="tool_use",
            ),
            response(text_block("That tool is unavailable.")),
        ]
    )
    client = AnthropicClient(client=fake, model="claude-test")

    result = client.ask_with_tools(
        [{"role": "user", "content": "Use missing."}],
        build_add_executor(),
    )

    assert result == "That tool is unavailable."
    result_block = fake.messages.create_calls[1]["messages"][-1]["content"][0]
    assert result_block == {
        "type": "tool_result",
        "tool_use_id": "tool_missing",
        "content": "Unknown tool: missing",
        "is_error": True,
    }


def test_ask_with_tools_enforces_round_limit_before_extra_execution():
    executions = []
    fake = QueuedAnthropicClient(
        [
            response(
                tool_block(
                    "tool_1",
                    "add",
                    {"left": 1, "right": 1},
                ),
                stop_reason="tool_use",
            ),
            response(
                tool_block(
                    "tool_2",
                    "add",
                    {"left": 2, "right": 2},
                ),
                stop_reason="tool_use",
            ),
        ]
    )
    executor = build_add_executor(
        lambda left, right: executions.append((left, right)) or left + right
    )
    client = AnthropicClient(client=fake, model="claude-test")

    with pytest.raises(RuntimeError, match="rounds exceeded"):
        client.ask_with_tools(
            [{"role": "user", "content": "Keep adding."}],
            executor,
            max_tool_rounds=1,
        )

    assert executions == [(1, 1)]


@pytest.mark.parametrize("max_tool_rounds", [0, -1, True, 1.5])
def test_ask_with_tools_rejects_invalid_round_limit(max_tool_rounds):
    client = AnthropicClient(
        client=FakeAnthropicClient(),
        model="claude-test",
    )

    with pytest.raises(ValueError, match="greater than zero"):
        client.ask_with_tools(
            [{"role": "user", "content": "Hi"}],
            build_add_executor(),
            max_tool_rounds=max_tool_rounds,
        )


def test_ask_with_tools_rejects_malformed_or_duplicate_calls():
    malformed = AnthropicClient(
        client=QueuedAnthropicClient(
            [
                response(
                    tool_block("tool_1", "add", "not-an-object"),
                    stop_reason="tool_use",
                ),
            ]
        ),
        model="claude-test",
    )

    with pytest.raises(RuntimeError, match="block is invalid"):
        malformed.ask_with_tools(
            [{"role": "user", "content": "Add."}],
            build_add_executor(),
        )

    duplicate = AnthropicClient(
        client=QueuedAnthropicClient(
            [
                response(
                    tool_block(
                        "tool_1",
                        "add",
                        {"left": 1, "right": 2},
                    ),
                    tool_block(
                        "tool_1",
                        "add",
                        {"left": 3, "right": 4},
                    ),
                    stop_reason="tool_use",
                ),
            ]
        ),
        model="claude-test",
    )

    with pytest.raises(RuntimeError, match="duplicate"):
        duplicate.ask_with_tools(
            [{"role": "user", "content": "Add."}],
            build_add_executor(),
        )


def test_ask_with_tools_uses_plain_ask_for_empty_registry():
    fake = FakeAnthropicClient()
    client = AnthropicClient(
        client=fake,
        model="claude-test",
    )

    result = client.ask_with_tools(
        [{"role": "user", "content": "Hi"}],
        ToolExecutor(ToolRegistry()),
    )

    assert result == "Hello world"
    assert "tools" not in fake.messages.create_call


def test_ask_with_tools_executes_multiple_calls_in_provider_order():
    fake = QueuedAnthropicClient(
        [
            response(
                tool_block(
                    "tool_1",
                    "add",
                    {"left": 1, "right": 2},
                ),
                tool_block(
                    "tool_2",
                    "add",
                    {"left": 3, "right": 4},
                ),
                stop_reason="tool_use",
            ),
            response(text_block("The answers are 3 and 7.")),
        ]
    )
    client = AnthropicClient(client=fake, model="claude-test")

    result = client.ask_with_tools(
        [{"role": "user", "content": "Add both pairs."}],
        build_add_executor(),
    )

    assert result == "The answers are 3 and 7."
    result_blocks = fake.messages.create_calls[1]["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in result_blocks] == [
        "tool_1",
        "tool_2",
    ]
    assert [block["content"] for block in result_blocks] == [
        "3",
        "7",
    ]


def test_ask_with_tools_rejects_duplicate_id_across_rounds():
    executions = []
    repeated_call = tool_block(
        "tool_repeated",
        "add",
        {"left": 1, "right": 2},
    )
    fake = QueuedAnthropicClient(
        [
            response(repeated_call, stop_reason="tool_use"),
            response(repeated_call, stop_reason="tool_use"),
        ]
    )
    client = AnthropicClient(client=fake, model="claude-test")
    executor = build_add_executor(
        lambda left, right: executions.append((left, right)) or left + right
    )

    with pytest.raises(RuntimeError, match="duplicate"):
        client.ask_with_tools(
            [{"role": "user", "content": "Add."}],
            executor,
        )

    assert executions == [(1, 2)]


def test_ask_with_tools_rejects_invalid_executor():
    client = AnthropicClient(
        client=FakeAnthropicClient(),
        model="claude-test",
    )

    with pytest.raises(TypeError, match="ToolExecutor"):
        client.ask_with_tools(
            [{"role": "user", "content": "Hi"}],
            object(),
        )


def test_ask_with_tools_rejects_inconsistent_tool_stop_reason():
    client = AnthropicClient(
        client=QueuedAnthropicClient(
            [
                response(
                    text_block("No tool block."),
                    stop_reason="tool_use",
                ),
            ]
        ),
        model="claude-test",
    )

    with pytest.raises(RuntimeError, match="without a tool call"):
        client.ask_with_tools(
            [{"role": "user", "content": "Use a tool."}],
            build_add_executor(),
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-a-list", "content must be a list"),
        ([SimpleNamespace(type="text", text=3)], "text block"),
    ],
)
def test_ask_with_tools_rejects_invalid_response_content(
    content,
    message,
):
    provider = QueuedAnthropicClient(
        [
            SimpleNamespace(
                content=content,
                stop_reason="end_turn",
            ),
        ]
    )
    client = AnthropicClient(client=provider, model="claude-test")

    with pytest.raises(RuntimeError, match=message):
        client.ask_with_tools(
            [{"role": "user", "content": "Hi"}],
            build_add_executor(),
        )
