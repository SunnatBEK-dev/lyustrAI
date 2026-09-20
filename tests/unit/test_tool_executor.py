import json

import pytest

from ai_sdk.observability import (
    InMemoryTraceCollector,
    Tracer,
    TraceStatus,
)
from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)


def make_schema():
    return ToolSchema(
        "add",
        "Add two integers.",
        [
            ToolParameter(
                "left",
                ToolParameterType.INTEGER,
                "Left number.",
            ),
            ToolParameter(
                "right",
                ToolParameterType.INTEGER,
                "Right number.",
            ),
        ],
    )


def test_registry_exposes_schemas_in_registration_order():
    registry = ToolRegistry()
    schema = make_schema()
    registry.register(schema, lambda left, right: left + right)

    assert registry.count() == 1
    assert registry.schemas() == [schema]
    assert registry.provider_schemas() == [schema.to_json_schema()]
    assert registry.get("missing") is None


def test_registry_rejects_duplicate_or_non_callable_handler():
    registry = ToolRegistry()
    schema = make_schema()
    registry.register(schema, lambda left, right: left + right)

    with pytest.raises(ValueError, match="already"):
        registry.register(schema, lambda: None)

    with pytest.raises(TypeError, match="callable"):
        ToolRegistry().register(schema, None)

    with pytest.raises(TypeError, match="ToolSchema"):
        ToolRegistry().register("invalid", lambda: None)


def test_executor_validates_executes_and_serializes_output():
    calls = []
    registry = ToolRegistry()

    def add(left, right):
        calls.append((left, right))
        return {"total": left + right}

    registry.register(make_schema(), add)
    result = ToolExecutor(registry).execute(
        ToolCall(
            id="call_one",
            name="add",
            arguments={"left": 2, "right": 3},
        )
    )

    assert result.is_error is False
    assert json.loads(result.content) == {"total": 5}
    assert calls == [(2, 3)]


def test_executor_does_not_call_handler_for_invalid_arguments():
    calls = []
    registry = ToolRegistry()
    registry.register(
        make_schema(),
        lambda **arguments: calls.append(arguments),
    )

    result = ToolExecutor(registry).execute(
        ToolCall(
            "call_invalid",
            "add",
            {"left": "two", "right": 3},
        )
    )

    assert result.is_error is True
    assert "integer" in result.content
    assert calls == []


def test_executor_returns_error_for_unknown_tool():
    result = ToolExecutor(ToolRegistry()).execute(
        ToolCall(
            "call_unknown",
            "missing",
            {},
        )
    )

    assert result.is_error is True
    assert result.content == "Unknown tool: missing"


def test_executor_contains_handler_and_serialization_errors():
    failing_registry = ToolRegistry()

    def fail(left, right):
        raise RuntimeError("private detail")

    failing_registry.register(make_schema(), fail)
    failed = ToolExecutor(failing_registry).execute(
        ToolCall(
            "call_failed",
            "add",
            {"left": 1, "right": 2},
        )
    )
    invalid_output_registry = ToolRegistry()
    invalid_output_registry.register(
        make_schema(),
        lambda left, right: {object()},
    )
    invalid_output = ToolExecutor(invalid_output_registry).execute(
        ToolCall(
            "call_output",
            "add",
            {"left": 1, "right": 2},
        )
    )

    assert failed.is_error is True
    assert failed.content == "Tool execution failed: RuntimeError"
    assert "private detail" not in failed.content
    assert invalid_output.is_error is True
    assert invalid_output.content == ("Tool execution failed: TypeError")


@pytest.mark.parametrize(
    ("call_id", "name", "arguments", "message"),
    [
        ("", "add", {}, "ID"),
        ("call", "", {}, "name"),
        ("call", "add", [], "object"),
    ],
)
def test_tool_call_rejects_invalid_data(
    call_id,
    name,
    arguments,
    message,
):
    with pytest.raises(ValueError, match=message):
        ToolCall(call_id, name, arguments)


def test_executor_preserves_plain_string_output():
    registry = ToolRegistry()
    registry.register(
        make_schema(),
        lambda left, right: "done",
    )

    result = ToolExecutor(registry).execute(
        ToolCall(
            "call_string",
            "add",
            {"left": 1, "right": 2},
        )
    )

    assert result.content == "done"


def test_executor_traces_safe_success_and_contained_error_metadata():
    collector = InMemoryTraceCollector()
    tracer = Tracer(collector)
    registry = ToolRegistry()
    registry.register(make_schema(), lambda left, right: left + right)
    executor = ToolExecutor(registry, tracer=tracer)

    success = executor.execute(ToolCall("call_ok", "add", {"left": 1, "right": 2}))
    failure = executor.execute(
        ToolCall("call_bad", "add", {"left": "secret", "right": 2})
    )

    first, second = collector.records()
    assert not success.is_error
    assert failure.is_error
    assert first.name == second.name == "tool.execute"
    assert first.attributes == {
        "tool.name": "add",
        "tool.is_error": False,
    }
    assert first.status is TraceStatus.OK
    assert second.status is TraceStatus.ERROR
    assert second.error_type == "ToolExecutionError"
    assert "secret" not in str(second.to_dict())


def test_executor_rejects_invalid_tracers():
    with pytest.raises(TypeError, match="tracer"):
        ToolExecutor(ToolRegistry(), tracer=object())

    executor = ToolExecutor(ToolRegistry())
    with pytest.raises(TypeError, match="tracer"):
        executor.execute(
            ToolCall("call", "missing", {}),
            tracer=object(),
        )
