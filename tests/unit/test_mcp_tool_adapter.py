import json

import pytest

from ai_sdk.mcp import (
    MCP_PROTOCOL_VERSION,
    BaseMCPTransport,
    MCPClient,
    MCPContentBlock,
    MCPDiscoveryResult,
    MCPImplementation,
    MCPInputRequest,
    MCPInputRequiredResult,
    MCPResourcePage,
    MCPResourceReadResult,
    MCPServerCapabilities,
    MCPTool,
    MCPToolAdapter,
    MCPToolAdapterError,
    MCPToolPage,
    MCPToolResult,
)
from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolHandlerError,
    ToolRegistry,
    ToolSchema,
)


class AdapterTransport(BaseMCPTransport):
    def __init__(self):
        self.requests = []
        self.tool_result = MCPToolResult([MCPContentBlock.text("remote success")])

    def open(self, *, timeout_seconds):
        pass

    def discover(self, context, *, timeout_seconds):
        return MCPDiscoveryResult(
            [MCP_PROTOCOL_VERSION],
            MCPServerCapabilities(tools={}),
        )

    def list_tools(self, context, *, cursor, timeout_seconds):
        return MCPToolPage([])

    def list_resources(
        self,
        context,
        *,
        cursor,
        timeout_seconds,
    ):
        return MCPResourcePage([])

    def call_tool(
        self,
        context,
        request,
        *,
        timeout_seconds,
    ):
        self.requests.append(request)
        return self.tool_result

    def read_resource(
        self,
        context,
        request,
        *,
        timeout_seconds,
    ):
        return MCPResourceReadResult([])

    def close(self):
        pass


def make_client(**kwargs):
    transport = AdapterTransport()
    client = MCPClient(
        transport,
        client_info=MCPImplementation("adapter-test", "1.0"),
        **kwargs,
    )
    return client, transport


def make_tool(name="search_docs", **schema_changes):
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "limit": {
                "type": "integer",
                "title": "Maximum results",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    schema.update(schema_changes)
    return MCPTool(
        name,
        schema,
        description="Search documents.",
    )


def test_adapter_registers_only_explicitly_approved_compatible_tools():
    client, transport = make_client()
    registry = ToolRegistry()
    tools = [
        make_tool("first_tool"),
        MCPTool("not-approved.tool", {"type": "object"}),
        make_tool("second_tool"),
    ]

    registered = MCPToolAdapter(client).register_approved(
        registry,
        tools,
        approved_names=["second_tool", "first_tool"],
    )

    assert registered == ("first_tool", "second_tool")
    assert registry.count() == 2
    assert registry.get("not-approved.tool") is None
    assert registry.provider_schemas()[0] == {
        "name": "first_tool",
        "description": "Search documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }

    client.open()
    result = ToolExecutor(registry).execute(
        ToolCall(
            "call-1",
            "first_tool",
            {"query": "Python", "limit": 2},
        )
    )

    assert result.content == "remote success"
    assert not result.is_error
    assert transport.requests[0].name == "first_tool"
    assert transport.requests[0].arguments == {
        "query": "Python",
        "limit": 2,
    }


def test_adapter_preserves_remote_tool_error_content():
    client, transport = make_client()
    transport.tool_result = MCPToolResult(
        [MCPContentBlock.text("remote validation failed")],
        is_error=True,
    )
    registry = ToolRegistry()
    MCPToolAdapter(client).register_approved(
        registry,
        [make_tool()],
        approved_names=["search_docs"],
    )
    client.open()

    result = ToolExecutor(registry).execute(
        ToolCall(
            "call-1",
            "search_docs",
            {"query": ""},
        )
    )

    assert result.content == "remote validation failed"
    assert result.is_error


def test_adapter_contains_tools_that_require_manual_host_input():
    client, transport = make_client(
        client_capabilities={"elicitation": {}},
    )
    transport.tool_result = MCPInputRequiredResult(
        {
            "confirm": MCPInputRequest(
                "elicitation/create",
                {"message": "Continue?", "requestedSchema": {}},
            )
        }
    )
    registry = ToolRegistry()
    MCPToolAdapter(client).register_approved(
        registry,
        [make_tool()],
        approved_names=["search_docs"],
    )
    client.open()

    result = ToolExecutor(registry).execute(
        ToolCall("call-1", "search_docs", {"query": "Python"})
    )

    assert result.is_error
    assert result.content == ("Remote MCP tool requires explicit host input.")


@pytest.mark.parametrize(
    ("tool_result", "expected"),
    [
        (
            MCPToolResult(structured_content={"count": 2}),
            '{"count": 2}',
        ),
        (
            MCPToolResult(
                [
                    MCPContentBlock(
                        "image",
                        {
                            "data": "YWJj",
                            "mimeType": "image/png",
                        },
                    )
                ]
            ),
            json.dumps(
                {
                    "type": "image",
                    "data": "YWJj",
                    "mimeType": "image/png",
                },
                sort_keys=True,
            ),
        ),
    ],
)
def test_adapter_serializes_structured_and_non_text_results(
    tool_result,
    expected,
):
    client, transport = make_client()
    transport.tool_result = tool_result
    registry = ToolRegistry()
    MCPToolAdapter(client).register_approved(
        registry,
        [MCPTool("remote", {"type": "object"})],
        approved_names=["remote"],
    )
    client.open()

    result = ToolExecutor(registry).execute(ToolCall("call-1", "remote", {}))

    assert result.content == expected
    assert not result.is_error


def test_adapter_validation_is_atomic_before_registry_changes():
    client, _ = make_client()
    registry = ToolRegistry()

    with pytest.raises(MCPToolAdapterError, match="primitive"):
        MCPToolAdapter(client).register_approved(
            registry,
            [
                make_tool("compatible"),
                make_tool(
                    "nested",
                    properties={"items": {"type": "array"}},
                    required=["items"],
                ),
            ],
            approved_names=["compatible", "nested"],
        )

    assert registry.count() == 0


def test_adapter_rejects_existing_registry_collision_atomically():
    client, _ = make_client()
    registry = ToolRegistry()
    registry.register(
        ToolSchema("existing", "Existing tool."),
        lambda: "local",
    )

    with pytest.raises(MCPToolAdapterError, match="conflict"):
        MCPToolAdapter(client).register_approved(
            registry,
            [
                MCPTool("new_tool", {"type": "object"}),
                MCPTool("existing", {"type": "object"}),
            ],
            approved_names=["new_tool", "existing"],
        )

    assert registry.count() == 1
    assert registry.get("new_tool") is None


@pytest.mark.parametrize(
    ("tools", "approved_names", "message"),
    [
        ("invalid", [], "sequence"),
        (["invalid"], [], "invalid definition"),
        (
            [
                MCPTool("same", {"type": "object"}),
                MCPTool("same", {"type": "object"}),
            ],
            ["same"],
            "unique",
        ),
        ([], "invalid", "sequence"),
        ([], [""], "cannot be empty"),
        ([], ["same", "same"], "unique"),
        ([], ["missing"], "not discovered"),
    ],
)
def test_adapter_rejects_invalid_selection(
    tools,
    approved_names,
    message,
):
    client, _ = make_client()

    with pytest.raises(MCPToolAdapterError, match=message):
        MCPToolAdapter(client).register_approved(
            ToolRegistry(),
            tools,
            approved_names=approved_names,
        )


@pytest.mark.parametrize(
    "tool",
    [
        MCPTool("invalid.name", {"type": "object"}),
        make_tool(unknownRoot=True),
        make_tool(properties=[]),
        make_tool(properties={1: {"type": "string"}}, required=[]),
        make_tool(required="query"),
        make_tool(required=[1]),
        make_tool(required=["query", "query"]),
        make_tool(required=["missing"]),
        make_tool(additionalProperties={}),
        make_tool(properties={"query": "invalid"}),
        make_tool(properties={"query": {"type": "string", "enum": ["one"]}}),
        make_tool(properties={"query": {"type": "array"}}),
        make_tool(
            properties={"invalid-name": {"type": "string"}},
            required=[],
        ),
        make_tool(properties={"query": {"type": "string", "description": 1}}),
    ],
)
def test_adapter_rejects_schemas_local_layer_cannot_validate(tool):
    client, _ = make_client()

    with pytest.raises(MCPToolAdapterError, match="incompatible"):
        MCPToolAdapter(client).register_approved(
            ToolRegistry(),
            [tool],
            approved_names=[tool.name],
        )


def test_adapter_supplies_safe_fallback_descriptions():
    client, _ = make_client()
    registry = ToolRegistry()
    tool = MCPTool(
        "lookup",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )

    MCPToolAdapter(client).register_approved(
        registry,
        [tool],
        approved_names=["lookup"],
    )

    schema = registry.get("lookup").schema
    assert schema.description == "Remote MCP tool lookup."
    assert schema.parameters[0].description == "MCP parameter query."


def test_adapter_rejects_invalid_runtime_objects():
    with pytest.raises(MCPToolAdapterError, match="client"):
        MCPToolAdapter(object())

    client, _ = make_client()
    with pytest.raises(MCPToolAdapterError, match="registry"):
        MCPToolAdapter(client).register_approved(
            object(),
            [],
            approved_names=[],
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        ToolHandlerError("")
