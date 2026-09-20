import pytest

from ai_sdk.mcp import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    MCP_PROTOCOL_VERSION,
    PROTOCOL_VERSION_META_KEY,
    BaseMCPTransport,
    MCPCapabilityError,
    MCPClient,
    MCPConnectionState,
    MCPContentBlock,
    MCPContinuation,
    MCPContinuationError,
    MCPDiscoveryResult,
    MCPImplementation,
    MCPInputRequest,
    MCPInputRequiredResult,
    MCPInputRoundsExceededError,
    MCPLifecycleError,
    MCPProtocolError,
    MCPRequestContext,
    MCPResource,
    MCPResourceContent,
    MCPResourcePage,
    MCPResourceReadResult,
    MCPServerCapabilities,
    MCPTimeoutError,
    MCPTool,
    MCPToolPage,
    MCPToolRequest,
    MCPToolResult,
    MCPTransportError,
    MCPValidationError,
)
from ai_sdk.observability import (
    InMemoryTraceCollector,
    Tracer,
    TraceStatus,
)


class RecordingTransport(BaseMCPTransport):
    def __init__(self):
        self.calls = []
        self.open_error = None
        self.close_error = None
        self.discovery_result = MCPDiscoveryResult(
            [MCP_PROTOCOL_VERSION],
            MCPServerCapabilities(tools={}, resources={}),
            server_info=MCPImplementation("test-server", "1.0"),
        )
        self.tools_result = MCPToolPage(
            [
                MCPTool("zeta", {"type": "object"}),
                MCPTool("alpha", {"type": "object"}),
            ],
            next_cursor="tools-page-2",
            ttl_ms=1000,
            cache_scope="private",
        )
        self.resources_result = MCPResourcePage(
            [
                MCPResource("file:///z.txt", "z.txt"),
                MCPResource("file:///a.txt", "a.txt"),
            ]
        )
        self.tool_result = MCPToolResult([MCPContentBlock.text("remote result")])
        self.resource_read_result = MCPResourceReadResult(
            [
                MCPResourceContent(
                    "file:///z.txt",
                    mime_type="text/plain",
                    text="Z content",
                )
            ],
            ttl_ms=500,
            cache_scope="private",
        )

    def open(self, *, timeout_seconds):
        self.calls.append(("open", timeout_seconds))
        if self.open_error is not None:
            raise self.open_error

    def discover(self, context, *, timeout_seconds):
        self.calls.append(("discover", context, timeout_seconds))
        if isinstance(self.discovery_result, BaseException):
            raise self.discovery_result
        return self.discovery_result

    def list_tools(self, context, *, cursor, timeout_seconds):
        self.calls.append(("list_tools", context, cursor, timeout_seconds))
        if isinstance(self.tools_result, BaseException):
            raise self.tools_result
        return self.tools_result

    def list_resources(
        self,
        context,
        *,
        cursor,
        timeout_seconds,
    ):
        self.calls.append(("list_resources", context, cursor, timeout_seconds))
        if isinstance(self.resources_result, BaseException):
            raise self.resources_result
        return self.resources_result

    def call_tool(
        self,
        context,
        request,
        *,
        timeout_seconds,
    ):
        self.calls.append(("call_tool", context, request, timeout_seconds))
        if isinstance(self.tool_result, BaseException):
            raise self.tool_result
        return self.tool_result

    def read_resource(
        self,
        context,
        request,
        *,
        timeout_seconds,
    ):
        self.calls.append(("read_resource", context, request, timeout_seconds))
        if isinstance(self.resource_read_result, BaseException):
            raise self.resource_read_result
        return self.resource_read_result

    def close(self):
        self.calls.append(("close",))
        if self.close_error is not None:
            raise self.close_error


def make_client(transport=None, **kwargs):
    timeout_seconds = kwargs.pop("timeout_seconds", 2.5)
    return MCPClient(
        transport or RecordingTransport(),
        client_info=MCPImplementation("test-client", "2.0"),
        timeout_seconds=timeout_seconds,
        **kwargs,
    )


def input_required(
    *,
    request_state="opaque-state",
    method="elicitation/create",
):
    return MCPInputRequiredResult(
        {
            "confirm": MCPInputRequest(
                method,
                {
                    "mode": "form",
                    "message": "Continue?",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                    },
                },
            )
        },
        request_state=request_state,
    )


def test_request_context_exports_required_stateless_metadata():
    capabilities = {"sampling": {"enabled": False}}
    context = MCPRequestContext(
        MCP_PROTOCOL_VERSION,
        MCPImplementation("client", "1.2.3"),
        capabilities,
    )

    capabilities["sampling"] = {"enabled": True}
    meta = context.to_meta()
    meta[CLIENT_CAPABILITIES_META_KEY]["changed"] = True

    assert context.to_meta() == {
        PROTOCOL_VERSION_META_KEY: MCP_PROTOCOL_VERSION,
        CLIENT_INFO_META_KEY: {
            "name": "client",
            "version": "1.2.3",
        },
        CLIENT_CAPABILITIES_META_KEY: {"sampling": {"enabled": False}},
    }


def test_open_only_opens_transport_without_implicit_discovery():
    transport = RecordingTransport()
    client = make_client(transport)

    client.open()

    assert client.state is MCPConnectionState.OPEN
    assert client.discovery is None
    assert transport.calls == [("open", 2.5)]


def test_explicit_discovery_validates_version_and_keeps_server_info():
    transport = RecordingTransport()
    client = make_client(transport)
    client.open()

    result = client.discover()

    _, context, timeout = transport.calls[-1]
    assert result is transport.discovery_result
    assert client.discovery is result
    assert result.server_info == MCPImplementation(
        "test-server",
        "1.0",
    )
    assert context is client.request_context
    assert context.to_meta()[PROTOCOL_VERSION_META_KEY] == (MCP_PROTOCOL_VERSION)
    assert timeout == 2.5


def test_tools_can_be_listed_without_discovery_and_order_is_preserved():
    transport = RecordingTransport()
    client = make_client(transport)
    client.open()

    page = client.list_tools(cursor="tools-page-1")

    assert [tool.name for tool in page.tools] == ["zeta", "alpha"]
    assert page.next_cursor == "tools-page-2"
    assert page.ttl_ms == 1000
    assert transport.calls[-1] == (
        "list_tools",
        client.request_context,
        "tools-page-1",
        2.5,
    )


def test_resources_are_listed_one_explicit_page_at_a_time():
    transport = RecordingTransport()
    client = make_client(transport)
    client.open()

    page = client.list_resources(cursor="resources-page-2")

    assert [resource.name for resource in page.resources] == [
        "z.txt",
        "a.txt",
    ]
    assert transport.calls[-1] == (
        "list_resources",
        client.request_context,
        "resources-page-2",
        2.5,
    )


def test_tool_call_passes_copied_arguments_and_returns_remote_error():
    transport = RecordingTransport()
    transport.tool_result = MCPToolResult(
        [MCPContentBlock.text("invalid city")],
        is_error=True,
    )
    client = make_client(transport)
    client.open()
    arguments = {"city": "Samarqand"}

    result = client.call_tool("get_weather", arguments)
    arguments["city"] = "changed"

    _, context, request, timeout = transport.calls[-1]
    assert result.is_error
    assert client.state is MCPConnectionState.OPEN
    assert context is client.request_context
    assert request.name == "get_weather"
    assert request.arguments == {"city": "Samarqand"}
    assert timeout == 2.5


def test_resource_read_returns_multiple_content_items_and_cache_hints():
    transport = RecordingTransport()
    transport.resource_read_result = MCPResourceReadResult(
        [
            MCPResourceContent(
                "file:///guide/one.txt",
                text="one",
                annotations={
                    "audience": ["user"],
                    "priority": 0.8,
                },
            ),
            MCPResourceContent(
                "file:///guide/two.bin",
                mime_type="application/octet-stream",
                blob="YWJj",
            ),
        ],
        ttl_ms=900,
        cache_scope="private",
    )
    client = make_client(transport)
    client.open()

    result = client.read_resource("file:///guide")

    _, context, request, timeout = transport.calls[-1]
    assert [content.uri for content in result.contents] == [
        "file:///guide/one.txt",
        "file:///guide/two.bin",
    ]
    assert result.ttl_ms == 900
    assert result.contents[0].annotations == {
        "audience": ["user"],
        "priority": 0.8,
    }
    assert context is client.request_context
    assert request.uri == "file:///guide"
    assert timeout == 2.5


@pytest.mark.parametrize(
    ("method_name", "capabilities", "message"),
    [
        (
            "list_tools",
            MCPServerCapabilities(resources={}),
            "tools",
        ),
        (
            "list_resources",
            MCPServerCapabilities(tools={}),
            "resources",
        ),
        (
            "call_tool",
            MCPServerCapabilities(resources={}),
            "tools",
        ),
        (
            "read_resource",
            MCPServerCapabilities(tools={}),
            "resources",
        ),
    ],
)
def test_discovered_capabilities_block_unsupported_operations(
    method_name,
    capabilities,
    message,
):
    transport = RecordingTransport()
    transport.discovery_result = MCPDiscoveryResult(
        [MCP_PROTOCOL_VERSION],
        capabilities,
    )
    client = make_client(transport)
    client.open()
    client.discover()

    with pytest.raises(MCPCapabilityError, match=message):
        if method_name == "call_tool":
            client.call_tool("tool")
        elif method_name == "read_resource":
            client.read_resource("file:///resource")
        else:
            getattr(client, method_name)()

    assert client.state is MCPConnectionState.OPEN
    assert len(transport.calls) == 2


def test_unsupported_discovered_version_fails_and_closes_transport():
    transport = RecordingTransport()
    transport.discovery_result = MCPDiscoveryResult(
        ["2025-11-25"],
        MCPServerCapabilities(),
    )
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPProtocolError, match="does not support"):
        client.discover()

    assert client.state is MCPConnectionState.FAILED
    assert transport.calls[-1] == ("close",)


@pytest.mark.parametrize(
    ("attribute", "method_name", "operation"),
    [
        ("discovery_result", "discover", "discovery"),
        ("tools_result", "list_tools", "tools/list"),
        (
            "resources_result",
            "list_resources",
            "resources/list",
        ),
        ("tool_result", "call_tool", "tools/call"),
        (
            "resource_read_result",
            "read_resource",
            "resources/read",
        ),
    ],
)
def test_request_timeout_fails_and_closes_transport(
    attribute,
    method_name,
    operation,
):
    transport = RecordingTransport()
    setattr(transport, attribute, TimeoutError("private timeout"))
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPTimeoutError, match=operation):
        if method_name == "call_tool":
            client.call_tool("tool")
        elif method_name == "read_resource":
            client.read_resource("file:///resource")
        else:
            getattr(client, method_name)()

    assert client.state is MCPConnectionState.FAILED
    assert transport.calls[-1] == ("close",)


def test_transport_failure_does_not_expose_exception_message():
    transport = RecordingTransport()
    transport.tools_result = RuntimeError("secret-token")
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPTransportError) as caught:
        client.list_tools()

    assert "RuntimeError" in str(caught.value)
    assert "secret-token" not in str(caught.value)
    assert client.state is MCPConnectionState.FAILED


def test_mcp_request_tracing_records_operation_without_remote_data():
    collector = InMemoryTraceCollector()
    tracer = Tracer(collector)
    transport = RecordingTransport()
    client = make_client(transport, tracer=tracer)
    client.open()

    page = client.list_tools()
    transport.tools_result = RuntimeError("private bearer token")
    with pytest.raises(MCPTransportError):
        client.list_tools()

    success, failure = collector.records()
    assert page.tools
    assert success.name == failure.name == "mcp.request"
    assert success.attributes == {
        "mcp.operation": "tools/list",
        "mcp.input_required": False,
    }
    assert success.status is TraceStatus.OK
    assert failure.status is TraceStatus.ERROR
    assert failure.error_type == "MCPTransportError"
    assert "private bearer token" not in str(failure.to_dict())


@pytest.mark.parametrize(
    ("attribute", "method_name", "message"),
    [
        ("discovery_result", "discover", "discovery"),
        ("tools_result", "list_tools", "tools/list"),
        (
            "resources_result",
            "list_resources",
            "resources/list",
        ),
        ("tool_result", "call_tool", "tools/call"),
        (
            "resource_read_result",
            "read_resource",
            "resources/read",
        ),
    ],
)
def test_invalid_transport_results_are_protocol_errors(
    attribute,
    method_name,
    message,
):
    transport = RecordingTransport()
    setattr(transport, attribute, object())
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPProtocolError, match=message):
        if method_name == "call_tool":
            client.call_tool("tool")
        elif method_name == "read_resource":
            client.read_resource("file:///resource")
        else:
            getattr(client, method_name)()

    assert client.state is MCPConnectionState.FAILED


def test_duplicate_tool_names_are_rejected_as_protocol_failure():
    transport = RecordingTransport()
    tool = MCPTool("same", {"type": "object"})
    transport.tools_result = MCPToolPage([tool, tool])
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPProtocolError, match="duplicate tool"):
        client.list_tools()

    assert client.state is MCPConnectionState.FAILED


def test_duplicate_resource_uris_are_rejected_as_protocol_failure():
    transport = RecordingTransport()
    first = MCPResource("file:///same.txt", "first")
    second = MCPResource("file:///same.txt", "second")
    transport.resources_result = MCPResourcePage([first, second])
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPProtocolError, match="duplicate resource"):
        client.list_resources()

    assert client.state is MCPConnectionState.FAILED


def test_operations_require_open_state_and_client_cannot_reopen():
    client = make_client()

    with pytest.raises(MCPLifecycleError, match="open transport"):
        client.list_tools()

    client.open()
    client.close()

    with pytest.raises(MCPLifecycleError, match="only open"):
        client.open()
    with pytest.raises(MCPLifecycleError, match="open transport"):
        client.discover()


def test_close_is_idempotent_and_context_manager_closes():
    transport = RecordingTransport()

    with make_client(transport) as client:
        assert client.state is MCPConnectionState.OPEN

    client.close()

    assert client.state is MCPConnectionState.CLOSED
    assert transport.calls == [("open", 2.5), ("close",)]


def test_close_before_open_does_not_touch_transport():
    transport = RecordingTransport()
    client = make_client(transport)

    client.close()

    assert client.state is MCPConnectionState.CLOSED
    assert transport.calls == []


@pytest.mark.parametrize(
    ("error", "error_type", "message"),
    [
        (
            TimeoutError("private timeout"),
            MCPTimeoutError,
            "timed out",
        ),
        (
            RuntimeError("private secret"),
            MCPTransportError,
            "RuntimeError",
        ),
    ],
)
def test_open_failure_is_contained_and_cleanup_is_attempted(
    error,
    error_type,
    message,
):
    transport = RecordingTransport()
    transport.open_error = error
    client = make_client(transport)

    with pytest.raises(error_type, match=message) as caught:
        client.open()

    assert "private" not in str(caught.value)
    assert client.state is MCPConnectionState.FAILED
    assert transport.calls[-1] == ("close",)


def test_close_failure_is_contained_and_state_still_closes():
    transport = RecordingTransport()
    transport.close_error = RuntimeError("private secret")
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPTransportError, match="RuntimeError") as caught:
        client.close()

    assert "private secret" not in str(caught.value)
    assert client.state is MCPConnectionState.CLOSED


def test_abort_ignores_cleanup_error_and_preserves_primary_failure():
    transport = RecordingTransport()
    transport.tools_result = TimeoutError("primary")
    transport.close_error = RuntimeError("cleanup")
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPTimeoutError, match="tools/list"):
        client.list_tools()

    assert client.state is MCPConnectionState.FAILED


def test_client_requires_transport_contract():
    with pytest.raises(MCPValidationError, match="transport"):
        make_client(object())


@pytest.mark.parametrize("timeout", [0, -1, True, "30"])
def test_client_rejects_invalid_timeout(timeout):
    with pytest.raises(MCPValidationError, match="timeout"):
        make_client(timeout_seconds=timeout)


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_client_rejects_invalid_input_round_limit(rounds):
    with pytest.raises(MCPValidationError, match="input rounds"):
        make_client(max_input_rounds=rounds)


@pytest.mark.parametrize("cursor", ["", " ", 1])
def test_client_rejects_invalid_cursor_without_transport_call(cursor):
    transport = RecordingTransport()
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPValidationError, match="cursor"):
        client.list_tools(cursor=cursor)

    assert client.state is MCPConnectionState.OPEN
    assert transport.calls == [("open", 2.5)]


@pytest.mark.parametrize(
    "name",
    ["", "contains space", "slash/name", "a" * 129],
)
def test_tool_rejects_invalid_mcp_name(name):
    with pytest.raises(MCPValidationError, match="tool name"):
        MCPTool(name, {"type": "object"})


def test_tool_copies_schema_and_requires_object_root():
    schema = {"type": "object", "properties": {}}
    tool = MCPTool("valid.name-v2", schema)
    schema["type"] = "string"

    assert tool.input_schema["type"] == "object"

    with pytest.raises(MCPValidationError, match="root type"):
        MCPTool("invalid_schema", {"type": "string"})


@pytest.mark.parametrize(
    ("uri", "name", "size", "message"),
    [
        ("relative/path", "name", None, "scheme"),
        ("https://[invalid", "name", None, "invalid"),
        ("file:///valid", " ", None, "name"),
        ("file:///valid", "name", -1, "size"),
        ("file:///valid", "name", True, "size"),
    ],
)
def test_resource_rejects_invalid_fields(uri, name, size, message):
    with pytest.raises(MCPValidationError, match=message):
        MCPResource(uri, name, size=size)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MCPImplementation("", "1"),
        lambda: MCPRequestContext(
            MCP_PROTOCOL_VERSION,
            MCPImplementation("client", "1"),
            [],
        ),
        lambda: MCPRequestContext(
            MCP_PROTOCOL_VERSION,
            "invalid",
        ),
        lambda: MCPRequestContext(
            MCP_PROTOCOL_VERSION,
            MCPImplementation("client", "1"),
            {1: "invalid"},
        ),
        lambda: MCPServerCapabilities(tools=[]),
        lambda: MCPDiscoveryResult(
            [],
            MCPServerCapabilities(),
        ),
        lambda: MCPDiscoveryResult(
            [MCP_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION],
            MCPServerCapabilities(),
        ),
        lambda: MCPDiscoveryResult(
            MCP_PROTOCOL_VERSION,
            MCPServerCapabilities(),
        ),
        lambda: MCPDiscoveryResult(
            [MCP_PROTOCOL_VERSION],
            "invalid",
        ),
        lambda: MCPDiscoveryResult(
            [MCP_PROTOCOL_VERSION],
            MCPServerCapabilities(),
            server_info="invalid",
        ),
        lambda: MCPToolPage(["invalid"]),
        lambda: MCPResourcePage(["invalid"]),
        lambda: MCPToolPage([], ttl_ms=-1),
        lambda: MCPResourcePage([], next_cursor=" "),
        lambda: MCPContentBlock("text", {}),
        lambda: MCPContentBlock.text(1),
        lambda: MCPContentBlock("custom", {"type": "changed"}),
        lambda: MCPContentBlock("custom", {"value": object()}),
        lambda: MCPToolResult([]),
        lambda: MCPToolResult(["invalid"]),
        lambda: MCPToolResult(
            [MCPContentBlock.text("result")],
            is_error="yes",
        ),
        lambda: MCPToolResult(structured_content=float("nan")),
        lambda: MCPToolRequest(
            "tool",
            input_responses={"": {}},
        ),
        lambda: MCPToolRequest("tool", request_state=1),
        lambda: MCPInputRequiredResult([]),
        lambda: MCPInputRequiredResult(
            {
                "": MCPInputRequest(
                    "roots/list",
                    {},
                )
            }
        ),
        lambda: MCPContinuation(
            "continuation",
            "tools/list",
            {},
            round=1,
        ),
        lambda: MCPContinuation(
            "continuation",
            "tools/call",
            {},
            round=0,
        ),
        lambda: MCPResourceContent(
            "file:///resource",
        ),
        lambda: MCPResourceContent(
            "file:///resource",
            text="text",
            blob="YQ==",
        ),
        lambda: MCPResourceContent(
            "file:///resource",
            text=1,
        ),
        lambda: MCPResourceContent(
            "file:///resource",
            blob=1,
        ),
        lambda: MCPResourceContent(
            "file:///resource",
            blob="not base64!",
        ),
        lambda: MCPResourceContent(
            "file:///resource",
            text="text",
            annotations={"priority": float("nan")},
        ),
        lambda: MCPResourceReadResult([]),
    ],
)
def test_models_reject_invalid_contracts(factory):
    with pytest.raises(MCPValidationError):
        factory()


def test_tool_input_required_is_manual_and_retry_echoes_state_once():
    transport = RecordingTransport()
    transport.tool_result = input_required()
    client = make_client(
        transport,
        client_capabilities={"elicitation": {}},
    )
    client.open()

    continuation = client.call_tool(
        "delete_files",
        {"count": 3},
    )

    assert isinstance(continuation, MCPContinuation)
    assert continuation.operation == "tools/call"
    assert continuation.round == 1
    assert continuation.input_requests["confirm"].method == ("elicitation/create")
    assert not hasattr(continuation, "request_state")

    transport.tool_result = MCPToolResult([MCPContentBlock.text("deleted")])
    responses = {
        "confirm": {
            "action": "accept",
            "content": {"approved": True},
        }
    }
    result = client.continue_request(continuation, responses)
    responses["confirm"]["action"] = "cancel"

    assert isinstance(result, MCPToolResult)
    assert result.content[0].data["text"] == "deleted"
    _, _, retry, timeout = transport.calls[-1]
    assert retry.name == "delete_files"
    assert retry.arguments == {"count": 3}
    assert retry.input_responses == {
        "confirm": {
            "action": "accept",
            "content": {"approved": True},
        }
    }
    assert retry.request_state == "opaque-state"
    assert timeout == 2.5
    assert client.state is MCPConnectionState.OPEN

    with pytest.raises(MCPContinuationError, match="consumed"):
        client.continue_request(continuation, {})


def test_continuation_requires_exact_keys_and_can_be_cancelled():
    transport = RecordingTransport()
    transport.resource_read_result = input_required()
    client = make_client(
        transport,
        client_capabilities={"elicitation": {}},
    )
    client.open()
    continuation = client.read_resource("file:///protected.txt")

    with pytest.raises(MCPContinuationError, match="invalid"):
        client.cancel_continuation(object())
    with pytest.raises(MCPContinuationError, match="exactly match"):
        client.continue_request(continuation, {})

    client.cancel_continuation(continuation)

    with pytest.raises(MCPContinuationError, match="consumed"):
        client.cancel_continuation(continuation)
    assert client.state is MCPConnectionState.OPEN
    assert [call[0] for call in transport.calls] == [
        "open",
        "read_resource",
    ]


def test_state_only_continuation_retries_resource_without_responses():
    transport = RecordingTransport()
    transport.resource_read_result = MCPInputRequiredResult(request_state="state-only")
    client = make_client(transport)
    client.open()
    continuation = client.read_resource("file:///large.txt")
    transport.resource_read_result = MCPResourceReadResult(
        [MCPResourceContent("file:///large.txt", text="ready")]
    )

    result = client.continue_request(continuation)

    assert isinstance(result, MCPResourceReadResult)
    retry = transport.calls[-1][2]
    assert retry.input_responses == {}
    assert retry.request_state == "state-only"


def test_undeclared_input_capability_is_a_protocol_failure():
    transport = RecordingTransport()
    transport.tool_result = input_required(method="sampling/createMessage")
    client = make_client(transport)
    client.open()

    with pytest.raises(MCPProtocolError, match="undeclared"):
        client.call_tool("writer")

    assert client.state is MCPConnectionState.FAILED
    assert transport.calls[-1] == ("close",)


def test_input_round_limit_stops_without_closing_transport():
    transport = RecordingTransport()
    transport.tool_result = input_required()
    client = make_client(
        transport,
        client_capabilities={"elicitation": {}},
        max_input_rounds=1,
    )
    client.open()
    continuation = client.call_tool("interactive")

    with pytest.raises(MCPInputRoundsExceededError, match="limit"):
        client.continue_request(
            continuation,
            {"confirm": {"action": "decline"}},
        )

    assert client.state is MCPConnectionState.OPEN
    with pytest.raises(MCPContinuationError, match="consumed"):
        client.continue_request(continuation, {})
