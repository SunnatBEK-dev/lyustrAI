import json
from base64 import b64encode
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from ai_sdk.mcp import (
    MCP_PROTOCOL_VERSION,
    MCPClient,
    MCPHTTPError,
    MCPImplementation,
    MCPInputRequiredResult,
    MCPRemoteError,
    MCPRequestContext,
    MCPResourceReadRequest,
    MCPToolRequest,
    MCPTransportResponseError,
    StreamableHTTPTransport,
)


class FakeResponse:
    def __init__(
        self,
        payload,
        *,
        content_type="application/json",
        status=200,
        raw=False,
    ):
        self.body = payload if raw else json.dumps(payload).encode("utf-8")
        self.headers = {"Content-Type": content_type}
        self.status = status
        self.closed = False

    def read(self, limit):
        return self.body[:limit]

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class FakeOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, *, timeout):
        self.calls.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def result_response(result, *, request_id=1, **kwargs):
    return FakeResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resultType": "complete", **result},
        },
        **kwargs,
    )


def make_context():
    return MCPRequestContext(
        MCP_PROTOCOL_VERSION,
        MCPImplementation("http-test", "1.0"),
        {"elicitation": {}},
    )


def request_headers(request):
    return {key.casefold(): value for key, value in request.header_items()}


def test_discovery_sends_stateless_headers_metadata_and_fresh_auth():
    opener = FakeOpener(
        result_response(
            {
                "supportedVersions": [MCP_PROTOCOL_VERSION],
                "capabilities": {"tools": {}, "resources": {}},
                "instructions": "Use approved tools only.",
                "ttlMs": 1000,
                "cacheScope": "private",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "server",
                        "version": "2.0",
                    }
                },
            },
            request_id=1,
        ),
        result_response(
            {
                "supportedVersions": [MCP_PROTOCOL_VERSION],
                "capabilities": {},
            },
            request_id=2,
        ),
    )
    tokens = iter(["Bearer first", "Bearer second"])
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        authorization_provider=lambda: next(tokens),
        opener=opener,
    )
    transport.open(timeout_seconds=3)

    first = transport.discover(make_context(), timeout_seconds=3)
    second = transport.discover(make_context(), timeout_seconds=4)

    assert first.server_info == MCPImplementation("server", "2.0")
    assert first.capabilities.supports_tools
    assert first.capabilities.supports_resources
    assert first.ttl_ms == 1000
    assert second.server_info is None
    first_request, first_timeout = opener.calls[0]
    second_request, second_timeout = opener.calls[1]
    first_headers = request_headers(first_request)
    second_headers = request_headers(second_request)
    assert first_request.get_method() == "POST"
    assert first_timeout == 3
    assert second_timeout == 4
    assert first_headers["accept"] == ("application/json, text/event-stream")
    assert first_headers["content-type"] == "application/json"
    assert first_headers["mcp-protocol-version"] == (MCP_PROTOCOL_VERSION)
    assert first_headers["mcp-method"] == "server/discover"
    assert "mcp-name" not in first_headers
    assert first_headers["authorization"] == "Bearer first"
    assert second_headers["authorization"] == "Bearer second"
    first_body = json.loads(first_request.data)
    second_body = json.loads(second_request.data)
    assert first_body["id"] == 1
    assert second_body["id"] == 2
    assert first_body["params"]["_meta"] == make_context().to_meta()


def test_tool_list_filters_invalid_header_schema_and_mirrors_values():
    opener = FakeOpener(
        result_response(
            {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "greeting": {
                                    "type": "string",
                                    "x-mcp-header": "Greeting",
                                },
                                "count": {
                                    "type": "integer",
                                    "x-mcp-header": "Count",
                                },
                                "enabled": {
                                    "type": "boolean",
                                    "x-mcp-header": "Enabled",
                                },
                                "context": {
                                    "type": "object",
                                    "properties": {
                                        "tenant": {
                                            "type": "string",
                                            "x-mcp-header": "Tenant",
                                        }
                                    },
                                },
                            },
                        },
                    },
                    {
                        "name": "excluded",
                        "inputSchema": {
                            "type": "object",
                            "x-mcp-header": "Root",
                        },
                    },
                ],
                "ttlMs": 500,
                "cacheScope": "public",
            },
            request_id=1,
        ),
        result_response(
            {
                "content": [
                    {"type": "text", "text": "found"},
                    {
                        "type": "image",
                        "data": "YWJj",
                        "mimeType": "image/png",
                    },
                ],
                "structuredContent": {"count": 1},
                "isError": False,
            },
            request_id=2,
        ),
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=2)

    page = transport.list_tools(
        make_context(),
        cursor="page-1",
        timeout_seconds=2,
    )
    result = transport.call_tool(
        make_context(),
        MCPToolRequest(
            "search",
            {
                "greeting": "Hello, dunyo 🌍",
                "count": 2,
                "enabled": False,
                "context": {"tenant": " acme "},
            },
        ),
        timeout_seconds=2,
    )

    assert [tool.name for tool in page.tools] == ["search"]
    assert page.ttl_ms == 500
    assert len(result.content) == 2
    assert result.structured_content == {"count": 1}
    list_body = json.loads(opener.calls[0][0].data)
    assert list_body["params"]["cursor"] == "page-1"
    call_request = opener.calls[1][0]
    headers = request_headers(call_request)
    assert headers["mcp-method"] == "tools/call"
    assert headers["mcp-name"] == "search"
    assert headers["mcp-param-count"] == "2"
    assert headers["mcp-param-enabled"] == "false"
    assert headers["mcp-param-greeting"] == ("=?base64?SGVsbG8sIGR1bnlvIPCfjI0=?=")
    assert headers["mcp-param-tenant"] == ("=?base64?IGFjbWUg?=")


@pytest.mark.parametrize(
    "invalid_schema",
    [
        {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "x-mcp-header": "Value",
                }
            },
        },
        {
            "type": "object",
            "properties": {
                "one": {
                    "type": "string",
                    "x-mcp-header": "Same",
                },
                "two": {
                    "type": "string",
                    "x-mcp-header": "same",
                },
            },
        },
        {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "x-mcp-header": "bad name",
                }
            },
        },
        {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "x-mcp-header": "Value",
                    },
                }
            },
        },
    ],
)
def test_tool_list_excludes_each_invalid_x_mcp_header(invalid_schema):
    opener = FakeOpener(
        result_response(
            {
                "tools": [
                    {
                        "name": "invalid",
                        "inputSchema": invalid_schema,
                    }
                ]
            }
        )
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)

    page = transport.list_tools(
        make_context(),
        cursor=None,
        timeout_seconds=1,
    )

    assert page.tools == ()


@pytest.mark.parametrize(
    ("schema", "arguments"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "x-mcp-header": "Value",
                    }
                },
            },
            {"value": 1},
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "x-mcp-header": "Value",
                    }
                },
            },
            {"value": 2**53},
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "boolean",
                        "x-mcp-header": "Value",
                    }
                },
            },
            {"value": "true"},
        ),
    ],
)
def test_tool_call_rejects_invalid_mirrored_argument(schema, arguments):
    opener = FakeOpener(
        result_response({"tools": [{"name": "tool", "inputSchema": schema}]})
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)
    transport.list_tools(
        make_context(),
        cursor=None,
        timeout_seconds=1,
    )

    with pytest.raises(MCPHTTPError, match="invalid value"):
        transport.call_tool(
            make_context(),
            MCPToolRequest("tool", arguments),
            timeout_seconds=1,
        )

    assert len(opener.calls) == 1


def test_resource_list_and_read_map_content_and_encode_unicode_name():
    uri = "file:///qo‘llanma/世界.md"
    opener = FakeOpener(
        result_response(
            {
                "resources": [
                    {
                        "uri": uri,
                        "name": "guide.md",
                        "title": "Guide",
                        "description": "A guide.",
                        "mimeType": "text/markdown",
                        "size": 12,
                    }
                ],
                "nextCursor": "next",
            },
            request_id=1,
        ),
        result_response(
            {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": "# Guide",
                        "annotations": {
                            "audience": ["user"],
                            "priority": 0.9,
                        },
                    },
                    {
                        "uri": "file:///image.png",
                        "mimeType": "image/png",
                        "blob": "YWJj",
                    },
                ],
                "ttlMs": 600,
                "cacheScope": "private",
            },
            request_id=2,
        ),
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)

    page = transport.list_resources(
        make_context(),
        cursor=None,
        timeout_seconds=1,
    )
    result = transport.read_resource(
        make_context(),
        MCPResourceReadRequest(uri),
        timeout_seconds=1,
    )

    assert page.resources[0].size == 12
    assert page.next_cursor == "next"
    assert result.contents[0].text == "# Guide"
    assert result.contents[0].annotations["priority"] == 0.9
    assert result.contents[1].blob == "YWJj"
    read_headers = request_headers(opener.calls[1][0])
    encoded_uri = b64encode(uri.encode()).decode()
    expected_name = f"=?base64?{encoded_uri}?="
    assert read_headers["mcp-name"] == expected_name


def test_sse_ignores_comments_and_notifications_before_final_response():
    final = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "resultType": "complete",
                "tools": [],
            },
        }
    )
    notification = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progress": 0.5},
        }
    )
    body = (
        f": keep-alive\n\ndata: {notification}\n\nevent: message\ndata: {final}\n\n"
    ).encode()
    opener = FakeOpener(
        FakeResponse(
            body,
            content_type="text/event-stream; charset=utf-8",
            raw=True,
        )
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)

    page = transport.list_tools(
        make_context(),
        cursor=None,
        timeout_seconds=1,
    )

    assert page.tools == ()


def test_sse_accepts_final_event_without_trailing_blank_line():
    body = (
        b'data: {"jsonrpc":"2.0","id":1,"result":{"resultType":"complete","tools":[]}}'
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(
            FakeResponse(
                body,
                content_type="text/event-stream",
                raw=True,
            )
        ),
    )
    transport.open(timeout_seconds=1)

    assert (
        transport.list_tools(
            make_context(),
            cursor=None,
            timeout_seconds=1,
        ).tools
        == ()
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b": only comment\n\n", "without a final"),
        (
            b'data: {"jsonrpc":"2.0","id":9,"method":"sampling/createMessage"}\n\n',
            "server request",
        ),
        (
            b'data: {"jsonrpc":"2.0","id":1,'
            b'"result":{"tools":[]}}\n\n'
            b'data: {"jsonrpc":"2.0",'
            b'"method":"notifications/progress"}\n\n',
            "continued",
        ),
        (b"data: \xff\n\n", "UTF-8"),
    ],
)
def test_sse_rejects_invalid_streams(body, message):
    opener = FakeOpener(
        FakeResponse(
            body,
            content_type="text/event-stream",
            raw=True,
        )
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match=message):
        transport.list_tools(
            make_context(),
            cursor=None,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be an object"),
        ({"jsonrpc": "1.0", "id": 1, "result": {}}, "version"),
        ({"jsonrpc": "2.0", "id": 2, "result": {}}, "ID"),
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
                "error": {},
            },
            "result or error",
        ),
        ({"jsonrpc": "2.0", "id": 1}, "result or error"),
        ({"jsonrpc": "2.0", "id": 1, "result": []}, "object"),
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"resultType": "input_required"},
            },
            "unsupported",
        ),
        (
            {"jsonrpc": "2.0", "id": 1, "error": []},
            "error is invalid",
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": True, "message": "bad"},
            },
            "error is invalid",
        ),
    ],
)
def test_json_rpc_envelope_is_strictly_validated(payload, message):
    opener = FakeOpener(FakeResponse(payload))
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match=message):
        transport.discover(make_context(), timeout_seconds=1)


def invoke_operation(transport, operation):
    if operation == "discover":
        return transport.discover(make_context(), timeout_seconds=1)
    if operation == "tools":
        return transport.list_tools(
            make_context(),
            cursor=None,
            timeout_seconds=1,
        )
    if operation == "resources":
        return transport.list_resources(
            make_context(),
            cursor=None,
            timeout_seconds=1,
        )
    if operation == "call":
        return transport.call_tool(
            make_context(),
            MCPToolRequest("tool"),
            timeout_seconds=1,
        )
    return transport.read_resource(
        make_context(),
        MCPResourceReadRequest("file:///resource"),
        timeout_seconds=1,
    )


@pytest.mark.parametrize(
    ("operation", "result", "message"),
    [
        ("discover", {}, "discovery result"),
        (
            "discover",
            {
                "supportedVersions": [MCP_PROTOCOL_VERSION],
                "capabilities": {"tools": True},
            },
            "capability",
        ),
        ("tools", {"tools": {}}, "tools/list result"),
        ("tools", {"tools": ["invalid"]}, "invalid tool"),
        (
            "tools",
            {"tools": [{"name": "tool", "inputSchema": []}]},
            "input schema",
        ),
        ("resources", {"resources": {}}, "resources/list result"),
        (
            "resources",
            {"resources": ["invalid"]},
            "invalid resource",
        ),
        ("call", {"content": {}}, "tools/call result"),
        ("call", {"content": ["invalid"]}, "content block"),
        ("call", {"content": [{}]}, "content type"),
        ("read", {"contents": {}}, "resources/read result"),
        ("read", {"contents": ["invalid"]}, "resource content"),
    ],
)
def test_operation_results_are_strictly_mapped(
    operation,
    result,
    message,
):
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(result_response(result)),
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match=message):
        invoke_operation(transport, operation)


def test_malformed_server_info_is_ignored():
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(
            result_response(
                {
                    "supportedVersions": [MCP_PROTOCOL_VERSION],
                    "capabilities": {},
                    "_meta": {
                        "io.modelcontextprotocol/serverInfo": {
                            "name": "",
                            "version": "1",
                        }
                    },
                }
            )
        ),
    )
    transport.open(timeout_seconds=1)

    assert (
        transport.discover(
            make_context(),
            timeout_seconds=1,
        ).server_info
        is None
    )


@pytest.mark.parametrize(
    ("response", "limit", "message"),
    [
        (
            FakeResponse(
                b"not-json",
                raw=True,
            ),
            100,
            "UTF-8 JSON",
        ),
        (
            FakeResponse({}, content_type="text/plain"),
            100,
            "content type",
        ),
        (
            FakeResponse(b"12345", raw=True),
            4,
            "size limit",
        ),
    ],
)
def test_http_response_framing_is_bounded(response, limit, message):
    opener = FakeOpener(response)
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
        max_response_bytes=limit,
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match=message):
        transport.discover(make_context(), timeout_seconds=1)


def test_non_json_http_error_uses_status_without_body_details():
    error = make_http_error(
        {"private": "secret"},
        status=401,
        content_type="text/plain",
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(error),
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match="status 401") as caught:
        transport.discover(make_context(), timeout_seconds=1)

    assert "secret" not in str(caught.value)


def make_http_error(payload, *, status=400, content_type="application/json"):
    headers = Message()
    headers["Content-Type"] = content_type
    return HTTPError(
        "https://example.com/mcp",
        status,
        "failure",
        headers,
        BytesIO(json.dumps(payload).encode()),
    )


def test_json_rpc_error_is_typed_and_keeps_client_transport_open():
    error_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32602,
            "message": "Resource not found",
        },
    }
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(make_http_error(error_payload)),
    )
    client = MCPClient(
        transport,
        client_info=MCPImplementation("client", "1.0"),
    )
    client.open()

    with pytest.raises(MCPRemoteError) as caught:
        client.read_resource("file:///missing.txt")

    assert caught.value.code == -32602
    assert caught.value.message == "Resource not found"
    assert client.state.value == "open"


def test_direct_transport_exposes_typed_json_rpc_error():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32022, "message": "Unsupported"},
    }
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(FakeResponse(payload)),
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPTransportResponseError) as caught:
        transport.discover(make_context(), timeout_seconds=1)

    assert caught.value.code == -32022


@pytest.mark.parametrize(
    ("error", "error_type", "message"),
    [
        (URLError(TimeoutError()), TimeoutError, None),
        (TimeoutError(), TimeoutError, None),
        (URLError("offline"), MCPHTTPError, "URLError"),
        (RuntimeError("secret"), MCPHTTPError, "RuntimeError"),
    ],
)
def test_network_failures_are_contained(error, error_type, message):
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(error),
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(error_type, match=message) as caught:
        transport.discover(make_context(), timeout_seconds=1)

    assert "secret" not in str(caught.value)
    assert "offline" not in str(caught.value)


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (lambda: "Bearer bad\nvalue", "header-safe"),
        (
            lambda: (_ for _ in ()).throw(RuntimeError("secret")),
            "RuntimeError",
        ),
    ],
)
def test_authorization_provider_failures_are_contained(provider, message):
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        authorization_provider=provider,
        opener=FakeOpener(result_response({})),
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match=message) as caught:
        transport.discover(make_context(), timeout_seconds=1)

    assert "secret" not in str(caught.value)


def test_authorization_provider_may_omit_header_for_one_request():
    opener = FakeOpener(
        result_response(
            {
                "supportedVersions": [MCP_PROTOCOL_VERSION],
                "capabilities": {},
            }
        )
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        authorization_provider=lambda: None,
        opener=opener,
    )
    transport.open(timeout_seconds=1)

    transport.discover(make_context(), timeout_seconds=1)

    assert "authorization" not in request_headers(opener.calls[0][0])


def test_non_json_request_metadata_is_rejected_before_network():
    opener = FakeOpener(result_response({}))
    context = make_context()
    context.client_capabilities["invalid"] = object()
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match="request body"):
        transport.discover(context, timeout_seconds=1)

    assert opener.calls == []


def test_missing_or_null_header_arguments_are_omitted():
    opener = FakeOpener(
        result_response(
            {
                "tools": [
                    {
                        "name": "tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "one": {
                                    "type": "string",
                                    "x-mcp-header": "One",
                                },
                                "two": {
                                    "type": "string",
                                    "x-mcp-header": "Two",
                                },
                            },
                        },
                    }
                ]
            },
            request_id=1,
        ),
        result_response(
            {"content": [{"type": "text", "text": "ok"}]},
            request_id=2,
        ),
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)
    transport.list_tools(
        make_context(),
        cursor=None,
        timeout_seconds=1,
    )

    result = transport.call_tool(
        make_context(),
        MCPToolRequest("tool", {"one": None}),
        timeout_seconds=1,
    )

    headers = request_headers(opener.calls[1][0])
    assert result.content[0].data["text"] == "ok"
    assert "mcp-param-one" not in headers
    assert "mcp-param-two" not in headers


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "ftp://example.com/mcp",
        "https:///missing-host",
        "https://user:pass@example.com/mcp",
        "https://example.com/mcp#fragment",
        "https://example.com:99999/mcp",
    ],
)
def test_transport_rejects_invalid_endpoint(endpoint):
    with pytest.raises(MCPHTTPError, match="endpoint"):
        StreamableHTTPTransport(endpoint)


def test_transport_validates_config_and_lifecycle():
    with pytest.raises(MCPHTTPError, match="callable"):
        StreamableHTTPTransport(
            "https://example.com/mcp",
            authorization_provider="token",
        )
    with pytest.raises(MCPHTTPError, match="response size"):
        StreamableHTTPTransport(
            "https://example.com/mcp",
            max_response_bytes=0,
        )
    with pytest.raises(MCPHTTPError, match="opener"):
        StreamableHTTPTransport(
            "https://example.com/mcp",
            opener=object(),
        )

    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(result_response({})),
    )
    with pytest.raises(MCPHTTPError, match="not open"):
        transport.discover(make_context(), timeout_seconds=1)
    transport.open(timeout_seconds=1)
    with pytest.raises(MCPHTTPError, match="already open"):
        transport.open(timeout_seconds=1)
    transport.close()
    with pytest.raises(MCPHTTPError, match="not open"):
        transport.discover(make_context(), timeout_seconds=1)


def test_tool_input_required_and_retry_use_new_id_and_exact_state():
    request_state = "opaque:\u2603:  "
    opener = FakeOpener(
        FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "resultType": "input_required",
                    "inputRequests": {
                        "confirm": {
                            "method": "elicitation/create",
                            "params": {
                                "mode": "form",
                                "message": "Delete files?",
                                "requestedSchema": {
                                    "type": "object",
                                    "properties": {"approved": {"type": "boolean"}},
                                },
                            },
                        }
                    },
                    "requestState": request_state,
                },
            }
        ),
        result_response(
            {
                "content": [{"type": "text", "text": "deleted"}],
                "isError": False,
            },
            request_id=2,
        ),
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=opener,
    )
    transport.open(timeout_seconds=1)
    original = MCPToolRequest("delete", {"count": 3})

    required = transport.call_tool(
        make_context(),
        original,
        timeout_seconds=1,
    )
    responses = {"confirm": {"action": "accept"}}
    retry = MCPToolRequest(
        original.name,
        original.arguments,
        input_responses=responses,
        request_state=required.request_state,
    )
    responses["confirm"]["action"] = "cancel"
    complete = transport.call_tool(
        make_context(),
        retry,
        timeout_seconds=1,
    )

    assert isinstance(required, MCPInputRequiredResult)
    assert required.input_requests["confirm"].params["message"] == ("Delete files?")
    assert required.request_state == request_state
    assert complete.content[0].data["text"] == "deleted"
    first = json.loads(opener.calls[0][0].data)
    second = json.loads(opener.calls[1][0].data)
    assert first["id"] == 1
    assert second["id"] == 2
    assert "inputResponses" not in first["params"]
    assert second["params"]["inputResponses"] == {"confirm": {"action": "accept"}}
    assert second["params"]["requestState"] == request_state


def test_sse_can_end_with_input_required_resource_result():
    body = (
        b'data: {"jsonrpc":"2.0","id":1,"result":'
        b'{"resultType":"input_required","requestState":"next"}}\n\n'
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(
            FakeResponse(
                body,
                content_type="text/event-stream",
                raw=True,
            )
        ),
    )
    transport.open(timeout_seconds=1)

    result = transport.read_resource(
        make_context(),
        MCPResourceReadRequest("file:///large.txt"),
        timeout_seconds=1,
    )

    assert isinstance(result, MCPInputRequiredResult)
    assert result.input_requests == {}
    assert result.request_state == "next"


@pytest.mark.parametrize(
    ("incomplete", "message"),
    [
        ({"inputRequests": []}, "inputRequests"),
        ({"inputRequests": None}, "inputRequests"),
        (
            {
                "inputRequests": {
                    "": {
                        "method": "roots/list",
                        "params": {},
                    }
                }
            },
            "invalid request",
        ),
        (
            {"inputRequests": {"one": {"method": "roots/list"}}},
            "invalid request",
        ),
        (
            {
                "inputRequests": {
                    "one": {
                        "method": "unknown/request",
                        "params": {},
                    }
                }
            },
            "invalid request",
        ),
        ({"requestState": 1}, "requestState"),
    ],
)
def test_input_required_framing_is_strict(incomplete, message):
    response = FakeResponse(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "resultType": "input_required",
                **incomplete,
            },
        }
    )
    transport = StreamableHTTPTransport(
        "https://example.com/mcp",
        opener=FakeOpener(response),
    )
    transport.open(timeout_seconds=1)

    with pytest.raises(MCPHTTPError, match=message):
        transport.call_tool(
            make_context(),
            MCPToolRequest("interactive"),
            timeout_seconds=1,
        )
