import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from threading import Thread

import pytest

from ai_sdk.mcp import (
    MCPClient,
    MCPContinuation,
    MCPImplementation,
    MCPToolAdapter,
    StreamableHTTPTransport,
)
from ai_sdk.tools import ToolCall, ToolExecutor, ToolRegistry


class LocalMCPHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        method = payload["method"]
        params = payload["params"]
        assert self.path == "/mcp"
        assert self.headers["Accept"] == ("application/json, text/event-stream")
        assert (
            self.headers["MCP-Protocol-Version"]
            == (params["_meta"]["io.modelcontextprotocol/protocolVersion"])
        )
        assert self.headers["Mcp-Method"] == method
        if method == "tools/call":
            assert self.headers["Mcp-Name"] == params["name"]
            assert self.headers["Mcp-Param-Tenant"] == (params["arguments"]["tenant"])
        if method == "resources/read":
            assert self.headers["Mcp-Name"] == params["uri"]

        self.requests.append(
            {
                "method": method,
                "authorization": self.headers["Authorization"],
            }
        )
        result = self._result(method, params)
        result_type = result.pop("resultType", "complete")
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"resultType": result_type, **result},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _result(method, params):
        if method == "server/discover":
            return {
                "supportedVersions": ["2026-07-28"],
                "capabilities": {"tools": {}, "resources": {}},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "search_docs",
                        "description": "Search documentation.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query.",
                                },
                                "tenant": {
                                    "type": "string",
                                    "description": "Tenant name.",
                                    "x-mcp-header": "Tenant",
                                },
                            },
                            "required": ["query", "tenant"],
                        },
                    }
                ]
            }
        if method == "tools/call":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": ("Found guide for " + params["arguments"]["query"]),
                    }
                ],
                "isError": False,
            }
        if method == "resources/read":
            if "inputResponses" not in params:
                return {
                    "resultType": "input_required",
                    "inputRequests": {
                        "confirm_read": {
                            "method": "elicitation/create",
                            "params": {
                                "mode": "form",
                                "message": "Read the protected guide?",
                                "requestedSchema": {
                                    "type": "object",
                                    "properties": {"approved": {"type": "boolean"}},
                                    "required": ["approved"],
                                },
                            },
                        }
                    },
                    "requestState": "local-opaque-state",
                }
            assert params["inputResponses"] == {
                "confirm_read": {
                    "action": "accept",
                    "content": {"approved": True},
                }
            }
            assert params["requestState"] == "local-opaque-state"
            return {
                "contents": [
                    {
                        "uri": params["uri"],
                        "mimeType": "text/markdown",
                        "text": "# Local HTTP MCP",
                    }
                ]
            }
        raise AssertionError(f"Unexpected method: {method}")

    def log_message(self, format, *args):
        pass


@pytest.mark.integration
def test_local_streamable_http_approved_tool_and_resource_workflow():
    LocalMCPHandler.requests = []
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        LocalMCPHandler,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token_numbers = count(1)
    endpoint = f"http://127.0.0.1:{server.server_port}/mcp"
    transport = StreamableHTTPTransport(
        endpoint,
        authorization_provider=lambda: f"Bearer local-{next(token_numbers)}",
    )
    client = MCPClient(
        transport,
        client_info=MCPImplementation("integration-client", "1.0"),
        client_capabilities={"elicitation": {}},
        timeout_seconds=3,
    )

    try:
        with client:
            discovery = client.discover()
            tools = client.list_tools()
            registry = ToolRegistry()
            MCPToolAdapter(client).register_approved(
                registry,
                tools.tools,
                approved_names=["search_docs"],
            )
            tool_result = ToolExecutor(registry).execute(
                ToolCall(
                    "call-1",
                    "search_docs",
                    {"query": "Python", "tenant": "local"},
                )
            )
            continuation = client.read_resource("file:///guides/python.md")
            assert isinstance(continuation, MCPContinuation)
            resource = client.continue_request(
                continuation,
                {
                    "confirm_read": {
                        "action": "accept",
                        "content": {"approved": True},
                    }
                },
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert discovery.capabilities.supports_tools
    assert tool_result.content == "Found guide for Python"
    assert not tool_result.is_error
    assert resource.contents[0].text == "# Local HTTP MCP"
    assert LocalMCPHandler.requests == [
        {
            "method": "server/discover",
            "authorization": "Bearer local-1",
        },
        {
            "method": "tools/list",
            "authorization": "Bearer local-2",
        },
        {
            "method": "tools/call",
            "authorization": "Bearer local-3",
        },
        {
            "method": "resources/read",
            "authorization": "Bearer local-4",
        },
        {
            "method": "resources/read",
            "authorization": "Bearer local-5",
        },
    ]
