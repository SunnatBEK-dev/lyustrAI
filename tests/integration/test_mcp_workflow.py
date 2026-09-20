import pytest

from ai_sdk.mcp import (
    MCP_PROTOCOL_VERSION,
    BaseMCPTransport,
    MCPClient,
    MCPConnectionState,
    MCPContentBlock,
    MCPDiscoveryResult,
    MCPImplementation,
    MCPResource,
    MCPResourceContent,
    MCPResourcePage,
    MCPResourceReadResult,
    MCPServerCapabilities,
    MCPTool,
    MCPToolAdapter,
    MCPToolPage,
    MCPToolResult,
)
from ai_sdk.tools import ToolCall, ToolExecutor, ToolRegistry


class InMemoryMCPTransport(BaseMCPTransport):
    def __init__(self):
        self.is_open = False
        self.received_meta = []

    def open(self, *, timeout_seconds):
        assert timeout_seconds == 5.0
        self.is_open = True

    def discover(self, context, *, timeout_seconds):
        assert self.is_open
        self.received_meta.append(context.to_meta())
        return MCPDiscoveryResult(
            [MCP_PROTOCOL_VERSION],
            MCPServerCapabilities(tools={}, resources={}),
            server_info=MCPImplementation("knowledge", "1.0"),
            instructions="Use the search tool before reading files.",
            ttl_ms=60_000,
            cache_scope="private",
        )

    def list_tools(self, context, *, cursor, timeout_seconds):
        assert self.is_open
        self.received_meta.append(context.to_meta())
        if cursor is None:
            return MCPToolPage(
                [
                    MCPTool(
                        "search_docs",
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    )
                ],
                next_cursor="page-2",
            )
        return MCPToolPage([MCPTool("health-check", {"type": "object"})])

    def list_resources(
        self,
        context,
        *,
        cursor,
        timeout_seconds,
    ):
        assert self.is_open
        assert cursor is None
        self.received_meta.append(context.to_meta())
        return MCPResourcePage(
            [
                MCPResource(
                    "file:///knowledge/python.md",
                    "python.md",
                    mime_type="text/markdown",
                )
            ],
            ttl_ms=10_000,
            cache_scope="private",
        )

    def call_tool(
        self,
        context,
        request,
        *,
        timeout_seconds,
    ):
        assert self.is_open
        assert request.name == "search_docs"
        assert request.arguments == {"query": "Python"}
        self.received_meta.append(context.to_meta())
        return MCPToolResult(
            [MCPContentBlock.text("Found Python guide.")],
            structured_content={"matches": 1},
        )

    def read_resource(
        self,
        context,
        request,
        *,
        timeout_seconds,
    ):
        assert self.is_open
        assert request.uri == "file:///knowledge/python.md"
        self.received_meta.append(context.to_meta())
        return MCPResourceReadResult(
            [
                MCPResourceContent(
                    request.uri,
                    mime_type="text/markdown",
                    text="# Python guide",
                )
            ],
            ttl_ms=10_000,
            cache_scope="private",
        )

    def close(self):
        self.is_open = False


@pytest.mark.integration
def test_stateless_mcp_discovery_and_catalog_workflow():
    transport = InMemoryMCPTransport()
    client = MCPClient(
        transport,
        client_info=MCPImplementation("ai-sdk", "0.1.0"),
        client_capabilities={"elicitation": {}},
        timeout_seconds=5,
    )

    with client:
        discovery = client.discover()
        first_tools = client.list_tools()
        second_tools = client.list_tools(cursor=first_tools.next_cursor)
        resources = client.list_resources()
        registry = ToolRegistry()
        registered = MCPToolAdapter(client).register_approved(
            registry,
            first_tools.tools + second_tools.tools,
            approved_names=["search_docs"],
        )
        tool_result = ToolExecutor(registry).execute(
            ToolCall(
                "call-1",
                "search_docs",
                {"query": "Python"},
            )
        )
        resource_result = client.read_resource(resources.resources[0].uri)

        assert discovery.capabilities.supports_tools
        assert discovery.capabilities.supports_resources
        assert first_tools.tools[0].name == "search_docs"
        assert second_tools.tools[0].name == "health-check"
        assert resources.resources[0].uri == ("file:///knowledge/python.md")
        assert registered == ("search_docs",)
        assert registry.count() == 1
        assert tool_result.content == "Found Python guide."
        assert not tool_result.is_error
        assert resource_result.contents[0].text == "# Python guide"

    assert client.state is MCPConnectionState.CLOSED
    assert not transport.is_open
    assert len(transport.received_meta) == 6
    assert all(
        meta["io.modelcontextprotocol/protocolVersion"] == MCP_PROTOCOL_VERSION
        for meta in transport.received_meta
    )
    assert all(
        meta["io.modelcontextprotocol/clientInfo"]
        == {"name": "ai-sdk", "version": "0.1.0"}
        for meta in transport.received_meta
    )
