import json
from collections.abc import Mapping, Sequence

from ai_sdk.mcp.client import MCPClient
from ai_sdk.mcp.model import (
    MCPContentBlock,
    MCPContinuation,
    MCPTool,
    MCPToolResult,
)
from ai_sdk.tools import (
    ToolHandlerError,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
    ToolValidationError,
)


class MCPToolAdapterError(ValueError):
    """Raised when an MCP tool cannot be safely registered."""


class MCPToolAdapter:
    """Register an explicit compatible subset of MCP tools locally."""

    _ROOT_SCHEMA_KEYS = {
        "$schema",
        "type",
        "title",
        "description",
        "properties",
        "required",
        "additionalProperties",
    }
    _PARAMETER_SCHEMA_KEYS = {
        "type",
        "title",
        "description",
        "x-mcp-header",
    }
    _PARAMETER_TYPES = {
        "string": ToolParameterType.STRING,
        "integer": ToolParameterType.INTEGER,
        "number": ToolParameterType.NUMBER,
        "boolean": ToolParameterType.BOOLEAN,
    }

    def __init__(self, client: MCPClient) -> None:
        if not isinstance(client, MCPClient):
            raise MCPToolAdapterError("MCP tool adapter client is invalid.")
        self._client = client

    def register_approved(
        self,
        registry: ToolRegistry,
        tools: Sequence[MCPTool],
        *,
        approved_names: Sequence[str],
    ) -> tuple[str, ...]:
        if not isinstance(registry, ToolRegistry):
            raise MCPToolAdapterError("MCP tool registry is invalid.")
        normalized_tools = self._normalize_tools(tools)
        approved = self._normalize_approved_names(approved_names)
        approved_set = set(approved)
        tool_by_name = {tool.name: tool for tool in normalized_tools}
        missing = sorted(approved_set - set(tool_by_name))
        if missing:
            raise MCPToolAdapterError(
                "Approved MCP tools were not discovered: " + ", ".join(missing) + "."
            )

        selected = [tool for tool in normalized_tools if tool.name in approved_set]
        schemas = [self._to_local_schema(tool) for tool in selected]
        collisions = [
            schema.name for schema in schemas if registry.get(schema.name) is not None
        ]
        if collisions:
            raise MCPToolAdapterError(
                "MCP tools conflict with registered tools: "
                + ", ".join(collisions)
                + "."
            )

        for tool, schema in zip(selected, schemas):
            registry.register(
                schema,
                self._make_handler(tool.name),
            )
        return tuple(tool.name for tool in selected)

    @staticmethod
    def _normalize_tools(
        tools: Sequence[MCPTool],
    ) -> tuple[MCPTool, ...]:
        if isinstance(tools, (str, bytes)):
            raise MCPToolAdapterError("MCP tools must be a sequence.")
        normalized = tuple(tools)
        if any(not isinstance(tool, MCPTool) for tool in normalized):
            raise MCPToolAdapterError("MCP tools contain an invalid definition.")
        names = [tool.name for tool in normalized]
        if len(names) != len(set(names)):
            raise MCPToolAdapterError("MCP tool names must be unique.")
        return normalized

    @staticmethod
    def _normalize_approved_names(
        approved_names: Sequence[str],
    ) -> tuple[str, ...]:
        if isinstance(approved_names, (str, bytes)):
            raise MCPToolAdapterError("Approved MCP tool names must be a sequence.")
        approved = tuple(approved_names)
        if any(not isinstance(name, str) or not name.strip() for name in approved):
            raise MCPToolAdapterError("Approved MCP tool names cannot be empty.")
        if len(approved) != len(set(approved)):
            raise MCPToolAdapterError("Approved MCP tool names must be unique.")
        return approved

    def _to_local_schema(self, tool: MCPTool) -> ToolSchema:
        schema = tool.input_schema
        unknown_root_keys = set(schema) - self._ROOT_SCHEMA_KEYS
        if unknown_root_keys:
            self._incompatible(tool)

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if (
            not isinstance(properties, Mapping)
            or isinstance(
                required,
                (str, bytes),
            )
            or not isinstance(required, Sequence)
        ):
            self._incompatible(tool)
        if any(not isinstance(name, str) for name in properties):
            self._incompatible(tool)
        if any(not isinstance(name, str) for name in required):
            self._incompatible(tool)
        if len(required) != len(set(required)):
            self._incompatible(tool)
        if set(required) - set(properties):
            self._incompatible(tool)
        additional = schema.get("additionalProperties", False)
        if not isinstance(additional, bool):
            self._incompatible(tool)

        required_set = set(required)
        parameters = [
            self._to_parameter(
                tool,
                name,
                parameter_schema,
                required=name in required_set,
            )
            for name, parameter_schema in properties.items()
        ]
        description = tool.description or tool.title or f"Remote MCP tool {tool.name}."
        try:
            return ToolSchema(tool.name, description, parameters)
        except ToolValidationError as error:
            raise MCPToolAdapterError(
                f"MCP tool is incompatible with the local tool layer: {tool.name}."
            ) from error

    def _to_parameter(
        self,
        tool: MCPTool,
        name: str,
        schema: object,
        *,
        required: bool,
    ) -> ToolParameter:
        if not isinstance(schema, Mapping):
            self._incompatible(tool)
        if set(schema) - self._PARAMETER_SCHEMA_KEYS:
            self._incompatible(tool)
        parameter_type = self._PARAMETER_TYPES.get(schema.get("type"))
        if parameter_type is None:
            self._incompatible(tool)
        description = schema.get("description") or schema.get("title")
        if description is None:
            description = f"MCP parameter {name}."
        if not isinstance(description, str) or not description.strip():
            self._incompatible(tool)
        try:
            return ToolParameter(
                name,
                parameter_type,
                description,
                required=required,
            )
        except ToolValidationError as error:
            raise MCPToolAdapterError(
                f"MCP tool is incompatible with the local tool layer: {tool.name}."
            ) from error

    @staticmethod
    def _incompatible(tool: MCPTool) -> None:
        raise MCPToolAdapterError(
            "MCP tool is incompatible with the local primitive "
            f"schema subset: {tool.name}."
        )

    def _make_handler(self, tool_name: str):
        def handler(**arguments: object) -> str:
            result = self._client.call_tool(tool_name, arguments)
            if isinstance(result, MCPContinuation):
                self._client.cancel_continuation(result)
                raise ToolHandlerError("Remote MCP tool requires explicit host input.")
            content = self._render_result(result)
            if result.is_error:
                raise ToolHandlerError(content or "Remote MCP tool returned an error.")
            return content

        return handler

    @staticmethod
    def _render_result(result: MCPToolResult) -> str:
        parts = [MCPToolAdapter._render_block(block) for block in result.content]
        if not parts and result.has_structured_content:
            return json.dumps(
                result.structured_content,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        return "\n".join(parts)

    @staticmethod
    def _render_block(block: MCPContentBlock) -> str:
        if block.type == "text":
            return block.data["text"]
        return json.dumps(
            block.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
