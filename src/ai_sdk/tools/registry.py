from collections.abc import Callable
from dataclasses import dataclass

from ai_sdk.tools.schema import ToolSchema

ToolHandler = Callable[..., object]


@dataclass(frozen=True)
class RegisteredTool:
    schema: ToolSchema
    handler: ToolHandler


class ToolRegistry:
    """Allow-list of schemas and handlers available for execution."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        schema: ToolSchema,
        handler: ToolHandler,
    ) -> None:
        if not isinstance(schema, ToolSchema):
            raise TypeError("Registered tool schema must be a ToolSchema.")

        if schema.name in self._tools:
            raise ValueError(f"Tool is already registered: {schema.name}")

        if not callable(handler):
            raise TypeError("Tool handler must be callable.")

        self._tools[schema.name] = RegisteredTool(
            schema=schema,
            handler=handler,
        )

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def schemas(self) -> list[ToolSchema]:
        return [tool.schema for tool in self._tools.values()]

    def provider_schemas(self) -> list[dict[str, object]]:
        return [schema.to_json_schema() for schema in self.schemas()]

    def count(self) -> int:
        return len(self._tools)
