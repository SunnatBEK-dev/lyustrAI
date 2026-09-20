import json

from ai_sdk.observability import (
    TraceCategory,
    Tracer,
    trace_span,
)
from ai_sdk.tools.model import (
    ToolCall,
    ToolHandlerError,
    ToolResult,
)
from ai_sdk.tools.registry import ToolRegistry
from ai_sdk.tools.schema import ToolValidationError


class ToolExecutor:
    """Validate and execute only handlers in an explicit registry."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tracer: Tracer | None = None,
    ) -> None:
        if tracer is not None and not isinstance(tracer, Tracer):
            raise TypeError("Tool tracer must be a Tracer.")
        self.registry = registry
        self.tracer = tracer

    def execute(
        self,
        call: ToolCall,
        *,
        tracer: Tracer | None = None,
    ) -> ToolResult:
        if tracer is not None and not isinstance(tracer, Tracer):
            raise TypeError("Tool tracer must be a Tracer.")
        active_tracer = tracer or self.tracer
        with trace_span(
            active_tracer,
            "tool.execute",
            TraceCategory.TOOL,
            {"tool.name": call.name},
        ) as span:
            result = self._execute(call)
            if span is not None:
                span.set_attribute("tool.is_error", result.is_error)
                if result.is_error:
                    span.set_error("ToolExecutionError")
            return result

    def _execute(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.name)

        if tool is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )

        try:
            arguments = tool.schema.validate_arguments(call.arguments)
            output = tool.handler(**arguments)
            content = self._serialize_output(output)
        except (ToolValidationError, ToolHandlerError) as error:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=(
                    error.content if isinstance(error, ToolHandlerError) else str(error)
                ),
                is_error=True,
            )
        except Exception as error:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=(f"Tool execution failed: {type(error).__name__}"),
                is_error=True,
            )

        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=content,
        )

    @staticmethod
    def _serialize_output(output: object) -> str:
        if isinstance(output, str):
            return output

        return json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
