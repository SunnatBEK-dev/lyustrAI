import json

import pytest

from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)

pytestmark = pytest.mark.integration


def test_schema_to_validated_tool_result_workflow():
    registry = ToolRegistry()
    schema = ToolSchema(
        "calculate_total",
        "Calculate a purchase total.",
        [
            ToolParameter(
                "quantity",
                ToolParameterType.INTEGER,
                "Number of items.",
            ),
            ToolParameter(
                "price",
                ToolParameterType.NUMBER,
                "Price per item.",
            ),
            ToolParameter(
                "include_tax",
                ToolParameterType.BOOLEAN,
                "Apply ten percent tax.",
                required=False,
            ),
        ],
    )

    def calculate_total(
        quantity,
        price,
        include_tax=False,
    ):
        subtotal = quantity * price
        return {
            "currency": "USD",
            "total": (subtotal * 1.1 if include_tax else subtotal),
        }

    registry.register(schema, calculate_total)
    executor = ToolExecutor(registry)
    result = executor.execute(
        ToolCall(
            id="call_total",
            name="calculate_total",
            arguments={
                "quantity": 3,
                "price": 4.0,
                "include_tax": True,
            },
        )
    )

    assert registry.provider_schemas()[0]["name"] == ("calculate_total")
    assert result.call_id == "call_total"
    assert result.name == "calculate_total"
    assert result.is_error is False
    assert json.loads(result.content) == {
        "currency": "USD",
        "total": pytest.approx(13.2),
    }
