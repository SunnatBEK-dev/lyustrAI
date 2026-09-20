import pytest

from ai_sdk.tools import (
    ToolParameter,
    ToolParameterType,
    ToolSchema,
    ToolValidationError,
)


def make_schema():
    return ToolSchema(
        name="calculate_total",
        description="Calculate an order total.",
        parameters=[
            ToolParameter(
                name="quantity",
                type=ToolParameterType.INTEGER,
                description="Number of items.",
            ),
            ToolParameter(
                name="price",
                type=ToolParameterType.NUMBER,
                description="Price per item.",
            ),
            ToolParameter(
                name="include_tax",
                type=ToolParameterType.BOOLEAN,
                description="Whether tax is included.",
                required=False,
            ),
        ],
    )


def test_tool_schema_exports_provider_neutral_json_schema():
    schema = make_schema()

    assert schema.to_json_schema() == {
        "name": "calculate_total",
        "description": "Calculate an order total.",
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity": {
                    "type": "integer",
                    "description": "Number of items.",
                },
                "price": {
                    "type": "number",
                    "description": "Price per item.",
                },
                "include_tax": {
                    "type": "boolean",
                    "description": "Whether tax is included.",
                },
            },
            "required": ["quantity", "price"],
            "additionalProperties": False,
        },
    }


def test_tool_schema_validates_and_copies_arguments():
    arguments = {
        "quantity": 2,
        "price": 4.5,
        "include_tax": True,
    }

    validated = make_schema().validate_arguments(arguments)
    arguments["quantity"] = 99

    assert validated == {
        "quantity": 2,
        "price": 4.5,
        "include_tax": True,
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"price": 2.0}, "Missing"),
        (
            {"quantity": 1, "price": 2.0, "extra": 3},
            "Unknown",
        ),
        ({"quantity": True, "price": 2.0}, "integer"),
        ({"quantity": 1, "price": False}, "number"),
        ({"quantity": 1, "price": float("inf")}, "number"),
    ],
)
def test_tool_schema_rejects_invalid_arguments(
    arguments,
    message,
):
    with pytest.raises(ToolValidationError, match=message):
        make_schema().validate_arguments(arguments)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "empty"),
        ("invalid-name", "identifier"),
        ("class", "identifier"),
        ("a" * 65, "identifier"),
    ],
)
def test_tool_schema_rejects_invalid_name(name, message):
    with pytest.raises(ToolValidationError, match=message):
        ToolSchema(name, "Description")


def test_tool_schema_rejects_invalid_parameters():
    parameter = ToolParameter(
        "value",
        ToolParameterType.STRING,
        "A value.",
    )

    with pytest.raises(ToolValidationError, match="unique"):
        ToolSchema(
            "duplicate",
            "Description",
            [parameter, parameter],
        )

    with pytest.raises(ToolValidationError, match="description"):
        ToolParameter(
            "value",
            ToolParameterType.STRING,
            " ",
        )

    with pytest.raises(ToolValidationError, match="supported"):
        ToolParameter("value", "array", "A value.")


def test_tool_schema_rejects_non_object_arguments():
    with pytest.raises(ToolValidationError, match="object"):
        make_schema().validate_arguments([])

    with pytest.raises(ToolValidationError, match="strings"):
        make_schema().validate_arguments({1: "invalid"})


def test_tool_schema_rejects_invalid_runtime_objects():
    with pytest.raises(ToolValidationError, match="ToolParameter"):
        ToolSchema("tool", "Description", ["invalid"])

    with pytest.raises(ToolValidationError, match="boolean"):
        ToolParameter(
            "value",
            ToolParameterType.STRING,
            "A value.",
            required="yes",
        )
