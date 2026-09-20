import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ToolValidationError(ValueError):
    """Raised when a tool schema or call arguments are invalid."""


class ToolParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: ToolParameterType
    description: str
    required: bool = True

    def __post_init__(self) -> None:
        _validate_identifier(
            self.name,
            label="Tool parameter name",
        )

        if not isinstance(self.description, str) or not self.description.strip():
            raise ToolValidationError("Tool parameter description cannot be empty.")

        if not isinstance(self.type, ToolParameterType):
            raise ToolValidationError("Tool parameter type is not supported.")

        if not isinstance(self.required, bool):
            raise ToolValidationError("Tool parameter required flag must be boolean.")

    def accepts(self, value: object) -> bool:
        if self.type is ToolParameterType.STRING:
            return isinstance(value, str)

        if self.type is ToolParameterType.BOOLEAN:
            return isinstance(value, bool)

        if self.type is ToolParameterType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)

        if isinstance(value, int) and not isinstance(value, bool):
            return True

        return isinstance(value, float) and isfinite(value)


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Sequence[ToolParameter] = (),
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(
            self,
            "parameters",
            tuple(parameters),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_identifier(
            self.name,
            label="Tool name",
        )

        if not isinstance(self.description, str) or not self.description.strip():
            raise ToolValidationError("Tool description cannot be empty.")

        if any(
            not isinstance(parameter, ToolParameter) for parameter in self.parameters
        ):
            raise ToolValidationError("Tool parameters must use ToolParameter objects.")

        parameter_names = [parameter.name for parameter in self.parameters]

        if len(parameter_names) != len(set(parameter_names)):
            raise ToolValidationError("Tool parameter names must be unique.")

    def validate_arguments(
        self,
        arguments: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(arguments, Mapping):
            raise ToolValidationError("Tool arguments must be an object.")

        validated = dict(arguments)

        if any(not isinstance(name, str) for name in validated):
            raise ToolValidationError("Tool argument names must be strings.")

        parameter_by_name = {parameter.name: parameter for parameter in self.parameters}
        unknown_names = sorted(set(validated) - set(parameter_by_name))

        if unknown_names:
            raise ToolValidationError(
                "Unknown tool arguments: " + ", ".join(unknown_names) + "."
            )

        missing_names = [
            parameter.name
            for parameter in self.parameters
            if parameter.required and parameter.name not in validated
        ]

        if missing_names:
            raise ToolValidationError(
                "Missing required tool arguments: " + ", ".join(missing_names) + "."
            )

        for name, value in validated.items():
            parameter = parameter_by_name[name]

            if not parameter.accepts(value):
                raise ToolValidationError(
                    f"Tool argument '{name}' must be {parameter.type.value}."
                )

        return validated

    def to_json_schema(self) -> dict[str, object]:
        required = [
            parameter.name for parameter in self.parameters if parameter.required
        ]

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    parameter.name: {
                        "type": parameter.type.value,
                        "description": parameter.description,
                    }
                    for parameter in self.parameters
                },
                "required": required,
                "additionalProperties": False,
            },
        }


def _validate_identifier(name: str, *, label: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ToolValidationError(f"{label} cannot be empty.")

    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) is None or keyword.iskeyword(
        name
    ):
        raise ToolValidationError(f"{label} must be a valid identifier.")
