import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_SAFE_COUNT_SUFFIXES = (
    "_count",
    ".count",
    "_length",
    ".length",
    "_size",
    ".size",
)
_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "arguments",
    "authorization",
    "content",
    "credential",
    "input_responses",
    "password",
    "prompt",
    "request_state",
    "requeststate",
    "secret",
    "token",
}
_MAX_ATTRIBUTES = 32
_MAX_ATTRIBUTE_KEY_LENGTH = 64
_MAX_ATTRIBUTE_STRING_LENGTH = 128
_REDACTED = "[REDACTED]"


TraceAttributeValue = str | int | float | bool


class TraceValidationError(ValueError):
    """Raised when trace metadata does not meet the safe contract."""


class TraceCategory(str, Enum):
    WORKFLOW = "workflow"
    LLM = "llm"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    AGENT = "agent"
    MCP = "mcp"


class TraceStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


def sanitize_trace_attributes(
    attributes: Mapping[str, object] | None,
) -> dict[str, TraceAttributeValue]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping):
        raise TraceValidationError("Trace attributes must be an object.")
    if len(attributes) > _MAX_ATTRIBUTES:
        raise TraceValidationError("Trace attributes exceed the configured limit.")

    sanitized: dict[str, TraceAttributeValue] = {}
    for key, value in attributes.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or len(key) > _MAX_ATTRIBUTE_KEY_LENGTH
        ):
            raise TraceValidationError(
                "Trace attribute keys must be short non-empty strings."
            )
        if _is_sensitive_key(key, value) or _looks_like_secret(value):
            sanitized[key] = _REDACTED
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise TraceValidationError("Trace attribute values must be scalar.")
        if isinstance(value, float) and not math.isfinite(value):
            raise TraceValidationError("Trace numeric attributes must be finite.")
        if isinstance(value, str):
            value = _truncate(value)
        sanitized[key] = value
    return sanitized


def _is_sensitive_key(key: str, value: object) -> bool:
    normalized = key.casefold().replace("-", "_")
    if (
        normalized.endswith(_SAFE_COUNT_SUFFIXES)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return False
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _looks_like_secret(value: object) -> bool:
    if not isinstance(value, str):
        return False
    folded = value.casefold().lstrip()
    return folded.startswith("bearer ") or folded.startswith("sk-")


def _truncate(value: str) -> str:
    if len(value) <= _MAX_ATTRIBUTE_STRING_LENGTH:
        return value
    return value[: _MAX_ATTRIBUTE_STRING_LENGTH - 1] + "…"


def _validate_identifier(
    value: object,
    pattern: re.Pattern[str],
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or pattern.fullmatch(value) is None
        or set(value) == {"0"}
    ):
        raise TraceValidationError(f"Trace {label} is invalid.")
    return value


def _validate_trace_id(value: object) -> str:
    return _validate_identifier(value, _TRACE_ID_PATTERN, "ID")


def _validate_span_id(value: object) -> str:
    return _validate_identifier(value, _SPAN_ID_PATTERN, "span ID")


def _validate_span_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise TraceValidationError("Trace span name must be a short non-empty string.")
    return value.strip()


def _validate_error_type(value: object) -> str:
    if not isinstance(value, str) or _ERROR_TYPE_PATTERN.fullmatch(value) is None:
        raise TraceValidationError("Trace error type is invalid.")
    return value


@dataclass(frozen=True, init=False)
class TraceRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    category: TraceCategory
    started_at_ns: int
    duration_ns: int
    status: TraceStatus
    attributes: Mapping[str, TraceAttributeValue]
    error_type: str | None

    def __init__(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        name: str,
        category: TraceCategory,
        started_at_ns: int,
        duration_ns: int,
        status: TraceStatus,
        attributes: Mapping[str, object] | None = None,
        error_type: str | None = None,
    ) -> None:
        validated_trace_id = _validate_trace_id(trace_id)
        validated_span_id = _validate_span_id(span_id)
        validated_parent = (
            None if parent_span_id is None else _validate_span_id(parent_span_id)
        )
        validated_name = _validate_span_name(name)
        if not isinstance(category, TraceCategory):
            raise TraceValidationError("Trace category is invalid.")
        if (
            not isinstance(started_at_ns, int)
            or isinstance(started_at_ns, bool)
            or started_at_ns < 0
            or not isinstance(duration_ns, int)
            or isinstance(duration_ns, bool)
            or duration_ns < 0
        ):
            raise TraceValidationError(
                "Trace timing values must be non-negative integers."
            )
        if not isinstance(status, TraceStatus):
            raise TraceValidationError("Trace status is invalid.")
        if status is TraceStatus.ERROR:
            try:
                validated_error_type = _validate_error_type(error_type)
            except TraceValidationError as error:
                raise TraceValidationError(
                    "Failed trace spans require an error type."
                ) from error
        elif error_type is not None:
            raise TraceValidationError(
                "Successful trace spans cannot contain an error type."
            )
        else:
            validated_error_type = None

        object.__setattr__(self, "trace_id", validated_trace_id)
        object.__setattr__(self, "span_id", validated_span_id)
        object.__setattr__(self, "parent_span_id", validated_parent)
        object.__setattr__(self, "name", validated_name)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "started_at_ns", started_at_ns)
        object.__setattr__(self, "duration_ns", duration_ns)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(sanitize_trace_attributes(attributes)),
        )
        object.__setattr__(self, "error_type", validated_error_type)

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "category": self.category.value,
            "started_at_ns": self.started_at_ns,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": dict(self.attributes),
            "error_type": self.error_type,
        }
