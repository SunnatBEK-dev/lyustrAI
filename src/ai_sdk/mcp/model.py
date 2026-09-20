import json
import re
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

MCP_PROTOCOL_VERSION = "2026-07-28"

PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
CLIENT_CAPABILITIES_META_KEY = "io.modelcontextprotocol/clientCapabilities"

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_INPUT_REQUEST_METHODS = {
    "elicitation/create",
    "sampling/createMessage",
    "roots/list",
}


class MCPValidationError(ValueError):
    """Raised when a local MCP contract is invalid."""


class MCPConnectionState(str, Enum):
    NEW = "new"
    OPENING = "opening"
    OPEN = "open"
    FAILED = "failed"
    CLOSED = "closed"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MCPValidationError(f"MCP {field_name} cannot be empty.")
    return value


def _copy_mapping(
    value: object,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MCPValidationError(f"MCP {field_name} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise MCPValidationError(f"MCP {field_name} keys must be strings.")
    return deepcopy(dict(value))


def _copy_json_value(
    value: object,
    field_name: str,
) -> object:
    copied = deepcopy(value)
    try:
        json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise MCPValidationError(
            f"MCP {field_name} must be valid JSON data."
        ) from error
    return copied


def _optional_text(
    value: object,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _validate_cache_hints(
    ttl_ms: object,
    cache_scope: object,
) -> tuple[int | None, str | None]:
    if ttl_ms is not None and (
        not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms < 0
    ):
        raise MCPValidationError("MCP cache TTL must be a non-negative integer.")
    return ttl_ms, _optional_text(cache_scope, "cache scope")


def _validate_tool_name(name: object) -> str:
    if not isinstance(name, str) or _TOOL_NAME_PATTERN.fullmatch(name) is None:
        raise MCPValidationError(
            "MCP tool name must use 1-128 ASCII letters, "
            "digits, underscores, hyphens, or dots."
        )
    return name


def _copy_input_responses(
    value: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    if value is None:
        return {}
    copied = _copy_mapping(value, "input responses")
    normalized: dict[str, dict[str, object]] = {}
    for key, response in copied.items():
        if not key.strip() or not isinstance(response, Mapping):
            raise MCPValidationError(
                "MCP input responses must map non-empty keys to objects."
            )
        normalized[key] = _copy_mapping(
            response,
            f"input response {key}",
        )
        _copy_json_value(normalized[key], f"input response {key}")
    return normalized


def _optional_request_state(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MCPValidationError("MCP request state must be an opaque string.")
    return value


def _validate_uri(uri: object, field_name: str) -> str:
    validated_uri = _require_text(uri, field_name)
    try:
        uri_scheme = urlsplit(validated_uri).scheme
    except ValueError as error:
        raise MCPValidationError(f"MCP {field_name} is invalid.") from error
    if not uri_scheme:
        raise MCPValidationError(f"MCP {field_name} must include a scheme.")
    return validated_uri


@dataclass(frozen=True)
class MCPImplementation:
    name: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.name, "implementation name")
        _require_text(self.version, "implementation version")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, init=False)
class MCPRequestContext:
    protocol_version: str
    client_info: MCPImplementation
    client_capabilities: dict[str, object]

    def __init__(
        self,
        protocol_version: str,
        client_info: MCPImplementation,
        client_capabilities: Mapping[str, object] | None = None,
    ) -> None:
        _require_text(protocol_version, "protocol version")
        if not isinstance(client_info, MCPImplementation):
            raise MCPValidationError("MCP client info is invalid.")

        capabilities = _copy_mapping(
            {} if client_capabilities is None else client_capabilities,
            "client capabilities",
        )
        object.__setattr__(
            self,
            "protocol_version",
            protocol_version,
        )
        object.__setattr__(self, "client_info", client_info)
        object.__setattr__(
            self,
            "client_capabilities",
            capabilities,
        )

    def to_meta(self) -> dict[str, object]:
        return {
            PROTOCOL_VERSION_META_KEY: self.protocol_version,
            CLIENT_INFO_META_KEY: self.client_info.to_dict(),
            CLIENT_CAPABILITIES_META_KEY: deepcopy(self.client_capabilities),
        }


@dataclass(frozen=True, init=False)
class MCPServerCapabilities:
    tools: dict[str, object] | None
    resources: dict[str, object] | None

    def __init__(
        self,
        *,
        tools: Mapping[str, object] | None = None,
        resources: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "tools",
            None if tools is None else _copy_mapping(tools, "tools capability"),
        )
        object.__setattr__(
            self,
            "resources",
            None
            if resources is None
            else _copy_mapping(
                resources,
                "resources capability",
            ),
        )

    @property
    def supports_tools(self) -> bool:
        return self.tools is not None

    @property
    def supports_resources(self) -> bool:
        return self.resources is not None


@dataclass(frozen=True, init=False)
class MCPDiscoveryResult:
    supported_versions: tuple[str, ...]
    capabilities: MCPServerCapabilities
    server_info: MCPImplementation | None
    instructions: str | None
    ttl_ms: int | None
    cache_scope: str | None

    def __init__(
        self,
        supported_versions: Sequence[str],
        capabilities: MCPServerCapabilities,
        *,
        server_info: MCPImplementation | None = None,
        instructions: str | None = None,
        ttl_ms: int | None = None,
        cache_scope: str | None = None,
    ) -> None:
        if isinstance(supported_versions, (str, bytes)):
            raise MCPValidationError("MCP supported versions must be a sequence.")
        versions = tuple(supported_versions)
        if not versions or any(
            not isinstance(version, str) or not version.strip() for version in versions
        ):
            raise MCPValidationError(
                "MCP supported versions must contain non-empty strings."
            )
        if len(set(versions)) != len(versions):
            raise MCPValidationError("MCP supported versions must be unique.")
        if not isinstance(capabilities, MCPServerCapabilities):
            raise MCPValidationError("MCP server capabilities are invalid.")
        if server_info is not None and not isinstance(
            server_info,
            MCPImplementation,
        ):
            raise MCPValidationError("MCP server info is invalid.")
        validated_ttl, validated_scope = _validate_cache_hints(
            ttl_ms,
            cache_scope,
        )

        object.__setattr__(self, "supported_versions", versions)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "server_info", server_info)
        object.__setattr__(
            self,
            "instructions",
            _optional_text(instructions, "server instructions"),
        )
        object.__setattr__(self, "ttl_ms", validated_ttl)
        object.__setattr__(self, "cache_scope", validated_scope)


@dataclass(frozen=True, init=False)
class MCPTool:
    name: str
    input_schema: dict[str, object]
    title: str | None
    description: str | None

    def __init__(
        self,
        name: str,
        input_schema: Mapping[str, object],
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        validated_name = _validate_tool_name(name)
        schema = _copy_mapping(input_schema, "tool input schema")
        if schema.get("type") != "object":
            raise MCPValidationError("MCP tool input schema root type must be object.")

        object.__setattr__(self, "name", validated_name)
        object.__setattr__(self, "input_schema", schema)
        object.__setattr__(
            self,
            "title",
            _optional_text(title, "tool title"),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(description, "tool description"),
        )


@dataclass(frozen=True, init=False)
class MCPResource:
    uri: str
    name: str
    title: str | None
    description: str | None
    mime_type: str | None
    size: int | None

    def __init__(
        self,
        uri: str,
        name: str,
        *,
        title: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        size: int | None = None,
    ) -> None:
        validated_uri = _validate_uri(uri, "resource URI")
        if size is not None and (
            not isinstance(size, int) or isinstance(size, bool) or size < 0
        ):
            raise MCPValidationError(
                "MCP resource size must be a non-negative integer."
            )

        object.__setattr__(self, "uri", validated_uri)
        object.__setattr__(self, "name", _require_text(name, "resource name"))
        object.__setattr__(
            self,
            "title",
            _optional_text(title, "resource title"),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(description, "resource description"),
        )
        object.__setattr__(
            self,
            "mime_type",
            _optional_text(mime_type, "resource MIME type"),
        )
        object.__setattr__(self, "size", size)


@dataclass(frozen=True, init=False)
class MCPToolPage:
    tools: tuple[MCPTool, ...]
    next_cursor: str | None
    ttl_ms: int | None
    cache_scope: str | None

    def __init__(
        self,
        tools: Sequence[MCPTool],
        *,
        next_cursor: str | None = None,
        ttl_ms: int | None = None,
        cache_scope: str | None = None,
    ) -> None:
        normalized = tuple(tools)
        if any(not isinstance(tool, MCPTool) for tool in normalized):
            raise MCPValidationError("MCP tool page contains an invalid tool.")
        validated_ttl, validated_scope = _validate_cache_hints(
            ttl_ms,
            cache_scope,
        )
        object.__setattr__(self, "tools", normalized)
        object.__setattr__(
            self,
            "next_cursor",
            _optional_text(next_cursor, "next cursor"),
        )
        object.__setattr__(self, "ttl_ms", validated_ttl)
        object.__setattr__(self, "cache_scope", validated_scope)


@dataclass(frozen=True, init=False)
class MCPResourcePage:
    resources: tuple[MCPResource, ...]
    next_cursor: str | None
    ttl_ms: int | None
    cache_scope: str | None

    def __init__(
        self,
        resources: Sequence[MCPResource],
        *,
        next_cursor: str | None = None,
        ttl_ms: int | None = None,
        cache_scope: str | None = None,
    ) -> None:
        normalized = tuple(resources)
        if any(not isinstance(resource, MCPResource) for resource in normalized):
            raise MCPValidationError("MCP resource page contains an invalid resource.")
        validated_ttl, validated_scope = _validate_cache_hints(
            ttl_ms,
            cache_scope,
        )
        object.__setattr__(self, "resources", normalized)
        object.__setattr__(
            self,
            "next_cursor",
            _optional_text(next_cursor, "next cursor"),
        )
        object.__setattr__(self, "ttl_ms", validated_ttl)
        object.__setattr__(self, "cache_scope", validated_scope)


@dataclass(frozen=True, init=False)
class MCPInputRequest:
    method: str
    params: dict[str, object]

    def __init__(
        self,
        method: str,
        params: Mapping[str, object],
    ) -> None:
        if method not in _INPUT_REQUEST_METHODS:
            raise MCPValidationError("MCP input request method is unsupported.")
        normalized_params = _copy_mapping(
            params,
            "input request parameters",
        )
        _copy_json_value(
            normalized_params,
            "input request parameters",
        )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "params", normalized_params)


@dataclass(frozen=True, init=False)
class MCPInputRequiredResult:
    input_requests: dict[str, MCPInputRequest]
    request_state: str | None

    def __init__(
        self,
        input_requests: Mapping[str, MCPInputRequest] | None = None,
        *,
        request_state: str | None = None,
    ) -> None:
        normalized: dict[str, MCPInputRequest] = {}
        if input_requests is not None:
            if not isinstance(input_requests, Mapping):
                raise MCPValidationError("MCP input requests must be an object.")
            for key, request in input_requests.items():
                if (
                    not isinstance(key, str)
                    or not key.strip()
                    or not isinstance(request, MCPInputRequest)
                ):
                    raise MCPValidationError(
                        "MCP input requests must map non-empty keys to requests."
                    )
                normalized[key] = request
        object.__setattr__(self, "input_requests", normalized)
        object.__setattr__(
            self,
            "request_state",
            _optional_request_state(request_state),
        )


@dataclass(frozen=True, init=False)
class MCPContinuation:
    continuation_id: str
    operation: str
    input_requests: dict[str, MCPInputRequest]
    round: int

    def __init__(
        self,
        continuation_id: str,
        operation: str,
        input_requests: Mapping[str, MCPInputRequest],
        *,
        round: int,
    ) -> None:
        validated_id = _require_text(
            continuation_id,
            "continuation ID",
        )
        if operation not in {"tools/call", "resources/read"}:
            raise MCPValidationError("MCP continuation operation is unsupported.")
        if not isinstance(round, int) or isinstance(round, bool) or round <= 0:
            raise MCPValidationError("MCP continuation round must be positive.")
        normalized = MCPInputRequiredResult(input_requests).input_requests
        object.__setattr__(self, "continuation_id", validated_id)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "input_requests", normalized)
        object.__setattr__(self, "round", round)


@dataclass(frozen=True, init=False)
class MCPToolRequest:
    name: str
    arguments: dict[str, object]
    input_responses: dict[str, dict[str, object]]
    request_state: str | None

    def __init__(
        self,
        name: str,
        arguments: Mapping[str, object] | None = None,
        *,
        input_responses: Mapping[
            str,
            Mapping[str, object],
        ]
        | None = None,
        request_state: str | None = None,
    ) -> None:
        validated_arguments = _copy_mapping(
            {} if arguments is None else arguments,
            "tool arguments",
        )
        _copy_json_value(validated_arguments, "tool arguments")
        object.__setattr__(self, "name", _validate_tool_name(name))
        object.__setattr__(self, "arguments", validated_arguments)
        object.__setattr__(
            self,
            "input_responses",
            _copy_input_responses(input_responses),
        )
        object.__setattr__(
            self,
            "request_state",
            _optional_request_state(request_state),
        )


@dataclass(frozen=True, init=False)
class MCPContentBlock:
    type: str
    data: dict[str, object]

    def __init__(
        self,
        type: str,
        data: Mapping[str, object],
    ) -> None:
        validated_type = _require_text(type, "content type")
        validated_data = _copy_mapping(data, "content data")
        if "type" in validated_data:
            raise MCPValidationError("MCP content data cannot redefine its type.")
        _copy_json_value(validated_data, "content data")
        if validated_type == "text" and not isinstance(
            validated_data.get("text"),
            str,
        ):
            raise MCPValidationError("MCP text content must contain text.")

        object.__setattr__(self, "type", validated_type)
        object.__setattr__(self, "data", validated_data)

    @classmethod
    def text(cls, text: str) -> "MCPContentBlock":
        if not isinstance(text, str):
            raise MCPValidationError("MCP text content must be a string.")
        return cls("text", {"text": text})

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, **deepcopy(self.data)}


_MISSING_STRUCTURED_CONTENT = object()


@dataclass(frozen=True, init=False)
class MCPToolResult:
    content: tuple[MCPContentBlock, ...]
    structured_content: object
    has_structured_content: bool
    is_error: bool

    def __init__(
        self,
        content: Sequence[MCPContentBlock] = (),
        *,
        structured_content: object = _MISSING_STRUCTURED_CONTENT,
        is_error: bool = False,
    ) -> None:
        normalized = tuple(content)
        if any(not isinstance(block, MCPContentBlock) for block in normalized):
            raise MCPValidationError(
                "MCP tool result contains an invalid content block."
            )
        has_structured = structured_content is not _MISSING_STRUCTURED_CONTENT
        if not normalized and not has_structured:
            raise MCPValidationError("MCP tool result must contain content.")
        if not isinstance(is_error, bool):
            raise MCPValidationError("MCP tool result error flag must be boolean.")
        copied_structured = (
            None
            if not has_structured
            else _copy_json_value(
                structured_content,
                "structured tool result",
            )
        )

        object.__setattr__(self, "content", normalized)
        object.__setattr__(
            self,
            "structured_content",
            copied_structured,
        )
        object.__setattr__(
            self,
            "has_structured_content",
            has_structured,
        )
        object.__setattr__(self, "is_error", is_error)


@dataclass(frozen=True, init=False)
class MCPResourceReadRequest:
    uri: str
    input_responses: dict[str, dict[str, object]]
    request_state: str | None

    def __init__(
        self,
        uri: str,
        *,
        input_responses: Mapping[
            str,
            Mapping[str, object],
        ]
        | None = None,
        request_state: str | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "uri",
            _validate_uri(uri, "resource URI"),
        )
        object.__setattr__(
            self,
            "input_responses",
            _copy_input_responses(input_responses),
        )
        object.__setattr__(
            self,
            "request_state",
            _optional_request_state(request_state),
        )


@dataclass(frozen=True, init=False)
class MCPResourceContent:
    uri: str
    mime_type: str | None
    text: str | None
    blob: str | None
    annotations: dict[str, object] | None

    def __init__(
        self,
        uri: str,
        *,
        mime_type: str | None = None,
        text: str | None = None,
        blob: str | None = None,
        annotations: Mapping[str, object] | None = None,
    ) -> None:
        if (text is None) == (blob is None):
            raise MCPValidationError(
                "MCP resource content must contain exactly one of text or blob."
            )
        if text is not None and not isinstance(text, str):
            raise MCPValidationError("MCP resource text must be a string.")
        if blob is not None:
            if not isinstance(blob, str):
                raise MCPValidationError("MCP resource blob must be a base64 string.")
            try:
                b64decode(blob, validate=True)
            except (BinasciiError, ValueError) as error:
                raise MCPValidationError(
                    "MCP resource blob must be valid base64."
                ) from error
        validated_annotations = (
            None
            if annotations is None
            else _copy_mapping(
                annotations,
                "resource annotations",
            )
        )
        if validated_annotations is not None:
            _copy_json_value(
                validated_annotations,
                "resource annotations",
            )

        object.__setattr__(
            self,
            "uri",
            _validate_uri(uri, "resource content URI"),
        )
        object.__setattr__(
            self,
            "mime_type",
            _optional_text(mime_type, "resource MIME type"),
        )
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "blob", blob)
        object.__setattr__(
            self,
            "annotations",
            validated_annotations,
        )


@dataclass(frozen=True, init=False)
class MCPResourceReadResult:
    contents: tuple[MCPResourceContent, ...]
    ttl_ms: int | None
    cache_scope: str | None

    def __init__(
        self,
        contents: Sequence[MCPResourceContent],
        *,
        ttl_ms: int | None = None,
        cache_scope: str | None = None,
    ) -> None:
        normalized = tuple(contents)
        if not normalized or any(
            not isinstance(content, MCPResourceContent) for content in normalized
        ):
            raise MCPValidationError(
                "MCP resource read result must contain valid content."
            )
        validated_ttl, validated_scope = _validate_cache_hints(
            ttl_ms,
            cache_scope,
        )

        object.__setattr__(self, "contents", normalized)
        object.__setattr__(self, "ttl_ms", validated_ttl)
        object.__setattr__(self, "cache_scope", validated_scope)
