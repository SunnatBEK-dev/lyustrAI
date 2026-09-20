import json
import re
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from threading import Lock
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    OpenerDirector,
    Request,
    build_opener,
)

from ai_sdk.mcp.model import (
    MCPContentBlock,
    MCPDiscoveryResult,
    MCPImplementation,
    MCPInputRequest,
    MCPInputRequiredResult,
    MCPRequestContext,
    MCPResource,
    MCPResourceContent,
    MCPResourcePage,
    MCPResourceReadRequest,
    MCPResourceReadResult,
    MCPServerCapabilities,
    MCPTool,
    MCPToolPage,
    MCPToolRequest,
    MCPToolResult,
)
from ai_sdk.mcp.transport import (
    BaseMCPTransport,
    MCPTransportResponseError,
)

AuthorizationProvider = Callable[[], str | None]

_HEADER_TOKEN_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_BASE64_SENTINEL_PATTERN = re.compile(
    r"^=\?base64\?.*\?=$",
    re.DOTALL,
)
_MISSING = object()
_MAX_SAFE_INTEGER = (2**53) - 1


class MCPHTTPError(RuntimeError):
    """Raised when Streamable HTTP framing or content is invalid."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object):
        return None


class StreamableHTTPTransport(BaseMCPTransport):
    """Stateless MCP 2026-07-28 Streamable HTTP transport."""

    def __init__(
        self,
        endpoint: str,
        *,
        authorization_provider: AuthorizationProvider | None = None,
        max_response_bytes: int = 10 * 1024 * 1024,
        opener: OpenerDirector | None = None,
    ) -> None:
        self._endpoint = self._validate_endpoint(endpoint)
        if authorization_provider is not None and not callable(authorization_provider):
            raise MCPHTTPError("MCP authorization provider must be callable.")
        if (
            not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or max_response_bytes <= 0
        ):
            raise MCPHTTPError("MCP maximum response size must be greater than zero.")
        if opener is not None and not hasattr(opener, "open"):
            raise MCPHTTPError("MCP HTTP opener is invalid.")

        self._authorization_provider = authorization_provider
        self._max_response_bytes = max_response_bytes
        self._opener = opener or build_opener(_NoRedirectHandler())
        self._request_ids = iter(range(1, 2**63))
        self._request_id_lock = Lock()
        self._is_open = False
        self._tool_header_specs: dict[
            str,
            tuple[tuple[str, tuple[str, ...], str], ...],
        ] = {}

    def open(self, *, timeout_seconds: float) -> None:
        if self._is_open:
            raise MCPHTTPError("MCP HTTP transport is already open.")
        self._is_open = True

    def discover(
        self,
        context: MCPRequestContext,
        *,
        timeout_seconds: float,
    ) -> MCPDiscoveryResult:
        result = self._post(
            context,
            method="server/discover",
            params={},
            timeout_seconds=timeout_seconds,
        )
        return self._parse_discovery(result)

    def list_tools(
        self,
        context: MCPRequestContext,
        *,
        cursor: str | None,
        timeout_seconds: float,
    ) -> MCPToolPage:
        params = {} if cursor is None else {"cursor": cursor}
        result = self._post(
            context,
            method="tools/list",
            params=params,
            timeout_seconds=timeout_seconds,
        )
        return self._parse_tool_page(result)

    def list_resources(
        self,
        context: MCPRequestContext,
        *,
        cursor: str | None,
        timeout_seconds: float,
    ) -> MCPResourcePage:
        params = {} if cursor is None else {"cursor": cursor}
        result = self._post(
            context,
            method="resources/list",
            params=params,
            timeout_seconds=timeout_seconds,
        )
        return self._parse_resource_page(result)

    def call_tool(
        self,
        context: MCPRequestContext,
        request: MCPToolRequest,
        *,
        timeout_seconds: float,
    ) -> MCPToolResult | MCPInputRequiredResult:
        params: dict[str, object] = {
            "name": request.name,
            "arguments": request.arguments,
        }
        self._add_retry_params(params, request)
        result = self._post(
            context,
            method="tools/call",
            name=request.name,
            params=params,
            timeout_seconds=timeout_seconds,
            extra_headers=self._tool_headers(request),
            allow_input_required=True,
        )
        if isinstance(result, MCPInputRequiredResult):
            return result
        return self._parse_tool_result(result)

    def read_resource(
        self,
        context: MCPRequestContext,
        request: MCPResourceReadRequest,
        *,
        timeout_seconds: float,
    ) -> MCPResourceReadResult | MCPInputRequiredResult:
        params: dict[str, object] = {"uri": request.uri}
        self._add_retry_params(params, request)
        result = self._post(
            context,
            method="resources/read",
            name=request.uri,
            params=params,
            timeout_seconds=timeout_seconds,
            allow_input_required=True,
        )
        if isinstance(result, MCPInputRequiredResult):
            return result
        return self._parse_resource_result(result)

    def close(self) -> None:
        self._is_open = False

    @staticmethod
    def _add_retry_params(
        params: dict[str, object],
        request: MCPToolRequest | MCPResourceReadRequest,
    ) -> None:
        if request.input_responses:
            params["inputResponses"] = request.input_responses
        if request.request_state is not None:
            params["requestState"] = request.request_state

    def _post(
        self,
        context: MCPRequestContext,
        *,
        method: str,
        params: Mapping[str, object],
        timeout_seconds: float,
        name: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_input_required: bool = False,
    ) -> Mapping[str, object] | MCPInputRequiredResult:
        self._require_open()
        request_id = self._next_request_id()
        body_params = dict(params)
        body_params["_meta"] = context.to_meta()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": body_params,
        }
        try:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise MCPHTTPError("MCP HTTP request body is not valid JSON.") from error

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": context.protocol_version,
            "Mcp-Method": method,
        }
        if name is not None:
            headers["Mcp-Name"] = self._encode_header_value(name)
        if extra_headers:
            headers.update(extra_headers)
        authorization = self._authorization()
        if authorization is not None:
            headers["Authorization"] = authorization

        http_request = Request(
            self._endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            response = self._opener.open(
                http_request,
                timeout=timeout_seconds,
            )
        except HTTPError as error:
            with error:
                return self._decode_response(
                    error,
                    request_id=request_id,
                    status=error.code,
                    allow_input_required=allow_input_required,
                )
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError("MCP HTTP request timed out.") from error
            raise MCPHTTPError("MCP HTTP request failed: URLError") from error
        except TimeoutError:
            raise
        except Exception as error:
            raise MCPHTTPError(
                f"MCP HTTP request failed: {type(error).__name__}"
            ) from error

        with response:
            status = response.getcode()
            return self._decode_response(
                response,
                request_id=request_id,
                status=status,
                allow_input_required=allow_input_required,
            )

    def _decode_response(
        self,
        response: BinaryIO,
        *,
        request_id: int,
        status: int,
        allow_input_required: bool,
    ) -> Mapping[str, object] | MCPInputRequiredResult:
        raw = response.read(self._max_response_bytes + 1)
        if len(raw) > self._max_response_bytes:
            raise MCPHTTPError("MCP HTTP response exceeded the configured size limit.")

        content_type = self._response_content_type(response)
        if status != 200:
            if content_type == "application/json" and raw:
                try:
                    message = self._decode_json(raw)
                    self._parse_envelope(
                        message,
                        request_id,
                        allow_input_required=allow_input_required,
                    )
                except MCPTransportResponseError:
                    raise
                except Exception:
                    pass
            raise MCPHTTPError(f"MCP HTTP request failed with status {status}.")

        if content_type == "application/json":
            message = self._decode_json(raw)
            return self._parse_envelope(
                message,
                request_id,
                allow_input_required=allow_input_required,
            )
        if content_type == "text/event-stream":
            return self._decode_sse(
                raw,
                request_id,
                allow_input_required=allow_input_required,
            )
        raise MCPHTTPError("MCP HTTP response content type is unsupported.")

    @staticmethod
    def _decode_json(raw: bytes) -> object:
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MCPHTTPError("MCP HTTP response is not valid UTF-8 JSON.") from error

    def _decode_sse(
        self,
        raw: bytes,
        request_id: int,
        *,
        allow_input_required: bool,
    ) -> Mapping[str, object] | MCPInputRequiredResult:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MCPHTTPError("MCP SSE response is not valid UTF-8.") from error

        final_result: Mapping[str, object] | MCPInputRequiredResult | None = None
        for data in self._sse_data_events(text):
            message = self._decode_json(data.encode("utf-8"))
            if final_result is not None:
                raise MCPHTTPError("MCP SSE stream continued after its final response.")
            if self._is_notification(message):
                continue
            if self._is_server_request(message):
                raise MCPHTTPError("MCP SSE stream contained a server request.")
            final_result = self._parse_envelope(
                message,
                request_id,
                allow_input_required=allow_input_required,
            )
        if final_result is None:
            raise MCPHTTPError("MCP SSE stream ended without a final response.")
        return final_result

    @staticmethod
    def _sse_data_events(text: str) -> tuple[str, ...]:
        events: list[str] = []
        data_lines: list[str] = []
        for line in text.splitlines():
            if not line:
                if data_lines:
                    events.append("\n".join(data_lines))
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
        if data_lines:
            events.append("\n".join(data_lines))
        return tuple(events)

    @staticmethod
    def _is_notification(message: object) -> bool:
        return (
            isinstance(message, Mapping)
            and message.get("jsonrpc") == "2.0"
            and isinstance(message.get("method"), str)
            and "id" not in message
        )

    @staticmethod
    def _is_server_request(message: object) -> bool:
        return (
            isinstance(message, Mapping)
            and isinstance(message.get("method"), str)
            and "id" in message
        )

    @staticmethod
    def _parse_envelope(
        message: object,
        request_id: int,
        *,
        allow_input_required: bool,
    ) -> Mapping[str, object] | MCPInputRequiredResult:
        if not isinstance(message, Mapping):
            raise MCPHTTPError("MCP JSON-RPC response must be an object.")
        if message.get("jsonrpc") != "2.0":
            raise MCPHTTPError("MCP JSON-RPC version is invalid.")
        if message.get("id") != request_id or isinstance(
            message.get("id"),
            bool,
        ):
            raise MCPHTTPError("MCP JSON-RPC response ID does not match the request.")
        has_result = "result" in message
        has_error = "error" in message
        if has_result == has_error:
            raise MCPHTTPError("MCP JSON-RPC response must contain result or error.")
        if has_error:
            error = message["error"]
            if not isinstance(error, Mapping):
                raise MCPHTTPError("MCP JSON-RPC error is invalid.")
            code = error.get("code")
            remote_message = error.get("message")
            if (
                not isinstance(code, int)
                or isinstance(code, bool)
                or not isinstance(remote_message, str)
                or not remote_message.strip()
            ):
                raise MCPHTTPError("MCP JSON-RPC error is invalid.")
            raise MCPTransportResponseError(code, remote_message)

        result = message["result"]
        if not isinstance(result, Mapping):
            raise MCPHTTPError("MCP JSON-RPC result must be an object.")
        result_type = result.get("resultType", "complete")
        if result_type == "input_required":
            if not allow_input_required:
                raise MCPHTTPError(
                    "MCP input_required result is unsupported for this operation."
                )
            return StreamableHTTPTransport._parse_input_required(result)
        if result_type != "complete":
            raise MCPHTTPError(f"MCP result type is unsupported: {result_type}.")
        return result

    @staticmethod
    def _parse_input_required(
        result: Mapping[str, object],
    ) -> MCPInputRequiredResult:
        raw_requests = result.get("inputRequests")
        requests: dict[str, MCPInputRequest] = {}
        if "inputRequests" in result:
            if not isinstance(raw_requests, Mapping):
                raise MCPHTTPError("MCP inputRequests must be an object.")
            for key, raw_request in raw_requests.items():
                if (
                    not isinstance(key, str)
                    or not key.strip()
                    or not isinstance(raw_request, Mapping)
                    or not isinstance(raw_request.get("params"), Mapping)
                ):
                    raise MCPHTTPError("MCP inputRequests contains an invalid request.")
                try:
                    requests[key] = MCPInputRequest(
                        raw_request.get("method"),
                        raw_request["params"],
                    )
                except (TypeError, ValueError) as error:
                    raise MCPHTTPError(
                        "MCP inputRequests contains an invalid request."
                    ) from error
        request_state = result.get("requestState")
        if "requestState" in result and not isinstance(
            request_state,
            str,
        ):
            raise MCPHTTPError("MCP requestState must be an opaque string.")
        return MCPInputRequiredResult(
            requests,
            request_state=request_state,
        )

    @staticmethod
    def _parse_discovery(
        result: Mapping[str, object],
    ) -> MCPDiscoveryResult:
        supported = result.get("supportedVersions")
        capabilities = result.get("capabilities")
        if (
            not isinstance(supported, Sequence)
            or isinstance(
                supported,
                (str, bytes),
            )
            or not isinstance(capabilities, Mapping)
        ):
            raise MCPHTTPError("MCP discovery result is invalid.")
        tools = StreamableHTTPTransport._capability(
            capabilities,
            "tools",
        )
        resources = StreamableHTTPTransport._capability(
            capabilities,
            "resources",
        )
        return MCPDiscoveryResult(
            supported,
            MCPServerCapabilities(
                tools=tools,
                resources=resources,
            ),
            server_info=StreamableHTTPTransport._server_info(result),
            instructions=result.get("instructions"),
            ttl_ms=result.get("ttlMs"),
            cache_scope=result.get("cacheScope"),
        )

    @staticmethod
    def _capability(
        capabilities: Mapping[str, object],
        name: str,
    ) -> Mapping[str, object] | None:
        if name not in capabilities:
            return None
        value = capabilities[name]
        if not isinstance(value, Mapping):
            raise MCPHTTPError(f"MCP {name} capability is invalid.")
        return value

    @staticmethod
    def _server_info(
        result: Mapping[str, object],
    ) -> MCPImplementation | None:
        meta = result.get("_meta")
        if not isinstance(meta, Mapping):
            return None
        raw = meta.get("io.modelcontextprotocol/serverInfo")
        if not isinstance(raw, Mapping):
            return None
        try:
            return MCPImplementation(raw.get("name"), raw.get("version"))
        except (TypeError, ValueError):
            return None

    def _parse_tool_page(
        self,
        result: Mapping[str, object],
    ) -> MCPToolPage:
        raw_tools = result.get("tools")
        if not isinstance(raw_tools, Sequence) or isinstance(
            raw_tools,
            (str, bytes),
        ):
            raise MCPHTTPError("MCP tools/list result is invalid.")

        tools: list[MCPTool] = []
        for raw in raw_tools:
            tool = self._parse_tool(raw)
            try:
                header_specs = self._extract_tool_headers(tool)
            except MCPHTTPError:
                self._tool_header_specs.pop(tool.name, None)
                continue
            self._tool_header_specs[tool.name] = header_specs
            tools.append(tool)
        return MCPToolPage(
            tools,
            next_cursor=result.get("nextCursor"),
            ttl_ms=result.get("ttlMs"),
            cache_scope=result.get("cacheScope"),
        )

    @staticmethod
    def _parse_tool(raw: object) -> MCPTool:
        if not isinstance(raw, Mapping):
            raise MCPHTTPError("MCP tools/list contains an invalid tool.")
        schema = raw.get("inputSchema")
        if not isinstance(schema, Mapping):
            raise MCPHTTPError("MCP tool input schema is invalid.")
        return MCPTool(
            raw.get("name"),
            schema,
            title=raw.get("title"),
            description=raw.get("description"),
        )

    @staticmethod
    def _parse_resource_page(
        result: Mapping[str, object],
    ) -> MCPResourcePage:
        raw_resources = result.get("resources")
        if not isinstance(raw_resources, Sequence) or isinstance(
            raw_resources,
            (str, bytes),
        ):
            raise MCPHTTPError("MCP resources/list result is invalid.")
        return MCPResourcePage(
            [StreamableHTTPTransport._parse_resource(raw) for raw in raw_resources],
            next_cursor=result.get("nextCursor"),
            ttl_ms=result.get("ttlMs"),
            cache_scope=result.get("cacheScope"),
        )

    @staticmethod
    def _parse_resource(raw: object) -> MCPResource:
        if not isinstance(raw, Mapping):
            raise MCPHTTPError("MCP resources/list contains an invalid resource.")
        return MCPResource(
            raw.get("uri"),
            raw.get("name"),
            title=raw.get("title"),
            description=raw.get("description"),
            mime_type=raw.get("mimeType"),
            size=raw.get("size"),
        )

    @staticmethod
    def _parse_tool_result(
        result: Mapping[str, object],
    ) -> MCPToolResult:
        raw_content = result.get("content")
        if not isinstance(raw_content, Sequence) or isinstance(
            raw_content,
            (str, bytes),
        ):
            raise MCPHTTPError("MCP tools/call result is invalid.")
        blocks = [
            StreamableHTTPTransport._parse_content_block(raw) for raw in raw_content
        ]
        is_error = result.get("isError", False)
        if "structuredContent" in result:
            return MCPToolResult(
                blocks,
                structured_content=result["structuredContent"],
                is_error=is_error,
            )
        return MCPToolResult(blocks, is_error=is_error)

    @staticmethod
    def _parse_content_block(raw: object) -> MCPContentBlock:
        if not isinstance(raw, Mapping):
            raise MCPHTTPError("MCP tool result content block is invalid.")
        content_type = raw.get("type")
        if not isinstance(content_type, str):
            raise MCPHTTPError("MCP tool result content type is invalid.")
        return MCPContentBlock(
            content_type,
            {key: value for key, value in raw.items() if key != "type"},
        )

    @staticmethod
    def _parse_resource_result(
        result: Mapping[str, object],
    ) -> MCPResourceReadResult:
        raw_contents = result.get("contents")
        if not isinstance(raw_contents, Sequence) or isinstance(
            raw_contents,
            (str, bytes),
        ):
            raise MCPHTTPError("MCP resources/read result is invalid.")
        return MCPResourceReadResult(
            [
                StreamableHTTPTransport._parse_resource_content(raw)
                for raw in raw_contents
            ],
            ttl_ms=result.get("ttlMs"),
            cache_scope=result.get("cacheScope"),
        )

    @staticmethod
    def _parse_resource_content(raw: object) -> MCPResourceContent:
        if not isinstance(raw, Mapping):
            raise MCPHTTPError("MCP resource content is invalid.")
        return MCPResourceContent(
            raw.get("uri"),
            mime_type=raw.get("mimeType"),
            text=raw.get("text") if "text" in raw else None,
            blob=raw.get("blob") if "blob" in raw else None,
            annotations=raw.get("annotations"),
        )

    def _tool_headers(
        self,
        request: MCPToolRequest,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        for header_name, path, value_type in self._tool_header_specs.get(
            request.name,
            (),
        ):
            value = self._path_value(request.arguments, path)
            if value is _MISSING or value is None:
                continue
            encoded = self._typed_header_value(
                value,
                value_type,
            )
            headers[f"Mcp-Param-{header_name}"] = self._encode_header_value(encoded)
        return headers

    @staticmethod
    def _extract_tool_headers(
        tool: MCPTool,
    ) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        specs: list[tuple[str, tuple[str, ...], str]] = []

        def visit(
            node: Mapping[str, object],
            path: tuple[str, ...],
            *,
            property_node: bool,
        ) -> None:
            if "x-mcp-header" in node:
                header_name = node["x-mcp-header"]
                value_type = node.get("type")
                if (
                    not property_node
                    or not isinstance(header_name, str)
                    or _HEADER_TOKEN_PATTERN.fullmatch(header_name) is None
                    or value_type not in {"string", "integer", "boolean"}
                ):
                    raise MCPHTTPError("MCP x-mcp-header annotation is invalid.")
                specs.append((header_name, path, value_type))

            for key, value in node.items():
                if key in {"x-mcp-header", "properties"}:
                    continue
                if StreamableHTTPTransport._contains_header_annotation(value):
                    raise MCPHTTPError("MCP x-mcp-header annotation is unreachable.")
            properties = node.get("properties")
            if properties is None:
                return
            if not isinstance(properties, Mapping):
                if StreamableHTTPTransport._contains_header_annotation(properties):
                    raise MCPHTTPError("MCP x-mcp-header annotation is unreachable.")
                return
            for name, child in properties.items():
                if not isinstance(name, str) or not isinstance(
                    child,
                    Mapping,
                ):
                    if StreamableHTTPTransport._contains_header_annotation(child):
                        raise MCPHTTPError("MCP x-mcp-header annotation is invalid.")
                    continue
                visit(
                    child,
                    path + (name,),
                    property_node=True,
                )

        visit(tool.input_schema, (), property_node=False)
        normalized_names = [name.casefold() for name, _, _ in specs]
        if len(normalized_names) != len(set(normalized_names)):
            raise MCPHTTPError("MCP x-mcp-header names must be unique.")
        return tuple(specs)

    @staticmethod
    def _contains_header_annotation(value: object) -> bool:
        if isinstance(value, Mapping):
            return "x-mcp-header" in value or any(
                StreamableHTTPTransport._contains_header_annotation(child)
                for child in value.values()
            )
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            return any(
                StreamableHTTPTransport._contains_header_annotation(child)
                for child in value
            )
        return False

    @staticmethod
    def _path_value(
        arguments: Mapping[str, object],
        path: tuple[str, ...],
    ) -> object:
        value: object = arguments
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                return _MISSING
            value = value[part]
        return value

    @staticmethod
    def _typed_header_value(value: object, value_type: str) -> str:
        if value_type == "string" and isinstance(value, str):
            return value
        if value_type == "boolean" and isinstance(value, bool):
            return "true" if value else "false"
        if (
            value_type == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)
            and -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER
        ):
            return str(value)
        raise MCPHTTPError("MCP header-mirrored tool argument has an invalid value.")

    def _authorization(self) -> str | None:
        if self._authorization_provider is None:
            return None
        try:
            value = self._authorization_provider()
        except Exception as error:
            raise MCPHTTPError(
                f"MCP authorization provider failed: {type(error).__name__}"
            ) from error
        if value is None:
            return None
        if not isinstance(value, str) or not self._plain_header_safe(value):
            raise MCPHTTPError("MCP authorization value is not header-safe.")
        return value

    @staticmethod
    def _encode_header_value(value: str) -> str:
        if (
            StreamableHTTPTransport._plain_header_safe(value)
            and _BASE64_SENTINEL_PATTERN.fullmatch(value) is None
        ):
            return value
        encoded = b64encode(value.encode("utf-8")).decode("ascii")
        return f"=?base64?{encoded}?="

    @staticmethod
    def _plain_header_safe(value: str) -> bool:
        return (
            bool(value)
            and value == value.strip(" \t")
            and all(
                character == "\t" or 0x20 <= ord(character) <= 0x7E
                for character in value
            )
        )

    @staticmethod
    def _response_content_type(response: object) -> str:
        headers = getattr(response, "headers", None)
        if headers is None:
            return ""
        value = headers.get("Content-Type", "")
        return value.split(";", 1)[0].strip().lower()

    def _next_request_id(self) -> int:
        with self._request_id_lock:
            return next(self._request_ids)

    def _require_open(self) -> None:
        if not self._is_open:
            raise MCPHTTPError("MCP HTTP transport is not open.")

    @staticmethod
    def _validate_endpoint(endpoint: object) -> str:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise MCPHTTPError("MCP HTTP endpoint cannot be empty.")
        try:
            parsed = urlsplit(endpoint)
            port = parsed.port
        except ValueError as error:
            raise MCPHTTPError("MCP HTTP endpoint is invalid.") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or port is not None
            and not 0 < port <= 65535
        ):
            raise MCPHTTPError("MCP HTTP endpoint is invalid.")
        return endpoint
