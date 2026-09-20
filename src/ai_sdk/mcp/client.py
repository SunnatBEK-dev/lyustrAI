from collections.abc import Callable, Mapping
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import TypeVar

from ai_sdk.mcp.model import (
    MCP_PROTOCOL_VERSION,
    MCPConnectionState,
    MCPContinuation,
    MCPDiscoveryResult,
    MCPImplementation,
    MCPInputRequiredResult,
    MCPRequestContext,
    MCPResourcePage,
    MCPResourceReadRequest,
    MCPResourceReadResult,
    MCPToolPage,
    MCPToolRequest,
    MCPToolResult,
    MCPValidationError,
)
from ai_sdk.mcp.transport import (
    BaseMCPTransport,
    MCPTransportResponseError,
)
from ai_sdk.observability import (
    TraceCategory,
    Tracer,
    trace_span,
)


class MCPClientError(RuntimeError):
    """Base error for MCP client runtime failures."""


class MCPLifecycleError(MCPClientError):
    """Raised when an operation is invalid for the client state."""


class MCPTimeoutError(MCPClientError):
    """Raised when an MCP transport operation times out."""


class MCPTransportError(MCPClientError):
    """Raised when the transport fails without exposing secrets."""


class MCPProtocolError(MCPClientError):
    """Raised when a transport returns an invalid MCP result."""


class MCPCapabilityError(MCPClientError):
    """Raised when discovery says an operation is unsupported."""


class MCPRemoteError(MCPClientError):
    """A valid JSON-RPC error response from an MCP server."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"MCP server error {code}: {message}")


class MCPContinuationError(MCPClientError):
    """Raised when a local MCP continuation cannot be resumed."""


class MCPInputRoundsExceededError(MCPClientError):
    """Raised when one logical MCP request exceeds its round limit."""


ResultT = TypeVar("ResultT")
MCPRoundRequest = MCPToolRequest | MCPResourceReadRequest
MCPRoundResult = MCPToolResult | MCPResourceReadResult


@dataclass(frozen=True)
class _PendingContinuation:
    continuation: MCPContinuation
    request: MCPRoundRequest
    request_state: str | None
    input_keys: frozenset[str]


class MCPClient:
    """Stateless MCP 2026-07-28 client with transport lifecycle."""

    def __init__(
        self,
        transport: BaseMCPTransport,
        *,
        client_info: MCPImplementation,
        client_capabilities: Mapping[str, object] | None = None,
        protocol_version: str = MCP_PROTOCOL_VERSION,
        timeout_seconds: float = 30.0,
        max_input_rounds: int = 10,
        tracer: Tracer | None = None,
    ) -> None:
        if not isinstance(transport, BaseMCPTransport):
            raise MCPValidationError("MCP transport is invalid.")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise MCPValidationError("MCP timeout must be greater than zero.")
        if (
            not isinstance(max_input_rounds, int)
            or isinstance(max_input_rounds, bool)
            or max_input_rounds <= 0
        ):
            raise MCPValidationError("MCP maximum input rounds must be positive.")
        if tracer is not None and not isinstance(tracer, Tracer):
            raise MCPValidationError("MCP tracer must be a Tracer.")

        self._transport = transport
        self._context = MCPRequestContext(
            protocol_version,
            client_info,
            client_capabilities,
        )
        self._timeout_seconds = float(timeout_seconds)
        self._max_input_rounds = max_input_rounds
        self.tracer = tracer
        self._state = MCPConnectionState.NEW
        self._transport_opened = False
        self._discovery: MCPDiscoveryResult | None = None
        self._pending_continuations: dict[
            str,
            _PendingContinuation,
        ] = {}

    @property
    def state(self) -> MCPConnectionState:
        return self._state

    @property
    def request_context(self) -> MCPRequestContext:
        return self._context

    @property
    def discovery(self) -> MCPDiscoveryResult | None:
        return self._discovery

    def open(self) -> None:
        if self._state is not MCPConnectionState.NEW:
            raise MCPLifecycleError("MCP client can only open from the new state.")

        self._state = MCPConnectionState.OPENING
        try:
            self._transport.open(timeout_seconds=self._timeout_seconds)
        except TimeoutError as error:
            self._abort()
            raise MCPTimeoutError("MCP transport open timed out.") from error
        except Exception as error:
            self._abort()
            raise MCPTransportError(
                f"MCP transport open failed: {type(error).__name__}"
            ) from error

        self._transport_opened = True
        self._state = MCPConnectionState.OPEN

    def discover(self) -> MCPDiscoveryResult:
        result = self._request(
            "server discovery",
            lambda: self._transport.discover(
                self._context,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        if not isinstance(result, MCPDiscoveryResult):
            self._fail_protocol("MCP discovery returned an invalid result.")
        if self._context.protocol_version not in result.supported_versions:
            self._fail_protocol(
                "MCP server does not support the requested protocol version."
            )

        self._discovery = result
        return result

    def list_tools(
        self,
        *,
        cursor: str | None = None,
    ) -> MCPToolPage:
        self._require_open()
        self._validate_cursor(cursor)
        if (
            self._discovery is not None
            and not self._discovery.capabilities.supports_tools
        ):
            raise MCPCapabilityError("MCP server did not advertise tools support.")

        page = self._request(
            "tools/list",
            lambda: self._transport.list_tools(
                self._context,
                cursor=cursor,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        if not isinstance(page, MCPToolPage):
            self._fail_protocol("MCP tools/list returned an invalid page.")
        names = [tool.name for tool in page.tools]
        if len(set(names)) != len(names):
            self._fail_protocol("MCP tools/list returned duplicate tool names.")
        return page

    def list_resources(
        self,
        *,
        cursor: str | None = None,
    ) -> MCPResourcePage:
        self._require_open()
        self._validate_cursor(cursor)
        if (
            self._discovery is not None
            and not self._discovery.capabilities.supports_resources
        ):
            raise MCPCapabilityError("MCP server did not advertise resources support.")

        page = self._request(
            "resources/list",
            lambda: self._transport.list_resources(
                self._context,
                cursor=cursor,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        if not isinstance(page, MCPResourcePage):
            self._fail_protocol("MCP resources/list returned an invalid page.")
        uris = [resource.uri for resource in page.resources]
        if len(set(uris)) != len(uris):
            self._fail_protocol("MCP resources/list returned duplicate resource URIs.")
        return page

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, object] | None = None,
    ) -> MCPToolResult | MCPContinuation:
        self._require_open()
        if (
            self._discovery is not None
            and not self._discovery.capabilities.supports_tools
        ):
            raise MCPCapabilityError("MCP server did not advertise tools support.")
        request = MCPToolRequest(name, arguments)
        result = self._request(
            "tools/call",
            lambda: self._transport.call_tool(
                self._context,
                request,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        return self._process_round_result(
            "tools/call",
            request,
            result,
            round=1,
        )

    def read_resource(
        self,
        uri: str,
    ) -> MCPResourceReadResult | MCPContinuation:
        self._require_open()
        if (
            self._discovery is not None
            and not self._discovery.capabilities.supports_resources
        ):
            raise MCPCapabilityError("MCP server did not advertise resources support.")
        request = MCPResourceReadRequest(uri)
        result = self._request(
            "resources/read",
            lambda: self._transport.read_resource(
                self._context,
                request,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        return self._process_round_result(
            "resources/read",
            request,
            result,
            round=1,
        )

    def continue_request(
        self,
        continuation: MCPContinuation,
        input_responses: Mapping[
            str,
            Mapping[str, object],
        ]
        | None = None,
    ) -> MCPRoundResult | MCPContinuation:
        self._require_open()
        pending = self._pending_continuation(continuation)
        if isinstance(pending.request, MCPToolRequest):
            retry: MCPRoundRequest = MCPToolRequest(
                pending.request.name,
                pending.request.arguments,
                input_responses=input_responses,
                request_state=pending.request_state,
            )
        else:
            retry = MCPResourceReadRequest(
                pending.request.uri,
                input_responses=input_responses,
                request_state=pending.request_state,
            )
        if set(retry.input_responses) != pending.input_keys:
            raise MCPContinuationError(
                "MCP input responses must exactly match the pending request keys."
            )

        self._pending_continuations.pop(continuation.continuation_id)
        if isinstance(retry, MCPToolRequest):
            result = self._request(
                "tools/call continuation",
                lambda: self._transport.call_tool(
                    self._context,
                    retry,
                    timeout_seconds=self._timeout_seconds,
                ),
            )
        else:
            result = self._request(
                "resources/read continuation",
                lambda: self._transport.read_resource(
                    self._context,
                    retry,
                    timeout_seconds=self._timeout_seconds,
                ),
            )
        return self._process_round_result(
            continuation.operation,
            retry,
            result,
            round=continuation.round + 1,
        )

    def cancel_continuation(
        self,
        continuation: MCPContinuation,
    ) -> None:
        self._require_open()
        self._pending_continuation(continuation)
        self._pending_continuations.pop(continuation.continuation_id)

    def close(self) -> None:
        if self._state is MCPConnectionState.CLOSED:
            return

        should_close_transport = (
            self._transport_opened or self._state is MCPConnectionState.OPENING
        )
        self._transport_opened = False
        self._state = MCPConnectionState.CLOSED
        self._pending_continuations.clear()

        if not should_close_transport:
            return
        try:
            self._transport.close()
        except Exception as error:
            raise MCPTransportError(
                f"MCP transport close failed: {type(error).__name__}"
            ) from error

    def __enter__(self) -> "MCPClient":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(
        self,
        operation: str,
        callback: Callable[[], ResultT],
    ) -> ResultT:
        self._require_open()
        with trace_span(
            self.tracer,
            "mcp.request",
            TraceCategory.MCP,
            {"mcp.operation": operation},
        ) as span:
            try:
                result = callback()
            except MCPTransportResponseError as error:
                raise MCPRemoteError(
                    error.code,
                    error.message,
                ) from error
            except TimeoutError as error:
                self._abort()
                raise MCPTimeoutError(f"MCP {operation} timed out.") from error
            except Exception as error:
                self._abort()
                raise MCPTransportError(
                    f"MCP {operation} failed: {type(error).__name__}"
                ) from error
            if span is not None:
                span.set_attribute(
                    "mcp.input_required",
                    isinstance(result, MCPInputRequiredResult),
                )
            return result

    def _process_round_result(
        self,
        operation: str,
        request: MCPRoundRequest,
        result: object,
        *,
        round: int,
    ) -> MCPRoundResult | MCPContinuation:
        if isinstance(result, MCPInputRequiredResult):
            if round > self._max_input_rounds:
                raise MCPInputRoundsExceededError(
                    "MCP input-required round limit was exceeded."
                )
            self._validate_input_capabilities(result)
            continuation_id = token_urlsafe(18)
            while continuation_id in self._pending_continuations:
                continuation_id = token_urlsafe(18)
            continuation = MCPContinuation(
                continuation_id,
                operation,
                result.input_requests,
                round=round,
            )
            self._pending_continuations[continuation.continuation_id] = (
                _PendingContinuation(
                    continuation,
                    request,
                    result.request_state,
                    frozenset(result.input_requests),
                )
            )
            return continuation
        if operation == "tools/call" and isinstance(
            result,
            MCPToolResult,
        ):
            return result
        if operation == "resources/read" and isinstance(
            result,
            MCPResourceReadResult,
        ):
            return result
        self._fail_protocol(f"MCP {operation} returned an invalid result.")

    def _pending_continuation(
        self,
        continuation: object,
    ) -> _PendingContinuation:
        if not isinstance(continuation, MCPContinuation):
            raise MCPContinuationError("MCP continuation is invalid.")
        pending = self._pending_continuations.get(continuation.continuation_id)
        if pending is None or pending.continuation is not continuation:
            raise MCPContinuationError(
                "MCP continuation is unknown or already consumed."
            )
        return pending

    def _validate_input_capabilities(
        self,
        result: MCPInputRequiredResult,
    ) -> None:
        capability_by_method = {
            "elicitation/create": "elicitation",
            "sampling/createMessage": "sampling",
            "roots/list": "roots",
        }
        missing = {
            capability_by_method[request.method]
            for request in result.input_requests.values()
            if capability_by_method[request.method]
            not in self._context.client_capabilities
        }
        if missing:
            self._fail_protocol("MCP server requested an undeclared client capability.")

    def _require_open(self) -> None:
        if self._state is not MCPConnectionState.OPEN:
            raise MCPLifecycleError("MCP client operation requires an open transport.")

    @staticmethod
    def _validate_cursor(cursor: str | None) -> None:
        if cursor is not None and (not isinstance(cursor, str) or not cursor.strip()):
            raise MCPValidationError("MCP cursor must be a non-empty string.")

    def _fail_protocol(self, message: str) -> None:
        self._abort()
        raise MCPProtocolError(message)

    def _abort(self) -> None:
        try:
            self._transport.close()
        except Exception:
            pass
        self._transport_opened = False
        self._pending_continuations.clear()
        self._state = MCPConnectionState.FAILED
