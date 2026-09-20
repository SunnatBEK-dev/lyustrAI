from collections.abc import Callable
from time import sleep as default_sleep

from ai_sdk.agents.model import (
    AgentEvent,
    AgentModelResponse,
    AgentStopReason,
)
from ai_sdk.agents.progress import (
    CancellationToken,
    WorkflowCancelledError,
)
from ai_sdk.agents.state import AgentState
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.retry import RetryPolicy
from ai_sdk.llm.types import LLMMessage
from ai_sdk.observability import (
    TraceCategory,
    Tracer,
    trace_span,
)
from ai_sdk.tools.executor import ToolExecutor
from ai_sdk.tools.schema import ToolSchema

AgentEventHandler = Callable[[AgentEvent], None]


class AgentRunner:
    """Provider-neutral policy for bounded tool-assisted runs."""

    def __init__(
        self,
        client: BaseToolLLMClient,
        executor: ToolExecutor,
        *,
        max_tool_rounds: int = 8,
        tracer: Tracer | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = default_sleep,
    ) -> None:
        if not isinstance(client, BaseToolLLMClient):
            raise TypeError("Agent client must support tool turns.")

        if not isinstance(executor, ToolExecutor):
            raise TypeError("Tool executor must be a ToolExecutor.")

        if (
            not isinstance(max_tool_rounds, int)
            or isinstance(max_tool_rounds, bool)
            or max_tool_rounds <= 0
        ):
            raise ValueError("Maximum tool rounds must be greater than zero.")
        if tracer is not None and not isinstance(tracer, Tracer):
            raise TypeError("Agent tracer must be a Tracer.")
        if retry_policy is not None and not isinstance(
            retry_policy,
            RetryPolicy,
        ):
            raise TypeError("Agent retry policy is invalid.")
        if not callable(sleep):
            raise TypeError("Agent retry sleep function is invalid.")

        self.client = client
        self.executor = executor
        self.max_tool_rounds = max_tool_rounds
        self.tracer = tracer or executor.tracer
        self.retry_policy = retry_policy
        self._sleep = sleep

    def run(
        self,
        messages: list[LLMMessage],
        *,
        on_event: AgentEventHandler | None = None,
        cancellation: CancellationToken | None = None,
    ) -> AgentState:
        self._validate_cancellation(cancellation)
        with trace_span(
            self.tracer,
            "agent.run",
            TraceCategory.AGENT,
            {
                "agent.message_count": len(messages),
                "agent.max_tool_rounds": self.max_tool_rounds,
            },
        ) as span:
            state = self._run(
                messages,
                on_event=on_event,
                cancellation=cancellation,
            )
            stop_reason = state.stop_reason
            if stop_reason is None:
                raise RuntimeError("Finished agent state is missing a stop reason.")
            if span is not None:
                span.set_attribute(
                    "agent.tool_round_count",
                    state.tool_rounds,
                )
                span.set_attribute(
                    "agent.stop_reason",
                    stop_reason.value,
                )
                if stop_reason is AgentStopReason.MAX_TOOL_ROUNDS:
                    span.set_error("MaxToolRoundsExceeded")
            return state

    def _run(
        self,
        messages: list[LLMMessage],
        *,
        on_event: AgentEventHandler | None = None,
        cancellation: CancellationToken | None = None,
    ) -> AgentState:
        if on_event is not None and not callable(on_event):
            raise TypeError("Agent event handler must be callable.")

        state = AgentState(messages)
        schemas = self.executor.registry.schemas()
        seen_call_ids: set[str] = set()

        while True:
            self._raise_if_cancelled(cancellation)
            with trace_span(
                self.tracer,
                "llm.tool_turn",
                TraceCategory.LLM,
                {
                    "llm.message_count": len(state.messages),
                    "llm.tool_schema_count": len(schemas),
                    "llm.prior_event_count": len(state.events),
                },
            ) as model_span:
                response, attempt_count = self._complete_tool_turn(
                    state,
                    schemas,
                    cancellation,
                )
                if model_span is not None and isinstance(
                    response,
                    AgentModelResponse,
                ):
                    model_span.set_attribute(
                        "llm.request_attempt_count",
                        attempt_count,
                    )
                    model_span.set_attribute(
                        "llm.tool_call_count",
                        len(response.tool_calls),
                    )

            if not isinstance(response, AgentModelResponse):
                raise TypeError("Agent client response is invalid.")

            self._raise_if_cancelled(cancellation)

            calls = response.tool_calls
            call_ids = [call.id for call in calls]

            if len(call_ids) != len(set(call_ids)) or any(
                call_id in seen_call_ids for call_id in call_ids
            ):
                raise RuntimeError("Agent received a duplicate tool call ID.")

            if calls and state.tool_rounds >= self.max_tool_rounds:
                event = AgentEvent(
                    iteration=len(state.events) + 1,
                    response=response,
                )
                self._record(state, event, on_event)
                state.finish(
                    AgentStopReason.MAX_TOOL_ROUNDS,
                    response.text,
                )
                return state

            result_items = []
            for call in calls:
                self._raise_if_cancelled(cancellation)
                result_items.append(
                    self.executor.execute(
                        call,
                        tracer=self.tracer,
                    )
                )
            self._raise_if_cancelled(cancellation)
            results = tuple(result_items)
            event = AgentEvent(
                iteration=len(state.events) + 1,
                response=response,
                tool_results=results,
            )
            self._record(state, event, on_event)

            if not calls:
                state.finish(
                    AgentStopReason.FINAL_RESPONSE,
                    response.text,
                )
                return state

            seen_call_ids.update(call_ids)

    def _complete_tool_turn(
        self,
        state: AgentState,
        schemas: list[ToolSchema],
        cancellation: CancellationToken | None,
    ) -> tuple[AgentModelResponse, int]:
        attempt = 1
        while True:
            self._raise_if_cancelled(cancellation)
            try:
                response = self.client.complete_tool_turn(
                    state.messages,
                    schemas,
                    tuple(state.events),
                )
                return response, attempt
            except Exception as error:
                self._raise_if_cancelled(cancellation)
                policy = self.retry_policy
                if (
                    policy is None
                    or attempt >= policy.max_attempts
                    or not policy.should_retry(error)
                ):
                    raise
                delay = policy.delay_before_retry(attempt)
                if cancellation is None:
                    self._sleep(delay)
                elif cancellation.wait(delay):
                    raise WorkflowCancelledError() from None
                attempt += 1

    def ask(
        self,
        messages: list[LLMMessage],
        *,
        on_event: AgentEventHandler | None = None,
        cancellation: CancellationToken | None = None,
    ) -> str:
        state = self.run(
            messages,
            on_event=on_event,
            cancellation=cancellation,
        )

        if state.stop_reason is AgentStopReason.MAX_TOOL_ROUNDS:
            raise RuntimeError("Maximum agent tool rounds exceeded.")

        return state.final_text or ""

    @staticmethod
    def _validate_cancellation(
        cancellation: CancellationToken | None,
    ) -> None:
        if cancellation is not None and not isinstance(
            cancellation,
            CancellationToken,
        ):
            raise TypeError("Agent cancellation token is invalid.")

    @staticmethod
    def _raise_if_cancelled(
        cancellation: CancellationToken | None,
    ) -> None:
        if cancellation is not None and cancellation.is_cancelled:
            raise WorkflowCancelledError()

    @staticmethod
    def _record(
        state: AgentState,
        event: AgentEvent,
        handler: AgentEventHandler | None,
    ) -> None:
        state.record(event)

        if handler is not None:
            handler(event)
