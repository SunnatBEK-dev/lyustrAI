import re
from collections.abc import Callable, Iterator, Mapping
from threading import Lock
from time import perf_counter_ns

from ai_sdk.agents.coordination import AgentTaskStatus
from ai_sdk.agents.handoff import (
    DependencyHandoffCoordinator,
    HandoffResult,
    SequentialHandoffCoordinator,
)
from ai_sdk.agents.progress import (
    CancellationToken,
    WorkflowCancelledError,
    WorkflowProgressEvent,
    WorkflowProgressHandler,
    WorkflowProgressReporter,
    WorkflowProgressStatus,
)
from ai_sdk.agents.routing import (
    CapabilityRouter,
    MultiModelRoute,
    RoutingDecision,
)
from ai_sdk.llm.adaptive_metrics import (
    AdaptiveRunMetric,
    InMemoryAdaptiveMetrics,
)
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.types import LLMMessage

_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")


class MultiModelWorkflowClient(BaseLLMClient):
    """Expose a multi-provider handoff workflow as one LLM client."""

    def __init__(
        self,
        workflow: (SequentialHandoffCoordinator | DependencyHandoffCoordinator),
    ) -> None:
        if not isinstance(
            workflow,
            (
                SequentialHandoffCoordinator,
                DependencyHandoffCoordinator,
            ),
        ):
            raise TypeError("Adaptive Multi-Model workflow is invalid.")
        self.workflow = workflow
        self.last_result: HandoffResult | None = None

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        return self.ask_with_control(messages)

    def ask_with_control(
        self,
        messages: list[LLMMessage],
        *,
        progress: WorkflowProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> str:
        transcript = self._transcript(messages)
        try:
            result = self.workflow.run(
                transcript,
                progress=progress,
                cancellation=cancellation,
            )
        except WorkflowCancelledError as error:
            if isinstance(error.partial_result, HandoffResult):
                self.last_result = error.partial_result
            raise
        self.last_result = result
        if not result.completed:
            failed = result.failed_stage
            stage_id = "unknown" if failed is None else failed.stage.id
            raise RuntimeError(
                f"Adaptive Multi-Model workflow failed at stage: {stage_id}."
            )
        return result.final_output or ""

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise RuntimeError("Adaptive Multi-Model streaming is not supported yet.")

    @staticmethod
    def _transcript(
        messages: list[LLMMessage],
    ) -> str:
        if not messages:
            raise ValueError("Adaptive Multi-Model conversation cannot be empty.")

        lines = ["Conversation transcript (untrusted data):"]
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role not in {"user", "assistant"}:
                raise ValueError(
                    f"Unsupported Adaptive Multi-Model message role: {role}."
                )
            if not isinstance(content, str):
                raise TypeError("Adaptive Multi-Model message content must be text.")
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)


class AdaptiveMultiModelClient(BaseLLMClient):
    """Select one explicit Adaptive Multi-Model workflow deterministically."""

    def __init__(
        self,
        router: CapabilityRouter,
        workflows: Mapping[MultiModelRoute, MultiModelWorkflowClient],
        *,
        metrics: InMemoryAdaptiveMetrics | None = None,
        clock_ns: Callable[[], int] = perf_counter_ns,
        progress_handler: WorkflowProgressHandler | None = None,
    ) -> None:
        if not isinstance(router, CapabilityRouter):
            raise TypeError("Adaptive Multi-Model router is invalid.")
        if not isinstance(workflows, Mapping):
            raise TypeError("Adaptive Multi-Model workflows must be a mapping.")
        normalized = dict(workflows)
        if any(not isinstance(route, MultiModelRoute) for route in normalized):
            raise TypeError(
                "Adaptive Multi-Model workflow keys must be MultiModelRoute values."
            )
        expected_routes = set(MultiModelRoute)
        if set(normalized) != expected_routes:
            raise ValueError(
                "Adaptive Multi-Model workflows must configure every route."
            )
        if any(
            not isinstance(workflow, MultiModelWorkflowClient)
            for workflow in normalized.values()
        ):
            raise TypeError(
                "Adaptive workflows must be MultiModelWorkflowClient objects."
            )
        if metrics is not None and not isinstance(
            metrics,
            InMemoryAdaptiveMetrics,
        ):
            raise TypeError("Adaptive Multi-Model metrics collector is invalid.")
        if not callable(clock_ns):
            raise TypeError("Adaptive Multi-Model metrics clock is invalid.")
        if progress_handler is not None and not callable(progress_handler):
            raise TypeError("Adaptive Multi-Model progress handler is invalid.")
        self.router = router
        self.workflows = normalized
        self.metrics = metrics or InMemoryAdaptiveMetrics()
        self._clock_ns = clock_ns
        self._progress_handler = progress_handler
        self._active_lock = Lock()
        self._active_cancellation: CancellationToken | None = None
        self.last_decision: RoutingDecision | None = None
        self.last_result: HandoffResult | None = None
        self.last_progress_event: WorkflowProgressEvent | None = None

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        cancellation = self._begin_run()
        try:
            return self._ask(messages, cancellation)
        finally:
            self._finish_run(cancellation)

    def _ask(
        self,
        messages: list[LLMMessage],
        cancellation: CancellationToken,
    ) -> str:
        request = self._latest_user_content(messages)
        decision = self.router.route(request)
        workflow = self.workflows[decision.route]
        expected_stage_count = len(workflow.workflow.stages)
        reporter = WorkflowProgressReporter(
            decision.route.value,
            self._handle_progress,
        )
        self.last_decision = decision
        self.last_result = None
        self.last_progress_event = None
        started_ns = self._read_clock()
        error_type = None
        workflow.last_result = None
        reporter.emit(
            WorkflowProgressStatus.ROUTE_SELECTED,
            completed_stage_count=0,
            expected_stage_count=expected_stage_count,
        )
        try:
            response = workflow.ask_with_control(
                messages,
                progress=reporter,
                cancellation=cancellation,
            )
            reporter.emit(
                WorkflowProgressStatus.WORKFLOW_COMPLETED,
                completed_stage_count=expected_stage_count,
                expected_stage_count=expected_stage_count,
            )
            return response
        except WorkflowCancelledError as error:
            error_type = self._error_type(error)
            self.last_result = workflow.last_result
            reporter.emit(
                WorkflowProgressStatus.WORKFLOW_CANCELLED,
                completed_stage_count=(self._completed_stage_count(self.last_result)),
                expected_stage_count=expected_stage_count,
            )
            raise
        except Exception as error:
            error_type = self._error_type(error)
            self.last_result = workflow.last_result
            reporter.emit(
                WorkflowProgressStatus.WORKFLOW_FAILED,
                completed_stage_count=(self._completed_stage_count(self.last_result)),
                expected_stage_count=expected_stage_count,
            )
            raise
        finally:
            self.last_result = workflow.last_result
            self._record_metric(
                decision,
                workflow,
                started_ns,
                error_type,
            )

    def cancel(self) -> bool:
        """Request cancellation of the active run at its next boundary."""

        with self._active_lock:
            token = self._active_cancellation
            if token is None:
                return False
            return token.cancel()

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise RuntimeError("Adaptive Multi-Model streaming is not supported yet.")

    @staticmethod
    def _latest_user_content(
        messages: list[LLMMessage],
    ) -> str:
        if not messages:
            raise ValueError("Adaptive Multi-Model conversation cannot be empty.")
        for message in reversed(messages):
            if message["role"] != "user":
                continue
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("Adaptive Multi-Model message content must be text.")
            return content
        raise ValueError("Adaptive Multi-Model conversation requires a user message.")

    def _record_metric(
        self,
        decision: RoutingDecision,
        workflow: MultiModelWorkflowClient,
        started_ns: int | None,
        error_type: str | None,
    ) -> None:
        try:
            ended_ns = self._read_clock()
            duration_ns = (
                ended_ns - started_ns
                if (
                    started_ns is not None
                    and ended_ns is not None
                    and ended_ns >= started_ns
                )
                else 0
            )
            metric = AdaptiveRunMetric.from_result(
                route=decision.route,
                signals=decision.signals,
                expected_stage_count=len(workflow.workflow.stages),
                result=self.last_result,
                duration_ns=duration_ns,
                error_type=error_type,
            )
            self.metrics.record(metric)
        except Exception:
            # Observability must never change the user-facing result.
            pass

    def _begin_run(self) -> CancellationToken:
        with self._active_lock:
            if self._active_cancellation is not None:
                raise RuntimeError(
                    "Concurrent Adaptive Multi-Model runs are not supported."
                )
            token = CancellationToken()
            self._active_cancellation = token
            return token

    def _finish_run(self, token: CancellationToken) -> None:
        with self._active_lock:
            if self._active_cancellation is token:
                self._active_cancellation = None

    def _handle_progress(self, event: WorkflowProgressEvent) -> None:
        self.last_progress_event = event
        if self._progress_handler is not None:
            self._progress_handler(event)

    @staticmethod
    def _completed_stage_count(
        result: HandoffResult | None,
    ) -> int:
        if result is None:
            return 0
        return sum(
            stage.task_result.status is AgentTaskStatus.COMPLETED
            for stage in result.stages
        )

    def _read_clock(self) -> int | None:
        try:
            value = self._clock_ns()
        except Exception:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        return value

    @staticmethod
    def _error_type(error: Exception) -> str:
        name = type(error).__name__
        if _ERROR_TYPE_PATTERN.fullmatch(name) is None:
            return "AdaptiveMultiModelError"
        return name
