from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ai_sdk.agents.model import AgentStopReason
from ai_sdk.agents.progress import (
    CancellationToken,
    WorkflowCancelledError,
)
from ai_sdk.agents.runner import AgentRunner
from ai_sdk.agents.state import AgentState

if TYPE_CHECKING:
    from ai_sdk.llm.retry import RetryPolicy
    from ai_sdk.observability import Tracer
    from ai_sdk.tools.executor import ToolExecutor


class CoordinationError(ValueError):
    """Raised when worker registration or assignment is invalid."""


class AgentTaskStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentWorker:
    name: str
    description: str
    runner: AgentRunner
    provider: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.name, label="Worker name")

        if not isinstance(self.description, str) or not self.description.strip():
            raise CoordinationError("Worker description cannot be empty.")

        if not isinstance(self.runner, AgentRunner):
            raise TypeError("Worker runner must be an AgentRunner.")

        if self.provider is not None:
            _validate_identifier(
                self.provider,
                label="Worker provider",
            )
            object.__setattr__(
                self,
                "provider",
                self.provider.casefold(),
            )

        object.__setattr__(
            self,
            "description",
            self.description.strip(),
        )


@dataclass(frozen=True)
class AgentTask:
    id: str
    worker_name: str
    instruction: str

    def __post_init__(self) -> None:
        _validate_identifier(self.id, label="Task ID")
        _validate_identifier(
            self.worker_name,
            label="Task worker name",
        )

        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise CoordinationError("Task instruction cannot be empty.")

        object.__setattr__(
            self,
            "instruction",
            self.instruction.strip(),
        )


@dataclass(frozen=True)
class AgentTaskResult:
    task: AgentTask
    status: AgentTaskStatus
    state: AgentState | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, AgentTask):
            raise TypeError("Task result task must be an AgentTask.")

        if not isinstance(self.status, AgentTaskStatus):
            raise TypeError("Task result status is invalid.")

        if self.state is not None and not isinstance(
            self.state,
            AgentState,
        ):
            raise TypeError("Task result state is invalid.")

        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise CoordinationError("Task result error must be a non-empty string.")

        if self.status is AgentTaskStatus.COMPLETED:
            if (
                self.state is None
                or self.state.stop_reason is not AgentStopReason.FINAL_RESPONSE
                or self.error is not None
            ):
                raise CoordinationError("Completed task result is inconsistent.")
        elif self.error is None:
            raise CoordinationError("Failed task result must contain an error.")

    @property
    def output(self) -> str | None:
        if self.state is None:
            return None

        return self.state.final_text


@dataclass(frozen=True, init=False)
class CoordinationResult:
    results: tuple[AgentTaskResult, ...]

    def __init__(
        self,
        results: Sequence[AgentTaskResult],
    ) -> None:
        normalized = tuple(results)

        if any(not isinstance(result, AgentTaskResult) for result in normalized):
            raise TypeError("Coordination results are invalid.")

        task_ids = [result.task.id for result in normalized]

        if len(task_ids) != len(set(task_ids)):
            raise CoordinationError("Coordination result task IDs must be unique.")

        object.__setattr__(self, "results", normalized)

    @property
    def completed(self) -> tuple[AgentTaskResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.status is AgentTaskStatus.COMPLETED
        )

    @property
    def failed(self) -> tuple[AgentTaskResult, ...]:
        return tuple(
            result for result in self.results if result.status is AgentTaskStatus.FAILED
        )

    def for_task(self, task_id: str) -> AgentTaskResult | None:
        return next(
            (result for result in self.results if result.task.id == task_id),
            None,
        )


class MultiAgentCoordinator:
    """Run explicit worker assignments sequentially and in order."""

    def __init__(
        self,
        workers: Sequence[AgentWorker] = (),
    ) -> None:
        self._workers: dict[str, AgentWorker] = {}

        for worker in workers:
            self.register(worker)

    def register(self, worker: AgentWorker) -> None:
        if not isinstance(worker, AgentWorker):
            raise TypeError("Registered worker must be an AgentWorker.")

        folded_name = worker.name.casefold()

        if any(name.casefold() == folded_name for name in self._workers):
            raise CoordinationError(f"Worker is already registered: {worker.name}")

        self._workers[worker.name] = worker

    def worker_names(self) -> tuple[str, ...]:
        return tuple(self._workers)

    def count(self) -> int:
        return len(self._workers)

    def run(
        self,
        tasks: Sequence[AgentTask],
        *,
        cancellation: CancellationToken | None = None,
    ) -> CoordinationResult:
        if cancellation is not None and not isinstance(
            cancellation,
            CancellationToken,
        ):
            raise TypeError("Coordinator cancellation token is invalid.")
        normalized_tasks = tuple(tasks)
        self._validate_tasks(normalized_tasks)
        results = [self._run_task(task, cancellation) for task in normalized_tasks]
        return CoordinationResult(results)

    def _validate_tasks(
        self,
        tasks: tuple[AgentTask, ...],
    ) -> None:
        if any(not isinstance(task, AgentTask) for task in tasks):
            raise TypeError("Coordinator tasks must be AgentTask objects.")

        task_ids = [task.id for task in tasks]

        if len(task_ids) != len(set(task_ids)):
            raise CoordinationError("Coordinator task IDs must be unique.")

        unknown_workers = sorted(
            {
                task.worker_name
                for task in tasks
                if task.worker_name not in self._workers
            }
        )

        if unknown_workers:
            raise CoordinationError(
                "Unknown task workers: " + ", ".join(unknown_workers) + "."
            )

    def _run_task(
        self,
        task: AgentTask,
        cancellation: CancellationToken | None,
    ) -> AgentTaskResult:
        worker = self._workers[task.worker_name]
        assignment = (
            f"Worker: {worker.name}\n"
            f"Responsibility: {worker.description}\n\n"
            f"Assigned task: {task.instruction}"
        )

        try:
            state = worker.runner.run(
                [
                    {
                        "role": "user",
                        "content": assignment,
                    }
                ],
                cancellation=cancellation,
            )
        except WorkflowCancelledError:
            raise
        except Exception as error:
            return AgentTaskResult(
                task=task,
                status=AgentTaskStatus.FAILED,
                error=(f"Worker execution failed: {type(error).__name__}"),
            )

        if state.stop_reason is AgentStopReason.FINAL_RESPONSE:
            return AgentTaskResult(
                task=task,
                status=AgentTaskStatus.COMPLETED,
                state=state,
            )

        reason = state.stop_reason.value if state.stop_reason is not None else "unknown"
        return AgentTaskResult(
            task=task,
            status=AgentTaskStatus.FAILED,
            state=state,
            error=f"Worker stopped: {reason}",
        )


def create_provider_worker(
    name: str,
    description: str,
    provider: str,
    executor: ToolExecutor | None = None,
    *,
    max_tool_rounds: int = 8,
    tracer: Tracer | None = None,
    retry_policy: RetryPolicy | None = None,
) -> AgentWorker:
    """Build a worker bound to one configured LLM provider."""
    from ai_sdk.llm.factory import (
        create_llm_client,
        normalize_llm_provider,
    )
    from ai_sdk.tools.executor import ToolExecutor
    from ai_sdk.tools.registry import ToolRegistry

    normalized_provider = normalize_llm_provider(provider)
    if executor is not None and not isinstance(
        executor,
        ToolExecutor,
    ):
        raise TypeError("Worker executor must be a ToolExecutor.")

    active_executor = executor or ToolExecutor(ToolRegistry())
    runner = AgentRunner(
        create_llm_client(normalized_provider),
        active_executor,
        max_tool_rounds=max_tool_rounds,
        tracer=tracer,
        retry_policy=retry_policy,
    )
    return AgentWorker(
        name=name,
        description=description,
        runner=runner,
        provider=normalized_provider,
    )


def _validate_identifier(name: str, *, label: str) -> None:
    if (
        not isinstance(name, str)
        or re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]{0,63}",
            name,
        )
        is None
    ):
        raise CoordinationError(f"{label} must be a valid identifier.")
