from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Event, Lock

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class WorkflowProgressStatus(str, Enum):
    ROUTE_SELECTED = "route_selected"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_BLOCKED = "stage_blocked"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"


@dataclass(frozen=True)
class WorkflowProgressEvent:
    """Content-free progress snapshot for one Adaptive Multi-Model run."""

    sequence: int
    status: WorkflowProgressStatus
    route: str
    completed_stage_count: int
    expected_stage_count: int
    stage_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence <= 0
        ):
            raise ValueError("Progress sequence must be positive.")
        if not isinstance(self.status, WorkflowProgressStatus):
            raise TypeError("Progress status is invalid.")
        self._validate_identifier(self.route, "route")
        if self.stage_id is not None:
            self._validate_identifier(self.stage_id, "stage ID")
        stage_statuses = {
            WorkflowProgressStatus.STAGE_STARTED,
            WorkflowProgressStatus.STAGE_COMPLETED,
            WorkflowProgressStatus.STAGE_FAILED,
            WorkflowProgressStatus.STAGE_BLOCKED,
        }
        if (self.status in stage_statuses) != (self.stage_id is not None):
            raise ValueError("Progress stage ID is inconsistent with its status.")
        if (
            not isinstance(self.completed_stage_count, int)
            or isinstance(self.completed_stage_count, bool)
            or self.completed_stage_count < 0
            or not isinstance(self.expected_stage_count, int)
            or isinstance(self.expected_stage_count, bool)
            or self.expected_stage_count <= 0
            or self.completed_stage_count > self.expected_stage_count
        ):
            raise ValueError("Progress stage counts are invalid.")

    @staticmethod
    def _validate_identifier(value: object, label: str) -> None:
        if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Progress {label} is invalid.")

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "status": self.status.value,
            "route": self.route,
            "stage_id": self.stage_id,
            "completed_stage_count": self.completed_stage_count,
            "expected_stage_count": self.expected_stage_count,
        }


WorkflowProgressHandler = Callable[[WorkflowProgressEvent], None]


class CancellationToken:
    """Thread-safe one-shot cooperative cancellation signal."""

    def __init__(self) -> None:
        self._event = Event()
        self._cancel_lock = Lock()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> bool:
        with self._cancel_lock:
            if self._event.is_set():
                return False
            self._event.set()
            return True

    def wait(self, timeout_seconds: float) -> bool:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError(
                "Cancellation wait timeout must be finite and non-negative."
            )
        return self._event.wait(timeout_seconds)


class WorkflowCancelledError(RuntimeError):
    """Raised at a safe workflow boundary after cancellation."""

    def __init__(self, partial_result: object | None = None) -> None:
        super().__init__("Adaptive Multi-Model workflow was cancelled.")
        self.partial_result = partial_result


class WorkflowProgressReporter:
    """Emit ordered progress without letting observers affect work."""

    def __init__(
        self,
        route: str,
        handler: WorkflowProgressHandler | None = None,
    ) -> None:
        WorkflowProgressEvent._validate_identifier(route, "route")
        if handler is not None and not callable(handler):
            raise TypeError("Workflow progress handler must be callable.")
        self.route = route
        self.handler = handler
        self._sequence = 0
        self._lock = Lock()

    def emit(
        self,
        status: WorkflowProgressStatus,
        *,
        completed_stage_count: int,
        expected_stage_count: int,
        stage_id: str | None = None,
    ) -> WorkflowProgressEvent:
        with self._lock:
            self._sequence += 1
            event = WorkflowProgressEvent(
                sequence=self._sequence,
                status=status,
                route=self.route,
                stage_id=stage_id,
                completed_stage_count=completed_stage_count,
                expected_stage_count=expected_stage_count,
            )
        if self.handler is not None:
            try:
                self.handler(event)
            except Exception:
                pass
        return event
