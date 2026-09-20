from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType

from ai_sdk.agents.coordination import AgentTaskStatus
from ai_sdk.agents.handoff import (
    DependencyHandoffResult,
    HandoffResult,
)
from ai_sdk.agents.routing import MultiModelRoute, RoutingSignal

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")


class AdaptiveMetricsValidationError(ValueError):
    """Raised when an Adaptive Multi-Model runtime metric is invalid."""


@dataclass(frozen=True, init=False)
class AdaptiveRunMetric:
    """Content-free facts about one Adaptive Multi-Model workflow run."""

    route: MultiModelRoute
    signals: tuple[RoutingSignal, ...]
    expected_stage_count: int
    executed_stage_ids: tuple[str, ...]
    failed_stage_ids: tuple[str, ...]
    blocked_stage_ids: tuple[str, ...]
    duration_ns: int
    completed: bool
    error_type: str | None

    def __init__(
        self,
        route: MultiModelRoute,
        signals: Sequence[RoutingSignal],
        expected_stage_count: int,
        executed_stage_ids: Sequence[str],
        failed_stage_ids: Sequence[str],
        blocked_stage_ids: Sequence[str],
        duration_ns: int,
        completed: bool,
        error_type: str | None = None,
    ) -> None:
        if not isinstance(route, MultiModelRoute):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model metric route is invalid."
            )
        normalized_signals = tuple(signals)
        if any(not isinstance(signal, RoutingSignal) for signal in normalized_signals):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model metric signals are invalid."
            )
        if len(normalized_signals) != len(set(normalized_signals)):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model metric signals must be unique."
            )
        if (
            not isinstance(expected_stage_count, int)
            or isinstance(expected_stage_count, bool)
            or expected_stage_count <= 0
        ):
            raise AdaptiveMetricsValidationError(
                "Expected stage count must be positive."
            )
        executed = self._stage_ids(
            executed_stage_ids,
            "executed",
        )
        failed = self._stage_ids(failed_stage_ids, "failed")
        blocked = self._stage_ids(blocked_stage_ids, "blocked")
        if not set(failed).issubset(executed):
            raise AdaptiveMetricsValidationError(
                "Failed stages must be executed stages."
            )
        if set(executed).intersection(blocked):
            raise AdaptiveMetricsValidationError(
                "Executed and blocked stages cannot overlap."
            )
        if len(executed) + len(blocked) > expected_stage_count:
            raise AdaptiveMetricsValidationError(
                "Stage totals exceed the expected stage count."
            )
        if (
            not isinstance(duration_ns, int)
            or isinstance(duration_ns, bool)
            or duration_ns < 0
        ):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model duration must be non-negative."
            )
        if not isinstance(completed, bool):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model completion flag is invalid."
            )
        if error_type is not None and (
            not isinstance(error_type, str)
            or _ERROR_TYPE_PATTERN.fullmatch(error_type) is None
        ):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model error type is invalid."
            )
        if completed and (
            len(executed) != expected_stage_count
            or failed
            or blocked
            or error_type is not None
        ):
            raise AdaptiveMetricsValidationError(
                "Completed Adaptive Multi-Model metric is inconsistent."
            )

        object.__setattr__(self, "route", route)
        object.__setattr__(self, "signals", normalized_signals)
        object.__setattr__(
            self,
            "expected_stage_count",
            expected_stage_count,
        )
        object.__setattr__(self, "executed_stage_ids", executed)
        object.__setattr__(self, "failed_stage_ids", failed)
        object.__setattr__(self, "blocked_stage_ids", blocked)
        object.__setattr__(self, "duration_ns", duration_ns)
        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "error_type", error_type)

    @classmethod
    def from_result(
        cls,
        *,
        route: MultiModelRoute,
        signals: Sequence[RoutingSignal],
        expected_stage_count: int,
        result: HandoffResult | None,
        duration_ns: int,
        error_type: str | None,
    ) -> AdaptiveRunMetric:
        stages = () if result is None else result.stages
        executed = tuple(stage.stage.id for stage in stages)
        failed = tuple(
            stage.stage.id
            for stage in stages
            if stage.task_result.status is AgentTaskStatus.FAILED
        )
        blocked = (
            result.blocked_stage_ids
            if isinstance(result, DependencyHandoffResult)
            else ()
        )
        return cls(
            route=route,
            signals=signals,
            expected_stage_count=expected_stage_count,
            executed_stage_ids=executed,
            failed_stage_ids=failed,
            blocked_stage_ids=blocked,
            duration_ns=duration_ns,
            completed=(result is not None and result.completed and error_type is None),
            error_type=error_type,
        )

    @staticmethod
    def _stage_ids(
        values: Sequence[str],
        label: str,
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise AdaptiveMetricsValidationError(
                f"Adaptive Multi-Model {label} stage IDs must be a sequence."
            )
        normalized = tuple(values)
        if any(
            not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None
            for value in normalized
        ):
            raise AdaptiveMetricsValidationError(
                f"Adaptive Multi-Model {label} stage ID is invalid."
            )
        if len(normalized) != len(set(normalized)):
            raise AdaptiveMetricsValidationError(
                f"Adaptive Multi-Model {label} stage IDs must be unique."
            )
        return normalized

    @property
    def executed_stage_count(self) -> int:
        return len(self.executed_stage_ids)

    @property
    def failed_stage_count(self) -> int:
        return len(self.failed_stage_ids)

    @property
    def blocked_stage_count(self) -> int:
        return len(self.blocked_stage_ids)

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000


@dataclass(frozen=True)
class AdaptiveMetricsReport:
    total_runs: int
    successful_runs: int
    failed_runs: int
    expected_stage_count: int
    executed_stage_count: int
    failed_stage_count: int
    blocked_stage_count: int
    mean_duration_ms: float
    max_duration_ms: float
    route_counts: Mapping[str, int]
    stage_execution_counts: Mapping[str, int]
    stage_failure_counts: Mapping[str, int]
    stage_blocked_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "expected_stage_count": self.expected_stage_count,
            "executed_stage_count": self.executed_stage_count,
            "failed_stage_count": self.failed_stage_count,
            "blocked_stage_count": self.blocked_stage_count,
            "mean_duration_ms": self.mean_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "route_counts": dict(self.route_counts),
            "stage_execution_counts": dict(self.stage_execution_counts),
            "stage_failure_counts": dict(self.stage_failure_counts),
            "stage_blocked_counts": dict(self.stage_blocked_counts),
        }


class InMemoryAdaptiveMetrics:
    """Keep bounded, content-free Adaptive Multi-Model runtime metrics in memory."""

    def __init__(self, max_records: int = 1_000) -> None:
        if (
            not isinstance(max_records, int)
            or isinstance(max_records, bool)
            or max_records <= 0
        ):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model metric limit must be positive."
            )
        self.max_records = max_records
        self._records: list[AdaptiveRunMetric] = []
        self._lock = Lock()

    def record(self, metric: AdaptiveRunMetric) -> None:
        if not isinstance(metric, AdaptiveRunMetric):
            raise AdaptiveMetricsValidationError(
                "Adaptive Multi-Model runtime metric is invalid."
            )
        with self._lock:
            self._records.append(metric)
            overflow = len(self._records) - self.max_records
            if overflow > 0:
                del self._records[:overflow]

    def records(self) -> tuple[AdaptiveRunMetric, ...]:
        with self._lock:
            return tuple(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def report(self) -> AdaptiveMetricsReport:
        records = self.records()
        route_counts: dict[str, int] = {}
        stage_execution_counts: dict[str, int] = {}
        stage_failure_counts: dict[str, int] = {}
        stage_blocked_counts: dict[str, int] = {}

        for metric in records:
            self._increment(route_counts, metric.route.value)
            for stage_id in metric.executed_stage_ids:
                self._increment(stage_execution_counts, stage_id)
            for stage_id in metric.failed_stage_ids:
                self._increment(stage_failure_counts, stage_id)
            for stage_id in metric.blocked_stage_ids:
                self._increment(stage_blocked_counts, stage_id)

        durations = [metric.duration_ms for metric in records]
        successful = sum(metric.completed for metric in records)
        return AdaptiveMetricsReport(
            total_runs=len(records),
            successful_runs=successful,
            failed_runs=len(records) - successful,
            expected_stage_count=sum(metric.expected_stage_count for metric in records),
            executed_stage_count=sum(metric.executed_stage_count for metric in records),
            failed_stage_count=sum(metric.failed_stage_count for metric in records),
            blocked_stage_count=sum(metric.blocked_stage_count for metric in records),
            mean_duration_ms=(sum(durations) / len(durations) if durations else 0.0),
            max_duration_ms=max(durations, default=0.0),
            route_counts=MappingProxyType(route_counts),
            stage_execution_counts=MappingProxyType(stage_execution_counts),
            stage_failure_counts=MappingProxyType(stage_failure_counts),
            stage_blocked_counts=MappingProxyType(stage_blocked_counts),
        )

    @staticmethod
    def _increment(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1
