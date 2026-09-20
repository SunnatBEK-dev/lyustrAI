from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from secrets import token_hex
from threading import Lock
from time import perf_counter_ns, time_ns
from typing import Protocol

from ai_sdk.observability.model import (
    TraceCategory,
    TraceRecord,
    TraceStatus,
    TraceValidationError,
    _validate_error_type,
    _validate_span_id,
    _validate_span_name,
    _validate_trace_id,
    sanitize_trace_attributes,
)

TraceIdFactory = Callable[[], str]
Clock = Callable[[], int]


class TraceCollector(Protocol):
    """Destination contract for completed trace records."""

    def emit(self, record: TraceRecord) -> None:
        """Accept one completed trace record."""


class InMemoryTraceCollector:
    """Thread-safe bounded in-memory trace collector."""

    def __init__(self, *, max_records: int = 1000) -> None:
        if (
            not isinstance(max_records, int)
            or isinstance(max_records, bool)
            or max_records <= 0
        ):
            raise TraceValidationError("Trace collector capacity must be positive.")
        self._records: deque[TraceRecord] = deque(maxlen=max_records)
        self._lock = Lock()

    def emit(self, record: TraceRecord) -> None:
        if not isinstance(record, TraceRecord):
            raise TraceValidationError("Trace collector record is invalid.")
        with self._lock:
            self._records.append(record)

    def records(
        self,
        *,
        trace_id: str | None = None,
    ) -> tuple[TraceRecord, ...]:
        with self._lock:
            records = tuple(self._records)
        if trace_id is not None:
            records = tuple(record for record in records if record.trace_id == trace_id)
        return tuple(sorted(records, key=lambda record: record.started_at_ns))

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


@dataclass(frozen=True)
class _ActiveSpan:
    trace_id: str
    span_id: str


class TraceSpan:
    """One mutable in-flight span that emits an immutable record."""

    def __init__(
        self,
        tracer: "Tracer",
        name: str,
        category: TraceCategory,
        attributes: Mapping[str, object] | None,
    ) -> None:
        self._tracer = tracer
        self._name = _validate_span_name(name)
        if not isinstance(category, TraceCategory):
            raise TraceValidationError("Trace category is invalid.")
        self._category = category
        self._attributes = sanitize_trace_attributes(attributes)
        self._status = TraceStatus.OK
        self._error_type: str | None = None
        self._trace_id = ""
        self._span_id = ""
        self._parent_span_id: str | None = None
        self._started_at_ns = 0
        self._started_monotonic_ns = 0
        self._token: Token[_ActiveSpan | None] | None = None
        self._entered = False
        self._finished = False

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_id

    def __enter__(self) -> "TraceSpan":
        if self._entered:
            raise TraceValidationError("Trace span cannot be reused.")
        self._entered = True
        parent = self._tracer._current.get()
        self._trace_id = (
            parent.trace_id if parent is not None else self._tracer._new_trace_id()
        )
        self._span_id = self._tracer._new_span_id()
        self._parent_span_id = None if parent is None else parent.span_id
        self._started_at_ns = self._tracer._wall_clock_ns()
        self._started_monotonic_ns = self._tracer._monotonic_clock_ns()
        self._token = self._tracer._current.set(
            _ActiveSpan(self._trace_id, self._span_id)
        )
        return self

    def set_attribute(self, key: str, value: object) -> None:
        if not self._entered or self._finished:
            raise TraceValidationError("Trace attributes require an active span.")
        updated: dict[str, object] = dict(self._attributes)
        updated[key] = value
        self._attributes = sanitize_trace_attributes(updated)

    def set_error(self, error_type: str) -> None:
        if not self._entered or self._finished:
            raise TraceValidationError(
                "Trace errors require an active span and error type."
            )
        validated_error_type = _validate_error_type(error_type)
        self._status = TraceStatus.ERROR
        self._error_type = validated_error_type

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: object,
    ) -> None:
        if error is not None:
            self.set_error(type(error).__name__)
        self._finish()

    def _finish(self) -> None:
        if not self._entered or self._finished:
            raise TraceValidationError("Trace span is not active.")
        self._finished = True
        ended = self._tracer._monotonic_clock_ns()
        duration = max(0, ended - self._started_monotonic_ns)
        if self._token is not None:
            self._tracer._current.reset(self._token)
        record = TraceRecord(
            trace_id=self._trace_id,
            span_id=self._span_id,
            parent_span_id=self._parent_span_id,
            name=self._name,
            category=self._category,
            started_at_ns=self._started_at_ns,
            duration_ns=duration,
            status=self._status,
            attributes=self._attributes,
            error_type=self._error_type,
        )
        self._tracer._emit(record)


class Tracer:
    """Provider-neutral nested span tracer with safe local metadata."""

    def __init__(
        self,
        collector: TraceCollector,
        *,
        trace_id_factory: TraceIdFactory | None = None,
        span_id_factory: TraceIdFactory | None = None,
        wall_clock_ns: Clock = time_ns,
        monotonic_clock_ns: Clock = perf_counter_ns,
    ) -> None:
        if not hasattr(collector, "emit") or not callable(collector.emit):
            raise TraceValidationError("Trace collector is invalid.")
        if trace_id_factory is not None and not callable(trace_id_factory):
            raise TraceValidationError("Trace ID factory is invalid.")
        if span_id_factory is not None and not callable(span_id_factory):
            raise TraceValidationError("Span ID factory is invalid.")
        if not callable(wall_clock_ns) or not callable(monotonic_clock_ns):
            raise TraceValidationError("Trace clock is invalid.")
        self._collector = collector
        self._trace_id_factory = trace_id_factory or (lambda: token_hex(16))
        self._span_id_factory = span_id_factory or (lambda: token_hex(8))
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_clock_ns = monotonic_clock_ns
        self._current: ContextVar[_ActiveSpan | None] = ContextVar(
            f"ai_sdk_trace_{id(self)}",
            default=None,
        )

    def span(
        self,
        name: str,
        category: TraceCategory,
        attributes: Mapping[str, object] | None = None,
    ) -> TraceSpan:
        return TraceSpan(self, name, category, attributes)

    def _new_trace_id(self) -> str:
        return _validate_trace_id(self._trace_id_factory())

    def _new_span_id(self) -> str:
        return _validate_span_id(self._span_id_factory())

    def _emit(self, record: TraceRecord) -> None:
        try:
            self._collector.emit(record)
        except Exception:
            pass


@contextmanager
def trace_span(
    tracer: Tracer | None,
    name: str,
    category: TraceCategory,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[TraceSpan | None]:
    if tracer is None:
        yield None
        return
    with tracer.span(name, category, attributes) as span:
        yield span
