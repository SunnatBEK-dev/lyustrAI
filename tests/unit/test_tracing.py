from itertools import cycle

import pytest

from ai_sdk.observability import (
    InMemoryTraceCollector,
    TraceCategory,
    Tracer,
    TraceRecord,
    TraceStatus,
    TraceValidationError,
    sanitize_trace_attributes,
    trace_span,
)

TRACE_ID = "1" * 32
OTHER_TRACE_ID = "2" * 32
ROOT_SPAN_ID = "a" * 16
CHILD_SPAN_ID = "b" * 16


def deterministic_tracer(collector=None):
    trace_ids = iter([TRACE_ID, OTHER_TRACE_ID])
    span_ids = iter([ROOT_SPAN_ID, CHILD_SPAN_ID, "c" * 16])
    wall_times = iter([100, 200, 300])
    monotonic_times = iter([10, 20, 50, 90, 100, 110])
    return Tracer(
        collector or InMemoryTraceCollector(),
        trace_id_factory=lambda: next(trace_ids),
        span_id_factory=lambda: next(span_ids),
        wall_clock_ns=lambda: next(wall_times),
        monotonic_clock_ns=lambda: next(monotonic_times),
    )


def test_nested_spans_propagate_context_timing_and_safe_attributes():
    collector = InMemoryTraceCollector()
    tracer = deterministic_tracer(collector)

    with tracer.span(
        "conversation.send",
        TraceCategory.WORKFLOW,
        {
            "prompt": "private question",
            "llm.message_count": 2,
            "custom": "Bearer private-token",
        },
    ) as root:
        with tracer.span(
            "llm.generate",
            TraceCategory.LLM,
        ) as child:
            child.set_attribute("llm.response_char_count", 12)
            child.set_attribute("label", "x" * 200)

    records = collector.records()
    root_record, child_record = records
    assert root.trace_id == child.trace_id == TRACE_ID
    assert root_record.span_id == ROOT_SPAN_ID
    assert root_record.parent_span_id is None
    assert root_record.duration_ns == 80
    assert child_record.parent_span_id == ROOT_SPAN_ID
    assert child_record.duration_ns == 30
    assert root_record.attributes == {
        "prompt": "[REDACTED]",
        "llm.message_count": 2,
        "custom": "[REDACTED]",
    }
    assert child_record.attributes["llm.response_char_count"] == 12
    assert len(child_record.attributes["label"]) == 128
    assert child.span_id == CHILD_SPAN_ID
    assert root_record.status is TraceStatus.OK
    assert root_record.to_dict()["duration_ms"] == 0.00008


def test_exception_records_only_type_and_resets_parent_context():
    collector = InMemoryTraceCollector()
    tracer = deterministic_tracer(collector)

    with pytest.raises(RuntimeError, match="private-api-key"):
        with tracer.span("tool.execute", TraceCategory.TOOL):
            raise RuntimeError("private-api-key")

    with tracer.span("mcp.request", TraceCategory.MCP):
        pass

    failed, next_root = collector.records()
    assert failed.status is TraceStatus.ERROR
    assert failed.error_type == "RuntimeError"
    assert "private-api-key" not in str(failed.to_dict())
    assert next_root.trace_id == OTHER_TRACE_ID
    assert next_root.parent_span_id is None


def test_logical_error_and_bounded_collector_keep_recent_records():
    collector = InMemoryTraceCollector(max_records=2)
    ids = cycle(["1" * 32])
    spans = iter(["1" * 16, "2" * 16, "3" * 16])
    clock = iter([1, 2, 3, 4, 5, 6])
    tracer = Tracer(
        collector,
        trace_id_factory=lambda: next(ids),
        span_id_factory=lambda: next(spans),
        wall_clock_ns=lambda: 1,
        monotonic_clock_ns=lambda: next(clock),
    )

    for index in range(3):
        with tracer.span(
            f"tool.operation_{index}",
            TraceCategory.TOOL,
        ) as span:
            if index == 2:
                span.set_error("ToolExecutionError")

    records = collector.records()
    assert [record.name for record in records] == [
        "tool.operation_1",
        "tool.operation_2",
    ]
    assert records[-1].status is TraceStatus.ERROR

    collector.clear()
    assert collector.records() == ()


def test_collector_validates_records_and_filters_by_trace():
    collector = InMemoryTraceCollector()
    tracer = deterministic_tracer(collector)
    with tracer.span("workflow.one", TraceCategory.WORKFLOW):
        pass
    with tracer.span("workflow.two", TraceCategory.WORKFLOW):
        pass

    assert len(collector.records(trace_id=TRACE_ID)) == 1
    assert len(collector.records(trace_id=OTHER_TRACE_ID)) == 1
    with pytest.raises(TraceValidationError, match="record"):
        collector.emit(object())


def test_collector_failure_never_changes_business_result():
    class FailingCollector:
        def emit(self, record):
            raise RuntimeError("collector unavailable")

    tracer = Tracer(FailingCollector())

    with tracer.span("workflow.run", TraceCategory.WORKFLOW):
        result = 42

    assert result == 42


def test_trace_span_without_tracer_is_a_noop():
    with trace_span(
        None,
        "workflow.run",
        TraceCategory.WORKFLOW,
    ) as span:
        assert span is None


@pytest.mark.parametrize(
    "attributes",
    [
        [],
        {"": 1},
        {"nested": {}},
        {"score": float("nan")},
        {f"key_{index}": index for index in range(33)},
    ],
)
def test_attribute_contract_rejects_unbounded_or_non_scalar_data(
    attributes,
):
    with pytest.raises(TraceValidationError):
        sanitize_trace_attributes(attributes)


def test_sensitive_count_name_requires_a_numeric_value():
    attributes = sanitize_trace_attributes(
        {
            "llm.prompt_token_count": 8,
            "prompt_count": "private prompt disguised as a count",
        }
    )

    assert attributes == {
        "llm.prompt_token_count": 8,
        "prompt_count": "[REDACTED]",
    }


def test_completed_trace_attributes_are_immutable():
    collector = InMemoryTraceCollector()
    tracer = deterministic_tracer(collector)

    with tracer.span(
        "workflow.run",
        TraceCategory.WORKFLOW,
        {"workflow.item_count": 1},
    ):
        pass

    record = collector.records()[0]
    with pytest.raises(TypeError):
        record.attributes["prompt"] = "private"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: InMemoryTraceCollector(max_records=0),
        lambda: Tracer(object()),
        lambda: TraceRecord(
            trace_id="0" * 32,
            span_id=ROOT_SPAN_ID,
            parent_span_id=None,
            name="operation",
            category=TraceCategory.WORKFLOW,
            started_at_ns=0,
            duration_ns=0,
            status=TraceStatus.OK,
        ),
        lambda: TraceRecord(
            trace_id=TRACE_ID,
            span_id=ROOT_SPAN_ID,
            parent_span_id=None,
            name="operation",
            category=TraceCategory.WORKFLOW,
            started_at_ns=0,
            duration_ns=0,
            status=TraceStatus.ERROR,
        ),
        lambda: TraceRecord(
            trace_id=TRACE_ID,
            span_id=ROOT_SPAN_ID,
            parent_span_id=None,
            name="operation",
            category="workflow",
            started_at_ns=0,
            duration_ns=0,
            status=TraceStatus.OK,
        ),
        lambda: TraceRecord(
            trace_id=TRACE_ID,
            span_id=ROOT_SPAN_ID,
            parent_span_id=None,
            name="operation",
            category=TraceCategory.WORKFLOW,
            started_at_ns=-1,
            duration_ns=0,
            status=TraceStatus.OK,
        ),
        lambda: TraceRecord(
            trace_id=TRACE_ID,
            span_id=ROOT_SPAN_ID,
            parent_span_id=None,
            name="operation",
            category=TraceCategory.WORKFLOW,
            started_at_ns=0,
            duration_ns=0,
            status="ok",
        ),
        lambda: TraceRecord(
            trace_id=TRACE_ID,
            span_id=ROOT_SPAN_ID,
            parent_span_id=None,
            name="operation",
            category=TraceCategory.WORKFLOW,
            started_at_ns=0,
            duration_ns=0,
            status=TraceStatus.OK,
            error_type="UnexpectedError",
        ),
        lambda: deterministic_tracer().span("", TraceCategory.LLM),
        lambda: deterministic_tracer().span("ok", "llm"),
        lambda: Tracer(
            InMemoryTraceCollector(),
            trace_id_factory="invalid",
        ),
        lambda: Tracer(
            InMemoryTraceCollector(),
            span_id_factory="invalid",
        ),
        lambda: Tracer(
            InMemoryTraceCollector(),
            wall_clock_ns="invalid",
        ),
    ],
)
def test_trace_contract_rejects_invalid_configuration(factory):
    with pytest.raises(TraceValidationError):
        factory()


def test_active_span_guards_mutation_and_reuse():
    tracer = deterministic_tracer()
    span = tracer.span("llm.generate", TraceCategory.LLM)

    with pytest.raises(TraceValidationError, match="active"):
        span.set_attribute("llm.message_count", 1)
    with span:
        with pytest.raises(TraceValidationError, match="error type"):
            span.set_error("private error detail")
    with pytest.raises(TraceValidationError, match="active"):
        span.set_error("RuntimeError")
    with pytest.raises(TraceValidationError, match="reused"):
        with span:
            pass
