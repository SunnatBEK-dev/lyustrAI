import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    CapabilityRouter,
    HandoffStage,
    MultiAgentCoordinator,
    MultiModelRoute,
    RoutingSignal,
    SequentialHandoffCoordinator,
)
from ai_sdk.llm.adaptive_metrics import (
    AdaptiveMetricsValidationError,
    AdaptiveRunMetric,
    InMemoryAdaptiveMetrics,
)
from ai_sdk.llm.adaptive_multi_model import (
    AdaptiveMultiModelClient,
    MultiModelWorkflowClient,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import ToolExecutor, ToolRegistry


class MetricsClient(BaseToolLLMClient):
    def __init__(self, response="done", error=None):
        self.response = response
        self.error = error

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        if self.error is not None:
            raise self.error
        return AgentModelResponse([AgentTextBlock(self.response)])


def workflow(provider):
    worker = AgentWorker(
        "worker",
        "Answer",
        AgentRunner(
            provider,
            ToolExecutor(ToolRegistry()),
        ),
    )
    return MultiModelWorkflowClient(
        SequentialHandoffCoordinator(
            MultiAgentCoordinator([worker]),
            [HandoffStage("final", "worker", "Answer")],
        )
    )


def adaptive_client(fast_provider, *, clock_ns):
    workflows = {
        route: workflow(
            fast_provider
            if route is MultiModelRoute.FAST
            else MetricsClient(route.value)
        )
        for route in MultiModelRoute
    }
    return AdaptiveMultiModelClient(
        CapabilityRouter(),
        workflows,
        clock_ns=clock_ns,
    )


def test_adaptive_client_collects_content_free_success_metrics():
    client = adaptive_client(
        MetricsClient("private answer"),
        clock_ns=iter([1_000_000, 4_000_000]).__next__,
    )

    assert (
        client.ask(
            [
                {
                    "role": "user",
                    "content": "private question",
                }
            ]
        )
        == "private answer"
    )

    metric = client.metrics.records()[0]
    assert metric.route is MultiModelRoute.FAST
    assert metric.signals == ()
    assert metric.completed is True
    assert metric.expected_stage_count == 1
    assert metric.executed_stage_ids == ("final",)
    assert metric.duration_ms == 3.0
    report = client.metrics.report()
    assert report.total_runs == 1
    assert report.successful_runs == 1
    assert report.route_counts == {"fast": 1}
    assert report.stage_execution_counts == {"final": 1}
    assert "private question" not in str(report.to_dict())
    assert "private answer" not in str(report.to_dict())


def test_adaptive_client_records_failed_stage_without_error_message():
    client = adaptive_client(
        MetricsClient(error=RuntimeError("private failure")),
        clock_ns=iter([10, 20]).__next__,
    )

    with pytest.raises(RuntimeError, match="stage: final"):
        client.ask([{"role": "user", "content": "hello"}])

    metric = client.metrics.records()[0]
    assert metric.completed is False
    assert metric.failed_stage_ids == ("final",)
    assert metric.error_type == "RuntimeError"
    report = client.metrics.report()
    assert report.failed_runs == 1
    assert report.stage_failure_counts == {"final": 1}
    assert "private failure" not in str(report.to_dict())


def test_broken_metrics_clock_never_changes_business_result():
    def broken_clock():
        raise RuntimeError("clock unavailable")

    client = adaptive_client(MetricsClient("answer"), clock_ns=broken_clock)

    assert (
        client.ask(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )
        == "answer"
    )
    assert client.metrics.records()[0].duration_ns == 0

    invalid_clock_client = adaptive_client(
        MetricsClient("answer"),
        clock_ns=lambda: "invalid",
    )
    assert (
        invalid_clock_client.ask(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )
        == "answer"
    )
    assert invalid_clock_client.metrics.records()[0].duration_ns == 0


def test_metrics_failures_and_unusual_errors_are_contained():
    class FailingMetrics(InMemoryAdaptiveMetrics):
        def record(self, metric):
            raise RuntimeError("metrics unavailable")

    template = adaptive_client(MetricsClient(), clock_ns=lambda: 1)
    client = AdaptiveMultiModelClient(
        CapabilityRouter(),
        template.workflows,
        metrics=FailingMetrics(),
        clock_ns=lambda: 1,
    )

    assert (
        client.ask(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )
        == "done"
    )

    unusual_error = type("invalid-error-name!", (Exception,), {})
    assert (
        AdaptiveMultiModelClient._error_type(unusual_error("private"))
        == "AdaptiveMultiModelError"
    )


def test_metrics_are_bounded_clearable_and_report_blocked_stages():
    metrics = InMemoryAdaptiveMetrics(max_records=1)
    metrics.record(
        AdaptiveRunMetric(
            route=MultiModelRoute.FULL,
            signals=(RoutingSignal.LONG_REQUEST,),
            expected_stage_count=3,
            executed_stage_ids=("context",),
            failed_stage_ids=("context",),
            blocked_stage_ids=("reasoning", "final"),
            duration_ns=2_000_000,
            completed=False,
            error_type="RuntimeError",
        )
    )
    blocked_report = metrics.report()
    assert blocked_report.stage_blocked_counts == {
        "reasoning": 1,
        "final": 1,
    }
    metrics.record(
        AdaptiveRunMetric(
            route=MultiModelRoute.FAST,
            signals=(),
            expected_stage_count=1,
            executed_stage_ids=("final",),
            failed_stage_ids=(),
            blocked_stage_ids=(),
            duration_ns=1_000_000,
            completed=True,
        )
    )

    assert len(metrics.records()) == 1
    assert metrics.report().route_counts == {"fast": 1}
    metrics.clear()
    assert metrics.report().total_runs == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: InMemoryAdaptiveMetrics(0),
        lambda: InMemoryAdaptiveMetrics().record(object()),
        lambda: AdaptiveRunMetric("fast", (), 1, (), (), (), 0, False),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            ("invalid",),
            1,
            (),
            (),
            (),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (
                RoutingSignal.LONG_REQUEST,
                RoutingSignal.LONG_REQUEST,
            ),
            1,
            (),
            (),
            (),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(MultiModelRoute.FAST, (), 0, (), (), (), 0, False),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            ("final",),
            ("other",),
            (),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            ("final",),
            (),
            ("final",),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            ("first",),
            (),
            ("second",),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            (),
            (),
            (),
            -1,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            (),
            (),
            (),
            0,
            "yes",
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            (),
            (),
            (),
            0,
            False,
            "invalid error",
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            "final",
            (),
            (),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            ("invalid id",),
            (),
            (),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            2,
            ("final", "final"),
            (),
            (),
            0,
            False,
        ),
        lambda: AdaptiveRunMetric(
            MultiModelRoute.FAST,
            (),
            1,
            ("final",),
            (),
            (),
            0,
            True,
            "RuntimeError",
        ),
    ],
)
def test_metrics_contract_rejects_invalid_data(factory):
    with pytest.raises(AdaptiveMetricsValidationError):
        factory()


def test_adaptive_client_validates_metrics_dependencies():
    client = adaptive_client(MetricsClient(), clock_ns=lambda: 1)

    with pytest.raises(TypeError, match="metrics collector"):
        AdaptiveMultiModelClient(
            CapabilityRouter(),
            client.workflows,
            metrics=object(),
        )
    with pytest.raises(TypeError, match="metrics clock"):
        AdaptiveMultiModelClient(
            CapabilityRouter(),
            client.workflows,
            clock_ns=object(),
        )
