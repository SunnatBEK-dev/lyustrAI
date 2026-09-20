from threading import Event, Thread

import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    CancellationToken,
    CapabilityRouter,
    DependencyHandoffCoordinator,
    HandoffStage,
    MultiAgentCoordinator,
    MultiModelRoute,
    SequentialHandoffCoordinator,
    WorkflowCancelledError,
    WorkflowProgressEvent,
    WorkflowProgressReporter,
    WorkflowProgressStatus,
)
from ai_sdk.llm.adaptive_multi_model import (
    AdaptiveMultiModelClient,
    MultiModelWorkflowClient,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import ToolExecutor, ToolRegistry


class ProgressClient(BaseToolLLMClient):
    def __init__(
        self,
        response="done",
        *,
        error=None,
        started=None,
        release=None,
    ):
        self.response = response
        self.error = error
        self.started = started
        self.release = release
        self.calls = 0

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        self.calls += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=3)
        if self.error is not None:
            raise self.error
        return AgentModelResponse([AgentTextBlock(self.response)])


def worker(name, client):
    return AgentWorker(
        name,
        "Progress test worker",
        AgentRunner(client, ToolExecutor(ToolRegistry())),
    )


def sequential_workflow(client, stage_ids=("final",)):
    agent = worker("worker", client)
    return MultiModelWorkflowClient(
        SequentialHandoffCoordinator(
            MultiAgentCoordinator([agent]),
            [HandoffStage(stage_id, "worker", "Run stage") for stage_id in stage_ids],
        )
    )


def adaptive_client_with(reasoning_workflow, *, progress_handler=None):
    return AdaptiveMultiModelClient(
        CapabilityRouter(),
        {
            MultiModelRoute.FAST: sequential_workflow(ProgressClient("fast")),
            MultiModelRoute.CONTEXT: sequential_workflow(ProgressClient("context")),
            MultiModelRoute.REASONING: reasoning_workflow,
            MultiModelRoute.FULL: sequential_workflow(ProgressClient("full")),
        },
        progress_handler=progress_handler,
    )


def test_adaptive_run_emits_ordered_content_free_progress():
    events = []
    client = adaptive_client_with(
        sequential_workflow(ProgressClient("answer")),
        progress_handler=events.append,
    )

    response = client.ask(
        [
            {
                "role": "user",
                "content": "private question",
            }
        ]
    )

    assert response == "fast"
    assert [event.status for event in events] == [
        WorkflowProgressStatus.ROUTE_SELECTED,
        WorkflowProgressStatus.STAGE_STARTED,
        WorkflowProgressStatus.STAGE_COMPLETED,
        WorkflowProgressStatus.WORKFLOW_COMPLETED,
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert events[-1] is client.last_progress_event
    assert client.cancel() is False
    assert "private question" not in str([event.to_dict() for event in events])


def test_active_run_can_be_cancelled_before_the_next_stage():
    started = Event()
    release = Event()
    provider = ProgressClient(
        "first output",
        started=started,
        release=release,
    )
    events = []
    provider_worker = worker("worker", provider)
    cancellable_workflow = MultiModelWorkflowClient(
        DependencyHandoffCoordinator(
            MultiAgentCoordinator([provider_worker]),
            [
                HandoffStage("first", "worker", "First"),
                HandoffStage(
                    "second",
                    "worker",
                    "Second",
                    depends_on=("first",),
                ),
            ],
        )
    )
    client = adaptive_client_with(
        cancellable_workflow,
        progress_handler=events.append,
    )
    outcome = {}

    def run():
        try:
            client.ask(
                [
                    {
                        "role": "user",
                        "content": "Nega bu ishlaydi?",
                    }
                ]
            )
        except Exception as error:
            outcome["error"] = error

    thread = Thread(target=run)
    thread.start()
    assert started.wait(timeout=3)

    with pytest.raises(RuntimeError, match="Concurrent"):
        client.ask([{"role": "user", "content": "Salom"}])
    assert client.cancel() is True
    assert client.cancel() is False
    release.set()
    thread.join(timeout=3)

    assert thread.is_alive() is False
    assert isinstance(outcome["error"], WorkflowCancelledError)
    assert provider.calls == 1
    assert client.last_result is not None
    assert client.last_result.stages == ()
    assert [event.status for event in events] == [
        WorkflowProgressStatus.ROUTE_SELECTED,
        WorkflowProgressStatus.STAGE_STARTED,
        WorkflowProgressStatus.WORKFLOW_CANCELLED,
    ]
    assert events[-1].status is (WorkflowProgressStatus.WORKFLOW_CANCELLED)
    metric = client.metrics.records()[0]
    assert metric.completed is False
    assert metric.executed_stage_ids == ()
    assert metric.error_type == "WorkflowCancelledError"
    assert client.cancel() is False


def test_cancelled_token_stops_before_first_stage():
    provider = ProgressClient()
    workflow = sequential_workflow(provider, ("first", "second"))
    cancellation = CancellationToken()
    cancellation.cancel()

    with pytest.raises(WorkflowCancelledError) as captured:
        workflow.ask_with_control(
            [{"role": "user", "content": "Question"}],
            cancellation=cancellation,
        )

    assert provider.calls == 0
    assert workflow.last_result is captured.value.partial_result
    assert workflow.last_result.stages == ()


def test_dependency_progress_reports_failed_and_blocked_stages():
    provider = ProgressClient(error=RuntimeError("private"))
    coordinator = MultiAgentCoordinator([worker("worker", provider)])
    workflow = DependencyHandoffCoordinator(
        coordinator,
        [
            HandoffStage("first", "worker", "Fail"),
            HandoffStage(
                "second",
                "worker",
                "Blocked",
                depends_on=("first",),
            ),
        ],
    )
    events = []
    reporter = WorkflowProgressReporter("full", events.append)

    result = workflow.run("Question", progress=reporter)

    assert result.completed is False
    assert result.blocked_stage_ids == ("second",)
    assert [event.status for event in events] == [
        WorkflowProgressStatus.STAGE_STARTED,
        WorkflowProgressStatus.STAGE_FAILED,
        WorkflowProgressStatus.STAGE_BLOCKED,
    ]
    assert provider.calls == 1


def test_progress_observer_failure_never_changes_result():
    def broken_handler(event):
        raise RuntimeError("observer unavailable")

    client = adaptive_client_with(
        sequential_workflow(ProgressClient()),
        progress_handler=broken_handler,
    )

    assert (
        client.ask(
            [
                {
                    "role": "user",
                    "content": "Salom",
                }
            ]
        )
        == "fast"
    )
    assert client.last_progress_event.status is (
        WorkflowProgressStatus.WORKFLOW_COMPLETED
    )


def test_progress_models_validate_contracts():
    token = CancellationToken()
    assert token.is_cancelled is False
    assert token.cancel() is True
    assert token.cancel() is False
    assert token.is_cancelled is True
    assert token.wait(0) is True
    with pytest.raises(ValueError, match="timeout"):
        token.wait(-1)
    with pytest.raises(ValueError, match="timeout"):
        token.wait(float("nan"))
    with pytest.raises(ValueError, match="timeout"):
        token.wait(float("inf"))

    with pytest.raises(TypeError, match="handler"):
        WorkflowProgressReporter("fast", object())
    with pytest.raises(ValueError, match="route"):
        WorkflowProgressReporter("bad route")
    with pytest.raises(TypeError, match="status"):
        WorkflowProgressEvent(1, "started", "fast", 0, 1)
    with pytest.raises(ValueError, match="sequence"):
        WorkflowProgressEvent(
            0,
            WorkflowProgressStatus.ROUTE_SELECTED,
            "fast",
            0,
            1,
        )
    with pytest.raises(ValueError, match="stage ID"):
        WorkflowProgressEvent(
            1,
            WorkflowProgressStatus.STAGE_STARTED,
            "fast",
            0,
            1,
        )
    with pytest.raises(ValueError, match="counts"):
        WorkflowProgressEvent(
            1,
            WorkflowProgressStatus.ROUTE_SELECTED,
            "fast",
            2,
            1,
        )


def test_handoff_rejects_invalid_progress_control():
    workflow = sequential_workflow(ProgressClient()).workflow

    with pytest.raises(TypeError, match="reporter"):
        workflow.run("Question", progress=object())
    with pytest.raises(TypeError, match="token"):
        workflow.run("Question", cancellation=object())
    with pytest.raises(TypeError, match="cancellation token"):
        MultiAgentCoordinator().run([], cancellation=object())


def test_adaptive_client_rejects_invalid_progress_handler():
    template = adaptive_client_with(sequential_workflow(ProgressClient()))

    with pytest.raises(TypeError, match="progress handler"):
        AdaptiveMultiModelClient(
            CapabilityRouter(),
            template.workflows,
            progress_handler=object(),
        )
    assert AdaptiveMultiModelClient._completed_stage_count(None) == 0
