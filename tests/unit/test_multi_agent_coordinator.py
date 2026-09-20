import pytest

import ai_sdk.llm.factory as llm_factory_module
from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentStopReason,
    AgentTask,
    AgentTaskResult,
    AgentTaskStatus,
    AgentTextBlock,
    AgentWorker,
    CoordinationError,
    CoordinationResult,
    MultiAgentCoordinator,
    create_provider_worker,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.retry import RetryPolicy
from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)


class RecordingClient(BaseToolLLMClient):
    def __init__(self, prefix="Done", error=None):
        self.prefix = prefix
        self.error = error
        self.turns = []

    def ask(self, messages):
        return self.prefix

    def stream(self, messages):
        yield self.prefix

    def complete_tool_turn(self, messages, schemas, events):
        self.turns.append((messages, schemas, events))
        if self.error:
            raise self.error
        return AgentModelResponse(
            [
                AgentTextBlock(f"{self.prefix} {len(self.turns)}"),
            ]
        )


class ScriptedClient(RecordingClient):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)

    def complete_tool_turn(self, messages, schemas, events):
        self.turns.append((messages, schemas, events))
        return self.responses.pop(0)


def empty_runner(client=None, *, max_tool_rounds=8):
    return AgentRunner(
        client or RecordingClient(),
        ToolExecutor(ToolRegistry()),
        max_tool_rounds=max_tool_rounds,
    )


def worker(name="researcher", client=None):
    return AgentWorker(
        name=name,
        description=f"{name} responsibilities",
        runner=empty_runner(client),
    )


def test_coordinator_runs_explicit_assignments_in_input_order():
    research_client = RecordingClient("Research")
    writing_client = RecordingClient("Draft")
    coordinator = MultiAgentCoordinator(
        [
            worker("researcher", research_client),
            worker("writer", writing_client),
        ]
    )
    tasks = [
        AgentTask("task_write", "writer", "Write summary"),
        AgentTask(
            "task_research",
            "researcher",
            "Collect facts",
        ),
        AgentTask("task_edit", "writer", "Edit summary"),
    ]

    result = coordinator.run(tasks)

    assert [item.task.id for item in result.results] == [
        "task_write",
        "task_research",
        "task_edit",
    ]
    assert [item.output for item in result.results] == [
        "Draft 1",
        "Research 1",
        "Draft 2",
    ]
    assert len(result.completed) == 3
    assert result.failed == ()
    assert result.results[0].state is not result.results[2].state
    assert writing_client.turns[0][2] == ()
    assert writing_client.turns[1][2] == ()
    assignment = writing_client.turns[0][0][0]["content"]
    assert "Worker: writer" in assignment
    assert "writer responsibilities" in assignment
    assert "Assigned task: Write summary" in assignment


def test_coordinator_contains_worker_failure_and_continues():
    failing = RecordingClient(error=RuntimeError("private provider detail"))
    healthy = RecordingClient("Healthy")
    coordinator = MultiAgentCoordinator(
        [
            worker("broken", failing),
            worker("healthy", healthy),
        ]
    )

    result = coordinator.run(
        [
            AgentTask("task_bad", "broken", "Fail"),
            AgentTask("task_good", "healthy", "Continue"),
        ]
    )

    assert [item.status for item in result.results] == [
        AgentTaskStatus.FAILED,
        AgentTaskStatus.COMPLETED,
    ]
    assert result.results[0].state is None
    assert result.results[0].error == ("Worker execution failed: RuntimeError")
    assert "private provider detail" not in result.results[0].error
    assert result.results[1].output == "Healthy 1"


def test_coordinator_marks_non_final_agent_stop_as_failed():
    registry = ToolRegistry()
    registry.register(
        ToolSchema(
            name="double",
            description="Double an integer.",
            parameters=[
                ToolParameter(
                    "value",
                    ToolParameterType.INTEGER,
                    "Integer to double.",
                ),
            ],
        ),
        lambda value: value * 2,
    )
    client = ScriptedClient(
        [
            AgentModelResponse(
                [
                    ToolCall("call_1", "double", {"value": 1}),
                ]
            ),
            AgentModelResponse(
                [
                    ToolCall("call_2", "double", {"value": 2}),
                ]
            ),
        ]
    )
    runner = AgentRunner(
        client,
        ToolExecutor(registry),
        max_tool_rounds=1,
    )
    coordinator = MultiAgentCoordinator(
        [
            AgentWorker("worker", "Run tools", runner),
        ]
    )

    result = coordinator.run(
        [
            AgentTask("task_1", "worker", "Keep doubling"),
        ]
    )

    task_result = result.results[0]
    assert task_result.status is AgentTaskStatus.FAILED
    assert task_result.state.stop_reason is (AgentStopReason.MAX_TOOL_ROUNDS)
    assert task_result.error == "Worker stopped: max_tool_rounds"


def test_coordinator_preflights_all_assignments_before_execution():
    client = RecordingClient()
    coordinator = MultiAgentCoordinator(
        [
            worker("known", client),
        ]
    )

    with pytest.raises(CoordinationError, match="Unknown"):
        coordinator.run(
            [
                AgentTask("task_1", "known", "Would run"),
                AgentTask("task_2", "missing", "Cannot run"),
            ]
        )

    assert client.turns == []

    with pytest.raises(CoordinationError, match="unique"):
        coordinator.run(
            [
                AgentTask("same", "known", "First"),
                AgentTask("same", "known", "Second"),
            ]
        )

    assert client.turns == []


def test_coordinator_registers_unique_named_workers():
    coordinator = MultiAgentCoordinator()
    coordinator.register(worker("researcher"))

    assert coordinator.worker_names() == ("researcher",)
    assert coordinator.count() == 1
    assert coordinator.run([]).results == ()

    with pytest.raises(CoordinationError, match="already"):
        coordinator.register(worker("Researcher"))

    with pytest.raises(TypeError, match="AgentWorker"):
        coordinator.register(object())


def test_provider_worker_factory_binds_provider_and_executor(
    monkeypatch,
):
    client = RecordingClient("Gemini")
    created_for = []

    def create_client(provider):
        created_for.append(provider)
        return client

    monkeypatch.setattr(
        llm_factory_module,
        "create_llm_client",
        create_client,
    )
    executor = ToolExecutor(ToolRegistry())
    retry_policy = RetryPolicy(max_attempts=2)

    provider_worker = create_provider_worker(
        "researcher",
        "Collect facts",
        " Gemini ",
        executor,
        max_tool_rounds=3,
        retry_policy=retry_policy,
    )

    assert created_for == ["gemini"]
    assert provider_worker.provider == "gemini"
    assert provider_worker.runner.client is client
    assert provider_worker.runner.executor is executor
    assert provider_worker.runner.max_tool_rounds == 3
    assert provider_worker.runner.retry_policy is retry_policy


def test_provider_worker_factory_can_use_empty_tool_registry(
    monkeypatch,
):
    client = RecordingClient("OpenAI")
    monkeypatch.setattr(
        llm_factory_module,
        "create_llm_client",
        lambda provider: client,
    )

    provider_worker = create_provider_worker(
        "writer",
        "Write a draft",
        "openai",
    )

    assert provider_worker.provider == "openai"
    assert provider_worker.runner.executor.registry.count() == 0


def test_provider_worker_factory_rejects_invalid_executor(
    monkeypatch,
):
    called = False

    def create_client(provider):
        nonlocal called
        called = True
        return RecordingClient()

    monkeypatch.setattr(
        llm_factory_module,
        "create_llm_client",
        create_client,
    )

    with pytest.raises(TypeError, match="executor"):
        create_provider_worker(
            "worker",
            "Do work",
            "anthropic",
            object(),
        )

    assert called is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AgentWorker("bad name", "Role", empty_runner()),
        lambda: AgentWorker("worker", " ", empty_runner()),
        lambda: AgentWorker("worker", "Role", object()),
        lambda: AgentWorker(
            "worker",
            "Role",
            empty_runner(),
            provider="bad provider",
        ),
        lambda: AgentTask("bad id", "worker", "Task"),
        lambda: AgentTask("task", "bad worker", "Task"),
        lambda: AgentTask("task", "worker", " "),
    ],
)
def test_worker_and_task_reject_invalid_fields(factory):
    with pytest.raises((CoordinationError, TypeError)):
        factory()


def test_coordination_result_exposes_lookup_and_partitions():
    completed_state = empty_runner().run(
        [
            {"role": "user", "content": "Complete"},
        ]
    )
    completed_task = AgentTask(
        "task_ok",
        "worker",
        "Complete",
    )
    failed_task = AgentTask(
        "task_failed",
        "worker",
        "Fail",
    )
    completed = AgentTaskResult(
        completed_task,
        AgentTaskStatus.COMPLETED,
        state=completed_state,
    )
    failed = AgentTaskResult(
        failed_task,
        AgentTaskStatus.FAILED,
        error="Worker execution failed: RuntimeError",
    )
    result = CoordinationResult([completed, failed])

    assert result.completed == (completed,)
    assert result.failed == (failed,)
    assert failed.output is None
    assert result.for_task("task_ok") is completed
    assert result.for_task("missing") is None


def test_result_models_reject_inconsistent_values():
    task = AgentTask("task", "worker", "Run")

    with pytest.raises(TypeError, match="task must"):
        AgentTaskResult(
            object(),
            AgentTaskStatus.FAILED,
            error="failed",
        )

    with pytest.raises(TypeError, match="status"):
        AgentTaskResult(task, "failed", error="failed")

    with pytest.raises(TypeError, match="state"):
        AgentTaskResult(
            task,
            AgentTaskStatus.FAILED,
            state=object(),
            error="failed",
        )

    with pytest.raises(CoordinationError, match="non-empty"):
        AgentTaskResult(
            task,
            AgentTaskStatus.FAILED,
            error=" ",
        )

    with pytest.raises(CoordinationError, match="inconsistent"):
        AgentTaskResult(task, AgentTaskStatus.COMPLETED)

    with pytest.raises(CoordinationError, match="contain an error"):
        AgentTaskResult(task, AgentTaskStatus.FAILED)

    with pytest.raises(TypeError, match="Coordination results"):
        CoordinationResult([object()])

    failed = AgentTaskResult(
        task,
        AgentTaskStatus.FAILED,
        error="failed",
    )

    with pytest.raises(CoordinationError, match="unique"):
        CoordinationResult([failed, failed])


def test_coordinator_rejects_non_task_input():
    coordinator = MultiAgentCoordinator([worker()])

    with pytest.raises(TypeError, match="AgentTask"):
        coordinator.run([object()])
