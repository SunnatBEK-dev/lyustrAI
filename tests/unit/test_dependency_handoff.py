import json

import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTaskStatus,
    AgentTextBlock,
    AgentWorker,
    CoordinationError,
    DependencyHandoffCoordinator,
    DependencyHandoffResult,
    HandoffOutputFormat,
    HandoffStage,
    MultiAgentCoordinator,
    SequentialHandoffCoordinator,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import ToolExecutor, ToolRegistry


class RecordingClient(BaseToolLLMClient):
    def __init__(self, response, error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        self.prompts.append(messages[0]["content"])
        if self.error is not None:
            raise self.error
        return AgentModelResponse(
            [
                AgentTextBlock(self.response),
            ]
        )


def worker(name, client):
    return AgentWorker(
        name,
        f"{name} responsibility",
        AgentRunner(
            client,
            ToolExecutor(ToolRegistry()),
        ),
    )


def payload(summary, *, facts=()):
    return json.dumps(
        {
            "summary": summary,
            "facts": list(facts),
            "uncertainties": [],
            "recommendations": [],
        }
    )


def test_dependency_workflow_uses_topological_order_and_inputs():
    context = RecordingClient(
        payload(
            "Context summary",
            facts=["Fact A"],
        )
    )
    reasoning = RecordingClient(
        payload(
            "Reasoning summary",
            facts=["Fact A"],
        )
    )
    final = RecordingClient("Final answer")
    workers = [
        worker("writer", final),
        worker("reasoner", reasoning),
        worker("context", context),
    ]
    stages = [
        HandoffStage(
            "final",
            "writer",
            "Write answer",
            depends_on=("context", "reasoning"),
        ),
        HandoffStage(
            "reasoning",
            "reasoner",
            "Analyze",
            output_format=HandoffOutputFormat.STRUCTURED,
            depends_on=("context",),
        ),
        HandoffStage(
            "context",
            "context",
            "Extract",
            output_format=HandoffOutputFormat.STRUCTURED,
        ),
    ]
    workflow = DependencyHandoffCoordinator(
        MultiAgentCoordinator(workers),
        stages,
    )

    result = workflow.run("Solve")

    assert result.completed is True
    assert result.final_output == "Final answer"
    assert result.blocked_stage_ids == ()
    assert [stage.id for stage in workflow.execution_stages] == [
        "context",
        "reasoning",
        "final",
    ]
    assert [stage.stage.id for stage in result.stages] == [
        "context",
        "reasoning",
        "final",
    ]
    assert '"stage_id":"context"' in reasoning.prompts[0]
    assert '"stage_id":"context"' in final.prompts[0]
    assert '"stage_id":"reasoning"' in final.prompts[0]
    assert "Required dependency handoffs" in final.prompts[0]


def test_failed_branch_blocks_dependents_but_not_independent_stage():
    failed = RecordingClient(
        "unused",
        RuntimeError("private failure"),
    )
    independent = RecordingClient("Independent output")
    final = RecordingClient("must not run")
    workers = [
        worker("failed", failed),
        worker("independent", independent),
        worker("writer", final),
    ]
    workflow = DependencyHandoffCoordinator(
        MultiAgentCoordinator(workers),
        [
            HandoffStage("failed", "failed", "Fail"),
            HandoffStage(
                "independent",
                "independent",
                "Continue",
            ),
            HandoffStage(
                "final",
                "writer",
                "Write",
                depends_on=("failed", "independent"),
            ),
        ],
    )

    result = workflow.run("Run")

    assert result.completed is False
    assert result.failed_stage.stage.id == "failed"
    assert result.blocked_stage_ids == ("final",)
    assert [stage.stage.id for stage in result.stages] == [
        "failed",
        "independent",
    ]
    assert result.stages[1].task_result.status is AgentTaskStatus.COMPLETED
    assert independent.prompts
    assert final.prompts == []


def test_oversized_dependency_input_fails_before_worker_runs():
    source = RecordingClient("large dependency output")
    target = RecordingClient("must not run")
    workflow = DependencyHandoffCoordinator(
        MultiAgentCoordinator(
            [
                worker("source", source),
                worker("target", target),
            ]
        ),
        [
            HandoffStage("source", "source", "Produce"),
            HandoffStage(
                "target",
                "target",
                "Consume",
                depends_on=("source",),
            ),
        ],
        max_handoff_chars=10,
    )

    result = workflow.run("Run")

    assert result.completed is False
    assert result.failed_stage.stage.id == "target"
    assert result.failed_stage.task_result.error == (
        "Dependency handoff validation failed."
    )
    assert result.blocked_stage_ids == ()
    assert target.prompts == []


def test_dependency_graph_rejects_unknown_nodes_and_cycles():
    coordinator = MultiAgentCoordinator(
        [
            worker("worker", RecordingClient("done")),
        ]
    )

    with pytest.raises(CoordinationError, match="Unknown"):
        DependencyHandoffCoordinator(
            coordinator,
            [
                HandoffStage(
                    "stage",
                    "worker",
                    "Run",
                    depends_on=("missing",),
                )
            ],
        )
    with pytest.raises(CoordinationError, match="cycle"):
        DependencyHandoffCoordinator(
            coordinator,
            [
                HandoffStage(
                    "one",
                    "worker",
                    "Run",
                    depends_on=("two",),
                ),
                HandoffStage(
                    "two",
                    "worker",
                    "Run",
                    depends_on=("one",),
                ),
            ],
        )


def test_dependency_stage_and_result_validation():
    coordinator = MultiAgentCoordinator(
        [
            worker("worker", RecordingClient("done")),
        ]
    )
    dependent = HandoffStage(
        "dependent",
        "worker",
        "Run",
        depends_on=("source",),
    )

    with pytest.raises(CoordinationError, match="cannot declare"):
        SequentialHandoffCoordinator(
            coordinator,
            [dependent],
        )
    with pytest.raises(TypeError, match="sequence"):
        HandoffStage(
            "stage",
            "worker",
            "Run",
            depends_on="source",
        )
    with pytest.raises(CoordinationError, match="unique"):
        HandoffStage(
            "stage",
            "worker",
            "Run",
            depends_on=("source", "source"),
        )
    with pytest.raises(CoordinationError, match="itself"):
        HandoffStage(
            "stage",
            "worker",
            "Run",
            depends_on=("stage",),
        )
    with pytest.raises(TypeError, match="non-empty"):
        DependencyHandoffResult([], 1, [None])
    with pytest.raises(CoordinationError, match="unique"):
        DependencyHandoffResult([], 2, ["one", "one"])
    executed = (
        SequentialHandoffCoordinator(
            coordinator,
            [HandoffStage("stage", "worker", "Run")],
        )
        .run("Run")
        .stages[0]
    )
    with pytest.raises(CoordinationError, match="also be blocked"):
        DependencyHandoffResult([executed], 2, ["stage"])
    with pytest.raises(CoordinationError, match="stage count"):
        DependencyHandoffResult([executed], 1, ["other"])


@pytest.mark.parametrize("input_value", ["", "   ", None])
def test_dependency_workflow_rejects_empty_request(input_value):
    workflow = DependencyHandoffCoordinator(
        MultiAgentCoordinator(
            [
                worker("worker", RecordingClient("done")),
            ]
        ),
        [HandoffStage("stage", "worker", "Run")],
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        workflow.run(input_value)
