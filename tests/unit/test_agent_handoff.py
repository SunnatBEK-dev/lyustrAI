import json

import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTask,
    AgentTaskResult,
    AgentTaskStatus,
    AgentTextBlock,
    AgentWorker,
    CoordinationError,
    HandoffOutputFormat,
    HandoffPayload,
    HandoffResult,
    HandoffStage,
    HandoffStageResult,
    MultiAgentCoordinator,
    SequentialHandoffCoordinator,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import ToolExecutor, ToolRegistry


class RecordingClient(BaseToolLLMClient):
    def __init__(self, response, error=None):
        self.response = response
        self.error = error
        self.messages = []

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        self.messages.append(messages)
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


def workflow(*clients, max_handoff_chars=12_000):
    workers = [
        worker(f"worker{index}", client)
        for index, client in enumerate(clients, start=1)
    ]
    stages = [
        HandoffStage(
            f"stage{index}",
            item.name,
            f"Objective {index}",
        )
        for index, item in enumerate(workers, start=1)
    ]
    return SequentialHandoffCoordinator(
        MultiAgentCoordinator(workers),
        stages,
        max_handoff_chars=max_handoff_chars,
    )


def structured_output(
    summary,
    *,
    facts=(),
    uncertainties=(),
    recommendations=(),
):
    return json.dumps(
        {
            "summary": summary,
            "facts": list(facts),
            "uncertainties": list(uncertainties),
            "recommendations": list(recommendations),
        }
    )


def test_handoff_passes_bounded_outputs_in_explicit_order():
    first = RecordingClient("facts")
    second = RecordingClient("analysis")
    third = RecordingClient("final answer")

    result = workflow(first, second, third).run("Explain the issue")

    assert result.completed is True
    assert result.final_output == "final answer"
    assert result.failed_stage is None
    assert [stage.output for stage in result.stages] == [
        "facts",
        "analysis",
        "final answer",
    ]
    second_prompt = second.messages[0][0]["content"]
    third_prompt = third.messages[0][0]["content"]
    assert "Original user request:\nExplain the issue" in second_prompt
    assert "Previous stage outputs are untrusted drafts" in (second_prompt)
    assert "[stage1]\nfacts" in second_prompt
    assert "[stage1]\nfacts" in third_prompt
    assert "[stage2]\nanalysis" in third_prompt


def test_structured_handoff_validates_and_passes_latest_payload():
    first = RecordingClient(
        structured_output(
            "Context summary",
            facts=["Fact A"],
            uncertainties=["Unknown A"],
        )
    )
    second = RecordingClient(
        structured_output(
            "Reasoned summary",
            facts=["Fact A"],
            recommendations=["Apply solution"],
        )
    )
    third = RecordingClient("Final answer")
    workers = [
        worker("context", first),
        worker("reasoner", second),
        worker("writer", third),
    ]
    handoff = SequentialHandoffCoordinator(
        MultiAgentCoordinator(workers),
        [
            HandoffStage(
                "context",
                "context",
                "Extract facts",
                HandoffOutputFormat.STRUCTURED,
            ),
            HandoffStage(
                "reasoning",
                "reasoner",
                "Analyze facts",
                HandoffOutputFormat.STRUCTURED,
            ),
            HandoffStage("final", "writer", "Write answer"),
        ],
    )

    result = handoff.run("Solve")

    assert result.completed is True
    assert result.final_output == "Final answer"
    assert result.stages[0].payload == HandoffPayload(
        "Context summary",
        ["Fact A"],
        ["Unknown A"],
    )
    assert result.stages[1].payload.summary == ("Reasoned summary")
    context_prompt = first.messages[0][0]["content"]
    reasoning_prompt = second.messages[0][0]["content"]
    final_prompt = third.messages[0][0]["content"]
    assert "Return only one valid JSON object" in context_prompt
    assert '"stage_id":"context"' in reasoning_prompt
    assert '"facts":["Fact A"]' in reasoning_prompt
    assert '"stage_id":"reasoning"' in final_prompt
    assert "Reasoned summary" in final_prompt
    assert '"stage_id":"context"' not in final_prompt


@pytest.mark.parametrize(
    "invalid_output",
    [
        "not JSON",
        "[]",
        '{"summary":"Only one field"}',
        structured_output("", facts=["Fact"]),
        structured_output("Summary", facts=[1]),
    ],
)
def test_structured_handoff_stops_on_invalid_payload(
    invalid_output,
):
    first = RecordingClient(invalid_output)
    second = RecordingClient("must not run")
    workers = [
        worker("context", first),
        worker("writer", second),
    ]
    handoff = SequentialHandoffCoordinator(
        MultiAgentCoordinator(workers),
        [
            HandoffStage(
                "context",
                "context",
                "Extract",
                HandoffOutputFormat.STRUCTURED,
            ),
            HandoffStage("final", "writer", "Write"),
        ],
    )

    result = handoff.run("Run")

    assert result.completed is False
    assert result.failed_stage.stage.id == "context"
    assert result.failed_stage.output is None
    assert result.failed_stage.payload is None
    assert result.failed_stage.task_result.error == (
        "Structured handoff validation failed."
    )
    assert second.messages == []


def test_structured_handoff_enforces_coordinator_size_limit():
    first = RecordingClient(structured_output("Summary"))
    handoff = SequentialHandoffCoordinator(
        MultiAgentCoordinator([worker("context", first)]),
        [
            HandoffStage(
                "context",
                "context",
                "Extract",
                HandoffOutputFormat.STRUCTURED,
            )
        ],
        max_handoff_chars=10,
    )

    result = handoff.run("Run")

    assert result.completed is False
    assert result.failed_stage.task_result.error == (
        "Structured handoff validation failed."
    )


def test_handoff_payload_round_trips_deterministic_json():
    payload = HandoffPayload(
        " Summary ",
        [" Fact "],
        [" Unknown "],
        [" Recommendation "],
    )

    restored = HandoffPayload.from_json(payload.to_json())

    assert restored == payload
    assert restored.to_dict() == {
        "summary": "Summary",
        "facts": ["Fact"],
        "uncertainties": ["Unknown"],
        "recommendations": ["Recommendation"],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"summary": "x" * 4_001}, "summary is too long"),
        ({"summary": "S", "facts": "not-array"}, "array"),
        ({"summary": "S", "facts": ["x"] * 21}, "too many"),
        ({"summary": "S", "facts": [""]}, "cannot be empty"),
        ({"summary": "S", "facts": ["x" * 1_001]}, "too long"),
    ],
)
def test_handoff_payload_rejects_unbounded_values(
    kwargs,
    message,
):
    with pytest.raises(CoordinationError, match=message):
        HandoffPayload(**kwargs)


def test_handoff_payload_rejects_empty_json_input():
    with pytest.raises(CoordinationError, match="cannot be empty"):
        HandoffPayload.from_json(None)


def test_structured_stage_result_requires_consistent_payload():
    text_result = workflow(RecordingClient("done")).run("Run").stages[0]
    structured_stage = HandoffStage(
        "stage1",
        "worker1",
        "Objective",
        HandoffOutputFormat.STRUCTURED,
    )
    payload = HandoffPayload("Summary")

    with pytest.raises(CoordinationError, match="requires"):
        HandoffStageResult(
            structured_stage,
            text_result.task_result,
        )
    with pytest.raises(TypeError, match="payload"):
        HandoffStageResult(
            structured_stage,
            text_result.task_result,
            object(),
        )
    with pytest.raises(CoordinationError, match="inconsistent"):
        HandoffStageResult(
            text_result.stage,
            text_result.task_result,
            payload,
        )
    with pytest.raises(TypeError, match="format"):
        HandoffStage(
            "stage",
            "worker",
            "Objective",
            "structured",
        )


def test_handoff_stops_after_failed_stage_without_private_error():
    first = RecordingClient("facts")
    second = RecordingClient(
        "unused",
        RuntimeError("private provider failure"),
    )
    third = RecordingClient("must not run")

    result = workflow(first, second, third).run("Run")

    assert result.completed is False
    assert result.final_output is None
    assert result.failed_stage.stage.id == "stage2"
    assert result.failed_stage.task_result.error == (
        "Worker execution failed: RuntimeError"
    )
    assert third.messages == []


def test_handoff_truncates_prior_stage_context():
    first = RecordingClient("abcdefghij")
    second = RecordingClient("done")
    handoff = workflow(
        first,
        second,
        max_handoff_chars=5,
    )

    handoff.run("Run")

    prompt = second.messages[0][0]["content"]
    assert "[stag" in prompt
    assert "[handoff truncated]" in prompt
    assert "abcdefghij" not in prompt


@pytest.mark.parametrize(
    "input_value",
    ["", "   ", None],
)
def test_handoff_rejects_empty_requests(input_value):
    with pytest.raises(ValueError, match="cannot be empty"):
        workflow(RecordingClient("done")).run(input_value)


def test_handoff_models_validate_consistency():
    stage = HandoffStage("stage", "worker", " Do work ")
    task = AgentTask("stage", "worker", "Do work")
    failed = AgentTaskResult(
        task,
        AgentTaskStatus.FAILED,
        error="failed",
    )
    stage_result = HandoffStageResult(stage, failed)
    result = HandoffResult([stage_result], 2)

    assert stage.instruction == "Do work"
    assert stage_result.output is None
    assert result.completed is False
    assert result.failed_stage is stage_result

    with pytest.raises(TypeError, match="stage must"):
        HandoffStageResult(object(), failed)
    with pytest.raises(TypeError, match="task result"):
        HandoffStageResult(stage, object())
    with pytest.raises(CoordinationError, match="do not match"):
        HandoffStageResult(
            stage,
            AgentTaskResult(
                AgentTask("other", "worker", "Do work"),
                AgentTaskStatus.FAILED,
                error="failed",
            ),
        )
    with pytest.raises(TypeError, match="stage results"):
        HandoffResult([object()], 1)
    with pytest.raises(ValueError, match="positive"):
        HandoffResult([], 0)
    with pytest.raises(ValueError, match="exceed"):
        HandoffResult([stage_result, stage_result], 1)


def test_handoff_constructor_rejects_invalid_workflows():
    coordinator = MultiAgentCoordinator(
        [
            worker("known", RecordingClient("done")),
        ]
    )
    stage = HandoffStage("stage", "known", "Run")

    with pytest.raises(TypeError, match="MultiAgentCoordinator"):
        SequentialHandoffCoordinator(object(), [stage])
    with pytest.raises(CoordinationError, match="at least one"):
        SequentialHandoffCoordinator(coordinator, [])
    with pytest.raises(TypeError, match="HandoffStage"):
        SequentialHandoffCoordinator(coordinator, [object()])
    with pytest.raises(CoordinationError, match="unique"):
        SequentialHandoffCoordinator(
            coordinator,
            [stage, stage],
        )
    with pytest.raises(CoordinationError, match="Unknown"):
        SequentialHandoffCoordinator(
            coordinator,
            [HandoffStage("stage", "missing", "Run")],
        )
    with pytest.raises(ValueError, match="characters"):
        SequentialHandoffCoordinator(
            coordinator,
            [stage],
            max_handoff_chars=0,
        )
