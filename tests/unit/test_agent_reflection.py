import copy
import json

import pytest

from ai_sdk.agents import (
    AgentEvent,
    AgentModelResponse,
    AgentPlan,
    AgentReflection,
    AgentState,
    AgentStopReason,
    AgentTextBlock,
    LLMAgentReflector,
    PlanStep,
    ReflectionValidationError,
    ReflectionVerdict,
)
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.tools import ToolCall, ToolResult


class FakeReflectionClient(BaseLLMClient):
    def __init__(self, response):
        self.response = response
        self.calls = []

    def ask(self, messages):
        self.calls.append(messages)
        return self.response

    def stream(self, messages):
        yield self.response


def valid_response(
    verdict="passed",
    summary="The work met its goal.",
    strengths=None,
    improvements=None,
):
    return json.dumps(
        {
            "verdict": verdict,
            "summary": summary,
            "strengths": strengths or ["Clear result"],
            "improvements": improvements or [],
        }
    )


def build_finished_state():
    state = AgentState(
        [
            {"role": "user", "content": "Double two."},
        ]
    )
    call = ToolCall(
        "call_1",
        "double",
        {"value": 2},
    )
    state.record(
        AgentEvent(
            1,
            AgentModelResponse([call]),
            [ToolResult("call_1", "double", "4")],
        )
    )
    state.record(
        AgentEvent(
            2,
            AgentModelResponse([AgentTextBlock("Four")]),
        )
    )
    state.finish(
        AgentStopReason.FINAL_RESPONSE,
        "Four",
    )
    return state


def build_completed_plan():
    plan = AgentPlan(
        "Finish task",
        [
            PlanStep("step_1", "Inspect"),
            PlanStep("step_2", "Implement"),
        ],
    )

    for result in ["Inspected", "Implemented"]:
        plan.start_next()
        plan.complete_current(result)

    return plan


def test_agent_reflection_normalizes_structured_feedback():
    reflection = AgentReflection(
        ReflectionVerdict.NEEDS_IMPROVEMENT,
        " Improve error handling. ",
        strengths=[" Clear flow "],
        improvements=[" Add timeout handling "],
    )

    assert reflection.summary == "Improve error handling."
    assert reflection.strengths == ("Clear flow",)
    assert reflection.improvements == ("Add timeout handling",)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AgentReflection("passed", "Summary"),
        lambda: AgentReflection(ReflectionVerdict.PASSED, " "),
        lambda: AgentReflection(
            ReflectionVerdict.PASSED,
            "Summary",
            strengths="not-a-list",
        ),
        lambda: AgentReflection(
            ReflectionVerdict.PASSED,
            "Summary",
            strengths=["same", " SAME "],
        ),
        lambda: AgentReflection(
            ReflectionVerdict.PASSED,
            "Summary",
            improvements=[2],
        ),
    ],
)
def test_agent_reflection_rejects_invalid_fields(factory):
    with pytest.raises(ReflectionValidationError):
        factory()


def test_reflector_reviews_finished_state_once_without_mutation():
    state = build_finished_state()
    original_messages = copy.deepcopy(state.messages)
    original_events = state.events.copy()
    client = FakeReflectionClient(
        valid_response(
            verdict="needs_improvement",
            improvements=["Explain the tool result"],
        )
    )

    reflection = LLMAgentReflector(client).reflect_state(state)

    assert reflection.verdict is (ReflectionVerdict.NEEDS_IMPROVEMENT)
    assert reflection.improvements == ("Explain the tool result",)
    assert len(client.calls) == 1
    prompt = client.calls[0][0]["content"]
    assert "completed agent_state" in prompt
    assert '"name": "double"' in prompt
    assert '"final_text": "Four"' in prompt
    assert state.messages == original_messages
    assert state.events == original_events
    assert state.final_text == "Four"


def test_reflector_reviews_terminal_plan_without_mutation():
    plan = build_completed_plan()
    original_steps = plan.steps
    client = FakeReflectionClient(valid_response())

    reflection = LLMAgentReflector(client).reflect_plan(plan)

    assert reflection.verdict is ReflectionVerdict.PASSED
    assert '"goal": "Finish task"' in (client.calls[0][0]["content"])
    assert '"status": "completed"' in (client.calls[0][0]["content"])
    assert plan.steps == original_steps


def test_reflector_accepts_failed_plan_as_terminal():
    plan = AgentPlan(
        "Attempt task",
        [PlanStep("step_1", "Run dependency")],
    )
    plan.start_next()
    plan.fail_current("Dependency unavailable")
    reflector = LLMAgentReflector(
        FakeReflectionClient(
            valid_response(
                verdict="failed",
                improvements=["Restore the dependency"],
            )
        )
    )

    reflection = reflector.reflect_plan(plan)

    assert reflection.verdict is ReflectionVerdict.FAILED


def test_reflector_rejects_unfinished_or_wrong_subjects():
    reflector = LLMAgentReflector(FakeReflectionClient(valid_response()))

    with pytest.raises(TypeError, match="AgentState"):
        reflector.reflect_state(object())

    with pytest.raises(ReflectionValidationError, match="finished"):
        reflector.reflect_state(
            AgentState(
                [
                    {"role": "user", "content": "Pending"},
                ]
            )
        )

    with pytest.raises(TypeError, match="AgentPlan"):
        reflector.reflect_plan(object())

    with pytest.raises(ReflectionValidationError, match="terminal"):
        reflector.reflect_plan(
            AgentPlan(
                "Pending",
                [PlanStep("step_1", "Wait")],
            )
        )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not JSON", "valid JSON"),
        ("[]", "invalid shape"),
        (
            '{"verdict":"passed","summary":"ok",'
            '"strengths":[],"improvements":[],"extra":true}',
            "invalid shape",
        ),
        (
            valid_response(verdict="unknown"),
            "verdict",
        ),
        (
            valid_response(summary=" "),
            "summary",
        ),
        (
            '{"verdict":"passed","summary":"ok","strengths":"bad","improvements":[]}',
            "strengths must be a list",
        ),
        (
            valid_response(strengths=["one", "two", "three"]),
            "too many strengths",
        ),
        (
            valid_response(improvements=["same", " SAME "]),
            "unique",
        ),
    ],
)
def test_reflector_rejects_invalid_llm_output(response, message):
    reflector = LLMAgentReflector(
        FakeReflectionClient(response),
        max_items=2,
    )

    with pytest.raises(ReflectionValidationError, match=message):
        reflector.reflect_state(build_finished_state())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_items", 0),
        ("max_items", True),
        ("max_input_chars", -1),
        ("max_input_chars", 1.5),
    ],
)
def test_reflector_rejects_invalid_limits(field, value):
    arguments = {field: value}

    with pytest.raises(ValueError, match="greater than zero"):
        LLMAgentReflector(
            FakeReflectionClient(valid_response()),
            **arguments,
        )


def test_reflector_rejects_invalid_client_or_response_type():
    with pytest.raises(TypeError, match="BaseLLMClient"):
        LLMAgentReflector(object())

    reflector = LLMAgentReflector(FakeReflectionClient(None))

    with pytest.raises(ReflectionValidationError, match="string"):
        reflector.reflect_state(build_finished_state())


def test_reflector_marks_truncated_snapshot():
    state = build_finished_state()
    state.messages[0]["content"] = "x" * 100
    client = FakeReflectionClient(valid_response())

    LLMAgentReflector(
        client,
        max_input_chars=20,
    ).reflect_state(state)

    prompt = client.calls[0][0]["content"]
    assert "truncated excerpt" in prompt
