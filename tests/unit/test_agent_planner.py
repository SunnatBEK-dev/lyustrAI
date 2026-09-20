import json

import pytest

from ai_sdk.agents import (
    AgentPlan,
    LLMAgentPlanner,
    PlanStatus,
    PlanStep,
    PlanValidationError,
)
from ai_sdk.llm.base import BaseLLMClient


class FakePlannerClient(BaseLLMClient):
    def __init__(self, response):
        self.response = response
        self.messages = None

    def ask(self, messages):
        self.messages = messages
        return self.response

    def stream(self, messages):
        yield self.response


def build_plan():
    return AgentPlan(
        goal="Build a feature",
        steps=[
            PlanStep("step_1", "Inspect requirements"),
            PlanStep("step_2", "Implement feature"),
        ],
    )


def test_agent_plan_completes_steps_sequentially():
    plan = build_plan()

    assert plan.status is PlanStatus.PENDING
    assert plan.current_step is None

    first = plan.start_next()

    assert first.status is PlanStatus.IN_PROGRESS
    assert plan.status is PlanStatus.IN_PROGRESS

    completed = plan.complete_current(" Requirements checked ")

    assert completed.status is PlanStatus.COMPLETED
    assert completed.outcome == "Requirements checked"
    assert plan.start_next().id == "step_2"
    plan.complete_current()

    assert plan.status is PlanStatus.COMPLETED
    assert [step.status for step in plan.steps] == [
        PlanStatus.COMPLETED,
        PlanStatus.COMPLETED,
    ]

    with pytest.raises(PlanValidationError, match="no pending"):
        plan.start_next()


def test_agent_plan_failure_stops_later_steps():
    plan = build_plan()
    plan.start_next()

    failed = plan.fail_current(" dependency unavailable ")

    assert failed.status is PlanStatus.FAILED
    assert failed.outcome == "dependency unavailable"
    assert plan.status is PlanStatus.FAILED

    with pytest.raises(PlanValidationError, match="failed plan"):
        plan.start_next()


def test_agent_plan_rejects_invalid_transitions():
    plan = build_plan()

    with pytest.raises(PlanValidationError, match="no active"):
        plan.complete_current()

    with pytest.raises(PlanValidationError, match="no active"):
        plan.fail_current("failed")

    with pytest.raises(PlanValidationError, match="error"):
        plan.fail_current(" ")

    plan.start_next()

    with pytest.raises(PlanValidationError, match="already"):
        plan.start_next()

    with pytest.raises(PlanValidationError, match="outcome"):
        plan.complete_current(3)


def test_agent_plan_rejects_invalid_structure():
    with pytest.raises(PlanValidationError, match="goal"):
        AgentPlan(" ", [PlanStep("step_1", "First")])

    with pytest.raises(PlanValidationError, match="at least one"):
        AgentPlan("Goal", [])

    with pytest.raises(PlanValidationError, match="steps are invalid"):
        AgentPlan("Goal", [object()])

    with pytest.raises(PlanValidationError, match="unique"):
        AgentPlan(
            "Goal",
            [
                PlanStep("same", "First"),
                PlanStep("same", "Second"),
            ],
        )

    with pytest.raises(PlanValidationError, match="Only one"):
        AgentPlan(
            "Goal",
            [
                PlanStep(
                    "step_1",
                    "First",
                    PlanStatus.IN_PROGRESS,
                ),
                PlanStep(
                    "step_2",
                    "Second",
                    PlanStatus.IN_PROGRESS,
                ),
            ],
        )


@pytest.mark.parametrize(
    "step",
    [
        lambda: PlanStep("", "Description"),
        lambda: PlanStep("step_1", " "),
        lambda: PlanStep(
            "step_1",
            "Description",
            "pending",
        ),
        lambda: PlanStep(
            "step_1",
            "Description",
            outcome=3,
        ),
    ],
)
def test_plan_step_rejects_invalid_fields(step):
    with pytest.raises(PlanValidationError):
        step()


def test_llm_planner_creates_bounded_provider_neutral_plan():
    client = FakePlannerClient(
        json.dumps(
            {
                "steps": [
                    "Inspect the repository",
                    "Implement the change",
                ],
            }
        )
    )
    planner = LLMAgentPlanner(client, max_steps=4)

    plan = planner.create_plan(" Add search support ")

    assert plan.goal == "Add search support"
    assert plan.steps == (
        PlanStep("step_1", "Inspect the repository"),
        PlanStep("step_2", "Implement the change"),
    )
    assert client.messages[0]["role"] == "user"
    assert 'Goal: "Add search support"' in (client.messages[0]["content"])
    assert "between 1 and 4" in client.messages[0]["content"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not JSON", "valid JSON"),
        ("[]", "only steps"),
        ('{"steps": [], "extra": true}', "only steps"),
        ('{"steps": []}', "non-empty list"),
        ('{"steps": ["valid", 2]}', "non-empty strings"),
        ('{"steps": ["same", " SAME "]}', "unique"),
        ('{"steps": ["one", "two", "three"]}', "too many"),
    ],
)
def test_llm_planner_rejects_invalid_responses(response, message):
    planner = LLMAgentPlanner(
        FakePlannerClient(response),
        max_steps=2,
    )

    with pytest.raises(PlanValidationError, match=message):
        planner.create_plan("Goal")


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5])
def test_llm_planner_rejects_invalid_max_steps(max_steps):
    with pytest.raises(ValueError, match="greater than zero"):
        LLMAgentPlanner(
            FakePlannerClient('{"steps": ["one"]}'),
            max_steps=max_steps,
        )


def test_llm_planner_rejects_invalid_client_goal_or_response_type():
    with pytest.raises(TypeError, match="BaseLLMClient"):
        LLMAgentPlanner(object())

    planner = LLMAgentPlanner(FakePlannerClient('{"steps": ["one"]}'))

    with pytest.raises(PlanValidationError, match="goal"):
        planner.create_plan(" ")

    invalid_response = LLMAgentPlanner(FakePlannerClient(None))

    with pytest.raises(PlanValidationError, match="string"):
        invalid_response.create_plan("Goal")
