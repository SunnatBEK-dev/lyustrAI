import json

import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentStopReason,
    AgentTextBlock,
    LLMAgentPlanner,
    LLMAgentReflector,
    PlanStatus,
    ReflectionVerdict,
)
from ai_sdk.llm.base import (
    BaseLLMClient,
    BaseToolLLMClient,
)
from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)

pytestmark = pytest.mark.integration


class PlanningClient(BaseLLMClient):
    def ask(self, messages):
        return json.dumps(
            {
                "steps": [
                    "Double 2",
                    "Double 3",
                ],
            }
        )

    def stream(self, messages):
        yield self.ask(messages)


class ReviewClient(BaseLLMClient):
    def ask(self, messages):
        return json.dumps(
            {
                "verdict": "passed",
                "summary": "Both plan steps completed.",
                "strengths": ["All outcomes are present"],
                "improvements": [],
            }
        )

    def stream(self, messages):
        yield self.ask(messages)


class StepClient(BaseToolLLMClient):
    def __init__(self):
        self.responses = [
            AgentModelResponse(
                [
                    ToolCall("call_1", "double", {"value": 2}),
                ]
            ),
            AgentModelResponse([AgentTextBlock("4")]),
            AgentModelResponse(
                [
                    ToolCall("call_2", "double", {"value": 3}),
                ]
            ),
            AgentModelResponse([AgentTextBlock("6")]),
        ]

    def ask(self, messages):
        return "unused"

    def stream(self, messages):
        yield "unused"

    def complete_tool_turn(self, messages, schemas, events):
        return self.responses.pop(0)


def test_plan_steps_can_be_executed_by_agent_runner():
    plan = LLMAgentPlanner(PlanningClient()).create_plan("Double two numbers")
    registry = ToolRegistry()
    executions = []
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
        lambda value: executions.append(value) or value * 2,
    )
    runner = AgentRunner(
        StepClient(),
        ToolExecutor(registry),
    )

    while plan.status is not PlanStatus.COMPLETED:
        step = plan.start_next()
        state = runner.run(
            [
                {
                    "role": "user",
                    "content": step.description,
                }
            ]
        )
        assert state.stop_reason is AgentStopReason.FINAL_RESPONSE
        plan.complete_current(state.final_text)

    assert executions == [2, 3]
    assert [step.outcome for step in plan.steps] == ["4", "6"]

    reflection = LLMAgentReflector(ReviewClient()).reflect_plan(plan)

    assert reflection.verdict is ReflectionVerdict.PASSED
    assert reflection.summary == "Both plan steps completed."
