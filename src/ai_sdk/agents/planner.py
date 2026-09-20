import json
from abc import ABC, abstractmethod

from ai_sdk.agents.plan import (
    AgentPlan,
    PlanStep,
    PlanValidationError,
)
from ai_sdk.llm.base import BaseLLMClient


class BaseAgentPlanner(ABC):
    @abstractmethod
    def create_plan(self, goal: str) -> AgentPlan:
        raise NotImplementedError


class LLMAgentPlanner(BaseAgentPlanner):
    """Create a bounded sequential plan from strict LLM JSON."""

    def __init__(
        self,
        client: BaseLLMClient,
        *,
        max_steps: int = 8,
    ) -> None:
        if not isinstance(client, BaseLLMClient):
            raise TypeError("Planner client must be a BaseLLMClient.")

        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise ValueError("Maximum plan steps must be greater than zero.")

        self.client = client
        self.max_steps = max_steps

    def create_plan(self, goal: str) -> AgentPlan:
        if not isinstance(goal, str) or not goal.strip():
            raise PlanValidationError("Agent plan goal cannot be empty.")

        normalized_goal = goal.strip()
        response = self.client.ask(
            [
                {
                    "role": "user",
                    "content": self._build_prompt(normalized_goal),
                }
            ]
        )
        descriptions = self._parse_steps(response)
        return AgentPlan(
            goal=normalized_goal,
            steps=[
                PlanStep(
                    id=f"step_{index}",
                    description=description,
                )
                for index, description in enumerate(
                    descriptions,
                    start=1,
                )
            ],
        )

    def _parse_steps(self, response: str) -> list[str]:
        if not isinstance(response, str):
            raise PlanValidationError("Planner response must be a string.")

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as error:
            raise PlanValidationError("Planner response must be valid JSON.") from error

        if not isinstance(payload, dict) or set(payload) != {"steps"}:
            raise PlanValidationError("Planner response must contain only steps.")

        raw_steps = payload["steps"]

        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanValidationError("Planner steps must be a non-empty list.")

        if len(raw_steps) > self.max_steps:
            raise PlanValidationError("Planner returned too many steps.")

        if any(not isinstance(step, str) or not step.strip() for step in raw_steps):
            raise PlanValidationError("Planner steps must be non-empty strings.")

        descriptions = [step.strip() for step in raw_steps]
        normalized = [description.casefold() for description in descriptions]

        if len(normalized) != len(set(normalized)):
            raise PlanValidationError("Planner steps must be unique.")

        return descriptions

    def _build_prompt(self, goal: str) -> str:
        encoded_goal = json.dumps(
            goal,
            ensure_ascii=False,
        )
        return (
            "Create a concise sequential execution plan for the goal "
            "below. Return JSON only, with exactly this shape: "
            '{"steps": ["first step", "second step"]}. '
            f"Use between 1 and {self.max_steps} unique, actionable "
            "steps. Do not execute the steps. Goal: "
            f"{encoded_goal}"
        )
