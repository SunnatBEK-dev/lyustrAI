from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum


class PlanValidationError(ValueError):
    """Raised when a plan or transition is invalid."""


class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    status: PlanStatus = PlanStatus.PENDING
    outcome: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise PlanValidationError("Plan step ID cannot be empty.")

        if not isinstance(self.description, str) or not self.description.strip():
            raise PlanValidationError("Plan step description cannot be empty.")

        if not isinstance(self.status, PlanStatus):
            raise PlanValidationError("Plan step status is invalid.")

        if self.outcome is not None and not isinstance(
            self.outcome,
            str,
        ):
            raise PlanValidationError("Plan step outcome must be a string.")


@dataclass(init=False)
class AgentPlan:
    goal: str
    _steps: list[PlanStep]

    def __init__(
        self,
        goal: str,
        steps: Sequence[PlanStep],
    ) -> None:
        if not isinstance(goal, str) or not goal.strip():
            raise PlanValidationError("Agent plan goal cannot be empty.")

        normalized_steps = list(steps)

        if not normalized_steps:
            raise PlanValidationError("Agent plan must contain at least one step.")

        if any(not isinstance(step, PlanStep) for step in normalized_steps):
            raise PlanValidationError("Agent plan steps are invalid.")

        step_ids = [step.id for step in normalized_steps]

        if len(step_ids) != len(set(step_ids)):
            raise PlanValidationError("Agent plan step IDs must be unique.")

        if sum(step.status is PlanStatus.IN_PROGRESS for step in normalized_steps) > 1:
            raise PlanValidationError("Only one plan step may be in progress.")

        self.goal = goal.strip()
        self._steps = normalized_steps

    @property
    def steps(self) -> tuple[PlanStep, ...]:
        return tuple(self._steps)

    @property
    def status(self) -> PlanStatus:
        statuses = {step.status for step in self._steps}

        if PlanStatus.FAILED in statuses:
            return PlanStatus.FAILED

        if statuses == {PlanStatus.COMPLETED}:
            return PlanStatus.COMPLETED

        if PlanStatus.IN_PROGRESS in statuses or PlanStatus.COMPLETED in statuses:
            return PlanStatus.IN_PROGRESS

        return PlanStatus.PENDING

    @property
    def current_step(self) -> PlanStep | None:
        return next(
            (step for step in self._steps if step.status is PlanStatus.IN_PROGRESS),
            None,
        )

    def start_next(self) -> PlanStep:
        if self.current_step is not None:
            raise PlanValidationError("A plan step is already in progress.")

        if self.status is PlanStatus.FAILED:
            raise PlanValidationError("A failed plan cannot start another step.")

        for index, step in enumerate(self._steps):
            if step.status is PlanStatus.PENDING:
                started = replace(
                    step,
                    status=PlanStatus.IN_PROGRESS,
                )
                self._steps[index] = started
                return started

        raise PlanValidationError("Agent plan has no pending steps.")

    def complete_current(
        self,
        outcome: str | None = None,
    ) -> PlanStep:
        normalized_outcome = self._normalize_outcome(outcome)
        return self._finish_current(
            PlanStatus.COMPLETED,
            normalized_outcome,
        )

    def fail_current(self, error: str) -> PlanStep:
        if not isinstance(error, str) or not error.strip():
            raise PlanValidationError("Plan step error cannot be empty.")

        return self._finish_current(
            PlanStatus.FAILED,
            error.strip(),
        )

    @staticmethod
    def _normalize_outcome(
        outcome: str | None,
    ) -> str | None:
        if outcome is None:
            return None

        if not isinstance(outcome, str):
            raise PlanValidationError("Plan step outcome must be a string.")

        normalized = outcome.strip()
        return normalized or None

    def _finish_current(
        self,
        status: PlanStatus,
        outcome: str | None,
    ) -> PlanStep:
        current = self.current_step

        if current is None:
            raise PlanValidationError("Agent plan has no active step.")

        index = self._steps.index(current)
        finished = replace(
            current,
            status=status,
            outcome=outcome,
        )
        self._steps[index] = finished
        return finished
