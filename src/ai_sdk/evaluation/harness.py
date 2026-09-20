import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")


class EvaluationValidationError(ValueError):
    """Raised when an evaluation contract is invalid."""


@dataclass(frozen=True)
class EvalCase:
    """One explicit text input and its expected output."""

    id: str
    input_text: str
    expected_output: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _validate_name(self.id, "case ID"))
        if not isinstance(self.input_text, str) or not self.input_text.strip():
            raise EvaluationValidationError("Evaluation input text cannot be empty.")
        if not isinstance(self.expected_output, str):
            raise EvaluationValidationError("Expected evaluation output must be text.")


class Evaluator(Protocol):
    """Provider-neutral contract for one deterministic quality score."""

    @property
    def name(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    def evaluate(self, case: EvalCase, actual_output: str) -> float: ...


@dataclass(frozen=True)
class _ConfiguredEvaluator:
    evaluator: Evaluator
    name: str
    threshold: float


@dataclass(frozen=True)
class ExactMatchEvaluator:
    """Score one when actual and expected text match exactly."""

    case_sensitive: bool = True
    strip: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.case_sensitive, bool):
            raise EvaluationValidationError(
                "Exact-match case sensitivity must be boolean."
            )
        if not isinstance(self.strip, bool):
            raise EvaluationValidationError("Exact-match stripping must be boolean.")

    @property
    def name(self) -> str:
        return "exact_match"

    @property
    def threshold(self) -> float:
        return 1.0

    def evaluate(self, case: EvalCase, actual_output: str) -> float:
        expected = case.expected_output
        actual = actual_output
        if self.strip:
            expected = expected.strip()
            actual = actual.strip()
        if not self.case_sensitive:
            expected = expected.casefold()
            actual = actual.casefold()
        return float(actual == expected)


@dataclass(frozen=True)
class EvalScore:
    """One normalized evaluator score and its pass threshold."""

    evaluator_name: str
    value: float
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluator_name",
            _validate_name(self.evaluator_name, "evaluator name"),
        )
        object.__setattr__(self, "value", _validate_score(self.value, "score"))
        object.__setattr__(
            self,
            "threshold",
            _validate_score(self.threshold, "threshold"),
        )

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluator": self.evaluator_name,
            "score": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EvalCaseResult:
    """Per-case scores without retaining the generated output."""

    case_id: str
    scores: tuple[EvalScore, ...]
    error_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _validate_name(self.case_id, "case ID"),
        )
        scores = tuple(self.scores)
        if any(not isinstance(score, EvalScore) for score in scores):
            raise EvaluationValidationError("Evaluation case scores are invalid.")
        score_names = tuple(score.evaluator_name for score in scores)
        if len(score_names) != len(set(score_names)):
            raise EvaluationValidationError(
                "Evaluation case score names must be unique."
            )
        if self.error_type is None and not scores:
            raise EvaluationValidationError(
                "Successful evaluation cases require scores."
            )
        if self.error_type is not None and (
            not isinstance(self.error_type, str)
            or _ERROR_TYPE_PATTERN.fullmatch(self.error_type) is None
        ):
            raise EvaluationValidationError("Evaluation error type is invalid.")
        object.__setattr__(self, "scores", scores)

    @property
    def passed(self) -> bool:
        return self.error_type is None and all(score.passed for score in self.scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "scores": [score.to_dict() for score in self.scores],
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate deterministic quality report for one eval run."""

    evaluator_names: tuple[str, ...]
    results: tuple[EvalCaseResult, ...]
    minimum_pass_rate: float

    def __post_init__(self) -> None:
        evaluator_names = tuple(
            _validate_name(name, "evaluator name") for name in self.evaluator_names
        )
        if not evaluator_names or len(evaluator_names) != len(set(evaluator_names)):
            raise EvaluationValidationError(
                "Evaluation report requires unique evaluator names."
            )
        results = tuple(self.results)
        if not results or any(
            not isinstance(result, EvalCaseResult) for result in results
        ):
            raise EvaluationValidationError("Evaluation report requires case results.")
        case_ids = tuple(result.case_id for result in results)
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationValidationError(
                "Evaluation report case IDs must be unique."
            )
        allowed_names = set(evaluator_names)
        if any(
            score.evaluator_name not in allowed_names
            for result in results
            for score in result.scores
        ):
            raise EvaluationValidationError(
                "Evaluation report contains an unknown evaluator score."
            )
        if any(
            result.error_type is None
            and {score.evaluator_name for score in result.scores} != allowed_names
            for result in results
        ):
            raise EvaluationValidationError(
                "Successful cases require every evaluator score."
            )
        minimum_pass_rate = _validate_score(
            self.minimum_pass_rate,
            "minimum pass rate",
        )
        if minimum_pass_rate == 0.0:
            raise EvaluationValidationError(
                "Minimum evaluation pass rate must be greater than zero."
            )
        object.__setattr__(self, "evaluator_names", evaluator_names)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "minimum_pass_rate", minimum_pass_rate)

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    @property
    def error_cases(self) -> int:
        return sum(result.error_type is not None for result in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases

    @property
    def passed(self) -> bool:
        return self.pass_rate >= self.minimum_pass_rate

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        return tuple(result.case_id for result in self.results if not result.passed)

    @property
    def mean_scores(self) -> Mapping[str, float]:
        totals = {name: 0.0 for name in self.evaluator_names}
        for result in self.results:
            for score in result.scores:
                totals[score.evaluator_name] += score.value
        return {name: totals[name] / self.total_cases for name in self.evaluator_names}

    def to_dict(self) -> dict[str, object]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "error_cases": self.error_cases,
            "pass_rate": self.pass_rate,
            "minimum_pass_rate": self.minimum_pass_rate,
            "passed": self.passed,
            "mean_scores": dict(self.mean_scores),
            "results": [result.to_dict() for result in self.results],
        }


class EvaluationRunner:
    """Run text eval cases sequentially with per-case failure isolation."""

    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        *,
        minimum_pass_rate: float = 1.0,
    ) -> None:
        evaluator_list = tuple(evaluators)
        if not evaluator_list:
            raise EvaluationValidationError(
                "Evaluation runner requires at least one evaluator."
            )

        configured: list[_ConfiguredEvaluator] = []
        for evaluator in evaluator_list:
            evaluate = getattr(evaluator, "evaluate", None)
            if not callable(evaluate):
                raise EvaluationValidationError("Evaluation evaluator is invalid.")
            configured.append(
                _ConfiguredEvaluator(
                    evaluator=evaluator,
                    name=_validate_name(
                        getattr(evaluator, "name", None),
                        "evaluator name",
                    ),
                    threshold=_validate_score(
                        getattr(evaluator, "threshold", None),
                        "evaluator threshold",
                    ),
                )
            )
        names = [item.name for item in configured]
        if len(names) != len(set(names)):
            raise EvaluationValidationError(
                "Evaluation evaluator names must be unique."
            )
        minimum_pass_rate = _validate_score(
            minimum_pass_rate,
            "minimum pass rate",
        )
        if minimum_pass_rate == 0.0:
            raise EvaluationValidationError(
                "Minimum evaluation pass rate must be greater than zero."
            )

        self.evaluators = evaluator_list
        self._configured = tuple(configured)
        self.evaluator_names = tuple(names)
        self.minimum_pass_rate = minimum_pass_rate

    def evaluate(
        self,
        cases: Sequence[EvalCase],
        target: Callable[[str], str],
    ) -> EvaluationReport:
        if not callable(target):
            raise EvaluationValidationError("Evaluation target must be callable.")
        case_list = tuple(cases)
        if not case_list:
            raise EvaluationValidationError("Evaluation dataset cannot be empty.")
        if any(not isinstance(case, EvalCase) for case in case_list):
            raise EvaluationValidationError("Evaluation dataset cases are invalid.")
        case_ids = tuple(case.id for case in case_list)
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationValidationError("Evaluation case IDs must be unique.")

        results = tuple(self._evaluate_case(case, target) for case in case_list)
        return EvaluationReport(
            evaluator_names=self.evaluator_names,
            results=results,
            minimum_pass_rate=self.minimum_pass_rate,
        )

    def _evaluate_case(
        self,
        case: EvalCase,
        target: Callable[[str], str],
    ) -> EvalCaseResult:
        scores: list[EvalScore] = []
        try:
            actual_output = target(case.input_text)
            if not isinstance(actual_output, str):
                raise TypeError("Evaluation target output must be text.")
            for configured in self._configured:
                scores.append(
                    EvalScore(
                        evaluator_name=configured.name,
                        value=configured.evaluator.evaluate(
                            case,
                            actual_output,
                        ),
                        threshold=configured.threshold,
                    )
                )
        except Exception as error:
            return EvalCaseResult(
                case_id=case.id,
                scores=tuple(scores),
                error_type=_safe_error_type(error),
            )
        return EvalCaseResult(
            case_id=case.id,
            scores=tuple(scores),
        )


def _validate_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EvaluationValidationError(f"Evaluation {label} is invalid.")
    normalized = value.strip()
    if _NAME_PATTERN.fullmatch(normalized) is None:
        raise EvaluationValidationError(f"Evaluation {label} is invalid.")
    return normalized


def _validate_score(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise EvaluationValidationError(
            f"Evaluation {label} must be between zero and one."
        )
    return float(value)


def _safe_error_type(error: Exception) -> str:
    error_type = type(error).__name__
    if _ERROR_TYPE_PATTERN.fullmatch(error_type) is None:
        return "EvaluationError"
    return error_type
