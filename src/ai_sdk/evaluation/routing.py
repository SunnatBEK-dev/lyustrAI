from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from ai_sdk.agents.routing import (
    CapabilityRouter,
    MultiModelRoute,
    RoutingDecision,
    RoutingSignal,
)
from ai_sdk.evaluation.harness import EvaluationValidationError

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")


@dataclass(frozen=True)
class RouteEvalCase:
    id: str
    input_text: str
    expected_route: MultiModelRoute

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _NAME_PATTERN.fullmatch(self.id) is None:
            raise EvaluationValidationError("Route evaluation case ID is invalid.")
        if not isinstance(self.input_text, str) or not self.input_text.strip():
            raise EvaluationValidationError("Route evaluation input cannot be empty.")
        if not isinstance(self.expected_route, MultiModelRoute):
            raise EvaluationValidationError(
                "Expected Adaptive Multi-Model route is invalid."
            )

    @property
    def expected_model_requests(self) -> int:
        return RoutingDecision(self.expected_route).estimated_model_requests


@dataclass(frozen=True)
class RouteEvalResult:
    case_id: str
    expected_route: MultiModelRoute
    actual_route: MultiModelRoute | None
    signals: tuple[RoutingSignal, ...]
    estimated_model_requests: int
    routing_latency_ms: float
    error_type: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or _NAME_PATTERN.fullmatch(self.case_id) is None
        ):
            raise EvaluationValidationError(
                "Route evaluation result case ID is invalid."
            )
        if not isinstance(self.expected_route, MultiModelRoute):
            raise EvaluationValidationError("Route result expected route is invalid.")
        if self.actual_route is not None and not isinstance(
            self.actual_route,
            MultiModelRoute,
        ):
            raise EvaluationValidationError("Route result actual route is invalid.")
        signals = tuple(self.signals)
        if any(not isinstance(signal, RoutingSignal) for signal in signals) or len(
            signals
        ) != len(set(signals)):
            raise EvaluationValidationError("Route result signals are invalid.")
        if (
            not isinstance(self.estimated_model_requests, int)
            or isinstance(self.estimated_model_requests, bool)
            or self.estimated_model_requests < 0
        ):
            raise EvaluationValidationError("Estimated model request count is invalid.")
        if (
            not isinstance(self.routing_latency_ms, (int, float))
            or isinstance(self.routing_latency_ms, bool)
            or not isfinite(self.routing_latency_ms)
            or self.routing_latency_ms < 0
        ):
            raise EvaluationValidationError("Routing latency is invalid.")
        if self.error_type is not None and (
            not isinstance(self.error_type, str)
            or _ERROR_TYPE_PATTERN.fullmatch(self.error_type) is None
        ):
            raise EvaluationValidationError("Route evaluation error type is invalid.")
        if self.error_type is None:
            if self.actual_route is None:
                raise EvaluationValidationError(
                    "Successful route result requires an actual route."
                )
            expected_requests = RoutingDecision(
                self.actual_route
            ).estimated_model_requests
            if self.estimated_model_requests != expected_requests:
                raise EvaluationValidationError(
                    "Route result request count is inconsistent."
                )
        elif (
            self.actual_route is not None
            or signals
            or self.estimated_model_requests != 0
        ):
            raise EvaluationValidationError(
                "Errored route result contains decision data."
            )
        object.__setattr__(self, "signals", signals)
        object.__setattr__(
            self,
            "routing_latency_ms",
            float(self.routing_latency_ms),
        )

    @property
    def passed(self) -> bool:
        return self.error_type is None and self.actual_route is self.expected_route

    @property
    def expected_model_requests(self) -> int:
        return RoutingDecision(self.expected_route).estimated_model_requests

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "expected_route": self.expected_route.value,
            "actual_route": (
                None if self.actual_route is None else self.actual_route.value
            ),
            "signals": [signal.value for signal in self.signals],
            "passed": self.passed,
            "expected_model_requests": (self.expected_model_requests),
            "estimated_model_requests": (self.estimated_model_requests),
            "routing_latency_ms": self.routing_latency_ms,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, init=False)
class RouteEvaluationReport:
    results: tuple[RouteEvalResult, ...]
    minimum_accuracy: float

    def __init__(
        self,
        results: Sequence[RouteEvalResult],
        minimum_accuracy: float = 1.0,
    ) -> None:
        normalized = tuple(results)
        if not normalized or any(
            not isinstance(result, RouteEvalResult) for result in normalized
        ):
            raise EvaluationValidationError("Route evaluation report requires results.")
        case_ids = [result.case_id for result in normalized]
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationValidationError("Route evaluation case IDs must be unique.")
        _validate_accuracy(minimum_accuracy)
        object.__setattr__(self, "results", normalized)
        object.__setattr__(
            self,
            "minimum_accuracy",
            float(minimum_accuracy),
        )

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def correct_cases(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def error_cases(self) -> int:
        return sum(result.error_type is not None for result in self.results)

    @property
    def accuracy(self) -> float:
        return self.correct_cases / self.total_cases

    @property
    def passed(self) -> bool:
        return self.error_cases == 0 and self.accuracy >= self.minimum_accuracy

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        return tuple(result.case_id for result in self.results if not result.passed)

    @property
    def expected_model_requests(self) -> int:
        return sum(result.expected_model_requests for result in self.results)

    @property
    def estimated_model_requests(self) -> int:
        return sum(result.estimated_model_requests for result in self.results)

    @property
    def request_count_delta(self) -> int:
        return self.estimated_model_requests - self.expected_model_requests

    @property
    def full_route_baseline_requests(self) -> int:
        return (
            self.total_cases
            * RoutingDecision(MultiModelRoute.FULL).estimated_model_requests
        )

    @property
    def estimated_request_savings(self) -> int:
        return self.full_route_baseline_requests - self.estimated_model_requests

    @property
    def mean_routing_latency_ms(self) -> float:
        return (
            sum(result.routing_latency_ms for result in self.results) / self.total_cases
        )

    @property
    def max_routing_latency_ms(self) -> float:
        return max(result.routing_latency_ms for result in self.results)

    @property
    def per_route_accuracy(self) -> Mapping[str, float]:
        values = {}
        for route in MultiModelRoute:
            matching = [
                result for result in self.results if result.expected_route is route
            ]
            if matching:
                values[route.value] = sum(result.passed for result in matching) / len(
                    matching
                )
        return values

    def to_dict(self) -> dict[str, object]:
        return {
            "total_cases": self.total_cases,
            "correct_cases": self.correct_cases,
            "error_cases": self.error_cases,
            "accuracy": self.accuracy,
            "minimum_accuracy": self.minimum_accuracy,
            "passed": self.passed,
            "failed_case_ids": list(self.failed_case_ids),
            "per_route_accuracy": dict(self.per_route_accuracy),
            "expected_model_requests": self.expected_model_requests,
            "estimated_model_requests": (self.estimated_model_requests),
            "request_count_delta": self.request_count_delta,
            "full_route_baseline_requests": (self.full_route_baseline_requests),
            "estimated_request_savings": (self.estimated_request_savings),
            "mean_routing_latency_ms": (self.mean_routing_latency_ms),
            "max_routing_latency_ms": self.max_routing_latency_ms,
            "results": [result.to_dict() for result in self.results],
        }


class RouteEvaluationRunner:
    """Evaluate deterministic route selection without provider calls."""

    def __init__(
        self,
        router: CapabilityRouter,
        *,
        minimum_accuracy: float = 1.0,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not isinstance(router, CapabilityRouter):
            raise EvaluationValidationError(
                "Route evaluator requires a CapabilityRouter."
            )
        _validate_accuracy(minimum_accuracy)
        if not callable(clock):
            raise EvaluationValidationError("Route evaluation clock must be callable.")
        self.router = router
        self.minimum_accuracy = float(minimum_accuracy)
        self.clock = clock

    def evaluate(
        self,
        cases: Sequence[RouteEvalCase],
    ) -> RouteEvaluationReport:
        case_list = tuple(cases)
        if not case_list or any(
            not isinstance(case, RouteEvalCase) for case in case_list
        ):
            raise EvaluationValidationError("Route evaluation dataset is invalid.")
        case_ids = [case.id for case in case_list]
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationValidationError("Route evaluation case IDs must be unique.")

        results = [self._evaluate_case(case) for case in case_list]
        return RouteEvaluationReport(
            results,
            self.minimum_accuracy,
        )

    def _evaluate_case(
        self,
        case: RouteEvalCase,
    ) -> RouteEvalResult:
        started_at = self._clock_value()
        try:
            decision = self.router.route(case.input_text)
            if not isinstance(decision, RoutingDecision):
                raise TypeError("Router returned an invalid decision.")
        except Exception as error:
            latency = self._elapsed_ms(started_at)
            return RouteEvalResult(
                case.id,
                case.expected_route,
                None,
                (),
                0,
                latency,
                _safe_error_type(error),
            )

        latency = self._elapsed_ms(started_at)
        return RouteEvalResult(
            case.id,
            case.expected_route,
            decision.route,
            decision.signals,
            decision.estimated_model_requests,
            latency,
        )

    def _elapsed_ms(self, started_at: float) -> float:
        finished_at = self._clock_value()
        if finished_at < started_at:
            raise EvaluationValidationError("Route evaluation clock moved backwards.")
        return (finished_at - started_at) * 1_000

    def _clock_value(self) -> float:
        value = self.clock()
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
        ):
            raise EvaluationValidationError("Route evaluation clock value is invalid.")
        return float(value)


def _validate_accuracy(value: float) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0
        or value > 1
    ):
        raise EvaluationValidationError("Minimum route accuracy must be within (0, 1].")


def _safe_error_type(error: Exception) -> str:
    name = type(error).__name__
    if _ERROR_TYPE_PATTERN.fullmatch(name) is None:
        return "RouteEvaluationError"
    return name


DEFAULT_ROUTE_EVAL_CASES = (
    RouteEvalCase("uz_fast_greeting", "Salom", MultiModelRoute.FAST),
    RouteEvalCase("uz_fast_short", "Qisqa javob ber", MultiModelRoute.FAST),
    RouteEvalCase("uz_fast_python", "Python nima?", MultiModelRoute.FAST),
    RouteEvalCase("en_fast_tuple", "What is a tuple?", MultiModelRoute.FAST),
    RouteEvalCase("en_fast_translate", "Translate hello", MultiModelRoute.FAST),
    RouteEvalCase("en_fast_thanks", "Thank you", MultiModelRoute.FAST),
    RouteEvalCase(
        "uz_context_sources",
        "Ushbu hujjatdagi manbalarni ko'rsat",
        MultiModelRoute.CONTEXT,
    ),
    RouteEvalCase(
        "uz_context_verify",
        "Dalillarni tekshirib ber",
        MultiModelRoute.CONTEXT,
    ),
    RouteEvalCase(
        "uz_context_facts",
        "Faktlarni ajratib ber",
        MultiModelRoute.CONTEXT,
    ),
    RouteEvalCase(
        "en_context_citation",
        "Cite the source for this fact",
        MultiModelRoute.CONTEXT,
    ),
    RouteEvalCase(
        "en_context_document",
        "Verify this document",
        MultiModelRoute.CONTEXT,
    ),
    RouteEvalCase(
        "rag_context",
        "Retrieved context:\nPython facts",
        MultiModelRoute.CONTEXT,
    ),
    RouteEvalCase(
        "uz_reasoning_why",
        "Nega bu yechim ishlaydi?",
        MultiModelRoute.REASONING,
    ),
    RouteEvalCase(
        "uz_reasoning_compare",
        "Ikki usulni taqqosla",
        MultiModelRoute.REASONING,
    ),
    RouteEvalCase(
        "uz_reasoning_plan",
        "Amalga oshirish rejasini tuz",
        MultiModelRoute.REASONING,
    ),
    RouteEvalCase(
        "en_reasoning_analysis",
        "Analyze this architecture",
        MultiModelRoute.REASONING,
    ),
    RouteEvalCase(
        "en_reasoning_plan",
        "Create an implementation plan",
        MultiModelRoute.REASONING,
    ),
    RouteEvalCase(
        "multi_part",
        "First question? Second question?",
        MultiModelRoute.REASONING,
    ),
    RouteEvalCase(
        "uz_full",
        "Manbalarni chuqur tahlil qilib taqqosla",
        MultiModelRoute.FULL,
    ),
    RouteEvalCase(
        "uz_full_document",
        "Hujjat asosida yechimni tahlil qil",
        MultiModelRoute.FULL,
    ),
    RouteEvalCase(
        "en_full_verify",
        "Verify the evidence and explain the reasoning",
        MultiModelRoute.FULL,
    ),
    RouteEvalCase(
        "en_full_sources",
        "Compare the sources and create a plan",
        MultiModelRoute.FULL,
    ),
    RouteEvalCase(
        "rag_full",
        "Retrieved context:\nFacts\n\nAnalyze the solution",
        MultiModelRoute.FULL,
    ),
    RouteEvalCase(
        "long_full",
        "Tafsilot " * 100,
        MultiModelRoute.FULL,
    ),
)
