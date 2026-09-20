import pytest

from ai_sdk.agents import (
    CapabilityRouter,
    MultiModelRoute,
)
from ai_sdk.evaluation import (
    DEFAULT_ROUTE_EVAL_CASES,
    EvaluationValidationError,
    RouteEvalCase,
    RouteEvalResult,
    RouteEvaluationReport,
    RouteEvaluationRunner,
)
from app.evaluate_routing import main


def test_default_bilingual_route_benchmark_passes_offline():
    report = RouteEvaluationRunner(CapabilityRouter()).evaluate(
        DEFAULT_ROUTE_EVAL_CASES
    )

    assert report.total_cases == 24
    assert report.correct_cases == 24
    assert report.error_cases == 0
    assert report.accuracy == 1.0
    assert report.passed is True
    assert report.failed_case_ids == ()
    assert report.expected_model_requests == 48
    assert report.estimated_model_requests == 48
    assert report.request_count_delta == 0
    assert report.full_route_baseline_requests == 72
    assert report.estimated_request_savings == 24
    assert report.per_route_accuracy == {
        "fast": 1.0,
        "context": 1.0,
        "reasoning": 1.0,
        "full": 1.0,
    }
    assert DEFAULT_ROUTE_EVAL_CASES[0].expected_model_requests == 1


def test_route_report_measures_mismatch_requests_and_latency():
    times = iter([1.0, 1.001, 2.0, 2.003])
    runner = RouteEvaluationRunner(
        CapabilityRouter(),
        minimum_accuracy=0.5,
        clock=lambda: next(times),
    )

    report = runner.evaluate(
        [
            RouteEvalCase("correct", "Salom", MultiModelRoute.FAST),
            RouteEvalCase(
                "mismatch",
                "Salom",
                MultiModelRoute.CONTEXT,
            ),
        ]
    )

    assert report.correct_cases == 1
    assert report.accuracy == 0.5
    assert report.passed is True
    assert report.failed_case_ids == ("mismatch",)
    assert report.expected_model_requests == 3
    assert report.estimated_model_requests == 2
    assert report.request_count_delta == -1
    assert report.full_route_baseline_requests == 6
    assert report.estimated_request_savings == 4
    assert report.mean_routing_latency_ms == pytest.approx(2.0)
    assert report.max_routing_latency_ms == pytest.approx(3.0)
    assert report.per_route_accuracy == {
        "fast": 1.0,
        "context": 0.0,
    }
    serialized = report.to_dict()
    assert serialized["passed"] is True
    assert serialized["results"][1]["actual_route"] == "fast"
    assert "Salom" not in str(serialized)


def test_route_failure_is_contained_and_does_not_leak_message():
    class FailingRouter(CapabilityRouter):
        def route(self, input_text):
            if input_text == "broken":
                raise RuntimeError("private routing detail")
            return super().route(input_text)

    times = iter([1.0, 1.001, 2.0, 2.002])
    report = RouteEvaluationRunner(
        FailingRouter(),
        minimum_accuracy=0.5,
        clock=lambda: next(times),
    ).evaluate(
        [
            RouteEvalCase("broken", "broken", MultiModelRoute.FAST),
            RouteEvalCase("healthy", "Salom", MultiModelRoute.FAST),
        ]
    )

    assert report.error_cases == 1
    assert report.passed is False
    assert report.results[0].actual_route is None
    assert report.results[0].signals == ()
    assert report.results[0].estimated_model_requests == 0
    assert report.results[0].error_type == "RuntimeError"
    assert report.results[1].passed is True
    assert "private routing detail" not in str(report.to_dict())


def test_route_runner_contains_invalid_decisions_and_error_names():
    class InvalidRouter(CapabilityRouter):
        def route(self, input_text):
            return object()

    invalid_times = iter([1.0, 1.001])
    invalid_report = RouteEvaluationRunner(
        InvalidRouter(),
        clock=lambda: next(invalid_times),
    ).evaluate(
        [
            RouteEvalCase("invalid", "input", MultiModelRoute.FAST),
        ]
    )
    unusual_error = type("x" * 129, (Exception,), {})

    class UnusualRouter(CapabilityRouter):
        def route(self, input_text):
            raise unusual_error("private")

    unusual_times = iter([2.0, 2.001])
    unusual_report = RouteEvaluationRunner(
        UnusualRouter(),
        clock=lambda: next(unusual_times),
    ).evaluate(
        [
            RouteEvalCase("unusual", "input", MultiModelRoute.FAST),
        ]
    )

    assert invalid_report.results[0].error_type == "TypeError"
    assert unusual_report.results[0].error_type == ("RouteEvaluationError")


def test_route_eval_case_and_result_validate_contracts():
    valid = RouteEvalResult(
        "case",
        MultiModelRoute.FAST,
        MultiModelRoute.FAST,
        (),
        1,
        0.1,
    )

    invalid_factories = [
        lambda: RouteEvalCase("", "input", MultiModelRoute.FAST),
        lambda: RouteEvalCase("case", " ", MultiModelRoute.FAST),
        lambda: RouteEvalCase("case", "input", "fast"),
        lambda: RouteEvalResult(
            "",
            MultiModelRoute.FAST,
            MultiModelRoute.FAST,
            (),
            1,
            0.1,
        ),
        lambda: RouteEvalResult(
            "case",
            "fast",
            MultiModelRoute.FAST,
            (),
            1,
            0.1,
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            "fast",
            (),
            1,
            0.1,
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            MultiModelRoute.FAST,
            ("invalid",),
            1,
            0.1,
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            MultiModelRoute.FAST,
            (),
            -1,
            0.1,
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            MultiModelRoute.FAST,
            (),
            1,
            float("nan"),
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            None,
            (),
            0,
            0.1,
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            MultiModelRoute.FAST,
            (),
            2,
            0.1,
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            None,
            (),
            0,
            0.1,
            "private error",
        ),
        lambda: RouteEvalResult(
            "case",
            MultiModelRoute.FAST,
            MultiModelRoute.FAST,
            (),
            1,
            0.1,
            "RuntimeError",
        ),
        lambda: RouteEvaluationReport([], 1.0),
        lambda: RouteEvaluationReport([valid, valid], 1.0),
        lambda: RouteEvaluationReport([valid], 0),
    ]

    for factory in invalid_factories:
        with pytest.raises(EvaluationValidationError):
            factory()


def test_route_runner_validates_dataset_clock_and_router():
    case = RouteEvalCase("case", "Salom", MultiModelRoute.FAST)

    with pytest.raises(EvaluationValidationError, match="CapabilityRouter"):
        RouteEvaluationRunner(object())
    with pytest.raises(EvaluationValidationError, match="clock"):
        RouteEvaluationRunner(CapabilityRouter(), clock=object())
    runner = RouteEvaluationRunner(CapabilityRouter())
    with pytest.raises(EvaluationValidationError, match="dataset"):
        runner.evaluate([])
    with pytest.raises(EvaluationValidationError, match="dataset"):
        runner.evaluate([object()])
    with pytest.raises(EvaluationValidationError, match="unique"):
        runner.evaluate([case, case])
    with pytest.raises(EvaluationValidationError, match="clock value"):
        RouteEvaluationRunner(
            CapabilityRouter(),
            clock=lambda: float("nan"),
        ).evaluate([case])
    backwards = iter([2.0, 1.0])
    with pytest.raises(EvaluationValidationError, match="backwards"):
        RouteEvaluationRunner(
            CapabilityRouter(),
            clock=lambda: next(backwards),
        ).evaluate([case])


def test_route_evaluation_cli_prints_compact_summary(capsys):
    main()

    output = capsys.readouterr().out
    assert "Accuracy: 100.0% (24/24)" in output
    assert "Estimated model requests: 48" in output
    assert "Estimated requests saved: 24" in output
    assert "Passed: True" in output
