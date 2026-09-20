import pytest

from ai_sdk.evaluation import (
    EvalCase,
    EvalCaseResult,
    EvalScore,
    EvaluationReport,
    EvaluationRunner,
    EvaluationValidationError,
    ExactMatchEvaluator,
)


class LengthEvaluator:
    name = "length_ratio"
    threshold = 0.5

    def evaluate(self, case, actual_output):
        if not case.expected_output:
            return float(not actual_output)
        return min(1.0, len(actual_output) / len(case.expected_output))


def test_runner_builds_per_case_and_aggregate_quality_report():
    outputs = {
        "first input": "first answer",
        "second input": "wrong",
    }
    cases = [
        EvalCase("first", "first input", "first answer"),
        EvalCase("second", "second input", "second answer"),
    ]
    runner = EvaluationRunner(
        [ExactMatchEvaluator(), LengthEvaluator()],
        minimum_pass_rate=0.5,
    )

    report = runner.evaluate(cases, outputs.__getitem__)

    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1
    assert report.error_cases == 0
    assert report.pass_rate == pytest.approx(0.5)
    assert report.passed is True
    assert report.failed_case_ids == ("second",)
    assert report.mean_scores == {
        "exact_match": pytest.approx(0.5),
        "length_ratio": pytest.approx(9 / 13),
    }
    assert report.results[0].scores[0].passed is True
    serialized = report.to_dict()
    assert serialized["passed"] is True
    assert serialized["results"][1]["scores"][0] == {
        "evaluator": "exact_match",
        "score": 0.0,
        "threshold": 1.0,
        "passed": False,
    }
    assert "first answer" not in str(serialized)
    assert "wrong" not in str(serialized)


def test_exact_match_can_normalize_case_and_outer_whitespace():
    evaluator = ExactMatchEvaluator(
        case_sensitive=False,
        strip=True,
    )
    case = EvalCase("greeting", "Say hello", "Hello")

    assert evaluator.evaluate(case, "  hello\n") == 1.0
    assert evaluator.name == "exact_match"
    assert evaluator.threshold == 1.0


def test_target_failure_is_isolated_and_does_not_expose_message():
    calls = []

    def target(input_text):
        calls.append(input_text)
        if input_text == "broken":
            raise RuntimeError("private generated output")
        return "ok"

    report = EvaluationRunner([ExactMatchEvaluator()]).evaluate(
        [
            EvalCase("broken", "broken", "unused"),
            EvalCase("healthy", "healthy", "ok"),
        ],
        target,
    )

    assert calls == ["broken", "healthy"]
    assert report.error_cases == 1
    assert report.results[0].error_type == "RuntimeError"
    assert report.results[0].scores == ()
    assert report.results[0].passed is False
    assert report.results[1].passed is True
    assert "private generated output" not in str(report.to_dict())


def test_evaluator_failure_preserves_prior_scores_and_continues():
    class ExplodingEvaluator:
        name = "exploding"
        threshold = 1.0

        def evaluate(self, case, actual_output):
            if case.id == "broken":
                raise LookupError("private evaluator detail")
            return 1.0

    report = EvaluationRunner([ExactMatchEvaluator(), ExplodingEvaluator()]).evaluate(
        [
            EvalCase("broken", "one", "one"),
            EvalCase("healthy", "two", "two"),
        ],
        lambda value: value,
    )

    assert report.results[0].error_type == "LookupError"
    assert len(report.results[0].scores) == 1
    assert report.results[1].passed is True
    assert report.mean_scores == {
        "exact_match": 1.0,
        "exploding": 0.5,
    }


def test_non_text_target_output_becomes_a_contained_case_error():
    report = EvaluationRunner([ExactMatchEvaluator()]).evaluate(
        [EvalCase("case", "input", "expected")],
        lambda value: 42,
    )

    assert report.error_cases == 1
    assert report.results[0].error_type == "TypeError"


def test_unusual_exception_class_name_uses_a_safe_fallback():
    unusual_error = type("x" * 129, (Exception,), {})

    def target(value):
        raise unusual_error("private")

    report = EvaluationRunner([ExactMatchEvaluator()]).evaluate(
        [EvalCase("case", "input", "expected")],
        target,
    )

    assert report.results[0].error_type == "EvaluationError"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EvalCase("", "input", "expected"),
        lambda: EvalCase(None, "input", "expected"),
        lambda: EvalCase("case", "  ", "expected"),
        lambda: EvalCase("case", "input", None),
        lambda: ExactMatchEvaluator(case_sensitive="yes"),
        lambda: ExactMatchEvaluator(strip=1),
        lambda: EvalScore("", 1.0, 1.0),
        lambda: EvalScore("score", -0.1, 1.0),
        lambda: EvalScore("score", 1.0, float("nan")),
        lambda: EvalCaseResult("case", ()),
        lambda: EvalCaseResult("case", (object(),), "TypeError"),
        lambda: EvalCaseResult("case", (), "private error"),
        lambda: EvalCaseResult(
            "case",
            (
                EvalScore("score", 1.0, 1.0),
                EvalScore("score", 0.0, 1.0),
            ),
        ),
    ],
)
def test_evaluation_models_reject_invalid_contracts(factory):
    with pytest.raises(EvaluationValidationError):
        factory()


@pytest.mark.parametrize(
    "evaluators",
    [
        [],
        [object()],
        [
            ExactMatchEvaluator(),
            ExactMatchEvaluator(),
        ],
    ],
)
def test_runner_rejects_invalid_evaluator_configuration(evaluators):
    with pytest.raises(EvaluationValidationError):
        EvaluationRunner(evaluators)


@pytest.mark.parametrize(
    "minimum_pass_rate",
    [0, -0.1, 1.1, True],
)
def test_runner_rejects_invalid_minimum_pass_rate(minimum_pass_rate):
    with pytest.raises(EvaluationValidationError):
        EvaluationRunner(
            [ExactMatchEvaluator()],
            minimum_pass_rate=minimum_pass_rate,
        )


@pytest.mark.parametrize(
    ("cases", "target"),
    [
        ([], lambda value: value),
        ([object()], lambda value: value),
        (
            [
                EvalCase("duplicate", "one", "one"),
                EvalCase("duplicate", "two", "two"),
            ],
            lambda value: value,
        ),
        ([EvalCase("case", "one", "one")], object()),
    ],
)
def test_runner_rejects_invalid_dataset_or_target(cases, target):
    runner = EvaluationRunner([ExactMatchEvaluator()])

    with pytest.raises(EvaluationValidationError):
        runner.evaluate(cases, target)


def test_invalid_evaluator_score_is_contained():
    class InvalidScoreEvaluator:
        name = "invalid_score"
        threshold = 0.5

        def evaluate(self, case, actual_output):
            return 2.0

    report = EvaluationRunner([InvalidScoreEvaluator()]).evaluate(
        [EvalCase("case", "input", "expected")],
        lambda value: "actual",
    )

    assert report.results[0].error_type == "EvaluationValidationError"


def test_report_rejects_inconsistent_direct_construction():
    score = EvalScore("exact_match", 1.0, 1.0)
    valid = EvaluationReport(
        evaluator_names=("exact_match",),
        results=(EvalCaseResult("case", (score,)),),
        minimum_pass_rate=1.0,
    )

    invalid_reports = [
        ((), valid.results, 1.0),
        (valid.evaluator_names, (), 1.0),
        (valid.evaluator_names, valid.results * 2, 1.0),
        (("other",), valid.results, 1.0),
        (("exact_match", "other"), valid.results, 1.0),
        (valid.evaluator_names, valid.results, 0.0),
    ]

    for evaluator_names, results, minimum_pass_rate in invalid_reports:
        with pytest.raises(EvaluationValidationError):
            EvaluationReport(
                evaluator_names=evaluator_names,
                results=results,
                minimum_pass_rate=minimum_pass_rate,
            )
