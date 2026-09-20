from ai_sdk.agents import CapabilityRouter
from ai_sdk.evaluation import (
    DEFAULT_ROUTE_EVAL_CASES,
    RouteEvaluationRunner,
)


def main() -> None:
    report = RouteEvaluationRunner(
        CapabilityRouter(),
        minimum_accuracy=1.0,
    ).evaluate(DEFAULT_ROUTE_EVAL_CASES)

    print("Adaptive Multi-Model route evaluation")
    print(
        f"Accuracy: {report.accuracy:.1%} ({report.correct_cases}/{report.total_cases})"
    )
    print(f"Estimated model requests: {report.estimated_model_requests}")
    print(f"Full-route baseline requests: {report.full_route_baseline_requests}")
    print(f"Estimated requests saved: {report.estimated_request_savings}")
    print(f"Mean local routing latency: {report.mean_routing_latency_ms:.3f} ms")
    print(f"Passed: {report.passed}")

    if report.failed_case_ids:
        print("Failed cases: " + ", ".join(report.failed_case_ids))


if __name__ == "__main__":
    main()
