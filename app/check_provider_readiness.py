from ai_sdk.config import PROJECT_ROOT  # noqa: F401
from ai_sdk.readiness import inspect_provider_readiness


def main() -> None:
    report = inspect_provider_readiness()
    print("AI configuration readiness (no network requests):")
    for provider in report.providers:
        status = "READY" if provider.ready else "NOT READY"
        detail = ""
        if provider.missing_variables:
            detail = " | missing: " + ", ".join(provider.missing_variables)
        print(f"- {provider.display_name}: {status}{detail}")

    ready_count = len(report.single_model_ready_providers)
    print(f"Single Model ready providers: {ready_count}/{len(report.providers)}")
    adaptive_status = "READY" if report.adaptive_ready else "NOT READY"
    print(f"Adaptive Multi-Model: {adaptive_status}")


if __name__ == "__main__":
    main()
