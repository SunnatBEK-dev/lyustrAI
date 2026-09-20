import pytest

from ai_sdk.llm.retry import (
    RetryPolicy,
    RetryValidationError,
)


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        ConnectionError("connection"),
        type("StatusError", (Exception,), {"status_code": 429})(),
        type("CodeError", (Exception,), {"code": 503})(),
        type("RateLimitError", (Exception,), {})(),
        type("OverloadedError", (Exception,), {})(),
        type("RetryableError", (Exception,), {})(),
    ],
)
def test_policy_retries_only_known_transient_errors(error):
    assert RetryPolicy().should_retry(error) is True


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("application"),
        ValueError("validation"),
        type("AuthError", (Exception,), {"status_code": 401})(),
        type("BadRequestError", (Exception,), {"code": 400})(),
    ],
)
def test_policy_rejects_permanent_errors(error):
    assert RetryPolicy().should_retry(error) is False


def test_policy_uses_bounded_deterministic_backoff():
    policy = RetryPolicy(
        max_attempts=5,
        initial_delay_seconds=0.5,
        max_delay_seconds=1.5,
        backoff_multiplier=2,
    )

    assert [policy.delay_before_retry(number) for number in range(1, 5)] == [
        0.5,
        1.0,
        1.5,
        1.5,
    ]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RetryPolicy(max_attempts=0),
        lambda: RetryPolicy(max_attempts=11),
        lambda: RetryPolicy(max_attempts=True),
        lambda: RetryPolicy(initial_delay_seconds=-1),
        lambda: RetryPolicy(initial_delay_seconds=float("nan")),
        lambda: RetryPolicy(
            initial_delay_seconds=2,
            max_delay_seconds=1,
        ),
        lambda: RetryPolicy(backoff_multiplier=0.5),
        lambda: RetryPolicy().delay_before_retry(0),
        lambda: RetryPolicy().delay_before_retry(3),
    ],
)
def test_policy_rejects_invalid_configuration(factory):
    with pytest.raises(RetryValidationError):
        factory()


def test_policy_rejects_non_exception_input():
    with pytest.raises(TypeError, match="exception"):
        RetryPolicy().should_retry("error")
