from __future__ import annotations

import math
from dataclasses import dataclass

_RETRYABLE_STATUS_CODES = frozenset(
    {
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    }
)
_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "OverloadedError",
        "RateLimitError",
        "RetryableError",
        "ServerError",
    }
)
_MAX_ATTEMPTS = 10


class RetryValidationError(ValueError):
    """Raised when retry configuration is unsafe or invalid."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded deterministic retry policy for transient LLM failures."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts <= 0
            or self.max_attempts > _MAX_ATTEMPTS
        ):
            raise RetryValidationError(
                "Retry maximum attempts must be between one and ten."
            )
        for value, label in (
            (self.initial_delay_seconds, "initial delay"),
            (self.max_delay_seconds, "maximum delay"),
            (self.backoff_multiplier, "backoff multiplier"),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise RetryValidationError(f"Retry {label} must be finite.")
        if self.initial_delay_seconds < 0:
            raise RetryValidationError("Retry initial delay cannot be negative.")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise RetryValidationError(
                "Retry maximum delay cannot be below initial delay."
            )
        if self.backoff_multiplier < 1:
            raise RetryValidationError("Retry backoff multiplier cannot be below one.")

    def should_retry(self, error: Exception) -> bool:
        if not isinstance(error, Exception):
            raise TypeError("Retry error must be an exception.")
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True

        status_code = getattr(error, "status_code", None)
        if status_code is None:
            status_code = getattr(error, "code", None)
        if (
            isinstance(status_code, int)
            and not isinstance(status_code, bool)
            and status_code in _RETRYABLE_STATUS_CODES
        ):
            return True
        return type(error).__name__ in _RETRYABLE_ERROR_NAMES

    def delay_before_retry(self, retry_number: int) -> float:
        """Return delay before a one-based retry after a failed attempt."""

        if (
            not isinstance(retry_number, int)
            or isinstance(retry_number, bool)
            or retry_number <= 0
            or retry_number >= self.max_attempts
        ):
            raise RetryValidationError(
                "Retry number is outside the configured attempt limit."
            )
        delay = self.initial_delay_seconds
        for _ in range(retry_number - 1):
            delay = min(
                delay * self.backoff_multiplier,
                self.max_delay_seconds,
            )
        return min(delay, self.max_delay_seconds)
