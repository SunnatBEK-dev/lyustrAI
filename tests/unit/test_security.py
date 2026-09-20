import pytest

from ai_sdk.security import redact_secrets


def test_redact_secrets_removes_configured_provider_keys():
    secret = "provider-secret-value"

    result = redact_secrets(
        f"Request failed with {secret}.",
        {"OPENAI_API_KEY": secret},
    )

    assert result == "Request failed with [REDACTED]."


def test_redact_secrets_removes_recognizable_unconfigured_token():
    token = "sk-" + "x" * 24

    result = redact_secrets(f"Authorization: {token}", {})

    assert result == "Authorization: [REDACTED]"


def test_redact_secrets_validates_input():
    with pytest.raises(TypeError, match="text"):
        redact_secrets(None)
