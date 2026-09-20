import pytest

import ai_sdk.llm.factory as factory_module
from ai_sdk.llm.factory import create_llm_client, normalize_llm_provider


def test_factory_creates_selected_provider(monkeypatch):
    anthropic_client = object()
    openai_client = object()
    gemini_client = object()
    monkeypatch.setattr(
        factory_module,
        "AnthropicClient",
        lambda: anthropic_client,
    )
    monkeypatch.setattr(
        factory_module,
        "OpenAIClient",
        lambda: openai_client,
    )
    monkeypatch.setattr(
        factory_module,
        "GeminiClient",
        lambda: gemini_client,
    )

    assert create_llm_client(" Anthropic ") is anthropic_client
    assert create_llm_client("OPENAI") is openai_client
    assert create_llm_client("Gemini") is gemini_client


def test_factory_uses_environment_provider(monkeypatch):
    expected = object()
    monkeypatch.setattr(factory_module, "DEFAULT_AI_PROVIDER", "openai")
    monkeypatch.setattr(
        factory_module,
        "OpenAIClient",
        lambda: expected,
    )

    assert create_llm_client() is expected


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (" Anthropic ", "anthropic"),
        ("OPENAI", "openai"),
        ("Gemini", "gemini"),
    ],
)
def test_provider_name_normalization(provider, expected):
    assert normalize_llm_provider(provider) == expected


@pytest.mark.parametrize("provider", ["", "unknown", 42])
def test_factory_rejects_invalid_provider(provider):
    with pytest.raises(RuntimeError):
        create_llm_client(provider)
