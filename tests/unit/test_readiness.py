import pytest

from ai_sdk.readiness import inspect_provider_readiness
from app.check_provider_readiness import main


def test_readiness_reports_direct_and_adaptive_configuration():
    report = inspect_provider_readiness(
        {
            "OPENAI_API_KEY": "private-openai-key",
            "OPENAI_MODEL": "gpt-test",
            "ANTHROPIC_API_KEY": "private-anthropic-key",
            "MODEL": "shared-test-model",
        }
    )

    assert report.single_model_ready_providers == (
        "anthropic",
        "openai",
    )
    assert report.adaptive_ready is False
    assert report.for_provider(" OpenAI ").ready is True
    gemini = report.for_provider("gemini")
    assert gemini.ready is False
    assert gemini.api_key_configured is False
    assert gemini.model_configured is True
    assert gemini.missing_variables == ("GEMINI_API_KEY",)
    serialized = report.to_dict()
    assert serialized["adaptive_ready"] is False
    assert "private-openai-key" not in str(serialized)
    assert "private-anthropic-key" not in str(serialized)
    assert "gpt-test" not in str(serialized)


def test_readiness_treats_blank_values_as_missing():
    report = inspect_provider_readiness(
        {
            "OPENAI_API_KEY": "  ",
            "OPENAI_MODEL": "",
        }
    )

    openai = report.for_provider("openai")
    assert openai.ready is False
    assert openai.missing_variables == (
        "OPENAI_API_KEY",
        "OPENAI_MODEL/MODEL",
    )
    with pytest.raises(KeyError, match="Unknown"):
        report.for_provider("other")
    with pytest.raises(TypeError, match="mapping"):
        inspect_provider_readiness([])


def test_readiness_cli_prints_status_without_secret_values(
    monkeypatch,
    capsys,
):
    variables = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "MODEL",
    )
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-value")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    main()

    output = capsys.readouterr().out
    assert "GPT (OpenAI): READY" in output
    assert "Claude (Anthropic): NOT READY" in output
    assert "Single Model ready providers: 1/3" in output
    assert "Adaptive Multi-Model: NOT READY" in output
    assert "private-test-value" not in output
    assert "test-model" not in output
