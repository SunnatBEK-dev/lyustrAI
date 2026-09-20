from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_PROVIDER_CONFIGURATION = (
    (
        "anthropic",
        "Claude (Anthropic)",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
    ),
    (
        "openai",
        "GPT (OpenAI)",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ),
    (
        "gemini",
        "Gemini (Google)",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ),
)


@dataclass(frozen=True)
class ProviderReadiness:
    """Secret-free configuration status for one provider."""

    provider: str
    display_name: str
    api_key_configured: bool
    model_configured: bool
    missing_variables: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.api_key_configured and self.model_configured

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "ready": self.ready,
            "api_key_configured": self.api_key_configured,
            "model_configured": self.model_configured,
            "missing_variables": list(self.missing_variables),
        }


@dataclass(frozen=True)
class ProviderReadinessReport:
    """Local configuration status without network requests or secrets."""

    providers: tuple[ProviderReadiness, ...]

    @property
    def adaptive_ready(self) -> bool:
        return all(provider.ready for provider in self.providers)

    @property
    def single_model_ready_providers(self) -> tuple[str, ...]:
        return tuple(provider.provider for provider in self.providers if provider.ready)

    def for_provider(self, name: str) -> ProviderReadiness:
        normalized = name.strip().casefold()
        for provider in self.providers:
            if provider.provider == normalized:
                return provider
        raise KeyError(f"Unknown AI provider: {normalized}")

    def to_dict(self) -> dict[str, object]:
        return {
            "adaptive_ready": self.adaptive_ready,
            "single_model_ready_providers": list(self.single_model_ready_providers),
            "providers": [provider.to_dict() for provider in self.providers],
        }


def inspect_provider_readiness(
    environment: Mapping[str, str] | None = None,
) -> ProviderReadinessReport:
    """Inspect required variable presence without returning values."""

    values = os.environ if environment is None else environment
    if not isinstance(values, Mapping):
        raise TypeError("AI readiness environment must be a mapping.")

    shared_model = _is_configured(values.get("MODEL"))
    providers: list[ProviderReadiness] = []
    for provider, display_name, key_name, model_name in _PROVIDER_CONFIGURATION:
        key_configured = _is_configured(values.get(key_name))
        model_configured = _is_configured(values.get(model_name)) or shared_model
        missing: list[str] = []
        if not key_configured:
            missing.append(key_name)
        if not model_configured:
            missing.append(f"{model_name}/MODEL")
        providers.append(
            ProviderReadiness(
                provider=provider,
                display_name=display_name,
                api_key_configured=key_configured,
                model_configured=model_configured,
                missing_variables=tuple(missing),
            )
        )
    return ProviderReadinessReport(tuple(providers))


def _is_configured(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
