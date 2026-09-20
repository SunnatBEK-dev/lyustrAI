from __future__ import annotations

import os
import re
from collections.abc import Mapping

_API_KEY_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
)
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
)


def redact_secrets(
    text: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Remove configured provider keys and recognizable tokens from text."""

    if not isinstance(text, str):
        raise TypeError("Secret redaction input must be text.")

    resolved_environment = os.environ if environment is None else environment
    redacted = text
    for variable in _API_KEY_VARIABLES:
        value = resolved_environment.get(variable, "")
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
