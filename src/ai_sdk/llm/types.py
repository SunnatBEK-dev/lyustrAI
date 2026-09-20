from typing import TypedDict


class LLMMessage(TypedDict):
    """Provider-neutral message sent to an LLM adapter."""

    role: str
    content: str
