from collections.abc import Sequence
from typing import Protocol

from ai_sdk.context.window import (
    RegexTokenCounter,
    TokenCounter,
)
from ai_sdk.llm.types import LLMMessage


class ConversationSummarizer(Protocol):
    """Contract for compressing messages excluded from context."""

    def summarize(
        self,
        messages: Sequence[LLMMessage],
    ) -> str: ...


class ExtractiveConversationSummarizer:
    """Retain recent excluded dialogue within a fixed summary budget."""

    def __init__(
        self,
        max_tokens: int,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("Summary token budget must be greater than zero.")

        self.max_tokens = max_tokens
        self.token_counter = token_counter or RegexTokenCounter()

    def summarize(
        self,
        messages: Sequence[LLMMessage],
    ) -> str:
        selected_lines = []
        used_tokens = 0

        for message in reversed(messages):
            content = " ".join(message["content"].split())

            if not content:
                continue

            line = f"{message['role'].capitalize()}: {content}"
            line_tokens = self.token_counter.count(line)
            remaining_tokens = self.max_tokens - used_tokens

            if line_tokens <= remaining_tokens:
                selected_lines.append(line)
                used_tokens += line_tokens
                continue

            if not selected_lines and remaining_tokens > 0:
                truncated = self._truncate_prefix(
                    line,
                    remaining_tokens,
                )

                if truncated:
                    selected_lines.append(truncated)

            break

        return "\n".join(reversed(selected_lines))

    def _truncate_prefix(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        low = 0
        high = len(text)
        best = ""

        while low <= high:
            midpoint = (low + high) // 2
            candidate = text[:midpoint].rstrip()

            if self.token_counter.count(candidate) <= max_tokens:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1

        return best
