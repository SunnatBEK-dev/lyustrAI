from dataclasses import dataclass, field

from ai_sdk.agents.model import (
    AgentEvent,
    AgentStopReason,
)
from ai_sdk.llm.types import LLMMessage


@dataclass
class AgentState:
    """Mutable state for one agent run."""

    messages: list[LLMMessage]
    events: list[AgentEvent] = field(default_factory=list)
    stop_reason: AgentStopReason | None = None
    final_text: str | None = None

    def __post_init__(self) -> None:
        self.messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in self.messages
        ]

    @property
    def is_finished(self) -> bool:
        return self.stop_reason is not None

    @property
    def tool_rounds(self) -> int:
        return sum(bool(event.tool_results) for event in self.events)

    def record(self, event: AgentEvent) -> None:
        if self.is_finished:
            raise RuntimeError("Finished agent state cannot record events.")

        if event.iteration != len(self.events) + 1:
            raise ValueError("Agent event iteration is out of sequence.")

        self.events.append(event)

    def finish(
        self,
        reason: AgentStopReason,
        final_text: str,
    ) -> None:
        if self.is_finished:
            raise RuntimeError("Agent state is already finished.")

        if not isinstance(reason, AgentStopReason):
            raise TypeError("Agent stop reason is invalid.")

        if not isinstance(final_text, str):
            raise TypeError("Agent final text must be a string.")

        self.stop_reason = reason
        self.final_text = final_text
