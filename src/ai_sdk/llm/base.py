from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from ai_sdk.llm.types import LLMMessage
from ai_sdk.tools.executor import ToolExecutor

if TYPE_CHECKING:
    from ai_sdk.agents.model import (
        AgentEvent,
        AgentModelResponse,
    )
    from ai_sdk.observability import Tracer
    from ai_sdk.tools.schema import ToolSchema


class BaseLLMClient(ABC):
    @abstractmethod
    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise NotImplementedError


class BaseToolLLMClient(BaseLLMClient):
    """LLM adapter that can complete one provider tool turn."""

    @abstractmethod
    def complete_tool_turn(
        self,
        messages: list[LLMMessage],
        schemas: list[ToolSchema],
        events: tuple[AgentEvent, ...],
    ) -> AgentModelResponse:
        raise NotImplementedError

    def ask_with_tools(
        self,
        messages: list[LLMMessage],
        executor: ToolExecutor,
        *,
        max_tool_rounds: int = 8,
        tracer: Tracer | None = None,
    ) -> str:
        from ai_sdk.agents.runner import AgentRunner

        runner = AgentRunner(
            client=self,
            executor=executor,
            max_tool_rounds=max_tool_rounds,
            tracer=tracer,
        )

        if executor.registry.count() == 0:
            return self.ask(messages)

        return runner.ask(messages)
