from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ai_sdk.tools.model import ToolCall, ToolResult


class AgentStopReason(str, Enum):
    FINAL_RESPONSE = "final_response"
    MAX_TOOL_ROUNDS = "max_tool_rounds"


@dataclass(frozen=True)
class AgentTextBlock:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Agent text must be a string.")


AgentResponseBlock = AgentTextBlock | ToolCall


@dataclass(frozen=True, init=False)
class AgentModelResponse:
    blocks: tuple[AgentResponseBlock, ...]

    def __init__(
        self,
        blocks: Sequence[AgentResponseBlock],
    ) -> None:
        normalized = tuple(blocks)

        if any(
            not isinstance(block, (AgentTextBlock, ToolCall)) for block in normalized
        ):
            raise TypeError("Agent response blocks are invalid.")

        object.__setattr__(self, "blocks", normalized)

    @property
    def text(self) -> str:
        return "".join(
            block.text for block in self.blocks if isinstance(block, AgentTextBlock)
        )

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(block for block in self.blocks if isinstance(block, ToolCall))


@dataclass(frozen=True, init=False)
class AgentEvent:
    iteration: int
    response: AgentModelResponse
    tool_results: tuple[ToolResult, ...]

    def __init__(
        self,
        iteration: int,
        response: AgentModelResponse,
        tool_results: Sequence[ToolResult] = (),
    ) -> None:
        if (
            not isinstance(iteration, int)
            or isinstance(iteration, bool)
            or iteration <= 0
        ):
            raise ValueError("Agent iteration must be greater than zero.")

        if not isinstance(response, AgentModelResponse):
            raise TypeError("Agent event response is invalid.")

        normalized_results = tuple(tool_results)

        if any(not isinstance(result, ToolResult) for result in normalized_results):
            raise TypeError("Agent tool results are invalid.")

        calls = response.tool_calls

        if normalized_results and (
            len(calls) != len(normalized_results)
            or any(
                call.id != result.call_id or call.name != result.name
                for call, result in zip(
                    calls,
                    normalized_results,
                )
            )
        ):
            raise ValueError("Agent tool results do not match tool calls.")

        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "response", response)
        object.__setattr__(
            self,
            "tool_results",
            normalized_results,
        )
