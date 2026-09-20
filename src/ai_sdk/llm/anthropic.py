from collections.abc import Iterator

from anthropic import Anthropic

from ai_sdk.agents.model import (
    AgentEvent,
    AgentModelResponse,
    AgentTextBlock,
)
from ai_sdk.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    MAX_TOKENS,
    TIMEOUT,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.tools.model import ToolCall
from ai_sdk.tools.schema import ToolSchema


class AnthropicClient(BaseToolLLMClient):
    def __init__(
        self,
        client: Anthropic | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = MAX_TOKENS,
        timeout: float = TIMEOUT,
    ) -> None:
        self.model = model or ANTHROPIC_MODEL
        self.max_tokens = max_tokens

        if not self.model:
            raise RuntimeError("ANTHROPIC_MODEL or MODEL is not configured.")

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or ANTHROPIC_API_KEY

        if not resolved_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        self.client = Anthropic(
            api_key=resolved_api_key,
            timeout=timeout,
            max_retries=0,
        )

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        parts = []

        for block in response.content:
            if block.type == "text":
                parts.append(block.text)

        return "".join(parts)

    def complete_tool_turn(
        self,
        messages: list[LLMMessage],
        schemas: list[ToolSchema],
        events: tuple[AgentEvent, ...],
    ) -> AgentModelResponse:
        """Complete one Claude turn for a provider-neutral agent."""
        provider_messages: list[dict[str, object]] = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]

        for event in events:
            provider_messages.append(
                {
                    "role": "assistant",
                    "content": self._assistant_blocks(event.response),
                }
            )
            provider_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }
                        for result in event.tool_results
                    ],
                }
            )

        request: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": provider_messages,
        }

        if schemas:
            request["tools"] = [schema.to_json_schema() for schema in schemas]

        response = self.client.messages.create(**request)
        parsed = self._parse_agent_response(response.content)

        if (
            getattr(response, "stop_reason", None) == "tool_use"
            and not parsed.tool_calls
        ):
            raise RuntimeError("Claude returned tool_use without a tool call.")

        return parsed

    @staticmethod
    def _assistant_blocks(
        response: AgentModelResponse,
    ) -> list[dict[str, object]]:
        blocks: list[dict[str, object]] = []

        for block in response.blocks:
            if isinstance(block, AgentTextBlock):
                blocks.append(
                    {
                        "type": "text",
                        "text": block.text,
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    }
                )

        return blocks

    @staticmethod
    def _parse_agent_response(
        blocks: object,
    ) -> AgentModelResponse:
        if not isinstance(blocks, list):
            raise RuntimeError("Claude response content must be a list.")

        parsed_blocks: list[AgentTextBlock | ToolCall] = []

        for block in blocks:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", None)

                if not isinstance(text, str):
                    raise RuntimeError("Claude text block is invalid.")

                parsed_blocks.append(AgentTextBlock(text))
                continue

            if block_type == "tool_use":
                try:
                    call = ToolCall(
                        id=getattr(block, "id", None),
                        name=getattr(block, "name", None),
                        arguments=getattr(block, "input", None),
                    )
                except (TypeError, ValueError) as error:
                    raise RuntimeError("Claude tool-use block is invalid.") from error

                parsed_blocks.append(call)

        return AgentModelResponse(parsed_blocks)

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
