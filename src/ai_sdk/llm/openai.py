import json
from collections.abc import Iterator

from openai import OpenAI

from ai_sdk.agents.model import (
    AgentEvent,
    AgentModelResponse,
    AgentTextBlock,
)
from ai_sdk.config import (
    MAX_TOKENS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TIMEOUT,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.tools.model import ToolCall, ToolResult
from ai_sdk.tools.schema import ToolSchema


class OpenAIClient(BaseToolLLMClient):
    """OpenAI Responses API adapter for text, streams, and tools."""

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int = MAX_TOKENS,
        timeout: float = TIMEOUT,
    ) -> None:
        self.model = model or OPENAI_MODEL
        self.max_output_tokens = max_output_tokens

        if not self.model:
            raise RuntimeError("OPENAI_MODEL or MODEL is not configured.")

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or OPENAI_API_KEY
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        self.client = OpenAI(
            api_key=resolved_api_key,
            timeout=timeout,
            max_retries=0,
        )

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            input=self._base_input(messages),
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise RuntimeError("OpenAI response output text is invalid.")
        return output_text

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        stream = self.client.responses.create(
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            input=self._base_input(messages),
            stream=True,
            store=False,
        )
        for event in stream:
            if getattr(event, "type", None) not in {
                "response.output_text.delta",
                "response.refusal.delta",
            }:
                continue
            delta = getattr(event, "delta", None)
            if not isinstance(delta, str):
                raise RuntimeError("OpenAI streaming text delta is invalid.")
            yield delta

    def complete_tool_turn(
        self,
        messages: list[LLMMessage],
        schemas: list[ToolSchema],
        events: tuple[AgentEvent, ...],
    ) -> AgentModelResponse:
        request: dict[str, object] = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "input": self._agent_input(messages, events),
            "store": False,
        }
        if schemas:
            request["tools"] = [self._tool_schema(schema) for schema in schemas]

        response = self.client.responses.create(**request)
        return self._parse_agent_response(getattr(response, "output", None))

    @staticmethod
    def _base_input(
        messages: list[LLMMessage],
    ) -> list[dict[str, object]]:
        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]

    @classmethod
    def _agent_input(
        cls,
        messages: list[LLMMessage],
        events: tuple[AgentEvent, ...],
    ) -> list[dict[str, object]]:
        provider_input = cls._base_input(messages)
        for event in events:
            for block in event.response.blocks:
                if isinstance(block, AgentTextBlock):
                    provider_input.append(
                        {
                            "role": "assistant",
                            "content": block.text,
                        }
                    )
                else:
                    provider_input.append(
                        {
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": json.dumps(
                                block.arguments,
                                ensure_ascii=False,
                            ),
                        }
                    )
            provider_input.extend(
                cls._tool_result_input(result) for result in event.tool_results
            )
        return provider_input

    @staticmethod
    def _tool_result_input(
        result: ToolResult,
    ) -> dict[str, object]:
        output = result.content
        if result.is_error:
            output = json.dumps(
                {
                    "is_error": True,
                    "content": result.content,
                },
                ensure_ascii=False,
            )
        return {
            "type": "function_call_output",
            "call_id": result.call_id,
            "output": output,
        }

    @staticmethod
    def _tool_schema(
        schema: ToolSchema,
    ) -> dict[str, object]:
        provider_schema = schema.to_json_schema()
        parameters = provider_schema["input_schema"]
        return {
            "type": "function",
            "name": provider_schema["name"],
            "description": provider_schema["description"],
            "parameters": parameters,
            "strict": all(parameter.required for parameter in schema.parameters),
        }

    @staticmethod
    def _parse_agent_response(
        output: object,
    ) -> AgentModelResponse:
        if not isinstance(output, list):
            raise RuntimeError("OpenAI response output must be a list.")

        parsed_blocks: list[AgentTextBlock | ToolCall] = []
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                OpenAIClient._append_message_blocks(
                    parsed_blocks,
                    getattr(item, "content", None),
                )
                continue
            if item_type != "function_call":
                continue

            arguments = getattr(item, "arguments", None)
            try:
                parsed_arguments = json.loads(arguments)
                call = ToolCall(
                    id=getattr(item, "call_id", None),
                    name=getattr(item, "name", None),
                    arguments=parsed_arguments,
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError("OpenAI function-call item is invalid.") from error
            parsed_blocks.append(call)

        return AgentModelResponse(parsed_blocks)

    @staticmethod
    def _append_message_blocks(
        parsed_blocks: list[AgentTextBlock | ToolCall],
        content: object,
    ) -> None:
        if not isinstance(content, list):
            raise RuntimeError("OpenAI message content must be a list.")
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "output_text":
                text = getattr(block, "text", None)
            elif block_type == "refusal":
                text = getattr(block, "refusal", None)
            else:
                continue
            if not isinstance(text, str):
                raise RuntimeError("OpenAI message text block is invalid.")
            parsed_blocks.append(AgentTextBlock(text))
