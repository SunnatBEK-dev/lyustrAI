from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy

from google import genai

from ai_sdk.agents.model import (
    AgentEvent,
    AgentModelResponse,
    AgentResponseBlock,
    AgentTextBlock,
)
from ai_sdk.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_TOKENS,
    TIMEOUT,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.tools.model import ToolCall, ToolResult
from ai_sdk.tools.schema import ToolSchema


class _GeminiModelResponse(AgentModelResponse):
    """Neutral response carrying private stateless Gemini steps."""

    def __init__(
        self,
        blocks: Sequence[AgentResponseBlock],
        provider_steps: Sequence[dict[str, object]],
    ) -> None:
        super().__init__(blocks)
        object.__setattr__(
            self,
            "provider_steps",
            tuple(deepcopy(provider_steps)),
        )


class GeminiClient(BaseToolLLMClient):
    """Gemini Interactions API adapter for text, streams, and tools."""

    def __init__(
        self,
        client: object | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int = MAX_TOKENS,
        timeout: float = TIMEOUT,
    ) -> None:
        self.model = model or GEMINI_MODEL
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

        if not self.model:
            raise RuntimeError("GEMINI_MODEL or MODEL is not configured.")

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or GEMINI_API_KEY
        if not resolved_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=resolved_api_key,
            http_options={
                "retry_options": {"attempts": 1},
            },
        )

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        request = self._request(messages)
        response = self.client.interactions.create(**request)
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise RuntimeError("Gemini interaction output text is invalid.")
        return output_text

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        request = self._request(messages)
        request["stream"] = True
        stream = self.client.interactions.create(**request)

        for event in stream:
            if getattr(event, "event_type", None) != "step.delta":
                continue
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", None) != "text":
                continue
            text = getattr(delta, "text", None)
            if not isinstance(text, str):
                raise RuntimeError("Gemini streaming text delta is invalid.")
            yield text

    def complete_tool_turn(
        self,
        messages: list[LLMMessage],
        schemas: list[ToolSchema],
        events: tuple[AgentEvent, ...],
    ) -> AgentModelResponse:
        system_instruction, provider_input = self._agent_input(
            messages,
            events,
        )
        request: dict[str, object] = {
            "model": self.model,
            "input": provider_input,
            "generation_config": {
                "max_output_tokens": self.max_output_tokens,
            },
            "store": False,
            "timeout": self.timeout,
        }
        if system_instruction:
            request["system_instruction"] = system_instruction
        if schemas:
            request["tools"] = [self._tool_schema(schema) for schema in schemas]

        response = self.client.interactions.create(**request)
        return self._parse_agent_response(getattr(response, "steps", None))

    def _request(
        self,
        messages: list[LLMMessage],
    ) -> dict[str, object]:
        system_instruction, provider_input = self._base_input(messages)
        request: dict[str, object] = {
            "model": self.model,
            "input": provider_input,
            "generation_config": {
                "max_output_tokens": self.max_output_tokens,
            },
            "store": False,
            "timeout": self.timeout,
        }
        if system_instruction:
            request["system_instruction"] = system_instruction
        return request

    @staticmethod
    def _base_input(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict[str, object]]]:
        instructions: list[str] = []
        provider_input: list[dict[str, object]] = []

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                instructions.append(content)
                continue
            if role == "user":
                step_type = "user_input"
            elif role == "assistant":
                step_type = "model_output"
            else:
                raise RuntimeError(f"Gemini message role is unsupported: {role}.")
            provider_input.append(
                {
                    "type": step_type,
                    "content": [{"type": "text", "text": content}],
                }
            )

        return "\n\n".join(instructions), provider_input

    @classmethod
    def _agent_input(
        cls,
        messages: list[LLMMessage],
        events: tuple[AgentEvent, ...],
    ) -> tuple[str, list[dict[str, object]]]:
        system_instruction, provider_input = cls._base_input(messages)
        for event in events:
            provider_steps = getattr(
                event.response,
                "provider_steps",
                None,
            )
            if provider_steps is None:
                provider_input.extend(cls._neutral_response_steps(event.response))
            else:
                provider_input.extend(deepcopy(provider_steps))
            provider_input.extend(
                cls._tool_result_input(result) for result in event.tool_results
            )
        return system_instruction, provider_input

    @staticmethod
    def _neutral_response_steps(
        response: AgentModelResponse,
    ) -> list[dict[str, object]]:
        steps: list[dict[str, object]] = []
        for block in response.blocks:
            if isinstance(block, AgentTextBlock):
                steps.append(
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": block.text,
                            }
                        ],
                    }
                )
            else:
                steps.append(
                    {
                        "type": "function_call",
                        "id": block.id,
                        "name": block.name,
                        "arguments": deepcopy(block.arguments),
                    }
                )
        return steps

    @staticmethod
    def _tool_result_input(
        result: ToolResult,
    ) -> dict[str, object]:
        provider_result: dict[str, object] = {
            "type": "function_result",
            "name": result.name,
            "call_id": result.call_id,
            "result": [
                {
                    "type": "text",
                    "text": result.content,
                }
            ],
        }
        if result.is_error:
            provider_result["is_error"] = True
        return provider_result

    @staticmethod
    def _tool_schema(
        schema: ToolSchema,
    ) -> dict[str, object]:
        provider_schema = schema.to_json_schema()
        return {
            "type": "function",
            "name": provider_schema["name"],
            "description": provider_schema["description"],
            "parameters": provider_schema["input_schema"],
        }

    @classmethod
    def _parse_agent_response(
        cls,
        steps: object,
    ) -> AgentModelResponse:
        if not isinstance(steps, list):
            raise RuntimeError("Gemini interaction steps must be a list.")

        parsed_blocks: list[AgentResponseBlock] = []
        provider_steps: list[dict[str, object]] = []
        for step in steps:
            provider_steps.append(cls._dump_step(step))
            step_type = getattr(step, "type", None)
            if step_type == "model_output":
                cls._append_model_output(
                    parsed_blocks,
                    getattr(step, "content", None),
                )
                continue
            if step_type != "function_call":
                continue

            try:
                call = ToolCall(
                    id=getattr(step, "id", None),
                    name=getattr(step, "name", None),
                    arguments=getattr(step, "arguments", None),
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError("Gemini function-call step is invalid.") from error
            parsed_blocks.append(call)

        return _GeminiModelResponse(
            parsed_blocks,
            provider_steps,
        )

    @staticmethod
    def _dump_step(step: object) -> dict[str, object]:
        model_dump = getattr(step, "model_dump", None)
        if not callable(model_dump):
            raise RuntimeError("Gemini interaction step cannot be preserved.")
        dumped = model_dump()
        if not isinstance(dumped, dict):
            raise RuntimeError("Gemini interaction step dump is invalid.")
        return dumped

    @staticmethod
    def _append_model_output(
        parsed_blocks: list[AgentResponseBlock],
        content: object,
    ) -> None:
        if not isinstance(content, list):
            raise RuntimeError("Gemini model output content must be a list.")
        for block in content:
            if getattr(block, "type", None) != "text":
                continue
            text = getattr(block, "text", None)
            if not isinstance(text, str):
                raise RuntimeError("Gemini model output text is invalid.")
            parsed_blocks.append(AgentTextBlock(text))
