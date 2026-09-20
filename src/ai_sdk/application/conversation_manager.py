from collections.abc import Iterator

from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.memory.base import BaseMemoryStore
from ai_sdk.memory.model import (
    LongTermMemory,
    MemorySearchResult,
)
from ai_sdk.observability import (
    TraceCategory,
    Tracer,
    trace_span,
)
from ai_sdk.storage.base import ConversationRepository
from ai_sdk.tools.executor import ToolExecutor


class ConversationManager:
    def __init__(
        self,
        conversation: Conversation,
        prompt_builder: PromptBuilder,
        client: BaseLLMClient,
        repository: ConversationRepository,
        memory_store: BaseMemoryStore | None = None,
        memory_retrieval_k: int = 3,
        tool_executor: ToolExecutor | None = None,
        max_tool_rounds: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        if memory_retrieval_k <= 0:
            raise ValueError("Memory retrieval top-k must be greater than zero.")

        if (
            not isinstance(max_tool_rounds, int)
            or isinstance(max_tool_rounds, bool)
            or max_tool_rounds <= 0
        ):
            raise ValueError("Maximum tool rounds must be greater than zero.")

        if tool_executor is not None and not isinstance(tool_executor, ToolExecutor):
            raise TypeError("Tool executor must be a ToolExecutor.")
        if tracer is not None and not isinstance(tracer, Tracer):
            raise TypeError("Conversation tracer must be a Tracer.")

        self.conversation = conversation
        self.prompt_builder = prompt_builder
        self.client = client
        self.repository = repository
        self.memory_store = memory_store
        self.memory_retrieval_k = memory_retrieval_k
        self.tool_executor = tool_executor
        self.max_tool_rounds = max_tool_rounds
        self.tracer = (
            tracer
            if tracer is not None
            else (None if tool_executor is None else tool_executor.tracer)
        )

    def _build_messages(
        self,
        text: str,
    ) -> list[LLMMessage]:
        return self.prompt_builder.build_messages(
            memory_results=self._recall_memories(text)
        )

    def remember(self, content: str) -> LongTermMemory:
        memory_store = self._require_memory_store()
        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError("Long-term memory content cannot be empty.")

        for memory in memory_store.list_memories():
            if memory.content.casefold() == normalized_content.casefold():
                return memory

        memory = LongTermMemory.create(normalized_content)
        memory_store.add(memory)
        return memory

    def list_memories(self) -> list[LongTermMemory]:
        return self._require_memory_store().list_memories()

    def forget(self, memory_id: str) -> bool:
        return self._require_memory_store().delete(memory_id)

    def _recall_memories(
        self,
        query: str,
    ) -> list[MemorySearchResult]:
        if self.memory_store is None or not query.strip():
            return []

        return self.memory_store.search(
            query=query,
            k=self.memory_retrieval_k,
        )

    def _require_memory_store(self) -> BaseMemoryStore:
        if self.memory_store is None:
            raise RuntimeError("Long-term memory is not configured.")

        return self.memory_store

    def _ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        with trace_span(
            self.tracer,
            "llm.generate",
            TraceCategory.LLM,
            {
                "llm.message_count": len(messages),
                "llm.tools_enabled": self.tool_executor is not None,
            },
        ) as span:
            if self.tool_executor is None:
                response = self.client.ask(messages)
            else:
                ask_with_tools = getattr(
                    self.client,
                    "ask_with_tools",
                    None,
                )

                if not callable(ask_with_tools):
                    raise RuntimeError("Configured LLM client does not support tools.")

                kwargs: dict[str, object] = {
                    "max_tool_rounds": self.max_tool_rounds,
                }
                if self.tracer is not None:
                    kwargs["tracer"] = self.tracer
                response = ask_with_tools(
                    messages,
                    self.tool_executor,
                    **kwargs,
                )
            if span is not None:
                span.set_attribute(
                    "llm.response_char_count",
                    len(response),
                )
            return response

    def send_message(
        self,
        text: str,
    ) -> str:
        with trace_span(
            self.tracer,
            "conversation.send",
            TraceCategory.WORKFLOW,
            {
                "conversation.message_count": len(self.conversation.messages),
                "conversation.input_length": (
                    len(text) if isinstance(text, str) else 0
                ),
            },
        ) as span:
            response = self._send_message(text)
            if span is not None:
                span.set_attribute(
                    "conversation.response_length",
                    len(response),
                )
            return response

    def _send_message(
        self,
        text: str,
    ) -> str:
        user_message = self.conversation.add_user(text)
        assistant_message = None

        try:
            messages = self._build_messages(text)

            response = self._ask(messages)

            assistant_message = self.conversation.add_assistant(response)

            self.repository.save(self.conversation)

            return response

        except Exception:
            if assistant_message is not None:
                self.conversation.delete_message(assistant_message.id)

            self.conversation.delete_message(user_message.id)

            raise

    def stream_message(
        self,
        text: str,
    ) -> Iterator[str]:
        with trace_span(
            self.tracer,
            "conversation.stream",
            TraceCategory.WORKFLOW,
            {
                "conversation.message_count": len(self.conversation.messages),
                "conversation.input_length": (
                    len(text) if isinstance(text, str) else 0
                ),
            },
        ) as span:
            response_length = 0
            for chunk in self._stream_message(text):
                response_length += len(chunk)
                yield chunk
            if span is not None:
                span.set_attribute(
                    "conversation.response_length",
                    response_length,
                )

    def _stream_message(
        self,
        text: str,
    ) -> Iterator[str]:
        if self.tool_executor is not None:
            raise RuntimeError(
                "Tool-enabled streaming is not supported. Use send_message instead."
            )

        user_message = self.conversation.add_user(text)

        chunks = []
        assistant_message = None

        try:
            messages = self._build_messages(text)

            with trace_span(
                self.tracer,
                "llm.stream",
                TraceCategory.LLM,
                {"llm.message_count": len(messages)},
            ) as span:
                for chunk in self.client.stream(messages):
                    chunks.append(chunk)
                    yield chunk
                if span is not None:
                    span.set_attribute(
                        "llm.response_char_count",
                        sum(len(chunk) for chunk in chunks),
                    )

            response = "".join(chunks)

            assistant_message = self.conversation.add_assistant(response)

            self.repository.save(self.conversation)

        except (Exception, GeneratorExit):
            if assistant_message is not None:
                self.conversation.delete_message(assistant_message.id)

            self.conversation.delete_message(user_message.id)

            raise
