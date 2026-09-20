from collections.abc import Sequence

from ai_sdk.context.summary import ConversationSummarizer
from ai_sdk.context.window import SlidingContextWindow
from ai_sdk.core.conversation import Conversation
from ai_sdk.llm.types import LLMMessage
from ai_sdk.memory.model import MemorySearchResult
from ai_sdk.retrieval.search import SearchResult


class PromptBuilder:
    def __init__(
        self,
        conversation: Conversation,
        context_window: SlidingContextWindow | None = None,
        summary_memory: ConversationSummarizer | None = None,
    ) -> None:
        if summary_memory is not None and context_window is None:
            raise ValueError("Summary memory requires a context window.")

        self.conversation = conversation
        self.context_window = context_window
        self.summary_memory = summary_memory

    def build_messages(
        self,
        retrieval_results: Sequence[SearchResult] = (),
        memory_results: Sequence[MemorySearchResult] = (),
    ) -> list[LLMMessage]:
        messages: list[LLMMessage] = []

        for message in self.conversation.history():
            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        if retrieval_results:
            self._augment_latest_user_message(
                messages,
                retrieval_results,
            )

        if self.context_window is not None:
            selection = self.context_window.partition(messages)
            messages = selection.included

            if self.summary_memory is not None and selection.excluded:
                summary = self.summary_memory.summarize(selection.excluded)

                if summary:
                    self._augment_latest_user_with_summary(
                        messages,
                        summary,
                    )

        if memory_results:
            self._augment_latest_user_with_memories(
                messages,
                memory_results,
            )

        return messages

    @staticmethod
    def _augment_latest_user_message(
        messages: list[LLMMessage],
        retrieval_results: Sequence[SearchResult],
    ) -> None:
        for message in reversed(messages):
            if message["role"] != "user":
                continue

            context = "\n\n".join(
                f"[{index}]\n{result.chunk.content}"
                for index, result in enumerate(
                    retrieval_results,
                    start=1,
                )
            )
            question = message["content"]
            message["content"] = (
                "Instructions:\n"
                "Use the retrieved context as reference data. "
                "Cite supporting context with [n]. If the "
                "context is insufficient, say so.\n\n"
                "Retrieved context:\n"
                f"{context}\n\n"
                "User question:\n"
                f"{question}"
            )
            return

        raise RuntimeError("Retrieval context requires a user message.")

    @staticmethod
    def _augment_latest_user_with_summary(
        messages: list[LLMMessage],
        summary: str,
    ) -> None:
        for message in reversed(messages):
            if message["role"] != "user":
                continue

            current_prompt = message["content"]
            message["content"] = (
                "Conversation memory from older turns "
                "(extractive and possibly incomplete):\n"
                f"{summary}\n\n"
                "Current prompt:\n"
                f"{current_prompt}"
            )
            return

        raise RuntimeError("Summary memory requires a user message.")

    @staticmethod
    def _augment_latest_user_with_memories(
        messages: list[LLMMessage],
        memory_results: Sequence[MemorySearchResult],
    ) -> None:
        for message in reversed(messages):
            if message["role"] != "user":
                continue

            memory_context = "\n".join(
                f"[M{index}] {result.memory.content}"
                for index, result in enumerate(
                    memory_results,
                    start=1,
                )
            )
            current_prompt = message["content"]
            message["content"] = (
                "Relevant user-approved long-term memory "
                "(use only when helpful):\n"
                f"{memory_context}\n\n"
                "Current prompt:\n"
                f"{current_prompt}"
            )
            return

        raise RuntimeError("Long-term memory requires a user message.")
