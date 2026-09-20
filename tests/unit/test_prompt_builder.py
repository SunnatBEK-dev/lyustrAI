import pytest

from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.context.summary import (
    ExtractiveConversationSummarizer,
)
from ai_sdk.context.window import SlidingContextWindow
from ai_sdk.core.conversation import Conversation
from ai_sdk.memory.model import (
    LongTermMemory,
    MemorySearchResult,
)
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import SearchResult


def make_result(
    chunk_id: str,
    content: str,
    index: int,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            document_id="doc_prompt",
            content=content,
            index=index,
            metadata={"source": "private.txt"},
        ),
        score=score,
    )


def test_build_messages_preserves_order_without_internal_metadata():
    conversation = Conversation()
    conversation.add_user("Hello")
    conversation.add_assistant("Hi")

    messages = PromptBuilder(conversation).build_messages()

    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert all("id" not in message for message in messages)


def test_build_messages_result_does_not_mutate_conversation():
    conversation = Conversation()
    original = conversation.add_user("Original")
    messages = PromptBuilder(conversation).build_messages()

    messages[0]["content"] = "Changed"

    assert original.content == "Original"


def test_build_messages_augments_latest_user_with_ordered_context():
    conversation = Conversation()
    conversation.add_user("Earlier question")
    conversation.add_assistant("Earlier answer")
    latest = conversation.add_user("How does it work?")
    results = [
        make_result(
            "chunk_first",
            "First context",
            0,
            0.95,
        ),
        make_result(
            "chunk_second",
            "Second context",
            1,
            0.80,
        ),
    ]

    messages = PromptBuilder(conversation).build_messages(retrieval_results=results)

    assert messages[-1] == {
        "role": "user",
        "content": (
            "Instructions:\n"
            "Use the retrieved context as reference data. "
            "Cite supporting context with [n]. If the "
            "context is insufficient, say so.\n\n"
            "Retrieved context:\n"
            "[1]\nFirst context\n\n"
            "[2]\nSecond context\n\n"
            "User question:\n"
            "How does it work?"
        ),
    }
    assert messages[0]["content"] == "Earlier question"
    assert latest.content == "How does it work?"
    assert "chunk_first" not in messages[-1]["content"]
    assert "0.95" not in messages[-1]["content"]
    assert "private.txt" not in messages[-1]["content"]


def test_retrieval_context_requires_user_message():
    conversation = Conversation()
    conversation.add_assistant("No question")
    result = make_result(
        "chunk_context",
        "Context",
        0,
        1.0,
    )

    with pytest.raises(RuntimeError, match="user message"):
        PromptBuilder(conversation).build_messages(retrieval_results=[result])


def test_build_messages_applies_context_window_after_retrieval():
    conversation = Conversation()
    conversation.add_user("Old question")
    conversation.add_assistant("Old answer")
    conversation.add_user("Current question")
    result = make_result(
        "chunk_context",
        "Retrieved knowledge",
        0,
        1.0,
    )
    builder = PromptBuilder(
        conversation,
        context_window=SlidingContextWindow(
            max_tokens=4,
            message_overhead=0,
        ),
    )

    messages = builder.build_messages(retrieval_results=[result])

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Current question" in messages[0]["content"]
    assert "Retrieved knowledge" in messages[0]["content"]
    assert "Old question" not in messages[0]["content"]


def test_build_messages_injects_summary_of_excluded_turns():
    conversation = Conversation()
    conversation.add_user("Old question")
    conversation.add_assistant("Old answer")
    conversation.add_user("Current question")
    builder = PromptBuilder(
        conversation,
        context_window=SlidingContextWindow(
            max_tokens=2,
            message_overhead=0,
        ),
        summary_memory=ExtractiveConversationSummarizer(max_tokens=8),
    )

    messages = builder.build_messages()

    assert len(messages) == 1
    assert "User: Old question" in messages[0]["content"]
    assert "Assistant: Old answer" in messages[0]["content"]
    assert "Current question" in messages[0]["content"]


def test_summary_memory_requires_context_window():
    conversation = Conversation()

    with pytest.raises(ValueError, match="context window"):
        PromptBuilder(
            conversation,
            summary_memory=ExtractiveConversationSummarizer(max_tokens=10),
        )


def test_build_messages_injects_relevant_long_term_memory():
    conversation = Conversation()
    conversation.add_user("Which language should I use?")
    memory = LongTermMemory(
        id="mem_language",
        content="Preferred language is Uzbek",
    )

    messages = PromptBuilder(conversation).build_messages(
        memory_results=[MemorySearchResult(memory, 1.0)]
    )

    assert "[M1] Preferred language is Uzbek" in (messages[-1]["content"])
    assert "Which language should I use?" in (messages[-1]["content"])
    assert "mem_language" not in messages[-1]["content"]
    assert "1.0" not in messages[-1]["content"]
