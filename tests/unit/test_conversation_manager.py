import pytest

from ai_sdk.application.conversation_manager import ConversationManager
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.memory.model import (
    LongTermMemory,
    MemorySearchResult,
)
from ai_sdk.observability import (
    InMemoryTraceCollector,
    Tracer,
    TraceStatus,
)
from ai_sdk.tools import ToolExecutor, ToolRegistry


class FakeClient:
    def __init__(
        self,
        response="Assistant response",
        chunks=None,
        error=None,
    ):
        self.response = response
        self.chunks = chunks or ["Assistant ", "response"]
        self.error = error
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        if self.error:
            raise self.error
        return self.response

    def stream(self, messages):
        self.received_messages = messages
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class FakeToolClient(FakeClient):
    def __init__(self, response="Tool-assisted response", error=None):
        super().__init__(response=response, error=error)
        self.tool_request = None

    def ask(self, messages):
        raise AssertionError("Plain ask must not be used with tools.")

    def ask_with_tools(
        self,
        messages,
        executor,
        *,
        max_tool_rounds,
    ):
        self.tool_request = (
            messages,
            executor,
            max_tool_rounds,
        )
        if self.error:
            raise self.error
        return self.response


class FakeRepository:
    def __init__(self, error=None):
        self.error = error
        self.saved = []

    def save(self, conversation):
        if self.error:
            raise self.error
        self.saved.append([message.to_dict() for message in conversation.history()])

    def load(self):
        return Conversation()


class FakeMemoryStore:
    def __init__(self, memories=None, results=None):
        self.memories = list(memories or [])
        self.results = list(results or [])
        self.searches = []

    def add(self, memory):
        self.memories.append(memory)

    def list_memories(self):
        return self.memories.copy()

    def delete(self, memory_id):
        before = len(self.memories)
        self.memories = [memory for memory in self.memories if memory.id != memory_id]
        return len(self.memories) < before

    def search(self, query, k=3):
        self.searches.append((query, k))
        return self.results[:k]

    def clear(self):
        self.memories.clear()

    def count(self):
        return len(self.memories)


def create_rag_manager(
    client=None,
    repository=None,
    memory_store=None,
    tool_executor=None,
    max_tool_rounds=8,
    tracer=None,
):
    conversation = Conversation()
    client = client or FakeClient()
    repository = repository or FakeRepository()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=repository,
        memory_store=memory_store,
        tool_executor=tool_executor,
        max_tool_rounds=max_tool_rounds,
        tracer=tracer,
    )
    return manager, conversation, client, repository


def test_send_message_orchestrates_and_persists_conversation():
    manager, conversation, client, repository = create_rag_manager()

    response = manager.send_message("User question")

    assert response == "Assistant response"
    assert client.received_messages == [
        {"role": "user", "content": "User question"},
    ]
    assert [message.content for message in conversation.history()] == [
        "User question",
        "Assistant response",
    ]
    assert len(repository.saved) == 1


def test_send_message_rolls_back_new_state_when_client_fails():
    manager, conversation, _, repository = create_rag_manager(
        client=FakeClient(error=RuntimeError("LLM failed")),
    )
    existing = conversation.add_user("Existing")

    with pytest.raises(RuntimeError, match="LLM failed"):
        manager.send_message("New")

    assert conversation.history() == [existing]
    assert repository.saved == []


def test_send_message_rolls_back_new_state_when_save_fails():
    manager, conversation, _, _ = create_rag_manager(
        repository=FakeRepository(error=OSError("save failed")),
    )
    existing = conversation.add_user("Existing")

    with pytest.raises(OSError, match="save failed"):
        manager.send_message("New")

    assert conversation.history() == [existing]


def test_stream_message_yields_chunks_and_persists_full_response():
    manager, conversation, client, repository = create_rag_manager(
        client=FakeClient(chunks=["A", "B", "C"]),
    )

    chunks = list(manager.stream_message("Question"))

    assert chunks == ["A", "B", "C"]
    assert client.received_messages == [
        {"role": "user", "content": "Question"},
    ]
    assert conversation.last_message().content == "ABC"
    assert len(repository.saved) == 1


def test_stream_message_traces_lengths_without_content():
    collector = InMemoryTraceCollector()
    manager, _, _, _ = create_rag_manager(
        tracer=Tracer(collector),
    )

    chunks = list(manager.stream_message("private question"))

    root, llm = collector.records()
    assert chunks == ["Assistant ", "response"]
    assert root.name == "conversation.stream"
    assert llm.name == "llm.stream"
    assert llm.parent_span_id == root.span_id
    assert root.attributes["conversation.response_length"] == 18
    assert llm.attributes["llm.response_char_count"] == 18
    assert root.status is llm.status is TraceStatus.OK
    assert "private question" not in str([record.to_dict() for record in (root, llm)])


def test_stream_message_rolls_back_when_stream_fails():
    manager, conversation, _, repository = create_rag_manager(
        client=FakeClient(
            chunks=["partial"],
            error=RuntimeError("stream failed"),
        ),
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        list(manager.stream_message("Question"))

    assert conversation.is_empty()
    assert repository.saved == []


def test_stream_message_rolls_back_when_consumer_closes_early():
    manager, conversation, _, repository = create_rag_manager(
        client=FakeClient(chunks=["first", "second"]),
    )
    stream = manager.stream_message("Question")

    assert next(stream) == "first"
    stream.close()

    assert conversation.is_empty()
    assert repository.saved == []


def test_stream_message_rolls_back_when_save_fails():
    manager, conversation, _, _ = create_rag_manager(
        repository=FakeRepository(error=OSError("save failed")),
    )

    with pytest.raises(OSError, match="save failed"):
        list(manager.stream_message("Question"))

    assert conversation.is_empty()


def test_manager_recalls_relevant_long_term_memory():
    memory = LongTermMemory(
        "mem_language",
        "Preferred language is Uzbek",
    )
    memory_store = FakeMemoryStore(
        memories=[memory],
        results=[MemorySearchResult(memory, 1.0)],
    )
    manager, _, client, _ = create_rag_manager(memory_store=memory_store)

    manager.send_message("Which language is preferred?")

    assert memory_store.searches == [("Which language is preferred?", 3)]
    assert "Preferred language is Uzbek" in (client.received_messages[-1]["content"])


def test_manager_manages_memories_without_duplicates():
    memory_store = FakeMemoryStore()
    manager, _, _, _ = create_rag_manager(memory_store=memory_store)

    first = manager.remember(" Preferred language is Uzbek ")
    duplicate = manager.remember("preferred language is uzbek")

    assert duplicate is first
    assert manager.list_memories() == [first]
    assert manager.forget(first.id) is True
    assert manager.forget(first.id) is False


def test_manager_rejects_unconfigured_or_invalid_memory():
    manager, _, _, _ = create_rag_manager()

    with pytest.raises(RuntimeError, match="not configured"):
        manager.list_memories()

    configured, _, _, _ = create_rag_manager(memory_store=FakeMemoryStore())

    with pytest.raises(ValueError, match="content"):
        configured.remember(" ")

    with pytest.raises(ValueError, match="greater than zero"):
        ConversationManager(
            conversation=Conversation(),
            prompt_builder=PromptBuilder(Conversation()),
            client=FakeClient(),
            repository=FakeRepository(),
            memory_retrieval_k=0,
        )


def test_manager_uses_tool_capable_client_and_persists_final_text():
    executor = ToolExecutor(ToolRegistry())
    client = FakeToolClient()
    manager, conversation, _, repository = create_rag_manager(
        client=client,
        tool_executor=executor,
        max_tool_rounds=4,
    )

    result = manager.send_message("Use a tool")

    assert result == "Tool-assisted response"
    assert client.tool_request == (
        [{"role": "user", "content": "Use a tool"}],
        executor,
        4,
    )
    assert [message.content for message in conversation.history()] == [
        "Use a tool",
        "Tool-assisted response",
    ]
    assert len(repository.saved) == 1


def test_manager_rolls_back_when_tool_client_fails():
    manager, conversation, _, repository = create_rag_manager(
        client=FakeToolClient(error=RuntimeError("tool loop failed")),
        tool_executor=ToolExecutor(ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="tool loop failed"):
        manager.send_message("Use a tool")

    assert conversation.is_empty()
    assert repository.saved == []


def test_manager_rejects_tools_for_non_tool_client():
    manager, conversation, _, repository = create_rag_manager(
        client=FakeClient(),
        tool_executor=ToolExecutor(ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="does not support tools"):
        manager.send_message("Use a tool")

    assert conversation.is_empty()
    assert repository.saved == []


def test_manager_rejects_streaming_when_tools_are_configured():
    manager, conversation, _, repository = create_rag_manager(
        client=FakeToolClient(),
        tool_executor=ToolExecutor(ToolRegistry()),
    )

    with pytest.raises(RuntimeError, match="streaming"):
        list(manager.stream_message("Use a tool"))

    assert conversation.is_empty()
    assert repository.saved == []


@pytest.mark.parametrize("max_tool_rounds", [0, -1, True, 1.5])
def test_manager_rejects_invalid_tool_round_limit(max_tool_rounds):
    with pytest.raises(ValueError, match="greater than zero"):
        create_rag_manager(max_tool_rounds=max_tool_rounds)


def test_manager_rejects_invalid_tool_executor():
    with pytest.raises(TypeError, match="ToolExecutor"):
        create_rag_manager(tool_executor=object())


def test_manager_rejects_invalid_tracer():
    with pytest.raises(TypeError, match="tracer"):
        create_rag_manager(tracer=object())
