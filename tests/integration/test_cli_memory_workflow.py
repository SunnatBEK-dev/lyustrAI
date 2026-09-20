import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.memory import JSONMemoryStore, LongTermMemory
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.in_memory import InMemoryVectorStore
from ai_sdk.retrieval.retriever import SemanticRetriever
from ai_sdk.storage.json import JSONConversationRepository
from app.cli import run_cli

pytestmark = pytest.mark.integration


class UnusedEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [[1.0] for _ in texts]


class RecordingLLMClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        raise NotImplementedError

    def stream(self, messages):
        self.received_messages = messages
        yield "Memory-aware answer"


def test_cli_manages_and_recalls_persistent_long_term_memory(
    tmp_path,
    capsys,
):
    memory_file = tmp_path / "memories.json"
    memory_store = JSONMemoryStore(memory_file)
    memory_store.add(
        LongTermMemory(
            "mem_language",
            "Preferred language is Uzbek",
        )
    )
    conversation = Conversation()
    client = RecordingLLMClient()
    manager = RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=JSONConversationRepository(tmp_path / "chat.json"),
        chunker=TextChunker(
            chunk_size=100,
            overlap=0,
        ),
        retriever=SemanticRetriever(
            embedding_client=UnusedEmbeddingClient(),
            vector_store=InMemoryVectorStore(),
        ),
        memory_store=memory_store,
        memory_retrieval_k=1,
    )
    commands = iter(
        [
            "/memories",
            "Which language is preferred?",
            "/forget mem_language",
            "/memories",
            "/remember Favorite editor is PyCharm",
            "/memories",
            "/exit",
        ]
    )

    run_cli(
        manager,
        input_fn=lambda _: next(commands),
    )
    output = capsys.readouterr().out

    assert "mem_language | Preferred language is Uzbek" in output
    assert "Memory-aware answer" in output
    assert "Forgot mem_language." in output
    assert "Long-term memory is empty." in output
    assert "Remembered mem_" in output
    assert "Favorite editor is PyCharm" in output
    assert "Preferred language is Uzbek" in (client.received_messages[-1]["content"])
    restored = JSONMemoryStore(memory_file)
    assert restored.count() == 1
    assert restored.list_memories()[0].content == ("Favorite editor is PyCharm")
