import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.observability import (
    InMemoryTraceCollector,
    Tracer,
    TraceStatus,
)
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.in_memory import (
    InMemoryVectorStore,
)
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)
from ai_sdk.storage.json import JSONConversationRepository

pytestmark = pytest.mark.integration


class KeywordEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        vectors = []

        for text in texts:
            if "python" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return vectors


class RecordingLLMClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        return "Grounded answer"

    def stream(self, messages):
        self.received_messages = messages
        yield "Grounded "
        yield "answer"


def test_full_offline_rag_workflow(tmp_path):
    repository = JSONConversationRepository(tmp_path / "chat.json")
    conversation = Conversation()
    client = RecordingLLMClient()
    collector = InMemoryTraceCollector()
    tracer = Tracer(collector)
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    manager = RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=repository,
        chunker=TextChunker(
            chunk_size=16,
            overlap=0,
        ),
        retriever=retriever,
        retrieval_k=1,
        tracer=tracer,
    )
    document = Document(
        id="doc_rag_workflow",
        content="Python functions Cooking recipes",
        metadata={"source": "guide.txt"},
    )

    chunks = manager.index_document(document)
    response_chunks = list(manager.stream_message("How do Python functions work?"))
    restored = repository.load()

    assert len(chunks) == 2
    assert response_chunks == ["Grounded ", "answer"]
    assert "Python functions" in (client.received_messages[-1]["content"])
    assert [message.content for message in restored.history()] == [
        "How do Python functions work?",
        "Grounded answer",
    ]
    records = collector.records()
    root = next(record for record in records if record.name == "conversation.stream")
    assert {
        "retrieval.search",
        "llm.stream",
    }.issubset({record.name for record in records})
    assert all(record.trace_id == root.trace_id for record in records)
    assert all(record.status is TraceStatus.OK for record in records)
    assert "How do Python functions work?" not in str(
        [record.to_dict() for record in records]
    )
