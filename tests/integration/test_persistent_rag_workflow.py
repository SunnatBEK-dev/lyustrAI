import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.json_store import JSONVectorStore
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)
from ai_sdk.storage.json import JSONConversationRepository

pytestmark = pytest.mark.integration


class KeywordEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [
            [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0] for text in texts
        ]


class RecordingLLMClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        return "Grounded after restart"

    def stream(self, messages):
        raise NotImplementedError


def test_rag_reuses_persisted_index_after_restart(tmp_path):
    vector_path = tmp_path / "vectors.json"
    embedding_client = KeywordEmbeddingClient()
    first_retriever = SemanticRetriever(
        embedding_client=embedding_client,
        vector_store=JSONVectorStore(vector_path),
    )
    document = Document(
        id="doc_persistent_rag",
        content="Python functions Cooking recipes",
    )
    chunks = TextChunker(
        chunk_size=16,
        overlap=0,
    ).split(document)
    first_retriever.index_document(
        document.id,
        chunks,
    )

    restarted_retriever = SemanticRetriever(
        embedding_client=embedding_client,
        vector_store=JSONVectorStore(vector_path),
    )
    conversation = Conversation()
    client = RecordingLLMClient()
    manager = RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=JSONConversationRepository(tmp_path / "chat.json"),
        chunker=TextChunker(),
        retriever=restarted_retriever,
        retrieval_k=1,
    )

    answer = manager.send_message("How do Python functions work?")

    assert answer == "Grounded after restart"
    assert len(chunks) == 2
    assert "Python functions" in (client.received_messages[-1]["content"])


def test_reindex_and_delete_document_persist_after_restart(
    tmp_path,
):
    vector_path = tmp_path / "vectors.json"
    chunker = TextChunker(
        chunk_size=100,
        overlap=0,
    )
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=JSONVectorStore(vector_path),
    )
    original = Document(
        id="doc_lifecycle",
        content="Python functions",
    )
    updated = Document(
        id="doc_lifecycle",
        content="Cooking recipes",
    )
    original_chunks = chunker.split(original)
    updated_chunks = chunker.split(updated)

    retriever.index_document(
        original.id,
        original_chunks,
    )
    restarted = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=JSONVectorStore(vector_path),
    )
    restarted.index_document(
        updated.id,
        updated_chunks,
    )

    after_reindex = JSONVectorStore(vector_path)
    assert after_reindex.count() == 1
    assert (
        after_reindex.search(
            [0.0, 1.0],
            k=1,
        )[0].chunk.id
        == updated_chunks[0].id
    )
    assert restarted.delete_document(updated.id) == 1
    assert JSONVectorStore(vector_path).count() == 0
