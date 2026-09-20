import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.json_store import JSONVectorStore
from ai_sdk.storage.json import JSONConversationRepository

pytestmark = pytest.mark.integration


class MisleadingEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [
            [0.0, 1.0] if text.startswith("Exact error") else [1.0, 0.0]
            for text in texts
        ]


class RecordingLLMClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        return "Hybrid grounded answer"

    def stream(self, messages):
        raise NotImplementedError


def test_hybrid_rag_recovers_exact_term_when_embeddings_miss(
    tmp_path,
):
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
        retriever=HybridRetriever(
            embedding_client=MisleadingEmbeddingClient(),
            vector_store=JSONVectorStore(tmp_path / "vectors.json"),
        ),
        retrieval_k=1,
    )
    manager.index_document(
        Document(
            id="doc_semantic",
            content="Generic semantic candidate",
            metadata={"source": "generic.txt"},
        )
    )
    manager.index_document(
        Document(
            id="doc_exact",
            content="Exact error ZX-81 means the token expired.",
            metadata={"source": "errors.txt"},
        )
    )

    response = manager.send_message("What does ZX-81 mean?")

    assert response == "Hybrid grounded answer"
    assert "Exact error ZX-81" in (client.received_messages[-1]["content"])
    assert "Generic semantic candidate" not in (client.received_messages[-1]["content"])
    assert manager.last_citations[0].document_id == ("doc_exact")
    assert manager.last_citations[0].source == "errors.txt"
