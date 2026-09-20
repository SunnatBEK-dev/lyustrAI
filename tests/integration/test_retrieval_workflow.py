import pytest

from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.in_memory import (
    InMemoryVectorStore,
)
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)

pytestmark = pytest.mark.integration


class KeywordEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        vectors = []

        for text in texts:
            lowered = text.lower()

            if "python" in lowered:
                vectors.append([1.0, 0.0])
            elif "cooking" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])

        return vectors


def test_document_to_semantic_result_workflow():
    document = Document(
        id="doc_workflow",
        content="Python functions Cooking recipes",
        metadata={"source": "guide.txt"},
    )
    chunks = TextChunker(
        chunk_size=16,
        overlap=0,
    ).split(document)
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )

    retriever.index(chunks)
    results = retriever.retrieve(
        "How do Python functions work?",
        k=1,
    )
    conversation = Conversation()
    conversation.add_user("How do Python functions work?")
    messages = PromptBuilder(conversation).build_messages(retrieval_results=results)

    assert len(chunks) == 2
    assert results[0].chunk.content == "Python functions"
    assert results[0].chunk.metadata == {"source": "guide.txt"}
    assert "Python functions" in messages[-1]["content"]
    assert "How do Python functions work?" in messages[-1]["content"]
    assert conversation.last_message().content == ("How do Python functions work?")
