from hashlib import sha256

import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.ingestion import (
    DirectorySynchronizer,
    create_default_ingestor,
)
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.json_store import JSONVectorStore
from ai_sdk.retrieval.retriever import SemanticRetriever
from ai_sdk.storage.json import JSONConversationRepository

pytestmark = pytest.mark.integration


class RecordingEmbeddingClient(BaseEmbeddingClient):
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class UnusedLLMClient:
    def ask(self, messages):
        raise NotImplementedError

    def stream(self, messages):
        raise NotImplementedError


def create_rag_manager(tmp_path, embedding_client):
    conversation = Conversation()

    return RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=UnusedLLMClient(),
        repository=JSONConversationRepository(tmp_path / "chat.json"),
        chunker=TextChunker(
            chunk_size=100,
            overlap=0,
        ),
        retriever=SemanticRetriever(
            embedding_client=embedding_client,
            vector_store=JSONVectorStore(tmp_path / "vectors.json"),
        ),
    )


def test_directory_sync_survives_restart_and_skips_unchanged_files(
    tmp_path,
):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    changed_path = directory / "changed.txt"
    removed_path = directory / "removed.md"
    changed_path.write_text("Old", encoding="utf-8")
    removed_path.write_text("Remove", encoding="utf-8")
    first_embedding = RecordingEmbeddingClient()
    first_manager = create_rag_manager(
        tmp_path,
        first_embedding,
    )
    first_result = DirectorySynchronizer(
        create_default_ingestor(),
        first_manager,
    ).sync(directory)
    removed_id = next(
        item.document_id
        for item in first_manager.document_catalog()
        if item.source == str(removed_path.resolve())
    )
    changed_path.write_text("Updated", encoding="utf-8")
    removed_path.unlink()

    second_embedding = RecordingEmbeddingClient()
    second_manager = create_rag_manager(
        tmp_path,
        second_embedding,
    )
    second_result = DirectorySynchronizer(
        create_default_ingestor(),
        second_manager,
    ).sync(directory)
    persisted_catalog = second_manager.document_catalog()

    assert len(first_result.indexed_documents) == 2
    assert len(first_embedding.calls) == 2
    assert len(second_result.indexed_documents) == 1
    assert second_result.removed_documents == (removed_id,)
    assert second_embedding.calls == [["Updated"]]
    assert len(persisted_catalog) == 1
    assert persisted_catalog[0].content_hash == sha256(b"Updated").hexdigest()
    assert persisted_catalog[0].ingestion_root == str(directory.resolve())

    third_embedding = RecordingEmbeddingClient()
    third_manager = create_rag_manager(
        tmp_path,
        third_embedding,
    )
    third_result = DirectorySynchronizer(
        create_default_ingestor(),
        third_manager,
    ).sync(directory)

    assert third_result.indexed_documents == ()
    assert len(third_result.unchanged_documents) == 1
    assert third_result.removed_documents == ()
    assert third_embedding.calls == []
