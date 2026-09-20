import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.json_store import JSONVectorStore
from ai_sdk.retrieval.retriever import SemanticRetriever
from ai_sdk.storage.json import JSONConversationRepository
from app.cli import load_document, run_cli

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
        raise NotImplementedError

    def stream(self, messages):
        self.received_messages = messages
        yield "Grounded CLI answer"


def test_cli_indexes_directory_uses_catalog_and_removes_document(
    tmp_path,
    capsys,
):
    guide_directory = tmp_path / "Knowledge base"
    guide_directory.mkdir()
    guide_path = guide_directory / "Python guide.txt"
    guide_path.write_text(
        "Python functions contain reusable logic.",
        encoding="utf-8",
    )
    cooking_path = guide_directory / "Cooking.md"
    cooking_path.write_text(
        "Cooking recipes combine ingredients.",
        encoding="utf-8",
    )
    (guide_directory / "ignored.docx").write_bytes(b"not supported")
    document_id = load_document(str(guide_path)).id
    vector_store = JSONVectorStore(tmp_path / "vectors.json")
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=vector_store,
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
        retriever=retriever,
        retrieval_k=1,
    )
    commands = iter(
        [
            f"/index {guide_directory}",
            "/documents",
            "How do Python functions work?",
            f"/remove {document_id}",
            "/exit",
        ]
    )

    run_cli(
        manager,
        input_fn=lambda _: next(commands),
    )
    output = capsys.readouterr().out

    assert f"Synchronized {guide_directory.resolve()}" in output
    assert "indexed=2, unchanged=0, removed=0, chunks=2." in output
    assert f"- {document_id}" in output
    assert "chunks=1" in output
    assert f"source={guide_path.resolve()}" in output
    assert f"source={cooking_path.resolve()}" in output
    assert "Grounded CLI answer" in output
    assert "Sources:" in output
    assert f"[1] {guide_path.resolve()}" in output
    assert f"Removed {document_id}: 1 chunks." in output
    assert (
        "Python functions contain reusable logic."
        in (client.received_messages[-1]["content"])
    )
    assert str(guide_path.resolve()) not in (client.received_messages[-1]["content"])
    assert (
        "Cite supporting context with [n]" in (client.received_messages[-1]["content"])
    )
    assert vector_store.count() == 1
