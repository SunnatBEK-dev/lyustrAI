from collections.abc import Sequence

from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.catalog import IndexedDocument
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    EmbeddedChunk,
    SearchResult,
)
from ai_sdk.retrieval.vector_store import (
    BaseVectorStore,
)


class SemanticRetriever:
    """Coordinate embedding generation and vector-store search."""

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        vector_store: BaseVectorStore,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def index(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        items = self._embed_chunks(chunks)

        if not items:
            return

        self.vector_store.add_many(items)

    def index_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
    ) -> None:
        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        chunk_list = list(chunks)

        if any(chunk.document_id != document_id for chunk in chunk_list):
            raise ValueError("Indexed chunks must belong to the requested document.")

        items = self._embed_chunks(chunk_list)
        self.vector_store.replace_document(
            document_id,
            items,
        )

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        return self.vector_store.delete_document(document_id)

    def list_documents(self) -> list[str]:
        return self.vector_store.document_ids()

    def document_catalog(
        self,
    ) -> list[IndexedDocument]:
        return self.vector_store.document_catalog()

    def _embed_chunks(
        self,
        chunks: Sequence[Chunk],
    ) -> list[EmbeddedChunk]:
        chunk_list = list(chunks)

        if not chunk_list:
            return []

        vectors = self.embedding_client.embed([chunk.content for chunk in chunk_list])

        if len(vectors) != len(chunk_list):
            raise RuntimeError(
                "Embedding client must return one vector for each chunk."
            )

        return list(
            zip(
                chunk_list,
                vectors,
                strict=True,
            )
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        query_vector = self.embedding_client.embed_one(query)

        return self.vector_store.search(
            query_vector=query_vector,
            k=k,
        )
