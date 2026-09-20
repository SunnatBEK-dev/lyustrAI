from collections.abc import Sequence

from ai_sdk.embeddings.base import EmbeddingVector
from ai_sdk.retrieval.catalog import (
    IndexedDocument,
    build_document_catalog,
)
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    EmbeddedChunk,
    SearchResult,
    bm25_search,
    top_k_search,
)
from ai_sdk.retrieval.vector_store import (
    BaseVectorStore,
)


class InMemoryVectorStore(BaseVectorStore):
    """Small learning-stage vector store backed by process memory."""

    def __init__(self) -> None:
        self._items: dict[str, EmbeddedChunk] = {}
        self._dimension: int | None = None

    def add(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
    ) -> None:
        if not vector:
            raise ValueError("Embedding vector cannot be empty.")

        stored_vector = [float(value) for value in vector]
        dimension = len(stored_vector)

        if self._dimension is not None and dimension != self._dimension:
            raise ValueError(
                "Embedding vector dimension does not match the vector store."
            )

        if self._dimension is None:
            self._dimension = dimension

        self._items[chunk.id] = (
            chunk,
            stored_vector,
        )

    def search(
        self,
        query_vector: EmbeddingVector,
        k: int = 5,
    ) -> list[SearchResult]:
        if self._dimension is not None and len(query_vector) != self._dimension:
            raise ValueError("Query vector dimension does not match the vector store.")

        return top_k_search(
            query_vector=query_vector,
            candidates=list(self._items.values()),
            k=k,
        )

    def replace_document(
        self,
        document_id: str,
        items: Sequence[EmbeddedChunk],
    ) -> None:
        item_list = list(items)
        self._validate_document_items(
            document_id,
            item_list,
        )
        previous_items = self._items.copy()
        previous_dimension = self._dimension

        try:
            self._delete_document_in_memory(document_id)

            for chunk, vector in item_list:
                self.add(chunk, vector)
        except Exception:
            self._items = previous_items
            self._dimension = previous_dimension
            raise

    def lexical_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        return bm25_search(
            query=query,
            candidates=[chunk for chunk, _ in self._items.values()],
            k=k,
        )

    def delete(self, chunk_id: str) -> bool:
        if chunk_id not in self._items:
            return False

        del self._items[chunk_id]

        if not self._items:
            self._dimension = None

        return True

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        self._validate_document_id(document_id)
        return self._delete_document_in_memory(document_id)

    def document_catalog(
        self,
    ) -> list[IndexedDocument]:
        return build_document_catalog(chunk for chunk, _ in self._items.values())

    def clear(self) -> None:
        self._items.clear()
        self._dimension = None

    def count(self) -> int:
        return len(self._items)

    def _delete_document_in_memory(
        self,
        document_id: str,
    ) -> int:
        chunk_ids = [
            chunk_id
            for chunk_id, (chunk, _) in self._items.items()
            if chunk.document_id == document_id
        ]

        for chunk_id in chunk_ids:
            del self._items[chunk_id]

        if not self._items:
            self._dimension = None

        return len(chunk_ids)

    @staticmethod
    def _validate_document_items(
        document_id: str,
        items: Sequence[EmbeddedChunk],
    ) -> None:
        InMemoryVectorStore._validate_document_id(document_id)

        if any(chunk.document_id != document_id for chunk, _ in items):
            raise ValueError(
                "Replacement chunks must belong to the requested document."
            )

    @staticmethod
    def _validate_document_id(
        document_id: str,
    ) -> None:
        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")
