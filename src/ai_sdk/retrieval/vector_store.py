from abc import ABC, abstractmethod
from collections.abc import Sequence

from ai_sdk.embeddings.base import EmbeddingVector
from ai_sdk.retrieval.catalog import IndexedDocument
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    EmbeddedChunk,
    SearchResult,
)


class BaseVectorStore(ABC):
    """Provider-neutral storage contract for embedded chunks."""

    @abstractmethod
    def add(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
    ) -> None:
        raise NotImplementedError

    def add_many(
        self,
        items: Sequence[EmbeddedChunk],
    ) -> None:
        for chunk, vector in items:
            self.add(chunk, vector)

    @abstractmethod
    def replace_document(
        self,
        document_id: str,
        items: Sequence[EmbeddedChunk],
    ) -> None:
        """Atomically replace every chunk for one document."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: EmbeddingVector,
        k: int = 5,
    ) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def lexical_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        """Return chunks ranked by exact lexical relevance."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """Delete all chunks for a document and return their count."""
        raise NotImplementedError

    def document_ids(self) -> list[str]:
        """Return indexed document IDs in deterministic order."""
        return [document.document_id for document in self.document_catalog()]

    @abstractmethod
    def document_catalog(
        self,
    ) -> list[IndexedDocument]:
        """Return source and chunk-count summaries."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
