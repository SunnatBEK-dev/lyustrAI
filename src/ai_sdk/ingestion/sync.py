from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ai_sdk.ingestion.ingestor import DocumentIngestor
from ai_sdk.retrieval.catalog import IndexedDocument
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.document import Document


class DirectoryIndex(Protocol):
    """Operations required to synchronize an ingestion directory."""

    def document_catalog(
        self,
    ) -> list[IndexedDocument]: ...

    def index_document(
        self,
        document: Document,
    ) -> Sequence[Chunk]: ...

    def delete_document(
        self,
        document_id: str,
    ) -> int: ...


@dataclass(frozen=True)
class DirectorySyncResult:
    """Summary of one incremental directory synchronization."""

    root: str
    indexed_documents: tuple[str, ...]
    unchanged_documents: tuple[str, ...]
    removed_documents: tuple[str, ...]
    indexed_chunks: int


class DirectorySynchronizer:
    """Keep a directory-backed document index up to date."""

    def __init__(
        self,
        ingestor: DocumentIngestor,
        index: DirectoryIndex,
    ) -> None:
        self.ingestor = ingestor
        self.index = index

    def sync(
        self,
        path: str | Path,
    ) -> DirectorySyncResult:
        root = Path(path).expanduser()

        if not root.exists():
            raise FileNotFoundError(f"Directory path does not exist: {root}")

        if not root.is_dir():
            raise ValueError("Directory synchronization requires a directory.")

        resolved_root = root.resolve()
        documents = self.ingestor.ingest(
            resolved_root,
            allow_empty=True,
        )
        existing_documents = [
            document
            for document in self.index.document_catalog()
            if self._belongs_to_root(
                document,
                resolved_root,
            )
        ]
        existing_by_source = {
            document.source: document for document in existing_documents
        }
        current_sources = {document.metadata["source"] for document in documents}
        indexed_documents: list[str] = []
        unchanged_documents: list[str] = []
        removed_document_ids = {
            document.document_id
            for document in existing_documents
            if document.source not in current_sources
        }
        indexed_chunks = 0

        for document in documents:
            source = document.metadata["source"]
            existing = existing_by_source.get(source)

            if self._is_unchanged(document, existing):
                unchanged_documents.append(document.id)
                continue

            chunks = self.index.index_document(document)
            indexed_documents.append(document.id)
            indexed_chunks += len(chunks)

            if existing is not None and existing.document_id != document.id:
                removed_document_ids.add(existing.document_id)

        removed_documents = []

        for document_id in sorted(removed_document_ids):
            self.index.delete_document(document_id)
            removed_documents.append(document_id)

        return DirectorySyncResult(
            root=str(resolved_root),
            indexed_documents=tuple(indexed_documents),
            unchanged_documents=tuple(unchanged_documents),
            removed_documents=tuple(removed_documents),
            indexed_chunks=indexed_chunks,
        )

    @staticmethod
    def _is_unchanged(
        document: Document,
        existing: IndexedDocument | None,
    ) -> bool:
        content_hash = document.metadata.get("content_hash")

        return (
            existing is not None
            and existing.document_id == document.id
            and content_hash is not None
            and existing.content_hash == content_hash
        )

    @staticmethod
    def _belongs_to_root(
        document: IndexedDocument,
        root: Path,
    ) -> bool:
        if document.ingestion_root is not None:
            return Path(document.ingestion_root).expanduser().resolve() == root

        source = Path(document.source).expanduser()

        if not source.is_absolute():
            return False

        return source.resolve().is_relative_to(root)
