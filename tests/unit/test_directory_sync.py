import pytest

from ai_sdk.ingestion import (
    DirectorySynchronizer,
    create_default_ingestor,
)
from ai_sdk.retrieval.catalog import IndexedDocument
from ai_sdk.retrieval.chunk import Chunk


class RecordingIndex:
    def __init__(self):
        self.catalog = []
        self.indexed = []
        self.deleted = []
        self.failing_content = None

    def document_catalog(self):
        return list(self.catalog)

    def index_document(self, document):
        if document.content == self.failing_content:
            raise RuntimeError("index failed")

        self.indexed.append(document.id)
        self.catalog = [
            item for item in self.catalog if item.document_id != document.id
        ]
        self.catalog.append(
            IndexedDocument(
                document_id=document.id,
                source=document.metadata["source"],
                chunk_count=1,
                content_hash=document.metadata["content_hash"],
                ingestion_root=document.metadata["ingestion_root"],
            )
        )
        return [
            Chunk(
                id=f"chunk_{document.id}",
                document_id=document.id,
                content=document.content,
                index=0,
                metadata=document.metadata,
            )
        ]

    def delete_document(self, document_id):
        matches = [item for item in self.catalog if item.document_id == document_id]
        self.catalog = [
            item for item in self.catalog if item.document_id != document_id
        ]
        self.deleted.append(document_id)
        return sum(item.chunk_count for item in matches)


def test_directory_sync_skips_unchanged_documents(tmp_path):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    (directory / "a.txt").write_text(
        "Alpha",
        encoding="utf-8",
    )
    (directory / "b.md").write_text(
        "Beta",
        encoding="utf-8",
    )
    index = RecordingIndex()
    synchronizer = DirectorySynchronizer(
        create_default_ingestor(),
        index,
    )

    first = synchronizer.sync(directory)
    second = synchronizer.sync(directory)

    assert len(first.indexed_documents) == 2
    assert first.indexed_chunks == 2
    assert second.indexed_documents == ()
    assert len(second.unchanged_documents) == 2
    assert second.indexed_chunks == 0
    assert len(index.indexed) == 2


def test_directory_sync_indexes_changes_and_removes_stale_files(
    tmp_path,
):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    changed_path = directory / "changed.txt"
    removed_path = directory / "removed.txt"
    changed_path.write_text("Old", encoding="utf-8")
    removed_path.write_text("Remove", encoding="utf-8")
    index = RecordingIndex()
    synchronizer = DirectorySynchronizer(
        create_default_ingestor(),
        index,
    )
    synchronizer.sync(directory)
    removed_id = next(
        item.document_id
        for item in index.catalog
        if item.source == str(removed_path.resolve())
    )
    index.indexed.clear()
    changed_path.write_text("New", encoding="utf-8")
    removed_path.unlink()
    (directory / "new.md").write_text(
        "Added",
        encoding="utf-8",
    )

    result = synchronizer.sync(directory)

    assert len(result.indexed_documents) == 2
    assert result.unchanged_documents == ()
    assert result.removed_documents == (removed_id,)
    assert result.indexed_chunks == 2
    assert index.deleted == [removed_id]
    assert len(index.catalog) == 2


def test_directory_sync_does_not_remove_stale_files_when_indexing_fails(
    tmp_path,
):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    changed_path = directory / "changed.txt"
    stale_path = directory / "stale.txt"
    changed_path.write_text("Old", encoding="utf-8")
    stale_path.write_text("Stale", encoding="utf-8")
    index = RecordingIndex()
    synchronizer = DirectorySynchronizer(
        create_default_ingestor(),
        index,
    )
    synchronizer.sync(directory)
    stale_id = next(
        item.document_id
        for item in index.catalog
        if item.source == str(stale_path.resolve())
    )
    changed_path.write_text("Fail", encoding="utf-8")
    stale_path.unlink()
    index.failing_content = "Fail"

    with pytest.raises(RuntimeError, match="index failed"):
        synchronizer.sync(directory)

    assert stale_id in {item.document_id for item in index.catalog}
    assert index.deleted == []


def test_directory_sync_upgrades_legacy_catalog_metadata(tmp_path):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    file_path = directory / "legacy.txt"
    file_path.write_text("Legacy", encoding="utf-8")
    ingestor = create_default_ingestor()
    document = ingestor.ingest(file_path)[0]
    index = RecordingIndex()
    index.catalog = [
        IndexedDocument(
            document_id=document.id,
            source=str(file_path.resolve()),
            chunk_count=1,
        ),
        IndexedDocument(
            document_id="doc_other",
            source=str(tmp_path / "other" / "file.txt"),
            chunk_count=1,
        ),
    ]

    result = DirectorySynchronizer(
        ingestor,
        index,
    ).sync(directory)

    upgraded = next(item for item in index.catalog if item.document_id == document.id)
    assert result.indexed_documents == (document.id,)
    assert upgraded.content_hash is not None
    assert upgraded.ingestion_root == str(directory.resolve())
    assert "doc_other" in {item.document_id for item in index.catalog}
    assert index.deleted == []
