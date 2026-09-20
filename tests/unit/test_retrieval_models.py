import pytest

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.document import Document


def test_document_create_generates_unique_identity():
    first = Document.create("First document")
    second = Document.create("Second document")

    assert first.id.startswith("doc_")
    assert first.id != second.id
    assert first.content == "First document"


def test_document_copies_metadata_from_caller():
    metadata = {"source": "guide.txt"}

    document = Document.create("Content", metadata)
    metadata["source"] = "changed.txt"

    assert document.metadata == {"source": "guide.txt"}


def test_document_rejects_blank_content():
    with pytest.raises(ValueError, match="content"):
        Document.create("   ")


def test_chunk_create_links_to_document_and_preserves_order():
    document = Document.create(
        "Full content",
        {"source": "guide.txt"},
    )

    chunk = Chunk.create(
        document_id=document.id,
        content="First piece",
        index=0,
        metadata=document.metadata,
    )

    assert chunk.id.startswith("chunk_")
    assert chunk.document_id == document.id
    assert chunk.content == "First piece"
    assert chunk.index == 0
    assert chunk.metadata == {"source": "guide.txt"}


@pytest.mark.parametrize(
    ("content", "index", "message"),
    [
        ("   ", 0, "content"),
        ("Valid", -1, "index"),
    ],
)
def test_chunk_rejects_invalid_retrieval_data(
    content,
    index,
    message,
):
    with pytest.raises(ValueError, match=message):
        Chunk.create(
            document_id="doc_valid",
            content=content,
            index=index,
        )
