import pytest

from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document


def test_short_document_becomes_one_linked_chunk():
    document = Document.create(
        "Short text",
        {"source": "guide.txt"},
    )

    chunks = TextChunker(
        chunk_size=20,
        overlap=5,
    ).split(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == document.id
    assert chunks[0].content == "Short text"
    assert chunks[0].index == 0
    assert chunks[0].metadata == {"source": "guide.txt"}


def test_split_uses_fixed_character_overlap():
    document = Document.create("ABCDEFGHIJ")

    chunks = TextChunker(
        chunk_size=4,
        overlap=1,
    ).split(document)

    assert [chunk.content for chunk in chunks] == [
        "ABCD",
        "DEFG",
        "GHIJ",
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_split_produces_stable_chunk_identities():
    document = Document(
        id="doc_stable",
        content="ABCDEFGHIJ",
    )
    chunker = TextChunker(
        chunk_size=4,
        overlap=1,
    )

    first = chunker.split(document)
    second = chunker.split(document)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.id.startswith("chunk_") for chunk in first)


def test_pdf_pages_are_chunked_independently_with_page_metadata():
    document = Document(
        id="doc_pdf",
        content="First page\f\fThird page",
        metadata={"source": "guide.pdf", "format": "pdf"},
    )

    chunks = TextChunker(chunk_size=20, overlap=5).split(document)

    assert [chunk.content for chunk in chunks] == [
        "First page",
        "Third page",
    ]
    assert [chunk.metadata["page"] for chunk in chunks] == ["1", "3"]
    assert chunks[0].id != chunks[1].id


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "size"),
        (4, -1, "negative"),
        (4, 4, "smaller"),
    ],
)
def test_chunker_rejects_invalid_window_settings(
    chunk_size,
    overlap,
    message,
):
    with pytest.raises(ValueError, match=message):
        TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )
