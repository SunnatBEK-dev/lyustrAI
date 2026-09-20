import pytest

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.in_memory import (
    InMemoryVectorStore,
)


def make_chunk(
    chunk_id: str,
    index: int = 0,
    document_id: str = "doc_store",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        content=f"Content {index}",
        index=index,
    )


def test_store_adds_copies_and_searches_vectors():
    store = InMemoryVectorStore()
    first = make_chunk("chunk_first", 0)
    second = make_chunk("chunk_second", 1)
    first_vector = [1.0, 0.0]

    store.add(first, first_vector)
    store.add(second, [0.0, 1.0])
    first_vector[0] = 0.0

    results = store.search([1.0, 0.0], k=1)

    assert store.count() == 2
    assert results[0].chunk is first
    assert results[0].score == pytest.approx(1.0)


def test_store_replaces_existing_chunk_without_growing():
    store = InMemoryVectorStore()
    original = make_chunk("chunk_same")
    replacement = Chunk(
        id="chunk_same",
        document_id="doc_store",
        content="Replacement",
        index=0,
    )

    store.add(original, [1.0, 0.0])
    store.add(replacement, [0.0, 1.0])

    assert store.count() == 1
    assert (
        store.search(
            [0.0, 1.0],
            k=1,
        )[0].chunk
        is replacement
    )


def test_store_searches_indexed_chunks_lexically():
    store = InMemoryVectorStore()
    exact = Chunk(
        id="chunk_exact",
        document_id="doc_store",
        content="Deployment error ZX-81",
        index=0,
    )
    generic = Chunk(
        id="chunk_generic",
        document_id="doc_store",
        content="General deployment guide",
        index=1,
    )
    store.add_many(
        [
            (generic, [1.0, 0.0]),
            (exact, [0.0, 1.0]),
        ]
    )

    results = store.lexical_search("ZX-81", k=1)

    assert results[0].chunk is exact


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([], "empty"),
        ([1.0, 2.0, 3.0], "dimension"),
    ],
)
def test_store_rejects_invalid_vector(
    vector,
    message,
):
    store = InMemoryVectorStore()
    store.add(make_chunk("chunk_first"), [1.0, 0.0])

    with pytest.raises(ValueError, match=message):
        store.add(make_chunk("chunk_invalid", 1), vector)

    assert store.count() == 1


def test_store_delete_and_clear_reset_dimension():
    store = InMemoryVectorStore()
    first = make_chunk("chunk_first")

    store.add(first, [1.0, 0.0])
    assert store.delete(first.id) is True
    assert store.delete(first.id) is False

    store.add(make_chunk("chunk_new"), [1.0, 2.0, 3.0])
    store.clear()

    assert store.count() == 0
    assert store.search([1.0], k=1) == []


def test_store_replaces_and_deletes_document_chunks():
    store = InMemoryVectorStore()
    old_first = make_chunk(
        "chunk_old_first",
        document_id="doc_replace",
    )
    old_second = make_chunk(
        "chunk_old_second",
        index=1,
        document_id="doc_replace",
    )
    other = make_chunk(
        "chunk_other",
        document_id="doc_other",
    )
    replacement = make_chunk(
        "chunk_replacement",
        document_id="doc_replace",
    )
    store.add_many(
        [
            (old_first, [1.0, 0.0]),
            (old_second, [0.8, 0.2]),
            (other, [0.0, 1.0]),
        ]
    )

    store.replace_document(
        "doc_replace",
        [(replacement, [1.0, 0.0])],
    )

    result_ids = {
        result.chunk.id
        for result in store.search(
            [1.0, 0.0],
            k=5,
        )
    }
    assert result_ids == {
        "chunk_replacement",
        "chunk_other",
    }
    assert store.document_ids() == [
        "doc_other",
        "doc_replace",
    ]
    assert store.delete_document("doc_replace") == 1
    assert store.delete_document("doc_replace") == 0
    assert store.count() == 1
    assert store.document_ids() == ["doc_other"]


def test_store_rolls_back_invalid_document_replacement():
    store = InMemoryVectorStore()
    original = make_chunk(
        "chunk_original",
        document_id="doc_replace",
    )
    other = make_chunk(
        "chunk_other",
        document_id="doc_other",
    )
    store.add_many(
        [
            (original, [1.0, 0.0]),
            (other, [0.0, 1.0]),
        ]
    )

    with pytest.raises(ValueError, match="dimension"):
        store.replace_document(
            "doc_replace",
            [
                (
                    make_chunk(
                        "chunk_invalid",
                        document_id="doc_replace",
                    ),
                    [1.0, 0.0, 0.0],
                )
            ],
        )

    with pytest.raises(ValueError, match="belong"):
        store.replace_document(
            "doc_replace",
            [(other, [0.0, 1.0])],
        )

    assert store.count() == 2
    assert (
        store.search(
            [1.0, 0.0],
            k=1,
        )[0].chunk
        is original
    )
