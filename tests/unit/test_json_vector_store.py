import json

import pytest

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.json_store import JSONVectorStore


def make_chunk(
    chunk_id: str,
    content: str,
    index: int = 0,
    document_id: str = "doc_json_store",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        content=content,
        index=index,
        metadata={"source": "qo‘llanma.txt"},
    )


def test_store_round_trip_preserves_chunks_and_searches(tmp_path):
    file_path = tmp_path / "nested" / "vectors.json"
    store = JSONVectorStore(file_path)
    python_chunk = make_chunk(
        "chunk_python",
        "Python functions",
    )
    cooking_chunk = make_chunk(
        "chunk_cooking",
        "Cooking recipes",
        1,
    )

    store.add_many(
        [
            (python_chunk, [1.0, 0.0]),
            (cooking_chunk, [0.0, 1.0]),
        ]
    )
    restarted = JSONVectorStore(file_path)
    results = restarted.search([1.0, 0.0], k=1)

    assert restarted.count() == 2
    assert restarted.document_ids() == ["doc_json_store"]
    assert results[0].chunk.id == "chunk_python"
    assert results[0].chunk.metadata == {"source": "qo‘llanma.txt"}
    assert results[0].score == pytest.approx(1.0)
    assert (
        restarted.lexical_search(
            "functions",
            k=1,
        )[0].chunk.id
        == "chunk_python"
    )
    with pytest.raises(ValueError, match="dimension"):
        restarted.search([1.0], k=1)

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1


def test_store_replacement_delete_and_clear_survive_restart(
    tmp_path,
):
    file_path = tmp_path / "vectors.json"
    store = JSONVectorStore(file_path)
    store.add(
        make_chunk("chunk_same", "Original"),
        [1.0, 0.0],
    )
    store.add(
        make_chunk("chunk_same", "Replacement"),
        [0.0, 1.0],
    )

    restarted = JSONVectorStore(file_path)

    assert restarted.count() == 1
    assert (
        restarted.search(
            [0.0, 1.0],
            k=1,
        )[0].chunk.content
        == "Replacement"
    )
    assert restarted.delete("chunk_same") is True
    assert restarted.delete("chunk_same") is False

    empty = JSONVectorStore(file_path)
    empty.add(
        make_chunk("chunk_three", "Three dimensions"),
        [1.0, 0.0, 0.0],
    )
    empty.clear()

    assert JSONVectorStore(file_path).count() == 0


def test_add_many_rolls_back_on_invalid_dimension(tmp_path):
    file_path = tmp_path / "vectors.json"
    original = make_chunk("chunk_original", "Original")
    store = JSONVectorStore(file_path)
    store.add(original, [1.0, 0.0])

    with pytest.raises(ValueError, match="dimension"):
        store.add_many(
            [
                (
                    make_chunk("chunk_valid", "Valid", 1),
                    [0.0, 1.0],
                ),
                (
                    make_chunk("chunk_invalid", "Invalid", 2),
                    [1.0, 0.0, 0.0],
                ),
            ]
        )

    assert store.count() == 1
    assert JSONVectorStore(file_path).count() == 1


def test_add_rolls_back_when_file_write_fails(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "vectors.json"
    original = make_chunk("chunk_original", "Original")
    store = JSONVectorStore(file_path)
    store.add(original, [1.0, 0.0])

    def fail_to_save():
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_save", fail_to_save)

    with pytest.raises(OSError, match="disk unavailable"):
        store.add(
            make_chunk("chunk_new", "New", 1),
            [0.0, 1.0],
        )

    assert store.count() == 1
    assert (
        store.search(
            [1.0, 0.0],
            k=1,
        )[0].chunk.id
        == "chunk_original"
    )
    assert JSONVectorStore(file_path).count() == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{invalid", "invalid JSON"),
        (json.dumps([]), "invalid format"),
        (
            json.dumps({"version": 99, "items": []}),
            "invalid format",
        ),
        (
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "chunk": {},
                            "vector": [1.0],
                        }
                    ],
                }
            ),
            "invalid format",
        ),
    ],
)
def test_store_rejects_corrupt_or_invalid_file(
    tmp_path,
    payload,
    message,
):
    file_path = tmp_path / "vectors.json"
    file_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        JSONVectorStore(file_path)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([], "empty"),
        ([float("inf")], "finite"),
        (["not-a-number"], "numbers"),
    ],
)
def test_store_rejects_invalid_vector(
    tmp_path,
    vector,
    message,
):
    store = JSONVectorStore(tmp_path / "vectors.json")

    with pytest.raises(ValueError, match=message):
        store.add(
            make_chunk("chunk_invalid", "Invalid"),
            vector,
        )

    assert store.count() == 0


def test_document_replacement_and_deletion_survive_restart(
    tmp_path,
):
    file_path = tmp_path / "vectors.json"
    store = JSONVectorStore(file_path)
    store.add_many(
        [
            (
                make_chunk(
                    "chunk_old_one",
                    "Old one",
                    document_id="doc_replace",
                ),
                [1.0, 0.0],
            ),
            (
                make_chunk(
                    "chunk_old_two",
                    "Old two",
                    index=1,
                    document_id="doc_replace",
                ),
                [0.8, 0.2],
            ),
            (
                make_chunk(
                    "chunk_other",
                    "Other",
                    document_id="doc_other",
                ),
                [0.0, 1.0],
            ),
        ]
    )
    replacement = make_chunk(
        "chunk_replacement",
        "Replacement",
        document_id="doc_replace",
    )

    store.replace_document(
        "doc_replace",
        [(replacement, [1.0, 0.0])],
    )
    restarted = JSONVectorStore(file_path)

    assert restarted.count() == 2
    assert restarted.document_ids() == [
        "doc_other",
        "doc_replace",
    ]
    assert {
        result.chunk.id
        for result in restarted.search(
            [1.0, 0.0],
            k=5,
        )
    } == {
        "chunk_replacement",
        "chunk_other",
    }
    assert restarted.delete_document("doc_replace") == 1
    assert restarted.delete_document("doc_replace") == 0
    final_store = JSONVectorStore(file_path)
    assert final_store.count() == 1
    assert final_store.document_ids() == ["doc_other"]
    assert (
        final_store.search(
            [0.0, 1.0],
            k=1,
        )[0].chunk.id
        == "chunk_other"
    )


def test_document_replacement_rolls_back_on_invalid_vector(
    tmp_path,
):
    file_path = tmp_path / "vectors.json"
    store = JSONVectorStore(file_path)
    original = make_chunk(
        "chunk_original",
        "Original",
        document_id="doc_replace",
    )
    other = make_chunk(
        "chunk_other",
        "Other",
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
                        "Invalid",
                        document_id="doc_replace",
                    ),
                    [1.0, 0.0, 0.0],
                )
            ],
        )

    restarted = JSONVectorStore(file_path)
    assert restarted.count() == 2
    assert (
        restarted.search(
            [1.0, 0.0],
            k=1,
        )[0].chunk.id
        == "chunk_original"
    )
