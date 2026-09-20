import json

import pytest

from ai_sdk.memory import (
    JSONMemoryStore,
    LongTermMemory,
)


def test_long_term_memory_creates_normalized_identity():
    memory = LongTermMemory.create("  Preferred language is Uzbek.  ")

    assert memory.id.startswith("mem_")
    assert memory.content == "Preferred language is Uzbek."


@pytest.mark.parametrize(
    ("memory_id", "content", "message"),
    [
        ("", "content", "ID"),
        ("mem_one", " ", "content"),
    ],
)
def test_long_term_memory_rejects_invalid_data(
    memory_id,
    content,
    message,
):
    with pytest.raises(ValueError, match=message):
        LongTermMemory(memory_id, content)


def test_json_memory_store_persists_searches_and_deletes(
    tmp_path,
):
    file_path = tmp_path / "memory" / "memories.json"
    language = LongTermMemory(
        "mem_language",
        "Preferred language is Uzbek",
    )
    food = LongTermMemory(
        "mem_food",
        "Favorite food is plov",
    )
    store = JSONMemoryStore(file_path)
    store.add(language)
    store.add(food)
    restarted = JSONMemoryStore(file_path)

    results = restarted.search(
        "Which language is preferred?",
        k=1,
    )

    assert restarted.count() == 2
    assert restarted.list_memories() == [language, food]
    assert results[0].memory == language
    assert results[0].score > 0.0
    assert restarted.search("unrelated", k=2) == []
    assert restarted.search("What is this?", k=2) == []
    assert restarted.delete("mem_language") is True
    assert restarted.delete("mem_language") is False
    assert JSONMemoryStore(file_path).list_memories() == [food]
    restarted.clear()
    assert JSONMemoryStore(file_path).count() == 0


def test_json_memory_store_rolls_back_failed_write(
    tmp_path,
    monkeypatch,
):
    store = JSONMemoryStore(tmp_path / "memories.json")
    existing = LongTermMemory("mem_existing", "Existing")
    store.add(existing)

    def fail_to_save():
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_save", fail_to_save)

    with pytest.raises(OSError, match="disk unavailable"):
        store.add(LongTermMemory("mem_new", "New"))

    assert store.list_memories() == [existing]


def test_json_memory_store_rolls_back_failed_delete_and_clear(
    tmp_path,
    monkeypatch,
):
    store = JSONMemoryStore(tmp_path / "memories.json")
    existing = LongTermMemory("mem_existing", "Existing")
    store.add(existing)

    def fail_to_save():
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_save", fail_to_save)

    with pytest.raises(OSError, match="disk unavailable"):
        store.delete(existing.id)

    assert store.list_memories() == [existing]

    with pytest.raises(OSError, match="disk unavailable"):
        store.clear()

    assert store.list_memories() == [existing]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{invalid", "invalid JSON"),
        (json.dumps([]), "invalid format"),
        (
            json.dumps({"version": 99, "memories": []}),
            "invalid format",
        ),
        (
            json.dumps(
                {
                    "version": 1,
                    "memories": [{"id": "", "content": "x"}],
                }
            ),
            "invalid format",
        ),
        (
            json.dumps({"version": 1, "memories": {}}),
            "invalid format",
        ),
        (
            json.dumps({"version": 1, "memories": ["invalid"]}),
            "invalid format",
        ),
        (
            json.dumps(
                {
                    "version": 1,
                    "memories": [
                        {"id": "mem_one", "content": "One"},
                        {"id": "mem_one", "content": "Duplicate"},
                    ],
                }
            ),
            "invalid format",
        ),
    ],
)
def test_json_memory_store_rejects_invalid_file(
    tmp_path,
    payload,
    message,
):
    file_path = tmp_path / "memories.json"
    file_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        JSONMemoryStore(file_path)


def test_json_memory_store_rejects_blank_delete_id(tmp_path):
    store = JSONMemoryStore(tmp_path / "memories.json")

    with pytest.raises(ValueError, match="ID"):
        store.delete(" ")

    with pytest.raises(ValueError, match="query"):
        store.search(" ")

    store.clear()
    assert store.count() == 0
