from ai_sdk.embeddings.cache import EmbeddingCache


class VectorWithToList:
    def tolist(self):
        return [0.25, 0.75]


def test_cache_crud_behavior(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.json")

    assert cache.has("msg_1") is False
    assert cache.get("msg_1") is None

    cache.set("msg_1", [1.0, 2.0])
    assert cache.has("msg_1") is True
    assert cache.get("msg_1") == [1.0, 2.0]

    assert cache.delete("msg_1") is True
    assert cache.delete("msg_1") is False


def test_set_converts_array_like_vector_to_json_compatible_list(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.json")

    cache.set("msg_1", VectorWithToList())

    assert cache.get("msg_1") == [0.25, 0.75]


def test_save_and_load_round_trip_uses_injected_path(tmp_path):
    file_path = tmp_path / "nested" / "embeddings.json"
    cache = EmbeddingCache(file_path)
    cache.set("msg_1", [0.1, 0.2])
    cache.save()

    restored = EmbeddingCache(file_path)
    restored.load()

    assert restored.get("msg_1") == [0.1, 0.2]


def test_load_missing_file_resets_cache(tmp_path):
    cache = EmbeddingCache(tmp_path / "missing.json")
    cache.set("stale", [1.0])

    cache.load()

    assert cache.cache == {}


def test_load_invalid_json_resets_cache(tmp_path):
    file_path = tmp_path / "embeddings.json"
    file_path.write_text("{invalid", encoding="utf-8")
    cache = EmbeddingCache(file_path)
    cache.set("stale", [1.0])

    cache.load()

    assert cache.cache == {}


def test_load_non_mapping_payload_resets_cache(tmp_path):
    file_path = tmp_path / "embeddings.json"
    file_path.write_text("[]", encoding="utf-8")
    cache = EmbeddingCache(file_path)
    cache.set("stale", [1.0])

    cache.load()

    assert cache.cache == {}


def test_clear_removes_all_cached_vectors(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.json")
    cache.set("msg_1", [1.0])

    cache.clear()

    assert cache.cache == {}
