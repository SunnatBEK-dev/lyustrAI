from collections.abc import Sequence

import pytest

from ai_sdk.embeddings.base import (
    BaseEmbeddingClient,
    EmbeddingBatch,
)


class FakeEmbeddingClient(BaseEmbeddingClient):
    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def embed(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        self.received_texts = list(texts)
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class BrokenEmbeddingClient(BaseEmbeddingClient):
    def embed(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        return []


def test_base_embedding_client_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbeddingClient()


def test_embed_contract_returns_one_vector_per_text():
    client = FakeEmbeddingClient()

    vectors = client.embed(["one", "three"])

    assert vectors == [[0.0, 3.0], [1.0, 5.0]]
    assert client.received_texts == ["one", "three"]


def test_embed_one_delegates_to_batch_contract():
    client = FakeEmbeddingClient()

    vector = client.embed_one("hello")

    assert vector == [0.0, 5.0]
    assert client.received_texts == ["hello"]


def test_embed_one_rejects_invalid_provider_cardinality():
    client = BrokenEmbeddingClient()

    with pytest.raises(RuntimeError, match="exactly one vector"):
        client.embed_one("hello")
