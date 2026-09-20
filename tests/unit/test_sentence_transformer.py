import pytest

from ai_sdk.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingClient,
)


class FakeArray:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def encode(self, sentences, *, convert_to_numpy):
        self.calls.append(
            {
                "sentences": sentences,
                "convert_to_numpy": convert_to_numpy,
            }
        )

        if self.error:
            raise self.error

        return self.result


def test_embed_converts_provider_batch_to_plain_float_vectors():
    model = FakeModel(result=FakeArray([[1, 2.5], [3, 4]]))
    client = SentenceTransformerEmbeddingClient(model=model)

    vectors = client.embed(("first", "second"))

    assert vectors == [[1.0, 2.5], [3.0, 4.0]]
    assert model.calls == [
        {
            "sentences": ["first", "second"],
            "convert_to_numpy": True,
        }
    ]


def test_embed_one_uses_the_shared_batch_adapter():
    model = FakeModel(result=[[0.1, 0.2]])
    client = SentenceTransformerEmbeddingClient(model=model)

    vector = client.embed_one("hello")

    assert vector == [0.1, 0.2]
    assert model.calls[0]["sentences"] == ["hello"]


def test_embed_empty_input_does_not_call_model():
    model = FakeModel(result=[])
    client = SentenceTransformerEmbeddingClient(model=model)

    assert client.embed([]) == []
    assert model.calls == []


def test_embed_propagates_provider_error():
    model = FakeModel(error=RuntimeError("model failed"))
    client = SentenceTransformerEmbeddingClient(model=model)

    with pytest.raises(RuntimeError, match="model failed"):
        client.embed(["hello"])
