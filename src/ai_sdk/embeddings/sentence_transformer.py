from collections.abc import Sequence
from typing import Any, Protocol

from ai_sdk.embeddings.base import (
    BaseEmbeddingClient,
    EmbeddingBatch,
)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
    ) -> Any:
        """Return one vector for each sentence."""


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    """SentenceTransformer adapter behind the embedding contract."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: SentenceTransformerModel | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def embed(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        input_texts = list(texts)

        if not input_texts:
            return []

        encoded = self._get_model().encode(
            input_texts,
            convert_to_numpy=True,
        )

        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()

        vectors = [[float(value) for value in vector] for vector in encoded]

        if len(vectors) != len(input_texts):
            raise RuntimeError(
                "Embedding provider returned a different "
                "number of vectors than input texts."
            )

        return vectors

    def _get_model(self) -> SentenceTransformerModel:
        if self._model is None:
            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )
            except ImportError as error:
                raise RuntimeError(
                    "SentenceTransformer support is not installed. "
                    "Install the 'embeddings' project extra."
                ) from error

            self._model = SentenceTransformer(self.model_name)

        return self._model
