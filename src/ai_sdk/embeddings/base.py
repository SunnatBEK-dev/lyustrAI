from abc import ABC, abstractmethod
from collections.abc import Sequence

EmbeddingVector = list[float]
EmbeddingBatch = list[EmbeddingVector]


class BaseEmbeddingClient(ABC):
    """Provider-neutral contract for generating embedding vectors."""

    @abstractmethod
    def embed(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatch:
        """Generate one embedding vector for each input text."""
        raise NotImplementedError

    def embed_one(
        self,
        text: str,
    ) -> EmbeddingVector:
        """Generate a single vector through the batch contract."""
        vectors = self.embed([text])

        if len(vectors) != 1:
            raise RuntimeError(
                "Embedding client must return exactly one vector for one input text."
            )

        return vectors[0]
