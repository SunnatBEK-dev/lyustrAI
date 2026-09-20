import pytest

from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.in_memory import InMemoryVectorStore


class MappingEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return [self.vectors[text] for text in texts]


def make_retriever():
    return HybridRetriever(
        embedding_client=MappingEmbeddingClient(
            {
                "Semantic candidate": [1.0, 0.0],
                "Exact error ZX-81": [0.0, 1.0],
                "ZX-81 help": [1.0, 0.0],
                "semantic question": [1.0, 0.0],
            }
        ),
        vector_store=InMemoryVectorStore(),
    )


def test_hybrid_retriever_promotes_exact_lexical_match():
    retriever = make_retriever()
    semantic = Chunk(
        id="chunk_semantic",
        document_id="doc_hybrid",
        content="Semantic candidate",
        index=0,
    )
    exact = Chunk(
        id="chunk_exact",
        document_id="doc_hybrid",
        content="Exact error ZX-81",
        index=1,
    )
    retriever.index([semantic, exact])

    results = retriever.retrieve("ZX-81 help", k=2)

    assert [result.chunk for result in results] == [
        exact,
        semantic,
    ]


def test_hybrid_retriever_falls_back_to_semantic_results():
    retriever = make_retriever()
    semantic = Chunk(
        id="chunk_semantic",
        document_id="doc_hybrid",
        content="Semantic candidate",
        index=0,
    )
    retriever.index([semantic])

    results = retriever.retrieve(
        "semantic question",
        k=1,
    )

    assert results[0].chunk is semantic


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"semantic_weight": -0.1}, "weight"),
        ({"candidate_multiplier": 0}, "multiplier"),
        ({"rank_constant": -1}, "constant"),
    ],
)
def test_hybrid_retriever_rejects_invalid_configuration(
    options,
    message,
):
    with pytest.raises(ValueError, match=message):
        HybridRetriever(
            embedding_client=MappingEmbeddingClient({}),
            vector_store=InMemoryVectorStore(),
            **options,
        )


def test_hybrid_retriever_rejects_invalid_query():
    retriever = make_retriever()

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(" ")

    with pytest.raises(ValueError, match="greater than zero"):
        retriever.retrieve("query", k=0)
