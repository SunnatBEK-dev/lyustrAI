import pytest

from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.evaluation import (
    EvalCase,
    EvaluationRunner,
    ExactMatchEvaluator,
)
from ai_sdk.evaluation.retrieval import (
    RetrievalComparator,
    RetrievalEvalCase,
    RetrievalEvaluator,
)
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.in_memory import (
    InMemoryVectorStore,
)
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)

pytestmark = pytest.mark.integration


class KeywordEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [
            [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0] for text in texts
        ]


def test_real_retriever_scores_offline_evaluation_dataset():
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    retriever.index(
        [
            Chunk(
                id="chunk_python",
                document_id="doc_guide",
                content="Python functions",
                index=0,
            ),
            Chunk(
                id="chunk_cooking",
                document_id="doc_guide",
                content="Cooking recipes",
                index=1,
            ),
        ]
    )
    cases = [
        RetrievalEvalCase(
            query="How do Python functions work?",
            expected_chunk_ids=frozenset({"chunk_python"}),
        ),
        RetrievalEvalCase(
            query="How should I start cooking?",
            expected_chunk_ids=frozenset({"chunk_cooking"}),
        ),
    ]

    report = RetrievalEvaluator(
        retriever=retriever,
        k=1,
    ).evaluate(cases)

    assert report.total_cases == 2
    assert report.hit_rate == pytest.approx(1.0)
    assert report.mean_recall == pytest.approx(1.0)
    assert report.mean_reciprocal_rank == pytest.approx(1.0)


def test_general_eval_harness_scores_a_real_sdk_component():
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    retriever.index(
        [
            Chunk(
                id="chunk_python",
                document_id="doc_guide",
                content="Python functions",
                index=0,
            ),
            Chunk(
                id="chunk_cooking",
                document_id="doc_guide",
                content="Cooking recipes",
                index=1,
            ),
        ]
    )

    def retrieve_first_chunk_id(query):
        return retriever.retrieve(query, k=1)[0].chunk.id

    report = EvaluationRunner(
        [ExactMatchEvaluator()],
        minimum_pass_rate=1.0,
    ).evaluate(
        [
            EvalCase(
                "python",
                "How do Python functions work?",
                "chunk_python",
            ),
            EvalCase(
                "cooking",
                "How should I start cooking?",
                "chunk_cooking",
            ),
        ],
        retrieve_first_chunk_id,
    )

    assert report.passed is True
    assert report.pass_rate == pytest.approx(1.0)
    assert report.mean_scores == {"exact_match": 1.0}


class ComparisonEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        vectors = []

        for text in texts:
            lowered = text.lower()

            if text.startswith("Exact error"):
                vectors.append([0.0, 0.0, 1.0, 0.0])
            elif "python" in lowered:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "cooking" in lowered:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])

        return vectors


def test_semantic_and_hybrid_retrievers_are_compared_on_same_cases():
    chunks = [
        Chunk(
            id="chunk_exact",
            document_id="doc_eval",
            content="Exact error ZX-81 means token expired",
            index=0,
        ),
        Chunk(
            id="chunk_python",
            document_id="doc_eval",
            content="Python functions",
            index=1,
        ),
        Chunk(
            id="chunk_cooking",
            document_id="doc_eval",
            content="Cooking recipes",
            index=2,
        ),
        Chunk(
            id="chunk_decoy",
            document_id="doc_eval",
            content="Generic troubleshooting",
            index=3,
        ),
    ]
    semantic = SemanticRetriever(
        embedding_client=ComparisonEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    hybrid = HybridRetriever(
        embedding_client=ComparisonEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    semantic.index(chunks)
    hybrid.index(chunks)
    cases = [
        RetrievalEvalCase(
            query="How do Python functions work?",
            expected_chunk_ids=frozenset({"chunk_python"}),
        ),
        RetrievalEvalCase(
            query="How should I start cooking?",
            expected_chunk_ids=frozenset({"chunk_cooking"}),
        ),
        RetrievalEvalCase(
            query="What does ZX-81 mean?",
            expected_chunk_ids=frozenset({"chunk_exact"}),
        ),
    ]

    comparison = RetrievalComparator(
        baseline=semantic,
        candidate=hybrid,
        k=1,
    ).evaluate(cases)

    assert comparison.baseline.hit_rate == pytest.approx(2 / 3)
    assert comparison.candidate.hit_rate == pytest.approx(1.0)
    assert comparison.hit_rate_delta == pytest.approx(1 / 3)
    assert comparison.mean_recall_delta == pytest.approx(1 / 3)
    assert comparison.mean_reciprocal_rank_delta == pytest.approx(1 / 3)
    assert comparison.candidate_improved is True
    assert comparison.candidate_regressed is False
