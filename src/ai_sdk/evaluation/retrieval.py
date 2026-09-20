from collections.abc import Sequence
from dataclasses import dataclass, field
from math import fsum
from typing import Protocol

from ai_sdk.retrieval.search import SearchResult


class _Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> Sequence[SearchResult]: ...


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A query and stable chunk or document relevance labels."""

    query: str
    expected_chunk_ids: frozenset[str] = field(default_factory=frozenset)
    expected_document_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Evaluation query cannot be empty.")

        expected_chunk_ids = frozenset(self.expected_chunk_ids)
        expected_document_ids = frozenset(self.expected_document_ids)

        if not expected_chunk_ids and not expected_document_ids:
            raise ValueError("Evaluation case must have at least one expected label.")

        if any(not chunk_id.strip() for chunk_id in expected_chunk_ids):
            raise ValueError("Expected chunk IDs cannot be empty.")
        if any(not document_id.strip() for document_id in expected_document_ids):
            raise ValueError("Expected document IDs cannot be empty.")

        object.__setattr__(
            self,
            "expected_chunk_ids",
            expected_chunk_ids,
        )
        object.__setattr__(
            self,
            "expected_document_ids",
            expected_document_ids,
        )


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Per-query retrieval outcome at the configured top-k."""

    case: RetrievalEvalCase
    retrieved_chunk_ids: tuple[str, ...]
    hit: bool
    recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalEvalReport:
    """Aggregate retrieval metrics for one evaluation dataset."""

    k: int
    results: tuple[RetrievalEvalResult, ...]
    hit_rate: float
    mean_recall: float
    mean_reciprocal_rank: float

    @property
    def total_cases(self) -> int:
        return len(self.results)


@dataclass(frozen=True)
class RetrievalComparisonReport:
    """Metric deltas between a baseline and candidate retriever."""

    baseline: RetrievalEvalReport
    candidate: RetrievalEvalReport

    def __post_init__(self) -> None:
        if self.baseline.k != self.candidate.k:
            raise ValueError("Compared retrieval reports must use the same top-k.")

        baseline_cases = tuple(result.case for result in self.baseline.results)
        candidate_cases = tuple(result.case for result in self.candidate.results)

        if baseline_cases != candidate_cases:
            raise ValueError("Compared retrieval reports must use the same dataset.")

    @property
    def hit_rate_delta(self) -> float:
        return self.candidate.hit_rate - self.baseline.hit_rate

    @property
    def mean_recall_delta(self) -> float:
        return self.candidate.mean_recall - self.baseline.mean_recall

    @property
    def mean_reciprocal_rank_delta(self) -> float:
        return self.candidate.mean_reciprocal_rank - self.baseline.mean_reciprocal_rank

    @property
    def candidate_improved(self) -> bool:
        deltas = self._deltas()
        return all(delta >= 0.0 for delta in deltas) and any(
            delta > 0.0 for delta in deltas
        )

    @property
    def candidate_regressed(self) -> bool:
        return any(delta < 0.0 for delta in self._deltas())

    def _deltas(self) -> tuple[float, float, float]:
        return (
            self.hit_rate_delta,
            self.mean_recall_delta,
            self.mean_reciprocal_rank_delta,
        )


class RetrievalEvaluator:
    """Measure Hit Rate, Recall, and MRR without an LLM judge."""

    def __init__(
        self,
        retriever: _Retriever,
        k: int = 5,
    ) -> None:
        if k <= 0:
            raise ValueError("Evaluation top-k must be greater than zero.")

        self.retriever = retriever
        self.k = k

    def evaluate(
        self,
        cases: Sequence[RetrievalEvalCase],
    ) -> RetrievalEvalReport:
        case_list = tuple(cases)

        if not case_list:
            raise ValueError("Evaluation dataset cannot be empty.")

        results = tuple(self._evaluate_case(case) for case in case_list)
        case_count = len(results)

        return RetrievalEvalReport(
            k=self.k,
            results=results,
            hit_rate=(fsum(result.hit for result in results) / case_count),
            mean_recall=(fsum(result.recall for result in results) / case_count),
            mean_reciprocal_rank=(
                fsum(result.reciprocal_rank for result in results) / case_count
            ),
        )

    def _evaluate_case(
        self,
        case: RetrievalEvalCase,
    ) -> RetrievalEvalResult:
        retrieved = tuple(self.retriever.retrieve(case.query, k=self.k))
        retrieved_chunk_ids = tuple(result.chunk.id for result in retrieved)
        relevant_chunk_ids = case.expected_chunk_ids.intersection(retrieved_chunk_ids)
        retrieved_document_ids = {result.chunk.document_id for result in retrieved}
        relevant_document_ids = case.expected_document_ids.intersection(
            retrieved_document_ids
        )
        reciprocal_rank = 0.0

        for rank, result in enumerate(retrieved, start=1):
            if (
                result.chunk.id in case.expected_chunk_ids
                or result.chunk.document_id in case.expected_document_ids
            ):
                reciprocal_rank = 1.0 / rank
                break

        relevant_count = len(relevant_chunk_ids) + len(relevant_document_ids)
        expected_count = len(case.expected_chunk_ids) + len(case.expected_document_ids)

        return RetrievalEvalResult(
            case=case,
            retrieved_chunk_ids=retrieved_chunk_ids,
            hit=relevant_count > 0,
            recall=relevant_count / expected_count,
            reciprocal_rank=reciprocal_rank,
        )


class RetrievalComparator:
    """Evaluate two retrievers on exactly the same labeled cases."""

    def __init__(
        self,
        baseline: _Retriever,
        candidate: _Retriever,
        k: int = 5,
    ) -> None:
        self.baseline_evaluator = RetrievalEvaluator(
            baseline,
            k=k,
        )
        self.candidate_evaluator = RetrievalEvaluator(
            candidate,
            k=k,
        )

    def evaluate(
        self,
        cases: Sequence[RetrievalEvalCase],
    ) -> RetrievalComparisonReport:
        case_list = tuple(cases)

        return RetrievalComparisonReport(
            baseline=self.baseline_evaluator.evaluate(case_list),
            candidate=self.candidate_evaluator.evaluate(case_list),
        )
