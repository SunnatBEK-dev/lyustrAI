from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ai_sdk.agents import CapabilityRouter
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.embeddings.sentence_transformer import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbeddingClient,
)
from ai_sdk.evaluation import (
    DEFAULT_ROUTE_EVAL_CASES,
    RetrievalComparator,
    RetrievalComparisonReport,
    RetrievalEvalCase,
    RouteEvaluationReport,
    RouteEvaluationRunner,
)
from ai_sdk.ingestion import create_default_ingestor
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.in_memory import InMemoryVectorStore
from ai_sdk.retrieval.retriever import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "docs" / "knowledge_base"
DEFAULT_DATASET = PROJECT_ROOT / "evals" / "knowledge_retrieval.json"


@dataclass(frozen=True)
class LabeledQuery:
    id: str
    query: str
    expected_sources: tuple[str, ...]


def load_dataset(path: Path) -> tuple[LabeledQuery, ...]:
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Knowledge evaluation dataset is invalid.") from error
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Knowledge evaluation dataset must be a non-empty list.")

    cases: list[LabeledQuery] = []
    seen_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Knowledge evaluation case must be an object.")
        case_id = raw_case.get("id")
        query = raw_case.get("query")
        expected_sources = raw_case.get("expected_sources")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in seen_ids
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(expected_sources, list)
            or not expected_sources
            or any(
                not isinstance(source, str) or not source.strip()
                for source in expected_sources
            )
        ):
            raise ValueError("Knowledge evaluation case fields are invalid.")
        seen_ids.add(case_id)
        cases.append(
            LabeledQuery(
                id=case_id,
                query=query,
                expected_sources=tuple(expected_sources),
            )
        )
    return tuple(cases)


def evaluate(
    corpus: Path,
    dataset: Path,
    *,
    k: int = 3,
    embedding_client: BaseEmbeddingClient | None = None,
) -> tuple[str, bool]:
    documents = create_default_ingestor().ingest(corpus)
    document_ids = {
        Path(document.metadata["source"]).name: document.id for document in documents
    }
    labeled_queries = load_dataset(dataset)
    unknown_sources = sorted(
        {
            source
            for case in labeled_queries
            for source in case.expected_sources
            if source not in document_ids
        }
    )
    if unknown_sources:
        raise ValueError(
            "Dataset references unknown sources: " + ", ".join(unknown_sources)
        )

    resolved_embedding_client = embedding_client or SentenceTransformerEmbeddingClient()
    semantic = SemanticRetriever(
        embedding_client=resolved_embedding_client,
        vector_store=InMemoryVectorStore(),
    )
    hybrid = HybridRetriever(
        embedding_client=resolved_embedding_client,
        vector_store=InMemoryVectorStore(),
    )
    chunker = TextChunker(chunk_size=700, overlap=100)
    for document in documents:
        chunks = chunker.split(document)
        semantic.index_document(document.id, chunks)
        hybrid.index_document(document.id, chunks)

    cases = tuple(
        RetrievalEvalCase(
            query=case.query,
            expected_document_ids=frozenset(
                document_ids[source] for source in case.expected_sources
            ),
        )
        for case in labeled_queries
    )
    comparison = RetrievalComparator(
        baseline=semantic,
        candidate=hybrid,
        k=k,
    ).evaluate(cases)
    route_report = RouteEvaluationRunner(
        CapabilityRouter(),
        minimum_accuracy=1.0,
    ).evaluate(DEFAULT_ROUTE_EVAL_CASES)

    candidate_misses = [
        labeled_queries[index].id
        for index, result in enumerate(comparison.candidate.results)
        if not result.hit
    ]
    baseline_misses = [
        labeled_queries[index].id
        for index, result in enumerate(comparison.baseline.results)
        if not result.hit
    ]
    candidate_rank_errors = [
        labeled_queries[index].id
        for index, result in enumerate(comparison.candidate.results)
        if result.hit and result.reciprocal_rank < 1.0
    ]
    baseline_rank_errors = [
        labeled_queries[index].id
        for index, result in enumerate(comparison.baseline.results)
        if result.hit and result.reciprocal_rank < 1.0
    ]
    passed = (
        comparison.candidate.hit_rate >= 0.90
        and comparison.candidate.mean_reciprocal_rank >= 0.80
        and not comparison.candidate_regressed
        and route_report.passed
    )
    report = _markdown_report(
        comparison=comparison,
        route_report=route_report,
        case_count=len(cases),
        k=k,
        baseline_misses=baseline_misses,
        candidate_misses=candidate_misses,
        baseline_rank_errors=baseline_rank_errors,
        candidate_rank_errors=candidate_rank_errors,
        passed=passed,
    )
    return report, passed


def _markdown_report(
    *,
    comparison: RetrievalComparisonReport,
    route_report: RouteEvaluationReport,
    case_count: int,
    k: int,
    baseline_misses: list[str],
    candidate_misses: list[str],
    baseline_rank_errors: list[str],
    candidate_rank_errors: list[str],
    passed: bool,
) -> str:
    baseline = comparison.baseline
    candidate = comparison.candidate
    return f"""# Evaluation report

This report is generated by `python app/evaluate_knowledge.py`. Retrieval uses
`{DEFAULT_MODEL_NAME}` and the committed English knowledge corpus. Provider
APIs are not called.

## Retrieval results

| Metric | Semantic baseline | Hybrid candidate | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@{k} | {baseline.hit_rate:.3f} | {candidate.hit_rate:.3f} | {comparison.hit_rate_delta:+.3f} |
| Recall@{k} | {baseline.mean_recall:.3f} | {candidate.mean_recall:.3f} | {comparison.mean_recall_delta:+.3f} |
| MRR | {baseline.mean_reciprocal_rank:.3f} | {candidate.mean_reciprocal_rank:.3f} | {comparison.mean_reciprocal_rank_delta:+.3f} |

- Dataset cases: {case_count}
- Semantic misses: {_list_or_none(baseline_misses)}
- Hybrid misses: {_list_or_none(candidate_misses)}
- Hybrid regression detected: {comparison.candidate_regressed}

## Observed failure categories

- Top-3 retrieval misses: semantic={_list_or_none(baseline_misses)}; hybrid={_list_or_none(candidate_misses)}
- Relevant source ranked below first: semantic={_list_or_none(baseline_rank_errors)}; hybrid={_list_or_none(candidate_rank_errors)}
- Routing misclassifications: {_list_or_none(list(route_report.failed_case_ids))}

## Routing results

- Expected routing decisions matched: {route_report.correct_cases}/{route_report.total_cases}
- Estimated provider requests: {route_report.estimated_model_requests}
- Always-FULL baseline: {route_report.full_route_baseline_requests}
- Estimated requests saved: {route_report.estimated_request_savings}
- Mean local routing latency: {route_report.mean_routing_latency_ms:.3f} ms

## Gate

Status: **{"PASS" if passed else "FAIL"}**

The gate requires hybrid Hit Rate@{k} >= 0.90, MRR >= 0.80, no retrieval
metric regression, and all {route_report.total_cases} expected internal routing
decisions to match.

## Limitations

- The corpus documents this project, so results do not generalize to unrelated domains.
- Retrieval metrics do not measure final generated-answer correctness.
- Request savings are workflow estimates, not token-cost measurements.
- Routing latency excludes provider and embedding latency.
"""


def _list_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the portfolio knowledge corpus."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=3)
    arguments = parser.parse_args()
    report, passed = evaluate(
        arguments.corpus,
        arguments.dataset,
        k=arguments.k,
    )
    print(report)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report, encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
