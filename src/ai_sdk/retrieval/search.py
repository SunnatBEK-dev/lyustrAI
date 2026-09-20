import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, log, sqrt

from ai_sdk.embeddings.base import EmbeddingVector
from ai_sdk.retrieval.chunk import Chunk

EmbeddedChunk = tuple[Chunk, EmbeddingVector]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Measure vector direction similarity without external math libraries."""
    if not left or not right:
        raise ValueError("Embedding vectors cannot be empty.")

    if len(left) != len(right):
        raise ValueError("Embedding vectors must have equal dimensions.")

    dot_product = fsum(
        left_value * right_value
        for left_value, right_value in zip(
            left,
            right,
            strict=True,
        )
    )
    left_norm = sqrt(fsum(value * value for value in left))
    right_norm = sqrt(fsum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    score = dot_product / (left_norm * right_norm)

    return max(-1.0, min(1.0, score))


def top_k_search(
    query_vector: Sequence[float],
    candidates: Sequence[EmbeddedChunk],
    k: int = 5,
) -> list[SearchResult]:
    """Return the most similar chunks in descending score order."""
    if k <= 0:
        raise ValueError("Top-k value must be greater than zero.")

    results = [
        SearchResult(
            chunk=chunk,
            score=cosine_similarity(
                query_vector,
                vector,
            ),
        )
        for chunk, vector in candidates
    ]
    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return results[:k]


def bm25_search(
    query: str,
    candidates: Sequence[Chunk],
    k: int = 5,
) -> list[SearchResult]:
    """Rank chunks by lexical relevance using dependency-free BM25."""
    if not query.strip():
        raise ValueError("Lexical search query cannot be empty.")

    if k <= 0:
        raise ValueError("Top-k value must be greater than zero.")

    candidate_list = list(candidates)

    if not candidate_list:
        return []

    query_terms = _tokenize(query)

    if not query_terms:
        return []

    tokenized_candidates = [_tokenize(chunk.content) for chunk in candidate_list]
    document_frequencies = Counter(
        term for terms in tokenized_candidates for term in set(terms)
    )
    document_count = len(candidate_list)
    average_length = sum(len(terms) for terms in tokenized_candidates) / document_count
    k1 = 1.5
    length_normalization = 0.75
    results = []

    for chunk, terms in zip(
        candidate_list,
        tokenized_candidates,
        strict=True,
    ):
        term_frequencies = Counter(terms)
        score = fsum(
            _bm25_term_score(
                term_frequency=term_frequencies[term],
                document_frequency=document_frequencies[term],
                document_length=len(terms),
                average_length=average_length,
                document_count=document_count,
                k1=k1,
                length_normalization=length_normalization,
            )
            for term in set(query_terms)
            if term_frequencies[term] > 0
        )

        if score > 0.0:
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return results[:k]


def fuse_ranked_results(
    semantic_results: Sequence[SearchResult],
    lexical_results: Sequence[SearchResult],
    *,
    semantic_weight: float = 0.7,
    k: int = 5,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Fuse semantic and lexical ranks into normalized RRF scores."""
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("Semantic weight must be between zero and one.")

    if k <= 0:
        raise ValueError("Top-k value must be greater than zero.")

    if rank_constant < 0:
        raise ValueError("Rank constant cannot be negative.")

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for results, weight in (
        (semantic_results, semantic_weight),
        (lexical_results, 1.0 - semantic_weight),
    ):
        if weight == 0.0:
            continue

        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            chunks.setdefault(chunk_id, result.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (
                rank_constant + rank
            )

    fused_results = [
        SearchResult(
            chunk=chunks[chunk_id],
            score=min(
                1.0,
                score * (rank_constant + 1),
            ),
        )
        for chunk_id, score in scores.items()
    ]
    fused_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return fused_results[:k]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _bm25_term_score(
    *,
    term_frequency: int,
    document_frequency: int,
    document_length: int,
    average_length: float,
    document_count: int,
    k1: float,
    length_normalization: float,
) -> float:
    inverse_document_frequency = log(
        1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    normalized_length = (
        document_length / average_length if average_length > 0.0 else 0.0
    )
    denominator = term_frequency + k1 * (
        1.0 - length_normalization + length_normalization * normalized_length
    )

    return inverse_document_frequency * term_frequency * (k1 + 1.0) / denominator
