import pytest

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    SearchResult,
    bm25_search,
    cosine_similarity,
    fuse_ranked_results,
    top_k_search,
)


def make_chunk(chunk_id: str, index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_search",
        content=f"Content {index}",
        index=index,
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
    ],
)
def test_cosine_similarity_known_directions(
    left,
    right,
    expected,
):
    assert cosine_similarity(
        left,
        right,
    ) == pytest.approx(expected)


def test_cosine_similarity_treats_zero_vector_as_no_similarity():
    assert (
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 0.0],
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([], [], "empty"),
        ([1.0], [1.0, 2.0], "dimensions"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left,
    right,
    message,
):
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)


def test_top_k_search_returns_highest_scoring_chunks():
    first = make_chunk("chunk_first", 0)
    second = make_chunk("chunk_second", 1)
    third = make_chunk("chunk_third", 2)

    results = top_k_search(
        query_vector=[1.0, 0.0],
        candidates=[
            (third, [0.0, 1.0]),
            (second, [0.8, 0.2]),
            (first, [1.0, 0.0]),
        ],
        k=2,
    )

    assert [result.chunk for result in results] == [
        first,
        second,
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score > 0.0


def test_top_k_search_preserves_input_order_for_equal_scores():
    first = make_chunk("chunk_first", 0)
    second = make_chunk("chunk_second", 1)

    results = top_k_search(
        query_vector=[1.0, 0.0],
        candidates=[
            (first, [1.0, 0.0]),
            (second, [1.0, 0.0]),
        ],
        k=2,
    )

    assert [result.chunk for result in results] == [
        first,
        second,
    ]


def test_top_k_search_returns_empty_for_no_candidates():
    assert (
        top_k_search(
            query_vector=[1.0],
            candidates=[],
            k=3,
        )
        == []
    )


def test_top_k_search_rejects_non_positive_k():
    with pytest.raises(ValueError, match="greater than zero"):
        top_k_search(
            query_vector=[1.0],
            candidates=[],
            k=0,
        )


def test_bm25_search_finds_exact_lexical_terms():
    exact = Chunk(
        id="chunk_exact",
        document_id="doc_search",
        content="Error code ZX-81 means the token expired.",
        index=0,
    )
    generic = Chunk(
        id="chunk_generic",
        document_id="doc_search",
        content="General authentication troubleshooting.",
        index=1,
    )

    results = bm25_search(
        "What causes zx-81?",
        [generic, exact],
        k=2,
    )

    assert [result.chunk for result in results] == [exact]
    assert results[0].score > 0.0


def test_bm25_search_returns_empty_when_terms_do_not_match():
    assert (
        bm25_search(
            "unrelated",
            [make_chunk("chunk_one", 0)],
        )
        == []
    )


@pytest.mark.parametrize(
    ("query", "k", "message"),
    [
        (" ", 1, "query"),
        ("query", 0, "greater than zero"),
    ],
)
def test_bm25_search_rejects_invalid_input(
    query,
    k,
    message,
):
    with pytest.raises(ValueError, match=message):
        bm25_search(query, [], k=k)


def test_rank_fusion_combines_semantic_and_lexical_evidence():
    semantic = make_chunk("chunk_semantic", 0)
    exact = make_chunk("chunk_exact", 1)

    results = fuse_ranked_results(
        semantic_results=[
            SearchResult(semantic, 0.99),
            SearchResult(exact, 0.20),
        ],
        lexical_results=[
            SearchResult(exact, 4.0),
        ],
        semantic_weight=0.7,
        k=2,
    )

    assert [result.chunk for result in results] == [
        exact,
        semantic,
    ]
    assert all(0.0 < result.score <= 1.0 for result in results)


@pytest.mark.parametrize(
    ("semantic_weight", "k", "rank_constant", "message"),
    [
        (1.1, 1, 60, "weight"),
        (0.7, 0, 60, "greater than zero"),
        (0.7, 1, -1, "constant"),
    ],
)
def test_rank_fusion_rejects_invalid_configuration(
    semantic_weight,
    k,
    rank_constant,
    message,
):
    with pytest.raises(ValueError, match=message):
        fuse_ranked_results(
            [],
            [],
            semantic_weight=semantic_weight,
            k=k,
            rank_constant=rank_constant,
        )
