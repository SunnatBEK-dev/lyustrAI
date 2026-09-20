import pytest

from ai_sdk.application.rag_response import (
    Citation,
    RAGResponse,
)
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import SearchResult


def make_result(source=None, page=None):
    metadata = {}

    if source is not None:
        metadata["source"] = source
    if page is not None:
        metadata["page"] = page

    return SearchResult(
        chunk=Chunk(
            id="chunk_citation",
            document_id="doc_citation",
            content="Citation context",
            index=0,
            metadata=metadata,
        ),
        score=0.875,
    )


def test_citation_maps_retrieval_source_and_identity():
    citation = Citation.from_search_result(
        1,
        make_result("guide.txt"),
    )

    assert citation == Citation(
        position=1,
        document_id="doc_citation",
        chunk_id="chunk_citation",
        source="guide.txt",
        score=0.875,
    )


def test_citation_falls_back_to_document_id():
    citation = Citation.from_search_result(
        2,
        make_result(),
    )

    assert citation.source == "doc_citation"
    assert RAGResponse(
        content="Answer",
        citations=(citation,),
    ).citations == (citation,)


def test_citation_rejects_non_positive_position():
    with pytest.raises(ValueError, match="greater than zero"):
        Citation.from_search_result(
            0,
            make_result(),
        )


def test_citation_preserves_pdf_page_number():
    citation = Citation.from_search_result(
        1,
        make_result("guide.pdf", "7"),
    )

    assert citation.page == 7

    with pytest.raises(ValueError, match="page"):
        Citation(
            position=1,
            document_id="doc",
            chunk_id="chunk",
            source="guide.pdf",
            score=1.0,
            page=0,
        )
