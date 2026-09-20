from dataclasses import dataclass

from ai_sdk.retrieval.search import SearchResult


@dataclass(frozen=True)
class Citation:
    """A local source reference for one retrieved chunk."""

    position: int
    document_id: str
    chunk_id: str
    source: str
    score: float
    page: int | None = None

    def __post_init__(self) -> None:
        if self.position <= 0:
            raise ValueError("Citation position must be greater than zero.")
        if self.page is not None and self.page <= 0:
            raise ValueError("Citation page must be greater than zero.")

    @classmethod
    def from_search_result(
        cls,
        position: int,
        result: SearchResult,
    ) -> "Citation":
        source = result.chunk.metadata.get("source")

        if not source or not source.strip():
            source = result.chunk.document_id

        page_value = result.chunk.metadata.get("page")
        page = int(page_value) if page_value is not None else None

        return cls(
            position=position,
            document_id=result.chunk.document_id,
            chunk_id=result.chunk.id,
            source=source,
            score=result.score,
            page=page,
        )


@dataclass(frozen=True)
class RAGResponse:
    """Generated content paired with its retrieved citations."""

    content: str
    citations: tuple[Citation, ...]
