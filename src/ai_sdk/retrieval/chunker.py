from hashlib import sha256

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.document import Document


class TextChunker:
    """Split document text with deterministic character windows."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        document: Document,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        if document.metadata.get("format") == "pdf":
            for page, content in enumerate(
                document.content.split("\f"),
                start=1,
            ):
                self._split_content(
                    document,
                    content,
                    chunks,
                    page=page,
                )
            return chunks

        self._split_content(document, document.content, chunks)
        return chunks

    def _split_content(
        self,
        document: Document,
        content: str,
        chunks: list[Chunk],
        *,
        page: int | None = None,
    ) -> None:
        step = self.chunk_size - self.overlap
        start = 0

        while start < len(content):
            end = min(
                start + self.chunk_size,
                len(content),
            )
            chunk_content = content[start:end]

            if chunk_content.strip():
                metadata = dict(document.metadata)
                if page is not None:
                    metadata["page"] = str(page)
                chunks.append(
                    Chunk(
                        id=self._create_chunk_id(
                            document_id=document.id,
                            start=start,
                            end=end,
                            content=chunk_content,
                            page=page,
                        ),
                        document_id=document.id,
                        content=chunk_content,
                        index=len(chunks),
                        metadata=metadata,
                    )
                )

            if end == len(content):
                break

            start += step

    @staticmethod
    def _create_chunk_id(
        document_id: str,
        start: int,
        end: int,
        content: str,
        page: int | None = None,
    ) -> str:
        page_identity = "" if page is None else f"\0page={page}"
        identity = f"{document_id}{page_identity}\0{start}\0{end}\0{content}"
        digest = sha256(identity.encode("utf-8")).hexdigest()[:12]

        return f"chunk_{digest}"
