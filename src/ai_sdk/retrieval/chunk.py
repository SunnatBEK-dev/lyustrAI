from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Chunk:
    """An ordered piece of a document prepared for retrieval."""

    id: str
    document_id: str
    content: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Chunk ID cannot be empty.")

        if not self.document_id:
            raise ValueError("Chunk document ID cannot be empty.")

        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty.")

        if self.index < 0:
            raise ValueError("Chunk index cannot be negative.")

        self.metadata = dict(self.metadata)

    @classmethod
    def create(
        cls,
        document_id: str,
        content: str,
        index: int,
        metadata: Mapping[str, str] | None = None,
    ) -> "Chunk":
        return cls(
            id=f"chunk_{uuid4().hex[:8]}",
            document_id=document_id,
            content=content,
            index=index,
            metadata=dict(metadata or {}),
        )
