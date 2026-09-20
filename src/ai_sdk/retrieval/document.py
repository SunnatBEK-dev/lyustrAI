from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Document:
    """A complete source that can later be split for retrieval."""

    id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Document ID cannot be empty.")

        if not self.content.strip():
            raise ValueError("Document content cannot be empty.")

        self.metadata = dict(self.metadata)

    @classmethod
    def create(
        cls,
        content: str,
        metadata: Mapping[str, str] | None = None,
    ) -> "Document":
        return cls(
            id=f"doc_{uuid4().hex[:8]}",
            content=content,
            metadata=dict(metadata or {}),
        )
