from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class LongTermMemory:
    """A durable user-provided fact or preference."""

    id: str
    content: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Long-term memory ID cannot be empty.")

        if not self.content.strip():
            raise ValueError("Long-term memory content cannot be empty.")

    @classmethod
    def create(cls, content: str) -> "LongTermMemory":
        return cls(
            id=f"mem_{uuid4().hex[:12]}",
            content=content.strip(),
        )


@dataclass(frozen=True)
class MemorySearchResult:
    memory: LongTermMemory
    score: float
