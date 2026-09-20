from abc import ABC, abstractmethod

from ai_sdk.memory.model import (
    LongTermMemory,
    MemorySearchResult,
)


class BaseMemoryStore(ABC):
    """Persistence and retrieval contract for long-term memory."""

    @abstractmethod
    def add(self, memory: LongTermMemory) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_memories(self) -> list[LongTermMemory]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[MemorySearchResult]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
