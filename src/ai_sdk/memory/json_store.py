import json
import re
from json import JSONDecodeError
from pathlib import Path
from tempfile import NamedTemporaryFile

from ai_sdk.memory.base import BaseMemoryStore
from ai_sdk.memory.model import (
    LongTermMemory,
    MemorySearchResult,
)
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import bm25_search


class JSONMemoryStore(BaseMemoryStore):
    """Persist long-term memories in an atomic local JSON file."""

    FORMAT_VERSION = 1
    COMMON_QUERY_TERMS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "be",
            "bu",
            "bilan",
            "da",
            "do",
            "does",
            "ham",
            "how",
            "is",
            "it",
            "kerak",
            "men",
            "menga",
            "my",
            "nima",
            "qanday",
            "qaysi",
            "should",
            "the",
            "this",
            "uchun",
            "va",
            "what",
            "which",
        }
    )

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        self._memories: dict[str, LongTermMemory] = {}
        self._load()

    def add(self, memory: LongTermMemory) -> None:
        previous_memories = self._memories.copy()
        self._memories[memory.id] = memory

        try:
            self._save()
        except Exception:
            self._memories = previous_memories
            raise

    def list_memories(self) -> list[LongTermMemory]:
        return list(self._memories.values())

    def delete(self, memory_id: str) -> bool:
        if not memory_id.strip():
            raise ValueError("Long-term memory ID cannot be empty.")

        if memory_id not in self._memories:
            return False

        previous_memories = self._memories.copy()
        del self._memories[memory_id]

        try:
            self._save()
        except Exception:
            self._memories = previous_memories
            raise

        return True

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[MemorySearchResult]:
        if not query.strip():
            raise ValueError("Memory search query cannot be empty.")

        normalized_query = self._normalize_query(query)

        if not normalized_query:
            return []

        chunks = [
            Chunk(
                id=memory.id,
                document_id=memory.id,
                content=memory.content,
                index=0,
            )
            for memory in self._memories.values()
        ]

        return [
            MemorySearchResult(
                memory=self._memories[result.chunk.id],
                score=result.score,
            )
            for result in bm25_search(
                query=normalized_query,
                candidates=chunks,
                k=k,
            )
        ]

    @classmethod
    def _normalize_query(cls, query: str) -> str:
        return " ".join(
            term
            for term in re.findall(
                r"\w+",
                query.casefold(),
            )
            if term not in cls.COMMON_QUERY_TERMS
        )

    def clear(self) -> None:
        if not self._memories:
            return

        previous_memories = self._memories.copy()
        self._memories.clear()

        try:
            self._save()
        except Exception:
            self._memories = previous_memories
            raise

    def count(self) -> int:
        return len(self._memories)

    def _load(self) -> None:
        if not self.file_path.exists():
            return

        try:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        except (JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Memory store file contains invalid JSON.") from error

        try:
            self._restore(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Memory store file has an invalid format.") from error

    def _restore(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Memory store payload must be an object.")

        if payload.get("version") != self.FORMAT_VERSION:
            raise ValueError("Memory store version is not supported.")

        records = payload.get("memories")

        if not isinstance(records, list):
            raise ValueError("Memory store records must be a list.")

        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Memory store record must be an object.")

            memory = LongTermMemory(
                id=record["id"],
                content=record["content"],
            )

            if memory.id in self._memories:
                raise ValueError("Long-term memory IDs must be unique.")

            self._memories[memory.id] = memory

    def _save(self) -> None:
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        payload = {
            "version": self.FORMAT_VERSION,
            "memories": [
                {
                    "id": memory.id,
                    "content": memory.content,
                }
                for memory in self._memories.values()
            ],
        }
        temporary_path: Path | None = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.file_path.parent,
                prefix=f".{self.file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                json.dump(
                    payload,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            temporary_path.replace(self.file_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
