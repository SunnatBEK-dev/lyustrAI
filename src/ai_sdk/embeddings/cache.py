import json
from pathlib import Path
from typing import Any

from ai_sdk.config import EMBEDDINGS_FILE


class EmbeddingCache:
    def __init__(
        self,
        file_path: Path = EMBEDDINGS_FILE,
    ) -> None:
        self.file_path = file_path
        self.cache: dict[str, list[float]] = {}

    def has(
        self,
        message_id: str,
    ) -> bool:
        return message_id in self.cache

    def get(
        self,
        message_id: str,
    ) -> list[float] | None:
        return self.cache.get(message_id)

    def set(
        self,
        message_id: str,
        vector: Any,
    ) -> None:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        self.cache[message_id] = [float(value) for value in vector]

    def delete(
        self,
        message_id: str,
    ) -> bool:
        if message_id not in self.cache:
            return False

        del self.cache[message_id]

        return True

    def clear(self) -> None:
        self.cache.clear()

    def save(self) -> None:
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.file_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.cache,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def load(self) -> None:
        if not self.file_path.exists():
            self.cache = {}
            return

        try:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if isinstance(data, dict):
                self.cache = data
            else:
                self.cache = {}

        except json.JSONDecodeError as e:
            print(f"Embedding cache parsing failed: {e}")
            self.cache = {}
