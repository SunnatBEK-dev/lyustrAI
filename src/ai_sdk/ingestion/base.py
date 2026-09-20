from abc import ABC, abstractmethod
from pathlib import Path

from ai_sdk.retrieval.document import Document


class BaseDocumentLoader(ABC):
    """Provider-neutral contract for loading one local document."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> Document:
        raise NotImplementedError
