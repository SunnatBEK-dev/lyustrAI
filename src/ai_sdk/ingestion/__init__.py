from ai_sdk.ingestion.base import BaseDocumentLoader
from ai_sdk.ingestion.ingestor import DocumentIngestor
from ai_sdk.ingestion.pdf import PDFDocumentLoader
from ai_sdk.ingestion.sync import (
    DirectorySynchronizer,
    DirectorySyncResult,
)
from ai_sdk.ingestion.text import TextDocumentLoader


def create_default_ingestor() -> DocumentIngestor:
    return DocumentIngestor(
        [
            TextDocumentLoader(),
            PDFDocumentLoader(),
        ]
    )


__all__ = [
    "BaseDocumentLoader",
    "DocumentIngestor",
    "DirectorySyncResult",
    "DirectorySynchronizer",
    "PDFDocumentLoader",
    "TextDocumentLoader",
    "create_default_ingestor",
]
