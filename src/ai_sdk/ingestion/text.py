from hashlib import sha256
from pathlib import Path

from ai_sdk.ingestion.base import BaseDocumentLoader
from ai_sdk.retrieval.document import Document


class TextDocumentLoader(BaseDocumentLoader):
    """Load UTF-8 text formats with a stable path-based identity."""

    SUPPORTED_EXTENSIONS = frozenset(
        {
            ".txt",
            ".md",
            ".markdown",
            ".rst",
        }
    )

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> Document:
        expanded_path = path.expanduser()

        if not expanded_path.exists():
            raise FileNotFoundError(f"Document file does not exist: {expanded_path}")

        if not expanded_path.is_file():
            raise ValueError(f"Document path is not a file: {expanded_path}")

        if not self.supports(expanded_path):
            raise ValueError(
                f"Unsupported document format: {expanded_path.suffix or '<none>'}"
            )

        resolved_path = expanded_path.resolve()

        try:
            content = resolved_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Document must be a UTF-8 text file.") from error

        path_digest = sha256(str(resolved_path).encode("utf-8")).hexdigest()[:12]
        content_hash = sha256(content.encode("utf-8")).hexdigest()

        return Document(
            id=f"doc_{path_digest}",
            content=content,
            metadata={
                "source": str(resolved_path),
                "format": (resolved_path.suffix.lower().lstrip(".")),
                "content_hash": content_hash,
            },
        )
