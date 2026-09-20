from hashlib import sha256
from pathlib import Path

from ai_sdk.ingestion.base import BaseDocumentLoader
from ai_sdk.retrieval.document import Document


class PDFDocumentLoader(BaseDocumentLoader):
    """Extract page-delimited text from a local PDF document."""

    SUPPORTED_EXTENSIONS = frozenset({".pdf"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> Document:
        resolved_path = self._validated_path(path)

        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                "PDF support requires the 'documents' optional dependency."
            ) from error

        try:
            reader = PdfReader(str(resolved_path), strict=False)
        except Exception as error:
            raise ValueError("PDF document could not be read.") from error

        if reader.is_encrypted:
            raise ValueError("Encrypted PDF documents are not supported.")

        pages: list[str] = []
        try:
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        except Exception as error:
            raise ValueError("PDF text extraction failed.") from error

        if not pages:
            raise ValueError("PDF document contains no pages.")

        if not any(page.strip() for page in pages):
            raise ValueError("PDF contains no extractable text; OCR is not supported.")

        content = "\f".join(pages)
        path_digest = sha256(str(resolved_path).encode("utf-8")).hexdigest()[:12]
        content_hash = sha256(resolved_path.read_bytes()).hexdigest()

        return Document(
            id=f"doc_{path_digest}",
            content=content,
            metadata={
                "source": str(resolved_path),
                "format": "pdf",
                "content_hash": content_hash,
                "page_count": str(len(pages)),
            },
        )

    def _validated_path(self, path: Path) -> Path:
        expanded_path = path.expanduser()
        if not expanded_path.exists():
            raise FileNotFoundError(f"Document file does not exist: {expanded_path}")
        if not expanded_path.is_file():
            raise ValueError("Document path is not a file.")
        if not self.supports(expanded_path):
            raise ValueError(
                f"Unsupported document format: {expanded_path.suffix or '<none>'}"
            )
        return expanded_path.resolve()
