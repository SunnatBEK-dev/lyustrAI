from hashlib import sha256

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ai_sdk.ingestion import (
    DocumentIngestor,
    PDFDocumentLoader,
    TextDocumentLoader,
    create_default_ingestor,
)


def write_text_pdf(path, pages):
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)

    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): font_reference,
                    }
                ),
            }
        )
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)

    with path.open("wb") as output:
        writer.write(output)


def test_text_loader_supports_document_formats(tmp_path):
    file_path = tmp_path / "Guide.MD"
    file_path.write_text(
        "# Python guide",
        encoding="utf-8",
    )
    loader = TextDocumentLoader()

    document = loader.load(file_path)

    assert loader.supports(file_path) is True
    assert document.content == "# Python guide"
    assert document.metadata == {
        "source": str(file_path.resolve()),
        "format": "md",
        "content_hash": sha256(b"# Python guide").hexdigest(),
    }


def test_ingestor_loads_supported_directory_files_in_order(
    tmp_path,
):
    directory = tmp_path / "knowledge"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "b.md").write_text(
        "Markdown",
        encoding="utf-8",
    )
    (directory / "a.txt").write_text(
        "Text",
        encoding="utf-8",
    )
    (nested / "c.rst").write_text(
        "RST",
        encoding="utf-8",
    )
    (directory / "ignored.docx").write_bytes(b"not supported")

    documents = create_default_ingestor().ingest(directory)

    assert [document.metadata["format"] for document in documents] == [
        "txt",
        "md",
        "rst",
    ]
    assert [document.content for document in documents] == ["Text", "Markdown", "RST"]
    assert {document.metadata["ingestion_root"] for document in documents} == {
        str(directory.resolve())
    }


def test_ingestor_can_disable_recursive_directory_scan(
    tmp_path,
):
    directory = tmp_path / "knowledge"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "top.txt").write_text(
        "Top",
        encoding="utf-8",
    )
    (nested / "nested.txt").write_text(
        "Nested",
        encoding="utf-8",
    )

    documents = create_default_ingestor().ingest(
        directory,
        recursive=False,
    )

    assert [document.content for document in documents] == ["Top"]


def test_ingestor_rejects_unsupported_direct_file(tmp_path):
    file_path = tmp_path / "guide.docx"
    file_path.write_bytes(b"not supported")
    ingestor = DocumentIngestor([TextDocumentLoader()])

    with pytest.raises(ValueError, match="Unsupported"):
        ingestor.ingest(file_path)


def test_pdf_loader_preserves_pages_and_metadata(tmp_path):
    file_path = tmp_path / "architecture.pdf"
    write_text_pdf(file_path, ["Routing design", "Cancellation policy"])

    document = PDFDocumentLoader().load(file_path)

    assert document.content.split("\f") == [
        "Routing design",
        "Cancellation policy",
    ]
    assert document.metadata["source"] == str(file_path.resolve())
    assert document.metadata["format"] == "pdf"
    assert document.metadata["page_count"] == "2"
    assert (
        document.metadata["content_hash"] == sha256(file_path.read_bytes()).hexdigest()
    )


def test_pdf_loader_rejects_corrupt_and_textless_documents(tmp_path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-corrupt")
    with pytest.raises(ValueError, match="could not be read"):
        PDFDocumentLoader().load(corrupt_path)

    blank_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with blank_path.open("wb") as output:
        writer.write(output)
    with pytest.raises(ValueError, match="OCR is not supported"):
        PDFDocumentLoader().load(blank_path)


def test_pdf_loader_rejects_encrypted_documents(tmp_path):
    file_path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("private")
    with file_path.open("wb") as output:
        writer.write(output)

    with pytest.raises(ValueError, match="Encrypted"):
        PDFDocumentLoader().load(file_path)


def test_ingestor_can_allow_empty_directory(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()

    documents = create_default_ingestor().ingest(
        directory,
        allow_empty=True,
    )

    assert documents == []
