# Retrieval and document grounding

The ingestion layer accepts UTF-8 Markdown and text documents plus PDFs with an
extractable text layer. PDF pages are separated before chunking, so chunks do
not cross page boundaries. Every PDF chunk carries its original page number.
Encrypted PDFs and image-only scanned PDFs are rejected with explicit errors;
OCR is outside the current release.

Documents are split into deterministic character windows with configurable
overlap. Chunk identifiers include the document identity, window offsets,
content, and PDF page number when applicable. Re-indexing an unchanged source
therefore produces stable identities, while changed content replaces the
document atomically.

The default retrieval path is hybrid. Semantic similarity comes from a
provider-neutral embedding client. BM25 supplies exact lexical evidence for
identifiers, error codes, names, and uncommon technical terms. Weighted
reciprocal-rank fusion combines both rankings, using semantic weight 0.7 by
default. The retriever expands the candidate pool before returning the final
top-k results.

The vector-store contract has in-memory and persistent JSON implementations.
The JSON adapter validates dimensions, preserves chunk metadata, writes changes
atomically, and supports document-level replacement and deletion. Directory
synchronization stores content hashes so unchanged files are not embedded
again and stale files can be removed from the index.

Retrieved chunks are inserted into the provider-neutral prompt as untrusted
context. Citations are constructed locally from the exact search results after
generation. Each citation includes a stable position, document ID, chunk ID,
source name, fused relevance score, and an optional PDF page. Absolute local
paths are reduced to filenames in the public web response.
