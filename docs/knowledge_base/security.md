# Local security boundaries

LyustrAI is a local, single-user portfolio application.
It binds to `127.0.0.1` by default and does not claim SaaS isolation. There is
no authentication, organization boundary, billing, or public multi-tenant data
model in version 0.1.0.

API credentials belong in `.env`, which is ignored by Git. `.env.example`
contains variable names and safe defaults but no key values. Readiness reports
expose only whether a key and model are configured. Built-in traces redact
content- and credential-related attribute names.

Uploaded files are limited to PDF, Markdown, and TXT and to 10 MB per file.
Filenames containing paths are rejected, names are normalized, and writes stay
inside `data/uploads`. Uploaded documents, conversations, vector indexes,
embedding caches, and memories are excluded from Git. The web application
returns source filenames rather than absolute local filesystem paths.

Same-origin browser requests are protected with a restrictive Content Security
Policy, frame denial, referrer suppression, and MIME sniffing protection. The
application does not execute document contents. PDF text extraction is treated
as untrusted data and scanned PDFs are not sent to an OCR service.

For a future hosted deployment, authentication, per-user storage, malware
scanning, rate limiting, durable audit logs, database transactions, and abuse
controls would be mandatory. They are explicitly outside this local release.
