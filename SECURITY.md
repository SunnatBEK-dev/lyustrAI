# Security policy

## Supported version

Security fixes are applied to the latest release on the `main` branch.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Email
`sunnatbekmail@gmail.com` with a concise description, reproduction steps, and
the affected version. You can expect an acknowledgement within five business
days.

## Local data and credentials

The application is designed for local, single-user use. API keys must remain
in `.env` and are never required in Git history. Conversation files, uploaded
documents, vector indexes, and memories under `data/` are ignored by Git.
