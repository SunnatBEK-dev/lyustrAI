from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_DATA_PATHS = frozenset({"data/.gitkeep"})
_PRIVATE_FILENAMES = frozenset(
    {
        ".env",
        "credentials.json",
        "secrets.json",
    }
)
_PRIVATE_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_SECRET_PATTERNS = {
    "OpenAI or Anthropic token": re.compile(
        r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"
    ),
    "Google API token": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "GitHub token": re.compile(
        r"\b(?:gh(?:p|o|u|s|r)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


@dataclass(frozen=True)
class AuditFinding:
    category: str
    location: str


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def repository_files() -> tuple[Path, ...]:
    output = _git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return tuple(
        PROJECT_ROOT / relative_path
        for relative_path in output.split("\0")
        if relative_path
    )


def audit_repository() -> tuple[AuditFinding, ...]:
    findings: list[AuditFinding] = []
    for path in repository_files():
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        if (
            relative_path.startswith("data/")
            and relative_path not in _ALLOWED_DATA_PATHS
        ):
            findings.append(AuditFinding("tracked runtime data", relative_path))
        if (
            path.name in _PRIVATE_FILENAMES
            or path.suffix.casefold() in _PRIVATE_SUFFIXES
        ):
            findings.append(AuditFinding("private credential file", relative_path))
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in _SECRET_PATTERNS.items():
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                findings.append(AuditFinding(label, f"{relative_path}:{line}"))

    history = _git("log", "--all", "-p", "--format=")
    for label, pattern in _SECRET_PATTERNS.items():
        if pattern.search(history):
            findings.append(
                AuditFinding(f"{label} in Git history", "reachable commits")
            )
    return tuple(findings)


def main() -> None:
    findings = audit_repository()
    if findings:
        print("Repository audit: FAIL")
        for finding in findings:
            print(f"- {finding.category}: {finding.location}")
        raise SystemExit(1)

    print("Repository audit: PASS")
    print(f"- Files inspected: {len(repository_files())}")
    print("- Runtime data tracked: none")
    print("- Recognizable secrets in files or Git history: none")


if __name__ == "__main__":
    main()
