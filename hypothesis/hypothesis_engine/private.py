"""The researcher's private papers: PDFs in a local folder (default `hypothesis/data/private/`).

Their text is extracted on this machine (pypdf) and stored in the library marked as private.
They are never looked up or downloaded anywhere, and `privacy.PrivateTextGuard` stops their
text leaving the machine beyond short quoted passages.

A paper's id is `private-` and the first 16 hex digits of the SHA-256 of its extracted text,
so renaming the file keeps the id and changing its words gives a new one. A PDF with no text
layer (a scan) or one pypdf cannot read is skipped with the reason; OCR is not done.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

ID_PREFIX = "private-"


@dataclass(frozen=True)
class PrivatePaper:
    id: str
    file_name: str
    title: str
    text: str


@dataclass(frozen=True)
class PrivateFolder:
    papers: list[PrivatePaper] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file name, reason)


def _tidy(page: str) -> str:
    lines = (" ".join(line.split()) for line in page.splitlines())
    return "\n".join(line for line in lines if line)


def _read(path: Path) -> tuple[str, str | None]:
    """(text, title from the PDF's metadata); raises ValueError when there is no readable text."""
    try:
        reader = PdfReader(path)
        pages = [_tidy(page.extract_text() or "") for page in reader.pages]
        title = reader.metadata.title if reader.metadata else None
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"not a readable PDF ({type(error).__name__})") from error
    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise ValueError("no text in it (a scan?); add a PDF with a text layer")
    return text, (" ".join(title.split()) or None) if isinstance(title, str) else None


def read_private_folder(folder: Path) -> PrivateFolder:
    """Every PDF directly in `folder`, by file name; a missing folder holds none."""
    result = PrivateFolder()
    if not folder.is_dir():
        return result
    seen: set[str] = set()
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # a broken file is reported as skipped instead
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        try:
            text, title = _read(path)
        except ValueError as error:
            result.skipped.append((path.name, str(error)))
            continue
        paper_id = ID_PREFIX + hashlib.sha256(text.encode()).hexdigest()[:16]
        if paper_id in seen:  # the same paper under two names
            continue
        seen.add(paper_id)
        result.papers.append(PrivatePaper(paper_id, path.name, title or text.split("\n", 1)[0], text))
    return result
