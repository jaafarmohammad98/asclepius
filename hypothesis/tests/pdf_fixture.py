"""Small PDFs written by hand for the private-folder tests: one page of text per paragraph."""

from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[str]], title: str | None = None) -> Path:
    """A PDF with one page per item of `pages`, each a list of lines; no lines: a page with no text."""
    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(len(pages))]
    info = f"<< /Title ({_escape(title)}) >>" if title else "<< >>"
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{n} 0 R" for n in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        stream = (
            "BT /F1 11 Tf 14 TL 72 720 Td " + " ".join(f"({_escape(line)}) Tj T*" for line in lines) + " ET"
        )
        if not lines:
            stream = "0 0 1 rg 72 72 200 200 re f"  # a picture-only page, like a scan
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        data = stream.encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
    objects.append(info.encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {len(objects)} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))
    return path
