"""Reading the researcher's private PDFs from a local folder."""

from __future__ import annotations

from pdf_fixture import write_pdf

from hypothesis_engine.private import read_private_folder

LINES = [
    "Allopregnanolone levels in men after finasteride",
    "Unpublished cohort, 2026.",
    "Cerebrospinal fluid allopregnanolone was lower in all twelve men.",
]


def test_a_pdf_gives_its_title_and_text(tmp_path):
    write_pdf(tmp_path / "cohort.pdf", [LINES[:2], LINES[2:]], title="Allopregnanolone after finasteride")
    folder = read_private_folder(tmp_path)
    assert folder.skipped == []
    (paper,) = folder.papers
    assert paper.title == "Allopregnanolone after finasteride"
    assert paper.file_name == "cohort.pdf"
    assert "Cerebrospinal fluid allopregnanolone was lower in all twelve men." in paper.text
    assert "Unpublished cohort, 2026." in paper.text
    assert paper.id.startswith("private-")


def test_a_pdf_without_a_title_is_named_by_its_first_line(tmp_path):
    write_pdf(tmp_path / "cohort.pdf", [LINES])
    (paper,) = read_private_folder(tmp_path).papers
    assert paper.title == LINES[0]


def test_the_id_follows_the_content_not_the_file_name(tmp_path):
    write_pdf(tmp_path / "a.pdf", [LINES])
    first = read_private_folder(tmp_path).papers[0].id
    (tmp_path / "a.pdf").rename(tmp_path / "renamed.pdf")
    assert read_private_folder(tmp_path).papers[0].id == first
    write_pdf(tmp_path / "renamed.pdf", [[*LINES, "A corrected sentence."]])
    assert read_private_folder(tmp_path).papers[0].id != first


def test_a_scanned_pdf_and_a_broken_file_are_skipped_with_a_reason(tmp_path):
    write_pdf(tmp_path / "scan.pdf", [[], []])
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf at all")
    write_pdf(tmp_path / "good.pdf", [LINES])
    (tmp_path / "notes.txt").write_text("not a PDF, not read")
    folder = read_private_folder(tmp_path)
    assert [p.file_name for p in folder.papers] == ["good.pdf"]
    reasons = dict(folder.skipped)
    assert set(reasons) == {"broken.pdf", "scan.pdf"}
    assert "no text" in reasons["scan.pdf"]
    assert "not a readable PDF" in reasons["broken.pdf"]


def test_the_same_pdf_twice_is_one_paper(tmp_path):
    write_pdf(tmp_path / "a.pdf", [LINES])
    write_pdf(tmp_path / "copy.pdf", [LINES])
    assert [p.file_name for p in read_private_folder(tmp_path).papers] == ["a.pdf"]


def test_a_missing_folder_holds_no_papers(tmp_path):
    folder = read_private_folder(tmp_path / "private")
    assert (folder.papers, folder.skipped) == ([], [])
