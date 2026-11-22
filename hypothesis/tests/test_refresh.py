"""The monthly refresh: new papers are added, stored ones are not downloaded again, and the
private folder's PDFs join the library. The version goes up only when something changed."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pdf_fixture import write_pdf

from hypothesis_engine.build import refresh_library
from hypothesis_engine.library import Library, LibraryError, write_build
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.topics import Topic

WHEN = datetime(2026, 12, 22, 9, 0, tzinfo=UTC)
TOPICS = [Topic("PSSD", "pssd", 10), Topic("PFS", "pfs", 10)]
QUOTE = "Cerebrospinal fluid allopregnanolone was lower in all twelve men."


def _paper(pmid: str) -> Paper:
    return Paper(pmid, f"Paper {pmid}", f"Abstract of {pmid}.", None, None, None, 2020)


class FakePubMed:
    """Search results by query; records every PMID it is asked to fetch."""

    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results
        self.fetched: list[str] = []

    def search(self, query: str, max_papers: int) -> list[str]:
        return self.results[query][:max_papers]

    def fetch(self, pmids: list[str]) -> list[Paper]:
        self.fetched.extend(pmids)
        return [_paper(pmid) for pmid in pmids]


def _library(tmp_path: Path) -> Path:
    """Version 1: two PSSD papers."""
    path = tmp_path / "library.sqlite"
    found_by = {"100": [("PSSD", 1)], "200": [("PSSD", 2)]}
    write_build(path, [_paper("100"), _paper("200")], found_by, "sha", "2026-11-22T00:00:00+00:00")
    return path


def _refresh(path: Path, pubmed: FakePubMed, private: Path):
    return refresh_library(path, TOPICS, pubmed, private, now=lambda: WHEN)


def test_new_papers_are_added_and_stored_ones_are_not_downloaded_again(tmp_path):
    path = _library(tmp_path)
    pubmed = FakePubMed({"pssd": ["300", "100", "200"], "pfs": ["200", "400"]})
    report = _refresh(path, pubmed, tmp_path / "private")
    assert sorted(pubmed.fetched) == ["300", "400"]
    assert (report.version, report.new_papers) == (2, 2)
    with Library.open(path) as library:
        assert library.version == 2
        assert [p.pmid for p in library.papers()] == ["100", "200", "300", "400"]
        assert library.topics("200") == ["PSSD", "PFS"]
        assert library.paper("300", version=1) is None  # version 1 is as it was


def test_refreshing_twice_in_a_row_adds_nothing_the_second_time(tmp_path):
    path = _library(tmp_path)
    (tmp_path / "private").mkdir()
    write_pdf(tmp_path / "private" / "cohort.pdf", [[QUOTE]])
    results = {"pssd": ["300", "100"], "pfs": ["400"]}
    _refresh(path, FakePubMed(results), tmp_path / "private")
    again = FakePubMed(results)
    report = _refresh(path, again, tmp_path / "private")
    assert again.fetched == []
    assert (report.version, report.new_papers, report.private_added, report.private_removed) == (
        None,
        0,
        0,
        0,
    )
    with Library.open(path) as library:
        assert library.version == 2


def test_a_paper_no_longer_found_by_its_search_stays_in_the_library(tmp_path):
    path = _library(tmp_path)
    _refresh(path, FakePubMed({"pssd": ["300"], "pfs": []}), tmp_path / "private")
    with Library.open(path) as library:
        assert [p.pmid for p in library.papers()] == ["100", "200", "300"]


def test_a_pdf_dropped_in_the_folder_is_searchable_after_refresh(tmp_path):
    path = _library(tmp_path)
    folder = tmp_path / "private"
    folder.mkdir()
    write_pdf(folder / "cohort.pdf", [["Unpublished cohort", QUOTE]])
    report = _refresh(path, FakePubMed({"pssd": ["100"], "pfs": []}), folder)
    assert (report.version, report.private_added) == (2, 1)
    with Library.open(path) as library:
        (paper_id,) = library.find("allopregnanolone was LOWER in all  twelve men")
        assert paper_id.startswith("private-")
        paper = library.private_paper(paper_id)
        assert paper is not None and QUOTE in paper.text
        documents = {d.id: d for d in library.documents()}
        assert documents[paper_id].private and not documents["100"].private
        assert library.find("Abstract of 100.") == ["100"]


def test_a_pdf_removed_from_the_folder_leaves_the_next_version(tmp_path):
    path = _library(tmp_path)
    folder = tmp_path / "private"
    folder.mkdir()
    write_pdf(folder / "cohort.pdf", [[QUOTE]])
    pubmed = FakePubMed({"pssd": ["100"], "pfs": []})
    _refresh(path, pubmed, folder)
    (folder / "cohort.pdf").unlink()
    report = _refresh(path, pubmed, folder)
    assert (report.version, report.private_removed) == (3, 1)
    with Library.open(path) as library:
        assert library.find(QUOTE) == []
        assert len(library.find(QUOTE, version=2)) == 1  # what version 2 could cite stays readable


def test_skipped_pdfs_are_reported(tmp_path):
    path = _library(tmp_path)
    folder = tmp_path / "private"
    folder.mkdir()
    write_pdf(folder / "scan.pdf", [[]])
    report = _refresh(path, FakePubMed({"pssd": [], "pfs": []}), folder)
    assert [name for name, _reason in report.skipped] == ["scan.pdf"]
    assert report.version is None


def test_refresh_needs_a_built_library(tmp_path):
    with pytest.raises(LibraryError, match="asclepius-library build"):
        _refresh(tmp_path / "library.sqlite", FakePubMed({"pssd": [], "pfs": []}), tmp_path)
