"""A small real library for the relevance test: the nine PubMed records of fixtures/build and
the three open-access full texts of fixtures/fulltext, all recorded 2026-11-22."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fulltext_fixture import DOWNLOAD_BATCH, OPEN_ACCESS, make_library
from test_build import Recorded as RecordedPubMed
from test_build import _build
from test_fulltext import Recorded as RecordedFullText

from hypothesis_engine.fulltext import fetch_full_texts
from hypothesis_engine.library import Library, record_lookup, record_texts, seal_texts, write_build
from hypothesis_engine.pubmed import Paper

AT = "2026-11-22T12:00:00+00:00"


def make_relevance_library(path: Path) -> None:
    with tempfile.TemporaryDirectory() as folder:
        records, texts = Path(folder) / "records.sqlite", Path(folder) / "texts.sqlite"
        _build(records, RecordedPubMed().pubmed)
        make_library(texts)
        downloads = RecordedFullText()
        fetch_full_texts(texts, downloads.europepmc, downloads.pubmed, download_batch=DOWNLOAD_BATCH)
        with Library.open(records) as library:
            papers = {p.pmid: p for p in library.papers()}
        with Library.open(texts) as library:
            full = {pmid: library.full_text(pmid) for pmid in OPEN_ACCESS}
    for pmid, text in full.items():
        assert text is not None
        if pmid not in papers:  # the full-text fixture's records are placeholders: title from the text
            papers[pmid] = Paper(pmid, text.text.split("\n", 1)[0], None, None, text.pmcid, None, None)
    found_by = {pmid: [("PSSD", rank)] for rank, pmid in enumerate(sorted(papers, key=int), start=1)}
    write_build(path, list(papers.values()), found_by, "sha", AT)
    copies = {pmid: (t.pmcid, t.licence) for pmid, t in full.items() if t is not None}
    record_lookup(path, list(copies), copies, AT)
    record_texts(path, [(pmid, t.text, t.licence) for pmid, t in full.items() if t is not None], AT)
    seal_texts(path, AT)
