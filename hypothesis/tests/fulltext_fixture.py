"""The small library the full-text tests start from, and where its recorded answers are kept:
Europe PMC's open-access lookups and PubMed Central's article XML."""

from __future__ import annotations

from pathlib import Path

from hypothesis_engine.library import write_build
from hypothesis_engine.pubmed import Paper

FOLDER = Path(__file__).parent / "fixtures" / "fulltext"
LOOKUP_BATCH = 2  # small, so the recording covers more than one lookup
DOWNLOAD_BATCH = 2  # and more than one download

# Read by hand from Europe PMC on 2026-11-22.
OPEN_ACCESS = {  # PMID -> (PMCID, licence)
    "32317131": ("PMC7253896", "cc by-nc-nd"),
    "39289881": ("PMC11450419", "cc by"),
    "41640943": ("PMC12864535", "cc by"),
}
FREE_TO_READ_ONLY = "42027367"  # in PMC, but not in the open-access subset
NO_COPY = "19439893"  # a 2009 letter, no copy anywhere
PMIDS = sorted([*OPEN_ACCESS, FREE_TO_READ_ONLY, NO_COPY], key=int)


def make_library(path: Path) -> None:
    papers = [Paper(pmid, f"Paper {pmid}", "An abstract.", None, None, None, 2020) for pmid in PMIDS]
    found_by = {pmid: [("PSSD", rank)] for rank, pmid in enumerate(PMIDS, start=1)}
    write_build(path, papers, found_by, "sha", "2026-11-22T00:00:00+00:00")
