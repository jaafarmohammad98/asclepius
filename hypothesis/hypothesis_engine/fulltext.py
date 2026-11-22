"""Open-access full text for the library's papers: `asclepius-library fulltext` (and `build`).

Two steps, each stored as it goes so an interrupted run resumes where it stopped:

1. Ask Europe PMC which papers of the current version have a copy in the open-access
   subset, `lookup_batch` at a time, and record each copy's PMCID and licence.
2. Download those copies' JATS XML from PubMed Central (NCBI efetch, `download_batch` at a
   time, at NCBI's rate limit) and store their readable text.

A paper is looked up once and a copy downloaded once; a rerun only does what is left. A
download batch that fails for a moment (after the usual retries), and any copy missing from
an answer, is left for the next run; several batches failing in a row mean PMC is
unreachable, so the run stops rather than wait on each. A lookup that still fails stops the
run at once (the lookups before it are kept). A copy PMC sends without body text (some
publishers do not allow the XML) counts as downloaded, without readable text.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hypothesis_engine.europepmc import EuropePMC
from hypothesis_engine.jats import articles, jats_text, licence_in
from hypothesis_engine.library import Coverage, Library, record_lookup, record_texts, to_download, unchecked
from hypothesis_engine.pubmed import PMC_BATCH, PubMed, PubMedError, TransientError

FAILURES_IN_A_ROW = 3


@dataclass(frozen=True)
class FullTextReport:
    coverage: Coverage
    failed: int  # copies not downloaded this run because of a failure; the next run tries them again


def batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_full_texts(
    path: Path,
    europepmc: EuropePMC,
    pubmed: PubMed,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[str], None] = lambda _message: None,
    download_batch: int = PMC_BATCH,
    failures_in_a_row: int = FAILURES_IN_A_ROW,
) -> FullTextReport:
    """Look up and download what the library at `path` still lacks; return its coverage."""
    stamp = lambda: now().isoformat(timespec="seconds")  # noqa: E731

    pmids = unchecked(path)
    if pmids:
        progress(f"looking up {len(pmids)} papers in Europe PMC")
    for batch in batches(pmids, europepmc.lookup_batch):
        copies = europepmc.open_access(batch)
        record_lookup(path, batch, {c.pmid: (c.pmcid, c.licence) for c in copies}, stamp())

    pending = to_download(path)
    if pending:
        progress(f"downloading {len(pending)} open-access copies from PubMed Central")
    failed = in_a_row = done = 0
    for group in batches(pending, download_batch):
        try:
            body = pubmed.fetch_pmc([pmcid for _pmid, pmcid, _licence in group])
        except TransientError as error:
            failed += len(group)
            in_a_row += 1
            if in_a_row >= failures_in_a_row:
                raise TransientError(f"{in_a_row} downloads failed in a row, last: {error}") from error
            continue
        in_a_row = 0
        try:
            found = articles(body)
        except ET.ParseError as error:
            raise PubMedError(f"efetch: unreadable PMC XML for {group[0][1]}..{group[-1][1]}") from error
        texts = []
        for pmid, pmcid, licence in group:
            article = found.get(pmcid)
            if article is None:  # an answer cut short: try again next run
                failed += 1
                continue
            texts.append((pmid, jats_text(article), licence or licence_in(article)))
        record_texts(path, texts, stamp())
        done += len(group)
        if done % 100 < download_batch:
            progress(f"downloaded {done} of {len(pending)}")

    with Library.open(path) as library:
        return FullTextReport(library.coverage(), failed)


def describe(report: FullTextReport) -> str:
    """One line for the researcher: how many papers have full text."""
    c = report.coverage
    share = f"{100 * c.with_text / c.papers:.0f}%" if c.papers else "0%"
    line = f"full text for {c.with_text} of {c.papers} papers ({share}); {c.open_access} have an open-access copy"
    if c.unreadable:
        line += f", {c.unreadable} of them without readable text"
    if report.failed:
        line += f"; {report.failed} could not be downloaded this time"
    left = c.to_download + (c.papers - c.checked)
    if left:
        line += f"; {left} still to fetch, run `asclepius-library fulltext` to continue"
    return line
