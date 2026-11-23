"""One run of the hypothesis engine on a findings export, as the `hypotheses` command and the web app
both make it: the evidence for each finding worth testing, then the LLM's hypothesis for each, every
request checked by one `PrivateTextGuard` before it leaves (stopping the run when it refuses one).

It also keeps, for every passage a hypothesis or a candidate's risk cites, the passage's text and
its paper's record (`Source`), so a reader can be shown the quote in its context and a link to the
paper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hypothesis_engine.evidence import EvidenceBundle, Finding, gather_evidence, is_intervention
from hypothesis_engine.hypotheses import Failure, Hypothesis, generate_hypotheses
from hypothesis_engine.library import Library
from hypothesis_engine.llm import LLM, RequestHook, Settings, connect
from hypothesis_engine.privacy import PrivateTextGuard, Quote
from hypothesis_engine.search import Embedder, Index

Connect = Callable[..., LLM]  # `llm.connect`: (settings, on_request=hook) -> LLM
Progress = Callable[[int, int], None]  # (findings done, findings to do)


@dataclass(frozen=True)
class Source:
    """A cited passage, and the record of its paper (None for what the library does not hold)."""

    passage_id: str
    paper_id: str
    private: bool
    title: str
    text: str
    doi: str | None = None
    pmcid: str | None = None
    journal: str | None = None
    year: int | None = None


@dataclass(frozen=True)
class Generated:
    """What one run made: a result per bundle, in order, and what it drew on."""

    findings: int  # finding rows read from the export
    bundles: list[EvidenceBundle]
    results: list[Hypothesis | Failure]  # results[i] is bundles[i]'s
    quoted: list[Quote]  # what of the private papers was sent
    sources: dict[str, Source]  # passage id -> the passage, for every passage cited
    library_version: int
    library_built_at: str
    search_model: str


def run_hypotheses(
    findings: Sequence[Finding],
    settings: Settings,
    library_path: Path,
    index_path: Path,
    embedder: Embedder,
    *,
    k: int = 5,
    connect_llm: Connect = connect,
    progress: Progress | None = None,
) -> Generated:
    """Hypotheses for `findings`. Raises `PrivateTextLeak` before sending a request the guard refuses,
    and `LibraryError` or `SearchIndexError` when the library or its index cannot be used."""
    quoted: list[Quote] = []
    interventions = [f.marker_name for f in findings if is_intervention(f)]
    with (
        Index.open(index_path, library_path, embedder) as index,
        Library.open(library_path) as library,
    ):
        guard = PrivateTextGuard.from_library(library)
        hook: RequestHook = lambda request: quoted.extend(guard.check(request.text()))  # noqa: E731
        llm = connect_llm(settings, on_request=hook)
        bundles = gather_evidence(findings, index, k)
        results = generate_hypotheses(
            bundles, llm, library, index, interventions=interventions, progress=progress
        )
        sources = _sources(results, index, library)
        version, built_at = library.version, library.built_at
    return Generated(len(findings), bundles, results, quoted, sources, version, built_at, index.model)


def cited(result: Hypothesis | Failure) -> list[str]:
    """The passage ids a result cites: its claims, entries and its candidates' risks."""
    if not isinstance(result, Hypothesis):
        return []
    ids = [c.passage_id for c in result.mechanism]
    ids += [c.passage_id for c in (*result.supporting, *result.contradicting)]
    ids += [r.passage_id for c in result.candidates for r in c.risks]
    return ids


def _sources(results: Sequence[Hypothesis | Failure], index: Index, library: Library) -> dict[str, Source]:
    """Every passage the results cite, read back from the index, with its paper's record."""
    sources: dict[str, Source] = {}
    for passage_id in (p for r in results for p in cited(r)):
        if passage_id in sources:
            continue
        passage = index.passage(passage_id)
        if passage is None:
            continue
        paper = None if passage.private else library.paper(passage.paper_id)
        sources[passage_id] = Source(
            passage.id,
            passage.paper_id,
            passage.private,
            passage.title,
            passage.text,
            *((paper.doi, paper.pmcid, paper.journal, paper.year) if paper else ()),
        )
    return sources
