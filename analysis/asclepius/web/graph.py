"""The knowledge graph in the web app (milestone 6): built from the paper library in the background
when the app opens, so the page is not kept waiting (a few seconds on the full library), and never
stored (spec G6).

A run's strong and suggestive findings add their finding links (M6-04b): the workspace says a run
has started (`expect`), and once it finishes hands over its findings (`add`); they are linked in
(`hypothesis_engine.graph.with_findings`) and the terms re-scored in the background, the map saying
it is updating until then, and labelled with the run's dataset fingerprint while they are in. A new
load, run or clear drops them (`drop`), back to the library-only graph, kept in memory, not rebuilt.

The bottleneck map (M6-04) draws it: `map_view` is every term with its bottleneck score and raw
metrics (`hypothesis_engine.bottlenecks`), each co-mention once and each syndrome link; the page
re-scores the terms itself when a weight slider moves. Each score is multiplied by the term's
posterior mean in the evidence log (`evidence`, spec B4, M6-11), as are the scores `current` gives
the leaderboard, so logging a result re-ranks both. `edge` is what is behind one link: the
passages it was counted from, each with its paper and where to read it.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any

import networkx as nx
from hypothesis_engine.bottlenecks import Bottleneck, Weights, bottlenecks
from hypothesis_engine.evidence import Finding
from hypothesis_engine.evidence_log import weighted
from hypothesis_engine.graph import (
    CO_MENTION,
    FINDING,
    LITERATURE,
    SYNDROME,
    SYNDROMES,
    library_graph,
    with_findings,
)
from hypothesis_engine.library import Library, LibraryError
from hypothesis_engine.passages import split
from hypothesis_engine.pipeline import Source
from hypothesis_engine.vocabulary import Vocabulary, VocabularyError, load_vocabulary

from asclepius.web.evidence_log import EvidenceStore
from asclepius.web.hypotheses import BUILD_LIBRARY, Engine, links

BUILDING, READY, FAILED = "building", "ready", "failed"
MOST_PASSAGES = 50  # shown for one link; a link of hundreds of papers says how many more there are
UPDATING = "Updating with this run's findings…"
AWAITING = "This run's findings join the map when it finishes."
SHOWN_FINGERPRINT = 12  # characters of a dataset's fingerprint, as the exports' file names have it


class Run:
    """The run whose findings are in the graph, or are being added."""

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint
        self.running = True  # the run is still going; then its findings are being added
        self.updating = True  # until they are in
        self.finding_links = 0


class KnowledgeGraph:
    """The graph of the engine's library, once built; until then, or if it cannot be, why not."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.state = BUILDING
        self.message = "The knowledge graph is being built from the paper library."
        self.graph: nx.DiGraph | None = None
        self.scores: list[Bottleneck] = []  # with the default weights
        self.version: int | None = None  # the library version it was built from
        self.vocabulary: Vocabulary | None = None
        self.run: Run | None = None  # whose findings are in `graph`
        self.evidence = EvidenceStore(engine.evidence)  # weighs the scores (spec B4)
        self._library: tuple[nx.DiGraph, list[Bottleneck]] | None = None  # the graph without them
        self._lock = threading.Lock()  # `run`, `graph` and `scores` change together
        self._done = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._build, daemon=True).start()

    def wait(self, timeout: float | None = None) -> str:
        """Its state, once built or failed (or still building, after `timeout` seconds)."""
        self._done.wait(timeout)
        return self.state

    def _build(self) -> None:
        try:
            vocabulary = load_vocabulary(self.engine.vocabulary)
            with Library.open(self.engine.library) as library:
                graph = library_graph(library, vocabulary)
                self.version = library.version
            scores = bottlenecks(graph)
            with self._lock:  # findings handed over while it was built are linked in once it is
                self.vocabulary, self._library = vocabulary, (graph, scores)
                self.graph, self.scores = graph, scores
            self.state, self.message = READY, f"{self.graph.number_of_nodes()} terms in the knowledge graph."
        except LibraryError as error:
            self.state, self.message = FAILED, f"{error}. {BUILD_LIBRARY}"
        except VocabularyError as error:
            self.state, self.message = FAILED, str(error)
        finally:
            self._done.set()

    def expect(self, fingerprint: str) -> Run:
        """A run of the dataset `fingerprint` has started: the graph drops any other run's findings and
        says it is updating until `add` is given this run's. Returns what to give `add`."""
        run = Run(fingerprint)
        with self._lock:
            self._drop()
            self.run = run
        return run

    def add(self, run: Run, findings: list[Finding]) -> None:
        """Link in `run`'s findings and re-score the terms, in the background (once the graph is
        built); nothing if another run, a load or a clear has come since `expect`."""
        with self._lock:
            run.running = False
        threading.Thread(target=self._add, args=(run, findings), daemon=True).start()

    def _add(self, run: Run, findings: list[Finding]) -> None:
        self._done.wait()
        with self._lock:
            library, vocabulary = self._library, self.vocabulary
        linked = None  # the graph with the findings, and its scores; none if it could not be built
        if library is not None and vocabulary is not None:
            graph = with_findings(library[0], findings, vocabulary)
            linked = graph, bottlenecks(graph)
        with self._lock:
            if self.run is not run:
                return
            if linked is not None:
                self.graph, self.scores = graph, scores = linked
                run.finding_links = sum(kind == FINDING for _, _, kind in graph.edges(data="kind"))
            run.updating = False

    def drop(self) -> None:
        """Drop the run's findings: the map shows the library's graph alone again."""
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        self.run = None
        if self._library is not None:
            self.graph, self.scores = self._library

    def current(self) -> tuple[nx.DiGraph, list[Bottleneck], list[Bottleneck]] | None:
        """As of one moment: the graph and its scores, a run's findings and all, each times its
        posterior mean in the evidence log (spec B4); and the scores of the library's graph alone,
        unweighted. None until built."""
        with self._lock:
            if self.state != READY or self.graph is None or self._library is None:
                return None
            graph, scores, library = self.graph, self.scores, self._library[1]
        return graph, weighted(scores, graph, self.evidence.log()), library

    def map_view(self) -> dict[str, Any]:
        """The bottleneck map's data once the graph is built; until then its state and why. `run` is
        the run whose findings are in (or being added), with its fingerprint's first characters."""
        with self._lock:
            graph, scores, run = self.graph, self.scores, self.run
            said = run and (run.running, run.updating, run.finding_links)  # as of one moment
        log = self.evidence.log()
        shown = None
        if run and said:
            running, updating, count = said
            shown = {
                "fingerprint": run.fingerprint[:SHOWN_FINGERPRINT],
                "updating": updating,
                "finding_links": count,
            }
        if self.state != READY or graph is None:
            return {"state": self.state, "message": self.message, "run": shown}
        message = self.message
        if shown:
            if running:
                message += f" {AWAITING}"
            elif updating:
                message += f" {UPDATING}"
            else:
                links = f"{count} finding link{'' if count == 1 else 's'}"
                message += f" With the findings of the run on dataset {shown['fingerprint']}: {links}."
        terms = {n for n, kind in graph.nodes(data="kind") if kind != SYNDROME}
        evidence = {b.name: log.posterior(b.name, graph.nodes[b.name]["papers"]).mean for b in scores}
        return {
            "state": READY,
            "message": message,
            "run": shown,
            "weights": asdict(Weights()),
            "syndromes": list(SYNDROMES),
            "terms": [
                asdict(b)
                | {"syndromes": list(b.syndromes), "overlap": b.overlap, "evidence": evidence[b.name]}
                for b in weighted(scores, graph, log)
            ],
            "co_mentions": [
                {"a": a, "b": b, "papers": data["papers"], "label": data["label"]}
                for a, b, data in graph.edges(data=True)
                if data["kind"] == CO_MENTION and a < b and a in terms and b in terms
            ],
            "links": [
                {
                    "syndrome": graph.nodes[a]["syndrome"],
                    "from": a,  # the syndrome's term, which `edge` takes
                    "term": b,
                    "kind": data["kind"],
                    "label": data["label"],
                }
                for a, b, data in graph.edges(data=True)
                if data["kind"] in (LITERATURE, FINDING)
            ],
        }

    def edge(self, a: str, b: str) -> dict[str, Any]:
        """The link a → b (a co-mention either way) and the passages behind it; KeyError if the
        graph has no such link, or is not built."""
        graph = self.graph
        if graph is None or not graph.has_edge(a, b):
            raise KeyError((a, b))
        data = graph.edges[a, b]
        ids = data["passage_ids"]
        with Library.open(self.engine.library) as library:
            passages = [p for p in _passages(library, ids[:MOST_PASSAGES], self.version) if p]
        return {
            "a": a,
            "b": b,
            "kind": data["kind"],
            "label": data["label"],
            "papers": data["papers"],
            "passages": passages,
            "more": max(0, len(ids) - MOST_PASSAGES),
            "findings": [
                {
                    k: getattr(f, k)
                    for k in ("marker_name", "comparison", "group_a", "group_b", "sex", "label")
                }
                for f in data.get("findings", ())
            ],
        }


def _passages(library: Library, ids: tuple[str, ...], version: int | None) -> list[dict[str, Any] | None]:
    """Each passage with its paper and links, as the library's `version` holds them (the one the graph
    was built from, though the library is rebuilt while the app runs); None if it does not."""
    by_paper: dict[str, dict[str, str]] = {}  # each paper's passages' texts, by passage id
    out: list[dict[str, Any] | None] = []
    for passage_id in ids:
        paper_id = passage_id.rsplit(":", 1)[0]
        if paper_id not in by_paper:
            document = library.document(paper_id, version)
            by_paper[paper_id] = {p.id: p.text for p in split(document)} if document else {}
        paper = library.paper(paper_id, version)
        text = by_paper[paper_id].get(passage_id)
        if paper is None or text is None:
            out.append(None)
            continue
        source = Source(
            passage_id=passage_id,
            paper_id=paper_id,
            private=False,  # only PubMed papers enter the graph (spec G5)
            title=paper.title,
            text=text,
            doi=paper.doi,
            pmcid=paper.pmcid,
            journal=paper.journal,
            year=paper.year,
        )
        out.append(
            {
                "passage_id": passage_id,
                "paper_id": paper_id,
                "title": paper.title,
                "journal": paper.journal,
                "year": paper.year,
                "text": text,
                "links": links(source),
            }
        )
    return out
