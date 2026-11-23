"""The knowledge graph in the web app (milestone 6): built from the paper library in the background
when the app opens, so the page is not kept waiting (a few seconds on the full library), and never
stored (spec G6). A run's findings add their links with `hypothesis_engine.graph.with_findings`.

The bottleneck map (M6-04) draws it: `map_view` is every term with its bottleneck score and raw
metrics (`hypothesis_engine.bottlenecks`), each co-mention once and each syndrome link; the page
re-scores the terms itself when a weight slider moves. `edge` is what is behind one link: the
passages it was counted from, each with its paper and where to read it.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any

import networkx as nx
from hypothesis_engine.bottlenecks import Bottleneck, Weights, bottlenecks
from hypothesis_engine.graph import CO_MENTION, FINDING, LITERATURE, SYNDROME, SYNDROMES, library_graph
from hypothesis_engine.library import Library, LibraryError
from hypothesis_engine.passages import split
from hypothesis_engine.pipeline import Source
from hypothesis_engine.vocabulary import VocabularyError, load_vocabulary

from asclepius.web.hypotheses import BUILD_LIBRARY, Engine, links

BUILDING, READY, FAILED = "building", "ready", "failed"
MOST_PASSAGES = 50  # shown for one link; a link of hundreds of papers says how many more there are


class KnowledgeGraph:
    """The graph of the engine's library, once built; until then, or if it cannot be, why not."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.state = BUILDING
        self.message = "The knowledge graph is being built from the paper library."
        self.graph: nx.DiGraph | None = None
        self.scores: list[Bottleneck] = []  # with the default weights
        self.version: int | None = None  # the library version it was built from
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
            self.graph, self.scores = graph, bottlenecks(graph)
            self.state, self.message = READY, f"{self.graph.number_of_nodes()} terms in the knowledge graph."
        except LibraryError as error:
            self.state, self.message = FAILED, f"{error}. {BUILD_LIBRARY}"
        except VocabularyError as error:
            self.state, self.message = FAILED, str(error)
        finally:
            self._done.set()

    def map_view(self) -> dict[str, Any]:
        """The bottleneck map's data once the graph is built; until then its state and why."""
        graph = self.graph
        if self.state != READY or graph is None:
            return {"state": self.state, "message": self.message}
        terms = {n for n, kind in graph.nodes(data="kind") if kind != SYNDROME}
        return {
            "state": READY,
            "message": self.message,
            "weights": asdict(Weights()),
            "syndromes": list(SYNDROMES),
            "terms": [
                asdict(b) | {"syndromes": list(b.syndromes), "overlap": b.overlap} for b in self.scores
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
        if self.graph is None or not self.graph.has_edge(a, b):
            raise KeyError((a, b))
        data = self.graph.edges[a, b]
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
