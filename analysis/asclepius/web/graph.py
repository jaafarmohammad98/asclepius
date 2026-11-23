"""The knowledge graph in the web app (milestone 6): built from the paper library in the background
when the app opens, so the page is not kept waiting (a few seconds on the full library), and never
stored (spec G6). A run's findings add their links with `hypothesis_engine.graph.with_findings`.
"""

from __future__ import annotations

import threading

import networkx as nx
from hypothesis_engine.graph import library_graph
from hypothesis_engine.library import Library, LibraryError
from hypothesis_engine.vocabulary import VocabularyError, load_vocabulary

from asclepius.web.hypotheses import BUILD_LIBRARY, Engine

BUILDING, READY, FAILED = "building", "ready", "failed"


class KnowledgeGraph:
    """The graph of the engine's library, once built; until then, or if it cannot be, why not."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.state = BUILDING
        self.message = "The knowledge graph is being built from the paper library."
        self.graph: nx.DiGraph | None = None
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
                self.graph = library_graph(library, vocabulary)
            self.state, self.message = READY, f"{self.graph.number_of_nodes()} terms in the knowledge graph."
        except LibraryError as error:
            self.state, self.message = FAILED, f"{error}. {BUILD_LIBRARY}"
        except VocabularyError as error:
            self.state, self.message = FAILED, str(error)
        finally:
            self._done.set()
