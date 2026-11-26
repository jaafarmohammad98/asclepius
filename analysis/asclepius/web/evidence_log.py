"""The evidence and hypothesis log in the web app (M6-11, spec B2 to B5): each term of the knowledge graph
as a hypothesis, with its prior, its posterior and the logged results behind them
(`hypothesis_engine.evidence_log`).

A result is logged from the page's form (`add_entry`): a term, supports or refutes, a study type and its
source, a PMID, a DOI or "unpublished observation". The log is kept in the data folder
(`Engine.evidence`), never online. It is exported as its JSON file and an exported log imported into it
(`import_log`), dropping entries it has already. Each term's bottleneck score is multiplied by its
posterior mean, so the bottleneck map and the leaderboard re-rank as results are logged
(`KnowledgeGraph.current` and `map_view`).

The prior is Beta(1, 1) plus the term's literature pseudo-supports, the posterior adds the logged
results. The study types' pseudo-counts are provisional and need expert review, and the page says so.

A log file that cannot be read is never written over, nor exported as empty: the views say why, and
logging, importing and exporting are refused until it is fixed or moved. Agreed 2026-11-26 (M6-11 review).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypothesis_engine.evidence_log import (
    DIRECTIONS,
    PROVISIONAL,
    STUDY_TYPES,
    UNPUBLISHED,
    Entry,
    EvidenceLog,
    EvidenceLogError,
    Posterior,
)
from hypothesis_engine.graph import SYNDROME

if TYPE_CHECKING:
    from asclepius.web.graph import KnowledgeGraph

EXPORT_NAME = "asclepius-evidence-log.json"


class EvidenceViewError(ValueError):
    """A result or an imported log the page cannot log, with what to change."""


class EvidenceStore:
    """The log kept at `path`, read once when the app opens; written whole after each change."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.error: str | None = None  # why the file cannot be read
        try:
            self._log = EvidenceLog.load(path)
        except EvidenceLogError as error:
            self._log = EvidenceLog()
            self.error = (
                f"The evidence log {path} cannot be read ({error}), so no logged result counts. "
                "Fix or move the file and start the app again; nothing is written over it."
            )

    def log(self) -> EvidenceLog:
        """The log as of now: a copy, which later changes leave as it is."""
        with self._lock:
            return EvidenceLog(self._log.entries)

    def add(self, entry: Entry) -> bool:
        """Log `entry` and keep the log; False, and nothing logged, when it has the entry already."""
        with self._lock:
            self._refuse_if_unreadable()
            changed = EvidenceLog(self._log.entries)
            if not changed.add(entry):
                return False
            changed.save(self.path)
            self._log = changed
            return True

    def merge(self, imported: EvidenceLog) -> int:
        """Add the entries of `imported` the log does not have, and keep it; how many were added."""
        with self._lock:
            self._refuse_if_unreadable()
            changed = EvidenceLog(self._log.entries)
            added = changed.merge(imported)
            if added:
                changed.save(self.path)
                self._log = changed
            return added

    def export(self) -> str:
        """The log as its JSON file; EvidenceViewError when the file cannot be read, never as empty."""
        with self._lock:
            self._refuse_if_unreadable()
            return self._log.to_json()

    def _refuse_if_unreadable(self) -> None:
        if self.error:
            raise EvidenceViewError(self.error)


def _beta(posterior: Posterior) -> dict[str, float]:
    return {"alpha": posterior.alpha, "beta": posterior.beta, "mean": posterior.mean}


def _source_url(source: str) -> str | None:
    if source.startswith("PMID:"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{source.removeprefix('PMID:')}/"
    if source.startswith("doi:"):
        return f"https://doi.org/{source.removeprefix('doi:')}"
    return None


def _entry_view(entry: Entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "node": entry.node,
        "direction": entry.direction,
        "study_type": STUDY_TYPES[entry.study_type].label,
        "weight": entry.weight,
        "source": entry.source,
        "url": _source_url(entry.source),
    }


def evidence_view(knowledge: KnowledgeGraph, message: str = "") -> dict[str, Any]:
    """Every term with its prior, posterior and logged entries, once the graph is built; until then,
    or if it cannot be, its state and why. Entries naming a term not in the graph are listed apart."""
    store = knowledge.evidence
    log = store.log()
    shared = {
        "provisional": PROVISIONAL,
        "study_types": [{"key": k, "label": t.label, "weight": t.weight} for k, t in STUDY_TYPES.items()],
        "directions": list(DIRECTIONS),
        "unpublished": UNPUBLISHED,
        "count": len(log.entries),
        "error": store.error,
        "message": message,
    }
    current = knowledge.current()
    if current is None:
        return {"state": knowledge.state, "graph_message": knowledge.message, **shared}
    graph = current[0]
    by_node: dict[str, list[Entry]] = {}
    for entry in log.entries:
        by_node.setdefault(entry.node, []).append(entry)
    terms = sorted(n for n, kind in graph.nodes(data="kind") if kind != SYNDROME)
    known = set(terms)
    return {
        "state": "ready",
        **shared,
        "nodes": [
            {
                "name": name,
                "kind": graph.nodes[name]["kind"],
                "papers": graph.nodes[name]["papers"],
                "prior": _beta(EvidenceLog().posterior(name, graph.nodes[name]["papers"])),
                "posterior": _beta(log.posterior(name, graph.nodes[name]["papers"])),
                "entries": [_entry_view(e) for e in by_node.get(name, [])],
            }
            for name in terms
        ],
        "elsewhere": [_entry_view(e) for e in log.entries if e.node not in known],
    }


def add_entry(request: object, knowledge: KnowledgeGraph) -> dict[str, Any]:
    """Log the result `request` names (`{"node", "direction", "study_type", "source"}`), a term of the
    graph; the view after. EvidenceViewError when it cannot be logged, or is logged already."""
    if not isinstance(request, dict):
        raise EvidenceViewError(
            "A result must be a JSON object naming its term, direction, study type and source."
        )
    current = knowledge.current()
    if current is None:
        raise EvidenceViewError("Wait for the knowledge graph to be built before logging a result.")
    node = request.get("node")
    graph = current[0]
    if not isinstance(node, str) or node not in graph or graph.nodes[node]["kind"] == SYNDROME:
        raise EvidenceViewError(f"{node!r} is not a term of the knowledge graph.")
    direction, study_type, source = (request.get(k) for k in ("direction", "study_type", "source"))
    if not (isinstance(direction, str) and isinstance(study_type, str) and isinstance(source, str)):
        raise EvidenceViewError("A result names its direction, study type and source as text.")
    try:
        entry = Entry.new(node, direction=direction, study_type=study_type, source=source)
    except EvidenceLogError as error:
        raise EvidenceViewError(str(error)) from error
    if not knowledge.evidence.add(entry):
        raise EvidenceViewError(
            f"This result is in the log already: {entry.source}, {entry.direction} {node}."
        )
    return evidence_view(knowledge, f"Logged: {entry.source} {entry.direction} {node}.")


def _entries(n: int) -> str:
    return f"{n} new entr{'y' if n == 1 else 'ies'}"


def import_log(text: str, knowledge: KnowledgeGraph) -> dict[str, Any]:
    """Add the entries of an exported log (`text`, its JSON) that the log does not have; the view
    after, saying how many were new. EvidenceViewError when it is not an evidence log."""
    try:
        imported = EvidenceLog.from_json(text)
    except EvidenceLogError as error:
        raise EvidenceViewError(str(error)) from error
    added = knowledge.evidence.merge(imported)
    total, had = len(imported.entries), len(imported.entries) - added
    message = (
        f"Imported {_entries(added)} of {total}; {had} {'was' if had == 1 else 'were'} in the log already."
    )
    return evidence_view(knowledge, message)
