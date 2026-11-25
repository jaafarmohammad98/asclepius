"""The drug repurposing leaderboard in the web app (M6-07, spec R3 to R6): the approved drugs acting on
the top bottleneck targets, ranked by TOPSIS as candidates for study, never as treatments.

The drugs are the cache `asclepius-library drugs` wrote (`hypothesis_engine.drugs`); nothing is fetched
here. Each target's bottleneck score and syndromes are the app's knowledge graph's as it is now, a run's
finding links included (M6-04b), not those of when the drugs were fetched; so a run re-ranks the drugs.
When the graph cannot be built, the cache's are used, and the message says so.

`hypothesis_engine.topsis` gives each drug's five criteria, stand-ins for unknowns included; the page
weights them and ranks by TOPSIS itself (`topsis.js`), so a slider re-ranks without asking the server.
A test checks the two rank alike. Each row has its risks with where they are stated, its ChEMBL record,
and per target the syndrome links whose passages the page opens (`KnowledgeGraph.edge`). The known
causes are listed apart, never ranked.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import networkx as nx
from hypothesis_engine.bottlenecks import Bottleneck
from hypothesis_engine.drugs import DrugCache, DrugsError, read_drug_cache
from hypothesis_engine.graph import FINDING, LITERATURE
from hypothesis_engine.topsis import COST, CRITERIA, Row, Weights, rank

from asclepius.web.graph import BUILDING, FAILED, READY, KnowledgeGraph

MISSING = "missing"
CACHED_SCORES = (
    "The knowledge graph could not be built, so each target's score is the one it had when the drug data "
    "was fetched, and its passages cannot be opened."
)


def rescored(cache: DrugCache, scores: list[Bottleneck]) -> DrugCache:
    """The cache with each target's score and syndromes as `scores` gives them; a target they do not
    name keeps its own."""
    by_name = {b.name: b for b in scores}
    return replace(
        cache,
        targets=tuple(
            replace(t, score=b.score, syndromes=b.syndromes) if (b := by_name.get(t.symbol)) else t
            for t in cache.targets
        ),
    )


def _links(graph: nx.DiGraph | None, symbol: str) -> list[dict[str, str]]:
    """The syndrome links to a target: the passages (or findings) that tied a syndrome to it."""
    if graph is None or symbol not in graph:
        return []
    return [
        {"syndrome": graph.nodes[a]["syndrome"], "from": a, "kind": data["kind"], "label": data["label"]}
        for a, _, data in graph.in_edges(symbol, data=True)
        if data["kind"] in (FINDING, LITERATURE)
    ]


def _row(row: Row, cache: DrugCache, graph: nx.DiGraph | None) -> dict[str, Any]:
    drug, safety = row.drug, row.drug.safety
    targets = {t.symbol: t for t in cache.targets}
    return {
        "chembl_id": drug.chembl_id,
        "name": drug.name,
        "url": drug.url,
        "label": row.label,
        "criteria": asdict(row.criteria),
        "risks": list(row.risks),
        "assumed": list(row.assumed),
        "bbb_note": row.bbb_note,
        "withdrawn": drug.withdrawn,
        "generic": safety.generic if safety else None,
        "label_url": safety.label.url if safety and safety.label else None,
        "mentions": [asdict(m) for m in safety.label.mentions] if safety and safety.label else [],
        "listed_risks": [asdict(r) for r in safety.risks] if safety else [],
        "unchecked": safety.unchecked if safety else None,
        "targets": [
            {
                "symbol": a.target,
                "action_type": a.action_type,
                "mechanism": a.mechanism,
                "score": targets[a.target].score if a.target in targets else None,
                "syndromes": list(targets[a.target].syndromes) if a.target in targets else [],
                "links": _links(graph, a.target),
            }
            for a in drug.actions
        ],
    }


def leaderboard_view(path: Path, knowledge: KnowledgeGraph) -> dict[str, Any]:
    """The leaderboard's data once the graph is built (or has failed); until then its state."""
    if knowledge.state == BUILDING:
        return {"state": BUILDING, "message": "Waiting for the knowledge graph to score the drugs' targets."}
    try:
        cache = read_drug_cache(path)
    except DrugsError as error:
        return {"state": MISSING, "message": str(error)}
    current = knowledge.current()
    graph = None
    message = f"{len(cache.ranked)} drugs ranked, drug data fetched {cache.fetched_at[:10]}."
    if current is not None:
        graph, scores = current
        cache = rescored(cache, scores)
    elif knowledge.state == FAILED:
        message += f" {CACHED_SCORES}"
    ranking = rank(cache)
    return {
        "state": READY,
        "message": message,
        "criteria": list(CRITERIA),
        "cost": sorted(COST),
        "weights": asdict(Weights()),
        "rows": [_row(r, cache, graph) for r in ranking.ranked],
        "known_causes": [
            {"name": d.name, "chembl_id": d.chembl_id, "known_cause": d.known_cause, "url": d.url}
            for d in ranking.known_causes
        ],
    }
