"""Bottleneck scores: where in the knowledge graph the three syndromes converge (spec G7, G8).

Every term of the graph (`graph.library_graph`) but the syndromes gets three metrics:

- *betweenness*: networkx's normalised betweenness centrality on the graph taken as undirected,
  syndromes included, counting hops, so a term between two syndromes lies on their paths;
- *overlap*: which of PSSD, PFS and PAS link to it, by literature or by a finding (a co-mention
  with a syndrome's name is not a link);
- *cascade*: its downstream reach, the sum over every other term it reaches by co-mentions of
  the strongest path's strength there: the product of its edges' strengths, an edge's strength
  being its papers over the most papers of any co-mention. Paths do not pass through a syndrome.

Each is scaled to 0 to 1 (overlap over the three syndromes, the others over the graph's highest)
and the score is their weighted sum, by default overlap 45%, cascade 35%, betweenness 20%, so a
generic hub does not outrank a point where the syndromes converge. A co-mention says two terms
were mentioned together, not that one acts on the other: a score is where to look, not a cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from hypothesis_engine.graph import CO_MENTION, FINDING, LITERATURE, SYNDROME, SYNDROMES


@dataclass(frozen=True)
class Weights:
    """How much each scaled metric counts; any non-negative numbers, used as shares of their sum."""

    overlap: float = 0.45
    cascade: float = 0.35
    betweenness: float = 0.20

    def __post_init__(self) -> None:
        values = (self.overlap, self.cascade, self.betweenness)
        if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) == 0:
            raise ValueError(f"weights must be non-negative and not all zero: {values}")

    @property
    def total(self) -> float:
        return self.overlap + self.cascade + self.betweenness


@dataclass(frozen=True)
class Bottleneck:
    """A term's score (0 to 1) beside its raw metrics."""

    name: str
    kind: str
    score: float
    betweenness: float  # normalised betweenness centrality, 0 to 1
    syndromes: tuple[str, ...]  # the syndromes linking to it: "PSSD", "PFS", "PAS"
    cascade: float  # summed strongest-path strength to the terms it reaches

    @property
    def overlap(self) -> int:
        return len(self.syndromes)


def bottlenecks(graph: nx.DiGraph, weights: Weights = Weights()) -> list[Bottleneck]:
    """The graph's terms, highest score first (then by name)."""
    syndromes = {n for n, kind in graph.nodes(data="kind") if kind == SYNDROME}
    terms = sorted(n for n in graph if n not in syndromes)
    if not terms:
        return []
    betweenness = nx.betweenness_centrality(graph.to_undirected(as_view=True))
    cascade = _cascade(graph.subgraph(terms))
    highest_betweenness = max(betweenness[t] for t in terms)
    highest_cascade = max(cascade.values())
    scored = []
    for term in terms:
        linked = tuple(
            sorted(
                graph.nodes[s]["syndrome"]
                for s in graph.predecessors(term)
                if s in syndromes and graph.edges[s, term]["kind"] in (LITERATURE, FINDING)
            )
        )
        score = (
            weights.overlap * len(linked) / len(SYNDROMES)
            + weights.cascade * _scaled(cascade[term], highest_cascade)
            + weights.betweenness * _scaled(betweenness[term], highest_betweenness)
        ) / weights.total
        scored.append(
            Bottleneck(term, graph.nodes[term]["kind"], score, betweenness[term], linked, cascade[term])
        )
    return sorted(scored, key=lambda b: (-b.score, b.name))


def _cascade(terms: nx.DiGraph) -> dict[str, float]:
    """Per term, the summed strength of its strongest co-mention path to every other term."""
    arcs = [
        (a, b, data["papers"])
        for a, b, data in terms.edges(data=True)
        if data["kind"] == CO_MENTION and data["papers"] > 0
    ]
    most_papers = max((p for _, _, p in arcs), default=0)
    strengths = nx.DiGraph()
    strengths.add_nodes_from(terms)
    # The strongest path, a product of strengths, is the shortest by -log strength.
    strengths.add_weighted_edges_from((a, b, -math.log(p / most_papers)) for a, b, p in arcs)
    reach = {}
    for term in strengths:
        lengths = nx.single_source_dijkstra_path_length(strengths, term)
        reach[term] = sum(math.exp(-d) for other, d in lengths.items() if other != term)
    return reach


def _scaled(value: float, highest: float) -> float:
    return value / highest if highest else 0.0
