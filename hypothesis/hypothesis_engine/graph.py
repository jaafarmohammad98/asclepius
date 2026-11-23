"""The knowledge graph: co-mentions, and the syndromes' links to terms (spec G4, G5, G6).

A directed networkx graph. Its nodes are vocabulary terms; three of them, the syndromes, link to
other terms (never to each other):

- a *literature* link, syndrome → term, when the term is found in a paper the syndrome's topic
  search found (`topics.toml`), or in a passage that names the syndrome;
- a *finding* link, syndrome → term, stronger, when the term is a significant analysis finding for
  that syndrome: strong or suggestive, of the syndrome's group against a comparator that is not
  another syndrome. The literature behind the link is kept beside its findings;
- a *co-mention*, term → term both ways, when two terms are found in one passage
  (`hypothesis_engine.comentions`). Labelled "co-mentioned in N papers", never causal.

Every node and link keeps what is behind it: `passage_ids` (and `papers`, how many distinct papers
they are from) and, for a finding link and its term, `findings`. A term whose seed citation fails
is left out, even from a finding (G5); `graph.graph["left_out"]` says which and why. The graph is
computed from the library when the app opens and never stored (G6).
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from hypothesis_engine.comentions import library_co_mentions
from hypothesis_engine.evidence import ENZYMES, RATIO, TESTED, Finding, is_intervention
from hypothesis_engine.library import Library
from hypothesis_engine.vocabulary import Vocabulary

# A syndrome's topic in `topics.toml` (and its group name in an analysis export), and its term
SYNDROMES = {
    "PSSD": "post-SSRI sexual dysfunction",
    "PFS": "post-finasteride syndrome",
    "PAS": "post-accutane syndrome",
}
CO_MENTION, LITERATURE, FINDING = "co-mention", "literature", "finding"
# The genes of the enzyme an enzyme-activity ratio stands for (`evidence.ENZYMES`): the ratio is a
# finding about them. The estrogen hydroxylation ratios have no one gene behind them, so none.
ENZYME_GENES = {
    "5-alpha reductase": ("SRD5A1", "SRD5A2"),
    "11-beta-hydroxysteroid dehydrogenase type 2": ("HSD11B2",),
    "COMT methylation": ("COMT",),
}


def library_graph(library: Library, vocabulary: Vocabulary, findings: Iterable[Finding] = ()) -> nx.DiGraph:
    """The graph of the library's current version, with the links of `findings` (an analysis
    export's, `evidence.read_findings`)."""
    found = library_co_mentions(library, vocabulary)
    graph = nx.DiGraph(left_out=found.left_out)
    for node in found.nodes.values():
        graph.add_node(
            node.name, kind=node.kind, papers=node.papers, passage_ids=node.passage_ids, findings=()
        )
    for edge in found.edges:
        data = {
            "kind": CO_MENTION,
            "papers": edge.papers,
            "passage_ids": edge.passage_ids,
            "label": edge.label,
        }
        graph.add_edge(edge.a, edge.b, **data)
        graph.add_edge(edge.b, edge.a, **data)
    topics: dict[str, list[str]] = {}  # by paper

    def found_by(paper: str) -> list[str]:
        if paper not in topics:
            topics[paper] = library.topics(paper)
        return topics[paper]

    for topic, name in SYNDROMES.items():
        if name in found.left_out:
            continue
        for node in [n for n in found.nodes.values() if n.kind != "syndrome"]:
            naming = graph.edges[name, node.name]["passage_ids"] if graph.has_edge(name, node.name) else ()
            ids = tuple(i for i in node.passage_ids if i in naming or topic in found_by(_paper(i)))
            if ids:
                _link(graph, topic, node.name, ids)
    return with_findings(graph, findings, vocabulary)


def with_findings(graph: nx.DiGraph, findings: Iterable[Finding], vocabulary: Vocabulary) -> nx.DiGraph:
    """A copy of `graph` with the links of `findings`; the terms they name join it if need be."""
    graph = graph.copy()
    vocabulary = vocabulary.without(graph.graph["left_out"])
    kinds = {t.name: t.kind for t in vocabulary.terms}
    for finding in findings:
        topic = _syndrome(finding)
        if topic is None or SYNDROMES[topic] in graph.graph["left_out"]:
            continue
        for term in sorted(_terms(finding, vocabulary)):
            if kinds[term] == "syndrome":
                continue
            if term not in graph:
                graph.add_node(term, kind=kinds[term], papers=0, passage_ids=(), findings=())
            graph.nodes[term]["findings"] += (finding,)
            link = graph.edges[SYNDROMES[topic], term] if graph.has_edge(SYNDROMES[topic], term) else None
            if link is None or link["kind"] == CO_MENTION:  # no literature link: no passage of it either
                _link(graph, topic, term, ())
                link = graph.edges[SYNDROMES[topic], term]
            link["findings"] += (finding,)
            link["kind"] = FINDING
            link["label"] = _label(link, topic)
    return graph


def _syndrome(finding: Finding) -> str | None:
    """The syndrome a finding is for: its group a, tested against a comparator not a syndrome."""
    significant = finding.label in TESTED and not is_intervention(finding)
    if significant and finding.group_a in SYNDROMES and finding.group_b not in SYNDROMES:
        return finding.group_a
    return None


def _terms(finding: Finding, vocabulary: Vocabulary) -> set[str]:
    terms = vocabulary.find(finding.marker_name)
    if finding.comparison == RATIO and finding.marker_id in ENZYMES:
        enzyme, _ = ENZYMES[finding.marker_id]
        terms |= {g for g in ENZYME_GENES.get(enzyme, ()) if g in {t.name for t in vocabulary.terms}}
    return terms


def _link(graph: nx.DiGraph, topic: str, term: str, passage_ids: tuple[str, ...]) -> None:
    name = SYNDROMES[topic]
    if name not in graph:
        graph.add_node(name, kind="syndrome", papers=0, passage_ids=(), findings=())
    graph.nodes[name]["syndrome"] = topic
    graph.add_edge(
        name,
        term,
        kind=LITERATURE,
        papers=len({_paper(i) for i in passage_ids}),
        passage_ids=passage_ids,
        findings=(),
    )
    graph.edges[name, term]["label"] = _label(graph.edges[name, term], topic)


def _label(link: dict[str, object], topic: str) -> str:
    papers = link["papers"]
    said = [f"found in {papers} {topic} paper{'' if papers == 1 else 's'}"] if papers else []
    if link["findings"]:
        count = len(link["findings"])  # type: ignore[arg-type]
        said.append(f"in {count} significant analysis finding{'' if count == 1 else 's'}")
    return "; ".join(said)


def _paper(passage_id: str) -> str:
    return passage_id.rsplit(":", 1)[0]
