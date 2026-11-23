"""Bottleneck scores: betweenness, syndrome overlap and cascade impact per node (spec G7, G8)."""

from __future__ import annotations

import networkx as nx
import pytest

from hypothesis_engine.bottlenecks import Weights, bottlenecks
from hypothesis_engine.graph import CO_MENTION, FINDING, LITERATURE, SYNDROME, SYNDROMES

PSSD, PFS, PAS = SYNDROMES["PSSD"], SYNDROMES["PFS"], SYNDROMES["PAS"]


def _graph(links: list[tuple[str, str, str]], co_mentions: list[tuple[str, str, int]]) -> nx.DiGraph:
    """A graph shaped as `graph.library_graph` makes it: syndrome → term links, co-mentions both ways."""
    graph = nx.DiGraph(left_out={})
    for topic, name in SYNDROMES.items():
        graph.add_node(name, kind=SYNDROME, syndrome=topic)
    for syndrome, term, kind in links:
        graph.add_node(term, kind="gene")
        graph.add_edge(syndrome, term, kind=kind, papers=1)
    for a, b, papers in co_mentions:
        for term in (a, b):
            if term not in graph:
                graph.add_node(term, kind="gene")
        graph.add_edge(a, b, kind=CO_MENTION, papers=papers)
        graph.add_edge(b, a, kind=CO_MENTION, papers=papers)
    return graph


# X is found in the papers of all three syndromes, with one co-mention; the hub H only in PSSD's, and
# co-mentioned with four other terms. Worked out by hand on the undirected graph's 9 nodes (28 pairs):
# - betweenness: H lies on 22 shortest paths (the 6 between leaves, and leaf to X, PSSD, PFS, PAS),
#   X on 13 (leaf to PFS and PAS, PSSD-PFS, PSSD-PAS, PFS-PAS, H to PFS and PAS): 22/28 and 13/28.
# - cascade, the strongest path to each other term, an edge's strength being papers / the most papers
#   (2): H reaches X at 0.5 and the leaves at 1 (4.5); X reaches H at 0.5 and the leaves through it at
#   0.5 (2.5); a leaf reaches H, the other leaves at 1 and X at 0.5 (4.5).
FIXTURE = _graph(
    [(PSSD, "X", LITERATURE), (PFS, "X", FINDING), (PAS, "X", LITERATURE), (PSSD, "H", LITERATURE)],
    [("X", "H", 1), ("H", "a", 2), ("H", "b", 2), ("H", "c", 2), ("H", "d", 2)],
)


def _by_name(graph: nx.DiGraph, weights: Weights | None = None):
    return {b.name: b for b in bottlenecks(graph, weights or Weights())}


def test_a_node_shared_by_all_three_syndromes_outranks_a_hub_touching_one():
    ranked = [b.name for b in bottlenecks(FIXTURE)]
    assert ranked[:2] == ["X", "H"]


def test_raw_metrics_come_with_the_score():
    found = _by_name(FIXTURE)
    x, h, a = found["X"], found["H"], found["a"]
    assert (x.syndromes, x.overlap, h.syndromes, a.overlap) == (("PAS", "PFS", "PSSD"), 3, ("PSSD",), 0)
    assert (x.betweenness, h.betweenness, a.betweenness) == pytest.approx((13 / 28, 22 / 28, 0))
    assert (x.cascade, h.cascade, a.cascade) == pytest.approx((2.5, 4.5, 4.5))


def test_the_score_is_the_weighted_sum_of_the_metrics_each_scaled_to_0_1():
    found = _by_name(FIXTURE)
    # overlap / 3, cascade / 4.5, betweenness / (22 / 28)
    assert found["X"].score == pytest.approx(0.45 * 1 + 0.35 * 2.5 / 4.5 + 0.20 * 13 / 22)
    assert found["H"].score == pytest.approx(0.45 / 3 + 0.35 * 1 + 0.20 * 1)
    assert found["a"].score == pytest.approx(0.35)
    assert PSSD not in found  # a syndrome is not a bottleneck of itself


def test_the_weights_are_parameters_and_count_as_shares_of_their_sum():
    hubs_first = [b.name for b in bottlenecks(FIXTURE, Weights(overlap=0, cascade=0, betweenness=1))]
    assert hubs_first[:2] == ["H", "X"]
    doubled = _by_name(FIXTURE, Weights(overlap=0.9, cascade=0.7, betweenness=0.4))
    assert doubled["X"].score == pytest.approx(_by_name(FIXTURE)["X"].score)


@pytest.mark.parametrize("weights", [(0, 0, 0), (-0.1, 0.6, 0.5), (float("nan"), 0.5, 0.5)])
def test_weights_that_weigh_nothing_are_refused(weights):
    with pytest.raises(ValueError):
        Weights(*weights)


def test_a_cascade_does_not_pass_through_a_syndrome():
    # Both terms are co-mentioned with PSSD; neither reaches the other through it.
    graph = _graph([], [("A", PSSD, 3), ("B", PSSD, 3)])
    found = _by_name(graph)
    assert (found["A"].cascade, found["B"].cascade) == (0, 0)


def test_only_a_literature_or_finding_link_counts_for_overlap_not_a_co_mention():
    graph = _graph([(PFS, "A", FINDING)], [("A", PSSD, 3)])
    assert _by_name(graph)["A"].syndromes == ("PFS",)


def test_a_graph_without_co_mentions_scores_by_overlap_alone():
    graph = _graph([(PSSD, "A", LITERATURE), (PFS, "A", LITERATURE)], [])
    graph.remove_node(PAS)
    assert bottlenecks(graph)[0].score == pytest.approx(0.45 * 2 / 3 + 0.20 * 1)
    assert bottlenecks(nx.DiGraph()) == []
