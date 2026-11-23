"""Co-mention edges: two vocabulary terms found in the same library passage (spec G3, G5).

An edge's weight is the number of distinct papers it was found in, and it keeps the ids of
every passage it was counted from, so each one can be opened. It says the terms were
mentioned together, nothing more: it is labelled "co-mentioned in N papers" and never
called causal (overview, rule 5).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations

from hypothesis_engine.library import Library
from hypothesis_engine.passages import Passage, split
from hypothesis_engine.vocabulary import Vocabulary, seed_problems


@dataclass(frozen=True)
class Edge:
    a: str  # the two terms' names, in sorted order
    b: str
    papers: int  # distinct papers
    passage_ids: tuple[str, ...]  # every passage it was counted from, in the order read

    @property
    def label(self) -> str:
        return f"co-mentioned in {self.papers} paper{'' if self.papers == 1 else 's'}"


@dataclass(frozen=True)
class Node:
    name: str
    kind: str
    papers: int  # distinct papers it is found in
    passage_ids: tuple[str, ...]  # every passage it is found in, in the order read


@dataclass(frozen=True)
class CoMentions:
    nodes: dict[str, Node]  # by name; only the terms found in some passage
    edges: tuple[Edge, ...]  # most papers first
    left_out: dict[str, str] = field(default_factory=dict)  # terms whose seed failed, and why


def co_mentions(passages: Iterable[Passage], vocabulary: Vocabulary) -> CoMentions:
    """The edges between the vocabulary's terms in `passages`."""
    kinds = {term.name: term.kind for term in vocabulary.terms}
    mentions: defaultdict[str, list[Passage]] = defaultdict(list)
    pairs: defaultdict[tuple[str, str], list[Passage]] = defaultdict(list)
    for passage in passages:
        names = sorted(vocabulary.find(passage.text))
        for name in names:
            mentions[name].append(passage)
        for pair in combinations(names, 2):
            pairs[pair].append(passage)
    nodes = {name: Node(name, kinds[name], _papers(ps), _ids(ps)) for name, ps in sorted(mentions.items())}
    edges = (Edge(a, b, _papers(ps), _ids(ps)) for (a, b), ps in pairs.items())
    return CoMentions(nodes, tuple(sorted(edges, key=lambda e: (-e.papers, e.a, e.b))))


def _papers(passages: list[Passage]) -> int:
    return len({p.paper_id for p in passages})


def _ids(passages: list[Passage]) -> tuple[str, ...]:
    return tuple(p.id for p in passages)


def library_co_mentions(library: Library, vocabulary: Vocabulary) -> CoMentions:
    """The co-mentions in every passage of the library's current version.

    Terms whose seed citation fails (`seed_problems`) are left out, so nothing uncited enters
    the graph (spec G5); `left_out` says which and why.
    """
    problems = seed_problems(vocabulary, library)
    cited = vocabulary.without(problems)
    passages = (p for document in library.documents() for p in split(document))
    return CoMentions(**{**co_mentions(passages, cited).__dict__, "left_out": problems})
