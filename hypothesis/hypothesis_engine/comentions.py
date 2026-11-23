"""Co-mention edges: two vocabulary terms found in the same library passage (spec G3, G5).

An edge's weight is the number of distinct papers it was found in, and it keeps the ids of
every passage it was counted from, so each one can be opened. It says the terms were
mentioned together, nothing more: it is labelled "co-mentioned in N papers" and never
called causal (overview, rule 5).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import combinations, groupby

from hypothesis_engine.library import Library
from hypothesis_engine.passages import Passage, split
from hypothesis_engine.vocabulary import Kind, Vocabulary


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
    kind: Kind
    papers: int  # distinct papers it is found in
    passage_ids: tuple[str, ...]  # every passage it is found in, in the order read


@dataclass(frozen=True)
class CoMentions:
    nodes: dict[str, Node]  # by name; only the terms found in some passage
    edges: tuple[Edge, ...]  # most papers first
    left_out: dict[str, str] = field(default_factory=dict)  # terms whose seed failed, and why


def co_mentions(passages: Iterable[Passage], vocabulary: Vocabulary) -> CoMentions:
    """The edges between the vocabulary's terms in `passages` (a paper's passages together).

    An abbreviation counts in a passage when the paper's title or one of its passages names the
    term in full (`Vocabulary.find_in_paper`).
    """
    kinds = {term.name: term.kind for term in vocabulary.terms}
    mentions: defaultdict[str, list[Passage]] = defaultdict(list)
    pairs: defaultdict[tuple[str, str], list[Passage]] = defaultdict(list)
    for _, group in groupby(passages, key=lambda p: p.paper_id):
        paper = list(group)
        _, *found = vocabulary.find_in_paper([paper[0].title, *(p.text for p in paper)])
        for passage, names in zip(paper, found, strict=True):
            for name in sorted(names):
                mentions[name].append(passage)
            for pair in combinations(sorted(names), 2):
                pairs[pair].append(passage)
    nodes = {name: Node(name, kinds[name], _papers(ps), _ids(ps)) for name, ps in sorted(mentions.items())}
    edges = (Edge(a, b, _papers(ps), _ids(ps)) for (a, b), ps in pairs.items())
    return CoMentions(nodes, tuple(sorted(edges, key=lambda e: (-e.papers, e.a, e.b))))


def _papers(passages: list[Passage]) -> int:
    return len({p.paper_id for p in passages})


def _ids(passages: list[Passage]) -> tuple[str, ...]:
    return tuple(p.id for p in passages)


def seed_problems(vocabulary: Vocabulary, library: Library) -> dict[str, str]:
    """Why each term whose seed citation fails cannot enter the graph, by name (spec G5).

    A seed holds when its PMID is a PubMed paper of the library's current version and the term
    is found in that paper's title or text, by the same rules as in passages.
    """
    problems = {}
    for term in vocabulary.terms:
        paper = library.document(term.pmid)
        if paper is None or paper.private:
            problems[term.name] = f"its paper {term.pmid} is not in the library"
        elif term.name not in vocabulary.find(f"{paper.title}\n{paper.text}"):
            problems[term.name] = f"it is not found in its paper {term.pmid}"
    return problems


def library_co_mentions(library: Library, vocabulary: Vocabulary) -> CoMentions:
    """The co-mentions in the passages of the library's current version's PubMed papers.

    Only what traces to a PMID enters the graph (spec G5): the researcher's private PDFs are not
    read, and terms whose seed citation fails (`seed_problems`) are left out; `left_out` says
    which and why.
    """
    problems = seed_problems(vocabulary, library)
    passages = (p for d in library.documents() if not d.private for p in split(d))
    return replace(co_mentions(passages, vocabulary.without(problems)), left_out=problems)
