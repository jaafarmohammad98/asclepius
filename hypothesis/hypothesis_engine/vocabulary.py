"""The curated vocabulary the knowledge graph's nodes come from (spec G1, G2, G5).

A term is a gene (its HGNC symbol), a hormone, a neurosteroid, a receptor, a symptom or a
syndrome, with the other names it goes by. A name with a capital letter or a digit in it (a
symbol: "AR", "IL-6", "5-HT") matches case-sensitively, so "Ar" or "bdnf" do not count, except
that one starting in lower case ("post-SSRI sexual dysfunction") may also start with a capital;
a name in lower case ("allopregnanolone", "androgen receptor") matches with case aside. Either
may take a plural "s". Names match whole: not inside a longer word or symbol, Greek letters
included ("AR" is not found in "ARX", "CAR" or "ARα"). Where names overlap, the one starting
first, then the longer, wins ("GABA-A receptor" over "GABA").

An ambiguous name (`Vocabulary.ambiguous`: "AR", "CAT", "REST"...) counts only when a gene word
(`GENE_WORDS`: "gene", "receptor", "expression", "polymorphism"...) is within
`GENE_WORD_WINDOW` words of it, so "AR (augmented reality)" is not the androgen receptor gene
and "AR gene expression" is.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, get_args

DEFAULT_VOCABULARY = Path(__file__).with_name("vocabulary.toml")
Kind = Literal["gene", "hormone", "neurosteroid", "receptor", "symptom", "syndrome"]
KINDS: tuple[Kind, ...] = get_args(Kind)
GENE_WORD_WINDOW = 3  # words either side of an ambiguous name
GENE_WORDS = frozenset(
    {
        "gene", "genes", "genetic", "receptor", "receptors", "expression", "expressed",
        "polymorphism", "polymorphisms", "variant", "variants", "allele", "alleles",
        "mrna", "snp", "snps", "knockout", "transcript", "transcripts",
    }
)  # fmt: skip

_PMID = re.compile(r"[1-9][0-9]*")
_NAME_CHARS = "A-Za-z0-9α-ωΑ-Ω"  # a name is not found next to one of these, as it would go on
_WORD = re.compile(r"[A-Za-z0-9]+")


class VocabularyError(Exception):
    """The vocabulary file is missing or malformed; the message names the file and the problem."""


@dataclass(frozen=True)
class Term:
    name: str  # the node's name: an HGNC symbol, or the usual name
    kind: Kind
    aliases: tuple[str, ...]
    pmid: str  # a library paper in which the term is found (spec G5)

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class Vocabulary:
    terms: tuple[Term, ...]
    ambiguous: frozenset[str] = frozenset()
    _matcher: _Matcher = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_matcher", _Matcher(self))  # a frozen dataclass's derived field

    def without(self, names: Iterable[str]) -> Vocabulary:
        """This vocabulary less the terms named."""
        gone = set(names)
        terms = tuple(t for t in self.terms if t.name not in gone)
        kept = {n for t in terms for n in t.names}
        return Vocabulary(terms, self.ambiguous & kept)

    def find(self, text: str) -> set[str]:
        """The names of the terms found in `text`."""
        return self._matcher.find(text)


def load_vocabulary(path: Path = DEFAULT_VOCABULARY) -> Vocabulary:
    def fail(problem: str) -> VocabularyError:
        return VocabularyError(f"{path}: {problem}")

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise fail(str(error)) from error
    entries = data.get("term")
    if not isinstance(entries, list) or not entries:
        raise fail("no [[term]] entries")
    terms: list[Term] = []
    term_of: dict[str, str] = {}  # each name (a term's own or an alias) to its term's name
    for number, entry in enumerate(entries, start=1):
        name, kind, aliases, pmid = (entry.get(k) for k in ("name", "kind", "aliases", "pmid"))
        if not isinstance(name, str) or not name.strip():
            raise fail(f"term {number} has no name")
        name = name.strip()
        if kind not in KINDS:
            raise fail(f"term {name!r} has kind {kind!r}, not one of {', '.join(KINDS)}")
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
            raise fail(f"term {name!r} has aliases that are not a list of names")
        if not isinstance(pmid, str) or not _PMID.fullmatch(pmid):
            raise fail(f"term {name!r} has no PMID")
        term = Term(name, cast(Kind, kind), tuple(a.strip() for a in aliases), pmid)
        for each in term.names:
            if each in term_of:
                raise fail(f"{each!r} names two terms")
            term_of[each] = name
        terms.append(term)
    ambiguous = data.get("ambiguous", [])
    if not isinstance(ambiguous, list) or not all(isinstance(a, str) for a in ambiguous):
        raise fail("ambiguous is not a list of names")
    for each in ambiguous:
        if each not in term_of:
            raise fail(f"ambiguous name {each!r} is no term's name")
    return Vocabulary(tuple(terms), frozenset(ambiguous))


def _folds(name: str) -> bool:
    """Whether the name matches with case aside: no capital letter and no digit in it."""
    return name == name.lower() and not any(c.isdigit() for c in name)


def _trie(names: Iterable[str]) -> str:
    """A pattern matching any of `names`, the longest that fits first, built as a prefix tree.

    Much faster than an alternation of a thousand names: at each position the regex engine
    follows one branch per character instead of trying every name.
    """
    root: dict[str, dict] = {}
    for name in names:
        node = root
        for char in name:
            node = node.setdefault(char, {})
        node[""] = {}  # the end of a name

    def pattern(node: dict[str, dict]) -> str:
        ends = "" in node
        branches = [re.escape(char) + pattern(child) for char, child in sorted(node.items()) if char]
        if not branches:
            return ""
        group = branches[0] if len(branches) == 1 and len(node) == 1 + ends else f"(?:{'|'.join(branches)})"
        return f"(?:{group})?" if ends else group

    return pattern(root)


class _Matcher:
    def __init__(self, vocabulary: Vocabulary) -> None:
        self._ambiguous = vocabulary.ambiguous
        self._exact: dict[str, str] = {}
        self._folded: dict[str, str] = {}
        for term in vocabulary.terms:
            for name in term.names:
                if _folds(name):
                    self._folded[name.casefold()] = term.name
                else:
                    self._exact[name] = term.name
        for name, term_name in list(self._exact.items()):
            if name[0].islower():  # it may also start a sentence
                self._exact.setdefault(name[0].upper() + name[1:], term_name)
        around = f"(?<![{_NAME_CHARS}])(?:{{}})(?![{_NAME_CHARS}])"
        self._patterns = []
        if self._exact:
            self._patterns.append(re.compile(around.format(f"{_trie(self._exact)}s?")))
        if self._folded:
            self._patterns.append(re.compile(around.format(f"{_trie(self._folded)}s?"), re.IGNORECASE))

    def _name(self, matched: str) -> tuple[str, str]:
        """The vocabulary name the matched text is (less a plural "s"), and its term's name."""
        for written in (matched, matched[:-1] if matched[-1] in "sS" else matched):
            if written in self._exact:
                return written, self._exact[written]
            if written.casefold() in self._folded:
                return written, self._folded[written.casefold()]
        raise AssertionError(matched)

    def _counts(self, match: re.Match[str], name: str) -> bool:
        """Whether the match counts: always, unless the name is ambiguous and no gene word is near."""
        if name not in self._ambiguous:
            return True
        text = match.string
        before = _WORD.findall(text, 0, match.start())[-GENE_WORD_WINDOW:]
        after = [
            m.group()
            for _, m in zip(range(GENE_WORD_WINDOW), _WORD.finditer(text, match.end()), strict=False)
        ]
        return any(word.lower() in GENE_WORDS for word in before + after)

    def find(self, text: str) -> set[str]:
        matches = sorted(
            (m for pattern in self._patterns for m in pattern.finditer(text)),
            key=lambda m: (m.start(), -len(m.group())),
        )
        found, reached = set(), 0
        for match in matches:  # where names overlap, the one starting first, then the longer, wins
            if match.start() >= reached:
                reached = match.end()
                name, term = self._name(match.group())
                if self._counts(match, name):
                    found.add(term)
        return found
