"""The curated vocabulary the knowledge graph's nodes come from (spec G1, G2, G5).

A term is a gene (its HGNC symbol), a hormone, a neurosteroid, a receptor, a symptom or a
syndrome, with the other names it goes by. Within a name, a word in lower case matches with case
aside ("androgen receptor" finds "Androgen Receptor") and a word with a capital letter or a digit
in it (a symbol: "AR", "IL-6", "SSRI") matches exactly, so "Ar" or "bdnf" do not count; a name
starting in lower case may also start with a capital ("mGluR5", "MGluR5"). A hyphen in a name
matches any hyphen, and any name may take a plural "s". Names match whole: not inside a longer
word or symbol, Greek letters included ("AR" is not found in "ARX", "CAR" or "ARα"). Where names
overlap, the longer wins ("GABA-A receptor" over "GABA").

An abbreviation (`Term.abbreviations`: "PSSD", "PFS") matches exactly, and counts only in a paper
that also names the term in full: elsewhere "PSSD" is post-stroke sexual dysfunction or a
nerve-testing device, and "PFS" progression-free survival.

An ambiguous name (`Vocabulary.ambiguous`: "AR", "CAT", "REST"...) counts only when a gene word
(`GENE_WORDS`: "gene", "receptor", "expression", "polymorphism"...) is within
`GENE_WORD_WINDOW` words of it, so "AR (augmented reality)" is not the androgen receptor gene
and "AR gene expression" is.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Sequence
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
_NAME_WORD = re.compile(f"[{_NAME_CHARS}]+")
_WORD = re.compile(r"[A-Za-z0-9]+")
_HYPHENS = "-‐‑"  # hyphen-minus, hyphen, non-breaking hyphen


class VocabularyError(Exception):
    """The vocabulary file is missing or malformed; the message names the file and the problem."""


@dataclass(frozen=True)
class Term:
    name: str  # the node's name: an HGNC symbol, or the usual name
    kind: Kind
    aliases: tuple[str, ...]
    pmid: str  # a library paper in which the term is found (spec G5)
    abbreviations: tuple[str, ...] = ()  # count only in a paper that names the term in full

    @property
    def names(self) -> tuple[str, ...]:
        """Its full names: its own and its aliases."""
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
        kept = {n for t in terms for n in (*t.names, *t.abbreviations)}
        return Vocabulary(terms, self.ambiguous & kept)

    def find(self, text: str) -> set[str]:
        """The names of the terms found in `text`, taken as a whole paper."""
        return self.find_in_paper([text])[0]

    def find_in_paper(self, texts: Sequence[str]) -> list[set[str]]:
        """The names of the terms found in each of one paper's texts (its title, its passages).

        An abbreviation counts in any of them when a full name of its term is in one of them.
        """
        found = [self._matcher.find(text) for text in texts]
        in_full = {term for each in found for term, abbreviated in each if not abbreviated}
        return [{term for term, abbreviated in each if not abbreviated or term in in_full} for each in found]


def _hyphens(name: str) -> str:
    return re.sub(f"[{_HYPHENS}]", "-", name)


def _keys(name: str, *, exactly: bool = False) -> list[str]:
    """What a name is looked up by: as written, any hyphen as "-"; and, when it has a lower-case
    word, also with case aside ("folded:" in front)."""
    keys = [_hyphens(name)]
    if not exactly and _folding(name):
        keys.append("folded:" + _hyphens(name).casefold())
    return keys


def _folding(name: str) -> set[int]:
    """The places of the letters in the name's lower-case words: those match with case aside."""
    places: set[int] = set()
    for word in _NAME_WORD.finditer(name):
        if word.group() == word.group().lower() and not any(c.isdigit() for c in word.group()):
            places.update(range(word.start(), word.end()))
    return places


def load_vocabulary(path: Path = DEFAULT_VOCABULARY) -> Vocabulary:
    def fail(problem: str) -> VocabularyError:
        return VocabularyError(f"{path}: {problem}")

    def names(value: object) -> bool:
        return isinstance(value, list) and all(isinstance(n, str) and n.strip() for n in value)

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise fail(str(error)) from error
    entries = data.get("term")
    if not isinstance(entries, list) or not entries:
        raise fail("no [[term]] entries")
    terms: list[Term] = []
    term_of: dict[str, str] = {}  # each name, any hyphen as "-", to its term's name
    alike: dict[
        str, list[tuple[str, str, bool]]
    ] = {}  # names alike with case aside: (name, term, abbreviated)
    for number, entry in enumerate(entries, start=1):
        name, kind, pmid = entry.get("name"), entry.get("kind"), entry.get("pmid")
        aliases, abbreviations = entry.get("aliases", []), entry.get("abbreviations", [])
        if not isinstance(name, str) or not name.strip():
            raise fail(f"term {number} has no name")
        name = name.strip()
        if kind not in KINDS:
            raise fail(f"term {name!r} has kind {kind!r}, not one of {', '.join(KINDS)}")
        if not names(aliases) or not names(abbreviations):
            raise fail(f"term {name!r} has aliases or abbreviations that are not a list of names")
        if not isinstance(pmid, str) or not _PMID.fullmatch(pmid):
            raise fail(f"term {name!r} has no PMID")
        term = Term(
            name,
            cast(Kind, kind),
            tuple(a.strip() for a in aliases),
            pmid,
            tuple(a.strip() for a in abbreviations),
        )
        for each, abbreviated in _all_names(term):
            for other, other_term, other_abbreviated in alike.get(_hyphens(each).casefold(), []):
                if other_term != name and (
                    _matches(other, each, exactly=other_abbreviated)
                    or _matches(each, other, exactly=abbreviated)
                ):
                    both = repr(each) if each == other else f"{other!r} and {each!r}"
                    raise fail(f"{both} name{'s' if each == other else ''} two terms")
            alike.setdefault(_hyphens(each).casefold(), []).append((each, name, abbreviated))
            term_of[_hyphens(each)] = name
        terms.append(term)
    ambiguous = data.get("ambiguous", [])
    if not isinstance(ambiguous, list) or not all(isinstance(a, str) for a in ambiguous):
        raise fail("ambiguous is not a list of names")
    for each in ambiguous:
        if _hyphens(each) not in term_of:
            raise fail(f"ambiguous name {each!r} is no term's name")
    return Vocabulary(tuple(terms), frozenset(ambiguous))


def _all_names(term: Term) -> list[tuple[str, bool]]:
    """The term's names and abbreviations, each with whether it is an abbreviation."""
    return [*((n, False) for n in term.names), *((n, True) for n in term.abbreviations)]


def _matches(name: str, text: str, *, exactly: bool) -> bool:
    """Whether `name` matches all of `text`, by the rules of the module docstring."""
    return re.fullmatch("".join(_atoms(name, exactly=exactly)[:-1]), text) is not None


def _atoms(name: str, *, exactly: bool) -> list[str]:
    """The name as one pattern piece per character, each matching as the module docstring says."""
    folding = set() if exactly else _folding(name)
    atoms = []
    for place, char in enumerate(name):
        if char in _HYPHENS:
            atoms.append(f"[{re.escape(_HYPHENS)}]")
        elif place in folding:
            atoms.append(f"(?i:{re.escape(char)})")
        elif place == 0 and char.islower() and not exactly:
            atoms.append(f"[{re.escape(char)}{re.escape(char.upper())}]")
        else:
            atoms.append(re.escape(char))
    plural = "(?i:s)" if len(name) - 1 in folding else "s"
    return atoms + [plural]


def _trie(names: Iterable[list[str]]) -> str:
    """A pattern matching any of `names` (each a list of atoms ending in an optional plural),
    the longest that fits first, built as a prefix tree.

    Much faster than an alternation of a thousand names: at each position the regex engine
    follows one branch per character instead of trying every name.
    """
    root: dict[str, dict] = {}
    for atoms in names:
        node = root
        for atom in atoms[:-1]:
            node = node.setdefault(atom, {})
        node[""] = {}  # a name ends here
        node.setdefault(atoms[-1], {})[""] = {}  # or here, with its plural

    def pattern(node: dict[str, dict]) -> str:
        ends = "" in node
        branches = [atom + pattern(child) for atom, child in sorted(node.items()) if atom]
        if not branches:
            return ""
        group = branches[0] if len(branches) == 1 else f"(?:{'|'.join(branches)})"
        return f"(?:{group})?" if ends else group

    return pattern(root)


class _Matcher:
    def __init__(self, vocabulary: Vocabulary) -> None:
        self._ambiguous = vocabulary.ambiguous
        self._names: dict[str, tuple[str, str, bool]] = {}  # key -> (name, term's name, abbreviated)
        atoms = []
        for term in vocabulary.terms:
            for name, abbreviated in _all_names(term):
                for key in _keys(name, exactly=abbreviated):
                    self._names[key] = (name, term.name, abbreviated)
                atoms.append(_atoms(name, exactly=abbreviated))
        found = _trie(atoms) if atoms else "(?!)"
        self._pattern = re.compile(f"(?<![{_NAME_CHARS}])(?:{found})(?![{_NAME_CHARS}])")

    def _name(self, matched: str) -> tuple[str, str, bool]:
        """The vocabulary name the matched text is (less a plural "s"), its term, and whether it
        is an abbreviation."""
        for written in (matched, matched[:-1]):
            written = _hyphens(written)
            for key in (written, written[0].lower() + written[1:], "folded:" + written.casefold()):
                if key in self._names:
                    return self._names[key]
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

    def find(self, text: str) -> set[tuple[str, bool]]:
        """(term's name, found by an abbreviation) for each term found in `text`."""
        found = set()
        for match in self._pattern.finditer(text):
            name, term, abbreviated = self._name(match.group())
            if self._counts(match, name):
                found.add((term, abbreviated))
        return found
