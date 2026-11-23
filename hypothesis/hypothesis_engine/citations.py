"""Citation checks (spec H1): a citation is shown only if its quote is found in its paper.

Every mechanism claim and every supporting or contradicting entry carries a paper identifier
and a quoted passage. `problem` checks one against the library version the hypothesis is
built from: the paper must be in it, and the quote must appear word for word (`quoted`) in
the paper's title or in its text (its full text when the version has it, else its abstract).
`hypotheses.py` sends citations that fail back to the LLM, then removes those still failing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from hypothesis_engine.library import Library

HYPHENS = "-\u00ad\u2010\u2011"  # hyphen-minus, soft hyphen, hyphen, non-breaking hyphen
_HYPHENS = re.escape(HYPHENS)
_IN_WORD = rf"(?:[{_HYPHENS}]\s*)?"  # within a quoted word: nothing, a hyphen, or a line-break hyphen
_BETWEEN_WORDS = rf"[\s{_HYPHENS}]+"  # where the quote has a space: a space or a hyphen
_ANY_GAP = rf"[\s{_HYPHENS}]*"  # where the quote has a hyphen
_PIECE = re.compile(rf"[^\s{_HYPHENS}]+|[\s{_HYPHENS}]+")


class Cited(Protocol):
    """A mechanism claim or a supporting or contradicting entry."""

    @property
    def passage_id(self) -> str: ...
    @property
    def paper_id(self) -> str: ...
    @property
    def quote(self) -> str: ...


@dataclass(frozen=True)
class Problem:
    """A citation that cannot be shown, and why."""

    what: str  # "a mechanism claim", "a supporting entry", "a contradicting entry"
    passage_id: str
    quote: str  # sent back to the LLM that wrote it; never written to the output
    why: str  # follows `what`: "cites paper 99999, which the library does not hold"

    def __str__(self) -> str:
        """For the output: what failed and why, without the quote."""
        return f"{self.what} citing passage {self.passage_id} {self.why}"


def problem(cited: Cited, what: str, library: Library, version: int) -> Problem | None:
    """Why the citation cannot be shown; None when it can."""

    def failed(why: str) -> Problem:
        return Problem(what, cited.passage_id, cited.quote, why)

    if not cited.paper_id.strip():
        return failed("carries no paper identifier")
    if not cited.quote.strip():
        return failed("carries no quoted passage")
    paper = library.document(cited.paper_id, version)
    if paper is None:
        return failed(f"cites paper {cited.paper_id}, which the library does not hold")
    if not (quoted(cited.quote, paper.title) or quoted(cited.quote, paper.text)):
        return failed(f"quotes words not found word for word in paper {cited.paper_id}")
    return None


def quoted(quote: str, text: str) -> bool:
    """Whether `quote` appears word for word in `text`, from the start of a word to the end of one.

    Case, white space and hyphenation do not count: a word broken across a line ("reduc-\\ntase")
    or hyphenated in one and not the other ("post-finasteride", "post finasteride",
    "postfinasteride") still matches. Words split or joined differently ("in creased") do not.
    """
    # runs of word characters, and the gaps (white space, hyphens) between them
    pieces = re.findall(_PIECE, _folded(quote))
    while pieces and _is_gap(pieces[0]):
        pieces.pop(0)
    while pieces and _is_gap(pieces[-1]):
        pieces.pop()
    if not pieces:
        return False
    pattern = ""
    for piece in pieces:
        if not _is_gap(piece):
            # inside a word, the paper may hyphenate where the quote does not, or break the line there
            pattern += _IN_WORD.join(re.escape(c) for c in piece)
        elif any(c in HYPHENS for c in piece):
            pattern += _ANY_GAP  # hyphenated in the quote: the paper may join, hyphenate or space it
        else:
            pattern += _BETWEEN_WORDS
    if pieces[0][0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if pieces[-1][-1].isalnum():
        pattern += r"(?!\w)"
    return re.search(pattern, _folded(text)) is not None


def _is_gap(piece: str) -> bool:
    return piece[0].isspace() or piece[0] in HYPHENS


def _folded(text: str) -> str:
    """Case aside, and NFKC: ligatures and other compatibility forms of PDF text ("ﬁ") as letters."""
    return unicodedata.normalize("NFKC", text).casefold()
