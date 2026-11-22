"""Papers split into passages: the pieces the search index returns and hypotheses quote.

A passage is a slice of the paper's text, character for character, so a quote taken from it
is found word for word in the paper (spec H1). Each line of the text (a paragraph of the
full text, a section of a structured abstract, the title the full text starts with) starts
a piece; a piece over `MAX_WORDS` is cut at sentence ends (or, for one very long sentence,
between words), and a piece under `MIN_WORDS` (a heading, a one-line paragraph) joins the
piece after it, or the one before it when it is last.

A passage's id is the paper's id, a colon and its number from 1 (`32317131:4`). The same
text always gives the same passages, so ids stay the same until the paper's text changes
(its abstract is corrected, or its full text is downloaded).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hypothesis_engine.library import Document

MIN_WORDS = 25
MAX_WORDS = 200

_LINE = re.compile(r"[^\n]*\S[^\n]*")
_SENTENCE_END = re.compile(r"[.!?]+(?=\s)")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Passage:
    id: str  # "<paper id>:<n>"
    paper_id: str  # a PMID, or a private paper's id
    private: bool
    title: str  # the paper's
    text: str


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    words: int


def _span(text: str, start: int, end: int) -> _Span:
    while text[start].isspace():
        start += 1
    while text[end - 1].isspace():
        end -= 1
    return _Span(start, end, len(text[start:end].split()))


def _sentences(text: str, line: _Span) -> list[_Span]:
    cuts = [line.start] + [m.end() for m in _SENTENCE_END.finditer(text, line.start, line.end)]
    if cuts[-1] < line.end:
        cuts.append(line.end)
    return [_span(text, a, b) for a, b in zip(cuts, cuts[1:], strict=False) if text[a:b].strip()]


def _cut_between_words(text: str, sentence: _Span) -> list[_Span]:
    words = list(_WORD.finditer(text, sentence.start, sentence.end))
    chunks = [words[i : i + MAX_WORDS] for i in range(0, len(words), MAX_WORDS)]
    return [_Span(c[0].start(), c[-1].end(), len(c)) for c in chunks]


def _pieces(text: str, line: _Span) -> list[_Span]:
    """The line as one piece, or cut into pieces of at most MAX_WORDS at sentence ends."""
    if line.words <= MAX_WORDS:
        return [line]
    units = [
        unit
        for sentence in _sentences(text, line)
        for unit in (_cut_between_words(text, sentence) if sentence.words > MAX_WORDS else [sentence])
    ]
    pieces = [units[0]]
    for unit in units[1:]:
        last = pieces[-1]
        if last.words + unit.words <= MAX_WORDS:
            pieces[-1] = _join(last, unit)
        else:
            pieces.append(unit)
    return pieces


def _join(a: _Span, b: _Span) -> _Span:
    return _Span(a.start, b.end, a.words + b.words)


def split(document: Document) -> list[Passage]:
    """The document's passages, in order; a document with no text is one passage, its title."""
    text = document.text
    lines = [_span(text, m.start(), m.end()) for m in _LINE.finditer(text)]
    spans: list[_Span] = []
    for piece in (p for line in lines for p in _pieces(text, line)):
        if spans and spans[-1].words < MIN_WORDS and spans[-1].words + piece.words <= MAX_WORDS:
            spans[-1] = _join(spans[-1], piece)
        else:
            spans.append(piece)
    if len(spans) > 1 and spans[-1].words < MIN_WORDS and spans[-2].words + spans[-1].words <= MAX_WORDS:
        spans[-2:] = [_join(spans[-2], spans[-1])]
    texts = [text[s.start : s.end] for s in spans] or [document.title]
    return [
        Passage(f"{document.id}:{n}", document.id, document.private, document.title, passage)
        for n, passage in enumerate(texts, start=1)
    ]
