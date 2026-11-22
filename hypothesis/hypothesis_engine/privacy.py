"""Private papers stay on the researcher's machine, except short quoted passages.

Everything the engine sends out (to an LLM, from M4-04 on) goes through
`PrivateTextGuard.check` first. It finds the runs of words a request shares with a private
paper (8 words or more in a row, ignoring case, spacing and punctuation) and refuses the
request when one run is longer than `max_quote_words` or all of them together are longer
than `max_total_words`. Otherwise it returns those passages, so the researcher can be shown
exactly what of their private papers is being sent.

Requests to PubMed and Europe PMC carry PMIDs only, and private papers have none, so those
requests cannot hold private text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from hypothesis_engine.library import Library
from hypothesis_engine.private import PrivatePaper

RUN = 8  # shorter shared runs are ordinary phrases, not quotes
MAX_QUOTE_WORDS = 60  # about two sentences
MAX_TOTAL_WORDS = 200

_WORD = re.compile(r"\w+")


class PrivateTextLeak(Exception):
    """A request holds more private-paper text than short quotes."""


@dataclass(frozen=True)
class Quote:
    text: str  # as it stands in the request
    words: int
    paper_ids: set[str]


class PrivateTextGuard:
    def __init__(
        self,
        papers: Iterable[PrivatePaper],
        max_quote_words: int = MAX_QUOTE_WORDS,
        max_total_words: int = MAX_TOTAL_WORDS,
    ) -> None:
        self._runs: dict[tuple[str, ...], set[str]] = {}
        for paper in papers:
            words = [w.casefold() for w in _WORD.findall(paper.text)]
            for start in range(len(words) - RUN + 1):
                self._runs.setdefault(tuple(words[start : start + RUN]), set()).add(paper.id)
        self.max_quote_words = max_quote_words
        self.max_total_words = max_total_words

    @classmethod
    def from_library(cls, library: Library) -> PrivateTextGuard:
        """A guard for the private papers of the library's current version."""
        return cls(library.private_papers())

    def check(self, outgoing: str) -> list[Quote]:
        """The private passages in `outgoing`, in order; raises PrivateTextLeak when they are too long."""
        if not self._runs:
            return []
        matches = list(_WORD.finditer(outgoing))
        words = [m.group().casefold() for m in matches]
        owners: list[set[str]] = [set() for _ in words]
        for start in range(len(words) - RUN + 1):
            found = self._runs.get(tuple(words[start : start + RUN]))
            if found:
                for i in range(start, start + RUN):
                    owners[i] |= found

        quotes = []
        i = 0
        while i < len(words):
            if not owners[i]:
                i += 1
                continue
            end = i
            ids: set[str] = set()
            while end < len(words) and owners[end]:
                ids |= owners[end]
                end += 1
            text = outgoing[matches[i].start() : matches[end - 1].end()]
            quotes.append(Quote(text, end - i, ids))
            i = end

        longest = max((q.words for q in quotes), default=0)
        if longest > self.max_quote_words:
            raise PrivateTextLeak(
                f"a passage of {longest} words from a private paper; at most {self.max_quote_words} words may be quoted"
            )
        total = sum(q.words for q in quotes)
        if total > self.max_total_words:
            raise PrivateTextLeak(
                f"{total} words from private papers; at most {self.max_total_words} words may be sent in one request"
            )
        return quotes
