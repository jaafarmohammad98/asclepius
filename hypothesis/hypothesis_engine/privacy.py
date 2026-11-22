"""Private papers stay on the researcher's machine, except short quoted passages.

Everything the engine sends out (to an LLM, from M4-04 on) goes through one
`PrivateTextGuard`'s `check` first. The guard finds the passages a request shares with a
private paper's title or text: runs of 8 words or more in a row (a whole paper shorter than
that counts as one run), ignoring case, spacing and punctuation. Runs separated by at most 3
other words (a citation marker, an inserted "[...]") are one passage. It refuses the
request when

- one passage is longer than `max_quote_words`,
- all of them together are longer than `max_total_words`, or
- the words of one paper this guard has let through, counted once however often they are
  sent, would pass `max_paper_words`; use one guard for one run of the engine.

Otherwise it returns the passages, so the researcher can be shown exactly what of their
private papers is being sent. It is a check against private text slipping into a request by
mistake, not against text disguised on purpose (a word changed every 7 words is not caught).

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
GAP = 3  # runs this many words apart or fewer are one passage
MAX_QUOTE_WORDS = 60  # about two sentences
MAX_TOTAL_WORDS = 200
MAX_PAPER_WORDS = 400

_WORD = re.compile(r"\w+")


class PrivateTextLeak(Exception):
    """A request holds more private-paper text than short quotes."""


@dataclass(frozen=True)
class Quote:
    """A run of words a request shares with private papers, for the researcher to see."""

    text: str  # as it stands in the request
    words: int
    paper_ids: set[str]


class PrivateTextGuard:
    """Checks outgoing text against the private papers' titles and text."""

    def __init__(
        self,
        papers: Iterable[PrivatePaper],
        max_quote_words: int = MAX_QUOTE_WORDS,
        max_total_words: int = MAX_TOTAL_WORDS,
        max_paper_words: int = MAX_PAPER_WORDS,
    ) -> None:
        # each run of words -> the (paper id, position in the paper) it starts at
        self._runs: dict[tuple[str, ...], list[tuple[str, int]]] = {}
        for paper in papers:
            words = [w.casefold() for w in _WORD.findall(f"{paper.title}\n{paper.text}")]
            size = min(RUN, len(words))
            for start in range(len(words) - size + 1):
                self._runs.setdefault(tuple(words[start : start + size]), []).append((paper.id, start))
        self._sizes = sorted({len(run) for run in self._runs})
        self._sent: dict[str, set[int]] = {}  # paper id -> positions of its words let through
        self.max_quote_words = max_quote_words
        self.max_total_words = max_total_words
        self.max_paper_words = max_paper_words

    @classmethod
    def from_library(cls, library: Library) -> PrivateTextGuard:
        """A guard for the private papers of the library's current version."""
        return cls(library.private_papers())

    def check(self, outgoing: str) -> list[Quote]:
        """The private passages in `outgoing`, in order; raises PrivateTextLeak when they are too long.

        A request it refuses is not counted as sent.
        """
        matches = list(_WORD.finditer(outgoing))
        words = [m.group().casefold() for m in matches]
        owners: list[set[str]] = [set() for _ in words]
        positions: dict[str, set[int]] = {}  # paper id -> positions of its words in this request
        for size in self._sizes:
            for start in range(len(words) - size + 1):
                for paper_id, at in self._runs.get(tuple(words[start : start + size]), ()):
                    for k in range(size):
                        owners[start + k].add(paper_id)
                    positions.setdefault(paper_id, set()).update(range(at, at + size))

        quotes = _passages(outgoing, matches, owners)
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
        sent = {paper_id: self._sent.get(paper_id, set()) | new for paper_id, new in positions.items()}
        for paper_id, all_sent in sent.items():
            if len(all_sent) > self.max_paper_words:
                raise PrivateTextLeak(
                    f"{len(all_sent)} words of {paper_id} in all; "
                    f"at most {self.max_paper_words} words of {paper_id} may be sent"
                )
        self._sent.update(sent)
        return quotes


def _passages(outgoing: str, matches: list[re.Match[str]], owners: list[set[str]]) -> list[Quote]:
    """Group the request's private words into passages; runs at most GAP words apart are one."""
    covered = [i for i, ids in enumerate(owners) if ids]
    groups: list[list[int]] = []
    for i in covered:
        if groups and i - groups[-1][-1] <= GAP + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [
        Quote(
            outgoing[matches[group[0]].start() : matches[group[-1]].end()],
            len(group),
            set().union(*(owners[i] for i in group)),
        )
        for group in groups
    ]
