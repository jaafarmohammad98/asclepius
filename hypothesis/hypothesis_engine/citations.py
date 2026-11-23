"""Citation verification (spec H1): no citation reaches the output unless its quote is in its paper.

Every mechanism claim and every supporting or contradicting entry carries a paper identifier
and a quoted passage. Each is checked against the library version the hypothesis was built
from: the paper must be in it, and the quote must appear word for word in the paper's title
or text (its full text when the version has it, else its abstract).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from hypothesis_engine.hypotheses import Citation, Claim, Failure, Hypothesis
from hypothesis_engine.library import Library

HYPHENS = "-­‐‑"  # hyphen-minus, soft hyphen, hyphen, non-breaking hyphen
_GAP = rf"[\s{re.escape(HYPHENS)}]*"  # what may lie between two characters of a quote


def verify_hypotheses(
    results: Iterable[Hypothesis | Failure], library: Library
) -> list[Hypothesis | Failure]:
    """Each result with its unverified citations removed; failures pass through unchanged."""
    return [verify(r, library) if isinstance(r, Hypothesis) else r for r in results]


def verify(hypothesis: Hypothesis, library: Library) -> Hypothesis | Failure:
    """The hypothesis keeping only the citations whose quote is found in their paper, and
    recording why each other one was removed (never its quote). Left with no mechanism claim,
    it is a `Failure`."""
    built_from = hypothesis.record.get("library_version")
    if not isinstance(built_from, int):
        raise ValueError("the hypothesis does not record the library version it was built from")
    version: int = built_from
    removed: list[str] = []

    def kept[C: (Claim, Citation)](cited: tuple[C, ...], what: str) -> tuple[C, ...]:
        verified = []
        for c in cited:
            problem = _problem(c, what, library, version)
            if problem:
                removed.append(problem)
            else:
                verified.append(c)
        return tuple(verified)

    mechanism = kept(hypothesis.mechanism, "a mechanism claim")
    supporting = kept(hypothesis.supporting, "a supporting entry")
    contradicting = kept(hypothesis.contradicting, "a contradicting entry")
    record = hypothesis.record | {"citations_removed": removed}
    if not mechanism:
        reason = "no mechanism claim is left once unverified citations are removed (spec H1)"
        return Failure(hypothesis.finding, (*removed, reason), record)
    return Hypothesis(hypothesis.finding, mechanism, supporting, contradicting, hypothesis.tests, record)


def _problem(cited: Claim | Citation, what: str, library: Library, version: int) -> str | None:
    """Why the citation (`what`: "a mechanism claim", ...) cannot be shown; None when it can."""
    if not cited.paper_id.strip():
        return f"{what} carries no paper identifier"
    if not cited.quote.strip():
        return f"{what} carries no quoted passage"
    paper = library.document(cited.paper_id, version)
    if paper is None:
        return f"{what} cites paper {cited.paper_id}, which the library does not hold"
    if not quoted(cited.quote, f"{paper.title}\n{paper.text}"):
        return f"{what}'s quote is not found word for word in paper {cited.paper_id}"
    return None


def quoted(quote: str, text: str) -> bool:
    """Whether `quote` appears word for word in `text`, from the start of a word to the end of one.

    Case, white space and hyphens do not count, so a word broken across a line ("reduc-\\ntase"),
    or hyphenated in one and not the other ("post-finasteride", "post finasteride"), still matches.
    """
    characters = [c for c in _normal(quote) if not c.isspace() and c not in HYPHENS]
    if not characters:
        return False
    pattern = _GAP.join(re.escape(c) for c in characters)
    if characters[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if characters[-1].isalnum():
        pattern += r"(?!\w)"
    return re.search(pattern, _normal(text)) is not None


def _normal(text: str) -> str:
    # NFKC: ligatures and other compatibility forms from PDF text ("ﬁ") read as their letters
    return unicodedata.normalize("NFKC", text).casefold()
