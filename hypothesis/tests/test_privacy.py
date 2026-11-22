"""Private papers leave the machine only as short quoted passages the researcher can see."""

from __future__ import annotations

import pytest

from hypothesis_engine.privacy import PrivateTextGuard, PrivateTextLeak
from hypothesis_engine.private import PrivatePaper

SENTENCES = [
    "Cerebrospinal fluid allopregnanolone was lower in all twelve men who had taken finasteride.",
    "Pregnenolone and progesterone did not differ from the controls in any of the samples.",
    "Three men reported that their symptoms began within a week of the first tablet.",
    "Dihydrotestosterone in plasma had returned to the normal range in every participant by then.",
    "We therefore propose that the deficit lies in the brain rather than in the blood.",
    "The cohort is small and was recruited from a single patient forum, which limits it.",
]
PAPER = PrivatePaper("private-0123456789abcdef", "cohort.pdf", "Cohort", "\n".join(SENTENCES))
GUARD = PrivateTextGuard([PAPER], max_quote_words=40, max_total_words=60)


def test_text_with_nothing_private_passes_and_quotes_nothing():
    assert GUARD.check("Finding: lower DHT in the PSSD group; effect size 0.4 (moderate).") == []


def test_a_short_quote_passes_and_is_returned_for_the_researcher_to_see():
    request = f'Supporting passage from {PAPER.id}: "{SENTENCES[0]}" What mechanism could explain it?'
    (quote,) = GUARD.check(request)
    assert quote.paper_ids == {PAPER.id}
    assert quote.text == SENTENCES[0].rstrip(".")


def test_spacing_and_case_do_not_hide_a_quote():
    (quote,) = GUARD.check(SENTENCES[1].upper().replace(" ", "\n  "))
    assert quote.paper_ids == {PAPER.id}


def test_a_long_run_of_private_text_is_refused():
    with pytest.raises(PrivateTextLeak, match="40 words"):
        GUARD.check("Context: " + " ".join(SENTENCES[:4]))


def test_many_short_quotes_that_add_up_to_too_much_are_refused():
    # each under the limit, together over it
    request = " Then, in another place entirely: ".join(f'"{s}"' for s in SENTENCES)
    with pytest.raises(PrivateTextLeak, match="60 words"):
        GUARD.check(request)


def test_a_guard_without_private_papers_lets_everything_through():
    assert PrivateTextGuard([]).check(" ".join(SENTENCES)) == []


def test_a_private_title_is_guarded_too():
    paper = PrivatePaper(
        "private-1", "a.pdf", "Allopregnanolone stayed low for eighteen months in one man", "Body."
    )
    (quote,) = PrivateTextGuard([paper]).check(
        "Title: Allopregnanolone stayed low for eighteen months in one man"
    )
    assert quote.paper_ids == {"private-1"}


def test_a_private_paper_shorter_than_a_run_is_guarded_whole():
    short = PrivatePaper("private-2", "b.pdf", "Note", "Dose halved, symptoms gone.")
    (quote,) = PrivateTextGuard([short]).check("They wrote: note. Dose halved, symptoms gone.")
    assert quote.paper_ids == {"private-2"}


def test_a_long_passage_broken_by_inserted_words_is_still_one_passage():
    words = " ".join(SENTENCES[:4]).split()
    # a citation marker every 30 words: each piece is under the limit, the passage is not
    broken = " ".join(w + (" [12]" if i % 30 == 29 else "") for i, w in enumerate(words))
    with pytest.raises(PrivateTextLeak, match="40 words"):
        GUARD.check(broken)


def test_one_guard_counts_what_has_been_sent_of_a_paper_across_requests():
    guard = PrivateTextGuard([PAPER], max_quote_words=40, max_total_words=60, max_paper_words=50)
    guard.check(f'"{SENTENCES[0]}" and "{SENTENCES[1]}"')
    guard.check(f'"{SENTENCES[0]}"')  # sent before: counts once
    with pytest.raises(PrivateTextLeak, match="50 words of private-0123456789abcdef"):
        guard.check(f'"{SENTENCES[2]}" and "{SENTENCES[3]}"')
    guard.check(f'"{SENTENCES[1]}"')  # a refused request was not counted as sent
