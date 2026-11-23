"""Papers split into passages: the pieces search returns and hypotheses quote."""

from __future__ import annotations

from hypothesis_engine.library import Document
from hypothesis_engine.passages import MAX_WORDS, MIN_WORDS, split


def _sentence(n: int, words: int = 12) -> str:
    return " ".join(f"w{n}x{i}" for i in range(words - 1)) + f" end{n}."


def _paragraph(start: int, sentences: int) -> str:
    return " ".join(_sentence(start + i) for i in range(sentences))


FULL_TEXT = "\n\n".join(
    [
        "Allopregnanolone after finasteride",  # the title, as the full text starts with it
        _paragraph(0, 4),  # the abstract, 48 words
        "Introduction",
        _paragraph(10, 3),
        _paragraph(20, 30),  # 360 words: longer than a passage may be
        "Methods",
        "Twelve men took part.",  # short, and last
    ]
)
PAPER = Document("32317131", "Allopregnanolone after finasteride", FULL_TEXT, private=False)


def test_every_word_of_the_paper_is_in_one_passage_in_order():
    passages = split(PAPER)
    assert " ".join(p.text for p in passages).split() == FULL_TEXT.split()


def test_passage_ids_name_the_paper_and_stay_the_same_for_the_same_text():
    passages = split(PAPER)
    assert [p.id for p in passages] == [f"32317131:{n}" for n in range(1, len(passages) + 1)]
    assert split(PAPER) == passages
    assert {p.paper_id for p in passages} == {"32317131"}


def test_headings_and_short_paragraphs_join_a_neighbour_and_long_ones_are_cut_at_sentences():
    passages = split(PAPER)
    assert all(MIN_WORDS <= len(p.text.split()) <= MAX_WORDS for p in passages)
    assert passages[0].text.startswith("Allopregnanolone after finasteride\n\nw0x0")
    assert any(p.text.startswith("Introduction\n\nw10x0") for p in passages)
    assert passages[-1].text.endswith("Methods\n\nTwelve men took part.")
    assert all(p.text.endswith(".") for p in passages)  # long paragraphs end at a sentence


def test_a_paragraph_is_not_cut_when_it_fits():
    passages = split(PAPER)
    assert any(p.text == f"Introduction\n\n{_paragraph(10, 3)}" for p in passages)


def test_a_structured_abstract_keeps_its_sections_one_per_line():
    abstract = "\n".join(f"{label}: {_paragraph(n * 10, 3)}" for n, label in enumerate(["AIMS", "RESULTS"]))
    passages = split(Document("1", "A study", abstract, private=False))
    assert [p.text for p in passages] == abstract.split("\n")


def test_a_paper_with_no_text_is_one_passage_holding_its_title():
    (passage,) = split(Document("19439893", "Finasteride-induced gynecomastia.", "", private=False))
    assert (passage.id, passage.text) == ("19439893:1", "Finasteride-induced gynecomastia.")


def test_a_short_paper_is_one_passage():
    (passage,) = split(Document("2", "Note", "Only a few words here.", private=False))
    assert passage.text == "Only a few words here."


def test_a_private_paper_s_passages_are_marked_private():
    paper = Document("private-0123456789abcdef", "Cohort", _paragraph(0, 3), private=True)
    assert all(p.private for p in split(paper))
    assert split(paper)[0].title == "Cohort"


def test_a_sentence_longer_than_a_passage_is_cut_between_words():
    paper = Document("3", "Run-on", " ".join(f"w{i}" for i in range(MAX_WORDS * 2 + 10)), private=False)
    passages = split(paper)
    assert all(len(p.text.split()) <= MAX_WORDS for p in passages)
    assert " ".join(p.text for p in passages).split() == paper.text.split()
