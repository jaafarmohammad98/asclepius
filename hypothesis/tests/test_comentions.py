"""Co-mention edges between vocabulary terms in library passages (spec G2, G3, G5)."""

from __future__ import annotations

import pytest

from hypothesis_engine.comentions import co_mentions, library_co_mentions, seed_problems
from hypothesis_engine.library import Library, write_build
from hypothesis_engine.passages import Passage
from hypothesis_engine.private import PrivatePaper
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.vocabulary import Term, Vocabulary

VOCABULARY = Vocabulary(
    terms=(
        Term("AR", "gene", ("androgen receptor",), "11111111"),
        Term("BDNF", "gene", (), "11111111"),
        Term("SRD5A2", "gene", ("5α-reductase type 2",), "11111111"),
        Term("allopregnanolone", "neurosteroid", (), "11111111"),
        Term("anhedonia", "symptom", (), "11111111"),
    ),
    ambiguous=frozenset({"AR"}),
)


def _passage(passage_id: str, text: str) -> Passage:
    paper_id = passage_id.split(":")[0]
    return Passage(passage_id, paper_id, False, f"Paper {paper_id}", text)


def test_an_edge_joins_two_terms_in_one_passage_weighted_by_distinct_papers():
    passages = [
        _passage("1:1", "BDNF fell as allopregnanolone fell."),
        _passage("1:2", "Again BDNF and allopregnanolone."),
        _passage("2:4", "Serum BDNF tracked allopregnanolone levels."),
        _passage("3:1", "BDNF alone."),
    ]
    (edge,) = co_mentions(passages, VOCABULARY).edges
    assert (edge.a, edge.b) == ("BDNF", "allopregnanolone")
    assert edge.papers == 2
    assert edge.passage_ids == ("1:1", "1:2", "2:4")
    assert edge.label == "co-mentioned in 2 papers"


def _nodes(*texts: str) -> set[str]:
    passages = [_passage(f"{n}:1", text) for n, text in enumerate(texts, start=1)]
    return set(co_mentions(passages, VOCABULARY).nodes)


def test_an_ambiguous_symbol_without_a_gene_word_is_no_node():
    assert _nodes("Patients tried AR (augmented reality) headsets for anhedonia.") == {"anhedonia"}


def test_an_ambiguous_symbol_with_a_gene_word_nearby_is_a_node():
    assert _nodes("AR gene expression was lower.") == {"AR"}
    assert _nodes("We genotyped the CAG repeat polymorphism of AR in 40 men.") == {"AR"}


def test_a_gene_word_too_far_away_does_not_count():
    assert _nodes("AR headsets were worn by one two three four five people with this gene.") == set()


def test_an_unambiguous_alias_of_an_ambiguous_symbol_needs_no_gene_word():
    assert _nodes("The androgen receptor was studied.") == {"AR"}


def test_symbols_match_case_sensitively_and_names_with_case_aside():
    assert _nodes("Bdnf and srd5a2 in mice.") == set()
    assert _nodes("Allopregnanolone and Anhedonias.") == {"allopregnanolone", "anhedonia"}


def test_a_name_inside_a_longer_word_or_symbol_does_not_count():
    assert _nodes("BDNFX, proBDNF and SRD5A21 were measured.") == set()


def test_an_alias_is_counted_as_its_term():
    assert _nodes("Levels of 5α-reductase type 2 fell.") == {"SRD5A2"}


def test_a_name_matched_with_case_aside_is_found_whatever_letter_form_the_text_uses():
    vocabulary = Vocabulary((Term("OPRM1", "gene", ("μ-opioid receptor",), "1"),))
    # the micro sign (U+00B5) where the name has the Greek letter mu (U+03BC)
    assert vocabulary.find("The µ-opioid receptors and the Μ-OPIOID RECEPTOR.") == {"OPRM1"}


def test_a_name_with_capitals_that_starts_in_lower_case_may_start_a_sentence():
    vocabulary = Vocabulary((Term("post-SSRI sexual dysfunction", "syndrome", (), "1"),))
    assert vocabulary.find("Post-SSRI sexual dysfunction persists.") == {"post-SSRI sexual dysfunction"}
    assert vocabulary.find("post-ssri sexual dysfunction") == set()


def test_where_names_overlap_the_longer_one_counts():
    vocabulary = Vocabulary(
        (
            Term("GABA", "hormone", (), "1"),
            Term("GABA-A receptor", "receptor", (), "1"),
            Term("estrogen receptor", "receptor", (), "1"),
            Term("ESR1", "gene", ("estrogen receptor alpha",), "1"),
        )
    )
    assert vocabulary.find("GABA-A receptors and estrogen receptor alpha.") == {"GABA-A receptor", "ESR1"}
    assert vocabulary.find("GABA and estrogen receptors.") == {"GABA", "estrogen receptor"}


def test_a_vocabulary_of_only_symbols_finds_no_stray_letters():
    assert Vocabulary((Term("BDNF", "gene", (), "1"),)).find("a s of BDNF") == {"BDNF"}
    assert Vocabulary((Term("anhedonia", "symptom", (), "1"),)).find("A s B") == set()


def test_a_plural_symbol_counts_and_is_still_ambiguous():
    assert _nodes("BDNFs rose; ARs (augmented realities) were used.") == {"BDNF"}


def test_a_plural_in_capitals_counts():
    assert _nodes("ANHEDONIAS AND ALLOPREGNANOLONE.") == {"anhedonia", "allopregnanolone"}


def test_a_symbol_followed_by_a_greek_letter_is_another_name():
    assert _nodes("BDNFα and BDNF-β were measured.") == {"BDNF"}  # the second, after a hyphen, counts
    assert _nodes("BDNFα was measured.") == set()


SEEDED = Vocabulary(
    terms=(
        Term("AR", "gene", ("androgen receptor",), "100"),
        Term("BDNF", "gene", (), "100"),
        Term("SET", "gene", (), "100"),
        Term("anhedonia", "symptom", (), "200"),
        Term("tinnitus", "symptom", (), "999"),
    ),
    ambiguous=frozenset({"SET"}),
)


@pytest.fixture
def seeded_library(tmp_path):
    path = tmp_path / "library.sqlite"
    papers = [
        Paper("100", "A study", "In a set of men, androgen receptor and BDNF fell.", None, None, None, 2020),
        Paper(
            "200",
            "Emotional blunting",
            "Anhedonia persisted after the drug was stopped.",
            None,
            None,
            None,
            2021,
        ),
    ]
    # a private PDF naming both: it has no PMID, so it is not counted (spec G5)
    private = [PrivatePaper("private-0123456789abcdef", "cohort.pdf", "Cohort", "Anhedonia and BDNF.")]
    found_by = {"100": [("PFS", 1)], "200": [("PSSD", 1)]}
    write_build(path, papers, found_by, "sha", "2026-11-23T00:00:00+00:00", private)
    with Library.open(path) as library:
        yield library


def test_a_seed_is_verified_when_its_paper_is_in_the_library_and_names_the_term(seeded_library):
    problems = seed_problems(SEEDED, seeded_library)
    assert set(problems) == {"SET", "tinnitus"}
    assert problems["tinnitus"] == "its paper 999 is not in the library"
    assert problems["SET"] == "it is not found in its paper 100"


def test_the_library_s_co_mentions_leave_out_terms_whose_seed_fails(seeded_library):
    found = library_co_mentions(seeded_library, SEEDED)
    assert set(found.nodes) == {"AR", "BDNF", "anhedonia"}
    (edge,) = found.edges
    assert (edge.a, edge.b, edge.passage_ids) == ("AR", "BDNF", ("100:1",))
    assert found.left_out == {
        "SET": "it is not found in its paper 100",
        "tinnitus": "its paper 999 is not in the library",
    }


def test_a_private_paper_s_passages_make_no_node_or_edge(seeded_library):
    found = library_co_mentions(seeded_library, SEEDED)
    assert found.nodes["anhedonia"].passage_ids == ("200:1",)
    assert found.nodes["BDNF"].passage_ids == ("100:1",)
