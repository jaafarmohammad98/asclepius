"""The vocabulary file and its seed citations (spec G1, G2, G5)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hypothesis_engine.comentions import library_co_mentions
from hypothesis_engine.library import Library, write_build
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.vocabulary import Term, Vocabulary, VocabularyError, load_vocabulary, seed_problems

GOOD = """
ambiguous = ["AR"]

[[term]]
name = "AR"
kind = "gene"
aliases = ["androgen receptor"]
pmid = "11111111"

[[term]]
name = "anhedonia"
kind = "symptom"
pmid = "22222222"
"""


def _load(tmp_path: Path, text: str):
    path = tmp_path / "vocabulary.toml"
    path.write_text(text, encoding="utf-8")
    return load_vocabulary(path)


def test_a_vocabulary_file_loads_its_terms_and_ambiguous_names(tmp_path):
    vocabulary = _load(tmp_path, GOOD)
    assert vocabulary.terms == (
        Term("AR", "gene", ("androgen receptor",), "11111111"),
        Term("anhedonia", "symptom", (), "22222222"),
    )
    assert vocabulary.ambiguous == frozenset({"AR"})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.replace('pmid = "22222222"', ""), "'anhedonia' has no PMID"),
        (lambda t: t.replace('"22222222"', '"PMC123"'), "'anhedonia' has no PMID"),
        (lambda t: t.replace('kind = "symptom"', 'kind = "mood"'), "'anhedonia' has kind 'mood'"),
        (lambda t: t.replace('name = "anhedonia"', 'name = " "'), "term 2 has no name"),
        (lambda t: t.replace('"androgen receptor"', '"anhedonia"'), "'anhedonia' names two terms"),
        (lambda t: t.replace('["AR"]', '["CAT"]'), "ambiguous name 'CAT' is no term's name"),
        (lambda t: t.replace("[[term]]", "[[terms]]"), re.escape("no [[term]] entries")),
        (lambda t: t + "\n[[term", "Expected"),
    ],
)
def test_a_malformed_vocabulary_file_is_refused_with_the_reason(tmp_path, change, message):
    with pytest.raises(VocabularyError, match=message):
        _load(tmp_path, change(GOOD))


def test_the_message_names_the_file(tmp_path):
    with pytest.raises(VocabularyError, match=re.escape(str(tmp_path / "vocabulary.toml"))):
        _load(tmp_path, "")


def test_the_shipped_vocabulary_loads():
    vocabulary = load_vocabulary()
    assert {t.kind for t in vocabulary.terms} >= {"gene", "hormone", "neurosteroid", "symptom", "syndrome"}


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
    write_build(path, papers, {"100": [("PFS", 1)], "200": [("PSSD", 1)]}, "sha", "2026-11-23T00:00:00+00:00")
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
