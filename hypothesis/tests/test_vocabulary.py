"""The vocabulary file (spec G1, G2, G5)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hypothesis_engine.vocabulary import Term, VocabularyError, load_vocabulary

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


def test_abbreviations_load(tmp_path):
    text = GOOD.replace('name = "anhedonia"', 'name = "post-SSRI sexual dysfunction"').replace(
        'kind = "symptom"', 'kind = "syndrome"\nabbreviations = ["PSSD"]'
    )
    (_, syndrome) = _load(tmp_path, text).terms
    assert syndrome.abbreviations == ("PSSD",)


@pytest.mark.parametrize("alias", ["Anhedonia", "ANHEDONIA"])
def test_a_name_matching_another_term_s_name_with_case_aside_is_refused(tmp_path, alias):
    with pytest.raises(VocabularyError, match=f"'{alias}' and 'anhedonia' name two terms"):
        _load(
            tmp_path,
            GOOD.replace('aliases = ["androgen receptor"]', f'aliases = ["androgen receptor", "{alias}"]'),
        )


def test_symbols_differing_in_case_are_different_names(tmp_path):
    text = GOOD.replace('aliases = ["androgen receptor"]', 'aliases = ["androgen receptor", "Ar"]')
    assert _load(tmp_path, text).terms[0].aliases == ("androgen receptor", "Ar")
