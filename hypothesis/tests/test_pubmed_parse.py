"""Reading PubMed efetch XML into paper records. The fixture is four real records, recorded 2026-11-22."""

from pathlib import Path

from hypothesis_engine.pubmed import parse_pubmed_xml

FIXTURES = Path(__file__).parent / "fixtures"


def _papers():
    return {p.pmid: p for p in parse_pubmed_xml((FIXTURES / "efetch_mixed.xml").read_bytes())}


def test_every_record_in_the_response_is_read():
    assert sorted(_papers()) == ["30422511", "42492846", "42628582", "43067474"]


def test_structured_abstract_keeps_its_section_labels():
    paper = _papers()["42628582"]
    assert paper.title.startswith("No Evidence of Post-finasteride Syndrome in 267,983 Person-years")
    assert paper.abstract is not None
    assert paper.abstract.startswith("OBJECTIVE: To investigate Post-Finasteride Syndrome")
    assert "\nMETHODS: Records of subjects enrolled in the PCPT were linked to Medi" in paper.abstract
    assert "\nCONCLUSION: In the largest placebo-controlled study" in paper.abstract


def test_identifiers_journal_and_year():
    paper = _papers()["42628582"]
    assert paper.doi == "10.1016/j.urology.2026.08.020"
    assert paper.pmcid == "PMC13625805"
    assert paper.journal == "Urology"
    assert paper.year == 2026


def test_unstructured_abstract_is_plain_text():
    paper = _papers()["42492846"]
    assert paper.abstract is not None
    assert paper.abstract.startswith("Steroid 5α-reductases (5αRs) catalyze the irreversible, NADP")
    assert ":" not in paper.abstract.split(" ")[0]


def test_record_without_an_abstract_is_marked_as_lacking_one():
    paper = _papers()["43067474"]
    assert paper.abstract is None
    assert paper.title.startswith("Critique of finasteride safety perspectives")
    assert paper.doi == "10.1016/j.ad.2026.104794"
    assert paper.pmcid is None


def test_book_chapter_is_read_with_the_book_as_its_journal():
    paper = _papers()["30422511"]
    assert paper.title == "Vitamin A Toxicity"
    assert paper.journal == "StatPearls"
    assert paper.year == 2026
    assert paper.abstract is not None
    assert "(eg, β-carotene)" in paper.abstract
    assert paper.doi is None
