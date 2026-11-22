"""Reading an open-access copy's JATS XML into the text later quote checks search (spec H1).

The fixture is a real PubMed Central efetch answer holding PMC11450419 and PMC7253896
(CC BY-NC-ND), recorded 2026-11-22.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from fulltext_fixture import FOLDER

from hypothesis_engine.jats import articles, jats_text, licence_in

ANSWER = (FOLDER / "efetch-694139824aea4225.xml").read_bytes()
ARTICLE = articles(ANSWER)["PMC7253896"]


def _text(xml: bytes) -> str | None:
    return jats_text(ET.fromstring(xml))


def _paragraphs() -> list[str]:
    text = jats_text(ARTICLE)
    assert text is not None
    return text.split("\n\n")


def test_every_article_in_an_answer_is_found_by_its_pmcid():
    assert sorted(articles(ANSWER)) == ["PMC11450419", "PMC7253896"]
    assert articles(b"<pmc-articleset/>") == {}


def test_the_text_starts_with_the_title_then_the_abstract():
    paragraphs = _paragraphs()
    assert paragraphs[0] == "Post-finasteride syndrome"  # its footnote markers are dropped
    assert any(
        p.startswith("Finasteride is a 5α-reductase enzyme inhibitor that has been approved")
        for p in paragraphs
    )


def test_body_sections_keep_their_titles_and_paragraphs_in_reading_order():
    paragraphs = _paragraphs()
    introduction = paragraphs.index("Introduction")
    assert paragraphs[introduction + 1].startswith(
        "Finasteride is an inhibitor of the enzyme 5α-reductase types 1 and 2"
    )
    assert paragraphs.index("Finasteride and infertility") > introduction


def test_a_sentence_reads_word_for_word_across_inline_markup():
    text = jats_text(ARTICLE)
    assert text is not None
    # the source has <sup> and <xref> inside this sentence; the citation number stays
    assert (
        "With a short half-life ranging from 4.7 to 7.1 h,1 it is able to significantly reduce serum" in text
    )


def test_the_reference_list_and_footnotes_are_left_out():
    text = jats_text(ARTICLE)
    assert text is not None
    assert "Finasteride – an update and review" not in text  # reference 1
    assert "How to cite this article" not in text  # a footnote
    assert "References" not in text.split("\n\n")


def test_reference_and_footnote_sections_inside_the_body_are_left_out():
    # Europe PMC's copies put them in <body>, as sections of their own
    xml = b"""<article><body><sec><title>Methods</title><p>We asked.</p></sec>
        <sec sec-type="ref-list"><title>References</title><ref-list><ref>Smith 2010</ref></ref-list></sec>
        <sec sec-type="fn-group"><title>Footnotes</title><fn-group><fn><p>How to cite</p></fn></fn-group></sec>
        </body></article>"""
    assert _text(xml) == "Methods\n\nWe asked."


def test_a_list_inside_a_paragraph_follows_it_once():
    xml = b"""<article><body><sec><title>Results</title>
        <p>Three groups were seen:<list><list-item><p>first</p></list-item><list-item><p>second</p></list-item></list></p>
        </sec></body></article>"""
    assert _text(xml) == "Results\n\nThree groups were seen:\n\nfirst\n\nsecond"


def test_tables_and_formulas_are_left_out_but_captions_kept():
    xml = b"""<article xmlns:mml="http://www.w3.org/1998/Math/MathML"><body>
        <p>The ratio <mml:math><mml:mi>x</mml:mi></mml:math> rose.</p>
        <table-wrap><label>Table 1</label><caption><title>Hormone levels</title></caption>
        <table><tr><td>12.5</td></tr></table></table-wrap></body></article>"""
    assert _text(xml) == "The ratio rose.\n\nHormone levels"


def test_a_copy_without_body_text_has_no_text():
    assert _text(b"<article><front><article-meta/></front></article>") is None
    assert _text(b"<article><body><sec><title>Only a heading</title></sec></body></article>") is None
    assert _text(b"<article><body/></article>") is None


def test_an_unreadable_answer_raises():
    with pytest.raises(ET.ParseError):
        articles(b"<pmc-articleset><article>")


def test_the_licence_stated_in_the_xml_is_read_when_the_lookup_gave_none():
    assert licence_in(ARTICLE) == "https://creativecommons.org/licenses/by-nc-nd/4.0/"
    xml = b"""<article><front><article-meta><permissions>
        <license license-type="open-access" xmlns:xlink="http://www.w3.org/1999/xlink"
        xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY</license-p></license>
        </permissions></article-meta></front></article>"""
    assert licence_in(ET.fromstring(xml)) == "https://creativecommons.org/licenses/by/4.0/"
    assert licence_in(ET.fromstring(b"<article/>")) is None


def test_an_old_paper_without_a_licence_element_gives_its_copyright_terms():
    xml = b"""<article><front><article-meta><permissions><copyright-statement>Copyright 2002 Edwards;
        licensee BioMed Central Ltd. This is an Open Access article: verbatim copying and
        redistribution of this article are permitted</copyright-statement></permissions></article-meta></front></article>"""
    licence = licence_in(ET.fromstring(xml))
    assert licence is not None
    assert licence.startswith(
        "Copyright 2002 Edwards; licensee BioMed Central Ltd. This is an Open Access article"
    )
