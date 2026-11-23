"""The knowledge graph: co-mentions, and syndrome links from the literature and from findings (spec G4-G6)."""

from __future__ import annotations

import pytest

from hypothesis_engine.evidence import ENZYMES, INTERVENTION, Finding
from hypothesis_engine.evidence import SYNDROMES as GROUPS
from hypothesis_engine.graph import ENZYME_GENES, SYNDROMES, library_graph
from hypothesis_engine.library import Library, write_build
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.topics import load_topics
from hypothesis_engine.vocabulary import Term, Vocabulary, load_vocabulary

PSSD, PFS = "post-SSRI sexual dysfunction", "post-finasteride syndrome"

VOCABULARY = Vocabulary(
    terms=(
        Term(PSSD, "syndrome", (), "100", abbreviations=("PSSD",)),
        Term(PFS, "syndrome", (), "200", abbreviations=("PFS",)),
        Term("BDNF", "gene", (), "100"),
        Term("SRD5A2", "gene", ("5α-reductase type 2",), "200"),
        Term("allopregnanolone", "neurosteroid", (), "200"),
        Term("anhedonia", "symptom", (), "100"),
        Term("tinnitus", "symptom", (), "999"),  # its seed paper is not in the library
    )
)


def _paper(pmid: str, title: str, abstract: str) -> Paper:
    return Paper(pmid, title, abstract, None, None, None, 2021)


@pytest.fixture
def library(tmp_path):
    papers = [
        _paper("100", "Post-SSRI sexual dysfunction", "In PSSD, BDNF fell and anhedonia persisted."),
        _paper("200", "Post-finasteride syndrome", "SRD5A2 and allopregnanolone in PFS. Tinnitus too."),
        _paper("300", "Rodent stress", "BDNF and allopregnanolone in rats."),
        # names PSSD in full, but the PSSD search did not find it
        _paper("400", "Neurosteroids", "Anhedonia in post-SSRI sexual dysfunction."),
    ]
    found_by = {
        "100": [("PSSD", 1)],
        "200": [("PFS", 1)],
        "300": [("neurosteroids", 1)],
        "400": [("neurosteroids", 2)],
    }
    write_build(tmp_path / "library.sqlite", papers, found_by, "sha", "2026-11-23T00:00:00+00:00", [])
    with Library.open(tmp_path / "library.sqlite") as opened:
        yield opened


def _kinds(graph, source: str) -> dict[str, str]:
    return {target: data["kind"] for _, target, data in graph.out_edges(source, data=True)}


def test_a_term_in_a_syndrome_s_papers_gets_a_literature_link_from_that_syndrome(library):
    graph = library_graph(library, VOCABULARY)
    assert _kinds(graph, PSSD) == {"BDNF": "literature", "anhedonia": "literature"}
    link = graph.edges[PSSD, "BDNF"]
    assert link["papers"] == 1
    assert link["passage_ids"] == ("100:1",)
    assert graph.nodes[PSSD]["syndrome"] == "PSSD"


def test_two_terms_in_one_passage_are_co_mentioned_both_ways(library):
    graph = library_graph(library, VOCABULARY)
    for a, b in (("BDNF", "allopregnanolone"), ("allopregnanolone", "BDNF")):
        assert graph.edges[a, b]["kind"] == "co-mention"
        assert graph.edges[a, b]["passage_ids"] == ("300:1",)
        assert graph.edges[a, b]["label"] == "co-mentioned in 1 paper"


def test_a_passage_naming_the_syndrome_counts_for_its_link_whatever_search_found_the_paper(library):
    graph = library_graph(library, VOCABULARY)
    link = graph.edges[PSSD, "anhedonia"]
    assert (link["kind"], link["papers"], link["passage_ids"]) == ("literature", 2, ("100:1", "400:1"))
    assert graph.edges["anhedonia", PSSD]["kind"] == "co-mention"


def _finding(marker: str, group_a: str, group_b: str, label: str, comparison: str = "group comparison"):
    return Finding(
        section="comparison",
        comparison=comparison,
        marker_id=marker.lower().replace(" ", ""),  # as the export writes it: "androsterone/etiocholanolone"
        marker_name=marker,
        sex="male",
        group_a=group_a,
        group_b=group_b,
        label=label,
        statistics={"effect_size": -0.6, "q_value": 0.01},
        labels=("thresholds provisional, needs expert review",),
        tool="asclepius 0.1.0",
        dataset_fingerprint="f" * 64,
        run_date="2026-11-23T00:00:00+00:00",
    )


def test_a_significant_finding_for_a_syndrome_makes_its_link_a_stronger_finding_link(library):
    strong = _finding("Allopregnanolone", "PFS", "healthy control", "strong")
    graph = library_graph(library, VOCABULARY, [strong])
    link = graph.edges[PFS, "allopregnanolone"]
    assert link["kind"] == "finding"
    assert link["findings"] == (strong,)
    assert link["passage_ids"] == ("200:1",)  # the literature behind it is kept
    assert graph.nodes["allopregnanolone"]["findings"] == (strong,)


def test_a_finding_links_a_term_the_syndrome_s_papers_never_name(library):
    suggestive = _finding("Allopregnanolone", "PSSD", "SSRI-exposed control", "suggestive")
    graph = library_graph(library, VOCABULARY, [suggestive])
    link = graph.edges[PSSD, "allopregnanolone"]
    assert (link["kind"], link["papers"], link["passage_ids"]) == ("finding", 0, ())


def test_an_enzyme_activity_ratio_links_the_enzyme_s_gene(library):
    ratio = _finding(
        "Androsterone / etiocholanolone", "PSSD", "healthy control", "strong", "enzyme-activity ratio"
    )
    graph = library_graph(library, VOCABULARY, [ratio])
    assert graph.edges[PSSD, "SRD5A2"]["kind"] == "finding"


def test_only_a_significant_finding_of_one_syndrome_against_controls_makes_a_link(library):
    others = [
        _finding("Allopregnanolone", "PSSD", "healthy control", "noise-level"),
        _finding("Allopregnanolone", "PSSD", "PFS", "strong"),  # which syndrome would it be for?
        _finding("Allopregnanolone", "fluoxetine", "other causal drugs", "strong"),
        _finding("Allopregnanolone", None, None, None, INTERVENTION),
    ]
    graph = library_graph(library, VOCABULARY, others)
    assert _kinds(graph, PSSD) == {"BDNF": "literature", "anhedonia": "literature"}
    assert _kinds(graph, PFS) == {"SRD5A2": "literature", "allopregnanolone": "literature"}


def test_a_term_left_out_for_its_seed_gets_no_node_even_from_a_finding(library):
    graph = library_graph(library, VOCABULARY, [_finding("Tinnitus", "PFS", "healthy control", "strong")])
    assert "tinnitus" not in graph


def test_no_node_or_link_is_without_a_passage_or_a_finding_behind_it(library):
    findings = [
        _finding("Allopregnanolone", "PSSD", "healthy control", "strong"),
        _finding(
            "Androsterone / etiocholanolone", "PFS", "healthy control", "suggestive", "enzyme-activity ratio"
        ),
    ]
    graph = library_graph(library, VOCABULARY, findings)
    assert "post-accutane syndrome" not in graph  # no vocabulary term, no paper, no finding
    for a, b, data in graph.edges(data=True):
        assert data["passage_ids"] or data["findings"], (a, b)
    for name, data in graph.nodes(data=True):
        assert data["passage_ids"] or data["findings"] or graph.out_degree(name), name


def test_the_syndromes_are_topics_of_the_library_and_terms_of_the_vocabulary():
    assert set(SYNDROMES) <= {t.name for t in load_topics()}
    assert set(SYNDROMES) == set(GROUPS)
    kinds = {t.name: t.kind for t in load_vocabulary().terms}
    assert all(kinds[name] == "syndrome" for name in SYNDROMES.values())
    assert set(ENZYME_GENES) <= {enzyme for enzyme, _ in ENZYMES.values()}
    assert all(gene in kinds for genes in ENZYME_GENES.values() for gene in genes)
