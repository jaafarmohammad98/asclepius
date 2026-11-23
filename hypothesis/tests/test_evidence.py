"""Evidence bundles (M4-05a): what the library says about each finding, before any LLM is asked.

The findings files in `fixtures/` are rows of real web-app exports (`asclepius.web.export`) of the
synthetic datasets; the self-report one has one row relabelled suggestive, since that run found none.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_search import FILLER, WordEmbedder

from hypothesis_engine import build
from hypothesis_engine.evidence import CONTRADICTING, SUPPORTING, gather_evidence, queries_for, read_findings
from hypothesis_engine.library import write_build
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.search import Index, build_index

FIXTURES = Path(__file__).parent / "fixtures"
LAB = FIXTURES / "findings_lab.csv"
SELF_REPORT = FIXTURES / "findings_self_report.csv"


def paper(pmid: str, title: str, abstract: str) -> Paper:
    return Paper(pmid, title, f"{abstract} {FILLER}", None, None, None, 2020)


PAPERS = [
    paper("10", "Neurosteroids in post-finasteride syndrome", "Androsterone was lower in men with PFS."),
    paper("11", "Androgen metabolites in PSSD", "Androsterone was higher in patients than in controls."),
    paper("12", "Paroxetine and mood", "Depression was reported more often after paroxetine."),
    paper("13", "Hair loss", "Minoxidil grew hair in most men."),
    Paper("14", "Lower androsterone in post-SSRI sexual dysfunction", None, None, None, None, 2021),
]


@pytest.fixture
def built(tmp_path):
    library = tmp_path / "library.sqlite"
    found_by = {p.pmid: [("PSSD", rank)] for rank, p in enumerate(PAPERS, start=1)}
    write_build(library, PAPERS, found_by, "sha", "2026-11-22T00:00:00+00:00", [])
    build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    return library, tmp_path / "index.sqlite"


@pytest.fixture
def index(built):
    library, path = built
    with Index.open(path, library, WordEmbedder()) as opened:
        yield opened


def test_a_noise_level_finding_produces_no_bundle(index):
    findings = read_findings(LAB)
    bundles = gather_evidence(findings, index)
    assert "16_oh_e1" not in {b.finding.marker_id for b in bundles}
    assert {b.finding.marker_id for b in bundles} == {"androsterone", "androsterone/etiocholanolone"}


def test_a_bundle_carries_the_group_statistics(index):
    (bundle, _ratio) = gather_evidence(read_findings(LAB), index)
    finding = bundle.as_dict()["finding"]
    assert (finding["group_a"], finding["group_b"], finding["sex"]) == ("PSSD", "healthy control", "male")
    assert (finding["n_a"], finding["n_b"]) == (28, 28)
    assert (finding["median_a"], finding["median_b"]) == (477.5, 1140.0)
    assert finding["effect_size"] == pytest.approx(-0.9043, abs=1e-4)
    assert finding["label"] == "suggestive"
    assert finding["tier_a_result"] == "Tier A too small to test"
    assert bundle.as_dict()["dataset_fingerprint"].startswith("bf8e9b77")


LEAKED_COLUMNS = ("case_id", "author_key", "report_id", "patient_value")
LEAKED_VALUES = ("case-4471", "u_quiet_fox", "report-0099", "612.25")


def test_no_bundle_contains_a_record_id_or_individual_value(index, tmp_path):
    # an export with columns it should never have: whatever they hold stays out of every bundle
    text = LAB.read_text(encoding="utf-8").splitlines()
    leaky = [",".join((text[0], *LEAKED_COLUMNS))]
    leaky += [",".join((line, *LEAKED_VALUES)) for line in text[1:]]
    path = tmp_path / "findings.csv"
    path.write_text("\n".join(leaky) + "\n", encoding="utf-8")

    bundles = gather_evidence(read_findings(path), index)
    assert bundles
    for bundle in bundles:
        written = json.dumps(bundle.as_dict())
        for leaked in (*LEAKED_COLUMNS, *LEAKED_VALUES):
            assert leaked not in written


def test_a_self_report_bundle_keeps_its_self_report_label(index):
    (bundle,) = gather_evidence(read_findings(SELF_REPORT), index)
    assert bundle.finding.marker_name == "depression"
    assert bundle.self_report
    assert "self-report, no controls" in bundle.as_dict()["labels"]
    (lab, _ratio) = gather_evidence(read_findings(LAB), index)
    assert not lab.self_report
    assert "self-report, no controls" not in lab.as_dict()["labels"]


def by_stance(finding):
    queries = queries_for(finding)
    return (
        [q.text for q in queries if q.stance == SUPPORTING],
        [q.text for q in queries if q.stance == CONTRADICTING],
    )


def test_a_marker_finding_is_searched_by_marker_direction_syndrome_and_drug():
    marker = next(f for f in read_findings(LAB) if f.marker_id == "androsterone")  # lower in PSSD
    supporting, contradicting = by_stance(marker)
    assert "lower Androsterone in post-SSRI sexual dysfunction than in healthy controls" in supporting
    assert "lower Androsterone after SSRI antidepressants" in supporting
    assert "higher Androsterone in post-SSRI sexual dysfunction than in healthy controls" in contradicting
    assert (
        "no difference in Androsterone between post-SSRI sexual dysfunction and healthy controls"
        in contradicting
    )


def test_a_ratio_finding_is_also_searched_by_the_enzyme_it_stands_for():
    ratio = next(f for f in read_findings(LAB) if f.marker_id == "androsterone/etiocholanolone")
    supporting, contradicting = by_stance(ratio)
    assert "lower 5-alpha reductase activity in post-SSRI sexual dysfunction" in supporting
    assert "higher 5-alpha reductase activity in post-SSRI sexual dysfunction" in contradicting


def test_a_ratio_that_falls_when_its_enzyme_rises_is_searched_the_other_way():
    ratio = next(f for f in read_findings(LAB) if f.marker_id == "androsterone/etiocholanolone")
    cortisol = replace(
        ratio,
        marker_id="24hr_free_cortisol/24hr_free_cortisone",
        marker_name="Free cortisol / free cortisone",
    )
    supporting, _ = by_stance(cortisol)  # a lower cortisol / cortisone ratio: more 11-beta-HSD2 at work
    assert (
        "higher 11-beta-hydroxysteroid dehydrogenase type 2 activity in post-SSRI sexual dysfunction"
        in supporting
    )


def test_a_self_report_finding_is_searched_by_drug_and_symptom():
    # depression, reported more often after paroxetine
    (finding,) = [f for f in read_findings(SELF_REPORT) if f.marker_name == "depression"]
    supporting, contradicting = by_stance(finding)
    assert "depression more often after paroxetine than after other drugs" in supporting
    assert "persistent depression after stopping paroxetine" in supporting
    assert "depression less often after paroxetine than after other drugs" in contradicting
    assert all("other causal drugs" not in q for q in supporting + contradicting)


def test_passages_are_split_into_candidate_support_and_contradiction(index):
    (marker, _ratio) = gather_evidence(read_findings(LAB), index, k=2)  # androsterone lower in PSSD
    assert [p.paper_id for p in marker.supporting][0] == "10"  # "Androsterone was lower in men with PFS"
    assert [p.paper_id for p in marker.contradicting][0] == "11"  # "Androsterone was higher in patients"
    assert len(marker.supporting) <= 2 and len(marker.contradicting) <= 2
    assert not {p.passage_id for p in marker.supporting} & {p.passage_id for p in marker.contradicting}


def test_each_passage_names_its_paper_and_the_search_that_found_it(index):
    (marker, _ratio) = gather_evidence(read_findings(LAB), index, k=2)
    passage = marker.as_dict()["supporting"][0]
    assert passage["passage_id"] == "10:1" and passage["paper_id"] == "10"
    assert passage["title"] == "Neurosteroids in post-finasteride syndrome"
    assert passage["text"].startswith("Androsterone was lower in men with PFS.")
    assert passage["query"] in [q.text for q in marker.queries]
    assert marker.as_dict()["library_version"] == 1 and marker.as_dict()["search_model"] == "test-words"


def test_an_unrelated_paper_is_not_offered_as_evidence(index):
    for bundle in gather_evidence(read_findings(LAB), index, k=2):
        assert "13" not in {p.paper_id for p in (*bundle.supporting, *bundle.contradicting)}


def where(built):
    library, path = built
    return ["--library", str(library), "--index", str(path)]


def test_the_evidence_command_writes_the_bundles_as_json(built, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build, "LocalEmbedder", lambda *_args, **_kwargs: WordEmbedder())
    out = tmp_path / "evidence.json"
    assert build.main(["evidence", str(LAB), *where(built), "--out", str(out), "-k", "2"]) == 0
    bundles = json.loads(out.read_text(encoding="utf-8"))
    assert [b["finding"]["marker_id"] for b in bundles] == ["androsterone", "androsterone/etiocholanolone"]
    assert "2 of 4 findings worth testing" in capsys.readouterr().err


def test_the_evidence_command_refuses_a_file_that_is_not_a_findings_export(built, tmp_path, capsys):
    other = tmp_path / "cases.csv"
    other.write_text("case_id,author_key\nc1,a1\n", encoding="utf-8")
    assert build.main(["evidence", str(other), *where(built), "--out", str(tmp_path / "e.json")]) == 1
    assert "not an analysis findings export" in capsys.readouterr().err
    assert not (tmp_path / "e.json").exists()


def test_a_paper_known_only_by_its_title_is_not_offered_as_evidence(index):
    # paper 14 has no abstract: its one passage is its title, with nothing to quote
    for bundle in gather_evidence(read_findings(LAB), index, k=5):
        assert "14" not in {p.paper_id for p in (*bundle.supporting, *bundle.contradicting)}


def test_a_marker_is_also_searched_on_its_own():
    marker = next(f for f in read_findings(LAB) if f.marker_id == "androsterone")
    supporting, contradicting = by_stance(marker)
    assert "lower Androsterone levels" in supporting and "higher Androsterone levels" in contradicting


def test_an_onset_or_trajectory_finding_is_searched_as_a_course_not_a_symptom():
    (symptom,) = [f for f in read_findings(SELF_REPORT) if f.marker_name == "depression"]
    recovered = replace(
        symptom,
        comparison="drug profile: trajectory",
        marker_id="trajectory:recovered",
        marker_name="recovered",
    )
    supporting, contradicting = by_stance(recovered)  # recovery reported more often after paroxetine
    assert "symptoms recovered more often after paroxetine than after other drugs" in supporting
    assert "symptoms recovered less often after paroxetine than after other drugs" in contradicting
    assert not [q for q in supporting + contradicting if "persistent" in q]
    onset = replace(
        symptom,
        comparison="drug profile: onset",
        marker_id="onset:during_use",
        marker_name="onset during use",
    )
    supporting, _ = by_stance(onset)
    assert supporting == ["symptom onset during use more often after paroxetine than after other drugs"]
