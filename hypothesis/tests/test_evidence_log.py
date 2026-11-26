"""The Bayesian evidence log: Beta posteriors per graph node (spec B1 to B5)."""

from __future__ import annotations

import pytest
from test_bottlenecks import FIXTURE

from hypothesis_engine.bottlenecks import bottlenecks
from hypothesis_engine.evidence_log import (
    REFUTES,
    SUPPORTS,
    UNPUBLISHED,
    Entry,
    EvidenceLog,
    EvidenceLogError,
    posterior,
    weighted,
)


def test_a_node_with_no_papers_and_no_entries_keeps_the_beta_1_1_prior():
    found = posterior(0, [])
    assert (found.alpha, found.beta, found.mean) == (1, 1, 0.5)


def test_each_paper_adds_a_quarter_support_up_to_two():
    assert posterior(3, []).alpha == 1.75
    assert posterior(8, []).alpha == 3
    assert posterior(40, []).alpha == 3  # capped at +2
    assert posterior(40, []).beta == 1


def test_a_logged_rct_supporting_a_node_adds_four_supports():
    rct = Entry.new("SRD5A2", SUPPORTS, "rct", "PMID:12345678")
    before, after = posterior(4, []), posterior(4, [rct])
    assert after.alpha - before.alpha == 4
    assert after.beta == before.beta
    assert after.mean == pytest.approx(6 / 7)  # Beta(1 + 1 + 4, 1)


def test_each_study_type_counts_by_the_provisional_table():
    weights = {"rct": 4, "cohort": 2, "case-series": 1, "in-vitro-or-animal": 0.5, "self-report": 0.25}
    for study, weight in weights.items():
        refuted = posterior(0, [Entry.new("COMT", REFUTES, study, "10.1000/x")])
        assert (refuted.alpha, refuted.beta) == (1, 1 + weight)


def test_a_logged_rct_raises_the_posterior_and_survives_export_and_import_into_a_fresh_folder(tmp_path):
    log = EvidenceLog.load(tmp_path / "one" / "evidence-log.json")
    before = log.posterior("SRD5A2", papers=4)
    assert log.add(Entry.new("SRD5A2", SUPPORTS, "rct", "PMID:12345678"))
    log.save(tmp_path / "one" / "evidence-log.json")

    fresh = EvidenceLog.load(tmp_path / "two" / "evidence-log.json")
    assert fresh.entries == ()
    assert fresh.merge(EvidenceLog.from_json(log.to_json())) == 1
    fresh.save(tmp_path / "two" / "evidence-log.json")

    reopened = EvidenceLog.load(tmp_path / "two" / "evidence-log.json")
    assert reopened.entries == log.entries
    assert reopened.posterior("SRD5A2", papers=4).alpha - before.alpha == 4
    assert reopened.posterior("COMT", papers=4) == before


def test_the_same_source_node_and_direction_is_logged_once():
    log = EvidenceLog()
    assert log.add(Entry.new("COMT", SUPPORTS, "cohort", "PMID:42"))
    assert not log.add(Entry.new("COMT", SUPPORTS, "rct", "pmid 42"))
    assert log.add(Entry.new("COMT", REFUTES, "cohort", "PMID:42"))  # the other direction
    assert log.add(Entry.new("HSD11B2", SUPPORTS, "cohort", "PMID:42"))  # another node
    assert len(log.entries) == 3


def test_merging_a_log_drops_its_duplicates_and_keeps_the_rest():
    mine, theirs = EvidenceLog(), EvidenceLog()
    mine.add(Entry.new("COMT", SUPPORTS, "cohort", "10.1000/ABC"))
    theirs.add(Entry.new("COMT", SUPPORTS, "case-series", "https://doi.org/10.1000/abc"))
    theirs.add(Entry.new("COMT", REFUTES, "self-report", "PMID:7"))
    assert mine.merge(theirs) == 1
    assert mine.merge(theirs) == 0
    assert [e.direction for e in mine.entries] == [SUPPORTS, REFUTES]


def test_unpublished_observations_are_kept_apart_unless_the_same_entry_comes_back():
    log = EvidenceLog()
    first = Entry.new("COMT", SUPPORTS, "self-report", UNPUBLISHED)
    assert log.add(first)
    assert log.add(Entry.new("COMT", SUPPORTS, "self-report", "Unpublished observation"))
    again = EvidenceLog.from_json(log.to_json())
    assert log.merge(again) == 0
    assert len(log.entries) == 2


@pytest.mark.parametrize(
    ("given", "kept"),
    [
        ("12345678", "PMID:12345678"),
        ("PMID: 12345678", "PMID:12345678"),
        ("10.1016/J.X.2020.01.002", "doi:10.1016/j.x.2020.01.002"),
        ("doi:10.1016/j.x", "doi:10.1016/j.x"),
        ("https://doi.org/10.1016/j.x", "doi:10.1016/j.x"),
    ],
)
def test_a_source_is_kept_in_one_form(given, kept):
    assert Entry.new("COMT", SUPPORTS, "rct", given).source == kept


@pytest.mark.parametrize(
    ("direction", "study", "source"),
    [
        ("supports", "rct", ""),  # B3: a PMID, a DOI or "unpublished observation"
        ("supports", "rct", "a blog post"),
        ("maybe", "rct", "PMID:1"),
        ("supports", "anecdote", "PMID:1"),
    ],
)
def test_an_entry_without_a_known_direction_study_type_or_source_is_refused(direction, study, source):
    with pytest.raises(EvidenceLogError):
        Entry.new("COMT", direction, study, source)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"version": 1, "entries": [{"id": "x"}]}',
        '{"version": 1, "entries": [{"id": "x", "node": "COMT", "direction": "maybe", "study_type": "rct", "source": "PMID:1"}]}',
        '{"version": 2, "entries": []}',
    ],
)
def test_an_unreadable_import_is_refused(text):
    with pytest.raises(EvidenceLogError):
        EvidenceLog.from_json(text)


def test_a_corrupt_log_file_is_refused_not_silently_emptied(tmp_path):
    path = tmp_path / "evidence-log.json"
    path.write_text("{")
    with pytest.raises(EvidenceLogError):
        EvidenceLog.load(path)


def _fixture(papers: dict[str, int]):
    graph = FIXTURE.copy()
    for name in graph:
        graph.nodes[name]["papers"] = papers.get(name, 0)
    return graph


# On the bottleneck fixture (test_bottlenecks), X scores 0.45 + 0.35 * 2.5/4.5 + 0.2 * 13/22 = 0.76263
# and the hub H 0.45/3 + 0.35 + 0.2 = 0.7. X has 8 papers behind it, Beta(3, 1), mean 3/4; H none, 1/2.
X_SCORE, H_SCORE = 0.45 + 0.35 * 2.5 / 4.5 + 0.2 * 13 / 22, 0.7


def test_the_posterior_mean_multiplies_the_bottleneck_score():
    graph = _fixture({"X": 8, "H": 0})
    found = {b.name: b for b in weighted(bottlenecks(graph), graph, EvidenceLog())}
    assert (found["X"].score, found["H"].score) == pytest.approx((X_SCORE * 3 / 4, H_SCORE / 2))
    assert (found["X"].cascade, found["X"].syndromes) == (2.5, ("PAS", "PFS", "PSSD"))  # raw metrics kept


def test_a_refuting_rct_re_ranks_the_bottlenecks():
    graph = _fixture({"X": 8, "H": 0})
    log = EvidenceLog()
    assert [b.name for b in weighted(bottlenecks(graph), graph, log)][:2] == ["X", "H"]
    log.add(Entry.new("X", REFUTES, "rct", "PMID:99"))
    ranked = weighted(bottlenecks(graph), graph, log)
    assert [b.name for b in ranked][:2] == ["H", "X"]
    assert ranked[1].score == pytest.approx(X_SCORE * 3 / 8)  # Beta(3, 5)
