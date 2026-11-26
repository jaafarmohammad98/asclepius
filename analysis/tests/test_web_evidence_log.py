"""The evidence log in the web app (M6-11, spec B2 to B5): each term with its prior and posterior and the
entries behind them, a result logged from the form, the log exported and imported."""

import json
import urllib.request

import pytest
from hypothesis_engine.drugs import write_drug_cache
from hypothesis_engine.evidence_log import Entry, EvidenceLog
from knowledge_fixture import RERANKED, knowledge_engine, serving
from test_web_server import api, call

from asclepius.web.evidence_log import EvidenceViewError, add_entry, evidence_view, import_log
from asclepius.web.graph import READY, KnowledgeGraph
from asclepius.web.leaderboard import leaderboard_view

RCT_AGAINST_BDNF = {"node": "BDNF", "direction": "refutes", "study_type": "rct", "source": "PMID 31000009"}


@pytest.fixture
def engine(tmp_path):
    return knowledge_engine(tmp_path)


@pytest.fixture
def knowledge(engine):
    built = KnowledgeGraph(engine)
    built.start()
    assert built.wait(timeout=30) == READY
    return built


def nodes(view):
    return {n["name"]: n for n in view["nodes"]}


def test_a_logged_result_shows_under_its_node_with_its_prior_and_posterior(knowledge):
    view = add_entry(RCT_AGAINST_BDNF, knowledge)
    bdnf = nodes(view)["BDNF"]
    # BDNF is in 3 papers: Beta(1 + 0.75, 1) before, and a refuting RCT adds 4 to beta
    assert bdnf["prior"] == {"alpha": 1.75, "beta": 1.0, "mean": pytest.approx(1.75 / 2.75)}
    assert bdnf["posterior"] == {"alpha": 1.75, "beta": 5.0, "mean": pytest.approx(1.75 / 6.75)}
    [entry] = bdnf["entries"]
    assert entry["direction"] == "refutes"
    assert entry["study_type"] == "randomised controlled trial"
    assert entry["weight"] == 4.0
    assert entry["source"] == "PMID:31000009"
    assert entry["url"] == "https://pubmed.ncbi.nlm.nih.gov/31000009/"
    assert nodes(view)["COMT"]["entries"] == []
    assert view["message"] == "Logged: PMID:31000009 refutes BDNF."
    assert evidence_view(knowledge)["nodes"] == view["nodes"]  # kept


def test_the_same_source_node_and_direction_twice_is_refused(knowledge):
    add_entry(RCT_AGAINST_BDNF, knowledge)
    with pytest.raises(EvidenceViewError, match="in the log already"):
        add_entry(RCT_AGAINST_BDNF | {"source": "pmid:31000009", "study_type": "cohort"}, knowledge)
    assert evidence_view(knowledge)["count"] == 1


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"source": "a blog post"}, "a source is a PMID, a DOI"),
        ({"node": "TNF"}, "not a term of the knowledge graph"),
        ({"node": "post-SSRI sexual dysfunction"}, "not a term of the knowledge graph"),
        ({"direction": "maybe"}, "a direction is supports or refutes"),
        ({"study_type": "anecdote"}, "a study type is one of"),
        ({"source": None}, "as text"),
    ],
)
def test_a_result_that_cannot_be_logged_is_refused_in_plain_words(knowledge, change, said):
    with pytest.raises(EvidenceViewError, match=said):
        add_entry(RCT_AGAINST_BDNF | change, knowledge)
    assert evidence_view(knowledge)["count"] == 0


def test_an_unpublished_observation_is_logged_each_time(knowledge):
    observed = RCT_AGAINST_BDNF | {"study_type": "self-report", "source": "Unpublished observation"}
    add_entry(observed, knowledge)
    view = add_entry(observed, knowledge)
    entries = nodes(view)["BDNF"]["entries"]
    assert [e["source"] for e in entries] == ["unpublished observation"] * 2
    assert entries[0]["url"] is None


def test_a_log_that_cannot_be_read_is_never_written_over(engine):
    engine.evidence.write_text("not json", encoding="utf-8")
    built = KnowledgeGraph(engine)
    built.start()
    assert built.wait(timeout=30) == READY
    assert "cannot be read" in evidence_view(built)["error"]
    with pytest.raises(EvidenceViewError, match="cannot be read"):
        add_entry(RCT_AGAINST_BDNF, built)
    with pytest.raises(EvidenceViewError, match="cannot be read"):
        import_log(EvidenceLog().to_json(), built)
    assert engine.evidence.read_text(encoding="utf-8") == "not json"


def test_an_exported_log_imports_into_another_dropping_what_it_has(tmp_path, knowledge):
    add_entry(RCT_AGAINST_BDNF, knowledge)
    add_entry(
        RCT_AGAINST_BDNF | {"node": "COMT", "direction": "supports", "source": "10.1234/ABC"}, knowledge
    )
    exported = knowledge.evidence.export()

    (tmp_path / "other").mkdir()
    other = KnowledgeGraph(knowledge_engine(tmp_path / "other"))
    other.start()
    assert other.wait(timeout=30) == READY
    add_entry(RCT_AGAINST_BDNF, other)  # it has this one already
    view = import_log(exported, other)
    assert view["message"] == "Imported 1 new entry of 2; 1 was in the log already."
    assert view["count"] == 2
    assert nodes(view)["COMT"]["entries"][0]["url"] == "https://doi.org/10.1234/abc"
    assert import_log(exported, other)["message"] == "Imported 0 new entries of 2; 2 were in the log already."
    reopened = KnowledgeGraph(other.engine)
    assert reopened.evidence.log().entries == other.evidence.log().entries  # kept in the data folder


def test_an_import_that_is_not_an_evidence_log_is_refused(knowledge):
    with pytest.raises(EvidenceViewError, match="not an evidence log"):
        import_log("[1, 2]", knowledge)


def test_an_imported_entry_on_a_term_this_graph_lacks_is_kept_and_listed_apart(knowledge):
    foreign = EvidenceLog([Entry.new("TNF", "supports", "cohort", "PMID:5")])
    view = import_log(foreign.to_json(), knowledge)
    assert [e["node"] for e in view["elsewhere"]] == ["TNF"]
    assert "TNF" not in nodes(view)


@pytest.fixture
def app(tmp_path):
    with serving(tmp_path) as (_, url):
        yield url


def test_the_page_logs_a_result_exports_the_log_and_imports_it(app):
    status, view = api(app, "/api/evidence")
    assert status == 200 and view["count"] == 0
    assert view["provisional"] == "provisional, needs expert review"
    assert [t["weight"] for t in view["study_types"]] == [4.0, 2.0, 1.0, 0.5, 0.25]
    status, view = api(app, "/api/evidence", "POST", RCT_AGAINST_BDNF)
    assert status == 200 and view["count"] == 1
    status, answer = api(app, "/api/evidence", "POST", RCT_AGAINST_BDNF)
    assert status == 400 and "in the log already" in answer["error"]

    status, exported = call(app + "/api/export/evidence-log.json")
    assert status == 200
    assert EvidenceLog.from_json(exported.decode()).entries[0].source == "PMID:31000009"
    status, view = api_text(app, "/api/evidence/import", exported)
    assert status == 200 and view["message"] == "Imported 0 new entries of 1; 1 was in the log already."
    status, answer = api_text(app, "/api/evidence/import", b"{}")
    assert status == 400 and "not an evidence log" in answer["error"]


def test_the_export_is_a_file_to_save(app):
    request = urllib.request.Request(app + "/api/export/evidence-log.json")
    with urllib.request.urlopen(request) as response:
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["Content-Disposition"] == 'attachment; filename="asclepius-evidence-log.json"'


def api_text(app, path, body):
    status, content = call(app + path, "POST", body, {"Content-Type": "application/json"})
    return status, json.loads(content)


BDNF_RAW = 0.45 * 2 / 3 + 0.35 + 0.20  # two of three syndromes, the highest cascade and betweenness


def test_each_term_on_the_map_is_weighted_by_its_posterior_mean(knowledge):
    before = {t["name"]: t for t in knowledge.map_view()["terms"]}
    assert before["BDNF"]["evidence"] == pytest.approx(1.75 / 2.75)  # literature alone: 3 papers
    assert before["BDNF"]["score"] == pytest.approx(BDNF_RAW * 1.75 / 2.75)
    assert [t["name"] for t in knowledge.map_view()["terms"]] == ["BDNF", "serotonin", "COMT"]
    add_entry(RCT_AGAINST_BDNF, knowledge)
    after = knowledge.map_view()["terms"]
    assert after[1]["name"] == "BDNF"  # 0.85 × 1.75 / 6.75 = 0.22, below serotonin's 0.5 × 0.6
    assert after[1]["evidence"] == pytest.approx(1.75 / 6.75)
    assert after[1]["score"] == pytest.approx(BDNF_RAW * 1.75 / 6.75)
    assert [t["name"] for t in after] == ["serotonin", "BDNF", "COMT"]


def test_a_logged_result_re_ranks_the_leaderboard(engine, knowledge):
    write_drug_cache(engine.drugs, RERANKED)

    def board():
        return [
            (r["name"], r["criteria"]["impact"]) for r in leaderboard_view(engine.drugs, knowledge)["rows"]
        ]

    assert [name for name, _ in board()] == ["alphadrug", "epsilondrug", "deltadrug", "gammadrug", "betadrug"]
    add_entry(RCT_AGAINST_BDNF, knowledge)
    after = board()
    assert [name for name, _ in after] == ["alphadrug", "epsilondrug", "deltadrug", "betadrug", "gammadrug"]
    assert dict(after)["alphadrug"] == pytest.approx(BDNF_RAW * 1.75 / 6.75)  # BDNF's weighted score


def test_a_log_that_cannot_be_read_is_not_exported_as_empty(tmp_path):
    knowledge_engine(tmp_path).evidence.write_text("not json", encoding="utf-8")
    with serving(tmp_path) as (_, url):
        status, answer = api(url, "/api/export/evidence-log.json")
    assert status == 400 and "cannot be read" in answer["error"]
