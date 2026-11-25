"""The drug repurposing leaderboard's data (M6-07, spec R3, R5, R6): each cached drug's criteria,
its targets scored by the app's knowledge graph, its risks and links; the known causes apart."""

import pytest
from knowledge_fixture import DRUGS, knowledge_engine, serving
from test_web_server import api

from asclepius.web.graph import READY, KnowledgeGraph
from asclepius.web.leaderboard import leaderboard_view


@pytest.fixture
def engine(tmp_path):
    return knowledge_engine(tmp_path)


@pytest.fixture
def knowledge(engine):
    built = KnowledgeGraph(engine)
    built.start()
    assert built.wait(timeout=30) == READY
    return built


def rows(view):
    return {r["name"]: r for r in view["rows"]}


def test_the_leaderboard_gives_the_criteria_and_default_weights(engine, knowledge):
    view = leaderboard_view(engine.drugs, knowledge)
    assert view["state"] == READY
    assert view["weights"] == {
        "impact": 0.30,
        "coverage": 0.25,
        "safety": 0.20,
        "bbb": 0.15,
        "readiness": 0.10,
    }
    assert set(rows(view)) == {"alphadrug", "betadrug", "gammadrug"}


def test_impact_and_coverage_come_from_the_app_s_graph_not_the_cache(engine, knowledge):
    alpha = rows(leaderboard_view(engine.drugs, knowledge))["alphadrug"]
    # BDNF in the graph: two of three syndromes, the highest cascade and betweenness (the cache says 0.1)
    assert alpha["criteria"]["impact"] == pytest.approx(0.45 * 2 / 3 + 0.35 + 0.20)
    assert alpha["criteria"]["coverage"] == 2  # PFS and PSSD; the cache says PAS
    assert alpha["criteria"]["bbb"] == pytest.approx(6.0)
    assert alpha["criteria"]["readiness"] == 1.0


def test_every_row_is_a_candidate_for_study_with_its_chembl_record(engine, knowledge):
    for row in leaderboard_view(engine.drugs, knowledge)["rows"]:
        assert row["label"] == "candidate for study"
        assert row["url"] == f"https://www.ebi.ac.uk/chembl/explore/compound/{row['chembl_id']}"


def test_a_row_lists_its_risks_with_where_they_are_stated(engine, knowledge):
    view = rows(leaderboard_view(engine.drugs, knowledge))
    beta = view["betadrug"]
    assert beta["risks"] == ["libido", "raises blood pressure"]
    assert beta["label_url"] == "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=set-b"
    assert beta["mentions"] == [
        {"concern": "libido", "section": "adverse_reactions", "excerpt": "decreased libido"}
    ]
    assert beta["listed_risks"] == [{"risk": "raises blood pressure", "source": "PMID:31000009"}]
    assert beta["criteria"]["safety"] == 2
    gamma = view["gammadrug"]
    assert gamma["risks"] == ["withdrawn from a market after approval"]
    assert gamma["assumed"] == ["safety penalty"]  # no US label: the worst seen, 2, then the withdrawal
    assert gamma["criteria"]["safety"] == 3
    assert gamma["bbb_note"] == "estimate, 4 of 6 terms"


def test_a_row_names_its_targets_and_the_syndrome_links_behind_them(engine, knowledge):
    gamma = rows(leaderboard_view(engine.drugs, knowledge))["gammadrug"]
    targets = {t["symbol"]: t for t in gamma["targets"]}
    assert set(targets) == {"BDNF", "COMT"}
    assert targets["BDNF"]["mechanism"] == "BDNF inhibitor"
    links = {(link["syndrome"], link["from"]) for link in targets["BDNF"]["links"]}
    assert links == {("PSSD", "post-SSRI sexual dysfunction"), ("PFS", "post-finasteride syndrome")}
    (comt,) = targets["COMT"]["links"]
    assert comt["label"] == "found in 1 PFS paper"
    assert knowledge.edge(comt["from"], "COMT")["passages"]  # the passages open


def test_the_known_causes_are_listed_apart_never_ranked(engine, knowledge):
    view = leaderboard_view(engine.drugs, knowledge)
    assert view["known_causes"] == [
        {
            "name": "SERTRALINE",
            "chembl_id": "CHEMBL_SERTRALINE",
            "known_cause": "SSRI",
            "url": "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL_SERTRALINE",
        }
    ]
    assert "SERTRALINE" not in rows(view)


def test_the_leaderboard_waits_for_the_graph(engine):
    view = leaderboard_view(engine.drugs, KnowledgeGraph(engine))  # never started
    assert view["state"] == "building"
    assert "rows" not in view


def test_without_drug_data_it_says_how_to_fetch_it(tmp_path, knowledge):
    view = leaderboard_view(tmp_path / "none.json", knowledge)
    assert view["state"] == "missing"
    assert "asclepius-library drugs" in view["message"]


def test_without_a_graph_the_cache_s_scores_are_used_and_said(tmp_path):
    engine = knowledge_engine(tmp_path)
    (tmp_path / "library.sqlite").unlink()
    broken = KnowledgeGraph(engine)
    broken.start()
    assert broken.wait(timeout=30) == "failed"
    view = leaderboard_view(engine.drugs, broken)
    assert view["state"] == READY
    alpha = rows(view)["alphadrug"]
    assert alpha["criteria"]["impact"] == pytest.approx(DRUGS.targets[0].score)
    assert alpha["targets"][0]["links"] == []
    assert "drug data was fetched" in view["message"]


def test_the_app_serves_the_leaderboard(tmp_path):
    with serving(tmp_path) as (_, url):
        status, view = api(url, "/api/leaderboard")
    assert status == 200
    assert view["state"] == READY and len(view["rows"]) == 3
