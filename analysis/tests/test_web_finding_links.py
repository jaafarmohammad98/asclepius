"""A finished run's findings in the app's knowledge graph (M6-04b): its strong and suggestive
findings become finding links on the bottleneck map, re-scored, labelled with the run's dataset,
until a new load, run or clear drops them."""

import time

import pytest
from knowledge_fixture import knowledge_engine, serving
from test_web_server import api, call

from asclepius.web.graph import KnowledgeGraph

PSSD = "post-SSRI sexual dysfunction"


@pytest.fixture
def app(tmp_path):
    with serving(tmp_path, with_androsterone=True) as (server, url):
        yield server, url


def run_example(url):
    assert api(url, "/api/example/lab-report", "POST")[0] == 200
    assert api(url, "/api/run", "POST", {"group_a": "PSSD", "group_b": "healthy control"})[0] == 202


def settled(url):
    """The map once it has finished updating with the run's findings."""
    deadline = time.monotonic() + 60
    while True:
        view = api(url, "/api/graph")[1]
        if api(url, "/api/status")[1]["state"] != "running" and not (view["run"] or {}).get("updating"):
            return view
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_a_finished_run_adds_its_finding_links_to_the_map(app):
    server, url = app
    before = api(url, "/api/graph")[1]
    assert before["run"] is None
    assert "androsterone" not in {t["name"] for t in before["terms"] if t["syndromes"]}

    run_example(url)
    view = settled(url)
    links = {(link["syndrome"], link["term"]): link for link in view["links"]}
    link = links[("PSSD", "androsterone")]
    assert link["kind"] == "finding"
    assert link["label"] == "in 2 significant analysis findings"  # the marker and its ratio
    fingerprint = server.workspace.data.fingerprint
    assert view["run"] == {"fingerprint": fingerprint[:12], "updating": False, "finding_links": 1}
    assert fingerprint[:12] in view["message"]
    ranked = {t["name"]: t for t in view["terms"]}
    assert ranked["androsterone"]["syndromes"] == ["PSSD"]  # re-scored with the link in

    status, edge = api(url, f"/api/graph/edge?a={PSSD.replace(' ', '%20')}&b=androsterone")
    assert status == 200 and edge["kind"] == "finding"
    assert {(f["marker_name"], f["label"]) for f in edge["findings"]} == {
        ("Androsterone", "strong"),
        ("Androsterone / etiocholanolone", "strong"),
    }


def test_loading_another_dataset_returns_the_map_to_the_library(app):
    server, url = app
    library = server.knowledge.graph
    run_example(url)
    settled(url)
    assert api(url, "/api/example/self-report", "POST")[0] == 200
    view = api(url, "/api/graph")[1]
    assert view["run"] is None
    assert {link["kind"] for link in view["links"]} == {"literature"}
    assert server.knowledge.graph is library  # kept in memory, not rebuilt


def test_clearing_the_files_drops_the_findings_too(app):
    _, url = app
    run_example(url)
    settled(url)
    assert api(url, "/api/clear", "POST")[0] == 200
    assert api(url, "/api/graph")[1]["run"] is None


def test_findings_waiting_for_the_graph_are_added_once_it_is_built(tmp_path):
    knowledge = KnowledgeGraph(knowledge_engine(tmp_path, with_androsterone=True))
    token = knowledge.expect("abcdef1234567890")
    assert knowledge.map_view()["state"] == "building"
    knowledge.add(token, [])
    knowledge.start()
    knowledge.wait(timeout=30)
    deadline = time.monotonic() + 30
    while knowledge.map_view()["run"]["updating"]:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert knowledge.map_view()["run"] == {
        "fingerprint": "abcdef123456",
        "updating": False,
        "finding_links": 0,
    }


def test_a_finished_run_can_be_exported_while_the_graph_is_still_updating(app, monkeypatch):
    from asclepius.web import workspace as workspace_module

    real = workspace_module.read_findings

    def slow(path):
        time.sleep(0.5)  # the hand-over to the graph takes a while
        return real(path)

    monkeypatch.setattr(workspace_module, "read_findings", slow)
    _, url = app
    run_example(url)
    while api(url, "/api/status")[1]["state"] == "running":
        time.sleep(0.01)
    assert call(url + "/api/export/findings.csv")[0] == 200  # not "let it finish"
    assert settled(url)["run"]["finding_links"] == 1
