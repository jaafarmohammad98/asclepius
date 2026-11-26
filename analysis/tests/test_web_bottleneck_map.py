"""The bottleneck map's data (M6-04): the scored terms with their raw metrics, the co-mentions and
syndrome links to draw, and what is behind any one link."""

import urllib.parse
import urllib.request

import pytest
from knowledge_fixture import knowledge_engine, serving
from test_web_server import api, call

from asclepius.web.graph import READY, KnowledgeGraph


@pytest.fixture
def knowledge(tmp_path):
    built = KnowledgeGraph(knowledge_engine(tmp_path))
    built.start()
    assert built.wait(timeout=30) == READY
    return built


def test_the_map_ranks_terms_with_their_raw_metrics(knowledge):
    view = knowledge.map_view()
    assert view["state"] == READY
    assert view["weights"] == {"overlap": 0.45, "cascade": 0.35, "betweenness": 0.20}
    assert view["syndromes"] == ["PSSD", "PFS", "PAS"]
    first = view["terms"][0]
    assert first["name"] == "BDNF"
    assert first["kind"] == "gene"
    assert first["syndromes"] == ["PFS", "PSSD"]
    # two of three syndromes, and the highest cascade and betweenness; times its posterior mean in the
    # evidence log, empty here: Beta(1 + 0.25 × 3 papers, 1) (M6-11)
    assert first["evidence"] == pytest.approx(1.75 / 2.75)
    assert first["score"] == pytest.approx((0.45 * 2 / 3 + 0.35 + 0.20) * 1.75 / 2.75)
    assert {"betweenness", "cascade"} <= set(first)
    assert [t["name"] for t in view["terms"]] == ["BDNF", "serotonin", "COMT"]


def test_each_co_mention_is_drawn_once_with_its_paper_count(knowledge):
    edges = {(e["a"], e["b"]): e for e in knowledge.map_view()["co_mentions"]}
    assert edges[("BDNF", "serotonin")]["papers"] == 2
    assert edges[("BDNF", "serotonin")]["label"] == "co-mentioned in 2 papers"
    assert edges[("BDNF", "COMT")]["label"] == "co-mentioned in 1 paper"
    assert ("serotonin", "BDNF") not in edges and ("COMT", "BDNF") not in edges


def test_the_syndrome_links_say_which_syndrome_and_how(knowledge):
    links = {(link["syndrome"], link["term"]): link for link in knowledge.map_view()["links"]}
    assert links[("PSSD", "BDNF")]["kind"] == "literature"
    assert links[("PSSD", "BDNF")]["label"] == "found in 1 PSSD paper"
    assert ("PFS", "COMT") in links and ("PSSD", "COMT") not in links


def test_a_co_mention_opens_the_passages_it_was_counted_from(knowledge):
    edge = knowledge.edge("serotonin", "BDNF")
    assert edge["label"] == "co-mentioned in 2 papers"
    passages = {p["paper_id"]: p for p in edge["passages"]}
    assert set(passages) == {"31000001", "31000003"}
    cited = passages["31000001"]
    assert cited["text"] == "BDNF and serotonin fell in PSSD."
    assert cited["title"] == "Post-SSRI sexual dysfunction"
    assert (cited["journal"], cited["year"]) == ("J Sex Med", 2022)
    assert {"label": "PubMed 31000001", "url": "https://pubmed.ncbi.nlm.nih.gov/31000001/"} in cited["links"]
    assert {"label": "DOI 10.1/pssd", "url": "https://doi.org/10.1/pssd"} in cited["links"]


def test_a_syndrome_link_opens_its_passages_too(knowledge):
    edge = knowledge.edge("post-finasteride syndrome", "COMT")
    assert edge["label"] == "found in 1 PFS paper"
    assert [p["paper_id"] for p in edge["passages"]] == ["31000002"]


def test_a_link_that_is_not_in_the_graph_is_refused(knowledge):
    with pytest.raises(KeyError):
        knowledge.edge("COMT", "serotonin")


def test_before_the_graph_is_built_the_map_says_why(tmp_path):
    waiting = KnowledgeGraph(knowledge_engine(tmp_path))  # not started
    view = waiting.map_view()
    assert view == {"state": "building", "message": waiting.message, "run": None}


@pytest.fixture
def app(tmp_path):
    with serving(tmp_path) as (_, url):
        yield url


def test_the_page_gets_the_map_and_a_link_s_passages(app):
    status, view = api(app, "/api/graph")
    assert status == 200 and view["terms"][0]["name"] == "BDNF"
    query = urllib.parse.urlencode({"a": "BDNF", "b": "serotonin"})
    status, edge = api(app, f"/api/graph/edge?{query}")
    assert status == 200 and edge["label"] == "co-mentioned in 2 papers"
    assert len(edge["passages"]) == 2


def test_a_link_not_in_the_graph_is_not_found(app):
    query = urllib.parse.urlencode({"a": "COMT", "b": "serotonin"})
    status, answer = api(app, f"/api/graph/edge?{query}")
    assert status == 404 and "COMT" in answer["error"]


def test_the_network_library_is_served_from_the_app_not_a_cdn(app):
    with urllib.request.urlopen(app + "/vendor/cytoscape.min.js") as response:
        assert response.headers["Content-Type"].startswith("text/javascript")
        assert b"cytoscape" in response.read(4000)
    status, page = call(app + "/")
    assert b'<script src="/vendor/cytoscape.min.js">' in page
    assert b"https://" not in page.split(b"<script")[1].split(b">")[0]


def test_only_terms_are_co_mentioned_and_only_syndromes_link(knowledge):
    view = knowledge.map_view()
    syndromes = {"post-SSRI sexual dysfunction", "post-finasteride syndrome"}
    assert not [e for e in view["co_mentions"] if {e["a"], e["b"]} & syndromes]
    assert {link["kind"] for link in view["links"]} == {"literature"}
    assert not [link for link in view["links"] if link["term"] in syndromes]
