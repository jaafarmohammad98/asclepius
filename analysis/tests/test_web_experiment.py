"""The lab experiment builder's data (M6-09, spec D1, D2, D4): factors picked from the leaderboard or
the bottleneck table, the design built on the server, each factor's known risks with it and in its CSV."""

import csv
import io
from dataclasses import replace

import pytest
from hypothesis_engine.drugs import read_drug_cache, write_drug_cache
from knowledge_fixture import knowledge_engine, serving
from test_web_server import api

from asclepius.web.experiment import ExperimentError, experiment_view
from asclepius.web.graph import READY, KnowledgeGraph


@pytest.fixture
def engine(tmp_path):
    return knowledge_engine(tmp_path)


@pytest.fixture
def knowledge(engine):
    built = KnowledgeGraph(engine)
    built.start()
    assert built.wait(timeout=30) == READY
    return built


def drug(name):
    return {"kind": "drug", "name": name}


def target(name):
    return {"kind": "target", "name": name}


def build(engine, knowledge, *factors, **more):
    return experiment_view({"factors": list(factors), **more}, engine.drugs, knowledge)


def runs(view):
    lines = [line for line in view["csv"].splitlines() if not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def test_three_leaderboard_drugs_take_the_8_run_full_factorial(engine, knowledge):
    view = build(engine, knowledge, drug("alphadrug"), drug("betadrug"), drug("gammadrug"), seed=11)
    assert view["label"] == "Full factorial 2^3, 8 runs"
    assert view["note"] == "Main effects and two-factor interactions are all separable."
    assert view["seed"] == 11
    assert len(view["runs"]) == 8 and len(runs(view)) == 8
    assert [r["run_order"] for r in view["runs"]] == list(range(1, 9))
    assert sorted(r["run_id"] for r in view["runs"]) == list(range(1, 9))
    assert view["filename"] == "asclepius-experiment-11.csv"


def test_the_same_seed_gives_the_same_run_order(engine, knowledge):
    a = build(engine, knowledge, drug("alphadrug"), target("COMT"), seed=5)
    b = build(engine, knowledge, drug("alphadrug"), target("COMT"), seed=5)
    assert a["runs"] == b["runs"] and a["csv"] == b["csv"]


def test_without_a_seed_one_is_drawn_and_recorded(engine, knowledge):
    view = build(engine, knowledge, drug("alphadrug"), target("COMT"))
    assert isinstance(view["seed"], int)
    assert {r["seed"] for r in runs(view)} == {str(view["seed"])}


def test_five_factors_take_the_16_run_resolution_v_half_fraction(engine, knowledge):
    picked = [drug("alphadrug"), drug("betadrug"), drug("gammadrug"), target("BDNF"), target("serotonin")]
    view = build(engine, knowledge, *picked, seed=42)
    assert view["label"] == "Fractional factorial 2^(5-1), resolution V, 16 runs"
    assert view["note"] == "Main effects and two-factor interactions are all separable."
    lines = view["csv"].splitlines()
    assert lines[0].startswith("# Safety warning")
    assert "safety review before any wet-lab or animal use" in lines[0]
    assert len(runs(view)) == 16
    assert {r["seed"] for r in runs(view)} == {"42"}


def test_each_factor_says_its_known_risks_with_the_design_and_in_the_csv(engine, knowledge):
    view = build(engine, knowledge, drug("alphadrug"), drug("betadrug"), drug("gammadrug"), target("BDNF"))
    said = {f["name"]: f["risks_said"] for f in view["factors"]}
    assert said == {
        "alphadrug": "none on its US label or listed",  # what was checked, never a bare "none"
        "betadrug": "libido; raises blood pressure",
        # no US label: its label risks are unknown, never none
        "gammadrug": "withdrawn from a market after approval; no US label: label risks unknown",
        "BDNF": "none listed: a target, not a compound; the risks of the compound used to act on it apply",
    }
    assert {f["name"]: f["risks"] for f in view["factors"]}["betadrug"] == ["libido", "raises blood pressure"]
    assert all(f["label"] == "candidate for study" for f in view["factors"])
    lines = view["csv"].splitlines()
    for name, text in said.items():
        assert f"# {name} (candidate for study), known risks: {text}" in lines


def test_an_unchecked_label_is_said_as_unknown(engine, knowledge):
    cache = read_drug_cache(engine.drugs)
    alpha = next(d for d in cache.drugs if d.name == "alphadrug")
    unchecked = replace(alpha, safety=replace(alpha.safety, unchecked="timeout"))
    write_drug_cache(
        engine.drugs, replace(cache, drugs=tuple(unchecked if d is alpha else d for d in cache.drugs))
    )
    view = build(engine, knowledge, drug("alphadrug"), target("COMT"))
    assert view["factors"][0]["risks_said"] == "label unchecked: openFDA could not answer"


def test_a_hormone_can_be_a_factor(engine, knowledge):
    view = build(engine, knowledge, target("serotonin"), target("COMT"))
    (serotonin,) = [f for f in view["factors"] if f["name"] == "serotonin"]
    assert serotonin["kind"] == "target" and "not a compound" in serotonin["risks_said"]


def test_the_factors_are_named_in_the_order_picked(engine, knowledge):
    view = build(engine, knowledge, target("COMT"), drug("betadrug"))
    assert [f["name"] for f in view["factors"]] == ["COMT", "betadrug"]
    assert [f["letter"] for f in view["factors"]] == ["A", "B"]
    assert list(runs(view)[0])[3:5] == ["COMT", "betadrug"]


@pytest.mark.parametrize(
    "factors, says",
    [
        ([drug("SERTRALINE"), target("COMT")], "known cause"),
        ([drug("nodrug"), target("COMT")], "not on the leaderboard"),
        ([target("NOPE"), target("COMT")], "not a term of the bottleneck table"),
        ([drug("alphadrug")], "2 to 8 factors"),
        ([drug("alphadrug"), drug("alphadrug")], "own name"),
        ([{"kind": "symptom", "name": "x"}, target("COMT")], "drug or a target"),
    ],
)
def test_a_factor_that_cannot_be_used_is_refused(engine, knowledge, factors, says):
    with pytest.raises(ExperimentError, match=says):
        build(engine, knowledge, *factors)


def test_more_than_eight_factors_is_refused(engine, knowledge):
    names = [drug(n) for n in ("alphadrug", "betadrug", "gammadrug")] + [target(t) for t in ("BDNF", "COMT")]
    names += [target("serotonin")] + [target(f"X{i}") for i in range(3)]
    with pytest.raises(ExperimentError, match="2 to 8 factors, not 9"):
        build(engine, knowledge, *names)


@pytest.mark.parametrize(
    "more, says",
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": "7"}, "seed"),
        ({"replicates": 0}, "replicate"),
        ({"replicates": 21}, "replicate"),
        ({"responses": []}, "response"),
        ({"responses": ["ok", ""]}, "response"),
        ({"responses": "libido"}, "response"),
    ],
)
def test_bad_settings_are_refused(engine, knowledge, more, says):
    with pytest.raises(ExperimentError, match=says):
        build(engine, knowledge, drug("alphadrug"), target("COMT"), **more)


def test_replicates_and_responses_shape_the_csv(engine, knowledge):
    view = build(
        engine, knowledge, drug("alphadrug"), target("COMT"), replicates=2, responses=["libido", "mood"]
    )
    assert view["replicates"] == 2
    header = list(runs(view)[0])
    assert header[-4:] == ["libido_1", "libido_2", "mood_1", "mood_2"]


def test_drug_factors_wait_for_the_knowledge_graph(engine):
    waiting = KnowledgeGraph(engine)  # never started
    with pytest.raises(ExperimentError, match="knowledge graph"):
        build(engine, waiting, drug("alphadrug"), drug("betadrug"))


def test_the_app_builds_experiments_and_refuses_bad_ones(tmp_path):
    with serving(tmp_path) as (_, url):
        status, view = api(
            url, "/api/experiment", "POST", {"factors": [drug("alphadrug"), target("COMT")], "seed": 3}
        )
        assert status == 200 and view["label"] == "Full factorial 2^2, 4 runs"
        status, refused = api(url, "/api/experiment", "POST", {"factors": [drug("SERTRALINE")]})
        assert status == 400 and "error" in refused
        status, refused = api(url, "/api/experiment", "POST", [1])
        assert status == 400
