"""The built-in examples: invented datasets of both kinds, loaded from the page with one click, so a
fresh install can be tried end to end before any real data is at hand."""

import time

import pytest
from test_web_server import api, app  # noqa: F401  (a fixture, shared from there)


def finish(app):  # noqa: F811
    deadline = time.monotonic() + 300
    while (run := api(app, "/api/status")[1])["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert run["state"] == "finished"


def labels(app):  # noqa: F811
    status, view = api(app, "/api/findings")
    assert status == 200
    return {f["label"] for s in view["sections"] for f in s.get("findings", [])}


def test_the_lab_report_example_runs_to_a_strong_finding(app):  # noqa: F811
    status, review = api(app, "/api/example/lab-report", "POST")
    assert status == 200
    assert review["kind"] == "lab-report" and "invented" in review["example"]
    assert review["choices"] == ["PSSD", "healthy control"]

    assert api(app, "/api/run", "POST", {"group_a": "PSSD", "group_b": "healthy control"})[0] == 202
    finish(app)
    assert "strong" in labels(app)  # so the hypotheses step has a finding to work on


def test_the_self_report_example_runs_to_a_suggestive_finding(app):  # noqa: F811
    status, review = api(app, "/api/example/self-report", "POST")
    assert status == 200
    assert review["kind"] == "self-report" and "invented" in review["example"]

    assert api(app, "/api/run", "POST", {})[0] == 202
    finish(app)
    assert "suggestive" in labels(app)  # self-reports are never strong (spec A5)


def test_an_example_replaces_the_files_picked_before(app, tmp_path):  # noqa: F811
    assert api(app, "/api/example/self-report", "POST")[0] == 200
    status, review = api(app, "/api/example/lab-report", "POST")
    assert status == 200 and review["kind"] == "lab-report"


@pytest.mark.parametrize("kind", ["blood-test", "..", ""])
def test_an_unknown_example_is_not_found(app, kind):  # noqa: F811
    assert api(app, f"/api/example/{kind}", "POST")[0] == 404
