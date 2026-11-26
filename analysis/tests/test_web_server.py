"""The web app over HTTP, as the page uses it: from picking a dataset file to a finished run."""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from asclepius import synthetic
from asclepius.run import Settings
from asclepius.web.export import read_findings_csv
from asclepius.web.server import make_server


@pytest.fixture
def app(tmp_path):
    server = make_server(tmp_path / "work", port=0, settings=Settings(permutations=999, bootstrap=200))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def call(url, method="GET", body=None, headers=None):
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def api(app, path, method="GET", payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    status, content = call(app + path, method, body, {"Content-Type": "application/json"})
    return status, json.loads(content)


def upload(app, role, name, content):
    query = urllib.parse.urlencode({"name": name})
    status, _ = call(f"{app}/api/files/{role}?{query}", "PUT", content)
    assert status == 204


def test_the_page_is_served(app):
    status, page = call(app + "/")
    assert status == 200
    assert b"<title>Asclepius</title>" in page


def test_the_page_asks_only_this_app_for_its_styles_fonts_and_icon(app):
    status, page = call(app + "/")
    assert b"fonts.googleapis" not in page and b"https://" not in page.split(b"<script>")[0]
    for path, kind in (
        ("/almanac.css", "text/css"),
        ("/favicon.svg", "image/svg+xml"),
        ("/vendor/fonts/newsreader.woff2", "font/woff2"),
        ("/vendor/fonts/hanken-grotesk.woff2", "font/woff2"),
        ("/vendor/fonts/ibm-plex-mono-400.woff2", "font/woff2"),
        ("/vendor/fonts/ibm-plex-mono-500.woff2", "font/woff2"),
    ):
        with urllib.request.urlopen(app + path) as response:
            assert response.status == 200 and response.headers["Content-Type"].startswith(kind), path
            assert response.read()
    with urllib.request.urlopen(app + "/almanac.css") as response:
        css = response.read().decode()
    assert "https://" not in css  # its fonts are the app's own


def test_a_dataset_file_goes_to_a_finished_run(app, tmp_path):
    out = synthetic.write_lab_reports(tmp_path / "lab", groups={"PSSD": 40, "healthy control": 40}, seed=1)
    assert api(app, "/api/clear", "POST")[0] == 200
    upload(app, "reports", "reports.csv", out.reports.read_bytes())
    upload(app, "metadata", "metadata.csv", out.metadata.read_bytes())

    status, review = api(app, "/api/load", "POST")
    assert status == 200 and review["kind"] == "lab-report"
    assert review["choices"] == ["PSSD", "healthy control"]

    status, _ = api(app, "/api/run", "POST", {"group_a": "PSSD", "group_b": "healthy control"})
    assert status == 202
    deadline = time.monotonic() + 120
    while (run := api(app, "/api/status")[1])["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert run["state"] == "finished" and run["done"] == run["total"] == 4

    # strict JSON, as a browser parses it: no NaN anywhere
    status, content = call(app + "/api/findings")
    assert status == 200
    findings = json.loads(content, parse_constant=lambda c: pytest.fail(f"{c} is not JSON"))
    assert [s["kind"] for s in findings["sections"]] == ["subgroups", "comparison", "syndromes", "ranges"]

    # both exports download as files, the CSV reloading to the findings shown
    for name, kind in (("findings.csv", "text/csv"), ("report.html", "text/html")):
        with urllib.request.urlopen(f"{app}/api/export/{name}") as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith(kind)
            assert response.headers["Content-Disposition"].startswith("attachment; filename=")
            content = response.read().decode()
        if name == "findings.csv":
            rows = [r for r in read_findings_csv(content) if r["row"] == "finding"]
            assert len(rows) == sum(len(s["findings"]) for s in findings["sections"])
        else:
            assert 'id="run-record"' in content


def test_nothing_is_exported_before_a_run(app):
    api(app, "/api/clear", "POST")
    status, answer = api(app, "/api/export/report.html")
    assert status == 400
    assert "before exporting" in answer["error"]


def test_a_loading_error_comes_back_in_plain_words(app):
    api(app, "/api/clear", "POST")
    upload(app, "reports", "march.csv", b"record_id\nr001\n")
    status, answer = api(app, "/api/load", "POST")
    assert status == 400
    assert answer["error"].startswith("march.csv: missing column(s)")


def test_requests_from_other_sites_are_refused(app):
    """Another web page open in the browser must not read or run anything (DNS rebinding, CSRF)."""
    status, _ = call(app + "/api/status", headers={"Host": "evil.example"})
    assert status == 403
    status, _ = call(app + "/api/clear", "POST", b"", {"Origin": "http://evil.example"})
    assert status == 403


@pytest.mark.parametrize(
    ("headers", "body"),
    [({"Content-Type": "application/json"}, b"[]"), ({"Content-Type": "application/json"}, b"{")],
)
def test_a_malformed_run_request_is_answered_in_plain_words(app, headers, body):
    status, content = call(app + "/api/run", "POST", body, headers)
    assert status == 400
    assert "error" in json.loads(content)


# ------------------------------------------------------------------ hypotheses (M4-09)


@pytest.fixture
def hypothesis_app(tmp_path):
    from test_web_hypotheses import build_library, engine

    (tmp_path / "library").mkdir()
    folder = build_library(tmp_path / "library")
    server = make_server(
        tmp_path / "work",
        port=0,
        settings=Settings(permutations=999, bootstrap=200),
        engine=engine(folder, settings="none"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


KEY = "sk-ant-server-test-key"


def test_the_settings_screen_stores_a_key_it_never_sends_back(hypothesis_app):
    status, shown = api(hypothesis_app, "/api/settings")
    assert status == 200 and shown["active"] is None
    form = {"backend": "anthropic", "anthropic": {"api_key": KEY, "model": "claude-opus-5-5"}}
    status, content = call(
        hypothesis_app + "/api/settings",
        "POST",
        json.dumps(form).encode(),
        {"Content-Type": "application/json"},
    )
    assert status == 200 and KEY.encode() not in content
    assert json.loads(content)["anthropic"]["key_set"] is True
    status, answer = api(
        hypothesis_app, "/api/settings", "POST", {"backend": "local", "local": {"model": ""}}
    )
    assert status == 400 and "no model is named" in answer["error"]


def test_a_finding_leads_to_its_hypothesis_and_its_quoted_source(hypothesis_app, tmp_path):
    app = hypothesis_app
    out = synthetic.write_lab_reports(
        tmp_path / "lab",
        groups={"PSSD": 40, "healthy control": 40},
        seed=1,
        planted=[synthetic.PlantedDifference("androsterone", 0.4)],
    )
    api(app, "/api/clear", "POST")
    upload(app, "reports", "reports.csv", out.reports.read_bytes())
    upload(app, "metadata", "metadata.csv", out.metadata.read_bytes())
    api(app, "/api/load", "POST")
    assert api(app, "/api/hypotheses", "POST")[0] == 400  # no run yet
    api(app, "/api/run", "POST", {"group_a": "PSSD", "group_b": "healthy control"})
    deadline = time.monotonic() + 120
    while api(app, "/api/status")[1]["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    local = {"backend": "local", "local": {"model": "llama3.1:8b"}}
    assert api(app, "/api/settings", "POST", local)[0] == 200
    assert api(app, "/api/settings/check", "POST")[1]["model"] == "fake-1"

    status, started = api(app, "/api/hypotheses", "POST")
    assert status == 202 and started["state"] == "running"
    while (view := api(app, "/api/hypotheses")[1])["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert view["state"] == "finished", view["message"]

    # finding -> hypothesis -> quoted source
    sections = api(app, "/api/findings")[1]["sections"]
    keys = {
        "|".join(
            v or ""
            for v in (
                s["kind"],
                f["comparison"],
                f["marker_id"],
                f.get("sex"),
                f.get("group_a"),
                f.get("group_b"),
            )
        )
        for s in sections
        for f in s["findings"]
    }
    hypothesis = view["hypotheses"][0]
    assert hypothesis["key"] in keys
    claim = hypothesis["mechanism"][0]
    source = view["sources"][claim["passage_id"]]
    assert claim["quote"] in source["text"] and source["links"][0]["url"].startswith(
        "https://pubmed.ncbi.nlm.nih.gov/"
    )

    with urllib.request.urlopen(f"{app}/api/export/hypotheses.html") as response:
        assert response.headers["Content-Type"].startswith("text/html")
        assert response.headers["Content-Disposition"].startswith(
            'attachment; filename="asclepius-hypotheses-'
        )
        assert 'id="run-record"' in response.read().decode()
