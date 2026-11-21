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
