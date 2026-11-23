"""Run an installed Asclepius through both built-in examples, as a researcher would from the page.

    python e2e.py                  # the app must be installed (`asclepius` on PATH)

Starts `asclepius --no-browser`, checks the extractor page is served, then for each example: loads
it, runs the analysis, downloads both exports, writes hypotheses with the LLM the environment names
(ANTHROPIC_API_KEY, ASCLEPIUS_LLM_MODEL...) and downloads them. The library must be built first
(`asclepius-library build`, then `asclepius-library index`). Exits 1 at the first step that fails.
Standard library only, so it runs before anything else is installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 8765
APP = f"http://127.0.0.1:{PORT}"
LAB_GROUPS = {"group_a": "PSSD", "group_b": "healthy control"}


def call(path: str, method: str = "GET", payload: object = None) -> tuple[int, bytes, str]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(APP + path, data=body, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def api(path: str, method: str = "GET", payload: object = None) -> dict:
    status, body, _ = call(path, method, payload)
    answer = json.loads(body) if body else {}
    if status >= 400:
        fail(f"{method} {path}: {status} {answer.get('error')}")
    return answer


def fail(message: str) -> None:
    print(f"FAILED: {message}", flush=True)
    raise SystemExit(1)


def wait(path: str, minutes: float) -> dict:
    deadline = time.monotonic() + minutes * 60
    while (state := api(path))["state"] == "running":
        if time.monotonic() > deadline:
            fail(f"{path} still running after {minutes} minutes")
        time.sleep(1)
    return state


def check_extractor() -> None:
    status, page, kind = call("/extractor/")
    if status != 200 or not kind.startswith("text/html") or b"<script" not in page:
        fail(f"the extractor page: {status} {kind}")
    status, _, kind = call("/extractor/sw.js")
    if status != 200 or not kind.startswith("text/javascript"):
        fail(f"/extractor/sw.js: {status} {kind}")
    print("ok: extractor page served", flush=True)


def run_example(kind: str) -> None:
    review = api(f"/api/example/{kind}", "POST")
    if "invented" not in review.get("example", ""):
        fail(f"{kind}: the review does not say the example is invented")
    api("/api/run", "POST", LAB_GROUPS if kind == "lab-report" else {})
    run = wait("/api/status", 30)
    if run["state"] != "finished":
        fail(f"{kind}: the run ended {run['state']}")
    view = api("/api/findings")
    labels = [f["label"] for s in view["sections"] for f in s.get("findings", [])]
    if not {"strong", "suggestive"} & set(labels):
        fail(f"{kind}: no strong or suggestive finding")
    for export in ("findings.csv", "report.html"):
        status, body, _ = call(f"/api/export/{export}")
        if status != 200 or not body:
            fail(f"{kind}: export {export} answered {status}")
    print(f"ok: {kind} example ran; {labels.count('strong')} strong, {labels.count('suggestive')} suggestive", flush=True)

    api("/api/hypotheses", "POST")
    written = wait("/api/hypotheses", 60)
    if written["state"] != "finished":
        fail(f"{kind}: hypotheses ended {written['state']}: {written.get('message')}")
    hypotheses = written["hypotheses"]
    if not hypotheses:
        fail(f"{kind}: no hypothesis written ({written.get('message')}; failed: {json.dumps(written.get('failed'))[:2000]})")
    first = hypotheses[0]
    claim = first["mechanism"][0]
    source = written["sources"][claim["passage_id"]]
    if claim["quote"] not in source["text"]:
        fail(f"{kind}: a shown quote is not in its passage")
    status, report, _ = call("/api/export/hypotheses.html")
    if status != 200 or b'id="run-record"' not in report:
        fail(f"{kind}: the hypotheses export answered {status}")
    print(f"ok: {kind}: {len(hypotheses)} hypotheses, first quote found in {source['links'][0]['url']}", flush=True)


def main() -> None:
    app = subprocess.Popen(["asclepius", "--no-browser", "--port", str(PORT)])
    try:
        for _ in range(60):
            try:
                call("/api/status")
                break
            except urllib.error.URLError:
                time.sleep(1)
        else:
            fail("the app did not start")
        print("ok: app started", flush=True)
        check_extractor()
        for kind in ("lab-report", "self-report"):
            run_example(kind)
    finally:
        app.terminate()
        app.wait()
    print("PASSED", flush=True)


if __name__ == "__main__":
    sys.exit(main())
