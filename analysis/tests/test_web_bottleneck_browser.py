"""The bottleneck map in Google Chrome (M6-04): drawn from the app alone, re-ranked by the weight
sliders as the Python scores rank (each times its posterior mean in the evidence log), and a clicked link showing its passages (`browser/bottleneck_map.mjs`);
after a run, its solid finding line showing its findings (M6-04b, `browser/finding_link.mjs`).

Skipped when node, the extractor's packages (`npm install` in `extractor`) or Chrome are absent.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis_engine.bottlenecks import Weights, bottlenecks
from hypothesis_engine.evidence_log import weighted
from knowledge_fixture import serving
from test_web_finding_links import run_example, settled

SCRIPT = Path(__file__).parent / "browser" / "bottleneck_map.mjs"
FINDING_SCRIPT = Path(__file__).parent / "browser" / "finding_link.mjs"
PLAYWRIGHT = Path(__file__).resolve().parents[2] / "extractor" / "node_modules" / "playwright-core"

pytestmark = pytest.mark.skipif(
    not (shutil.which("node") and PLAYWRIGHT.is_dir()), reason="needs node and `npm install` in extractor"
)


def test_the_map_works_in_the_browser(tmp_path):
    with serving(tmp_path) as (server, url):
        graph, log = server.knowledge.graph, server.knowledge.evidence.log()
        ranked = weighted(bottlenecks(graph), graph, log)  # each score times its posterior mean (M6-11)
        cascade_only = bottlenecks(graph, Weights(overlap=0, cascade=1, betweenness=0))
        expected = {
            "default": [b.name for b in ranked],
            "top_score": f"{ranked[0].score:.2f}",
            "cascade_only": [b.name for b in weighted(cascade_only, graph, log)],
        }
        done = subprocess.run(
            ["node", str(SCRIPT), url, json.dumps(expected)], capture_output=True, text=True, timeout=120
        )
    if done.returncode == 77:
        pytest.skip("Google Chrome is not installed")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASSED" in done.stdout


def test_a_run_s_finding_line_opens_in_the_browser(tmp_path):
    with serving(tmp_path, with_androsterone=True) as (server, url):
        run_example(url)
        view = settled(url)
        expected = {
            "fingerprint": view["run"]["fingerprint"],
            "from": "post-SSRI sexual dysfunction",
            "term": "androsterone",
            "label": "in 2 significant analysis findings",
        }
        done = subprocess.run(
            ["node", str(FINDING_SCRIPT), url, json.dumps(expected)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if done.returncode == 77:
        pytest.skip("Google Chrome is not installed")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASSED" in done.stdout
