"""The lab experiment builder in Google Chrome (M6-09): five drugs ticked in the leaderboard download a
16-run resolution V CSV, the server's to the byte, with each factor's risks (`browser/experiment.mjs`).

Skipped when node, the extractor's packages (`npm install` in `extractor`) or Chrome are absent.
"""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis_engine.drugs import write_drug_cache
from hypothesis_engine.safety import Label, Safety
from knowledge_fixture import DRUGS, GOOD_MPO, _drug, serving

from asclepius.web.experiment import experiment_view

SCRIPT = Path(__file__).parent / "browser" / "experiment.mjs"
PLAYWRIGHT = Path(__file__).resolve().parents[2] / "extractor" / "node_modules" / "playwright-core"

pytestmark = pytest.mark.skipif(
    not (shutil.which("node") and PLAYWRIGHT.is_dir()), reason="needs node and `npm install` in extractor"
)

MORE = tuple(  # two more drugs, so five can be ticked
    _drug(name, ("BDNF",), GOOD_MPO, Safety(Label(f"set-{name}", None, "NDA9", ()), (), ()))
    for name in ("deltadrug", "epsilondrug")
)


def test_five_leaderboard_drugs_download_a_16_run_csv(tmp_path):
    with serving(tmp_path) as (server, url):
        write_drug_cache(server.engine.drugs, replace(DRUGS, drugs=DRUGS.drugs + MORE))
        names = ["alphadrug", "betadrug", "gammadrug", "deltadrug", "epsilondrug"]
        request = {"factors": [{"kind": "drug", "name": n} for n in names], "seed": 42}
        view = experiment_view(request, server.engine.drugs, server.knowledge)
        assert view["label"] == "Fractional factorial 2^(5-1), resolution V, 16 runs"
        expected = {key: view[key] for key in ("label", "note", "csv", "filename", "seed")} | {"drugs": names}
        done = subprocess.run(
            ["node", str(SCRIPT), url, json.dumps(expected)], capture_output=True, text=True, timeout=120
        )
    if done.returncode == 77:
        pytest.skip("Google Chrome is not installed")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASSED" in done.stdout
