"""The drug repurposing leaderboard in Google Chrome (M6-07): ranked as Python ranks, re-ranked by the
weight sliders without asking the server, filtered, sorted, its known causes apart, and a target's
passages opened (`browser/leaderboard.mjs`).

Skipped when node, the extractor's packages (`npm install` in `extractor`) or Chrome are absent.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis_engine.drugs import read_drug_cache
from hypothesis_engine.topsis import Weights, rank
from knowledge_fixture import serving

from asclepius.web.leaderboard import rescored

SCRIPT = Path(__file__).parent / "browser" / "leaderboard.mjs"
PLAYWRIGHT = Path(__file__).resolve().parents[2] / "extractor" / "node_modules" / "playwright-core"

pytestmark = pytest.mark.skipif(
    not (shutil.which("node") and PLAYWRIGHT.is_dir()), reason="needs node and `npm install` in extractor"
)


def test_the_leaderboard_works_in_the_browser(tmp_path):
    with serving(tmp_path) as (server, url):
        cache = rescored(read_drug_cache(server.engine.drugs), server.knowledge.scores)

        def names(weights):
            return [r.drug.name for r in rank(cache, weights).ranked]

        expected = {
            "default": names(Weights()),
            "safety_only": names(Weights(impact=0, coverage=0, safety=1, bbb=0, readiness=0)),
        }
        assert expected["default"] != expected["safety_only"]  # so the sliders are seen to re-rank
        done = subprocess.run(
            ["node", str(SCRIPT), url, json.dumps(expected)], capture_output=True, text=True, timeout=120
        )
    if done.returncode == 77:
        pytest.skip("Google Chrome is not installed")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASSED" in done.stdout
