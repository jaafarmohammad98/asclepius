"""The evidence and hypothesis log in Google Chrome (M6-11): a refuting RCT logged in the form re-ranks the
leaderboard and the bottleneck map, the log downloads as the server keeps it, and an imported log merges
(`browser/evidence_log.mjs`).

Skipped when node, the extractor's packages (`npm install` in `extractor`) or Chrome are absent.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis_engine.drugs import write_drug_cache
from hypothesis_engine.evidence_log import Entry, EvidenceLog
from knowledge_fixture import RERANKED, serving

SCRIPT = Path(__file__).parent / "browser" / "evidence_log.mjs"
PLAYWRIGHT = Path(__file__).resolve().parents[2] / "extractor" / "node_modules" / "playwright-core"

pytestmark = pytest.mark.skipif(
    not (shutil.which("node") and PLAYWRIGHT.is_dir()), reason="needs node and `npm install` in extractor"
)


def test_a_logged_result_re_ranks_the_leaderboard_in_the_browser(tmp_path):
    other = tmp_path / "other-researcher.json"  # another researcher's log, to import
    other.write_text(EvidenceLog([Entry.new("COMT", "supports", "cohort", "10.1234/xyz")]).to_json())
    with serving(tmp_path) as (server, url):
        write_drug_cache(server.engine.drugs, RERANKED)
        expected = {
            # as test_web_evidence_log.py's Python test ranks them, before and after the refuting RCT
            "before": ["alphadrug", "epsilondrug", "deltadrug", "gammadrug", "betadrug"],
            "after": ["alphadrug", "epsilondrug", "deltadrug", "betadrug", "gammadrug"],
            "map_after": ["serotonin", "BDNF", "COMT"],
            "import": str(other),
        }
        done = subprocess.run(
            ["node", str(SCRIPT), url, json.dumps(expected)], capture_output=True, text=True, timeout=120
        )
        kept = EvidenceLog.load(server.engine.evidence)
    if done.returncode == 77:
        pytest.skip("Google Chrome is not installed")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASSED" in done.stdout
    assert [(e.node, e.direction, e.source) for e in kept.entries] == [
        ("BDNF", "refutes", "PMID:31000009"),
        ("COMT", "supports", "doi:10.1234/xyz"),
    ]
