"""The hypothesis engine's privacy fixtures (M4-08) are what this engine exports today.

`hypothesis/tests/test_outgoing_requests.py` checks that no record id or individual value of two
synthetic datasets reaches an LLM, starting from the findings this engine exported from them. The
hypothesis engine cannot run this one, so the datasets and exports are committed there; this test
writes them again and fails when they differ, so a change here cannot leave them stale.
"""

import csv
import importlib.util
import io
from pathlib import Path

HYPOTHESIS_TESTS = Path(__file__).parents[2] / "hypothesis" / "tests"
COMMITTED = HYPOTHESIS_TESTS / "fixtures" / "privacy"
REGENERATE = "cd hypothesis && uv run --project ../analysis python tests/privacy_fixture.py"


def _script():
    spec = importlib.util.spec_from_file_location("privacy_fixture", HYPOTHESIS_TESTS / "privacy_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _content(path: Path) -> list[dict[str, str]]:
    """The file's rows, without the run date, which changes with every run."""
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    for row in rows:
        row.pop("run_date", None)
    return rows


def test_the_privacy_fixtures_are_current(tmp_path):
    _script().write(tmp_path)
    written = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.csv"))
    assert written == sorted(p.relative_to(COMMITTED) for p in COMMITTED.rglob("*.csv")), REGENERATE
    stale = [str(name) for name in written if _content(tmp_path / name) != _content(COMMITTED / name)]
    assert stale == [], f"the hypothesis engine's privacy fixtures are stale; write them again: {REGENERATE}"
