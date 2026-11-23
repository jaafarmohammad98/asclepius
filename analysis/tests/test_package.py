"""The one thing a researcher installs: a wheel with the app, the hypothesis engine, the extractor's
built page and the examples, and nothing from any data folder (`package/build.py`).

Slow: it builds the extractor page (npm) and the wheel. Skipped when npm or git is absent.
"""

from __future__ import annotations

import importlib.util
import shutil
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (shutil.which("npm") and shutil.which("git")), reason="building the wheel needs npm and git"
)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("package_build", REPO / "package" / "build.py")
    assert spec and spec.loader
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    path = build.build_wheel(tmp_path_factory.mktemp("dist"))
    with zipfile.ZipFile(path) as archive:
        yield path, archive.namelist(), archive


def dist_info(names, archive, name):
    (path,) = [n for n in names if n.endswith(f".dist-info/{name}")]
    return archive.read(path).decode()


def test_it_is_one_wheel_named_asclepius(wheel):
    path, _, _ = wheel
    assert path.name.startswith("asclepius-") and path.name.endswith("-py3-none-any.whl")


def test_it_holds_the_app_the_engine_the_extractor_page_and_the_examples(wheel):
    _, names, _ = wheel
    for name in (
        "asclepius/web/server.py",
        "asclepius/web/index.html",
        "asclepius/examples.py",
        "asclepius/synthetic.py",
        "hypothesis_engine/pipeline.py",
        "hypothesis_engine/topics.toml",
        "hypothesis_engine/interventions.toml",
        "asclepius/web/extractor/index.html",
        "asclepius/web/extractor/sw.js",
    ):
        assert name in names
    assert any(n.startswith("asclepius/web/extractor/assets/pdf.worker") for n in names)


def test_it_holds_no_data_tests_or_build_leftovers(wheel):
    _, names, _ = wheel
    for name in names:
        top = name.split("/")[0]
        assert top in ("asclepius", "hypothesis_engine") or top.endswith(".dist-info"), name
        parts = name.split("/")
        assert not {"data", "tests", "__pycache__", "node_modules", "private"} & set(parts), name
        assert not name.endswith((".csv", ".sqlite", ".pdf", ".json", ".pyc")), name


def test_one_command_starts_the_app_and_one_builds_the_library(wheel):
    _, names, archive = wheel
    scripts = dist_info(names, archive, "entry_points.txt")
    assert "asclepius = asclepius.web.server:main" in scripts
    assert "asclepius-library = hypothesis_engine.build:main" in scripts


def test_its_dependencies_are_the_tested_versions(wheel):
    _, names, archive = wheel
    metadata = Parser().parsestr(dist_info(names, archive, "METADATA"))
    requires = metadata.get_all("Requires-Dist") or []
    pinned = {r.split("==")[0].strip().lower() for r in requires if "==" in r}
    assert {"pandas", "numpy", "anthropic", "fastembed", "pypdf"} <= pinned
    assert not any("asclepius" in r for r in requires)  # the engine is inside, not a dependency
    assert all("==" in r for r in requires)
