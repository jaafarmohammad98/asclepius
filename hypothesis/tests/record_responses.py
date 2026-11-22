"""Re-record the responses `test_build.py` and `test_fulltext.py` replay. Needs the network;
run only by hand:

    uv run python tests/record_responses.py build      # PubMed answers, fixtures/build
    uv run python tests/record_responses.py fulltext   # Europe PMC answers, fixtures/fulltext

Then update the literal values in those tests (and `fulltext_fixture.py`) if the answers changed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fulltext_fixture  # noqa: E402
from replay import recording  # noqa: E402

from hypothesis_engine import europepmc  # noqa: E402
from hypothesis_engine.build import build_library  # noqa: E402
from hypothesis_engine.fulltext import fetch_full_texts  # noqa: E402
from hypothesis_engine.pubmed import PubMed, RateLimiter, http_transport  # noqa: E402
from hypothesis_engine.topics import load_topics  # noqa: E402

FOLDER = Path(__file__).parent / "fixtures" / "build"
FETCH_BATCH = 4  # small, so the recording covers more than one efetch request


def record_full_text() -> None:
    folder = fulltext_fixture.FOLDER
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*"):
        old.unlink()
    scratch = folder.parent / "_recording.sqlite"
    scratch.unlink(missing_ok=True)
    fulltext_fixture.make_library(scratch)
    lookup = europepmc.EuropePMC(
        recording(europepmc.http_transport(), folder),
        RateLimiter(europepmc.RATE),
        lookup_batch=fulltext_fixture.LOOKUP_BATCH,
    )
    pmc = PubMed(recording(http_transport(), folder), RateLimiter(3))
    print(fetch_full_texts(scratch, lookup, pmc, download_batch=fulltext_fixture.DOWNLOAD_BATCH))
    scratch.unlink()


def record_build() -> None:
    for old in [*FOLDER.glob("esearch-*"), *FOLDER.glob("efetch-*")]:
        old.unlink()
    scratch = Path(__file__).parent / "fixtures" / "_recording.sqlite"
    scratch.unlink(missing_ok=True)
    pubmed = PubMed(recording(http_transport(), FOLDER), RateLimiter(3), fetch_batch=FETCH_BATCH)
    report = build_library(scratch, load_topics(FOLDER / "topics.toml"), pubmed)
    scratch.unlink()
    print(report)


def main() -> None:
    which = sys.argv[1:] or ["build", "fulltext"]
    for name in which:
        {"build": record_build, "fulltext": record_full_text}[name]()
    shutil.rmtree(Path(__file__).parent / "fixtures" / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
