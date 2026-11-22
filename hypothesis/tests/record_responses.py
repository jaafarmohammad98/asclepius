"""Re-record the responses `test_build.py` replays. Needs the network; run only by hand:

    uv run python tests/record_responses.py

Then update the literal counts in `test_build.py` if PubMed's answers changed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from replay import recording  # noqa: E402

from hypothesis_engine.build import build_library  # noqa: E402
from hypothesis_engine.pubmed import PubMed, RateLimiter, http_transport  # noqa: E402
from hypothesis_engine.topics import load_topics  # noqa: E402

FOLDER = Path(__file__).parent / "fixtures" / "build"
FETCH_BATCH = 4  # small, so the recording covers more than one efetch request


def main() -> None:
    for old in [*FOLDER.glob("esearch-*"), *FOLDER.glob("efetch-*")]:
        old.unlink()
    scratch = Path(__file__).parent / "fixtures" / "_recording.sqlite"
    scratch.unlink(missing_ok=True)
    pubmed = PubMed(recording(http_transport(), FOLDER), RateLimiter(3), fetch_batch=FETCH_BATCH)
    report = build_library(scratch, load_topics(FOLDER / "topics.toml"), pubmed)
    scratch.unlink()
    print(report)
    shutil.rmtree(Path(__file__).parent / "fixtures" / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
