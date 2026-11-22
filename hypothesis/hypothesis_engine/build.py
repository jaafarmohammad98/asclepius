"""Build the paper library: `asclepius-library build`, then `asclepius-library fulltext`.

`build` runs each topic search, fetches every paper found (once, however many topics found
it) and stores them as the library's next version. Nothing is written until every request
has succeeded, so a failed build leaves the previous library as it was. It then downloads
the open-access full text (see `fulltext.py`) and prints how many papers have it.

`fulltext` does only that second step, for the current version. It picks up where an
interrupted run stopped.

Set NCBI_API_KEY to an NCBI API key (free, from an NCBI account) to send 10 requests a
second instead of 3. It is only ever sent to NCBI.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hypothesis_engine import europepmc
from hypothesis_engine.fulltext import describe, fetch_full_texts
from hypothesis_engine.library import LibraryError, write_build
from hypothesis_engine.pubmed import (
    PubMed,
    PubMedError,
    RateLimiter,
    TransientError,
    http_transport,
    rate_for,
)
from hypothesis_engine.topics import DEFAULT_TOPICS, Topic, TopicsError, load_topics

# inside hypothesis/, whatever folder the command runs from (git ignores hypothesis/data/)
DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "data" / "library.sqlite"


@dataclass(frozen=True)
class BuildReport:
    version: int
    papers: int
    lacking_abstract: int
    found_per_topic: dict[str, int]  # topic -> PMIDs its search returned
    missing: int  # PMIDs a search returned that efetch had no usable record for


def _fingerprint(topics: Sequence[Topic]) -> str:
    """Identifies the exact searches a build ran, so two versions can be told apart."""
    text = "\n".join(f"{t.name}\t{t.query}\t{t.max_papers}" for t in topics)
    return hashlib.sha256(text.encode()).hexdigest()


def build_library(
    path: Path,
    topics: Sequence[Topic],
    pubmed: PubMed,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[str], None] = lambda _message: None,
) -> BuildReport:
    """Search, fetch and store; return what the new library version holds."""
    found_by: dict[str, list[tuple[str, int]]] = {}  # PMID -> (topic, rank), first-found order
    found_per_topic: dict[str, int] = {}
    for topic in topics:
        pmids = pubmed.search(topic.query, topic.max_papers)
        found_per_topic[topic.name] = len(pmids)
        progress(f"{topic.name}: {len(pmids)} papers")
        for rank, pmid in enumerate(pmids, start=1):
            found_by.setdefault(pmid, []).append((topic.name, rank))

    progress(f"fetching {len(found_by)} paper records")
    fetched = {paper.pmid: paper for paper in pubmed.fetch(list(found_by))}
    papers = [fetched[pmid] for pmid in found_by if pmid in fetched]

    version = write_build(path, papers, found_by, _fingerprint(topics), now().isoformat(timespec="seconds"))
    return BuildReport(
        version=version,
        papers=len(papers),
        lacking_abstract=sum(1 for p in papers if p.abstract is None),
        found_per_topic=found_per_topic,
        missing=len(found_by) - len(papers),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asclepius-library", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="download paper records from PubMed, then their full text")
    fulltext = commands.add_parser("fulltext", help="download the open-access full text still missing")
    for command in (build, fulltext):
        command.add_argument(
            "--library", type=Path, default=DEFAULT_LIBRARY, help=f"SQLite file (default {DEFAULT_LIBRARY})"
        )
    build.add_argument(
        "--topics", type=Path, default=DEFAULT_TOPICS, help="topic searches (default: the shipped list)"
    )
    args = parser.parse_args(argv)

    say = lambda message: print(message, file=sys.stderr)  # noqa: E731
    downloading = args.command == "fulltext"
    try:
        api_key = os.environ.get("NCBI_API_KEY") or None
        pubmed = PubMed(http_transport(api_key), RateLimiter(rate_for(api_key)))
        if not downloading:
            topics = load_topics(args.topics)
            report = build_library(args.library, topics, pubmed, progress=say)
            say(
                f"library version {report.version} at {args.library}: {report.papers} papers, "
                f"{report.lacking_abstract} without an abstract"
                + (f", {report.missing} found by a search but not fetchable" if report.missing else "")
            )
        downloading = True
        lookup = europepmc.EuropePMC(europepmc.http_transport(), RateLimiter(europepmc.RATE))
        say(describe(fetch_full_texts(args.library, lookup, pubmed, progress=say)))
    except KeyboardInterrupt:
        if downloading:
            say("asclepius-library: stopped; `asclepius-library fulltext` continues the download")
        else:
            say("asclepius-library: stopped before the build was stored; the library is as it was")
        return 130
    except (TopicsError, LibraryError, PubMedError, TransientError, europepmc.EuropePMCError) as error:
        say(f"asclepius-library: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
